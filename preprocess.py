"""
Audio preprocessing: offline augmentation + GPU mel cache.

Phase 1 — Parselmouth augmentation (CPU, per file):
    For every audio file, one pitch+formant-shifted variant is saved as a
    16-kHz audio tensor alongside the source:
        <stem>_aug.pt   random direction: pitch Uniform(1.10,1.30) or Uniform(0.70,0.90)
                                          formant Uniform(1.05,1.15) or Uniform(0.85,0.95)

Phase 2 — GPU mel cache (batched):
    For every clean audio file, a log-mel spectrogram is computed on the GPU:
        <stem>.pt   100-band log-mel at 24 kHz

Mel parameters match the Vocos vocoder (vocos-mel-24khz):
    sample_rate=24000, n_fft=1024, hop_length=256, win_length=1024, n_mels=100
    log scale: log(mel.clamp(min=1e-7))

Already-existing .pt files are skipped (safe to interrupt and resume).

Usage:
    python preprocess.py --data-root datasets/wav48_silence_trimmed
    python preprocess.py --data-root datasets/wav48_silence_trimmed --batch-size 64 --num-workers 8
    python preprocess.py --data-root datasets/wav48_silence_trimmed --skip-aug
"""

import argparse
import random
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from augmentation import augment_pitch_and_formant

# ── Mel parameters (must match dataset.py / train.py) ────────────────────────
MEL_SR = 24_000
AUG_SR = 16_000   # target SR for augmented audio tensors
N_MELS = 100
N_FFT  = 1024
HOP    = 256
WIN    = 1024

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
_AUG_SUFFIXES    = ("_aug",)


# ── Phase 1: Parselmouth augmentation ────────────────────────────────────────

def _load_to_16k(path: str) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != AUG_SR:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, AUG_SR).squeeze(0)
    return wav


def _augment_file_worker(args: tuple[Path, float, float]) -> None:
    path, pitch_ratio, formant_ratio = args
    out_pt = path.with_name(path.stem + "_aug.pt")
    if out_pt.exists():
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        augment_pitch_and_formant(str(path), tmp_path, formant_ratio, pitch_ratio)
        wav = _load_to_16k(tmp_path)
        torch.save(wav.clone(), out_pt)
    except Exception as e:
        print(f"  [WARN] {path.name} → {out_pt.name}: {e}")
        out_pt.unlink(missing_ok=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_augmentation(root: Path, seed: int, num_workers: int = 8) -> None:
    files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS
        and not any(p.stem.endswith(s) for s in _AUG_SUFFIXES)
    )
    todo = [p for p in files if not p.with_name(p.stem + "_aug.pt").exists()]
    print(f"\nPhase 1 — Augmentation: {len(todo)} / {len(files)} files need augmenting.")
    if not todo:
        return

    rng = random.Random(seed)
    tasks = []
    for path in todo:
        pitch_range   = rng.choice([(1.10, 1.30), (0.70, 0.90)])
        formant_range = rng.choice([(1.05, 1.15), (0.85, 0.95)])
        pitch_ratio   = rng.uniform(*pitch_range)
        formant_ratio = rng.uniform(*formant_range)
        tasks.append((path, pitch_ratio, formant_ratio))

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(_augment_file_worker, tasks), total=len(tasks), desc="Augmenting", unit="file"))


# ── Phase 2: GPU mel cache ────────────────────────────────────────────────────

class WavDataset(Dataset):
    """Recursively finds clean audio files; skips augmented stems and existing mels."""

    def __init__(self, root: str) -> None:
        all_files = sorted(
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in AUDIO_EXTENSIONS
            and not any(p.stem.endswith(s) for s in _AUG_SUFFIXES)
        )
        self.files = [p for p in all_files if not p.with_suffix(".pt").exists()]
        print(f"\nPhase 2 — Mel cache: {len(all_files)} audio files — "
              f"{len(self.files)} left to process.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)
        return wav, sr, str(path)


def collate_fn(batch: list) -> tuple:
    wavs, srs, paths = zip(*batch)
    resampled_wavs = []
    for w, sr in zip(wavs, srs):
        if sr != MEL_SR:
            w = torchaudio.functional.resample(w, sr, MEL_SR)
        resampled_wavs.append(w)
    
    lengths = [w.shape[0] for w in resampled_wavs]
    max_len = max(lengths)
    padded  = torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in resampled_wavs])
    return padded, lengths, MEL_SR, paths


def run_mel_cache(
    root: str, batch_size: int, num_workers: int, device: torch.device
) -> None:
    ds = WavDataset(root)
    if len(ds) == 0:
        print("  Nothing to do — all mel .pt files already exist.")
        return

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        shuffle=False,
    )

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=MEL_SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN, n_mels=N_MELS, power=1.0, center=True,
    ).to(device)

    with torch.no_grad():
        for padded, lengths, native_sr, paths in tqdm(loader, desc="Mel cache", unit="batch"):
            padded = padded.to(device)
            # padded is already resampled to MEL_SR in collate_fn

            mel_padded = torch.log(mel_transform(padded).clamp(min=1e-7))
            mel_padded = mel_padded.cpu()

            for mel, path, orig_len in zip(mel_padded, paths, lengths):
                # orig_len is already at MEL_SR
                t_mel = 1 + orig_len // HOP
                torch.save(mel[:, :t_mel].clone(), Path(path).with_suffix(".pt"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess audio: Parselmouth augmentation + GPU mel cache"
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Root directory containing speaker sub-folders with audio files",
    )
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument(
        "--skip-aug", action="store_true",
        help="Skip Phase 1 (Parselmouth augmentation) and only run mel cache",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root = Path(args.data_root)
    if not args.skip_aug:
        run_augmentation(root, args.seed, args.num_workers)

    run_mel_cache(args.data_root, args.batch_size, args.num_workers, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
