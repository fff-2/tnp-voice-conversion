"""
GPU-accelerated mel-spectrogram preprocessing for voice conversion.

Reads all audio files in a directory tree, computes log-mel spectrograms on
the GPU in batches, and saves a .pt tensor alongside each source file.

Mel parameters match the Vocos vocoder (vocos-mel-24khz):
    sample_rate=24000, n_fft=1024, hop_length=256, win_length=1024, n_mels=100
    log scale: log(mel.clamp(min=1e-5))

Already-preprocessed files (.pt exists) are skipped automatically, so the
script is safe to interrupt and resume.

Usage:
    python preprocess.py --data-root datasets/wav48_silence_trimmed
    python preprocess.py --data-root datasets/wav48_silence_trimmed --batch-size 64 --num-workers 8
"""

import argparse
import math
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Mel parameters (must match dataset.py / train.py) ────────────────────────
MEL_SR   = 24_000
N_MELS   = 100
N_FFT    = 1024
HOP      = 256
WIN      = 1024

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


# ── Dataset ───────────────────────────────────────────────────────────────────

class WavDataset(Dataset):
    """Recursively finds audio files; __getitem__ reads with soundfile."""

    def __init__(self, root: str) -> None:
        all_files = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
        )
        # Skip files whose .pt already exists (safe resume)
        self.files = [p for p in all_files if not p.with_suffix(".pt").exists()]
        print(f"Found {len(all_files)} audio files — {len(self.files)} left to process.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)        # [C, T]
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)   # mono [1, T]
        wav = wav.squeeze(0)                  # [T]
        return wav, sr, str(path)


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: list) -> tuple:
    """Pad variable-length waveforms to the batch maximum and stack."""
    wavs, srs, paths = zip(*batch)

    # All files in a batch must share the same native sample rate so that
    # a single GPU resampling kernel can handle the whole batch.
    assert len(set(srs)) == 1, (
        f"Mixed sample rates in batch {set(srs)} — "
        "ensure all files in the dataset share the same sample rate."
    )

    lengths = [w.shape[0] for w in wavs]
    max_len = max(lengths)
    padded  = torch.stack([F.pad(w, (0, max_len - w.shape[0])) for w in wavs])
    # padded: [B, T_max]
    return padded, lengths, srs[0], paths


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU mel preprocessing")
    parser.add_argument(
        "--data-root", required=True,
        help="Root directory containing speaker sub-folders with audio files",
    )
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = WavDataset(args.data_root)
    if len(ds) == 0:
        print("Nothing to do — all .pt files already exist.")
        return

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        shuffle=False,
    )

    # GPU mel transform (initialized once, shared across all batches)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=MEL_SR,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        n_mels=N_MELS,
    ).to(device)

    resampler = None   # created lazily once native_sr is known

    with torch.no_grad():
        for padded, lengths, native_sr, paths in tqdm(loader, desc="Preprocessing", unit="batch"):
            # ── Move to GPU ───────────────────────────────────────────────────
            padded = padded.to(device)   # [B, T_max] at native_sr

            # ── Resample native SR → 24 kHz on GPU ────────────────────────────
            if native_sr != MEL_SR:
                if resampler is None:
                    resampler = torchaudio.transforms.Resample(
                        native_sr, MEL_SR
                    ).to(device)
                padded = resampler(padded)   # [B, T_max_24k]

            # ── Mel + log compression on GPU ──────────────────────────────────
            mel_padded = mel_transform(padded)              # [B, N_MELS, T_mel_padded]
            mel_padded = torch.log(mel_padded.clamp(min=1e-5))

            # ── Move back to CPU for saving ───────────────────────────────────
            mel_padded = mel_padded.cpu()

            # ── Trim padding and save each file ───────────────────────────────
            for mel, path, orig_len in zip(mel_padded, paths, lengths):
                # Compute how many mel frames the original (unpadded) audio produces.
                # With center=True (torchaudio default): frames = 1 + T_24k // hop
                t_24k = math.ceil(orig_len * MEL_SR / native_sr)
                t_mel = 1 + t_24k // HOP
                mel = mel[:, :t_mel].clone()   # clone breaks view into padded batch storage

                out_path = Path(path).with_suffix(".pt")
                torch.save(mel, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
