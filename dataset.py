"""
Generic speaker dataset for voice conversion training.

Expected folder layout — one sub-folder per speaker, audio files inside:

    data/
    ├── alice/
    │   ├── clip_01.wav
    │   ├── clip_02.wav
    │   └── ...
    ├── bob/
    │   ├── clip_01.wav
    │   └── ...
    └── ...

Supported audio formats: .wav  .flac  .mp3  .ogg
Minimum utterances per speaker: N_CTX + 2  (default: 7)
"""

import random
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000
MEL_SAMPLE_RATE = 24_000   # resample to this before computing mel (matches vocos-mel-24khz)
N_MELS = 100
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


class SpeakerDataset(Dataset):
    """
    Builds random cross-speaker pairs from a folder of speaker sub-directories.

    Each __getitem__ returns:
        source_audio   [T_src]             source speaker waveform @ 16 kHz
        target_audio   [T_tgt]             target speaker waveform @ 16 kHz
        context_mels   [N_CTX, N_MELS, T]  reference mels from target speaker (zero-padded)
        ctx_mel_lens   list[N_CTX]         unpadded T length of each reference mel
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        max_sec: float = 8.0,
        n_ctx: int = 5,
        min_utts: int = 7,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.max_samples = int(max_sec * SAMPLE_RATE)
        self.n_ctx = n_ctx

        # Collect per-speaker file lists
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Data root not found: {root_path}")

        speakers: dict[str, list[Path]] = {}
        for spk_dir in sorted(root_path.iterdir()):
            if not spk_dir.is_dir():
                continue
            files = sorted(
                f for f in spk_dir.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS
            )
            if len(files) >= min_utts:
                speakers[spk_dir.name] = files

        if len(speakers) < 2:
            raise ValueError(
                f"Need at least 2 speakers with ≥{min_utts} utterances each. "
                f"Found {len(speakers)} valid speaker(s) in {root_path}."
            )

        # Deterministic train/val split by speaker
        spk_list = sorted(speakers.keys())
        rng = random.Random(seed)
        rng.shuffle(spk_list)
        cut = max(1, int(len(spk_list) * (1 - val_ratio)))
        if split == "train":
            chosen = spk_list[:cut]
        else:
            chosen = spk_list[cut:] or spk_list[-1:]   # at least 1 val speaker

        self.speakers: dict[str, list[Path]] = {s: speakers[s] for s in chosen}
        self.spk_names = list(self.speakers.keys())

        # Build index: each entry is (src_spk, tgt_spk, src_idx, tgt_idx)
        # We create N_PAIRS pairs per speaker combination
        N_PAIRS = 50
        self.pairs: list[tuple[str, str, int, int]] = []
        rng2 = random.Random(seed + 1)
        for i, src in enumerate(self.spk_names):
            for tgt in self.spk_names:
                if src == tgt:
                    continue
                src_files = self.speakers[src]
                tgt_files = self.speakers[tgt]
                for _ in range(N_PAIRS):
                    self.pairs.append((
                        src, tgt,
                        rng2.randrange(len(src_files)),
                        rng2.randrange(len(tgt_files)),
                    ))

        rng2.shuffle(self.pairs)

        print(f"SpeakerDataset [{split}]: {len(self.spk_names)} speakers, "
              f"{len(self.pairs)} pairs")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load(self, path: Path) -> torch.Tensor:
        """Load audio file → mono float32 tensor [T] @ 16 kHz, trimmed to max_samples."""
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)                    # [C, T]
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)               # [1, T]
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.squeeze(0)                              # [T]
        if wav.shape[0] > self.max_samples:
            start = random.randint(0, wav.shape[0] - self.max_samples)
            wav = wav[start: start + self.max_samples]
        return wav

    @staticmethod
    def _mel(audio: torch.Tensor) -> torch.Tensor:
        """[T] → [N_MELS, T_mel] log-compressed mel at 24000 Hz (on CPU)."""
        audio_24k = torchaudio.functional.resample(audio.unsqueeze(0), SAMPLE_RATE, MEL_SAMPLE_RATE).squeeze(0)
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=MEL_SAMPLE_RATE, n_fft=1024, hop_length=256, win_length=1024, n_mels=N_MELS,
        )
        mel = transform(audio_24k.unsqueeze(0)).squeeze(0)   # [N_MELS, T_mel]
        return torch.log(mel.clamp(min=1e-5))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        src_spk, tgt_spk, src_idx, tgt_idx = self.pairs[idx]

        source_audio = self._load(self.speakers[src_spk][src_idx])
        target_audio = self._load(self.speakers[tgt_spk][tgt_idx])

        # N_CTX reference utterances from target speaker (excluding tgt_idx)
        tgt_files = self.speakers[tgt_spk]
        ctx_indices = [i for i in range(len(tgt_files)) if i != tgt_idx]
        ctx_indices = random.sample(ctx_indices, min(self.n_ctx, len(ctx_indices)))
        max_mel_frames = int(self.max_samples * (MEL_SAMPLE_RATE / SAMPLE_RATE) / 256)

        mels = []
        mel_lens = []
        for i in ctx_indices:
            cache = tgt_files[i].with_suffix(".pt")
            if cache.exists():
                mel = torch.load(cache, weights_only=True).float()
                if mel.shape[-1] > max_mel_frames:
                    start = random.randint(0, mel.shape[-1] - max_mel_frames)
                    mel = mel[:, start : start + max_mel_frames]
            else:
                mel = self._mel(self._load(tgt_files[i]))
            mel_lens.append(mel.shape[-1])  # unpadded T for this mel
            mels.append(mel)
        max_T = max(m.shape[-1] for m in mels)
        context_mels = torch.stack([
            F.pad(m, (0, max_T - m.shape[-1])) for m in mels
        ])   # [N_CTX, N_MELS, T_ctx]

        return {
            "source_audio": source_audio,
            "target_audio": target_audio,
            "context_mels": context_mels,
            "ctx_mel_lens": mel_lens,   # list[N_CTX] of ints: unpadded T per ref mel
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length tensors to batch maximum."""

    def pad1d(tensors: list[torch.Tensor]) -> torch.Tensor:
        L = max(t.shape[0] for t in tensors)
        return torch.stack([F.pad(t, (0, L - t.shape[0])) for t in tensors])

    def pad_mels(mels: list[torch.Tensor]) -> torch.Tensor:
        # mels[i]: [N_CTX, N_MELS, T_i]
        L = max(m.shape[-1] for m in mels)
        return torch.stack([F.pad(m, (0, L - m.shape[-1])) for m in mels])

    return {
        "source_audio": pad1d([b["source_audio"] for b in batch]),
        "target_audio": pad1d([b["target_audio"] for b in batch]),
        "context_mels": pad_mels([b["context_mels"] for b in batch]),
        "target_lengths": [b["target_audio"].shape[0] for b in batch],
        # list[B] of list[N_CTX]: unpadded mel-frame count per reference utterance.
        # Used to build key_padding_mask for cross-attention (True = padding position).
        "ctx_mel_lens": [b["ctx_mel_lens"] for b in batch],
    }
