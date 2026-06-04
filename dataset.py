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
Minimum utterances per speaker: n_ctx + 1  (auto-derived from n_ctx)
"""

import random
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000
MEL_SAMPLE_RATE = (
    24_000  # resample to this before computing mel (matches vocos-mel-24khz)
)
N_MELS = 100
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


class SpeakerDataset(Dataset):
    """
    Builds random cross-speaker pairs from a folder of speaker sub-directories.

    Each __getitem__ returns:
        source_audio   [T_src]             source speaker waveform @ 16 kHz
        audio_content  [T_content]          target speaker waveform @ 16 kHz (content to reconstruct)
        context_mels   [N_CTX, N_MELS, T]  reference mels from target speaker (zero-padded)
        ctx_mel_lens   list[N_CTX]         unpadded T length of each reference mel
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        max_sec: float = 8.0,
        n_ctx: int = 2,
        min_utts: int | None = None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.max_samples = int(max_sec * SAMPLE_RATE)
        self.n_ctx = n_ctx
        if min_utts is None:
            min_utts = n_ctx + 1  # 1 content clip + n_ctx context clips

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
            chosen = spk_list[cut:] or spk_list[-1:]  # at least 1 val speaker

        self.speakers: dict[str, list[Path]] = {s: speakers[s] for s in chosen}
        self.spk_names = list(self.speakers.keys())

        # Build index: each entry is (spk, utt_idx).
        # source = augmented version of utt_idx; audio_content = clean version.
        N_ITEMS = 200  # items per speaker (cap); speakers with fewer files use all without duplication
        self.pairs: list[tuple[str, int]] = []
        rng2 = random.Random(seed + 1)
        for spk in self.spk_names:
            files = self.speakers[spk]
            n = len(files)
            if n >= N_ITEMS:
                indices = rng2.sample(range(n), N_ITEMS)
            else:
                indices = list(range(n))
            for idx in indices:
                self.pairs.append((spk, idx))

        rng2.shuffle(self.pairs)

        print(
            f"SpeakerDataset [{split}]: {len(self.spk_names)} speakers, "
            f"{len(self.pairs)} items"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load(self, path: Path) -> torch.Tensor:
        """Load audio file → mono float32 tensor [T] @ 16 kHz, trimmed to max_samples."""
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)  # [C, T]
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)  # [1, T]
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.squeeze(0)  # [T]
        if wav.shape[0] > self.max_samples:
            start = random.randint(0, wav.shape[0] - self.max_samples)
            wav = wav[start : start + self.max_samples]
        return wav

    def _load_aug(self, path: Path) -> torch.Tensor | None:
        """Load full augmented audio tensor (16 kHz) from _aug.pt, untrimed.
        Returns None if the augmented file is missing."""
        aug_pt = path.with_name(f"{path.stem}_aug.pt")
        if aug_pt.exists():
            return torch.load(aug_pt, weights_only=True)
        return None

    @staticmethod
    def _mel(audio: torch.Tensor) -> torch.Tensor:
        """[T] → [N_MELS, T_mel] log-compressed mel at 24000 Hz (on CPU)."""
        audio_24k = torchaudio.functional.resample(
            audio.unsqueeze(0), SAMPLE_RATE, MEL_SAMPLE_RATE
        ).squeeze(0)
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=MEL_SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=N_MELS,
            power=1.0,
            center=True,
        )
        mel = transform(audio_24k.unsqueeze(0)).squeeze(0)  # [N_MELS, T_mel]
        return torch.log(mel.clamp(min=1e-7))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        spk, utt_idx = self.pairs[idx]
        spk_files = self.speakers[spk]

        # Load clean audio (full, untrimed) and pick one shared start offset.
        path = spk_files[utt_idx]
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.squeeze(0)  # [T] @ 16 kHz

        start = random.randint(0, max(0, wav.shape[0] - self.max_samples))
        audio_content = wav[start : start + self.max_samples]

        # Apply the same start to the augmented tensor so source and content
        # cover the same portion of the utterance.
        aug_wav = self._load_aug(path)
        if aug_wav is not None:
            aug_start = min(start, max(0, aug_wav.shape[0] - self.max_samples))
            source_audio = aug_wav[aug_start : aug_start + self.max_samples]
        else:
            source_audio = audio_content  # fallback: clean

        # N_CTX clean reference utterances from the same speaker (excluding utt_idx).
        ctx_indices = [i for i in range(len(spk_files)) if i != utt_idx]
        ctx_indices = random.sample(ctx_indices, min(self.n_ctx, len(ctx_indices)))
        max_mel_frames = int(self.max_samples * (MEL_SAMPLE_RATE / SAMPLE_RATE) / 256)

        mels = []
        mel_lens = []
        ctx_audios_list = []
        ctx_audio_lens = []
        for i in ctx_indices:
            ctx_audio = self._load(spk_files[i])  # [T] @ 16kHz
            ctx_audio_lens.append(ctx_audio.shape[0])
            ctx_audios_list.append(ctx_audio)

            cache = spk_files[i].with_suffix(".pt")
            if cache.exists():
                mel = torch.load(cache, weights_only=True).float()
                if mel.shape[-1] > max_mel_frames:
                    start = random.randint(0, mel.shape[-1] - max_mel_frames)
                    mel = mel[:, start : start + max_mel_frames]
            else:
                mel = self._mel(ctx_audio)
            mel_lens.append(mel.shape[-1])
            mels.append(mel)

        max_T = max(m.shape[-1] for m in mels)
        context_mels = torch.stack(
            [F.pad(m, (0, max_T - m.shape[-1])) for m in mels]
        )  # [N_CTX, N_MELS, T_ctx]

        max_T_audio = max(a.shape[0] for a in ctx_audios_list)
        context_audios = torch.stack(
            [F.pad(a, (0, max_T_audio - a.shape[0])) for a in ctx_audios_list]
        )  # [N_CTX, T_max_audio]

        return {
            "source_audio": source_audio,
            "audio_content": audio_content,
            "context_mels": context_mels,
            "ctx_mel_lens": mel_lens,  # list[N_CTX] unpadded mel-frame counts (for visualization)
            "context_audios": context_audios,  # [N_CTX, T_max_audio] @ 16kHz
            "ctx_audio_lens": ctx_audio_lens,  # list[N_CTX] unpadded sample counts
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

    def pad_ctx_audios(audios: list[torch.Tensor]) -> torch.Tensor:
        # audios[i]: [N_CTX, T_i]
        L = max(a.shape[-1] for a in audios)
        return torch.stack([F.pad(a, (0, L - a.shape[-1])) for a in audios])

    return {
        "source_audio": pad1d([b["source_audio"] for b in batch]),
        "audio_content": pad1d([b["audio_content"] for b in batch]),
        "context_mels": pad_mels([b["context_mels"] for b in batch]),
        "context_audios": pad_ctx_audios(
            [b["context_audios"] for b in batch]
        ),  # [B, N_CTX, T]
        "source_lengths": [b["source_audio"].shape[0] for b in batch],
        "content_lengths": [b["audio_content"].shape[0] for b in batch],
        "ctx_mel_lens": [b["ctx_mel_lens"] for b in batch],  # for visualization
        "ctx_audio_lens": [b["ctx_audio_lens"] for b in batch],  # for mask construction
    }
