"""
Offline voice conversion — convert a WAV file using a trained checkpoint.

Usage:
    python convert.py \
        --source  my_voice.wav \
        --reference  target_speaker.wav \
        --checkpoint checkpoints/best.pt \
        --output  converted.wav

You can pass multiple --reference files to get a more robust speaker embedding:
    python convert.py \
        --source my_voice.wav \
        --reference ref1.wav ref2.wav ref3.wav \
        --checkpoint checkpoints/best.pt \
        --output converted.wav
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.model import VoiceConversionModel

SAMPLE_RATE = 16_000
VOCODER_SR = 24_000          # vocos output sample rate
N_MELS = 100
CHUNK_SAMPLES = 16_000 * 4   # process 4-second chunks (content encoder at 16 kHz)


def load_audio(path: str, device: torch.device) -> torch.Tensor:
    """Load any audio file → mono float32 [1, T] @ 16 kHz on device."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)      # [1, T]
    if sr != SAMPLE_RATE:
        wav = AF.resample(wav, sr, SAMPLE_RATE)
    return wav.to(device)                    # [1, T]


def audio_to_mel(audio: torch.Tensor, device: torch.device) -> torch.Tensor:
    """[1, T @ 16kHz] → [1, N_MELS, T_mel] log mel at 24000 Hz, on device."""
    audio_24k = AF.resample(audio, SAMPLE_RATE, VOCODER_SR)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=VOCODER_SR, n_fft=1024, hop_length=256, win_length=1024, n_mels=N_MELS,
    ).to(device)
    mel = mel_transform(audio_24k)           # [1, N_MELS, T_mel]
    return torch.log(mel.clamp(min=1e-5))


def convert(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model …")
    model = VoiceConversionModel(device=device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Checkpoint loaded: {args.checkpoint}")

    # ── Compute context vector C from reference audio(s) ─────────────────────
    print(f"Computing speaker context from {len(args.reference)} reference file(s) …")
    ref_mels = []
    for ref_path in args.reference:
        ref_audio = load_audio(ref_path, device)   # [1, T]
        ref_mel = audio_to_mel(ref_audio, device)  # [1, N_MELS, T_mel]
        ref_mels.append(ref_mel)

    C = model.compute_context(ref_mels)            # [1, 256]
    print(f"Context vector computed: {C.shape}")

    # ── Load source audio ─────────────────────────────────────────────────────
    print(f"Loading source: {args.source}")
    source = load_audio(args.source, device)       # [1, T_total]
    T_total = source.shape[-1]
    print(f"Source duration: {T_total / SAMPLE_RATE:.2f}s ({T_total} samples)")

    # Reset DFN state once for the full conversion pass
    model.content_encoder.reset_dfn_state(batch_size=1)

    # ── Process in chunks ─────────────────────────────────────────────────────
    output_chunks = []
    step = CHUNK_SAMPLES
    for start in range(0, T_total, step):
        end = min(start + step, T_total)
        chunk = source[:, start:end]               # [1, chunk_len]

        # Pad last chunk if shorter than expected (HuBERT needs ≥ 1 frame)
        if chunk.shape[-1] < 400:
            print(f"Skipping final {chunk.shape[-1]}-sample chunk (too short).")
            break

        wav_out = model.convert_chunk(chunk, C)    # [1, 1, T_out]
        output_chunks.append(wav_out.squeeze(0))   # [1, T_out]

        pct = min(end, T_total) / T_total * 100
        print(f"  {pct:.0f}%  [{start}:{end}] → {wav_out.shape[-1]} samples out")

    # ── Concatenate and save ──────────────────────────────────────────────────
    if not output_chunks:
        print("No output produced — source audio may be too short.")
        return

    output = torch.cat(output_chunks, dim=-1)      # [1, T_total_out]
    output = output.cpu().clamp(-1.0, 1.0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), output[0].numpy(), VOCODER_SR)
    print(f"\nSaved: {out_path}  ({output.shape[-1] / VOCODER_SR:.2f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline voice conversion")
    parser.add_argument(
        "--source", required=True,
        help="Source audio file to convert (WAV/FLAC/MP3)",
    )
    parser.add_argument(
        "--reference", required=True, nargs="+",
        help="One or more reference audio files from the target speaker",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best.pt",
        help="Path to trained checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--output", default="converted.wav",
        help="Output WAV file path (default: converted.wav)",
    )
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
