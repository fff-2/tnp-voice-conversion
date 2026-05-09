"""
Training script for the voice conversion pipeline.

Trains the three trainable modules (ContextEncoder, CrossAttentionFusion, MelDecoder)
against frozen HuBERT content encoder and Vocos vocoder.

VRAM optimizations:
    - AMP (torch.amp.autocast) for bf16 forward/backward
    - Gradient accumulation (physical batch=40, accumulate=2 → effective batch=80)

Dataset layout (place datasets inside the datasets/ folder):
    VCTK:       datasets/wav48_silence_trimmed/
    LibriSpeech: datasets/LibriSpeech/train-clean-100/
    Custom:     datasets/<any-name>/<speaker>/clip.wav

Usage:
    python train.py --data-root datasets/wav48_silence_trimmed
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from loguru import logger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.model import VoiceConversionModel
from dataset import SpeakerDataset, collate_fn

# ── Hyperparameters ───────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
VOCODER_SR = 24_000  # mel computation and vocoder output sample rate
BATCH_SIZE = 32  # physical batch per GPU step — increase to fill VRAM
GRAD_ACCUM = 2  # effective batch = BATCH_SIZE * GRAD_ACCUM = 32
MAX_STEPS = 100_000
SAVE_EVERY = 1000
LOG_EVERY = 50
CSV_LOG_EVERY = 50
WARMUP_STEPS = 1_000
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_AUDIO_SEC = 8.0  # longer clips → more HuBERT activations → more VRAM
N_CTX = 5  # number of context utterances per training sample
N_MELS = 100


# ── LR schedule: linear warmup → cosine decay ────────────────────────────────


def get_lr(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


# ── Training loop ─────────────────────────────────────────────────────────────


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VoiceConversionModel(device=device)
    model.train()
    logger.info(f"Trainable parameters: {model.trainable_param_count():,}")

    # ── Optimizer & AMP ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),
    )
    # ── Checkpoint resume ─────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    best_loss = float("inf")
    last_val_loss = float("inf")

    csv_path = output_dir / args.csv_log
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "learning_rate"])

    ckpt_path = output_dir / "latest.pt"
    if ckpt_path.exists() and not args.reset:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        best_loss = ckpt.get("best_loss", best_loss)
        logger.info(f"Resumed from step {step}")

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    train_ds = SpeakerDataset(args.data_root, split="train", max_sec=MAX_AUDIO_SEC)
    val_ds = SpeakerDataset(args.data_root, split="val", max_sec=MAX_AUDIO_SEC)
    val_subset = torch.utils.data.Subset(val_ds, range(50))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ── Mel transform on GPU for target mel computation ───────────────────────
    # Resample 16 kHz audio to 24000 Hz before mel (matches vocos-mel-24khz params)
    mel_resampler = torchaudio.transforms.Resample(SAMPLE_RATE, VOCODER_SR).to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=VOCODER_SR,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=N_MELS,
    ).to(device)

    def compute_target_mel(audio: torch.Tensor) -> torch.Tensor:
        """[B, T @ 16kHz] → [B, T_mel, N_MELS] log-compressed, channels-last."""
        mel = mel_transform(mel_resampler(audio))  # [B, N_MELS, T_mel]
        mel = mel.transpose(1, 2)  # [B, T_mel, N_MELS]
        return torch.log(mel.clamp(min=1e-5))

    # ── Training loop ─────────────────────────────────────────────────────────
    optimizer.zero_grad()
    running_loss = 0.0
    accum_count = 0

    while step < MAX_STEPS:
        for batch in train_loader:
            if step >= MAX_STEPS:
                break

            source = batch["source_audio"].to(device)  # [B, T_src]
            target = batch["target_audio"].to(device)  # [B, T_tgt]
            ctx_mels = batch["context_mels"].to(device)  # [B, N_CTX, N_MELS, T_ctx]

            B, N, M, T_ctx = ctx_mels.shape
            # Flatten context batch for context encoder
            ctx_flat = ctx_mels.view(B * N, M, T_ctx)  # [B*N, N_MELS, T_ctx]

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                # ── Context encoding ──────────────────────────────────────────
                C_all = model.context_encoder(ctx_flat)  # [B*N, D_MODEL]
                C = C_all.view(B, N, -1).mean(dim=1)  # [B, D_MODEL]

                # ── Content encoding (frozen, no grad) ────────────────────────
                with torch.no_grad():
                    content = model.content_encoder(target)  # [B, T_frames, 769]

                # ── Cross-attention + decode ──────────────────────────────────
                fused = model.cross_attention(content, C)  # [B, T_frames, D_MODEL]
                pred_mel = model.decoder(fused)  # [B, T_mel_pred, N_MELS]

                # ── Target mel ────────────────────────────────────────────────
                with torch.no_grad():
                    tgt_mel = compute_target_mel(target)  # [B, T_mel_tgt, N_MELS]

                # ── Align lengths & compute loss ──────────────────────────────
                T = min(pred_mel.shape[1], tgt_mel.shape[1])
                loss = F.l1_loss(pred_mel[:, :T, :], tgt_mel[:, :T, :])
                loss = loss / GRAD_ACCUM

            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM
            accum_count += 1

            # ── Optimizer step every GRAD_ACCUM mini-batches ─────────────────
            if accum_count % GRAD_ACCUM == 0:
                # Update LR
                lr = get_lr(step, WARMUP_STEPS, MAX_STEPS, LR)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_params(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                # ── Logging ───────────────────────────────────────────────────
                if step % LOG_EVERY == 0:
                    avg = running_loss / (LOG_EVERY * GRAD_ACCUM)
                    running_loss = 0.0
                    logger.info(f"step={step:6d}  loss={avg:.4f}  lr={lr:.2e}")

                    if step % CSV_LOG_EVERY == 0:
                        with open(csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([step, avg, last_val_loss, lr])

                # ── Checkpointing ─────────────────────────────────────────────
                if step % SAVE_EVERY == 0:
                    avg_loss = _validate(
                        model,
                        val_loader,
                        device,
                        mel_transform,
                        mel_resampler,
                        step=step,
                        output_dir=output_dir,
                    )
                    last_val_loss = avg_loss
                    logger.info(f"Validation loss @ step {step}: {avg_loss:.4f}")

                    ckpt = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "best_loss": best_loss,
                    }
                    torch.save(ckpt, output_dir / "latest.pt")

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(ckpt, output_dir / "best.pt")
                        logger.info(f"New best model saved (loss={best_loss:.4f})")

                    model.train()  # restore training mode after validation

    logger.info(f"Training complete. Best validation loss: {best_loss:.4f}")


@torch.no_grad()
def _validate(
    model: VoiceConversionModel,
    loader: DataLoader,
    device: torch.device,
    mel_transform,
    mel_resampler,
    step: int = 0,
    output_dir: Path = None,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    samples_saved = False

    for batch in loader:
        source = batch["source_audio"].to(device)
        target = batch["target_audio"].to(device)
        ctx_mels = batch["context_mels"].to(device)

        B, N, M, T_ctx = ctx_mels.shape
        ctx_flat = ctx_mels.view(B * N, M, T_ctx)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            C_all = model.context_encoder(ctx_flat)
            C = C_all.view(B, N, -1).mean(dim=1)
            content = model.content_encoder(target)
            fused = model.cross_attention(content, C)
            pred_mel = model.decoder(fused)
            tgt_mel = mel_transform(mel_resampler(target)).transpose(1, 2)
            tgt_mel = torch.log(tgt_mel.clamp(min=1e-5))
            T = min(pred_mel.shape[1], tgt_mel.shape[1])
            loss = F.l1_loss(pred_mel[:, :T, :], tgt_mel[:, :T, :])

        total_loss += loss.item()
        n += 1

        # ── Audio samples: first batch only ──────────────────────────────────
        if not samples_saved and output_dir is not None:
            samples_saved = True
            sample_dir = output_dir / "samples" / f"step_{step}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Raw waveforms at 16 kHz (first item of first batch)
            sf.write(
                str(sample_dir / "source.wav"), source[0, :].cpu().numpy(), SAMPLE_RATE
            )
            sf.write(
                str(sample_dir / "target.wav"), target[0, :].cpu().numpy(), SAMPLE_RATE
            )

            # Converted: source content + target speaker C → vocoder (24000 Hz out)
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                src_content = model.content_encoder(source[0:1])
                src_fused = model.cross_attention(src_content, C[0:1])
                src_mel = model.decoder(src_fused)
                wav = model.vocoder(src_mel.transpose(1, 2))  # [1, 1, T_wav]
            sf.write(
                str(sample_dir / "converted.wav"),
                wav[0, 0, :].cpu().numpy(),
                VOCODER_SR,
            )

        if n >= 50:  # cap validation batches for speed
            break

    return total_loss / max(1, n)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train voice conversion model")
    parser.add_argument(
        "--data-root",
        default="datasets/wav48_silence_trimmed",
        help="Speaker folder root (default: datasets/wav48_silence_trimmed)",
    )
    parser.add_argument(
        "--output-dir", default="checkpoints", help="Directory for saving checkpoints"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="DataLoader worker count"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing checkpoint and train from scratch",
    )
    parser.add_argument(
        "--csv-log",
        default="training_log.csv",
        help="CSV filename inside --output-dir for logging (default: training_log.csv)",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
