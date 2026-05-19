"""
Training script for the voice conversion pipeline.

Trains TNPUnifiedTransformer against frozen ContentEncoder (HuBERT/DFN3/crepe)
and Vocos vocoder.

Strict TNP-D: context (HuBERT+F0, mel) and target (HuBERT+F0) tokens are
concatenated into one sequence with a block attention mask — context sees only
context, each target token sees all context plus itself only.  No variational
bottleneck; optimised solely on masked L1 reconstruction loss.

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
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.model import VoiceConversionModel
from dataset import SpeakerDataset, collate_fn

# ── Hyperparameters ───────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
VOCODER_SR = 24_000  # mel computation and vocoder output sample rate
BATCH_SIZE = 32  # physical batch per GPU step — increase to fill VRAM
GRAD_ACCUM = 2  # effective batch = BATCH_SIZE * GRAD_ACCUM = 64
MAX_STEPS = 100_000
SAVE_EVERY = 2500
LOG_EVERY = 50
CSV_LOG_EVERY = 50
WARMUP_STEPS = 1_000
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_AUDIO_SEC = 8.0  # longer clips → more HuBERT activations → more VRAM
N_CTX = 1  # number of context utterances per training sample
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

    # ── Optimizer ─────────────────────────────────────────────────────────────
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
    train_ds = SpeakerDataset(
        args.data_root, split="train", max_sec=MAX_AUDIO_SEC, n_ctx=N_CTX
    )
    val_ds = SpeakerDataset(
        args.data_root, split="val", max_sec=MAX_AUDIO_SEC, n_ctx=N_CTX
    )
    val_subset = torch.utils.data.Subset(val_ds, range(3, 53))

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

    # ── Training loop ─────────────────────────────────────────────────────────
    optimizer.zero_grad()
    running_loss = 0.0
    accum_count = 0

    while step < MAX_STEPS:
        for batch in train_loader:
            if step >= MAX_STEPS:
                break

            source_audio = batch["source_audio"].to(device)  # [B, T] augmented
            content_audio = batch["audio_content"].to(device)  # [B, T] clean target
            ctx_audios = batch["context_audios"].to(device)  # [B, N, T_ctx]
            content_lengths = batch["content_lengths"]  # list[int]

            B = source_audio.shape[0]

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                # model.forward() handles all context/content extraction and masking
                pred_mel, tgt_mel = model(
                    source_audio,
                    ctx_audios,
                    content_audio,
                    ctx_audio_lens=batch["ctx_audio_lens"],
                    content_lengths=content_lengths,
                )

                # ── Masked L1 reconstruction loss ─────────────────────────────
                T = min(pred_mel.shape[1], tgt_mel.shape[1])
                mel_lengths = [
                    1 + math.ceil(n_samples * VOCODER_SR / SAMPLE_RATE) // 256
                    for n_samples in content_lengths
                ]
                mask = torch.zeros(B, T, device=device, dtype=torch.bool)
                for i, ml in enumerate(mel_lengths):
                    mask[i, : min(ml, T)] = True
                loss_raw = F.l1_loss(
                    pred_mel[:, :T, :], tgt_mel[:, :T, :], reduction="none"
                )
                recon_loss = (loss_raw * mask.unsqueeze(-1)).sum() / (
                    mask.sum() * N_MELS + 1e-8
                )
                loss = recon_loss / GRAD_ACCUM

            loss.backward()
            running_loss += recon_loss.item()
            accum_count += 1

            # ── Optimizer step every GRAD_ACCUM mini-batches ─────────────────
            if accum_count % GRAD_ACCUM == 0:
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
                    avg_recon = running_loss / (LOG_EVERY * GRAD_ACCUM)
                    running_loss = 0.0
                    logger.info(f"step={step:6d}  recon={avg_recon:.4f}  lr={lr:.2e}")

                    if step % CSV_LOG_EVERY == 0:
                        with open(csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([step, avg_recon, last_val_loss, lr])

                # ── Checkpointing ─────────────────────────────────────────────
                if step % SAVE_EVERY == 0:
                    avg_loss = _validate(
                        model,
                        val_loader,
                        device,
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
    step: int = 0,
    output_dir: Path = None,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    samples_saved = False

    for batch in loader:
        source = batch["source_audio"].to(device)
        content_audio = batch["audio_content"].to(device)
        ctx_audios = batch["context_audios"].to(device)  # [B, N, T_ctx]
        ctx_mels = batch["context_mels"].to(
            device
        )  # [B, N, N_MELS, T_mel] — visualization only

        source_lengths = batch["source_lengths"]
        content_lengths = batch["content_lengths"]
        ctx_mel_lens = batch["ctx_mel_lens"]

        B = content_audio.shape[0]

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            # Validation: use clean content_audio as both source and target
            pred_mel, tgt_mel = model(
                content_audio,
                ctx_audios,
                content_audio,
                ctx_audio_lens=batch["ctx_audio_lens"],
                content_lengths=content_lengths,
            )

            T = min(pred_mel.shape[1], tgt_mel.shape[1])
            mel_lengths = [
                1 + math.ceil(n_samples * VOCODER_SR / SAMPLE_RATE) // 256
                for n_samples in content_lengths
            ]
            mask = torch.zeros(B, T, device=device, dtype=torch.bool)
            for i, ml in enumerate(mel_lengths):
                mask[i, : min(ml, T)] = True
            loss_raw = F.l1_loss(
                pred_mel[:, :T, :], tgt_mel[:, :T, :], reduction="none"
            )
            loss = (loss_raw * mask.unsqueeze(-1)).sum() / (mask.sum() * N_MELS + 1e-8)

        total_loss += loss.item()
        num_batches += 1

        # ── Audio samples: first batch only ──────────────────────────────────
        if not samples_saved and output_dir is not None:
            samples_saved = True
            sample_dir = output_dir / "samples" / f"step_{step}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            src_len = source_lengths[0]
            tgt_len = content_lengths[0]
            sf.write(
                str(sample_dir / "source.wav"),
                source[0, :src_len].cpu().numpy(),
                SAMPLE_RATE,
            )
            sf.write(
                str(sample_dir / "target.wav"),
                content_audio[0, :tgt_len].cpu().numpy(),
                SAMPLE_RATE,
            )

            # F0 stats for pitch shifting (source speaker → target speaker range)
            sample_f0_stats = None
            if model.content_encoder._crepe_available:
                with torch.no_grad():
                    src_f0 = model.content_encoder._extract_f0(source[0:1, :src_len])
                    tgt_f0 = model.content_encoder._extract_f0(
                        content_audio[0:1, :tgt_len]
                    )
                src_v = src_f0[0, :, 0].cpu().numpy()
                tgt_v = tgt_f0[0, :, 0].cpu().numpy()
                src_v = src_v[src_v > 0.0]
                tgt_v = tgt_v[tgt_v > 0.0]
                if len(src_v) > 1 and len(tgt_v) > 1:
                    # Robust estimators: median + MAD-based std
                    # Prevents residual octave errors from inflating the spread.
                    def _robust(v):
                        c = float(np.median(v))
                        s = float(max(np.median(np.abs(v - c)) * 1.4826, 5.0))
                        return c, s
                    sc, ss = _robust(src_v)
                    tc, ts = _robust(tgt_v)
                    sample_f0_stats = (sc, ss, tc, ts)

            # Converted: source content + target speaker context → vocoder
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                # Build per-speaker context from raw reference audio
                N = ctx_audios.shape[1]
                ref_list = [
                    ctx_audios[0, n : n + 1, : batch["ctx_audio_lens"][0][n]]
                    for n in range(N)
                ]
                C_sample = model.compute_context(ref_list)  # [1, T_total, D_MODEL]

                model.content_encoder.reset_dfn_state(batch_size=1)
                wav = model.convert_chunk(
                    source[0:1, :src_len], C_sample, f0_stats=sample_f0_stats
                )
            sf.write(
                str(sample_dir / "converted.wav"),
                wav[0, 0, :].cpu().numpy(),
                model.VOCODER_SR,
            )

            # context.wav: first reference clip decoded through Vocos
            ctx_len_frames = ctx_mel_lens[0][0]
            ctx_mel_sample = ctx_mels[0, 0, :, :ctx_len_frames].unsqueeze(0).float()
            ctx_wav = model.vocoder(ctx_mel_sample)
            sf.write(
                str(sample_dir / "context.wav"),
                ctx_wav[0, 0, :].cpu().numpy(),
                model.VOCODER_SR,
            )

        if num_batches >= 50:
            break

    return total_loss / max(1, num_batches)


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
