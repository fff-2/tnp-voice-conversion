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
import math
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
SAVE_EVERY = 1
LOG_EVERY = 1
CSV_LOG_EVERY = 1
WARMUP_STEPS = 1_000
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_AUDIO_SEC = 8.0  # longer clips → more HuBERT activations → more VRAM
N_CTX = 5  # number of context utterances per training sample
N_MELS = 100

# ── KL annealing schedule ──────────────────────────────────────────────────────
# beta=0 for the first KL_ANNEAL_START steps (pure reconstruction warmup),
# then linearly ramps to KL_BETA_MAX by KL_ANNEAL_END.  Prevents posterior collapse.
KL_ANNEAL_START = 5_000
KL_ANNEAL_END = 20_000
KL_BETA_MAX = 0.01

HUBERT_NOISE_STD = (
    0.01  # std of additive Gaussian noise on HuBERT features (training only)
)


# ── LR schedule: linear warmup → cosine decay ────────────────────────────────


def get_lr(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def get_beta(step: int) -> float:
    """Linear KL annealing: 0 → KL_BETA_MAX over [KL_ANNEAL_START, KL_ANNEAL_END]."""
    if step < KL_ANNEAL_START:
        return 0.0
    if step >= KL_ANNEAL_END:
        return KL_BETA_MAX
    # return KL_BETA_MAX * (step - KL_ANNEAL_START) / (KL_ANNEAL_END - KL_ANNEAL_START)
    return 0.0001


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
            csv.writer(f).writerow(
                ["step", "train_loss", "val_loss", "learning_rate", "kl_loss", "beta"]
            )

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

    # ── Mel transform on GPU for target mel computation ───────────────────────
    # Resample 16 kHz audio to 24000 Hz before mel (matches vocos-mel-24khz params)
    mel_resampler = torchaudio.transforms.Resample(SAMPLE_RATE, VOCODER_SR).to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=VOCODER_SR,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=N_MELS,
        power=1.0,
        center=True,
    ).to(device)

    def compute_target_mel(audio: torch.Tensor) -> torch.Tensor:
        """[B, T @ 16kHz] → [B, T_mel, N_MELS] log-compressed, channels-last."""
        mel = mel_transform(mel_resampler(audio))  # [B, N_MELS, T_mel]
        mel = mel.transpose(1, 2)  # [B, T_mel, N_MELS]
        return torch.log(mel.clamp(min=1e-7))

    # ── Training loop ─────────────────────────────────────────────────────────
    optimizer.zero_grad()
    running_loss = 0.0
    running_kl = 0.0
    accum_count = 0

    while step < MAX_STEPS:
        for batch in train_loader:
            if step >= MAX_STEPS:
                break

            source_audio = batch["source_audio"].to(device)  # [B, T] augmented
            content_audio = batch["audio_content"].to(device)  # [B, T] clean target
            ctx_mels = batch["context_mels"].to(device)  # [B, N_CTX, N_MELS, T_ctx]
            content_lengths = batch[
                "content_lengths"
            ]  # list[int], unpadded sample counts
            ctx_mel_lens = batch["ctx_mel_lens"]  # list[B] of list[N_CTX] ints

            B, N, M, T_ctx = ctx_mels.shape
            # Flatten context batch for context encoder
            ctx_flat = ctx_mels.view(B * N, M, T_ctx)  # [B*N, N_MELS, T_ctx]

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                # ── Context encoding ──────────────────────────────────────────
                # Self-attention padding mask for the context encoder Transformer.
                # Shape [B*N, T_ctx]: flat index i*N+n covers reference mel n of
                # batch item i; positions beyond ctx_mel_lens[i][n] are zero-padding.
                ctx_enc_mask = torch.ones(B * N, T_ctx, dtype=torch.bool, device=device)
                for i in range(B):
                    for n in range(N):
                        valid = ctx_mel_lens[i][n]
                        ctx_enc_mask[i * N + n, :valid] = False

                C_all, mu_all, lv_all = model.context_encoder(
                    ctx_flat, src_key_padding_mask=ctx_enc_mask
                )  # each [B*N, T_ctx, d_model]
                T_ctx_enc = C_all.shape[1]  # == T_ctx (no temporal stride)
                # Concatenate N reference sequences along the time axis (TNP style),
                # matching encode_references() used at inference.
                C = C_all.view(B, N * T_ctx_enc, -1)  # [B, N*T_ctx, d_model]
                mu_C = mu_all.view(B, N * T_ctx_enc, -1)  # [B, N*T_ctx, d_model]
                lv_C = lv_all.view(B, N * T_ctx_enc, -1)  # [B, N*T_ctx, d_model]

                # ── Cross-attention padding mask ──────────────────────────────
                # True = position is zero-padding; prevents it from polluting attention.
                # Each item i has N mels; mel n occupies positions [n*T_ctx_enc, (n+1)*T_ctx_enc).
                # Valid frames = ctx_mel_lens[i][n]; remainder is zero-padding.
                ctx_mask = torch.ones(B, N * T_ctx_enc, dtype=torch.bool, device=device)
                for i in range(B):
                    for n in range(N):
                        valid = ctx_mel_lens[i][n]
                        ctx_mask[i, n * T_ctx_enc : n * T_ctx_enc + valid] = False

                # ── KL divergence: q(z|C) vs N(0,I) ──────────────────────────
                # Compute in float32 to avoid bfloat16 precision loss in exp().
                # Masked so that zero-padded context positions are excluded.
                mu_f = mu_C.float()
                lv_f = lv_C.float()
                kl_raw = -0.5 * (
                    1.0 + lv_f - mu_f.pow(2) - lv_f.exp()
                )  # [B, N*T_ctx, d_model]
                valid_ctx = (~ctx_mask).float()  # [B, N*T_ctx]
                kl_loss = (kl_raw * valid_ctx.unsqueeze(-1)).sum() / (
                    valid_ctx.sum() * model.D_MODEL + 1e-8
                )

                # ── Content encoding (frozen, no grad) ────────────────────────
                # source_audio is the Parselmouth-augmented version of the target
                # utterance.  The model must rely on clean context C to recover
                # speaker identity rather than copying it from content features.
                with torch.no_grad():
                    content = model.content_encoder(source_audio)  # [B, T_frames, 769]

                """  # ── HuBERT Gaussia[n noise (information bottleneck) ────────────
                # Additive noise perturbs features without zeroing entire channels.
                # F0 (last dim) is left intact so pitch correction is preserved.
                hubert_feat = content[..., :768]  # [B, T_frames, 768]
                f0_feat = content[..., 768:]  # [B, T_frames, 1]
                if model.training:
                    hubert_feat = (
                        hubert_feat + torch.randn_like(hubert_feat) * HUBERT_NOISE_STD
                    )
                content = torch.cat(
                    [hubert_feat, f0_feat], dim=-1
                )  # [B, T_frames, 769] """

                # ── Cross-attention + decode ──────────────────────────────────
                fused = model.cross_attention(
                    content, C, key_padding_mask=ctx_mask
                )  # [B, T_frames, d_model]
                pred_mel = model.decoder(fused)  # [B, T_mel_pred, N_MELS]

                # ── Target mel (always from unperturbed content_audio) ────────
                with torch.no_grad():
                    tgt_mel = compute_target_mel(
                        content_audio
                    )  # [B, T_mel_tgt, N_MELS]

                # ── Masked L1 loss (ignore padded frames) ─────────────────────
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
                beta = get_beta(step)
                loss = (recon_loss + beta * kl_loss) / GRAD_ACCUM

            loss.backward()
            running_loss += recon_loss.item()
            running_kl += kl_loss.item()
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
                    avg_recon = running_loss / (LOG_EVERY * GRAD_ACCUM)
                    avg_kl = running_kl / (LOG_EVERY * GRAD_ACCUM)
                    running_loss = 0.0
                    running_kl = 0.0
                    cur_beta = get_beta(step)
                    logger.info(
                        f"step={step:6d}  recon={avg_recon:.4f}"
                        f"  kl={avg_kl:.4f}  beta={cur_beta:.4f}  lr={lr:.2e}"
                    )

                    if step % CSV_LOG_EVERY == 0:
                        with open(csv_path, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [step, avg_recon, last_val_loss, lr, avg_kl, cur_beta]
                            )

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
    num_batches = 0
    samples_saved = False

    for batch in loader:
        source = batch["source_audio"].to(device)
        content_audio = batch["audio_content"].to(device)
        ctx_mels = batch["context_mels"].to(device)

        source_lengths = batch["source_lengths"]
        content_lengths = batch["content_lengths"]
        ctx_mel_lens = batch["ctx_mel_lens"]  # list[B] of list[N_CTX] ints
        B, N, M, T_ctx = ctx_mels.shape
        ctx_flat = ctx_mels.view(B * N, M, T_ctx)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            ctx_enc_mask = torch.ones(B * N, T_ctx, dtype=torch.bool, device=device)
            for i in range(B):
                for n in range(N):
                    valid = ctx_mel_lens[i][n]
                    ctx_enc_mask[i * N + n, :valid] = False

            C_all, _, _ = model.context_encoder(
                ctx_flat, src_key_padding_mask=ctx_enc_mask
            )  # [B*N, T_ctx, d_model]  (mu/lv not needed at eval)
            T_ctx_enc = C_all.shape[1]
            C = C_all.view(B, N * T_ctx_enc, -1)  # [B, N*T_ctx, d_model]

            ctx_mask = torch.ones(B, N * T_ctx_enc, dtype=torch.bool, device=device)
            for i in range(B):
                for n in range(N):
                    valid = ctx_mel_lens[i][n]
                    ctx_mask[i, n * T_ctx_enc : n * T_ctx_enc + valid] = False

            content = model.content_encoder(content_audio)
            fused = model.cross_attention(content, C, key_padding_mask=ctx_mask)
            pred_mel = model.decoder(fused)
            tgt_mel = mel_transform(mel_resampler(content_audio)).transpose(1, 2)
            tgt_mel = torch.log(tgt_mel.clamp(min=1e-7))
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

            # Raw waveforms at 16 kHz (first item of first batch, unpadded)
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
                src_voiced = src_f0[0, :, 0].cpu().numpy()
                tgt_voiced = tgt_f0[0, :, 0].cpu().numpy()
                src_v = src_voiced[src_voiced > 0.0]
                tgt_v = tgt_voiced[tgt_voiced > 0.0]
                if len(src_v) > 1 and len(tgt_v) > 1:
                    sample_f0_stats = (
                        float(src_v.mean()),
                        float(max(src_v.std(), 5.0)),
                        float(tgt_v.mean()),
                        float(max(tgt_v.std(), 5.0)),
                    )

            # Converted: source content + target speaker C → vocoder (24000 Hz out)
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                src_content = model.content_encoder(
                    source[0:1, :src_len], f0_stats=sample_f0_stats
                )
                src_fused = model.cross_attention(
                    src_content, C[0:1], key_padding_mask=ctx_mask[0:1]
                )
                src_mel = model.decoder(src_fused)
                wav = model.vocoder(src_mel.transpose(1, 2))  # [1, 1, T_wav]
            sf.write(
                str(sample_dir / "converted.wav"),
                wav[0, 0, :].cpu().numpy(),
                VOCODER_SR,
            )

            # context.wav: first reference clip decoded through Vocos
            ctx_len_frames = ctx_mel_lens[0][0]
            ctx_mel_sample = ctx_mels[0, 0, :, :ctx_len_frames].unsqueeze(0).float()
            ctx_wav = model.vocoder(ctx_mel_sample)  # [1, 1, T_wav]
            sf.write(
                str(sample_dir / "context.wav"),
                ctx_wav[0, 0, :].cpu().numpy(),
                VOCODER_SR,
            )

        if num_batches >= 50:  # cap validation batches for speed
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
