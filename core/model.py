import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor

from core.modules.context_encoder import ContextEncoder
from core.modules.content_encoder import ContentEncoder
from core.modules.cross_attention import CrossAttentionFusion
from core.modules.decoder import MelDecoder
from core.vocoder import VocosVocoder


class VoiceConversionModel(nn.Module):
    """
    Full voice conversion pipeline — deterministic TNP-D style.

    Trainable:  ContextEncoder, CrossAttentionFusion, MelDecoder  (~5.8M params)
    Frozen:     ContentEncoder (DFN3 + HuBERT + torchcrepe), VocosVocoder

    Context set: N (HuBERT+F0, mel) pairs extracted from reference utterances of
    the target speaker.  The ContextEncoder learns the content→acoustic mapping
    rule for that speaker.  No stochastic sampling; no KL divergence loss.

    Content encoder operates at 16 kHz.
    Mel spectrogram (context encoder, decoder target, vocoder) operates at 24000 Hz.

    Inference:
        1. compute_context(reference_audios) → C  (once per speaker)
        2. convert_chunk(audio_chunk, C) → waveform at 24000 Hz  (per streaming chunk)
    """

    N_MELS = 100
    D_MODEL = 256
    HUBERT_DIM = 769           # 768 HuBERT + 1 F0
    CONTENT_SR = 16_000        # sample rate for content encoder (HuBERT / DFN3)
    VOCODER_SR = 24_000        # sample rate for mel computation and vocoder output

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Trainable modules
        self.context_encoder = ContextEncoder(
            hubert_dim=self.HUBERT_DIM,
            n_mels=self.N_MELS,
            d_model=self.D_MODEL,
            nhead=4,
            num_layers=4,
            dim_feedforward=1024,
        )
        self.cross_attention = CrossAttentionFusion(
            hubert_dim=768,
            f0_dim=1,
            d_model=self.D_MODEL,
            nhead=4,
        )
        self.decoder = MelDecoder(
            d_model=self.D_MODEL,
            n_mels=self.N_MELS,
            n_conv_blocks=3,
            n_transformer_layers=2,
        )

        # Frozen modules
        self.content_encoder = ContentEncoder(device=device)
        self.vocoder = VocosVocoder(device=device)

        # Resample 16 kHz audio → 24000 Hz for mel computation
        self.resampler = torchaudio.transforms.Resample(
            self.CONTENT_SR, self.VOCODER_SR
        ).to(device)

        # Mel transform matching vocos-mel-24khz training parameters
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.VOCODER_SR,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=self.N_MELS,
            power=1.0,
            center=True,
        ).to(device)

        self.to(device)

    # ── Ensure frozen modules stay in eval mode even when model.train() is called ──

    def train(self, mode: bool = True):
        super().train(mode)
        self.content_encoder.eval()
        for p in self.content_encoder.parameters():
            p.requires_grad_(False)
        self.vocoder.eval()
        for p in self.vocoder.parameters():
            p.requires_grad_(False)
        return self

    # ── Mel helpers ───────────────────────────────────────────────────────────

    def _compute_mel(self, audio: Tensor) -> Tensor:
        """[B, T @ 16kHz] → [B, T_mel, N_MELS] log-compressed, channels-last."""
        mel = self.mel_transform(self.resampler(audio))   # [B, N_MELS, T_mel]
        mel = mel.transpose(1, 2)                          # [B, T_mel, N_MELS]
        return torch.log(mel.clamp(min=1e-7))

    def _compute_mel_channels_first(self, audio: Tensor) -> Tensor:
        """[B, T @ 16kHz] → [B, N_MELS, T_mel] log-compressed, channels-first."""
        mel = self.mel_transform(self.resampler(audio))   # [B, N_MELS, T_mel]
        return torch.log(mel.clamp(min=1e-7))

    def _align_mel_to_content_rate(self, mel_cf: Tensor, T_target: int) -> Tensor:
        """Downsample mel [B, N_MELS, T_mel] → [B, T_target, N_MELS] at HuBERT rate."""
        mel_down = F.interpolate(
            mel_cf.float(), size=T_target, mode="linear", align_corners=False
        )                                                  # [B, N_MELS, T_target]
        return mel_down.transpose(1, 2)                    # [B, T_target, N_MELS]

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        source_audio: Tensor,                     # [B, T_src]     augmented, 16kHz
        context_audios: Tensor,                   # [B, N, T_ctx]  reference waveforms, 16kHz
        target_audio: Tensor,                     # [B, T_tgt]     clean, 16kHz
        ctx_audio_lens: list | None = None,       # list[B] of list[N] sample counts
        content_lengths: list | None = None,      # list[B] sample counts for source
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (pred_mel, target_mel) for reconstruction loss.
        No mu / log_var — fully deterministic (TNP-D).
        """
        B, N, T_ctx = context_audios.shape
        ctx_flat = context_audios.view(B * N, T_ctx)

        # ── Reference content + mel extraction (frozen, no grad) ──────────
        self.content_encoder.reset_dfn_state(batch_size=B * N)
        with torch.no_grad():
            ctx_content = self.content_encoder(
                ctx_flat, f0_audio_16k=ctx_flat
            )                                          # [B*N, T_h, 769]
            ctx_mel_cf = self._compute_mel_channels_first(ctx_flat)
            #                                            [B*N, N_MELS, T_mel]

        T_h = ctx_content.shape[1]
        ctx_mel_al = self._align_mel_to_content_rate(ctx_mel_cf, T_h)
        #                                              [B*N, T_h, N_MELS]
        ctx_pairs = torch.cat([ctx_content, ctx_mel_al], dim=-1)
        #                       [B*N, T_h, HUBERT_DIM + N_MELS]

        # ── Build padding masks from audio lengths ─────────────────────────
        # ctx_enc_mask:   [B*N, T_h]  — ContextEncoder self-attention
        # ctx_cross_mask: [B, N*T_h]  — CrossAttentionFusion key_padding_mask
        ctx_enc_mask   = torch.ones(B * N, T_h,     dtype=torch.bool, device=self.device)
        ctx_cross_mask = torch.ones(B,     N * T_h, dtype=torch.bool, device=self.device)
        if ctx_audio_lens is not None:
            for i in range(B):
                for n in range(N):
                    flen = min((ctx_audio_lens[i][n] // 320) + 1, T_h)
                    ctx_enc_mask[i * N + n, :flen]               = False
                    ctx_cross_mask[i, n * T_h : n * T_h + flen]  = False
        else:
            ctx_enc_mask.fill_(False)
            ctx_cross_mask.fill_(False)

        # ── ContextEncoder → C ────────────────────────────────────────────
        C_all = self.context_encoder(
            ctx_pairs, src_key_padding_mask=ctx_enc_mask
        )                                              # [B*N, T_h, D_MODEL]
        C = C_all.view(B, N * T_h, -1)               # [B, N*T_h, D_MODEL]

        # ── Source content extraction (frozen, no grad) ────────────────────
        self.content_encoder.reset_dfn_state(batch_size=B)
        with torch.no_grad():
            content = self.content_encoder(
                source_audio,
                f0_audio_16k=target_audio,
                lengths=content_lengths,
            )                                          # [B, T_frames, 769]

        fused = self.cross_attention(content, C, key_padding_mask=ctx_cross_mask)

        # ── Decoder with content padding mask ─────────────────────────────
        T_frames = fused.shape[1]
        content_mask = torch.zeros(B, T_frames, dtype=torch.bool, device=self.device)
        if content_lengths is not None:
            for i, L in enumerate(content_lengths):
                flen = (L // 320) + 1
                if flen < T_frames:
                    content_mask[i, flen:] = True

        pred_mel   = self.decoder(fused, key_padding_mask=content_mask)
        target_mel = self._compute_mel(target_audio)
        T = min(pred_mel.shape[1], target_mel.shape[1])
        return pred_mel[:, :T, :], target_mel[:, :T, :]

    # ── Inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_context(self, reference_audios: list) -> Tensor:
        """
        Encode a list of reference waveforms into a speaker context tensor.

        Args:
            reference_audios: list of N tensors, each [1, T_i] at 16kHz
        Returns:
            C: [1, T_total, D_MODEL]
        """
        self.eval()
        ctx_pairs_list = []
        for wav in reference_audios:
            self.content_encoder.reset_dfn_state(batch_size=1)
            content = self.content_encoder(wav, f0_audio_16k=wav)  # [1, T_h, 769]
            mel_cf  = self._compute_mel_channels_first(wav)         # [1, N_MELS, T_mel]
            T_h     = content.shape[1]
            mel_al  = self._align_mel_to_content_rate(mel_cf, T_h) # [1, T_h, N_MELS]
            pair    = torch.cat([content, mel_al], dim=-1)          # [1, T_h, 869]
            ctx_pairs_list.append(pair)
        return self.context_encoder.encode_references(ctx_pairs_list)

    @torch.no_grad()
    def convert_chunk(
        self,
        audio_chunk: Tensor,
        C: Tensor,
        ctx_mask: Tensor | None = None,
        f0_stats: tuple | None = None,
    ) -> Tensor:
        """
        Args:
            audio_chunk: [1, T]              16 kHz PCM float32
            C:           [1, T_ctx, D_MODEL] pre-cached context sequence
            ctx_mask:    [1, T_ctx]          bool, True = padding (optional)
            f0_stats:    (src_mean, src_std, tgt_mean, tgt_std) floats, or None
        Returns:
            waveform:    [1, 1, T_out]       24000 Hz float32
        """
        self.eval()
        content = self.content_encoder(audio_chunk, f0_stats=f0_stats)
        fused = self.cross_attention(content, C, key_padding_mask=ctx_mask)
        mel = self.decoder(fused)
        wav = self.vocoder(mel.transpose(1, 2))
        return wav

    @torch.no_grad()
    def convert_chunk_streaming(
        self,
        predenoised_audio: Tensor,
        C: Tensor,
        skip_head: int = 0,
        return_length: int | None = None,
        ctx_mask: Tensor | None = None,
        hubert_stats: tuple | None = None,
        f0_stats: tuple | None = None,
    ) -> Tensor:
        """
        Streaming variant of convert_chunk for real-time mic conversion.

        The caller feeds a ring buffer of recent denoised audio (~1.5 s) so that
        HuBERT sees enough temporal context.  Only the content frames from
        ``skip_head`` to ``skip_head + return_length`` are forwarded through the
        decoder and vocoder, producing audio for exactly the newest block.

        Args:
            predenoised_audio: [1, T]  denoised audio ring buffer at 16 kHz
            C:                 [1, T_ctx, D_MODEL]  pre-cached speaker context
            skip_head:         number of content frames to discard from front
            return_length:     number of content frames to keep (None = all after skip_head)
            ctx_mask:          optional padding mask for C
            hubert_stats:      (mean, std) each [1, 1, 768] — EMA stats for stable
                               streaming normalization
            f0_stats:          (src_mean, src_std, tgt_mean, tgt_std) floats
        Returns:
            waveform: [1, 1, T_out]  24000 Hz float32
        """
        self.eval()
        content = self.content_encoder(
            predenoised_audio,
            skip_denoise=True,
            hubert_stats=hubert_stats,
            f0_stats=f0_stats,
        )
        if return_length is not None:
            content = content[:, skip_head : skip_head + return_length, :]
        elif skip_head > 0:
            content = content[:, skip_head:, :]
        fused = self.cross_attention(content, C, key_padding_mask=ctx_mask)
        mel = self.decoder(fused)
        wav = self.vocoder(mel.transpose(1, 2))
        return wav

    @torch.no_grad()
    def convert_from_features(
        self,
        hubert_feat: Tensor,
        f0: Tensor,
        C: Tensor,
        skip_head: int = 0,
        return_length: int | None = None,
        ctx_mask: Tensor | None = None,
        hubert_stats: tuple | None = None,
        f0_stats: tuple | None = None,
    ) -> Tensor:
        """
        Convert using pre-extracted HuBERT and F0 features (zero-redundancy path).

        Args:
            hubert_feat:   [1, T_frames, 768]  raw HuBERT features (unnormalised)
            f0:            [1, T_frames, 1]    raw F0 in Hz (0.0 for unvoiced)
            C:             [1, T_ctx, D_MODEL]  pre-cached speaker context
            skip_head:     content frames to discard from front
            return_length: content frames to keep (None = all after skip_head)
            ctx_mask:      optional padding mask for C
            hubert_stats:  (mean, std) each [1, 1, 768] — EMA stats
            f0_stats:      (src_mean, src_std, tgt_mean, tgt_std) floats
        Returns:
            waveform: [1, 1, T_out]  24000 Hz float32
        """
        self.eval()

        # Align lengths (HuBERT and crepe may differ by 1 frame)
        T = min(hubert_feat.shape[1], f0.shape[1])
        hubert_feat = hubert_feat[:, :T, :]
        f0 = f0[:, :T, :]

        # Normalise HuBERT
        if hubert_stats is not None:
            hub_mean, hub_std = hubert_stats
            hubert_norm = (hubert_feat - hub_mean) / (hub_std + 1e-5)
        else:
            hubert_norm = F.instance_norm(
                hubert_feat.transpose(1, 2)
            ).transpose(1, 2)

        # F0 shifting
        if f0_stats is not None:
            src_mean, src_std, tgt_mean, tgt_std = f0_stats
            voiced_mask = (f0 > 0.0).float()
            f0_shifted = (f0 - src_mean) / (src_std + 1e-5) * tgt_std + tgt_mean
            f0 = voiced_mask * f0_shifted.clamp(min=50.0) + (1.0 - voiced_mask) * f0

        f0 = torch.log1p(f0)
        content = torch.cat([hubert_norm, f0], dim=-1)   # [1, T_all, 769]

        fused = self.cross_attention(content, C, key_padding_mask=ctx_mask)
        mel = self.decoder(fused)                         # [1, T_mel_all, N_MELS]

        if return_length is not None:
            T_content = content.shape[1]
            T_mel = mel.shape[1]
            scale = T_mel / T_content
            mel_start = int(round(skip_head * scale))
            mel_end   = int(round((skip_head + return_length) * scale))
            mel = mel[:, mel_start:mel_end, :]

        wav = self.vocoder(mel.transpose(1, 2))
        return wav

    def get_trainable_params(self) -> list:
        return (
            list(self.context_encoder.parameters())
            + list(self.cross_attention.parameters())
            + list(self.decoder.parameters())
        )

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())
