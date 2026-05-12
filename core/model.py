import torch
import torch.nn as nn
import torchaudio
from torch import Tensor

from core.modules.context_encoder import ContextEncoder
from core.modules.content_encoder import ContentEncoder
from core.modules.cross_attention import CrossAttentionFusion
from core.modules.decoder import MelDecoder
from core.vocoder import VocosVocoder


class VoiceConversionModel(nn.Module):
    """
    Full voice conversion pipeline.

    Trainable:  ContextEncoder, CrossAttentionFusion, MelDecoder  (~5.8M params)
    Frozen:     ContentEncoder (DFN3 + HuBERT + torchcrepe), VocosVocoder

    Content encoder operates at 16 kHz.
    Mel spectrogram (context encoder, decoder target, vocoder) operates at 24000 Hz.

    Inference:
        1. compute_context(reference_mels) → C  (once per speaker)
        2. convert_chunk(audio_chunk, C) → waveform at 24000 Hz  (per streaming chunk)
    """

    N_MELS = 100
    D_MODEL = 256
    CONTENT_SR = 16_000    # sample rate for content encoder (HuBERT / DFN3)
    VOCODER_SR = 24_000    # sample rate for mel computation and vocoder output

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Trainable modules
        self.context_encoder = ContextEncoder(
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

        # Mel transform matching vocos-mel-24khz training parameters:
        # torchaudio defaults (htk scale, no norm) + power=1.0 + log(x.clamp(1e-7))
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

    # ── Helper ────────────────────────────────────────────────────────────────

    def _compute_mel(self, audio: Tensor) -> Tensor:
        """
        Args:    audio: [B, T]  16 kHz PCM float32
        Returns: mel:   [B, T_mel, N_MELS]  log-compressed, channels-last
        """
        audio_24k = self.resampler(audio)             # [B, T_24k]
        mel = self.mel_transform(audio_24k)           # [B, N_MELS, T_mel]
        mel = mel.transpose(1, 2)                     # [B, T_mel, N_MELS]
        return torch.log(mel.clamp(min=1e-7))

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        source_audio: Tensor,             # [B, T_src]          source waveform, 16 kHz
        context_mel: Tensor,              # [B, N_MELS, T_ctx]  target reference mel
        target_audio: Tensor,             # [B, T_tgt]          target waveform (loss)
        ctx_mask: Tensor | None = None,   # [B, T_ctx]  bool, True = padding
    ) -> tuple:
        """Returns (pred_mel, target_mel, mu, log_var) for ELBO training."""
        z, mu, log_var = self.context_encoder(context_mel)
        content = self.content_encoder(source_audio, f0_audio_16k=target_audio)
        fused = self.cross_attention(content, z, key_padding_mask=ctx_mask)
        pred_mel = self.decoder(fused)
        target_mel = self._compute_mel(target_audio)
        T = min(pred_mel.shape[1], target_mel.shape[1])
        return pred_mel[:, :T, :], target_mel[:, :T, :], mu, log_var

    # ── Inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_context(self, reference_mels: list) -> Tensor:
        """
        Args:
            reference_mels: list of tensors, each [1, N_MELS, T_i]
        Returns:
            C: [1, D_MODEL] on self.device
        """
        self.eval()
        return self.context_encoder.encode_references(reference_mels)

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

        This follows the RVC pattern:
            input_wav = [context_history | new_block]
            HuBERT features = extract from full input_wav
            keep only features[skip_head : skip_head + return_length]
            → decoder → vocoder → output audio for new_block only

        Args:
            predenoised_audio: [1, T]  denoised audio ring buffer at 16 kHz
            C:                 [1, T_ctx, D_MODEL]  pre-cached speaker context
            skip_head:         number of content frames to discard from front
            return_length:     number of content frames to keep (None = all after skip_head)
            ctx_mask:          optional padding mask for C
            hubert_stats:      (mean, std) each [1, 1, 768] — EMA stats for stable
                               streaming normalization
            f0_stats:          (src_mean, src_std, tgt_mean, tgt_std) floats — shifts
                               voiced F0 from source to target speaker range
        Returns:
            waveform: [1, 1, T_out]  24000 Hz float32
        """
        self.eval()
        # skip_denoise=True: caller manages DFN GRU state separately
        content = self.content_encoder(
            predenoised_audio,
            skip_denoise=True,
            hubert_stats=hubert_stats,
            f0_stats=f0_stats,
        )
        # Trim to only the new-block frames (RVC skip_head / return_length)
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

        The caller has already run _extract_hubert and _extract_f0 (e.g. for EMA
        stats update).  This method applies normalisation, F0 shifting, frame
        trimming, cross-attention, decoder, and vocoder — without re-running the
        heavy feature extractors.

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
        import torch.nn.functional as F

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
        content = torch.cat([hubert_norm, f0], dim=-1)  # [1, T_all, 769]

        # Pass ALL frames through cross-attention and decoder so the decoder's
        # self-attention sees full temporal context (critical for quality).
        fused = self.cross_attention(content, C, key_padding_mask=ctx_mask)
        mel = self.decoder(fused)  # [1, T_mel_all, n_mels]

        # Trim mel to only the new-block portion before vocoder.
        # mel has ~1.875× more frames than content (decoder upsampling).
        if return_length is not None:
            T_content = content.shape[1]
            T_mel = mel.shape[1]
            scale = T_mel / T_content
            mel_start = int(round(skip_head * scale))
            mel_end = int(round((skip_head + return_length) * scale))
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
