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
            content_dim=769,    # 768 HuBERT + 1 F0
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
        # torchaudio defaults (power=2, htk scale, no norm) + log(x.clamp(1e-5))
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.VOCODER_SR,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=self.N_MELS,
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
        return torch.log(mel.clamp(min=1e-5))

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        source_audio: Tensor,   # [B, T_src]   source speaker waveform, 16 kHz
        context_mel: Tensor,    # [B, N_MELS, T_ctx]  target speaker reference mel
        target_audio: Tensor,   # [B, T_tgt]   target speaker waveform (for loss)
    ) -> tuple:
        C = self.context_encoder(context_mel)
        content = self.content_encoder(source_audio)
        fused = self.cross_attention(content, C)
        pred_mel = self.decoder(fused)
        target_mel = self._compute_mel(target_audio)
        T = min(pred_mel.shape[1], target_mel.shape[1])
        return pred_mel[:, :T, :], target_mel[:, :T, :]

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
    def convert_chunk(self, audio_chunk: Tensor, C: Tensor) -> Tensor:
        """
        Args:
            audio_chunk: [1, T]              16 kHz PCM float32
            C:           [1, T_ctx, D_MODEL] pre-cached context sequence
        Returns:
            waveform:    [1, 1, T_out]       24000 Hz float32
        """
        self.eval()
        content = self.content_encoder(audio_chunk)
        fused = self.cross_attention(content, C)
        mel = self.decoder(fused)
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
