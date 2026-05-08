import torch
import torch.nn as nn
import torchaudio
from torch import Tensor

from core.modules.context_encoder import ContextEncoder
from core.modules.content_encoder import ContentEncoder
from core.modules.cross_attention import CrossAttentionFusion
from core.modules.decoder import MelDecoder
from core.vocoder import HiFiGANVocoder


class VoiceConversionModel(nn.Module):
    """
    Full voice conversion pipeline.

    Trainable:  ContextEncoder, CrossAttentionFusion, MelDecoder  (~5.8M params)
    Frozen:     ContentEncoder (DFN3 + HuBERT + torchcrepe), HiFiGANVocoder

    Training forward pass:
        forward(source_audio, context_mel, target_audio) → (pred_mel, target_mel)

    Inference:
        1. compute_context(reference_mels) → C  (once per speaker)
        2. convert_chunk(audio_chunk, C) → waveform  (per streaming chunk)
    """

    N_MELS = 80
    D_MODEL = 256

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
        self.vocoder = HiFiGANVocoder(device=device)

        # Mel transform used during training to produce ground-truth mels.
        # hop_length=160 @ 16kHz → 100 Hz frame rate, matching HiFiGAN.
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=1024,
            hop_length=160,
            win_length=1024,
            n_mels=self.N_MELS,
            f_min=0.0,
            f_max=8000.0,
            power=1.0,
            norm="slaney",
            mel_scale="slaney",
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
        Returns: mel:   [B, T_mel, N_MELS]  log1p-compressed, channels-last
        """
        mel = self.mel_transform(audio)   # [B, N_MELS, T_mel]
        mel = mel.transpose(1, 2)         # [B, T_mel, N_MELS]
        return torch.log1p(mel)           # log compression, avoids log(0)

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        source_audio: Tensor,   # [B, T_src]   source speaker waveform, 16 kHz
        context_mel: Tensor,    # [B, N_MELS, T_ctx]  target speaker reference mel
        target_audio: Tensor,   # [B, T_tgt]   target speaker waveform (for loss)
    ) -> tuple:
        """
        Returns:
            pred_mel:   [B, T, N_MELS]  predicted mel from decoder
            target_mel: [B, T, N_MELS]  ground-truth mel from target_audio
            (lengths trimmed to min(T_pred, T_tgt))
        """
        # Encode target speaker identity
        C = self.context_encoder(context_mel)          # [B, D_MODEL]

        # Encode source content (frozen path)
        content = self.content_encoder(source_audio)   # [B, T_frames, 769]

        # Fuse content with target identity
        fused = self.cross_attention(content, C)       # [B, T_frames, D_MODEL]

        # Decode to mel
        pred_mel = self.decoder(fused)                 # [B, T_frames*2, N_MELS]

        # Ground-truth mel from target speaker audio
        target_mel = self._compute_mel(target_audio)   # [B, T_mel, N_MELS]

        # Trim to minimum length to align both mels for loss computation
        T = min(pred_mel.shape[1], target_mel.shape[1])
        return pred_mel[:, :T, :], target_mel[:, :T, :]

    # ── Inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_context(self, reference_mels: list) -> Tensor:
        """
        Server-side: compute and return the cached context vector C from
        one or more reference utterances.

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
        Convert a single audio chunk using a pre-cached context vector.

        Args:
            audio_chunk: [1, T]       16 kHz PCM float32
            C:           [1, D_MODEL] pre-cached context vector
        Returns:
            waveform:    [1, 1, T_out] converted audio float32
                         T_out ≈ T (for T=3200: T_out = 3200)
        """
        self.eval()
        content = self.content_encoder(audio_chunk)    # [1, T_frames, 769]
        fused = self.cross_attention(content, C)       # [1, T_frames, D_MODEL]
        mel = self.decoder(fused)                      # [1, T_frames*2, N_MELS]
        wav = self.vocoder(mel.transpose(1, 2))        # [1, 1, T_frames*2 * 160]
        return wav

    def get_trainable_params(self) -> list:
        """Returns parameters from trainable modules only (for optimizer)."""
        return (
            list(self.context_encoder.parameters())
            + list(self.cross_attention.parameters())
            + list(self.decoder.parameters())
        )

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())
