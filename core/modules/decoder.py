import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Conv1dBlock(nn.Module):
    """Residual 1D conv block: Conv1d + LayerNorm + GELU + residual."""

    def __init__(self, d_model: int = 256, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        # padding = same-length output
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d_model]
        residual = x
        # Conv1d expects [B, C, T]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, d_model]
        h = F.gelu(self.norm(h + residual))
        return self.drop(h)


class MelDecoder(nn.Module):
    """
    Decodes cross-attention output to mel spectrogram with 2× temporal upsampling.

    Input:  [B, T_frames, d_model]   at HuBERT rate (50 Hz)
    Output: [B, T_frames*2, n_mels]  at vocoder rate (100 Hz)

    Trainable.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_mels: int = 100,
        n_conv_blocks: int = 3,
        n_transformer_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv_blocks = nn.ModuleList(
            [Conv1dBlock(d_model, dropout=dropout) for _ in range(n_conv_blocks)]
        )
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_transformer_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        # Upsample HuBERT 50 Hz frames → Vocos 93.75 Hz (24000 Hz / hop 256)
        self.upsample = nn.Upsample(scale_factor=24000 / (256 * 50), mode="linear", align_corners=False)
        self.out_proj = nn.Linear(d_model, n_mels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x:   [B, T_frames, d_model]
        Returns:
            mel: [B, T_frames*2, n_mels]
        """
        for block in self.conv_blocks:
            x = block(x)                               # [B, T_frames, d_model]

        for layer in self.transformer_layers:
            x = layer(x)                               # [B, T_frames, d_model]

        x = self.out_norm(x)                           # [B, T_frames, d_model]

        # Upsample time dimension
        x = x.transpose(1, 2)                          # [B, d_model, T_frames]
        x = self.upsample(x)                           # [B, d_model, T_frames*2]
        x = x.transpose(1, 2)                          # [B, T_frames*2, d_model]

        mel = self.out_proj(x)                         # [B, T_frames*2, n_mels]
        return mel
