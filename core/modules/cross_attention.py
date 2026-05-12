import torch
import torch.nn as nn
from torch import Tensor


class ContinuousPitchEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # d_model must be even
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # MLP to map sine/cosine waves into a learnable non-linear space
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, 1] (log1p(f0) values, 0 ~ 7.6)

        # Scale-up: expand dynamic range so waves oscillate nicely across dimensions
        x_scaled = x * 10.0

        # Sinusoidal transformation
        sinusoid_inp = x_scaled * self.inv_freq  # [B, T, d_model//2]
        emb = torch.cat(
            [sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1
        )  # [B, T, d_model]

        # Non-linear mapping
        return self.mlp(emb)


class CrossAttentionFusion(nn.Module):
    """
    Single cross-attention block fusing source content with target context.

      Q = projected source content   [B, T_frames, d_model]
      K = V = context sequence C     [B, T_ctx, d_model]
    """

    def __init__(
        self,
        hubert_dim: int = 768,
        f0_dim: int = 1,  # Kept for signature compatibility
        d_model: int = 256,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hubert_proj = nn.Linear(hubert_dim, d_model)

        # Replace weak Linear with powerful ContinuousPitchEmbedding
        self.f0_embed = ContinuousPitchEmbedding(d_model=d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            kdim=d_model,
            vdim=d_model,
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        content: Tensor,
        C: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            content:          [B, T_frames, content_dim]   source content features
            C:                [B, T_ctx,    d_model]        target context sequence
            key_padding_mask: [B, T_ctx]    bool, True = padding (ignored by attention)
        Returns:
            fused:            [B, T_frames, d_model]
        """
        hubert_feat = content[..., :768]
        f0_feat = content[..., 768:]

        # Pitch embedding now provides rich non-linear harmonics in the 256-d space
        Q = self.hubert_proj(hubert_feat) + self.f0_embed(
            f0_feat
        )  # [B, T_frames, d_model]

        attn_out, _ = self.attn(
            query=Q,
            key=C,
            value=C,
            need_weights=False,
            key_padding_mask=key_padding_mask,  # masks zero-padded context frames
        )  # [B, T_frames, d_model]
        x = self.out_norm(Q + attn_out)  # [B, T_frames, d_model]
        x = self.ffn_norm(x + self.ffn(x))  # [B, T_frames, d_model]
        return x
