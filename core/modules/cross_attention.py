import torch
import torch.nn as nn
from torch import Tensor


class CrossAttentionFusion(nn.Module):
    """
    Single cross-attention block fusing source content with target context.

      Q = projected source content   [B, T_frames, d_model]
      K = V = context sequence C     [B, T_ctx, d_model]

    Fix 1 — C is now a full sequence of reference frames [B, T_ctx, d_model]
    rather than a single pooled vector. Softmax over T_ctx > 1 makes attention
    weights meaningful: each content frame learns to attend to the most relevant
    reference frames for speaker conditioning.

    Trainable.
    """

    def __init__(
        self,
        hubert_dim: int = 768,
        f0_dim: int = 1,
        d_model: int = 256,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hubert_proj = nn.Linear(hubert_dim, d_model)
        self.f0_proj = nn.Linear(f0_dim, d_model)
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
        Q = self.hubert_proj(hubert_feat) + self.f0_proj(f0_feat) # [B, T_frames, d_model]
        attn_out, _ = self.attn(
            query=Q,
            key=C,
            value=C,
            need_weights=False,
            key_padding_mask=key_padding_mask,  # masks zero-padded context frames
        )                                  # [B, T_frames, d_model]
        x = self.out_norm(Q + attn_out)    # [B, T_frames, d_model]
        x = self.ffn_norm(x + self.ffn(x)) # [B, T_frames, d_model]
        return x
