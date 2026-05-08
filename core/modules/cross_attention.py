import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttentionFusion(nn.Module):
    """
    Single cross-attention block fusing source content with target context.

      Q = projected source content  [B, T_frames, d_model]
      K = V = context vector C      [B, 1, d_model]  (single context token)

    Trainable.
    """

    def __init__(
        self,
        content_dim: int = 769,   # 768 HuBERT + 1 F0
        d_model: int = 256,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(content_dim, d_model)
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

    def forward(self, content: Tensor, C: Tensor) -> Tensor:
        """
        Args:
            content: [B, T_frames, content_dim]   source content features
            C:       [B, d_model]                 target context vector
        Returns:
            fused:   [B, T_frames, d_model]
        """
        Q = self.q_proj(content)           # [B, T_frames, d_model]
        KV = C.unsqueeze(1)                # [B, 1, d_model]
        attn_out, _ = self.attn(
            query=Q,
            key=KV,
            value=KV,
            need_weights=False,
        )                                  # [B, T_frames, d_model]
        # Pre-norm residual (Q is already in d_model space)
        x = self.out_norm(Q + attn_out)    # [B, T_frames, d_model]
        x = self.ffn_norm(x + self.ffn(x)) # [B, T_frames, d_model]
        return x
