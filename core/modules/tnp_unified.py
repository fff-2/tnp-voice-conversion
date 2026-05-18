"""
TNP-D unified transformer for voice conversion.

Context (HuBERT+F0, mel) and target (HuBERT+F0) tokens are concatenated
into one sequence and processed by a shared Transformer with a block
attention mask that enforces the TNP-D conditional-independence constraints:

    Context → Context : full attention   (speaker style encoding)
    Context → Target  : blocked          (no look-ahead)
    Target  → Context : full attention   (each target reads all context)
    Target  → Target  : diagonal only    (targets are conditionally independent)

The output target portion is upsampled from HuBERT rate (50 fps) to mel
rate (93.75 fps) before the final mel projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TNPUnifiedTransformer(nn.Module):
    CTX_IN_DIM = 869   # 769 HuBERT+F0  +  100 mel
    TGT_IN_DIM = 769   # 768 HuBERT     +    1 F0
    MEL_SCALE  = 24_000 / (256 * 50)    # ≈ 1.875  HuBERT fps → mel fps

    def __init__(
        self,
        n_mels: int = 100,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_mels  = n_mels

        self.ctx_proj = nn.Linear(self.CTX_IN_DIM, d_model)
        self.tgt_proj = nn.Linear(self.TGT_IN_DIM, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm  = nn.LayerNorm(d_model)
        # Upsample target frames from HuBERT rate → mel rate before projection
        self.upsample  = nn.Upsample(
            scale_factor=self.MEL_SCALE, mode="linear", align_corners=False
        )
        self.mel_proj  = nn.Linear(d_model, n_mels)

    # ── Mask construction ─────────────────────────────────────────────────────

    @staticmethod
    def build_tnp_mask(L_ctx: int, L_tgt: int, device: torch.device) -> Tensor:
        """
        Additive attention mask [L_ctx+L_tgt, L_ctx+L_tgt].
        0 = attend freely, -inf = blocked.

        Block layout:
            [ ctx×ctx: 0     | ctx×tgt: -inf ]
            [ tgt×ctx: 0     | tgt×tgt: diag ]
        """
        L    = L_ctx + L_tgt
        mask = torch.zeros(L, L, device=device)

        # Context rows cannot see target columns
        mask[:L_ctx, L_ctx:] = float("-inf")

        # Target rows: only the diagonal within the target block is 0
        tgt_block = torch.full((L_tgt, L_tgt), float("-inf"), device=device)
        tgt_block.fill_diagonal_(0.0)
        mask[L_ctx:, L_ctx:] = tgt_block

        return mask

    # ── Public API ────────────────────────────────────────────────────────────

    def encode_context(self, ctx_pairs: Tensor) -> Tensor:
        """Project raw context pairs — call once per speaker, cache the result.

        Args:
            ctx_pairs: [B, L_ctx, CTX_IN_DIM]
        Returns:
            [B, L_ctx, d_model]
        """
        return self.ctx_proj(ctx_pairs)

    def forward(
        self,
        ctx_encoded: Tensor,                         # [B, L_ctx, d_model]
        tgt_content: Tensor,                         # [B, T_hub, TGT_IN_DIM]
        ctx_key_padding_mask: Tensor | None = None,  # [B, L_ctx]  True=pad
        content_lengths: list | None = None,         # list[B] samples @16kHz
    ) -> Tensor:                                     # [B, T_mel, n_mels]
        B     = ctx_encoded.shape[0]
        L_ctx = ctx_encoded.shape[1]
        T_hub = tgt_content.shape[1]

        tgt_encoded = self.tgt_proj(tgt_content)                      # [B, T_hub, d_model]
        combined    = torch.cat([ctx_encoded, tgt_encoded], dim=1)    # [B, L_ctx+T_hub, d_model]

        attn_mask = self.build_tnp_mask(L_ctx, T_hub, combined.device)

        # Full padding mask [B, L_ctx + T_hub] — float additive (0 = attend, -inf = block)
        # Must match attn_mask dtype to avoid PyTorch deprecation warning.
        pad_mask: Tensor | None = None
        if ctx_key_padding_mask is not None or content_lengths is not None:
            pad_mask = torch.zeros(
                B, L_ctx + T_hub, dtype=attn_mask.dtype, device=combined.device
            )
            if ctx_key_padding_mask is not None:
                # ctx_key_padding_mask is bool (True = padded); convert to -inf
                pad_mask[:, :L_ctx].masked_fill_(ctx_key_padding_mask, float("-inf"))
            if content_lengths is not None:
                for i, n_samples in enumerate(content_lengths):
                    hub_len = n_samples // 320 + 1
                    if hub_len < T_hub:
                        pad_mask[i, L_ctx + hub_len:] = float("-inf")

        out = self.transformer(
            combined,
            mask=attn_mask,
            src_key_padding_mask=pad_mask,
            is_causal=False,
        )                                                              # [B, L_ctx+T_hub, d_model]
        out = self.out_norm(out)

        # Extract target portion only — loss is computed here
        tgt_out = out[:, L_ctx:, :]                                   # [B, T_hub, d_model]

        # HuBERT rate (50 fps) → mel rate (93.75 fps)
        tgt_up = self.upsample(tgt_out.transpose(1, 2)).transpose(1, 2)  # [B, T_mel, d_model]

        return self.mel_proj(tgt_up)                                  # [B, T_mel, n_mels]
