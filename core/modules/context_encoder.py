import torch
import torch.nn as nn
from torch import Tensor


class ContextEncoder(nn.Module):
    """
    Encodes reference (HuBERT+F0, mel) context pairs into a deterministic
    sequence z [B, T, d_model] for TNP cross-attention.

    Input: concatenated [HuBERT+F0 (769), mel (100)] = 869 dims per frame,
    channels-last.  No variational bottleneck.  No positional encoding.
    No mean pooling — full sequence is returned for cross-attention (TNP style).
    """

    def __init__(
        self,
        hubert_dim: int = 769,      # 768 HuBERT + 1 F0
        n_mels: int = 100,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        input_dim = hubert_dim + n_mels              # 869
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        ctx_pairs: Tensor,                         # [B, T, hubert_dim + n_mels]
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:                                   # [B, T, d_model]
        """
        Args:
            ctx_pairs:            [B, T, input_dim]  channels-last, already concatenated
            src_key_padding_mask: [B, T]  bool, True = padded position
        Returns:
            z: [B, T, d_model]  deterministic context sequence
        """
        x = self.input_proj(ctx_pairs)                                      # [B, T, d_model]
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)     # [B, T, d_model]
        return self.out_norm(x)

    @torch.no_grad()
    def encode_references(self, ctx_pairs_list: list) -> Tensor:
        """
        Inference helper: encode a list of pre-built context pair tensors and
        concatenate their frame sequences along the time axis (TNP style).

        Args:
            ctx_pairs_list: list of N tensors, each [1, T_i, hubert_dim + n_mels]
        Returns:
            C: [1, T_total, d_model]
        """
        self.eval()
        encoded = [self.forward(p) for p in ctx_pairs_list]
        return torch.cat(encoded, dim=1)           # [1, T_total, d_model]
