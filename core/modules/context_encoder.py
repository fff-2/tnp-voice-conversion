import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, max_len: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # register as buffer so it moves with .to(device) but is not a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ContextEncoder(nn.Module):
    """
    Encodes reference mel spectrograms from the target speaker into a single
    deterministic context vector C via self-attention + mean pooling.

    Trainable. No probabilistic components.
    """

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable than Post-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, mel: Tensor) -> Tensor:
        """
        Args:
            mel: [B, n_mels, T_ctx]  channels-first (torchaudio convention)
        Returns:
            C:   [B, d_model]        deterministic context vector
        """
        x = mel.permute(0, 2, 1)       # [B, T_ctx, n_mels]
        x = self.input_proj(x)         # [B, T_ctx, d_model]
        x = self.pe(x)                 # [B, T_ctx, d_model]
        x = self.encoder(x)            # [B, T_ctx, d_model]
        x = self.out_norm(x)           # [B, T_ctx, d_model]
        C = x.mean(dim=1)              # [B, d_model]  — deterministic mean pool
        return C

    @torch.no_grad()
    def encode_references(self, mels: list) -> Tensor:
        """
        Inference helper: encode multiple reference utterances, average their
        context vectors to produce a robust speaker embedding.

        Args:
            mels: list of N tensors, each [1, n_mels, T_i]  (variable length OK)
        Returns:
            C:   [1, d_model]
        """
        self.eval()
        vectors = [self.forward(m) for m in mels]   # list of [1, d_model]
        return torch.stack(vectors, dim=0).mean(dim=0, keepdim=True).squeeze(0).unsqueeze(0)
