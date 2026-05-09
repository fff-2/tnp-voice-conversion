import torch
import torch.nn as nn
from torch import Tensor


class ContextEncoder(nn.Module):
    """
    Encodes reference mel spectrograms from the target speaker into a sequence
    of context representations [B, T_ctx, d_model] for cross-attention.

    Fix 1 — No mean pooling: outputs the full sequence so CrossAttentionFusion
    can attend over all reference frames (TNP style).
    Fix 5 — No positional encoding: speaker identity is time-invariant;
    absolute position forces the model to overfit on phoneme timing in the
    reference clip rather than learning speaker timbre.

    Trainable. No probabilistic components.
    """

    def __init__(
        self,
        n_mels: int = 100,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
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
            C:   [B, T_ctx, d_model]  sequence of context representations
        """
        x = mel.permute(0, 2, 1)       # [B, T_ctx, n_mels]
        x = self.input_proj(x)         # [B, T_ctx, d_model]
        x = self.encoder(x)            # [B, T_ctx, d_model]
        x = self.out_norm(x)           # [B, T_ctx, d_model]
        return x                       # no mean pool — full sequence for cross-attention

    @torch.no_grad()
    def encode_references(self, mels: list) -> Tensor:
        """
        Inference helper: encode multiple reference utterances and concatenate
        their frame sequences along the time axis, giving cross-attention access
        to the full set of reference frames (TNP style).

        Args:
            mels: list of N tensors, each [1, n_mels, T_i]  (variable length OK)
        Returns:
            C:   [1, T_total, d_model]  where T_total = sum(T_i)
        """
        self.eval()
        encoded = [self.forward(m) for m in mels]   # list of [1, T_i, d_model]
        return torch.cat(encoded, dim=1)             # [1, T_total, d_model]
