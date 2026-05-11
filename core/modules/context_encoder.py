import torch
import torch.nn as nn
from torch import Tensor


class ContextEncoder(nn.Module):
    """
    Encodes reference mel spectrograms into a variational context sequence
    [B, T_ctx, d_model] for TNP cross-attention.

    Variational bottleneck: the Transformer output is projected to mu and
    log_var.  During training, z is sampled via the reparameterization trick;
    during eval, z = mu (deterministic, stable inference).

    No positional encoding (speaker identity is time-invariant).
    No mean pooling (full sequence for cross-attention, TNP style).
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
        self.d_model = d_model
        self.input_proj = nn.Linear(n_mels, d_model)
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

        # Variational bottleneck heads
        self.mu_proj = nn.Linear(d_model, d_model)
        self.lv_proj = nn.Linear(d_model, d_model)

        # Near-identity init for mu_proj and near-zero variance for lv_proj so
        # that a checkpoint resumed from the deterministic version continues
        # with z ≈ mu and std ≈ 0.08 — no disruption to learned representations.
        nn.init.eye_(self.mu_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.zeros_(self.lv_proj.weight)
        nn.init.constant_(self.lv_proj.bias, -5.0)

    def forward(
        self, mel: Tensor, src_key_padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            mel:                  [B, n_mels, T_ctx]  channels-first
            src_key_padding_mask: [B, T_ctx]  bool, True = padded position
        Returns:
            z:       [B, T_ctx, d_model]  sampled latent (= mu during eval)
            mu:      [B, T_ctx, d_model]  distribution mean
            log_var: [B, T_ctx, d_model]  log variance, clamped to [-10, 4]
        """
        x = mel.permute(0, 2, 1)                                           # [B, T_ctx, n_mels]
        x = self.input_proj(x)                                             # [B, T_ctx, d_model]
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)    # [B, T_ctx, d_model]
        x = self.out_norm(x)

        mu = self.mu_proj(x)                                               # [B, T_ctx, d_model]
        log_var = self.lv_proj(x).clamp(-10.0, 4.0)                       # [B, T_ctx, d_model]

        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        else:
            z = mu   # deterministic at eval — stable inference

        return z, mu, log_var

    @torch.no_grad()
    def encode_references(self, mels: list) -> Tensor:
        """
        Inference helper: encode multiple reference utterances and concatenate
        their frame sequences along the time axis (TNP style).

        Args:
            mels: list of N tensors, each [1, n_mels, T_i]
        Returns:
            C:   [1, T_total, d_model]  z sequences (= mu in eval mode)
        """
        self.eval()
        encoded = [self.forward(m)[0] for m in mels]   # z = mu at eval
        return torch.cat(encoded, dim=1)               # [1, T_total, d_model]
