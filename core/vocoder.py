import torch
import torch.nn as nn
from torch import Tensor
from vocos import Vocos


class VocosVocoder(nn.Module):
    """
    Frozen Vocos vocoder (charactr/vocos-mel-24khz).

    Input:  log-mel [B, 100, T_mel]  24000 Hz / n_fft 1024 / hop 256
    Output: waveform [B, 1, T_wav]   24000 Hz float32 in [-1, 1]

    Log scale: log(mel.clamp(min=1e-7)) — matches vocos safe_log default.
    All parameters are frozen.
    """

    SAMPLE_RATE = 24_000

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        self._freeze()

    def _freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, mel: Tensor) -> Tensor:
        """
        Args:
            mel: [B, n_mels, T_mel]  log-compressed, channels-first
        Returns:
            wav: [B, 1, T_wav]  waveform float32
        """
        with torch.amp.autocast("cuda", enabled=False):
            wav = self.model.decode(mel.float())   # Vocos ISTFT doesn't support FP16
        return wav.unsqueeze(1)        # [B, 1, T_wav]
