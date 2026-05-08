import torch
import torch.nn as nn
import torchaudio
from torch import Tensor


class HiFiGANVocoder(nn.Module):
    """
    Frozen HiFiGAN vocoder wrapper using torchaudio's pretrained model.

    Input:  mel spectrogram [B, n_mels, T_mel]  channels-first, 100 Hz frame rate
    Output: waveform        [B, 1, T_mel * 160]  16 kHz float32 in [-1, 1]

    Frame rate 100 Hz × hop_size 160 = 16 000 Hz sample rate.
    All parameters are frozen (requires_grad=False, eval mode).
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        bundle = torchaudio.pipelines.HIFIGAN_16K_100HZ
        self.model = bundle.get_vocoder().to(device)   # ~50 MB download, cached
        self._freeze()

    def _freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    # Keep eval mode even when parent calls .train()
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, mel: Tensor) -> Tensor:
        """
        Args:
            mel: [B, n_mels, T_mel]  log-compressed mel, channels-first
        Returns:
            wav: [B, 1, T_mel * 160]  waveform float32
        """
        return self.model(mel)
