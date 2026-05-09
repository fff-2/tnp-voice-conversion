import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
import numpy as np
from torch import Tensor

try:
    import torchcrepe
    _CREPE_AVAILABLE = True
except ImportError:
    _CREPE_AVAILABLE = False

try:
    from df.enhance import enhance, init_df
    _DFN_AVAILABLE = True
except ImportError:
    _DFN_AVAILABLE = False


class ContentEncoder(nn.Module):
    """
    Frozen content encoder: DeepFilterNet3 → HuBERT → torchcrepe F0 → concat.

    Sample-rate pipeline:
        input 16 kHz
          → resample 16k→48k → DeepFilterNet3 → resample 48k→16k
          → HuBERT BASE (layer 6) → [B, T_frames, 768]
          → torchcrepe (hop=320) → [B, T_frames, 1]
          → concat → [B, T_frames, 769]

    All sub-models are frozen (requires_grad=False, eval mode).
    The DFN GRU hidden state is stateful across chunks: callers must invoke
    reset_dfn_state() at the start of each new audio stream.
    """

    SR_DFN = 48_000
    SR_HUB = 16_000
    HUBERT_LAYER_IDX = 5   # 0-based index into extract_features list → layer 6
    HOP = 320              # samples @ 16kHz; 20ms; matches HuBERT frame stride

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # ── DeepFilterNet3 ────────────────────────────────────────────────────
        if _DFN_AVAILABLE:
            self.dfn_model, self.dfn_state, _ = init_df()
            self.dfn_model = self.dfn_model.to(device)
            self._freeze(self.dfn_model)
        else:
            self.dfn_model = None
            self.dfn_state = None

        # ── HuBERT BASE ───────────────────────────────────────────────────────
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.hubert = bundle.get_model().to(device)
        self._freeze(self.hubert)

        # torchcrepe is function-based (no nn.Module to freeze)
        self._crepe_available = _CREPE_AVAILABLE

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    def reset_dfn_state(self, batch_size: int = 1) -> None:
        """
        Reset DeepFilterNet GRU hidden state for a new audio stream.
        Must be called at the start of each WebSocket connection.
        """
        if self.dfn_model is not None:
            self.dfn_model.reset_h0(batch_size=batch_size, device=self.device)

    def _denoise(self, audio_16k: Tensor) -> Tensor:
        """
        Denoise audio using DeepFilterNet3.

        Args:
            audio_16k: [B, T]  mono 16 kHz float32
        Returns:
            denoised:  [B, T]  same shape, denoised
        """
        if self.dfn_model is None or self.dfn_state is None:
            return audio_16k   # passthrough if DFN not installed

        B, T = audio_16k.shape
        # Resample 16k → 48k for DFN
        audio_48k = AF.resample(audio_16k, self.SR_HUB, self.SR_DFN)  # [B, T*3]
        # enhance() expects [B, T] float32
        enhanced_48k = enhance(self.dfn_model, self.dfn_state, audio_48k)  # [B, T*3]
        # Resample 48k → 16k
        denoised = AF.resample(enhanced_48k, self.SR_DFN, self.SR_HUB)  # [B, T]
        # Ensure exact length (resampling may add/remove 1 sample due to rounding)
        if denoised.shape[-1] > T:
            denoised = denoised[..., :T]
        elif denoised.shape[-1] < T:
            denoised = torch.nn.functional.pad(denoised, (0, T - denoised.shape[-1]))
        return denoised

    @torch.no_grad()
    def _extract_hubert(self, audio_16k: Tensor) -> Tensor:
        """
        Extract HuBERT layer-6 hidden states.

        Args:
            audio_16k: [B, T]  normalized float32 16 kHz
        Returns:
            features:  [B, T_frames, 768]
        """
        # extract_features returns (list_of_layer_features, lengths)
        # list has 12 elements for HUBERT_BASE (one per transformer layer)
        features_list, _ = self.hubert.extract_features(audio_16k)
        return features_list[self.HUBERT_LAYER_IDX]   # [B, T_frames, 768]

    @torch.no_grad()
    def _extract_f0(self, audio_16k: Tensor) -> Tensor:
        """
        Extract fundamental frequency with torchcrepe.

        Args:
            audio_16k: [B, T]  16 kHz mono
        Returns:
            f0:        [B, T_frames, 1]  F0 in Hz (0.0 for unvoiced)
        """
        if not self._crepe_available:
            # Fallback: return zeros if torchcrepe not installed
            B = audio_16k.shape[0]
            T_frames = (audio_16k.shape[-1] // self.HOP) + 1
            return torch.zeros(B, T_frames, 1, device=self.device)

        pitch = torchcrepe.predict(
            audio_16k,
            sample_rate=self.SR_HUB,
            hop_length=self.HOP,
            fmin=50.0,
            fmax=2006.0,
            model="tiny",                          # fast inference
            decoder=torchcrepe.decode.argmax,      # Fix 3: frame-independent, causal
            return_periodicity=False,
            batch_size=None,
            device=self.device,
            pad=True,
        )                                          # [B, T_frames]
        pitch = torch.nan_to_num(pitch, nan=0.0)   # 0.0 for unvoiced frames
        return pitch.unsqueeze(-1)                 # [B, T_frames, 1]

    def forward(self, audio_16k: Tensor) -> Tensor:
        """
        Full content encoding pipeline.

        Args:
            audio_16k: [B, T]  raw PCM float32 at 16 kHz
        Returns:
            content:   [B, T_frames, 769]  HuBERT(768) ‖ F0(1)
        """
        # Step 1: denoise
        audio = self._denoise(audio_16k)           # [B, T]

        # Step 2: HuBERT features
        hubert_feat = self._extract_hubert(audio)  # [B, T_frames, 768]

        # Step 3: F0
        f0 = self._extract_f0(audio)               # [B, T_frames, 1]

        # Align lengths (HuBERT and crepe may differ by 1 frame at edges)
        T = min(hubert_feat.shape[1], f0.shape[1])
        hubert_feat = hubert_feat[:, :T, :]
        f0 = f0[:, :T, :]

        # Step 4: log-scale F0 then concatenate
        # Fix 2: raw F0 spans [0, 2006] Hz vs HuBERT features in [-3, +3].
        # log1p maps F0 to [0, ~7.6], preventing F0 from dominating the projection.
        f0 = torch.log1p(f0)
        content = torch.cat([hubert_feat, f0], dim=-1)  # [B, T_frames, 769]
        return content
