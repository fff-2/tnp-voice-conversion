import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor

from core.modules.content_encoder import ContentEncoder
from core.modules.tnp_unified import TNPUnifiedTransformer
from core.vocoder import VocosVocoder


class VoiceConversionModel(nn.Module):
    """
    Full voice conversion pipeline — strict TNP-D style.

    Trainable:  TNPUnifiedTransformer (~26.1M params)
    Frozen:     ContentEncoder (DFN3 + HuBERT + torchcrepe), VocosVocoder

    Context set: N (HuBERT+F0, mel) pairs from reference utterances.
    Target:      ContentVec+F0 from source audio.

    Both are concatenated into one sequence and processed by a single
    Transformer with a TNP-D block-attention mask:
        context tokens → attend only to context
        target  tokens → attend to all context + self (diagonal)

    Training:
        pred_mel, target_mel = model(source_audio, context_audios, target_audio)

    Inference:
        C   = model.compute_context(reference_audios)   # once per speaker
        wav = model.convert_chunk(audio_chunk, C)       # per streaming chunk
    """

    N_MELS = 100
    D_MODEL = 512
    CONTENT_SR = 16_000
    VOCODER_SR = 24_000

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Trainable
        self.tnp = TNPUnifiedTransformer(
            n_mels=self.N_MELS,
            d_model=self.D_MODEL,
            nhead=8,
            num_layers=8,
            dim_feedforward=2048,
        )

        # Frozen
        self.content_encoder = ContentEncoder(device=device)
        self.vocoder = VocosVocoder(device=device)

        self.resampler = torchaudio.transforms.Resample(
            self.CONTENT_SR, self.VOCODER_SR
        ).to(device)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.VOCODER_SR,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=self.N_MELS,
            power=1.0,
            center=True,
        ).to(device)

        self.to(device)

    # ── Keep frozen modules in eval mode even when model.train() is called ────

    def train(self, mode: bool = True):
        super().train(mode)
        self.content_encoder.eval()
        for p in self.content_encoder.parameters():
            p.requires_grad_(False)
        self.vocoder.eval()
        for p in self.vocoder.parameters():
            p.requires_grad_(False)
        return self

    # ── Mel helpers ───────────────────────────────────────────────────────────

    def _compute_mel(self, audio: Tensor) -> Tensor:
        """[B, T @ 16kHz] → [B, T_mel, N_MELS] log-compressed, channels-last."""
        mel = self.mel_transform(self.resampler(audio))  # [B, N_MELS, T_mel]
        return torch.log(mel.transpose(1, 2).clamp(min=1e-7))

    def _compute_mel_channels_first(self, audio: Tensor) -> Tensor:
        """[B, T @ 16kHz] → [B, N_MELS, T_mel] log-compressed."""
        return torch.log(self.mel_transform(self.resampler(audio)).clamp(min=1e-7))

    def _align_mel_to_content_rate(self, mel_cf: Tensor, T_target: int) -> Tensor:
        """[B, N_MELS, T_mel] → [B, T_target, N_MELS] downsampled to HuBERT rate."""
        mel_down = F.interpolate(
            mel_cf.float(), size=T_target, mode="linear", align_corners=False
        )
        return mel_down.transpose(1, 2)

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        source_audio: Tensor,  # [B, T_src]     augmented, 16kHz
        context_audios: Tensor,  # [B, N, T_ctx]  reference waveforms, 16kHz
        target_audio: Tensor,  # [B, T_tgt]     clean, 16kHz
        ctx_audio_lens: list | None = None,  # list[B] of list[N] sample counts
        content_lengths: list | None = None,  # list[B] sample counts for source
    ) -> tuple[Tensor, Tensor]:
        """Returns (pred_mel, target_mel) for reconstruction loss."""
        B, N, T_ctx = context_audios.shape
        ctx_flat = context_audios.view(B * N, T_ctx)

        # ── Reference features (frozen, no grad) ─────────────────────────────
        # skip_denoise=True: training data is clean studio audio; DFN3 cold-start
        # on clean audio only adds noise to the training signal with no benefit.
        # DFN3 is still used at inference (convert.py / mic_convert.py) where the
        # input may be noisy, preceded by a 1-s silence warm-up.
        with torch.no_grad():
            ctx_content = self.content_encoder(
                ctx_flat, f0_audio_16k=ctx_flat, skip_denoise=True
            )  # [B*N, T_h, 769]
            ctx_mel_cf = self._compute_mel_channels_first(ctx_flat)
            #                                            [B*N, N_MELS, T_mel]

        T_h = ctx_content.shape[1]
        ctx_mel_al = self._align_mel_to_content_rate(ctx_mel_cf, T_h)
        #                                              [B*N, T_h, N_MELS]
        ctx_pairs = torch.cat([ctx_content, ctx_mel_al], dim=-1)
        #                       [B*N, T_h, 869]

        # Project context and reshape: [B*N, T_h, D] → [B, N*T_h, D]
        ctx_encoded = self.tnp.encode_context(ctx_pairs)  # [B*N, T_h, D_MODEL]
        ctx_encoded = ctx_encoded.view(B, N * T_h, -1)  # [B, N*T_h, D_MODEL]

        # ── Context key-padding mask [B, N*T_h] ──────────────────────────────
        ctx_key_padding_mask = torch.ones(
            B, N * T_h, dtype=torch.bool, device=self.device
        )
        if ctx_audio_lens is not None:
            for i in range(B):
                for n in range(N):
                    flen = min((ctx_audio_lens[i][n] // 320) + 1, T_h)
                    ctx_key_padding_mask[i, n * T_h : n * T_h + flen] = False
        else:
            ctx_key_padding_mask.fill_(False)

        # ── Source content (frozen, no grad) ─────────────────────────────────
        with torch.no_grad():
            content = self.content_encoder(
                source_audio,
                f0_audio_16k=target_audio,
                lengths=content_lengths,
                skip_denoise=True,
            )  # [B, T_frames, 769]

        # ── Unified TNP-D transformer ─────────────────────────────────────────
        pred_mel = self.tnp(
            ctx_encoded,
            content,
            ctx_key_padding_mask=ctx_key_padding_mask,
            content_lengths=content_lengths,
        )  # [B, T_mel, N_MELS]

        target_mel = self._compute_mel(target_audio)
        T = min(pred_mel.shape[1], target_mel.shape[1])
        return pred_mel[:, :T, :], target_mel[:, :T, :]

    # ── Inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_context(self, reference_audios: list) -> Tensor:
        """
        Pre-encode reference waveforms into cached context embeddings.

        Args:
            reference_audios: list of N tensors, each [1, T_i] at 16kHz
        Returns:
            ctx_encoded: [1, N*T_h, D_MODEL]
        """
        self.eval()
        encoded_list = []
        for wav in reference_audios:
            # enhance() resets the GRU at the start of every call, so a separate
            # warm-up call followed by a content call would cold-start twice.
            # Denoise in one pass; forward with skip_denoise=True avoids a second reset.
            denoised = self.content_encoder._denoise(wav)               # one enhance() call
            content = self.content_encoder(denoised, skip_denoise=True)  # [1, T_h, 769]
            mel_cf = self._compute_mel_channels_first(denoised)         # [1, N_MELS, T_mel]
            T_h = content.shape[1]
            mel_al = self._align_mel_to_content_rate(mel_cf, T_h)  # [1, T_h, N_MELS]
            pair = torch.cat([content, mel_al], dim=-1)  # [1, T_h, 869]
            encoded_list.append(self.tnp.encode_context(pair))  # [1, T_h, D_MODEL]
        return torch.cat(encoded_list, dim=1)  # [1, N*T_h, D_MODEL]

    @torch.no_grad()
    def convert_chunk(
        self,
        audio_chunk: Tensor,
        C: Tensor,
        ctx_mask: Tensor | None = None,
        f0_stats: tuple | None = None,
    ) -> Tensor:
        """
        Args:
            audio_chunk: [1, T]              16 kHz PCM float32
            C:           [1, T_ctx, D_MODEL] pre-cached context (from compute_context)
            ctx_mask:    [1, T_ctx]          bool, True = padding (optional)
            f0_stats:    (src_mean, src_std, tgt_mean, tgt_std) or None
        Returns:
            waveform: [1, 1, T_out]  24000 Hz float32
        """
        self.eval()
        content = self.content_encoder(audio_chunk, f0_stats=f0_stats)
        mel = self.tnp(C, content, ctx_key_padding_mask=ctx_mask)
        return self.vocoder(mel.transpose(1, 2))

    @torch.no_grad()
    def convert_chunk_streaming(
        self,
        predenoised_audio: Tensor,
        C: Tensor,
        skip_head: int = 0,
        return_length: int | None = None,
        ctx_mask: Tensor | None = None,
        hubert_stats: tuple | None = None,
        f0_stats: tuple | None = None,
    ) -> Tensor:
        """
        Streaming variant: caller passes a ring buffer of recent denoised audio.
        Content frames are sliced *before* the TNP transformer so only the
        requested window is processed.
        """
        self.eval()
        content = self.content_encoder(
            predenoised_audio,
            skip_denoise=True,
            hubert_stats=hubert_stats,
            f0_stats=f0_stats,
        )
        if return_length is not None:
            content = content[:, skip_head : skip_head + return_length, :]
        elif skip_head > 0:
            content = content[:, skip_head:, :]
        mel = self.tnp(C, content, ctx_key_padding_mask=ctx_mask)
        return self.vocoder(mel.transpose(1, 2))

    @torch.no_grad()
    def convert_from_features(
        self,
        hubert_feat: Tensor,
        f0: Tensor,
        C: Tensor,
        skip_head: int = 0,
        return_length: int | None = None,
        ctx_mask: Tensor | None = None,
        hubert_stats: tuple | None = None,
        f0_stats: tuple | None = None,
    ) -> Tensor:
        """
        Convert using pre-extracted HuBERT and F0 features (zero-redundancy path).
        Content is sliced before TNP so only the requested window is forwarded.
        """
        self.eval()

        T = min(hubert_feat.shape[1], f0.shape[1])
        hubert_feat = hubert_feat[:, :T, :]
        f0 = f0[:, :T, :]

        if hubert_stats is not None:
            hub_mean, hub_std = hubert_stats
            hubert_norm = (hubert_feat - hub_mean) / (hub_std + 1e-5)
        else:
            hubert_norm = F.instance_norm(hubert_feat.transpose(1, 2)).transpose(1, 2)

        if f0_stats is not None:
            src_log_mean, src_log_std, tgt_log_mean, tgt_log_std = f0_stats
            voiced_mask = (f0 > 0.0).float()
            log_f0 = torch.log(f0.clamp(min=1.0))
            log_f0_shifted = (log_f0 - src_log_mean) / (src_log_std + 1e-5) * tgt_log_std + tgt_log_mean
            f0_shifted = torch.exp(log_f0_shifted).clamp(min=50.0, max=800.0)
            f0 = voiced_mask * f0_shifted + (1.0 - voiced_mask) * f0

        f0 = torch.log1p(f0)
        content = torch.cat([hubert_norm, f0], dim=-1)  # [1, T, 769]

        if return_length is not None:
            content = content[:, skip_head : skip_head + return_length, :]
        elif skip_head > 0:
            content = content[:, skip_head:, :]

        mel = self.tnp(C, content, ctx_key_padding_mask=ctx_mask)
        return self.vocoder(mel.transpose(1, 2))

    def get_trainable_params(self) -> list:
        return list(self.tnp.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())
