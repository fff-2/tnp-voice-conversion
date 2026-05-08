"""
Real-time microphone voice conversion — no server needed.

Phase 1: Record the target speaker's voice from the microphone (or load a WAV).
Phase 2: Stream your own microphone in real-time → converted to target speaker's voice.

Install audio backend:
    pip install sounddevice
    # WSL2 (if no audio):  sudo apt install libportaudio2 pulseaudio
    # Windows:             pip install sounddevice  (PortAudio included)

Usage:
    # Record 5s of target voice from mic, then start converting:
    python mic_convert.py --checkpoint checkpoints/best.pt

    # Use an existing WAV as reference instead of recording:
    python mic_convert.py --checkpoint checkpoints/best.pt --reference alice.wav

    # List audio devices to find the right device index:
    python mic_convert.py --list-devices
"""

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF

try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not found.")
    print("  Install: pip install sounddevice")
    print("  WSL2:    sudo apt install libportaudio2 && pip install sounddevice")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.model import VoiceConversionModel

# ── Constants ─────────────────────────────────────────────────────────────────

SR = 16_000
VOCODER_SR = 24_000   # vocos output sample rate; output is resampled back to SR for playback
N_MELS = 100
CHUNK = 1024          # samples per audio callback frame  (~64 ms)
PROC_SAMPLES = 3200   # accumulated before each inference (~200 ms)
OVERLAP = 1600        # left-context fed to DFN/HuBERT to reduce edge artefacts
JITTER_TARGET = 5     # chunks to pre-buffer before playback starts


# ── Audio helpers ─────────────────────────────────────────────────────────────

def list_devices() -> None:
    print(sd.query_devices())


def record_reference(seconds: int, device_in=None) -> np.ndarray:
    """
    Record `seconds` of audio from the microphone.
    Returns float32 mono numpy array [T] @ SR.
    """
    print(f"\n[REF] Will record {seconds}s of target speaker voice.")
    print("[REF] Get ready …", end="", flush=True)
    for i in range(3, 0, -1):
        time.sleep(1)
        print(f" {i}", end="", flush=True)
    print("\n[REF] *** SPEAK NOW ***", flush=True)

    audio = sd.rec(
        int(seconds * SR),
        samplerate=SR,
        channels=1,
        dtype="float32",
        device=device_in,
    )
    # Progress bar while recording
    for elapsed in range(seconds):
        time.sleep(1)
        bar = "█" * (elapsed + 1) + "░" * (seconds - elapsed - 1)
        print(f"\r[REF] [{bar}] {elapsed+1}/{seconds}s", end="", flush=True)
    sd.wait()
    print("\n[REF] Recording done.")
    return audio.squeeze()   # [T]


def load_wav_reference(path: str, device: torch.device) -> torch.Tensor:
    """Load a WAV file → mono float32 tensor [1, T] @ SR on device."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = AF.resample(wav, sr, SR)
    return wav.to(device)    # [1, T]


def audio_to_mel(audio: torch.Tensor, device: torch.device) -> torch.Tensor:
    """[1, T @ 16kHz] → [1, N_MELS, T_mel] log mel at 24000 Hz."""
    audio_24k = AF.resample(audio, SR, VOCODER_SR)
    tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=VOCODER_SR, n_fft=1024, hop_length=256, win_length=1024, n_mels=N_MELS,
    ).to(device)
    mel = tf(audio_24k)
    return torch.log(mel.clamp(min=1e-5))


# ── Real-time converter ───────────────────────────────────────────────────────

class RealtimeConverter:
    """
    Two-thread real-time voice conversion pipeline.

    Thread 1 (sounddevice callback, called by PortAudio):
        - Writes incoming mic samples to mic_queue
        - Reads from out_deque and writes to speaker output

    Thread 2 (processing_thread):
        - Drains mic_queue, accumulates to PROC_SAMPLES
        - Runs model.convert_chunk()
        - Pushes output chunks into out_deque (jitter buffer)
    """

    def __init__(
        self,
        model: VoiceConversionModel,
        C: torch.Tensor,
        device: torch.device,
        device_in=None,
        device_out=None,
    ) -> None:
        self.model = model
        self.C = C
        self.torch_device = device
        self.device_in = device_in
        self.device_out = device_out

        self.mic_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=60)
        self.out_deque: deque[np.ndarray] = deque(maxlen=JITTER_TARGET * 3)

        self._ready = threading.Event()   # set once jitter buffer is pre-filled
        self._stop = threading.Event()

        self._accum = np.zeros(0, dtype=np.float32)   # mic accumulation buffer
        self._prev_ctx = np.zeros(OVERLAP, dtype=np.float32)  # DFN left-context
        self._pre_fill_count = 0

    # ── sounddevice callback (runs in PortAudio audio thread) ─────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,    # [CHUNK, 1]
        outdata: np.ndarray,   # [CHUNK, 1]
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            pass   # ignore overflow/underflow warnings in the hot path

        # Push mic input
        try:
            self.mic_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass   # drop under sustained backpressure

        # Pull converted output
        if self._ready.is_set() and self.out_deque:
            chunk = self.out_deque.popleft()
            n = min(len(chunk), frames)
            outdata[:n, 0] = chunk[:n]
            if n < frames:
                outdata[n:, 0] = 0.0
        else:
            outdata[:, 0] = 0.0   # silence while pre-filling

    # ── Processing thread (Python thread, runs inference on GPU) ──────────────

    def _processing_thread(self) -> None:
        while not self._stop.is_set():
            # Drain mic_queue into accumulation buffer
            try:
                samples = self.mic_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self._accum = np.concatenate([self._accum, samples])

            # Process whenever we have enough samples
            while len(self._accum) >= PROC_SAMPLES:
                chunk = self._accum[:PROC_SAMPLES]
                self._accum = self._accum[PROC_SAMPLES:]

                # Prepend left-context for DFN GRU continuity + HuBERT edge quality
                with_ctx = np.concatenate([self._prev_ctx, chunk])
                self._prev_ctx = chunk[-OVERLAP:].copy()

                # GPU inference
                audio_t = (
                    torch.from_numpy(with_ctx)
                    .unsqueeze(0)
                    .to(self.torch_device)
                )                                           # [1, PROC+OVERLAP]
                wav = self.model.convert_chunk(audio_t, self.C)   # [1, 1, T_out @ 24000 Hz]

                # Resample vocoder output (24000 Hz) → playback rate (16000 Hz)
                wav_16k = AF.resample(wav.squeeze(0), VOCODER_SR, SR)   # [1, T_16k]
                out_np = wav_16k.squeeze().cpu().numpy().astype(np.float32)
                np.clip(out_np, -1.0, 1.0, out=out_np)

                # Discard the context prefix from the output
                trim = int(len(out_np) * PROC_SAMPLES / (PROC_SAMPLES + OVERLAP))
                out_trimmed = out_np[-trim:]

                # Split into CHUNK-sized slices and push to jitter buffer
                for i in range(0, len(out_trimmed), CHUNK):
                    self.out_deque.append(out_trimmed[i: i + CHUNK])

                self._pre_fill_count += 1
                if self._pre_fill_count >= JITTER_TARGET and not self._ready.is_set():
                    self._ready.set()
                    print("[CONVERT] Buffer ready — playback started.")

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Start real-time conversion. Blocks until Ctrl+C."""
        # Reset DFN GRU state for this session
        self.model.content_encoder.reset_dfn_state(batch_size=1)

        proc = threading.Thread(
            target=self._processing_thread, daemon=True, name="inference"
        )
        proc.start()

        print(f"\n[CONVERT] Streaming started (chunk={CHUNK}, proc={PROC_SAMPLES}).")
        print("[CONVERT] Speak into the microphone. Press Ctrl+C to stop.\n")

        try:
            with sd.Stream(
                samplerate=SR,
                channels=1,
                dtype="float32",
                blocksize=CHUNK,
                device=(self.device_in, self.device_out),
                callback=self._audio_callback,
            ):
                while not self._stop.is_set():
                    time.sleep(0.2)
                    # Print live buffer depth every 2 seconds
                    if int(time.time()) % 2 == 0:
                        print(
                            f"\r[CONVERT] buffer depth: {len(self.out_deque):2d} chunks  ",
                            end="",
                            flush=True,
                        )
        except KeyboardInterrupt:
            print("\n[CONVERT] Stopping …")
        finally:
            self._stop.set()
            proc.join(timeout=2.0)
            print("[CONVERT] Done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time microphone voice conversion (no server)"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best.pt",
        help="Trained model checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--reference", default=None,
        help="WAV file of target speaker (skip mic recording if provided)",
    )
    parser.add_argument(
        "--record-seconds", type=int, default=5,
        help="Seconds to record target speaker from mic (default: 5)",
    )
    parser.add_argument(
        "--device-in", type=int, default=None,
        help="Input audio device index (default: system default)",
    )
    parser.add_argument(
        "--device-out", type=int, default=None,
        help="Output audio device index (default: system default)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print available audio devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device}")

    print("[INIT] Loading model …")
    model = VoiceConversionModel(device=device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print(f"[INIT] Checkpoint loaded: {ckpt_path}")
    else:
        print(f"[INIT] WARNING: checkpoint not found ({ckpt_path}), using random weights")
    model.eval()

    # ── Compute context vector C ──────────────────────────────────────────────
    if args.reference:
        # Load from file
        print(f"[REF] Loading reference: {args.reference}")
        ref_audio = load_wav_reference(args.reference, device)  # [1, T]
    else:
        # Record from microphone
        ref_np = record_reference(args.record_seconds, device_in=args.device_in)
        ref_audio = torch.from_numpy(ref_np).unsqueeze(0).to(device)  # [1, T]

    ref_mel = audio_to_mel(ref_audio, device)          # [1, N_MELS, T_mel]
    C = model.compute_context([ref_mel])               # [1, 256]
    print(f"[REF] Context vector ready: {C.shape}, norm={C.norm().item():.3f}")

    # ── Start real-time conversion ────────────────────────────────────────────
    converter = RealtimeConverter(
        model=model,
        C=C,
        device=device,
        device_in=args.device_in,
        device_out=args.device_out,
    )
    converter.run()


if __name__ == "__main__":
    main()
