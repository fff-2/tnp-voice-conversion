"""
Real-time microphone voice conversion — no server needed.

Phase 1: Record the target speaker's voice from the microphone (or load a WAV).
Phase 2: Stream your own microphone in real-time → converted to target speaker's voice.

Usage:
    python mic_convert.py --checkpoint checkpoints/best.pt
    python mic_convert.py --checkpoint checkpoints/best.pt --pa        # WSL2 / WSLg
    python mic_convert.py --checkpoint checkpoints/best.pt --reference alice.wav
    python mic_convert.py --list-devices [--pa]
"""

import argparse
import fcntl
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

try:
    import sounddevice as sd

    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.model import VoiceConversionModel

# ── Audio ──────────────────────────────────────────────────────────────────────
SR = 16_000
VOCODER_SR = 24_000
N_MELS = 100

# ── Streaming ─────────────────────────────────────────────────────────────────
# BLOCK / 320 = 15 HuBERT frames.  15 × 1.875 = 28.125 mel frames.
# We round to 28 mel frames → 28 × 256 = 7168 samples @ 24 kHz.
BLOCK = 4800           # new samples per iteration (300 ms @ 16 kHz, 15 HuBERT frames)
BLOCK_FRAMES = BLOCK // 320   # 15 HuBERT frames per block

# Ring buffer: keep ~1.5 s of denoised history so HuBERT has enough context.
# Total window fed to HuBERT = RING_SIZE.  Only the last BLOCK_FRAMES frames
# from HuBERT output are forwarded to decoder/vocoder (RVC skip_head pattern).
RING_SAMPLES = 16000   # 1.0 s @ 16 kHz (enough HuBERT context, lower latency)
RING_FRAMES = RING_SAMPLES // 320   # 50 HuBERT frames of context

# SOLA (Similarity Overlap-Add) crossfade — finds optimal overlap position
# for smooth chunk boundaries (technique from RVC GUI).
SOLA_SEARCH = 320      # search window for best overlap (samples @ 16 kHz, 20 ms)
SOLA_OVERLAP = 640     # overlap region for crossfade (samples @ 16 kHz, 40 ms)

CHUNK = 960            # I/O granularity (60 ms)
JITTER_TARGET = 1      # chunks to pre-buffer before playback (lower = less latency)

# ── EMA normalisation ─────────────────────────────────────────────────────────
EMA_MOMENTUM = 0.95


# ── Audio helpers ──────────────────────────────────────────────────────────────


def list_devices(use_pa: bool = False) -> None:
    if use_pa:
        print("=== Input sources ===")
        subprocess.run(["pactl", "list", "sources", "short"])
        print("\n=== Output sinks ===")
        subprocess.run(["pactl", "list", "sinks", "short"])
    else:
        if not _SD_AVAILABLE:
            print("sounddevice not available — use --pa for PulseAudio devices")
            return
        print(sd.query_devices())


def _countdown(seconds: int) -> None:
    print(f"\n[REF] Will record {seconds}s of target speaker voice.")
    print("[REF] Get ready …", end="", flush=True)
    for i in range(3, 0, -1):
        time.sleep(1)
        print(f" {i}", end="", flush=True)
    print("\n[REF] *** SPEAK NOW ***", flush=True)


def record_reference(seconds: int, device_in=None) -> np.ndarray:
    """Record via sounddevice → float32 mono [T] @ SR."""
    _countdown(seconds)
    audio = sd.rec(
        int(seconds * SR), samplerate=SR, channels=1, dtype="float32", device=device_in
    )
    for elapsed in range(seconds):
        time.sleep(1)
        bar = "█" * (elapsed + 1) + "░" * (seconds - elapsed - 1)
        print(f"\r[REF] [{bar}] {elapsed + 1}/{seconds}s", end="", flush=True)
    sd.wait()
    print("\n[REF] Recording done.")
    return audio.squeeze()


def record_reference_pa(seconds: int, device_in: str | None = None) -> np.ndarray:
    """Record via parec (PulseAudio) → float32 mono [T] @ SR."""
    _countdown(seconds)
    cmd = ["parec", "--channels=1", f"--rate={SR}", "--format=float32le"]
    if device_in:
        cmd += ["--device", device_in]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    chunks: list[bytes] = []
    try:
        for elapsed in range(seconds):
            chunks.append(proc.stdout.read(SR * 4))  # 1 s of float32
            bar = "█" * (elapsed + 1) + "░" * (seconds - elapsed - 1)
            print(f"\r[REF] [{bar}] {elapsed + 1}/{seconds}s", end="", flush=True)
    finally:
        proc.terminate()
        proc.wait()
    print("\n[REF] Recording done.")
    return np.frombuffer(b"".join(chunks), dtype=np.float32).copy()


def load_wav_reference(path: str, device: torch.device) -> torch.Tensor:
    """WAV file → mono float32 [1, T] @ SR on device."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = AF.resample(wav, sr, SR)
    return wav.to(device)



# ── Real-time converter ────────────────────────────────────────────────────────


class RealtimeConverter:
    """
    Streaming voice conversion — simplified.

    Uses model.convert_chunk() directly (same code path as convert.py).
    No ring buffer, no EMA, no separate feature extraction.
    DFN GRU state is maintained automatically across blocks.

    Per-block pipeline:
      1. convert_chunk(block, C, f0_stats)  — full pipeline in one call
      2. Resample 24 kHz → 16 kHz
      3. Linear crossfade at boundaries
      4. Write to output
    """

    def __init__(
        self,
        model: VoiceConversionModel,
        C: torch.Tensor,
        device: torch.device,
        ref_audio: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.C = C
        self.dev = device

        self.mic_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=60)
        self.out_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=30)
        self._ready = threading.Event()
        self._stop = threading.Event()

        # Crossfade state
        self._xfade_buf = np.zeros(SOLA_OVERLAP, dtype=np.float32)
        self._fade_in = np.linspace(0.0, 1.0, SOLA_OVERLAP, dtype=np.float32)
        self._fade_out = 1.0 - self._fade_in

        # Accumulation buffer for _infer_thread
        self._accum = np.zeros(0, dtype=np.float32)

        # F0 stats
        self._src_f0_mean: float = 0.0
        self._src_f0_std: float = 1.0
        self.tgt_f0_mean: float = 0.0
        self.tgt_f0_std: float = 1.0
        self._n_chunks: int = 0

        if ref_audio is not None:
            self._init_target_stats(ref_audio)

    def _init_target_stats(self, ref_audio: torch.Tensor) -> None:
        if not self.model.content_encoder._crepe_available:
            print("[REF] torchcrepe not available — F0 shifting disabled")
            return
        with torch.no_grad():
            f0_t = self.model.content_encoder._extract_f0(ref_audio)
        f0 = f0_t[0, :, 0].cpu().numpy()
        voiced = f0 > 0.0
        if voiced.sum() > 1:
            self.tgt_f0_mean = float(f0[voiced].mean())
            self.tgt_f0_std = float(max(f0[voiced].std(), 5.0))
            print(f"[REF] Target F0: {self.tgt_f0_mean:.1f} ± {self.tgt_f0_std:.1f} Hz")

    def _get_f0_stats(self):
        if (
            self.model.content_encoder._crepe_available
            and self.tgt_f0_mean > 10.0
            and self._src_f0_mean > 10.0
        ):
            # Ratio shift: f0_out = f0 * (tgt_mean / src_mean).
            # Encoded into Z-score format so content_encoder.forward needs no changes:
            #   (f0 - src_mean) / 1.0 * ratio + tgt_mean  =  f0 * ratio
            ratio = self.tgt_f0_mean / self._src_f0_mean
            return (self._src_f0_mean, 1.0, self.tgt_f0_mean, ratio)
        return None

    def _update_src_f0(self, audio_t: torch.Tensor) -> None:
        """EMA update of source speaker F0 stats from this block."""
        if not self.model.content_encoder._crepe_available:
            return
        with torch.no_grad():
            f0_t = self.model.content_encoder._extract_f0(audio_t)
        f0 = f0_t[0, :, 0].cpu().numpy()
        voiced = f0 > 0.0
        if voiced.sum() > 1:
            cm = float(f0[voiced].mean())
            cs = float(max(f0[voiced].std(), 5.0))
            α = 0.05
            if self._src_f0_mean == 0.0:
                self._src_f0_mean, self._src_f0_std = cm, cs
            else:
                self._src_f0_mean = (1 - α) * self._src_f0_mean + α * cm
                self._src_f0_std  = (1 - α) * self._src_f0_std  + α * cs

    def _process_block(self, block: np.ndarray) -> np.ndarray:
        """Process one BLOCK of audio through model.convert_chunk() — same as convert.py."""
        t0 = time.time()

        audio_t = torch.from_numpy(block).unsqueeze(0).to(self.dev)

        self._update_src_f0(audio_t)
        f0_stats = self._get_f0_stats()

        # Run the exact same pipeline as convert.py
        wav = self.model.convert_chunk(audio_t, self.C, f0_stats=f0_stats)
        t_infer = time.time()

        # Resample 24 kHz → 16 kHz
        out = (
            AF.resample(wav.squeeze(0), VOCODER_SR, SR)
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        np.clip(out, -1.0, 1.0, out=out)

        # CRITICAL: enforce output == BLOCK samples.
        # Model produces slightly varying output lengths per block.
        # If output > BLOCK, the excess accumulates in pacat's pipe and
        # eventually blocks the write() call forever → "먹통".
        raw_len = len(out)
        if len(out) > BLOCK:
            out = out[:BLOCK]
        elif len(out) < BLOCK:
            out = np.pad(out, (0, BLOCK - len(out)))

        # Simple linear crossfade at boundary
        n = min(SOLA_OVERLAP, len(out))
        out[:n] = self._xfade_buf[:n] * self._fade_out[:n] + out[:n] * self._fade_in[:n]
        self._xfade_buf = np.zeros(SOLA_OVERLAP, dtype=np.float32)
        self._xfade_buf[:min(SOLA_OVERLAP, len(out))] = out[-min(SOLA_OVERLAP, len(out)):].copy()

        self._n_chunks += 1
        dt = time.time() - t0
        budget = BLOCK / SR
        if self._n_chunks <= 3 or self._n_chunks % 20 == 0:
            print(
                f"\n[TIMING] total={dt*1000:.0f}/{budget*1000:.0f}ms "
                f"infer={1000*(t_infer-t0):.0f}ms "
                f"f0={'ON' if f0_stats else 'OFF'} "
                f"out={raw_len}/{BLOCK}"
            )

        return out

    def _infer_thread(self) -> None:
        self.model.content_encoder.reset_dfn_state(batch_size=1)
        while not self._stop.is_set():
            # Drop stale input if too far behind, but keep model state
            if self.mic_q.qsize() > 20:
                dropped = 0
                while self.mic_q.qsize() > 5:
                    try:
                        self.mic_q.get_nowait()
                        dropped += 1
                    except queue.Empty:
                        break
                if dropped:
                    self._accum = np.zeros(0, dtype=np.float32)

            try:
                samples = self.mic_q.get(timeout=0.1)
            except queue.Empty:
                continue

            self._accum = np.concatenate([self._accum, samples])

            while len(self._accum) >= BLOCK:
                out = self._process_block(self._accum[:BLOCK])
                self._accum = self._accum[BLOCK:]
                for i in range(0, len(out), CHUNK):
                    piece = out[i : i + CHUNK]
                    if len(piece) > 0:
                        try:
                            self.out_q.put_nowait(piece)
                        except queue.Full:
                            pass  # drop excess rather than blocking
                if not self._ready.is_set():
                    self._ready.set()
                    print("[CONVERT] Playback started.")

    # ── sounddevice run ────────────────────────────────────────────────────────

    def _sd_callback(self, indata, outdata, frames, time_info, status) -> None:
        try:
            self.mic_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass
        if self._ready.is_set() and not self.out_q.empty():
            chunk = self.out_q.get_nowait()
            n = min(len(chunk), frames)
            outdata[:n, 0] = chunk[:n]
            if n < frames:
                outdata[n:, 0] = 0.0
        else:
            outdata[:, 0] = 0.0

    def run(self, device_in=None, device_out=None) -> None:
        """Run with sounddevice. Blocks until Ctrl+C."""
        infer = threading.Thread(target=self._infer_thread, daemon=True, name="infer")
        infer.start()
        print(f"\n[CONVERT] Streaming (BLOCK={BLOCK} = {BLOCK/SR*1000:.0f}ms).")
        print("[CONVERT] Speak into the microphone. Press Ctrl+C to stop.\n")
        try:
            with sd.Stream(
                samplerate=SR,
                channels=1,
                dtype="float32",
                blocksize=CHUNK,
                device=(device_in, device_out),
                callback=self._sd_callback,
            ):
                while not self._stop.is_set():
                    time.sleep(0.2)
                    if int(time.time()) % 2 == 0:
                        print(
                            f"\r[CONVERT] buf={self.out_q.qsize():2d}  ",
                            end="",
                            flush=True,
                        )
        except KeyboardInterrupt:
            print("\n[CONVERT] Stopping …")
        finally:
            self._stop.set()
            infer.join(timeout=2.0)
            print("[CONVERT] Done.")

    # ── PulseAudio run ─────────────────────────────────────────────────────────

    def run_pa(
        self, device_in: str | None = None, device_out: str | None = None
    ) -> None:
        """Run with PulseAudio (parec/pacat). Blocks until Ctrl+C.

        Architecture (3 threads):
          reader  → mic_q → infer → out_q → writer → pacat
          
        Critical: infer thread NEVER blocks on I/O. If writer can't
        keep up (WSL2 audio bridge stall), infer drops output rather
        than stalling. This prevents mic_q from growing.
        """
        # Output queue: small buffer between infer and writer.
        # NOT the same as self.out_q (used by sounddevice path).
        pa_out_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=10)

        in_cmd = [
            "parec",
            "--channels=1",
            f"--rate={SR}",
            "--format=float32le",
            "--latency-msec=60",
        ]
        out_cmd = [
            "pacat",
            "--channels=1",
            f"--rate={SR}",
            "--format=float32le",
            "--latency-msec=60",
        ]
        if device_in:
            in_cmd += ["--device", device_in]
        if device_out:
            out_cmd += ["--device", device_out]

        parec = subprocess.Popen(in_cmd, stdout=subprocess.PIPE)
        pacat_ref = [subprocess.Popen(out_cmd, stdin=subprocess.PIPE, bufsize=0)]

        def _reader() -> None:
            n_bytes = CHUNK * 4
            while not self._stop.is_set():
                data = parec.stdout.read(n_bytes)
                if not data:
                    break
                try:
                    self.mic_q.put_nowait(np.frombuffer(data, dtype=np.float32).copy())
                except queue.Full:
                    pass

        def _infer() -> None:
            """Process mic → model. NEVER blocks on I/O."""
            self.model.content_encoder.reset_dfn_state(batch_size=1)
            accum = np.zeros(0, dtype=np.float32)
            blk = 0

            while not self._stop.is_set():
                try:
                    samples = self.mic_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                accum = np.concatenate([accum, samples])

                while len(accum) >= BLOCK:
                    out = self._process_block(accum[:BLOCK])
                    accum = accum[BLOCK:]
                    blk += 1

                    # Put output into writer queue. If full, drop (never block).
                    try:
                        pa_out_q.put_nowait(out.astype(np.float32).tobytes())
                    except queue.Full:
                        # Drop oldest, put newest
                        try:
                            pa_out_q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            pa_out_q.put_nowait(out.astype(np.float32).tobytes())
                        except queue.Full:
                            pass

        write_count = [0]  # mutable counter for writer thread
        write_blocked = [False]

        def _writer() -> None:
            """Write audio to pacat with non-blocking I/O; full process restart on stall."""
            def _make_nonblocking(p):
                fd = p.stdin.fileno()
                fcntl.fcntl(fd, fcntl.F_SETFL,
                            fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
                return fd

            def _flush_pa_q():
                while True:
                    try:
                        pa_out_q.get_nowait()
                    except queue.Empty:
                        break

            def _hard_restart():
                print("\n[WRITER] Audio broken — restarting process …")
                parec.terminate()
                pacat_ref[0].terminate()
                os.execv(sys.executable, [sys.executable] + sys.argv)

            fd = _make_nonblocking(pacat_ref[0])
            stall_since = None

            while not self._stop.is_set():
                try:
                    data = pa_out_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                write_blocked[0] = True
                try:
                    fd = pacat_ref[0].stdin.fileno()
                    written = os.write(fd, data)
                    if written == len(data):
                        write_count[0] += 1
                        stall_since = None
                    else:
                        if stall_since is None:
                            stall_since = time.time()
                        elif time.time() - stall_since > 2.0:
                            _hard_restart()
                        else:
                            _flush_pa_q()
                except BlockingIOError:
                    if stall_since is None:
                        stall_since = time.time()
                    elif time.time() - stall_since > 2.0:
                        _hard_restart()
                    else:
                        _flush_pa_q()
                except OSError:
                    _hard_restart()
                finally:
                    write_blocked[0] = False

        rt = threading.Thread(target=_reader, daemon=True, name="pa-in")
        it = threading.Thread(target=_infer, daemon=True, name="infer")
        wt = threading.Thread(target=_writer, daemon=True, name="pa-out")
        rt.start()
        it.start()
        wt.start()

        print(
            f"\n[CONVERT] Streaming via PulseAudio (BLOCK={BLOCK} = {BLOCK/SR*1000:.0f}ms)."
        )
        print("[CONVERT] Speak into the microphone. Press Ctrl+C to stop.\n")
        try:
            last_wc = 0
            while not self._stop.is_set():
                time.sleep(2.0)
                wc = write_count[0]
                print(
                    f"\r[STATUS] mic_q={self.mic_q.qsize():2d} "
                    f"out_q={pa_out_q.qsize():2d} "
                    f"writes={wc} (+{wc-last_wc}) "
                    f"blocked={write_blocked[0]} "
                    f"threads: r={rt.is_alive()} i={it.is_alive()} w={wt.is_alive()}  ",
                    end="", flush=True,
                )
                last_wc = wc
        except KeyboardInterrupt:
            print("\n[CONVERT] Stopping …")
        finally:
            self._stop.set()
            parec.terminate()
            pacat_ref[0].terminate()
            it.join(timeout=2.0)
            rt.join(timeout=1.0)
            wt.join(timeout=1.0)
            print("[CONVERT] Done.")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time microphone voice conversion"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best.pt", help="Model checkpoint path"
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="WAV file of target speaker (skips mic recording)",
    )
    parser.add_argument(
        "--record-seconds",
        type=int,
        default=5,
        help="Seconds to record target speaker (default: 5)",
    )
    parser.add_argument(
        "--device-in",
        default=None,
        help="Input device: int index (sounddevice) or name (--pa)",
    )
    parser.add_argument(
        "--device-out",
        default=None,
        help="Output device: int index (sounddevice) or name (--pa)",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="Print audio devices and exit"
    )
    parser.add_argument(
        "--pa",
        action="store_true",
        help="Use PulseAudio backend (parec/pacat) — for WSL2/WSLg",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices(use_pa=args.pa)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device}")
    print("[INIT] Loading model …")
    model = VoiceConversionModel(device=device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[INIT] Checkpoint loaded: {ckpt_path}")
    else:
        print(
            f"[INIT] WARNING: checkpoint not found ({ckpt_path}), using random weights"
        )
    model.eval()

    if args.reference:
        print(f"[REF] Loading reference: {args.reference}")
        ref_audio = load_wav_reference(args.reference, device)
    else:
        if args.pa:
            ref_np = record_reference_pa(args.record_seconds, device_in=args.device_in)
        else:
            if not _SD_AVAILABLE:
                print(
                    "[ERROR] sounddevice not available. Use --pa or --reference <wav>."
                )
                sys.exit(1)
            ref_np = record_reference(
                args.record_seconds,
                device_in=int(args.device_in) if args.device_in is not None else None,
            )
        ref_audio = torch.from_numpy(ref_np).unsqueeze(0).to(device)
        # Save recording so that auto-restart skips the mic recording step.
        _ref_tmp = "/tmp/mic_convert_ref.wav"
        sf.write(_ref_tmp, ref_np, SR)
        sys.argv = [sys.argv[0], "--reference", _ref_tmp] + sys.argv[1:]

    C = model.compute_context([ref_audio])
    print(f"[REF] Context vector ready: {C.shape}, norm={C.norm().item():.3f}")

    converter = RealtimeConverter(model=model, C=C, device=device, ref_audio=ref_audio)

    if args.pa:
        converter.run_pa(device_in=args.device_in, device_out=args.device_out)
    else:
        if not _SD_AVAILABLE:
            print("[ERROR] sounddevice not available. Use --pa.")
            sys.exit(1)
        converter.run(
            device_in=int(args.device_in) if args.device_in is not None else None,
            device_out=int(args.device_out) if args.device_out is not None else None,
        )


if __name__ == "__main__":
    main()
