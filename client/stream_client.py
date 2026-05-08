"""
Windows PyAudio client for real-time voice conversion.

Four-thread architecture:
    mic_thread   — reads microphone → mic_queue
    send_loop    — mic_queue → WebSocket (asyncio)
    recv_loop    — WebSocket → jitter_buffer (asyncio)
    play_thread  — jitter_buffer → speaker output

Install on Windows:
    pip install pipwin && pipwin install pyaudio
    pip install websockets requests numpy

Usage:
    # Register a target speaker first:
    python stream_client.py --server-ip 172.x.x.x --speaker-id alice --register-wav alice_ref.wav

    # Then start streaming:
    python stream_client.py --server-ip 172.x.x.x --speaker-id alice
"""

import argparse
import asyncio
import queue
import struct
import threading
import time
from collections import deque

import numpy as np
import requests
import websockets

try:
    import pyaudio
    _PYAUDIO_AVAILABLE = True
except ImportError:
    _PYAUDIO_AVAILABLE = False
    print("[WARNING] PyAudio not installed. Install: pipwin install pyaudio")

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
CHUNK = 1024              # samples per mic/speaker frame
FORMAT = pyaudio.paFloat32 if _PYAUDIO_AVAILABLE else None
CHANNELS = 1
BYTES_PER_SAMPLE = 4      # float32
CHUNK_BYTES = CHUNK * BYTES_PER_SAMPLE   # 4096 bytes per frame to server
JITTER_TARGET = 5         # number of chunks to pre-fill before playback starts
JITTER_MAX = 10           # drop oldest when buffer exceeds this


# ── Jitter Buffer ─────────────────────────────────────────────────────────────

class JitterBuffer:
    """
    Thread-safe playback buffer with pre-fill gate and overflow protection.

    pre-fill: blocks play_thread until target_size chunks are buffered.
    overflow:  drops oldest chunk when buffer exceeds max_size.
    """

    def __init__(self, target_size: int = JITTER_TARGET, max_size: int = JITTER_MAX) -> None:
        self._buf: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self.target_size = target_size
        self.max_size = max_size

    def put(self, chunk: bytes) -> None:
        with self._lock:
            if len(self._buf) >= self.max_size:
                self._buf.popleft()   # drop oldest to prevent runaway growth
            self._buf.append(chunk)
            if len(self._buf) >= self.target_size:
                self._ready.set()

    def get(self) -> bytes | None:
        with self._lock:
            return self._buf.popleft() if self._buf else None

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Block until pre-fill target is reached. Returns True if ready."""
        return self._ready.wait(timeout=timeout)

    def depth(self) -> int:
        with self._lock:
            return len(self._buf)


# ── Stream Client ─────────────────────────────────────────────────────────────

class StreamClient:
    """
    Four-thread PyAudio WebSocket voice conversion client.

    Thread layout:
        mic_thread  : PyAudio input callback → mic_queue
        send_loop   : mic_queue → WebSocket binary send  (asyncio coroutine)
        recv_loop   : WebSocket recv → split → jitter_buffer  (asyncio coroutine)
        play_thread : jitter_buffer.get() → PyAudio output
        Main thread : asyncio event loop (runs send_loop + recv_loop concurrently)
    """

    def __init__(
        self,
        server_ip: str,
        server_port: int = 8000,
        speaker_id: str = "default",
        chunk: int = CHUNK,
    ) -> None:
        if not _PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio is required. Install: pipwin install pyaudio")

        self.uri = f"ws://{server_ip}:{server_port}/ws/{speaker_id}"
        self.register_url = f"http://{server_ip}:{server_port}/register"
        self.chunk = chunk
        self.chunk_bytes = chunk * BYTES_PER_SAMPLE

        self.mic_queue: queue.Queue[bytes] = queue.Queue(maxsize=20)
        self.jitter = JitterBuffer(target_size=JITTER_TARGET, max_size=JITTER_MAX)

        self._stop = threading.Event()
        self._audio = pyaudio.PyAudio()

    # ── Speaker registration ──────────────────────────────────────────────────

    def register_speaker(self, wav_path: str, speaker_id: str) -> None:
        """HTTP POST to /register with a reference WAV file."""
        print(f"Registering speaker '{speaker_id}' from {wav_path} …")
        with open(wav_path, "rb") as f:
            resp = requests.post(
                self.register_url,
                data={"speaker_id": speaker_id},
                files={"audio_file": ("reference.wav", f, "audio/wav")},
                timeout=30,
            )
        resp.raise_for_status()
        data = resp.json()
        print(f"Registered: {data}")

    # ── Mic thread ────────────────────────────────────────────────────────────

    def _mic_thread(self) -> None:
        stream = self._audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.chunk,
        )
        print("[MIC] Recording started.")
        while not self._stop.is_set():
            data = stream.read(self.chunk, exception_on_overflow=False)
            try:
                self.mic_queue.put_nowait(data)
            except queue.Full:
                pass   # drop frame under backpressure
        stream.stop_stream()
        stream.close()
        print("[MIC] Recording stopped.")

    # ── Play thread ───────────────────────────────────────────────────────────

    def _play_thread(self) -> None:
        print("[PLAY] Waiting for jitter buffer to fill …")
        if not self.jitter.wait_ready(timeout=10.0):
            print("[PLAY] WARNING: jitter buffer pre-fill timeout, starting anyway.")
        stream = self._audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=self.chunk,
        )
        print("[PLAY] Playback started.")
        silence = b"\x00" * self.chunk_bytes
        while not self._stop.is_set():
            data = self.jitter.get()
            if data is not None:
                stream.write(data)
            else:
                # Buffer underrun: output silence to avoid audio glitches
                stream.write(silence)
        stream.stop_stream()
        stream.close()
        print("[PLAY] Playback stopped.")

    # ── Async WebSocket session ───────────────────────────────────────────────

    async def _send_loop(self, ws) -> None:
        """Drain mic_queue and forward binary frames to the server."""
        loop = asyncio.get_event_loop()
        while not self._stop.is_set():
            try:
                # Non-blocking get with short timeout so we can check _stop
                data = await loop.run_in_executor(
                    None,
                    lambda: self.mic_queue.get(timeout=0.1),
                )
                await ws.send(data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SEND] Error: {e}")
                break

    async def _recv_loop(self, ws) -> None:
        """Receive converted audio frames and feed into the jitter buffer."""
        async for message in ws:
            if not isinstance(message, bytes):
                print(f"[RECV] Unexpected message type: {type(message)}")
                continue
            # Server may send more than one CHUNK worth of audio per response.
            # Split into CHUNK-sized playback slices.
            offset = 0
            while offset < len(message):
                end = min(offset + self.chunk_bytes, len(message))
                self.jitter.put(message[offset:end])
                offset = end

    async def _ws_session(self) -> None:
        print(f"[WS] Connecting to {self.uri} …")
        try:
            async with websockets.connect(
                self.uri,
                max_size=2 ** 20,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            ) as ws:
                print("[WS] Connected.")
                await asyncio.gather(
                    self._send_loop(ws),
                    self._recv_loop(ws),
                )
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[WS] Connection closed: {e}")
        except Exception as e:
            print(f"[WS] Error: {e}")
        finally:
            self._stop.set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch all threads and run the asyncio WebSocket session."""
        self._stop.clear()

        mic = threading.Thread(target=self._mic_thread, daemon=True, name="mic")
        play = threading.Thread(target=self._play_thread, daemon=True, name="play")
        mic.start()
        play.start()

        try:
            asyncio.run(self._ws_session())
        except KeyboardInterrupt:
            print("\n[MAIN] Interrupted by user.")
        finally:
            self.stop()
            mic.join(timeout=2.0)
            play.join(timeout=2.0)
            self._audio.terminate()

    def stop(self) -> None:
        self._stop.set()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time voice conversion client (Windows, PyAudio)"
    )
    parser.add_argument(
        "--server-ip", required=True,
        help="WSL2 server IP address (find with: wsl hostname -I)",
    )
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--speaker-id", default="target_speaker",
                        help="Speaker ID to convert to (must be registered first)")
    parser.add_argument(
        "--register-wav", default=None,
        help="Path to reference WAV file for speaker registration (optional)",
    )
    args = parser.parse_args()

    client = StreamClient(
        server_ip=args.server_ip,
        server_port=args.server_port,
        speaker_id=args.speaker_id,
    )

    if args.register_wav:
        client.register_speaker(args.register_wav, args.speaker_id)

    client.start()


if __name__ == "__main__":
    main()
