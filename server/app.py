"""
FastAPI + WebSocket server for real-time voice conversion.

Endpoints:
    POST /register          — register a target speaker (compute + cache context C)
    WebSocket /ws/{id}      — stream source audio, receive converted audio

Binary WebSocket protocol:
    Client → Server: 4096 bytes = 1024 × float32 PCM samples @ 16 kHz (raw, no header)
    Server → Client: 12800 bytes = 3200 × float32 PCM samples @ 16 kHz (raw, no header)
    The server accumulates PROC_SAMPLES=3200 input samples before each inference call.

Start the server:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import io
import sys
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from loguru import logger

# Allow running from repo root: `uvicorn server.app:app`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.model import VoiceConversionModel

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
VOCODER_SR = 24_000          # mel computation and vocoder output sample rate
CLIENT_CHUNK = 1024          # float32 samples per WebSocket frame from client
PROC_SAMPLES = 3200          # samples accumulated before each inference (200 ms)
OVERLAP_SAMPLES = 1600       # left-context overlap to reduce edge artifacts (100 ms)
N_MELS = 100

# ── Global state ──────────────────────────────────────────────────────────────

app = FastAPI(title="Voice Conversion Server")
_model: Optional[VoiceConversionModel] = None
_device: Optional[torch.device] = None
_speaker_contexts: dict[str, torch.Tensor] = {}   # speaker_id → C [1, 256]
_executor: Optional[ThreadPoolExecutor] = None

# Mel transform for speaker registration (converts reference WAV → mel)
_mel_transform: Optional[torchaudio.transforms.MelSpectrogram] = None


# ── Startup / shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    global _model, _device, _executor, _mel_transform

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {_device}")

    logger.info("Loading VoiceConversionModel …")
    _model = VoiceConversionModel(device=_device)

    ckpt_path = Path("checkpoints/best.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=_device)
        _model.load_state_dict(state["model"], strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning("No checkpoint found — running with random trainable weights")

    _model.eval()

    _mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=VOCODER_SR, n_fft=1024, hop_length=256, win_length=1024, n_mels=N_MELS,
    ).to(_device)

    _executor = ThreadPoolExecutor(max_workers=4)
    logger.info("Server ready.")


@app.on_event("shutdown")
async def shutdown() -> None:
    if _executor:
        _executor.shutdown(wait=False)


# ── Speaker registration ──────────────────────────────────────────────────────

def _compute_context_sync(wav_bytes: bytes, speaker_id: str) -> torch.Tensor:
    """
    Blocking: decode WAV bytes → mel → context vector C.
    Runs inside ThreadPoolExecutor so it doesn't block the event loop.
    """
    buf = io.BytesIO(wav_bytes)
    waveform, sr = torchaudio.load(buf)     # [C, T]
    # Downmix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)   # [1, T]
    # Resample to 16 kHz if necessary
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.to(_device)   # [1, T]

    # Resample to 24000 Hz for mel (matches vocos-mel-24khz training params)
    waveform_24k = torchaudio.functional.resample(waveform, SAMPLE_RATE, VOCODER_SR)
    mel = _mel_transform(waveform_24k)             # [1, N_MELS, T_mel]
    mel = torch.log(mel.clamp(min=1e-5))

    # Compute context vector
    C = _model.compute_context([mel])  # [1, D_MODEL]
    logger.info(f"Registered speaker '{speaker_id}', C.shape={C.shape}")
    return C


@app.post("/register")
async def register_speaker(
    speaker_id: str = Form(...),
    audio_file: UploadFile = File(...),
) -> JSONResponse:
    """
    Register a target speaker by uploading a reference WAV file.
    The context vector C is computed and cached server-side.

    Form fields:
        speaker_id  (str)  — unique identifier for this speaker
        audio_file  (file) — WAV file with reference speech (any duration ≥ 2 s)
    """
    wav_bytes = await audio_file.read()
    loop = asyncio.get_event_loop()
    try:
        C = await loop.run_in_executor(
            _executor, _compute_context_sync, wav_bytes, speaker_id
        )
        _speaker_contexts[speaker_id] = C
        return JSONResponse({"speaker_id": speaker_id, "status": "ok", "context_dim": C.shape[-1]})
    except Exception as e:
        logger.exception(f"Registration failed for '{speaker_id}': {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/speakers")
async def list_speakers() -> JSONResponse:
    return JSONResponse({"speakers": list(_speaker_contexts.keys())})


# ── WebSocket streaming ───────────────────────────────────────────────────────

def _run_inference_sync(audio_chunk: torch.Tensor, C: torch.Tensor) -> bytes:
    """
    Blocking inference: content encode → cross-attn → decode → vocode.
    Runs in ThreadPoolExecutor.
    Returns converted audio as raw float32 bytes (little-endian).
    """
    wav = _model.convert_chunk(audio_chunk, C)       # [1, 1, T_out]
    wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
    # Clip to [-1, 1] to prevent clipping artefacts downstream
    wav_np = np.clip(wav_np, -1.0, 1.0)
    return wav_np.tobytes()


@app.websocket("/ws/{speaker_id}")
async def websocket_endpoint(websocket: WebSocket, speaker_id: str) -> None:
    """
    Real-time voice conversion WebSocket endpoint.

    The client sends 4096-byte binary frames (1024 float32 PCM @ 16 kHz).
    The server accumulates PROC_SAMPLES=3200 samples, runs inference, and
    returns 12800 bytes (3200 float32 samples) of converted audio.
    """
    await websocket.accept()

    # Validate speaker registration
    if speaker_id not in _speaker_contexts:
        await websocket.send_json({"error": f"Speaker '{speaker_id}' not registered. POST /register first."})
        await websocket.close(code=1008)
        return

    C = _speaker_contexts[speaker_id]   # [1, D_MODEL]
    loop = asyncio.get_event_loop()

    # Per-connection audio accumulation buffer
    audio_buffer: deque[float] = deque()
    # Left-context from the previous processing window (to reduce edge artefacts)
    prev_context: np.ndarray = np.zeros(OVERLAP_SAMPLES, dtype=np.float32)

    # Reset DeepFilterNet GRU state for this connection
    _model.content_encoder.reset_dfn_state(batch_size=1)

    logger.info(f"WebSocket connected: speaker='{speaker_id}'")
    try:
        async for message in websocket.iter_bytes():
            # Decode incoming PCM bytes → float32 samples
            samples = np.frombuffer(message, dtype=np.float32).copy()
            audio_buffer.extend(samples.tolist())

            # Process when we have enough samples
            while len(audio_buffer) >= PROC_SAMPLES:
                # Extract PROC_SAMPLES from buffer
                chunk = np.array([audio_buffer.popleft() for _ in range(PROC_SAMPLES)],
                                  dtype=np.float32)

                # Prepend left-context overlap for better boundary quality
                chunk_with_ctx = np.concatenate([prev_context, chunk])
                prev_context = chunk[-OVERLAP_SAMPLES:].copy()

                # Convert to tensor
                audio_tensor = torch.from_numpy(chunk_with_ctx).unsqueeze(0).to(_device)  # [1, T]

                # Run inference in thread pool (non-blocking)
                converted_bytes = await loop.run_in_executor(
                    _executor, _run_inference_sync, audio_tensor, C
                )

                # The output contains PROC_SAMPLES + OVERLAP_SAMPLES worth of audio.
                # Return only the PROC_SAMPLES portion (drop the context prefix).
                # hop=160, T_frames = (PROC_SAMPLES + OVERLAP_SAMPLES) // 320 (HuBERT stride)
                # We trim to the last PROC_SAMPLES samples.
                out_f32 = np.frombuffer(converted_bytes, dtype=np.float32)
                # Keep only the tail corresponding to PROC_SAMPLES output
                out_len = int(len(out_f32) * PROC_SAMPLES / (PROC_SAMPLES + OVERLAP_SAMPLES))
                out_trimmed = out_f32[-out_len:].astype(np.float32)
                await websocket.send_bytes(out_trimmed.tobytes())

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: speaker='{speaker_id}'")
    except Exception as e:
        logger.exception(f"WebSocket error for '{speaker_id}': {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
