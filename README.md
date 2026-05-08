# Real-Time Deterministic Voice Conversion Pipeline

Few-shot, real-time voice conversion. Record a few seconds of a target speaker — the system converts your live microphone input into that voice with low latency.

**Architecture:** Deterministic attention-based conditioning — no VAEs, no probabilistic sampling, fully stable for streaming.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Training Strategy](#training-strategy)
- [Setup](#setup)
- [Dataset — `dataset.py`](#dataset--datasetpy)
- [Training — `train.py`](#training--trainpy)
- [Real-Time Inference — `mic_convert.py`](#real-time-inference--mic_convertpy)
- [Offline Inference — `convert.py`](#offline-inference--convertpy)
- [Networked Mode](#networked-mode-optional)
- [Implementation Notes](#implementation-notes)
- [Verification](#verification)

---

## Quick Start

```bash
# 1. Create environment (Python 3.12, portaudio, ffmpeg via conda-forge)
conda env create -f environment.yml
conda activate voice

# 2. Install PyTorch for CUDA 12.8
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Download VCTK (~11 GB)
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip -d datasets/

python train.py                                                    # train
python mic_convert.py --checkpoint checkpoints/best.pt            # real-time
python convert.py --source me.wav --reference alice.wav --output out.wav  # offline
```

---

## Project Structure

<details>
<summary>File map</summary>

```
voice/
├── environment.yml             # Conda environment (Python 3.12, PyTorch + CUDA 12.8)
├── datasets/                   # All datasets go here
├── dataset.py                  # Generic speaker-folder dataset for training
├── train.py                    # Training loop (AMP + gradient accumulation)
├── convert.py                  # Offline file-to-file voice conversion
├── mic_convert.py              # Real-time microphone conversion (no server needed)
│
├── checkpoints/                # Created by train.py
│   ├── best.pt
│   ├── latest.pt
│   └── samples/step_N/         # Qualitative audio saved every SAVE_EVERY steps
│       ├── source.wav
│       ├── target.wav
│       └── converted.wav
│
├── core/
│   ├── modules/
│   │   ├── context_encoder.py  # Transformer mean-pool → C [256]
│   │   ├── content_encoder.py  # DeepFilterNet3 + HuBERT + torchcrepe F0
│   │   ├── cross_attention.py  # Q=content, K/V=C
│   │   └── decoder.py          # Conv1d + Transformer + 1.875× upsample → Mel [100]
│   ├── model.py                # Full pipeline wrapper
│   └── vocoder.py              # Vocos vocoder wrapper (frozen, 24 kHz)
│
├── server/app.py               # Optional FastAPI + WebSocket server
└── client/stream_client.py     # Optional PyAudio client for networked server
```

</details>

---

## Architecture

```
TARGET SPEAKER (reference audio)
        │
        ▼
┌─────────────────────┐
│   Context Encoder   │  Transformer (4 layers) + Mean Pool
│     (Trainable)     │  [B, 100, T_ctx] → C [B, 256]
└─────────────────────┘
        │  C (computed once, cached per speaker)
        ▼
SOURCE SPEAKER (live mic / audio file)
        │
        ▼
┌─────────────────────┐
│   Content Encoder   │  DeepFilterNet3 → HuBERT (layer 6) → torchcrepe F0
│      (Frozen)       │  [B, T @ 16 kHz] → [B, T_frames, 769]
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Cross-Attention    │  Q=content, K=V=C  →  [B, T_frames, 256]
│    (Trainable)      │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    Mel Decoder      │  Conv1d × 3 + Transformer × 2 + 1.875× upsample
│    (Trainable)      │  [B, T_frames, 256] → [B, T_mel, 100]
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Vocos Vocoder     │  charactr/vocos-mel-24khz (frozen)
│      (Frozen)       │  [B, 100, T_mel] → [B, 1, T_wav @ 24 kHz]
└─────────────────────┘
        │
        ▼
   CONVERTED AUDIO
```

| Module | Trainable | Parameters |
|---|---|---|
| ContextEncoder | Yes | ~3.1 M |
| CrossAttentionFusion | Yes | ~460 K |
| MelDecoder | Yes | ~3.2 M |
| ContentEncoder (DFN3 + HuBERT + crepe) | No | ~122 M |
| VocosVocoder | No | ~13.4 M |
| **Total trainable** | | **~6.8 M (~26 MB fp32)** |

---

## Training Strategy

### The non-parallel data problem

Voice conversion requires separating *what is said* (phonetic content) from *who says it* (speaker identity), then recombining them. The direct approach — comparing converted audio against a ground-truth recording of a different speaker saying the same sentence — requires **parallel data**: sentence-aligned recordings across every speaker pair. Parallel corpora are rare and expensive to collect.

VCTK and LibriSpeech are **non-parallel**: each speaker says different sentences. Computing a frame-wise L1 loss between "Speaker A saying *apple*" and "Speaker B saying *banana*" is meaningless — the mel spectrograms have nothing to compare.

### Self-reconstruction

The training loop uses a **self-reconstruction** objective to sidestep this entirely. The content encoder receives the **target audio** (not the source), and the loss is computed against the same target utterance:

```
Training forward pass
─────────────────────────────────────────────────────────────────────
target_audio  ──→  [ContentEncoder (frozen)]  ──→  content features
                        (HuBERT discards speaker acoustics,
                         preserves phonetics)

context_mels  ──→  [ContextEncoder]  ──→  C  (target speaker embedding)

content + C   ──→  [CrossAttention + Decoder]  ──→  pred_mel

Loss:  L1( pred_mel,  mel(target_audio) )   ← both sides are the target
```

Because both prediction and ground truth derive from the same utterance, the loss is phonetically valid. The model must learn to inject speaker identity from `C` to reconstruct `target_mel` — it cannot shortcut by aligning with a different sentence.

At **inference time** the roles switch to the intended conversion task:

```
Inference forward pass
─────────────────────────────────────────────────────────────────────
source_audio  ──→  [ContentEncoder]  ──→  content features (source phonetics)
reference     ──→  [ContextEncoder]  ──→  C  (target speaker embedding)

content + C   ──→  [CrossAttention + Decoder]  ──→  converted mel
```

### Why HuBERT makes this work

HuBERT is pre-trained with a masked-prediction objective on large-scale speech, so its internal representations correlate with phonemes more than with speaker acoustics. Feeding `target_audio` through HuBERT produces features that carry the *content* of the utterance while discarding much of the speaker-specific spectral shape. Without that bottleneck the model could learn to copy the input and ignore `C` entirely.

### Copy-synthesis risk

HuBERT layer-6 features are not perfectly speaker-agnostic — some identity leaks through. If cross-attention exploits that residual signal, the model learns *copy synthesis*: reconstruction loss looks good but cross-speaker conversion fails at inference because `C` is never truly consulted.

**Signs:** converted audio sounds like the source, not the target; validation loss is low but listening tests are poor.

**Mitigations (in increasing complexity):**

1. **Information bottleneck** — narrow FC layer or vector quantization (VQ) between content encoder and cross-attention
2. **Speaker perturbation** — randomly pitch-shift or formant-shift `target_audio` before the content encoder, breaking absolute spectral cues
3. **Speaker adversarial loss** — classifier head on content features trained adversarially to strip speaker identity from the content path

---

## Setup

**Hardware:** NVIDIA GPU with ≥8 GB VRAM (tested on RTX 5060 Ti 16 GB) · CUDA driver ≥12.8 · ≥16 GB system RAM · WSL2 Ubuntu (training) or Windows (mic client)

<details>
<summary>Environment setup (two steps)</summary>

**Step 1 — create the conda env** (Python 3.12, portaudio, ffmpeg, and all pip deps except PyTorch):

```bash
conda env create -f environment.yml
conda activate voice
```

**Step 2 — install PyTorch for CUDA 12.8** (must be run after activating the env):

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

PyTorch is not included in `environment.yml` so you can target the exact CUDA version your driver supports. Change `cu128` to match your installed CUDA if needed (e.g. `cu121` for CUDA 12.1). Check your driver's supported CUDA version with `nvidia-smi`.

If the conda solver hangs, switch to libmamba first:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -f environment.yml
```

</details>

<details>
<summary>Common setup errors</summary>

| Error | Fix |
|---|---|
| `CommandNotFoundError: conda activate` | `conda init bash`, then reopen terminal |
| `prefix already exists: .../envs/voice` | `conda env remove -n voice` first |
| `OSError: PortAudio library not found` | `portaudio` is in `environment.yml` — re-run `conda env create` |
| `torch.cuda.is_available()` returns `False` | Confirm you ran Step 2 above, and that your driver supports CUDA 12.8 (`nvidia-smi` → CUDA Version row) |
| `No module named 'torch'` | Step 2 pip install was not run, or run outside the `voice` conda env |

</details>

---

## Dataset — `dataset.py`

`SpeakerDataset` loads audio from any folder of speaker sub-directories. No parallel recordings, no matched filenames, no special naming convention. Any sample rate is auto-resampled to 16 kHz. Minimum: **2 speakers**, **7 files each**.

For each training sample it picks a random source speaker, a random target speaker, and `N_CTX=5` reference utterances from the target speaker. Context mels are computed at 24 kHz (100-band log-mel, `n_fft=1024`, `hop=256`) to match the Vocos vocoder.

<details>
<summary>Dataset options and download commands</summary>

**Option A — VCTK (default, recommended)** — 110 speakers, high quality, multiple accents.

```bash
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip -d datasets/
python train.py   # no --data-root flag needed
```

> To use mic1 files only, add `and "mic2" not in f.stem` to the file-discovery loop in `dataset.py` (~line 40).

**Option B — LibriSpeech** — 251 speakers, already in the right layout.

```bash
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz -C datasets/
python train.py --data-root datasets/LibriSpeech/train-clean-100
```

**Option C — Custom recordings**

```
datasets/my_data/
├── alice/   ← folder name = speaker identity (≥7 .wav/.flac/.mp3/.ogg files)
├── bob/
└── ...      (≥2 speakers)
```

```bash
python train.py --data-root datasets/my_data
```

| Dataset | Speakers | Size | `--data-root` |
|---|---|---|---|
| VCTK *(default)* | 110 | ~11 GB | `datasets/VCTK-Corpus-0.92/wav48_silence_trimmed` |
| LibriSpeech `train-clean-100` | 251 | ~6.3 GB | `datasets/LibriSpeech/train-clean-100` |

</details>

<details>
<summary>Direct Python API</summary>

```python
from dataset import SpeakerDataset, collate_fn
from torch.utils.data import DataLoader

ds     = SpeakerDataset("datasets/VCTK-Corpus-0.92/wav48_silence_trimmed", split="train")
loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=True)

batch = next(iter(loader))
print(batch["source_audio"].shape)   # [B, T_src]
print(batch["target_audio"].shape)   # [B, T_tgt]
print(batch["context_mels"].shape)   # [B, N_CTX, 100, T_ctx]
```

</details>

---

## Training — `train.py`

Trains ContextEncoder, CrossAttentionFusion, and MelDecoder with FP16 AMP and gradient accumulation. ContentEncoder and Vocos remain frozen throughout.

```bash
python train.py                               # VCTK default
python train.py --data-root datasets/my_data  # custom dataset
python train.py --reset                       # ignore existing checkpoint
```

Checkpoints are written to `checkpoints/`:
- `latest.pt` — every 1 000 steps, used for resuming
- `best.pt` — whenever validation loss improves, used for inference

To resume training from the last checkpoint, just run `python train.py` again — it picks up `checkpoints/latest.pt` automatically.

### Qualitative audio samples

Every `SAVE_EVERY` steps, validation saves three WAV files to `checkpoints/samples/step_{N}/`:

| File | Content |
|---|---|
| `source.wav` | Raw source speaker audio from the first validation batch (16 kHz) |
| `target.wav` | Raw target speaker audio from the first validation batch (16 kHz) |
| `converted.wav` | Source content + target speaker C, decoded through Vocos (24 kHz) |

`converted.wav` feeds **source** audio into the content encoder (not target), mirroring real inference rather than the self-reconstruction loss path. Use it to track cross-speaker generalisation: early in training it will sound like the source; as training progresses it should shift toward the target speaker's voice characteristics.

<details>
<summary>CLI flags and key constants</summary>

| Flag | Default | Description |
|---|---|---|
| `--data-root` | `datasets/VCTK-Corpus-0.92/wav48_silence_trimmed` | Speaker folder root |
| `--output-dir` | `checkpoints` | Checkpoint and sample output directory |
| `--num-workers` | `8` | DataLoader worker processes |
| `--reset` | off | Train from scratch, ignoring existing checkpoint |

Key constants at the top of `train.py`:

```python
BATCH_SIZE    = 40       # physical batch per GPU step
GRAD_ACCUM    = 1        # effective batch = BATCH_SIZE × GRAD_ACCUM
MAX_AUDIO_SEC = 8.0      # clip length — increase to use more VRAM
MAX_STEPS     = 100_000
LR            = 1e-4
WARMUP_STEPS  = 1_000
SAVE_EVERY    = 1_000    # validation + checkpoint + audio sample interval
```

</details>

---

## Real-Time Inference — `mic_convert.py`

Runs the full pipeline locally — no server required. Phase 1 computes the target speaker embedding **C** from a recording or WAV file. Phase 2 streams your microphone through the model in real time.

```bash
python mic_convert.py --list-devices                              # list device indices
python mic_convert.py --checkpoint checkpoints/best.pt            # record 5s reference from mic
python mic_convert.py --checkpoint checkpoints/best.pt --reference alice.wav  # use WAV
```

Press `Ctrl+C` to stop.

<details>
<summary>All flags and latency breakdown</summary>

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/best.pt` | Trained model checkpoint |
| `--reference` | *(none)* | WAV file to use as reference instead of recording |
| `--record-seconds` | `5` | Seconds to record from mic for the reference |
| `--device-in` | system default | Microphone device index |
| `--device-out` | system default | Speaker device index |
| `--list-devices` | — | Print available audio devices and exit |

**Pipeline:**

```
Microphone  →  mic_queue  →  Inference thread  →  Jitter buffer  →  Speaker
[1024 samples/callback]   [accumulate 3200]   [5-chunk pre-fill]
```

| Stage | Time |
|---|---|
| Mic accumulation (3 200 samples) | ~200 ms |
| GPU inference (DFN3 + HuBERT + decode + vocoder) | ~25 ms |
| Jitter buffer pre-fill (5 chunks) | ~320 ms |
| **Total steady-state** | **~250 ms** |

To reduce latency, lower `PROC_SAMPLES` in `mic_convert.py` (e.g. `1600` = 100 ms) at the cost of slightly worse chunk-boundary quality.

Vocos outputs at 24 kHz; the processing thread resamples back to 16 kHz for playback so you can use a standard 16 kHz output device.

</details>

---

## Offline Inference — `convert.py`

Converts an audio file without a microphone or server. Useful for evaluating a checkpoint before moving to real-time use.

```bash
python convert.py --source me.wav --reference alice.wav --output converted.wav
python convert.py --source me.wav --reference alice_1.wav alice_2.wav alice_3.wav --output converted.wav
```

<details>
<summary>All flags</summary>

| Flag | Default | Description |
|---|---|---|
| `--source` | *(required)* | Audio file to convert |
| `--reference` | *(required)* | One or more reference WAV files from the target speaker |
| `--checkpoint` | `checkpoints/best.pt` | Trained model checkpoint |
| `--output` | `converted.wav` | Output file path |

Audio is processed in 4-second chunks. Long files are fully supported. Output is saved at 24 kHz (Vocos native sample rate).

</details>

---

## Networked Mode (Optional)

For use across machines — e.g. a WSL2 GPU server with a Windows mic client. Not required for local use.

<details>
<summary>Server and client setup</summary>

**Start server (WSL2):**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Register a speaker and stream (Windows):**

```powershell
pip install pipwin && pipwin install pyaudio
pip install websockets requests numpy

wsl hostname -I   # find WSL2 IP

python client/stream_client.py `
    --server-ip 172.26.x.x `
    --speaker-id alice `
    --register-wav alice.wav
```

</details>

---

## Implementation Notes

<details>
<summary>Critical details for modifying the code</summary>

**Frozen modules must stay in eval mode**
`VoiceConversionModel.train()` is overridden to re-call `.eval()` on `content_encoder` and `vocoder` immediately after `super().train()`. PyTorch's `.train()` propagates to all submodules — without this override it would accidentally enable dropout and BatchNorm in HuBERT and Vocos. If you add a new frozen submodule, add it to that override.

**Dual sample rates**
The content encoder (DFN3 + HuBERT + crepe) operates at **16 kHz**. The mel computation and Vocos vocoder operate at **24 kHz**. Audio is resampled 16 kHz → 24 kHz before the mel transform; the vocoder outputs native 24 kHz audio. Do not change the mel parameters (`n_fft=1024`, `hop_length=256`, `n_mels=100`) — they must match Vocos's training configuration exactly.

**DeepFilterNet3 runs at 48 kHz**
DFN3 operates internally at 48 kHz. The content encoder resamples around it:
```
16 kHz → 48 kHz → DeepFilterNet3 → 16 kHz → HuBERT
```
Feeding 16 kHz directly into DFN3 produces silent garbage with no error.

**DeepFilterNet3 GRU state**
`reset_dfn_state()` must be called **once per audio session** — when a WebSocket connection opens or when `mic_convert.py` starts. Do not call it between chunks; the GRU hidden state carries temporal context across chunks.

**HuBERT layer index**
`HUBERT_BASE.extract_features()` returns a list of 12 tensors (one per transformer layer). Index `5` (0-based) is transformer layer 6 — the correct layer for mid-level linguistic content.

**F0 frame alignment**
`torchcrepe.predict(..., hop_length=320)` and HuBERT both have a 320-sample stride, producing `T // 320` frames each. If you ever change one, change both — mismatched strides cause silent feature misalignment at concatenation.

**Vocos input format**
The decoder outputs `[B, T_mel, 100]` (channels-last). Vocos expects `[B, 100, T_mel]` (channels-first). Always transpose before calling the vocoder: `vocoder(mel.transpose(1, 2))`.

**Mel decoder upsample factor**
HuBERT produces ~50 frames/s (stride 320 @ 16 kHz). Vocos requires ~93.75 frames/s (24 000 Hz / hop 256). The decoder uses `scale_factor = 24000 / (256 × 50) = 1.875` to bridge this gap.

**Gradient accumulation — scale all loss terms**
The training loss is divided by `GRAD_ACCUM` before `.backward()`. If you add a second loss term (e.g. speaker loss), apply the same scaling: `(l1 + 0.1 * spk_loss) / GRAD_ACCUM`. Forgetting this makes the effective learning rate `GRAD_ACCUM×` too large.

**FP16 NaN losses**
Occasional NaN losses during early training are normal AMP behaviour — the GradScaler detects overflow, skips the optimizer step, and reduces the loss scale automatically. Weights are unaffected. If NaN losses persist past step ~2 000, lower `LR` or switch to `bfloat16`.

</details>

---

## Verification

```bash
conda activate voice

# CUDA check
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Model shape smoke test (no checkpoint needed)
python - <<'EOF'
import torch
from core.model import VoiceConversionModel
m   = VoiceConversionModel(torch.device("cuda"))
audio = torch.randn(1, 3200).cuda()
C     = torch.randn(1, 256).cuda()
out   = m.convert_chunk(audio, C)
print("Output shape:", out.shape)   # [1, 1, T_wav @ 24 kHz]
EOF

# Dataset smoke test
python - <<'EOF'
from dataset import SpeakerDataset
ds = SpeakerDataset("datasets/VCTK-Corpus-0.92/wav48_silence_trimmed", split="train")
s  = ds[0]
print("source_audio:", s["source_audio"].shape)
print("target_audio:", s["target_audio"].shape)
print("context_mels:", s["context_mels"].shape)   # [N_CTX, 100, T_ctx]
EOF
```

---

## License

For research and personal use. Pre-trained models (HuBERT, Vocos, DeepFilterNet) are subject to their respective upstream licenses.
