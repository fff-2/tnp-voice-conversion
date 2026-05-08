# Real-Time Deterministic Voice Conversion Pipeline

A few-shot, real-time voice conversion system. Record a few seconds of a target speaker's voice, and the system converts your live microphone input into that speaker's voice with low latency.

**Architecture:** Deterministic attention-based conditioning — no VAEs, no probabilistic sampling, fully stable for live streaming.

---

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate voice

# 2. Download VCTK into datasets/
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip -d datasets/

# 3. Train (default --data-root points to datasets/VCTK-Corpus-0.92/wav48_silence_trimmed)
python train.py

# 4a. Real-time mic conversion (record reference from mic)
python mic_convert.py --checkpoint checkpoints/best.pt

# 4b. Real-time mic conversion (use a WAV as reference)
python mic_convert.py --checkpoint checkpoints/best.pt --reference alice.wav

# 4c. Offline file conversion
python convert.py --source my_voice.wav --reference alice.wav --output out.wav
```

---

## Project Structure

```
voice/
├── environment.yml             # Conda environment (Python 3.10, PyTorch + CUDA 12.4)
│
├── datasets/                   # ★ All datasets go here
│   └── VCTK-Corpus-0.92/
│       └── wav48_silence_trimmed/
│           ├── p225/
│           └── ...
│
├── dataset.py                  # ★ Generic speaker-folder dataset for training
├── train.py                    # ★ Training loop (AMP + gradient accumulation)
├── convert.py                  # ★ Offline file-to-file voice conversion
├── mic_convert.py              # ★ Real-time microphone conversion (no server needed)
│
├── checkpoints/                # Saved model checkpoints (created by train.py)
│   ├── best.pt
│   └── latest.pt
│
├── core/
│   ├── modules/
│   │   ├── context_encoder.py  # Deterministic mean-pooling transformer → C [256]
│   │   ├── content_encoder.py  # DeepFilterNet3 + HuBERT + torchcrepe F0
│   │   ├── cross_attention.py  # Cross-attention: Q=content, K/V=C
│   │   └── decoder.py          # Conv1d + Transformer + 2× upsample → Mel [80]
│   ├── model.py                # Full pipeline wrapper (train + inference)
│   └── vocoder.py              # HiFi-GAN wrapper (frozen)
│
├── server/
│   └── app.py                  # Optional: FastAPI + WebSocket server
└── client/
    └── stream_client.py        # Optional: PyAudio client for networked server
```

---

## Architecture

```
TARGET SPEAKER (reference audio)
        │
        ▼
┌─────────────────────┐
│   Context Encoder   │  Transformer (4 layers) + Mean Pool
│     (Trainable)     │  [B, 80, T_ctx] → C [B, 256]
└─────────────────────┘
        │  C (computed once, cached)
        ▼
SOURCE SPEAKER (live mic / audio file)
        │
        ▼
┌─────────────────────┐
│   Content Encoder   │  DeepFilterNet3 → HuBERT (layer 6) → torchcrepe F0
│      (Frozen)       │  [B, T] → [B, T_frames, 769]
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
│    Mel Decoder      │  Conv1d × 3 + Transformer × 2 + 2× upsample
│    (Trainable)      │  [B, T_frames, 256] → [B, T_mel, 80]
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   HiFi-GAN Vocoder  │  torchaudio HIFIGAN_16K_100HZ
│      (Frozen)       │  [B, 80, T_mel] → [B, 1, T_wav]
└─────────────────────┘
        │
        ▼
   CONVERTED AUDIO
```

| Module | Trainable | Parameters |
|---|---|---|
| ContextEncoder | Yes | ~3.1M |
| CrossAttentionFusion | Yes | ~460K |
| MelDecoder | Yes | ~2.2M |
| ContentEncoder (DFN3 + HuBERT + crepe) | No | ~122M |
| HiFiGANVocoder | No | ~13.9M |
| **Total trainable** | | **~5.8M (~22 MB fp32)** |

---

## Hardware Requirements

- **GPU:** NVIDIA RTX 5060 Ti (16 GB VRAM) or equivalent
- **CUDA:** Driver ≥ 12.x (tested on 13.2)
- **OS:** WSL2 Ubuntu for training and inference · Windows for mic client
- **RAM:** ≥ 16 GB system RAM

---

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate voice
```

If the solver hangs or fails with conflicts, switch to the faster libmamba solver first:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -f environment.yml
```

### 2. Audio backend for microphone scripts

`sounddevice` is included in `environment.yml`. If you get a PortAudio error at runtime:

```bash
# WSL2 / Ubuntu
sudo apt install libportaudio2

# Windows — sounddevice already includes PortAudio, no extra step needed
```

### Common setup errors

| Error message | Fix |
|---|---|
| `CommandNotFoundError: conda activate` | Run `conda init bash`, then reopen terminal |
| `PackagesNotFoundError: pytorch-cuda=12.4` | `conda config --add channels pytorch` and `--add channels nvidia` |
| `prefix already exists: .../envs/voice` | `conda env remove -n voice` first |
| `OSError: PortAudio library not found` | `sudo apt install libportaudio2` |
| `torch.cuda.is_available()` returns `False` | `conda install pytorch=2.4.* pytorch-cuda=12.4 -c pytorch -c nvidia` |

---

## `dataset.py` — Training Data

`dataset.py` defines `SpeakerDataset`, a generic dataset class that loads audio from any folder of speaker sub-directories. **No dataset download is required** — your own recordings work fine. No parallel recordings, no matched filenames, no special naming convention required.

All datasets are stored in the `datasets/` folder at the project root.

### Option A — VCTK (default, recommended)

110 speakers, high quality, multiple accents. This is the default dataset — `train.py` points to it automatically.

```bash
# 1. Download (~11 GB)
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip

# 2. Extract into datasets/
unzip VCTK-Corpus-0.92.zip -d datasets/

# datasets/ structure after extraction:
# datasets/VCTK-Corpus-0.92/wav48_silence_trimmed/
#   ├── p225/
#   │   ├── p225_001_mic1.flac   ← both mic1 and mic2 used (~800 files/speaker)
#   │   ├── p225_001_mic2.flac
#   │   └── ...
#   ├── p226/
#   └── ...                      (110 speakers total)

# 3. Train — no --data-root needed, default points here
python train.py
```

Files are automatically resampled from 48 kHz → 16 kHz.

> **Optional — mic1 only:** To skip mic2 duplicates, add `and "mic2" not in f.stem` to the file-discovery loop in `dataset.py` (line ~40). Training works fine either way.

### Option B — LibriSpeech

251 speakers, already in the right folder layout.

```bash
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz -C datasets/

# datasets/LibriSpeech/train-clean-100/
#   ├── 19/       ← speaker ID
#   │   ├── 198/  ← chapter sub-folder (SpeakerDataset recurses into sub-folders)
#   │   │   ├── 19-198-0000.flac
#   │   │   └── ...

python train.py --data-root datasets/LibriSpeech/train-clean-100
```

### Option C — Your own recordings

Record yourself and others, place them in speaker sub-folders inside `datasets/`:

```
datasets/
└── my_data/
    ├── alice/              ← folder name = speaker identity
    │   ├── clip_01.wav
    │   └── ...             ← minimum 7 audio files per speaker
    ├── bob/
    │   └── ...
    └── ...                 ← minimum 2 speakers total
```

**Supported formats:** `.wav` `.flac` `.mp3` `.ogg`  
**Sample rate:** any — automatically resampled to 16 kHz  
**Content:** clips do not need to say the same thing

```bash
python train.py --data-root datasets/my_data
```

### Supported public datasets at a glance

| Dataset | Speakers | Size | `--data-root` |
|---|---|---|---|
| [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) *(default)* | 110 | ~11 GB | `datasets/VCTK-Corpus-0.92/wav48_silence_trimmed` |
| [LibriSpeech](https://www.openslr.org/12) `train-clean-100` | 251 | ~6.3 GB | `datasets/LibriSpeech/train-clean-100` |
| [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | 1 | ~2.6 GB | single speaker — combine with others |

### What `SpeakerDataset` does internally

For each training sample:
1. Picks a random **source speaker** and a random **target speaker**
2. Picks a random audio file from each
3. Picks `N_CTX=5` additional files from the target speaker as reference context
4. Returns `source_audio`, `target_audio`, and `context_mels` (log-mel spectrograms of the references)

The 90 / 10 train / val speaker split is deterministic (seeded), so the same speakers always go to the same split.

### Using it directly in code

```python
from dataset import SpeakerDataset, collate_fn
from torch.utils.data import DataLoader

ds = SpeakerDataset("data", split="train")
loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=True)

batch = next(iter(loader))
print(batch["source_audio"].shape)   # [B, T_src]
print(batch["target_audio"].shape)   # [B, T_tgt]
print(batch["context_mels"].shape)   # [B, N_CTX, 80, T_ctx]
```

---

## `train.py` — Training

Trains the three trainable modules (ContextEncoder, CrossAttentionFusion, MelDecoder) with AMP mixed precision and gradient accumulation.

```bash
# VCTK default — no flags needed after downloading into datasets/
python train.py

# Any other dataset
python train.py --data-root datasets/my_data
```

| Flag | Default | Description |
|---|---|---|
| `--data-root` | `datasets/VCTK-Corpus-0.92/wav48_silence_trimmed` | Root folder containing speaker sub-directories |
| `--output-dir` | `checkpoints` | Where to save `.pt` checkpoint files |
| `--num-workers` | `4` | DataLoader worker processes |
| `--reset` | off | Ignore existing checkpoint, train from scratch |

**Key constants (edit at the top of `train.py`):**

```python
BATCH_SIZE    = 32       # physical batch per GPU step
GRAD_ACCUM    = 1        # effective batch = BATCH_SIZE × GRAD_ACCUM
MAX_AUDIO_SEC = 8.0      # clip length — increase to fill more VRAM
MAX_STEPS     = 100_000
LR            = 1e-4
WARMUP_STEPS  = 1_000
```

Checkpoints:
- `checkpoints/latest.pt` — saved every 1 000 steps, used for resuming
- `checkpoints/best.pt` — saved whenever validation loss improves, used for inference

```bash
# Resume from latest checkpoint automatically
python train.py

# Start fresh
python train.py --reset
```

---

## `mic_convert.py` — Real-Time Microphone Conversion

Runs the full pipeline locally on your GPU with no server. Two phases:

- **Phase 1:** Record a few seconds of the target speaker from the microphone (or load a WAV file) → computes context vector **C**
- **Phase 2:** Your microphone is streamed in real time → converted to the target speaker's voice → played through your speakers

```
Microphone
    │  [1024 samples @ 16 kHz per callback]
    ▼
mic_queue
    │
    ▼
Inference thread   ← accumulates 3200 samples, runs model.convert_chunk()
    │
    ▼
Jitter buffer  (5-chunk pre-fill before playback starts)
    │
    ▼
Speaker output
```

### Commands

```bash
# See available mic/speaker device indices
python mic_convert.py --list-devices

# Record 5s of target voice from mic, then convert
python mic_convert.py --checkpoint checkpoints/best.pt

# Use a WAV file as reference (skip mic recording)
python mic_convert.py --checkpoint checkpoints/best.pt --reference alice.wav

# Record a longer reference for better quality
python mic_convert.py --checkpoint checkpoints/best.pt --record-seconds 10

# Specify audio devices by index
python mic_convert.py --checkpoint checkpoints/best.pt --device-in 1 --device-out 3
```

Press `Ctrl+C` to stop.

### All flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/best.pt` | Trained model checkpoint |
| `--reference` | *(none)* | WAV file to use as reference instead of recording from mic |
| `--record-seconds` | `5` | How many seconds to record from mic for the reference |
| `--device-in` | system default | Input (microphone) device index |
| `--device-out` | system default | Output (speaker) device index |
| `--list-devices` | — | Print available audio devices and exit |

### Latency

| Stage | Time |
|---|---|
| Mic accumulation (3 200 samples) | ~200 ms |
| GPU inference (DFN3 + HuBERT + decode + vocoder) | ~25 ms |
| Jitter buffer pre-fill (5 chunks) | ~320 ms |
| **Total steady-state** | **~250 ms** |

To reduce latency, lower `PROC_SAMPLES` in `mic_convert.py` (e.g. `1600` = 100 ms window) at the cost of slightly lower quality at chunk boundaries.

---

## `convert.py` — Offline File Conversion

Converts an audio file to a target speaker's voice without a microphone or server. Useful for testing a trained checkpoint before doing real-time use.

```bash
# Single reference file
python convert.py \
    --source   my_voice.wav \
    --reference alice.wav \
    --checkpoint checkpoints/best.pt \
    --output   converted.wav

# Multiple reference files → more robust speaker embedding
python convert.py \
    --source   my_voice.wav \
    --reference alice_01.wav alice_02.wav alice_03.wav \
    --output   converted.wav
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--source` | *(required)* | Audio file to convert (any format) |
| `--reference` | *(required)* | One or more reference WAV files from the target speaker |
| `--checkpoint` | `checkpoints/best.pt` | Trained model checkpoint |
| `--output` | `converted.wav` | Output file path |

The script processes audio in 4-second chunks, printing progress as it goes. Long files are fully supported.

---

## Networked Mode (Optional)

For use across machines — e.g. a WSL2 GPU server with a Windows mic client. Not needed for local use.

### Start server (WSL2)

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Register a speaker and stream (Windows)

```powershell
pip install pipwin && pipwin install pyaudio
pip install websockets requests numpy

wsl hostname -I   # find WSL2 IP

python client/stream_client.py `
    --server-ip 172.26.x.x `
    --speaker-id alice `
    --register-wav alice.wav
```

---

## Implementation Notes

### Freezing frozen modules
`VoiceConversionModel.train()` is overridden to re-apply `.eval()` and `requires_grad_(False)` on `ContentEncoder` and `HiFiGANVocoder` after every call to `.train()`. This prevents PyTorch's mode propagation from accidentally enabling gradients in frozen submodules during the training loop.

### DeepFilterNet3 sample rate
DFN3 operates at 48 kHz internally. `content_encoder.py` resamples around it:
```
16 kHz → 48 kHz → DeepFilterNet3 → 16 kHz → HuBERT
```

### DeepFilterNet3 GRU state
DFN3 has stateful GRU layers. `reset_dfn_state()` is called **once per session** and must persist across all chunks of that session. Resetting between chunks breaks temporal continuity.

### HuBERT layer index
`HUBERT_BASE.extract_features()` returns 12 tensors. Index `5` (0-based) = transformer layer 6, which encodes mid-level linguistic content.

### F0 frame alignment
`torchcrepe.predict(..., hop_length=320)` matches HuBERT's 320-sample stride. Both produce `T // 320` frames, allowing direct concatenation into `[B, T_frames, 769]`.

---

## Verification

```bash
conda activate voice

# Check CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Model shape smoke test (no checkpoint needed)
python - <<'EOF'
import torch
from core.model import VoiceConversionModel
m = VoiceConversionModel(torch.device("cuda"))
audio = torch.randn(1, 3200).cuda()
C     = torch.randn(1, 256).cuda()
out   = m.convert_chunk(audio, C)
print("Output shape:", out.shape)   # expect torch.Size([1, 1, 3200])
EOF

# Dataset smoke test
python - <<'EOF'
from dataset import SpeakerDataset
ds = SpeakerDataset("datasets/VCTK-Corpus-0.92/wav48_silence_trimmed", split="train")
s  = ds[0]
print("source_audio:", s["source_audio"].shape)
print("target_audio:", s["target_audio"].shape)
print("context_mels:", s["context_mels"].shape)
EOF
```

---

## License

For research and personal use. Pre-trained models (HuBERT, HiFi-GAN, DeepFilterNet) are subject to their respective upstream licenses.
