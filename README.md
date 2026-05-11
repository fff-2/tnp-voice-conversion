# Real-Time Voice Conversion Pipeline — Variational TNP

Few-shot, real-time voice conversion. Record a few seconds of a target speaker — the system converts your live microphone input into that voice with low latency.

**Architecture:** Variational Transformer Neural Process — the context encoder outputs a stochastic latent sequence `z ~ q(z|C) = N(μ, σ²)` trained with an ELBO (masked L1 reconstruction + KL divergence). At inference `z = μ`, fully deterministic and stable for streaming.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Training Strategy](#training-strategy)
- [Setup](#setup)
- [Dataset — `dataset.py`](#dataset--datasetpy)
- [Preprocessing — `preprocess.py`](#preprocessing--preprocesspy)
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

# 4. Preprocess: generate augmented audio tensors + cache mels on GPU
python preprocess.py --data-root datasets/wav48_silence_trimmed

python train.py                                                    # train
python mic_convert.py --checkpoint checkpoints/best.pt            # real-time
python convert.py --source me.wav --reference alice.wav --output out.wav # offline
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
├── preprocess.py               # GPU-accelerated mel preprocessing (optional, speeds up training)
├── train.py                    # Training loop (AMP + gradient accumulation)
├── convert.py                  # Offline file-to-file voice conversion
├── mic_convert.py              # Real-time microphone conversion (no server needed)
│
├── checkpoints/                # Created by train.py
│   ├── best.pt
│   ├── latest.pt
│   └── samples/step_N/         # Qualitative audio saved every SAVE_EVERY steps
│       ├── source.wav
│       ├── context.wav      
│       ├── target.wav
│       └── converted.wav
│
├── core/
│   ├── modules/
│   │   ├── context_encoder.py  # Transformer, no PE/mean-pool → μ, log σ² → z [B, T_ctx, 256]
│   │   ├── content_encoder.py  # DeepFilterNet3 + HuBERT (InstanceNorm) + torchcrepe F0
│   │   ├── cross_attention.py  # Q=content, K/V=C, key_padding_mask
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
│   Context Encoder   │  Transformer (4 layers), no positional encoding
│     (Trainable)     │  [B, 100, T_ctx] → μ, log σ² → z [B, T_ctx, 256]
│                     │  (training: z sampled; inference: z = μ, deterministic)
└─────────────────────┘
        │  z (computed once, cached per speaker)
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
│  Cross-Attention    │  Q=content, K=V=C sequence (TNP style)
│    (Trainable)      │  each content frame attends over all T_ctx ref frames
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
| ContextEncoder (Transformer + μ/log σ² heads) | Yes | ~3.5 M |
| CrossAttentionFusion | Yes | ~1.0 M |
| MelDecoder | Yes | ~2.6 M |
| ContentEncoder (DFN3 + HuBERT + crepe) | No | ~122 M |
| VocosVocoder | No | ~13.4 M |
| **Total trainable** | | **~7.1 M (~27 MB fp32)** |

---

## Training Strategy

### The non-parallel data problem

Voice conversion requires separating *what is said* (phonetic content) from *who says it* (speaker identity), then recombining them. The direct approach — comparing converted audio against a ground-truth recording of a different speaker saying the same sentence — requires **parallel data**: sentence-aligned recordings across every speaker pair. Parallel corpora are rare and expensive to collect.

VCTK and LibriSpeech are **non-parallel**: each speaker says different sentences. Computing a frame-wise L1 loss between "Speaker A saying *apple*" and "Speaker B saying *banana*" is meaningless — the mel spectrograms have nothing to compare.

### Self-reconstruction with offline augmentation

The training loop uses a **self-reconstruction** objective to sidestep the parallel-data problem. For each training sample, one speaker and one utterance are selected:

- `source_audio` — the **Parselmouth-augmented** version of the utterance (pitch + formant shifted). Fed into the content encoder.
- `audio_content` — the **clean** version of the same utterance. Used as the mel reconstruction target.
- `context_mels` — N_CTX=5 **clean** reference utterances from the same speaker (always different clips).

```
Training forward pass
─────────────────────────────────────────────────────────────────────
source_audio   ──→  [ContentEncoder (frozen)]  ──→  content features
  (augmented)       (HuBERT + InstanceNorm strips per-sample timbre)

context_mels   ──→  [ContextEncoder Transformer]
  (clean)           ──→  μ [B, T_ctx, 256],  log σ² [B, T_ctx, 256]
                    ──→  z = μ + ε·σ,  ε ~ N(0, I)   ← reparameterization trick

content + z    ──→  [CrossAttention + Decoder]  ──→  pred_mel

ELBO loss:  L1( pred_mel,  mel(audio_content) )          ← reconstruction
          + β · KL( q(z|C) || N(0,I) )                  ← KL divergence
          β: 0 for steps 0–5000, linear ramp → 0.01 by step 20000 (KL annealing)
```

Because prediction and ground truth come from the same speaker, the loss is phonetically valid. The augmented source has different pitch and formant characteristics from the clean context clips — the model cannot copy timbre from content features and must consult `C` to reconstruct the correct spectral shape.

At **inference time** the roles switch to the intended conversion task:

```
Inference forward pass
─────────────────────────────────────────────────────────────────────
source_audio  ──→  [ContentEncoder]  ──→  content features (source phonetics)
reference     ──→  [ContextEncoder]  ──→  C  (target speaker embedding)

content + C   ──→  [CrossAttention + Decoder]  ──→  converted mel
```

### Why HuBERT makes this work

HuBERT is pre-trained with a masked-prediction objective on large-scale speech, so its internal representations correlate with phonemes more than with speaker acoustics. Feeding `audio_content` through HuBERT produces features that carry the *content* of the utterance while discarding much of the speaker-specific spectral shape. Without that bottleneck the model could learn to copy the input and ignore `C` entirely.

### Copy-synthesis risk

HuBERT layer-6 features are not perfectly speaker-agnostic — some identity leaks through. If cross-attention exploits that residual signal, the model learns *copy synthesis*: reconstruction loss looks good but cross-speaker conversion fails at inference because `C` is never truly consulted.

**Signs:** converted audio sounds like the source, not the target; validation loss is low but listening tests are poor.

**Mitigation 1 — HuBERT Instance Normalization:**
The content encoder applies `F.instance_norm` to HuBERT features before the F0 concat. It normalises each `(sample, channel)` pair to mean=0 / std=1 across the time dimension, stripping the per-sample spectral bias that encodes speaker timbre. This is a hard constraint: the model *cannot* reconstruct speaker identity from content features alone and must consult `C` via cross-attention. See the Implementation Notes for technical details.

**Mitigation 2 — Offline Pitch + Formant Augmentation:**
`preprocess.py` pre-generates one Parselmouth-augmented variant per audio file (`_aug.pt`). Pitch and formant shift directions are sampled independently at random: pitch `Uniform(1.10, 1.30)` or `Uniform(0.70, 0.90)`; formant `Uniform(1.05, 1.15)` or `Uniform(0.85, 0.95)`. During training, the augmented audio is fed into the content encoder while the clean audio is used as the reconstruction target. The pitch shift forces the model to recover the correct pitch from context `z` rather than copying it from content features; the formant shift alters the vocal tract shape, making it harder to extract speaker identity from the content stream.

**Mitigation 3 — Variational KL Bottleneck (Stochastic TNP):**
The context encoder applies a variational bottleneck after its Transformer layers. Instead of producing a single deterministic representation, it outputs `μ` and `log σ²` (each `[B, T_ctx, 256]`) and samples `z = μ + ε·σ` via the reparameterization trick. The ELBO adds a KL divergence penalty between `q(z|C) = N(μ, σ²)` and the isotropic prior `p(z) = N(0, I)`:

```
KL = -0.5 · mean( 1 + log σ² − μ² − σ² )   [closed-form, per valid frame]
```

This regularises the context representation toward the prior — the encoder cannot produce arbitrarily sharp, data-memorising embeddings. At inference `z = μ` exactly (model in `eval()` mode), so the pipeline is fully deterministic and streaming-safe. The two linear projection heads (`μ_proj`, `log σ²_proj`) are initialised as near-identity so resuming from a deterministic checkpoint causes zero disruption to learned representations on the first step.

**KL Annealing** prevents posterior collapse (the encoder collapsing to the prior with `μ = 0`, `σ = 1` to zero out the KL term): β starts at 0 for the first 5 000 steps (pure reconstruction warmup), then linearly ramps to `β_max = 0.01` by step 20 000 and remains constant thereafter.

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

For each training sample it picks a speaker and an utterance from that speaker:

- `source_audio` — the **Parselmouth-augmented** version of that utterance (`_aug.pt`), loaded as a 16 kHz audio tensor. Feeds into the content encoder. Falls back to the clean file if the augmented tensor has not been generated yet.
- `audio_content` — the **clean** version of the same utterance. Used as the reconstruction target (mel ground truth).
- `context_mels` — `N_CTX=5` **clean** reference utterances from the same speaker (always different clips), loaded from cached mel `.pt` files where available.

Context mels are computed at 24 kHz (100-band log-mel, `n_fft=1024`, `hop=256`, `power=1.0`) to match the Vocos vocoder.

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
print(batch["source_audio"].shape)   # [B, T_src]  — zero-padded to batch max
print(batch["audio_content"].shape)  # [B, T_content] — target speaker content clip
print(batch["context_mels"].shape)   # [B, N_CTX, 100, T_ctx]
print(batch["source_lengths"])       # list[B]: unpadded sample count per source
print(batch["content_lengths"])      # list[B]: unpadded sample count per content clip
print(batch["ctx_mel_lens"])         # list[B] of list[N_CTX]: unpadded T per ref mel
```

</details>

---

## Preprocessing — `preprocess.py`

`preprocess.py` runs two phases in sequence. Both are **safe to interrupt and resume** — existing `.pt` files are skipped automatically.

```bash
python preprocess.py --data-root datasets/wav48_silence_trimmed
python preprocess.py --data-root datasets/wav48_silence_trimmed --batch-size 64  # faster on high-VRAM GPUs
python preprocess.py --data-root datasets/wav48_silence_trimmed --skip-aug       # mel cache only
```

### Phase 1 — Parselmouth augmentation (CPU)

For every audio file, one pitch+formant-shifted variant is created with Praat's *Change gender* algorithm and saved as a 16-kHz audio tensor alongside the source:

```
p225_001_mic1.flac  →  p225_001_mic1_aug.pt   (float32 tensor, 16 kHz)
```

This process is CPU-bound and automatically runs in parallel across multiple CPU cores using a `ProcessPoolExecutor`. You can control the number of parallel processes using the `--num-workers` flag.

The shift direction for pitch and formant is sampled independently and at random:

| Parameter | High direction | Low direction |
|---|---|---|
| Pitch ratio | `Uniform(1.10, 1.30)` | `Uniform(0.70, 0.90)` |
| Formant ratio | `Uniform(1.05, 1.15)` | `Uniform(0.85, 0.95)` |

`dataset.py` loads `_aug.pt` as `source_audio` (the content encoder input) and the original clean file as `audio_content` (the reconstruction target). If `_aug.pt` does not exist, clean audio is used as a fallback.

### Phase 2 — GPU mel cache (batched)

Computing mel spectrograms on the CPU inside the DataLoader is the primary bottleneck on large datasets (VCTK: ~44 000 files, >2 hours per epoch on CPU). Phase 2 converts every **clean** audio file to a cached log-mel tensor on the GPU, reducing `dataset.py` context-mel loading to a simple `torch.load` call.

1. Loads waveforms on the CPU with `soundfile.read` (batched, pin_memory)
2. Resamples to 24 kHz and applies `MelSpectrogram` on the GPU in a single batched pass
3. Trims padding, applies `log(mel.clamp(1e-7))`, saves as `<stem>.pt` next to the source file

<details>
<summary>CLI flags</summary>

| Flag | Default | Description |
|---|---|---|
| `--data-root` | *(required)* | Root directory of audio files |
| `--batch-size` | `32` | GPU batch size for Phase 2 — increase to fill VRAM |
| `--num-workers` | `8` | CPU worker count for Phase 1 (augmentation) and Phase 2 (DataLoader) |
| `--seed` | `42` | RNG seed for augmentation direction sampling |
| `--skip-aug` | off | Skip Phase 1 and run mel cache only |

</details>

<details>
<summary>Disk space</summary>

**Phase 1 (`_aug.pt`):** Each file stores a `float32` audio tensor at 16 kHz. For a 5-second clip: `5 × 16 000 × 4 bytes ≈ 320 KB` per file. Full VCTK adds roughly **14 GB**.

**Phase 2 (`<stem>.pt`):** Each file stores a `float32` mel tensor `[100, T_mel]`. For a 5-second clip: `100 × 470 × 4 bytes ≈ 188 KB`. Full VCTK adds roughly **8–10 GB**.

</details>

---

## Training — `train.py`

Trains ContextEncoder, CrossAttentionFusion, and MelDecoder with bfloat16 AMP and gradient accumulation. ContentEncoder and Vocos remain frozen throughout.

```bash
python train.py                               # VCTK default
python train.py --data-root datasets/my_data  # custom dataset
python train.py --reset                       # ignore existing checkpoint
```

Checkpoints are written to `checkpoints/`:
- `latest.pt` — every 1 000 steps, used for resuming
- `best.pt` — whenever validation loss improves, used for inference

To resume training from the last checkpoint, just run `python train.py` again — it picks up `checkpoints/latest.pt` automatically.

### Training log CSV

Every 50 steps (`CSV_LOG_EVERY`), a row is appended to `checkpoints/training_log.csv`:

| Column | Description |
|---|---|
| `step` | Optimizer step number |
| `train_loss` | Average masked L1 reconstruction loss over the last 50 steps |
| `val_loss` | Most recent validation loss (carries forward between checkpoints) |
| `learning_rate` | Current LR after warmup / cosine schedule |
| `kl_loss` | Average KL divergence (per valid frame-element) over the last 50 steps |
| `beta` | Current KL annealing weight β (0 → 0.01 between steps 5 000 and 20 000) |

The file is created with a header on first run and **appended** on resume — rows are never overwritten. Change the filename with `--csv-log`:

```bash
python train.py --csv-log run2.csv   # writes to checkpoints/run2.csv
```

Quick inspection:

```python
import pandas as pd
df = pd.read_csv("checkpoints/training_log.csv")
df.plot(x="step", y=["train_loss", "val_loss"])
```

### Qualitative audio samples

Every `SAVE_EVERY` steps, validation saves four WAV files to `checkpoints/samples/step_{N}/`:

| File | Content |
|---|---|
| `source.wav` | Augmented source audio from the first validation batch (16 kHz) |
| `target.wav` | Clean audio content from the first validation batch (16 kHz) |
| `context.wav`| First clean reference context clip used for speaker conditioning, decoded through Vocos (24 kHz) |
| `converted.wav` | Augmented source content + clean context C, decoded through Vocos (24 kHz) |

`converted.wav` mirrors the training path: augmented audio into the content encoder, clean context clips as speaker reference. Use it to track how well the model reconstructs clean speech from perturbed input. With F0 shifting applied, the converted output should match the clean target's pitch register.

<details>
<summary>CLI flags and key constants</summary>

| Flag | Default | Description |
|---|---|---|
| `--data-root` | `datasets/VCTK-Corpus-0.92/wav48_silence_trimmed` | Speaker folder root |
| `--output-dir` | `checkpoints` | Checkpoint and sample output directory |
| `--num-workers` | `8` | DataLoader worker processes |
| `--reset` | off | Train from scratch, ignoring existing checkpoint |
| `--csv-log` | `training_log.csv` | CSV filename inside `--output-dir` for loss logging |

Key constants at the top of `train.py`:

```python
BATCH_SIZE    = 32       # physical batch per GPU step
GRAD_ACCUM    = 2        # effective batch = BATCH_SIZE × GRAD_ACCUM = 64
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
[1024 samples/callback]   [accumulate 2560]   [5-chunk pre-fill]
```

| Stage | Time |
|---|---|
| Mic accumulation (2 560 samples) | ~160 ms |
| GPU inference (DFN3 + HuBERT + decode + vocoder) | ~25 ms |
| Jitter buffer pre-fill (5 chunks) | ~320 ms |
| **Total steady-state** | **~210 ms** |

`PROC_SAMPLES` must be a multiple of **2 560** to avoid fractional Vocos mel frames (see *Streaming chunk size constraint* in Implementation Notes). Vocos outputs at 24 kHz; the processing thread resamples back to 16 kHz for playback so you can use a standard 16 kHz output device.

</details>

---python convert.py --source me.wav --reference alice.wav --output converted.wav

## Offline Inference — `convert.py`

Converts an audio file without a microphone or server. Useful for evaluating a checkpoint before moving to real-time use.

```bash

python convert.py --source me.wav --reference alice_1.wav alice_2.wav alice_3.wav --output converted.wav
```

**F0 shifting** is applied automatically: `convert.py` extracts F0 statistics from both the source audio and all reference files (concatenated), then maps the source speaker's pitch contour into the target speaker's fundamental frequency range. This is printed at runtime:

```
F0 shifting: src=120.3±18.4 Hz → tgt=210.7±22.1 Hz
```

If torchcrepe is not installed or either speaker has no voiced frames, F0 shifting is skipped silently and a message is printed instead.

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
The content encoder (DFN3 + HuBERT + crepe) operates at **16 kHz**. The mel computation and Vocos vocoder operate at **24 kHz**. Audio is resampled 16 kHz → 24 kHz before the mel transform; the vocoder outputs native 24 kHz audio. Do not change the mel parameters (`n_fft=1024`, `hop_length=256`, `n_mels=100`, `power=1.0`) — they must match Vocos's training configuration exactly.

**DeepFilterNet3 runs at 48 kHz**
DFN3 operates internally at 48 kHz. The content encoder resamples around it:
```
16 kHz → 48 kHz → DeepFilterNet3 → 16 kHz → HuBERT
```
Feeding 16 kHz directly into DFN3 produces silent garbage with no error.

**DeepFilterNet3 GRU state**
`reset_dfn_state()` must be called **once per audio session** — when a WebSocket connection opens or when `mic_convert.py` starts. Do not call it between chunks; the GRU hidden state carries temporal context across chunks.

In streaming mode the DFN GRU must only advance over **new** audio. `mic_convert.py` calls `content_encoder._denoise()` on the raw non-overlapping chunk before prepending the denoised overlap for HuBERT context. Feeding the overlap prefix into DFN would process the same timeline twice, destroying the GRU state. The dedicated `VoiceConversionModel.convert_chunk_streaming()` method accepts pre-denoised audio and takes `skip_denoise=True` through to `ContentEncoder.forward()` so DFN is never called twice.

**HuBERT layer index**
`HUBERT_BASE.extract_features()` returns a list of 12 tensors (one per transformer layer). Index `5` (0-based) is transformer layer 6 — the correct layer for mid-level linguistic content.

**Audio loading — soundfile, not torchaudio.load**
`convert.py` and `mic_convert.py` load audio with `soundfile.read()` directly. Recent versions of torchaudio default to `torchcodec` as the backend, which is not installed in this environment. `soundfile` natively handles WAV, FLAC, OGG, and AIFF without any codec dependency. `dataset.py` has always used `soundfile`; the inference scripts were updated to match.

**Offline pitch + formant augmentation**
`preprocess.py` Phase 1 generates `<stem>_aug.pt` — a 16-kHz audio tensor produced by Praat's *Change gender* algorithm with randomised pitch and formant ratios. Pitch direction (up or down) and formant direction are sampled independently so the combination covers a wide range of voice characteristics. The augmented tensor is stored once and loaded by `dataset.py` at training time, avoiding the cost of running Parselmouth or `torchaudio.functional.pitch_shift` inside the training loop. If `_aug.pt` is missing for a given file, `_load_aug()` silently falls back to clean audio.

**F0 log scaling and cross-speaker Z-score shift**
Raw F0 from torchcrepe spans `[0, 2006]` Hz while HuBERT features fall roughly in `[-3, +3]`. `torch.log1p(f0)` maps F0 to `[0, ~7.6]`, preventing it from dominating the linear projection.

`ContentEncoder.forward()` accepts an optional `f0_stats=(src_mean, src_std, tgt_mean, tgt_std)` tuple. When provided, voiced frames (f0 > 0) are Z-score shifted before `log1p`:
```python
voiced_mask = (f0 > 0.0).float()
f0_shifted = (f0 - src_mean) / (src_std + 1e-5) * tgt_std + tgt_mean
f0 = voiced_mask * f0_shifted.clamp(min=0.0) + (1.0 - voiced_mask) * f0
```
Unvoiced frames (f0 == 0) are left unchanged. This shifts the source speaker's pitch contour into the target speaker's fundamental frequency range — without it, a male-to-female conversion will have the correct timbre but the wrong pitch register.

`f0_stats` is used in three places:
- **`convert.py`** — extracted from source and concatenated reference audio once before the chunk loop; applied to every chunk.
- **`mic_convert.py`** — target stats extracted from reference at startup; source stats tracked via EMA over the streaming session, applied after `STATS_WARMUP=15` chunks.
- **Training `_validate` samples** — extracted from the unpadded source and target audio of the first validation batch item; applied only to the `converted.wav` sample, not to the loss computation.

**F0 decoder — argmax, not Viterbi**
`torchcrepe` is configured with `decoder=torchcrepe.decode.argmax`. Viterbi is a global dynamic-programming algorithm that requires the full sequence and is incompatible with chunk-by-chunk streaming. Argmax is frame-independent and causal.

**F0 frame alignment**
`torchcrepe.predict(..., hop_length=320)` and HuBERT both have a 320-sample stride, producing `T // 320` frames each. If you ever change one, change both — mismatched strides cause silent feature misalignment at concatenation.

**Context encoder — variational bottleneck, no positional encoding, no mean pooling**
Speaker identity (timbre, vocal tract shape) is time-invariant. Absolute positional encoding was removed so the model cannot overfit on the temporal position of phonemes in the reference clip. Mean pooling was also removed: `ContextEncoder` outputs a full sequence so `CrossAttentionFusion` can attend over all reference frames (TNP style).

After the Transformer and `LayerNorm`, two linear heads project to `μ [B, T_ctx, 256]` and `log σ² [B, T_ctx, 256]`. The sampled latent `z = μ + ε·σ` replaces the old deterministic output as the K/V sequence for cross-attention. In `eval()` mode `z = μ` — no randomness in inference. The `log σ²` output is clamped to `[-10, 4]` (σ ∈ `[0.007, 7.4]`) to prevent numerical explosion before the KL `exp()`.

The two new projection heads are initialised specially to avoid disrupting a resumed checkpoint: `μ_proj` starts as an identity matrix (so `μ = x` on step 0), `log σ²_proj` starts with zero weights and bias `−5` (so `σ ≈ 0.08` — nearly no noise). The training signal immediately corrects these from the first gradient step.

**HuBERT Instance Normalization**
`F.instance_norm` is applied to HuBERT features `[B, 768, T_frames]` before the F0 concat. It normalises each `(sample, channel)` pair to mean=0 / std=1 across the time dimension, stripping the per-sample spectral bias that encodes speaker timbre. Combined with the offline pitch+formant augmentation, this is the primary hard constraint preventing copy-synthesis from the content stream.

**Masked L1 loss**
The training loss is computed only over valid (non-padded) mel frames. `collate_fn` returns `content_lengths` (original audio sample counts for the content clip); the training loop converts these to mel frame counts and builds a boolean mask before calling `F.l1_loss(..., reduction="none")`. Padding regions contain `log(1e-7) ≈ −16.1` and would otherwise waste model capacity if included in the loss.

**ELBO — KL divergence and annealing**
The total training loss is the ELBO: `recon_loss + β · kl_loss`. The KL term is also masked — padded context positions are excluded via `~ctx_mask` before summing:
```python
kl_raw   = -0.5 * (1.0 + lv_f - mu_f.pow(2) - lv_f.exp())  # [B, N*T_ctx, d_model]
valid    = (~ctx_mask).float()                                # 1 = valid, 0 = padding
kl_loss  = (kl_raw * valid.unsqueeze(-1)).sum() / (valid.sum() * D_MODEL + 1e-8)
```
KL is computed in float32 (explicit `.float()` cast inside the bfloat16 autocast block) to avoid precision loss in the `exp()` call. `β = get_beta(step)` implements linear annealing: 0 for steps 0–5 000, linear ramp to 0.01 by step 20 000, constant at 0.01 thereafter. Both `recon_loss` and the KL term are divided by `GRAD_ACCUM` before `.backward()` to keep gradient scale correct under accumulation.

**Batch padding and lengths**
`collate_fn` zero-pads `source_audio` and `audio_content` to the batch maximum length. `source_lengths` and `content_lengths` are returned alongside the padded tensors so callers can recover the unpadded region. When saving validation audio samples, `source[0, :source_lengths[0]]` and `content_audio[0, :content_lengths[0]]` are used — feeding the full padded tensor (including zero-padding) into HuBERT causes garbage features for the silent tail, producing silence in the converted output after the real audio ends.

**Vocos input format**
The decoder outputs `[B, T_mel, 100]` (channels-last). Vocos expects `[B, 100, T_mel]` (channels-first). Always transpose before calling the vocoder: `vocoder(mel.transpose(1, 2))`.

**Mel decoder upsample factor**
HuBERT produces ~50 frames/s (stride 320 @ 16 kHz). Vocos requires ~93.75 frames/s (24 000 Hz / hop 256). The decoder uses `scale_factor = 24000 / (256 × 50) = 1.875` to bridge this gap.

**Streaming chunk size constraint**
`nn.Upsample(scale_factor=1.875)` floors its output size. For N HuBERT frames, the Vocos input has `floor(N × 1.875)` mel frames, which is only an integer when `N` is a multiple of 8 (since 1.875 = 15/8). The streaming constants are chosen so that:
```
PROC_SAMPLES = 2560 → 2560 / 320 = 8 HuBERT frames
8 × 1.875 = 15 Vocos mel frames  (exact — no floor loss)
15 × hop 256 = 3840 @ 24 kHz  →  × (16/24) = 2560 @ 16 kHz = PROC_SAMPLES ✓
```
The 1 280-sample HuBERT overlap is trimmed at the **content tensor** level (4 frames discarded before the decoder), so the decoder always sees exactly 8 frames regardless of overlap size. Do not set `PROC_SAMPLES` to a value where `PROC_SAMPLES / 320` is not a multiple of 8 — it will cause a fractional mel frame count, floor rounding, and a permanent per-chunk sample deficit that accumulates as audio drift.

**Gradient accumulation — scale all loss terms**
The training loss is divided by `GRAD_ACCUM` before `.backward()`. If you add a second loss term (e.g. speaker loss), apply the same scaling: `(l1 + 0.1 * spk_loss) / GRAD_ACCUM`. Forgetting this makes the effective learning rate `GRAD_ACCUM×` too large.

**bfloat16 AMP — no GradScaler needed**
Training uses `torch.bfloat16` via `torch.amp.autocast`. Unlike float16, bfloat16 has the same exponent range as float32 so activations never overflow to `inf`/`NaN`. `GradScaler` has been removed — it only exists to work around float16 underflow and is a no-op for bfloat16.

**HuBERT normalization — training vs streaming**
HuBERT features must be normalised per-channel to strip residual speaker timbre. The mechanism differs between training and streaming:

*Training* — `F.instance_norm` over the full sequence: normalises each `(sample, channel)` pair across all T frames, giving mean=0 / std=1 per channel. Stable because sequences are 100–400 frames long.

*Streaming* — only 8 frames per chunk. Computing `F.instance_norm` on 8 frames is catastrophic: during silence the denominator is near-zero and the output explodes; even during speech the statistics are wildly noisy frame-to-frame. Instead, `ContentEncoder.forward()` accepts `hubert_stats=(mean, std)` as `[1, 1, 768]` tensors:
```python
hubert_norm = (hubert_feat - hub_mean) / (hub_std + 1e-5)
```
`mic_convert.py` maintains a per-channel EMA (`momentum=0.95`, τ ≈ 3 s) of the raw HuBERT channel statistics, updated each chunk via `content_encoder.extract_streaming_stats()`. The EMA stats are passed into `convert_chunk_streaming()` → `content_encoder.forward()` after a `STATS_WARMUP=15` chunk (~2.4 s) period, during which the pipeline falls back to per-chunk `instance_norm`.

**TNP training / inference consistency**
At inference `encode_references()` calls `self.eval()` first (so `z = μ`, deterministic), then **concatenates** N reference sequences along the time axis: `torch.cat(encoded, dim=1)` → `[1, N·T_ctx, 256]`. The training loop mirrors this exactly — context encoder output is reshaped, not averaged:
```python
# train.py — context encoder returns (z, mu, log_var), each [B*N, T_ctx, d_model]
C_all, mu_all, lv_all = model.context_encoder(ctx_flat, src_key_padding_mask=ctx_enc_mask)
C    = C_all.view(B, N * T_ctx_enc, -1)   # [B, N·T_ctx, d_model]  ← concatenate, not mean
mu_C = mu_all.view(B, N * T_ctx_enc, -1)  # for KL computation
lv_C = lv_all.view(B, N * T_ctx_enc, -1)
```
During validation the model is in `eval()`, so `z = μ` and `_validate()` discards `mu_all`/`lv_all` with `C_all, _, _ = ...`.

**Two separate padding masks for context — self-attention and cross-attention**
Context mels are zero-padded in two places: within `__getitem__` (different mels padded to the item-level maximum T) and in `collate_fn` (items padded to the batch-level maximum T). Padding must be excluded from *both* the context encoder's self-attention and the cross-attention fusion. These require masks of different shapes and must be built separately.

`dataset.py` records the unpadded length of every reference mel as `ctx_mel_lens: list[N_CTX]` per item. `collate_fn` aggregates it into `ctx_mel_lens: list[B][N_CTX]`. The training loop builds both masks:
```python
# ── 1. Context encoder self-attention mask — shape [B*N, T_ctx] ──────────────
# ctx_flat is [B*N, N_MELS, T_ctx]: flat index i*N+n = reference n of item i.
ctx_enc_mask = torch.ones(B * N, T_ctx, dtype=torch.bool, device=device)
for i in range(B):
    for n in range(N):
        ctx_enc_mask[i * N + n, :ctx_mel_lens[i][n]] = False
C_all = model.context_encoder(ctx_flat, src_key_padding_mask=ctx_enc_mask)

# ── 2. Cross-attention key_padding_mask — shape [B, N*T_ctx] ─────────────────
# C is [B, N*T_ctx, d_model]: N reference sequences concatenated along time.
ctx_mask = torch.ones(B, N * T_ctx_enc, dtype=torch.bool, device=device)
for i in range(B):
    for n in range(N):
        ctx_mask[i, n * T_ctx_enc : n * T_ctx_enc + ctx_mel_lens[i][n]] = False
model.cross_attention(content, C, key_padding_mask=ctx_mask)
```
Without the self-attention mask, zero-padded frames in the context encoder corrupt the speaker embeddings before they even reach cross-attention. At inference, `encode_references()` processes one utterance at a time — no padding, both masks are `None`.

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
from core.modules.context_encoder import ContextEncoder

m     = VoiceConversionModel(torch.device("cuda"))
audio = torch.randn(1, 3200).cuda()

# Variational context encoder: forward() returns (z, mu, log_var)
enc  = ContextEncoder().cuda().eval()
mel  = torch.randn(1, 100, 150).cuda()
z, mu, lv = enc(mel)
print("z shape:", z.shape)          # [1, 150, 256]
print("z == mu at eval:", torch.allclose(z, mu))   # True

# Full pipeline via pre-cached context
C   = torch.randn(1, 750, 256).cuda()   # [1, T_ctx, D_MODEL]
out = m.convert_chunk(audio, C)
print("Output shape:", out.shape)   # [1, 1, T_wav @ 24 kHz]
EOF

# Dataset smoke test
python - <<'EOF'
from dataset import SpeakerDataset
ds = SpeakerDataset("datasets/VCTK-Corpus-0.92/wav48_silence_trimmed", split="train")
s  = ds[0]
print("source_audio:", s["source_audio"].shape)
print("audio_content:", s["audio_content"].shape)
print("context_mels:", s["context_mels"].shape)   # [N_CTX, 100, T_ctx]
print("ctx_mel_lens:", s["ctx_mel_lens"])          # list[N_CTX] of ints
EOF
```

---

## License

For research and personal use. Pre-trained models (HuBERT, Vocos, DeepFilterNet) are subject to their respective upstream licenses.
