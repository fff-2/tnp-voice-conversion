# Real-Time Voice Conversion Pipeline — Deterministic TNP-D

Few-shot, real-time voice conversion for both **speech and singing**. Record a few seconds of a target speaker — the system converts your live microphone input into that voice with low latency. The model is trained on a mixture of VCTK speech data and JVS-MuSiC singing data, enabling vocal conversion across speaking and singing registers.

**Architecture:** Deterministic Transformer Neural Process (TNP-D) — the context encoder learns a content→acoustic mapping function directly from (ContentVec+F0, mel) context pairs extracted from reference utterances. No stochastic sampling; training optimises a single masked L1 reconstruction loss.

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

python train.py --reset                                            # train from scratch
python mic_convert.py --checkpoint checkpoints/best.pt            # real-time
                                 --output out.wav # offline
```

---

## Project Structure

<details>
<summary>File map</summary>

```
voice/
├── environment.yml             # Conda environment (Python 3.12, PyTorch + CUDA 12.8)
├── datasets/                   # All datasets go here
├── augmentation.py             # Parselmouth pitch+formant augmentation (used by preprocess.py)
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
│   │   ├── context_encoder.py  # Transformer, no PE/mean-pool; (content, mel) pairs → z [B, T_h, 256]
│   │   ├── content_encoder.py  # DeepFilterNet3 + ContentVec (InstanceNorm) + torchcrepe F0
│   │   ├── cross_attention.py  # Q=hubert_proj(h)+f0_proj(f0), K/V=C, separate projections
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
TARGET SPEAKER (N reference utterances)
        │
        ├──→ [ContentEncoder: ContentVec+F0]  ──→ content_ctx [B*N, T_h, 769]
        │         (frozen, no grad)
        └──→ [Mel transform, downsample]  ──→ mel_ctx     [B*N, T_h, 100]
                                                               │
                                            concat ──→ ctx_pairs [B*N, T_h, 869]
                                                               │
                                                               ▼
                                                 ┌─────────────────────┐
                                                 │   Context Encoder   │  Transformer (4 layers)
                                                 │     (Trainable)     │  no positional encoding
                                                 │                     │  [B*N, T_h, 869] → z [B*N, T_h, 256]
                                                 └─────────────────────┘
                                                        │  C = reshape to [B, N·T_h, 256]
                                                        │  (computed once, cached per speaker)
                                                        ▼
SOURCE SPEAKER (augmented audio)
        │
        ▼
┌─────────────────────┐
│   Content Encoder   │  DeepFilterNet3 → ContentVec (last layer) → torchcrepe F0
│      (Frozen)       │  [B, T @ 16 kHz] → [B, T_frames, 769]
└─────────────────────┘
        │  content_tgt (Query)
        ▼
┌─────────────────────┐
│  Cross-Attention    │  Q = content_tgt,  K = V = C  (TNP style)
│    (Trainable)      │  Q: hubert_proj(h) + f0_proj(f0)
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
| ContextEncoder (Transformer, input 869→256) | Yes | ~3.2 M |
| CrossAttentionFusion (hubert_proj + f0_proj + MHA + FFN) | Yes | ~1.0 M |
| MelDecoder | Yes | ~2.6 M |
| ContentEncoder (DFN3 + ContentVec + crepe) | No | ~94 M |
| VocosVocoder | No | ~13.4 M |
| **Total trainable** | | **~6.8 M (~26 MB fp32)** |

**Temporal alignment:** ContentVec outputs at 50 fps (stride 320 @ 16 kHz). Mel is computed at ~93.75 fps (hop 256 @ 24 kHz). Reference mels are downsampled to ContentVec rate via `F.interpolate(mode='linear')` inside `model.forward()` before concatenation with content features.

---

## Training Strategy

### The non-parallel data problem

Voice conversion requires separating *what is said* (phonetic content) from *who says it* (speaker identity), then recombining them. The direct approach — comparing converted audio against a ground-truth recording of a different speaker saying the same sentence — requires **parallel data**: sentence-aligned recordings across every speaker pair. Parallel corpora are rare and expensive to collect.

VCTK and LibriSpeech are **non-parallel**: each speaker says different sentences. Computing a frame-wise L1 loss between "Speaker A saying *apple*" and "Speaker B saying *banana*" is meaningless — the mel spectrograms have nothing to compare.

### Self-reconstruction with offline augmentation

The training loop uses a **self-reconstruction** objective to sidestep the parallel-data problem. For each training sample, one speaker and one utterance are selected:

- `source_audio` — the **Parselmouth-augmented** version of the utterance (pitch + formant shifted). Used to extract **phonetic content** (ContentVec).
- `audio_content` — the **clean** version of the same utterance. Used as the reconstruction target (mel ground truth) and to extract the **true pitch contour** (F0).
- `context_audios` — N_CTX=2 **clean** reference utterances from the same speaker (always different clips). Each provides a (ContentVec+F0, mel) pair showing how the target speaker maps content to acoustics.

```
Training forward pass (TNP-D)
─────────────────────────────────────────────────────────────────────
context_audios ──→ [ContentEncoder: ContentVec+F0]  ──→ content_ctx [B*N, T_h, 769]
  (clean refs)  └→ [Mel transform + downsample]  ──→ mel_ctx     [B*N, T_h, 100]
                                                            │
                                                 concat → ctx_pairs [B*N, T_h, 869]
                                                            │
                                              ┌─────────────────────┐
                                              │  Context Encoder    │  Learns the mapping rule
                                              │  (Transformer)      │  content → acoustic
                                              └─────────────────────┘
                                                            │ C  [B, N·T_h, 256]

source_audio   ──→ [ContentEncoder: ContentVec]  ──→  phonetic features  (Query)
  (augmented)          (InstanceNorm strips per-sample timbre)

audio_content  ──→ [ContentEncoder: Crepe]   ──→  F0 (true pitch contour)  (Query)
  (clean)

phonetics + F0 ──→ [CrossAttention + Decoder]  ──→  pred_mel
   + C (context)

Loss: Masked L1( pred_mel,  mel(audio_content) )          ← reconstruction only
```

Because prediction and ground truth come from the same speaker, the loss is phonetically valid. The augmented source has different pitch and formant characteristics from the clean context clips — the model cannot copy timbre from content features and must consult `C` to reconstruct the correct spectral shape.

At **inference time** the roles switch to the intended conversion task:

```
Inference forward pass
─────────────────────────────────────────────────────────────────────
reference_audios  ──→  [ContentEncoder + Mel]  ──→  C  (target speaker context)
source_audio      ──→  [ContentEncoder]        ──→  content features (source phonetics)

content + C   ──→  [CrossAttention + Decoder]  ──→  converted mel
```

### Why TNP-D: learning a mapping function, not an embedding

A classic speaker embedding (d-vector, x-vector) compresses all N reference utterances into a single fixed-size vector. That vector must encode the speaker's full acoustic identity in limited dimensions, discarding temporal detail.

TNP-D instead treats the N reference utterances as a **context set** of input→output pairs: each pair `(content_ctx_i, mel_ctx_i)` is a direct observation of how the target speaker maps phonetic content to acoustic output at frame i. The ContextEncoder's Transformer attends over all these pairs jointly and outputs a full sequence `C [B, N·T_h, 256]` — a *function representation* rather than a fixed-size code. CrossAttentionFusion then uses the source content as a query against this function representation to predict the target speaker's mel for each source frame.

This formulation is strictly more expressive: the model can exploit fine-grained co-variation between content and acoustics in the reference set, rather than averaging it into a single vector.

### Why ContentVec makes this work

ContentVec is a fine-tuned HuBERT model trained with an explicit speaker disentanglement objective: a teacher model conditioned on a *different* speaker's voice guides the student to produce representations that are invariant to speaker identity. The resulting features correlate strongly with phonetic content while discarding speaker-specific spectral shape — making them a better content bottleneck than vanilla HuBERT for voice conversion. Unlike HuBERT layer-6 (which leaks some speaker identity), ContentVec's last layer is trained to suppress it by design.

### Copy-synthesis risk

ContentVec features still carry some residual speaker signal. If cross-attention exploits that residual, the model learns *copy synthesis*: reconstruction loss looks good but cross-speaker conversion fails at inference because `C` is never truly consulted.

**Signs:** converted audio sounds like the source, not the target; validation loss is low but listening tests are poor.

**Mitigation 1 — ContentVec Instance Normalization:**
The content encoder applies `F.instance_norm` to ContentVec features before the F0 concat. It normalises each `(sample, channel)` pair to mean=0 / std=1 across the time dimension, stripping the per-sample spectral bias that encodes speaker timbre. This is a hard constraint: the model *cannot* reconstruct speaker identity from content features alone and must consult `C` via cross-attention.

**Mitigation 2 — Offline Pitch + Formant Augmentation:**
`preprocess.py` pre-generates one Parselmouth-augmented variant per audio file (`_aug.pt`). Pitch and formant shift directions are sampled independently at random: pitch `Uniform(1.10, 1.30)` or `Uniform(0.70, 0.90)`; formant `Uniform(1.05, 1.15)` or `Uniform(0.85, 0.95)`. During training, this augmented audio is fed into the ContentVec content encoder. The aggressive pitch and formant shifts alter the vocal tract shape and fundamental frequency, effectively destroying the speaker identity in the source audio so it doesn't leak through the ContentVec features. Meanwhile, the *true* pitch contour is extracted from the clean target audio (`audio_content`) and provided directly to the decoder. This ensures the decoder learns to perfectly respect the F0 input (instead of blurring the pitch to minimize L1 error), while still being forced to rely entirely on `C` for the speaker's vocal tract and timbre.

**Mitigation 3 — TNP-D Context Pairs as an Explicit Mapping Bottleneck:**
By requiring the ContextEncoder to distil the target speaker's voice from (content, mel) pairs rather than mel alone, the model is forced to learn a *conditional* mapping rule. Any content information present in the source that was not seen in the reference context pairs cannot be attributed to the target speaker — the context set acts as an explicit prior over what acoustic features are speaker-specific vs. content-specific.

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
- `audio_content` — the **clean** version of the same utterance. Used as the reconstruction target (mel ground truth) and for F0 extraction.
- `context_audios` — N_CTX=2 **clean** reference utterances from the same speaker (always different clips), returned as raw 16 kHz waveforms. Used to extract (ContentVec+F0, mel) context pairs in `model.forward()`.
- `context_mels` — pre-cached log-mel spectrograms of the same N_CTX reference clips (loaded from `.pt` files where available). Kept in the batch for validation visualization only — not used in the forward pass.

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

**Option C — JVS-MuSiC** — 100 Japanese speakers, each with a unique solo-singing recording at 24 kHz.

Download from the [official release](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music) and place the extracted folder at `datasets/jvs_music_ver1/`. Then run the preparation script before preprocessing:

```bash
# 1. Segment song_unique/wav/raw.wav per speaker on silence/breath boundaries.
#    Deletes song_common/ (shared-song data) and outputs seg_001.wav … per speaker.
python prepare_jvs_music.py --data-root datasets/jvs_music_ver1

# 2. (Optional) dry-run first to preview segment counts
python prepare_jvs_music.py --data-root datasets/jvs_music_ver1 --dry-run

# 3. Cache augmentation + mel as usual
python preprocess.py --data-root datasets/jvs_music_ver1

# 4. Train on mixed speech + singing
python train.py --data-root datasets/jvs_music_ver1   # or combine with VCTK via a symlinked root
```

`prepare_jvs_music.py` produces **1,006 segments** (2–8 s each, 24 kHz) across 100 speakers. The 24 kHz native sample rate is preserved end-to-end by Phase 2 of `preprocess.py`, so no quality is lost compared with the downsampled 16 kHz → 24 kHz path used for VCTK. Since ContentVec and torchcrepe operate at 16 kHz, `dataset.py` resamples on load — no changes to the data pipeline are needed.

> **Why singing data matters:** The model's torchcrepe F0 extractor covers `[50, 1100]` Hz to include the soprano singing range (~1100 Hz). Training on JVS-MuSiC teaches the ContextEncoder to encode a singer's vocal tract shape and register from sung context clips, enabling cross-speaker singing voice conversion alongside speech conversion.

**Option D — Custom recordings**

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
| JVS-MuSiC (singing) | 100 | ~0.5 GB (after segmentation) | `datasets/jvs_music_ver1` |

</details>

<details>
<summary>Direct Python API</summary>

```python
from dataset import SpeakerDataset, collate_fn
from torch.utils.data import DataLoader

ds     = SpeakerDataset("datasets/VCTK-Corpus-0.92/wav48_silence_trimmed", split="train")
loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=True)

batch = next(iter(loader))
print(batch["source_audio"].shape)    # [B, T_src]  — zero-padded to batch max
print(batch["audio_content"].shape)   # [B, T_content]
print(batch["context_audios"].shape)  # [B, N_CTX, T_ctx]  — raw 16 kHz reference audio
print(batch["context_mels"].shape)    # [B, N_CTX, 100, T_mel]  — cached mels (visualization)
print(batch["content_lengths"])       # list[B]: unpadded sample count per content clip
print(batch["ctx_audio_lens"])        # list[B] of list[N_CTX]: unpadded sample count per ref
print(batch["ctx_mel_lens"])          # list[B] of list[N_CTX]: unpadded mel frames per ref
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
python train.py --reset                       # train from scratch (required: new architecture)
python train.py --data-root datasets/my_data  # custom dataset
python train.py                               # resume from latest.pt
```

> **Note:** Checkpoints from the previous VAE architecture are **incompatible** with the TNP-D model (`ContextEncoder.input_proj` changed from 256×100 to 256×869). Always use `--reset` when starting fresh after the architecture change.

Checkpoints are written to `checkpoints/`:
- `latest.pt` — every 1 000 steps, used for resuming
- `best.pt` — whenever validation loss improves, used for inference

### Training log CSV

Every 50 steps (`CSV_LOG_EVERY`), a row is appended to `checkpoints/training_log.csv`:

| Column | Description |
|---|---|
| `step` | Optimizer step number |
| `train_loss` | Average masked L1 reconstruction loss over the last 50 steps |
| `val_loss` | Most recent validation loss (carries forward between checkpoints) |
| `learning_rate` | Current LR after warmup / cosine schedule |

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
BATCH_SIZE    = 16       # physical batch per GPU step
GRAD_ACCUM    = 4        # effective batch = BATCH_SIZE × GRAD_ACCUM = 64
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
Microphone  →  mic_queue  →  Inference thread  →  out_queue  →  Speaker
[960 samples/callback]    [accumulate 4800]
```

| Stage | Time |
|---|---|
| Mic accumulation (`BLOCK` = 4 800 samples) | ~300 ms |
| GPU inference (DFN3 + ContentVec + decode + vocoder) | ~25 ms |
| **Total steady-state** | **~325 ms** |

Vocos outputs at 24 kHz; the inference thread resamples back to 16 kHz before writing to the output queue. Output is clipped to exactly `BLOCK` samples per iteration to prevent drift.

</details>

---

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
`VoiceConversionModel.train()` is overridden to re-call `.eval()` on `content_encoder` and `vocoder` immediately after `super().train()`. PyTorch's `.train()` propagates to all submodules — without this override it would accidentally enable dropout and BatchNorm in ContentVec and Vocos. If you add a new frozen submodule, add it to that override.

**Dual sample rates**
The content encoder (DFN3 + ContentVec + crepe) operates at **16 kHz**. The mel computation and Vocos vocoder operate at **24 kHz**. Audio is resampled 16 kHz → 24 kHz before the mel transform; the vocoder outputs native 24 kHz audio. Do not change the mel parameters (`n_fft=1024`, `hop_length=256`, `n_mels=100`, `power=1.0`) — they must match Vocos's training configuration exactly.

**DeepFilterNet3 runs at 48 kHz**
DFN3 operates internally at 48 kHz. The content encoder resamples around it:
```
16 kHz → 48 kHz → DeepFilterNet3 → 16 kHz → ContentVec
```
Feeding 16 kHz directly into DFN3 produces silent garbage with no error.

**DeepFilterNet3 GRU state**
`reset_dfn_state()` must be called **once per audio session** — when a WebSocket connection opens or when `mic_convert.py` starts. Do not call it between chunks; the GRU hidden state carries temporal context across chunks.

In streaming mode the DFN GRU must only advance over **new** audio. `mic_convert.py` calls `content_encoder._denoise()` on the raw non-overlapping chunk before prepending the denoised overlap for ContentVec context. Feeding the overlap prefix into DFN would process the same timeline twice, destroying the GRU state. The dedicated `VoiceConversionModel.convert_chunk_streaming()` method accepts pre-denoised audio and takes `skip_denoise=True` through to `ContentEncoder.forward()` so DFN is never called twice.

`model.forward()` calls `reset_dfn_state(batch_size=B*N)` before encoding N reference utterances, then `reset_dfn_state(batch_size=B)` before encoding source audio. This ensures independent GRU state for each call.

**ContentVec layer**
`HubertModel.from_pretrained("lengyue233/content-vec-best")` returns a standard `transformers` HuBERT model. `last_hidden_state` (the final transformer layer) is used — ContentVec is trained to maximise speaker disentanglement at the last layer, unlike speech HuBERT where layer 6 is preferred. Weights (~360 MB) are downloaded to `~/.cache/huggingface/` on first run.

**Audio loading — soundfile, not torchaudio.load**
`convert.py` and `mic_convert.py` load audio with `soundfile.read()` directly. Recent versions of torchaudio default to `torchcodec` as the backend, which is not installed in this environment. `soundfile` natively handles WAV, FLAC, OGG, and AIFF without any codec dependency. `dataset.py` has always used `soundfile`; the inference scripts were updated to match.

**Offline pitch + formant augmentation**
`preprocess.py` Phase 1 generates `<stem>_aug.pt` — a 16-kHz audio tensor produced by Praat's *Change gender* algorithm with randomised pitch and formant ratios. Pitch direction (up or down) and formant direction are sampled independently so the combination covers a wide range of voice characteristics. The augmented tensor is stored once and loaded by `dataset.py` at training time, avoiding the cost of running Parselmouth or `torchaudio.functional.pitch_shift` inside the training loop. If `_aug.pt` is missing for a given file, `_load_aug()` silently falls back to clean audio.

**F0 log scaling and cross-speaker Z-score shift**
Raw F0 from torchcrepe spans `[0, 1100]` Hz (fmax=1100 to cover singing soprano range) while ContentVec features fall roughly in `[-3, +3]`. `torch.log1p(f0)` maps F0 to `[0, ~7.6]` before it is passed to `f0_proj`.

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

**F0 routing — Training vs Inference**
To prevent the decoder from ignoring F0 inputs, it must receive the exact target pitch during training. `ContentEncoder.forward()` accepts an `f0_audio_16k` argument.
- *Training*: `f0_audio_16k` is set to the clean target audio (`audio_content`). The model routes the *true* pitch contour to the decoder without any math, teaching the decoder to trust and trace the F0 harmonics sharply.
- *Inference*: The target audio doesn't exist yet, so `f0_audio_16k` is omitted. The F0 is extracted from the source audio and mathematically shifted using the `f0_stats` Z-score calculation to match the target speaker's range. From the decoder's perspective, both pipelines provide a perfectly valid target pitch contour.

For reference context encoding, `content_encoder(ctx_flat, f0_audio_16k=ctx_flat)` — F0 is extracted from the same clean reference audio, no shift applied.

**Separate ContentVec / F0 projection in CrossAttentionFusion (Signal Drowning Fix)**
The content vector fed to `CrossAttentionFusion` has shape `[B, T_frames, 769]` — 768 ContentVec channels followed by 1 log-F0 channel. A naïve single `nn.Linear(769, d_model)` projection applies the same Xavier initialisation variance (`~1/769`) to all 769 inputs. Because ContentVec contributes 768 columns while F0 contributes only 1, the total output variance driven by ContentVec is **768× larger** than the F0 signal at step 0 — the F0 is effectively drowned out, and the model converges to a blurry local minimum that ignores pitch entirely.

To fix this, the projection is split into two independent heads:
```python
self.hubert_proj = nn.Linear(768, d_model)   # variance ∝ 1/768
self.f0_proj     = nn.Linear(1,   d_model)   # variance ∝ 1/1  ← 768× larger weights
Q = self.hubert_proj(content[..., :768]) + self.f0_proj(content[..., 768:])
```
The `f0_proj` fan-in of 1 means its weights are initialised ~28× larger in magnitude, amplifying the F0 signal to match the aggregate ContentVec signal power. The final expressible function is mathematically identical to a single `nn.Linear(769, d_model)` with manually set row-wise variances — but the split makes the initialisation automatic and PyTorch-idiomatic.

**F0 decoder — argmax, not Viterbi**
`torchcrepe` is configured with `decoder=torchcrepe.decode.argmax`. Viterbi is a global dynamic-programming algorithm that requires the full sequence and is incompatible with chunk-by-chunk streaming. Argmax is frame-independent and causal.

**F0 frame alignment**
`torchcrepe.predict(..., hop_length=320)` and ContentVec both have a 320-sample stride, producing `T // 320` frames each. If you ever change one, change both — mismatched strides cause silent feature misalignment at concatenation.

**Context encoder — deterministic TNP-D, no positional encoding, no mean pooling**
Speaker identity (timbre, vocal tract shape) is time-invariant. Absolute positional encoding was removed so the model cannot overfit on the temporal position of phonemes in the reference clip. Mean pooling was also removed: `ContextEncoder` outputs a full sequence so `CrossAttentionFusion` can attend over all reference frames (TNP style).

The input to `ContextEncoder` is a concatenated `(ContentVec+F0, mel)` pair of shape `[B, T_h, 869]`, where mel has been downsampled from 93.75 fps to ContentVec rate (50 fps) via `F.interpolate`. The `input_proj = nn.Linear(869, 256)` maps this to d_model. No variational bottleneck — the output `z` is fully deterministic, making both training and inference identical in behaviour.

**ContentVec Instance Normalization**
`F.instance_norm` is applied to ContentVec features `[B, 768, T_frames]` before the F0 concat. It normalises each `(sample, channel)` pair to mean=0 / std=1 across the time dimension, stripping any residual per-sample spectral bias. ContentVec's training objective already reduces speaker leakage; instance norm is a second hard constraint ensuring the content stream cannot bypass it.

**Masked L1 loss**
The training loss is computed only over valid (non-padded) mel frames. `collate_fn` returns `content_lengths` (original audio sample counts for the content clip); the training loop converts these to mel frame counts and builds a boolean mask before calling `F.l1_loss(..., reduction="none")`. Padding regions contain `log(1e-7) ≈ −16.1` and would otherwise waste model capacity if included in the loss.

**Batch padding and lengths**
`collate_fn` zero-pads `source_audio` and `audio_content` to the batch maximum length. `source_lengths` and `content_lengths` are returned alongside the padded tensors so callers can recover the unpadded region. When saving validation audio samples, `source[0, :source_lengths[0]]` and `content_audio[0, :content_lengths[0]]` are used — feeding the full padded tensor (including zero-padding) into ContentVec causes garbage features for the silent tail, producing silence in the converted output after the real audio ends.

**Two separate padding masks for context — self-attention and cross-attention**
Context audios are zero-padded in two places: within `__getitem__` (clips within an item padded to the item-level maximum T) and in `collate_fn` (items padded to the batch-level maximum T). Padding must be excluded from *both* the context encoder's self-attention and the cross-attention fusion. These require masks of different shapes.

`model.forward()` builds both masks internally from `ctx_audio_lens` (unpadded sample counts per reference):
```python
# ctx_enc_mask:   [B*N, T_h]   — for ContextEncoder self-attention
# ctx_cross_mask: [B, N*T_h]   — for CrossAttentionFusion key_padding_mask
for i in range(B):
    for n in range(N):
        flen = min((ctx_audio_lens[i][n] // 320) + 1, T_h)
        ctx_enc_mask[i * N + n, :flen]               = False
        ctx_cross_mask[i, n * T_h : n * T_h + flen]  = False
```
At inference, `encode_references()` processes one utterance at a time — no padding, no mask needed.

**Vocos input format**
The decoder outputs `[B, T_mel, 100]` (channels-last). Vocos expects `[B, 100, T_mel]` (channels-first). Always transpose before calling the vocoder: `vocoder(mel.transpose(1, 2))`.

**Mel decoder upsample factor**
ContentVec produces ~50 frames/s (stride 320 @ 16 kHz). Vocos requires ~93.75 frames/s (24 000 Hz / hop 256). The decoder uses `scale_factor = 24000 / (256 × 50) = 1.875` to bridge this gap.

**Streaming chunk size**
`BLOCK = 4800` samples (300 ms @ 16 kHz) → 15 ContentVec frames per block. `15 × 1.875 = 28.125` mel frames, which floors to 28. The output waveform is therefore slightly shorter than the input block on some iterations; `_process_block()` clips or zero-pads to exactly `BLOCK` samples before the crossfade so that no drift accumulates in the output pipe.

*Strictly exact alignment* requires `BLOCK / 320` to be a multiple of 8 (so that `N × 1.875` is an integer). The next exact value above the current `BLOCK=4800` is `BLOCK=5120` (16 frames → 30 mel frames exactly). Changing `BLOCK` also requires adjusting the `CHUNK` I/O granularity so that `BLOCK` is an integer multiple of `CHUNK`.

**Gradient accumulation — scale all loss terms**
The training loss is divided by `GRAD_ACCUM` before `.backward()`. If you add a second loss term (e.g. speaker loss), apply the same scaling: `(l1 + 0.1 * spk_loss) / GRAD_ACCUM`. Forgetting this makes the effective learning rate `GRAD_ACCUM×` too large.

**bfloat16 AMP — no GradScaler needed**
Training uses `torch.bfloat16` via `torch.amp.autocast`. Unlike float16, bfloat16 has the same exponent range as float32 so activations never overflow to `inf`/`NaN`. `GradScaler` has been removed — it only exists to work around float16 underflow and is a no-op for bfloat16.

**ContentVec normalization — training vs streaming**
ContentVec features must be normalised per-channel to strip residual speaker timbre. The mechanism differs between training and streaming:

*Training* — `F.instance_norm` over the full sequence: normalises each `(sample, channel)` pair across all T frames, giving mean=0 / std=1 per channel. Stable because sequences are 100–400 frames long.

*Streaming* — `mic_convert.py` calls `model.convert_chunk()` with blocks of 15 ContentVec frames. `ContentEncoder.forward()` applies `F.instance_norm` over the full block sequence, which is short but stable enough at 15 frames for practical use. For sub-8-frame blocks, `instance_norm` would be noisy and the EMA `hubert_stats` path in `ContentEncoder.forward()` should be used instead.

**TNP training / inference consistency**
At inference `encode_references()` processes each reference audio through `compute_context()`, which calls ContentEncoder + mel + align + concat → ContextEncoder for each clip independently, then **concatenates** the N sequences along the time axis: `torch.cat(encoded, dim=1)` → `[1, N·T_h, 256]`. The training loop mirrors this exactly — context encoder output is reshaped, not averaged:
```python
# model.forward() — context encoder output [B*N, T_h, d_model]
C_all = self.context_encoder(ctx_pairs, src_key_padding_mask=ctx_enc_mask)
C = C_all.view(B, N * T_h, -1)   # [B, N·T_h, d_model]  ← concatenate, not mean
```

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = VoiceConversionModel(device)

# Deterministic context encoder: forward() returns a single Tensor
enc = ContextEncoder().to(device).eval()
ctx_pairs = torch.randn(2, 50, 869).to(device)   # [B, T_h, 769+100]
z = enc(ctx_pairs)
print("z shape:", z.shape)          # [2, 50, 256]
assert isinstance(z, torch.Tensor), "expected Tensor, not tuple"

# Determinism check: same input → same output
z2 = enc(ctx_pairs)
print("Deterministic:", torch.allclose(z, z2))   # True

# Full pipeline via pre-cached context
audio = torch.randn(1, 3200).to(device)
C     = torch.randn(1, 750, 256).to(device)   # [1, T_ctx, D_MODEL]
out   = m.convert_chunk(audio, C)
print("Output shape:", out.shape)   # [1, 1, T_wav @ 24 kHz]
EOF

# Dataset smoke test
python - <<'EOF'
from dataset import SpeakerDataset
ds = SpeakerDataset("datasets/VCTK-Corpus-0.92/wav48_silence_trimmed", split="train")
s  = ds[0]
print("source_audio:",   s["source_audio"].shape)
print("audio_content:",  s["audio_content"].shape)
print("context_audios:", s["context_audios"].shape)   # [N_CTX, T_audio]
print("context_mels:",   s["context_mels"].shape)     # [N_CTX, 100, T_mel]
print("ctx_audio_lens:", s["ctx_audio_lens"])          # list[N_CTX] of ints
print("ctx_mel_lens:",   s["ctx_mel_lens"])            # list[N_CTX] of ints
EOF
```

---

## License

For research and personal use. Pre-trained models (ContentVec, Vocos, DeepFilterNet) are subject to their respective upstream licenses.
