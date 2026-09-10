# Real-Time Voice Conversion — Deterministic TNP-D

Few-shot, real-time voice conversion for **speech and singing**. Record a few seconds of a target speaker, and the system converts live microphone input into that voice at ~325 ms latency. Training mixes VCTK (speech) and JVS-MuSiC (singing) so conversion works across both registers.

**Architecture in one paragraph:** a Deterministic Transformer Neural Process (TNP-D). Context tokens `(ContentVec+F0, mel)` and target tokens `(ContentVec+F0)` are concatenated into a single sequence and processed by one shared Transformer under a block attention mask: context tokens see only context; each target token sees all context plus itself only (conditional independence). There is no stochastic sampling — training optimises masked L1 reconstruction plus temporal and spectral delta regularisation.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [How Training Works](#how-training-works)
- [Setup](#setup)
- [Data Pipeline](#data-pipeline)
  - [`dataset.py`](#datasetpy--speaker-folder-loader)
  - [`preprocess.py`](#preprocesspy--augmentation--mel-cache)
- [Training — `train.py`](#training--trainpy)
- [Inference](#inference)
  - [Offline — `convert.py`](#offline--convertpy)
  - [Real-time — `mic_convert.py`](#real-time--mic_convertpy)
  - [Networked mode](#networked-mode-optional)
- [Implementation Notes](#implementation-notes)
- [Verification](#verification)
- [License](#license)

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

# 4. Cache augmented audio + mels on GPU
python preprocess.py --data-root datasets/wav48_silence_trimmed

# 5. Train, then convert
python train.py --reset                                                       # from scratch
python mic_convert.py --checkpoint checkpoints/best.pt                        # real-time mic
python convert.py --source me.wav --reference alice.wav --output out.wav      # offline file
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
├── preprocess.py               # Offline augmentation + GPU mel cache
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
│   │   ├── tnp_unified.py      # TNPUnifiedTransformer: ctx_proj + tgt_proj + 8-layer Transformer + mel_proj
│   │   └── content_encoder.py  # DeepFilterNet3 + ContentVec (InstanceNorm) + torchcrepe F0
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
        ├──→ ContentEncoder (frozen) ──→ ctx_content [B*N, T_h, 769]
        └──→ Mel + F.interpolate     ──→ ctx_mel     [B*N, T_h, 100]
                        └── concat ──→ ctx_pairs  [B*N, T_h, 869]
                                              │  ctx_proj
                                              ▼
                                       ctx_enc [B*N, T_h, 512]
                                              │  reshape
                                              ▼
                                       [B, N·T_h, 512]  ───────────────────┐
                                                                            │  cat(dim=1)
SOURCE SPEAKER                                                              │
        │                                                                   │
        ▼                                                                   │
ContentEncoder (frozen)                                                     │
[B, T @ 16 kHz] → content [B, T_hub, 769]                                   │
        │  hubert_proj(768→512) + f0_proj(1→512)                            │
        ▼                                                                   │
  [B, T_hub, 512]  ─────────────────────────────────────────────────────►  │
                                                                            ▼
                                                           [B, N·T_h + T_hub, 512]
                                                                            │
                                           ┌────────────────────────────────────────┐
                                           │       TNPUnifiedTransformer            │
                                           │     (d=512, 8 heads, 8 layers)         │
                                           │                                        │
                                           │  TNP-D attention mask:                 │
                                           │  ┌─────────────┬─────────────┐         │
                                           │  │ ctx → ctx   │ ctx → tgt   │         │
                                           │  │   full  ○   │  blocked ✗  │         │
                                           │  ├─────────────┼─────────────┤         │
                                           │  │ tgt → ctx   │ tgt → tgt   │         │
                                           │  │   full  ○   │ diagonal ◑  │         │
                                           │  └─────────────┴─────────────┘         │
                                           └────────────────────────────────────────┘
                                                                            │
                                                          take target [B, T_hub, 512]
                                                                            │
                                                        1.875× upsample + mel_proj
                                                                            ▼
                                                                    [B, T_mel, 100]
                                                                            │
                                           ┌────────────────────────────────────────┐
                                           │      Vocos Vocoder (frozen)            │
                                           │  [B, 100, T_mel] → [B, 1, T_wav]       │
                                           └────────────────────────────────────────┘
                                                                            │
                                                                    CONVERTED AUDIO
```

### Parameter budget

| Module | Trainable | Parameters |
|---|---|---|
| `ctx_proj` — Linear(869 → 512) | Yes | 445,440 |
| `hubert_proj` — Linear(768 → 512) | Yes | 393,728 |
| `f0_proj` — Linear(1 → 512) | Yes | 1,024 |
| Transformer × 8 layers (d=512, heads=8, ff=2048) | Yes | 25,219,072 |
| `out_norm` + `mel_proj` — Linear(512 → 100) | Yes | 52,324 |
| ContentEncoder (DFN3 + ContentVec + crepe) | No | ~94.4 M |
| VocosVocoder | No | ~13.5 M |
| **Total trainable** | | **26,111,588 (~99.6 MB fp32)** |

### Temporal alignment

ContentVec runs at 50 fps (stride 320 @ 16 kHz); mel runs at 93.75 fps (hop 256 @ 24 kHz). Reference mels are downsampled to the ContentVec rate with `F.interpolate(mode='linear')` before concatenation, and target features are upsampled back inside `TNPUnifiedTransformer` with `scale_factor = 24000 / (256 × 50) = 1.875` before `mel_proj`.

---

## How Training Works

### The non-parallel data problem

Voice conversion needs to separate *what is said* (phonetic content) from *who says it* (speaker identity), then recombine them. Comparing converted audio against a ground-truth recording of another speaker saying the same sentence would require **parallel data** — sentence-aligned recordings across every speaker pair — which is rare and expensive.

VCTK and LibriSpeech are **non-parallel**: every speaker says different sentences. A frame-wise L1 loss between "Speaker A saying *apple*" and "Speaker B saying *banana*" is meaningless.

### Self-reconstruction with offline augmentation

Training uses a **self-reconstruction** objective instead. For each sample, one speaker and one utterance are picked, and three roles are assigned:

| Role | Audio used | Purpose |
|---|---|---|
| `source_audio` | **Parselmouth-augmented** version of the utterance (`_aug.pt`) | Phonetic content via ContentVec |
| `audio_content` | **Clean** version of the same utterance | Mel reconstruction target **and** true F0 contour |
| `context_audios` | `N_CTX=2` **clean** reference clips from the same speaker (always different clips) | `(ContentVec+F0, mel)` context pairs |

```
Training forward pass (TNP-D)
─────────────────────────────────────────────────────────────────────
context_audios ──→ ContentEncoder (ContentVec+F0)  ──→ ctx_content [B*N, T_h, 769]
  (clean refs)  └→ Mel + downsample                ──→ ctx_mel     [B*N, T_h, 100]
                                       concat → ctx_pairs [B*N, T_h, 869]
                                                   │ ctx_proj + reshape
                                                   ▼  ctx_enc [B, N·T_h, 512]

source_audio   ──→ ContentEncoder (ContentVec + InstanceNorm)  ──→  [B, T_hub, 769]
  (augmented)
audio_content  ──→ ContentEncoder (crepe F0)  ──→  F0 appended to ContentVec
  (clean)
                    hubert_proj + f0_proj → [B, T_hub, 512]

──────── cat(dim=1) → [B, N·T_h + T_hub, 512] ───────────────────────
        │
  TNPUnifiedTransformer (d=512, 8 heads, 8 layers, TNP-D mask)
        │
  take target portion → upsample → mel_proj → pred_mel [B, T_mel, 100]
```

Because prediction and ground truth come from the same speaker, the loss is phonetically valid. The augmented source has different pitch and formant characteristics from the clean context clips, so the model cannot copy timbre out of the content features — it must consult the context set `C` to recover the correct spectral shape.

At **inference** the roles switch to the intended conversion task:

```
Inference forward pass
─────────────────────────────────────────────────────────────────────
reference_audios  ──→  ContentEncoder + Mel  ──→  ctx_proj  ──→  C [1, N·T_h, 512]
                                                  (cached once per speaker)

source_audio  ──→  ContentEncoder  ──→  hubert_proj + f0_proj  ──→  [1, T_hub, 512]
                                │
                                └── concat with C → TNPUnifiedTransformer → converted mel
```

### Why TNP-D instead of a speaker embedding

A classic speaker embedding (d-vector, x-vector) compresses all N reference utterances into one fixed-size vector that must encode the speaker's full acoustic identity in limited dimensions, discarding temporal detail.

TNP-D treats the N references as a **context set** of input→output pairs: each `(content_ctx_i, mel_ctx_i)` is a direct observation of how the target speaker maps phonetic content to acoustics at frame *i*. Context tokens attend over all reference frames to build a function representation, and each target token attends to the whole context while staying conditionally independent of other target tokens. This is strictly more expressive than a fixed-size embedding — fine-grained co-variation between content and acoustics survives instead of being averaged away.

### Why ContentVec makes this work

ContentVec is a HuBERT fine-tuned with an explicit speaker-disentanglement objective: a teacher conditioned on a *different* speaker's voice guides the student toward speaker-invariant representations. The resulting features track phonetic content while discarding speaker-specific spectral shape, which makes a better content bottleneck than vanilla HuBERT. Unlike speech HuBERT (where layer 6 is preferred), ContentVec's **last layer** is the one trained to suppress speaker identity.

### Copy-synthesis risk and its three mitigations

ContentVec features still carry some residual speaker signal. If the model exploits it, it learns *copy synthesis*: reconstruction loss looks good, but cross-speaker conversion fails because `C` is never truly consulted.

**Symptoms:** converted audio sounds like the source rather than the target; validation loss is low but listening tests are poor.

1. **ContentVec Instance Normalization** — `F.instance_norm` is applied to ContentVec features before the F0 concat, normalising each `(sample, channel)` pair to mean=0 / std=1 across time. This strips the per-sample spectral bias that encodes timbre, making it a hard constraint: speaker identity *cannot* be reconstructed from content features alone.

2. **Offline pitch + formant augmentation** — `preprocess.py` pre-generates one Parselmouth-augmented variant per file. The aggressive shifts alter vocal tract shape and fundamental frequency, effectively destroying source speaker identity before ContentVec sees it. Meanwhile the *true* pitch contour is taken from the clean `audio_content` and handed straight to the decoder, so the decoder learns to respect F0 sharply (rather than blurring pitch to minimise L1) while relying entirely on `C` for timbre. Shift ranges are listed under [Phase 1](#phase-1--parselmouth-augmentation-cpu).

3. **Context pairs as an explicit mapping bottleneck** — because the context is distilled from `(content, mel)` pairs rather than mel alone, the model must learn a *conditional* mapping rule. Content present in the source but absent from the reference pairs cannot be attributed to the target speaker; the context set acts as an explicit prior over what is speaker-specific vs. content-specific.

---

## Setup

**Hardware:** NVIDIA GPU with ≥8 GB VRAM (tested on RTX 5060 Ti 16 GB) · CUDA driver ≥12.8 · ≥16 GB system RAM · WSL2 Ubuntu (training) or Windows (mic client).

<details>
<summary>Environment setup (two steps)</summary>

**Step 1 — create the conda env** (Python 3.12, portaudio, ffmpeg, and all pip deps except PyTorch):

```bash
conda env create -f environment.yml
conda activate voice
```

**Step 2 — install PyTorch for CUDA 12.8** (must run after activating the env):

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

PyTorch is deliberately excluded from `environment.yml` so you can target the exact CUDA version your driver supports. Swap `cu128` as needed (e.g. `cu121`); check with `nvidia-smi`.

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
| `torch.cuda.is_available()` returns `False` | Confirm Step 2 ran, and that your driver supports CUDA 12.8 (`nvidia-smi` → CUDA Version row) |
| `No module named 'torch'` | Step 2 was skipped, or run outside the `voice` conda env |

</details>

---

## Data Pipeline

### `dataset.py` — speaker-folder loader

`SpeakerDataset` reads any folder of speaker sub-directories. No parallel recordings, no matched filenames, no naming convention. Any sample rate is auto-resampled to 16 kHz. Minimum: **2 speakers, 7 files each**.

Each sample returns the three roles described in [Self-reconstruction](#self-reconstruction-with-offline-augmentation), as raw 16 kHz waveforms. If `_aug.pt` has not been generated yet, `source_audio` silently falls back to the clean file. One extra field, `context_mels`, carries pre-cached log-mels of the same reference clips — used for **validation visualization only**, not in the forward pass.

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

**Option C — JVS-MuSiC** — 100 Japanese speakers, each with a unique solo-singing recording at 24 kHz. Download from the [official release](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music), extract to `datasets/jvs_music_ver1/`, then:

```bash
# 1. Segment song_unique/wav/raw.wav per speaker on silence/breath boundaries.
#    Deletes song_common/ (shared-song data) and outputs seg_001.wav … per speaker.
python prepare_jvs_music.py --data-root datasets/jvs_music_ver1

# 2. (Optional) preview segment counts first
python prepare_jvs_music.py --data-root datasets/jvs_music_ver1 --dry-run

# 3. Cache augmentation + mel as usual
python preprocess.py --data-root datasets/jvs_music_ver1

# 4. Train on mixed speech + singing
python train.py --data-root datasets/jvs_music_ver1   # or combine with VCTK via a symlinked root
```

`prepare_jvs_music.py` produces **1,006 segments** (2–8 s each, 24 kHz) across 100 speakers. The native 24 kHz rate is preserved end-to-end by Phase 2 of `preprocess.py`, so nothing is lost relative to VCTK's 16 kHz → 24 kHz path. ContentVec and torchcrepe run at 16 kHz, and `dataset.py` resamples on load — no pipeline changes needed.

> **Why singing data matters:** the torchcrepe F0 extractor covers `[50, 800]` Hz, spanning speech and light singing registers. Training on JVS-MuSiC teaches the context encoder to represent a singer's vocal tract shape and register from sung reference clips, enabling cross-speaker singing conversion alongside speech.

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

### `preprocess.py` — augmentation + mel cache

Two phases run in sequence. Both are **safe to interrupt and resume** — existing `.pt` files are skipped.

```bash
python preprocess.py --data-root datasets/wav48_silence_trimmed
python preprocess.py --data-root datasets/wav48_silence_trimmed --batch-size 64  # faster on high-VRAM GPUs
python preprocess.py --data-root datasets/wav48_silence_trimmed --skip-aug       # mel cache only
```

#### Phase 1 — Parselmouth augmentation (CPU)

One pitch+formant-shifted variant per file, produced with Praat's *Change gender* algorithm and saved as a 16 kHz float32 tensor next to the source:

```
p225_001_mic1.flac  →  p225_001_mic1_aug.pt
```

Pitch and formant shift directions are sampled **independently at random**, so the combination covers a wide range of voice characteristics:

| Parameter | High direction | Low direction |
|---|---|---|
| Pitch ratio | `Uniform(1.10, 1.30)` | `Uniform(0.70, 0.90)` |
| Formant ratio | `Uniform(1.05, 1.15)` | `Uniform(0.85, 0.95)` |

This phase is CPU-bound and parallelised across cores with `ProcessPoolExecutor` (`--num-workers`). Caching once here avoids running Parselmouth or `torchaudio.functional.pitch_shift` inside the training loop.

#### Phase 2 — GPU mel cache (batched)

Computing mels on CPU inside the DataLoader is the primary bottleneck on large datasets (VCTK: ~44,000 files, >2 h per epoch). Phase 2 converts every **clean** file to a cached log-mel tensor on the GPU, reducing context-mel loading in `dataset.py` to a `torch.load` call.

1. Load waveforms on CPU with `soundfile.read` (batched, pin_memory)
2. Resample to 24 kHz and apply `MelSpectrogram` on GPU in one batched pass
3. Trim padding, apply `log(mel.clamp(1e-7))`, save as `<stem>.pt` next to the source file

<details>
<summary>CLI flags</summary>

| Flag | Default | Description |
|---|---|---|
| `--data-root` | *(required)* | Root directory of audio files |
| `--batch-size` | `32` | GPU batch size for Phase 2 — increase to fill VRAM |
| `--num-workers` | `8` | CPU workers for Phase 1 (augmentation) and Phase 2 (DataLoader) |
| `--seed` | `42` | RNG seed for augmentation direction sampling |
| `--skip-aug` | off | Skip Phase 1 and run mel cache only |

</details>

<details>
<summary>Disk space</summary>

**Phase 1 (`_aug.pt`)** — float32 audio at 16 kHz. A 5-second clip: `5 × 16 000 × 4 B ≈ 320 KB`. Full VCTK adds roughly **14 GB**.

**Phase 2 (`<stem>.pt`)** — float32 mel `[100, T_mel]`. A 5-second clip: `100 × 470 × 4 B ≈ 188 KB`. Full VCTK adds roughly **8–10 GB**.

</details>

---

## Training — `train.py`

Trains `TNPUnifiedTransformer` with bfloat16 AMP and gradient accumulation. `ContentEncoder` and `VocosVocoder` stay frozen throughout.

```bash
python train.py --reset                       # train from scratch (required: new architecture)
python train.py --data-root datasets/my_data  # custom dataset
python train.py                               # resume from latest.pt
```

> **Note:** checkpoints from any earlier architecture are **incompatible** — the trainable part changed from three modules (`ContextEncoder`, `CrossAttentionFusion`, `MelDecoder`) to a single `TNPUnifiedTransformer`. Use `--reset` when starting fresh.

### Loss

```
total = recon_loss + 0.5 × (loss_delta_time + loss_delta_freq)
```

| Term | Meaning |
|---|---|
| `recon_loss` | Masked L1 between predicted and target mel — absolute spectral shape |
| `loss_delta_time` | Masked L1 on frame-to-frame differences — temporal contour / prosody |
| `loss_delta_freq` | Masked L1 on mel-bin-to-bin differences — spectral contour / timbre |

All three are computed over valid (non-padded) frames only; see [Combined delta loss](#implementation-notes) for the exact masking code.

### Checkpoints

- `latest.pt` — every 2,500 steps, used for resuming
- `best.pt` — whenever validation loss improves, used for inference

### Training log CSV

Every 50 steps (`CSV_LOG_EVERY`) a row is appended to `checkpoints/training_log.csv`:

| Column | Description |
|---|---|
| `step` | Optimizer step number |
| `train_total` | Average combined loss over the last 50 steps |
| `train_recon` | Average masked L1 reconstruction component |
| `train_delta` | Average combined delta component (`0.5×(delta_time + delta_freq)`) |
| `val_loss` | Most recent validation loss (carried forward between checkpoints) |
| `learning_rate` | Current LR after warmup / cosine schedule |

The header is written on first run and rows are **appended** on resume — never overwritten. Rename with `--csv-log run2.csv` (written inside `--output-dir`).

```python
import pandas as pd
df = pd.read_csv("checkpoints/training_log.csv")
df.plot(x="step", y=["train_total", "train_recon", "train_delta", "val_loss"])
```

### Qualitative audio samples

Every `SAVE_EVERY` steps, validation writes four WAVs to `checkpoints/samples/step_{N}/`:

| File | Content |
|---|---|
| `source.wav` | Augmented source audio from the first validation batch (16 kHz) |
| `target.wav` | Clean audio content from the first validation batch (16 kHz) |
| `context.wav` | First clean reference context clip, decoded through Vocos (24 kHz) |
| `converted.wav` | Augmented source content + clean context `C`, decoded through Vocos (24 kHz) |

`converted.wav` mirrors the training path, so it tracks how well the model reconstructs clean speech from perturbed input. With F0 shifting applied, its pitch register should match the clean target.

<details>
<summary>CLI flags and key constants</summary>

| Flag | Default | Description |
|---|---|---|
| `--data-root` | `datasets/VCTK-Corpus-0.92/wav48_silence_trimmed` | Speaker folder root |
| `--output-dir` | `checkpoints` | Checkpoint and sample output directory |
| `--num-workers` | `8` | DataLoader worker processes |
| `--reset` | off | Train from scratch, ignoring existing checkpoints |
| `--csv-log` | `training_log.csv` | CSV filename inside `--output-dir` |

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

## Inference

Both entry points share the same F0 handling: source and target F0 statistics are estimated, then the source pitch contour is Z-score shifted in log(Hz) into the target speaker's range. Without this, a male-to-female conversion has the right timbre but the wrong register. If torchcrepe is unavailable or either speaker has no voiced frames, shifting is skipped and a message is printed.

### Offline — `convert.py`

Converts a file without a microphone or server — useful for evaluating a checkpoint before going real-time.

```bash
python convert.py --source me.wav --reference alice_1.wav alice_2.wav alice_3.wav --output converted.wav
```

Statistics are taken from the source and from all references concatenated, then applied to every chunk, and reported at runtime:

```
F0 shifting: src=120.3Hz (±2.6 st) → tgt=210.7Hz (±1.7 st)
```

<details>
<summary>All flags</summary>

| Flag | Default | Description |
|---|---|---|
| `--source` | *(required)* | Audio file to convert |
| `--reference` | *(required)* | One or more reference WAVs from the target speaker |
| `--checkpoint` | `checkpoints/best.pt` | Trained model checkpoint |
| `--output` | `converted.wav` | Output file path |

The full source is denoised by DFN3 in a single pass before chunking, so there is one cold-start transient at the start of the file rather than one per 4-second chunk. Output is saved at 24 kHz (Vocos native).

</details>

### Real-time — `mic_convert.py`

Runs the full pipeline locally, no server required. Phase 1 computes the target context `C` from a recording or WAV file; Phase 2 streams the microphone through the model. Press `Ctrl+C` to stop.

```bash
python mic_convert.py --list-devices                                          # list device indices
python mic_convert.py --checkpoint checkpoints/best.pt                        # record 5s reference from mic
python mic_convert.py --checkpoint checkpoints/best.pt --reference alice.wav  # use a WAV instead
```

Target F0 stats come from the reference at startup; source stats are tracked by EMA over the session and applied after `STATS_WARMUP=15` chunks.

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

```
Microphone  →  mic_queue  →  Inference thread  →  out_queue  →  Speaker
[960 samples/callback]    [accumulate 4800]
```

| Stage | Time |
|---|---|
| Mic accumulation (`BLOCK` = 4,800 samples) | ~300 ms |
| GPU inference (`_denoise_streaming` + ContentVec + TNP + vocoder) | ~25 ms |
| **Total steady-state** | **~325 ms** |

Vocos outputs 24 kHz; the inference thread resamples back to 16 kHz before writing to the output queue, and clips to exactly `BLOCK` samples per iteration to prevent drift.

</details>

### Networked mode (optional)

For cross-machine use — e.g. a WSL2 GPU server with a Windows mic client. Not required for local use.

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
<summary>Frozen modules, sample rates, and denoising</summary>

**Frozen modules must stay in eval mode.** `VoiceConversionModel.train()` is overridden to re-call `.eval()` on `content_encoder` and `vocoder` right after `super().train()`. PyTorch's `.train()` propagates to all submodules, so without the override it would enable dropout and BatchNorm inside ContentVec and Vocos. Add any new frozen submodule to that override.

**Dual sample rates.** The content encoder (DFN3 + ContentVec + crepe) runs at **16 kHz**; mel computation and Vocos run at **24 kHz**. Audio is resampled 16 → 24 kHz before the mel transform. Do not change the mel parameters (`n_fft=1024`, `hop_length=256`, `n_mels=100`, `power=1.0`) — they must match Vocos's training configuration exactly.

**DeepFilterNet3 runs at 48 kHz.** The content encoder resamples around it: `16 kHz → 48 kHz → DFN3 → 16 kHz → ContentVec`. Feeding 16 kHz directly into DFN3 produces silent garbage with no error.

**DeepFilterNet3 GRU state.** The library's `enhance()` calls `model.reset_h0()` at the **start of every call** — it is built for offline enhancement of a complete file, not streaming. Calling `_denoise()` on consecutive chunks therefore cold-starts the GRU each time, causing ~0.5 s of attenuation at every boundary. Each call site handles this differently:

- **`convert.py` (offline):** the full source is passed to `_denoise()` in **one** call before the chunk loop; chunks are sliced from the pre-denoised tensor via `convert_chunk_streaming(..., skip_denoise=True)`.
- **`compute_context()` (reference encoding):** each reference waveform is denoised in **one** call, then forwarded with `skip_denoise=True`.
- **`mic_convert.py` (real-time):** `_denoise_streaming()` suppresses the internal `reset_h0`, letting GRU state carry across blocks. `reset_dfn_state()` is called **once** at stream start.
- **Training `model.forward()`:** skips DFN3 entirely (`skip_denoise=True`) — training data is clean studio audio, so denoising adds nothing and the cold-start transient would corrupt the signal. Since DFN3's job is to make noisy audio look clean before ContentVec, the train/inference mismatch is minimal.

**Audio loading — `soundfile`, not `torchaudio.load`.** Recent torchaudio defaults to the `torchcodec` backend, which is not installed here. `soundfile.read()` handles WAV, FLAC, OGG, and AIFF with no codec dependency. `dataset.py` always used it; the inference scripts were updated to match.

</details>

<details>
<summary>Content and F0 features</summary>

**ContentVec layer.** `HubertModel.from_pretrained("lengyue233/content-vec-best")` returns a standard `transformers` HuBERT; `last_hidden_state` is used, because ContentVec maximises speaker disentanglement at the final layer (unlike speech HuBERT, where layer 6 is preferred). Weights (~360 MB) download to `~/.cache/huggingface/` on first run.

**ContentVec normalization — training vs streaming.** `F.instance_norm` normalises each `(sample, channel)` pair across time, stripping residual per-sample spectral bias. In *training* sequences are 100–400 frames, so this is stable. In *streaming*, `mic_convert.py` pre-denoises each block and calls `convert_chunk_streaming()` with 15-frame blocks — short, but stable enough in practice. For sub-8-frame blocks, use the EMA `hubert_stats` path in `ContentEncoder.forward()` instead.

**F0 log scaling.** Raw torchcrepe F0 spans `[0, 800]` Hz while ContentVec features sit roughly in `[-3, +3]`, so `torch.log1p(f0)` maps F0 to `[0, ~6.7]` before `f0_proj`. A nanmedian spike filter (kernel 5) is applied after periodicity gating; unvoiced frames are marked NaN before the median window so they cannot pull voiced pitch down at voiced/silence boundaries — nanmedian returns a non-NaN value as long as one frame in the window is voiced.

**Cross-speaker Z-score shift.** `ContentEncoder.forward()` accepts an optional `f0_stats=(src_log_mean, src_log_std, tgt_log_mean, tgt_log_std)` tuple, all in log(Hz). Voiced frames are shifted before `log1p`:

```python
voiced_mask = (f0 > 0.0).float()
log_f0 = torch.log(f0.clamp(min=1.0))          # log(Hz); unvoiced→0, masked out
log_f0_shifted = (log_f0 - src_log_mean) / (src_log_std + 1e-5) * tgt_log_std + tgt_log_mean
f0_shifted = torch.exp(log_f0_shifted).clamp(min=50.0, max=800.0)
f0 = voiced_mask * f0_shifted + (1.0 - voiced_mask) * f0
```

Working in log(Hz) makes the shift multiplicative in Hz, so semitone intervals and vibrato depth are exactly preserved; a linear Z-score in Hz would distort interval ratios (a perfect fifth's 3:2 would compress or expand with absolute pitch). Statistics use **robust estimators** (median + MAD×1.4826), resisting octave-error outliers — `extract_f0_stats` returns `(log_center, log_spread)`. Applied in three places: `convert.py` (once before the chunk loop), `mic_convert.py` (EMA over the session), and `_validate` samples (first validation batch item, `converted.wav` only — never the loss).

**F0 routing — training vs inference.** To stop the decoder from ignoring F0, it must see the exact target pitch during training. `ContentEncoder.forward()` takes an `f0_audio_16k` argument:

- *Training* — set to the clean target (`audio_content`), routing the true contour to the decoder with no math and teaching it to trace F0 harmonics sharply.
- *Inference* — omitted, since the target audio doesn't exist yet; F0 comes from the source and is Z-score shifted as above.

Either way the decoder receives a valid target pitch contour. For reference context encoding, `content_encoder(ctx_flat, f0_audio_16k=ctx_flat)` — F0 comes from the same clean reference audio, unshifted.

**F0 decoder — argmax, not Viterbi.** `torchcrepe` uses `decoder=torchcrepe.decode.argmax`. Viterbi is global dynamic programming that needs the full sequence and is incompatible with chunk-by-chunk streaming; argmax is frame-independent and causal.

**F0 frame alignment.** `torchcrepe.predict(..., hop_length=320)` and ContentVec both use a 320-sample stride, producing `T // 320` frames each. Change one and you must change the other — mismatched strides cause silent misalignment at concatenation.

**Split projection keeps F0 visible.** The target vector `[B, T_hub, 769]` is 768 ContentVec channels + 1 log-F0 channel. A single `nn.Linear(769, 512)` would initialise with Xavier variance `∝ 1/769`, making the aggregate ContentVec signal 768× the F0 signal at step 0 — easy to ignore pitch early in training. Instead `hubert_proj = nn.Linear(768, 512)` and `f0_proj = nn.Linear(1, 512)` are added element-wise; Xavier gives `f0_proj` weights ~28× larger (fan-in 1 vs 768), equalising F0's representational weight at init.

</details>

<details>
<summary>Transformer, masking, and loss</summary>

**No positional encoding, strict conditional independence.** Speaker identity (timbre, vocal tract shape) is time-invariant, so position should not affect the mapping; dropping PE also prevents overfitting to where phonemes sit in the reference clip. Context tokens come from `ctx_proj = nn.Linear(869, 512)`; target tokens from the split projections above. Both pass through 8 shared layers (d=512, 8 heads, ff=2048). There is no variational bottleneck — training and inference behave identically.

**Training / inference consistency.** At inference, `compute_context()` encodes each reference independently and **concatenates** along time: `torch.cat(encoded_list, dim=1)` → `[1, N·T_h, 512]`. Training mirrors this exactly — `ctx_proj` output is reshaped, never averaged:

```python
# model.forward()
ctx_encoded = self.tnp.encode_context(ctx_pairs)  # [B*N, T_h, D_MODEL]
ctx_encoded = ctx_encoded.view(B, N * T_h, -1)    # [B, N·T_h, D_MODEL] — concatenate, not mean
```

`ctx_proj` is only a linear projection; the actual self-attention across context tokens happens inside `TNPUnifiedTransformer.transformer` during the joint forward pass, not in a separate pre-encoding step.

**Padding masks.** `collate_fn` zero-pads `source_audio`, `audio_content`, and context audios to the batch maximum and returns `source_lengths`, `content_lengths`, `ctx_audio_lens`, and `ctx_mel_lens`. `model.forward()` builds a single float additive padding mask `[B, N*T_h + T_hub]` — 0 for valid, `-inf` for padded — matching the dtype of the TNP-D attention mask so PyTorch's Transformer sees a consistent float mask throughout (no bool/float mismatch). At inference `compute_context()` handles one utterance at a time, so no mask is needed.

```python
# ctx_key_padding_mask: [B, N*T_h]  True=padded (bool), built in model.forward()
# pad_mask in TNPUnifiedTransformer.forward(): [B, N*T_h + T_hub]  float, 0/-inf
```

When saving validation audio, use `source[0, :source_lengths[0]]` and `content_audio[0, :content_lengths[0]]` — feeding the padded tensor into ContentVec yields garbage features for the silent tail and silence after the real audio ends.

**Combined delta loss.** All three terms are masked to valid frames; padding regions contain `log(1e-7) ≈ −16.1`. The mask is applied to predictions and targets before `reduction="sum"` (cleaner than post-hoc multiplication), and each term is normalised by its own valid-pair count.

```python
# recon: absolute spectral shape
recon_loss = F.l1_loss(pred_mel * mask, tgt_mel * mask, reduction="sum") / (mask.sum() * N_MELS)

# delta_time: frame-to-frame change (temporal contour / prosody)
pred_dt = pred_mel[:, 1:T, :] - pred_mel[:, :T-1, :]
loss_delta_time = F.l1_loss(pred_dt * mask_dt, tgt_dt * mask_dt, reduction="sum") / (mask_dt.sum() * N_MELS)

# delta_freq: mel-bin-to-bin change (spectral contour / timbre)
pred_df = pred_mel[:, :T, 1:] - pred_mel[:, :T, :-1]
loss_delta_freq = F.l1_loss(pred_df * mask_df, tgt_df * mask_df, reduction="sum") / (mask_df.sum() * (N_MELS-1))

total_loss = recon_loss + 0.5 * (loss_delta_time + loss_delta_freq)
```

**Gradient accumulation — scale all loss terms.** The loss is divided by `GRAD_ACCUM` before `.backward()`. Any additional term needs the same scaling: `(l1 + 0.1 * spk_loss) / GRAD_ACCUM`. Forgetting it makes the effective learning rate `GRAD_ACCUM×` too large.

**bfloat16 AMP — no GradScaler.** Training uses `torch.bfloat16` via `torch.amp.autocast`. bfloat16 shares float32's exponent range, so activations never overflow to `inf`/`NaN`. `GradScaler` exists only to work around float16 underflow and has been removed.

</details>

<details>
<summary>Vocoder and streaming granularity</summary>

**Vocos input format.** The decoder outputs `[B, T_mel, 100]` (channels-last); Vocos expects `[B, 100, T_mel]` (channels-first). Always transpose: `vocoder(mel.transpose(1, 2))`.

**Streaming chunk size.** `BLOCK = 4800` samples (300 ms @ 16 kHz) → 15 ContentVec frames → `15 × 1.875 = 28.125` mel frames, which floors to 28. The output waveform is therefore slightly shorter than the input block on some iterations, so `_process_block()` clips or zero-pads to exactly `BLOCK` samples before the crossfade and no drift accumulates.

*Strictly exact* alignment requires `BLOCK / 320` to be a multiple of 8 (so `N × 1.875` is an integer). The next exact value above 4,800 is `BLOCK = 5120` (16 frames → 30 mel frames). Changing `BLOCK` also means adjusting the `CHUNK` I/O granularity so `BLOCK` stays an integer multiple of `CHUNK`.

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
from core.modules.tnp_unified import TNPUnifiedTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = VoiceConversionModel(device)
print("Trainable params:", m.trainable_param_count())   # 26,111,588

# TNP-D mask correctness
L_ctx, L_tgt = 80, 60
mask = TNPUnifiedTransformer.build_tnp_mask(L_ctx, L_tgt, device)
assert (mask[:L_ctx, L_ctx:] == float("-inf")).all(), "ctx must not see tgt"
assert (mask[L_ctx:, :L_ctx] == 0).all(),            "tgt must see all ctx"
assert (mask[L_ctx:, L_ctx:].diagonal() == 0).all(), "tgt diagonal must be open"
print("TNP-D mask: OK")

# Determinism check
m.eval()
audio = torch.randn(1, 3200).to(device)
C = m.compute_context([torch.randn(1, 12000).to(device)])
out1 = m.convert_chunk(audio, C)
out2 = m.convert_chunk(audio, C)
print("Deterministic:", torch.allclose(out1, out2))     # True
print("Output shape:", out1.shape)                      # [1, 1, T_wav @ 24 kHz]
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
print("ctx_audio_lens:", s["ctx_audio_lens"])         # list[N_CTX] of ints
print("ctx_mel_lens:",   s["ctx_mel_lens"])           # list[N_CTX] of ints
EOF
```

---

## License

For research and personal use. Pre-trained models (ContentVec, Vocos, DeepFilterNet) are subject to their respective upstream licenses.
