# Real-Time Voice Conversion Pipeline — Deterministic TNP-D

Few-shot, real-time voice conversion for both **speech and singing**. Record a few seconds of a target speaker — the system converts your live microphone input into that voice with low latency. The model is trained on a mixture of VCTK speech data and JVS-MuSiC singing data, enabling vocal conversion across speaking and singing registers.

**Architecture:** Deterministic Transformer Neural Process (TNP-D) — context (ContentVec+F0, mel) tokens and target (ContentVec+F0) tokens are concatenated into one sequence and processed by a single shared Transformer with a block attention mask: context tokens see only context, each target token sees all context plus only itself (conditional independence). No stochastic sampling; training optimises a combined loss of masked L1 reconstruction + temporal and spectral delta regularisation.

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
SOURCE SPEAKER (augmented audio)                                            │
        │                                                                   │
        ▼                                                                   │
ContentEncoder (frozen)                                                     │
[B, T @ 16 kHz] → content [B, T_hub, 769]                                  │
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
                                           │  ┌─────────────┬─────────────┐        │
                                           │  │ ctx → ctx   │ ctx → tgt   │        │
                                           │  │   full  ○   │  blocked ✗  │        │
                                           │  ├─────────────┼─────────────┤        │
                                           │  │ tgt → ctx   │ tgt → tgt   │        │
                                           │  │   full  ○   │ diagonal ◑  │        │
                                           │  └─────────────┴─────────────┘        │
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
                                           │  [B, 100, T_mel] → [B, 1, T_wav]      │
                                           └────────────────────────────────────────┘
                                                                            │
                                                                    CONVERTED AUDIO
```

| Module | Trainable | Parameters |
|---|---|---|
| ctx_proj  Linear(869 → 512) | Yes | 445,440 |
| hubert_proj  Linear(768 → 512) | Yes | 393,728 |
| f0_proj  Linear(1 → 512) | Yes | 1,024 |
| Transformer × 8 layers (d=512, heads=8, ff=2048) | Yes | 25,219,072 |
| out_norm + mel_proj  Linear(512 → 100) | Yes | 52,324 |
| ContentEncoder (DFN3 + ContentVec + crepe) | No | ~94.4 M |
| VocosVocoder | No | ~13.5 M |
| **Total trainable** | | **26,111,588 (~99.6 MB fp32)** |

**Temporal alignment:** ContentVec outputs at 50 fps (stride 320 @ 16 kHz). Mel is computed at ~93.75 fps (hop 256 @ 24 kHz). Reference mels are downsampled to ContentVec rate via `F.interpolate(mode='linear')` before concatenation. After the transformer, target features are upsampled back to mel rate (`scale_factor = 1.875`) inside `TNPUnifiedTransformer` before `mel_proj`.

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
context_audios ──→ ContentEncoder (ContentVec+F0)  ──→ ctx_content [B*N, T_h, 769]
  (clean refs)  └→ Mel + downsample               ──→ ctx_mel     [B*N, T_h, 100]
                                       concat → ctx_pairs [B*N, T_h, 869]
                                                   │ ctx_proj + reshape
                                                   ▼  ctx_enc [B, N·T_h, 512]

source_audio   ──→ ContentEncoder (ContentVec+InstanceNorm)  ──→  [B, T_hub, 769]
  (augmented)

audio_content  ──→ ContentEncoder (Crepe F0)  ──→  F0 appended to ContentVec
  (clean)

                    hubert_proj + f0_proj → [B, T_hub, 512]

──────── cat(dim=1) ─────────────────────────────────────────────────────
[B, N·T_h + T_hub, 512]
        │
  TNPUnifiedTransformer (d=512, 8 heads, 8 layers, TNP-D mask)
        │
  take target portion → upsample → mel_proj → pred_mel [B, T_mel, 100]

Loss: total = recon + 0.5 × (delta_time + delta_freq)
        recon      = Masked L1( pred_mel, mel(audio_content) )
        delta_time = Masked L1 on frame-to-frame mel differences   (temporal contour)
        delta_freq = Masked L1 on mel-bin-to-bin differences        (spectral contour)
```

Because prediction and ground truth come from the same speaker, the loss is phonetically valid. The augmented source has different pitch and formant characteristics from the clean context clips — the model cannot copy timbre from content features and must consult `C` to reconstruct the correct spectral shape.

At **inference time** the roles switch to the intended conversion task:

```
Inference forward pass
─────────────────────────────────────────────────────────────────────
reference_audios  ──→  ContentEncoder + Mel  ──→  ctx_proj  ──→  C [1, N·T_h, 512]
                                                  (cached once per speaker)

source_audio  ──→  ContentEncoder  ──→  tgt_proj  ──→  [1, T_hub, 512]
                                │
                                └── concat with C → TNPUnifiedTransformer → converted mel
```

### Why TNP-D: learning a mapping function, not an embedding

A classic speaker embedding (d-vector, x-vector) compresses all N reference utterances into a single fixed-size vector. That vector must encode the speaker's full acoustic identity in limited dimensions, discarding temporal detail.

TNP-D instead treats the N reference utterances as a **context set** of input→output pairs: each pair `(content_ctx_i, mel_ctx_i)` is a direct observation of how the target speaker maps phonetic content to acoustic output at frame i. The unified Transformer processes both context and target tokens jointly — context tokens attend over all reference frames to build a function representation, and each target token attends to the entire context while remaining conditionally independent of other target tokens. This is strictly more expressive than a fixed-size embedding: the model exploits fine-grained co-variation between content and acoustics in the reference set, not averaged into a single vector.

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

> **Why singing data matters:** The model's torchcrepe F0 extractor covers `[50, 800]` Hz, spanning speech and light singing registers. Training on JVS-MuSiC teaches the ContextEncoder to encode a singer's vocal tract shape and register from sung context clips, enabling cross-speaker singing voice conversion alongside speech conversion.

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

Trains `TNPUnifiedTransformer` with bfloat16 AMP and gradient accumulation. `ContentEncoder` and `VocosVocoder` remain frozen throughout. The loss combines three masked L1 terms:

```
total = recon_loss + 0.5 × (loss_delta_time + loss_delta_freq)
```

- **recon_loss** — L1 between predicted mel and target mel (absolute spectral shape)
- **loss_delta_time** — L1 on frame-to-frame differences (temporal contour / prosody)
- **loss_delta_freq** — L1 on mel-bin-to-bin differences (spectral contour / timbre)

```bash
python train.py --reset                       # train from scratch (required: new architecture)
python train.py --data-root datasets/my_data  # custom dataset
python train.py                               # resume from latest.pt
```

> **Note:** Checkpoints from any previous architecture are **incompatible** — the trainable module changed from three separate modules (`ContextEncoder`, `CrossAttentionFusion`, `MelDecoder`) to a single `TNPUnifiedTransformer`. Always use `--reset` when starting fresh.

Checkpoints are written to `checkpoints/`:
- `latest.pt` — every 2 500 steps, used for resuming
- `best.pt` — whenever validation loss improves, used for inference

### Training log CSV

Every 50 steps (`CSV_LOG_EVERY`), a row is appended to `checkpoints/training_log.csv`:

| Column | Description |
|---|---|
| `step` | Optimizer step number |
| `train_total` | Average combined loss over the last 50 steps (`recon + 0.5×(delta_time + delta_freq)`) |
| `train_recon` | Average masked L1 reconstruction component |
| `train_delta` | Average combined delta component (`0.5×(delta_time + delta_freq)`) |
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
df.plot(x="step", y=["train_total", "train_recon", "train_delta", "val_loss"])
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
Microphone  →  mic_queue  →  Inference thread  →  out_queue  →  Speaker
[960 samples/callback]    [accumulate 4800]
```

| Stage | Time |
|---|---|
| Mic accumulation (`BLOCK` = 4 800 samples) | ~300 ms |
| GPU inference (`_denoise_streaming` + ContentVec + TNP + vocoder) | ~25 ms |
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
F0 shifting: src=120.3Hz (±2.6 st) → tgt=210.7Hz (±1.7 st)
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

The full source audio is denoised by DFN3 in a single pass before chunking, so there is only one cold-start transient at the very start of the file rather than one per 4-second chunk. Output is saved at 24 kHz (Vocos native sample rate).

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
`enhance()` (the DeepFilterNet library function) calls `model.reset_h0()` at the **start of every call** — it is designed for offline enhancement of a complete audio file, not streaming. Calling `_denoise()` twice on consecutive chunks therefore cold-starts the GRU on the second chunk, causing ~0.5 s of attenuation at every call boundary.

Each call site is handled differently:

- **`convert.py` (offline):** the full source audio is passed to `_denoise()` in **one call** before the chunk loop. DFN3 has a single cold start at the beginning; all 4-second chunks are sliced from the pre-denoised tensor. `convert_chunk_streaming()` (which accepts pre-denoised audio via `skip_denoise=True`) is used instead of `convert_chunk()`.
- **`compute_context()` (reference encoding):** each reference waveform is denoised in **one** `_denoise()` call, then forwarded with `skip_denoise=True` so no second `enhance()` call is made.
- **`mic_convert.py` (real-time):** `_denoise_streaming()` suppresses the internal `reset_h0` inside `enhance()`, allowing the GRU state to carry over across blocks. `reset_dfn_state()` is called **once** at stream start. `convert_chunk_streaming()` receives the pre-denoised block.
- **Training `model.forward()`:** skips DFN3 entirely (`skip_denoise=True`) — training data is clean studio audio, so denoising adds no benefit and the cold-start transient would only corrupt the training signal. DFN3 is active only at inference, where the input may be a live microphone. Since DFN3's goal is to make noisy audio look like clean audio before ContentVec, the train/inference distribution mismatch is minimal.

**ContentVec layer**
`HubertModel.from_pretrained("lengyue233/content-vec-best")` returns a standard `transformers` HuBERT model. `last_hidden_state` (the final transformer layer) is used — ContentVec is trained to maximise speaker disentanglement at the last layer, unlike speech HuBERT where layer 6 is preferred. Weights (~360 MB) are downloaded to `~/.cache/huggingface/` on first run.

**Audio loading — soundfile, not torchaudio.load**
`convert.py` and `mic_convert.py` load audio with `soundfile.read()` directly. Recent versions of torchaudio default to `torchcodec` as the backend, which is not installed in this environment. `soundfile` natively handles WAV, FLAC, OGG, and AIFF without any codec dependency. `dataset.py` has always used `soundfile`; the inference scripts were updated to match.

**Offline pitch + formant augmentation**
`preprocess.py` Phase 1 generates `<stem>_aug.pt` — a 16-kHz audio tensor produced by Praat's *Change gender* algorithm with randomised pitch and formant ratios. Pitch direction (up or down) and formant direction are sampled independently so the combination covers a wide range of voice characteristics. The augmented tensor is stored once and loaded by `dataset.py` at training time, avoiding the cost of running Parselmouth or `torchaudio.functional.pitch_shift` inside the training loop. If `_aug.pt` is missing for a given file, `_load_aug()` silently falls back to clean audio.

**F0 log scaling and cross-speaker Z-score shift**
Raw F0 from torchcrepe spans `[0, 800]` Hz (fmax=800) while ContentVec features fall roughly in `[-3, +3]`. `torch.log1p(f0)` maps F0 to `[0, ~6.7]` before it is passed to `f0_proj`.

A nanmedian spike filter (kernel size 5) is applied after periodicity gating. Unvoiced frames are marked NaN before the median window so they cannot pull voiced pitch values down at voiced/silence boundaries — nanmedian returns a non-NaN value as long as at least one frame in the window is voiced.

`ContentEncoder.forward()` accepts an optional `f0_stats=(src_log_mean, src_log_std, tgt_log_mean, tgt_log_std)` tuple (all values in log(Hz)). When provided, voiced frames (f0 > 0) are Z-score shifted in log domain before `log1p`:
```python
voiced_mask = (f0 > 0.0).float()
log_f0 = torch.log(f0.clamp(min=1.0))          # log(Hz); unvoiced→0, masked out
log_f0_shifted = (log_f0 - src_log_mean) / (src_log_std + 1e-5) * tgt_log_std + tgt_log_mean
f0_shifted = torch.exp(log_f0_shifted).clamp(min=50.0, max=800.0)
f0 = voiced_mask * f0_shifted + (1.0 - voiced_mask) * f0
```
Unvoiced frames (f0 == 0) are left unchanged. This shifts the source speaker's pitch contour into the target speaker's fundamental frequency range — without it, a male-to-female conversion will have the correct timbre but the wrong pitch register.

F0 statistics are computed and applied in **log(Hz) domain**. Log-domain Z-score shift is equivalent to a multiplicative (ratio-preserving) transformation in Hz, so semitone intervals and vibrato depth are exactly preserved. A linear Z-score shift in Hz would distort interval ratios — e.g. a perfect fifth (3:2 ratio) would be compressed or expanded depending on absolute pitch.

Statistics use **robust estimators** (median + MAD×1.4826) in log(Hz), resisting octave-error outliers. `extract_f0_stats` returns `(log_center, log_spread)`; the shift in `ContentEncoder.forward()` converts to log, applies Z-score, then exponentiates back to Hz before `log1p`.

`f0_stats` is used in three places:
- **`convert.py`** — extracted from source and concatenated reference audio once before the chunk loop; applied to every chunk.
- **`mic_convert.py`** — target stats extracted from reference at startup; source stats tracked via EMA over the streaming session, applied after `STATS_WARMUP=15` chunks.
- **Training `_validate` samples** — extracted from the unpadded source and target audio of the first validation batch item; applied only to the `converted.wav` sample, not to the loss computation.

**F0 routing — Training vs Inference**
To prevent the decoder from ignoring F0 inputs, it must receive the exact target pitch during training. `ContentEncoder.forward()` accepts an `f0_audio_16k` argument.
- *Training*: `f0_audio_16k` is set to the clean target audio (`audio_content`). The model routes the *true* pitch contour to the decoder without any math, teaching the decoder to trust and trace the F0 harmonics sharply.
- *Inference*: The target audio doesn't exist yet, so `f0_audio_16k` is omitted. The F0 is extracted from the source audio and mathematically shifted using the `f0_stats` Z-score calculation to match the target speaker's range. From the decoder's perspective, both pipelines provide a perfectly valid target pitch contour.

For reference context encoding, `content_encoder(ctx_flat, f0_audio_16k=ctx_flat)` — F0 is extracted from the same clean reference audio, no shift applied.

**Split projection keeps F0 visible — `hubert_proj` + `f0_proj`**
The target content vector `[B, T_hub, 769]` has 768 ContentVec channels and 1 log-F0 channel. A single `nn.Linear(769, 512)` would initialise with Xavier uniform variance `∝ 1/769`; the aggregate ContentVec signal would be 768× the F0 signal at step 0, making it easy for the model to ignore pitch entirely in early training. Instead two separate projections are used: `hubert_proj = nn.Linear(768, 512)` for ContentVec and `f0_proj = nn.Linear(1, 512)` for log-F0, added element-wise. Xavier uniform gives `f0_proj` weights ~28× larger (fan-in 1 vs 768), ensuring F0 has equal representational weight at initialisation.

**F0 decoder — argmax, not Viterbi**
`torchcrepe` is configured with `decoder=torchcrepe.decode.argmax`. Viterbi is a global dynamic-programming algorithm that requires the full sequence and is incompatible with chunk-by-chunk streaming. Argmax is frame-independent and causal.

**F0 frame alignment**
`torchcrepe.predict(..., hop_length=320)` and ContentVec both have a 320-sample stride, producing `T // 320` frames each. If you ever change one, change both — mismatched strides cause silent feature misalignment at concatenation.

**TNPUnifiedTransformer — no positional encoding, strict conditional independence**
No positional encoding is used. Speaker identity (timbre, vocal tract shape) is time-invariant, so position should not affect the mapping. Without PE, the model cannot overfit on the temporal position of phonemes in the reference clip.

Context tokens are projected from `(ContentVec+F0, mel)` pairs `[B*N, T_h, 869]` via `ctx_proj = nn.Linear(869, 512)`. Target tokens use two separate projections added element-wise: ContentVec `[B, T_hub, 768]` via `hubert_proj = nn.Linear(768, 512)` and log-F0 `[B, T_hub, 1]` via `f0_proj = nn.Linear(1, 512)`. After concatenation both pass through 8 shared Transformer layers (d=512, 8 heads, ff=2048). No variational bottleneck — the model is fully deterministic, training and inference behave identically.

**ContentVec Instance Normalization**
`F.instance_norm` is applied to ContentVec features `[B, 768, T_frames]` before the F0 concat. It normalises each `(sample, channel)` pair to mean=0 / std=1 across the time dimension, stripping any residual per-sample spectral bias. ContentVec's training objective already reduces speaker leakage; instance norm is a second hard constraint ensuring the content stream cannot bypass it.

**Combined delta loss**
The training loss has three masked L1 components, all computed only over valid (non-padded) frames:

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

`collate_fn` returns `content_lengths` (original audio sample counts); the training loop converts these to mel frame counts and builds a boolean mask. Padding regions contain `log(1e-7) ≈ −16.1` and are excluded from all three terms. The mask is applied to both predictions and targets before `reduction="sum"` (cleaner than post-hoc multiplication) and each term is normalised by its own valid-pair count.

**Batch padding and lengths**
`collate_fn` zero-pads `source_audio` and `audio_content` to the batch maximum length. `source_lengths` and `content_lengths` are returned alongside the padded tensors so callers can recover the unpadded region. When saving validation audio samples, `source[0, :source_lengths[0]]` and `content_audio[0, :content_lengths[0]]` are used — feeding the full padded tensor (including zero-padding) into ContentVec causes garbage features for the silent tail, producing silence in the converted output after the real audio ends.

**Padding masks in the unified Transformer**
Context audios are zero-padded in `collate_fn` to the batch-level maximum. `model.forward()` builds a single float additive padding mask `[B, N*T_h + T_hub]` from `ctx_audio_lens` and `content_lengths`. Valid positions are 0; padded positions are -inf. This matches the dtype of the TNP-D attention mask so PyTorch's Transformer sees a consistent float mask throughout — no bool/float mismatch.

```python
# ctx_key_padding_mask: [B, N*T_h]  True=padded (bool), built in model.forward()
# pad_mask in TNPUnifiedTransformer.forward(): [B, N*T_h + T_hub]  float, 0/-inf
```
At inference, `compute_context()` processes one utterance at a time — no padding, no mask needed.

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

*Streaming* — `mic_convert.py` pre-denoises each block with `_denoise_streaming()` then calls `model.convert_chunk_streaming()` with blocks of 15 ContentVec frames. `ContentEncoder.forward()` applies `F.instance_norm` over the full block sequence, which is short but stable enough at 15 frames for practical use. For sub-8-frame blocks, `instance_norm` would be noisy and the EMA `hubert_stats` path in `ContentEncoder.forward()` should be used instead.

**TNP training / inference consistency**
At inference `compute_context()` calls ContentEncoder + mel + align + concat → `ctx_proj` for each reference clip independently, then **concatenates** along the time axis: `torch.cat(encoded_list, dim=1)` → `[1, N·T_h, 256]`. Training mirrors this exactly — `ctx_proj` output is reshaped, not averaged:
```python
# model.forward()
ctx_encoded = self.tnp.encode_context(ctx_pairs)  # [B*N, T_h, D_MODEL]
ctx_encoded = ctx_encoded.view(B, N * T_h, -1)    # [B, N·T_h, D_MODEL] — concatenate, not mean
```
Note that `ctx_proj` is just a linear projection; the actual self-attention across context tokens happens inside `TNPUnifiedTransformer.transformer` during the joint forward pass, not in a separate pre-encoding step.

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
print("ctx_audio_lens:", s["ctx_audio_lens"])          # list[N_CTX] of ints
print("ctx_mel_lens:",   s["ctx_mel_lens"])            # list[N_CTX] of ints
EOF
```

---

## License

For research and personal use. Pre-trained models (ContentVec, Vocos, DeepFilterNet) are subject to their respective upstream licenses.
