"""
Prepare JVS-MuSiC dataset for training.

Steps per speaker:
  1. Delete song_common/ (unwanted shared-song data)
  2. Segment song_unique/wav/raw.wav on silence/breath boundaries
     using librosa energy-based splitting
  3. Save only segments in [MIN_DUR, MAX_DUR] seconds directly
     inside the speaker folder as seg_001.wav, seg_002.wav, …
  4. Delete song_unique/ directory

Target structure after processing:
    datasets/jvs_music_ver1/
        jvs001/
            seg_001.wav
            seg_002.wav
            …
        jvs002/
            …

Usage:
    python prepare_jvs_music.py --data-root datasets/jvs_music_ver1
    python prepare_jvs_music.py --data-root datasets/jvs_music_ver1 --dry-run
"""

import argparse
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MIN_DUR = 2.0   # seconds — discard shorter segments
MAX_DUR = 8.0   # seconds — split longer segments
TOP_DB  = 35    # silence threshold (dB below peak)
GAP_SEC = 0.25  # merge intervals separated by less than this (breath pauses)


def _split_on_silence(y: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """
    Returns a list of (start_sample, end_sample) for speech/singing regions.
    Intervals closer than GAP_SEC are merged first.
    Remaining segments longer than MAX_DUR are recursively halved at the
    longest internal silence.
    """
    intervals = librosa.effects.split(y, top_db=TOP_DB, frame_length=2048, hop_length=512)
    if len(intervals) == 0:
        return []

    # Merge intervals separated by a breath-length pause
    gap = int(GAP_SEC * sr)
    merged: list[tuple[int, int]] = []
    s, e = int(intervals[0][0]), int(intervals[0][1])
    for ns, ne in intervals[1:]:
        ns, ne = int(ns), int(ne)
        if ns - e < gap:
            e = ne
        else:
            merged.append((s, e))
            s, e = ns, ne
    merged.append((s, e))

    # Split segments that exceed MAX_DUR at the longest internal silence
    result: list[tuple[int, int]] = []
    for s, e in merged:
        result.extend(_split_long(y, sr, s, e))
    return result


def _split_long(y: np.ndarray, sr: int, s: int, e: int, depth: int = 0) -> list[tuple[int, int]]:
    """Recursively split a segment at its longest silence if it exceeds MAX_DUR."""
    dur = (e - s) / sr
    if dur <= MAX_DUR or depth >= 6:
        return [(s, e)]

    chunk = y[s:e]
    frame_len = 2048
    hop = 512
    rms = librosa.feature.rms(y=chunk, frame_length=frame_len, hop_length=hop)[0]
    peak = rms.max()
    if peak == 0:
        return [(s, e)]
    silent = rms < (peak * 10 ** (-TOP_DB / 20))

    # Find the longest run of silent frames as split point
    best_run_len, best_run_center = 0, -1
    run_start, run_len = None, 0
    for i, sil in enumerate(silent):
        if sil:
            if run_start is None:
                run_start = i
            run_len += 1
        else:
            if run_start is not None and run_len > best_run_len:
                best_run_len = run_len
                best_run_center = run_start + run_len // 2
            run_start, run_len = None, 0
    if run_start is not None and run_len > best_run_len:
        best_run_center = run_start + run_len // 2

    if best_run_center <= 0:
        return [(s, e)]

    mid = s + best_run_center * hop
    # Guard: split must create two non-trivial halves
    if mid - s < int(0.5 * sr) or e - mid < int(0.5 * sr):
        return [(s, e)]

    left  = _split_long(y, sr, s, mid, depth + 1)
    right = _split_long(y, sr, mid, e, depth + 1)
    return left + right


def process_speaker(spk_dir: Path, dry_run: bool = False) -> int:
    raw_wav = spk_dir / "song_unique" / "wav" / "raw.wav"
    if not raw_wav.exists():
        print(f"  [SKIP] {spk_dir.name}: raw.wav not found")
        return 0

    y, sr = sf.read(str(raw_wav), dtype="float32", always_2d=False)
    intervals = _split_on_silence(y, sr)

    segments = [(s, e) for s, e in intervals if MIN_DUR <= (e - s) / sr <= MAX_DUR]
    n_kept = len(segments)

    if not dry_run:
        # Delete song_common
        song_common = spk_dir / "song_common"
        if song_common.exists():
            shutil.rmtree(song_common)

        # Write segments
        for i, (s, e) in enumerate(segments, 1):
            out_path = spk_dir / f"seg_{i:03d}.wav"
            sf.write(str(out_path), y[s:e], sr)

        # Delete song_unique
        shutil.rmtree(spk_dir / "song_unique")
    else:
        durs = [(e - s) / sr for s, e in intervals]
        print(f"  {spk_dir.name}: {len(intervals)} intervals → keep {n_kept} "
              f"({[f'{d:.1f}s' for d in durs]})")

    return n_kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare JVS-MuSiC for training")
    parser.add_argument("--data-root", default="datasets/jvs_music_ver1")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without changing any files")
    args = parser.parse_args()

    root = Path(args.data_root)
    spk_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("jvs"))

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(spk_dirs)} speakers in {root}")
    total = 0
    for spk_dir in spk_dirs:
        n = process_speaker(spk_dir, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"  {spk_dir.name}: {n} segments saved")
        total += n

    print(f"\nTotal segments {'(would be saved)' if args.dry_run else 'saved'}: {total}")


if __name__ == "__main__":
    main()
