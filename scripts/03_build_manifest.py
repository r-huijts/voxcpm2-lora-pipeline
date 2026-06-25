#!/usr/bin/env python3
"""
03_build_manifest.py — Turn clips + transcripts into VoxCPM2 train/val manifests.

Does the unglamorous-but-critical work the fine-tuning docs insist on:
  * Trims TRAILING silence to < 0.5s. Long trailing silence is THE most common
    cause of "generation never stops" after fine-tuning. Non-negotiable.
  * Filters clips outside the 3-30s window.
  * Drops clips whose transcript is empty or missing.
  * Adds `ref_audio` to ~40% of samples (another clip from the same speaker),
    so the model keeps both zero-shot and reference-based cloning ability.
  * Writes a `duration` field to speed up the dataloader's length filtering.
  * Splits into train/val.

Writes trimmed clips to a new dir so your originals stay intact.

Usage:
    python 03_build_manifest.py --clips_dir clips/ --out_dir dataset/
    python 03_build_manifest.py --clips_dir clips/ --out_dir dataset/ \
        --ref_ratio 0.4 --val_frac 0.05 --min_sec 3 --max_sec 30

Requires: pydub, ffmpeg.
    pip install pydub
"""
import argparse
import json
import random
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_leading_silence


def trim_trailing_silence(audio: AudioSegment, silence_thresh_db: int,
                          keep_ms: int) -> AudioSegment:
    """Trim trailing silence to at most keep_ms. Leading silence trimmed too."""
    # Leading
    lead = detect_leading_silence(audio, silence_threshold=silence_thresh_db)
    # Trailing: detect on the reversed segment
    trail = detect_leading_silence(audio.reverse(),
                                   silence_threshold=silence_thresh_db)
    start = max(0, lead - keep_ms)
    end = len(audio) - max(0, trail - keep_ms)
    if end <= start:
        return audio  # all silence by this threshold; leave as-is
    return audio[start:end]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips_dir", required=True, type=Path,
                    help="Dir with clip_XXXX.wav and clip_XXXX.txt")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Output dir for trimmed clips + manifests.")
    ap.add_argument("--ref_ratio", type=float, default=0.4,
                    help="Fraction of samples given a same-speaker ref_audio. "
                         "Docs recommend 0.3-0.5.")
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="Fraction held out for validation.")
    ap.add_argument("--min_sec", type=float, default=3.0)
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--silence_db", type=int, default=-40,
                    help="dBFS threshold for silence trimming.")
    ap.add_argument("--keep_ms", type=int, default=150,
                    help="Silence to keep at each edge after trimming. "
                         "Keeps it under the 0.5s danger line.")
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    if not args.clips_dir.is_dir():
        raise SystemExit(f"Clips dir not found: {args.clips_dir}")

    trimmed_dir = args.out_dir / "audio"
    trimmed_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(args.clips_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav in {args.clips_dir}")

    samples = []
    dropped = {"no_text": 0, "too_short": 0, "too_long": 0}

    for wav in wavs:
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            dropped["no_text"] += 1
            continue
        text = txt.read_text(encoding="utf-8").strip()
        text = " ".join(text.split())
        if not text:
            dropped["no_text"] += 1
            continue

        audio = AudioSegment.from_file(wav)
        audio = audio.set_channels(1).set_frame_rate(args.target_sr)
        audio = trim_trailing_silence(audio, args.silence_db, args.keep_ms)

        dur = len(audio) / 1000.0
        if dur < args.min_sec:
            dropped["too_short"] += 1
            continue
        if dur > args.max_sec:
            dropped["too_long"] += 1
            continue

        out_wav = trimmed_dir / wav.name
        audio.export(out_wav, format="wav")
        samples.append({
            "audio": str(out_wav.resolve()),
            "text": text,
            "duration": round(dur, 2),
        })

    if not samples:
        raise SystemExit("No usable samples after filtering. Check transcripts "
                         "and duration window.")

    # --- Assign ref_audio to a fraction of samples ---
    # ref must be a DIFFERENT clip from the same speaker (here: any other clip).
    all_paths = [s["audio"] for s in samples]
    n_ref = int(len(samples) * args.ref_ratio)
    ref_indices = set(random.sample(range(len(samples)), n_ref)) if n_ref else set()

    for i, s in enumerate(samples):
        if i in ref_indices and len(all_paths) > 1:
            choices = [p for p in all_paths if p != s["audio"]]
            s["ref_audio"] = random.choice(choices)

    # --- Train/val split ---
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_frac)) if args.val_frac > 0 else 0
    val = samples[:n_val]
    train = samples[n_val:]

    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if val:
        with val_path.open("w", encoding="utf-8") as f:
            for s in val:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    total_min = sum(s["duration"] for s in samples) / 60.0
    n_with_ref = sum(1 for s in samples if "ref_audio" in s)

    print(f"\nManifest built in {args.out_dir}")
    print(f"  train: {len(train)} samples -> {train_path.name}")
    print(f"  val:   {len(val)} samples -> {val_path.name}"
          if val else "  val:   (none)")
    print(f"  total audio: {total_min:.1f} min")
    print(f"  with ref_audio: {n_with_ref} ({n_with_ref/len(samples)*100:.0f}%)")
    print(f"  dropped: {dropped}")
    if total_min < 5:
        print("  NOTE: under 5 min usable. Fine for a test LoRA; consider more "
              "for production.")


if __name__ == "__main__":
    main()
