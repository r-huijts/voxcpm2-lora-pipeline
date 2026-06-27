#!/usr/bin/env python3
"""
03_stitch.py — Concatenate generated chunks with natural pauses.

Reads the manifest.json written by 02_generate.py and joins the chunk wavs in
order, inserting silence at each break:
  - "short" gap  -> short pause (within a paragraph)
  - "long"  gap  -> longer pause (between paragraphs)
  - "none"       -> no trailing pause (last chunk)

Each chunk's leading/trailing silence is trimmed first so the inserted pauses
are the ONLY pauses at the seams (the model sometimes leaves ragged edges).
A short equal-power crossfade smooths each join. Optional final EBU R128
loudness normalization.

No global speed change is applied — by design. If you want to slow the whole
thing, do it afterward with: ffmpeg -i final.wav -filter:a "atempo=0.85" out.wav

Usage:
    python 03_stitch.py --run-dir /workspace/narration/run01 \
        --output /workspace/narration/run01/final.wav

    # tune pauses (override manifest config), add loudness mastering
    python 03_stitch.py --run-dir ... --output final.wav \
        --short-ms 200 --long-ms 600 --loudnorm --lufs -16

Requires: numpy, soundfile. Optional: ffmpeg (for --loudnorm).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def trim_silence(audio: np.ndarray, sr: int, thresh_db: float = -40.0,
                 keep_ms: int = 30) -> np.ndarray:
    """Trim leading/trailing silence below thresh_db, keeping a small margin."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    amp = np.abs(audio)
    if amp.max() <= 0:
        return audio
    thresh = (10 ** (thresh_db / 20.0)) * amp.max()
    above = np.where(amp > thresh)[0]
    if len(above) == 0:
        return audio
    keep = int(sr * keep_ms / 1000)
    start = max(0, above[0] - keep)
    end = min(len(audio), above[-1] + keep)
    return audio[start:end]


def silence(sr: int, ms: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000), dtype=np.float32)


def equal_power_crossfade(a: np.ndarray, b: np.ndarray, sr: int,
                          ms: int = 40) -> np.ndarray:
    """Join a+b with an equal-power crossfade of `ms` over the overlap."""
    n = int(sr * ms / 1000)
    if n <= 0 or len(a) < n or len(b) < n:
        return np.concatenate([a, b])
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    head, tail = a[:-n], a[-n:]
    overlap = tail * fade_out + b[:n] * fade_in
    return np.concatenate([head, overlap, b[n:]])


def loudnorm(audio: np.ndarray, sr: int, lufs: float, tp: float = -1.0) -> np.ndarray:
    """
    EBU R128 loudness normalization via pyloudnorm (ITU-R BS.1770 meter).
    Measures integrated loudness, applies the gain to hit `lufs`, then
    guards the true peak to <= tp dBFS with a simple limiter ceiling.
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if not np.isfinite(loudness):
        return audio  # silent or unmeasurable; leave as-is
    normalized = pyln.normalize.loudness(audio, loudness, lufs)

    # True-peak guard: scale down if we exceed the ceiling.
    ceiling = 10 ** (tp / 20.0)
    peak = np.abs(normalized).max()
    if peak > ceiling and peak > 0:
        normalized = normalized * (ceiling / peak)
    return normalized.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Dir with chunk wavs + manifest.json from 02_generate.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--gap-scale", type=float, default=None,
                    help="Global multiplier on all per-chunk gaps "
                         "(else from manifest config; default 1.0).")
    ap.add_argument("--crossfade-ms", type=int, default=None,
                    help="Crossfade at every seam (else from manifest config).")
    ap.add_argument("--no-trim", action="store_true",
                    help="Don't trim per-chunk leading/trailing silence.")
    ap.add_argument("--loudnorm", action="store_true",
                    help="Apply final EBU R128 loudness normalization (ffmpeg).")
    ap.add_argument("--lufs", type=float, default=-16.0,
                    help="Target integrated loudness (-23 broadcast, -16 podcast).")
    args = ap.parse_args()

    manifest_path = args.run_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No manifest.json in {args.run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    if not items:
        sys.exit("Manifest has no items.")

    cfg = manifest.get("config", {})
    gap_scale = args.gap_scale if args.gap_scale is not None else cfg.get("gap_scale", 1.0)
    crossfade_ms = args.crossfade_ms if args.crossfade_ms is not None else cfg.get("crossfade_ms", 40)

    print(f"Stitching {len(items)} chunks "
          f"(gap_scale={gap_scale}, crossfade={crossfade_ms}ms)")

    timeline = None
    sr = None

    for item in items:
        wav_path = args.run_dir / item["file"]
        if not wav_path.exists():
            sys.exit(f"Missing chunk audio: {wav_path}")
        audio, file_sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr is None:
            sr = file_sr
        elif file_sr != sr:
            sys.exit(f"Sample-rate mismatch in {wav_path} ({file_sr} vs {sr}).")

        if not args.no_trim:
            audio = trim_silence(audio, sr)

        if timeline is None:
            timeline = audio
        else:
            timeline = equal_power_crossfade(timeline, audio, sr, crossfade_ms)

        # Per-chunk numeric gap, scaled globally.
        gap_ms = float(item.get("gap_after_ms", 300)) * gap_scale
        if gap_ms > 0:
            timeline = np.concatenate([timeline, silence(sr, int(round(gap_ms)))])

    # Optional loudness normalization (in-memory, pyloudnorm).
    if args.loudnorm:
        print(f"Loudness normalizing to {args.lufs} LUFS (EBU R128)...")
        timeline = loudnorm(timeline, sr, args.lufs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, timeline, sr)

    dur = len(timeline) / sr
    print(f"\nDone: {args.output}  ({dur:.1f}s)")
    print("To slow overall tempo if needed:")
    print(f"  ffmpeg -i {args.output} -filter:a \"atempo=0.85\" "
          f"{args.output.with_name(args.output.stem + '_slow.wav')}")


if __name__ == "__main__":
    main()
