#!/usr/bin/env python3
"""
03_stitch.py — Concatenate generated chunks with natural pauses.

Reads the manifest.json written by 02_generate_nanovllm.py and joins the chunk wavs in
order, inserting silence at each break:
  - "short" gap  -> short pause (within a paragraph)
  - "long"  gap  -> longer pause (between paragraphs)
  - "none"       -> no trailing pause (last chunk)

Each chunk's leading/trailing silence is trimmed first so the inserted pauses
are the ONLY pauses at the seams (the model sometimes leaves ragged edges).
A short equal-power crossfade smooths each join. Optional final EBU R128
loudness normalization.

CANDIDATE SELECTION
    If a selection.json is present in the run-dir (or passed via --selection),
    it maps chunk id -> chosen candidate version, e.g. {"3": 2, "4": 1}. For a
    chunk with a pick, chunk_NNNN_v<K>.wav is stitched instead of the plain
    chunk_NNNN.wav. Chunks without a pick use the plain file as before. This
    lets you generate several candidates per chunk (02's interactive `cand`
    command), listen, and hand-pick the best of each for the final stitch.

No global speed change is applied — by design. If you want to slow the whole
thing, do it afterward with: ffmpeg -i final.wav -filter:a "atempo=0.85" out.wav

Usage:
    python 03_stitch.py --run-dir /workspace/narration/run01 \
        --output /workspace/narration/run01/final.wav

    # with hand-picked candidates
    python 03_stitch.py --run-dir ... --output final.wav \
        --selection /workspace/narration/run01/selection.json

    # tune pauses (override manifest config), add loudness mastering
    python 03_stitch.py --run-dir ... --output final.wav \
        --gap-scale 0.85 --loudnorm --lufs -16

Requires: numpy, soundfile. Optional: ffmpeg (for --loudnorm).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from _pipeline_config import load_voice_config, apply_config_defaults


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


def resolve_chunk_file(run_dir: Path, item: dict, selection: dict) -> Path:
    """
    Decide which wav to use for a chunk. If the chunk id has a pick in the
    selection map, use chunk_NNNN_v<K>.wav; otherwise the plain file from the
    manifest. Falls back to the plain file if the selected candidate is missing.
    """
    plain = run_dir / item["file"]
    cid = item.get("id")
    if cid is None or cid not in selection:
        return plain
    version = selection[cid]
    # Derive the versioned name from the manifest filename stem.
    stem = Path(item["file"]).stem  # e.g. chunk_0003
    cand = run_dir / f"{stem}_v{version}.wav"
    if cand.exists():
        return cand
    print(f"  WARNING chunk {cid}: selected v{version} not found "
          f"({cand.name}); falling back to {plain.name}")
    return plain


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Dir with chunk wavs + manifest.json from 02_generate_nanovllm.py.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--gap-scale", type=float, default=None,
                    help="Global multiplier on all per-chunk gaps "
                         "(else from manifest config; default 1.0).")
    ap.add_argument("--crossfade-ms", type=int, default=None,
                    help="Crossfade at every seam (else from manifest config).")
    ap.add_argument("--no-trim", action="store_true",
                    help="Don't trim per-chunk leading/trailing silence.")
    ap.add_argument("--loudnorm", action=argparse.BooleanOptionalAction, default=False,
                    help="Apply final EBU R128 loudness normalization (ffmpeg).")
    ap.add_argument("--lufs", type=float, default=-16.0,
                    help="Target integrated loudness (-23 broadcast, -16 podcast).")
    ap.add_argument("--selection", type=Path, default=None,
                    help="Path to selection.json mapping chunk id -> chosen "
                         "candidate version, e.g. {\"3\": 2, \"4\": 1}. For each "
                         "chunk, uses chunk_NNNN_vK.wav when a pick exists, else "
                         "falls back to the plain chunk_NNNN.wav. Defaults to "
                         "selection.json in --run-dir if present.")

    CONFIGURABLE = {"gap_scale", "crossfade_ms", "lufs", "loudnorm"}
    ap.add_argument("--config", type=Path, default=Path("voice.json"),
                    help="Shared per-voice defaults JSON (see scripts/_pipeline_config.py "
                         "and scripts/voice.example.json). Keys: gap_scale, crossfade_ms, "
                         "lufs, loudnorm. CLI flags always override it (pass "
                         "--no-loudnorm to force it off for one run even if voice.json "
                         "sets it); a gap_scale/crossfade_ms set here also takes "
                         "priority over the values baked into this run's manifest.json.")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=Path("voice.json"))
    config = load_voice_config(pre.parse_known_args()[0].config)
    apply_config_defaults(ap, config, CONFIGURABLE)

    args = ap.parse_args()

    if args.lufs != -16.0 and not args.loudnorm:
        print(f"WARNING: --lufs {args.lufs} has no effect without --loudnorm.",
              file=sys.stderr)

    manifest_path = args.run_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"No manifest.json in {args.run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    if not items:
        sys.exit("Manifest has no items.")

    # Load the candidate selection map, if any.
    sel_path = args.selection
    sel_source = "--selection"
    if sel_path is None:
        default_sel = args.run_dir / "selection.json"
        sel_path = default_sel if default_sel.exists() else None
        sel_source = "auto-detected"
    selection = {}
    if sel_path is not None:
        if not sel_path.exists():
            sys.exit(f"Selection file not found: {sel_path}")
        raw_sel = json.loads(sel_path.read_text(encoding="utf-8"))
        for k, v in raw_sel.items():
            if str(k).startswith("_"):
                continue  # allow comment keys
            selection[int(k)] = int(v)
        print(f"Selection loaded ({len(selection)} picks) from {sel_path.name} ({sel_source})")

    cfg = manifest.get("config", {})
    manifest_gap_scale = cfg.get("gap_scale", 1.0)
    manifest_crossfade_ms = cfg.get("crossfade_ms", 40)
    gap_scale = args.gap_scale if args.gap_scale is not None else manifest_gap_scale
    crossfade_ms = args.crossfade_ms if args.crossfade_ms is not None else manifest_crossfade_ms

    if args.gap_scale is not None and args.gap_scale != manifest_gap_scale:
        print(f"NOTE: using gap_scale={gap_scale} (from --gap-scale or voice.json), "
              f"overriding this run's manifest value of {manifest_gap_scale}.", file=sys.stderr)
    if args.crossfade_ms is not None and args.crossfade_ms != manifest_crossfade_ms:
        print(f"NOTE: using crossfade_ms={crossfade_ms} (from --crossfade-ms or voice.json), "
              f"overriding this run's manifest value of {manifest_crossfade_ms}.", file=sys.stderr)

    print(f"Stitching {len(items)} chunks "
          f"(gap_scale={gap_scale}, crossfade={crossfade_ms}ms)")

    timeline = None
    sr = None

    for item in items:
        wav_path = resolve_chunk_file(args.run_dir, item, selection)
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