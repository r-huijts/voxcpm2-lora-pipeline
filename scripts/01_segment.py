#!/usr/bin/env python3
"""
01_segment.py — Split long source recordings into training-length clips.

Cuts on silence so clips land on natural speech boundaries, then enforces the
3-30s window VoxCPM2 wants. Clips shorter than MIN are dropped; clips longer
than MAX are hard-split at the longest internal silence (falling back to a
time cut if no usable silence exists).

Usage:
    python 01_segment.py --input_dir raw/ --output_dir clips/
    python 01_segment.py --input_dir raw/ --output_dir clips/ \
        --min_sec 3 --max_sec 25 --silence_db -35 --min_silence_ms 400

Requires: pydub, ffmpeg on PATH.
    pip install pydub
"""
import argparse
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")


def iter_audio_files(input_dir: Path):
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in AUDIO_SUFFIXES:
            yield p


def split_on_silence_windowed(
    audio: AudioSegment,
    min_sec: float,
    max_sec: float,
    silence_thresh_db: int,
    min_silence_ms: int,
    keep_silence_ms: int,
) -> list[AudioSegment]:
    """
    Return a list of AudioSegments respecting [min_sec, max_sec].

    Strategy: find non-silent regions, greedily merge adjacent regions until
    adding the next would exceed max_sec, emit the accumulated segment. Regions
    longer than max_sec on their own are hard-split by time.
    """
    min_ms = int(min_sec * 1000)
    max_ms = int(max_sec * 1000)

    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
        seek_step=10,
    )
    if not nonsilent:
        return []

    # Pad each region a touch so we don't clip word onsets/offsets.
    padded = []
    for start, end in nonsilent:
        start = max(0, start - keep_silence_ms)
        end = min(len(audio), end + keep_silence_ms)
        padded.append((start, end))

    segments: list[AudioSegment] = []
    cur_start, cur_end = padded[0]

    for start, end in padded[1:]:
        prospective = end - cur_start
        if prospective <= max_ms:
            cur_end = end  # merge
        else:
            segments.extend(_emit(audio, cur_start, cur_end, min_ms, max_ms))
            cur_start, cur_end = start, end

    segments.extend(_emit(audio, cur_start, cur_end, min_ms, max_ms))
    return segments


def _emit(audio, start, end, min_ms, max_ms) -> list[AudioSegment]:
    """Emit one accumulated region, hard-splitting by time if over max."""
    span = end - start
    if span < min_ms:
        return []  # too short, drop
    if span <= max_ms:
        return [audio[start:end]]

    # Region itself exceeds max: hard-split by time into near-equal pieces.
    pieces = []
    n = (span // max_ms) + 1
    step = span // n
    for k in range(n):
        s = start + k * step
        e = end if k == n - 1 else start + (k + 1) * step
        if (e - s) >= min_ms:
            pieces.append(audio[s:e])
    return pieces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--min_sec", type=float, default=3.0)
    ap.add_argument("--max_sec", type=float, default=25.0,
                    help="Stay a touch under 30 to leave headroom.")
    ap.add_argument("--silence_db", type=int, default=-35,
                    help="dBFS below which audio counts as silence. "
                         "Quieter rooms: -40. Noisier: -30.")
    ap.add_argument("--min_silence_ms", type=int, default=400,
                    help="Minimum silence length to cut on.")
    ap.add_argument("--keep_silence_ms", type=int, default=120,
                    help="Padding kept around each clip edge.")
    ap.add_argument("--target_sr", type=int, default=16000,
                    help="Output sample rate. 16k matches the VoxCPM2 encoder; "
                         "the dataloader resamples anyway, so this is just to "
                         "keep clips small.")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_audio_files(args.input_dir))
    if not files:
        raise SystemExit(f"No audio files in {args.input_dir}")

    total_clips = 0
    total_seconds = 0.0
    counter = 1

    for src in files:
        print(f"\n=== {src.name} ===")
        audio = AudioSegment.from_file(src)
        audio = audio.set_channels(1).set_frame_rate(args.target_sr)

        segments = split_on_silence_windowed(
            audio,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            silence_thresh_db=args.silence_db,
            min_silence_ms=args.min_silence_ms,
            keep_silence_ms=args.keep_silence_ms,
        )

        if not segments:
            print("  No clips produced. Try a different --silence_db "
                  "(e.g. -40 for quiet recordings, -30 for noisy).")
            continue

        for seg in segments:
            out = args.output_dir / f"clip_{counter:04d}.wav"
            seg.export(out, format="wav")
            dur = len(seg) / 1000.0
            print(f"  {out.name}  {dur:5.1f}s")
            counter += 1
            total_clips += 1
            total_seconds += dur

    print(f"\nDone. {total_clips} clips, {total_seconds/60:.1f} min total "
          f"in {args.output_dir}")
    if total_seconds < 5 * 60:
        print("NOTE: under 5 minutes of audio. Usable for a quick LoRA but "
              "more (10-20 min) usually gives a more robust voice.")


if __name__ == "__main__":
    main()
