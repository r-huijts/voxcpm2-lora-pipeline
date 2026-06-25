#!/usr/bin/env python3
"""
00_segment_srt.py — Slice long WAVs into training clips using their SRT timing.

Use this INSTEAD of 01_segment.py + 02_transcribe.py when you already have
timestamped subtitles. It groups consecutive SRT cues into [min_sec, max_sec]
windows on sentence boundaries, cuts the audio at those exact timestamps, and
writes each clip with its transcript already attached — no re-transcription.

Pairing: for each <name>.wav it looks for <name>.srt in the same dir (or in
--srt_dir). So 1.wav <-> 1.srt, clip_0003.wav <-> clip_0003.srt, etc.

Output: clip_0001.wav + clip_0001.txt pairs, ready for 03_build_manifest.py.

Usage:
    python 00_segment_srt.py --input_dir raw/ --output_dir clips/
    python 00_segment_srt.py --input_dir raw/ --srt_dir srt/ --output_dir clips/ \
        --min_sec 3 --max_sec 20

Requires: pydub, ffmpeg.
    pip install pydub
"""
import argparse
import re
from pathlib import Path

from pydub import AudioSegment

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
SENTENCE_END = re.compile(r"[.!?]['\"”’)]?\s*$")


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    """Return [(start_s, end_s, text), ...] for non-empty cues."""
    raw = path.read_text(encoding="utf-8-sig")  # tolerate BOM
    blocks = re.split(r"\n\s*\n", raw.strip())
    cues = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        # find the timestamp line (usually line 2, but be tolerant)
        ts_line = None
        ts_idx = None
        for i, ln in enumerate(lines):
            if TS_RE.search(ln):
                ts_line = ln
                ts_idx = i
                break
        if ts_line is None:
            continue
        m = TS_RE.search(ts_line)
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        text = " ".join(lines[ts_idx + 1:]).strip()
        text = " ".join(text.split())
        if not text:
            continue
        cues.append((start, end, text))
    return cues


def group_cues(cues, min_sec, max_sec):
    """
    Greedily merge consecutive cues into windows <= max_sec, preferring to
    close a window at a sentence end. Windows shorter than min_sec are merged
    forward (or dropped if nothing follows).
    """
    groups = []
    cur = None  # [start, end, text]

    for start, end, text in cues:
        if cur is None:
            cur = [start, end, text]
            continue

        prospective = end - cur[0]
        ends_sentence = bool(SENTENCE_END.search(cur[2]))

        # Close the current window if we're at/over max, OR we've passed min
        # and the current window already ends on a sentence boundary.
        if prospective > max_sec:
            if (cur[1] - cur[0]) >= min_sec:
                groups.append(cur)
                cur = [start, end, text]
            else:
                # too short but adding overflows — take it anyway, better a
                # slightly long clip than a sub-min one
                cur = [cur[0], end, f"{cur[2]} {text}".strip()]
        elif (cur[1] - cur[0]) >= min_sec and ends_sentence:
            groups.append(cur)
            cur = [start, end, text]
        else:
            cur = [cur[0], end, f"{cur[2]} {text}".strip()]

    if cur is not None:
        if (cur[1] - cur[0]) >= min_sec or not groups:
            groups.append(cur)
        else:
            # tail too short: glue onto previous group
            groups[-1][1] = cur[1]
            groups[-1][2] = f"{groups[-1][2]} {cur[2]}".strip()

    return groups


def find_srt(wav: Path, srt_dir: Path | None) -> Path | None:
    candidates = []
    if srt_dir:
        candidates.append(srt_dir / f"{wav.stem}.srt")
    candidates.append(wav.with_suffix(".srt"))
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", required=True, type=Path,
                    help="Dir with the long WAVs (and SRTs, unless --srt_dir).")
    ap.add_argument("--srt_dir", type=Path, default=None,
                    help="Optional separate dir holding the .srt files.")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--min_sec", type=float, default=3.0)
    ap.add_argument("--max_sec", type=float, default=20.0)
    ap.add_argument("--pad_ms", type=int, default=80,
                    help="Padding added at each clip edge so word onsets/"
                         "offsets aren't clipped by tight subtitle timing.")
    ap.add_argument("--target_sr", type=int, default=16000)
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wavs = [p for p in sorted(args.input_dir.iterdir())
            if p.suffix.lower() in AUDIO_SUFFIXES]
    if not wavs:
        raise SystemExit(f"No audio in {args.input_dir}")

    counter = 1
    total_clips = 0
    total_seconds = 0.0
    missing_srt = []

    for wav in wavs:
        srt = find_srt(wav, args.srt_dir)
        if srt is None:
            missing_srt.append(wav.name)
            continue

        print(f"\n=== {wav.name}  (srt: {srt.name}) ===")
        cues = parse_srt(srt)
        if not cues:
            print("  No usable cues. Skipping.")
            continue

        audio = AudioSegment.from_file(wav)
        audio = audio.set_channels(1).set_frame_rate(args.target_sr)
        dur_ms = len(audio)

        groups = group_cues(cues, args.min_sec, args.max_sec)

        for start, end, text in groups:
            s_ms = max(0, int(start * 1000) - args.pad_ms)
            e_ms = min(dur_ms, int(end * 1000) + args.pad_ms)
            if e_ms <= s_ms:
                continue
            seg = audio[s_ms:e_ms]
            dur = len(seg) / 1000.0

            out_wav = args.output_dir / f"clip_{counter:04d}.wav"
            out_txt = args.output_dir / f"clip_{counter:04d}.txt"
            seg.export(out_wav, format="wav")
            out_txt.write_text(text, encoding="utf-8")

            print(f"  clip_{counter:04d}  {dur:5.1f}s  {text[:60]}"
                  f"{'...' if len(text) > 60 else ''}")
            counter += 1
            total_clips += 1
            total_seconds += dur

    print(f"\nDone. {total_clips} clips, {total_seconds/60:.1f} min total "
          f"in {args.output_dir}")
    if missing_srt:
        print(f"WARNING: no SRT found for {len(missing_srt)} file(s): "
              f"{', '.join(missing_srt)}")
    print("\nNext: python scripts/03_build_manifest.py "
          f"--clips_dir {args.output_dir} --out_dir dataset/")
    print("(Skip 01_segment and 02_transcribe — the SRT already gave you "
          "aligned text.)")


if __name__ == "__main__":
    main()
