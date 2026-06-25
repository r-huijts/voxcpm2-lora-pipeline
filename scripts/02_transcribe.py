#!/usr/bin/env python3
"""
02_transcribe.py — Transcribe each clip to a matching .txt sidecar.

Produces clip_0001.txt next to clip_0001.wav. The manifest builder (step 3)
reads these. Uses faster-whisper (CTranslate2) for speed; Dutch by default.

ACCURACY MATTERS: the docs are blunt that mismatched transcripts degrade both
cloning quality and text adherence. Whisper is good but not perfect on names
and numbers — after this runs, spot-check the .txt files, especially proper
nouns (place names, rider names) and any digits. Fix them by hand. This script
is a first pass, not the final word.

Usage:
    python 02_transcribe.py --clips_dir clips/ --language nl --model large-v3
    python 02_transcribe.py --clips_dir clips/ --language nl --model medium \
        --device cpu --compute_type int8

Requires:
    pip install faster-whisper
"""
import argparse
from pathlib import Path

from faster_whisper import WhisperModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips_dir", required=True, type=Path)
    ap.add_argument("--language", default="nl",
                    help="ISO code. 'nl' for Dutch. Leave empty to auto-detect.")
    ap.add_argument("--model", default="large-v3",
                    help="faster-whisper model size. large-v3 best for names; "
                         "medium is faster.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute_type", default="float16",
                    help="float16 on GPU; int8 or int8_float16 to save VRAM; "
                         "int8 on CPU.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-transcribe clips that already have a .txt.")
    args = ap.parse_args()

    if not args.clips_dir.is_dir():
        raise SystemExit(f"Clips dir not found: {args.clips_dir}")

    wavs = sorted(args.clips_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files in {args.clips_dir}")

    print(f"Loading faster-whisper '{args.model}' on {args.device} "
          f"({args.compute_type})...")
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)
    print("Model loaded.\n")

    lang = args.language if args.language else None
    done = 0
    skipped = 0

    for wav in wavs:
        txt_path = wav.with_suffix(".txt")
        if txt_path.exists() and not args.overwrite:
            skipped += 1
            continue

        segments, info = model.transcribe(
            str(wav),
            language=lang,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        text = " ".join(text.split())  # collapse whitespace

        txt_path.write_text(text, encoding="utf-8")
        print(f"  {wav.name} -> {text[:70]}{'...' if len(text) > 70 else ''}")
        done += 1

    print(f"\nDone. Transcribed {done}, skipped {skipped} (already had .txt).")
    print("REVIEW STEP: open the .txt files and fix names, numbers, and any "
          "obvious errors before building the manifest. Transcript accuracy "
          "directly affects cloning quality.")


if __name__ == "__main__":
    main()
