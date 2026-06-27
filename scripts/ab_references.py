#!/usr/bin/env python3
"""
ab_references.py — A/B multiple reference clips on the same plan (Mode 1).

Generates the SAME plan once per reference clip, each into its own out-dir, then
(optionally) stitches each so you can listen and pick the reference that gives
the best voice. The reference clip strongly shapes Mode 1 output — different
clips settle differently and carry different expressiveness.

Point it at a directory of candidate reference clips (e.g. 5 hand-picked clips),
or pass them explicitly. Each clip is converted to 16k mono internally by the
generator.

Usage:
    python ab_references.py \
        --plan plan.json \
        --lora /workspace/voxcpm2-lora-pipeline/checkpoints/lora/step_0000999 \
        --refs-dir /workspace/narration/ref_candidates \
        --out-root /workspace/narration/ab_refs \
        --stitch

    # explicit clips instead of a directory:
    python ab_references.py --plan plan.json --lora ... \
        --refs clipA.wav clipB.wav clipC.wav --out-root ... --stitch

    # pass-through generation params:
    python ab_references.py ... --cfg 1.6 --timesteps 30 --first-n 6

This is a thin wrapper: it shells out to 02_generate.py (and 03_stitch.py with
--stitch) per reference, so all their flags/behaviour stay identical.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEN = HERE / "02_generate.py"
STITCH = HERE / "03_stitch.py"
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a")


def collect_refs(refs_dir, refs):
    clips = []
    if refs_dir:
        clips += [p for p in sorted(Path(refs_dir).iterdir())
                  if p.suffix.lower() in AUDIO_SUFFIXES]
    if refs:
        clips += [Path(r) for r in refs]
    # de-dup, preserve order
    seen, out = set(), []
    for c in clips:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def maybe_trim_plan(plan_path: Path, first_n: int, work_dir: Path) -> Path:
    """If --first-n is set, write a trimmed plan (first N chunks) for quick A/B."""
    if not first_n:
        return plan_path
    import json
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    chunks = plan.get("chunks", [])[:first_n]
    if chunks:
        chunks[-1]["gap_after_ms"] = 0  # clean end on the trimmed plan
    plan["chunks"] = chunks
    work_dir.mkdir(parents=True, exist_ok=True)
    trimmed = work_dir / "plan_first_n.json"
    trimmed.write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"Trimmed plan to first {first_n} chunks: {trimmed}")
    return trimmed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--lora", required=True, type=Path)
    ap.add_argument("--refs-dir", type=Path, default=None,
                    help="Directory of candidate reference clips.")
    ap.add_argument("--refs", nargs="*", default=None,
                    help="Explicit reference clip paths.")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="Parent dir; each reference gets a subdir under here.")
    ap.add_argument("--stitch", action="store_true",
                    help="Also stitch each run to final.wav.")
    ap.add_argument("--first-n", type=int, default=0,
                    help="Only generate the first N chunks (fast A/B).")
    # pass-through generation params
    ap.add_argument("--cfg", type=float, default=1.6)
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--gap-scale", type=float, default=None,
                    help="Passed to stitch (if --stitch).")
    args = ap.parse_args()

    clips = collect_refs(args.refs_dir, args.refs)
    if not clips:
        sys.exit("No reference clips found (use --refs-dir or --refs).")
    missing = [c for c in clips if not c.exists()]
    if missing:
        sys.exit(f"Missing clips: {missing}")

    plan_path = maybe_trim_plan(args.plan, args.first_n, args.out_root)

    print(f"A/B over {len(clips)} reference clips:\n  " +
          "\n  ".join(c.name for c in clips) + "\n")

    for clip in clips:
        tag = clip.stem
        out_dir = args.out_root / f"ref_{tag}"
        print(f"\n=== reference: {clip.name} -> {out_dir} ===")
        gen_cmd = [
            sys.executable, str(GEN),
            "--plan", str(plan_path),
            "--lora", str(args.lora),
            "--reference", str(clip),
            "--out-dir", str(out_dir),
            "--cfg", str(args.cfg),
            "--timesteps", str(args.timesteps),
        ]
        subprocess.run(gen_cmd, check=True)

        if args.stitch:
            stitch_cmd = [
                sys.executable, str(STITCH),
                "--run-dir", str(out_dir),
                "--output", str(out_dir / "final.wav"),
            ]
            if args.gap_scale is not None:
                stitch_cmd += ["--gap-scale", str(args.gap_scale)]
            subprocess.run(stitch_cmd, check=True)

    print(f"\nDone. Compare the final.wav (or chunks) under {args.out_root}/ref_*/")
    print("Pick the reference whose voice settles fastest and sounds most like him.")


if __name__ == "__main__":
    main()
