#!/usr/bin/env python3
"""
generate_and_stitch.py — run 02_generate_nanovllm.py then 03_stitch.py back
to back.

Stage 1 (01_chunk.py) always needs a manual pause to review/edit plan.json.
Stages 2 and 3 don't — you run them one after another every time. This just
saves retyping --run-dir/--output for the stitch step.

Every flag 02_generate_nanovllm.py understands is accepted here and forwarded
to it unchanged (--plan, --lora, --reference, --cfg, --controllable, ...).
A small set of stitch-only flags are also accepted and forwarded to
03_stitch.py; --run-dir defaults to --out-dir and --output defaults to
<out-dir>/final.wav if not given explicitly. voice.json is picked up by each
underlying script exactly as if you'd run them separately.

Usage:
    python scripts/generate_and_stitch.py \\
        --plan plan.json --out-dir run01 \\
        --loudnorm --lufs -16

    # with voice.json set up, --lora/--reference/tuning flags can be omitted,
    # same as running 02_generate_nanovllm.py directly

If generation fails, stitching is skipped.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATE = HERE / "02_generate_nanovllm.py"
STITCH = HERE / "03_stitch.py"


def main():
    argv = sys.argv[1:]

    # Peek at --out-dir/--config without fully parsing 02's ~25 other flags;
    # both are needed here, everything else is forwarded to 02 untouched.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--out-dir", type=Path)
    peek.add_argument("--config", type=Path, default=None)
    peek_args, _ = peek.parse_known_args(argv)
    if peek_args.out_dir is None:
        sys.exit("--out-dir is required (forwarded to 02_generate_nanovllm.py).")

    # Stitch-only flags, parsed here so defaults can be computed; anything not
    # matched (--plan, --lora, --reference, --cfg, ...) passes through as-is.
    stitch_ap = argparse.ArgumentParser(add_help=False)
    stitch_ap.add_argument("--run-dir", type=Path, default=None)
    stitch_ap.add_argument("--output", type=Path, default=None)
    stitch_ap.add_argument("--gap-scale", type=float, default=None)
    stitch_ap.add_argument("--crossfade-ms", type=int, default=None)
    stitch_ap.add_argument("--no-trim", action="store_true", default=False)
    stitch_ap.add_argument("--loudnorm", action=argparse.BooleanOptionalAction, default=None)
    stitch_ap.add_argument("--lufs", type=float, default=None)
    stitch_ap.add_argument("--selection", type=Path, default=None)
    stitch_args, generate_argv = stitch_ap.parse_known_args(argv)

    run_dir = stitch_args.run_dir or peek_args.out_dir
    output = stitch_args.output or (run_dir / "final.wav")

    print(f"[1/2] Generating -> {peek_args.out_dir}")
    result = subprocess.run([sys.executable, str(GENERATE), *generate_argv])
    if result.returncode != 0:
        sys.exit(f"Generation failed (exit {result.returncode}); stitching skipped.")

    stitch_argv = ["--run-dir", str(run_dir), "--output", str(output)]
    if peek_args.config is not None:
        stitch_argv += ["--config", str(peek_args.config)]
    if stitch_args.gap_scale is not None:
        stitch_argv += ["--gap-scale", str(stitch_args.gap_scale)]
    if stitch_args.crossfade_ms is not None:
        stitch_argv += ["--crossfade-ms", str(stitch_args.crossfade_ms)]
    if stitch_args.no_trim:
        stitch_argv += ["--no-trim"]
    if stitch_args.loudnorm is True:
        stitch_argv += ["--loudnorm"]
    elif stitch_args.loudnorm is False:
        stitch_argv += ["--no-loudnorm"]
    if stitch_args.lufs is not None:
        stitch_argv += ["--lufs", str(stitch_args.lufs)]
    if stitch_args.selection is not None:
        stitch_argv += ["--selection", str(stitch_args.selection)]

    print(f"[2/2] Stitching -> {output}")
    result = subprocess.run([sys.executable, str(STITCH), *stitch_argv])
    if result.returncode != 0:
        sys.exit(f"Stitching failed (exit {result.returncode}).")


if __name__ == "__main__":
    main()
