#!/usr/bin/env python3
"""
patch_nanovllm.py — Re-apply the two nano-vllm-voxcpm fixes needed for
single-sequence LoRA inference. Idempotent: safe to run any number of times.
Run after every (re)install of nano-vllm-voxcpm, or bake into a Dockerfile.

Two upstream bugs are patched:

  1. lora_ops/triton_ops/lora_shrink_op.py
     `_SMALL_M_THRESHOLD = 32` routes small batches (single-sequence decode,
     M < 32) into `_lora_shrink_small_m_kernel`, whose `tl.dot` on a 1xK row
     violates Triton's M>=16 constraint and crashes. Setting the threshold to
     0 disables the small-m fast path entirely, falling through to the regular
     `_lora_shrink_kernel`, which pads M up to BLOCK_M internally and works.

  2. engine/model_runner.py
     In run_model(), `self.graphs.get("lora")` is read before the
     `enforce_eager` short-circuit. When enforce_eager=True, capture_cudagraph()
     is skipped so `self.graphs` never exists -> AttributeError. Guarding with
     getattr(self, "graphs", {}) makes the line safe in eager mode.

Usage:
    python patch_nanovllm.py            # apply (idempotent)
    python patch_nanovllm.py --check    # report status, change nothing
    python patch_nanovllm.py --revert   # restore from .orig backups
"""
import argparse
import shutil
import sys
from pathlib import Path


def _find_package_root() -> Path:
    try:
        import nanovllm_voxcpm
    except ImportError:
        sys.exit("nano-vllm-voxcpm is not installed in this environment.")
    return Path(nanovllm_voxcpm.__file__).resolve().parent


# (relative path, marker that means "already patched", old line, new line)
PATCHES = [
    (
        "lora_ops/triton_ops/lora_shrink_op.py",
        "_SMALL_M_THRESHOLD = 0",
        "_SMALL_M_THRESHOLD = 32",
        "_SMALL_M_THRESHOLD = 0",
    ),
    (
        "engine/model_runner.py",
        'getattr(self, "graphs", {}).get("lora")',
        'has_lora_graph = has_active_lora and bool(self.graphs.get("lora"))',
        'has_lora_graph = has_active_lora and bool(getattr(self, "graphs", {}).get("lora"))',
    ),
]


def _status(path: Path, patched_marker: str, old: str) -> str:
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8")
    if patched_marker in text:
        return "patched"
    if old in text:
        return "unpatched"
    return "unknown"  # neither form found — file changed upstream


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Report patch status only; make no changes.")
    ap.add_argument("--revert", action="store_true",
                    help="Restore original files from .orig backups.")
    args = ap.parse_args()

    root = _find_package_root()
    print(f"nano-vllm-voxcpm at: {root}\n")

    any_problem = False

    for rel, marker, old, new in PATCHES:
        path = root / rel
        backup = path.with_suffix(path.suffix + ".orig")

        if args.revert:
            if backup.exists():
                shutil.copy2(backup, path)
                print(f"[revert]  {rel}  <- {backup.name}")
            else:
                print(f"[revert]  {rel}  no .orig backup, skipped")
            continue

        st = _status(path, marker, old)

        if args.check:
            print(f"[{st:>9}] {rel}")
            if st in ("missing", "unknown"):
                any_problem = True
            continue

        # apply mode
        if st == "patched":
            print(f"[skip]    {rel}  already patched")
        elif st == "unpatched":
            if not backup.exists():
                shutil.copy2(path, backup)
            text = path.read_text(encoding="utf-8")
            text = text.replace(old, new)
            path.write_text(text, encoding="utf-8")
            print(f"[patched] {rel}  (backup: {backup.name})")
        elif st == "missing":
            print(f"[ERROR]   {rel}  file not found")
            any_problem = True
        else:  # unknown
            print(f"[ERROR]   {rel}  neither original nor patched line found; "
                  f"upstream may have changed — inspect manually")
            any_problem = True

    if any_problem:
        print("\nOne or more patches could not be applied cleanly. "
              "The package version may differ from what these patches target.")
        sys.exit(1)

    if not args.check and not args.revert:
        print("\nDone. Both fixes in place. Safe to re-run anytime.")


if __name__ == "__main__":
    main()