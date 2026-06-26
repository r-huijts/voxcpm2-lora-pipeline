#!/usr/bin/env python3
"""
02_generate.py — Generate one audio file per chunk from a reviewed plan.

Reads the JSON plan from 01_chunk.py (after you've edited it), and synthesizes
each chunk with the trained LoRA in Controllable Cloning mode (Mode 1):
  - the LoRA supplies the voice/timbre
  - a fixed reference clip is re-anchored on EVERY chunk to fight drift
  - each chunk's per-chunk control tag steers cadence: (tag)text
  - cfg_value defaults low (1.6) for long-form stability

Outputs: <out_dir>/chunk_0001.wav, chunk_0002.wav, ... plus a manifest.json
that records each chunk's gap_after, so the stitcher (03_stitch.py) knows the
pause pattern without re-reading the plan.

Usage:
    python 02_generate.py --plan plan.json \
        --lora /workspace/voxcpm2-lora-pipeline/checkpoints/lora/step_0000999 \
        --reference /workspace/voxcpm_project/references/ref_voice.wav \
        --out-dir /workspace/narration/run01

    # tuning
    python 02_generate.py --plan plan.json --lora ... --reference ... \
        --out-dir ... --cfg 1.5 --timesteps 24

Requires: voxcpm, soundfile, ffmpeg (for reference conversion).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from voxcpm import VoxCPM

# LoRAConfig location varies by version — same fallback as 05_infer.py.
try:
    from voxcpm.model.voxcpm2 import LoRAConfig
except ImportError:
    from voxcpm import LoRAConfig

torch.set_float32_matmul_precision("high")

BASE = "openbmb/VoxCPM2"


def load_lora_config(lora_path: Path):
    """
    Build a LoRAConfig from the checkpoint's own lora_config.json so the adapter
    is created at the trained rank (otherwise the loader defaults to r=8 and
    weight loading fails on an r=32 checkpoint).
    """
    cfg_file = lora_path / "lora_config.json"
    if not cfg_file.exists():
        return None
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    cfg = data.get("lora_config", data)
    return LoRAConfig(**cfg)


def load_model(lora_path: Path) -> VoxCPM:
    print(f"Loading base + LoRA: {lora_path}")
    lora_config = load_lora_config(lora_path)
    if lora_config is None:
        sys.exit(f"No lora_config.json under {lora_path} — cannot match rank.")
    return VoxCPM.from_pretrained(
        BASE,
        lora_config=lora_config,
        lora_weights_path=str(lora_path),
        load_denoiser=False,
    )


def convert_reference(src: Path, dst: Path) -> Path:
    """16 kHz mono WAV reference, as the encoder expects."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def apply_control(text: str, control: str) -> str:
    """Controllable Cloning convention: (instruction)text, no space."""
    control = (control or "").strip()
    return f"({control}){text}" if control else text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--lora", required=True, type=Path,
                    help="LoRA checkpoint dir (e.g. .../step_0000999).")
    ap.add_argument("--reference", required=True, type=Path,
                    help="Reference voice clip (timbre anchor, re-used per chunk).")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cfg", type=float, default=1.6,
                    help="cfg_value. 1.5-1.6 recommended for long-form stability.")
    ap.add_argument("--timesteps", type=int, default=20,
                    help="inference_timesteps (detail vs speed).")
    ap.add_argument("--normalize", action="store_true", default=False,
                    help="Let the model normalize text. OFF by default because "
                         "the chunking LLM already expanded numbers/names.")
    ap.add_argument("--start-at", type=int, default=1,
                    help="Resume: skip chunks with id < this (1-indexed).")
    args = ap.parse_args()

    if not args.plan.exists():
        sys.exit(f"Plan not found: {args.plan}")
    if not args.reference.exists():
        sys.exit(f"Reference not found: {args.reference}")
    if not (args.lora / "lora_config.json").exists():
        sys.exit(f"No lora_config.json in {args.lora}")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    chunks = plan.get("chunks", [])
    if not chunks:
        sys.exit("Plan has no chunks.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Convert the reference once; re-anchored on every chunk.
    ref16k = convert_reference(args.reference, args.out_dir / "reference_16k.wav")

    model = load_model(args.lora)
    print(f"Model loaded. Generating {len(chunks)} chunks "
          f"(cfg={args.cfg}, timesteps={args.timesteps}, "
          f"normalize={args.normalize})\n")

    manifest = {
        "config": plan.get("config", {}),
        "register": plan.get("register"),
        "items": [],
    }

    for c in chunks:
        cid = int(c["id"])
        text = c["text"]
        control = c.get("control", "")
        gap_after = c.get("gap_after", "short")
        wav_name = f"chunk_{cid:04d}.wav"
        wav_path = args.out_dir / wav_name

        # Always record in manifest so the stitcher has the full pattern,
        # even for chunks skipped on resume.
        manifest["items"].append({
            "id": cid, "file": wav_name, "gap_after": gap_after,
            "control": control,
        })

        if cid < args.start_at:
            print(f"[{cid:03d}] skipped (resume)")
            continue

        controlled = apply_control(text, control)
        print(f"[{cid:03d}] ({control}) {text[:64]}"
              f"{'...' if len(text) > 64 else ''}")

        wav = model.generate(
            text=controlled,
            reference_wav_path=str(ref16k),
            cfg_value=args.cfg,
            inference_timesteps=args.timesteps,
            normalize=args.normalize,
        )
        sf.write(wav_path, wav, model.tts_model.sample_rate)

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nDone. {len(chunks)} chunks in {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Next: python 03_stitch.py --run-dir {args.out_dir} "
          f"--output {args.out_dir / 'final.wav'}")


if __name__ == "__main__":
    main()
