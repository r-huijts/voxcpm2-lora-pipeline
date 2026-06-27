#!/usr/bin/env python3
"""
02_generate_hifi.py — Hi-Fi (continuation) cloning variant of 02_generate.py.

Same plan in, same manifest out (the stitcher 03_stitch.py works unchanged).
The differences from the Controllable-Cloning generator:

  - Hi-Fi mode: passes prompt_wav_path + prompt_text + reference_wav_path.
    The reference's exact transcript anchors generation as a continuation,
    giving tighter timbre and often faster voice settling (which can help the
    short-chunk garbling problem).
  - Control tags are IGNORED in Hi-Fi mode, so this script does NOT apply them.
    Use this run to hear the voice with NO cadence steering at all.

Why a separate script: Hi-Fi and Controllable are different enough that one
flag-laden file gets confusing. Run both, A/B the output.

Usage:
    python 02_generate_hifi.py \
        --plan plan.json \
        --lora /workspace/voxcpm2-lora-pipeline/checkpoints/lora/step_0000999 \
        --reference /workspace/voxcpm_project/references/ref_voice.wav \
        --ref-transcript /workspace/voxcpm_project/references/ref_voice.txt \
        --out-dir /workspace/narration/run01_hifi

Requires: voxcpm, soundfile, ffmpeg.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch
from voxcpm import VoxCPM

try:
    from voxcpm.model.voxcpm2 import LoRAConfig
except ImportError:
    from voxcpm import LoRAConfig

torch.set_float32_matmul_precision("high")

BASE = "openbmb/VoxCPM2"


def load_lora_config(lora_path: Path):
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
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--lora", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path,
                    help="Reference voice clip.")
    ap.add_argument("--ref-transcript", required=True, type=Path,
                    help="EXACT transcript of the reference clip (Hi-Fi needs it).")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cfg", type=float, default=1.6)
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--normalize", action="store_true", default=False)
    ap.add_argument("--start-at", type=int, default=1)
    args = ap.parse_args()

    for p, label in [(args.plan, "plan"), (args.reference, "reference"),
                     (args.ref_transcript, "ref-transcript")]:
        if not p.exists():
            sys.exit(f"{label} not found: {p}")
    if not (args.lora / "lora_config.json").exists():
        sys.exit(f"No lora_config.json in {args.lora}")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    chunks = plan.get("chunks", [])
    if not chunks:
        sys.exit("Plan has no chunks.")

    prompt_text = args.ref_transcript.read_text(encoding="utf-8").strip()
    if not prompt_text:
        sys.exit("Reference transcript is empty.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref16k = convert_reference(args.reference, args.out_dir / "reference_16k.wav")

    model = load_model(args.lora)
    print(f"Model loaded. Hi-Fi mode, {len(chunks)} chunks "
          f"(cfg={args.cfg}, timesteps={args.timesteps}, "
          f"control tags IGNORED in Hi-Fi)\n")

    manifest = {
        "config": plan.get("config", {}),
        "register": plan.get("register"),
        "mode": "hifi",
        "items": [],
    }

    for c in chunks:
        cid = int(c["id"])
        text = c["text"]  # no control tag applied in Hi-Fi
        gap_after_ms = c.get("gap_after_ms", 300)
        wav_name = f"chunk_{cid:04d}.wav"
        wav_path = args.out_dir / wav_name

        manifest["items"].append({
            "id": cid, "file": wav_name, "gap_after_ms": gap_after_ms,
            "control": c.get("control", ""),
        })

        if cid < args.start_at:
            print(f"[{cid:03d}] skipped (resume)")
            continue

        print(f"[{cid:03d}] {text[:70]}{'...' if len(text) > 70 else ''}")

        wav = model.generate(
            text=text,
            prompt_wav_path=str(ref16k),
            prompt_text=prompt_text,
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
    print(f"Next: python 03_stitch.py --run-dir {args.out_dir} "
          f"--output {args.out_dir / 'final.wav'}")


if __name__ == "__main__":
    main()
