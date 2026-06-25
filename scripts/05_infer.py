#!/usr/bin/env python3
"""
05_infer.py — Generate with a fine-tuned LoRA, and A/B multiple checkpoints.

Two uses:
  1. Single checkpoint, single text -> one wav.
  2. --compare: run the SAME text through every checkpoint in a directory, so
     you can pick the best by ear (the docs warn val-loss doesn't track quality).

After you pick a winner, point your main long-form script at it by adding
`lora_weights_path=...` to its VoxCPM.from_pretrained(...) call, then run Mode 1
(the trained voice supplies timbre; the control tag still steers cadence).

Usage:
    # one checkpoint
    python 05_infer.py --lora step_0000500 \
        --text "Goeiedag. Kent u de Col de la Croix?" --output test.wav

    # compare all checkpoints under a dir on one line of text
    python 05_infer.py --compare checkpoints/lora/ \
        --text "Goeiedag. Kent u de Col de la Croix?" --out_dir ab_test/

    # with a style/pace control tag (works because this is control-capable)
    python 05_infer.py --lora checkpoints/lora/latest \
        --text "(rustig tempo, duidelijke pauzes)Goeiedag." --output test.wav
"""
import argparse
from pathlib import Path

import soundfile as sf
from voxcpm import VoxCPM

BASE = "openbmb/VoxCPM2"


def find_checkpoints(root: Path) -> list[Path]:
    """Return checkpoint dirs containing lora_weights.* under root."""
    found = []
    for p in sorted(root.rglob("lora_config.json")):
        found.append(p.parent)
    return found


def load_model(lora_path: Path) -> VoxCPM:
    print(f"Loading base + LoRA: {lora_path}")
    return VoxCPM.from_pretrained(BASE, lora_weights_path=str(lora_path),
                                  load_denoiser=False)


def generate(model: VoxCPM, text: str, out: Path,
             cfg_value: float, timesteps: int, normalize: bool) -> None:
    wav = model.generate(
        text=text,
        cfg_value=cfg_value,
        inference_timesteps=timesteps,
        normalize=normalize,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wav, model.tts_model.sample_rate)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", required=True)
    ap.add_argument("--lora", type=Path,
                    help="Single LoRA checkpoint dir.")
    ap.add_argument("--compare", type=Path,
                    help="Root dir; runs text through every checkpoint found.")
    ap.add_argument("--output", type=Path, default=Path("lora_out.wav"),
                    help="Output wav (single-checkpoint mode).")
    ap.add_argument("--out_dir", type=Path, default=Path("ab_test"),
                    help="Output dir (--compare mode).")
    ap.add_argument("--cfg_value", type=float, default=1.6)
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no_normalize", dest="normalize", action="store_false")
    args = ap.parse_args()

    if not args.lora and not args.compare:
        raise SystemExit("Pass either --lora <ckpt> or --compare <dir>.")

    if args.compare:
        ckpts = find_checkpoints(args.compare)
        if not ckpts:
            raise SystemExit(f"No LoRA checkpoints under {args.compare}")
        print(f"Comparing {len(ckpts)} checkpoints on identical text.\n")
        for ckpt in ckpts:
            model = load_model(ckpt)
            tag = ckpt.name
            out = args.out_dir / f"{tag}.wav"
            generate(model, args.text, out,
                     args.cfg_value, args.timesteps, args.normalize)
            del model
        print(f"\nDone. Listen through {args.out_dir}/ and pick the best by ear.")
        print("Earlier checkpoints that already sound right are preferable — "
              "later ones risk overfitting (voice stops following the text).")
    else:
        model = load_model(args.lora)
        generate(model, args.text, args.output,
                 args.cfg_value, args.timesteps, args.normalize)


if __name__ == "__main__":
    main()
