#!/usr/bin/env python3
"""
02_generate_nanovllm.py — Generate one audio file per chunk from a reviewed plan.

Drop-in replacement for 02_generate.py that uses nano-vllm-voxcpm instead of
plain voxcpm. Key differences:

  VOICE SEED
    A short neutral Dutch sentence is synthesised once at startup using your
    reference clip as the prompt. The resulting audio is registered server-side
    via add_prompt(); every chunk then references it by prompt_id. The model
    never sees the parenthetical voice-design cue again — chunk 1 starts clean.
    This also solves the "Goeiedag" garbling: the seed absorbs the warm-up
    instability so it never touches the real audio.

  PROSODY CARRY-OVER
    After each chunk the last PROSODY_TAIL_SECONDS of audio are encoded via
    server.encode_latents() and passed as ref_audio_latents on the next call.
    The model sees both the timbre anchor (prompt_id) and the immediately
    preceding intonation contour — so intonation flows across chunk seams
    instead of resetting at each boundary.

  LORA
    Your fine-tuned LoRA checkpoint is loaded once at server init via
    LoRAConfig (read from lora_config.json in the checkpoint dir). The
    checkpoint must contain *.safetensors weight files — if yours are .pt,
    convert first:
      python -c "from safetensors.torch import save_file; import torch; \
        save_file(torch.load('lora_weights.pt'), 'lora_weights.safetensors')"

  MANIFEST
    Output manifest.json is identical to 02_generate.py — 03_stitch.py
    works unchanged.

Usage:
    python 02_generate_nanovllm.py \\
        --plan plan.json \\
        --lora /workspace/voxcpm2-lora-pipeline/checkpoints/lora/step_0000999 \\
        --reference /workspace/voxcpm_project/references/ref_voice.wav \\
        --out-dir /workspace/narration/run01

    # tuning knobs
    python 02_generate_nanovllm.py --plan plan.json --lora ... --reference ... \\
        --out-dir ... --cfg 2.0 --timesteps 30 --prosody-tail 6.0

Requires: nano-vllm-voxcpm, soundfile, torchaudio
    pip install nano-vllm-voxcpm soundfile torchaudio
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
import torch

from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig

torch.set_float32_matmul_precision("high")

BASE_MODEL = "openbmb/VoxCPM2"

# Short neutral Dutch seed — long enough (~3-4 s of speech) to give the
# AudioVAE a stable voice anchor, short enough not to burn context budget.
# This text is registered as prompt_text alongside the seed audio; it is
# never included in the final output.
SEED_TEXT = (
    "Goedemiddag. Dit is een korte inleiding om de stem te kalibreren. "
    "We beginnen zo meteen met het eigenlijke verslag."
)


# ── helpers ────────────────────────────────────────────────────────────────

def load_lora_config(lora_path: Path) -> LoRAConfig:
    cfg_file = lora_path / "lora_config.json"
    if not cfg_file.exists():
        sys.exit(f"No lora_config.json found in {lora_path}")
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    # The training pipeline wraps config under "lora_config" key; handle both.
    cfg = data.get("lora_config", data)
    # Map training checkpoint fields to nano-vllm-voxcpm's LoRAConfig schema.
    # Training uses: r, alpha, dropout
    # Inference uses: max_lora_rank, max_loras (alpha and dropout dropped)
    mapped = {
        "enable_lm":           cfg.get("enable_lm", True),
        "enable_dit":          cfg.get("enable_dit", True),
        "enable_proj":         cfg.get("enable_proj", False),
        "max_lora_rank":       cfg.get("r", 32),
        "max_loras":           1,
        "target_modules_lm":   cfg.get("target_modules_lm",
                                       ["q_proj", "k_proj", "v_proj", "o_proj"]),
        "target_modules_dit":  cfg.get("target_modules_dit",
                                       ["q_proj", "k_proj", "v_proj", "o_proj"]),
        "target_proj_modules": cfg.get("target_proj_modules", []),
    }
    return LoRAConfig(**mapped)


def wav_to_bytes(path: Path, target_sr: int) -> bytes:
    """Load any audio file, resample to target_sr, convert to mono WAV bytes."""
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    buf = io.BytesIO()
    torchaudio.save(buf, wav, target_sr, format="wav")
    return buf.getvalue()


def ndarray_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def trim_silence(
    audio: np.ndarray,
    sr: int,
    thresh_db: float = -40.0,
    keep_ms: int = 40,
    max_trim_ms: int = 800,
) -> np.ndarray:
    """Trim leading/trailing silence, keeping a small natural pad."""
    amp = np.abs(audio)
    if amp.max() <= 0:
        return audio
    thresh = (10 ** (thresh_db / 20.0)) * amp.max()
    above = np.where(amp > thresh)[0]
    if len(above) == 0:
        return audio
    keep = int(sr * keep_ms / 1000)
    max_trim = int(sr * max_trim_ms / 1000)
    start = max(0, min(above[0], max_trim) - keep)
    end = min(len(audio), max(above[-1], len(audio) - max_trim) + keep)
    return audio[start:end]


def collect_chunks(generator) -> np.ndarray:
    parts = []
    for c in generator:
        if c is None:
            continue
        arr = np.asarray(c, dtype=np.float32).reshape(-1)
        if arr.size:
            parts.append(arr)
    if not parts:
        raise RuntimeError("Empty audio returned from generator.")
    return np.concatenate(parts)


def apply_control(text: str, control: str) -> str:
    """Controllable Cloning convention: (instruction)text, no space."""
    control = (control or "").strip()
    return f"({control}){text}" if control else text


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, type=Path,
                    help="plan.json from 01_chunk.py (reviewed and edited).")
    ap.add_argument("--lora", required=True, type=Path,
                    help="LoRA checkpoint dir (must contain lora_config.json "
                         "and *.safetensors weights).")
    ap.add_argument("--reference", required=True, type=Path,
                    help="Reference voice clip (WAV/FLAC/MP3) to clone from.")
    ap.add_argument("--reference-text", default="",
                    help="Transcript of the reference clip (inline). Strongly "
                         "recommended — improves cloning quality and prevents "
                         "the model speaking stray words. Leave empty for "
                         "zero-shot cloning from audio alone.")
    ap.add_argument("--reference-text-file", type=Path, default=None,
                    help="Path to a .txt file containing the reference "
                         "transcript. Takes precedence over --reference-text.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory for chunk wavs + manifest.json.")
    ap.add_argument("--cfg", type=float, default=2.0,
                    help="cfg_value / guidance scale (default 2.0).")
    ap.add_argument("--timesteps", type=int, default=20,
                    help="inference_timesteps (default 30; no speed pressure "
                         "so use 30 for quality).")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature (default 1.0).")
    ap.add_argument("--max-generate-length", type=int, default=2000,
                    help="Maximum generation steps per chunk (default 2000).")
    ap.add_argument("--prosody-tail", type=float, default=6.0,
                    help="Seconds of previous chunk audio to carry forward as "
                         "ref_audio_latents (default 6.0). Reduce to 4.0 if "
                         "you hit max_model_len errors.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="Fraction of VRAM for nano-vllm (default 0.90).")
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="LM context length (default 4096). Increase + VRAM "
                         "if you get context overflow errors.")
    ap.add_argument("--no-control", action="store_true", default=False,
                    help="Ignore per-chunk control instructions entirely.")
    ap.add_argument("--simple-control", default=None,
                    help="Override all per-chunk control instructions with "
                         "one fixed tag, e.g. 'dry, measured'.")
    ap.add_argument("--start-at", type=int, default=1,
                    help="Resume: skip chunks with id < this (1-indexed).")
    args = ap.parse_args()

    # ── validate inputs ────────────────────────────────────────────────────
    if not args.plan.exists():
        sys.exit(f"Plan not found: {args.plan}")
    if not args.reference.exists():
        sys.exit(f"Reference not found: {args.reference}")
    lora_cfg_file = args.lora / "lora_config.json"
    if not lora_cfg_file.exists():
        sys.exit(f"No lora_config.json in {args.lora}")
    safetensors = list(args.lora.glob("*.safetensors"))
    if not safetensors:
        sys.exit(
            f"No *.safetensors files found in {args.lora}.\n"
            "nano-vllm-voxcpm requires safetensors format.\n"
            "Convert your checkpoint:\n"
            "  python -c \"from safetensors.torch import save_file; import torch; "
            "save_file(torch.load('lora_weights.pt'), 'lora_weights.safetensors')\""
        )

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    chunks = plan.get("chunks", [])
    if not chunks:
        sys.exit("Plan has no chunks.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ─────────────────────────────────────────────────────────
    lora_config = load_lora_config(args.lora)
    print(f"LoRA config loaded from {lora_cfg_file.name}")

    print(f"\nLoading {BASE_MODEL} + LoRA ({args.lora.name})...")
    print("(First run will snapshot-download ~9 GB of weights.)\n")

    server = VoxCPM.from_pretrained(
        model=BASE_MODEL,
        inference_timesteps=args.timesteps,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        devices=[0],
        lora_config=lora_config,
    )

    model_info = server.get_model_info()
    sample_rate = int(model_info["sample_rate"])
    print(f"Model ready. Sample rate: {sample_rate} Hz")

    # ── activate the LoRA ──────────────────────────────────────────────────
    # Init-time lora_config only ALLOCATES slots. The adapter weights must be
    # registered by path, and lora_name must be passed on every generate()
    # call to actually apply the adapter. Without this you get the base model.
    LORA_NAME = "voice"
    server.register_lora(LORA_NAME, str(args.lora))
    print(f"LoRA registered and active: '{LORA_NAME}' -> {args.lora}\n")

    # ── reference voice prompt ─────────────────────────────────────────────
    # Clone directly from the reference clip. We register the reference audio
    # + its transcript as a prompt, then every chunk clones from it via
    # prompt_id. No intermediate "seed" synthesis — that produced a clone of a
    # clone, and at low cfg the seed transcript bled into chunk 1.
    print(f"Loading reference clip: {args.reference.name}")
    ref_bytes = wav_to_bytes(args.reference, sample_rate)

    # Resolve the reference transcript: file takes precedence over inline.
    reference_text = args.reference_text
    if args.reference_text_file is not None:
        if not args.reference_text_file.exists():
            sys.exit(f"Reference text file not found: {args.reference_text_file}")
        reference_text = args.reference_text_file.read_text(encoding="utf-8").strip()
        print(f"Reference transcript loaded from {args.reference_text_file.name} "
              f"({len(reference_text)} chars)")

    if reference_text.strip():
        # Best path: register reference audio + transcript as a stored prompt.
        prompt_id = server.add_prompt(ref_bytes, "wav", reference_text)
        ref_latents = None
        print(f"Reference registered with transcript. prompt_id={prompt_id}\n")
    else:
        # Zero-shot fallback: no transcript, so we can't use add_prompt
        # (it requires matching text). Encode latents and pass them per-chunk.
        prompt_id = None
        ref_latents = server.encode_latents(ref_bytes, "wav")
        print("No reference transcript given — zero-shot cloning from audio.\n")

    # ── generate chunks ────────────────────────────────────────────────────
    import time

    n_total = len(chunks)
    n_to_generate = sum(1 for c in chunks if int(c["id"]) >= args.start_at)
    print(f"Generating {n_total} chunks "
          f"(cfg={args.cfg}, timesteps={args.timesteps}, "
          f"temperature={args.temperature}, "
          f"prosody_tail={args.prosody_tail}s)...\n")

    manifest = {
        "config": plan.get("config", {}),
        "register": plan.get("register"),
        "items": [],
    }

    prev_ref_latents: bytes | None = None
    t_start = time.time()
    n_done = 0
    total_audio_s = 0.0

    for c in chunks:
        cid = int(c["id"])
        text = c["text"]
        control = c.get("control", "")

        if args.no_control:
            control = ""
        elif args.simple_control is not None:
            control = args.simple_control

        gap_after_ms = c.get("gap_after_ms", 300)
        wav_name = f"chunk_{cid:04d}.wav"
        wav_path = args.out_dir / wav_name

        # Control tags are NOT applied in this backend: VoxCPM2 via
        # nano-vllm-voxcpm reads the (instruction)text parenthetical aloud
        # instead of interpreting it. Inter-chunk continuity is handled by
        # ref_audio_latents prosody carry-over instead. We still record the
        # tag in the manifest for reference.
        manifest["items"].append({
            "id": cid,
            "file": wav_name,
            "gap_after_ms": gap_after_ms,
            "control": control,
        })

        if cid < args.start_at:
            print(f"[{cid:03d}/{n_total:03d}] skipped (resume)")
            continue

        ref_carry = "yes" if prev_ref_latents else "no"
        print(f"[{cid:03d}/{n_total:03d}] ref_carry={ref_carry} | "
              f"{text[:55]}{'...' if len(text) > 55 else ''}")

        t_chunk = time.time()
        if prompt_id is not None:
            gen = server.generate(
                target_text=text,
                prompt_id=prompt_id,
                ref_audio_latents=prev_ref_latents,
                cfg_value=args.cfg,
                temperature=args.temperature,
                max_generate_length=args.max_generate_length,
                lora_name=LORA_NAME,
            )
        else:
            # Zero-shot: use the reference latents as the prosody/voice ref on
            # every chunk (no stored prompt available without a transcript).
            gen = server.generate(
                target_text=text,
                ref_audio_latents=prev_ref_latents or ref_latents,
                cfg_value=args.cfg,
                temperature=args.temperature,
                max_generate_length=args.max_generate_length,
                lora_name=LORA_NAME,
            )
        wav = collect_chunks(gen)
        wav = trim_silence(wav, sample_rate)
        sf.write(wav_path, wav, sample_rate, subtype="PCM_16")

        # Progress stats.
        chunk_wall = time.time() - t_chunk
        chunk_audio_s = len(wav) / sample_rate
        total_audio_s += chunk_audio_s
        n_done += 1
        elapsed = time.time() - t_start
        avg_s_per_chunk = elapsed / n_done
        remaining = n_to_generate - n_done
        eta_s = avg_s_per_chunk * remaining
        rtf = chunk_wall / chunk_audio_s if chunk_audio_s > 0 else 0.0
        eta_str = (f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
                   if eta_s >= 60 else f"{int(eta_s)}s")
        print(f"         audio={chunk_audio_s:.1f}s wall={chunk_wall:.1f}s "
              f"RTF={rtf:.2f} ETA={eta_str}")

        # Encode the tail of this chunk for prosody carry-over to the next.
        tail_samples = int(args.prosody_tail * sample_rate)
        tail = wav[-tail_samples:] if wav.size > tail_samples else wav
        prev_ref_latents = server.encode_latents(
            ndarray_to_wav_bytes(tail, sample_rate), "wav"
        )

    # ── write manifest ─────────────────────────────────────────────────────
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_wall = time.time() - t_start
    avg_rtf = total_wall / total_audio_s if total_audio_s > 0 else 0.0
    print(f"\nDone. {n_total} chunks | "
          f"{total_audio_s:.1f}s audio | "
          f"wall {total_wall:.1f}s | avg RTF {avg_rtf:.2f}")
    print(f"Manifest: {manifest_path}")
    print(f"Next: python 03_stitch.py --run-dir {args.out_dir} "
          f"--output {args.out_dir / 'final.wav'}")

    server.stop()


if __name__ == "__main__":
    main()