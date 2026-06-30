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

  ASR RETRY LOOP  (new)
    After each chunk is generated, faster-whisper transcribes the audio and
    jiwer computes Word Error Rate (WER) against the input text. If WER exceeds
    --wer-threshold the chunk is regenerated (up to --max-retries times). The
    attempt with the lowest WER is kept. This replicates ElevenLabs' Request
    Stitching quality gate — bad chunks are caught and retried automatically
    instead of surfacing in the final stitch.

    Install deps once:
        pip install faster-whisper jiwer

    Disable entirely with --no-asr. Tune aggressiveness with:
        --wer-threshold 0.20   (default 0.15 — higher = more permissive)
        --max-retries 2        (default 2, matching ElevenLabs behaviour)
        --whisper-model base   (default; use large-v3 for precision QC)

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

    # with ASR quality gate
    python 02_generate_nanovllm.py --plan plan.json --lora ... --reference ... \\
        --out-dir ... --wer-threshold 0.15 --max-retries 2 --whisper-model base

    # disable ASR gate
    python 02_generate_nanovllm.py --plan plan.json --lora ... --reference ... \\
        --out-dir ... --no-asr

    # tuning knobs
    python 02_generate_nanovllm.py --plan plan.json --lora ... --reference ... \\
        --out-dir ... --cfg 2.0 --timesteps 30 --prosody-tail 6.0

Requires: nano-vllm-voxcpm, soundfile, torchaudio, faster-whisper, jiwer
    pip install nano-vllm-voxcpm soundfile torchaudio faster-whisper jiwer
"""
import argparse
import io
import json
import os
import re
import sys
import warnings
from pathlib import Path

# ── silence harmless third-party noise ─────────────────────────────────────
# torch weight_norm deprecation, torchaudio TorchCodec-migration warnings, and
# the nano-vllm "non-writable NumPy array" UserWarning are all cosmetic and do
# not affect output. Suppress them so the generation log stays readable. Set
# VOXCPM_VERBOSE=1 to see them again.
if not os.environ.get("VOXCPM_VERBOSE"):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # torchaudio reads this to stop emitting the StreamReader/Writer deprecations.
    os.environ.setdefault("TORCHAUDIO_NO_DEPRECATION_WARNING", "1")

import numpy as np
import soundfile as sf
import torchaudio
import torch


def _ensure_nanovllm_patched() -> None:
    """
    Re-apply the two nano-vllm-voxcpm source fixes required for single-sequence
    LoRA inference, in case the package was reinstalled and reverted to stock.
    Idempotent and silent when already patched. Must run BEFORE importing
    nanovllm_voxcpm so the corrected kernel source is what gets imported.

    Fix 1 (lora_shrink_op.py): _SMALL_M_THRESHOLD 32 -> 0. The small-m LoRA
           kernel's 1xK tl.dot violates Triton's M>=16 rule at batch < 16;
           disabling that path routes to the regular kernel, which works.
    Fix 2 (model_runner.py): guard self.graphs access with getattr so eager
           mode (enforce_eager=True) doesn't AttributeError before the
           enforce_eager short-circuit.

    See patch_nanovllm.py for the standalone version + backups + --revert.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("nanovllm_voxcpm")
        if spec is None or not spec.submodule_search_locations:
            return  # not installed; the real import below will raise clearly
        root = Path(list(spec.submodule_search_locations)[0])
    except Exception:
        return

    edits = [
        (
            root / "lora_ops" / "triton_ops" / "lora_shrink_op.py",
            "_SMALL_M_THRESHOLD = 32",
            "_SMALL_M_THRESHOLD = 0",
        ),
        (
            root / "engine" / "model_runner.py",
            'has_lora_graph = has_active_lora and bool(self.graphs.get("lora"))',
            'has_lora_graph = has_active_lora and bool(getattr(self, "graphs", {}).get("lora"))',
        ),
    ]

    for path, old, new in edits:
        try:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            if new in text:
                continue  # already patched
            if old in text:
                backup = path.with_suffix(path.suffix + ".orig")
                if not backup.exists():
                    backup.write_text(text, encoding="utf-8")
                path.write_text(text.replace(old, new), encoding="utf-8")
                print(f"[self-patch] applied fix to {path.name}")
        except Exception as e:
            print(f"[self-patch] WARNING: could not patch {path.name}: {e}",
                  file=sys.stderr)


_ensure_nanovllm_patched()

from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig

torch.set_float32_matmul_precision("high")

BASE_MODEL = "openbmb/VoxCPM2"

SEED_TEXT = (
    "Goedemiddag. Dit is een korte inleiding om de stem te kalibreren. "
    "We beginnen zo meteen met het eigenlijke verslag."
)


# ── ASR quality gate ───────────────────────────────────────────────────────

def _load_asr(whisper_model: str):
    """Lazy-load faster-whisper. Returns None if not installed."""
    try:
        from faster_whisper import WhisperModel
        print(f"[asr] Loading faster-whisper '{whisper_model}'...")
        model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
        print(f"[asr] Ready.\n")
        return model
    except ImportError:
        print(
            "[asr] WARNING: faster-whisper not installed. ASR retry disabled.\n"
            "         Install with: pip install faster-whisper jiwer",
            file=sys.stderr,
        )
        return None


def _transcribe(asr_model, audio: np.ndarray, sr: int) -> str:
    """Transcribe audio array to text using faster-whisper."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    segments, _ = asr_model.transcribe(buf, language="nl", beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


def _normalize_for_wer(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace. Done in plain Python so
    we don't depend on jiwer's transform API, which changed incompatibly between
    2.x / 3.x / 4.x (truth_transform -> reference_transform, plus a 3.0 bug where
    the renamed kwarg produced wrong results). We hand jiwer already-clean
    strings and let it just count edits.
    """
    text = text.lower()
    # Drop anything that isn't a letter, digit, or whitespace (Unicode-aware,
    # so Dutch accented chars in rider names survive).
    text = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return " ".join(text.split())


def _word_levenshtein_wer(reference: str, hypothesis: str) -> float:
    """Pure-Python word-level WER fallback (no jiwer)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    m, n = len(ref_words), len(hyp_words)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / m


def _compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate between reference text and ASR hypothesis. Normalization is
    applied in Python first (see _normalize_for_wer), then jiwer just counts
    edits on the clean strings — version-agnostic. Falls back to a pure-Python
    Levenshtein WER if jiwer isn't installed.
    """
    ref = _normalize_for_wer(reference)
    hyp = _normalize_for_wer(hypothesis)
    if not ref.split():
        return 0.0 if not hyp.split() else 1.0
    try:
        from jiwer import wer
        return wer(ref, hyp)
    except ImportError:
        return _word_levenshtein_wer(ref, hyp)


def generate_with_retry(
    server,
    text: str,
    prompt_id,
    ref_latents,
    zero_shot_latents,
    cfg: float,
    temperature: float,
    max_generate_length: int,
    lora_name: str,
    asr_model,
    wer_threshold: float,
    max_retries: int,
    sample_rate: int,
    wer_reference: str | None = None,
) -> tuple[np.ndarray, float, int]:
    """
    Generate audio for one chunk, retrying if WER exceeds threshold.

    `text` is what the model synthesises (may include a (control) parenthetical
    and inline [tags]). `wer_reference`, if given, is the clean spoken text used
    for WER scoring — without the parenthetical or non-verbal tags, since the
    model should not voice those. Falls back to `text` when not provided.

    Returns (best_audio, best_wer, attempts_used).
    best_wer is -1.0 if ASR was skipped.
    """
    wer_target = wer_reference if wer_reference is not None else text
    def _generate_once(ref_audio_latents) -> np.ndarray:
        if prompt_id is not None:
            gen = server.generate(
                target_text=text,
                prompt_id=prompt_id,
                ref_audio_latents=ref_audio_latents,
                cfg_value=cfg,
                temperature=temperature,
                max_generate_length=max_generate_length,
                lora_name=lora_name,
            )
        else:
            gen = server.generate(
                target_text=text,
                ref_audio_latents=ref_audio_latents or zero_shot_latents,
                cfg_value=cfg,
                temperature=temperature,
                max_generate_length=max_generate_length,
                lora_name=lora_name,
            )
        return collect_chunks(gen)

    best_audio = None
    best_wer = float("inf")
    attempts = 0

    for attempt in range(1, max_retries + 2):  # +2: initial attempt + max_retries
        attempts = attempt
        wav = _generate_once(ref_latents)
        wav = trim_silence(wav, sample_rate)

        if asr_model is None:
            # No ASR — accept immediately.
            return wav, -1.0, attempts

        transcript = _transcribe(asr_model, wav, sample_rate)
        current_wer = _compute_wer(wer_target, transcript)

        if best_audio is None or current_wer < best_wer:
            best_audio = wav
            best_wer = current_wer

        wer_pct = f"{current_wer * 100:.1f}%"
        if current_wer <= wer_threshold:
            if attempt > 1:
                print(f"         [asr] attempt {attempt}: WER={wer_pct} ✓ accepted")
            else:
                print(f"         [asr] WER={wer_pct} ✓")
            return best_audio, best_wer, attempts

        # Threshold exceeded.
        if attempt <= max_retries:
            print(f"         [asr] attempt {attempt}: WER={wer_pct} > "
                  f"{wer_threshold * 100:.0f}% — retrying...")
        else:
            print(f"         [asr] attempt {attempt}: WER={wer_pct} — "
                  f"retries exhausted, keeping best ({best_wer * 100:.1f}%)")

    return best_audio, best_wer, attempts


# ── helpers ────────────────────────────────────────────────────────────────

def load_lora_config(lora_path: Path) -> LoRAConfig:
    cfg_file = lora_path / "lora_config.json"
    if not cfg_file.exists():
        sys.exit(f"No lora_config.json found in {lora_path}")
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    cfg = data.get("lora_config", data)
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
    control = (control or "").strip()
    return f"({control}){text}" if control else text


_LEADING_PAREN_RE = re.compile(r"^\s*\([^)]*\)\s*")
_TAG_RE = re.compile(r"\[[^\]]+\]")


def clean_for_wer(text: str) -> str:
    """
    Strip the leading (control) parenthetical and any inline [non-verbal] tags
    so WER is scored against only the words the model should actually speak.
    """
    text = _LEADING_PAREN_RE.sub("", text)
    text = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def concat_latents(*blobs: bytes | None, feat_dim: int) -> bytes | None:
    """
    Concatenate one or more raw-float32 latent blobs (as returned by
    server.encode_latents) into a single blob, in order. None blobs are
    skipped. Used for regrounding: original reference latents + previous-chunk
    tail latents share the one ref_audio_latents slot, so the model sees the
    true voice anchor AND the prosody carry-over on the same chunk.

    Each blob is float32 of shape (frames * feat_dim,) where frames is a
    multiple of patch_size; vertical concatenation preserves that invariant.
    """
    parts = []
    for b in blobs:
        if b is None:
            continue
        arr = np.frombuffer(b, dtype=np.float32).reshape(-1, feat_dim)
        if arr.shape[0]:
            parts.append(arr)
    if not parts:
        return None
    return np.concatenate(parts, axis=0).astype(np.float32).tobytes()


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--lora", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--reference-text", default="")
    ap.add_argument("--reference-text-file", type=Path, default=None)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cfg", type=float, default=1.6,
                    help="Guidance scale (default 1.6 — more stable for long-form "
                         "narration; raise to 2.0–2.5 for stricter text adherence "
                         "at the cost of potential buzzing on difficult inputs).")
    ap.add_argument("--timesteps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-generate-length", type=int, default=2000)
    ap.add_argument("--prosody-tail", type=float, default=6.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-control", action="store_true", default=False)
    ap.add_argument("--simple-control", default=None)
    ap.add_argument("--start-at", type=int, default=1)
    ap.add_argument("--controllable", action="store_true", default=False,
                    help="Use Controllable Cloning instead of Hi-Fi. Drops the "
                         "reference transcript (timbre via encoded latents only) "
                         "so the per-chunk (control instruction) parenthetical is "
                         "honoured by the model. Trades a little voice fidelity "
                         "for active style/intonation control. Hi-Fi (default) "
                         "ignores control instructions entirely.")
    ap.add_argument("--reground", default="every",
                    help="Controllable mode only. How often to re-anchor the "
                         "ORIGINAL reference voice into the ref_audio_latents slot "
                         "to stop timbre drift. 'every' (default) = every chunk "
                         "sees [original reference + previous-chunk tail], the most "
                         "stable option. An integer N = hard reground to the pure "
                         "original reference every N chunks, plain carry-over in "
                         "between. '0' or 'off' = never reground (pure carry-over; "
                         "the old drifting behaviour).")
    ap.add_argument("--reground-anchor-frames", type=int, default=200,
                    help="Cap on the original-reference anchor length in latent "
                         "frames when regrounding (default 200, ~a few seconds). "
                         "Protects max_model_len when anchor + tail + a long chunk "
                         "combine. Set 0 to disable the cap and use the full "
                         "reference.")

    # ASR retry gate
    asr_group = ap.add_argument_group("ASR quality gate (faster-whisper + jiwer)")
    asr_group.add_argument("--no-asr", action="store_true", default=False,
                           help="Disable ASR transcription and WER retry entirely.")
    asr_group.add_argument("--whisper-model", default="base",
                           help="faster-whisper model size: tiny/base/small/medium/"
                                "large-v3 (default: base). Use large-v3 for "
                                "precise QC at the cost of speed.")
    asr_group.add_argument("--wer-threshold", type=float, default=0.15,
                           help="WER above which a chunk is retried (default 0.15 "
                                "= 15%%). Higher = more permissive.")
    asr_group.add_argument("--max-retries", type=int, default=2,
                           help="Max regeneration attempts per chunk before "
                                "keeping the best result (default 2).")

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
            "Convert: python -c \"from safetensors.torch import save_file; "
            "import torch; save_file(torch.load('lora_weights.pt'), "
            "'lora_weights.safetensors')\""
        )

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    chunks = plan.get("chunks", [])
    if not chunks:
        sys.exit("Plan has no chunks.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ASR model ─────────────────────────────────────────────────────
    asr_model = None
    if not args.no_asr:
        asr_model = _load_asr(args.whisper_model)
        if asr_model is not None:
            print(f"[asr] WER threshold={args.wer_threshold * 100:.0f}%  "
                  f"max-retries={args.max_retries}\n")

    # ── load TTS model ─────────────────────────────────────────────────────
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

    LORA_NAME = "voice"
    server.register_lora(LORA_NAME, str(args.lora))
    print(f"LoRA registered and active: '{LORA_NAME}' -> {args.lora}\n")

    # ── reference voice prompt ─────────────────────────────────────────────
    print(f"Loading reference clip: {args.reference.name}")
    ref_bytes = wav_to_bytes(args.reference, sample_rate)

    reference_text = args.reference_text
    if args.reference_text_file is not None:
        if not args.reference_text_file.exists():
            sys.exit(f"Reference text file not found: {args.reference_text_file}")
        reference_text = args.reference_text_file.read_text(encoding="utf-8").strip()
        print(f"Reference transcript loaded from {args.reference_text_file.name} "
              f"({len(reference_text)} chars)")

    if args.controllable and reference_text.strip():
        print("--controllable set: ignoring reference transcript so per-chunk "
              "control instructions stay active (Controllable Cloning mode).")
        reference_text = ""

    if reference_text.strip():
        prompt_id = server.add_prompt(ref_bytes, "wav", reference_text)
        zero_shot_latents = None
        print(f"Reference registered with transcript. prompt_id={prompt_id} "
              f"(Hi-Fi mode — control instructions ignored)\n")
    else:
        prompt_id = None
        zero_shot_latents = server.encode_latents(ref_bytes, "wav")
        mode = "Controllable Cloning" if args.controllable else "zero-shot"
        print(f"Timbre via encoded latents ({mode} — control instructions "
              f"active).\n")

    # ── regrounding setup (Controllable mode only) ─────────────────────────
    # The original reference latents anchor timbre; in Controllable mode they
    # share the single ref_audio_latents slot with the prosody carry-over.
    # Without regrounding the slot holds only the previous chunk's tail, so the
    # voice clones a clone and drifts. We re-inject the original reference here.
    ref_anchor_latents = zero_shot_latents  # original reference (bytes) or None
    # feat_dim is in the model_info dict we already fetched at startup. Fall back
    # to attribute paths, then to the model default (64) if all else fails.
    feat_dim = None
    try:
        feat_dim = int(model_info["feat_dim"])
    except Exception:
        for getter in (
            lambda: int(server.llm.feat_dim),
            lambda: int(server.config.model_config.feat_dim),
        ):
            try:
                feat_dim = getter()
                break
            except Exception:
                continue
    if feat_dim is None and ref_anchor_latents is not None:
        # Last resort: infer from the reference blob length. VoxCPM2 feat_dim
        # is 64; verify the blob divides evenly before trusting it.
        n_floats = len(ref_anchor_latents) // 4  # float32
        if n_floats % 64 == 0:
            feat_dim = 64
    if feat_dim is not None:
        print(f"feat_dim={feat_dim} (for regrounding latent concatenation).")

    # Cap the regrounding anchor so [anchor + tail + long chunk] can't overflow
    # max_model_len. The reference clip is usually short, but a hard cap is
    # cheap insurance. Trim to the FIRST anchor_cap_frames latent frames
    # (a multiple of patch_size).
    if (args.controllable and ref_anchor_latents is not None
            and feat_dim is not None and args.reground_anchor_frames > 0):
        try:
            arr = np.frombuffer(ref_anchor_latents, dtype=np.float32).reshape(-1, feat_dim)
            cap = args.reground_anchor_frames
            if arr.shape[0] > cap:
                ref_anchor_latents = arr[:cap].astype(np.float32).tobytes()
                print(f"Reground anchor trimmed to first {cap} latent frames "
                      f"(was {arr.shape[0]}).")
        except Exception as e:
            print(f"WARNING: could not trim reground anchor: {e}")

    # Parse --reground into a mode: "every" | "off" | int N.
    reground_raw = str(args.reground).strip().lower()
    if reground_raw in ("off", "none", "0"):
        reground_mode, reground_n = "off", 0
    elif reground_raw == "every":
        reground_mode, reground_n = "every", 1
    else:
        try:
            reground_n = int(reground_raw)
            reground_mode = "n" if reground_n > 0 else "off"
        except ValueError:
            sys.exit(f"--reground must be 'every', 'off', or an integer; "
                     f"got {args.reground!r}")

    if args.controllable and ref_anchor_latents is not None:
        if feat_dim is None:
            print("WARNING: could not read feat_dim from server; regrounding "
                  "disabled (falling back to pure carry-over).")
            reground_mode = "off"
        elif reground_mode == "every":
            print("Regrounding: ORIGINAL reference re-anchored on EVERY chunk "
                  "(reference + carry-over tail share the ref slot).\n")
        elif reground_mode == "n":
            print(f"Regrounding: hard reset to ORIGINAL reference every "
                  f"{reground_n} chunks; pure carry-over in between.\n")
        else:
            print("Regrounding: OFF (pure carry-over — timbre may drift).\n")

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
        "mode": ("controllable" if args.controllable else "hifi"),
        "reground": (reground_mode if args.controllable else None),
        "cfg": args.cfg,
        "items": [],
    }

    prev_ref_latents: bytes | None = None
    t_start = time.time()
    n_done = 0
    total_audio_s = 0.0
    total_retries = 0
    wer_log = []  # (chunk_id, wer, attempts)

    for c in chunks:
        cid = int(c["id"])
        text = c["text"]
        control = c.get("control", "")

        if args.no_control:
            control = ""
        elif args.simple_control is not None:
            control = args.simple_control

        # In Controllable mode the parenthetical is honoured by the model, so
        # prepend it to the text: "(dry, measured)De renner...". In Hi-Fi mode
        # the model would just read the parenthetical aloud, so we never inject
        # it there — the control tag is recorded in the manifest only.
        if args.controllable and control.strip():
            target_text = apply_control(text, control)
        else:
            target_text = text

        gap_after_ms = c.get("gap_after_ms", 300)
        wav_name = f"chunk_{cid:04d}.wav"
        wav_path = args.out_dir / wav_name

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
        ctrl_str = f" ctrl='{control}'" if (args.controllable and control.strip()) else ""

        # ── decide what goes in the ref_audio_latents slot ─────────────────
        # Hi-Fi: timbre comes from prompt_id, so the slot is pure prosody
        #   carry-over (previous tail), unchanged from before.
        # Controllable: the slot is the ONLY voice anchor, so we reground the
        #   original reference into it according to --reground.
        reground_tag = ""
        if not args.controllable or ref_anchor_latents is None or reground_mode == "off":
            chunk_ref_latents = prev_ref_latents
        elif reground_mode == "every":
            # Original reference + previous tail, every chunk. True anchor +
            # prosody continuity in one slot.
            chunk_ref_latents = concat_latents(
                ref_anchor_latents, prev_ref_latents, feat_dim=feat_dim
            )
            reground_tag = " reground=ref+tail"
        else:  # mode == "n": hard reset every N chunks
            if (n_done % reground_n) == 0 or prev_ref_latents is None:
                chunk_ref_latents = ref_anchor_latents
                reground_tag = " reground=hard"
            else:
                chunk_ref_latents = prev_ref_latents

        print(f"[{cid:03d}/{n_total:03d}] ref_carry={ref_carry}{ctrl_str}{reground_tag} | "
              f"{text[:55]}{'...' if len(text) > 55 else ''}")

        t_chunk = time.time()

        wav, chunk_wer, attempts = generate_with_retry(
            server=server,
            text=target_text,
            prompt_id=prompt_id,
            ref_latents=chunk_ref_latents,
            zero_shot_latents=zero_shot_latents,
            cfg=args.cfg,
            temperature=args.temperature,
            max_generate_length=args.max_generate_length,
            lora_name=LORA_NAME,
            asr_model=asr_model,
            wer_threshold=args.wer_threshold,
            max_retries=args.max_retries,
            sample_rate=sample_rate,
            wer_reference=clean_for_wer(target_text),
        )

        sf.write(wav_path, wav, sample_rate, subtype="PCM_16")

        # Progress stats.
        chunk_wall = time.time() - t_chunk
        chunk_audio_s = len(wav) / sample_rate
        total_audio_s += chunk_audio_s
        n_done += 1
        retries_this_chunk = attempts - 1
        total_retries += retries_this_chunk
        wer_log.append((cid, chunk_wer, attempts))

        elapsed = time.time() - t_start
        avg_s_per_chunk = elapsed / n_done
        remaining = n_to_generate - n_done
        eta_s = avg_s_per_chunk * remaining
        rtf = chunk_wall / chunk_audio_s if chunk_audio_s > 0 else 0.0
        eta_str = (f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
                   if eta_s >= 60 else f"{int(eta_s)}s")

        wer_str = (f" WER={chunk_wer * 100:.1f}%"
                   if chunk_wer >= 0 else "")
        retry_str = (f" retries={retries_this_chunk}"
                     if retries_this_chunk > 0 else "")
        print(f"         audio={chunk_audio_s:.1f}s wall={chunk_wall:.1f}s "
              f"RTF={rtf:.2f}{wer_str}{retry_str} ETA={eta_str}")

        # Encode tail for prosody carry-over.
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

    if asr_model is not None and wer_log:
        valid_wers = [(cid, w, a) for cid, w, a in wer_log if w >= 0]
        if valid_wers:
            avg_wer = sum(w for _, w, _ in valid_wers) / len(valid_wers)
            worst = max(valid_wers, key=lambda x: x[1])
            print(f"ASR summary: avg WER={avg_wer * 100:.1f}% | "
                  f"total retries={total_retries} | "
                  f"worst chunk={worst[0]} ({worst[1] * 100:.1f}% WER, "
                  f"{worst[2]} attempts)")

            wer_log_path = args.out_dir / "wer_log.json"
            wer_log_data = {
                "avg_wer": round(avg_wer, 4),
                "total_retries": total_retries,
                "threshold": args.wer_threshold,
                "whisper_model": args.whisper_model,
                "chunks": [
                    {"id": cid, "wer": round(w, 4), "attempts": a}
                    for cid, w, a in valid_wers
                ],
            }
            wer_log_path.write_text(
                json.dumps(wer_log_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"WER log:  {wer_log_path}")

    print(f"Manifest: {manifest_path}")
    print(f"Next: python 03_stitch.py --run-dir {args.out_dir} "
          f"--output {args.out_dir / 'final.wav'}")

    server.stop()


if __name__ == "__main__":
    main()