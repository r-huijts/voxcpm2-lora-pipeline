#!/usr/bin/env python3
"""
02_generate_nanovllm.py — Generate one audio file per chunk from a reviewed plan.

Uses nano-vllm-voxcpm (a quantized serving backend) to generate audio for each
chunk of a plan.json produced by 01_chunk.py. Highlights:

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
    Output manifest.json is the format 03_stitch.py expects.

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

    # generate 3 candidate takes of chunk 7 (chunk_0007_v1..v3.wav), plain
    # file untouched -- listen and record your pick in selection.json, no
    # need for --interactive
    python 02_generate_nanovllm.py --plan plan.json --lora ... --reference ... \\
        --out-dir ... --only-chunks 7 --candidates 3

Requires: nano-vllm-voxcpm, soundfile, torchaudio, faster-whisper, jiwer
    pip install nano-vllm-voxcpm soundfile torchaudio faster-whisper jiwer
"""
import argparse
import io
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

from _pipeline_config import default_voice_config_path, load_voice_config, apply_config_defaults
from _plan_schema import resolve_source_text, resolve_spoken_text
from _carryover import should_carry_over
from _duration_gate import (
    BUILTIN_DEFAULT_SEC_PER_WORD,
    check_dragging,
    check_duration,
    resolve_duration_target,
    select_best_attempt,
)
from _streaming import collect_chunks
from _audio_metrics import compute_metrics, derive_flags, rank_candidates
from _console import (
    INDENT, INDENT2, candidate_table, candidate_take_line, chunk_header,
    command_table, console, dim, duration_note, error, info,
    metrics_advisory_line, metrics_warning, plain, quality_line, rule,
    success, warn, warn_err,
)
from _journal import append_chunk_record, journal_path, read_chunk_records, reset_journal

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
    """Lazy-load faster-whisper. Hard-fails if missing: silently running
    without the WER gate halves the quality checks without the user noticing.
    Pass --no-asr to run without it deliberately."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit(
            "[asr] faster-whisper is not installed, so the ASR/WER quality "
            "gate cannot run.\n"
            "      Install it:   pip install faster-whisper jiwer\n"
            "      Or pass --no-asr to deliberately generate without the gate."
        )
    info(f"[asr] Loading faster-whisper '{whisper_model}'...")
    model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
    success("[asr] Ready.\n")
    return model


def _transcribe(asr_model, audio: np.ndarray, sr: int) -> str:
    """Transcribe audio array to text using faster-whisper."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    segments, _ = asr_model.transcribe(buf, language="nl", beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


def resolve_control(c: dict, args) -> str:
    """The chunk's effective control tag after --no-control/--simple-control."""
    control = c.get("control", "")
    if args.no_control:
        return ""
    if args.simple_control is not None:
        return args.simple_control
    return control


def wer_log_entry(cid: int, wer: float, attempts: int, sec_per_word: float,
                  duration_ok: bool, duration_reason: str,
                  audio_metrics: dict | None = None,
                  audio_flags: list | None = None) -> dict:
    """One chunk's wer_log.json entry -- also what goes in the journal."""
    entry = {
        "id": cid,
        "wer": round(wer, 4) if wer >= 0 else None,
        "attempts": attempts,
        "sec_per_word": round(sec_per_word, 4) if sec_per_word else None,
        "duration_ok": duration_ok,
        "duration_reason": duration_reason,
    }
    if audio_metrics is not None:
        entry["audio_metrics"] = audio_metrics
        entry["audio_flags"] = audio_flags or []
    return entry


def analyze_chunk_audio(wav, sr: int) -> tuple[dict | None, list[str]]:
    """
    Advisory waveform metrics + flags for an accepted take (see
    _audio_metrics.py). Never a retry trigger, and a failure here must not
    kill the run -- the chunk itself is already accepted.
    """
    try:
        metrics = compute_metrics(wav, sr)
        return metrics, derive_flags(metrics)
    except Exception as e:
        metrics_warning(f"WARNING: waveform analysis failed: {e}")
        return None, []


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
    duration_gate: bool = True,
    target_sec_per_word: float = BUILTIN_DEFAULT_SEC_PER_WORD,
    duration_floor: float = 0.5,
    runaway_ratio: float = 2.0,
    dragging_ratio: float | None = None,
) -> tuple:
    """
    Generate audio for one chunk, retrying if WER exceeds threshold OR the
    duration-ratio gate fails (see _duration_gate.py: too short/rushed, or a
    "runaway" generation that never cleanly stopped -- VoxCPM's documented
    failure mode that otherwise fills VRAM silently). The duration gate does
    NOT require ASR and still runs when asr_model is None / --no-asr is set.

    `text` is what the model synthesises (may include a (control) parenthetical
    and inline [tags]). `wer_reference`, if given, is the clean spoken text used
    for BOTH WER scoring and the duration gate's word count — without the
    parenthetical or non-verbal tags, since the model should not voice those.
    Falls back to `text` when not provided.

    Best-attempt selection when no single attempt cleanly passes both gates:
    prefer attempts passing BOTH (lowest WER among those, ties broken by
    closeness to the expected duration); else the lowest-WER attempt that at
    least passes duration; else the lowest-WER attempt overall.

    Returns (best_audio, best_wer, attempts_used, accepted_transcript,
    best_sec_per_word, best_duration_ok, best_duration_reason).
    best_wer is -1.0 if ASR was skipped; accepted_transcript is "" then too.
    best_duration_reason is "ok" / "too_short" / "runaway", or "skipped" when
    duration_gate is False.
    """
    wer_target = wer_reference if wer_reference is not None else text
    words = len(wer_target.split())
    # Known before generation starts (target_sec_per_word is resolved by the
    # caller), so the live progress ticker can show "X.Xs / ~Y.Ys expected"
    # instead of just a bare running total.
    expected_seconds = words * target_sec_per_word if words else None

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
        return collect_chunks(
            gen,
            sample_rate=sample_rate,
            progress_label=f"attempt {attempt}",
            expected_seconds=expected_seconds,
        )

    attempts_data = []
    attempts = 0

    for attempt in range(1, max_retries + 2):  # +2: initial attempt + max_retries
        attempts = attempt
        wav = _generate_once(ref_latents)
        wav = trim_silence(wav, sample_rate)
        # The duration gate must judge the TRUE spoken length, not the
        # safety-capped trim above (max_trim_ms=800 by design, so it never
        # eats real speech) -- for a genuine runaway with >800ms of trailing
        # junk, that cap leaves residual silence in `wav`, which previously
        # made the gate's own measurement read longer than what 03_stitch.py
        # (whose trim is uncapped) keeps, causing false "runaway" flags and
        # wasted retries on chunks that were already fine. Re-trim (uncapped)
        # a COPY purely for measurement; `wav` itself -- what's transcribed,
        # kept, and written to disk -- is unchanged.
        audio_seconds = len(trim_silence(wav, sample_rate, max_trim_ms=60_000)) / sample_rate

        if asr_model is not None:
            transcript = _transcribe(asr_model, wav, sample_rate)
            current_wer = _compute_wer(wer_target, transcript)
            wer_ok = current_wer <= wer_threshold
        else:
            transcript = ""
            current_wer = -1.0
            wer_ok = True

        if duration_gate:
            dur_ok, dur_reason, sec_per_word, expected_seconds = check_duration(
                audio_seconds, words, target_sec_per_word, duration_floor, runaway_ratio,
            )
        else:
            dur_ok, dur_reason = True, "skipped"
            sec_per_word = audio_seconds / words if words else 0.0
            expected_seconds = words * target_sec_per_word if words else 0.0

        if dur_ok and check_dragging(audio_seconds, expected_seconds, dragging_ratio):
            duration_note(f"NOTE: dragging ({sec_per_word:.2f}s/word, "
                          f"{audio_seconds:.1f}s vs ~{expected_seconds:.1f}s expected) — informational only.")

        attempts_data.append(dict(
            audio=wav, wer=current_wer, transcript=transcript,
            wer_ok=wer_ok, dur_ok=dur_ok, dur_reason=dur_reason,
            sec_per_word=sec_per_word, expected_seconds=expected_seconds,
        ))

        status_bits = []
        if asr_model is not None:
            status_bits.append(f"WER={current_wer * 100:.1f}%")
        if duration_gate and words > 0:
            status_bits.append(f"{sec_per_word:.2f}s/word (~{expected_seconds:.1f}s expected)")
        status = " ".join(status_bits)

        if wer_ok and dur_ok:
            quality_line(attempt, status, passed=True, retrying=False)
            break

        reasons = []
        if not wer_ok:
            reasons.append(f"WER>{wer_threshold * 100:.0f}%")
        if not dur_ok:
            reasons.append("RUNAWAY (hit max length, no clean stop)"
                            if dur_reason == "runaway" else "too short/rushed")
        quality_line(attempt, status, passed=False, retrying=(attempt <= max_retries),
                    reasons=", ".join(reasons))

    best = select_best_attempt(attempts_data, target_sec_per_word)
    if not (best["wer_ok"] and best["dur_ok"]) and any(
        a["dur_reason"] == "runaway" for a in attempts_data
    ):
        duration_note(f"RUNAWAY (hit max length, no clean stop) on all "
                      f"attempts — keeping least-bad ({best['sec_per_word']:.2f}s/word).",
                      severe=True)

    return (best["audio"], best["wer"], attempts, best["transcript"],
            best["sec_per_word"], best["dur_ok"], best["dur_reason"])


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


def _diff_tokens(text: str) -> list[str]:
    """
    Tokenize for diffing: keep original case (so proper nouns are detectable),
    strip surrounding punctuation but keep internal accents/hyphens.
    """
    text = _LEADING_PAREN_RE.sub("", text)
    text = _TAG_RE.sub(" ", text)
    toks = []
    for raw in text.split():
        # strip leading/trailing punctuation, keep inner chars (Évenepoel, Van-der)
        t = raw.strip(".,;:!?\"'()[]…—–")
        if t:
            toks.append(t)
    return toks


def pronunciation_diff(reference_text: str, transcript: str) -> list[dict]:
    """
    Word-align the chunk's intended text against the ASR transcript of the
    ACCEPTED audio, and return the substitutions — i.e. words the model was
    asked to say that Whisper heard as something else. These are the candidate
    mispronunciations.

    Comparison is case-insensitive for matching (so capitalization alone isn't
    flagged) but the ORIGINAL-cased reference word is reported, so proper nouns
    stay recognizable and can be flagged. Insertions and deletions are ignored —
    only true substitutions (said X, heard Y) are pronunciation signal.

    Returns a list of {ref, heard, is_proper} dicts, in order of appearance.
    """
    ref_orig = _diff_tokens(reference_text)
    hyp_orig = _diff_tokens(transcript)
    ref_lc = [w.lower() for w in ref_orig]
    hyp_lc = [w.lower() for w in hyp_orig]

    # Standard Levenshtein alignment with backtrace over words.
    m, n = len(ref_lc), len(hyp_lc)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_lc[i - 1] == hyp_lc[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)

    subs = []
    i, j = m, n
    while i > 0 and j > 0:
        cost = 0 if ref_lc[i - 1] == hyp_lc[j - 1] else 1
        if dp[i][j] == dp[i - 1][j - 1] + cost:
            if cost == 1:  # substitution
                ref_w = ref_orig[i - 1]
                subs.append({
                    "ref": ref_w,
                    "heard": hyp_orig[j - 1],
                    "is_proper": ref_w[:1].isupper(),
                })
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1  # deletion (ignored)
        else:
            j -= 1  # insertion (ignored)
    subs.reverse()
    return subs


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
    lora_action = ap.add_argument("--lora", required=True, type=Path)
    reference_action = ap.add_argument("--reference", required=True, type=Path)
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
    ap.add_argument("--only-chunks", default=None,
                    help="Regenerate ONLY these chunk IDs, leaving all other "
                         "existing chunk wavs untouched. Comma-separated, e.g. "
                         "'7' or '4,7'. Requires a previous run's wavs in "
                         "--out-dir. The manifest is preserved; only the named "
                         "chunks' audio is replaced. Use this to fix a few bad "
                         "chunks without re-running the whole column.")
    ap.add_argument("--candidates", type=int, default=None,
                    help="Requires --only-chunks. Instead of overwriting the plain "
                         "chunk_NNNN.wav for each named chunk, generate this many "
                         "candidate takes as chunk_NNNN_v1.wav..vK.wav (the plain "
                         "file is left untouched). Listen and record your pick in "
                         "selection.json, e.g. {\"7\": 2}, then re-run 03_stitch.py. "
                         "Same idea as --interactive's 'cand' command, without "
                         "needing to go interactive.")
    ap.add_argument("--controllable", action=argparse.BooleanOptionalAction, default=False,
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

    # Duration-ratio gate: catches rushed/truncated chunks (WER can't see
    # delivery) and "runaway" generations that never cleanly stop (a
    # documented VoxCPM failure mode that otherwise fills VRAM silently).
    # Independent of ASR -- runs even with --no-asr.
    dur_group = ap.add_argument_group("Duration-ratio gate")
    dur_group.add_argument("--no-duration-gate", dest="duration_gate",
                           action="store_false", default=True,
                           help="Disable the duration-ratio gate entirely "
                                "(restores prior behavior exactly).")
    dur_group.add_argument("--sec-per-word-target", type=float, default=None,
                           help="Seconds-per-word baseline to judge chunk duration "
                                "against. Default: use a conservative built-in value "
                                "until --baseline-min-chunks chunks are accepted, "
                                "then the running median of accepted chunks.")
    dur_group.add_argument("--baseline-min-chunks", type=int, default=5,
                           help="Accepted chunks needed before trusting the running "
                                "median over the built-in default (default 5). Early "
                                "chunks may themselves be flawed.")
    dur_group.add_argument("--duration-floor", type=float, default=0.5,
                           help="A chunk is rejected as too short/rushed when its "
                                "audio is under this fraction of "
                                "(words * target_sec_per_word) (default 0.5).")
    dur_group.add_argument("--runaway-ratio", type=float, default=2.0,
                           help="A chunk is flagged RUNAWAY when its audio exceeds "
                                "this multiple of the expected duration (default "
                                "2.0) -- catches a generation that never cleanly "
                                "stopped, judged by ratio-to-expected, never by "
                                "absolute length.")
    dur_group.add_argument("--duration-ceiling", type=float, default=None,
                           help="Optional informational-only warning (no retry) when "
                                "a chunk exceeds this multiple of the expected "
                                "duration but isn't bad enough to count as RUNAWAY. "
                                "Off by default.")

    ap.add_argument("--interactive", action="store_true", default=False,
                    help="After the run, keep the model loaded and drop into a "
                         "prompt for regenerating individual chunks fast (no "
                         "reload). Commands: '<id>' regenerates a chunk, "
                         "'<id> --cfg 1.7 --temp 0.9' overrides settings, "
                         "'reload' re-reads the plan (pick up lexicon/plan edits), "
                         "'list' shows chunks, 'quit' exits. Pair with "
                         "--only-chunks to skip the initial full run.")

    CONFIGURABLE = {"lora", "reference", "reference_text_file", "cfg", "timesteps",
                     "temperature", "max_generate_length", "prosody_tail",
                     "gpu_memory_utilization", "max_model_len", "controllable",
                     "reground", "reground_anchor_frames", "whisper_model",
                     "wer_threshold", "max_retries", "sec_per_word_target",
                     "baseline_min_chunks", "duration_floor", "runaway_ratio",
                     "duration_ceiling"}
    default_config_path = default_voice_config_path(__file__)
    ap.add_argument("--config", type=Path, default=default_config_path,
                    help="Shared per-voice defaults JSON (see scripts/_pipeline_config.py "
                         "and scripts/voice.example.json). Looked up in the current "
                         "directory first, then next to this script -- so it's found "
                         "whether you run from a project dir or from scripts/. Lets "
                         "--lora/--reference/tuning flags be set once per project "
                         "instead of retyped every run. CLI flags always override it "
                         "(pass --no-controllable to force Hi-Fi for one run even if "
                         "voice.json sets controllable: true).")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=default_config_path)
    config = load_voice_config(pre.parse_known_args()[0].config)
    applied = apply_config_defaults(ap, config, CONFIGURABLE)
    if "lora" in applied:
        lora_action.required = False
    if "reference" in applied:
        reference_action.required = False

    args = ap.parse_args()

    if not args.controllable and args.reground != "every":
        warn_err(f"WARNING: --reground {args.reground!r} is ignored in Hi-Fi mode "
                f"(regrounding only applies with --controllable).")
    if args.reference_text.strip() and args.reference_text_file is not None:
        warn_err(f"NOTE: --reference-text-file ({args.reference_text_file}) overrides "
                f"--reference-text.")

    # Parse --only-chunks into a set of ints (or None for "all").
    only_chunks = None
    if args.only_chunks is not None:
        try:
            only_chunks = {int(x) for x in args.only_chunks.split(",") if x.strip()}
        except ValueError:
            sys.exit(f"--only-chunks must be comma-separated integers; "
                     f"got {args.only_chunks!r}")
        if not only_chunks:
            sys.exit("--only-chunks was empty.")

    if args.candidates is not None:
        if only_chunks is None:
            sys.exit("--candidates requires --only-chunks (which chunk(s) to generate "
                      "candidates for).")
        if args.candidates < 1:
            sys.exit(f"--candidates must be >= 1; got {args.candidates}.")

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
            info(f"[asr] WER threshold={args.wer_threshold * 100:.0f}%  "
                f"max-retries={args.max_retries}\n")

    # ── load TTS model ─────────────────────────────────────────────────────
    lora_config = load_lora_config(args.lora)
    info(f"LoRA config loaded from {lora_cfg_file.name}")
    rule(f"Loading {BASE_MODEL} + LoRA ({args.lora.name})")
    dim("(First run will snapshot-download ~9 GB of weights.)\n")

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
    success(f"Model ready. Sample rate: {sample_rate} Hz")

    LORA_NAME = "voice"
    server.register_lora(LORA_NAME, str(args.lora))
    success(f"LoRA registered and active: '{LORA_NAME}' -> {args.lora}\n")

    # ── reference voice prompt ─────────────────────────────────────────────
    info(f"Loading reference clip: {args.reference.name}")
    ref_bytes = wav_to_bytes(args.reference, sample_rate)

    reference_text = args.reference_text
    if args.reference_text_file is not None:
        if not args.reference_text_file.exists():
            sys.exit(f"Reference text file not found: {args.reference_text_file}")
        reference_text = args.reference_text_file.read_text(encoding="utf-8").strip()
        info(f"Reference transcript loaded from {args.reference_text_file.name} "
            f"({len(reference_text)} chars)")

    if args.controllable and reference_text.strip():
        dim("--controllable set: ignoring reference transcript so per-chunk "
            "control instructions stay active (Controllable Cloning mode).")
        reference_text = ""

    if reference_text.strip():
        prompt_id = server.add_prompt(ref_bytes, "wav", reference_text)
        zero_shot_latents = None
        success(f"Reference registered with transcript. prompt_id={prompt_id} "
               f"(Hi-Fi mode — control instructions ignored)\n")
    else:
        prompt_id = None
        zero_shot_latents = server.encode_latents(ref_bytes, "wav")
        mode = "Controllable Cloning" if args.controllable else "zero-shot"
        success(f"Timbre via encoded latents ({mode} — control instructions "
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
        dim(f"feat_dim={feat_dim} (for regrounding latent concatenation).")

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
                dim(f"Reground anchor trimmed to first {cap} latent frames "
                    f"(was {arr.shape[0]}).")
        except Exception as e:
            warn(f"WARNING: could not trim reground anchor: {e}")

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
            warn("WARNING: could not read feat_dim from server; regrounding "
                "disabled (falling back to pure carry-over).")
            reground_mode = "off"
        elif reground_mode == "every":
            info("Regrounding: ORIGINAL reference re-anchored on EVERY chunk "
                "(reference + carry-over tail share the ref slot).\n")
        elif reground_mode == "n":
            info(f"Regrounding: hard reset to ORIGINAL reference every "
                f"{reground_n} chunks; pure carry-over in between.\n")
        else:
            warn("Regrounding: OFF (pure carry-over — timbre may drift).\n")

    # ── generate chunks ────────────────────────────────────────────────────
    n_total = len(chunks)
    n_to_generate = sum(1 for c in chunks if int(c["id"]) >= args.start_at)
    if only_chunks is not None:
        warn(f"REGENERATING ONLY chunks {sorted(only_chunks)} — all other "
            f"existing wavs in {args.out_dir} are kept untouched.")
        n_to_generate = len(only_chunks)
    rule(f"Generating {n_total} chunks")
    info(f"cfg={args.cfg}, timesteps={args.timesteps}, "
        f"temperature={args.temperature}, prosody_tail={args.prosody_tail}s\n")

    manifest = {
        "config": plan.get("config", {}),
        "register": plan.get("register"),
        "mode": ("controllable" if args.controllable else "hifi"),
        "reground": (reground_mode if args.controllable else None),
        "cfg": args.cfg,
        "items": [],
    }
    # Every manifest field derives from the plan + flags, not from generation
    # results -- so build ALL items and write manifest.json up front. A run
    # that crashes partway then still leaves a complete manifest next to
    # whatever wavs it managed to produce, instead of no manifest at all.
    # generation_complete stays False until the loop finishes, so the stitcher
    # can warn that some listed wavs may be missing or stale (e.g. left over
    # from an earlier run into the same out-dir).
    manifest["generation_complete"] = False
    for c in chunks:
        cid = int(c["id"])
        manifest["items"].append({
            "id": cid,
            "file": f"chunk_{cid:04d}.wav",
            "gap_after_ms": c.get("gap_after_ms", 300),
            "control": resolve_control(c, args),
            "source_text": resolve_source_text(c),
            "spoken_text": resolve_spoken_text(c),
        })
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Per-chunk journal (journal.jsonl): appended + flushed as each chunk is
    # accepted, so a crash mid-run loses no bookkeeping -- a resumed run
    # (--start-at / --only-chunks) rebuilds finished chunks' wer_log entries
    # and the duration baseline from it. A fresh full run supersedes whatever
    # a previous run journaled into this out-dir.
    jpath = journal_path(args.out_dir)
    fresh_full_run = only_chunks is None and args.start_at <= 1
    prior_journal = {} if fresh_full_run else read_chunk_records(jpath)
    if fresh_full_run:
        reset_journal(jpath)
    elif prior_journal:
        info(f"Journal: {len(prior_journal)} completed chunk(s) recorded by "
            f"prior run(s) ({jpath.name}).")

    prev_ref_latents: bytes | None = None
    t_start = time.time()
    n_done = 0
    total_audio_s = 0.0
    total_retries = 0
    wer_log = []  # list of wer_log_entry() dicts, one per generated chunk
    pron_diffs = []  # [{id, ref, heard, is_proper}, ...] across accepted chunks

    # Duration-gate baseline: seed from prior state in this out-dir (e.g. an
    # --only-chunks rerun, or a crash resume), so a partial run still benefits
    # from the established pace instead of restarting at the conservative
    # built-in default. Per chunk id the journal outranks wer_log.json: the
    # journal always reflects the most recent run to touch this out-dir (a
    # fresh full run resets it), while wer_log.json can survive from a
    # superseded earlier run.
    wer_log_path = args.out_dir / "wer_log.json"
    seed_entries: dict[int, dict] = {}
    if wer_log_path.exists():
        try:
            prior_log = json.loads(wer_log_path.read_text(encoding="utf-8"))
            for e in prior_log.get("chunks", []):
                seed_entries[int(e["id"])] = e
        except Exception as e:
            warn_err(f"WARNING: could not seed duration baseline from "
                    f"{wer_log_path.name}: {e}")
    for jcid, rec in prior_journal.items():
        if rec.get("wer"):
            seed_entries[jcid] = rec["wer"]
    accepted_sec_per_word: list[float] = [
        e["sec_per_word"] for e in seed_entries.values()
        if isinstance(e.get("sec_per_word"), (int, float))
    ]
    if accepted_sec_per_word:
        info(f"Duration baseline seeded from prior run(s): "
            f"{len(accepted_sec_per_word)} chunk(s).")

    def _encode_tail_from_wav(path: Path) -> bytes:
        """Carry-over latents from an existing chunk wav's last prosody_tail s."""
        existing, _sr = sf.read(path, dtype="float32")
        if existing.ndim > 1:
            existing = existing.mean(axis=1)
        tail_samples = int(args.prosody_tail * sample_rate)
        tail = existing[-tail_samples:] if existing.size > tail_samples else existing
        return server.encode_latents(ndarray_to_wav_bytes(tail, sample_rate), "wav")

    for idx, c in enumerate(chunks):
        prev_c = chunks[idx - 1] if idx > 0 else None
        cid = int(c["id"])
        text = resolve_spoken_text(c)
        control = resolve_control(c, args)

        # In Controllable mode the parenthetical is honoured by the model, so
        # prepend it to the text: "(dry, measured)De renner...". In Hi-Fi mode
        # the model would just read the parenthetical aloud, so we never inject
        # it there — the control tag is recorded in the manifest only.
        if args.controllable and control.strip():
            target_text = apply_control(text, control)
        else:
            target_text = text

        wav_name = f"chunk_{cid:04d}.wav"
        wav_path = args.out_dir / wav_name

        if cid < args.start_at:
            # If this is the last skipped chunk before generation resumes,
            # rebuild the carry-over tail from its existing wav so the first
            # resumed chunk keeps prosody continuity across the crash/stop.
            nxt = chunks[idx + 1] if idx + 1 < len(chunks) else None
            resumes_next = nxt is not None and int(nxt["id"]) >= args.start_at
            if resumes_next and wav_path.exists():
                try:
                    prev_ref_latents = _encode_tail_from_wav(wav_path)
                    dim(f"[{cid:03d}/{n_total:03d}] skipped (resume); "
                        f"carry-over rebuilt from existing wav")
                except Exception as e:
                    prev_ref_latents = None
                    warn(f"[{cid:03d}/{n_total:03d}] skipped (resume); "
                        f"WARNING could not read for carry-over: {e}")
            else:
                dim(f"[{cid:03d}/{n_total:03d}] skipped (resume)")
            continue

        # --only-chunks: regenerate just the named chunks. For a chunk NOT in
        # the set, load its existing wav, recompute the carry-over tail from it
        # (so the next targeted chunk still gets correct prosody continuity),
        # and skip generation. The existing audio is left on disk untouched.
        if only_chunks is not None and cid not in only_chunks:
            if wav_path.exists():
                try:
                    prev_ref_latents = _encode_tail_from_wav(wav_path)
                    dim(f"[{cid:03d}/{n_total:03d}] kept (existing); carry-over refreshed")
                except Exception as e:
                    # Stale latents from an OLDER chunk must not masquerade as
                    # this chunk's prosody -- clear them; the next generated
                    # chunk simply starts without a carry-over tail.
                    prev_ref_latents = None
                    warn(f"[{cid:03d}/{n_total:03d}] kept (existing); "
                        f"WARNING could not read for carry-over: {e}; "
                        f"carry-over cleared")
            else:
                prev_ref_latents = None
                warn(f"[{cid:03d}/{n_total:03d}] kept — but no existing wav at "
                    f"{wav_name}; carry-over cleared")
            continue

        # Carry-over should only flow WITHIN a rhetorical thought: suppressed
        # when the previous chunk's position is "final" or this chunk's is
        # "opening" (see should_carry_over), unless the plan explicitly
        # overrides it via the previous chunk's carryover_after field.
        carry_ok, carry_reason = should_carry_over(prev_c, c)
        tail_latents = prev_ref_latents if carry_ok else None

        ctrl_str = f" ctrl='{control}'" if (args.controllable and control.strip()) else ""

        # ── decide what goes in the ref_audio_latents slot ─────────────────
        # Hi-Fi: timbre comes from prompt_id, so the slot is pure prosody
        #   carry-over (previous tail, subject to the suppression above).
        # Controllable: the slot is the ONLY voice anchor, so we reground the
        #   original reference into it according to --reground; suppressing
        #   the tail never drops the anchor itself.
        reground_tag = ""
        if not args.controllable or ref_anchor_latents is None or reground_mode == "off":
            chunk_ref_latents = tail_latents
        elif reground_mode == "every":
            # Original reference + previous tail (if carried), every chunk.
            # True anchor + prosody continuity in one slot.
            chunk_ref_latents = concat_latents(
                ref_anchor_latents, tail_latents, feat_dim=feat_dim
            )
            reground_tag = " reground=ref+tail" if tail_latents else " reground=ref-only"
        else:  # mode == "n": hard reset every N chunks
            if (n_done % reground_n) == 0 or tail_latents is None:
                chunk_ref_latents = ref_anchor_latents
                reground_tag = " reground=hard"
            else:
                chunk_ref_latents = tail_latents

        chunk_header(cid, n_total, bool(tail_latents),
                    "" if tail_latents else carry_reason, ctrl_str, reground_tag,
                    text[:55] + ("..." if len(text) > 55 else ""))

        t_chunk = time.time()

        target_sec_per_word, target_source = resolve_duration_target(
            accepted_sec_per_word, args.sec_per_word_target, args.baseline_min_chunks,
        )
        if args.duration_gate:
            dim(f"[duration] target: {target_source}", INDENT)

        # --candidates: generate K versioned takes instead of one "kept" take.
        # The plain chunk_NNNN.wav is left untouched (same contract as
        # --interactive's 'cand' command); pick a winner via selection.json.
        # Deliberately does NOT feed accepted_sec_per_word / prev_ref_latents /
        # wer_log -- these are exploratory takes, not an accepted result.
        if args.candidates:
            info(f"[{cid:03d}/{n_total:03d}] generating {args.candidates} candidates "
                f"(chunk_{cid:04d}_v1..v{args.candidates}.wav; plain file untouched)")
            cand_entries = []
            for version in range(1, args.candidates + 1):
                v_wav, v_wer, v_attempts, _tr, v_spw, v_dur_ok, v_dur_reason = generate_with_retry(
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
                    duration_gate=args.duration_gate,
                    target_sec_per_word=target_sec_per_word,
                    duration_floor=args.duration_floor,
                    runaway_ratio=args.runaway_ratio,
                    dragging_ratio=args.duration_ceiling,
                )
                version_path = args.out_dir / f"chunk_{cid:04d}_v{version}.wav"
                sf.write(version_path, v_wav, sample_rate, subtype="PCM_16")
                v_metrics, v_flags = analyze_chunk_audio(v_wav, sample_rate)
                cand_entries.append({
                    "version": version,
                    "file": version_path.name,
                    "wer": round(v_wer, 4) if v_wer >= 0 else None,
                    "sec_per_word": round(v_spw, 4) if v_spw else None,
                    "duration_ok": v_dur_ok,
                    "duration_reason": v_dur_reason,
                    "flags": v_flags,
                    "metrics": v_metrics,
                })
                candidate_take_line(version, len(v_wav) / sample_rate, v_wer, v_spw,
                                    args.duration_gate, v_flags, version_path.name)
            # Pre-listening sort order (see _audio_metrics.rank_candidates):
            # a filter for where to START listening, not a verdict -- the ear
            # test still decides what goes in selection.json.
            ranked = rank_candidates(cand_entries)
            cand_report_path = args.out_dir / f"chunk_{cid:04d}_candidates.json"
            cand_report_path.write_text(
                json.dumps({"id": cid, "ranking": [e["version"] for e in ranked],
                            "candidates": cand_entries},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            console.print(candidate_table(cid, ranked))
            info(f"details: {cand_report_path.name}")
            dim(f"Listen to chunk_{cid:04d}_v1..v{args.candidates}.wav and "
               f"record your pick in selection.json (e.g. {{\"{cid}\": 2}}), "
               f"then re-run 03_stitch.py.\n")
            continue

        wav, chunk_wer, attempts, accepted_transcript, sec_per_word, duration_ok, duration_reason = (
            generate_with_retry(
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
                duration_gate=args.duration_gate,
                target_sec_per_word=target_sec_per_word,
                duration_floor=args.duration_floor,
                runaway_ratio=args.runaway_ratio,
                dragging_ratio=args.duration_ceiling,
            )
        )
        if args.duration_gate and sec_per_word > 0:
            accepted_sec_per_word.append(sec_per_word)

        sf.write(wav_path, wav, sample_rate, subtype="PCM_16")

        # Progress stats.
        chunk_wall = time.time() - t_chunk
        chunk_audio_s = len(wav) / sample_rate
        total_audio_s += chunk_audio_s
        n_done += 1
        retries_this_chunk = attempts - 1
        total_retries += retries_this_chunk
        audio_metrics, audio_flags = analyze_chunk_audio(wav, sample_rate)
        entry = wer_log_entry(cid, chunk_wer, attempts, sec_per_word,
                              duration_ok, duration_reason,
                              audio_metrics=audio_metrics,
                              audio_flags=audio_flags)
        wer_log.append(entry)

        # Journal this chunk NOW (flushed to disk): if the run crashes later,
        # its bookkeeping survives for the resumed run to pick up.
        append_chunk_record(jpath, cid, entry)

        # Pronunciation diff from the ACCEPTED attempt's transcript only.
        if accepted_transcript:
            for sub in pronunciation_diff(clean_for_wer(text), accepted_transcript):
                pron_diffs.append({"id": cid, **sub})

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
        dim(f"audio={chunk_audio_s:.1f}s wall={chunk_wall:.1f}s "
           f"RTF={rtf:.2f}{wer_str}{retry_str} ETA={eta_str}", INDENT)
        if audio_flags:
            metrics_advisory_line(audio_flags)

        # Encode tail for prosody carry-over. A failure here must not kill the
        # run (the chunk itself is already accepted and on disk) nor leave a
        # stale tail -- clear it and continue.
        try:
            tail_samples = int(args.prosody_tail * sample_rate)
            tail = wav[-tail_samples:] if wav.size > tail_samples else wav
            prev_ref_latents = server.encode_latents(
                ndarray_to_wav_bytes(tail, sample_rate), "wav"
            )
        except Exception as e:
            prev_ref_latents = None
            warn_err(f"WARNING could not encode carry-over tail: {e}; "
                    f"carry-over cleared for next chunk", INDENT)

    # ── mark generation complete ───────────────────────────────────────────
    manifest["generation_complete"] = True
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_wall = time.time() - t_start
    avg_rtf = total_wall / total_audio_s if total_audio_s > 0 else 0.0

    rule("Done")
    success(f"{n_total} chunks | "
           f"{total_audio_s:.1f}s audio | "
           f"wall {total_wall:.1f}s | avg RTF {avg_rtf:.2f}")

    if wer_log:
        # Duration data is independent of ASR, so this now writes even with
        # --no-asr; WER-specific fields are simply None in that case.
        valid_wers = [e for e in wer_log if e["wer"] is not None]
        if valid_wers:
            avg_wer = sum(e["wer"] for e in valid_wers) / len(valid_wers)
            worst = max(valid_wers, key=lambda e: e["wer"])
            info(f"ASR summary: avg WER={avg_wer * 100:.1f}% | "
                f"total retries={total_retries} | "
                f"worst chunk={worst['id']} ({worst['wer'] * 100:.1f}% WER, "
                f"{worst['attempts']} attempts)")

        duration_failures = [e for e in wer_log if not e["duration_ok"]]
        if args.duration_gate and duration_failures:
            runaways = [e for e in duration_failures
                        if e["duration_reason"] == "runaway"]
            warn(f"Duration gate: {len(duration_failures)}/{len(wer_log)} chunk(s) "
                f"never cleanly passed"
                + (f" ({len(runaways)} RUNAWAY)" if runaways else "") + ".")

        new_entries = {e["id"]: e for e in wer_log}

        # Partial runs (--only-chunks, --start-at) must merge, not replace:
        # first from the journal (covers chunks a crashed prior run completed
        # but never got into wer_log.json), then from the prior wer_log.json
        # itself (covers out-dirs from before the journal existed).
        merged = dict(new_entries)
        for jcid, rec in prior_journal.items():
            if jcid not in merged and rec.get("wer"):
                merged[jcid] = rec["wer"]
        if not fresh_full_run and wer_log_path.exists():
            try:
                prior = json.loads(wer_log_path.read_text(encoding="utf-8"))
                for prior_entry in prior.get("chunks", []):
                    if prior_entry["id"] not in merged:
                        merged[prior_entry["id"]] = prior_entry
            except Exception as e:
                warn(f"WARNING: could not merge prior wer_log: {e}")

        chunks_sorted = [merged[k] for k in sorted(merged)]
        all_wers = [e["wer"] for e in chunks_sorted if e.get("wer") is not None]
        wer_log_data = {
            "avg_wer": round(sum(all_wers) / len(all_wers), 4) if all_wers else None,
            "total_retries": total_retries,
            "threshold": args.wer_threshold if asr_model is not None else None,
            "whisper_model": args.whisper_model if asr_model is not None else None,
            "duration_gate": args.duration_gate,
            "duration_floor": args.duration_floor if args.duration_gate else None,
            "runaway_ratio": args.runaway_ratio if args.duration_gate else None,
            "regenerated": sorted(only_chunks) if only_chunks else "all",
            "chunks": chunks_sorted,
        }
        wer_log_path.write_text(
            json.dumps(wer_log_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        info(f"WER log:  {wer_log_path}")

    # ── pronunciation diff ─────────────────────────────────────────────────
    # Words the model was asked to say that Whisper heard differently, from
    # ACCEPTED audio only. Manual fuel for building a respelling lexicon — the
    # tool only records; you decide what (if anything) to do with each entry.
    if pron_diffs:
        # Tally by reference word (case-insensitive grouping, original case kept).
        from collections import Counter
        counter = Counter()
        display = {}
        heard_examples = {}
        proper = {}
        for d in pron_diffs:
            key = d["ref"].lower()
            counter[key] += 1
            display.setdefault(key, d["ref"])
            proper[key] = proper.get(key, False) or d["is_proper"]
            heard_examples.setdefault(key, [])
            if d["heard"] not in heard_examples[key]:
                heard_examples[key].append(d["heard"])

        # Sort: proper nouns first, then by frequency.
        ordered = sorted(
            counter.keys(),
            key=lambda k: (not proper[k], -counter[k], k),
        )

        diff_json = {
            "summary": "Words asked-for vs heard by ASR, accepted audio only. "
                       "Candidate mispronunciations for manual lexicon building.",
            "total_substitutions": len(pron_diffs),
            "unique_words": len(ordered),
            "words": [
                {
                    "ref": display[k],
                    "count": counter[k],
                    "is_proper": proper[k],
                    "heard_as": heard_examples[k],
                    "chunks": sorted({d["id"] for d in pron_diffs
                                      if d["ref"].lower() == k}),
                }
                for k in ordered
            ],
        }
        diff_json_path = args.out_dir / "pronunciation_diff.json"
        diff_json_path.write_text(
            json.dumps(diff_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Human-readable version — the one you actually scan.
        lines = [
            "PRONUNCIATION DIFF — accepted audio only",
            "Words the model was asked to say vs what Whisper heard.",
            "Proper nouns (capitalized) listed first; these are usual lexicon targets.",
            "This is evidence only — decide manually what to respell.",
            "",
            f"{'WORD':<24} {'COUNT':>5}  {'PROPER':<7} HEARD AS",
            "-" * 72,
        ]
        for k in ordered:
            heard = ", ".join(heard_examples[k][:4])
            lines.append(
                f"{display[k]:<24} {counter[k]:>5}  "
                f"{'yes' if proper[k] else '':<7} {heard}"
            )
        diff_txt_path = args.out_dir / "pronunciation_diff.txt"
        diff_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        n_proper = sum(1 for k in ordered if proper[k])
        info(f"Pronunciation diff: {diff_txt_path}")
        info(f"{len(ordered)} unique mismatched words "
            f"({n_proper} proper nouns) — review for lexicon candidates.", INDENT2)
    elif asr_model is not None:
        dim("Pronunciation diff: no mismatches found across accepted chunks "
           "(no pronunciation_diff.json/.txt written).")

    info(f"Manifest: {manifest_path}")
    success(f"Next: python scripts/03_stitch.py --run-dir {args.out_dir} "
           f"--output {args.out_dir / 'final.wav'}")

    # ── interactive regeneration loop ──────────────────────────────────────
    if args.interactive:
        import shlex

        def _carry_from_prev(prev_id: int) -> bytes | None:
            """Rebuild carry-over latents from the preceding chunk's existing wav."""
            if prev_id < 1:
                return None
            pj = plan_lookup.get(prev_id)
            if pj is None:
                return None
            pw = args.out_dir / f"chunk_{prev_id:04d}.wav"
            if not pw.exists():
                return None
            try:
                return _encode_tail_from_wav(pw)
            except Exception as e:
                warn(f"WARNING could not rebuild carry-over from {pw.name}: "
                    f"{e}; regenerating without carry-over", INDENT2)
                return None

        def _regen(cid: int, cfg_v: float, temp_v: float, version: int | None = None):
            c = plan_lookup.get(cid)
            if c is None:
                error(f"no chunk with id {cid} in the plan.", INDENT2)
                return
            text = resolve_spoken_text(c)
            control = resolve_control(c, args)
            if args.controllable and control.strip():
                target_text = apply_control(text, control)
            else:
                target_text = text

            # carry-over from the previous chunk id in the plan order, subject
            # to the same rhetorical-boundary suppression as the batch loop.
            ids_sorted = sorted(plan_lookup)
            pos = ids_sorted.index(cid)
            prev_id = ids_sorted[pos - 1] if pos > 0 else 0
            prev_c = plan_lookup.get(prev_id)
            carry_ok, carry_reason = should_carry_over(prev_c, c)
            prev_latents = _carry_from_prev(prev_id) if carry_ok else None

            # ref slot: mirror the batch logic (including carry-over suppression)
            if not args.controllable or ref_anchor_latents is None or reground_mode == "off":
                chunk_ref = prev_latents
            elif reground_mode == "every":
                chunk_ref = concat_latents(ref_anchor_latents, prev_latents, feat_dim=feat_dim)
            else:
                chunk_ref = ref_anchor_latents if prev_latents is None else prev_latents

            tag = f" v{version}" if version else ""
            carry_str = "yes" if prev_latents else f"no ({carry_reason})"
            info(f"regen chunk {cid}{tag} @ cfg={cfg_v} temp={temp_v} "
                f"carry={carry_str}: "
                f"{text[:50]}{'...' if len(text) > 50 else ''}", INDENT2)
            regen_target_spw, regen_target_source = resolve_duration_target(
                accepted_sec_per_word, args.sec_per_word_target, args.baseline_min_chunks,
            )
            if args.duration_gate:
                dim(f"[duration] target: {regen_target_source}", INDENT2 * 2)
            wav, wer, att, _tr, spw, dok, dreason = generate_with_retry(
                server=server, text=target_text, prompt_id=prompt_id,
                ref_latents=chunk_ref, zero_shot_latents=zero_shot_latents,
                cfg=cfg_v, temperature=temp_v,
                max_generate_length=args.max_generate_length,
                lora_name=LORA_NAME, asr_model=asr_model,
                wer_threshold=args.wer_threshold, max_retries=args.max_retries,
                sample_rate=sample_rate, wer_reference=clean_for_wer(target_text),
                duration_gate=args.duration_gate,
                target_sec_per_word=regen_target_spw,
                duration_floor=args.duration_floor,
                runaway_ratio=args.runaway_ratio,
                dragging_ratio=args.duration_ceiling,
            )
            if args.duration_gate and spw > 0 and version is None:
                # Only feed the running baseline from the "canonical" regen
                # (not throwaway candidate takes), same spirit as the batch loop.
                accepted_sec_per_word.append(spw)
            if version is None:
                outp = args.out_dir / f"chunk_{cid:04d}.wav"
            else:
                outp = args.out_dir / f"chunk_{cid:04d}_v{version}.wav"
            sf.write(outp, wav, sample_rate, subtype="PCM_16")
            metrics, flags = analyze_chunk_audio(wav, sample_rate)
            if version is None:
                # A canonical regen replaces chunk_NNNN.wav, so journal the new
                # take's quality data too -- otherwise the journal (and any
                # later merge into wer_log.json) certifies audio that is no
                # longer on disk. Candidate takes (_vK) aren't journaled.
                append_chunk_record(
                    jpath, cid,
                    wer_log_entry(cid, wer, att, spw, dok, dreason,
                                  audio_metrics=metrics, audio_flags=flags),
                )
            wtxt = f" WER={wer*100:.1f}%" if wer >= 0 else ""
            ftxt = f" [{', '.join(flags)}]" if flags else ""
            (warn if flags else success)(
                f"wrote {outp.name} ({len(wav)/sample_rate:.1f}s{wtxt}{ftxt})", INDENT2)

        def _candidates(cid: int, k: int, cfg_v: float, temp_v: float):
            """Generate k candidate versions of a chunk as _v1.._vk."""
            info(f"generating {k} candidates of chunk {cid} "
                f"@ cfg={cfg_v} temp={temp_v} ...", INDENT2)
            for v in range(1, k + 1):
                _regen(cid, cfg_v, temp_v, version=v)
            dim(f"done. Listen to chunk_{cid:04d}_v1.. and record your pick "
               f"in selection.json (e.g. {{\"{cid}\": 2}}).", INDENT2)

        # Build/refresh the id->chunk lookup from the current plan.
        def _load_plan_lookup():
            p = json.loads(args.plan.read_text(encoding="utf-8"))
            return {int(c["id"]): c for c in p.get("chunks", [])}

        plan_lookup = _load_plan_lookup()

        console.print()
        rule("INTERACTIVE MODE — model stays loaded")
        console.print(command_table())
        console.print()

        while True:
            try:
                raw = console.input("[bold cyan]regen> [/]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not raw:
                continue
            if raw in ("quit", "exit", "q"):
                break
            if raw == "reload":
                plan_lookup = _load_plan_lookup()
                info(f"plan reloaded ({len(plan_lookup)} chunks).", INDENT2)
                continue
            if raw == "list":
                for k in sorted(plan_lookup):
                    t = resolve_spoken_text(plan_lookup[k])
                    plain(f"{k:3d}  {t[:60]}{'...' if len(t) > 60 else ''}", INDENT2)
                continue
            # candidate command: cand <id> <k> [--cfg X] [--temp Y]
            if raw.startswith("cand"):
                try:
                    parts = shlex.split(raw)
                    cid = int(parts[1])
                    k = int(parts[2])
                    cfg_v, temp_v = args.cfg, args.temperature
                    i = 3
                    while i < len(parts):
                        if parts[i] == "--cfg" and i + 1 < len(parts):
                            cfg_v = float(parts[i + 1]); i += 2
                        elif parts[i] in ("--temp", "--temperature") and i + 1 < len(parts):
                            temp_v = float(parts[i + 1]); i += 2
                        else:
                            i += 1
                except (ValueError, IndexError):
                    error("usage: cand <id> <k> [--cfg X] [--temp Y]", INDENT2)
                    continue
                _candidates(cid, k, cfg_v, temp_v)
                continue
            # parse: <id> [--cfg X] [--temp Y]
            try:
                parts = shlex.split(raw)
                cid = int(parts[0])
                cfg_v, temp_v = args.cfg, args.temperature
                i = 1
                while i < len(parts):
                    if parts[i] in ("--cfg",) and i + 1 < len(parts):
                        cfg_v = float(parts[i + 1]); i += 2
                    elif parts[i] in ("--temp", "--temperature") and i + 1 < len(parts):
                        temp_v = float(parts[i + 1]); i += 2
                    else:
                        i += 1
            except (ValueError, IndexError):
                error("usage: <id> [--cfg X] [--temp Y] | cand <id> <k> | reload | list | quit", INDENT2)
                continue
            _regen(cid, cfg_v, temp_v)

    server.stop()


if __name__ == "__main__":
    main()