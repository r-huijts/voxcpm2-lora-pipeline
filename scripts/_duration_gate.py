"""Duration-ratio gate for 02_generate_nanovllm.py.

The WER gate verifies WORDS, not delivery -- a chunk can score perfect WER
while being rushed or truncated. This checks generated audio duration
against a per-voice baseline of seconds-per-word, as a RATIO (never an
absolute threshold, which false-alarms across content of different length).

The same ratio-to-expected-duration check also catches the opposite failure:
a generation that "never stops" (a documented VoxCPM failure mode that fills
VRAM) shows up as audio far LONGER than its word count would expect, long
before the caller's max_generate_length cap is hit in wall-clock terms.
"""
import statistics

# Rough Dutch narration baseline (~150 wpm / dry, measured delivery). Only
# used until enough real chunks have been accepted to trust a running
# median -- see resolve_duration_target().
BUILTIN_DEFAULT_SEC_PER_WORD = 0.40


def resolve_duration_target(
    accepted_sec_per_word: list,
    cli_target: float | None,
    baseline_min_chunks: int,
    builtin_default: float = BUILTIN_DEFAULT_SEC_PER_WORD,
) -> tuple:
    """
    Resolve the seconds-per-word target to judge THIS chunk against.

    Order: explicit --sec-per-word-target always wins. Otherwise use the
    conservative built-in default until baseline_min_chunks chunks have been
    accepted (early accepted chunks may themselves be flawed, so a running
    median isn't trusted too soon); after that, use the running median of
    accepted chunks' sec_per_word.

    Returns (target, source_label) -- the label is for logging, so a
    rejection is explainable after the fact.
    """
    if cli_target is not None:
        return cli_target, f"cli override {cli_target:.3f}s/word"
    n = len(accepted_sec_per_word)
    if n < baseline_min_chunks:
        return builtin_default, f"built-in default {builtin_default:.3f}s/word ({n}/{baseline_min_chunks} baseline chunks so far)"
    median = statistics.median(accepted_sec_per_word)
    return median, f"running median {median:.3f}s/word (n={n})"


def check_duration(
    audio_seconds: float,
    words: int,
    target_sec_per_word: float,
    duration_floor: float,
    runaway_ratio: float,
) -> tuple:
    """
    Judge one generated attempt's duration against expectation.

    Returns (ok, reason, sec_per_word, expected_seconds):
      reason is "ok", "too_short" (rushed/truncated -- below duration_floor
      * expected), or "runaway" (far longer than expected -- above
      runaway_ratio * expected; the model likely never cleanly stopped).

    Never judged on absolute duration alone -- always relative to the
    word-count-derived expectation, since a long chunk can be entirely
    valid if its text is long.
    """
    if words <= 0:
        return True, "ok", 0.0, 0.0
    sec_per_word = audio_seconds / words
    expected_seconds = words * target_sec_per_word
    if audio_seconds < duration_floor * expected_seconds:
        return False, "too_short", sec_per_word, expected_seconds
    if audio_seconds > runaway_ratio * expected_seconds:
        return False, "runaway", sec_per_word, expected_seconds
    return True, "ok", sec_per_word, expected_seconds


def check_dragging(audio_seconds: float, expected_seconds: float, dragging_ratio: float | None) -> bool:
    """Optional, off-by-default soft warning for a chunk that's noticeably
    slower than expected but not bad enough to be a runaway. Informational
    only -- never triggers a retry."""
    if dragging_ratio is None or expected_seconds <= 0:
        return False
    return audio_seconds > dragging_ratio * expected_seconds


def select_best_attempt(attempts_data: list, target_sec_per_word: float) -> dict:
    """
    Pick the best of several generation attempts (each a dict with at least
    wer, wer_ok, dur_ok, sec_per_word). Combined criterion: prefer attempts
    passing BOTH the WER and duration gates (lowest WER among those, ties
    broken by closeness to the expected pace); else the lowest-WER attempt
    that at least passes duration; else the lowest-WER attempt overall.
    """
    def _rank_key(a):
        wer_component = a["wer"] if a["wer"] >= 0 else 0.0
        duration_deviation = abs(a["sec_per_word"] - target_sec_per_word)
        return (wer_component, duration_deviation)

    both_ok = [a for a in attempts_data if a["wer_ok"] and a["dur_ok"]]
    if both_ok:
        return min(both_ok, key=_rank_key)
    dur_ok_only = [a for a in attempts_data if a["dur_ok"]]
    if dur_ok_only:
        return min(dur_ok_only, key=_rank_key)
    return min(attempts_data, key=_rank_key)
