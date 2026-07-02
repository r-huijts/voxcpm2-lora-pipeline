#!/usr/bin/env python3
"""
test_duration_gate.py — regression test for _duration_gate.py, the
duration-ratio quality gate in 02_generate_nanovllm.py.

Run after touching _duration_gate.py:
    python scripts/test_duration_gate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _duration_gate import (
    BUILTIN_DEFAULT_SEC_PER_WORD,
    check_dragging,
    check_duration,
    resolve_duration_target,
    select_best_attempt,
)


def _attempt(wer, wer_ok, dur_ok, sec_per_word):
    return {"wer": wer, "wer_ok": wer_ok, "dur_ok": dur_ok, "sec_per_word": sec_per_word,
            "audio": f"audio(wer={wer},dur_ok={dur_ok})"}


def test_resolve_cli_override_always_wins():
    target, label = resolve_duration_target([0.3, 0.3, 0.3, 0.3, 0.3], cli_target=0.5,
                                             baseline_min_chunks=5)
    assert target == 0.5
    assert "cli override" in label


def test_resolve_uses_builtin_default_before_baseline():
    target, label = resolve_duration_target([0.2, 0.2], cli_target=None, baseline_min_chunks=5)
    assert target == BUILTIN_DEFAULT_SEC_PER_WORD
    assert "built-in default" in label
    assert "2/5" in label


def test_resolve_switches_to_running_median_after_baseline():
    accepted = [0.30, 0.32, 0.31, 0.29, 0.33]  # 5 chunks, median 0.31
    target, label = resolve_duration_target(accepted, cli_target=None, baseline_min_chunks=5)
    assert abs(target - 0.31) < 1e-9
    assert "running median" in label


def test_truncated_chunk_detected_even_at_perfect_wer():
    # 20 words expected at 0.4s/word = 8s; audio is only 2s (far short).
    ok, reason, spw, expected = check_duration(
        audio_seconds=2.0, words=20, target_sec_per_word=0.4,
        duration_floor=0.5, runaway_ratio=2.0,
    )
    assert ok is False
    assert reason == "too_short"
    assert expected == 8.0


def test_normal_length_chunk_not_flagged():
    # 20 words at 0.4s/word = 8s expected; audio is 7.5s -- well within floor.
    ok, reason, spw, expected = check_duration(
        audio_seconds=7.5, words=20, target_sec_per_word=0.4,
        duration_floor=0.5, runaway_ratio=2.0,
    )
    assert ok is True
    assert reason == "ok"


def test_runaway_detected_via_ratio_not_absolute_length():
    # 5 words expected at 0.4s/word = 2s; audio is 30s -- way over ratio,
    # even though 30s alone isn't "absurd" for some other chunk's word count.
    ok, reason, spw, expected = check_duration(
        audio_seconds=30.0, words=5, target_sec_per_word=0.4,
        duration_floor=0.5, runaway_ratio=2.0,
    )
    assert ok is False
    assert reason == "runaway"


def test_long_chunk_with_long_text_not_falsely_flagged_as_runaway():
    # A long chunk (100 words) legitimately produces long audio (35s), well
    # under the ratio ceiling relative to ITS OWN expectation (40s).
    ok, reason, spw, expected = check_duration(
        audio_seconds=35.0, words=100, target_sec_per_word=0.4,
        duration_floor=0.5, runaway_ratio=2.0,
    )
    assert ok is True
    assert reason == "ok"


def test_zero_words_never_flagged():
    ok, reason, spw, expected = check_duration(
        audio_seconds=1.0, words=0, target_sec_per_word=0.4,
        duration_floor=0.5, runaway_ratio=2.0,
    )
    assert ok is True


def test_dragging_off_by_default():
    assert check_dragging(10.0, expected_seconds=5.0, dragging_ratio=None) is False


def test_dragging_flagged_when_enabled():
    assert check_dragging(10.0, expected_seconds=5.0, dragging_ratio=1.5) is True
    assert check_dragging(6.0, expected_seconds=5.0, dragging_ratio=1.5) is False


def test_select_prefers_both_gates_passing():
    attempts = [
        _attempt(wer=0.30, wer_ok=False, dur_ok=True, sec_per_word=0.40),
        _attempt(wer=0.10, wer_ok=True, dur_ok=True, sec_per_word=0.40),  # both ok
        _attempt(wer=0.05, wer_ok=True, dur_ok=False, sec_per_word=0.05),  # lower WER but fails duration
    ]
    best = select_best_attempt(attempts, target_sec_per_word=0.40)
    assert best["wer"] == 0.10 and best["dur_ok"] is True


def test_select_lowest_wer_among_both_ok():
    attempts = [
        _attempt(wer=0.12, wer_ok=True, dur_ok=True, sec_per_word=0.40),
        _attempt(wer=0.08, wer_ok=True, dur_ok=True, sec_per_word=0.41),
    ]
    best = select_best_attempt(attempts, target_sec_per_word=0.40)
    assert best["wer"] == 0.08


def test_select_falls_back_to_duration_ok_when_none_pass_both():
    # No attempt passes both -- prefer the lowest-WER attempt that at least
    # passes duration, even if a worse-duration attempt had lower WER.
    attempts = [
        _attempt(wer=0.05, wer_ok=False, dur_ok=False, sec_per_word=0.05),
        _attempt(wer=0.25, wer_ok=False, dur_ok=True, sec_per_word=0.39),
    ]
    best = select_best_attempt(attempts, target_sec_per_word=0.40)
    assert best["dur_ok"] is True
    assert best["wer"] == 0.25


def test_select_falls_back_to_lowest_wer_overall_when_none_pass_duration():
    attempts = [
        _attempt(wer=0.30, wer_ok=False, dur_ok=False, sec_per_word=0.05),
        _attempt(wer=0.10, wer_ok=True, dur_ok=False, sec_per_word=0.90),
    ]
    best = select_best_attempt(attempts, target_sec_per_word=0.40)
    assert best["wer"] == 0.10


def test_select_ties_broken_by_closeness_to_expected_pace():
    # Equal WER (e.g. ASR disabled -> both -1 -> tied key 0.0); the one
    # closer to the target pace should win.
    attempts = [
        _attempt(wer=-1.0, wer_ok=True, dur_ok=True, sec_per_word=0.55),
        _attempt(wer=-1.0, wer_ok=True, dur_ok=True, sec_per_word=0.41),
    ]
    best = select_best_attempt(attempts, target_sec_per_word=0.40)
    assert best["sec_per_word"] == 0.41


def main():
    tests = [
        test_resolve_cli_override_always_wins,
        test_resolve_uses_builtin_default_before_baseline,
        test_resolve_switches_to_running_median_after_baseline,
        test_truncated_chunk_detected_even_at_perfect_wer,
        test_normal_length_chunk_not_flagged,
        test_runaway_detected_via_ratio_not_absolute_length,
        test_long_chunk_with_long_text_not_falsely_flagged_as_runaway,
        test_zero_words_never_flagged,
        test_dragging_off_by_default,
        test_dragging_flagged_when_enabled,
        test_select_prefers_both_gates_passing,
        test_select_lowest_wer_among_both_ok,
        test_select_falls_back_to_duration_ok_when_none_pass_both,
        test_select_falls_back_to_lowest_wer_overall_when_none_pass_duration,
        test_select_ties_broken_by_closeness_to_expected_pace,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
    if failures:
        sys.exit(f"\n{failures}/{len(tests)} test(s) failed.")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
