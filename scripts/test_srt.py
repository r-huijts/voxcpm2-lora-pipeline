#!/usr/bin/env python3
"""
test_srt.py — regression test for _srt.py (SRT timecode/cue formatting used
by 03_stitch.py's timeline output).

Run after touching _srt.py:
    python scripts/test_srt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _srt import build_srt, format_srt_timestamp


def test_zero():
    assert format_srt_timestamp(0.0) == "00:00:00,000"


def test_simple_seconds_and_ms():
    assert format_srt_timestamp(1.234) == "00:00:01,234"


def test_minutes_and_hours_carry():
    # 1h 1m 1.234s = 3661.234s
    assert format_srt_timestamp(3661.234) == "01:01:01,234"


def test_ms_rounds_up_into_next_second():
    assert format_srt_timestamp(1.9997) == "00:00:02,000"


def test_negative_clamped_to_zero():
    assert format_srt_timestamp(-0.5) == "00:00:00,000"


def test_build_srt_basic_two_cues():
    entries = [
        {"text": "Eerste zin.", "start": 0.0, "end": 2.5},
        {"text": "Tweede zin.", "start": 2.7, "end": 5.0},
    ]
    srt = build_srt(entries)
    lines = srt.split("\n")
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,500"
    assert lines[2] == "Eerste zin."
    assert lines[3] == ""
    assert lines[4] == "2"
    assert lines[5] == "00:00:02,700 --> 00:00:05,000"
    assert lines[6] == "Tweede zin."


def test_build_srt_empty_text_falls_back_to_blank_not_crash():
    entries = [{"text": "", "start": 0.0, "end": 1.0}]
    srt = build_srt(entries)
    assert "1\n00:00:00,000 --> 00:00:01,000\n" in srt


def main():
    tests = [
        test_zero,
        test_simple_seconds_and_ms,
        test_minutes_and_hours_carry,
        test_ms_rounds_up_into_next_second,
        test_negative_clamped_to_zero,
        test_build_srt_basic_two_cues,
        test_build_srt_empty_text_falls_back_to_blank_not_crash,
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
