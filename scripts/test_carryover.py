#!/usr/bin/env python3
"""
test_carryover.py — regression test for should_carry_over() (_carryover.py),
used by 02_generate_nanovllm.py to suppress prosody carry-over at rhetorical
boundaries.

Run after touching _carryover.py:
    python scripts/test_carryover.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _carryover import should_carry_over

CONTINUING = {"position": "continuing"}
OPENING = {"position": "opening"}
FINAL = {"position": "final"}


def test_no_previous_chunk():
    ok, reason = should_carry_over(None, OPENING)
    assert ok is False
    assert "no previous" in reason


def test_final_then_anything_suppresses():
    # A chunk following a "final" chunk generates with no prosody tail.
    for curr in (OPENING, CONTINUING, FINAL):
        ok, reason = should_carry_over(FINAL, curr)
        assert ok is False, f"expected suppression after final, curr={curr}"
        assert "final" in reason


def test_opening_curr_suppresses_regardless_of_prev():
    # An "opening" chunk generates with no inherited tail, even if the
    # previous chunk was mid-thought.
    for prev in (CONTINUING, OPENING):
        ok, reason = should_carry_over(prev, OPENING)
        assert ok is False, f"expected suppression into opening, prev={prev}"


def test_continuing_carries_over():
    # Mid-thought "continuing" chunks still carry over.
    ok, reason = should_carry_over(CONTINUING, CONTINUING)
    assert ok is True
    ok, reason = should_carry_over(OPENING, CONTINUING)
    assert ok is True
    ok, reason = should_carry_over(CONTINUING, FINAL)
    assert ok is True


def test_plan_override_forces_on_after_final():
    prev = {"position": "final", "carryover_after": True}
    ok, reason = should_carry_over(prev, OPENING)
    assert ok is True
    assert "override" in reason


def test_plan_override_forces_off_between_continuing():
    prev = {"position": "continuing", "carryover_after": False}
    ok, reason = should_carry_over(prev, CONTINUING)
    assert ok is False
    assert "override" in reason


def main():
    tests = [
        test_no_previous_chunk,
        test_final_then_anything_suppresses,
        test_opening_curr_suppresses_regardless_of_prev,
        test_continuing_carries_over,
        test_plan_override_forces_on_after_final,
        test_plan_override_forces_off_between_continuing,
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
