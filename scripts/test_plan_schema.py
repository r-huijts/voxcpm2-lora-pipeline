#!/usr/bin/env python3
"""
test_plan_schema.py — regression test for the plan-chunk field fallback
chains in _plan_schema.py, used by 02_generate_nanovllm.py and 03_stitch.py
to stay compatible with old-format plans (text-only) after 01_chunk.py
switched to script-built source_text/spoken_text.

Run after touching _plan_schema.py:
    python scripts/test_plan_schema.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _plan_schema import resolve_source_text, resolve_spoken_text


def test_new_format_chunk():
    chunk = {"source_text": "Origineel.", "spoken_text": "Origineel gesproken."}
    assert resolve_spoken_text(chunk) == "Origineel gesproken."
    assert resolve_source_text(chunk) == "Origineel."


def test_old_format_chunk_text_only():
    chunk = {"text": "Oude stijl tekst."}
    assert resolve_spoken_text(chunk) == "Oude stijl tekst."
    assert resolve_source_text(chunk) == "Oude stijl tekst."


def test_partial_new_format_chunk():
    # source_text present but spoken_text missing (e.g. --no-lexicon path
    # producing an empty transform) -- spoken falls back to source.
    chunk = {"source_text": "Alleen origineel."}
    assert resolve_spoken_text(chunk) == "Alleen origineel."
    assert resolve_source_text(chunk) == "Alleen origineel."


def test_empty_chunk_never_crashes():
    assert resolve_spoken_text({}) == ""
    assert resolve_source_text({}) == ""


def main():
    tests = [
        test_new_format_chunk,
        test_old_format_chunk_text_only,
        test_partial_new_format_chunk,
        test_empty_chunk_never_crashes,
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
