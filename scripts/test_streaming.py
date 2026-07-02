#!/usr/bin/env python3
"""
test_streaming.py — regression test for _streaming.py's collect_chunks(),
which drains 02_generate_nanovllm.py's server.generate() streaming
generator and (when asked) prints live in-place progress during the long
silent gap of a single chunk's generation.

Only needs numpy. Run after touching _streaming.py:
    python scripts/test_streaming.py
"""
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from _streaming import collect_chunks


class FakeClock:
    """Deterministic clock: advances by `step` seconds each call."""
    def __init__(self, step=1.0):
        self.t = 0.0
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


def _fake_generator(n_chunks, samples_per_chunk, include_none=False):
    for i in range(n_chunks):
        if include_none and i == 1:
            yield None
        yield np.zeros(samples_per_chunk, dtype=np.float32) + 0.01


def test_concatenates_all_chunks_in_order():
    result = collect_chunks(_fake_generator(4, 100))
    assert len(result) == 400


def test_none_chunks_are_skipped_not_crashed():
    result = collect_chunks(_fake_generator(3, 100, include_none=True))
    assert len(result) == 300  # the None yield contributes nothing


def test_empty_generator_raises():
    try:
        collect_chunks(iter([]))
        raise AssertionError("expected RuntimeError for an empty generator")
    except RuntimeError:
        pass


def test_no_progress_output_without_label_or_sample_rate():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        collect_chunks(_fake_generator(5, 1000), sample_rate=None, progress_label="attempt 1")
        collect_chunks(_fake_generator(5, 1000), sample_rate=16000, progress_label=None)
    assert buf.getvalue() == "", "no progress line should print unless BOTH label and sample_rate are given"


def test_progress_prints_when_label_and_sample_rate_given():
    clock = FakeClock(step=1.0)  # 1 simulated second per generator step
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        collect_chunks(
            _fake_generator(6, 8000),  # 8000 samples/chunk
            sample_rate=16000,
            progress_label="attempt 1",
            expected_seconds=3.0,
            progress_interval_s=0.5,  # throttle well below the 1s clock step -> prints every chunk
            clock=clock,
        )
    output = buf.getvalue()
    assert "[attempt 1] generating..." in output
    assert "expected" in output
    assert output.count("\r") >= 1


def test_progress_throttled_by_interval():
    # step=0.1s per chunk, interval=1.0s -> most chunks should NOT trigger a
    # print; only roughly one per ~10 chunks should.
    clock = FakeClock(step=0.1)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        collect_chunks(
            _fake_generator(10, 100),
            sample_rate=16000,
            progress_label="attempt 1",
            progress_interval_s=1.0,
            clock=clock,
        )
    prints = buf.getvalue().count("[attempt 1] generating...")
    assert prints <= 2, f"expected throttling to suppress most updates, got {prints} prints"


def main():
    tests = [
        test_concatenates_all_chunks_in_order,
        test_none_chunks_are_skipped_not_crashed,
        test_empty_generator_raises,
        test_no_progress_output_without_label_or_sample_rate,
        test_progress_prints_when_label_and_sample_rate_given,
        test_progress_throttled_by_interval,
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
