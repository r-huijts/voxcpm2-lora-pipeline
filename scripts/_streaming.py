"""Streaming-generation helpers for 02_generate_nanovllm.py.

Only depends on numpy + stdlib time, so it's testable without the heavy
nano-vllm-voxcpm/torch stack.
"""
import time

import numpy as np


def collect_chunks(generator, sample_rate: int | None = None,
                    progress_label: str | None = None,
                    expected_seconds: float | None = None,
                    progress_interval_s: float = 2.0,
                    clock=time.time) -> np.ndarray:
    """
    Drain server.generate()'s streaming generator into one array. When
    progress_label is given (and sample_rate is known), prints a live,
    in-place-updating line every ~progress_interval_s seconds -- audio
    generation runs at several seconds of wall time per second of audio, so
    without this there's a long silent gap between chunks with no signal
    that anything is happening.

    clock is injectable for deterministic testing; real callers should
    leave it at the default (time.time).
    """
    parts = []
    show_progress = progress_label is not None and sample_rate is not None
    total_samples = 0
    t_start = clock()
    last_print = 0.0
    for c in generator:
        if c is None:
            continue
        arr = np.asarray(c, dtype=np.float32).reshape(-1)
        if arr.size:
            parts.append(arr)
            if show_progress:
                total_samples += arr.size
                now = clock()
                if now - last_print >= progress_interval_s:
                    audio_s = total_samples / sample_rate
                    elapsed = now - t_start
                    expected_str = f"/~{expected_seconds:.1f}s expected" if expected_seconds else ""
                    print(f"\r         [{progress_label}] generating... "
                          f"{audio_s:.1f}s audio{expected_str} ({elapsed:.0f}s elapsed)",
                          end="", flush=True)
                    last_print = now
    if show_progress and last_print > 0.0:
        print()  # end the in-place progress line before the next print
    if not parts:
        raise RuntimeError("Empty audio returned from generator.")
    return np.concatenate(parts)
