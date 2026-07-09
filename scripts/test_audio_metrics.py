#!/usr/bin/env python3
"""
test_audio_metrics.py — regression test for _audio_metrics.py, the advisory
waveform quality metrics (technical sanity, boundary, prosody-flatness,
loudness-consistency, candidate ranking).

All on synthetic waveforms — no model, no soundfile, numpy only.

Run after touching _audio_metrics.py:
    python scripts/test_audio_metrics.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _audio_metrics import (
    compute_metrics,
    derive_flags,
    loudness_outliers,
    rank_candidates,
    rms_dbfs,
)

SR = 16000


def tone(freq_hz: float, dur_s: float, amp: float = 0.5, sr: int = SR,
         fade_ms: float = 0.0) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    x = amp * np.sin(2 * np.pi * freq_hz * t)
    if fade_ms > 0:
        n = min(len(x), int(fade_ms / 1000.0 * sr))
        ramp = np.linspace(0.0, 1.0, n)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def glide(f_start: float, f_end: float, dur_s: float, amp: float = 0.5,
          sr: int = SR) -> np.ndarray:
    """Tone whose pitch sweeps f_start -> f_end (phase-integrated)."""
    t = np.arange(int(dur_s * sr)) / sr
    freq = np.linspace(f_start, f_end, len(t))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return amp * np.sin(phase)


def silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(dur_s * sr))


def test_levels_on_sine():
    m = compute_metrics(tone(150, 2.0, amp=0.5, fade_ms=100), SR)
    assert abs(m["peak_dbfs"] - (-6.02)) < 0.3, m["peak_dbfs"]
    assert abs(m["rms_dbfs"] - (-9.03)) < 0.5, m["rms_dbfs"]
    assert m["clip_ratio"] == 0.0
    assert "clipping" not in derive_flags(m)
    assert "low_level" not in derive_flags(m)


def test_clipping_flagged():
    x = np.clip(tone(150, 2.0, amp=1.5), -1.0, 1.0)
    m = compute_metrics(x, SR)
    assert m["clip_ratio"] > 0.01, m["clip_ratio"]
    assert "clipping" in derive_flags(m)


def test_low_level_flagged():
    # Quiet but still above the -40 dB silence threshold (frames stay active).
    m = compute_metrics(tone(150, 2.0, amp=0.016), SR)
    assert "low_level" in derive_flags(m), m["active_rms_dbfs"]


def test_all_silent_flags_low_level_not_crash():
    m = compute_metrics(silence(1.0), SR)
    assert m["f0_median_hz"] is None
    assert m["energy_cv"] is None
    assert "low_level" in derive_flags(m)


def test_empty_audio_not_crash():
    m = compute_metrics(np.zeros(0), SR)
    assert m["duration_s"] == 0.0
    assert derive_flags(m) == []  # nothing to measure, nothing to claim


def test_f0_median_accuracy_and_monotone_flagged():
    m = compute_metrics(tone(150, 2.0, amp=0.5, fade_ms=100), SR)
    assert m["f0_median_hz"] is not None
    assert abs(m["f0_median_hz"] - 150.0) < 5.0, m["f0_median_hz"]
    assert m["voiced_fraction"] > 0.5
    assert m["f0_std_semitones"] < 0.5, m["f0_std_semitones"]
    assert "flat_pitch" in derive_flags(m)


def test_pitch_glide_not_flagged_flat():
    m = compute_metrics(glide(120, 240, 2.0), SR)
    assert m["f0_std_semitones"] is not None
    assert m["f0_std_semitones"] > 2.0, m["f0_std_semitones"]
    assert "flat_pitch" not in derive_flags(m)


def test_constant_energy_flagged_modulated_not():
    flat = compute_metrics(tone(150, 2.0, amp=0.5), SR)
    assert "flat_energy" in derive_flags(flat), flat["energy_cv"]
    t = np.arange(int(2.0 * SR)) / SR
    am = (0.25 + 0.75 * np.abs(np.sin(2 * np.pi * 1.5 * t))) * np.sin(2 * np.pi * 150 * t) * 0.5
    mod = compute_metrics(am, SR)
    assert mod["energy_cv"] > 0.25, mod["energy_cv"]
    assert "flat_energy" not in derive_flags(mod)


def test_edge_silence_measured():
    x = np.concatenate([silence(0.2), tone(150, 1.5, fade_ms=100), silence(0.3)])
    m = compute_metrics(x, SR)
    assert abs(m["lead_silence_ms"] - 200) < 50, m["lead_silence_ms"]
    assert abs(m["trail_silence_ms"] - 300) < 50, m["trail_silence_ms"]


def test_abrupt_end_flagged_faded_end_not():
    cut = compute_metrics(tone(150, 2.0), SR)  # ends at full amplitude
    assert cut["edge_end_ratio"] > 0.9, cut["edge_end_ratio"]
    assert "abrupt_end" in derive_flags(cut)
    faded = compute_metrics(tone(150, 2.0, fade_ms=300), SR)
    assert "abrupt_end" not in derive_flags(faded), faded["edge_end_ratio"]


def test_loudness_outliers_flags_the_quiet_chunk():
    by_id = {1: -20.0, 2: -20.5, 3: -26.0, 4: -19.8, 5: -20.2}
    out = loudness_outliers(by_id)
    assert set(out) == {3}, out
    assert out[3] < -3.0, out[3]


def test_loudness_outliers_needs_neighbours_and_skips_none():
    assert loudness_outliers({1: -20.0, 2: -30.0}) == {}  # 1 neighbour each
    out = loudness_outliers({1: -20.0, 2: None, 3: -20.3, 4: -27.0, 5: -20.1})
    assert 2 not in out
    assert 4 in out


def test_rank_candidates_order():
    v1 = {"version": 1, "wer": 0.05, "duration_ok": True, "flags": [],
          "metrics": {"f0_std_semitones": 2.0}}
    v2 = {"version": 2, "wer": 0.0, "duration_ok": True, "flags": ["flat_pitch"],
          "metrics": {"f0_std_semitones": 0.4}}
    v3 = {"version": 3, "wer": 0.10, "duration_ok": False, "flags": [],
          "metrics": {"f0_std_semitones": 3.0}}
    ranked = rank_candidates([v2, v3, v1])
    assert [e["version"] for e in ranked] == [1, 2, 3], ranked


def test_rank_candidates_tie_broken_by_livelier_pitch():
    a = {"version": 1, "wer": 0.0, "duration_ok": True, "flags": [],
         "metrics": {"f0_std_semitones": 1.5}}
    b = {"version": 2, "wer": 0.0, "duration_ok": True, "flags": [],
         "metrics": {"f0_std_semitones": 3.5}}
    ranked = rank_candidates([a, b])
    assert [e["version"] for e in ranked] == [2, 1]


def test_rank_candidates_missing_fields_not_crash():
    ranked = rank_candidates([{"version": 1}, {"version": 2, "wer": None,
                                               "metrics": None, "flags": None}])
    assert len(ranked) == 2


def test_rms_dbfs_helper():
    assert abs(rms_dbfs(tone(150, 1.0, amp=0.5)) - (-9.03)) < 0.3
    assert rms_dbfs(np.zeros(100)) == -200.0
    assert rms_dbfs(np.zeros(0)) == -200.0


def test_metrics_json_serializable():
    import json
    m = compute_metrics(tone(150, 1.0, fade_ms=50), SR)
    json.dumps(m)  # raises if any numpy scalar leaked through


def main():
    tests = [
        test_levels_on_sine,
        test_clipping_flagged,
        test_low_level_flagged,
        test_all_silent_flags_low_level_not_crash,
        test_empty_audio_not_crash,
        test_f0_median_accuracy_and_monotone_flagged,
        test_pitch_glide_not_flagged_flat,
        test_constant_energy_flagged_modulated_not,
        test_edge_silence_measured,
        test_abrupt_end_flagged_faded_end_not,
        test_loudness_outliers_flags_the_quiet_chunk,
        test_loudness_outliers_needs_neighbours_and_skips_none,
        test_rank_candidates_order,
        test_rank_candidates_tie_broken_by_livelier_pitch,
        test_rank_candidates_missing_fields_not_crash,
        test_rms_dbfs_helper,
        test_metrics_json_serializable,
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
