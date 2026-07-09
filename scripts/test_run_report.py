#!/usr/bin/env python3
"""
test_run_report.py — regression test for _run_report.py (Sub-task 5b: QA
sheet aggregating wer_log.json + timeline + selection into one report).

Run after touching _run_report.py:
    python scripts/test_run_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run_report import build_report_rows, compute_totals, format_report_txt

TIMELINE = [
    {"id": 1, "duration": 4.0},
    {"id": 2, "duration": 3.0},
    {"id": 3, "duration": 6.0},
]


def test_row_pulls_wer_log_fields():
    wer_by_id = {1: {"wer": 0.05, "attempts": 1, "sec_per_word": 0.35,
                      "duration_ok": True, "duration_reason": "ok"}}
    rows = build_report_rows(TIMELINE[:1], wer_by_id, {}, wer_threshold=0.15)
    r = rows[0]
    assert r["wer"] == 0.05
    assert r["retries"] == 0
    assert abs(r["wpm"] - (60.0 / 0.35)) < 1e-9
    assert r["duration"] == 4.0
    assert r["warnings"] == []


def test_missing_wer_log_entry_leaves_fields_none_not_crash():
    rows = build_report_rows(TIMELINE, {}, {}, wer_threshold=0.15)
    assert len(rows) == 3
    for r in rows:
        assert r["wer"] is None
        assert r["retries"] is None
        assert r["warnings"] == []


def test_wer_over_threshold_produces_warning():
    wer_by_id = {2: {"wer": 0.30, "attempts": 3, "sec_per_word": 0.40,
                      "duration_ok": True, "duration_reason": "ok"}}
    rows = build_report_rows(TIMELINE, wer_by_id, {}, wer_threshold=0.15)
    r2 = next(r for r in rows if r["id"] == 2)
    assert r2["retries"] == 2
    assert any("WER" in w for w in r2["warnings"])


def test_duration_gate_failure_produces_warning():
    wer_by_id = {3: {"wer": 0.02, "attempts": 3, "sec_per_word": 1.2,
                      "duration_ok": False, "duration_reason": "runaway"}}
    rows = build_report_rows(TIMELINE, wer_by_id, {}, wer_threshold=0.15)
    r3 = next(r for r in rows if r["id"] == 3)
    assert any("RUNAWAY" in w for w in r3["warnings"])


def test_candidate_version_surfaced():
    rows = build_report_rows(TIMELINE, {}, selection={2: 3}, wer_threshold=0.15)
    r2 = next(r for r in rows if r["id"] == 2)
    assert r2["candidate_version"] == 3
    r1 = next(r for r in rows if r["id"] == 1)
    assert r1["candidate_version"] is None


def test_totals_aggregate_correctly():
    wer_by_id = {
        1: {"wer": 0.10, "attempts": 1, "sec_per_word": 0.40, "duration_ok": True, "duration_reason": "ok"},
        2: {"wer": 0.20, "attempts": 2, "sec_per_word": 0.30, "duration_ok": True, "duration_reason": "ok"},
    }
    rows = build_report_rows(TIMELINE, wer_by_id, {}, wer_threshold=0.15)
    totals = compute_totals(rows, total_audio_seconds=13.0)
    assert totals["chunk_count"] == 3
    assert totals["total_audio_seconds"] == 13.0
    assert abs(totals["mean_wer"] - 0.15) < 1e-9
    assert totals["total_retries"] == 1  # only chunk 2 had a retry
    assert totals["gate_failures"] == 1  # chunk 2 exceeded threshold


def test_totals_handle_no_wer_log_at_all():
    rows = build_report_rows(TIMELINE, {}, {}, wer_threshold=None)
    totals = compute_totals(rows, total_audio_seconds=13.0)
    assert totals["mean_wer"] is None
    assert totals["mean_wpm"] is None
    assert totals["total_retries"] == 0
    assert totals["gate_failures"] == 0


def test_audio_flags_pulled_through_but_not_gate_failures():
    wer_by_id = {1: {"wer": 0.05, "attempts": 1, "sec_per_word": 0.35,
                      "duration_ok": True, "duration_reason": "ok",
                      "audio_flags": ["flat_pitch", "clipping"]}}
    rows = build_report_rows(TIMELINE, wer_by_id, {}, wer_threshold=0.15)
    r1 = next(r for r in rows if r["id"] == 1)
    assert r1["audio_flags"] == ["flat_pitch", "clipping"]
    assert r1["warnings"] == []  # advisory stays out of the hard-gate column
    totals = compute_totals(rows, total_audio_seconds=13.0)
    assert totals["gate_failures"] == 0
    assert totals["acoustic_advisories"] == 1


def test_loudness_outlier_flagged_from_timeline_rms():
    timeline = [{"id": i, "duration": 4.0, "rms_dbfs": v} for i, v in
                [(1, -20.0), (2, -20.4), (3, -26.5), (4, -19.9), (5, -20.1)]]
    rows = build_report_rows(timeline, {}, {}, wer_threshold=0.15)
    r3 = next(r for r in rows if r["id"] == 3)
    assert any(f.startswith("loudness_outlier") for f in r3["audio_flags"]), rows
    for r in rows:
        if r["id"] != 3:
            assert not any(f.startswith("loudness_outlier") for f in r["audio_flags"])


def test_no_rms_data_no_outlier_crash():
    rows = build_report_rows(TIMELINE, {}, {}, wer_threshold=0.15)
    assert all(r["audio_flags"] == [] for r in rows)


def test_format_report_txt_shows_advisory():
    wer_by_id = {2: {"wer": 0.05, "attempts": 1, "sec_per_word": 0.35,
                      "duration_ok": True, "duration_reason": "ok",
                      "audio_flags": ["abrupt_end"]}}
    rows = build_report_rows(TIMELINE, wer_by_id, {}, wer_threshold=0.15)
    totals = compute_totals(rows, total_audio_seconds=13.0)
    txt = format_report_txt("run01", "final.wav", rows, totals)
    assert "advisory: abrupt_end" in txt
    assert "acoustic advisories=1" in txt


def test_format_report_txt_lists_every_chunk_and_a_totals_line():
    rows = build_report_rows(TIMELINE, {}, {}, wer_threshold=0.15)
    totals = compute_totals(rows, total_audio_seconds=13.0)
    txt = format_report_txt("run01", "final.wav", rows, totals)
    for cid in (1, 2, 3):
        assert f"{cid:>4}" in txt or str(cid) in txt
    assert "Totals:" in txt
    assert "3 chunks" in txt


def main():
    tests = [
        test_row_pulls_wer_log_fields,
        test_missing_wer_log_entry_leaves_fields_none_not_crash,
        test_wer_over_threshold_produces_warning,
        test_duration_gate_failure_produces_warning,
        test_candidate_version_surfaced,
        test_totals_aggregate_correctly,
        test_totals_handle_no_wer_log_at_all,
        test_audio_flags_pulled_through_but_not_gate_failures,
        test_loudness_outlier_flagged_from_timeline_rms,
        test_no_rms_data_no_outlier_crash,
        test_format_report_txt_shows_advisory,
        test_format_report_txt_lists_every_chunk_and_a_totals_line,
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
