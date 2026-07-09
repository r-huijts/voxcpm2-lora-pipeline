#!/usr/bin/env python3
"""
test_console.py — regression test for _console.py, the rich-based console
formatting for 02_generate_nanovllm.py.

Swaps in a Console(record=True) so assertions can inspect rendered plain
text without a real terminal. The main risk this guards against: rich's
markup parser treats "[...]" as a style tag, and both our own log prefixes
("[quality]", "[metrics]") and plan text (inline [non-verbal] tags) contain
literal brackets -- every _console.py function must render those as text,
never raise MarkupError.

Run after touching _console.py:
    python scripts/test_console.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console

import _console as c


def _record():
    """A fresh recording console, swapped into the module under test."""
    console = Console(record=True, width=200, highlight=False)
    c.console = console
    return console


def test_bracket_prefixes_render_as_literal_text_not_markup():
    console = _record()
    c.info("[quality] this must not crash on the brackets")
    c.warn("[metrics] advisory: flat_pitch")
    out = console.export_text()
    assert "[quality] this must not crash on the brackets" in out
    assert "[metrics] advisory: flat_pitch" in out


def test_plan_text_with_inline_tags_does_not_crash():
    # Real plan text can contain [stilte]-style non-verbal tags -- these
    # must render literally too, not be swallowed as (invalid) markup.
    console = _record()
    c.chunk_header(4, 29, True, "", "", "", "Dat was een [stilte] pauze.")
    out = console.export_text()
    assert "[stilte]" in out


def test_info_success_warn_error_dim_render_message():
    console = _record()
    c.info("info msg")
    c.success("success msg")
    c.warn("warn msg")
    c.error("error msg")
    c.dim("dim msg")
    out = console.export_text()
    for msg in ("info msg", "success msg", "warn msg", "error msg", "dim msg"):
        assert msg in out


def test_indent_is_applied():
    console = _record()
    c.info("indented", indent=c.INDENT)
    out = console.export_text()
    assert (c.INDENT + "indented") in out


def test_warn_err_goes_to_a_separate_stderr_console():
    console = _record()
    err_console = Console(record=True, width=200, highlight=False)
    c.err_console = err_console
    c.warn_err("stderr-routed warning")
    assert "stderr-routed warning" not in console.export_text()
    assert "stderr-routed warning" in err_console.export_text()


def test_chunk_header_shows_carry_and_text():
    console = _record()
    c.chunk_header(4, 29, False, "default", " ctrl='dry'", " reground=ref-only",
                   "Niet zomaar een truitje...")
    out = console.export_text()
    assert "[004/029]" in out
    assert "no" in out
    assert "default" in out
    assert "ctrl='dry'" in out
    assert "Niet zomaar een truitje..." in out


def test_quality_line_passed():
    console = _record()
    c.quality_line(1, "WER=5.0% 0.35s/word", passed=True, retrying=False)
    out = console.export_text()
    assert "attempt 1" in out
    assert "✓" in out


def test_quality_line_accepted_after_retry():
    console = _record()
    c.quality_line(2, "WER=5.0%", passed=True, retrying=False)
    out = console.export_text()
    assert "accepted" in out


def test_quality_line_retrying_and_exhausted():
    console = _record()
    c.quality_line(1, "WER=20.0%", passed=False, retrying=True, reasons="WER>15%")
    c.quality_line(3, "WER=20.0%", passed=False, retrying=False, reasons="WER>15%")
    out = console.export_text()
    assert "retrying..." in out
    assert "retries exhausted" in out


def test_wer_cell_thresholds():
    assert c.wer_cell(0.02).plain == "2.0%"
    assert c.wer_cell(0.10).plain == "10.0%"
    assert c.wer_cell(0.20).plain == "20.0%"
    assert c.wer_cell(None).plain == "n/a"
    assert c.wer_cell(-1.0).plain == "n/a"
    assert c.wer_cell(0.02).style == "green"
    assert c.wer_cell(0.10).style == "yellow"
    assert c.wer_cell(0.20).style == "red"


def test_duration_gate_cell():
    assert c.duration_gate_cell(True, "ok").plain == "ok"
    assert c.duration_gate_cell(False, "runaway").plain == "runaway"
    assert c.duration_gate_cell(False, None).plain == "n/a"


def test_flag_badges_empty_and_populated():
    assert c.flag_badges([]).plain == "—"
    t = c.flag_badges(["flat_pitch", "clipping"])
    assert "flat_pitch" in t.plain
    assert "clipping" in t.plain


def test_candidate_table_lists_every_version_ranked_first():
    console = _record()
    ranked = [
        {"version": 3, "wer": 0.05, "sec_per_word": 0.38, "duration_ok": True,
         "duration_reason": "ok", "flags": [], "metrics": {"duration_s": 7.2}},
        {"version": 2, "wer": 0.05, "sec_per_word": 0.35, "duration_ok": True,
         "duration_reason": "ok", "flags": [], "metrics": {"duration_s": 6.7}},
        {"version": 1, "wer": 0.10, "sec_per_word": 0.42, "duration_ok": True,
         "duration_reason": "ok", "flags": ["flat_pitch"], "metrics": {"duration_s": 7.9}},
    ]
    console.print(c.candidate_table(4, ranked))
    out = console.export_text()
    assert "v3" in out and "v2" in out and "v1" in out
    assert "flat_pitch" in out
    assert "chunk 0004" in out


def test_candidate_table_handles_missing_metrics():
    console = _record()
    ranked = [{"version": 1, "wer": None, "sec_per_word": None, "duration_ok": False,
               "duration_reason": "runaway", "flags": [], "metrics": None}]
    console.print(c.candidate_table(7, ranked))  # must not raise
    out = console.export_text()
    assert "v1" in out
    assert "n/a" in out


def test_candidate_take_line_with_flags():
    console = _record()
    c.candidate_take_line(3, 6.5, 0.0, 0.25, True, ["abrupt_start"], "chunk_0005_v3.wav")
    out = console.export_text()
    assert "v3:" in out
    assert "0.0%" in out
    assert "abrupt_start" in out
    assert "chunk_0005_v3.wav" in out


def test_metrics_advisory_line():
    console = _record()
    c.metrics_advisory_line(["flat_pitch", "loudness_outlier(-17.1dB)"])
    out = console.export_text()
    assert "flat_pitch" in out
    assert "loudness_outlier(-17.1dB)" in out
    assert "not a gate" in out


def test_command_table_lists_all_commands():
    console = _record()
    console.print(c.command_table())
    out = console.export_text()
    for cmd in ("<id>", "cand <id> <k>", "reload", "list", "quit"):
        assert cmd in out


def test_chunk_header_has_leading_blank_line_for_separation():
    console = _record()
    c.info("previous chunk's tail line")
    c.chunk_header(5, 29, True, "", "", "", "next chunk text")
    lines = console.export_text().splitlines()
    idx = next(i for i, l in enumerate(lines) if "[005/029]" in l)
    assert lines[idx - 1].strip() == "", "expected a blank line before the chunk header"


def test_rule_has_blank_lines_on_both_sides():
    console = _record()
    c.info("before")
    c.rule("Section")
    c.info("after")
    lines = console.export_text().splitlines()
    rule_idx = next(i for i, l in enumerate(lines) if "Section" in l)
    assert lines[rule_idx - 1].strip() == ""
    assert lines[rule_idx + 1].strip() == ""


def test_show_candidate_table_is_indented_and_framed():
    console = _record()
    ranked = [{"version": 1, "wer": 0.0, "sec_per_word": 0.3, "duration_ok": True,
               "duration_reason": "ok", "flags": [], "metrics": {"duration_s": 5.0}}]
    c.info("prior line")
    c.show_candidate_table(9, ranked)
    c.info("next line")
    out = console.export_text()
    lines = out.splitlines()
    assert any(l.strip() == "" for l in lines)  # framed by blank lines
    table_lines = [l for l in lines if "v1" in l]
    assert table_lines and table_lines[0].startswith("  "), "expected a left margin"


def test_show_command_table_has_left_margin():
    console = _record()
    c.show_command_table()
    out = console.export_text()
    line = next(l for l in out.splitlines() if "<id>" in l)
    assert line.startswith("  ")


def main():
    tests = [
        test_bracket_prefixes_render_as_literal_text_not_markup,
        test_plan_text_with_inline_tags_does_not_crash,
        test_info_success_warn_error_dim_render_message,
        test_indent_is_applied,
        test_warn_err_goes_to_a_separate_stderr_console,
        test_chunk_header_shows_carry_and_text,
        test_quality_line_passed,
        test_quality_line_accepted_after_retry,
        test_quality_line_retrying_and_exhausted,
        test_wer_cell_thresholds,
        test_duration_gate_cell,
        test_flag_badges_empty_and_populated,
        test_candidate_table_lists_every_version_ranked_first,
        test_candidate_table_handles_missing_metrics,
        test_candidate_take_line_with_flags,
        test_metrics_advisory_line,
        test_command_table_lists_all_commands,
        test_chunk_header_has_leading_blank_line_for_separation,
        test_rule_has_blank_lines_on_both_sides,
        test_show_candidate_table_is_indented_and_framed,
        test_show_command_table_has_left_margin,
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
