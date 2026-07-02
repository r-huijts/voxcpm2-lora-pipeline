"""run_report aggregation for 03_stitch.py (Sub-task 5b).

Aggregates data 03_stitch.py already has in hand -- timeline placement
(Task 5), wer_log.json's per-chunk WER/duration-gate metrics (Task 2), and
the candidate selection map -- into one glanceable QA sheet. Pure
aggregation of already-computed fields, not new measurement.
"""
import statistics


def build_report_rows(timeline_entries: list, wer_by_id: dict, selection: dict,
                       wer_threshold) -> list:
    """
    timeline_entries: [{"id", "duration", ...}, ...] from the stitch timeline.
    wer_by_id: chunk id -> wer_log.json chunk entry (wer/attempts/sec_per_word/
      duration_ok/duration_reason), or {} if wer_log.json wasn't found.
    selection: chunk id -> chosen candidate version (int), or {}.
    wer_threshold: the run's WER threshold (for the "exceeded" warning), or
      None if unknown (no warning is synthesized in that case).
    """
    rows = []
    for e in timeline_entries:
        cid = e["id"]
        w = wer_by_id.get(cid, {})
        wer = w.get("wer")
        attempts = w.get("attempts")
        retries = (attempts - 1) if isinstance(attempts, int) else None
        sec_per_word = w.get("sec_per_word")
        duration_ok = w.get("duration_ok")
        duration_reason = w.get("duration_reason")
        wpm = (60.0 / sec_per_word) if sec_per_word else None

        warnings = []
        if wer is not None and wer_threshold is not None and wer > wer_threshold:
            warnings.append(f"WER {wer * 100:.1f}% > threshold {wer_threshold * 100:.0f}%")
        if duration_ok is False:
            warnings.append("RUNAWAY (hit max length, no clean stop)"
                             if duration_reason == "runaway" else "too short/rushed")

        rows.append({
            "id": cid,
            "wer": wer,
            "wpm": wpm,
            "sec_per_word": sec_per_word,
            "duration": e.get("duration"),
            "retries": retries,
            "duration_ok": duration_ok,
            "duration_reason": duration_reason,
            "candidate_version": selection.get(cid),
            "warnings": warnings,
        })
    return rows


def compute_totals(rows: list, total_audio_seconds: float) -> dict:
    wers = [r["wer"] for r in rows if r["wer"] is not None]
    wpms = [r["wpm"] for r in rows if r["wpm"] is not None]
    retries = [r["retries"] for r in rows if r["retries"] is not None]
    return {
        "chunk_count": len(rows),
        "total_audio_seconds": round(total_audio_seconds, 2),
        "mean_wpm": round(statistics.mean(wpms), 1) if wpms else None,
        "median_wpm": round(statistics.median(wpms), 1) if wpms else None,
        "mean_wer": round(statistics.mean(wers), 4) if wers else None,
        "total_retries": sum(retries) if retries else 0,
        "gate_failures": sum(1 for r in rows if r["warnings"]),
    }


def format_report_txt(run_dir_label: str, output_label: str, rows: list, totals: dict) -> str:
    lines = [f"Run report — {run_dir_label}", f"Output: {output_label}", ""]
    header = (f"{'id':>4}  {'WER':>7}  {'WPM':>6}  {'dur(s)':>7}  "
              f"{'retries':>7}  {'dur_gate':>10}  {'cand':>5}  warnings")
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        wer_str = f"{r['wer'] * 100:.1f}%" if r["wer"] is not None else "n/a"
        wpm_str = f"{r['wpm']:.0f}" if r["wpm"] is not None else "n/a"
        dur_str = f"{r['duration']:.1f}" if r["duration"] is not None else "n/a"
        retries_str = str(r["retries"]) if r["retries"] is not None else "n/a"
        gate_str = "ok" if r["duration_ok"] else (r["duration_reason"] or "n/a")
        cand_str = str(r["candidate_version"]) if r["candidate_version"] is not None else "-"
        warn_str = "; ".join(r["warnings"])
        lines.append(f"{r['id']:>4}  {wer_str:>7}  {wpm_str:>6}  {dur_str:>7}  "
                      f"{retries_str:>7}  {gate_str:>10}  {cand_str:>5}  {warn_str}")
    lines.append("-" * len(header))
    mean_wer_str = f"{totals['mean_wer'] * 100:.1f}%" if totals["mean_wer"] is not None else "n/a"
    lines.append(
        f"Totals: {totals['chunk_count']} chunks | {totals['total_audio_seconds']}s audio | "
        f"mean WPM={totals['mean_wpm']} | median WPM={totals['median_wpm']} | "
        f"mean WER={mean_wer_str} | total retries={totals['total_retries']} | "
        f"gate failures={totals['gate_failures']}"
    )
    return "\n".join(lines) + "\n"
