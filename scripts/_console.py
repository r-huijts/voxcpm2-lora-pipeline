"""
_console.py — rich-based console output for 02_generate_nanovllm.py.

The generation log is dozens of print() call sites spread across the batch
loop, --candidates mode, and --interactive mode, all plain text. This
centralizes styling so it reads as one system instead of ad hoc prefixes:
color communicates severity (info/success/warn/error) at a glance, and a
real Table replaces --candidates mode's five-line-per-take dump with a
sortable comparison.

Pure rich + stdlib -- no torch/nano-vllm dependency -- so it's testable
locally like the rest of this repo's helper modules (Console(record=True)
+ export_text() lets a test assert on rendered content without a real tty).

markup=False everywhere a message might embed literal square brackets: plan
text legitimately contains inline [non-verbal] tags (see _TAG_RE in
02_generate_nanovllm.py), and our own log lines use bracket prefixes like
"[quality]" -- both would crash rich's markup parser if it tried to
interpret them as style tags.
"""
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

INDENT = " " * 9   # matches the original per-chunk sub-line indent
INDENT2 = " " * 2  # interactive-mode command indent


def info(msg: str, indent: str = "") -> None:
    console.print(indent + msg, style="cyan", markup=False)


def success(msg: str, indent: str = "") -> None:
    console.print(indent + msg, style="green", markup=False)


def warn(msg: str, indent: str = "") -> None:
    console.print(indent + msg, style="yellow", markup=False)


def warn_err(msg: str, indent: str = "") -> None:
    """Same as warn(), but to stderr -- for the handful of warnings that
    were already routed to stderr (e.g. a fallback path being taken)."""
    err_console.print(indent + msg, style="yellow", markup=False)


def error(msg: str, indent: str = "") -> None:
    console.print(indent + msg, style="bold red", markup=False)


def dim(msg: str, indent: str = "") -> None:
    console.print(indent + msg, style="dim", markup=False)


def plain(msg: str, indent: str = "") -> None:
    console.print(indent + msg, markup=False)


def blank() -> None:
    """One blank output line -- named so call sites read as intentional
    paragraph breaks, not stray print()s."""
    console.print()


def rule(title: str = "") -> None:
    """A section divider with breathing room on both sides -- otherwise the
    line right above and right below a rule sit flush against it."""
    blank()
    console.rule(title, style="cyan")
    blank()


def chunk_header(cid: int, n_total: int, carry_ok: bool, carry_note: str,
                  ctrl_str: str, reground_tag: str, text_preview: str) -> None:
    """
    Leads with a blank line: with 20-30+ chunks per run, back-to-back chunk
    blocks with zero vertical separation read as one solid wall of text.
    One blank line between chunks (and, via the leading blank, between the
    last chunk's tail and the next chunk's header) is enough to let the eye
    find chunk boundaries without inflating the log length much.
    """
    blank()
    t = Text()
    t.append(f"[{cid:03d}/{n_total:03d}] ", style="bold cyan")
    t.append("carry=", style="dim")
    t.append("yes" if carry_ok else "no", style="green" if carry_ok else "yellow")
    if carry_note:
        t.append(f" ({carry_note})", style="dim")
    if ctrl_str:
        t.append(ctrl_str, style="magenta")
    if reground_tag:
        t.append(reground_tag, style="dim")
    t.append(" | ")
    t.append(text_preview)
    console.print(t)


def quality_line(attempt: int, status: str, passed: bool, retrying: bool,
                  reasons: str = "", indent: str = INDENT) -> None:
    """One generate_with_retry() attempt's result: WER/duration status plus
    a pass/retry/exhausted verdict, color-coded."""
    t = Text(indent)
    t.append("[quality] ", style="dim")
    t.append(f"attempt {attempt}: ", style="bold")
    t.append(status)
    if passed:
        t.append("  ✓" + (" accepted" if attempt > 1 else ""), style="bold green")
    elif retrying:
        t.append(f"  — {reasons} — retrying...", style="yellow")
    else:
        t.append(f"  — {reasons} — retries exhausted, keeping best", style="bold red")
    console.print(t)


def duration_note(msg: str, severe: bool = False, indent: str = INDENT) -> None:
    t = Text(indent)
    t.append("[duration] ", style="dim")
    t.append(msg, style="bold red" if severe else "yellow")
    console.print(t)


def metrics_warning(msg: str, indent: str = INDENT) -> None:
    t = Text(indent)
    t.append("[metrics] ", style="dim")
    t.append(msg, style="yellow")
    console.print(t)


def metrics_advisory_line(flags: list[str], indent: str = INDENT) -> None:
    t = Text(indent)
    t.append("[metrics] ", style="dim")
    t.append("advisory: ", style="dim")
    t.append(flag_badges(flags))
    t.append(" (triage hint, not a gate -- worth a listen)", style="dim")
    console.print(t)


def flag_badges(flags: list[str]) -> Text:
    """Inline colored badges for advisory flags, for a line or table cell."""
    if not flags:
        return Text("—", style="dim")
    t = Text()
    for i, f in enumerate(flags):
        if i:
            t.append(", ")
        t.append(f, style="yellow")
    return t


def wer_cell(wer) -> Text:
    if wer is None or (isinstance(wer, (int, float)) and wer < 0):
        return Text("n/a", style="dim")
    pct = wer * 100
    style = "green" if pct <= 5 else ("yellow" if pct <= 15 else "red")
    return Text(f"{pct:.1f}%", style=style)


def duration_gate_cell(ok, reason) -> Text:
    if ok:
        return Text("ok", style="green")
    return Text(reason or "n/a", style="bold red")


def candidate_take_line(version: int, duration_s: float, wer, spw, show_pace: bool,
                        flags: list[str], filename: str, indent: str = INDENT) -> None:
    """Live per-take line printed as each candidate finishes generating --
    the full ranked comparison follows in candidate_table() once all are done."""
    t = Text(indent)
    t.append(f"v{version}: ", style="bold")
    t.append(f"{duration_s:.1f}s ")
    t.append(wer_cell(wer))
    if show_pace and spw:
        t.append(f" {spw:.2f}s/word")
    if flags:
        t.append("  [")
        t.append(flag_badges(flags))
        t.append("]")
    t.append(f" -> {filename}", style="dim")
    console.print(t)


def candidate_table(cid: int, ranked: list[dict]) -> Table:
    """
    One row per candidate take, in the suggested-listening-order (best
    first, per _audio_metrics.rank_candidates): duration gate, then fewest
    advisory flags, then WER, then liveliest pitch contour. A pre-listening
    filter for where to start, never a verdict -- the top row is bolded,
    not auto-selected.
    """
    table = Table(title=f"chunk {cid:04d} — {len(ranked)} candidate take(s), "
                        f"best listening order first",
                  header_style="bold cyan", show_lines=True, padding=(0, 2),
                  title_style="bold")
    table.add_column("#", justify="right")
    table.add_column("take", justify="center")
    table.add_column("duration", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("pace (s/word)", justify="right")
    table.add_column("duration gate", justify="center")
    table.add_column("advisory flags")
    for rank, e in enumerate(ranked, start=1):
        dur = (e.get("metrics") or {}).get("duration_s")
        dur_str = f"{dur:.1f}s" if dur is not None else "n/a"
        spw = e.get("sec_per_word")
        spw_str = f"{spw:.2f}" if spw is not None else "n/a"
        table.add_row(
            str(rank), f"v{e['version']}", dur_str,
            wer_cell(e.get("wer")), spw_str,
            duration_gate_cell(e.get("duration_ok"), e.get("duration_reason")),
            flag_badges(e.get("flags") or []),
            style="bold" if rank == 1 else None,
        )
    return table


def show_candidate_table(cid: int, ranked: list[dict]) -> None:
    """candidate_table(), framed with blank lines and a left margin so it
    doesn't sit flush against the per-take lines above it."""
    blank()
    console.print(Padding(candidate_table(cid, ranked), (0, 0, 0, 2)))
    blank()


def command_table() -> Table:
    """Interactive-mode command reference, in place of six plain print()s."""
    table = Table(header_style="bold cyan", show_lines=False, box=None,
                  padding=(0, 3, 1, 0))
    table.add_column("command", style="bold")
    table.add_column("does")
    rows = [
        ("<id>", "regenerate that chunk"),
        ("<id> --cfg 1.7 --temp 0.9", "regenerate with overrides"),
        ("cand <id> <k>", "generate k candidate versions (_v1.._vk)"),
        ("cand <id> <k> --cfg 1.7 --temp 0.9", "candidates with settings"),
        ("reload", "re-read plan.json (after lexicon/plan edits)"),
        ("list", "show chunk ids + text starts"),
        ("quit", "exit"),
    ]
    for cmd, desc in rows:
        table.add_row(cmd, desc)
    return table


def show_command_table() -> None:
    """command_table(), with a left margin matching show_candidate_table()."""
    console.print(Padding(command_table(), (0, 0, 1, 2)))
