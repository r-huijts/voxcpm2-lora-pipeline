"""SRT subtitle formatting for 03_stitch.py's timeline output.

Kept separate from the audio-manipulation code in 03_stitch.py so the pure
timecode/formatting logic is unit-testable without numpy/soundfile.
"""


def format_srt_timestamp(seconds: float) -> str:
    """Seconds -> SRT timecode HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_srt(entries: list) -> str:
    """
    entries: list of {"text": str, "start": float, "end": float}, in order.
    Returns the full .srt file content (standard numbered-cue format).
    """
    lines = []
    for i, e in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_timestamp(e['start'])} --> {format_srt_timestamp(e['end'])}")
        lines.append(e["text"] or "")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"
