"""Field-fallback helpers for plan.json / manifest.json chunk objects.

Chunks may be old-format (only `text`, written directly by the LLM) or
new-format (`source_text` + `spoken_text`, script-built -- see 01_chunk.py's
module docstring for why the LLM stopped writing spoken text directly).
Every downstream consumer must go through these helpers instead of assuming
either field exists, so old runs/plans keep working unchanged.
"""


def resolve_spoken_text(chunk: dict) -> str:
    """The text that should actually be generated/spoken."""
    return chunk.get("spoken_text") or chunk.get("text") or chunk.get("source_text") or ""


def resolve_source_text(chunk: dict) -> str:
    """The original wording for audit/reporting/SRT -- never lexicon- or
    normalization-altered."""
    return chunk.get("source_text") or chunk.get("text") or chunk.get("spoken_text") or ""
