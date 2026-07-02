"""should_carry_over() — conditional prosody carry-over at rhetorical boundaries.

Used by 02_generate_nanovllm.py (both the batch loop and the interactive
regen path) to decide whether a chunk should inherit the previous chunk's
prosody tail. Carry-over should only flow WITHIN a rhetorical thought, not
across it -- a falling "final" cadence shouldn't bleed into the next
opening sentence.
"""


def should_carry_over(prev_chunk: dict | None, curr_chunk: dict) -> tuple[bool, str]:
    """
    Decide whether prev_chunk's prosody tail should flow into curr_chunk's
    generation. Suppressed when prev_chunk's position is "final" (a falling
    cadence shouldn't bleed into what follows) or curr_chunk's position is
    "opening" (a fresh thought shouldn't inherit the prior thought's
    prosody).

    prev_chunk's explicit "carryover_after" (bool), if present, overrides
    both position-derived checks -- a human forcing it in the plan wins.

    Returns (carry_ok, reason) so callers can log why.
    """
    if prev_chunk is None:
        return False, "no previous chunk"
    override = prev_chunk.get("carryover_after")
    if override is not None:
        return bool(override), f"carryover_after={override!r} (plan override)"
    if prev_chunk.get("position") == "final":
        return False, "prev final"
    if curr_chunk.get("position") == "opening":
        return False, "curr opening"
    return True, "default"
