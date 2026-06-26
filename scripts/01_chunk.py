#!/usr/bin/env python3
"""
01_chunk.py — Turn a full Dutch column into a reviewable chunk plan (JSON).

An LLM (via Portkey) reads the WHOLE column first, decides the overall delivery
register, then splits it into "delivery units" — spans spoken as one continuous
breath. For each chunk it returns:
  - text:    the chunk, with numbers expanded and hard names respelled phonetically
  - control: a per-chunk style/pace tag, chosen against the whole arc
  - gap_after: "short" (within paragraph) | "long" (between paragraphs) | "none" (last)

The output JSON is meant to be EDITED by hand before generation. Nothing is
final until you've read it.

Usage:
    export PORTKEY_API_KEY=...                  # or pass --api-key
    python 01_chunk.py --input column.txt --output plan.json
    python 01_chunk.py --input column.txt --output plan.json \
        --model gpt-4o --config-id pc-xxxx

Requires: portkey_ai
    pip install portkey-ai
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import pysbd
from portkey_ai import Portkey


def split_sentences(column: str) -> list[dict]:
    """
    Deterministically split the column into sentences with pySBD (Dutch),
    preserving paragraph structure. Returns a list of
    {"para": int, "sent": int, "text": str} so the LLM receives clean,
    pre-numbered sentences and only has to GROUP them — not find boundaries
    (which LLMs occasionally botch on abbreviations/numbers).
    """
    seg = pysbd.Segmenter(language="nl", clean=False)
    column = column.replace("\r\n", "\n").strip()
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", column) if p.strip()]

    rows = []
    for pi, para in enumerate(paragraphs, start=1):
        for si, sent in enumerate(seg.segment(para), start=1):
            s = sent.strip()
            if s:
                rows.append({"para": pi, "sent": si, "text": s})
    return rows


def format_sentences_for_llm(rows: list[dict]) -> str:
    """Render the numbered sentence list the LLM will group."""
    lines = []
    cur_para = None
    for r in rows:
        if r["para"] != cur_para:
            cur_para = r["para"]
            lines.append(f"\n[ALINEA {cur_para}]")
        lines.append(f"  P{r['para']}S{r['sent']}: {r['text']}")
    return "\n".join(lines).strip()


SYSTEM_PROMPT = """\
Je bent een audioregisseur die teksten voorbereidt voor tekst-naar-spraak \
synthese met een Nederlandse stem (stijl: Mart Smeets, droge wielercommentaar).

Je krijgt een column die al is opgesplitst in genummerde zinnen (P<alinea>S<zin>). \
De zinsgrenzen staan VAST — die hoef je niet te bepalen. Jouw taak is de zinnen \
GROEPEREN tot "delivery units" en voorbereiden voor uitspraak.

Doe het volgende, in deze volgorde:

1. LEES ALLE ZINNEN en bepaal het overkoepelende register (toon, tempo, ironie). \
Dit bepaalt de basis voor alle fragmenten.

2. GROEPEER de genummerde zinnen tot fragmenten — spans die als één doorlopende \
ademhaling worden uitgesproken. Regels:
   - Een fragment bevat één of meer HELE zinnen. Splits nooit binnen een zin.
   - Korte zinnen die bij elkaar horen (een opbouw + clou, een opsomming) \
     groepeer je SAMEN in één fragment.
   - Een lange zin mag een eigen fragment zijn.
   - Een losse, korte volzin mag een eigen kort fragment zijn — kort is prima \
     zolang het een hele gedachte is.
   - Gebruik je oor: waar zou een verteller ademhalen? Dat is de grens.

3. Per fragment: kies een control-tag (Nederlandse stijl/tempo-instructie) die \
past bij DAT moment binnen de hele boog. Bijvoorbeeld: "rustig, uitnodigend", \
"droog, met lichte spot", "trager, nadrukkelijk", "zakelijk opsommend".

4. NORMALISEER de tekst van elk fragment voor uitspraak:
   - Schrijf getallen volledig uit in het Nederlands. Komma-decimalen: \
     "271,7" -> "tweehonderdeenenzeventig komma zeven". "7,3" -> "zeven komma \
     drie". "65,7" -> "vijfenzestig komma zeven".
   - Herspel lastige eigennamen fonetisch naar Nederlandse uitspraak. \
     Voorbeelden: "Pogačar" -> "Pogatsjar", "Narváez" -> "Narwa-es", \
     "Carapaz" -> "Karapas". Verzonnen teamnamen ("Ruudissimo-Colnago", \
     "Deux-Roubaix", "Flandrién Koenigsberg Elite") schrijf je zoals ze \
     uitgesproken moeten worden.
   - Laat gewone Nederlandse woorden ongemoeid.

5. gap_after per fragment:
   - "short" als het volgende fragment in DEZELFDE alinea staat
   - "long" als er een alinea-grens volgt (de P-nummers veranderen)
   - "none" voor het allerlaatste fragment

Geef per fragment ook "sentences" terug: de lijst van zin-IDs (zoals "P1S1") die \
je hebt samengevoegd, zodat we kunnen controleren dat alle zinnen precies één \
keer zijn gebruikt.

Geef UITSLUITEND geldige JSON terug, zonder uitleg, zonder markdown:
{
  "register": "<korte beschrijving van het overkoepelende register>",
  "chunks": [
    {"id": 1, "sentences": ["P1S1","P1S2"], "text": "...", \
"control": "...", "gap_after": "short"},
    ...
  ]
}
"""


def build_client(api_key: str, config_id: str | None) -> Portkey:
    kwargs = {"api_key": api_key}
    if config_id:
        kwargs["config"] = config_id
    return Portkey(**kwargs)


def call_llm(client: Portkey, model: str, sentences_block: str) -> str:
    """Single chat completion; returns raw assistant text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sentences_block},
    ]
    # max_tokens for most providers; some newer OpenAI models need
    # max_completion_tokens. Try both, mirroring the reference tester.
    for limit in ({"max_tokens": 8000}, {"max_completion_tokens": 8000}):
        try:
            resp = client.chat.completions.create(
                messages=messages, model=model, **limit
            )
            break
        except Exception as e:
            msg = str(e).lower()
            retryable = any(
                n in msg for n in (
                    "max_tokens", "max_completion_tokens", "unsupported",
                    "unknown parameter", "extra_forbidden",
                )
            )
            if not retryable:
                raise
    else:
        raise RuntimeError("Both token-limit parameters were rejected.")

    return resp.choices[0].message.content


def parse_plan(raw: str) -> dict:
    """Extract JSON from the model output, tolerating stray fences/prose."""
    text = raw.strip()
    # Strip markdown fences if the model added them despite instructions.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # If there's leading/trailing prose, grab the outermost JSON object.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    return json.loads(text)


def validate_plan(plan: dict, expected_sentence_ids: set[str] | None = None) -> list[str]:
    """Return a list of warnings (empty if clean). Non-fatal sanity checks."""
    warnings = []
    chunks = plan.get("chunks", [])
    if not chunks:
        warnings.append("No chunks returned.")
        return warnings
    valid_gaps = {"short", "long", "none"}
    for i, c in enumerate(chunks):
        cid = c.get("id", i + 1)
        if not c.get("text", "").strip():
            warnings.append(f"Chunk {cid}: empty text.")
        if not c.get("control", "").strip():
            warnings.append(f"Chunk {cid}: missing control tag.")
        if c.get("gap_after") not in valid_gaps:
            warnings.append(f"Chunk {cid}: gap_after={c.get('gap_after')!r} "
                            f"(expected one of {valid_gaps}).")
    if chunks[-1].get("gap_after") != "none":
        warnings.append("Last chunk's gap_after should be 'none'.")
    # Flag any digits that survived normalization.
    for c in chunks:
        if any(ch.isdigit() for ch in c.get("text", "")):
            warnings.append(f"Chunk {c.get('id')}: still contains digits — "
                            f"check number expansion.")
    # Coverage: every pySBD sentence used exactly once.
    if expected_sentence_ids is not None:
        used = []
        for c in chunks:
            used.extend(c.get("sentences", []))
        used_set = set(used)
        missing = expected_sentence_ids - used_set
        extra = used_set - expected_sentence_ids
        dupes = {s for s in used if used.count(s) > 1}
        if missing:
            warnings.append(f"Sentences dropped (not in any chunk): "
                            f"{sorted(missing)}")
        if extra:
            warnings.append(f"Unknown sentence IDs in chunks: {sorted(extra)}")
        if dupes:
            warnings.append(f"Sentences used more than once: {sorted(dupes)}")
    return warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Column .txt")
    ap.add_argument("--output", required=True, type=Path, help="Plan .json")
    ap.add_argument("--model", default="gpt-4o",
                    help="Model slug as configured in Portkey.")
    ap.add_argument("--config-id", default=None,
                    help="Optional Portkey config ID.")
    ap.add_argument("--api-key", default=os.environ.get("PORTKEY_API_KEY"),
                    help="Portkey API key (or set PORTKEY_API_KEY).")
    ap.add_argument("--short-pause-ms", type=int, default=220)
    ap.add_argument("--long-pause-ms", type=int, default=550)
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("No Portkey API key. Pass --api-key or set PORTKEY_API_KEY.")
    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    column = args.input.read_text(encoding="utf-8").strip()
    if not column:
        sys.exit("Input file is empty.")

    # Deterministic sentence split first (pySBD, Dutch) — the LLM groups these.
    rows = split_sentences(column)
    if not rows:
        sys.exit("No sentences found after splitting.")
    expected_ids = {f"P{r['para']}S{r['sent']}" for r in rows}
    sentences_block = format_sentences_for_llm(rows)
    print(f"Split into {len(rows)} sentences across "
          f"{rows[-1]['para']} paragraphs (pySBD).")

    print(f"Grouping into delivery units via {args.model}...")
    client = build_client(args.api_key, args.config_id)
    raw = call_llm(client, args.model, sentences_block)

    try:
        plan = parse_plan(raw)
    except json.JSONDecodeError as e:
        # Save the raw output so nothing is lost when parsing fails.
        dump = args.output.with_suffix(".raw.txt")
        dump.write_text(raw, encoding="utf-8")
        sys.exit(f"Could not parse JSON: {e}\nRaw model output saved to {dump}")

    # Inject pause config (kept with the plan so the stitcher is self-contained).
    plan.setdefault("config", {})
    plan["config"]["short_pause_ms"] = args.short_pause_ms
    plan["config"]["long_pause_ms"] = args.long_pause_ms

    warnings = validate_plan(plan, expected_sentence_ids=expected_ids)

    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n = len(plan.get("chunks", []))
    print(f"\nWrote {n} chunks to {args.output}")
    print(f"Register: {plan.get('register', '(none)')}")
    if warnings:
        print("\nWarnings (review before generating):")
        for w in warnings:
            print(f"  - {w}")
    print(f"\nReview and edit {args.output}, then run 02_generate.py.")


if __name__ == "__main__":
    main()
