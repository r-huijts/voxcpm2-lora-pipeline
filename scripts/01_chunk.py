#!/usr/bin/env python3
"""
01_chunk.py — Turn a full Dutch column into a reviewable chunk plan (JSON).

An LLM (via Portkey) reads the WHOLE column first, decides the overall delivery
register, then splits it into "delivery units" — spans spoken as one continuous
breath. For each chunk it returns:
  - text:    the chunk, with numbers expanded and (sparingly) non-verbal tags
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


def _load_dotenv():
    """
    Load KEY=VALUE lines from a .env file (repo root, then narrate/) into the
    environment, without overwriting anything already set. Lets you keep
    PORTKEY_API_KEY in a gitignored .env instead of pasting it each run.
    """
    here = Path(__file__).resolve().parent
    candidates = [here.parent / ".env", here / ".env"]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            # Strip optional surrounding quotes.
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), val)


_load_dotenv()


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

Je bent een audioregisseur die een Nederlandse column voorbereidt voor \
tekst-naar-spraak synthese met een gekloonde stem \
(stijl: Mart Smeets — droog, zakelijk wielercommentaar; weinig pathos, \
veel precisie, ironie zit in de timing).

Je krijgt de column als een lijst van genummerde zinnen (P<alinea>S<zin>). \
De zinsgrenzen staan VAST. Jouw taak is de zinnen GROEPEREN tot \
"delivery units" en de tekst voorbereiden voor uitspraak.

════════════════════════════════════════════════════════
STAP 1 — LEES HET GEHEEL
════════════════════════════════════════════════════════
Lees alle zinnen en stel vast:
  - Het overkoepelende register: toon, ironie-niveau, spreektempo.
  - De retorische structuur: waar zitten de clous, de opbouwen, de \
dramatische wendingen?

Dit register is de baseline voor de hele column. Individuele fragmenten \
mogen er tijdelijk van afwijken, maar keren er altijd naar terug.

════════════════════════════════════════════════════════
STAP 2 — GROEPEER tot delivery units
════════════════════════════════════════════════════════
Een delivery unit = een span die als één doorlopende ademhaling wordt \
uitgesproken. Regels:

  a. Gebruik ALLEEN hele zinnen. Splits nooit binnen een zin.

  b. Groepeer zinnen die samen één gedachte of retorische beweging vormen \
     (opbouw + clou, vraag + antwoord, opsomming, tegenstelling).

  c. Een lange zin mag een eigen fragment zijn. Een korte, volledige \
     gedachte ook — maar zie regel (d).

  d. MINIMUMLENGTE — isoleer nooit een fragment van minder dan ~6 woorden. \
     De TTS-stem heeft minstens ~1,5 seconde spraak nodig om te \
     stabiliseren; een te kort fragment klinkt vervormd of instabiel. \
     Voeg korte zinnetjes (begroetingen, antwoorden, afsluiters) altijd \
     samen met de aangrenzende zin. Dramatische pauzes creëer je met \
     gap_after_ms, niet met losse mini-fragmenten.

  e. Let op retorische staccato: reeksen van korte zinnen die samen één \
     sfeer neerzetten (bv. "Rome. De Eeuwige Stad. Je loopt er rond…") \
     horen bij elkaar in één fragment — ook al zijn het meerdere zinnen.

  f. Gebruik je oor: waar zou een verteller ademhalen? Dáár is de grens.

════════════════════════════════════════════════════════
STAP 3 — POSITIE in de gedachtegang
════════════════════════════════════════════════════════
Elke fragment krijgt één positie:

  "opening"    — begin van een nieuwe gedachte of alinea
  "continuing" — midden in een doorlopende gedachte; leunt vooruit naar \
                 het volgende fragment en mag NIET volledig afsluiten
  "final"      — einde van een gedachte of alinea; de stem mag dalen en \
                 afsluiten

De positie beschrijft de retorische functie, niet de interpunctie. \
Een fragment dat op een punt eindigt kan "continuing" zijn als de gedachte \
in het volgende fragment doorgaat.

════════════════════════════════════════════════════════
STAP 4 — CONTROL INSTRUCTION
════════════════════════════════════════════════════════
Schrijf per fragment een korte Engelstalige control instruction.
Dit is een directe aanwijzing aan de TTS-stem — een compacte technische
cue, geen beschrijving voor een menselijke lezer.

Regels:
  - Maximaal acht woorden. Kommalijst van eigenschappen, geen zinnen.
  - Gebruik uitsluitend technische leveringsbeschrijvingen: tempo, energie,
    volume, toon. Voorbeelden van goede woorden: dry, measured, brisk,
    unhurried, slow, light, heavy, forward, settled, composed, clipped.
  - Geen dramatische of interpretatieve instructies zoals "ironic", "wry",
    "deadpan", "conspiratorial", "climax", "weight", "finality", "lands
    harder". De ironie zit in de tekst en de timing — niet in de stem die
    opdracht krijgt het te spelen. De TTS voert dramatische instructies
    letterlijk en overdreven uit.
  - De "position" gebruik je alleen als redeneersteiger om de juiste toon
    te kiezen — het woord "continuing", "final" of "opening" verschijnt
    NOOIT in de control instruction zelf.
      • "continuing": kies energie die past bij een gedachte die nog
        loopt (bv. "dry, brisk", "measured, forward")
      • "final": kies rust en gewicht (bv. "slow, dry, settled",
        "measured, heavy")
      • "opening": licht en open (bv. "measured, dry", "light, brisk")

Goed: "dry, measured, deliberate"
Fout: "Measured and slightly wry; delivered with understated irony —
no falling tone at the end."

Kijk bij het toewijzen van control instructions naar de semantische
samenhang tussen opeenvolgende fragmenten. Fragmenten die samen één
gedachte vormen krijgen tags die prosodisch op elkaar aansluiten —
vergelijkbaar tempo, vergelijkbare energie. Zo ontstaat een natuurlijke
beweging binnen elke gedachtegang: opbouw, draag, afsluiting.

Gebruik tempo als het voornaamste verbindingsmiddel:
  - Aaneengesloten fragmenten binnen één gedachte: consistent tempo,
    geen plotse versnelling of vertraging tussen hen.
  - Het sluitende fragment van een gedachte: iets langzamer en zwaarder
    dan de fragmenten ervoor.
  - Het openingsfragment van een nieuwe gedachte: iets lichter en opener
    dan het sluitende fragment ervoor.

Lees tot slot alle control instructions als reeks terug. Ze moeten samen
een coherente boog vormen over de column — tempo en register verschuiven
geleidelijk en doelbewust. De reeks tags is het pacing-script voor het
geheel.
Het allerlaatste fragment krijgt altijd een control instruction die
afsluiting en rust uitdrukt (bv. "slow, dry, settled, heavy").
Geen uitzonderingen.

════════════════════════════════════════════════════════
STAP 5 — NON-VERBALE TAGS (zeer spaarzaam)
════════════════════════════════════════════════════════
Je mag — uitsluitend waar de tekst het echt verdient — een non-verbale tag
inline in de fragmenttekst plaatsen. De TTS-stem zet deze tags om in een
hoorbaar, niet-talig geluid (een ademhaling, een korte lach, een zucht).

SYNTAX:
  - Engelstalige tag, tussen rechte haken, kleine letters: [zucht] wordt
    NIET gebruikt — gebruik de Engelse vorm. Toegestane tags:
        [sigh]        — een korte, droge zucht
        [breath]      — een hoorbare ademhaling vóór een nieuwe gedachte
        [laugh]       — een korte, ingehouden lach (zelden)
        [exhale]      — een uitademing, berusting
  - Plaats de tag exact op de positie in de tekst waar het geluid hoort,
    niet aan het begin van het fragment als een soort label.
        Goed:  "Hij won. [exhale] Natuurlijk won hij."
        Fout:  "[sigh] Hij won opnieuw zonder enige tegenstand."
        (de tweede plaatst de tag mechanisch vooraan; dat klinkt onecht)

IJZEREN REGELS — overtreed deze nooit:
  1. MAXIMAAL één tag per fragment. Liever geen.
  2. De meeste fragmenten krijgen GEEN tag. Een hele column met drie of
     vier tags in totaal is ruim voldoende. Tags zijn een schaars
     kruidmiddel, geen vaste ingrediënt.
  3. Gebruik een tag alleen als het non-verbale geluid betekenis draagt:
     een droge zucht ná een voorspelbare overwinning, een ademhaling vóór
     een wending. Nooit ter decoratie.
  4. Stapel nooit tags ([sigh][breath]) en zet nooit twee tags in één zin.
  5. Bij twijfel: GEEN tag. De ironie zit in de woorden en de timing; het
     non-verbale geluid is slechts een zeldzame, welbewuste onderstreping.
  6. Kleine letters, exact zoals hierboven gespeld. Geen varianten als
     [Sigh], [laughter], [sighs].

Plaats de tags terwijl je STAP 5 (normalisatie) uitvoert, in dezelfde
fragmenttekst. De tag telt niet mee als "los cijfer" of als naam — het is
gewoon onderdeel van de uitspreektekst.

════════════════════════════════════════════════════════
STAP 6 — NORMALISEER de tekst voor uitspraak
════════════════════════════════════════════════════════
Schrijf de tekst van elk fragment uitspreekvriendelijk:

  GETALLEN — schrijf altijd voluit in het Nederlands:
    • Kardinaal:   "214"    → "tweehonderdveertien"
    • Decimaal:    "420,3"  → "vierhonderdtwintig komma drie"
    • Ordinals:    "1e"     → "eerste", "3e" → "derde"
    • Tijden:      "14:30"  → "veertien uur dertig"
    • Jaren:       "2026"   → "tweeduizend zesentwintig"
    • Procenten:   "8%"     → "acht procent"
    • Snelheid:    "45 km/u"→ "vijfenveertig kilometer per uur"

  AFKORTINGEN — schrijf voluit of spel letter voor letter:
    • "UCI"  → "U-C-I"
    • "ASO"  → "A-S-O"
    • "nr."  → "nummer"
    • "ca."  → "circa"
    • "bv."  → "bijvoorbeeld"
    • "km"   → "kilometer" (wanneer als maatstaf gebruikt)

  EIGENNAMEN — laat ONGEWIJZIGD. Schrijf namen precies zoals ze in \
  de originele tekst staan. Geen fonetische herspelling.

  OVERIG:
    • Gewone Nederlandse woorden: ongemoeid laten.
    • Begint een fragment na normalisatie met een uitgeschreven getal: \
      zet een hoofdletter op het eerste woord.
    • Na normalisatie mogen er GEEN losse cijfers (0–9) meer in de tekst \
      staan. Controleer dit expliciet.
  VASTE VERVANGINGEN: 
    "Phoenix Poule" → "Phoenix Poel"

════════════════════════════════════════════════════════
STAP 7 — GAP AFTER (ms)
════════════════════════════════════════════════════════
De stilte NA dit fragment, in milliseconden. Richtlijnen:

  "continuing", gedachte loopt direct door  :   0 –  150 ms
  Gewone grens binnen een gedachte          : 200 –  350 ms
  Einde van een gedachte / alinea-grens     : 450 –  700 ms
  Dramatische beat vóór een clou of reveal  : 600 –  900 ms
  Allerlaatste fragment                     : altijd 0

Kies een concreet getal. Ronde getallen zijn prima.

════════════════════════════════════════════════════════
UITVOER — uitsluitend geldige JSON, geen uitleg, geen markdown
════════════════════════════════════════════════════════
{
  "register": "<één zin: overkoepelend register van de hele column>",
  "chunks": [
    {
      "id": 1,
      "sentences": ["P1S1", "P1S2"],
      "text": "<genormaliseerde uitspraakvriendelijke tekst, eventueel met max één non-verbale tag>",
      "position": "opening",
      "control": "<Engelse control instruction voor dit fragment>",
      "gap_after_ms": 300
    }
  ]
}

Zorg dat elke zin-ID exact één keer voorkomt over alle chunks.
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


# Allowed non-verbal tags. Anything in [brackets] not on this list is a warning.
ALLOWED_TAGS = {"[sigh]", "[breath]", "[laugh]", "[exhale]"}
_TAG_RE = re.compile(r"\[[^\]]+\]")


def validate_plan(plan: dict, expected_sentence_ids: set[str] | None = None) -> list[str]:
    """Return a list of warnings (empty if clean). Non-fatal sanity checks."""
    warnings = []
    chunks = plan.get("chunks", [])
    if not chunks:
        warnings.append("No chunks returned.")
        return warnings
    valid_positions = {"opening", "continuing", "final"}
    total_tags = 0
    for i, c in enumerate(chunks):
        cid = c.get("id", i + 1)
        text = c.get("text", "")
        if not text.strip():
            warnings.append(f"Chunk {cid}: empty text.")
        if not c.get("control", "").strip():
            warnings.append(f"Chunk {cid}: missing control tag.")
        if c.get("position") not in valid_positions:
            warnings.append(f"Chunk {cid}: position={c.get('position')!r} "
                            f"(expected one of {valid_positions}).")
        gap = c.get("gap_after_ms")
        if not isinstance(gap, (int, float)):
            warnings.append(f"Chunk {cid}: gap_after_ms={gap!r} is not a number.")
        elif gap < 0 or gap > 3000:
            warnings.append(f"Chunk {cid}: gap_after_ms={gap} out of sane "
                            f"range (0-3000).")
        # Non-verbal tag checks.
        tags_in_chunk = _TAG_RE.findall(text)
        if len(tags_in_chunk) > 1:
            warnings.append(f"Chunk {cid}: {len(tags_in_chunk)} non-verbal tags "
                            f"in one chunk (max 1): {tags_in_chunk}")
        for t in tags_in_chunk:
            if t not in ALLOWED_TAGS:
                warnings.append(f"Chunk {cid}: unknown non-verbal tag {t!r} "
                                f"(allowed: {sorted(ALLOWED_TAGS)})")
        total_tags += len(tags_in_chunk)
    if chunks and chunks[-1].get("gap_after_ms") not in (0, 0.0):
        warnings.append("Last chunk's gap_after_ms should be 0.")
    # Tags should be rare. Flag if the LLM got tag-happy.
    if total_tags > max(4, len(chunks) // 8):
        warnings.append(f"{total_tags} non-verbal tags across {len(chunks)} "
                        f"chunks — likely too many; tags should be a rare "
                        f"seasoning. Review and trim.")
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
    ap.add_argument("--gap-scale", type=float, default=1.0,
                    help="Global multiplier on all per-chunk gaps at stitch time "
                         "(1.2 = 20%% longer pauses everywhere).")
    ap.add_argument("--crossfade-ms", type=int, default=40,
                    help="Crossfade floor at every seam, even zero-gap ones.")
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

    # Stitch config: gaps are per-chunk (gap_after_ms). These are global knobs.
    plan.setdefault("config", {})
    plan["config"]["gap_scale"] = args.gap_scale
    plan["config"]["crossfade_ms"] = args.crossfade_ms

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