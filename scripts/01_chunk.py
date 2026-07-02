#!/usr/bin/env python3
"""
01_chunk.py — Turn a full Dutch column into a reviewable chunk plan (JSON).

An LLM (via Portkey) reads the WHOLE column first, decides the overall delivery
register, then splits it into "delivery units" — spans spoken as one continuous
breath. The LLM makes STRUCTURAL decisions only: for each chunk it returns
  - sentences: which P#S# sentence IDs belong to this delivery unit
  - position:  "opening" | "continuing" | "final" in the thought-arc
  - control:   a per-chunk style/pace tag, chosen against the whole arc
  - gap_after_ms: the pause after this chunk, in milliseconds

The LLM does NOT write the spoken text. This is deliberate: an LLM asked to
reproduce text verbatim while also editing it (grouping, expanding numbers,
inserting tags) can silently paraphrase, drop a clause, or "improve" a
sentence -- undetectable by a coverage check that only verifies sentence IDs.
Instead, the SCRIPT builds two fields per chunk from the RAW pySBD sentences,
after the LLM has only chosen which IDs go together:
  - source_text:  the original sentences verbatim, byte-for-byte from the
                   pySBD split -- the audit reference, never touched by the
                   lexicon or normalization.
  - spoken_text:  source_text run through deterministic transforms (lexicon
                   respellings, then normalize_dutch() for numbers/
                   abbreviations) -- what actually gets generated.

The output JSON is meant to be EDITED by hand before generation. Nothing is
final until you've read it.

Usage:
    export PORTKEY_API_KEY=...                  # or pass --api-key
    python 01_chunk.py --input column.txt --output plan.json
    python 01_chunk.py --input column.txt --output plan.json \
        --model gpt-4o --config-id pc-xxxx

Requires: portkey_ai, num2words
    pip install portkey-ai num2words
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import pysbd
from portkey_ai import Portkey

from _pipeline_config import load_voice_config, apply_config_defaults
from _dutch_numbers import normalize_dutch


def _load_dotenv():
    """
    Load KEY=VALUE lines from a .env file (repo root, then scripts/) into the
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


def load_lexicon(path: Path | None) -> dict[str, str]:
    """
    Load a pronunciation lexicon: a flat JSON object mapping the original word
    (as it appears in the column) to a phonetic respelling that the TTS voice
    pronounces correctly. Example:

        {
          "klassementsman": "klassements-man",
          "Roglič": "Roglietsj"
        }

    Only CONFIRMED respellings belong here — entries you've verified by ear
    (e.g. with 02_generate_nanovllm.py's --only-chunks on the affected chunk). A wrong
    respelling can move the error rather than fix it, so this file is a record
    of wins, not guesses. Returns {} if no path or file is given.
    """
    if path is None:
        return {}
    if not path.exists():
        sys.exit(f"Lexicon file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"Lexicon is not valid JSON: {e}")
    if not isinstance(data, dict):
        sys.exit("Lexicon must be a JSON object of {original: respelling}.")
    # Coerce values to str; drop empty keys and underscore-prefixed comment keys
    # (e.g. "_comment") so notes can live in the file without being applied.
    return {str(k): str(v) for k, v in data.items()
            if str(k).strip() and not str(k).startswith("_")}


def apply_lexicon(text: str, lexicon: dict[str, str]) -> tuple[str, list[tuple[str, str, int]]]:
    """
    Apply the lexicon to the raw column text with whole-word, case-sensitive
    replacement, BEFORE sentence splitting and the LLM call — so the LLM and the
    TTS only ever see the respelled form.

    Whole-word means the match is bounded by non-letter characters (Unicode
    aware, so accented names like Pogačar are matched as whole words and inner
    accents are preserved). Longer keys are applied first so a multi-word key
    isn't pre-empted by a shorter overlapping one.

    Returns (new_text, applied) where applied is a list of
    (original, respelling, count) for the ones that actually fired — so the
    caller can report what changed.
    """
    if not lexicon:
        return text, []
    applied = []
    # Apply longer keys first (handles multi-word names before their parts).
    for original in sorted(lexicon, key=len, reverse=True):
        respelling = lexicon[original]
        # Word boundary via lookarounds that treat any non-letter (incl. start/
        # end of string, spaces, punctuation) as a boundary. \w is ASCII-biased,
        # so use an explicit letter class with Unicode.
        pattern = re.compile(
            r"(?<![^\W\d_])" + re.escape(original) + r"(?![^\W\d_])",
            flags=re.UNICODE,
        )
        new_text, n = pattern.subn(respelling, text)
        if n > 0:
            applied.append((original, respelling, n))
            text = new_text
    return text, applied


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


def sentence_lookup(rows: list[dict]) -> dict[str, dict]:
    """Map 'P#S#' ID -> its row (raw text + natural order), for building
    source_text straight from the RAW pySBD split -- never from LLM output."""
    return {f"P{r['para']}S{r['sent']}": {**r, "_order": i} for i, r in enumerate(rows)}


def build_source_text(sentence_ids: list[str], lookup: dict[str, dict]) -> str:
    """Join the raw sentences for these IDs verbatim, in natural (para, sent)
    order regardless of the order the LLM listed them in. IDs unknown to the
    lookup are skipped (the coverage check separately flags unknown IDs)."""
    known = [lookup[sid] for sid in sentence_ids if sid in lookup]
    known.sort(key=lambda r: r["_order"])
    return " ".join(r["text"] for r in known)


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


# Generic, safe-to-commit style profile. The VoxCPM2 model card warns against
# using a real person's identity in generation instructions; a project that
# clones a specific real voice should override this locally (never commit the
# override) via, in priority order: the --style-profile flag, a "style_profile"
# key in voice.json (gitignored), the NARRATE_STYLE_PROFILE env var (settable
# via the gitignored .env), or this shipped default.
GENERIC_STYLE_PROFILE = (
    "veteran Dutch public-broadcast cycling columnist: dry, measured, "
    "restrained, lightly ironic; precision over pathos, irony in the timing"
)


def default_style_profile() -> str:
    return os.environ.get("NARRATE_STYLE_PROFILE", GENERIC_STYLE_PROFILE)


SYSTEM_PROMPT_TEMPLATE = """\

Je bent een audioregisseur die een Nederlandse column voorbereidt voor \
tekst-naar-spraak synthese met een gekloonde stem \
(stijl: __STYLE_PROFILE__).

Je krijgt de column als een lijst van genummerde zinnen (P<alinea>S<zin>). \
De zinsgrenzen staan VAST. Jouw taak is uitsluitend STRUCTUREEL: de zinnen \
GROEPEREN tot "delivery units", en per unit de positie, control instruction \
en pauze bepalen. Je herschrijft, normaliseert of respelt de tekst zelf NIET \
— dat gebeurt na jouw output, deterministisch, door het script.

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

  f. AFSLUITERS NOOIT ISOLEREN — een kort, afsluitend fragment (een dalende \
     slotzin, een groet, een sign-off zoals "Bon courage." of "Tot morgen.", \
     een korte terugblik aan het einde van een gedachte) mag NOOIT als eigen \
     fragment staan. Voeg het ALTIJD samen met het VOORAFGAANDE fragment, \
     zodat de slotzin meelift op de aanloop van een langere generatie. \
     Reden: de TTS-stem stopt vroegtijdig wanneer een kort fragment met een \
     dalende, "afgeronde" cadans op zichzelf staat — de stem "denkt" dat het \
     klaar is en kapt de zin af. Een afsluiter die achter de voorafgaande \
     zin hangt, wordt wél volledig uitgesproken. Dit geldt in het bijzonder \
     voor het ALLERLAATSTE fragment van de hele tekst: dat is nooit een losse \
     korte slotzin, maar hangt aan de zin ervoor. Een lange slotzin (meer dan \
     ~twaalf woorden) mag wel op zichzelf staan; het risico geldt alleen \
     korte afsluiters.

  g. Gebruik je oor: waar zou een verteller ademhalen? Dáár is de grens.

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
STAP 5 — GAP AFTER (ms)
════════════════════════════════════════════════════════
De stilte NA dit fragment, in milliseconden.

Wees aan de royale kant met deze pauzes. Deze stem levert droog, bedachtzaam
wielercommentaar; de ironie en het gewicht zitten in de timing. Lucht tussen
delivery units laat een zin landen voordat de volgende begint. Liever iets te
veel ruimte dan een gehaaste, opeengepakte voordracht. Bij twijfel: kies de
langere pauze.

Richtlijnen (royaal bemeten):

  "continuing", gedachte loopt direct door  : 100 –  250 ms
  Gewone grens binnen een gedachte          : 350 –  550 ms
  Einde van een gedachte / alinea-grens     : 650 –  950 ms
  Dramatische beat vóór een clou of reveal  : 850 – 1200 ms
  Allerlaatste fragment                     : altijd 0

Kies een concreet getal. Ronde getallen zijn prima. De globale gap_scale-knop
kan alles achteraf nog proportioneel bijregelen, dus mik hier op de natuurlijke
voordracht en niet op een totale tijdsduur.

════════════════════════════════════════════════════════
UITVOER — uitsluitend geldige JSON, geen uitleg, geen markdown
════════════════════════════════════════════════════════
{
  "register": "<één zin: overkoepelend register van de hele column>",
  "chunks": [
    {
      "id": 1,
      "sentences": ["P1S1", "P1S2"],
      "position": "opening",
      "control": "<Engelse control instruction voor dit fragment>",
      "gap_after_ms": 300
    }
  ]
}

Geef GEEN "text"-veld terug — de tekst wordt door het script zelf opgebouwd
uit de originele zinnen, exact zoals aangeleverd. Jouw taak is uitsluitend
groeperen (sentences), positioneren (position), de control instruction en
de pauze (gap_after_ms).

Zorg dat elke zin-ID exact één keer voorkomt over alle chunks.
"""


def build_system_prompt(style_profile: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.replace("__STYLE_PROFILE__", style_profile)


def build_client(api_key: str, config_id: str | None) -> Portkey:
    kwargs = {"api_key": api_key}
    if config_id:
        kwargs["config"] = config_id
    return Portkey(**kwargs)


def call_llm(client: Portkey, model: str, sentences_block: str, style_profile: str) -> str:
    """Single chat completion; returns raw assistant text."""
    messages = [
        {"role": "system", "content": build_system_prompt(style_profile)},
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


def validate_plan(plan: dict, expected_sentence_ids: set[str] | None = None,
                   rows: list[dict] | None = None) -> list[str]:
    """Return a list of warnings (empty if clean). Non-fatal sanity checks."""
    warnings = []
    chunks = plan.get("chunks", [])
    if not chunks:
        warnings.append("No chunks returned.")
        return warnings
    valid_positions = {"opening", "continuing", "final"}
    for i, c in enumerate(chunks):
        cid = c.get("id", i + 1)
        if not c.get("spoken_text", "").strip():
            warnings.append(f"Chunk {cid}: empty spoken_text.")
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
    if chunks and chunks[-1].get("gap_after_ms") not in (0, 0.0):
        warnings.append("Last chunk's gap_after_ms should be 0.")
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
    # Round-trip check: concatenating chunks' source_text (in narrative
    # order) must reproduce the original column's sentences exactly. This is
    # a self-check on the SCRIPT's own construction (build_source_text), not
    # on the LLM -- the LLM never touches wording, so a mismatch here means a
    # bug in chunk construction, not a paraphrase to catch.
    if rows is not None:
        lookup = sentence_lookup(rows)
        ordered = sorted(
            chunks,
            key=lambda c: min(
                (lookup[s]["_order"] for s in c.get("sentences", []) if s in lookup),
                default=-1,
            ),
        )
        rebuilt = " ".join(c.get("source_text", "") for c in ordered if c.get("source_text"))
        expected_full = " ".join(r["text"] for r in rows)
        if rebuilt != expected_full:
            warnings.append(
                "source_text round-trip check FAILED: concatenated chunks' "
                "source_text does not exactly match the original column "
                "sentences. This indicates a bug in chunk construction (not "
                "the LLM) -- do not trust this plan until investigated."
            )
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
    ap.add_argument("--lexicon", type=Path,
                    default=Path(__file__).resolve().parent / "lexicon.json",
                    help="Pronunciation lexicon JSON ({original: respelling}), "
                         "applied to the column before chunking. Defaults to "
                         "lexicon.json next to this script. Missing default file "
                         "is silently skipped; an explicitly-passed missing file "
                         "is an error.")
    ap.add_argument("--no-lexicon", action="store_true", default=False,
                    help="Skip lexicon application even if lexicon.json exists.")
    ap.add_argument("--style-profile", default=default_style_profile(),
                    help="The '(stijl: ...)' descriptor interpolated into the system "
                         "prompt. Defaults to NARRATE_STYLE_PROFILE (env or .env) if "
                         "set, else a generic, safe-to-commit profile. A project "
                         "cloning a specific real voice should override this via "
                         "voice.json or NARRATE_STYLE_PROFILE -- never commit a real "
                         "person's name here (see README).")

    CONFIGURABLE = {"model", "config_id", "gap_scale", "crossfade_ms", "lexicon",
                     "style_profile"}
    ap.add_argument("--config", type=Path, default=Path("voice.json"),
                    help="Shared per-voice defaults JSON (see scripts/_pipeline_config.py "
                         "and scripts/voice.example.json). Keys: model, config_id, "
                         "gap_scale, crossfade_ms, lexicon, style_profile. CLI flags "
                         "always override it.")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=Path("voice.json"))
    config = load_voice_config(pre.parse_known_args()[0].config)
    apply_config_defaults(ap, config, CONFIGURABLE)

    args = ap.parse_args()

    if not args.api_key:
        sys.exit("No Portkey API key. Pass --api-key or set PORTKEY_API_KEY.")
    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}")

    column = args.input.read_text(encoding="utf-8").strip()
    if not column:
        sys.exit("Input file is empty.")

    # Sentence split FIRST, on the RAW column -- before any lexicon or
    # normalization touches it. source_text (built below, per chunk) comes
    # straight from these rows, so the pronunciation layer can never
    # contaminate the audit reference. The LLM also sees these raw sentences;
    # respelling/number-expansion has no bearing on its structural decisions.
    rows = split_sentences(column)
    if not rows:
        sys.exit("No sentences found after splitting.")
    expected_ids = {f"P{r['para']}S{r['sent']}" for r in rows}
    sentences_block = format_sentences_for_llm(rows)
    print(f"Split into {len(rows)} sentences across "
          f"{rows[-1]['para']} paragraphs (pySBD).")

    print(f"Grouping into delivery units via {args.model}...")
    client = build_client(args.api_key, args.config_id)
    raw = call_llm(client, args.model, sentences_block, args.style_profile)

    try:
        plan = parse_plan(raw)
    except json.JSONDecodeError as e:
        # Save the raw output so nothing is lost when parsing fails.
        dump = args.output.with_suffix(".raw.txt")
        dump.write_text(raw, encoding="utf-8")
        sys.exit(f"Could not parse JSON: {e}\nRaw model output saved to {dump}")

    # Load the pronunciation lexicon once (applied per chunk below, AFTER
    # source_text is built -- never before, so it can't touch the audit copy).
    lexicon = {}
    if not args.no_lexicon:
        lex_path = args.lexicon
        default_lexicon_path = Path(__file__).resolve().parent / "lexicon.json"
        if lex_path.exists() or lex_path != default_lexicon_path:
            # Missing default lexicon is silently skipped; an explicitly
            # passed missing path is an error (load_lexicon exits).
            lexicon = load_lexicon(lex_path)

    # Build source_text (raw, verbatim) and spoken_text (lexicon + deterministic
    # number/abbreviation normalization) for each chunk. The LLM never sees or
    # writes this text -- see the module docstring for why.
    lookup = sentence_lookup(rows)
    lexicon_hits = {}  # original -> total count across all chunks, for the summary
    for c in plan.get("chunks", []):
        c.pop("text", None)  # untrusted if the LLM included it despite instructions
        source_text = build_source_text(c.get("sentences", []), lookup)
        c["source_text"] = source_text
        lexed_text, applied = apply_lexicon(source_text, lexicon)
        for original, respelling, n in applied:
            lexicon_hits[original] = (respelling, lexicon_hits.get(original, (respelling, 0))[1] + n)
        c["spoken_text"] = normalize_dutch(lexed_text)

    if lexicon:
        if lexicon_hits:
            print(f"Lexicon applied ({args.lexicon.name}):")
            for original, (respelling, n) in lexicon_hits.items():
                print(f"  {original!r} → {respelling!r}  ×{n}")
        else:
            print(f"Lexicon loaded ({len(lexicon)} entries) — "
                  f"no matches in this column.")

    # Stitch config: gaps are per-chunk (gap_after_ms). These are global knobs.
    plan.setdefault("config", {})
    plan["config"]["gap_scale"] = args.gap_scale
    plan["config"]["crossfade_ms"] = args.crossfade_ms

    warnings = validate_plan(plan, expected_sentence_ids=expected_ids, rows=rows)

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
    print(f"\nReview and edit {args.output}, then run 02_generate_nanovllm.py.")


if __name__ == "__main__":
    main()
