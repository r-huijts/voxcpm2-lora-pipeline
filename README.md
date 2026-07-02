# voxcpm2-lora-pipeline — long-form voice narration

**The problem:** making a written column sound like it's actually being read
aloud by a specific person — at full article length — is harder than it
looks. A generic TTS voice doesn't sound like anyone in particular. Feed a
whole article to VoxCPM2 (or most long-form TTS models) in one shot and it
"rushes" — pace and timbre drift over the length of the piece. Names and
numbers get mispronounced. And a single giant block of text throws away
everything that makes spoken delivery sound human: pacing, breath, the shape
of a sentence building to a punchline.

**What this pipeline does about it:**
- **Clones a specific voice with a LoRA** — real identity, not just a
  reference-clip approximation.
- **Has an LLM read the whole column first** and split it into variable-size
  "delivery units" — complete spoken thoughts — instead of fixed-size chunks.
  Each chunk stays short enough that VoxCPM2 never has room to drift, while
  short punches that land together stay together and long builds stay whole.
- **Respells hard names and numbers** before they're ever spoken, so
  pronunciation doesn't degrade cloning quality.
- **Inserts pauses at stitch time** — short within a paragraph, longer between
  them, sized to the rhetorical weight of each break — instead of asking the
  model to generate silence.

The voice is swappable: point a different LoRA checkpoint + reference clip at
it (see Requirements) and it narrates written text in a different cloned
voice using the same mechanism.

Three stages, run on a GPU pod (see Pipeline below for the full diagram). The
LoRA itself is trained separately, once per voice, outside this repo — here
you only need a finished checkpoint.

## Requirements

- **VoxCPM2 + `nano-vllm-voxcpm`** installed on the pod, plus the rest of
  `requirements.txt`: `pip install -r requirements.txt`
- **ffmpeg** on `PATH` (used for reference-clip conversion and optional
  loudness mastering)
- **A trained LoRA checkpoint** for your voice — a `checkpoints/lora/...`
  directory containing `lora_config.json` + `*.safetensors` weights
- **A reference clip** of the target voice (a few seconds of clean audio, plus
  its transcript) for Stage 2 to anchor timbre against
- **A Portkey API key** for Stage 1 only (the chunking LLM call) — see Stage 1
  below for how to set it

## Pipeline

```
column.txt
   │  01_chunk.py            (LLM via Portkey: group + position + control + gap;
   │                          script builds source_text/spoken_text deterministically)
   ▼
plan.json   ← YOU REVIEW AND EDIT THIS
   │  02_generate_nanovllm.py   (LoRA, reference re-anchor + ASR quality gate per chunk)
   ▼
run-dir/chunk_*.wav + manifest.json
   │  03_stitch.py           (trim, crossfade, insert short/long pauses)
   ▼
final.wav
```

## Reusing settings across runs — `voice.json`

`--lora`, `--reference`, and most tuning flags (`--cfg`, `--gap-scale`,
`--whisper-model`, ...) are stable for a given voice/project, not per-run. Copy
`scripts/voice.example.json` to `voice.json` in the directory you run the
pipeline from and fill in your paths:

```bash
cp scripts/voice.example.json voice.json && nano voice.json
```

All three scripts auto-load `voice.json`, checked in this order: the current
directory first, then next to the script itself (e.g. `scripts/voice.json`) —
so it's found whether you keep one shared `voice.json` in `scripts/` and run
commands from anywhere, or a separate `voice.json` per project directory.
Override with `--config <path>` to point at a specific file explicitly. Found
values fill in defaults for the flags listed in the file — anything you still
pass on the command line wins over the config, and anything in the config wins
over the script's built-in default. Only genuinely per-run values
(`--input`/`--output`, `--plan`, `--out-dir`, `--run-dir`) are never read from
`voice.json`, so it can't accidentally clobber a specific run.

`voice.json` holds real filesystem paths, not secrets — keep `PORTKEY_API_KEY`
in `.env` instead (see Stage 1 below). `voice.json` is gitignored by default.

`controllable` and `loudnorm` can also be set in `voice.json` (e.g. if a
project always wants Controllable Cloning or always wants loudness mastering).
Both accept an explicit `--no-controllable` / `--no-loudnorm` on the command
line to force them off for a single run even when `voice.json` sets them
`true`.

## Cloning a real person's voice: ship generic, override locally

`01_chunk.py`'s LLM prompt describes a delivery *style* (a `(stijl: ...)`
persona descriptor), not a name — the committed default is a generic,
safe-to-share profile:

> veteran Dutch public-broadcast cycling columnist: dry, measured, restrained,
> lightly ironic; precision over pathos, irony in the timing

If you're cloning a specific real person's voice, override this **locally,
never in a committed file** — the VoxCPM2 model card warns against shipping
generation instructions that name a real person. Two equivalent ways, in
priority order (`--style-profile` flag beats both):

```bash
# 1. voice.json (gitignored) -- the recommended way, alongside your other
#    per-project settings:
{"style_profile": "your real descriptor here"}

# 2. NARRATE_STYLE_PROFILE in .env (also gitignored):
echo 'NARRATE_STYLE_PROFILE=your real descriptor here' >> .env
```

`03_stitch.py` also notes in its console output that the result is
AI-generated (provenance documentation, not an audio watermark) — on by
default, disable with `--no-label-ai-generated` or `NARRATE_LABEL_AI_GENERATED=0`.

## Stage 1 — chunk (LLM)

```bash
pip install portkey-ai pysbd
export PORTKEY_API_KEY=...
python scripts/01_chunk.py --input column.txt --output plan.json --model gpt-4o
# optional: --config-id pc-xxxx  --gap-scale 1.0  --crossfade-ms 40
```

**Avoid pasting the key each run:** copy `.env.example` to `.env` in the repo
root and put your key there. `01_chunk.py` loads it automatically; `.env` is
gitignored so it never gets pushed.

```bash
cp .env.example .env && nano .env   # set PORTKEY_API_KEY
```

Two steps inside Stage 1:
1. **pySBD** splits the column into sentences deterministically (Dutch, rule-based,
   handles abbreviations/numbers). The LLM does NOT find sentence boundaries —
   that's the part LLMs occasionally botch.
2. The **LLM makes structural decisions only** — which sentences group into a
   delivery unit, its position in the thought-arc, its control tag, its pause.
   It does **not** write the spoken text. An LLM asked to reproduce text
   verbatim while also editing it (grouping, expanding numbers) can silently
   paraphrase or drop a clause, undetectable by an ID-coverage check alone —
   so the SCRIPT builds the wording itself from the raw sentences instead.

The script then runs a **coverage check**: every pySBD sentence must appear in
exactly one chunk, AND a self-check that concatenating every chunk's
`source_text` reproduces the original column exactly (this catches bugs in the
script's own chunk-building, since the LLM never touches wording at all).

Produces `plan.json`. Each chunk carries:
- `position`: `opening` | `continuing` | `final` — the chunk's role in the
  thought-arc. Drives the control tag, the stitch-time gap defaults, and (in
  Stage 2) whether prosody carries over from the previous chunk.
- `control`: style **plus an intonation hint** derived from position — a `final`
  chunk gets a falling close ("dalende afsluiting"), a `continuing` chunk is told
  not to resolve ("doorlopend, niet afsluiten"). Targets the "every chunk sounds
  like a full stop" problem on the Mode 1 path.
- `gap_after_ms`: a per-chunk numeric pause, sized to the rhetorical weight of the
  break — ~0-150 ms mid-thought, 200-350 between sentences, 450-700 at a thought
  end, 600-900 before a punchline.
- `sentences`: the pySBD sentence IDs grouped into this chunk (for the coverage
  check).
- `source_text`: the original sentences, verbatim — the audit reference, never
  touched by the lexicon or number normalization.
- `spoken_text`: `source_text` run through the lexicon then `normalize_dutch()`
  (deterministic, conservative — numbers/times/years/percentages/abbreviations
  are expanded only where unambiguous; alphanumeric codes and thousands-grouped
  numbers are left alone rather than guessed). This is what Stage 2 generates.
- `carryover_after` (optional, you add it by hand): overrides Stage 2's default
  rule for whether this chunk's prosody carries into the next one — see Stage 2.

**Review `spoken_text`** — that's what gets spoken. The `position` field makes
the LLM's structural judgment inspectable; editing a pause is just changing the
`gap_after_ms` integer.

**Pronunciation lexicon.** `scripts/lexicon.json` holds CONFIRMED respellings
(`{"klassementsman": "klassements-man"}`) applied to the column before chunking.
Only add entries you've verified by ear — a wrong respelling can move the error
rather than fix it. Skip it entirely with `--no-lexicon`.

**Screening respelling candidates before spending a generation cycle.**
`scripts/lexicon_prefilter.py` is a standalone, optional diagnostic tool — not
wired into the pipeline — that checks whether a candidate respelling produces
a *sane phoneme sequence* via espeak-ng, before you burn a GPU cycle testing
it:

```bash
python scripts/lexicon_prefilter.py --word klassementsman \
    --candidates "klassements-man" "klassemensman" "klassement man"
```

It catches consonant-cluster collapse and gross garbage — the
"klassementsman" → "klassemensman" class of bug — cheaply and offline. **It
does not catch stress/emphasis placement**: espeak-ng's stress model isn't
VoxCPM's, so a candidate can pass this filter cleanly and still land with
wrong emphasis. Stress-focused respellings still go straight to the ear test.
This is a coarse net for a cheap, common failure mode, not a pronunciation
oracle — **ear verification via `--only-chunks` in `02_generate_nanovllm.py`
remains mandatory** before any candidate goes into `lexicon.json`. Requires
`espeak-ng` + `phonemizer` (see `requirements.txt`); without them every
candidate is flagged `espeak_unavailable` and left in its original order —
the rest of the pipeline is unaffected either way.

## Stage 2 — generate

```bash
python scripts/02_generate_nanovllm.py \
  --plan plan.json \
  --lora /workspace/voxcpm2-lora-pipeline/checkpoints/lora/step_0000999 \
  --reference /workspace/voxcpm_project/references/ref_voice.wav \
  --out-dir /workspace/narration/run01
# tuning: --cfg 1.5  --timesteps 24  --start-at 12 (resume)
# with voice.json set up, --lora/--reference/tuning flags can be omitted
```

Hi-Fi mode (default): the reference clip + its transcript anchor the voice
directly; control tags are ignored. Pass `--controllable` for Controllable
Cloning instead — the LoRA gives the voice, the reference clip is re-anchored
each chunk to fight drift (`--reground`), and the per-chunk control tag steers
cadence. Writes `chunk_0001.wav ...` and `manifest.json`.

**Prosody carry-over is suppressed at rhetorical boundaries.** By default each
chunk inherits a short prosody tail from the previous one for continuity, but
that's turned off when the previous chunk's `position` is `final` (its falling
cadence shouldn't bleed into what follows) or the current chunk's is `opening`
(a fresh thought shouldn't inherit the prior one's prosody) — in Controllable
mode with regrounding this only drops the tail, the voice anchor stays. Force
either way for one transition with the previous chunk's `carryover_after: true
/ false` in `plan.json`.

**ASR quality gate.** After each chunk, faster-whisper transcribes it and jiwer
scores Word Error Rate against the intended text; chunks over `--wer-threshold`
(default 0.15) are regenerated up to `--max-retries` times and the best attempt
is kept. Disable with `--no-asr`.

**Duration-ratio gate.** WER checks words, not delivery — a chunk can score
perfect WER while rushed, truncated, or "runaway" (VoxCPM's documented
never-stops failure mode, which otherwise silently fills VRAM). Each attempt's
audio length is checked against `words × target seconds-per-word`, as a RATIO
(never an absolute threshold): under `--duration-floor` (default 0.5×) is
rushed/truncated, over `--runaway-ratio` (default 2.0×) is a suspected runaway —
both retry like the WER gate, logged to `wer_log.json` alongside WER. The
per-voice pace target starts at a conservative built-in default and switches to
the running median of accepted chunks after `--baseline-min-chunks` (default 5);
override it directly with `--sec-per-word-target`. Disable entirely with
`--no-duration-gate`.

**Fixing individual chunks.** `--only-chunks 4,7` regenerates just those chunk
IDs in an existing `--out-dir`, leaving the rest untouched. `--interactive`
keeps the model loaded after the run and drops into a prompt for fast one-off
regeneration without a reload.

**Running Stage 2 + 3 together.** Unlike chunk → generate, generate → stitch
never needs a manual pause in between, so `scripts/generate_and_stitch.py`
runs both with one command — it accepts every flag from both scripts, forwards
each to the right one, and defaults `--run-dir`/`--output` from `--out-dir`:

```bash
python scripts/generate_and_stitch.py \
  --plan plan.json --out-dir /workspace/narration/run01 \
  --loudnorm --lufs -16
```

## Stage 3 — stitch

```bash
python scripts/03_stitch.py \
  --run-dir /workspace/narration/run01 \
  --output /workspace/narration/run01/final.wav
# mastering: --loudnorm --lufs -16   (-23 broadcast, -16 podcast; --lufs has no
#   effect unless --loudnorm is also passed)
# scale all pauses: --gap-scale 1.2  (20% longer everywhere; overrides both the
#   manifest's baked-in value AND voice.json if both are set)
```

Trims each chunk's ragged edges, crossfades the seams (40 ms equal-power floor,
applied even at zero-gap seams so flowing chunks butt together cleanly), inserts
each chunk's `gap_after_ms` pause (optionally scaled by `--gap-scale`). Loudness
mastering (`--loudnorm`) uses **pyloudnorm** (EBU R128) with a true-peak guard.
**No speed change is applied.** If still a touch fast, slow it afterward:

```bash
ffmpeg -i final.wav -filter:a "atempo=0.85" final_slow.wav   # pitch preserved
```

**Candidate selection.** If you generated multiple takes per chunk (Stage 2's
interactive `cand` command), drop a `selection.json` (`{"3": 2, "4": 1}`,
chunk id → chosen version) in `--run-dir` and it's picked up automatically, or
point at one explicitly with `--selection`.

**Every stitch also writes:**
- `timeline.json` and `<output-stem>.srt`, next to `--output` — sample-accurate
  start/end/duration/gap for every chunk in the final audio (accounts for trim
  + crossfade overlap, not naive nominal lengths — a targeted regen from a
  timecode you heard a problem at starts here), and one subtitle cue per
  chunk from the same timeline, captioned with the chunk's original
  (pre-normalization) wording.
- `run_report.txt` / `.json`, in `--run-dir` — a QA sheet combining the
  timeline with `wer_log.json` (WER, pace, retries, duration-gate pass/fail,
  RUNAWAY flags) and any candidate picks — one glanceable table instead of
  three separate files. Regenerates every time you re-stitch, e.g. after
  updating `selection.json`.

## Notes

- **Stage 1 is the only part that needs Portkey / an LLM.** Stages 2–3 are local
  to the pod and the model.
- **Resume generation** with `--start-at N` if a long run is interrupted; the
  manifest still records every chunk's gap so the stitcher has the full pattern.
- **The LoRA loader** reads the checkpoint's own `lora_config.json` to match the
  trained rank. Don't let it default.
- **Control tags nudge, they don't command** — per the research, their effect on
  pace is real but stochastic. The chunking and pauses do the heavy lifting.
