# voxcpm2-lora-pipeline — long-form voice narration

Turns a plain-text column into a finished narration audio file, spoken in a
specific cloned voice (currently: Mart Smeets). It's built on
[VoxCPM2](https://github.com/OpenBMB/VoxCPM), a voice-cloning TTS model, using
a LoRA fine-tuned on that voice's recordings for timbre, plus an LLM-driven
chunking step for natural pacing. Three stages, run on a GPU pod (see
Pipeline below for the full diagram).

The LoRA itself is trained separately, once per voice, outside this repo —
here you only need a finished checkpoint (see Requirements below).

The design principle: **variable-size chunks decided by an LLM that reads the
whole column**, not fixed rules. Each chunk is one "delivery unit" — a complete
spoken thought. Short punches that land together stay together; long builds stay
whole; numbers and hard names are respelled for correct pronunciation. Pauses are
inserted at stitch time (short within a paragraph, long between paragraphs), so
the model never has to generate silence.

Why not one long generation: VoxCPM2 accelerates ("rushes") on long single-shot
text. Chunking removes that at the source — each chunk is too short to drift.

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
   │  01_chunk.py            (LLM via Portkey: chunk + respell + tag + gap)
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

All three scripts auto-load `./voice.json` (override with `--config <path>`)
and use it to fill in defaults for the flags listed in the file — anything you
still pass on the command line wins over the config, and anything in the config
wins over the script's built-in default. Only genuinely per-run values
(`--input`/`--output`, `--plan`, `--out-dir`, `--run-dir`) are never read from
`voice.json`, so it can't accidentally clobber a specific run.

`voice.json` holds real filesystem paths, not secrets — keep `PORTKEY_API_KEY`
in `.env` instead (see Stage 1 below). `voice.json` is gitignored by default.

`controllable` and `loudnorm` can also be set in `voice.json` (e.g. if a
project always wants Controllable Cloning or always wants loudness mastering).
Both accept an explicit `--no-controllable` / `--no-loudnorm` on the command
line to force them off for a single run even when `voice.json` sets them
`true`.

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
2. The **LLM groups** those clean sentences into delivery units, respells
   numbers/names, and tags cadence + gaps.

The script then runs a **coverage check**: every pySBD sentence must appear in
exactly one chunk. If the LLM drops or duplicates a sentence while grouping, you
get a warning before generating — not a hole in the audio.

Produces `plan.json`. Each chunk carries:
- `position`: `opening` | `continuing` | `final` — the chunk's role in the
  thought-arc. This drives everything else.
- `control`: style **plus an intonation hint** derived from position — a `final`
  chunk gets a falling close ("dalende afsluiting"), a `continuing` chunk is told
  not to resolve ("doorlopend, niet afsluiten"). Targets the "every chunk sounds
  like a full stop" problem on the Mode 1 path.
- `gap_after_ms`: a per-chunk numeric pause, sized to the rhetorical weight of the
  break — ~0-150 ms mid-thought, 200-350 between sentences, 450-700 at a thought
  end, 600-900 before a punchline.
- `sentences`: the pySBD sentence IDs grouped into this chunk (for the coverage
  check).

**Review it.** The `position` field makes the LLM's judgment inspectable — you can
see at a glance which chunks it thinks resolve vs. flow, and correct that directly.
Editing a pause is just changing the `gap_after_ms` integer.

**Pronunciation lexicon.** `scripts/lexicon.json` holds CONFIRMED respellings
(`{"klassementsman": "klassements-man"}`) applied to the column before chunking.
Only add entries you've verified by ear — a wrong respelling can move the error
rather than fix it. Skip it entirely with `--no-lexicon`.

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

**ASR quality gate.** After each chunk, faster-whisper transcribes it and jiwer
scores Word Error Rate against the intended text; chunks over `--wer-threshold`
(default 0.15) are regenerated up to `--max-retries` times and the best attempt
is kept. Disable with `--no-asr`.

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

## Notes

- **Stage 1 is the only part that needs Portkey / an LLM.** Stages 2–3 are local
  to the pod and the model.
- **Resume generation** with `--start-at N` if a long run is interrupted; the
  manifest still records every chunk's gap so the stitcher has the full pattern.
- **The LoRA loader** reads the checkpoint's own `lora_config.json` to match the
  trained rank. Don't let it default.
- **Control tags nudge, they don't command** — per the research, their effect on
  pace is real but stochastic. The chunking and pauses do the heavy lifting.
