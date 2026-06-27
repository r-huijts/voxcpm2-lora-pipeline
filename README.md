# narrate/ — LLM-chunked long-form narration with the trained LoRA

Three stages, run on the pod. Turns a full Dutch column into finished narration
in the trained Mart-Smeets voice, with natural pauses and per-chunk cadence.

The design principle: **variable-size chunks decided by an LLM that reads the
whole column**, not fixed rules. Each chunk is one "delivery unit" — a complete
spoken thought. Short punches that land together stay together; long builds stay
whole; numbers and hard names are respelled for correct pronunciation. Pauses are
inserted at stitch time (short within a paragraph, long between paragraphs), so
the model never has to generate silence.

Why not one long generation: VoxCPM2 accelerates ("rushes") on long single-shot
text. Chunking removes that at the source — each chunk is too short to drift.

## Pipeline

```
column.txt
   │  01_chunk.py   (LLM via Portkey: chunk + respell + tag + gap)
   ▼
plan.json   ← YOU REVIEW AND EDIT THIS
   │  02_generate.py   (999 LoRA, Mode 1, reference re-anchor per chunk)
   ▼
run-dir/chunk_*.wav + manifest.json
   │  03_stitch.py   (trim, crossfade, insert short/long pauses)
   ▼
final.wav
```

## Stage 1 — chunk (LLM)

```bash
pip install portkey-ai pysbd
export PORTKEY_API_KEY=...
python narrate/01_chunk.py --input column.txt --output plan.json --model gpt-4o
# optional: --config-id pc-xxxx  --short-pause-ms 220  --long-pause-ms 550
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

## Stage 2 — generate

```bash
python narrate/02_generate.py \
  --plan plan.json \
  --lora /workspace/voxcpm2-lora-pipeline/checkpoints/lora/step_0000999 \
  --reference /workspace/voxcpm_project/references/ref_voice.wav \
  --out-dir /workspace/narration/run01
# tuning: --cfg 1.5  --timesteps 24  --start-at 12 (resume)
```

Mode 1 (Controllable Cloning): the LoRA gives the voice, the reference clip is
re-anchored on every chunk to fight drift, the per-chunk control tag steers
cadence. `normalize` is OFF by default — the LLM already expanded numbers/names.
Writes `chunk_0001.wav ...` and `manifest.json`.

## Stage 3 — stitch

```bash
python narrate/03_stitch.py \
  --run-dir /workspace/narration/run01 \
  --output /workspace/narration/run01/final.wav
# mastering: --loudnorm --lufs -16   (-23 broadcast, -16 podcast)
# scale all pauses: --gap-scale 1.2  (20% longer everywhere)
```

Trims each chunk's ragged edges, crossfades the seams (40 ms equal-power floor,
applied even at zero-gap seams so flowing chunks butt together cleanly), inserts
each chunk's `gap_after_ms` pause (optionally scaled by `--gap-scale`). Loudness
mastering (`--loudnorm`) uses **pyloudnorm** (EBU R128) with a true-peak guard.
**No speed change is applied.** If still a touch fast, slow it afterward:

```bash
ffmpeg -i final.wav -filter:a "atempo=0.85" final_slow.wav   # pitch preserved
```

## Notes

- **Stage 1 is the only part that needs Portkey / an LLM.** Stages 2–3 are local
  to the pod and the model.
- **Resume generation** with `--start-at N` if a long run is interrupted; the
  manifest still records every chunk's gap so the stitcher has the full pattern.
- **The LoRA loader** reads the checkpoint's own `lora_config.json` to match the
  trained rank (r=32) — same fix as `scripts/05_infer.py`. Don't let it default.
- **Control tags nudge, they don't command** — per the research, their effect on
  pace is real but stochastic. The chunking and pauses do the heavy lifting.
