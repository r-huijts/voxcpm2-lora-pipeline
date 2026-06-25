# VoxCPM2 LoRA Voice-Cloning Pipeline

End-to-end: raw recordings → trained LoRA → fine-tuned voice in your long-form
synthesis script. Five steps. The first three are data prep (where the real work
is); the last two are training and inference (mostly config).

The guiding principle from the VoxCPM2 docs: **a LoRA is only as good as its
data.** Clean audio, exact transcripts, and trimmed trailing silence matter more
than any hyperparameter.

---

## Prerequisites

**On the training box (RunPod, etc.):**
```bash
git clone https://github.com/OpenBMB/VoxCPM.git      # for the training script
cd VoxCPM && pip install -e .
pip install pydub faster-whisper tensorboardX argbind transformers librosa
# ffmpeg must be on PATH (apt-get install ffmpeg)
```

**Model snapshot** (so training reads a local path):
```bash
python -c "from modelscope import snapshot_download; \
  snapshot_download('OpenBMB/VoxCPM2', local_dir='/workspace/models/VoxCPM2')"
```

**Hardware:** ~20 GB VRAM for LoRA at batch_size=16. A single 4090/A40/A100 is fine.

---

## Step 1 — Segment raw audio into clips

### If you already have SRT subtitles (preferred — skip transcription)

When each recording has a timestamped `.srt`, use the SRT segmenter instead.
It slices on subtitle timing, groups cues into 3–20s sentence-aligned clips,
and writes the transcript with each clip — no Whisper, no proofreading round two.

```bash
python scripts/00_segment_srt.py --input_dir raw/ --output_dir clips/
# SRTs in a separate folder? add:  --srt_dir srt/
```

Pairs `<name>.wav` with `<name>.srt`. Output is `clip_XXXX.wav` + `clip_XXXX.txt`
pairs. **Then skip straight to Step 3** — Steps 1 (silence segmenter) and 2
(transcription) below are only for raw audio with no subtitles.

### If you only have raw audio (no transcripts)

Cut long recordings into 3–25s clips on silence boundaries.

```bash
python scripts/01_segment.py --input_dir raw/ --output_dir clips/
```

If you get no clips or bad cuts, tune `--silence_db` (-40 quiet room, -30 noisy)
and `--min_silence_ms`. Aim for **10–20 minutes** of total clip audio for a
robust narration voice; 5 minutes works for a quick test.

**Cover your prosodic range.** If you'll generate questions and rhetorical
build-ups (you will), make sure the training clips contain them. A LoRA trained
only on flat declaratives narrates everything flat.

---

## Step 2 — Transcribe

```bash
python scripts/02_transcribe.py --clips_dir clips/ --language nl --model large-v3
```

Produces a `.txt` next to each `.wav`.

**Then review by hand.** Whisper stumbles on names and numbers — exactly the
words your WWII/cycling material is full of (Pogačar, Villars-sur-Ollon, unit
designations, years). Mismatched transcripts degrade cloning quality and text
adherence. Twenty minutes of proofreading here is worth more than any training
tweak. If you already have exact transcripts, skip this step and drop your
`.txt` files next to the clips instead.

---

## Step 3 — Build the manifest

Trims trailing silence (<0.5s — the #1 cause of runaway generation), filters the
3–30s window, adds same-speaker `ref_audio` to ~40% of samples, splits train/val.

```bash
python scripts/03_build_manifest.py --clips_dir clips/ --out_dir dataset/
```

Outputs `dataset/train.jsonl`, `dataset/val.jsonl`, and trimmed audio under
`dataset/audio/`. Your original clips are left untouched.

Check the printed summary: total minutes, % with ref_audio (~40%), and the
`dropped` counts. A high `no_text` count means missing/empty transcripts; a high
`too_short`/`too_long` means re-run step 1 with a different window.

---

## Step 4 — Train

Edit the four PATHS at the top of `scripts/lora_config.yaml`, then:

```bash
bash scripts/04_train.sh                 # single GPU
NPROC=4 bash scripts/04_train.sh         # multi-GPU
```

Monitor in another terminal:
```bash
tensorboard --logdir /workspace/voxcpm_lora/logs/lora
```

**What to watch and when to stop:**
- `loss/diff` should fall, then flatten. When it stops improving, you're near done.
- Listen to the sample audio under TensorBoard's AUDIO tab.
- **1–3 epochs is usually enough for single-speaker.** Overfitting can appear in
  a few hundred steps.
- **Overfit tell:** the model starts ignoring your text — same audio regardless
  of input. If you see/hear that, the best checkpoint is an EARLIER one.
- The config keeps the guardrails on: `training_cfg_rate: 0.1` (never 0),
  `weight_decay: 0.01`. Leave them.

Checkpoints land under `save_path` every `save_interval` steps. Keep several.

---

## Step 5 — Pick the best checkpoint, then wire it in

A/B every checkpoint on identical text and choose by ear:

```bash
python scripts/05_infer.py --compare checkpoints/lora/ \
    --text "Goeiedag. Kent u de Col de la Croix? Tadej Pogačar reed iedereen eraf." \
    --out_dir ab_test/
```

Use text loaded with the failure modes you care about: a hard name (Pogačar), a
question, a declarative. Listen through `ab_test/`. Prefer the earliest
checkpoint that already sounds right.

Single-checkpoint generate, with a cadence control tag:
```bash
python scripts/05_infer.py --lora checkpoints/lora/latest \
    --text "(rustig tempo, duidelijke pauzes)Goeiedag." --output test.wav
```

---

## Wiring the LoRA into your long-form script

One change. In `synthesis/voxcpm2_longform.py`, the model load becomes:

```python
model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    lora_weights_path="/workspace/voxcpm_lora/checkpoints/lora/latest",
    load_denoiser=False,
)
```

Then run **Mode 1 (Controllable Cloning)**. The trained voice supplies the
timbre, so:
- the chunk-opening voice-pop largely disappears (no per-chunk reference
  re-derivation),
- you keep the Mode 1 cadence the control tag gives you,
- you no longer strictly need a reference clip for timbre — though passing one
  still works and can reinforce consistency.

That's the combination Modes 1–4 couldn't give you at once: stable timbre AND
working cadence control.

---

## Quick troubleshooting

| Symptom | Fix |
|---|---|
| OOM during training | raise `grad_accum_steps`, lower `max_batch_tokens` or `batch_size` |
| Generation never stops | trailing silence in data (rebuild manifest); raise `loss/stop` weight; `retry_badcase=True` at inference |
| Voice ignores the text | overfit — use an earlier checkpoint; confirm `training_cfg_rate: 0.1` |
| LoRA seems to do nothing at inference | inference `r`/`alpha`/`enable_*` must match training config exactly |
| Weak voice likeness | raise `r` to 64, set `alpha` to `2*r`, or train more steps |
