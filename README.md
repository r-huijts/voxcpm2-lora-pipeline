# voxcpm2-lora-pipeline

Tooling for cloning a voice with [VoxCPM2](https://github.com/OpenBMB/VoxCPM)
and using it for long-form, multi-chunk narration. Built for a Dutch narration
voice, but nothing here is language-specific.

It does two things:

1. **A LoRA fine-tuning pipeline** — turn raw recordings into a trained
   single-speaker voice (data prep → manifest → train → evaluate).
2. **A long-form synthesis driver** — chunk long text and generate it with
   stable cadence, with four selectable cloning modes.

## Why this exists

Chunked TTS voice cloning has two recurring problems: the voice subtly *drifts*
at the start of each chunk (cold-start re-derivation from the reference), and
prosody *resets* at chunk seams (each chunk generated without cross-chunk
context). The synthesis driver gives you modes that trade these off differently;
the LoRA pipeline is the path that resolves both at once — a trained voice gives
stable timbre AND lets you keep cadence control.

See **[PIPELINE.md](PIPELINE.md)** for the full step-by-step runbook.

## Repo layout

```
scripts/
  01_segment.py         segment raw audio into 3-25s clips on silence
  02_transcribe.py      faster-whisper transcription -> .txt sidecars
  03_build_manifest.py  trim silence, filter, mix ref_audio, build JSONL
  04_train.sh           launch LoRA training (single/multi-GPU)
  05_infer.py           generate + A/B compare checkpoints
  lora_config.yaml      training config (r=32, anti-overfit guardrails)
synthesis/
  voxcpm2_longform.py   interactive long-form synthesis, 4 cloning modes
PIPELINE.md             full runbook
requirements.txt
```

## Quickstart (RunPod or any CUDA box)

```bash
# 1. clone this repo
git clone https://github.com/<you>/voxcpm2-lora-pipeline.git
cd voxcpm2-lora-pipeline

# 2. install upstream VoxCPM (provides the model + training scripts)
git clone https://github.com/OpenBMB/VoxCPM.git
cd VoxCPM && pip install -e . && cd ..

# 3. install this repo's data-prep deps
pip install -r requirements.txt
# ffmpeg must also be on PATH:  apt-get install -y ffmpeg

# 4. grab a local VoxCPM2 snapshot
python -c "from modelscope import snapshot_download; \
  snapshot_download('OpenBMB/VoxCPM2', local_dir='models/VoxCPM2')"
```

Then follow [PIPELINE.md](PIPELINE.md) from Step 1.

## The four synthesis modes (synthesis/voxcpm2_longform.py)

| Mode | Name | Timbre source | Cadence control | Notes |
|------|------|---------------|-----------------|-------|
| 1 | Controllable Cloning | reference clip | yes (control tag) | best cadence; some chunk-start drift |
| 2 | Hi-Fi Cloning | reference + transcript | no | tightest timbre; flatter cadence |
| 3 | Voice Design | none (described) | yes | invents a voice from a description |
| 4 | Chained Continuation | prev-chunk tail | no | best cross-seam glue; Hi-Fi cadence |

Once you have a LoRA, load it in `from_pretrained(..., lora_weights_path=...)`
and run **Mode 1** — the trained voice supplies timbre, so drift drops and you
keep cadence control. That combination is the whole point.

## Credits & license

Built on [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) (Apache-2.0). This
tooling is released under the MIT License — see [LICENSE](LICENSE).
