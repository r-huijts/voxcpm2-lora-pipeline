#!/usr/bin/env bash
# 04_train.sh — Launch VoxCPM2 LoRA fine-tuning.
#
# Assumes you have cloned the VoxCPM repo (for scripts/train_voxcpm_finetune.py)
# and installed it. Edit VOXCPM_REPO and CONFIG below.
#
# Usage:
#   bash 04_train.sh                 # single GPU
#   NPROC=4 bash 04_train.sh         # multi-GPU via torchrun

set -euo pipefail

VOXCPM_REPO="${VOXCPM_REPO:-/workspace/VoxCPM}"
CONFIG="${CONFIG:-scripts/lora_config.yaml}"
NPROC="${NPROC:-1}"

TRAIN_SCRIPT="${VOXCPM_REPO}/scripts/train_voxcpm_finetune.py"

# --- Sanity checks ---
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: training script not found at ${TRAIN_SCRIPT}"
  echo "Clone the repo first:  git clone https://github.com/OpenBMB/VoxCPM.git"
  echo "and set VOXCPM_REPO to its path."
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found at ${CONFIG}"
  exit 1
fi

echo "Config:        ${CONFIG}"
echo "Train script:  ${TRAIN_SCRIPT}"
echo "GPUs:          ${NPROC}"
echo

# --- TensorBoard reminder ---
TB_DIR="$(grep -E '^tensorboard:' "${CONFIG}" | awk '{print $2}')"
echo "Monitor training in another terminal with:"
echo "  tensorboard --logdir ${TB_DIR}"
echo "Watch loss/diff (should fall then flatten) and the sample audio under"
echo "the AUDIO tab. Stop when samples sound right — usually 1-3 epochs."
echo

# --- Launch ---
if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" \
    "${TRAIN_SCRIPT}" --config_path "${CONFIG}"
else
  python "${TRAIN_SCRIPT}" --config_path "${CONFIG}"
fi

echo
echo "Training finished. Checkpoints are under the save_path in your config."
echo "Pick the best by EAR, not by loss alone — evaluate a few checkpoints"
echo "around convergence with 05_infer.py."
