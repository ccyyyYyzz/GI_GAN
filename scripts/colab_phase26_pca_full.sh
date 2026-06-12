#!/usr/bin/env bash
set -euo pipefail

ROOT="${NS_MC_GAN_GI_ROOT:-/content/drive/MyDrive/ns_mc_gan_gi}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-5000}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
DEVICE="${DEVICE:-cuda}"

python -m src.phase26_pca_oracle_full \
  --drive_root "$ROOT" \
  --device "$DEVICE" \
  --train_samples "$TRAIN_SAMPLES" \
  --eval_samples "$EVAL_SAMPLES"
