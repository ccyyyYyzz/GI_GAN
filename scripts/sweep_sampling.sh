#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-eval}"
RATIOS=(0.01 0.02 0.05 0.10)

for RATIO in "${RATIOS[@]}"; do
  TAG="sr_${RATIO/./}"
  OUT_DIR="E:/ns_mc_gan_gi/outputs/${TAG}"
  if [[ "${MODE}" == "train" ]]; then
    python -m src.train \
      --config configs/default.yaml \
      --sampling_ratio "${RATIO}" \
      --device cuda \
      --output_dir "${OUT_DIR}"
  else
    python -m src.eval \
      --config configs/default.yaml \
      --checkpoint "${OUT_DIR}/best_ssim.pt" \
      --sampling_ratio "${RATIO}" \
      --device cuda \
      --output_dir "${OUT_DIR}"
  fi
done
