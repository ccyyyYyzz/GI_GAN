#!/usr/bin/env bash
set -euo pipefail

python -m src.eval \
  --config configs/default.yaml \
  --checkpoint E:/ns_mc_gan_gi/outputs/sr_005/best_ssim.pt \
  --sampling_ratio 0.05 \
  --device cuda \
  --output_dir E:/ns_mc_gan_gi/outputs/sr_005
