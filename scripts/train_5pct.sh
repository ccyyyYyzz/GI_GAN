#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --config configs/default.yaml \
  --sampling_ratio 0.05 \
  --device cuda \
  --output_dir E:/ns_mc_gan_gi/outputs/sr_005
