#!/usr/bin/env bash
set -euo pipefail

ROOT="${NS_MC_GAN_GI_ROOT:-/content/drive/MyDrive/ns_mc_gan_gi}"
DEVICE="${DEVICE:-cuda}"
CONFIGS="${CONFIGS:-current_hq_rad5_pilot,nafnet_small_rad5_pilot,unrolled_ista_rad5_pilot,current_hq_scr5_pilot,nafnet_small_scr5_pilot,unrolled_ista_scr5_pilot}"

python -m src.phase26_prepare_arch_pilot --drive_root "$ROOT" --device "$DEVICE"
python -m src.run_phase26_arch_pilot --drive_root "$ROOT" --device "$DEVICE" --configs "$CONFIGS" --skip_existing
