#!/usr/bin/env bash
set -euo pipefail

ROOT="${NS_MC_GAN_GI_ROOT:-/content/drive/MyDrive/ns_mc_gan_gi}"

python -m src.aggregate_phase26_arch_pilot --drive_root "$ROOT"
python -m src.phase26_gate_decision --drive_root "$ROOT"
python -m src.make_phase26_limit_arch_report --drive_root "$ROOT"
