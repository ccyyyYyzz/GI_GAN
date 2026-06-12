#!/usr/bin/env bash
set -euo pipefail

ROOT="${NS_MC_GAN_GI_ROOT:-/content/drive/MyDrive/ns_mc_gan_gi}"

python -m compileall -q src
bash scripts/colab_phase26_pca_full.sh
bash scripts/colab_phase26_arch_pilot.sh
bash scripts/colab_phase26_gate_report.sh
