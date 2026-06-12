#!/usr/bin/env bash
set -euo pipefail

echo "== GPU =="
nvidia-smi || true

echo "== Python =="
python --version

echo "== Install dependencies =="
python -m pip install -U pip
python -m pip install "numpy<2" tqdm matplotlib scikit-image PyYAML tensorboard

echo "== Torch =="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo "== Prepare local folders =="
mkdir -p /content/ns_mc_gan_gi_data
mkdir -p /content/ns_mc_gan_gi_outputs
mkdir -p /content/ns_mc_gan_gi_archives
mkdir -p /content/ns_mc_gan_gi_local_configs

echo "Local Colab setup complete. Outputs will be temporary until downloaded."
