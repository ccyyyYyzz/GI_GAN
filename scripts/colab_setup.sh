#!/usr/bin/env bash
set -euo pipefail

echo "== GPU =="
nvidia-smi || true

echo "== Python =="
python --version

if [[ ! -d /content/drive/MyDrive ]]; then
  echo "Google Drive is not mounted. Run drive.mount('/content/drive') first." >&2
  exit 2
fi

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

echo "== Prepare Drive folders =="
mkdir -p /content/drive/MyDrive/ns_mc_gan_gi/data
mkdir -p /content/drive/MyDrive/ns_mc_gan_gi/outputs_phase10_colab
mkdir -p /content/drive/MyDrive/ns_mc_gan_gi/outputs_phase11_colab
mkdir -p /content/drive/MyDrive/ns_mc_gan_gi/colab_archives

echo "Colab setup complete."
