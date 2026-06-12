#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/colab/hadamard5_medium_noise001_colab.yaml}"
OUTPUT_DIR="$(python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["output_dir"])
PY
)"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Output dir missing: $OUTPUT_DIR" >&2
  exit 2
fi

ARCHIVE_DIR="/content/drive/MyDrive/ns_mc_gan_gi/colab_archives"
mkdir -p "$ARCHIVE_DIR"
METHOD="$(basename "$OUTPUT_DIR")"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$ARCHIVE_DIR/${METHOD}_${STAMP}.tar.gz"

tar -czf "$ARCHIVE" -C "$(dirname "$OUTPUT_DIR")" "$METHOD"
echo "$ARCHIVE"
