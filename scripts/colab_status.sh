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

echo "Output: $OUTPUT_DIR"
ls -lah "$OUTPUT_DIR" || true

if [[ -f "$OUTPUT_DIR/per_epoch_metrics.csv" ]]; then
  echo "== Latest epochs =="
  python - "$OUTPUT_DIR/per_epoch_metrics.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")))
for row in rows[-5:]:
    print({
        "epoch": row.get("epoch"),
        "psnr": row.get("val_model_psnr"),
        "ssim": row.get("val_model_ssim"),
        "hq": row.get("hq_score"),
        "loss": row.get("train_total_loss"),
    })
PY
fi

if [[ -f "$OUTPUT_DIR/eval_metrics.json" ]]; then
  echo "== Final eval metrics =="
  python - "$OUTPUT_DIR/eval_metrics.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(data, indent=2))
PY
fi
