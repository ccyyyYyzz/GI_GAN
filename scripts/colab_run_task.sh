#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/colab/hadamard5_medium_noise001_colab.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

OUTPUT_DIR="$(python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["output_dir"])
PY
)"

mkdir -p "$OUTPUT_DIR"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"

if [[ -f "$OUTPUT_DIR/last.pt" ]]; then
  echo "Found last.pt; resuming full optimizer state."
  python -m src.train --config "$CONFIG" --device cuda --resume_checkpoint "$OUTPUT_DIR/last.pt" --resume_mode full
else
  python -m src.train --config "$CONFIG" --device cuda
fi

python -m src.eval_auto --output_dir "$OUTPUT_DIR" --config "$CONFIG" --device cuda
python -m src.analyze_convergence --output_dir "$OUTPUT_DIR"

echo "Task complete: $CONFIG"
