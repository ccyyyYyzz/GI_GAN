#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-mnist5}"

case "$TASK" in
  hadamard5_medium)
    CONFIG="configs/colab/hadamard5_medium_noise001_colab.yaml"
    ;;
  rademacher10)
    CONFIG="configs/colab/rademacher10_full_noise001_colab.yaml"
    ;;
  scrambled10)
    CONFIG="configs/colab/scrambled_hadamard10_full_noise001_colab.yaml"
    ;;
  mnist5)
    CONFIG="configs/colab/mnist_hadamard5_full_colab.yaml"
    ;;
  fashion5)
    CONFIG="configs/colab/fashion_hadamard5_full_colab.yaml"
    ;;
  hadamard5_push)
    CONFIG="configs/colab/hadamard5_push_hq_colab.yaml"
    ;;
  configs/*)
    CONFIG="$TASK"
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    echo "Valid: hadamard5_medium, rademacher10, scrambled10, mnist5, fashion5, hadamard5_push, or configs/colab/*.yaml" >&2
    exit 2
    ;;
esac

LOCAL_CONFIG="$(python - "$CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
data = yaml.safe_load(config_path.read_text())
original_output = Path(str(data["output_dir"]))
run_name = original_output.name
data["dataset_root"] = "/content/ns_mc_gan_gi_data"
data["output_dir"] = f"/content/ns_mc_gan_gi_outputs/{run_name}"

local_config_dir = Path("/content/ns_mc_gan_gi_local_configs")
local_config_dir.mkdir(parents=True, exist_ok=True)
local_config = local_config_dir / config_path.name.replace("_colab.yaml", "_local.yaml")
local_config.write_text(yaml.safe_dump(data, sort_keys=False))
print(local_config)
PY
)"

OUTPUT_DIR="$(python - "$LOCAL_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

data = yaml.safe_load(Path(sys.argv[1]).read_text())
print(data["output_dir"])
PY
)"

echo "Config: $LOCAL_CONFIG"
echo "Output: $OUTPUT_DIR"

if [[ -f "$OUTPUT_DIR/last.pt" ]]; then
  echo "Resuming from $OUTPUT_DIR/last.pt"
  python -m src.train --config "$LOCAL_CONFIG" --device cuda --resume_checkpoint "$OUTPUT_DIR/last.pt" --resume_mode full
else
  python -m src.train --config "$LOCAL_CONFIG" --device cuda
fi

python -m src.eval_auto --output_dir "$OUTPUT_DIR" --config "$LOCAL_CONFIG" --device cuda
python -m src.analyze_convergence --output_dir "$OUTPUT_DIR"

echo "Done. Local output written to: $OUTPUT_DIR"
echo "Remember to pack and download it before the runtime disconnects."
