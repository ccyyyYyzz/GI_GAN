#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_COLAB_BIN="/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab"
if [[ -z "${COLAB_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_COLAB_BIN" ]]; then
    COLAB_BIN="$DEFAULT_COLAB_BIN"
  else
    COLAB_BIN="colab"
  fi
fi

COLAB_AUTH="${COLAB_AUTH:-oauth2}"
COLAB_GPU="${COLAB_GPU:-T4}"
COLAB_TIMEOUT="${COLAB_TIMEOUT:-180}"
COLAB_ACCOUNT_LABEL="${COLAB_ACCOUNT_LABEL:-default}"

mkdir -p logs
timestamp="$(date -u +%Y%m%d_%H%M%SZ)"
log_path="logs/colab_smoke_test_${timestamp}.log"

{
  echo "timestamp_utc=$timestamp"
  echo "repo_root=$REPO_ROOT"
  echo "colab_bin=$COLAB_BIN"
  echo "colab_auth=$COLAB_AUTH"
  echo "colab_gpu=$COLAB_GPU"
  echo "colab_account_label=$COLAB_ACCOUNT_LABEL"
  echo "colab_home=$HOME"
  "$COLAB_BIN" version
  "$COLAB_BIN" --auth "$COLAB_AUTH" run --gpu "$COLAB_GPU" --timeout "$COLAB_TIMEOUT" scripts/colab_smoke_test.py
} 2>&1 | tee "$log_path"

exit "${PIPESTATUS[0]}"
