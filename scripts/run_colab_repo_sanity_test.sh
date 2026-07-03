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
session="repo-sanity-${timestamp}"
archive="/tmp/ghost_repo_sanity_${timestamp}.tgz"
log_path="logs/colab_repo_sanity_test_${timestamp}.log"

exec > >(tee "$log_path") 2>&1

cleanup() {
  local cleanup_status=$?
  if [[ "${session_started:-0}" == "1" ]]; then
    "$COLAB_BIN" --auth "$COLAB_AUTH" stop -s "$session" || true
  fi
  rm -f "$archive"
  exit "$cleanup_status"
}
trap cleanup EXIT

echo "timestamp_utc=$timestamp"
echo "repo_root=$REPO_ROOT"
echo "colab_bin=$COLAB_BIN"
echo "colab_auth=$COLAB_AUTH"
echo "colab_gpu=$COLAB_GPU"
echo "colab_account_label=$COLAB_ACCOUNT_LABEL"
echo "colab_home=$HOME"
echo "session=$session"

"$COLAB_BIN" version
tar --exclude="__pycache__" --exclude="*.pyc" -czf "$archive" \
  src/__init__.py \
  src/measurement.py \
  src/models.py \
  scripts/colab_repo_sanity_test.py
ls -lh "$archive"

"$COLAB_BIN" --auth "$COLAB_AUTH" new -s "$session" --gpu "$COLAB_GPU"
session_started=1
"$COLAB_BIN" --auth "$COLAB_AUTH" upload -s "$session" "$archive" /content/ghost_repo_sanity.tgz

cat <<'PY' | "$COLAB_BIN" --auth "$COLAB_AUTH" exec -s "$session" --timeout "$COLAB_TIMEOUT"
import os
import runpy
import tarfile

os.chdir("/content")
with tarfile.open("/content/ghost_repo_sanity.tgz", "r:gz") as archive:
    archive.extractall("/content", filter="data")
try:
    runpy.run_path("/content/scripts/colab_repo_sanity_test.py", run_name="__main__")
except SystemExit as exc:
    if exc.code not in (0, None):
        raise
PY

if ! grep -q "repo_sanity_ok: true" "$log_path"; then
  echo "Repo sanity success sentinel was not found in $log_path"
  exit 1
fi
