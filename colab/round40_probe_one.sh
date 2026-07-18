#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  printf 'usage: %s SESSION\n' "$0" >&2
  exit 2
fi
C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2
ROOT=/mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab
HOME="$H" timeout 240 "$C" --auth oauth2 exec --session "$1" \
  --file "$ROOT/round40_probe.py" --timeout 120
