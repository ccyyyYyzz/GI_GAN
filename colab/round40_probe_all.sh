#!/bin/bash
set -euo pipefail
C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2
ROOT=/mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab
bash "$ROOT/round38_rebind.sh" >/dev/null
for session in gan_r38_gan gan_r38_token gan_r38_vqae; do
  printf '== %s ==\n' "$session"
  HOME="$H" timeout 240 "$C" --auth oauth2 exec --session "$session" \
    --file "$ROOT/round40_probe.py" --timeout 120
done
