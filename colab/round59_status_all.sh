#!/bin/bash
set -euo pipefail
C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2
SCRIPT=/mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab/round59_status_vm.py
bash /mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab/rebind_pro2_current.sh
for session in gan_r38_gan gan_r38_token gan_r38_vqae; do
  echo "===== ${session}"
  HOME="$H" timeout 360 "$C" --auth oauth2 exec --session "$session" --file "$SCRIPT" --timeout 300
done
