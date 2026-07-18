#!/bin/bash
set -euo pipefail

C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2
LOCAL_ROOT=/mnt/e/GAN_FCC_WORK/experiments/gan_gi_journal_round40/colab
ROOT=/mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab

bash "$ROOT/round38_rebind.sh" >/dev/null
for session in gan_r38_gan gan_r38_token gan_r38_vqae; do
  local_dir="$LOCAL_ROOT/$session"
  mkdir -p "$local_dir"
  for name in summary.json partial_results.json driver.log launch_receipt.txt; do
    remote="/content/gan_r40_results/$session/$name"
    local_path="$local_dir/$name"
    if HOME="$H" timeout 240 "$C" --auth oauth2 download \
      --session "$session" "$remote" "$local_path"; then
      printf 'FETCHED %s %s\n' "$session" "$name"
    else
      printf 'PENDING %s %s\n' "$session" "$name"
    fi
  done
done
