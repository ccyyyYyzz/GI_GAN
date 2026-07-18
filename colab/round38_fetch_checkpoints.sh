#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s SESSION\n' "$0" >&2
  exit 2
fi

session=$1
C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2
LOCAL_ROOT=/mnt/e/GAN_FCC_WORK/experiments/gan_gi_journal_round38/colab
ROOT=/mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab

bash "$ROOT/round38_rebind.sh" >/dev/null
case "$session" in
  gan_r38_gan)
    checkpoints=(checkpoint_spatial_lp0.pt checkpoint_spatial_lp0.005.pt checkpoint_spatial_lp0.015.pt)
    ;;
  gan_r38_token)
    checkpoints=(checkpoint_spectral_lp0.pt checkpoint_spectral_lp0.005.pt checkpoint_spectral_lp0.015.pt)
    ;;
  gan_r38_vqae)
    checkpoints=(
      checkpoint_spatial_lp0.pt checkpoint_spatial_lp0.005.pt checkpoint_spatial_lp0.015.pt
      checkpoint_spectral_lp0.pt checkpoint_spectral_lp0.005.pt checkpoint_spectral_lp0.015.pt
    )
    ;;
  *)
    printf 'unknown session: %s\n' "$session" >&2
    exit 2
    ;;
esac

local_dir="$LOCAL_ROOT/$session"
mkdir -p "$local_dir"
for name in "${checkpoints[@]}"; do
  HOME="$H" timeout 240 "$C" --auth oauth2 download --session "$session" \
    "/content/gan_r38_results/$session/$name" "$local_dir/$name"
  printf 'FETCHED %s %s\n' "$session" "$name"
done
