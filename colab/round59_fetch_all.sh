#!/bin/bash
set -euo pipefail
C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2
PACK=/mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab/round59_pack_vm.py
DEST=/mnt/e/GAN_FCC_WORK/experiments/gan_gi_journal_round59
bash /mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab/rebind_pro2_current.sh
mkdir -p "$DEST"
sessions=(gan_r38_gan gan_r38_token gan_r38_vqae)
lanes=(0 1 2)
for index in 0 1 2; do
  session="${sessions[$index]}"
  lane="${lanes[$index]}"
  echo "===== packing ${session} lane${lane}"
  HOME="$H" timeout 360 "$C" --auth oauth2 exec --session "$session" --file "$PACK" --timeout 300
  HOME="$H" timeout 360 "$C" --auth oauth2 download --session "$session" "/content/round59_raw_fiber_lane${lane}.zip" "$DEST/round59_raw_fiber_lane${lane}.zip"
  HOME="$H" timeout 360 "$C" --auth oauth2 download --session "$session" "/content/round59_raw_fiber_lane${lane}.zip.sha256" "$DEST/round59_raw_fiber_lane${lane}.zip.sha256"
  archive="$DEST/round59_raw_fiber_lane${lane}.zip"
  sidecar="$archive.sha256"
  expected=$(awk 'NR==1 {print $1}' "$sidecar")
  actual=$(sha256sum "$archive" | awk '{print $1}')
  test "$actual" = "$expected" || {
    echo "ROUND59_ARCHIVE_SHA256_MISMATCH:lane${lane}:$actual!=$expected" >&2
    exit 1
  }
  unzip -t "$archive" >/dev/null
  echo "ROUND59_ARCHIVE_VERIFIED lane${lane} $actual"
done
