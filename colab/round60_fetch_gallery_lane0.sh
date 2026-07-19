#!/bin/bash
set -euo pipefail

C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2
PACK=/mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab/round60_pack_gallery_vm.py
DEST=/mnt/e/GAN_FCC_WORK/experiments/gan_gi_journal_round60
SESSION=gan_r38_gan
ARCHIVE=round60_raw_y_qualitative_gallery_lane0.zip

bash /mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab/rebind_pro2_current.sh
mkdir -p "$DEST"
HOME="$H" timeout 360 "$C" --auth oauth2 exec --session "$SESSION" \
  --file "$PACK" --timeout 300
HOME="$H" timeout 360 "$C" --auth oauth2 download --session "$SESSION" \
  "/content/$ARCHIVE" "$DEST/$ARCHIVE"
HOME="$H" timeout 360 "$C" --auth oauth2 download --session "$SESSION" \
  "/content/$ARCHIVE.sha256" "$DEST/$ARCHIVE.sha256"

expected=$(awk 'NR==1 {print $1}' "$DEST/$ARCHIVE.sha256")
actual=$(sha256sum "$DEST/$ARCHIVE" | awk '{print $1}')
test "$actual" = "$expected" || {
  echo "ROUND60_ARCHIVE_SHA256_MISMATCH:$actual!=$expected" >&2
  exit 1
}
unzip -t "$DEST/$ARCHIVE" >/dev/null
echo "ROUND60_ARCHIVE_VERIFIED $actual"
