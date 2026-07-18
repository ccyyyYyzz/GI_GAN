#!/bin/bash
set -u
C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2

create() {
  local session="$1"
  echo "== creating $session =="
  HOME="$H" timeout 240 "$C" --auth oauth2 new --session "$session" --gpu L4
}

create gan_r38_gan
create gan_r38_vqae
create gan_r38_token
echo "== receipt =="
HOME="$H" timeout 240 "$C" --auth oauth2 sessions
