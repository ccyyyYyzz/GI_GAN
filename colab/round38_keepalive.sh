#!/bin/bash
set -euo pipefail
C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2
STAGE=/var/tmp/gan_r38_stage
mkdir -p "$STAGE"

for session in gan_r38_gan gan_r38_vqae gan_r38_token; do
  line=$(HOME="$H" timeout 240 "$C" --auth oauth2 sessions 2>&1 | grep -F "[$session]")
  endpoint=$(printf '%s' "$line" | sed -nE 's/^\[[^]]+\] ([A-Za-z0-9._-]+) \|.*/\1/p')
  if [ -z "$endpoint" ]; then
    echo "ENDPOINT_PARSE_FAIL $session"
    exit 1
  fi
  HOME="$H" setsid nohup timeout 86400 "$C" --auth oauth2 \
    keep-alive "$endpoint" "$session" >"$STAGE/keepalive_$session.log" 2>&1 < /dev/null &
  sleep 2
  pgrep -af "keep-alive $endpoint $session" >/dev/null
  echo "KEEPALIVE_OK $session $endpoint"
done
