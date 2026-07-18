#!/bin/bash
set -euo pipefail
C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2
STAGE=/var/tmp/gan_r40_stage
ROOT=/mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab
BRANCH=codex/gan-gi-journal-poc-20260718
mkdir -p "$STAGE"
bash "$ROOT/round38_rebind.sh" >/dev/null

start_one() {
  local session="$1" source_arm="$2" rotations="$3" adv_weights="$4" control="$5"
  local control_args=""
  if [[ "$control" == "yes" ]]; then
    control_args="--control-dev /content/data_control/seed1_dev.pt --control-val /content/data_control/seed1_val.pt"
  fi
  cat >"$STAGE/start_$session.py" <<PYEOF
import datetime
import os
import subprocess
import sys

repo = '/content/GI_GAN'
subprocess.run(['git', 'fetch', 'origin', '$BRANCH'], cwd=repo, check=True)
subprocess.run(['git', 'checkout', '$BRANCH'], cwd=repo, check=True)
subprocess.run(['git', 'pull', '--ff-only', 'origin', '$BRANCH'], cwd=repo, check=True)
out = '/content/gan_r40_results/$session'
os.makedirs(out, exist_ok=True)
cmd = [
    sys.executable, '-u', repo + '/train_fiber_residual_phase_gan.py',
    '--primary-dev', '/content/data_primary/seed0_dev.pt',
    '--primary-val', '/content/data_primary/seed0_val.pt',
    '--config', '/content/data_primary/config_used.yaml',
    '--source-arm', '$source_arm',
    '--rotation-scales', '$rotations',
    '--adv-weights', '$adv_weights',
    '--lpips-weight', '0.003',
    '--steps', '1500',
    '--batch-size', '32',
    '--bootstrap-reps', '5000',
    '--output-dir', out,
]
cmd.extend('$control_args'.split())
with open(out + '/driver.log', 'w') as log, open(os.devnull, 'r') as null:
    subprocess.run(
        ['setsid', '-f', *cmd],
        cwd=repo,
        stdin=null,
        stdout=log,
        stderr=subprocess.STDOUT,
        close_fds=True,
        check=True,
        env=dict(os.environ, PYTHONUNBUFFERED='1'),
    )
receipt = datetime.datetime.now(datetime.timezone.utc).isoformat() + '\n' + ' '.join(cmd) + '\n'
open(out + '/launch_receipt.txt', 'w').write(receipt)
print('BACKGROUND_LAUNCHED', out)
PYEOF
  printf '== start %s ==\n' "$session"
  HOME="$H" timeout 300 "$C" --auth oauth2 exec --session "$session" \
    --file "$STAGE/start_$session.py" --timeout 180 | grep -F BACKGROUND_LAUNCHED
}

start_one gan_r38_gan gan 0.25 0,0.0005,0.0015 no
start_one gan_r38_token gan 0.5 0,0.0005,0.0015 no
start_one gan_r38_vqae vqae_control 0.25,0.5 0 yes
printf 'ROUND40_TRAINING_STARTED\n'
