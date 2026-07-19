#!/bin/bash
set -u

C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
H=/var/tmp/codex-colab-accounts/pro2
VENVPY=/var/tmp/codex-colab-tools/colab-cli-venv/bin/python
STOP_SCRIPT=/mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab/round59_stop_vm.py

bash /mnt/e/GAN_FCC_WORK/active_code/completion_gan_round18/colab/rebind_pro2_current.sh || true
stop_failures=0
for session in gan_r38_gan gan_r38_token gan_r38_vqae; do
  echo "===== STOP ${session}"
  HOME="$H" timeout 120 "$C" --auth oauth2 exec --session "$session" \
    --file "$STOP_SCRIPT" --timeout 90 || stop_failures=$((stop_failures + 1))
done
if [ "$stop_failures" -gt 0 ]; then
  echo "REMOTE_STOP_INCOMPLETE count=$stop_failures; enforcing the hard boundary by unassigning all three target VMs"
fi

cat > /tmp/gan_round59_release.py <<'PYEOF'
from colab_cli.common import state

targets = {
    "gpu-l4-s-kkb-ass1b1-1469d7cs8m0t3",
    "gpu-l4-s-kkb-ass1a1-jdo9vzzcc9te",
    "gpu-l4-s-kkb-ass1a0-1y0i33h879rwe",
}
for assignment in list(state.client.list_assignments()):
    if assignment.endpoint not in targets:
        continue
    try:
        state.client.unassign(assignment.endpoint)
        print("RELEASED", assignment.endpoint)
    except Exception as exc:
        print("RELEASE_FAIL", assignment.endpoint, type(exc).__name__, str(exc)[:160])

remaining = [item.endpoint for item in state.client.list_assignments()]
print("REMAINING_ASSIGNMENTS", remaining)
if any(endpoint in targets for endpoint in remaining):
    raise SystemExit("ROUND59_PRO2_RELEASE_INCOMPLETE")
PYEOF

HOME="$H" timeout 240 "$VENVPY" /tmp/gan_round59_release.py
for endpoint in \
  gpu-l4-s-kkb-ass1b1-1469d7cs8m0t3 \
  gpu-l4-s-kkb-ass1a1-jdo9vzzcc9te \
  gpu-l4-s-kkb-ass1a0-1y0i33h879rwe; do
  pkill -f "keep-alive ${endpoint}" 2>/dev/null \
    && echo "KEEPALIVE_KILLED ${endpoint}" \
    || echo "NO_KEEPALIVE ${endpoint}"
done
