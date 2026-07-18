#!/bin/bash
set -u
VENVPY=/var/tmp/codex-colab-tools/colab-cli-venv/bin/python

cat > /tmp/gan_rebind.py <<'PYEOF'
from colab_cli.common import state
from colab_cli.state import SessionState

mapping = {
    "gpu-l4-s-kkb-ass1a0-1y0i33h879rwe": "gan_r38_vqae",
    "gpu-l4-s-kkb-ass1a1-jdo9vzzcc9te": "gan_r38_token",
    "gpu-l4-s-kkb-ass1b1-1469d7cs8m0t3": "gan_r38_gan",
}
count = 0
for assignment in state.client.list_assignments():
    name = mapping.get(assignment.endpoint)
    if name is None:
        continue
    state.store.add(
        SessionState(
            name=name,
            token=assignment.runtime_proxy_info.token,
            url=assignment.runtime_proxy_info.url,
            endpoint=assignment.endpoint,
            variant=assignment.variant.name,
            accelerator=assignment.accelerator.name,
        )
    )
    print("REBOUND", name, assignment.endpoint)
    count += 1
print("TOTAL_REBOUND", count)
PYEOF

HOME=/var/tmp/codex-colab-accounts/pro2 timeout 240 "$VENVPY" /tmp/gan_rebind.py
