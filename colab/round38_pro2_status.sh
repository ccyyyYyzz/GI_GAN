#!/bin/bash
set -euo pipefail
C=/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab
P=/var/tmp/codex-colab-tools/colab-cli-venv/bin/python
H=/var/tmp/codex-colab-accounts/pro2

echo "== live assignments =="
HOME="$H" timeout 240 "$P" - <<'PYEOF'
from colab_cli.common import state
assignments = state.client.list_assignments()
print("COUNT", len(assignments))
for assignment in assignments:
    print(assignment.endpoint, assignment.accelerator.name, assignment.variant.name)
PYEOF

echo "== stored sessions =="
HOME="$H" timeout 240 "$C" --auth oauth2 sessions
