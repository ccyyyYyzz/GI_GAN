from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path


TARGETS = (
    "/content/GI_GAN/",
    "/content/gan_r59_raw_fiber",
    "/content/gan_r60_qualitative_gallery",
)


def matching_processes() -> dict[int, str]:
    matches: dict[int, str] = {}
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(target in command for target in TARGETS):
            matches[int(entry.name)] = command
    return matches


terminated: list[dict[str, object]] = []
for pid, command in matching_processes().items():
    try:
        os.kill(pid, signal.SIGTERM)
        terminated.append({"pid": pid, "signal": "SIGTERM", "command": command})
    except ProcessLookupError:
        pass

deadline = time.monotonic() + 5.0
while time.monotonic() < deadline and matching_processes():
    time.sleep(0.25)

for pid, command in matching_processes().items():
    try:
        os.kill(pid, signal.SIGKILL)
        terminated.append({"pid": pid, "signal": "SIGKILL", "command": command})
    except ProcessLookupError:
        pass

kill_deadline = time.monotonic() + 2.0
while time.monotonic() < kill_deadline and matching_processes():
    time.sleep(0.10)

remaining = matching_processes()
status = (
    "ROUND59_REMOTE_PROJECT_PROCESSES_STOPPED"
    if not remaining
    else "ROUND59_REMOTE_PROJECT_PROCESSES_REMAIN"
)

print(
    json.dumps(
        {
            "status": status,
            "terminated": terminated,
            "remaining_matches": remaining,
        },
        indent=2,
        sort_keys=True,
    )
)
if remaining:
    sys.exit(2)
