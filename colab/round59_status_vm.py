from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path


with zipfile.ZipFile("/content/gan_rate_bundle.zip") as archive:
    lane = int(json.loads(archive.read("manifest.json").decode("utf-8"))["seed"])
root = Path(f"/content/gan_r59_raw_fiber/lane{lane}")
payload: dict[str, object] = {"lane": lane, "root_exists": root.exists()}
payload["gpu"] = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
).strip()
payload["runner_processes"] = [
    line
    for line in subprocess.check_output(["ps", "-eo", "pid,etime,cmd"], text=True).splitlines()
    if "raw_fiber" in line or "diagnose_fiber_orthogonal" in line
]
if root.exists():
    payload["files"] = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())
    for marker in ("ROUND59_STARTED.json", "ROUND59_COMPLETE.json", "ROUND59_FAILED.json"):
        path = root / marker
        if path.is_file():
            payload[marker] = json.loads(path.read_text(encoding="utf-8"))
    for rate in ("05", "10"):
        log = root / f"rate{rate}/fohi.log"
        if log.is_file():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            payload[f"rate{rate}_log_tail"] = lines[-12:]
        summary = root / f"rate{rate}/fohi/summary.json"
        if summary.is_file():
            data = json.loads(summary.read_text(encoding="utf-8"))
            payload[f"rate{rate}_summary"] = {
                "status": data.get("status"),
                "final_target": data.get("final_target"),
                "runtime_seconds": data.get("runtime_seconds"),
                "fohi_vs_structural": data.get("fohi_vs_structural"),
                "raw_measurement_residual_certificate": data.get("raw_measurement_residual_certificate"),
            }
launcher = Path(f"/content/round59_raw_fiber_lane{lane}_launcher.log")
if launcher.is_file():
    payload["launcher_tail"] = launcher.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
print(json.dumps(payload, indent=2, sort_keys=True))
