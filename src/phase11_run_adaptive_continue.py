from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .phase11_common import ROOT11, ensure_dir, read_json, write_json


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> int:
    print("Running:", " ".join(cmd))
    return int(subprocess.run(cmd, cwd=str(cwd) if cwd else None).returncode)


def main() -> None:
    ensure_dir(ROOT11)
    plan = read_json(ROOT11 / "adaptive_continue_plan.json")
    tasks = [item for item in plan.get("plan", []) if item.get("should_run")]
    rows = []
    if not plan:
        rows.append({"method": "adaptive_continue", "status": "skipped", "reason": "missing_plan"})
    for item in tasks:
        method = item["method"]
        config_path = item.get("config_path")
        row = {
            "method": method,
            "config_path": config_path,
            "start_time": now(),
            "end_time": "",
            "status": "running",
            "return_codes": [],
        }
        if not config_path or not Path(config_path).exists():
            row.update({"status": "failed", "reason": "missing_config", "end_time": now()})
            rows.append(row)
            continue
        output_dir = None
        try:
            import yaml

            cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            output_dir = cfg.get("output_dir")
        except Exception:
            output_dir = None
        commands = [
            [sys.executable, "-m", "src.train", "--config", config_path, "--device", "cuda"],
            [sys.executable, "-m", "src.eval_auto", "--output_dir", output_dir or "", "--config", config_path, "--device", "cuda"],
            [sys.executable, "-m", "src.analyze_convergence", "--output_dir", output_dir or ""],
        ]
        status = "completed"
        for cmd in commands:
            if "" in cmd:
                code = 2
            else:
                code = run_cmd(cmd)
            row["return_codes"].append(code)
            if code != 0:
                status = "failed"
                break
        row["status"] = status
        row["end_time"] = now()
        rows.append(row)
    if plan and not tasks:
        rows.append({"method": "adaptive_continue", "status": "skipped", "reason": "no_should_run_tasks", "start_time": now(), "end_time": now()})
    write_json({"tasks": rows, "updated_at": now()}, ROOT11 / "adaptive_continue_status.json")
    print(f"Adaptive continuation status written to: {ROOT11 / 'adaptive_continue_status.json'}")


if __name__ == "__main__":
    main()
