from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .utils import ensure_dir


DEFAULT_STATUS_PATH = Path("E:/ns_mc_gan_gi/outputs_phase10/overnight_status.json")
DEFAULT_LOG_DIR = Path("E:/ns_mc_gan_gi/outputs_phase10/logs")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable Phase 10 command queue.")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--status_path", default=str(DEFAULT_STATUS_PATH))
    parser.add_argument("--log_dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--init_only", action="store_true", help="Write/sync pending status rows and exit.")
    parser.add_argument("--max_priority", type=int, default=None, help="Run only tasks with priority <= this value.")
    return parser.parse_args()


def read_queue(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    tasks = data if isinstance(data, list) else data.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"Queue must be a list or contain a tasks list: {path}")
    indexed = list(enumerate(tasks))
    indexed.sort(key=lambda item: (int(item[1].get("priority", item[0] + 1)), item[0]))
    return [dict(task) for _, task in indexed]


def load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"created_at": now_iso(), "tasks": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_status(path: Path, status: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    status["updated_at"] = now_iso()
    with path.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def task_name(task: dict[str, Any]) -> str:
    name = str(task.get("name", "")).strip()
    if not name:
        raise ValueError(f"Task has no name: {task}")
    return name


def outputs_exist(required_outputs: list[str]) -> bool:
    if not required_outputs:
        return True
    return all(Path(str(path)).exists() for path in required_outputs)


def base_record(task: dict[str, Any], log_dir: Path) -> dict[str, Any]:
    name = task_name(task)
    return {
        "task_name": name,
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "return_code": None,
        "cmd": str(task.get("cmd", "")),
        "required_outputs": [str(p) for p in task.get("required_outputs", []) or []],
        "stdout_log": str(log_dir / f"{name}_stdout.txt"),
        "stderr_log": str(log_dir / f"{name}_stderr.txt"),
        "notes": task.get("notes", ""),
        "attempts": 0,
    }


def status_map(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("task_name")): row for row in status.get("tasks", [])}


def sync_tasks(status: dict[str, Any], tasks: list[dict[str, Any]], log_dir: Path) -> None:
    existing = status_map(status)
    rows = []
    for task in tasks:
        name = task_name(task)
        row = existing.get(name, base_record(task, log_dir))
        base = base_record(task, log_dir)
        for key in ["cmd", "required_outputs", "stdout_log", "stderr_log", "notes"]:
            row[key] = base[key]
        rows.append(row)
    status["tasks"] = rows


def should_resume_skip(row: dict[str, Any]) -> bool:
    return row.get("status") in {"completed", "skipped_existing"} and outputs_exist(
        row.get("required_outputs", []) or []
    )


def run_task(
    task: dict[str, Any],
    row: dict[str, Any],
    cwd: Path | None,
    *,
    status_path: Path | None = None,
    status_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = [str(p) for p in task.get("required_outputs", []) or []]
    if should_resume_skip(row):
        row["status"] = "skipped_existing"
        row["end_time"] = now_iso()
        row["return_code"] = 0
        return row
    if bool(task.get("skip_if_outputs_exist", False)) and outputs_exist(required):
        row["status"] = "skipped_existing"
        row["start_time"] = row.get("start_time") or now_iso()
        row["end_time"] = now_iso()
        row["return_code"] = 0
        return row

    cmd = str(task.get("cmd", "")).strip()
    if not cmd:
        row["status"] = "failed_allowed" if bool(task.get("allow_fail", False)) else "failed"
        row["return_code"] = 2
        row["end_time"] = now_iso()
        return row

    stdout_path = Path(row["stdout_log"])
    stderr_path = Path(row["stderr_log"])
    ensure_dir(stdout_path.parent)
    max_retries = int(task.get("max_retries", 0) or 0)
    allow_fail = bool(task.get("allow_fail", False))
    attempts = 0
    return_code = 1
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        row["status"] = "running"
        row["start_time"] = row.get("start_time") or now_iso()
        row["attempts"] = int(row.get("attempts", 0) or 0) + 1
        if status_path is not None and status_doc is not None:
            write_status(status_path, status_doc)
        mode = "a" if attempt > 0 else "w"
        with stdout_path.open(mode, encoding="utf-8", errors="replace") as out, stderr_path.open(
            mode, encoding="utf-8", errors="replace"
        ) as err:
            if attempt > 0:
                out.write(f"\n\n=== retry {attempt} at {now_iso()} ===\n")
                err.write(f"\n\n=== retry {attempt} at {now_iso()} ===\n")
            completed = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None, stdout=out, stderr=err)
            return_code = int(completed.returncode)
        if return_code == 0 and outputs_exist(required):
            row["status"] = "completed"
            break
        if return_code == 0 and not outputs_exist(required):
            return_code = 3
        if attempt >= max_retries:
            row["status"] = "failed_allowed" if allow_fail else "failed"
    row["end_time"] = now_iso()
    row["return_code"] = return_code
    row["attempts_this_run"] = attempts
    return row


def main() -> None:
    args = parse_args()
    queue_path = Path(args.queue)
    status_path = Path(args.status_path)
    log_dir = ensure_dir(args.log_dir)
    tasks = read_queue(queue_path)
    status = load_status(status_path)
    status["queue"] = str(queue_path)
    sync_tasks(status, tasks, log_dir)
    write_status(status_path, status)
    if args.init_only:
        print(f"Initialized overnight status at: {status_path}")
        return
    if args.max_priority is not None:
        tasks = [task for task in tasks if int(task.get("priority", 0) or 0) <= int(args.max_priority)]
    rows = status_map(status)
    cwd = Path(args.cwd) if args.cwd else None
    for task in tasks:
        name = task_name(task)
        row = rows[name]
        print(f"[{now_iso()}] {name}: {row.get('status')}")
        rows[name] = run_task(task, row, cwd, status_path=status_path, status_doc=status)
        for idx, current in enumerate(status["tasks"]):
            if current.get("task_name") == name:
                status["tasks"][idx] = rows[name]
                break
        write_status(status_path, status)
        print(f"[{now_iso()}] {name}: {rows[name].get('status')} rc={rows[name].get('return_code')}")
    write_status(status_path, status)
    print(f"Overnight status written to: {status_path}")


if __name__ == "__main__":
    main()
