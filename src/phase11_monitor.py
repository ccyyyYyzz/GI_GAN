from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import ensure_dir


ROOT10 = Path("E:/ns_mc_gan_gi/outputs_phase10")
ROOT11 = Path("E:/ns_mc_gan_gi/outputs_phase11")
MONITOR_DIR = ROOT11 / "monitoring"
STATUS_PATH = ROOT10 / "overnight_status.json"
PROCESS_PATH = ROOT11 / "phase11_full_overnight_process.json"
ERROR_KEYWORDS = [
    "OOM",
    "CUDA out of memory",
    "Traceback",
    "RuntimeError",
    "NaN",
    "loss is nan",
    "KeyboardInterrupt",
    "PermissionError",
    "FileNotFoundError",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Phase 11 running monitor snapshot.")
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--status_path", default=str(STATUS_PATH))
    parser.add_argument("--process_path", default=str(PROCESS_PATH))
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    for _ in range(5):
        try:
            text = path.read_text(encoding="utf-8")
            text = text.lstrip("\ufeff")
            if text.strip():
                return json.loads(text)
        except Exception:
            time.sleep(0.2)
    return {}


def train_process_running(method: str) -> bool:
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*src.train*{method}*' }} | Select-Object -First 1 -ExpandProperty ProcessId",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def tail_lines(path: str | Path, n: int = 40) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return lines[-n:]


def process_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def nvidia_smi() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return result.stderr.strip()
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def read_status(status_path: Path) -> tuple[list[dict], dict | None]:
    status = read_json(status_path)
    tasks = status.get("tasks", [])
    running = next((task for task in tasks if task.get("status") == "running"), None)
    return tasks, running


def read_latest_metrics(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "per_epoch_metrics.csv"
    if not path.exists():
        return {}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    return rows[-1] if rows else {}


def detect_errors(lines: list[str]) -> list[str]:
    joined = "\n".join(lines)
    return [kw for kw in ERROR_KEYWORDS if kw.lower() in joined.lower()]


def plot_early_curve(output_dir: Path, monitor_dir: Path) -> str:
    path = output_dir / "per_epoch_metrics.csv"
    if not path.exists():
        return ""
    try:
        import matplotlib.pyplot as plt

        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        epochs = [float(row["epoch"]) for row in rows if row.get("epoch")]
        psnr = [float(row["val_model_psnr"]) for row in rows if row.get("val_model_psnr")]
        ssim = [float(row["val_model_ssim"]) for row in rows if row.get("val_model_ssim")]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        if epochs and psnr:
            ax1.plot(epochs[-len(psnr) :], psnr, marker="o", label="PSNR")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("PSNR")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        if epochs and ssim:
            ax2.plot(epochs[-len(ssim) :], ssim, color="tab:orange", marker="s", label="SSIM")
        ax2.set_ylabel("SSIM")
        fig.tight_layout()
        out = monitor_dir / "early_convergence_hadamard10.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return str(out)
    except Exception as exc:
        note = monitor_dir / "early_convergence_hadamard10.txt"
        note.write_text(f"Could not render early convergence plot: {exc}\n", encoding="utf-8")
        return str(note)


def recommendation(pid_is_running: bool, current_task: str, errors: list[str], latest: dict[str, Any]) -> str:
    if errors:
        return "errors_detected_review_logs_do_not_kill_without_diagnosis"
    if pid_is_running and current_task:
        return "background_training_running_do_not_start_duplicate_training"
    if not pid_is_running:
        return "background_process_not_running_check_status_and_resume_pending_or_failed_tasks"
    if latest:
        return "training_outputs_present_continue_monitoring"
    return "monitoring_only_wait_for_training_outputs"


def write_running_status(snapshot: dict[str, Any]) -> None:
    lines = [
        "# Phase 11B Running Status",
        "",
        f"- timestamp: {snapshot['timestamp']}",
        f"- pid: {snapshot['pid']}",
        f"- pid_running: {snapshot['pid_running']}",
        f"- current_task: {snapshot['current_task']}",
        f"- current_epoch: {snapshot['current_epoch']}",
        f"- latest_train_loss: {snapshot['latest_train_loss']}",
        f"- latest_val_psnr: {snapshot['latest_val_psnr']}",
        f"- latest_val_ssim: {snapshot['latest_val_ssim']}",
        f"- latest_hq_score: {snapshot['latest_hq_score']}",
        f"- best_hq_exists: {snapshot['best_hq_exists']}",
        f"- close_to_10pct_threshold: {snapshot['close_to_10pct_threshold']}",
        f"- detected_error_keywords: {', '.join(snapshot['detected_error_keywords']) if snapshot['detected_error_keywords'] else 'none'}",
        f"- recommendation: {snapshot['recommendation']}",
        f"- early_convergence_plot: {snapshot['early_convergence_plot']}",
        "",
        "## Waiting For",
        "",
        "- The current train task to finish and runner to advance to eval/analyze.",
        "- Final `eval_metrics.json` after `eval_hadamard10_full_noise001`.",
        "- Updated aggregate/report/assets after the queue reaches aggregation tasks.",
    ]
    (ROOT11 / "RUNNING_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(MONITOR_DIR)
    process = read_json(args.process_path)
    pid = args.pid if args.pid is not None else process.get("pid")
    pid = int(pid) if pid is not None else None
    tasks, running = read_status(Path(args.status_path))
    current_task = running.get("task_name", "") if running else ""
    if not current_task and train_process_running("hadamard10_full_noise001"):
        current_task = "train_hadamard10_full_noise001"
        running = {
            "task_name": current_task,
            "stdout_log": str(ROOT10 / "logs" / "train_hadamard10_full_noise001_stdout.txt"),
            "stderr_log": str(ROOT10 / "logs" / "train_hadamard10_full_noise001_stderr.txt"),
        }
    output_dir = ROOT10 / "hadamard10_full_noise001" if current_task == "train_hadamard10_full_noise001" else ROOT10 / current_task.replace("train_", "")
    latest = read_latest_metrics(output_dir)
    stdout_log = running.get("stdout_log", "") if running else process.get("stdout", "")
    stderr_log = running.get("stderr_log", "") if running else process.get("stderr", "")
    stdout_tail = tail_lines(stdout_log, 40)
    stderr_tail = tail_lines(stderr_log, 40)
    errors = detect_errors(stdout_tail + stderr_tail)
    pid_is_running = process_running(pid)
    psnr = latest.get("val_model_psnr", "")
    ssim = latest.get("val_model_ssim", "")
    hq = latest.get("hq_score", "")
    try:
        close_to_threshold = float(psnr) >= 21.5 or float(ssim) >= 0.62
    except Exception:
        close_to_threshold = False
    early_plot = plot_early_curve(output_dir, MONITOR_DIR) if latest else ""
    snapshot = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pid": pid,
        "pid_running": pid_is_running,
        "current_task": current_task,
        "task_counts": {status: sum(1 for task in tasks if task.get("status") == status) for status in sorted({task.get("status") for task in tasks})},
        "current_epoch": latest.get("epoch", ""),
        "latest_train_loss": latest.get("train_total_loss", ""),
        "latest_val_psnr": psnr,
        "latest_val_ssim": ssim,
        "latest_hq_score": hq,
        "best_hq_exists": (output_dir / "best_hq.pt").exists(),
        "close_to_10pct_threshold": close_to_threshold,
        "last_log_lines_stdout": stdout_tail,
        "last_log_lines_stderr": stderr_tail,
        "detected_error_keywords": errors,
        "gpu_status_if_available": nvidia_smi(),
        "early_convergence_plot": early_plot,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
    }
    snapshot["recommendation"] = recommendation(pid_is_running, current_task, errors, latest)
    (MONITOR_DIR / "monitor_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    lines = [
        "# Phase 11B Monitor Snapshot",
        "",
        f"- timestamp: {snapshot['timestamp']}",
        f"- pid: {pid}",
        f"- pid_running: {pid_is_running}",
        f"- current_task: {current_task}",
        f"- current_epoch: {snapshot['current_epoch']}",
        f"- latest_train_loss: {snapshot['latest_train_loss']}",
        f"- latest_val_psnr: {psnr}",
        f"- latest_val_ssim: {ssim}",
        f"- latest_hq_score: {hq}",
        f"- best_hq_exists: {snapshot['best_hq_exists']}",
        f"- close_to_10pct_threshold: {close_to_threshold}",
        f"- detected_error_keywords: {', '.join(errors) if errors else 'none'}",
        f"- gpu_status_if_available: {snapshot['gpu_status_if_available']}",
        f"- recommendation: {snapshot['recommendation']}",
        f"- early_convergence_plot: {early_plot}",
        f"- stdout_log: {stdout_log}",
        f"- stderr_log: {stderr_log}",
        "",
        "## Last Stdout Lines",
        "",
        *[f"- {line}" for line in stdout_tail[-10:]],
        "",
        "## Last Stderr Lines",
        "",
        *[f"- {line}" for line in stderr_tail[-10:]],
    ]
    (MONITOR_DIR / "monitor_snapshot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_running_status(snapshot)
    print(f"Monitor snapshot written to: {MONITOR_DIR / 'monitor_snapshot.json'}")
    print(f"Running status written to: {ROOT11 / 'RUNNING_STATUS.md'}")


if __name__ == "__main__":
    main()
