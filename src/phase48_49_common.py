from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .utils import ensure_dir, load_config, save_config, save_json


TASKS: dict[str, dict[str, Any]] = {
    "rad5": {
        "task_dir": "rademacher5_hq_noise001_colab",
        "display": "STL-10 Rad-5",
        "sampling_family": "rademacher",
        "sampling_pct": 5,
        "requires_exact_A": True,
    },
    "scr5": {
        "task_dir": "scrambled_hadamard5_hq_noise001_colab",
        "display": "STL-10 Scr-5",
        "sampling_family": "scrambled_hadamard",
        "sampling_pct": 5,
        "requires_exact_A": False,
    },
    "rad10": {
        "task_dir": "rademacher10_full_noise001_colab",
        "display": "STL-10 Rad-10",
        "sampling_family": "rademacher",
        "sampling_pct": 10,
        "requires_exact_A": True,
    },
    "scr10": {
        "task_dir": "scrambled_hadamard10_full_noise001_colab",
        "display": "STL-10 Scr-10",
        "sampling_family": "scrambled_hadamard",
        "sampling_pct": 10,
        "requires_exact_A": False,
    },
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_sha256s(root: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(root)
    output_path = Path(output_path) if output_path else root / "SHA256SUMS.txt"
    rows: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.resolve() == output_path.resolve():
            continue
        rows.append(f"{file_sha256(path)}  {path.relative_to(root).as_posix()}")
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return output_path


def write_environment(output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    lines = [
        f"created_at: {now_iso()}",
        f"python: {sys.version.replace(os.linesep, ' ')}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
        lines.append(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
    try:
        pip = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        lines.append("\n[pip_freeze]\n" + pip.stdout.strip())
    except Exception as exc:  # pragma: no cover - diagnostic only
        lines.append(f"pip_freeze_error: {exc}")
    path = Path(output_dir) / "environment.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_markdown_table(path: str | Path, rows: list[dict[str, Any]], title: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows were produced.\n", encoding="utf-8")
        return path
    keys = sorted({key for row in rows for key in row.keys()})
    lines = [f"# {title}", "", "|" + "|".join(keys) + "|", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(key, "")) for key in keys) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def find_bundle_file(bundle_root: str | Path, task_key: str, names: list[str]) -> Path | None:
    bundle_root = Path(bundle_root)
    task_dir = TASKS[task_key]["task_dir"]
    prefixes = [
        bundle_root / task_dir,
        bundle_root / "configs" / task_dir,
        bundle_root / "checkpoints" / task_dir,
        bundle_root / "metrics" / task_dir,
        bundle_root / "exact_A" / task_dir,
        bundle_root / task_key,
        bundle_root,
    ]
    for prefix in prefixes:
        for name in names:
            path = prefix / name
            if path.exists():
                return path
    return None


def load_bundle_task(bundle_root: str | Path, task_key: str) -> dict[str, Any]:
    config_path = find_bundle_file(
        bundle_root,
        task_key,
        ["resolved_config.yaml", "config_used.yaml", "run_config.yaml", f"{TASKS[task_key]['task_dir']}.yaml"],
    )
    checkpoint_path = find_bundle_file(
        bundle_root,
        task_key,
        ["last.pt", "best_hq.pt", "best_score.pt", "best_psnr.pt", "checkpoint.pt"],
    )
    metrics_path = find_bundle_file(bundle_root, task_key, ["eval_metrics.json", "best_hq_metrics.json"])
    exact_path = find_bundle_file(
        bundle_root,
        task_key,
        ["measurement_operator_exact.pt", f"{TASKS[task_key]['task_dir']}_measurement_operator_exact.pt"],
    )
    if config_path is None:
        raise FileNotFoundError(f"Missing resolved config for {task_key} in bundle {bundle_root}.")
    if checkpoint_path is None:
        raise FileNotFoundError(f"Missing no-leak checkpoint for {task_key} in bundle {bundle_root}.")
    if TASKS[task_key]["requires_exact_A"] and exact_path is None:
        raise FileNotFoundError(f"Missing required exact-A for {task_key} in bundle {bundle_root}.")
    config = load_config(config_path)
    config["phase48_49_source_config_path"] = str(config_path)
    config["phase48_49_source_checkpoint_path"] = str(checkpoint_path)
    if exact_path is not None:
        config["measurement_operator_exact_path"] = str(exact_path)
        config["exact_A_required"] = bool(TASKS[task_key]["requires_exact_A"])
    return {
        "task_key": task_key,
        "metadata": TASKS[task_key],
        "config": config,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "exact_A_path": exact_path,
    }


def copy_required_bundle_leaf(bundle_root: str | Path, out_dir: str | Path, task_key: str) -> dict[str, str]:
    info = load_bundle_task(bundle_root, task_key)
    out_dir = ensure_dir(out_dir)
    copied: dict[str, str] = {}
    for label, source in [
        ("source_config", info["config_path"]),
        ("source_checkpoint", info["checkpoint_path"]),
        ("source_metrics", info["metrics_path"]),
        ("source_exact_A", info["exact_A_path"]),
    ]:
        if source is None:
            continue
        source = Path(source)
        target = out_dir / f"{label}_{source.name}"
        shutil.copy2(source, target)
        copied[label] = str(target)
    return copied


def save_run_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "run_config.yaml"
    save_config(config, path)
    return path


def write_session_manifest(output_dir: str | Path, session_name: str, payload: dict[str, Any]) -> Path:
    output_dir = ensure_dir(output_dir)
    lines = [
        f"# {session_name} Manifest",
        "",
        f"- created_at: {now_iso()}",
        f"- session_name: {session_name}",
    ]
    for key, value in payload.items():
        lines.append(f"- {key}: {value}")
    path = Path(output_dir) / "SESSION_MANIFEST.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json({"session_name": session_name, "created_at": now_iso(), **payload}, Path(output_dir) / "MANIFEST.json")
    download = [
        f"# {session_name} Download Instructions",
        "",
        "The Colab notebook packages this directory as `<session_name>_outputs.zip`.",
        "",
        "If the zip is larger than the chunk limit, download all `.part_###` files and the split manifest, then merge locally with:",
        "",
        "```powershell",
        ".\\scripts\\phase48_49\\phase48_49_merge_colab_parts.ps1",
        ".\\scripts\\phase48_49\\phase48_49_import_colab_outputs.ps1",
        "```",
        "",
        "Keep `SESSION_STATUS.json`, `SESSION_MANIFEST.md`, `MANIFEST.json`, `SHA256SUMS.txt`, `run_config.yaml`, `environment.txt`, and `command_log.txt` with the imported output.",
    ]
    (Path(output_dir) / "DOWNLOAD_INSTRUCTIONS.md").write_text("\n".join(download) + "\n", encoding="utf-8")
    return path
