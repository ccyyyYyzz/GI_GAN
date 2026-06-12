from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import yaml

from .checkpoint_utils import find_best_checkpoint
from .utils import ensure_dir, load_config, save_config


ROOT9 = Path("E:/ns_mc_gan_gi/outputs_phase9")
ROOT10 = Path("E:/ns_mc_gan_gi/outputs_phase10")
ROOT11 = Path("E:/ns_mc_gan_gi/outputs_phase11")
CONFIG10 = Path("configs/phase10")
CONFIG11 = Path("configs/phase11")

KNOWN_PHASE10_METHODS = [
    "hadamard10_full_noise001",
    "hadamard10_full_nonoise",
    "hadamard5_medium_noise001",
    "hadamard5_full_noise001",
    "rademacher10_full_noise001",
    "scrambled_hadamard10_full_noise001",
    "lowfreq_no_dc10_control",
    "mnist_hadamard5_full",
    "fashion_hadamard5_full",
    "cifar10_gray_hadamard10_medium",
    "continuous_physical_hq10_full",
]


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(rows: list[dict[str, Any]], path: str | Path, fields: list[str] | None = None) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if fields is None:
        fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_md_table(rows: list[dict[str, Any]], path: str | Path, fields: list[str]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(field, "")).replace("|", "/") for field in fields) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def as_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        value = float(value)
        if math.isnan(value):
            return None
        return value
    except Exception:
        return None


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "passed"}


def method_config_path(method: str, phase: str = "phase10") -> Path:
    root = CONFIG11 if phase == "phase11" else CONFIG10
    return root / f"{method}.yaml"


def method_output_dir(method: str, phase: str = "phase10") -> Path:
    config_path = method_config_path(method, phase=phase)
    if config_path.exists():
        config = load_config(config_path)
        if config.get("output_dir"):
            return Path(config["output_dir"])
    return (ROOT11 if phase == "phase11" else ROOT10) / method


def read_metrics_for_output(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    return read_json(output_dir / "eval_metrics.json") or read_json(output_dir / "val_metrics_latest.json")


def read_phase10_results() -> list[dict[str, str]]:
    return read_csv_rows(ROOT10 / "phase10_results.csv")


def phase10_row(method: str) -> dict[str, str]:
    for row in read_phase10_results():
        if row.get("method") == method and row.get("phase") == "phase10":
            return row
    return {}


def metric_from_row(row: dict[str, Any], key: str) -> float | None:
    return as_float(row.get(key))


def hq_score(config: dict[str, Any], model: dict[str, Any]) -> float | None:
    psnr = as_float(model.get("psnr"))
    ssim = as_float(model.get("ssim"))
    if psnr is None or ssim is None:
        return None
    rel = as_float(model.get("rel_meas_error")) or 0.0
    return psnr + float(config.get("score_ssim_weight", 20.0)) * ssim - float(
        config.get("score_relmeas_weight", 0.0)
    ) * rel


def threshold_flags(config: dict[str, Any], model: dict[str, Any]) -> tuple[bool, bool, bool]:
    psnr = as_float(model.get("psnr"))
    ssim = as_float(model.get("ssim"))
    if psnr is None or ssim is None:
        return False, False, False
    dataset = str(config.get("dataset_name", "")).lower()
    ratio = float(config.get("sampling_ratio", 0.0) or 0.0)
    stl10_10 = dataset == "stl10" and ratio >= 0.099 and psnr >= 22.0 and ssim >= 0.65
    stl10_5 = dataset == "stl10" and ratio <= 0.051 and psnr >= 20.0 and ssim >= 0.60
    simple = dataset in {"mnist", "fashion_mnist"} and ratio <= 0.051 and psnr >= 25.0 and ssim >= 0.80
    return stl10_10, stl10_5, simple


def read_convergence(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    return read_json(output_dir / "convergence_summary.json")


def last_epoch(output_dir: str | Path, fallback: int = 0) -> int:
    output_dir = Path(output_dir)
    for path in [output_dir / "per_epoch_metrics.csv", output_dir / "eval_history.csv"]:
        rows = read_csv_rows(path)
        if rows:
            value = as_float(rows[-1].get("epoch"))
            if value is not None:
                return int(value)
    return int(fallback)


def safe_copy(src: str | Path, dst: str | Path) -> bool:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copyfile(src, dst)
    return True


def copy_config_with_updates(source: str | Path, dest: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    config = load_config(source)
    config.update(updates)
    save_config(config, dest)
    return config


def plot_bar(rows: list[dict[str, Any]], key: str, path: str | Path, ylabel: str, label_key: str = "method") -> None:
    path = Path(path)
    try:
        import matplotlib.pyplot as plt

        filtered = [row for row in rows if as_float(row.get(key)) is not None]
        labels = [str(row.get(label_key, "")) for row in filtered]
        values = [float(row[key]) for row in filtered]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.75), 4.5))
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        fig.tight_layout()
        ensure_dir(path.parent)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        ensure_dir(path.parent)
        path.with_suffix(".txt").write_text(f"Could not render plot: {exc}\n", encoding="utf-8")
