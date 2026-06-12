from __future__ import annotations

import csv
import json
import math
import random
import shutil
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .phase14_common import PHASE14, PHASE14_IMPORTS, ensure_dir, read_csv, row_from_output, write_csv, write_md_table


ABLATION_PACK = PHASE14 / "ablation_pack"
PHASE12_REGISTRY = Path("E:/ns_mc_gan_gi/outputs_phase12/final_result_registry.csv")
PHASE12_DC = Path("E:/ns_mc_gan_gi/outputs_phase12/dc_row_control/dc_row_results.csv")
PHASE14_RESULTS = PHASE14 / "phase14_final_results.csv"

MAIN_METHOD_IDS = [
    "stl10_rademacher10_colab_full",
    "stl10_scrambled10_colab_full",
    "stl10_hadamard10_local_full",
    "stl10_hadamard5_local_medium",
    "mnist_hadamard5_colab_full",
    "fashion_hadamard5_colab_full",
]

PHASE14_METHOD_IDS = [
    "stl10_rademacher5_colab_full",
    "stl10_scrambled5_colab_full",
]


def out_dir() -> Path:
    return ensure_dir(ABLATION_PACK)


def load_main_rows(include_phase14: bool = True) -> list[dict[str, Any]]:
    rows = read_csv(PHASE12_REGISTRY)
    by_id = {row.get("method_id"): row for row in rows}
    selected = [by_id[mid] for mid in MAIN_METHOD_IDS if mid in by_id]
    if include_phase14:
        for exp in PHASE14_IMPORTS:
            row = row_from_output(exp)
            if row.get("status") == "completed":
                selected.append(row)
    return selected


def slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_eval_targets(include_phase14: bool = True) -> list[dict[str, Any]]:
    source_rows = read_csv(PHASE14_RESULTS) if PHASE14_RESULTS.exists() else load_main_rows(include_phase14)
    ids = set(MAIN_METHOD_IDS)
    if include_phase14:
        ids.update(PHASE14_METHOD_IDS)
    targets = []
    for row in source_rows:
        method_id = row.get("method_id", "")
        if method_id not in ids or row.get("status") != "completed":
            continue
        ckpt = Path(str(row.get("best_checkpoint_path", "")))
        base_dir = ckpt.parent if ckpt.exists() else Path(str(row.get("eval_metrics_path", ""))).parent
        if method_id in {"stl10_scrambled5_colab_full", "stl10_rademacher5_colab_full"} and (base_dir / "last.pt").exists():
            ckpt = base_dir / "last.pt"
        config = base_dir / "resolved_config.yaml"
        if not ckpt.exists() or not config.exists():
            continue
        targets.append(
            {
                "method_id": method_id,
                "method": row.get("display_name") or method_id,
                "slug": slug(method_id),
                "dataset": row.get("dataset", ""),
                "sampling_ratio": row.get("sampling_ratio", ""),
                "pattern_type": row.get("pattern_type", ""),
                "checkpoint": str(ckpt),
                "config": str(config),
                "source_output_dir": str(base_dir),
            }
        )
    return targets


def f(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def classify(model_psnr: float | None, back_psnr: float | None, delta_psnr: float | None, delta_ssim: float | None) -> str:
    if model_psnr is not None and back_psnr is not None and model_psnr < back_psnr - 0.1:
        return "model_degrades_backprojection"
    if (delta_psnr is not None and delta_psnr >= 0.3) or (delta_ssim is not None and delta_ssim >= 0.03):
        return "model_refinement_helpful"
    if (delta_psnr is not None and delta_psnr < 0.3) and (delta_ssim is not None and delta_ssim < 0.03):
        return "backprojection_dominated"
    return "insufficient_data"


def plot_bar(rows: list[dict[str, Any]], key: str, path: Path, title: str, ylabel: str, group_key: str = "method") -> None:
    if not rows:
        return
    labels = [str(r.get(group_key, r.get("method_id", ""))) for r in rows]
    values = [f(r.get(key)) if f(r.get(key)) is not None else np.nan for r in rows]
    fig_w = max(7.0, min(16.0, 0.8 * len(labels) + 3.0))
    plt.figure(figsize=(fig_w, 4.5))
    plt.bar(range(len(labels)), values)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_lines(rows: list[dict[str, Any]], x_key: str, y_key: str, path: Path, title: str, ylabel: str) -> None:
    if not rows:
        return
    plt.figure(figsize=(8.5, 5.0))
    methods = sorted({r.get("method", r.get("method_id", "")) for r in rows})
    for method in methods:
        subset = [r for r in rows if r.get("method", r.get("method_id", "")) == method]
        subset = sorted(subset, key=lambda r: f(r.get(x_key)) if f(r.get(x_key)) is not None else 999)
        xs = [f(r.get(x_key)) for r in subset]
        ys = [f(r.get(y_key)) for r in subset]
        plt.plot(xs, ys, marker="o", label=method)
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    ensure_dir(path.parent)
    plt.savefig(path, dpi=180)
    plt.close()


def write_rows(base: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    target = out_dir()
    write_csv(target / f"{base}.csv", rows, fields)
    write_md_table(target / f"{base}.md", rows, fields)


def merge_existing_rows(base: str, rows: list[dict[str, Any]], replace_method_ids: set[str] | None = None) -> list[dict[str, Any]]:
    if not replace_method_ids:
        return rows
    existing_path = out_dir() / f"{base}.csv"
    if not existing_path.exists():
        return rows
    existing = read_csv(existing_path)
    kept = [row for row in existing if row.get("method_id") not in replace_method_ids]
    return kept + rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def per_sample_rows(method_row: dict[str, Any]) -> list[dict[str, Any]]:
    metrics_path = Path(str(method_row.get("eval_metrics_path", "")))
    candidates: list[Path] = []
    if metrics_path.exists():
        data = read_json(metrics_path)
        ps = data.get("per_sample_metrics")
        if ps:
            candidates.append(Path(str(ps).replace("/content/drive/MyDrive/ns_mc_gan_gi", "E:/ns_mc_gan_gi")))
    sample_path = Path(str(method_row.get("sample_image_path", "")))
    if sample_path.exists():
        candidates.append(sample_path.parent.parent / "eval_samples_individual" / "per_sample_metrics.csv")
    if metrics_path.exists():
        candidates.append(metrics_path.parent / "eval_samples_individual" / "per_sample_metrics.csv")
    for csv_path in candidates:
        if csv_path.exists():
            return read_csv(csv_path)
    return []


def bootstrap_ci(values: list[float], resamples: int = 1000, seed: int = 42) -> tuple[float | str, float | str]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return "", ""
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def copy_sample_grid(method_row: dict[str, Any], label: str, suffix: str) -> str:
    sample = Path(str(method_row.get("sample_image_path", "")))
    if not sample.exists():
        return ""
    dst = out_dir() / "sample_grids" / f"{label}_{suffix}.png"
    ensure_dir(dst.parent)
    shutil.copy2(sample, dst)
    return str(dst)


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)
