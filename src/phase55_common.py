from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from .phase48_49_common import write_csv, write_markdown_table
from .utils import ensure_dir


PHASE53C_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import")
PHASE53D_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase53D_local_preflight")
PHASE48_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase48_49_colab_import")
PHASE51A_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase51A_colab_import")
PHASE55_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase55_cross_audit")
TASKS = ["rad5", "scr5", "rad10", "scr10"]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_text(path: str | Path) -> str:
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def fmt(value: Any, digits: int = 3) -> str:
    v = to_float(value)
    return "n/a" if math.isnan(v) else f"{v:.{digits}f}"


def mean(values: list[Any]) -> float:
    vals = [to_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def max_value(values: list[Any]) -> float:
    vals = [to_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return max(vals) if vals else float("nan")


def min_value(values: list[Any]) -> float:
    vals = [to_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return min(vals) if vals else float("nan")


def write_rows(root: Path, stem: str, rows: list[dict[str, Any]], title: str) -> None:
    ensure_dir(root)
    write_csv(root / f"{stem}.csv", rows)
    write_markdown_table(root / f"{stem}.md", rows, title)


def add_metric(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    group: str,
    metric: str,
    value: Any,
    task: str = "",
    family: str = "",
    model: str = "",
    source: str = "",
    note: str = "",
) -> None:
    rows.append(
        {
            "phase": phase,
            "group": group,
            "metric": metric,
            "task": task,
            "family": family,
            "model": model,
            "value": value,
            "source": source,
            "note": note,
        }
    )


def copy_if_exists(source: Path, dest_dir: Path, dest_name: str | None = None) -> Path | None:
    if not source.exists():
        return None
    ensure_dir(dest_dir)
    dest = dest_dir / (dest_name or source.name)
    shutil.copy2(source, dest)
    return dest


def best_rows_by_task(rows: list[dict[str, str]], *, include_models: list[str], metric: str = "auc") -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        model = str(row.get("model", ""))
        if include_models and not any(token in model for token in include_models):
            continue
        task = str(row.get("task", ""))
        if not task:
            continue
        if task not in out or to_float(row.get(metric)) > to_float(out[task].get(metric)):
            out[task] = row
    return out


def classify_optional_gan(rows: list[dict[str, str]]) -> tuple[str, str]:
    if not rows:
        return "exploratory only", "optional GAN table not found"
    psnrs = [to_float(r.get("psnr")) for r in rows if str(r.get("status", "")).startswith("ran")]
    rels = [to_float(r.get("rel_meas_err")) for r in rows if str(r.get("status", "")).startswith("ran")]
    if not psnrs:
        return "do not cite", "GAN pilot did not run successfully"
    # Session24 has no baseline perceptual metric; keep conservative.
    if any(math.isnan(v) for v in psnrs):
        return "exploratory only", "PSNR fields incomplete"
    if max(rels) if rels else float("nan") > 0.01:
        return "exploratory only", "RelMeasErr is not clearly controlled"
    return "supplement only", "GAN ran only as Scr-5 pilot and lacks LPIPS/FID/KID improvement evidence"

