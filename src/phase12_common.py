from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .utils import ensure_dir


PHASE10 = Path("E:/ns_mc_gan_gi/outputs_phase10")
PHASE11 = Path("E:/ns_mc_gan_gi/outputs_phase11")
PHASE12 = Path("E:/ns_mc_gan_gi/outputs_phase12")


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "method_id": "stl10_hadamard10_local_full",
        "display_name": "STL-10 Lowfreq Hadamard 10%",
        "path": PHASE10 / "hadamard10_full_noise001",
        "source": "local",
        "measurement_family": "lowfreq_hadamard",
        "run_scale": "full",
        "gpu": "local RTX 4060 Laptop",
        "preferred_for_paper": True,
        "notes": "Primary local STL-10 lowfreq Hadamard 10% full result.",
    },
    {
        "method_id": "stl10_rademacher10_colab_full",
        "display_name": "STL-10 Rademacher 10%",
        "path": PHASE10 / "rademacher10_full_noise001_colab_import",
        "source": "colab_import",
        "measurement_family": "rademacher",
        "run_scale": "imported_full",
        "gpu": "Colab L4",
        "preferred_for_paper": True,
        "notes": "Primary STL-10 Rademacher 10% Colab result; checkpoint was merged and checked.",
    },
    {
        "method_id": "stl10_scrambled10_colab_full",
        "display_name": "STL-10 Scrambled Hadamard 10%",
        "path": PHASE10 / "scrambled_hadamard10_full_noise001_colab_import",
        "source": "colab_import",
        "measurement_family": "scrambled_hadamard",
        "run_scale": "imported_full",
        "gpu": "Colab L4",
        "preferred_for_paper": True,
        "notes": "Primary STL-10 scrambled Hadamard 10% Colab result; checkpoint was merged and checked.",
    },
    {
        "method_id": "stl10_hadamard5_local_medium",
        "display_name": "STL-10 Lowfreq Hadamard 5% medium",
        "path": PHASE10 / "hadamard5_medium_noise001",
        "source": "local",
        "measurement_family": "lowfreq_hadamard",
        "run_scale": "medium",
        "gpu": "local RTX 4060 Laptop",
        "preferred_for_paper": True,
        "notes": "Medium STL-10 5% run; do not label as full.",
    },
    {
        "method_id": "stl10_hadamard5_colab_medium",
        "display_name": "STL-10 Lowfreq Hadamard 5% medium Colab",
        "path": PHASE10 / "hadamard5_medium_noise001_colab_import",
        "source": "colab_import",
        "measurement_family": "lowfreq_hadamard",
        "run_scale": "medium",
        "gpu": "Colab L4",
        "preferred_for_paper": False,
        "notes": "Reproducibility counterpart for the local medium STL-10 5% run.",
    },
    {
        "method_id": "mnist_hadamard5_colab_full",
        "display_name": "MNIST Lowfreq Hadamard 5%",
        "path": PHASE10 / "mnist_hadamard5_full",
        "source": "colab_import",
        "measurement_family": "lowfreq_hadamard",
        "run_scale": "imported_full",
        "gpu": "Colab L4",
        "preferred_for_paper": True,
        "notes": "Simple-domain MNIST 5% sanity and quality result.",
    },
    {
        "method_id": "fashion_hadamard5_colab_full",
        "display_name": "Fashion-MNIST Lowfreq Hadamard 5%",
        "path": PHASE10 / "fashion_hadamard5_full_colab_import" / "fashion_hadamard5_full",
        "source": "colab_import",
        "measurement_family": "lowfreq_hadamard",
        "run_scale": "imported_full",
        "gpu": "Colab L4",
        "preferred_for_paper": True,
        "sha256_if_known": "adbdd3fc66439dbd0aab15b5f03cca10e0709ef77dd66dd05e036d3a0fb8f2fa",
        "notes": "Simple-domain Fashion-MNIST 5% Colab result; checkpoint SHA verified.",
    },
    {
        "method_id": "fashion_hadamard5_local",
        "display_name": "Fashion-MNIST Lowfreq Hadamard 5% local",
        "path": PHASE10 / "fashion_hadamard5_full",
        "source": "running_local",
        "measurement_family": "lowfreq_hadamard",
        "run_scale": "running",
        "gpu": "local RTX 4060 Laptop",
        "preferred_for_paper": False,
        "notes": "Local Fashion-MNIST reproducibility run; use as primary only if completed cleanly.",
    },
    {
        "method_id": "stl10_rademacher10_local_incomplete",
        "display_name": "STL-10 Rademacher 10% local incomplete",
        "path": PHASE10 / "rademacher10_full_noise001",
        "source": "local",
        "measurement_family": "rademacher",
        "run_scale": "short",
        "gpu": "local RTX 4060 Laptop",
        "preferred_for_paper": False,
        "notes": "Stopped early; not a completed full result.",
        "optional": True,
    },
    {
        "method_id": "stl10_scrambled10_local_incomplete",
        "display_name": "STL-10 Scrambled Hadamard 10% local incomplete",
        "path": PHASE10 / "scrambled_hadamard10_full_noise001",
        "source": "local",
        "measurement_family": "scrambled_hadamard",
        "run_scale": "short",
        "gpu": "local RTX 4060 Laptop",
        "preferred_for_paper": False,
        "notes": "Stopped early; not a completed full result.",
        "optional": True,
    },
    {
        "method_id": "phase11_hadamard5_push_hq",
        "display_name": "Phase 11 Hadamard5 push HQ",
        "path": PHASE11 / "hadamard5_push_hq",
        "source": "local",
        "measurement_family": "lowfreq_hadamard",
        "run_scale": "probe",
        "gpu": "local RTX 4060 Laptop",
        "preferred_for_paper": False,
        "notes": "Optional Phase 11 result if present.",
        "optional": True,
    },
]


FIELDS = [
    "method_id",
    "display_name",
    "source",
    "dataset",
    "sampling_ratio",
    "pattern_type",
    "measurement_family",
    "noise_std",
    "epochs",
    "epochs_actual",
    "run_scale",
    "gpu",
    "psnr",
    "ssim",
    "mse",
    "backproj_psnr",
    "backproj_ssim",
    "delta_psnr",
    "delta_ssim",
    "rel_meas_err",
    "hq_score",
    "threshold_type",
    "threshold_reached",
    "preferred_for_paper",
    "best_checkpoint_path",
    "checkpoint_exists",
    "sha256_if_known",
    "eval_metrics_path",
    "sample_image_path",
    "convergence_summary_path",
    "limited_examples",
    "status",
    "notes",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_md_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        values = [format_cell(row.get(field, "")) for field in fields]
        lines.append("|" + "|".join(values) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    lines = ["\\begin{tabular}{" + "l" * len(fields) + "}", " \\hline"]
    lines.append(" & ".join(escape_tex(field) for field in fields) + " \\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(escape_tex(format_cell(row.get(field, ""))) for field in fields) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "/").replace("\n", " ")


def escape_tex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def as_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def latest_epoch_row(output_dir: Path) -> dict[str, Any]:
    rows = read_csv(output_dir / "per_epoch_metrics.csv")
    if rows:
        return rows[-1]
    rows = read_csv(output_dir / "eval_history.csv")
    if rows:
        return rows[-1]
    return {}


def metric_data(output_dir: Path) -> tuple[dict[str, Any], Path | None]:
    for name in ["eval_metrics.json", "val_metrics_latest.json", "best_hq_metrics.json"]:
        path = output_dir / name
        data = read_json(path)
        if data:
            return data, path
    return {}, None


def latest_metric_data(output_dir: Path) -> tuple[dict[str, Any], Path | None]:
    for name in ["val_metrics_latest.json", "best_hq_metrics.json", "eval_metrics.json"]:
        path = output_dir / name
        data = read_json(path)
        if data:
            return data, path
    return {}, None


def config_data(output_dir: Path) -> dict[str, Any]:
    data = read_yaml(output_dir / "resolved_config.yaml")
    if data:
        return data
    # Fallback for imported directories that kept only report/metrics.
    name = output_dir.name
    candidates = [
        Path("configs/phase10") / f"{name}.yaml",
        Path("configs/colab") / f"{name}_colab.yaml",
    ]
    if name == "fashion_hadamard5_full":
        candidates.extend([Path("configs/phase10/fashion_hadamard5_full.yaml"), Path("configs/colab/fashion_hadamard5_full_colab.yaml")])
    for path in candidates:
        data = read_yaml(path)
        if data:
            return data
    return {}


def sample_image(output_dir: Path) -> tuple[str, bool]:
    candidates = [
        output_dir / "eval_samples" / "recon_grid.png",
        output_dir / "samples" / "epoch_040.png",
        output_dir / "samples" / "epoch_060.png",
        output_dir / "samples" / "epoch_080.png",
    ]
    sample_dir = output_dir / "samples"
    if sample_dir.exists():
        candidates.extend(sorted(sample_dir.glob("epoch_*.png"), reverse=True))
        candidates.extend(sorted(sample_dir.glob("step_*.png"), reverse=True))
    for path in candidates:
        if path.exists():
            return str(path), path.name == "recon_grid.png"
    return "", True


def threshold(dataset: str, ratio: float | None) -> tuple[str, bool | None, float | None, float | None]:
    dataset = dataset.lower()
    if dataset == "stl10" and ratio is not None and ratio >= 0.099:
        return "stl10_10pct", None, 22.0, 0.65
    if dataset == "stl10" and ratio is not None and ratio <= 0.051:
        return "stl10_5pct", None, 20.0, 0.60
    if dataset in {"mnist", "fashion_mnist"} and ratio is not None and ratio <= 0.051:
        return "simple_5pct", None, 25.0, 0.80
    return "", None, None, None


def hq_reached(dataset: str, ratio: float | None, psnr: float | None, ssim: float | None) -> tuple[str, bool]:
    threshold_type, _, psnr_min, ssim_min = threshold(dataset, ratio)
    if psnr is None or ssim is None or psnr_min is None or ssim_min is None:
        return threshold_type, False
    return threshold_type, psnr >= psnr_min and ssim >= ssim_min


def process_running(pattern: str) -> list[dict[str, str]]:
    cmd = ["powershell", "-NoProfile", "-Command", f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match '{pattern}' }} | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"]
    try:
        raw = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="ignore")
        if not raw.strip():
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def gpu_busy_for_training() -> bool:
    return bool(process_running("python -m src\\.train"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_from_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(exp["path"])
    config = config_data(output_dir)
    if exp.get("source") == "running_local":
        metrics, metrics_path = latest_metric_data(output_dir)
    else:
        metrics, metrics_path = metric_data(output_dir)
    model = metrics.get("model", {}) if metrics else {}
    back = metrics.get("backprojection", {}) if metrics else {}
    improve = metrics.get("improvement", {}) if metrics else {}
    latest = latest_epoch_row(output_dir)
    if not metrics and latest:
        model = {
            "psnr": latest.get("val_model_psnr"),
            "ssim": latest.get("val_model_ssim"),
            "mse": latest.get("val_model_mse"),
            "rel_meas_error": latest.get("val_model_rel_meas_err"),
        }
        back = {
            "psnr": latest.get("val_backproj_psnr"),
            "ssim": latest.get("val_backproj_ssim"),
            "mse": latest.get("val_backproj_mse"),
        }
    dataset = str(config.get("dataset_name", "") or infer_dataset(exp["method_id"]))
    ratio = as_float(config.get("sampling_ratio", infer_ratio(exp["method_id"])))
    psnr = as_float(model.get("psnr"))
    ssim = as_float(model.get("ssim"))
    threshold_type, reached = hq_reached(dataset, ratio, psnr, ssim)
    checkpoint = output_dir / "best_hq.pt"
    if not checkpoint.exists():
        checkpoint = output_dir / "best_score.pt"
    sample, is_recon_grid = sample_image(output_dir)
    status = "completed" if metrics_path and checkpoint.exists() else "missing_checkpoint" if metrics_path else "missing_metrics"
    if exp.get("source") == "running_local":
        if process_running("python -m src\\.train --config configs/phase10/fashion_hadamard5_full\\.yaml"):
            status = "running"
        else:
            target_epochs = as_float(config.get("epochs"))
            actual_epochs = as_float(latest.get("epoch"))
            if target_epochs is not None and actual_epochs is not None and actual_epochs < target_epochs:
                status = "stopped_incomplete"
    if exp.get("run_scale") == "short":
        status = "incomplete"
    hq_score = as_float(latest.get("hq_score")) if latest else None
    if hq_score is None:
        hq_score = None if psnr is None or ssim is None else psnr + 20.0 * ssim - 20.0 * (as_float(model.get("rel_meas_error")) or 0.0)
    return {
        "method_id": exp["method_id"],
        "display_name": exp["display_name"],
        "source": exp["source"],
        "dataset": dataset,
        "sampling_ratio": ratio if ratio is not None else "",
        "pattern_type": config.get("pattern_type", infer_pattern(exp["method_id"])),
        "measurement_family": exp["measurement_family"],
        "noise_std": config.get("noise_std", ""),
        "epochs": config.get("epochs", ""),
        "epochs_actual": latest.get("epoch", ""),
        "run_scale": exp["run_scale"],
        "gpu": exp["gpu"],
        "psnr": psnr if psnr is not None else "",
        "ssim": ssim if ssim is not None else "",
        "mse": as_float(model.get("mse")) if as_float(model.get("mse")) is not None else "",
        "backproj_psnr": as_float(back.get("psnr")) if as_float(back.get("psnr")) is not None else "",
        "backproj_ssim": as_float(back.get("ssim")) if as_float(back.get("ssim")) is not None else "",
        "delta_psnr": as_float(improve.get("delta_psnr")) if as_float(improve.get("delta_psnr")) is not None else diff(psnr, as_float(back.get("psnr"))),
        "delta_ssim": as_float(improve.get("delta_ssim")) if as_float(improve.get("delta_ssim")) is not None else diff(ssim, as_float(back.get("ssim"))),
        "rel_meas_err": as_float(model.get("rel_meas_error")) if as_float(model.get("rel_meas_error")) is not None else "",
        "hq_score": hq_score if hq_score is not None else "",
        "threshold_type": threshold_type,
        "threshold_reached": reached,
        "preferred_for_paper": bool(exp.get("preferred_for_paper")) and status in {"completed", "running"},
        "best_checkpoint_path": str(checkpoint) if checkpoint.exists() else "",
        "checkpoint_exists": checkpoint.exists(),
        "sha256_if_known": exp.get("sha256_if_known", ""),
        "eval_metrics_path": str(metrics_path) if metrics_path else "",
        "sample_image_path": sample,
        "convergence_summary_path": str(output_dir / "convergence_summary.md") if (output_dir / "convergence_summary.md").exists() else "",
        "limited_examples": not is_recon_grid,
        "status": status,
        "notes": exp.get("notes", ""),
    }


def diff(a: float | None, b: float | None) -> float | str:
    if a is None or b is None:
        return ""
    return a - b


def infer_dataset(method_id: str) -> str:
    if "mnist_" in method_id and "fashion" not in method_id:
        return "mnist"
    if "fashion" in method_id:
        return "fashion_mnist"
    return "stl10"


def infer_ratio(method_id: str) -> float:
    return 0.05 if "5" in method_id else 0.1


def infer_pattern(method_id: str) -> str:
    if "rademacher" in method_id:
        return "rademacher"
    return "lowfreq_hadamard"


def load_registry() -> list[dict[str, str]]:
    return read_csv(PHASE12 / "final_result_registry.csv")


def round_float(value: Any, digits: int = 4) -> str:
    v = as_float(value)
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True
