from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import yaml


PHASE10 = Path("E:/ns_mc_gan_gi/outputs_phase10")
PHASE12 = Path("E:/ns_mc_gan_gi/outputs_phase12")
PHASE14 = Path("E:/ns_mc_gan_gi/outputs_phase14")
COLAB_DOWNLOADS = Path("E:/ns_mc_gan_gi/colab_downloads")

PHASE14_IMPORTS = [
    {
        "method_id": "stl10_rademacher5_colab_full",
        "display_name": "STL-10 Rademacher 5%",
        "source": "colab_import",
        "measurement_family": "rademacher",
        "gpu": "Colab L4/A100/T4",
        "path": PHASE14 / "rademacher5_hq_noise001_colab_import",
        "config": Path("configs/phase14_colab/rademacher5_hq_noise001_colab.yaml"),
        "notes": "Phase 14 STL-10 5% HQ result; trained on Colab, not local.",
    },
    {
        "method_id": "stl10_scrambled5_colab_full",
        "display_name": "STL-10 Scrambled Hadamard 5%",
        "source": "colab_import",
        "measurement_family": "scrambled_hadamard",
        "gpu": "Colab L4/A100/T4",
        "path": PHASE14 / "scrambled_hadamard5_hq_noise001_colab_import",
        "config": Path("configs/phase14_colab/scrambled_hadamard5_hq_noise001_colab.yaml"),
        "notes": "Phase 14 STL-10 5% HQ result; trained on Colab, not local.",
    },
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    fields = fields or sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        values = [format_cell(row.get(field, "")) for field in fields]
        lines.append("|" + "|".join(values) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value).replace("|", "/").replace("\n", " ")


def as_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        backup = dst.with_name(dst.name + "_backup")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(dst), str(backup))
    shutil.copytree(src, dst)


def metric_file(output_dir: Path) -> Path | None:
    for name in ("eval_metrics.json", "best_hq_metrics.json", "val_metrics_latest.json"):
        path = output_dir / name
        if path.exists():
            return path
    return None


def config_for(output_dir: Path, fallback: Path | None = None) -> dict[str, Any]:
    for path in (output_dir / "resolved_config.yaml", fallback):
        if path:
            data = read_yaml(path)
            if data:
                return data
    return {}


def sample_image(output_dir: Path) -> str:
    candidates = [
        output_dir / "eval_samples" / "recon_grid.png",
        output_dir / "samples" / "epoch_080.png",
        output_dir / "samples" / "epoch_060.png",
        output_dir / "samples" / "epoch_040.png",
    ]
    sample_dir = output_dir / "samples"
    if sample_dir.exists():
        candidates.extend(sorted(sample_dir.glob("epoch_*.png"), reverse=True))
        candidates.extend(sorted(sample_dir.glob("step_*.png"), reverse=True))
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def row_from_output(exp: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(exp["path"])
    metrics_path = metric_file(output_dir)
    metrics = read_json(metrics_path) if metrics_path else {}
    model = metrics.get("model", {}) if metrics else {}
    back = metrics.get("backprojection", {}) if metrics else {}
    improve = metrics.get("improvement", {}) if metrics else {}
    cfg = config_for(output_dir, exp.get("config"))
    psnr = as_float(model.get("psnr"))
    ssim = as_float(model.get("ssim"))
    ratio = as_float(cfg.get("sampling_ratio", 0.05))
    ckpt = first_existing(output_dir, ["best_hq.pt", "best_score.pt", "best_psnr.pt", "last.pt"])
    threshold_reached = bool(psnr is not None and ssim is not None and psnr >= 20.0 and ssim >= 0.60)
    return {
        "method_id": exp["method_id"],
        "display_name": exp["display_name"],
        "source": exp.get("source", ""),
        "dataset": cfg.get("dataset_name", "stl10"),
        "sampling_ratio": ratio if ratio is not None else "",
        "pattern_type": cfg.get("pattern_type", ""),
        "measurement_family": exp.get("measurement_family", ""),
        "noise_std": cfg.get("noise_std", ""),
        "epochs": cfg.get("epochs", ""),
        "gpu": exp.get("gpu", ""),
        "psnr": psnr if psnr is not None else "",
        "ssim": ssim if ssim is not None else "",
        "mse": as_float(model.get("mse")) if as_float(model.get("mse")) is not None else "",
        "backproj_psnr": as_float(back.get("psnr")) if as_float(back.get("psnr")) is not None else "",
        "backproj_ssim": as_float(back.get("ssim")) if as_float(back.get("ssim")) is not None else "",
        "delta_psnr": as_float(improve.get("delta_psnr")) if as_float(improve.get("delta_psnr")) is not None else "",
        "delta_ssim": as_float(improve.get("delta_ssim")) if as_float(improve.get("delta_ssim")) is not None else "",
        "rel_meas_err": as_float(model.get("rel_meas_error")) if as_float(model.get("rel_meas_error")) is not None else "",
        "threshold_type": "stl10_5pct",
        "threshold_reached": threshold_reached,
        "best_checkpoint_path": str(ckpt) if ckpt else "",
        "checkpoint_exists": bool(ckpt),
        "checkpoint_sha256": sha256_file(ckpt) if ckpt else "",
        "eval_metrics_path": str(metrics_path) if metrics_path else "",
        "sample_image_path": sample_image(output_dir),
        "status": "completed" if metrics_path and ckpt else "missing",
        "notes": exp.get("notes", ""),
    }


def first_existing(output_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = output_dir / name
        if path.exists():
            return path
    return None


def load_phase12_rows() -> list[dict[str, str]]:
    return read_csv(PHASE12 / "final_result_registry.csv")
