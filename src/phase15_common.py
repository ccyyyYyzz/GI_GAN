from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E_ROOT = Path("E:/ns_mc_gan_gi")
NOLEAK_DOWNLOADS = E_ROOT / "colab_downloads" / "noleak"
PHASE14 = E_ROOT / "outputs_phase14"
PHASE15 = E_ROOT / "outputs_phase15"
IMPORTED_NOLEAK = PHASE15 / "imported_noleak"


METHODS: list[dict[str, Any]] = [
    {
        "method_id": "mnist_hadamard5_full_colab",
        "download_dir": "mnist_hadamard5_full_colab",
        "display_name": "MNIST Hadamard 5%",
        "dataset": "MNIST",
        "measurement_family": "lowfreq_hadamard",
        "pattern_type": "lowfreq_hadamard",
        "sampling_ratio": 0.05,
        "noise_std": 0.01,
        "source": "Colab no-leak import",
        "run_type": "strict no-leak final",
        "notes": "Simple-domain sanity result.",
    },
    {
        "method_id": "fashion_hadamard5_full_colab",
        "download_dir": "fashion_hadamard5_full_colab",
        "display_name": "Fashion-MNIST Hadamard 5%",
        "dataset": "Fashion-MNIST",
        "measurement_family": "lowfreq_hadamard",
        "pattern_type": "lowfreq_hadamard",
        "sampling_ratio": 0.05,
        "noise_std": 0.01,
        "source": "Colab no-leak import",
        "run_type": "strict no-leak final",
        "notes": "Simple-domain sanity result.",
    },
    {
        "method_id": "scrambled_hadamard5_hq_noise001_colab",
        "download_dir": "scrambled_hadamard5_hq_noise001_colab",
        "display_name": "STL-10 Scrambled Hadamard 5%",
        "dataset": "STL-10",
        "measurement_family": "scrambled_hadamard",
        "pattern_type": "scrambled_hadamard",
        "sampling_ratio": 0.05,
        "noise_std": 0.01,
        "source": "Colab no-leak import",
        "run_type": "strict no-leak final",
        "notes": "Primary STL-10 5% strict no-leak method.",
    },
    {
        "method_id": "rademacher5_hq_noise001_colab",
        "download_dir": "rademacher5_hq_noise001_colab",
        "display_name": "STL-10 Rademacher 5%",
        "dataset": "STL-10",
        "measurement_family": "rademacher",
        "pattern_type": "rademacher",
        "sampling_ratio": 0.05,
        "noise_std": 0.01,
        "source": "Colab no-leak import",
        "run_type": "strict no-leak final",
        "notes": "Primary STL-10 5% strict no-leak method; exact A exported.",
    },
    {
        "method_id": "scrambled_hadamard10_full_noise001_colab",
        "download_dir": "scrambled_hadamard10_full_noise001_colab",
        "display_name": "STL-10 Scrambled Hadamard 10%",
        "dataset": "STL-10",
        "measurement_family": "scrambled_hadamard",
        "pattern_type": "scrambled_hadamard",
        "sampling_ratio": 0.10,
        "noise_std": 0.01,
        "source": "Colab no-leak import",
        "run_type": "strict no-leak final",
        "notes": "Primary STL-10 10% strict no-leak method.",
    },
    {
        "method_id": "rademacher10_full_noise001_colab",
        "download_dir": "rademacher10_full_noise001_colab",
        "display_name": "STL-10 Rademacher 10%",
        "dataset": "STL-10",
        "measurement_family": "rademacher",
        "pattern_type": "rademacher",
        "sampling_ratio": 0.10,
        "noise_std": 0.01,
        "source": "Colab no-leak import",
        "run_type": "strict no-leak final",
        "notes": "Primary STL-10 10% strict no-leak method; exact A exported.",
    },
]


FIELDS_REGISTRY = [
    "method_id",
    "display_name",
    "dataset",
    "sampling_ratio",
    "measurement_family",
    "pattern_type",
    "noise_std",
    "source",
    "run_type",
    "strict_noleak",
    "exact_A_available",
    "exact_A_path",
    "checkpoint_path",
    "checkpoint_sha256",
    "sha_verified",
    "eval_metrics_path",
    "psnr",
    "ssim",
    "mse",
    "backproj_psnr",
    "backproj_ssim",
    "delta_psnr",
    "delta_ssim",
    "rel_meas_err",
    "epochs",
    "run_scale",
    "threshold_type",
    "threshold_reached",
    "preferred_for_paper",
    "exclusion_reason",
    "notes",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
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
    if fields is None:
        fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value).replace("|", "/").replace("\n", " ")


def write_md_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(format_cell(row.get(field, "")) for field in fields) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    cols = "l" * len(fields)
    lines = [r"\begin{tabular}{" + cols + "}", r"\hline"]
    lines.append(" & ".join(escape_tex(field) for field in fields) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        lines.append(" & ".join(escape_tex(format_cell(row.get(field, ""))) for field in fields) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(value))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def method_by_id(method_id: str) -> dict[str, Any]:
    for method in METHODS:
        if method["method_id"] == method_id:
            return method
    raise KeyError(method_id)


def method_source_dir(method: dict[str, Any]) -> Path:
    return NOLEAK_DOWNLOADS / method["download_dir"]


def method_import_dir(method: dict[str, Any]) -> Path:
    return IMPORTED_NOLEAK / method["method_id"]


def find_large_zip(path: Path) -> Path | None:
    candidates = [p for p in path.glob("*.zip") if p.stat().st_size > 100 * 1024 * 1024]
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0] if candidates else None


def first_existing(base: Path, names: list[str]) -> Path | None:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def threshold_for(dataset: str, sampling_ratio: float) -> tuple[str, float, float]:
    dataset_l = str(dataset).lower()
    ratio = float(sampling_ratio)
    if "stl" in dataset_l and ratio >= 0.099:
        return "stl10_10pct", 22.0, 0.65
    if "stl" in dataset_l:
        return "stl10_5pct", 20.0, 0.60
    return "simple_domain_5pct", 25.0, 0.80


def load_metrics(output_dir: Path) -> dict[str, Any]:
    return read_json(output_dir / "eval_metrics.json")


def metric_value(metrics: dict[str, Any], section: str, key: str) -> float | str:
    value = as_float((metrics.get(section) or {}).get(key))
    return value if value is not None else ""


def checkpoint_path(output_dir: Path) -> Path | None:
    return first_existing(output_dir, ["last.pt", "best_hq.pt", "best_score.pt", "best_ssim.pt", "best_psnr.pt"])


def generated_sha_manifest(output_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(output_dir).as_posix()
        files.append({"path": rel, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"root": str(output_dir), "file_count": len(files), "files": files}


def manifest_sha_for(output_dir: Path, rel_path: str) -> str:
    manifest = read_json(output_dir / "sha256_manifest.json")
    for item in manifest.get("files", []):
        if item.get("path") == rel_path:
            return str(item.get("sha256", ""))
    return ""


def registry_row(method: dict[str, Any], output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or method_import_dir(method)
    metrics = load_metrics(output_dir)
    cfg = read_yaml(output_dir / "resolved_config.yaml")
    dataset = cfg.get("dataset_name") or method["dataset"]
    ratio = as_float(cfg.get("sampling_ratio")) or float(method["sampling_ratio"])
    threshold_type, threshold_psnr, threshold_ssim = threshold_for(str(dataset), ratio)
    psnr = metric_value(metrics, "model", "psnr")
    ssim = metric_value(metrics, "model", "ssim")
    ckpt = checkpoint_path(output_dir)
    exact_a = output_dir / "measurement_operator_exact.pt"
    ckpt_sha = sha256_file(ckpt) if ckpt and ckpt.exists() else ""
    rel_ckpt = ckpt.relative_to(output_dir).as_posix() if ckpt and ckpt.exists() else ""
    sha_verified = bool(ckpt_sha and manifest_sha_for(output_dir, rel_ckpt) == ckpt_sha)
    threshold_reached = (
        isinstance(psnr, float)
        and isinstance(ssim, float)
        and psnr >= threshold_psnr
        and ssim >= threshold_ssim
    )
    run_scale = cfg.get("phase14_run_scale") or cfg.get("phase10_run_scale") or cfg.get("run_scale", "")
    return {
        "method_id": method["method_id"],
        "display_name": method["display_name"],
        "dataset": method["dataset"],
        "sampling_ratio": ratio,
        "measurement_family": method["measurement_family"],
        "pattern_type": cfg.get("pattern_type", method["pattern_type"]),
        "noise_std": as_float(cfg.get("noise_std")) if cfg else method["noise_std"],
        "source": method["source"],
        "run_type": method["run_type"],
        "strict_noleak": True,
        "exact_A_available": exact_a.exists(),
        "exact_A_path": str(exact_a) if exact_a.exists() else "",
        "checkpoint_path": str(ckpt) if ckpt else "",
        "checkpoint_sha256": ckpt_sha,
        "sha_verified": sha_verified,
        "eval_metrics_path": str(output_dir / "eval_metrics.json") if (output_dir / "eval_metrics.json").exists() else "",
        "psnr": psnr,
        "ssim": ssim,
        "mse": metric_value(metrics, "model", "mse"),
        "backproj_psnr": metric_value(metrics, "backprojection", "psnr"),
        "backproj_ssim": metric_value(metrics, "backprojection", "ssim"),
        "delta_psnr": metric_value(metrics, "improvement", "delta_psnr"),
        "delta_ssim": metric_value(metrics, "improvement", "delta_ssim"),
        "rel_meas_err": metric_value(metrics, "model", "rel_meas_error"),
        "epochs": cfg.get("epochs", ""),
        "run_scale": run_scale,
        "threshold_type": threshold_type,
        "threshold_reached": threshold_reached,
        "preferred_for_paper": True,
        "exclusion_reason": "",
        "notes": method["notes"],
    }


def load_registry(path: Path | None = None) -> list[dict[str, str]]:
    return read_csv(path or (PHASE15 / "noleak_registry.csv"))


def row_for(registry: list[dict[str, Any]], method_id: str) -> dict[str, Any] | None:
    for row in registry:
        if row.get("method_id") == method_id:
            return row
    return None


def backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}_backup_{now_stamp()}")
    shutil.move(str(path), str(backup))
    return backup


def unzip_to(zip_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def numeric(value: Any) -> float:
    out = as_float(value)
    return float("nan") if out is None else out
