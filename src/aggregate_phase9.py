from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from .utils import ensure_dir, load_config


ROOT = Path("E:/ns_mc_gan_gi/outputs_phase9")
CONFIG_ROOT = Path("configs/phase9")

FIELDS = [
    "method",
    "dataset_name",
    "sampling_ratio",
    "noise_std",
    "pattern_type",
    "matrix_normalization",
    "hadamard_include_dc",
    "backprojection_mode",
    "model_type",
    "epochs_actual",
    "limit_train_samples",
    "limit_val_samples",
    "backproj_psnr",
    "backproj_ssim",
    "model_psnr",
    "model_ssim",
    "model_mse",
    "model_rel_meas_err",
    "rel_meas_err_unclamped",
    "rel_meas_err_clamped",
    "hq_score",
    "reaches_stl10_10pct_hq",
    "reaches_stl10_5pct_hq",
    "reaches_simple_domain_hq",
    "checkpoint",
    "sample_image",
    "status",
]

RUNS = [
    ("hadamard10_probe_nonoise", CONFIG_ROOT / "hadamard10_probe_nonoise.yaml"),
    ("hadamard10_probe_noise001", CONFIG_ROOT / "hadamard10_probe_noise001.yaml"),
    ("rademacher10_probe_noise001", CONFIG_ROOT / "rademacher10_probe_noise001.yaml"),
    ("scrambled_hadamard10_probe_noise001", CONFIG_ROOT / "scrambled_hadamard10_probe_noise001.yaml"),
    ("hadamard5_probe_noise001", CONFIG_ROOT / "hadamard5_probe_noise001.yaml"),
    ("fashion_hadamard5_hq", CONFIG_ROOT / "fashion_hadamard5_hq.yaml"),
    ("mnist_hadamard5_hq", CONFIG_ROOT / "mnist_hadamard5_hq.yaml"),
]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def checkpoint_for(output_dir: Path) -> str:
    for name in ["best_hq.pt", "best_score.pt", "best_ssim.pt", "last.pt", "best_overfit.pt"]:
        p = output_dir / name
        if p.exists():
            return str(p)
    return ""


def sample_for(output_dir: Path) -> str:
    for p in [
        output_dir / "eval_samples" / "recon_grid.png",
        output_dir / "overfit_examples_best.png",
        output_dir / "samples" / "epoch_000.png",
    ]:
        if p.exists():
            return str(p)
    samples = sorted((output_dir / "samples").glob("epoch_*.png")) if (output_dir / "samples").exists() else []
    return str(samples[-1]) if samples else ""


def hq_flags(config: dict, model: dict) -> tuple[bool, bool, bool]:
    psnr = float(model.get("psnr", float("nan")))
    ssim = float(model.get("ssim", float("nan")))
    dataset = str(config.get("dataset_name", "stl10")).lower()
    ratio = float(config.get("sampling_ratio", 0.0))
    stl10_10 = dataset == "stl10" and ratio >= 0.10 and psnr >= 22.0 and ssim >= 0.65
    stl10_5 = dataset == "stl10" and ratio <= 0.051 and psnr >= 20.0 and ssim >= 0.60
    simple = dataset in {"mnist", "fashion_mnist"} and psnr >= 25.0 and ssim >= 0.80
    return stl10_10, stl10_5, simple


def row_for_run(method: str, config_path: Path) -> dict:
    config = load_config(config_path) if config_path.exists() else {}
    output_dir = Path(config.get("output_dir", ROOT / method))
    metrics_path = output_dir / "eval_metrics.json"
    latest_path = output_dir / "val_metrics_latest.json"
    metrics = read_json(metrics_path) or read_json(latest_path)
    history = read_last_csv(output_dir / "eval_history.csv")
    back = metrics.get("backprojection", {}) if metrics else {}
    model = metrics.get("model", {}) if metrics else {}
    stl10_10, stl10_5, simple = hq_flags(config, model) if metrics else (False, False, False)
    hq_score = ""
    if metrics:
        hq_score = (
            float(model.get("psnr", 0.0))
            + float(config.get("score_ssim_weight", 20.0)) * float(model.get("ssim", 0.0))
            - float(config.get("score_relmeas_weight", 0.0)) * float(model.get("rel_meas_error", 0.0))
        )
    return {
        "method": method,
        "dataset_name": config.get("dataset_name", ""),
        "sampling_ratio": config.get("sampling_ratio", ""),
        "noise_std": config.get("noise_std", ""),
        "pattern_type": config.get("pattern_type", ""),
        "matrix_normalization": config.get("matrix_normalization", ""),
        "hadamard_include_dc": config.get("hadamard_include_dc", ""),
        "backprojection_mode": config.get("backprojection_mode", ""),
        "model_type": config.get("model_type", ""),
        "epochs_actual": history.get("epoch", ""),
        "limit_train_samples": config.get("limit_train_samples", ""),
        "limit_val_samples": config.get("limit_val_samples", ""),
        "backproj_psnr": back.get("psnr", ""),
        "backproj_ssim": back.get("ssim", ""),
        "model_psnr": model.get("psnr", ""),
        "model_ssim": model.get("ssim", ""),
        "model_mse": model.get("mse", ""),
        "model_rel_meas_err": model.get("rel_meas_error", ""),
        "rel_meas_err_unclamped": model.get("rel_meas_err_unclamped", ""),
        "rel_meas_err_clamped": model.get("rel_meas_err_clamped", ""),
        "hq_score": hq_score,
        "reaches_stl10_10pct_hq": stl10_10,
        "reaches_stl10_5pct_hq": stl10_5,
        "reaches_simple_domain_hq": simple,
        "checkpoint": checkpoint_for(output_dir),
        "sample_image": sample_for(output_dir),
        "status": "completed" if metrics else "missing",
    }


def sanity_row() -> dict:
    path = ROOT / "sanity_hadamard" / "hadamard_sanity.json"
    data = read_json(path)
    status = data.get("status", "missing") if data else "missing"
    return {
        "method": "sanity_hadamard",
        "dataset_name": "stl10",
        "sampling_ratio": "",
        "noise_std": 0.0,
        "pattern_type": "hadamard",
        "matrix_normalization": "orthonormal_rows",
        "hadamard_include_dc": True,
        "backprojection_mode": "hadamard_zero_filled",
        "model_type": "none",
        "epochs_actual": "",
        "limit_train_samples": "",
        "limit_val_samples": "",
        "backproj_psnr": "",
        "backproj_ssim": "",
        "model_psnr": "",
        "model_ssim": "",
        "model_mse": "",
        "model_rel_meas_err": data.get("full_sampling_rel_error", "") if data else "",
        "rel_meas_err_unclamped": "",
        "rel_meas_err_clamped": "",
        "hq_score": "",
        "reaches_stl10_10pct_hq": False,
        "reaches_stl10_5pct_hq": False,
        "reaches_simple_domain_hq": False,
        "checkpoint": "",
        "sample_image": "",
        "status": status,
    }


def overfit_row() -> dict:
    output_dir = ROOT / "overfit_hadamard_10pct"
    row = read_last_csv(output_dir / "overfit_metrics.csv")
    config = load_config(CONFIG_ROOT / "overfit_hadamard_10pct.yaml")
    psnr = float(row.get("train_psnr", "nan")) if row else float("nan")
    ssim = float(row.get("train_ssim", "nan")) if row else float("nan")
    return {
        "method": "overfit_hadamard_10pct",
        "dataset_name": config.get("dataset_name", ""),
        "sampling_ratio": config.get("sampling_ratio", ""),
        "noise_std": config.get("noise_std", ""),
        "pattern_type": config.get("pattern_type", ""),
        "matrix_normalization": config.get("matrix_normalization", ""),
        "hadamard_include_dc": config.get("hadamard_include_dc", ""),
        "backprojection_mode": config.get("backprojection_mode", ""),
        "model_type": config.get("model_type", ""),
        "epochs_actual": row.get("epoch", "") if row else "",
        "limit_train_samples": config.get("limit_train_samples", ""),
        "limit_val_samples": config.get("limit_val_samples", ""),
        "backproj_psnr": "",
        "backproj_ssim": "",
        "model_psnr": row.get("train_psnr", "") if row else "",
        "model_ssim": row.get("train_ssim", "") if row else "",
        "model_mse": row.get("train_mse", "") if row else "",
        "model_rel_meas_err": row.get("rel_meas_error", "") if row else "",
        "rel_meas_err_unclamped": row.get("rel_meas_err_unclamped", "") if row else "",
        "rel_meas_err_clamped": row.get("rel_meas_err_clamped", "") if row else "",
        "hq_score": "",
        "reaches_stl10_10pct_hq": False,
        "reaches_stl10_5pct_hq": False,
        "reaches_simple_domain_hq": False,
        "checkpoint": checkpoint_for(output_dir),
        "sample_image": sample_for(output_dir),
        "status": "completed" if row else "missing",
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict], path: Path) -> None:
    headers = FIELDS
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(h, "")) for h in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict], key: str, path: Path, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt

        filtered = [row for row in rows if row.get("status") == "completed" and row.get(key) not in {"", None}]
        labels = [row["method"] for row in filtered]
        values = [float(row[key]) for row in filtered]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4))
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"Could not render plot: {exc}\n", encoding="utf-8")


def main() -> None:
    ensure_dir(ROOT)
    rows = [sanity_row(), overfit_row()]
    rows.extend(row_for_run(method, config_path) for method, config_path in RUNS)
    write_csv(rows, ROOT / "phase9_results.csv")
    write_md(rows, ROOT / "phase9_results.md")

    src = ROOT / "sanity_hadamard" / "backproj_quality_table.csv"
    if src.exists():
        shutil.copyfile(src, ROOT / "phase9_backprojection_quality.csv")
    else:
        (ROOT / "phase9_backprojection_quality.csv").write_text("status\nmissing\n", encoding="utf-8")

    plot(rows, "model_psnr", ROOT / "phase9_psnr.png", "PSNR")
    plot(rows, "model_ssim", ROOT / "phase9_ssim.png", "SSIM")
    plot(rows, "hq_score", ROOT / "phase9_hq_score.png", "HQ score")
    print(f"Phase 9 aggregate written to: {ROOT / 'phase9_results.csv'}")


if __name__ == "__main__":
    main()
