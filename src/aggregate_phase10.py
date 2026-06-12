from __future__ import annotations

import csv
import json
from pathlib import Path

from .checkpoint_utils import find_best_checkpoint
from .utils import ensure_dir, load_config


ROOT = Path("E:/ns_mc_gan_gi/outputs_phase10")
PHASE9_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase9")
CONFIG_ROOT = Path("configs/phase10")

FIELDS = [
    "method",
    "phase",
    "dataset_name",
    "sampling_ratio",
    "noise_std",
    "pattern_type",
    "matrix_normalization",
    "hadamard_include_dc",
    "backprojection_mode",
    "model_type",
    "epochs_target",
    "epochs_actual",
    "limit_train_samples",
    "limit_val_samples",
    "is_short_train",
    "backproj_psnr",
    "backproj_ssim",
    "backproj_mse",
    "model_psnr",
    "model_ssim",
    "model_mse",
    "model_rel_meas_err",
    "rel_meas_err_unclamped",
    "rel_meas_err_clamped",
    "delta_model_vs_backproj_psnr",
    "delta_model_vs_backproj_ssim",
    "hq_score",
    "reaches_stl10_10pct_hq",
    "reaches_stl10_5pct_hq",
    "reaches_simple_domain_hq",
    "checkpoint",
    "sample_image",
    "convergence_summary",
    "continue_training_recommended",
    "status",
]

RUNS = [
    ("hadamard10_full_noise001", CONFIG_ROOT / "hadamard10_full_noise001.yaml"),
    ("hadamard10_full_nonoise", CONFIG_ROOT / "hadamard10_full_nonoise.yaml"),
    ("hadamard5_full_noise001", CONFIG_ROOT / "hadamard5_full_noise001.yaml"),
    ("hadamard5_medium_noise001", CONFIG_ROOT / "hadamard5_medium_noise001.yaml"),
    ("rademacher10_full_noise001", CONFIG_ROOT / "rademacher10_full_noise001.yaml"),
    ("scrambled_hadamard10_full_noise001", CONFIG_ROOT / "scrambled_hadamard10_full_noise001.yaml"),
    ("lowfreq_no_dc10_control", CONFIG_ROOT / "lowfreq_no_dc10_control.yaml"),
    ("mnist_hadamard5_full", CONFIG_ROOT / "mnist_hadamard5_full.yaml"),
    ("fashion_hadamard5_full", CONFIG_ROOT / "fashion_hadamard5_full.yaml"),
    ("cifar10_gray_hadamard10_medium", CONFIG_ROOT / "cifar10_gray_hadamard10_medium.yaml"),
    ("continuous_physical_hq10_full", CONFIG_ROOT / "continuous_physical_hq10_full.yaml"),
]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_last_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def as_float(value):
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def hq_score(config: dict, model: dict) -> float | str:
    psnr = as_float(model.get("psnr"))
    ssim = as_float(model.get("ssim"))
    if psnr is None or ssim is None:
        return ""
    rel = as_float(model.get("rel_meas_error")) or 0.0
    return psnr + float(config.get("score_ssim_weight", 20.0)) * ssim - float(
        config.get("score_relmeas_weight", 0.0)
    ) * rel


def hq_flags(config: dict, model: dict) -> tuple[bool, bool, bool]:
    psnr = as_float(model.get("psnr"))
    ssim = as_float(model.get("ssim"))
    if psnr is None or ssim is None:
        return False, False, False
    dataset = str(config.get("dataset_name", "stl10")).lower()
    ratio = float(config.get("sampling_ratio", 0.0) or 0.0)
    stl10_10 = dataset == "stl10" and ratio >= 0.099 and psnr >= 22.0 and ssim >= 0.65
    stl10_5 = dataset == "stl10" and ratio <= 0.051 and psnr >= 20.0 and ssim >= 0.60
    simple = dataset in {"mnist", "fashion_mnist"} and psnr >= 25.0 and ssim >= 0.80
    return stl10_10, stl10_5, simple


def sample_for(output_dir: Path) -> str:
    candidates = [
        output_dir / "eval_samples" / "recon_grid.png",
        output_dir / "samples" / "epoch_000.png",
    ]
    samples = sorted((output_dir / "samples").glob("epoch_*.png")) if (output_dir / "samples").exists() else []
    if samples:
        candidates.append(samples[-1])
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def epoch_actual(output_dir: Path) -> str:
    row = read_last_csv(output_dir / "per_epoch_metrics.csv") or read_last_csv(output_dir / "eval_history.csv")
    return str(row.get("epoch", "")) if row else ""


def convergence_for(output_dir: Path) -> tuple[str, str]:
    md = output_dir / "convergence_summary.md"
    js = read_json(output_dir / "convergence_summary.json")
    return (str(md) if md.exists() else "", str(js.get("continue_training_recommended", "")) if js else "")


def row_for_run(method: str, config_path: Path) -> dict:
    config = load_config(config_path) if config_path.exists() else {}
    output_dir = Path(config.get("output_dir", ROOT / method))
    metrics = read_json(output_dir / "eval_metrics.json") or read_json(output_dir / "val_metrics_latest.json")
    back = metrics.get("backprojection", {}) if metrics else {}
    model = metrics.get("model", {}) if metrics else {}
    improve = metrics.get("improvement", {}) if metrics else {}
    stl10_10, stl10_5, simple = hq_flags(config, model) if metrics else (False, False, False)
    ckpt = find_best_checkpoint(output_dir)
    conv_path, cont = convergence_for(output_dir)
    return {
        "method": method,
        "phase": "phase10",
        "dataset_name": config.get("dataset_name", ""),
        "sampling_ratio": config.get("sampling_ratio", ""),
        "noise_std": config.get("noise_std", ""),
        "pattern_type": config.get("pattern_type", ""),
        "matrix_normalization": config.get("matrix_normalization", ""),
        "hadamard_include_dc": config.get("hadamard_include_dc", ""),
        "backprojection_mode": config.get("backprojection_mode", ""),
        "model_type": config.get("model_type", ""),
        "epochs_target": config.get("epochs", ""),
        "epochs_actual": epoch_actual(output_dir),
        "limit_train_samples": config.get("limit_train_samples", ""),
        "limit_val_samples": config.get("limit_val_samples", ""),
        "is_short_train": bool(config.get("phase9_run_scope", "") == "short_train" or int(config.get("epochs", 0) or 0) < 20),
        "backproj_psnr": back.get("psnr", ""),
        "backproj_ssim": back.get("ssim", ""),
        "backproj_mse": back.get("mse", ""),
        "model_psnr": model.get("psnr", ""),
        "model_ssim": model.get("ssim", ""),
        "model_mse": model.get("mse", ""),
        "model_rel_meas_err": model.get("rel_meas_error", ""),
        "rel_meas_err_unclamped": model.get("rel_meas_err_unclamped", ""),
        "rel_meas_err_clamped": model.get("rel_meas_err_clamped", ""),
        "delta_model_vs_backproj_psnr": improve.get("delta_psnr", ""),
        "delta_model_vs_backproj_ssim": improve.get("delta_ssim", ""),
        "hq_score": hq_score(config, model) if metrics else "",
        "reaches_stl10_10pct_hq": stl10_10,
        "reaches_stl10_5pct_hq": stl10_5,
        "reaches_simple_domain_hq": simple,
        "checkpoint": "" if ckpt is None else str(ckpt),
        "sample_image": sample_for(output_dir),
        "convergence_summary": conv_path,
        "continue_training_recommended": cont,
        "status": "completed" if metrics else "missing",
    }


def phase9_rows() -> list[dict]:
    path = PHASE9_ROOT / "phase9_results.csv"
    if not path.exists():
        return [
            {
                **{field: "" for field in FIELDS},
                "method": "phase9_results",
                "phase": "phase9",
                "status": "missing",
            }
        ]
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for src in csv.DictReader(f):
            row = {field: "" for field in FIELDS}
            for field in FIELDS:
                if field in src:
                    row[field] = src[field]
            row["phase"] = "phase9"
            row["epochs_target"] = src.get("epochs_target", "")
            row["epochs_actual"] = src.get("epochs_actual", "")
            row["backproj_mse"] = src.get("backproj_mse", "")
            row["delta_model_vs_backproj_psnr"] = ""
            row["delta_model_vs_backproj_ssim"] = ""
            row["is_short_train"] = "true"
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict], path: Path) -> None:
    headers = FIELDS
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(h, "")).replace("|", "/") for h in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def completed(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "completed"]


def plot_bar(rows: list[dict], key: str, path: Path, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt

        filtered = [row for row in completed(rows) if as_float(row.get(key)) is not None]
        labels = [f"{row['phase']}:{row['method']}" for row in filtered]
        values = [float(row[key]) for row in filtered]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4.5))
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"Could not render plot: {exc}\n", encoding="utf-8")


def plot_backproj_vs_model(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        filtered = [
            row
            for row in completed(rows)
            if as_float(row.get("backproj_psnr")) is not None and as_float(row.get("model_psnr")) is not None
        ]
        labels = [f"{row['phase']}:{row['method']}" for row in filtered]
        back = [float(row["backproj_psnr"]) for row in filtered]
        model = [float(row["model_psnr"]) for row in filtered]
        xs = range(len(labels))
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.5))
        ax.bar([x - 0.18 for x in xs], back, width=0.36, label="backprojection")
        ax.bar([x + 0.18 for x in xs], model, width=0.36, label="model")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("PSNR")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"Could not render plot: {exc}\n", encoding="utf-8")


def main() -> None:
    ensure_dir(ROOT)
    rows = phase9_rows()
    rows.extend(row_for_run(method, config_path) for method, config_path in RUNS)
    write_csv(rows, ROOT / "phase10_results.csv")
    write_md(rows, ROOT / "phase10_results.md")
    plot_bar(rows, "model_psnr", ROOT / "phase10_psnr.png", "model PSNR")
    plot_bar(rows, "model_ssim", ROOT / "phase10_ssim.png", "model SSIM")
    plot_bar(rows, "hq_score", ROOT / "phase10_hq_score.png", "HQ score")
    plot_bar(rows, "model_rel_meas_err", ROOT / "phase10_relmeaserr.png", "relative measurement error")
    plot_backproj_vs_model(rows, ROOT / "phase10_backproj_vs_model.png")
    print(f"Phase 10 aggregate written to: {ROOT / 'phase10_results.csv'}")


if __name__ == "__main__":
    main()
