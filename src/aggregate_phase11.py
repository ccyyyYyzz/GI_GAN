from __future__ import annotations

from pathlib import Path

from .checkpoint_utils import find_best_checkpoint
from .phase11_common import (
    CONFIG10,
    CONFIG11,
    ROOT10,
    ROOT11,
    as_float,
    hq_score,
    last_epoch,
    plot_bar,
    read_convergence,
    read_csv_rows,
    read_metrics_for_output,
    safe_copy,
    threshold_flags,
    write_csv_rows,
    write_md_table,
)
from .utils import ensure_dir, load_config


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
    "run_scale",
    "is_short_train",
    "backproj_psnr",
    "backproj_ssim",
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
    "convergence_continue_recommended",
    "checkpoint",
    "sample_image",
    "status",
]

PHASE11_METHODS = [
    ("hadamard10_seed43", CONFIG11 / "hadamard10_seed43.yaml", ROOT11 / "hadamard10_seed43"),
    ("hadamard10_seed44", CONFIG11 / "hadamard10_seed44.yaml", ROOT11 / "hadamard10_seed44"),
    ("hadamard5_push_hq", CONFIG11 / "hadamard5_push_hq.yaml", ROOT11 / "hadamard5_push_hq"),
    ("hadamard10_continue_noise001", CONFIG11 / "hadamard10_continue_noise001.yaml", ROOT10 / "hadamard10_full_noise001"),
    ("hadamard5_continue_noise001", CONFIG11 / "hadamard5_continue_noise001.yaml", ROOT10 / "hadamard5_medium_noise001"),
    ("hadamard5_confirm_full_noise001", CONFIG11 / "hadamard5_confirm_full_noise001.yaml", ROOT11 / "hadamard5_confirm_full_noise001"),
]


def sample_for(output_dir: Path) -> str:
    for path in [
        output_dir / "eval_samples" / "recon_grid.png",
        output_dir / "samples" / "epoch_000.png",
    ]:
        if path.exists():
            return str(path)
    samples = sorted((output_dir / "samples").glob("epoch_*.png")) if (output_dir / "samples").exists() else []
    return str(samples[-1]) if samples else ""


def row_from_config(method: str, phase: str, config_path: Path, output_dir: Path) -> dict:
    config = load_config(config_path) if config_path.exists() else {}
    metrics = read_metrics_for_output(output_dir)
    back = metrics.get("backprojection", {}) if metrics else {}
    model = metrics.get("model", {}) if metrics else {}
    improve = metrics.get("improvement", {}) if metrics else {}
    stl10_10, stl10_5, simple = threshold_flags(config, model) if metrics else (False, False, False)
    conv = read_convergence(output_dir)
    ckpt = find_best_checkpoint(output_dir)
    hq = hq_score(config, model) if metrics else ""
    return {
        "method": method,
        "phase": phase,
        "dataset_name": config.get("dataset_name", ""),
        "sampling_ratio": config.get("sampling_ratio", ""),
        "noise_std": config.get("noise_std", ""),
        "pattern_type": config.get("pattern_type", ""),
        "matrix_normalization": config.get("matrix_normalization", ""),
        "hadamard_include_dc": config.get("hadamard_include_dc", ""),
        "backprojection_mode": config.get("backprojection_mode", ""),
        "model_type": config.get("model_type", ""),
        "epochs_target": config.get("epochs", ""),
        "epochs_actual": last_epoch(output_dir, fallback=0) or "",
        "limit_train_samples": config.get("limit_train_samples", ""),
        "limit_val_samples": config.get("limit_val_samples", ""),
        "run_scale": config.get("phase11_run_scale", config.get("phase10_run_scale", "")),
        "is_short_train": bool(int(config.get("epochs", 0) or 0) < 20),
        "backproj_psnr": back.get("psnr", ""),
        "backproj_ssim": back.get("ssim", ""),
        "model_psnr": model.get("psnr", ""),
        "model_ssim": model.get("ssim", ""),
        "model_mse": model.get("mse", ""),
        "model_rel_meas_err": model.get("rel_meas_error", ""),
        "rel_meas_err_unclamped": model.get("rel_meas_err_unclamped", ""),
        "rel_meas_err_clamped": model.get("rel_meas_err_clamped", ""),
        "delta_model_vs_backproj_psnr": improve.get("delta_psnr", ""),
        "delta_model_vs_backproj_ssim": improve.get("delta_ssim", ""),
        "hq_score": hq if hq is not None else "",
        "reaches_stl10_10pct_hq": stl10_10,
        "reaches_stl10_5pct_hq": stl10_5,
        "reaches_simple_domain_hq": simple,
        "convergence_continue_recommended": conv.get("continue_training_recommended", ""),
        "checkpoint": "" if ckpt is None else str(ckpt),
        "sample_image": sample_for(output_dir),
        "status": "completed" if metrics else "missing",
    }


def phase10_rows() -> list[dict]:
    rows = []
    for row in read_csv_rows(ROOT10 / "phase10_results.csv"):
        out = {field: "" for field in FIELDS}
        for field in FIELDS:
            if field in row:
                out[field] = row[field]
        rows.append(out)
    return rows


def best_row(rows: list[dict]) -> dict | None:
    completed = [row for row in rows if row.get("status") == "completed" and as_float(row.get("hq_score")) is not None]
    return max(completed, key=lambda row: float(row["hq_score"]), default=None)


def threshold_status(rows: list[dict]) -> str:
    best = best_row(rows)
    stl10_10 = any(str(row.get("reaches_stl10_10pct_hq")).lower() == "true" for row in rows if row.get("status") == "completed")
    stl10_5 = any(str(row.get("reaches_stl10_5pct_hq")).lower() == "true" for row in rows if row.get("status") == "completed")
    simple = any(str(row.get("reaches_simple_domain_hq")).lower() == "true" for row in rows if row.get("status") == "completed")
    lines = [
        "# Phase 11 Threshold Status",
        "",
        f"- best_method: {best.get('method') if best else 'missing'}",
        f"- best_psnr: {best.get('model_psnr') if best else 'missing'}",
        f"- best_ssim: {best.get('model_ssim') if best else 'missing'}",
        f"- best_hq_score: {best.get('hq_score') if best else 'missing'}",
        f"- stl10_10pct_hq_reached: {stl10_10}",
        f"- stl10_5pct_hq_reached: {stl10_5}",
        f"- simple_domain_hq_reached: {simple}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(ROOT11)
    rows = phase10_rows()
    rows.extend(row_from_config(method, "phase11", config_path, output_dir) for method, config_path, output_dir in PHASE11_METHODS)
    write_csv_rows(rows, ROOT11 / "phase11_summary.csv", FIELDS)
    write_md_table(rows, ROOT11 / "phase11_summary.md", FIELDS)
    plot_bar(rows, "model_psnr", ROOT11 / "phase11_psnr.png", "model PSNR")
    plot_bar(rows, "model_ssim", ROOT11 / "phase11_ssim.png", "model SSIM")
    plot_bar(rows, "hq_score", ROOT11 / "phase11_hq_score.png", "HQ score")
    (ROOT11 / "phase11_threshold_status.md").write_text(threshold_status(rows), encoding="utf-8")
    print(f"Phase 11 summary written to: {ROOT11 / 'phase11_summary.csv'}")


if __name__ == "__main__":
    main()
