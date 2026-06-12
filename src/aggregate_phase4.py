from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .utils import ensure_dir, load_config


PHASE4_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase4")
PHASE3_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase3")
PHASE2_CLEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_clean_phase2")

FIELDS = [
    "method",
    "sampling_ratio",
    "m",
    "n",
    "pattern_mode",
    "pattern_init",
    "warm_start",
    "freeze_generator_epochs",
    "lr_patterns",
    "pattern_tau",
    "pattern_tau_final",
    "lambda_pattern_energy",
    "lambda_pattern_decorrelation",
    "lambda_pattern_binary",
    "lambda_pattern_secrip",
    "lambda_pattern_contrast",
    "target_contrast",
    "model_psnr",
    "model_ssim",
    "model_mse",
    "model_rel_meas_err",
    "backproj_psnr",
    "backproj_ssim",
    "backproj_mse",
    "backproj_rel_meas_err",
    "delta_psnr",
    "delta_ssim",
    "score",
    "pattern_mean",
    "pattern_std",
    "row_std_mean",
    "binary_fraction_005_095",
    "mean_abs_offdiag_corr",
    "secant_rip_eval_loss",
    "checkpoint_best_ssim",
    "checkpoint_best_psnr",
    "checkpoint_best_score",
    "sample_image",
    "pattern_image",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Phase 4 learned-pattern results.")
    parser.add_argument("--phase4_dir", default=str(PHASE4_ROOT))
    parser.add_argument("--phase3_dir", default=str(PHASE3_ROOT))
    parser.add_argument("--phase2_clean_dir", default=str(PHASE2_CLEAN_ROOT))
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_config(run_dir: Path) -> dict:
    resolved = run_dir / "resolved_config.yaml"
    if resolved.exists():
        return load_config(resolved)
    return {}


def row_from_run(method: str, ratio: float, run_dir: Path, learned: bool) -> dict:
    metrics = read_json(run_dir / "eval_metrics.json") or read_json(run_dir / "best_score_metrics.json")
    config = read_config(run_dir)
    img_size = int(config.get("img_size", 64))
    n = img_size * img_size
    sampling_ratio = float(config.get("sampling_ratio", ratio))
    m = int(round(sampling_ratio * n))
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "method": method,
            "sampling_ratio": sampling_ratio,
            "m": m,
            "n": n,
            "pattern_mode": config.get("pattern_mode", "fixed" if not learned else ""),
            "pattern_init": config.get("pattern_init", ""),
            "warm_start": bool(
                config.get("load_generator_checkpoint")
                or config.get("load_discriminator_checkpoint")
                or config.get("load_pattern_checkpoint")
            ),
            "freeze_generator_epochs": config.get("freeze_generator_epochs", ""),
            "lr_patterns": config.get("lr_patterns", ""),
            "pattern_tau": config.get("pattern_tau", ""),
            "pattern_tau_final": config.get("pattern_tau_final", ""),
            "lambda_pattern_energy": config.get("lambda_pattern_energy", ""),
            "lambda_pattern_decorrelation": config.get("lambda_pattern_decorrelation", ""),
            "lambda_pattern_binary": config.get("lambda_pattern_binary", ""),
            "lambda_pattern_secrip": config.get("lambda_pattern_secrip", ""),
            "lambda_pattern_contrast": config.get("lambda_pattern_contrast", ""),
            "target_contrast": config.get("target_contrast", ""),
            "checkpoint_best_ssim": str(run_dir / "best_ssim.pt") if (run_dir / "best_ssim.pt").exists() else "",
            "checkpoint_best_psnr": str(run_dir / "best_psnr.pt") if (run_dir / "best_psnr.pt").exists() else "",
            "checkpoint_best_score": str(run_dir / "best_score.pt") if (run_dir / "best_score.pt").exists() else "",
            "sample_image": str(run_dir / "eval_samples" / "recon_grid.png")
            if (run_dir / "eval_samples" / "recon_grid.png").exists()
            else "",
            "pattern_image": str(run_dir / "eval_patterns" / "final_patterns.png")
            if (run_dir / "eval_patterns" / "final_patterns.png").exists()
            else "",
            "status": "ok" if metrics else "missing",
        }
    )
    if not metrics:
        return row
    back = metrics.get("backprojection", {})
    model = metrics.get("model", {})
    improve = metrics.get("improvement", {})
    pattern = metrics.get("pattern", {})
    ssim_weight = float(config.get("score_ssim_weight", 10.0))
    row.update(
        {
            "model_psnr": model.get("psnr", ""),
            "model_ssim": model.get("ssim", ""),
            "model_mse": model.get("mse", ""),
            "model_rel_meas_err": model.get("rel_meas_error", ""),
            "backproj_psnr": back.get("psnr", ""),
            "backproj_ssim": back.get("ssim", ""),
            "backproj_mse": back.get("mse", ""),
            "backproj_rel_meas_err": back.get("rel_meas_error", ""),
            "delta_psnr": improve.get("delta_psnr", ""),
            "delta_ssim": improve.get("delta_ssim", ""),
            "score": float(model.get("psnr", 0.0)) + ssim_weight * float(model.get("ssim", 0.0)),
            "pattern_mean": pattern.get("mean", ""),
            "pattern_std": pattern.get("std", ""),
            "row_std_mean": pattern.get("row_std_mean", ""),
            "binary_fraction_005_095": pattern.get("binary_fraction_005_095", ""),
            "mean_abs_offdiag_corr": pattern.get("mean_abs_offdiag_corr", ""),
            "secant_rip_eval_loss": pattern.get("secant_rip_eval_loss", ""),
        }
    )
    return row


def tuning_specs(phase4_dir: Path, phase3_dir: Path, phase2_dir: Path):
    return [
        ("Fixed Rademacher", 0.05, phase2_dir / "quick_5pct", False),
        ("Phase 3 Binary STE", 0.05, phase3_dir / "learned_binary_5pct", True),
        ("Phase 3 Continuous", 0.05, phase3_dir / "learned_continuous_5pct", True),
        ("Phase 4 Matched Binary", 0.05, phase4_dir / "matched_binary_5pct", True),
        ("Phase 4 Matched Binary Slow", 0.05, phase4_dir / "matched_binary_slow_5pct", True),
        ("Phase 4 Matched Binary No Freeze", 0.05, phase4_dir / "matched_binary_no_freeze_5pct", True),
        ("Phase 4 Continuous Contrast", 0.05, phase4_dir / "continuous_contrast_5pct", True),
        ("Phase 4 Continuous To Binary", 0.05, phase4_dir / "continuous_to_binary_5pct", True),
    ]


def best_sweep_specs(phase4_dir: Path, phase2_dir: Path):
    return [
        ("Fixed Rademacher", 0.02, phase2_dir / "quick_2pct", False),
        ("Fixed Rademacher", 0.05, phase2_dir / "quick_5pct", False),
        ("Fixed Rademacher", 0.10, phase2_dir / "quick_10pct", False),
        ("Phase 4 Best", 0.02, phase4_dir / "best_2pct", True),
        ("Phase 4 Best", 0.05, phase4_dir / "best_5pct", True),
        ("Phase 4 Best", 0.10, phase4_dir / "best_10pct", True),
    ]


def write_csv(rows: list[dict], path: Path, fields: list[str] = FIELDS) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def write_markdown(rows: list[dict], path: Path, title: str, cols: list[str]) -> None:
    lines = [f"# {title}", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stats_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "method",
        "sampling_ratio",
        "pattern_mode",
        "pattern_init",
        "pattern_mean",
        "pattern_std",
        "row_std_mean",
        "binary_fraction_005_095",
        "mean_abs_offdiag_corr",
        "secant_rip_eval_loss",
        "pattern_image",
        "status",
    ]
    write_csv([row for row in rows if row.get("pattern_mode") != "fixed"], path, fields)


def plot_bars(rows: list[dict], metric: str, ylabel: str, path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get(metric) != ""]
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [row["method"] for row in ok_rows]
        values = [float(row[metric]) for row in ok_rows]
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"Plot unavailable: {exc}\n", encoding="utf-8")


def plot_lines(rows: list[dict], metric: str, ylabel: str, path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get(metric) != ""]
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for method in sorted({row["method"] for row in ok_rows}):
            series = sorted(
                [row for row in ok_rows if row["method"] == method],
                key=lambda row: float(row["sampling_ratio"]),
            )
            ax.plot(
                [float(row["sampling_ratio"]) for row in series],
                [float(row[metric]) for row in series],
                marker="o",
                label=method,
            )
        ax.set_xlabel("sampling_ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"Plot unavailable: {exc}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase4_dir = ensure_dir(args.phase4_dir)
    phase3_dir = Path(args.phase3_dir)
    phase2_dir = Path(args.phase2_clean_dir)
    tuning_rows = [row_from_run(*spec) for spec in tuning_specs(phase4_dir, phase3_dir, phase2_dir)]
    sweep_rows = [row_from_run(*spec) for spec in best_sweep_specs(phase4_dir, phase2_dir)]

    write_csv(tuning_rows, phase4_dir / "phase4_tuning_results.csv")
    write_markdown(
        tuning_rows,
        phase4_dir / "phase4_tuning_results.md",
        "Phase 4 Tuning Results",
        ["method", "model_psnr", "model_ssim", "score", "pattern_std", "row_std_mean", "status"],
    )
    write_stats_csv(tuning_rows + sweep_rows, phase4_dir / "phase4_pattern_stats.csv")
    plot_bars(
        [row for row in tuning_rows if abs(float(row.get("sampling_ratio", 0.0)) - 0.05) < 1e-9],
        "model_psnr",
        "5% PSNR",
        phase4_dir / "phase4_fixed_vs_phase3_vs_phase4_5pct.png",
    )

    write_csv(sweep_rows, phase4_dir / "phase4_best_sweep_results.csv")
    write_markdown(
        sweep_rows,
        phase4_dir / "phase4_best_sweep_results.md",
        "Phase 4 Best Sweep Results",
        ["method", "sampling_ratio", "model_psnr", "model_ssim", "score", "status"],
    )
    plot_lines(sweep_rows, "model_psnr", "PSNR", phase4_dir / "phase4_best_sweep_psnr.png")
    plot_lines(sweep_rows, "model_ssim", "SSIM", phase4_dir / "phase4_best_sweep_ssim.png")
    print(f"Wrote Phase 4 aggregation to: {phase4_dir}")


if __name__ == "__main__":
    main()
