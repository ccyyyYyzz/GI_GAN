from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .utils import ensure_dir, load_config


PHASE5_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase5")
PHASE4_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase4")
PHASE2_CLEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_clean_phase2")

FIELDS = [
    "method",
    "sampling_ratio",
    "m",
    "n",
    "effective_A_mode",
    "pattern_mode",
    "pattern_init",
    "warm_start",
    "epoch0_psnr",
    "epoch0_ssim",
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
    "A_rel_fro_error",
    "A_max_abs_error",
    "A_cosine",
    "y_rel_error",
    "x_data_rel_error",
    "pattern_mean",
    "pattern_std",
    "row_std_mean",
    "binary_fraction_005_095",
    "mean_abs_offdiag_corr",
    "secant_rip_eval_loss",
    "checkpoint_best_score",
    "checkpoint_best_ssim",
    "sample_image",
    "pattern_image",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Phase 5 exact-operator results.")
    parser.add_argument("--phase5_dir", default=str(PHASE5_ROOT))
    parser.add_argument("--phase4_dir", default=str(PHASE4_ROOT))
    parser.add_argument("--phase2_clean_dir", default=str(PHASE2_CLEAN_ROOT))
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_config(run_dir: Path) -> dict:
    resolved = run_dir / "resolved_config.yaml"
    if resolved.exists():
        return load_config(resolved)
    return {}


def read_calibration(phase5_dir: Path) -> dict:
    return read_json(phase5_dir / "operator_calibration_5pct.json") or {}


def row_from_run(
    method: str,
    ratio: float,
    run_dir: Path,
    learned: bool,
    calibration: dict | None = None,
) -> dict:
    metrics = read_json(run_dir / "eval_metrics.json") or read_json(run_dir / "best_score_metrics.json")
    epoch0 = read_json(run_dir / "eval_epoch000_metrics.json")
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
            "effective_A_mode": config.get("effective_A_mode", "fixed" if not learned else ""),
            "pattern_mode": config.get("pattern_mode", "fixed" if not learned else ""),
            "pattern_init": config.get("pattern_init", ""),
            "warm_start": bool(
                config.get("load_generator_checkpoint")
                or config.get("load_discriminator_checkpoint")
                or config.get("load_pattern_checkpoint")
            ),
            "checkpoint_best_score": str(run_dir / "best_score.pt")
            if (run_dir / "best_score.pt").exists()
            else "",
            "checkpoint_best_ssim": str(run_dir / "best_ssim.pt")
            if (run_dir / "best_ssim.pt").exists()
            else "",
            "sample_image": str(run_dir / "eval_samples" / "recon_grid.png")
            if (run_dir / "eval_samples" / "recon_grid.png").exists()
            else "",
            "pattern_image": str(run_dir / "eval_patterns" / "final_patterns.png")
            if (run_dir / "eval_patterns" / "final_patterns.png").exists()
            else "",
            "status": "ok" if metrics else "missing",
        }
    )
    if epoch0:
        row["epoch0_psnr"] = epoch0.get("model", {}).get("psnr", "")
        row["epoch0_ssim"] = epoch0.get("model", {}).get("ssim", "")
    if calibration:
        for key in [
            "A_rel_fro_error",
            "A_max_abs_error",
            "A_cosine",
            "y_rel_error",
            "x_data_rel_error",
        ]:
            row[key] = calibration.get(key, "")
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
            "score": float(model.get("psnr", 0.0))
            + ssim_weight * float(model.get("ssim", 0.0)),
            "pattern_mean": pattern.get("mean", ""),
            "pattern_std": pattern.get("std", ""),
            "row_std_mean": pattern.get("row_std_mean", ""),
            "binary_fraction_005_095": pattern.get("binary_fraction_005_095", ""),
            "mean_abs_offdiag_corr": pattern.get("mean_abs_offdiag_corr", ""),
            "secant_rip_eval_loss": pattern.get("secant_rip_eval_loss", ""),
        }
    )
    return row


def tuning_specs(phase5_dir: Path, phase4_dir: Path, phase2_dir: Path):
    return [
        ("Fixed Rademacher", 0.05, phase2_dir / "quick_5pct", False, None),
        ("Phase 4 Best", 0.05, phase4_dir / "matched_binary_no_freeze_5pct", True, None),
        ("Phase 5 Exact Binary", 0.05, phase5_dir / "exact_binary_5pct", True, "calibration"),
        ("Phase 5 Exact Binary Slow", 0.05, phase5_dir / "exact_binary_slow_5pct", True, None),
        ("Phase 5 Exact Binary FreezeG", 0.05, phase5_dir / "exact_binary_freezeG_5pct", True, None),
        ("Phase 5 Centered Vs Exact", 0.05, phase5_dir / "centered_vs_exact_5pct", True, None),
    ]


def best_sweep_specs(phase5_dir: Path, phase2_dir: Path):
    return [
        ("Fixed Rademacher", 0.01, phase2_dir / "quick_1pct", False),
        ("Fixed Rademacher", 0.02, phase2_dir / "quick_2pct", False),
        ("Fixed Rademacher", 0.05, phase2_dir / "quick_5pct", False),
        ("Fixed Rademacher", 0.10, phase2_dir / "quick_10pct", False),
        ("Phase 5 Best", 0.01, phase5_dir / "best_1pct", True),
        ("Phase 5 Best", 0.02, phase5_dir / "best_2pct", True),
        ("Phase 5 Best", 0.05, phase5_dir / "best_5pct", True),
        ("Phase 5 Best", 0.10, phase5_dir / "best_10pct", True),
    ]


def extreme_specs(phase5_dir: Path):
    return [
        ("Fixed Rademacher 0.5%", 0.005, phase5_dir / "fixed_0p5pct", False),
        ("Phase 5 Exact 0.5%", 0.005, phase5_dir / "extreme_0p5pct", True),
    ]


def write_csv(rows: list[dict], path: Path, fields: list[str] = FIELDS) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
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


def plot_bars(rows: list[dict], metric: str, ylabel: str, path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get(metric) != ""]
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([row["method"] for row in ok_rows], [float(row[metric]) for row in ok_rows])
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


def write_pattern_stats(rows: list[dict], path: Path) -> None:
    fields = [
        "method",
        "sampling_ratio",
        "effective_A_mode",
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


def write_epoch0_equivalence(rows: list[dict], path: Path) -> None:
    fields = [
        "method",
        "effective_A_mode",
        "epoch0_psnr",
        "epoch0_ssim",
        "A_rel_fro_error",
        "A_max_abs_error",
        "A_cosine",
        "y_rel_error",
        "x_data_rel_error",
        "status",
    ]
    write_csv(rows, path, fields)


def write_noise_summary(phase5_dir: Path) -> None:
    sweep_root = phase5_dir / "noise_sweep"
    lines = ["# Phase 5 Noise Sweep Summary", ""]
    if not sweep_root.exists():
        lines.append("No noise sweep outputs found.")
        (phase5_dir / "phase5_noise_sweep_summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return
    found = False
    for csv_path in sorted(sweep_root.glob("*/noise_sweep_metrics.csv")):
        found = True
        lines.extend([f"## {csv_path.parent.name}", ""])
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        cols = ["noise_std", "model_psnr", "model_ssim", "model_rel_meas_err", "score", "status"]
        lines.append("|" + "|".join(cols) + "|")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in rows:
            lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
        lines.append("")
    if not found:
        lines.append("No noise sweep CSV files found.")
    (phase5_dir / "phase5_noise_sweep_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    phase5_dir = ensure_dir(args.phase5_dir)
    phase4_dir = Path(args.phase4_dir)
    phase2_dir = Path(args.phase2_clean_dir)
    calibration = read_calibration(phase5_dir)

    tuning_rows = []
    for method, ratio, run_dir, learned, cal_ref in tuning_specs(phase5_dir, phase4_dir, phase2_dir):
        tuning_rows.append(
            row_from_run(method, ratio, run_dir, learned, calibration if cal_ref else None)
        )
    sweep_rows = [row_from_run(*spec) for spec in best_sweep_specs(phase5_dir, phase2_dir)]
    extreme_rows = [row_from_run(*spec) for spec in extreme_specs(phase5_dir)]
    all_rows = tuning_rows + sweep_rows + extreme_rows

    write_csv(tuning_rows, phase5_dir / "phase5_tuning_results.csv")
    write_markdown(
        tuning_rows,
        phase5_dir / "phase5_tuning_results.md",
        "Phase 5 Tuning Results",
        ["method", "model_psnr", "model_ssim", "score", "epoch0_psnr", "A_rel_fro_error", "status"],
    )
    plot_bars(
        tuning_rows,
        "model_psnr",
        "5% PSNR",
        phase5_dir / "phase5_fixed_vs_phase4_vs_phase5_5pct.png",
    )
    write_epoch0_equivalence(tuning_rows, phase5_dir / "phase5_epoch0_equivalence.csv")
    write_pattern_stats(all_rows, phase5_dir / "phase5_pattern_stats.csv")

    write_csv(sweep_rows, phase5_dir / "phase5_best_sweep_results.csv")
    write_markdown(
        sweep_rows,
        phase5_dir / "phase5_best_sweep_results.md",
        "Phase 5 Best Sweep Results",
        ["method", "sampling_ratio", "model_psnr", "model_ssim", "score", "status"],
    )
    plot_lines(sweep_rows, "model_psnr", "PSNR", phase5_dir / "phase5_best_sweep_psnr.png")
    plot_lines(sweep_rows, "model_ssim", "SSIM", phase5_dir / "phase5_best_sweep_ssim.png")
    write_noise_summary(phase5_dir)
    print(f"Wrote Phase 5 aggregation to: {phase5_dir}")


if __name__ == "__main__":
    main()
