from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .utils import ensure_dir, load_config


PHASE3_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase3")
PHASE2_CLEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_clean_phase2")

FIELDS = [
    "method",
    "sampling_ratio",
    "m",
    "n",
    "pattern_mode",
    "use_learned_patterns",
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
    "pattern_mean",
    "pattern_std",
    "binary_fraction_005_095",
    "mean_abs_offdiag_corr",
    "secant_rip_eval_loss",
    "checkpoint",
    "sample_image",
    "pattern_image",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Phase 3 learned-pattern results.")
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


def build_row(method: str, ratio: float, run_dir: Path, learned: bool) -> dict:
    metrics = read_json(run_dir / "eval_metrics.json") or read_json(run_dir / "best_metrics.json")
    config = read_config(run_dir)
    img_size = int(config.get("img_size", 64))
    n = img_size * img_size
    m = int(round(float(config.get("sampling_ratio", ratio)) * n))
    checkpoint = run_dir / "best_ssim.pt"
    sample = run_dir / "eval_samples" / "recon_grid.png"
    pattern_image = run_dir / "eval_patterns" / "final_patterns.png"

    row = {field: "" for field in FIELDS}
    row.update(
        {
            "method": method,
            "sampling_ratio": float(config.get("sampling_ratio", ratio)),
            "m": m,
            "n": n,
            "pattern_mode": config.get("pattern_mode", "fixed" if not learned else ""),
            "use_learned_patterns": bool(config.get("use_learned_patterns", learned)),
            "checkpoint": str(checkpoint) if checkpoint.exists() else "",
            "sample_image": str(sample) if sample.exists() else "",
            "pattern_image": str(pattern_image) if pattern_image.exists() else "",
            "status": "ok" if metrics else "missing",
        }
    )
    if not metrics:
        return row
    back = metrics.get("backprojection", {})
    model = metrics.get("model", {})
    improve = metrics.get("improvement", {})
    pattern = metrics.get("pattern", {})
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
            "pattern_mean": pattern.get("mean", ""),
            "pattern_std": pattern.get("std", ""),
            "binary_fraction_005_095": pattern.get("binary_fraction_005_095", ""),
            "mean_abs_offdiag_corr": pattern.get("mean_abs_offdiag_corr", ""),
            "secant_rip_eval_loss": pattern.get("secant_rip_eval_loss", ""),
        }
    )
    return row


def main_specs(phase3_dir: Path, phase2_clean_dir: Path):
    return [
        ("Fixed Rademacher", 0.02, phase2_clean_dir / "quick_2pct", False),
        ("Fixed Rademacher", 0.05, phase2_clean_dir / "quick_5pct", False),
        ("Fixed Rademacher", 0.10, phase2_clean_dir / "quick_10pct", False),
        ("Learned Binary STE", 0.02, phase3_dir / "learned_binary_2pct", True),
        ("Learned Binary STE", 0.05, phase3_dir / "learned_binary_5pct", True),
        ("Learned Binary STE", 0.10, phase3_dir / "learned_binary_10pct", True),
        ("Learned Continuous", 0.05, phase3_dir / "learned_continuous_5pct", True),
    ]


def ablation_specs(phase3_dir: Path):
    return [
        ("Learned Binary STE", 0.05, phase3_dir / "learned_binary_5pct", True),
        ("No Secant-RIP", 0.05, phase3_dir / "binary_5pct_no_secrip", True),
        ("No Decorrelation", 0.05, phase3_dir / "binary_5pct_no_decorrelation", True),
        ("No Energy Constraint", 0.05, phase3_dir / "binary_5pct_no_energy", True),
    ]


def write_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_stats_csv(rows: list[dict], path: Path) -> None:
    stats_fields = [
        "method",
        "sampling_ratio",
        "pattern_mode",
        "pattern_mean",
        "pattern_std",
        "binary_fraction_005_095",
        "mean_abs_offdiag_corr",
        "secant_rip_eval_loss",
        "pattern_image",
        "status",
    ]
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_fields)
        writer.writeheader()
        for row in rows:
            if row.get("use_learned_patterns") in (True, "True", "true"):
                writer.writerow({field: row.get(field, "") for field in stats_fields})


def fmt(value) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def write_main_markdown(rows: list[dict], path: Path, title: str) -> None:
    cols = [
        "method",
        "sampling_ratio",
        "model_psnr",
        "model_ssim",
        "model_rel_meas_err",
        "pattern_mean",
        "mean_abs_offdiag_corr",
        "status",
    ]
    lines = [f"# {title}", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_lines(rows: list[dict], metric: str, ylabel: str, path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get(metric) != ""]
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for method in sorted({row["method"] for row in ok_rows}):
            series = sorted(
                [row for row in ok_rows if row["method"] == method],
                key=lambda r: float(r["sampling_ratio"]),
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
    except Exception:
        path.with_suffix(".txt").write_text("Plot unavailable.\n", encoding="utf-8")


def plot_bars(rows: list[dict], metric: str, ylabel: str, path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get(metric) != ""]
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [row["method"] for row in ok_rows]
        values = [float(row[metric]) for row in ok_rows]
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        path.with_suffix(".txt").write_text("Plot unavailable.\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    phase3_dir = ensure_dir(args.phase3_dir)
    phase2_clean_dir = Path(args.phase2_clean_dir)
    main_rows = [build_row(*spec) for spec in main_specs(phase3_dir, phase2_clean_dir)]
    ablation_rows = [build_row(*spec) for spec in ablation_specs(phase3_dir)]

    write_csv(main_rows, phase3_dir / "phase3_main_results.csv")
    write_main_markdown(main_rows, phase3_dir / "phase3_main_results.md", "Phase 3 Main Results")
    plot_lines(main_rows, "model_psnr", "PSNR", phase3_dir / "phase3_fixed_vs_learned_psnr.png")
    plot_lines(main_rows, "model_ssim", "SSIM", phase3_dir / "phase3_fixed_vs_learned_ssim.png")
    plot_lines(
        main_rows,
        "model_rel_meas_err",
        "relative measurement error",
        phase3_dir / "phase3_fixed_vs_learned_relmeaserr.png",
    )

    write_csv(ablation_rows, phase3_dir / "phase3_pattern_ablation_results.csv")
    write_main_markdown(
        ablation_rows,
        phase3_dir / "phase3_pattern_ablation_results.md",
        "Phase 3 Pattern Ablation Results",
    )
    plot_bars(ablation_rows, "model_psnr", "PSNR", phase3_dir / "phase3_pattern_ablation_psnr.png")
    plot_bars(ablation_rows, "model_ssim", "SSIM", phase3_dir / "phase3_pattern_ablation_ssim.png")
    plot_bars(
        ablation_rows,
        "mean_abs_offdiag_corr",
        "mean abs offdiag corr",
        phase3_dir / "phase3_pattern_ablation_corr.png",
    )
    write_stats_csv(main_rows + ablation_rows, phase3_dir / "phase3_pattern_stats.csv")
    print(f"Wrote Phase 3 aggregation to: {phase3_dir}")


if __name__ == "__main__":
    main()

