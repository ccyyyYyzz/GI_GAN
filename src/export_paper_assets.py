from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import ensure_dir


PHASE5_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase5")
PHASE6_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase6")
ASSET_ROOT = PHASE6_ROOT / "paper_assets"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_fig(fig, path_base: Path) -> None:
    ensure_dir(path_base.parent)
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_sampling_sweep(rows: list[dict], path_base: Path) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method in sorted({row["method"] for row in ok}):
        part = sorted([row for row in ok if row["method"] == method], key=lambda r: float(r["sampling_ratio"]))
        ax.plot(
            [float(row["sampling_ratio"]) * 100.0 for row in part],
            [float(row["score"]) for row in part],
            marker="o",
            label=method,
        )
    ax.set_xlabel("Sampling ratio (%)")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, path_base)


def plot_noise_sweep(path_base: Path) -> None:
    root = PHASE5_ROOT / "noise_sweep"
    csvs = sorted(root.glob("*/noise_sweep_metrics.csv"))
    if not csvs:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for csv_path in csvs:
        rows = read_csv(csv_path)
        ok = [row for row in rows if row.get("status") == "ok"]
        if not ok:
            continue
        ax.plot(
            [float(row["noise_std"]) for row in ok],
            [float(row["score"]) for row in ok],
            marker="o",
            label=csv_path.parent.name,
        )
    ax.set_xlabel("Noise std")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, path_base)


def plot_phase6_controls(rows: list[dict], path_base: Path) -> None:
    ok = [row for row in rows if row.get("status") == "ok" and row.get("score") not in ("", None)]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(ok) * 1.1), 4.5))
    labels = [row["method"] for row in ok]
    scores = [float(row["score"]) for row in ok]
    ax.bar(range(len(ok)), scores)
    ax.set_ylabel("Score")
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    save_fig(fig, path_base)


def write_latex(rows: list[dict], path: Path, cols: list[str]) -> None:
    ensure_dir(path.parent)
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline"]
    lines.append(" & ".join(cols).replace("_", "\\_") + " \\\\")
    lines.append("\\hline")
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            try:
                value = f"{float(value):.4f}"
            except Exception:
                value = str(value or "missing")
            values.append(value.replace("_", "\\_"))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def main() -> None:
    asset_root = ensure_dir(ASSET_ROOT)
    sweep_rows = read_csv(PHASE5_ROOT / "phase5_best_sweep_results.csv")
    control_rows = read_csv(PHASE6_ROOT / "phase6_control_results.csv")
    diag_rows = read_csv(PHASE6_ROOT / "phase6_pattern_diagnostics.csv")
    noise_rows = []
    for csv_path in sorted((PHASE5_ROOT / "noise_sweep").glob("*/noise_sweep_metrics.csv")):
        for row in read_csv(csv_path):
            row = dict(row)
            row["method"] = csv_path.parent.name
            noise_rows.append(row)

    plot_sampling_sweep(sweep_rows, asset_root / "fig_sampling_sweep_phase5")
    plot_noise_sweep(asset_root / "fig_noise_sweep_phase5")
    plot_phase6_controls(control_rows, asset_root / "fig_phase6_controls")
    copy_if_exists(
        PHASE6_ROOT / "pattern_trainable_alpha1_5pct" / "eval_pattern_diagnostics" / "pattern_change_grid.png",
        asset_root / "fig_pattern_change_grid.png",
    )
    copy_if_exists(
        PHASE6_ROOT / "pattern_trainable_alpha1_5pct" / "eval_samples" / "recon_grid.png",
        asset_root / "fig_reconstruction_grid_best.png",
    )
    write_latex(
        sweep_rows,
        asset_root / "table_main_results.tex",
        ["method", "sampling_ratio", "model_psnr", "model_ssim", "score"],
    )
    write_latex(
        control_rows,
        asset_root / "table_phase6_controls.tex",
        ["method", "model_psnr", "model_ssim", "score", "hard_flip_fraction", "A_rel_fro_delta"],
    )
    write_latex(
        diag_rows,
        asset_root / "table_pattern_diagnostics.tex",
        ["method", "hard_flip_fraction", "A_rel_fro_delta", "secant_rip_delta", "offdiag_corr_delta"],
    )
    write_latex(
        noise_rows,
        asset_root / "table_noise_sweep.tex",
        ["method", "noise_std", "model_psnr", "model_ssim", "score"],
    )
    print(f"Wrote paper assets to: {asset_root}")


if __name__ == "__main__":
    main()
