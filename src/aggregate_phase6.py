from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import ensure_dir


PHASE6_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase6")
PHASE5_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase5")
PHASE2_ROOT = Path("E:/ns_mc_gan_gi/outputs_clean_phase2")

FIELDS = [
    "method",
    "sampling_ratio",
    "model_psnr",
    "model_ssim",
    "model_mse",
    "model_rel_meas_err",
    "score",
    "epoch0_psnr",
    "epoch0_ssim",
    "delta_from_epoch0_psnr",
    "delta_from_epoch0_ssim",
    "delta_from_epoch0_score",
    "hard_flip_fraction",
    "hard_flip_count",
    "rows_with_any_flip_fraction",
    "A_rel_fro_delta",
    "A_cosine_initial_final",
    "soft_l2_delta",
    "logits_l2_delta",
    "secant_rip_initial",
    "secant_rip_final",
    "secant_rip_delta",
    "offdiag_corr_initial",
    "offdiag_corr_final",
    "offdiag_corr_delta",
    "freeze_patterns",
    "freeze_generator_all",
    "pattern_logit_abs_init",
    "lr_patterns",
    "pattern_update_every",
    "attribution_note",
    "checkpoint",
    "sample_image",
    "pattern_image",
    "pattern_change_image",
    "status",
]


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score(metrics: dict | None, weight: float = 10.0):
    if not metrics:
        return ""
    model = metrics.get("model", {})
    if "psnr" not in model or "ssim" not in model:
        return ""
    return float(model["psnr"]) + weight * float(model["ssim"])


def specs(phase6: Path):
    return [
        ("Fixed Rademacher", 0.05, PHASE2_ROOT / "quick_5pct", False),
        ("Phase 5 Best Exact Slow", 0.05, PHASE5_ROOT / "exact_binary_slow_5pct", True),
        ("G-only Fine-tune", 0.05, phase6 / "g_only_finetune_5pct", True),
        ("Pattern Trainable Alpha6", 0.05, phase6 / "pattern_trainable_alpha6_5pct", True),
        ("Pattern Trainable Alpha2", 0.05, phase6 / "pattern_trainable_alpha2_5pct", True),
        ("Pattern Trainable Alpha1", 0.05, phase6 / "pattern_trainable_alpha1_5pct", True),
        ("Pattern Trainable Alpha0.5", 0.05, phase6 / "pattern_trainable_alpha0p5_5pct", True),
        ("Pattern-only Alpha1", 0.05, phase6 / "pattern_only_alpha1_5pct", True),
        ("Soft Signed Train", 0.05, phase6 / "soft_signed_train_5pct", True),
        ("Best Long 5%", 0.05, phase6 / "best_long_5pct", True),
    ]


def diagnostics_for(run_dir: Path) -> dict:
    return (
        read_json(run_dir / "eval_pattern_diagnostics" / "pattern_diagnostics.json")
        or read_json(run_dir / "pattern_diagnostics" / "pattern_diagnostics.json")
        or {}
    )


def row_from_run(method: str, ratio: float, run_dir: Path, learned: bool) -> dict:
    metrics = read_json(run_dir / "eval_metrics.json") or read_json(run_dir / "best_score_metrics.json")
    epoch0 = read_json(run_dir / "eval_epoch000_metrics.json")
    diagnostics = diagnostics_for(run_dir)
    config = read_json(run_dir / "resolved_config.json") or {}
    if not config:
        yaml_path = run_dir / "resolved_config.yaml"
        if yaml_path.exists():
            try:
                import yaml

                config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            except Exception:
                config = {}
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "method": method,
            "sampling_ratio": ratio,
            "status": "ok" if metrics else "missing",
            "checkpoint": str(run_dir / "best_score.pt")
            if (run_dir / "best_score.pt").exists()
            else str(run_dir / "best_ssim.pt")
            if (run_dir / "best_ssim.pt").exists()
            else "",
            "sample_image": str(run_dir / "eval_samples" / "recon_grid.png")
            if (run_dir / "eval_samples" / "recon_grid.png").exists()
            else str(run_dir / "samples" / "epoch_000.png")
            if (run_dir / "samples" / "epoch_000.png").exists()
            else "",
            "pattern_image": str(run_dir / "eval_patterns" / "final_patterns.png")
            if (run_dir / "eval_patterns" / "final_patterns.png").exists()
            else "",
            "pattern_change_image": str(
                run_dir / "eval_pattern_diagnostics" / "pattern_change_grid.png"
            )
            if (run_dir / "eval_pattern_diagnostics" / "pattern_change_grid.png").exists()
            else str(run_dir / "pattern_diagnostics" / "pattern_change_grid.png")
            if (run_dir / "pattern_diagnostics" / "pattern_change_grid.png").exists()
            else "",
        }
    )
    if metrics:
        model = metrics.get("model", {})
        row.update(
            {
                "model_psnr": model.get("psnr", ""),
                "model_ssim": model.get("ssim", ""),
                "model_mse": model.get("mse", ""),
                "model_rel_meas_err": model.get("rel_meas_error", ""),
                "score": score(metrics),
            }
        )
    if epoch0:
        model0 = epoch0.get("model", {})
        row["epoch0_psnr"] = model0.get("psnr", "")
        row["epoch0_ssim"] = model0.get("ssim", "")
        if row["model_psnr"] != "" and row["epoch0_psnr"] != "":
            row["delta_from_epoch0_psnr"] = float(row["model_psnr"]) - float(row["epoch0_psnr"])
        if row["model_ssim"] != "" and row["epoch0_ssim"] != "":
            row["delta_from_epoch0_ssim"] = float(row["model_ssim"]) - float(row["epoch0_ssim"])
        if row["score"] != "" and row["epoch0_psnr"] != "" and row["epoch0_ssim"] != "":
            row["delta_from_epoch0_score"] = float(row["score"]) - (
                float(row["epoch0_psnr"]) + 10.0 * float(row["epoch0_ssim"])
            )
    for key in [
        "hard_flip_fraction",
        "hard_flip_count",
        "rows_with_any_flip_fraction",
        "A_rel_fro_delta",
        "A_cosine_initial_final",
        "soft_l2_delta",
        "logits_l2_delta",
        "secant_rip_initial",
        "secant_rip_final",
        "secant_rip_delta",
        "offdiag_corr_initial",
        "offdiag_corr_final",
        "offdiag_corr_delta",
    ]:
        row[key] = diagnostics.get(key, "")
    row["attribution_note"] = diagnostics.get("pattern_attribution_note", "")
    row["freeze_patterns"] = diagnostics.get("freeze_patterns", config.get("freeze_patterns", ""))
    row["freeze_generator_all"] = diagnostics.get(
        "freeze_generator_all", config.get("freeze_generator_all", "")
    )
    row["pattern_logit_abs_init"] = diagnostics.get(
        "pattern_logit_abs_init", config.get("pattern_logit_abs_init", "")
    )
    row["lr_patterns"] = diagnostics.get("lr_patterns", config.get("lr_patterns", ""))
    row["pattern_update_every"] = diagnostics.get(
        "pattern_update_every", config.get("pattern_update_every", "")
    )
    return row


def write_csv(rows: list[dict], path: Path, fields: list[str] = FIELDS) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value) -> str:
    if value in ("", None):
        return "missing"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def write_markdown(rows: list[dict], path: Path, title: str, cols: list[str]) -> None:
    lines = [f"# {title}", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_score(rows: list[dict], path: Path) -> None:
    ok = [row for row in rows if row.get("status") == "ok" and row.get("score") != ""]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(ok) * 1.05), 4.5))
    labels = [row["method"] for row in ok]
    values = [float(row["score"]) for row in ok]
    ax.bar(range(len(ok)), values)
    ax.set_ylabel("Score")
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scatter(rows: list[dict], x_key: str, path: Path, xlabel: str) -> None:
    ok = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("score") != "" and row.get(x_key) not in ("", "missing")
    ]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for row in ok:
        ax.scatter(float(row[x_key]), float(row["score"]), label=row["method"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_pattern_diagnostics_csv(rows: list[dict], path: Path) -> None:
    cols = [
        "method",
        "hard_flip_fraction",
        "A_rel_fro_delta",
        "secant_rip_delta",
        "offdiag_corr_delta",
        "attribution_note",
        "status",
    ]
    write_csv(rows, path, cols)


def write_latex(rows: list[dict], path: Path, cols: list[str]) -> None:
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline"]
    lines.append(" & ".join(cols).replace("_", "\\_") + " \\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(fmt(row.get(col, "")).replace("_", "\\_") for col in cols) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    phase6 = ensure_dir(PHASE6_ROOT)
    rows = [row_from_run(*spec) for spec in specs(phase6)]
    for seed_dir in sorted(phase6.glob("best_seed*_5pct")):
        rows.append(row_from_run(seed_dir.name, 0.05, seed_dir, True))

    write_csv(rows, phase6 / "phase6_control_results.csv")
    write_markdown(
        rows,
        phase6 / "phase6_control_results.md",
        "Phase 6 Control Results",
        [
            "method",
            "model_psnr",
            "model_ssim",
            "score",
            "hard_flip_fraction",
            "A_rel_fro_delta",
            "status",
        ],
    )
    write_pattern_diagnostics_csv(rows, phase6 / "phase6_pattern_diagnostics.csv")
    plot_score(rows, phase6 / "phase6_control_score_bar.png")
    plot_scatter(rows, "hard_flip_fraction", phase6 / "phase6_pattern_flip_vs_score.png", "Hard Flip Fraction")
    plot_scatter(rows, "A_rel_fro_delta", phase6 / "phase6_A_drift_vs_score.png", "A Rel Fro Delta")
    plot_scatter(rows, "secant_rip_delta", phase6 / "phase6_secant_rip_vs_score.png", "Secant-RIP Delta")

    paired = read_csv(phase6 / "paired_5pct" / "paired_summary.csv")
    if paired:
        write_markdown(
            paired,
            phase6 / "phase6_paired_summary.md",
            "Phase 6 Paired Bootstrap Summary",
            [
                "method_a",
                "method_b",
                "score_mean_delta",
                "score_ci_low",
                "score_ci_high",
                "score_p_gt_0",
                "status",
            ],
        )
    else:
        (phase6 / "phase6_paired_summary.md").write_text(
            "# Phase 6 Paired Bootstrap Summary\n\nNo paired bootstrap results found.\n",
            encoding="utf-8",
        )
    write_latex(
        rows,
        phase6 / "phase6_latex_table_controls.tex",
        ["method", "model_psnr", "model_ssim", "score", "hard_flip_fraction", "A_rel_fro_delta"],
    )
    write_latex(
        paired,
        phase6 / "phase6_latex_table_paired.tex",
        ["method_a", "method_b", "score_mean_delta", "score_ci_low", "score_ci_high", "score_p_gt_0"],
    )
    print(f"Wrote Phase 6 aggregation to: {phase6}")


if __name__ == "__main__":
    main()
