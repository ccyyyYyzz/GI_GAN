from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase70_gauge_gan_paper_expansion"
PH69A = ROOT / "outputs_phase69A_gauge_gan_signal_diagnostic"
PH69B = ROOT / "outputs_phase69B_controlled_gauge_cgan_pilot"
CERT = ROOT / "results" / "cert_package_20260612" / "cache"
A_SCR5 = CERT / "A_scr5.npy"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    lines = [
        "| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |",
        "| " + " | ".join("-" * widths[c] for c in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns) + " |")
    return "\n".join(lines)


def f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def phase69b_repro() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = read_csv(PH69B / "evaluation_metrics.csv")
    comp = read_csv(PH69B / "paired_comparison_C_vs_B.csv")
    beta = read_csv(PH69B / "beta_calibration.csv")
    rel = read_csv(PH69B / "relmeaserr_certificate_table.csv")
    lpips = read_csv(PH69B / "lpips_or_dists_results.csv")
    auc = read_csv(PH69A / "critic_auc_results.csv")
    ctrl = read_csv(PH69A / "shortcut_control_results.csv")
    split = json.loads((PH69B / "split_manifest.json").read_text(encoding="utf-8"))
    b_summary = json.loads((PH69B / "pilot" / "armB" / "training_summary.json").read_text(encoding="utf-8"))
    c_summary = json.loads((PH69B / "pilot" / "armC" / "training_summary.json").read_text(encoding="utf-8"))

    rows = []
    by_arm = {row["arm"]: row for row in metrics}
    lp_by_arm = {row["arm"]: row for row in lpips if row.get("arm")}
    for arm in ["A", "B", "C"]:
        row = by_arm[arm]
        rows.append(
            {
                "arm": arm,
                "psnr_mean": row["psnr_mean"],
                "ssim_mean": row["ssim_mean"],
                "relmeaserr_mean": row["relmeaserr_unclipped_float64_mean"],
                "lpips_mean": lp_by_arm.get(arm, {}).get("lpips_mean", ""),
                "rapsd_distance_mean": row["rapsd_distance_mean"],
                "gradient_error_mean": row["gradient_mean_abs_error_mean"],
                "highfreq_error_mean": row["highfreq_ratio_abs_error_mean"],
                "p0_l2_mean": row["p0_l2_mean"],
            }
        )
    write_csv(OUT / "phase69B_repro_metrics.csv", rows)
    text = [
        "# Phase69B Reproduction Summary",
        "",
        f"Source: `{PH69B}`",
        "",
        "## A/B/C Metrics",
        "",
        table(rows, ["arm", "psnr_mean", "ssim_mean", "relmeaserr_mean", "lpips_mean", "rapsd_distance_mean", "gradient_error_mean", "highfreq_error_mean"]),
        "",
        "## C vs B",
        "",
        table(comp, ["metric", "direction", "mean_B", "mean_C", "improvement_positive_means_C_better", "ci_low", "ci_high", "ci_excludes_zero_in_favor_of_C"]),
        "",
        "## Beta Calibration",
        "",
        table(beta, ["adv_to_rec_ratio", "selected_beta0", "candidate_0p3_beta0", "candidate_beta0", "candidate_3_beta0", "candidate_sweep_run"]),
        "",
        "## Split / Checkpoint Provenance",
        "",
        f"- train full sorted SHA256: `{split['train_full_sorted_sha256']}`",
        f"- eval full sorted SHA256: `{split['eval_full_sorted_sha256']}`",
        f"- Arm B best checkpoint: `{b_summary['best_checkpoint']}`",
        f"- Arm C best checkpoint: `{c_summary['best_checkpoint']}`",
        f"- Arm C D last mean accuracy: `{c_summary['d_accuracy_last_mean']}`",
        "",
        "## RelMeasErr Certificate",
        "",
        table(rel, ["arm", "relmeaserr_unclipped_float64_mean", "relmeaserr_unclipped_float64_median", "certificate_operator"]),
        "",
        "No first-paper result was modified.",
    ]
    write_text(OUT / "PHASE69B_REPRO_SUMMARY.md", "\n".join(text) + "\n")
    return rows, comp, auc, ctrl


def gauge_equality_check() -> list[dict[str, Any]]:
    A = np.load(A_SCR5).astype(np.float64)
    data = np.load(PH69A / "gauge_dataset_cache_float16.npz")
    rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        real = data[f"{split}_real"].astype(np.float64)
        fake = data[f"{split}_fake"].astype(np.float64)
        y = data[f"{split}_y"].astype(np.float64)
        b = (y @ A) / 1.001
        ab = b @ A.T
        ar = real @ A.T
        af = fake @ A.T
        denom = np.linalg.norm(ab, axis=1).clip(1e-12)
        er = np.linalg.norm(ar - ab, axis=1) / denom
        ef = np.linalg.norm(af - ab, axis=1) / denom
        erf = np.linalg.norm(ar - af, axis=1) / denom
        rows.extend(
            [
                {
                    "split": split,
                    "check": "A_real_gauge_minus_A_Blambda_y",
                    "median_relative_error": float(np.median(er)),
                    "max_relative_error": float(np.max(er)),
                    "n": int(real.shape[0]),
                },
                {
                    "split": split,
                    "check": "A_fake_gauge_minus_A_Blambda_y",
                    "median_relative_error": float(np.median(ef)),
                    "max_relative_error": float(np.max(ef)),
                    "n": int(fake.shape[0]),
                },
                {
                    "split": split,
                    "check": "A_real_gauge_minus_A_fake_gauge",
                    "median_relative_error": float(np.median(erf)),
                    "max_relative_error": float(np.max(erf)),
                    "n": int(fake.shape[0]),
                },
            ]
        )
    write_csv(OUT / "gauge_equality_check.csv", rows)
    write_text(
        OUT / "GAUGE_EQUALITY_CHECK_REPORT.md",
        "\n".join(
            [
                "# Gauge Equality Check",
                "",
                "The B_lambda gauge is row-equalized / residual-shortcut-free. It should not be described as exactly feasible, and the deployed reconstruction remains `hat{x}=Pi_y^lambda(v_theta)`.",
                "",
                table(rows, ["split", "check", "median_relative_error", "max_relative_error", "n"]),
                "",
                "Interpretation: both real and fake canonical images share the same measured-row component up to numerical/cache precision, so a discriminator cannot use the direct residual shortcut.",
                "",
            ]
        ),
    )
    return rows


def seed_and_regime_reports(comp: list[dict[str, Any]]) -> None:
    # Current Phase69B is a single paired Scr-5 seed. Produce an honest seed
    # table and mark conference-level seed validation as incomplete.
    metric_map = {row["metric"]: row for row in comp}
    seed_rows = []
    for metric in ["lpips", "rapsd_distance", "gradient_mean_abs_error", "highfreq_ratio_abs_error", "psnr", "relmeaserr_unclipped_float64"]:
        row = metric_map[metric]
        seed_rows.append(
            {
                "regime": "scr5",
                "seed_id": "phase69B_seed0",
                "metric": metric,
                "C_minus_B": row["mean_C_minus_B"],
                "C_better_effect": row["improvement_positive_means_C_better"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "ci_excludes_zero_in_favor_of_C": row["ci_excludes_zero_in_favor_of_C"],
            }
        )
    write_csv(OUT / "scr5_seed_results.csv", seed_rows)
    write_text(
        OUT / "scr5_seed_summary.md",
        "\n".join(
            [
                "# Scr-5 Seed Summary",
                "",
                "Available evidence: one controlled paired Scr-5 seed from Phase69B.",
                "",
                table(seed_rows, ["metric", "C_minus_B", "C_better_effect", "ci_low", "ci_high", "ci_excludes_zero_in_favor_of_C"]),
                "",
                "Seed-readiness verdict: insufficient for conference claim. The observed seed is positive for LPIPS/RAPSD/gradient/high-frequency metrics, but the required 3/3 seed consistency has not been run.",
                "",
            ]
        ),
    )
    # Scr-10 and Rad-5 not run by Phase70 because Scr-5 seed gate is incomplete.
    scr10 = [
        {
            "regime": "scr10",
            "status": "not_run",
            "reason": "Scr-5 minimum 3-seed gate is incomplete; running Scr-10 would not yet support a paper-level regime claim.",
        }
    ]
    rad5 = [
        {
            "regime": "rad5",
            "status": "not_run",
            "reason": "Optional robustness regime deferred until paired Scr-5 seeds and Scr-10 minimum regime are complete.",
        }
    ]
    write_csv(OUT / "scr10_results.csv", scr10)
    write_text(OUT / "scr10_summary.md", "# Scr-10 Summary\n\nScr-10 was not run in Phase70. The Scr-5 3-seed gate is incomplete, so Scr-10 remains a required next experiment.\n")
    write_csv(OUT / "rad5_results.csv", rad5)
    write_text(OUT / "rad5_summary.md", "# Rad-5 Summary\n\nRad-5 was not run. It remains an optional measurement-ensemble robustness check after Scr-5/Scr-10 validation.\n")

    # Seed figures
    fig, ax = plt.subplots(figsize=(6, 3.5))
    regimes = ["Scr-5\nobserved", "Scr-10\nnot run", "Rad-5\nnot run"]
    evidence = [1.0, 0.0, 0.0]
    colors = ["tab:blue", "lightgray", "lightgray"]
    ax.bar(regimes, evidence, color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("paired evidence available")
    ax.set_title("Regime Coverage for Phase70")
    for idx, label in enumerate(["1 seed", "gate incomplete", "deferred"]):
        ax.text(idx, evidence[idx] + 0.04, label, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "scr10_vs_scr5_regime_plot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["metric"] for r in seed_rows]
    vals = [float(r["C_better_effect"]) for r in seed_rows]
    colors = ["tab:green" if v > 0 else "tab:red" for v in vals]
    ax.bar(range(len(names)), vals, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("C improvement over B")
    ax.set_title("Scr-5 single-seed C vs B")
    fig.tight_layout()
    fig.savefig(OUT / "scr5_C_vs_B_seed_plot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_rows = [r for r in seed_rows if r["metric"] in {"lpips", "rapsd_distance", "gradient_mean_abs_error", "highfreq_ratio_abs_error"}]
    names = [r["metric"] for r in plot_rows]
    vals = np.array([float(r["C_better_effect"]) for r in plot_rows])
    lows = np.array([float(r["ci_low"]) for r in plot_rows])
    highs = np.array([float(r["ci_high"]) for r in plot_rows])
    ax.bar(range(len(names)), vals, yerr=np.vstack([vals - lows, highs - vals]), capsize=3)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Within-seed bootstrap C vs B")
    fig.tight_layout()
    fig.savefig(OUT / "scr5_metric_ci_bars.png", dpi=180)
    plt.close(fig)


def shortcut_report(auc: list[dict[str, Any]], ctrl: list[dict[str, Any]]) -> None:
    rows = []
    for row in auc + ctrl:
        rows.append(
            {
                "model": row["model"],
                "auc": float(row["auc"]),
                "auc_ci_low": float(row["auc_ci_low"]),
                "auc_ci_high": float(row["auc_ci_high"]),
            }
        )
    write_csv(OUT / "shortcut_ablation_auc.csv", rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["model"] for r in rows]
    aucs = np.array([r["auc"] for r in rows])
    lows = np.array([r["auc_ci_low"] for r in rows])
    highs = np.array([r["auc_ci_high"] for r in rows])
    ax.bar(range(len(rows)), aucs, yerr=np.vstack([aucs - lows, highs - aucs]), capsize=3)
    ax.axhline(0.5, color="black", linewidth=1)
    ax.axhline(0.65, color="tab:green", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("Shortcut and Gauge Ablations")
    fig.tight_layout()
    for name in ["shortcut_ablation_auc", "fig2_shortcut_auc"]:
        fig.savefig(OUT / f"{name}.png", dpi=180)
        fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    write_text(
        OUT / "SHORTCUT_ABLATION_REPORT.md",
        "\n".join(
            [
                "# Shortcut Ablation Report",
                "",
                table(rows, ["model", "auc", "auc_ci_low", "auc_ci_high"]),
                "",
                "The residual-fed logistic control reaches AUC about 0.98, showing that a naive discriminator can cheat. The gauge-only PatchGAN remains high (about 0.847) while excluding direct residual/correction features, so the gauge removes the obvious residual shortcut while retaining a realism signal.",
                "",
            ]
        ),
    )


def beta_frontier(metrics: list[dict[str, Any]], comp: list[dict[str, Any]]) -> None:
    by_arm = {r["arm"]: r for r in metrics}
    beta = read_csv(PH69B / "beta_calibration.csv")[0]
    lp = {r["arm"]: r for r in read_csv(PH69B / "lpips_or_dists_results.csv") if r.get("arm")}
    rows = [
        {
            "beta": 0.0,
            "source": "Arm B supervised",
            "psnr": by_arm["B"]["psnr_mean"],
            "lpips": lp["B"]["lpips_mean"],
            "rapsd_distance": by_arm["B"]["rapsd_distance_mean"],
            "relmeaserr": by_arm["B"]["relmeaserr_mean"],
            "run_status": "observed",
        },
        {
            "beta": beta["candidate_0p3_beta0"],
            "source": "candidate_0p3_beta0",
            "psnr": "",
            "lpips": "",
            "rapsd_distance": "",
            "relmeaserr": "",
            "run_status": "not_run",
        },
        {
            "beta": beta["selected_beta0"],
            "source": "Arm C cGAN",
            "psnr": by_arm["C"]["psnr_mean"],
            "lpips": lp["C"]["lpips_mean"],
            "rapsd_distance": by_arm["C"]["rapsd_distance_mean"],
            "relmeaserr": by_arm["C"]["relmeaserr_mean"],
            "run_status": "observed",
        },
        {
            "beta": beta["candidate_3_beta0"],
            "source": "candidate_3_beta0",
            "psnr": "",
            "lpips": "",
            "rapsd_distance": "",
            "relmeaserr": "",
            "run_status": "not_run",
        },
    ]
    write_csv(OUT / "beta_frontier.csv", rows)
    obs = [r for r in rows if r["run_status"] == "observed"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot([float(r["psnr"]) for r in obs], [float(r["lpips"]) for r in obs], marker="o")
    axes[0].set_xlabel("PSNR")
    axes[0].set_ylabel("LPIPS lower better")
    axes[1].plot([float(r["psnr"]) for r in obs], [float(r["rapsd_distance"]) for r in obs], marker="o")
    axes[1].set_xlabel("PSNR")
    axes[1].set_ylabel("RAPSD distance")
    axes[2].plot([float(r["beta"]) for r in obs], [float(r["relmeaserr"]) for r in obs], marker="o")
    axes[2].set_xlabel("beta")
    axes[2].set_ylabel("RelMeasErr")
    fig.suptitle("Observed beta frontier: beta=0 vs beta0")
    fig.tight_layout()
    for name in ["beta_frontier_plot", "fig5_perception_distortion_frontier"]:
        fig.savefig(OUT / f"{name}.png", dpi=180)
        fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    write_text(
        OUT / "BETA_FRONTIER_REPORT.md",
        "\n".join(
            [
                "# Beta / Perception-Distortion Frontier",
                "",
                "Observed points currently include beta=0 (supervised Arm B) and beta0 (cGAN Arm C). Candidate 0.3 beta0 and 3 beta0 were recorded but not run, so the frontier is not complete.",
                "",
                table(rows, ["beta", "source", "psnr", "lpips", "rapsd_distance", "relmeaserr", "run_status"]),
                "",
                "Operating point used in Phase69B satisfies the guardrail: PSNR loss is about 0.003 dB and RelMeasErr remains comparable, while LPIPS and spectral/texture metrics improve.",
                "",
            ]
        ),
    )


def other_figures(metrics: list[dict[str, Any]], comp: list[dict[str, Any]]) -> None:
    # fig1 method gauge
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    boxes = [
        (0.05, 0.62, "x, y"),
        (0.32, 0.76, "Real gauge\nP0 x + Bλ y"),
        (0.32, 0.48, "Fake gauge\nP0 vθ + Bλ y"),
        (0.62, 0.62, "Dψ( x~ )\nno residual features"),
        (0.32, 0.16, "Deployment\nx̂ = Πλ_y(vθ)"),
    ]
    for x, y, text in boxes:
        rect = plt.Rectangle((x, y), 0.22, 0.14, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.11, y + 0.07, text, ha="center", va="center", fontsize=10)
    for start, end in [((0.27, 0.69), (0.32, 0.83)), ((0.27, 0.69), (0.32, 0.55)), ((0.54, 0.83), (0.62, 0.69)), ((0.54, 0.55), (0.62, 0.69)), ((0.43, 0.48), (0.43, 0.30))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(0.5, 0.04, "GAN is an adversarial prior branch; Πλ_y remains the bucket-signal accountability operator.", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_method_gauge.png", dpi=180)
    fig.savefig(OUT / "fig1_method_gauge.pdf")
    plt.close(fig)

    copy_if_exists(PH69B / "visual_grid_A_B_C.png", OUT / "fig3_abc_visual_grid.png")
    copy_if_exists(PH69B / "visual_grid_A_B_C.pdf", OUT / "fig3_abc_visual_grid.pdf")
    copy_if_exists(PH69B / "rapsd_comparison.png", OUT / "fig7_rapsd_curves.png")
    copy_if_exists(PH69B / "rapsd_comparison.pdf", OUT / "fig7_rapsd_curves.pdf")

    # fig4 LPIPS/RAPSD bars
    lp = {r["arm"]: r for r in read_csv(PH69B / "lpips_or_dists_results.csv") if r.get("arm")}
    by_arm = {r["arm"]: r for r in metrics}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    arms = ["A", "B", "C"]
    axes[0].bar(arms, [float(lp[a]["lpips_mean"]) for a in arms])
    axes[0].set_ylabel("LPIPS lower better")
    axes[1].bar(arms, [float(by_arm[a]["rapsd_distance_mean"]) for a in arms])
    axes[1].set_ylabel("RAPSD distance lower better")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_lpips_rapsd_seed_bars.png", dpi=180)
    fig.savefig(OUT / "fig4_lpips_rapsd_seed_bars.pdf")
    plt.close(fig)

    # fig6 RelMeasErr
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(arms, [float(by_arm[a]["relmeaserr_mean"]) for a in arms])
    ax.set_ylabel("RelMeasErr unclipped float64")
    ax.set_title("Measurement certificate stays with Πλ_y")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_relmeaserr_certificate.png", dpi=180)
    fig.savefig(OUT / "fig6_relmeaserr_certificate.pdf")
    plt.close(fig)


def readiness_reports(comp: list[dict[str, Any]], gauge_rows: list[dict[str, Any]]) -> None:
    metric_map = {r["metric"]: r for r in comp}
    success_metrics = [m for m in ["lpips", "rapsd_distance", "gradient_mean_abs_error", "highfreq_ratio_abs_error"] if str(metric_map[m]["ci_excludes_zero_in_favor_of_C"]).lower() == "true"]
    decision = "workshop/project-supplement candidate only"
    missing = [
        "Run at least 3 paired Scr-5 seeds with identical B/C data order and checkpoint selection.",
        "Run at least one Scr-10 paired regime; 3 seeds preferred.",
        "Run Rad-5 robustness or explicitly scope the paper to Scrambled-Hadamard.",
        "Complete beta sweep: 0, 0.3 beta0, beta0, 3 beta0.",
        "Freeze all scripts/configs in a reproducible package with command logs.",
    ]
    claims = [
        "Gauge-equalized discriminator input removes the direct measurement residual shortcut.",
        "A gauge-only PatchGAN can still detect a realism signal in Scr-5 mean-mode outputs.",
        "In the single controlled Scr-5 pilot, cGAN fine-tuning improves LPIPS and spectral/texture metrics relative to a budget-matched supervised fine-tune.",
        "Measurement accountability remains with Pi_y^lambda; GAN is not a certificate.",
    ]
    limitations = [
        "Only one Scr-5 paired seed is available; this is not enough for a conference-level stability claim.",
        "Scr-10 and Rad-5 regimes were not run in Phase70.",
        "The beta frontier has only beta=0 and beta0 observed.",
        "DISTS and KID were not used for the decision; KID is not reliable at this small sample count.",
        "This branch must not modify the first paper's main table, title, or abstract.",
    ]
    outline = [
        "1. Motivation: adversarial priors without measurement shortcuts.",
        "2. Gauge construction: P0 x + B_lambda y and P0 v_theta + B_lambda y.",
        "3. Shortcut audit: residual-fed vs gauge-only discriminators.",
        "4. Controlled cGAN pilot: A mean, B supervised, C gauge-cGAN.",
        "5. Perceptual/spectral results and measurement certificate.",
        "6. Limitations and multi-regime seed plan.",
    ]
    answers = [
        ("Is C better than B across seeds?", "Not established. Current evidence is one paired Scr-5 seed only."),
        ("Is effect stable or small/fragile?", "Positive but currently fragile/underpowered until seed and regime checks are complete."),
        ("Does GAN preserve RelMeasErr?", "Approximately; C mean RelMeasErr is 0.005568 vs B 0.005563. Certificate stays with Pi_y^lambda."),
        ("Does improvement live in perceptual/spectral/high-frequency metrics?", f"Yes in the available seed: {', '.join(success_metrics)}."),
        ("Does shortcut-free gauge work?", "Yes structurally and empirically: residual-fed AUC is about 0.98, gauge-only AUC about 0.847, and gauge equality checks show shared measured-row components."),
        ("Is this enough for workshop?", "Yes as a cautious workshop/project-supplement candidate if framed as single-seed pilot evidence."),
        ("What is missing for strong conference?", "; ".join(missing)),
        ("Should further training continue?", "Yes: paired Scr-5 seeds, Scr-10, Rad-5, and beta sweep are the next gates."),
        ("How does this support the first paper?", "It provides a separate adversarial-prior branch while preserving the first paper's measurement-certified core."),
        ("First paper main results unchanged?", "Yes."),
    ]
    write_text(OUT / "CLAIMS_FOR_GAUGE_GAN_PAPER.md", "# Claims For Gauge-GAN Paper\n\n" + "\n".join(f"- {c}" for c in claims) + "\n")
    write_text(OUT / "LIMITATIONS_GAUGE_GAN_PAPER.md", "# Limitations\n\n" + "\n".join(f"- {l}" for l in limitations) + "\n")
    write_text(OUT / "PAPER_OUTLINE_GAUGE_GAN.md", "# Paper Outline\n\n" + "\n".join(f"- {o}" for o in outline) + "\n")
    write_text(OUT / "WORKSHOP_READINESS.md", "# Workshop Readiness\n\nDecision: workshop/project-supplement candidate only.\n\n" + "\n".join(f"- {q} {a}" for q, a in answers) + "\n")
    write_text(OUT / "STRONG_CONFERENCE_GAP_LIST.md", "# Strong Conference Gap List\n\n" + "\n".join(f"- {m}" for m in missing) + "\n")
    write_text(
        OUT / "PHASE70_GAUGE_GAN_PAPER_REPORT.md",
        "\n".join(
            [
                "# Phase70 Gauge-Equalized Adversarial Prior Paper Expansion",
                "",
                f"Generated: {datetime.now().isoformat(timespec='seconds')}",
                "",
                f"Decision: {decision}.",
                "",
                "## Required Answers",
                "",
                table([{"question": q, "answer": a} for q, a in answers], ["question", "answer"]),
                "",
                "## Allowed Claims",
                "",
                "\n".join(f"- {c}" for c in claims),
                "",
                "## Limitations",
                "",
                "\n".join(f"- {l}" for l in limitations),
                "",
                "## Gauge Equality",
                "",
                table(gauge_rows, ["split", "check", "median_relative_error", "max_relative_error", "n"]),
                "",
                "No first-paper main result, checkpoint, table, title, or abstract was modified.",
                "",
            ]
        ),
    )


def manifest() -> None:
    names = sorted(p.name for p in OUT.iterdir() if p.is_file())
    write_text(
        OUT / "MANIFEST.md",
        "\n".join(
            [
                "# Phase70 Manifest",
                "",
                f"Output directory: `{OUT}`",
                "",
                "## Files",
                "",
                "\n".join(f"- `{name}`" for name in names),
                "",
                "## Main Result Integrity",
                "",
                "Phase70 is an evidence/readiness package. It did not modify the first paper's main results or overwrite any existing checkpoint.",
                "",
            ]
        ),
    )


def main() -> int:
    ensure_dir(OUT)
    write_text(OUT / "RUNLOG.md", f"# Phase70 Runlog\n- {datetime.now().isoformat(timespec='seconds')} start\n")
    metrics, comp, auc, ctrl = phase69b_repro()
    gauge_rows = gauge_equality_check()
    seed_and_regime_reports(comp)
    shortcut_report(auc, ctrl)
    beta_frontier(metrics, comp)
    other_figures(metrics, comp)
    readiness_reports(comp, gauge_rows)
    manifest()
    with (OUT / "RUNLOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {datetime.now().isoformat(timespec='seconds')} complete no_first_paper_changes=true\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
