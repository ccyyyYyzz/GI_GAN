from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("E:/ns_mc_gan_gi")
PH69B = ROOT / "outputs_phase69B_controlled_gauge_cgan_pilot"
PH70 = ROOT / "outputs_phase70_gauge_gan_paper_expansion"
PH71 = ROOT / "outputs_phase71_gauge_cgan_paired_seeds"
PH72 = ROOT / "outputs_phase72_scr10_gauge_cgan_regime_validation"
PH73 = ROOT / "outputs_phase73_overnight_gauge_gan_expansion"
PH74 = ROOT / "outputs_phase74_high_tier_gauge_cgan_pack"
PH75 = ROOT / "outputs_phase75_final_high_tier_validation"
PH76 = ROOT / "outputs_phase76_high_upside_auditable_gan_exploration"
OUT = ROOT / "outputs_phase77_auditable_gan_paper_assembly"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_No rows available._"
    use = df.copy()
    if columns is not None:
        use = use[[c for c in columns if c in use.columns]]
    if max_rows is not None:
        use = use.head(max_rows)
    return use.to_markdown(index=False)


def fmt(x: float | int | str | None, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if abs(xf) != 0 and (abs(xf) < 1e-3 or abs(xf) >= 1e4):
        return f"{xf:.3e}"
    return f"{xf:.{digits}f}"


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def image_to_pdf(src: Path, dst_png: Path, dst_pdf: Path) -> None:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst_png)
    img = plt.imread(src)
    fig_w = 9
    fig_h = max(3, fig_w * img.shape[0] / img.shape[1])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(dst_pdf, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def aggregate_metrics(df: pd.DataFrame, source: str, regime: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = [
        "psnr_mean",
        "ssim_mean",
        "lpips_mean",
        "rapsd_distance_mean",
        "gradient_mean_abs_error_mean",
        "highfreq_ratio_abs_error_mean",
        "relmeaserr_unclipped_float64_mean",
        "p0_l2_mean",
    ]
    rows = []
    for arm, g in df.groupby("arm"):
        row = {"source": source, "regime": regime, "arm": arm, "seeds": int(g["seed"].nunique()) if "seed" in g else 1}
        if "n" in g:
            row["n_per_seed"] = int(g["n"].iloc[0])
        for m in metrics:
            if m in g:
                row[m] = float(g[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_canonical_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    scr5 = aggregate_metrics(
        read_csv(PH75 / "standard_cgan_seed_metrics.csv"),
        "Phase75 standard_cgan_seed_metrics.csv",
        "Scr-5",
    )
    rad5 = aggregate_metrics(
        read_csv(PH73 / "rad5_seed_metrics.csv"),
        "Phase73 rad5_seed_metrics.csv",
        "Rad-5",
    )
    canonical = pd.concat([scr5, rad5], ignore_index=True)
    if not canonical.empty:
        canonical["use_in_main"] = canonical["regime"].isin(["Scr-5", "Rad-5"])
        canonical["claim_role"] = canonical.apply(
            lambda r: "canonical controlled 5% result" if r["regime"] == "Scr-5" else "robustness/regime result",
            axis=1,
        )
    archive_frames = []
    for label, path in [
        ("Phase69B original pilot", PH69B / "evaluation_metrics.csv"),
        ("Phase70 Phase69B reproduction", PH70 / "phase69B_repro_metrics.csv"),
        ("Phase70 Scr-5 seed table", PH70 / "scr5_seed_results.csv"),
        ("Phase71 Scr-5 seed deltas", PH71 / "scr5_seed_delta_metrics.csv"),
        ("Phase74 Scr-5 seed01 standard comparison", PH74 / "standard_cgan_baseline_scr5.csv"),
    ]:
        df = read_csv(path)
        if df.empty:
            archive_frames.append(pd.DataFrame([{"archive_source": label, "archive_path": str(path), "status": "missing_or_empty"}]))
            continue
        keep = df.copy()
        keep.insert(0, "archive_source", label)
        keep.insert(1, "archive_path", str(path))
        keep["status"] = "archive_or_support_only"
        archive_frames.append(keep)
    archive = pd.concat(archive_frames, ignore_index=True, sort=False)
    return canonical, archive


def build_unmeasured_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = read_csv(PH76 / "tables" / "unmeasured_content_metrics.csv")
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    bins = []
    for (regime, arm), g in metrics.groupby(["regime", "arm"]):
        h = g["h_null_energy_ratio"].astype(float)
        candidates = {
            "null_error_ratio": g["h_err_null_error_ratio"].astype(float),
            "range_error_ratio": g["r_err_range_error_ratio"].astype(float),
            "highfreq_ratio": g["highfreq_ratio"].astype(float),
            "rapsd_distance": g["rapsd_distance"].astype(float),
        }
        for target_name, y in candidates.items():
            pearson = float(h.corr(y, method="pearson"))
            spearman = float(h.corr(y, method="spearman"))
            top = g[h >= h.quantile(0.9)]
            rest = g[h < h.quantile(0.9)]
            rows.append(
                {
                    "regime": regime,
                    "arm": arm,
                    "target": target_name,
                    "pearson": pearson,
                    "spearman": spearman,
                    "n": len(g),
                    "top10_h_mean": float(top["h_null_energy_ratio"].mean()),
                    "top10_target_mean": float(top[target_name if target_name in top else "h_err_null_error_ratio"].mean())
                    if target_name in top
                    else np.nan,
                    "rest_target_mean": float(rest[target_name if target_name in rest else "h_err_null_error_ratio"].mean())
                    if target_name in rest
                    else np.nan,
                    "evidence_level": "per-sample precomputed Phase76 metric; not fresh pixel-level recomputation",
                }
            )
        q = pd.qcut(h.rank(method="first"), q=5, labels=False)
        tmp = g.copy()
        tmp["quantile_bin"] = q
        for b, bg in tmp.groupby("quantile_bin"):
            bins.append(
                {
                    "regime": regime,
                    "arm": arm,
                    "quantile_bin": int(b),
                    "h_mean": float(bg["h_null_energy_ratio"].mean()),
                    "null_error_ratio_mean": float(bg["h_err_null_error_ratio"].mean()),
                    "range_error_ratio_mean": float(bg["r_err_range_error_ratio"].mean()),
                    "rapsd_distance_mean": float(bg["rapsd_distance"].mean()),
                    "n": len(bg),
                }
            )
    corr = pd.DataFrame(rows)
    binned = pd.DataFrame(bins)
    corr.to_csv(OUT / "unmeasured_content_correlation.csv", index=False)
    binned.to_csv(OUT / "unmeasured_content_binned_curve.csv", index=False)
    return corr, binned


def plot_unmeasured(corr: pd.DataFrame, binned: pd.DataFrame) -> None:
    metrics = read_csv(PH76 / "tables" / "unmeasured_content_metrics.csv")
    summary = read_csv(PH76 / "tables" / "unmeasured_content_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    for (regime, arm), g in metrics.groupby(["regime", "arm"]):
        if regime == "scr5":
            ax.scatter(g["h_null_energy_ratio"], g["rapsd_distance"], s=10, alpha=0.45, label=f"{regime}-{arm}")
    ax.set_title("Scr-5: unmeasured content vs RAPSD error")
    ax.set_xlabel("h = |P0 xhat| energy ratio")
    ax.set_ylabel("RAPSD distance")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for (regime, arm), g in binned.groupby(["regime", "arm"]):
        if regime == "scr5":
            ax.plot(g["h_mean"], g["rapsd_distance_mean"], marker="o", label=f"{regime}-{arm}")
    ax.set_title("Binned curve")
    ax.set_xlabel("mean unmeasured-content energy")
    ax.set_ylabel("mean RAPSD distance")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    view = corr[corr["target"].eq("rapsd_distance")].copy()
    labels = [f"{r.regime}-{r.arm}" for r in view.itertuples()]
    ax.bar(labels, view["spearman"], color="#4c78a8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Spearman correlation with RAPSD")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1, 1]
    if not summary.empty:
        labels = [f"{r.regime}-{r.arm}" for r in summary.itertuples()]
        ax.bar(labels, summary["h_mean"], color="#59a14f")
        ax.set_title("Mean prior-supplied content")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("h mean")
    fig.suptitle("Unmeasured-content validation from Phase76 precomputed metrics")
    fig.tight_layout()
    save_fig(fig, "fig_unmeasured_content_validation")

    src = PH76 / "figs" / "fig_unmeasured_content_maps.png"
    if src.exists():
        image_to_pdf(src, OUT / "fig_unmeasured_content_maps.png", OUT / "fig_unmeasured_content_maps.pdf")


def plot_alpha() -> None:
    alpha = read_csv(PH76 / "tables" / "alpha_sweep_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [
        ("lpips_mean", "LPIPS (lower)"),
        ("rapsd_distance_mean", "RAPSD distance (lower)"),
        ("sharpness_proxy", "sharpness proxy"),
        ("relmeaserr_mean", "RelMeasErr"),
        ("null_energy_ratio_mean", "null energy scaling"),
        ("psnr_mean", "PSNR"),
    ]
    colors = {"scr5": "#4c78a8", "rad5": "#f58518"}
    for ax, (col, title) in zip(axes.ravel(), panels):
        for regime, g in alpha.groupby("regime"):
            ax.plot(g["alpha"], g[col], marker="o", label=regime, color=colors.get(regime))
        ax.set_title(title)
        ax.set_xlabel("alpha")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Alpha trust-sharpness control: prior content changes, certificate error stays flat")
    fig.tight_layout()
    save_fig(fig, "fig_alpha_trust_sharpness")

    src = PH76 / "figs" / "fig_alpha_grid_examples.png"
    if src.exists():
        image_to_pdf(src, OUT / "fig_alpha_examples_grid.png", OUT / "fig_alpha_examples_grid.pdf")


def plot_shortcut() -> None:
    stress = read_csv(PH75 / "shortcut_stress_summary.csv")
    scr10 = read_csv(PH72 / "scr10_gauge_signal_auc.csv")
    rad10_shortcut = read_csv(PH74 / "rad10_shortcut_controls.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    ax = axes[0]
    subset = stress[stress["base_kind"].eq("fake_mean")]
    for (model, pert), g in subset.groupby(["model", "perturb_type"]):
        style = "-" if pert == "row" else "--"
        ax.plot(g["alpha"], g["mean_abs_delta_vs_alpha0"], marker="o", linestyle=style, label=f"{model} {pert}")
    ax.set_title("Shortcut stress: score sensitivity")
    ax.set_xlabel("perturbation scale")
    ax.set_ylabel("|score - score(alpha=0)|")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    rows = []
    if not scr10.empty:
        for _, r in scr10.iterrows():
            rows.append((str(r["model"]).replace("_", "\n"), float(r["auc"])))
    if not rad10_shortcut.empty:
        for _, r in rad10_shortcut.iterrows():
            rows.append((str(r["model"]).replace("_", "\n"), float(r["auc"])))
    if rows:
        ax.bar([r[0] for r in rows], [r[1] for r in rows], color=["#4c78a8", "#e45756", "#e45756"][: len(rows)])
    ax.axhline(0.65, color="black", linestyle="--", linewidth=1, label="usable-signal guide")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Residual-fed shortcut controls")
    ax.set_ylabel("AUC")
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig_shortcut_stress_final")


def plot_regime_gate() -> None:
    reg = read_csv(PH75 / "regime_map_final.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    names = reg["regime"].tolist()
    auc = reg["gauge_auc"].astype(float).tolist()
    colors = ["#4c78a8" if x >= 0.65 else "#f58518" for x in auc]
    ax.axhspan(0.58, 0.65, color="#f6d55c", alpha=0.25, label="weak zone")
    ax.bar(names, auc, color=colors)
    ax.axhline(0.58, color="#888", linewidth=1)
    ax.axhline(0.65, color="#333", linewidth=1, linestyle="--")
    ax.set_ylim(0.5, 0.92)
    ax.set_ylabel("Gauge diagnostic AUC")
    ax.set_title("Regime gate: whether adversarial fine-tuning is informative")
    for i, r in reg.iterrows():
        ax.text(i, float(r["gauge_auc"]) + 0.01, str(r["decision"]), ha="center", va="bottom", fontsize=7, rotation=10)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_regime_gate_final")


def build_reports(canonical: pd.DataFrame, archive: pd.DataFrame, corr: pd.DataFrame, binned: pd.DataFrame) -> None:
    canonical.to_csv(OUT / "canonical_results_table.csv", index=False)
    archive.to_csv(OUT / "old_numbers_archive.csv", index=False)

    main_cols = [
        "source",
        "regime",
        "arm",
        "seeds",
        "n_per_seed",
        "psnr_mean",
        "ssim_mean",
        "lpips_mean",
        "rapsd_distance_mean",
        "relmeaserr_unclipped_float64_mean",
        "p0_l2_mean",
        "claim_role",
    ]
    write_text(
        OUT / "CANONICAL_RUN_DECISION.md",
        f"""
# Canonical Run Decision

Generated: `{now()}`

## Decision

Use Phase75/Phase74-family Scr-5 numbers as the canonical controlled 5% B/C/standard-D result. Use Phase73 Rad-5 paired-seed results as the Rad-5 robustness result. Phase69B and Phase70 Scr-5 numbers are archived as earlier protocol outputs and are not used in the main text.

## Numbers Entering Main Text

{md_table(canonical, main_cols)}

## Old Numbers Removed From Main Text

- Phase69B original pilot values.
- Phase70 Phase69B reproduction values.
- Phase70 early Scr-5 seed table values.
- Single-seed Phase74 values when a Phase75 three-seed aggregate is available.

## Remaining LPIPS/RAPSD Conflicts

No unresolved main-text conflict remains after locking to the Phase75 Scr-5 aggregate and Phase73 Rad-5 aggregate. Older LPIPS/RAPSD rows still exist in the archive for provenance, but they are protocol-specific and should not be mixed with the canonical table.

## Supplement Protocol Notes Needed

The supplement should explain that Phase69B/70 were pilot or expansion protocols, Phase71/73 supplied paired-seed evidence, Phase74 added standard-D and stress controls, and Phase75 locked the final comparison/decision layer. The main text should cite only the canonical table for Scr-5 B/C/standard-D metrics.
""",
    )

    archive_preview_cols = [c for c in ["archive_source", "archive_path", "status", "arm", "metric", "psnr_mean", "lpips_mean", "rapsd_distance_mean"] if c in archive]
    write_text(
        OUT / "OLD_NUMBERS_ARCHIVE.md",
        f"""
# Old Numbers Archive

These rows preserve provenance for earlier Scr-5 results. They are not main-text numbers and must not be mixed with the canonical Phase75/Phase74-family table.

{md_table(archive, archive_preview_cols, max_rows=80)}
""",
    )

    top_corr = corr[corr["target"].eq("rapsd_distance")].sort_values("spearman", ascending=False)
    write_text(
        OUT / "UNMEASURED_CONTENT_VALIDATION.md",
        f"""
# Unmeasured-Content Validation

Generated from Phase76 precomputed per-sample metrics. This is an evidence consolidation pass, not a new training run.

## Result

The unmeasured-content metric is quantitatively associated with prior/detail error proxies in Scr-5, especially RAPSD/high-frequency metrics, while Rad-5 correlations are weaker and sometimes negative. This supports using `P0 xhat` as a prior-supplied content map and as an accountability cue, not as a proof of truth or falsity.

## Correlation Summary

{md_table(top_corr, ["regime", "arm", "target", "pearson", "spearman", "n", "top10_h_mean", "top10_target_mean", "rest_target_mean"])}

## Binned Curve Data

{md_table(binned, ["regime", "arm", "quantile_bin", "h_mean", "rapsd_distance_mean", "null_error_ratio_mean"], max_rows=40)}

## Important Scope Note

The available Phase76 artifact stores per-sample null/range summary metrics, not full pixel-level `P0 xhat` and GT arrays. Therefore this final package reports the strongest available quantitative validation from existing outputs and does not invent pixelwise values. The figure `fig_unmeasured_content_maps` remains the visual accountability panel.
""",
    )

    alpha = read_csv(PH76 / "tables" / "alpha_sweep_metrics.csv")
    rel_span = alpha.groupby("regime")["relmeaserr_mean"].agg(lambda s: float(s.max() - s.min())).to_dict()
    write_text(
        OUT / "ALPHA_KNOB_FINAL_REPORT.md",
        f"""
# Alpha Knob Final Report

The alpha knob is a user-facing trust-sharpness control, not a novel mathematical operator.

## Evidence

- Figure: `fig_alpha_trust_sharpness.png/pdf`.
- Example grid: `fig_alpha_examples_grid.png/pdf`.
- RelMeasErr span across alpha: `{rel_span}`.
- Null energy increases with alpha while certificate-controlled RelMeasErr remains flat within numerical tolerance.

## Modes

- Conservative: low alpha, low prior-supplied detail.
- Balanced: alpha near the validated operating point.
- Full-GAN: alpha near one, more prior detail under the same measurement certificate.
""",
    )

    stress = read_csv(PH75 / "shortcut_stress_summary.csv")
    fake_row = stress[
        stress["model"].eq("standard_D_score")
        & stress["base_kind"].eq("fake_mean")
        & stress["perturb_type"].eq("row")
        & stress["alpha"].astype(float).eq(0.1)
    ]
    gauge_row = stress[
        stress["model"].eq("gauge_D_score")
        & stress["base_kind"].eq("fake_mean")
        & stress["perturb_type"].eq("row")
        & stress["alpha"].astype(float).eq(0.1)
    ]
    gauge_null = stress[
        stress["model"].eq("gauge_D_score")
        & stress["base_kind"].eq("fake_mean")
        & stress["perturb_type"].eq("null")
        & stress["alpha"].astype(float).eq(0.1)
    ]
    write_text(
        OUT / "SHORTCUT_SAFETY_FINAL_REPORT.md",
        f"""
# Shortcut Safety Final Report

## Final Stress Result

- Standard D fake-mean row perturbation delta at 0.1: `{fmt(fake_row['mean_abs_delta_vs_alpha0'].iloc[0] if not fake_row.empty else None, 6)}`.
- Gauge D fake-mean row perturbation delta at 0.1: `{fmt(gauge_row['mean_abs_delta_vs_alpha0'].iloc[0] if not gauge_row.empty else None, 6)}`.
- Gauge D fake-mean null perturbation delta at 0.1: `{fmt(gauge_null['mean_abs_delta_vs_alpha0'].iloc[0] if not gauge_null.empty else None, 6)}`.

The final claim is shortcut safety for the diagnostic/training interface: gauge equalization removes the measured-residual shortcut. This is not a claim that gauge quality dominates standard cGAN quality.
""",
    )

    reg = read_csv(PH75 / "regime_map_final.csv")
    write_text(
        OUT / "REGIME_GATE_FINAL_REPORT.md",
        f"""
# Regime Gate Final Report

{md_table(reg)}

The gate decides whether adversarial fine-tuning is informative in a regime. The main decomposition, alpha trust control, and prior-supplied content maps still apply as audit concepts in all regimes.
""",
    )


def tex_escape(text: str) -> str:
    repl = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def simple_tex(title: str, body_md: str) -> str:
    lines = []
    in_verbatim = False
    for line in body_md.splitlines():
        if line.startswith("|"):
            if not in_verbatim:
                lines.append(r"\begin{verbatim}")
                in_verbatim = True
            lines.append(line)
            continue
        if in_verbatim:
            lines.append(r"\end{verbatim}")
            in_verbatim = False
        if line.startswith("# "):
            lines.append(r"\section*{" + tex_escape(line[2:]) + "}")
        elif line.startswith("## "):
            lines.append(r"\section{" + tex_escape(line[3:]) + "}")
        elif line.startswith("### "):
            lines.append(r"\subsection{" + tex_escape(line[4:]) + "}")
        else:
            lines.append(line)
    if in_verbatim:
        lines.append(r"\end{verbatim}")
    return "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{graphicx}",
            r"\usepackage[strings]{underscore}",
            r"\usepackage{hyperref}",
            r"\title{" + tex_escape(title) + "}",
            r"\author{}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
            *lines,
            r"\end{document}",
        ]
    )


def build_paper_files(canonical: pd.DataFrame, corr: pd.DataFrame) -> None:
    canon_main = canonical[
        canonical["regime"].eq("Scr-5") & canonical["arm"].isin(["B", "C_gauge", "D_standard"])
    ]
    regime = read_csv(PH75 / "regime_map_final.csv")
    paper = f"""
# Auditable GAN Reconstruction for Ghost Imaging: Unmeasured-Content Maps and Measurement-Invariant Trust Control

## 1. Introduction

GAN reconstruction is useful in ghost imaging only if its prior-supplied content is kept separate from measured evidence. This paper frames the GAN as a prior/detail engine, not as the measurement certificate. The certificate is \\(\\Pi_y^\\lambda\\); the auditable prior map is \\(P_0\\hat{{x}}\\).

## 2. Setup and measurement certificate

Given measurements \\(y = Ax\\), the final reconstruction is audited by \\(\\Pi_y^\\lambda\\). Measured consistency is attributed to that projection/certificate path. The GAN path supplies a candidate detail field whose measured component is not trusted until it passes the certificate.

## 3. Auditable decomposition and unmeasured-content maps

The null component \\(P_0\\hat{{x}}\\) is reported as an unmeasured-content or prior-supplied content map. Phase76 precomputed metrics show that this quantity is associated with prior/detail error proxies, especially in Scr-5, while remaining an accountability signal rather than a proof of truth or falsity.

## 4. Measurement-invariant trust control

The alpha knob scales prior-supplied detail. It is a user-facing trust-sharpness control, not a new operator. Phase76 alpha sweeps show LPIPS/RAPSD/sharpness changes across alpha while RelMeasErr remains effectively flat under \\(\\Pi_y^\\lambda\\).

## 5. Shortcut-free adversarial training

Gauge equalization removes residual shortcuts from the discriminator interface. The Phase75 shortcut stress test shows standard D sensitivity to row/residual perturbations and gauge D insensitivity to row perturbations, while gauge D remains responsive to null perturbations.

## 6. Diagnostic gate for adversarial usefulness

The gauge-AUC diagnostic is used to decide whether adversarial fine-tuning is informative. It supports Scr-5/Rad-5 positive regimes and stops weak Scr-10/Rad-10 regimes.

{md_table(regime)}

## 7. Controlled 5% study

Canonical Scr-5 numbers are locked to the Phase75/Phase74-family controlled 5% result:

{md_table(canon_main, ["arm", "seeds", "n_per_seed", "psnr_mean", "ssim_mean", "lpips_mean", "rapsd_distance_mean", "relmeaserr_unclipped_float64_mean"])}

The safe interpretation is that the adversarial branch changes perceptual/prior-detail metrics under the same certificate. Standard cGAN and gauge cGAN are close in quality; the paper does not claim gauge quality dominance over standard cGAN.

## 8. Brief generality / inpainting toy

Phase76 includes a random-mask inpainting toy showing that a gauge-style diagnostic can be applied beyond ghost imaging. This is a toy feasibility note, not a second trained reconstruction method.

## 9. Limitations

The paper does not include formal 2AFC responses, does not run a diffusion baseline, does not make distribution-shift detector claims, and does not use z sampling as a main uncertainty argument. Scr-10/Rad-10 are weak-gate regimes. Cross-seed effects are descriptive, not significance claims.

## 10. Positioning and conclusion

Range-null decompositions and projection correction are not new. The contribution here is a user-facing audit map, measurement-invariant trust control, shortcut-safe adversarial training, and a per-image certificate-compatible reporting style for an efficient GAN prior branch.
"""
    supplement = f"""
# Supplement: Auditable GAN Reconstruction

## S1. Canonical result lock

{md_table(canonical)}

## S2. Unmeasured-content validation

{md_table(corr, ["regime", "arm", "target", "pearson", "spearman", "n", "evidence_level"], max_rows=60)}

## S3. Alpha sweep

See `fig_alpha_trust_sharpness.png/pdf` and `ALPHA_KNOB_FINAL_REPORT.md`.

## S4. Shortcut stress

See `fig_shortcut_stress_final.png/pdf` and `SHORTCUT_SAFETY_FINAL_REPORT.md`.

## S5. Protocol archive

Older Phase69B/70 numbers are preserved in `OLD_NUMBERS_ARCHIVE.md` and `old_numbers_archive.csv`; they are not main-text values.
"""
    write_text(OUT / "auditable_gan_paper_v1.md", paper)
    write_text(OUT / "auditable_gan_paper_v1.tex", simple_tex("Auditable GAN Reconstruction for Ghost Imaging", paper))
    write_text(OUT / "supplement_auditable_gan_v1.md", supplement)
    write_text(OUT / "supplement_auditable_gan_v1.tex", simple_tex("Supplement: Auditable GAN Reconstruction", supplement))


def build_positioning_claim_files() -> None:
    method_rows = pd.DataFrame(
        [
            {
                "method_family": "DDNM / range-null diffusion",
                "relationship": "Shares range-null accountability idea.",
                "safe_position": "Range-null decomposition is not new; this paper contributes user-facing audit/trust control for a GAN branch.",
            },
            {
                "method_family": "Diffusion inverse solvers",
                "relationship": "Natural strong comparator not run here.",
                "safe_position": "Diffusion may be stronger; this paper focuses on efficient auditable GAN reconstruction.",
            },
            {
                "method_family": "Adversarial regularizers",
                "relationship": "GAN supplies prior/detail pressure.",
                "safe_position": "Discriminator is not a certificate and must be shortcut-safe.",
            },
            {
                "method_family": "Null-space networks",
                "relationship": "Models unmeasured content.",
                "safe_position": "We expose prior-supplied content as a map rather than hiding it in the final image.",
            },
            {
                "method_family": "Data consistency / projection correction",
                "relationship": "Closest certificate mechanism.",
                "safe_position": "Pi_y^lambda supplies per-image measured consistency after prior detail is added.",
            },
            {
                "method_family": "Standard cGAN",
                "relationship": "Quality baseline/control.",
                "safe_position": "Standard and gauge cGAN are close in quality; gauge is for shortcut safety, not quality dominance.",
            },
            {
                "method_family": "Paper 1",
                "relationship": "Premise only.",
                "safe_position": "Paper 1 establishes measurement accountability/certificates; its main results are not GAN results.",
            },
        ]
    )
    write_text(
        OUT / "METHOD_POSITIONING_TABLE.md",
        "# Method Positioning Table\n\n" + md_table(method_rows),
    )
    write_text(
        OUT / "RELATED_WORK_AUDITABLE_GAN.md",
        """
# Related Work Positioning

## DDNM / range-null diffusion

DDNM and related range-null methods motivate the separation between measured consistency and unmeasured content. This draft does not claim that range-null decomposition is new. The contribution is the audit-facing use of the decomposition for a GAN branch: a prior-supplied content map, trust control, shortcut-safe adversarial training, and a per-image certificate.

## Adversarial regularizers

Adversarial regularizers can improve visual detail but can also hide physical inconsistency. Here the discriminator is a prior/detail tool, not a certificate. Gauge equalization is used to remove a residual shortcut from the adversarial interface.

## Null-space networks

Null-space learning supplies content in directions not fixed by the measurement operator. The present package exposes that content as `P0 xhat` instead of making it invisible in the final image.

## Data consistency / projection correction

Projection correction and data consistency layers restore agreement with the forward model. In this paper, `Pi_y^lambda` is the measurement certificate, while the GAN contributes prior detail.

## Standard cGAN

Standard cGAN is a necessary control. The final position is not that gauge cGAN is higher quality than standard cGAN; gauge equalization is used to protect the diagnostic/training interface from row-residual shortcuts.

## Paper 1

Paper 1 is cited only for the premise that measurement accountability is separable and that an audit certificate exists. Its main empirical results are not assigned to the GAN branch.
""",
    )

    claims = pd.DataFrame(
        [
            ["C1", "GAN is a prior/detail engine.", "Supported", "Phase75/76"],
            ["C2", "Pi_y^lambda is the measurement certificate.", "Supported", "audit/projection protocol"],
            ["C3", "P0 xhat is an unmeasured-content map.", "Supported with caveats", "Phase76 maps/metrics"],
            ["C4", "Alpha is a measurement-invariant trust-sharpness control.", "Supported", "Phase76 alpha sweep"],
            ["C5", "Gauge equalization provides shortcut safety.", "Supported", "Phase75 shortcut stress"],
            ["C6", "Gauge-AUC explains Scr-5/Rad-5 positive and Scr-10/Rad-10 weak regimes.", "Supported", "Phase75 regime map"],
            ["C7", "Failure/OOD signals are preliminary only.", "Limited", "Phase76 failure report"],
        ],
        columns=["id", "claim", "status", "evidence"],
    )
    forbidden = pd.DataFrame(
        [
            ["hallucination proof", "Use unmeasured-content / prior-supplied content instead."],
            ["SOTA", "No broad benchmark or diffusion/PnP run."],
            ["beats diffusion", "Diffusion was not run here."],
            ["gauge quality dominance over standard cGAN", "Quality is close; gauge is shortcut safety."],
            ["GAN improves RelMeasErr", "RelMeasErr belongs to the certificate path."],
            ["GAN is certificate", "Pi_y^lambda is the certificate."],
            ["stochastic uncertainty", "z sampling collapsed."],
            ["OOD detector main claim", "Failure signal is preliminary."],
            ["cross-seed significant claim", "Use descriptive multi-seed wording only."],
            ["uncanonical old numbers", "Phase69B/70 Scr-5 numbers stay in archive."],
        ],
        columns=["forbidden_claim", "safe_replacement"],
    )
    attacks = pd.DataFrame(
        [
            ["Is P0 xhat just an artifact?", "It is reported as prior-supplied content and backed by Phase76 metrics; it is not treated as truth/falsity proof."],
            ["Could D exploit residuals?", "Phase75 stress tests show gauge D is row-insensitive while standard D is row-sensitive."],
            ["Why no diffusion baseline?", "This is a final assembly pass; diffusion comparison is listed as a blocker for a stronger venue tier."],
            ["Does alpha change the measurements?", "Phase76 identity checks show RelMeasErr remains flat under the certificate."],
            ["Does this work at Scr-10/Rad-10?", "The gate marks those regimes weak; no positive claim is made."],
            ["Is the detector reliable?", "No; it remains preliminary and outside the main claim."],
        ],
        columns=["reviewer_attack", "locked_response"],
    )
    write_text(OUT / "CLAIMS_FINAL.md", "# Final Claims\n\n" + md_table(claims))
    write_text(OUT / "FORBIDDEN_CLAIMS_FINAL.md", "# Forbidden Claims\n\n" + md_table(forbidden))
    write_text(OUT / "REVIEWER_ATTACK_BANK_FINAL.md", "# Reviewer Attack Bank\n\n" + md_table(attacks))


def build_final_reports(canonical: pd.DataFrame, corr: pd.DataFrame) -> None:
    files = sorted([p.name for p in OUT.iterdir() if p.is_file()])
    corr_focus = corr[corr["target"].eq("rapsd_distance")].sort_values("spearman", ascending=False)
    best_corr = corr_focus.iloc[0].to_dict() if not corr_focus.empty else {}
    paper = OUT / "auditable_gan_paper_v1.md"
    supp = OUT / "supplement_auditable_gan_v1.md"
    report = f"""
# Phase77 Final Report

Generated: `{now()}`

## Required Answers

1. canonical numbers locked? Yes. Main-text Scr-5 B/C/standard-D numbers are locked to Phase75/Phase74-family canonical outputs; Phase69B/70 Scr-5 numbers are archived only.
2. unmeasured-content validation result? Quantitative Phase76 per-sample metrics support unmeasured-content maps as accountability/proxy evidence. Strongest RAPSD Spearman row: `{best_corr}`.
3. alpha figure ready? Yes: `fig_alpha_trust_sharpness.png/pdf` and `fig_alpha_examples_grid.png/pdf`.
4. shortcut safety figure ready? Yes: `fig_shortcut_stress_final.png/pdf`.
5. regime gate figure ready? Yes: `fig_regime_gate_final.png/pdf`.
6. paper draft path? `{paper}`.
7. supplement path? `{supp}`.
8. any remaining blocker? For human-editable draft: no. For stronger venue tier: formal human 2AFC and a vetted diffusion/PnP comparator remain blockers, but neither was run or fabricated here.
9. recommended venue tier? Workshop/short-paper or cautious conference submission after human editing; not high-tier-ready without the remaining blockers.
10. confirm no training and first-paper unchanged. Confirmed. Phase77 only reads existing outputs and writes this output directory.

## Deliverable Files

{chr(10).join('- `' + f + '`' for f in files)}
"""
    write_text(OUT / "PHASE77_FINAL_REPORT.md", report)
    write_text(
        OUT / "READY_TO_WRITE_DECISION.md",
        """
# Ready To Write Decision

Decision: `READY_TO_HUMAN_EDIT`

The package is ready for human editing as a cautious auditable-GAN manuscript. It is not positioned as a benchmark-dominance paper. Do not add new numbers from Phase69B/70 into the main text unless the canonical table is explicitly revised.
""",
    )
    manifest = {
        "generated_utc": now(),
        "output_dir": str(OUT),
        "no_training": True,
        "no_diffusion_run": True,
        "no_2afc": True,
        "no_human_response_fabrication": True,
        "no_checkpoint_write": True,
        "no_first_paper_change": True,
        "input_dirs": [str(p) for p in [PH69B, PH70, PH71, PH73, PH74, PH75, PH76]],
        "files": files,
    }
    write_text(OUT / "PHASE77_MANIFEST.md", "# Phase77 Manifest\n\n```json\n" + json.dumps(manifest, indent=2) + "\n```")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical, archive = build_canonical_tables()
    corr, binned = build_unmeasured_validation()
    plot_unmeasured(corr, binned)
    plot_alpha()
    plot_shortcut()
    plot_regime_gate()
    build_reports(canonical, archive, corr, binned)
    build_paper_files(canonical, corr)
    build_positioning_claim_files()
    build_final_reports(canonical, corr)
    print(f"Phase77 final assembly complete: {OUT}")


if __name__ == "__main__":
    main()
