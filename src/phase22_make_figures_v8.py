from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from .phase20_common import (
    BLUE,
    GRAY,
    GREEN,
    LIGHT_BLUE,
    LIGHT_GRAY,
    LIGHT_GREEN,
    LIGHT_ORANGE,
    LIGHT_RED,
    METHOD_LABEL,
    METHOD_ORDER,
    ORANGE,
    PURPLE,
    RED,
    STL_METHODS,
    as_float,
    crop_sample,
    ensure_dir,
    registry_by_id,
    setup_matplotlib,
    table,
)


OUT = Path("E:/ns_mc_gan_gi/outputs_phase22_submission_v8")
FIG_DIR = OUT / "figures"


def save_figure(fig: plt.Figure, stem: str, exts: tuple[str, ...] = ("pdf", "png", "svg"), dpi: int = 300) -> None:
    out = ensure_dir(FIG_DIR)
    for ext in exts:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(out / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def arrow(ax, p0, p1, color=GRAY, lw=1.1, rad=0.0) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def box(ax, xy, w, h, text, fc, ec, fs=7.4) -> None:
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.014,rounding_size=0.025", fc=fc, ec=ec, lw=1.0))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _hadamard(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1]])
    h = _hadamard(n // 2)
    return np.block([[h, h], [h, -h]])


def fig1_mechanism_v8() -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 5, figsize=(7.35, 3.05))
    titles = [
        "(a) Acquisition",
        "(b) Underdetermined set",
        "(c) Data + residual",
        "(d) Insert + project",
        "(e) Audit",
    ]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(titles[i], fontsize=7.8, pad=5, fontweight="bold" if i == 0 else None)

    rng = np.random.default_rng(22)
    ax = axes[0]
    for j, x0 in enumerate([0.04, 0.13, 0.22]):
        ax.imshow(rng.choice([-1, 1], size=(8, 8)), cmap="gray", extent=(x0, x0 + 0.20, 0.64 - 0.055 * j, 0.84 - 0.055 * j), interpolation="nearest")
        ax.add_patch(Rectangle((x0, 0.64 - 0.055 * j), 0.20, 0.20, fill=False, ec=BLUE, lw=0.75))
    ax.add_patch(Ellipse((0.56, 0.64), 0.22, 0.30, fc=LIGHT_GRAY, ec=GRAY, lw=1.0))
    ax.text(0.56, 0.64, "object", ha="center", va="center", fontsize=7.1)
    ax.add_patch(Circle((0.86, 0.64), 0.068, fc=LIGHT_BLUE, ec=BLUE, lw=1.0))
    ax.text(0.86, 0.64, "bucket", ha="center", va="center", fontsize=6.5)
    arrow(ax, (0.40, 0.66), (0.47, 0.65), BLUE)
    arrow(ax, (0.68, 0.64), (0.78, 0.64), BLUE)
    ax.text(0.52, 0.22, "patterns + scalar\nmeasurements", ha="center", fontsize=6.9, color=BLUE)

    ax = axes[1]
    ax.add_patch(Ellipse((0.50, 0.57), 0.78, 0.38, angle=-14, fc=LIGHT_BLUE, ec=BLUE, lw=1.1))
    ax.plot([0.16, 0.84], [0.34, 0.78], color=GREEN, lw=1.45)
    arrow(ax, (0.27, 0.41), (0.74, 0.72), GREEN)
    ax.scatter([0.35, 0.62], [0.50, 0.62], s=32, color=[BLUE, ORANGE], edgecolor="white", lw=0.5, zorder=3)
    ax.text(0.50, 0.24, r"$\mathcal{C}_y$", ha="center", fontsize=9.5, color=BLUE)
    ax.text(0.50, 0.12, "null-space\ndirections", ha="center", fontsize=6.8, color=GREEN)

    ax = axes[2]
    box(ax, (0.07, 0.64), 0.40, 0.16, r"$x_{\rm data}$", LIGHT_BLUE, BLUE, 8.0)
    box(ax, (0.55, 0.64), 0.37, 0.16, r"$G_\theta$", LIGHT_ORANGE, ORANGE, 8.0)
    arrow(ax, (0.47, 0.72), (0.55, 0.72), ORANGE)
    ax.text(0.27, 0.49, "measured\ncomponent", ha="center", fontsize=6.9, color=BLUE)
    ax.text(0.74, 0.49, "neural\nresidual", ha="center", fontsize=6.9, color=ORANGE)
    ax.add_patch(Rectangle((0.13, 0.21), 0.72, 0.13, fc="#F7F7F7", ec=GRAY, lw=0.9))
    ax.text(0.49, 0.275, "separate fixed and missing parts", ha="center", va="center", fontsize=6.8)

    ax = axes[3]
    box(ax, (0.08, 0.66), 0.32, 0.15, r"$P_N$", LIGHT_GREEN, GREEN, 8.2)
    box(ax, (0.60, 0.66), 0.32, 0.15, r"$\Pi_y$", LIGHT_GREEN, GREEN, 8.2)
    arrow(ax, (0.40, 0.735), (0.60, 0.735), GREEN)
    ax.text(0.24, 0.48, "insert missing\ncomponent", ha="center", fontsize=6.8, color=GREEN)
    ax.text(0.76, 0.48, "project to\nmeasurements", ha="center", fontsize=6.8, color=GREEN)
    ax.add_patch(Ellipse((0.50, 0.28), 0.68, 0.21, angle=-8, fc=LIGHT_BLUE, ec=BLUE, lw=0.9))
    ax.text(0.50, 0.28, r"back to $\mathcal{C}_y$", ha="center", va="center", fontsize=7.0)

    ax = axes[4]
    box(ax, (0.14, 0.66), 0.32, 0.15, r"$\hat{x}$", LIGHT_BLUE, BLUE, 9.2)
    box(ax, (0.58, 0.66), 0.32, 0.15, r"$A\hat{x}$", LIGHT_RED, RED, 8.0)
    arrow(ax, (0.46, 0.735), (0.58, 0.735), RED)
    arrow(ax, (0.74, 0.66), (0.36, 0.36), RED, rad=-0.18)
    ax.add_patch(FancyBboxPatch((0.17, 0.22), 0.64, 0.17, boxstyle="round,pad=0.015,rounding_size=0.025", fc=LIGHT_GREEN, ec=GREEN, lw=0.9))
    ax.text(0.49, 0.305, "compare with y\nRelMeasErr", ha="center", va="center", fontsize=6.9, color=GREEN)

    fig.text(0.5, 0.026, "The neural component fills unmeasured structure, while the projection and audit keep the reconstruction tied to the bucket measurements.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.018, right=0.995, top=0.82, bottom=0.21, wspace=0.055)
    save_figure(fig, "fig1_mechanism_v8")


def fig2_primary_metrics_v8() -> None:
    setup_matplotlib()
    reg = registry_by_id()
    groups = [
        (["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab"], "STL-10 5%", ["Rad-5", "Scr-5"], 20.0, 0.60),
        (["rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"], "STL-10 10%", ["Rad-10", "Scr-10"], 22.0, 0.65),
        (["mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"], "Simple 5%", ["MNIST-5", "Fashion-5"], 25.0, 0.80),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.9), sharey="row")
    for i, (mids, title, labels, psnr_thr, ssim_thr) in enumerate(groups):
        x = np.arange(len(mids))
        colors = [BLUE, GREEN] if i < 2 else [PURPLE, ORANGE]
        vals_by_row = [[as_float(reg[mid]["psnr"]) for mid in mids], [as_float(reg[mid]["ssim"]) for mid in mids]]
        for row, (vals, thr, ylabel) in enumerate(zip(vals_by_row, [psnr_thr, ssim_thr], ["PSNR (dB)", "SSIM"])):
            ax = axes[row, i]
            ax.bar(x, vals, color=colors, width=0.60)
            ax.axhline(thr, color=RED, ls="--", lw=0.95)
            ax.text(0.02, thr + (0.30 if row == 0 else 0.018), "threshold", color=RED, fontsize=6.3, ha="left")
            top = max(vals + [thr]) + (1.15 if row == 0 else 0.08)
            ax.set_ylim(0, top if row == 0 else min(1.05, top))
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(x)
            if row == 0:
                ax.set_title(title)
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(labels)
            for xi, yi in zip(x, vals):
                ax.text(xi, yi + (0.25 if row == 0 else 0.014), f"{yi:.2f}" if row == 0 else f"{yi:.3f}", ha="center", fontsize=7.0)
            if i == 0:
                ax.set_ylabel(ylabel)
        axes[0, i].text(-0.13, 1.07, f"({chr(ord('a') + i)})", transform=axes[0, i].transAxes, fontsize=9, fontweight="bold")
    fig.text(0.5, 0.012, "Dashed lines are predefined operational thresholds used to summarize reconstruction quality in this study.", ha="center", fontsize=7.2)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.13, wspace=0.22, hspace=0.12)
    save_figure(fig, "fig2_primary_metrics_v8")


def fig3_qualitative_reconstruction_v8() -> None:
    setup_matplotlib()
    rows = [(mid, 0) for mid in STL_METHODS]
    cols = ["GT", "Backprojection", "Reconstruction", "Absolute error"]
    fig = plt.figure(figsize=(7.35, 6.7))
    gs = fig.add_gridspec(len(rows), 5, width_ratios=[1, 1, 1, 1, 0.055], wspace=0.06, hspace=0.09)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(len(rows))])
    cax = fig.add_subplot(gs[:, 4])
    err_arrays = []
    cells_by_row = []
    for mid, sample_idx in rows:
        cells = crop_sample(mid, sample_idx)
        gt = np.asarray(cells["GT"].convert("L"), dtype=np.float32) / 255.0
        recon = np.asarray(cells["Reconstruction"].convert("L"), dtype=np.float32) / 255.0
        err = np.abs(gt - recon)
        err_arrays.append(err)
        cells_by_row.append(cells)
    vmax = max(0.12, float(np.quantile(np.concatenate([e.ravel() for e in err_arrays]), 0.99)))
    last_im = None
    for ridx, ((mid, _), cells, err) in enumerate(zip(rows, cells_by_row, err_arrays)):
        for cidx, col in enumerate(cols):
            ax = axes[ridx, cidx]
            if col == "Absolute error":
                last_im = ax.imshow(err, cmap="inferno", vmin=0, vmax=vmax)
            else:
                ax.imshow(cells[col], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ridx == 0:
                ax.set_title(col, fontsize=9)
            if cidx == 0:
                ax.set_ylabel(METHOD_LABEL[mid], rotation=0, labelpad=32, va="center", fontsize=8.5)
    if last_im is not None:
        fig.colorbar(last_im, cax=cax).ax.tick_params(labelsize=6.5)
    fig.text(0.5, 0.014, "Representative evaluation samples are enlarged for visibility. Error maps use a shared 99th-percentile scale.", ha="center", fontsize=7.2)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.92, bottom=0.08)
    save_figure(fig, "fig3_qualitative_reconstruction_v8")


def fig4_measurement_attribution_v8() -> None:
    setup_matplotlib()
    attr = {r["method_id"]: r for r in table("attribution")}
    display = STL_METHODS + [r["method_id"] for r in table("attribution") if r["method_id"].startswith("stl10_hadamard")]
    display = list(dict.fromkeys(display))
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.85))

    ax = axes[0, 0]
    rng = np.random.default_rng(4)
    patterns = [
        ("Rademacher", rng.choice([-1, 1], size=(16, 16))),
        ("Scrambled\nHadamard", _hadamard(16)[rng.permutation(16), :]),
        ("Low-frequency\nHadamard", _hadamard(16)),
    ]
    for i, (label, arr) in enumerate(patterns):
        ax.imshow(arr, cmap="gray", extent=(i, i + 0.8, 0.18, 0.98), interpolation="nearest")
        ax.text(i + 0.4, 0.04, label, ha="center", va="top", fontsize=7.0)
    ax.set_xlim(-0.05, 2.85)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Pattern examples")
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    labels = [METHOD_LABEL.get(mid, "Lowfreq-10" if "10" in mid else "Lowfreq-5") for mid in display]
    x = np.arange(len(display))
    ax = axes[0, 1]
    bp = [as_float(attr[mid]["backproj_psnr"]) for mid in display]
    model = [as_float(attr[mid]["model_psnr"]) for mid in display]
    ax.bar(x - 0.18, bp, width=0.36, color=LIGHT_BLUE, edgecolor=BLUE, label="BP")
    ax.bar(x + 0.18, model, width=0.36, color=BLUE, label="Model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Backprojection vs final model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1, 0]
    delta = [as_float(attr[mid]["delta_psnr"]) for mid in display]
    ax.bar(x, delta, color=[ORANGE if "rademacher" in attr[mid]["measurement_family"] else GREEN if "scrambled" in attr[mid]["measurement_family"] else PURPLE for mid in display], width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"$\Delta$PSNR (dB)")
    ax.set_title("Neural gain")
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1, 1]
    ax.axvspan(0, 10.5, color=LIGHT_ORANGE, alpha=0.32)
    ax.axvspan(10.5, 17.0, color=LIGHT_GREEN, alpha=0.26)
    ax.axvspan(17.0, 25.5, color=LIGHT_BLUE, alpha=0.24)
    ax.text(7.5, 15.0, "weak init.\nlarge gain", ha="center", fontsize=6.8)
    ax.text(13.7, 10.2, "stronger init.\nmoderate gain", ha="center", fontsize=6.8)
    ax.text(20.4, 5.2, "strong init.\nsmall gain", ha="center", fontsize=6.8)
    for mid, label in zip(display, labels):
        fam = attr[mid]["measurement_family"]
        color = ORANGE if "rademacher" in fam else GREEN if "scrambled" in fam else PURPLE
        xx = as_float(attr[mid]["backproj_psnr"])
        yy = as_float(attr[mid]["delta_psnr"])
        ax.scatter([xx], [yy], s=48, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(xx + 0.12, yy + 0.12, label, fontsize=6.8)
    ax.set_xlabel("BP PSNR (dB)")
    ax.set_ylabel(r"$\Delta$PSNR (dB)")
    ax.set_title("Regime map")
    ax.grid(alpha=0.25)
    ax.text(-0.12, 1.05, "(d)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.text(0.5, 0.012, "Final PSNR alone hides whether performance comes from physical initialization, neural refinement, or both.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.13, wspace=0.30, hspace=0.50)
    save_figure(fig, "fig4_measurement_attribution_v8")


def fig5_inference_ablation_v8() -> None:
    setup_matplotlib()
    ab = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    modes = [("full_model", "Full"), ("no_dc_project", "-DC"), ("no_null_project", "-Null"), ("stage1_only", "Stage1"), ("raw_weights", "Raw"), ("ema_weights", "EMA")]
    colors = [BLUE, RED, "#A7AEB8", ORANGE, PURPLE, GREEN]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), sharey=True)
    for idx, (ax, mid) in enumerate(zip(axes.ravel(), STL_METHODS)):
        vals = [as_float(ab[(mid, mode)]["psnr"]) for mode, _ in modes]
        x = np.arange(len(modes))
        ax.bar(x, vals, color=colors, width=0.68)
        for xi, yi in zip(x, vals):
            ax.text(xi, yi + 0.35, f"{yi:.1f}", ha="center", fontsize=6.8)
        ax.set_ylim(0, 26)
        ax.set_title(METHOD_LABEL[mid])
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in modes])
        if idx % 2 == 0:
            ax.set_ylabel("PSNR (dB)")
        ax.grid(axis="y", alpha=0.25)
        ax.text(-0.12, 1.06, f"({chr(ord('a') + idx)})", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.text(0.5, 0.012, "The limited metric change of -Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.", ha="center", fontsize=6.8)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.12, hspace=0.30)
    save_figure(fig, "fig5_inference_ablation_v8")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(STL_METHODS))
    rel = [as_float(ab[(mid, "no_dc_project")]["delta_vs_full_relmeaserr"]) for mid in STL_METHODS]
    ax.bar(x, rel, color=RED, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in STL_METHODS])
    ax.set_ylabel("RelMeasErr increase")
    ax.set_title("No-DC projection increases measurement inconsistency")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "figS1_relmeaserr_ablation_v8", exts=("pdf", "png"))


def fig6_robustness_baselines_v8() -> None:
    setup_matplotlib()
    noise = table("noise")
    perturb = table("perturbation")
    baseline = table("baseline")
    stats = {r["method_id"]: r for r in table("statistics")}
    reg = registry_by_id()
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.72))

    ax = axes[0, 0]
    for mid, color in zip(STL_METHODS, [BLUE, GREEN, PURPLE, ORANGE]):
        sub = sorted([r for r in noise if r["method_id"] == mid], key=lambda r: as_float(r["noise_std"]))
        ax.plot([as_float(r["noise_std"]) for r in sub], [as_float(r["psnr"]) for r in sub], marker="o", lw=1.35, color=color, label=METHOD_LABEL[mid])
    ax.set_title("Finite noise sweep")
    ax.set_xlabel("Noise std")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=6.6)
    ax.text(-0.13, 1.06, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[0, 1]
    mids = STL_METHODS[:2]
    modes = [("shuffle_coefficients", "Shuffle"), ("wrong_sample", "Wrong-y")]
    x = np.arange(len(mids))
    for i, (mode, label) in enumerate(modes):
        vals = []
        for mid in mids:
            row = next(r for r in perturb if r["method_id"] == mid and r["perturbation_mode"] == mode)
            vals.append(as_float(row["psnr_drop_from_normal"]))
        ax.bar(x + (i - 0.5) * 0.34, vals, width=0.34, label=label, color=ORANGE if i == 0 else RED)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids])
    ax.set_ylabel("PSNR drop (dB)")
    ax.set_title("Measurement perturbation")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1, 0]
    tv_best = {}
    for mid in STL_METHODS:
        sub = [r for r in baseline if r["method_id"] == mid and r["baseline"] == "tv_pgd"]
        if sub:
            tv_best[mid] = max(sub, key=lambda r: as_float(r["psnr"]))
    mids = [mid for mid in STL_METHODS if mid in tv_best]
    x = np.arange(len(mids))
    ax.bar(x - 0.18, [as_float(reg[mid]["psnr"]) for mid in mids], width=0.36, color=BLUE, label="Ours")
    ax.bar(x + 0.18, [as_float(tv_best[mid]["psnr"]) for mid in mids], width=0.36, color="#9B8A4E", label="CS-TV")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids])
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("STL-10 CS-TV baseline")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1, 1]
    mids = STL_METHODS
    x = np.arange(len(mids))
    mean = [as_float(stats[mid]["mean_psnr"]) for mid in mids]
    low = [as_float(stats[mid]["ci95_psnr_low"]) for mid in mids]
    high = [as_float(stats[mid]["ci95_psnr_high"]) for mid in mids]
    yerr = np.array([[m - l for m, l in zip(mean, low)], [h - m for h, m in zip(high, mean)]])
    ax.errorbar(x, mean, yerr=yerr, fmt="o", color=BLUE, ecolor=GRAY, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids])
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Bootstrap 95% CI")
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "(d)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.text(0.5, 0.012, "These diagnostics support finite-noise stability and measurement dependence; they do not imply universal robustness.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.13, wspace=0.30, hspace=0.45)
    save_figure(fig, "fig6_robustness_baselines_v8")


def main() -> None:
    fig1_mechanism_v8()
    fig2_primary_metrics_v8()
    fig3_qualitative_reconstruction_v8()
    fig4_measurement_attribution_v8()
    fig5_inference_ablation_v8()
    fig6_robustness_baselines_v8()
    print({"figures": str(FIG_DIR)})


if __name__ == "__main__":
    main()
