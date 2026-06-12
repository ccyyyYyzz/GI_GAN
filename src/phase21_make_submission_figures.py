from __future__ import annotations

import math
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


OUT = Path("E:/ns_mc_gan_gi/outputs_phase21_submission_polish")


def save_figure(fig: plt.Figure, stem: str, exts: tuple[str, ...] = ("pdf", "png", "svg"), dpi: int = 300) -> None:
    out = ensure_dir(OUT / "figures")
    for ext in exts:
        path = out / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
    plt.close(fig)


def arrow(ax, p0, p1, color=GRAY, lw=1.15, rad=0.0) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def box(ax, xy, w, h, text, fc, ec, fs=8.0) -> None:
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.018,rounding_size=0.030", fc=fc, ec=ec, lw=1.05))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def fig1_mechanism() -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 5, figsize=(7.35, 3.05))
    titles = ["Acquisition", "Feasible set", "Data solution", "Null-space residual", "Projection"]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(titles[i], fontsize=8.5, pad=6)
        ax.text(-0.03, 1.04, f"({chr(ord('a') + i)})", transform=ax.transAxes, fontsize=9, fontweight="bold")

    rng = np.random.default_rng(21)
    ax = axes[0]
    for j, x0 in enumerate([0.05, 0.14, 0.23]):
        ax.imshow(rng.choice([0, 1], size=(8, 8)), cmap="gray", extent=(x0, x0 + 0.19, 0.61 - j * 0.05, 0.80 - j * 0.05), interpolation="nearest")
        ax.add_patch(Rectangle((x0, 0.61 - j * 0.05), 0.19, 0.19, fill=False, ec=BLUE, lw=0.8))
    ax.add_patch(Ellipse((0.58, 0.64), 0.22, 0.28, fc=LIGHT_GRAY, ec=GRAY, lw=1.0))
    ax.text(0.58, 0.64, "$x$", ha="center", va="center", fontsize=10.5)
    ax.add_patch(Circle((0.86, 0.64), 0.065, fc=LIGHT_BLUE, ec=BLUE, lw=1.0))
    ax.text(0.86, 0.64, "$y_i$", ha="center", va="center", fontsize=8.5)
    arrow(ax, (0.39, 0.66), (0.47, 0.65), BLUE)
    arrow(ax, (0.69, 0.64), (0.79, 0.64), BLUE)
    ax.text(0.52, 0.23, r"$y_i=\langle a_i,x\rangle+\epsilon_i$", ha="center", fontsize=7.7, color=BLUE)

    ax = axes[1]
    ax.add_patch(Ellipse((0.50, 0.57), 0.75, 0.36, angle=-14, fc=LIGHT_BLUE, ec=BLUE, lw=1.1))
    ax.plot([0.17, 0.84], [0.34, 0.78], color=GREEN, lw=1.5)
    arrow(ax, (0.30, 0.43), (0.75, 0.72), GREEN)
    ax.scatter([0.38, 0.62], [0.52, 0.61], s=32, color=[BLUE, ORANGE], edgecolor="white", lw=0.5, zorder=3)
    ax.text(0.50, 0.23, r"$\mathcal{C}_y=\{x:Ax=y\}$", ha="center", fontsize=7.3)
    ax.text(0.50, 0.12, r"$\mathrm{Null}(A)$, $m\ll n$", ha="center", fontsize=7.3, color=GREEN)

    ax = axes[2]
    box(ax, (0.12, 0.58), 0.76, 0.21, r"$x_{\rm data}$", LIGHT_BLUE, BLUE, 9.0)
    ax.text(0.50, 0.40, "measured row-space\ncomponent", ha="center", fontsize=7.7, color=BLUE)
    ax.text(0.50, 0.16, r"$A^T(AA^T+\lambda I)^{-1}y$", ha="center", fontsize=6.4)

    ax = axes[3]
    box(ax, (0.08, 0.64), 0.36, 0.18, r"$G_\theta(x_{\rm data},z)$", LIGHT_ORANGE, ORANGE, 7.1)
    box(ax, (0.56, 0.64), 0.36, 0.18, r"$P_N$", LIGHT_GREEN, GREEN, 9.0)
    arrow(ax, (0.44, 0.73), (0.56, 0.73), ORANGE)
    ax.add_patch(Rectangle((0.28, 0.30), 0.44, 0.14, fc="#F7F7F7", ec=GRAY, lw=0.9))
    ax.text(0.50, 0.37, "complete missing\nstructure", ha="center", va="center", fontsize=7.4)
    arrow(ax, (0.74, 0.64), (0.59, 0.44), GREEN)

    ax = axes[4]
    box(ax, (0.10, 0.63), 0.34, 0.18, r"$\Pi_y$", LIGHT_GREEN, GREEN, 9.5)
    box(ax, (0.57, 0.63), 0.33, 0.18, r"$\hat{x}$", LIGHT_BLUE, BLUE, 9.5)
    arrow(ax, (0.44, 0.72), (0.57, 0.72), GREEN)
    ax.add_patch(FancyBboxPatch((0.21, 0.31), 0.58, 0.16, boxstyle="round,pad=0.015,rounding_size=0.025", fc=LIGHT_RED, ec=RED, lw=0.9))
    ax.text(0.50, 0.39, r"$A\hat{x}\approx y$", ha="center", va="center", fontsize=8.4, color=RED)
    ax.text(0.50, 0.18, "RelMeasErr check", ha="center", fontsize=7.4, color=GREEN)

    fig.text(0.5, 0.025, "Measurement-consistent null-space reconstruction: preserve measured data, complete missing structure, then project back to the bucket measurements.", ha="center", fontsize=7.3)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.82, bottom=0.22, wspace=0.06)
    save_figure(fig, "fig1_mechanism")


def fig2_primary_metrics() -> None:
    setup_matplotlib()
    reg = registry_by_id()
    groups = [
        (["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab"], "STL-10 5%", 20.0, 0.60),
        (["rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"], "STL-10 10%", 22.0, 0.65),
        (["mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"], "MNIST/Fashion 5%", 25.0, 0.80),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.9), sharey="row")
    for i, (mids, title, psnr_thr, ssim_thr) in enumerate(groups):
        x = np.arange(len(mids))
        colors = [BLUE, GREEN] if i < 2 else [PURPLE, ORANGE]
        psnr = [as_float(reg[mid]["psnr"]) for mid in mids]
        ssim = [as_float(reg[mid]["ssim"]) for mid in mids]
        for row, vals, thr, ylabel in [(0, psnr, psnr_thr, "PSNR (dB)"), (1, ssim, ssim_thr, "SSIM")]:
            ax = axes[row, i]
            ax.bar(x, vals, color=colors, width=0.62)
            ax.axhline(thr, color=RED, ls="--", lw=1.0)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(x)
            if row == 0:
                ax.set_title(title)
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids])
                ax.set_ylim(0, 1.04)
            for xi, yi in zip(x, vals):
                ax.text(xi, yi + (0.25 if row == 0 else 0.012), f"{yi:.2f}" if row == 0 else f"{yi:.3f}", ha="center", fontsize=7.3)
            if i == 0:
                ax.set_ylabel(ylabel)
        axes[0, i].text(-0.13, 1.07, f"({chr(ord('a') + i)})", transform=axes[0, i].transAxes, fontsize=9, fontweight="bold")
    fig.text(0.5, 0.012, "Dashed lines are internal engineering thresholds, not theoretical limits.", ha="center", fontsize=7.3)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.13, wspace=0.20, hspace=0.12)
    save_figure(fig, "fig2_primary_metrics")


def fig3_qualitative_reconstruction() -> None:
    setup_matplotlib()
    rows = [(mid, 0) for mid in STL_METHODS]
    cols = ["GT", "Backprojection", "Reconstruction", "Absolute error"]
    fig = plt.figure(figsize=(7.35, 6.7))
    gs = fig.add_gridspec(len(rows), 5, width_ratios=[1, 1, 1, 1, 0.055], wspace=0.06, hspace=0.09)
    axes = np.array([[fig.add_subplot(gs[ridx, cidx]) for cidx in range(4)] for ridx in range(len(rows))])
    cax = fig.add_subplot(gs[:, 4])
    last_im = None
    for ridx, (mid, sample_idx) in enumerate(rows):
        cells = crop_sample(mid, sample_idx)
        gt = np.asarray(cells["GT"].convert("L"), dtype=np.float32) / 255.0
        recon = np.asarray(cells["Reconstruction"].convert("L"), dtype=np.float32) / 255.0
        err = np.abs(gt - recon)
        vmax = max(0.12, float(np.quantile(err, 0.99)))
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
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.ax.tick_params(labelsize=6.5)
    fig.text(0.5, 0.014, "Images are enlarged for visibility. Error maps are contrast-enhanced with a 99th-percentile scale.", ha="center", fontsize=7.4)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.92, bottom=0.08)
    save_figure(fig, "fig3_qualitative_reconstruction")


def _hadamard(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1]])
    h = _hadamard(n // 2)
    return np.block([[h, h], [h, -h]])


def fig4_measurement_attribution() -> None:
    setup_matplotlib()
    attr = {r["method_id"]: r for r in table("attribution")}
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.85))
    ax = axes[0, 0]
    rng = np.random.default_rng(3)
    patterns = [
        ("Rademacher", rng.choice([-1, 1], size=(16, 16))),
        ("Scrambled\nHadamard", _hadamard(16)[rng.permutation(16), :]),
        ("Low-frequency\nHadamard", _hadamard(16)),
    ]
    for i, (label, arr) in enumerate(patterns):
        ax.imshow(arr, cmap="gray", extent=(i, i + 0.8, 0.18, 0.98), interpolation="nearest")
        ax.text(i + 0.4, 0.04, label, ha="center", va="top", fontsize=7.2)
    ax.set_xlim(-0.05, 2.85)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Pattern examples")
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[0, 1]
    x = np.arange(len(STL_METHODS))
    bp = [as_float(attr[mid]["backproj_psnr"]) for mid in STL_METHODS]
    model = [as_float(attr[mid]["model_psnr"]) for mid in STL_METHODS]
    ax.bar(x - 0.18, bp, width=0.36, color=LIGHT_BLUE, edgecolor=BLUE, label="BP")
    ax.bar(x + 0.18, model, width=0.36, color=BLUE, label="Model")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in STL_METHODS])
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Backprojection vs final model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1, 0]
    delta = [as_float(attr[mid]["delta_psnr"]) for mid in STL_METHODS]
    ax.bar(x, delta, color=[ORANGE, GREEN, ORANGE, GREEN], width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in STL_METHODS])
    ax.set_ylabel(r"$\Delta$PSNR (dB)")
    ax.set_title("Neural gain")
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1, 1]
    for mid in STL_METHODS:
        label = METHOD_LABEL[mid]
        color = BLUE if label.startswith("Rad") else GREEN
        xx = as_float(attr[mid]["backproj_psnr"])
        yy = as_float(attr[mid]["delta_psnr"])
        ax.scatter([xx], [yy], s=50, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(xx + 0.10, yy + 0.10, label, fontsize=7.2)
    ax.axvspan(0, 10.5, color=LIGHT_ORANGE, alpha=0.30)
    ax.axvspan(10.5, 16.0, color=LIGHT_GREEN, alpha=0.28)
    ax.text(7.7, 13.0, "weak initialization\nlarge gain", ha="center", fontsize=7.2)
    ax.text(14.1, 8.3, "stronger initialization\nsimilar final", ha="center", fontsize=7.2)
    ax.set_xlabel("BP PSNR (dB)")
    ax.set_ylabel(r"$\Delta$PSNR (dB)")
    ax.set_title("Regime map")
    ax.grid(alpha=0.25)
    ax.text(-0.12, 1.05, "(d)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.text(0.5, 0.012, "Final PSNR hides measurement-family regimes: Rademacher starts weak and gains more, while scrambled Hadamard starts stronger and reaches similar final quality.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.11, wspace=0.27, hspace=0.43)
    save_figure(fig, "fig4_measurement_attribution")


def fig5_inference_ablation() -> None:
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
    fig.text(0.5, 0.012, "Removing measurement-consistency projection gives the strongest degradation; no-null removal has limited metric effect for these trained checkpoints and is not used as sole evidence for null-space necessity.", ha="center", fontsize=6.6)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.12, hspace=0.30)
    save_figure(fig, "fig5_inference_ablation")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(STL_METHODS))
    rel = [as_float(ab[(mid, "no_dc_project")]["delta_vs_full_relmeaserr"]) for mid in STL_METHODS]
    ax.bar(x, rel, color=RED, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in STL_METHODS])
    ax.set_ylabel("RelMeasErr increase")
    ax.set_title("No-DC projection increases measurement inconsistency")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "figS_relmeaserr_ablation", exts=("pdf", "png"))


def fig6_validation_summary() -> None:
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
    ax.set_title("Noise sweep")
    ax.set_xlabel("Noise std")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=6.6)
    ax.text(-0.13, 1.06, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[0, 1]
    modes = [("shuffle_coefficients", "Shuffle"), ("wrong_sample", "Wrong-y")]
    mids = STL_METHODS[:2]
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
    ax.legend(frameon=False)
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
    ax.set_title("Ours vs CS-TV")
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

    fig.text(0.5, 0.012, "These tests support finite-noise stability and measurement dependence; they do not imply universal robustness.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.13, wspace=0.30, hspace=0.45)
    save_figure(fig, "fig6_validation_summary")


def main() -> None:
    fig1_mechanism()
    fig2_primary_metrics()
    fig3_qualitative_reconstruction()
    fig4_measurement_attribution()
    fig5_inference_ablation()
    fig6_validation_summary()
    print({"figures": str(OUT / "figures")})


if __name__ == "__main__":
    main()
