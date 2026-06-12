from __future__ import annotations

import math

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
    OUT,
    PURPLE,
    RED,
    STL_METHODS,
    as_float,
    crop_sample,
    registry_by_id,
    save_figure,
    setup_matplotlib,
    table,
)


def arrow(ax, p0, p1, color=GRAY, lw=1.2, rad=0.0) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def box(ax, xy, w, h, text, fc, ec, fs=8.0) -> None:
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.018,rounding_size=0.03", fc=fc, ec=ec, lw=1.1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def fig1_mechanism() -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 5, figsize=(7.3, 2.85))
    titles = ["acquisition", "feasible set", "data solution", "neural residual", "projection"]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(titles[i], fontsize=8.4)
        ax.text(-0.02, 1.04, f"({chr(ord('a') + i)})", transform=ax.transAxes, fontsize=9, fontweight="bold")

    rng = np.random.default_rng(4)
    ax = axes[0]
    for j, x0 in enumerate([0.05, 0.13, 0.21]):
        ax.imshow(rng.choice([0, 1], size=(8, 8)), cmap="gray", extent=(x0, x0 + 0.20, 0.60 - j * 0.045, 0.80 - j * 0.045), interpolation="nearest")
        ax.add_patch(Rectangle((x0, 0.60 - j * 0.045), 0.20, 0.20, fill=False, ec=BLUE, lw=0.8))
    ax.add_patch(Ellipse((0.57, 0.64), 0.21, 0.26, fc=LIGHT_GRAY, ec=GRAY, lw=1))
    ax.text(0.57, 0.64, "$x$", ha="center", va="center", fontsize=10)
    ax.add_patch(Circle((0.85, 0.64), 0.062, fc=LIGHT_BLUE, ec=BLUE, lw=1))
    ax.text(0.85, 0.64, "$y$", ha="center", va="center", fontsize=8)
    arrow(ax, (0.36, 0.66), (0.46, 0.65), BLUE)
    arrow(ax, (0.68, 0.64), (0.78, 0.64), BLUE)
    ax.text(0.52, 0.22, r"$y=Ax+\epsilon$", ha="center", fontsize=8.5, color=BLUE)

    ax = axes[1]
    ax.add_patch(Ellipse((0.50, 0.56), 0.72, 0.34, angle=-15, fc=LIGHT_BLUE, ec=BLUE, lw=1.1))
    ax.plot([0.18, 0.82], [0.34, 0.77], color=GREEN, lw=1.5)
    arrow(ax, (0.28, 0.41), (0.75, 0.72), GREEN)
    ax.scatter([0.39, 0.62], [0.52, 0.60], s=28, color=[BLUE, ORANGE], zorder=3)
    ax.text(0.50, 0.20, r"$\mathcal{C}_y=\{x:Ax=y\}$", ha="center", fontsize=7.4)
    ax.text(0.50, 0.09, r"$\mathrm{Null}(A)$ directions", ha="center", fontsize=7.0, color=GREEN)

    ax = axes[2]
    box(ax, (0.13, 0.55), 0.74, 0.22, r"$x_{\rm data}$", LIGHT_BLUE, BLUE, 9)
    ax.text(0.50, 0.34, "measured\nrow-space\npreserved", ha="center", fontsize=8, color=BLUE)
    ax.text(0.50, 0.11, r"$A^T(AA^T+\lambda I)^{-1}y$", ha="center", fontsize=6.7)

    ax = axes[3]
    box(ax, (0.08, 0.63), 0.36, 0.18, r"$G_\theta$", LIGHT_ORANGE, ORANGE, 9)
    box(ax, (0.56, 0.63), 0.36, 0.18, r"$P_N$", LIGHT_GREEN, GREEN, 9)
    arrow(ax, (0.44, 0.72), (0.56, 0.72), ORANGE)
    box(ax, (0.20, 0.27), 0.60, 0.18, "null-space\ncompleted", "#F7F7F7", GRAY, 8)
    arrow(ax, (0.74, 0.63), (0.57, 0.45), GREEN)

    ax = axes[4]
    box(ax, (0.10, 0.60), 0.36, 0.18, r"$\Pi_y$", LIGHT_GREEN, GREEN, 9)
    box(ax, (0.56, 0.60), 0.34, 0.18, r"$\hat{x}$", LIGHT_BLUE, BLUE, 9)
    arrow(ax, (0.46, 0.69), (0.56, 0.69), GREEN)
    ax.add_patch(FancyBboxPatch((0.28, 0.25), 0.44, 0.18, boxstyle="round,pad=0.02,rounding_size=0.03", fc=LIGHT_RED, ec=RED, lw=1.0))
    ax.text(0.50, 0.34, "measurement\nchecked", ha="center", va="center", fontsize=8, color=RED)
    ax.text(0.50, 0.12, r"$A\hat{x}\approx y$", ha="center", fontsize=8.4, color=GREEN)

    fig.text(0.5, 0.025, "Mechanism: measured row-space is preserved, null-space content is completed, and the output is checked against the bucket measurements.", ha="center", fontsize=7.3)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.82, bottom=0.21, wspace=0.07)
    save_figure(fig, "fig1_mechanism")


def fig2_main_results() -> None:
    setup_matplotlib()
    reg = registry_by_id()
    groups = [
        (["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab"], "STL-10 5%", 20.0, 0.60),
        (["rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"], "STL-10 10%", 22.0, 0.65),
        (["mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"], "MNIST/Fashion 5%", 25.0, 0.80),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.85), sharey="row")
    for i, (mids, title, psnr_thr, ssim_thr) in enumerate(groups):
        x = np.arange(len(mids))
        colors = [BLUE, GREEN] if i < 2 else [PURPLE, ORANGE]
        psnr = [as_float(reg[mid]["psnr"]) for mid in mids]
        ssim = [as_float(reg[mid]["ssim"]) for mid in mids]
        axes[0, i].bar(x, psnr, color=colors, width=0.62)
        axes[0, i].axhline(psnr_thr, color=RED, ls="--", lw=1)
        axes[0, i].set_title(title)
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels([])
        axes[0, i].grid(axis="y", alpha=0.25)
        for xi, yi in zip(x, psnr):
            axes[0, i].text(xi, yi + 0.25, f"{yi:.2f}", ha="center", fontsize=7.4)
        axes[1, i].bar(x, ssim, color=colors, width=0.62)
        axes[1, i].axhline(ssim_thr, color=RED, ls="--", lw=1)
        axes[1, i].set_ylim(0, 1.04)
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels([METHOD_LABEL[mid] for mid in mids])
        axes[1, i].grid(axis="y", alpha=0.25)
        for xi, yi in zip(x, ssim):
            axes[1, i].text(xi, yi + 0.012, f"{yi:.3f}", ha="center", fontsize=7.4)
        axes[0, i].text(-0.13, 1.07, f"({chr(ord('a') + i)})", transform=axes[0, i].transAxes, fontsize=9, fontweight="bold")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[1, 0].set_ylabel("SSIM")
    fig.text(0.5, 0.012, "Dashed lines are internal engineering thresholds, not theoretical limits.", ha="center", fontsize=7.3)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.13, wspace=0.20, hspace=0.12)
    save_figure(fig, "fig2_main_results")


def fig3_qualitative_reconstruction() -> None:
    setup_matplotlib()
    rows = []
    for mid in STL_METHODS:
        rows.extend([(mid, 0), (mid, 1)])
    cols = ["GT", "Backprojection", "Reconstruction", "Absolute error"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.25, 11.35))
    for ridx, (mid, sample_idx) in enumerate(rows):
        cells = crop_sample(mid, sample_idx)
        gt = np.asarray(cells["GT"].convert("L"), dtype=np.float32) / 255.0
        recon = np.asarray(cells["Reconstruction"].convert("L"), dtype=np.float32) / 255.0
        err = np.abs(gt - recon)
        for cidx, col in enumerate(cols):
            ax = axes[ridx, cidx]
            if col == "Absolute error":
                ax.imshow(err, cmap="inferno", vmin=0, vmax=max(0.18, float(np.quantile(err, 0.98))))
            else:
                ax.imshow(cells[col], cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ridx == 0:
                ax.set_title(col, fontsize=9)
            if cidx == 0:
                label = METHOD_LABEL[mid] if sample_idx == 0 else f"{METHOD_LABEL[mid]} (2)"
                ax.set_ylabel(label, rotation=0, labelpad=34, va="center", fontsize=8.5)
    fig.text(0.5, 0.012, "Images are enlarged for visibility. Qualitative samples are visual evidence; metrics are reported in the tables.", ha="center", fontsize=7.4)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.96, bottom=0.045, wspace=0.05, hspace=0.055)
    save_figure(fig, "fig3_qualitative_reconstruction")


def _hadamard(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1]])
    h = _hadamard(n // 2)
    return np.block([[h, h], [h, -h]])


def fig4_measurement_attribution() -> None:
    setup_matplotlib()
    reg = registry_by_id()
    attr = {r["method_id"]: r for r in table("attribution")}
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.85))

    ax = axes[0, 0]
    rng = np.random.default_rng(0)
    patterns = [
        ("Rademacher", rng.choice([-1, 1], size=(16, 16))),
        ("Scrambled\nHadamard", _hadamard(16)[rng.permutation(16)[:16], :]),
        ("Lowfreq\nHadamard", _hadamard(16)),
    ]
    for i, (label, arr) in enumerate(patterns):
        ax.imshow(arr, cmap="gray", extent=(i, i + 0.8, 0.15, 0.95), interpolation="nearest")
        ax.text(i + 0.4, 0.03, label, ha="center", va="top", fontsize=7.4)
    ax.set_xlim(-0.05, 2.85)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Pattern examples")
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[0, 1]
    mids = STL_METHODS
    x = np.arange(len(mids))
    bp = [as_float(attr[mid]["backproj_psnr"]) for mid in mids]
    model = [as_float(attr[mid]["model_psnr"]) for mid in mids]
    ax.bar(x - 0.18, bp, width=0.36, color=LIGHT_BLUE, edgecolor=BLUE, label="BP")
    ax.bar(x + 0.18, model, width=0.36, color=BLUE, label="Model")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids])
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("BP vs model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1, 0]
    delta = [as_float(attr[mid]["delta_psnr"]) for mid in mids]
    ax.bar(x, delta, color=[ORANGE, GREEN, ORANGE, GREEN], width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids])
    ax.set_ylabel(r"$\Delta$PSNR (dB)")
    ax.set_title("Neural gain")
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1, 1]
    label_map = {
        "rademacher5_hq_noise001_colab": "Rad-5",
        "scrambled_hadamard5_hq_noise001_colab": "Scr-5",
        "rademacher10_full_noise001_colab": "Rad-10",
        "scrambled_hadamard10_full_noise001_colab": "Scr-10",
        "stl10_hadamard5_local_medium": "Lowfreq-5",
        "stl10_hadamard10_local_full": "Lowfreq-10",
        "mnist_hadamard5_full_colab": "MNIST",
        "fashion_hadamard5_full_colab": "Fashion",
    }
    for row in table("attribution"):
        mid = row["method_id"]
        if mid not in label_map:
            continue
        label = label_map[mid]
        color = BLUE if label.startswith("Rad") else GREEN if label.startswith("Scr") else ORANGE if label.startswith("Low") else PURPLE
        xx = as_float(row["backproj_psnr"])
        yy = as_float(row["delta_psnr"])
        if math.isfinite(xx) and math.isfinite(yy):
            ax.scatter([xx], [yy], s=45, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(xx + 0.12, yy + 0.10, label, fontsize=6.8)
    ax.set_xlabel("BP PSNR (dB)")
    ax.set_ylabel(r"$\Delta$PSNR (dB)")
    ax.set_title("Regime map")
    ax.grid(alpha=0.25)
    ax.text(-0.12, 1.05, "(d)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.text(0.5, 0.012, "Rademacher has weak BP but large neural gain; scrambled Hadamard has stronger BP and similar final quality. Low-frequency Hadamard is diagnostic.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.11, wspace=0.27, hspace=0.43)
    save_figure(fig, "fig4_measurement_attribution")


def fig5_inference_ablation() -> None:
    setup_matplotlib()
    ab = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    modes = [("full_model", "Full"), ("no_dc_project", "-DC"), ("no_null_project", "-Null"), ("stage1_only", "Stage1"), ("raw_weights", "Raw"), ("ema_weights", "EMA")]
    colors = [BLUE, RED, "#ADB5BD", ORANGE, PURPLE, GREEN]
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.2), sharey=True)
    for idx, (ax, mid) in enumerate(zip(axes.ravel(), STL_METHODS)):
        vals = [as_float(ab[(mid, mode)]["psnr"]) for mode, _ in modes]
        x = np.arange(len(modes))
        ax.bar(x, vals, color=colors, width=0.68)
        for xi, yi in zip(x, vals):
            ax.text(xi, yi + 0.22, f"{yi:.1f}", ha="center", fontsize=6.7)
        ax.set_title(METHOD_LABEL[mid])
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in modes])
        if idx % 2 == 0:
            ax.set_ylabel("PSNR (dB)")
        ax.grid(axis="y", alpha=0.25)
        ax.text(-0.12, 1.06, f"({chr(ord('a') + idx)})", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.text(0.5, 0.012, "Removing the measurement-consistency/DC projection causes the strongest degradation.", ha="center", fontsize=7.2)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.12, hspace=0.30)
    save_figure(fig, "fig5_inference_ablation")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(STL_METHODS))
    rel = [as_float(ab[(mid, "no_dc_project")]["delta_vs_full_relmeaserr"]) for mid in STL_METHODS]
    ax.bar(x, rel, color=RED, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in STL_METHODS])
    ax.set_ylabel("RelMeasErr increase")
    ax.set_title("No-DC projection measurement-error increase")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "figS_ablation_relmeaserr", exts=("pdf", "png"))


def fig6_robustness_baselines() -> None:
    setup_matplotlib()
    noise = table("noise")
    perturb = table("perturbation")
    baseline = table("baseline")
    stats = {r["method_id"]: r for r in table("statistics")}
    reg = registry_by_id()
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.75))

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
    for mid in METHOD_ORDER:
        sub = [r for r in baseline if r["method_id"] == mid and r["baseline"] == "tv_pgd"]
        if sub:
            tv_best[mid] = max(sub, key=lambda r: as_float(r["psnr"]))
    mids = [mid for mid in METHOD_ORDER if mid in tv_best]
    x = np.arange(len(mids))
    ax.bar(x - 0.18, [as_float(reg[mid]["psnr"]) for mid in mids], width=0.36, color=BLUE, label="Ours")
    ax.bar(x + 0.18, [as_float(tv_best[mid]["psnr"]) for mid in mids], width=0.36, color="#9B8A4E", label="CS-TV")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids], fontsize=7)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Ours vs CS-TV")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    ax = axes[1, 1]
    mids = METHOD_ORDER
    x = np.arange(len(mids))
    mean = [as_float(stats[mid]["mean_psnr"]) for mid in mids]
    low = [as_float(stats[mid]["ci95_psnr_low"]) for mid in mids]
    high = [as_float(stats[mid]["ci95_psnr_high"]) for mid in mids]
    yerr = np.array([[m - l for m, l in zip(mean, low)], [h - m for h, m in zip(high, mean)]])
    ax.errorbar(x, mean, yerr=yerr, fmt="o", color=BLUE, ecolor=GRAY, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids], fontsize=6.5, rotation=20)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Bootstrap CI")
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "(d)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.text(0.5, 0.012, "CS-TV is a TV-regularized compressed-sensing baseline solved by projected gradient descent.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.13, wspace=0.30, hspace=0.45)
    save_figure(fig, "fig6_robustness_baselines")


def main() -> None:
    fig1_mechanism()
    fig2_main_results()
    fig3_qualitative_reconstruction()
    fig4_measurement_attribution()
    fig5_inference_ablation()
    fig6_robustness_baselines()
    print({"figures": str(OUT / "figures")})


if __name__ == "__main__":
    main()
