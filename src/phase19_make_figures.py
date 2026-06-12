from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from .phase19_common import (
    BLUE,
    GRAY,
    GREEN,
    LIGHT_BLUE,
    LIGHT_GRAY,
    LIGHT_GREEN,
    LIGHT_ORANGE,
    METHOD_LABEL,
    METHOD_ORDER,
    ORANGE,
    OUT,
    PURPLE,
    RED,
    STL_METHODS,
    as_float,
    crop_sample,
    fmt,
    registry_by_id,
    save_figure,
    setup_matplotlib,
    table,
)


def arrow(ax, p0, p1, color=GRAY, lw=1.2, rad=0.0) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def box(ax, xy, w, h, text, fc, ec, fs=8.0) -> None:
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.035", fc=fc, ec=ec, lw=1.1)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def fig0_graphical_abstract() -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, (0.04, 0.35), 0.22, 0.30, "bucket\nmeasurements\n$y=Ax+\\epsilon$", LIGHT_BLUE, BLUE, 9)
    box(ax, (0.39, 0.35), 0.26, 0.30, "data solution\n+ null-space\nneural residual", LIGHT_ORANGE, ORANGE, 8.5)
    box(ax, (0.74, 0.35), 0.22, 0.30, "STL-10\n5% and 10%\nhigh quality", LIGHT_GREEN, GREEN, 9)
    arrow(ax, (0.26, 0.50), (0.39, 0.50), BLUE, 1.5)
    arrow(ax, (0.65, 0.50), (0.74, 0.50), GREEN, 1.5)
    ax.text(0.52, 0.20, r"$\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))]$", ha="center", fontsize=10.5)
    ax.text(0.52, 0.08, "constrained completion: preserve row space, complete null space, check measurements", ha="center", fontsize=8, color="#444444")
    save_figure(fig, "fig0_graphical_abstract")


def fig1_mechanism() -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.7))
    titles = ["acquisition", "feasible set", "data solution", "null residual", "projection"]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(titles[i], fontsize=8.2)
        ax.text(-0.02, 1.04, chr(ord("a") + i), transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = axes[0]
    rng = np.random.default_rng(1)
    for j, x0 in enumerate([0.06, 0.13, 0.20]):
        ax.imshow(rng.choice([0, 1], size=(8, 8)), cmap="gray", extent=(x0, x0 + 0.20, 0.58 - j * 0.04, 0.78 - j * 0.04), interpolation="nearest")
        ax.add_patch(Rectangle((x0, 0.58 - j * 0.04), 0.20, 0.20, fill=False, ec=BLUE, lw=0.8))
    ax.add_patch(Ellipse((0.55, 0.65), 0.20, 0.25, fc=LIGHT_GRAY, ec=GRAY, lw=1))
    ax.text(0.55, 0.65, "$x$", ha="center", va="center", fontsize=10)
    ax.add_patch(Circle((0.84, 0.65), 0.065, fc=LIGHT_BLUE, ec=BLUE, lw=1))
    ax.text(0.84, 0.65, "$y$", ha="center", va="center", fontsize=8)
    arrow(ax, (0.34, 0.67), (0.44, 0.66), BLUE)
    arrow(ax, (0.66, 0.65), (0.77, 0.65), BLUE)
    ax.text(0.50, 0.24, r"$y=Ax+\epsilon$", ha="center", fontsize=8.2)

    ax = axes[1]
    ax.add_patch(Ellipse((0.50, 0.56), 0.70, 0.34, angle=-15, fc=LIGHT_BLUE, ec=BLUE, lw=1.1))
    ax.plot([0.18, 0.82], [0.35, 0.76], color=GREEN, lw=1.4)
    arrow(ax, (0.28, 0.42), (0.75, 0.71), GREEN)
    ax.scatter([0.39, 0.62], [0.52, 0.60], s=24, color=[BLUE, ORANGE])
    ax.text(0.50, 0.21, r"$\mathcal{C}_y=\{x:Ax=y\}$", ha="center", fontsize=7.2)
    ax.text(0.50, 0.10, "null-space directions", ha="center", fontsize=6.8, color=GREEN)

    ax = axes[2]
    box(ax, (0.14, 0.55), 0.72, 0.22, r"$x_{\rm data}$", LIGHT_BLUE, BLUE, 9)
    ax.text(0.50, 0.36, "row-space\npreserved", ha="center", fontsize=8, color=BLUE)
    ax.text(0.50, 0.15, r"$A^T(AA^T+\lambda I)^{-1}y$", ha="center", fontsize=6.7)

    ax = axes[3]
    box(ax, (0.10, 0.62), 0.35, 0.18, r"$G_\theta$", LIGHT_ORANGE, ORANGE, 9)
    box(ax, (0.55, 0.62), 0.35, 0.18, r"$P_N$", LIGHT_GREEN, GREEN, 9)
    arrow(ax, (0.45, 0.71), (0.55, 0.71), ORANGE)
    box(ax, (0.25, 0.27), 0.50, 0.18, "null-space\ncompleted", "#F5F5F5", GRAY, 8)
    arrow(ax, (0.72, 0.62), (0.56, 0.45), GREEN)

    ax = axes[4]
    box(ax, (0.12, 0.58), 0.35, 0.18, r"$\Pi_y$", LIGHT_GREEN, GREEN, 9)
    box(ax, (0.55, 0.58), 0.32, 0.18, r"$\hat{x}$", LIGHT_BLUE, BLUE, 9)
    arrow(ax, (0.47, 0.67), (0.55, 0.67), GREEN)
    ax.text(0.50, 0.34, "measurement\nchecked", ha="center", fontsize=8, color=GREEN)
    ax.text(0.50, 0.14, r"$A\hat{x}\approx y$", ha="center", fontsize=8.3)

    fig.text(0.5, 0.02, "Mechanism: physical row-space data are preserved, missing null-space content is completed, and the output is checked against the bucket measurements.", ha="center", fontsize=7.2)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.82, bottom=0.20, wspace=0.08)
    save_figure(fig, "fig1_mechanism")


def fig2_main_metrics() -> None:
    setup_matplotlib()
    reg = registry_by_id()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharey="row")
    groups = [
        (["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab"], "STL-10 5%", 20.0, 0.60),
        (["rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"], "STL-10 10%", 22.0, 0.65),
        (["mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"], "Simple domains 5%", 25.0, 0.80),
    ]
    for i, (mids, title, psnr_thr, ssim_thr) in enumerate(groups):
        x = np.arange(len(mids))
        psnr = [as_float(reg[mid]["psnr"]) for mid in mids]
        ssim = [as_float(reg[mid]["ssim"]) for mid in mids]
        colors = [BLUE, GREEN] if i < 2 else [PURPLE, ORANGE]
        axes[0, i].bar(x, psnr, color=colors, width=0.62)
        axes[0, i].axhline(psnr_thr, color=RED, ls="--", lw=1.0)
        for xi, yi in zip(x, psnr):
            axes[0, i].text(xi, yi + 0.25, f"{yi:.2f}", ha="center", fontsize=7)
        axes[0, i].set_title(title)
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels([])
        axes[0, i].grid(axis="y", alpha=0.25)
        axes[1, i].bar(x, ssim, color=colors, width=0.62)
        axes[1, i].axhline(ssim_thr, color=RED, ls="--", lw=1.0)
        for xi, yi in zip(x, ssim):
            axes[1, i].text(xi, yi + 0.012, f"{yi:.3f}", ha="center", fontsize=7)
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels([METHOD_LABEL[mid] for mid in mids])
        axes[1, i].set_ylim(0, 1.04)
        axes[1, i].grid(axis="y", alpha=0.25)
        axes[0, i].text(-0.15, 1.07, chr(ord("a") + i), transform=axes[0, i].transAxes, fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[1, 0].set_ylabel("SSIM")
    fig.text(0.5, 0.012, "Dashed lines are internal engineering thresholds, not theoretical limits.", ha="center", fontsize=7.4)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.13, wspace=0.22, hspace=0.12)
    save_figure(fig, "fig2_main_metrics")


def fig3_qualitative_grid_v2() -> None:
    setup_matplotlib()
    cols = ["GT", "BP", "Recon", "Error"]
    rows = []
    for mid in STL_METHODS:
        rows.append((mid, 0))
        rows.append((mid, 1))
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.2, 11.2))
    for ridx, (mid, sample_row) in enumerate(rows):
        cells = crop_sample(mid, sample_row)
        for cidx, col in enumerate(cols):
            ax = axes[ridx, cidx]
            ax.imshow(cells[col], cmap="magma" if col == "Error" else "gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ridx == 0:
                ax.set_title(col, fontsize=9)
            if cidx == 0:
                label = METHOD_LABEL[mid] if sample_row == 0 else f"{METHOD_LABEL[mid]} (2)"
                ax.set_ylabel(label, rotation=0, labelpad=34, va="center", fontsize=8.5)
    fig.text(0.5, 0.012, "Qualitative examples are enlarged from saved strict no-leak evaluation grids; they are visual evidence, not additional metrics.", ha="center", fontsize=7.3)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.96, bottom=0.045, wspace=0.05, hspace=0.06)
    save_figure(fig, "fig3_qualitative_grid_v2")


def fig4_measurement_regime_map() -> None:
    setup_matplotlib()
    rows = table("attribution")
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
    color_map = {"Rad": BLUE, "Scr": GREEN, "Lowfreq": ORANGE, "MNIST": PURPLE, "Fashion": "#9B6A44"}
    points = []
    for row in rows:
        mid = row["method_id"]
        if mid not in label_map:
            continue
        label = label_map[mid]
        family = "Rad" if label.startswith("Rad") else "Scr" if label.startswith("Scr") else "Lowfreq" if label.startswith("Lowfreq") else label
        points.append((label, as_float(row["backproj_psnr"]), as_float(row["delta_psnr"]), color_map.get(family, GRAY)))
    fig, ax = plt.subplots(figsize=(6.9, 4.7))
    for label, x, y, color in points:
        ax.scatter([x], [y], s=60, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(x + 0.12, y + 0.15, label, fontsize=7.6)
    ax.axvspan(0, 10.5, color=LIGHT_BLUE, alpha=0.45)
    ax.axvspan(10.5, 16.5, color=LIGHT_GREEN, alpha=0.35)
    ax.axvspan(16.5, 26, color=LIGHT_ORANGE, alpha=0.35)
    ax.text(5.2, 1.0, "weak initialization\nlarge gain", ha="center", fontsize=8)
    ax.text(13.5, 1.0, "stronger initialization\nmoderate gain", ha="center", fontsize=8)
    ax.text(20.5, 1.0, "strong initialization\nsmaller gain", ha="center", fontsize=8)
    ax.set_xlabel("Backprojection PSNR (dB)")
    ax.set_ylabel(r"Neural gain $\Delta$PSNR (dB)")
    ax.set_title("Measurement regime map")
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.13)
    save_figure(fig, "fig4_measurement_regime_map")


def fig5_inference_ablation() -> None:
    setup_matplotlib()
    ab = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    modes = [("full_model", "Full"), ("no_dc_project", "-DC"), ("no_null_project", "-Null"), ("stage1_only", "Stage1"), ("raw_weights", "Raw"), ("ema_weights", "EMA")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), sharey=True)
    colors = [BLUE, RED, "#B7BBC2", ORANGE, PURPLE, GREEN]
    for idx, (ax, mid) in enumerate(zip(axes.ravel(), STL_METHODS)):
        vals = [as_float(ab[(mid, mode)]["psnr"]) for mode, _ in modes]
        x = np.arange(len(modes))
        ax.bar(x, vals, color=colors, width=0.68)
        for xi, yi in zip(x, vals):
            ax.text(xi, yi + 0.25, f"{yi:.1f}", ha="center", fontsize=6.7)
        ax.set_title(METHOD_LABEL[mid])
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in modes])
        if idx % 2 == 0:
            ax.set_ylabel("PSNR (dB)")
        ax.grid(axis="y", alpha=0.25)
        ax.text(-0.12, 1.06, chr(ord("a") + idx), transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.text(0.5, 0.012, "No-DC projection removal is the strongest measurement-consistency ablation; no-null removal has limited metric effect for these checkpoints.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.12, hspace=0.30)
    save_figure(fig, "fig5_inference_ablation")


def fig6_validation_summary() -> None:
    setup_matplotlib()
    noise = table("noise")
    ab = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    perturb = table("perturbation")
    baseline = table("baseline")
    reg = registry_by_id()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = axes[0, 0]
    for mid, color in zip(STL_METHODS, [BLUE, GREEN, PURPLE, ORANGE]):
        sub = sorted([r for r in noise if r["method_id"] == mid], key=lambda r: as_float(r["noise_std"]))
        ax.plot([as_float(r["noise_std"]) for r in sub], [as_float(r["psnr"]) for r in sub], marker="o", lw=1.4, color=color, label=METHOD_LABEL[mid])
    ax.set_title("Finite noise sweep")
    ax.set_xlabel("Noise std")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=6.7)
    ax.text(-0.13, 1.06, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = axes[0, 1]
    x = np.arange(len(STL_METHODS))
    psnr_drop = [abs(as_float(ab[(mid, "no_dc_project")]["delta_vs_full_psnr"])) for mid in STL_METHODS]
    rel_inc = [as_float(ab[(mid, "no_dc_project")]["delta_vs_full_relmeaserr"]) for mid in STL_METHODS]
    ax.bar(x - 0.18, psnr_drop, width=0.36, color=RED, label="PSNR drop")
    ax2 = ax.twinx()
    ax2.plot(x + 0.18, rel_inc, color=ORANGE, marker="o", lw=1.2, label="RelMeasErr inc.")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in STL_METHODS], fontsize=7)
    ax.set_ylabel("PSNR drop (dB)")
    ax2.set_ylabel("RelMeasErr increase")
    ax.set_title("No-DC projection stress test")
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = axes[1, 0]
    mids = STL_METHODS[:2]
    modes = [("shuffle_coefficients", "Shuffle"), ("wrong_sample", "Wrong-y")]
    x = np.arange(len(mids))
    for i, (mode, lab) in enumerate(modes):
        vals = []
        for mid in mids:
            row = next(r for r in perturb if r["method_id"] == mid and r["perturbation_mode"] == mode)
            vals.append(as_float(row["psnr_drop_from_normal"]))
        ax.bar(x + (i - 0.5) * 0.34, vals, width=0.34, label=lab, color=ORANGE if i == 0 else RED)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids])
    ax.set_ylabel("PSNR drop (dB)")
    ax.set_title("Measurement perturbation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "c", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = axes[1, 1]
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
    ax.set_title("Ours vs CS-TV (PGD)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "d", transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.text(0.5, 0.012, "Validation emphasizes physical dependence: finite noise, projection stress, measurement perturbation, and a CS-TV compressed-sensing baseline.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.91, top=0.91, bottom=0.12, wspace=0.42, hspace=0.42)
    save_figure(fig, "fig6_validation_summary")


def main() -> None:
    fig0_graphical_abstract()
    fig1_mechanism()
    fig2_main_metrics()
    fig3_qualitative_grid_v2()
    fig4_measurement_regime_map()
    fig5_inference_ablation()
    fig6_validation_summary()
    print({"figures": str(OUT / "figures")})


if __name__ == "__main__":
    main()
