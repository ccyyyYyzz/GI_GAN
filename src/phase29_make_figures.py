from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from .phase20_common import (
    BLUE,
    GRAY,
    GREEN,
    LIGHT_BLUE,
    LIGHT_GRAY,
    LIGHT_GREEN,
    LIGHT_ORANGE,
    METHOD_LABEL,
    ORANGE,
    PURPLE,
    RED,
    STL_METHODS,
    as_float,
    ensure_dir,
    registry_by_id,
    setup_matplotlib,
    table,
)


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase29_final_submission_polish"
FIG_DIR = OUT / "figures"
PROJECT_FIG_DIR = OUT / "latex_project_final" / "figures"
PHASE28_FIG_DIR = ROOT / "outputs_phase28_paragraph_polish" / "figures"
PHASE23_FIG_DIR = ROOT / "outputs_phase23_top_journal_rewrite" / "figures"


def save_figure(fig: plt.Figure, stem: str, exts: tuple[str, ...] = ("pdf", "png", "svg"), dpi: int = 300) -> None:
    ensure_dir(FIG_DIR)
    ensure_dir(PROJECT_FIG_DIR)
    for ext in exts:
        path = FIG_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        shutil.copy2(path, PROJECT_FIG_DIR / path.name)
    plt.close(fig)


def arrow(ax, start, end, color=GRAY, lw=1.4, rad=0.0, scale=10, alpha=1.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            lw=lw,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


def panel_box(ax, xy, w, h, label: str, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            fc="white",
            ec="#D6D9DE",
            lw=0.9,
        )
    )
    ax.text(xy[0] + 0.022, xy[1] + h - 0.040, label, fontsize=9, fontweight="bold")
    ax.text(xy[0] + 0.060, xy[1] + h - 0.040, title, fontsize=8.7, fontweight="bold")


def fig1_mechanism_final() -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.35, 5.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panels = {
        "a": (0.035, 0.575, 0.290, 0.365, "(a)", "Optical acquisition"),
        "b": (0.355, 0.575, 0.290, 0.365, "(b)", "Measurement geometry"),
        "c": (0.675, 0.575, 0.290, 0.365, "(c)", "Physical data solution"),
        "d": (0.125, 0.095, 0.350, 0.355, "(d)", "Neural null-space completion"),
        "e": (0.535, 0.095, 0.350, 0.355, "(e)", "Measurement audit"),
    }
    for x, y, w, h, lab, title in panels.values():
        panel_box(ax, (x, y), w, h, lab, title)

    # (a) Optical acquisition
    rng = np.random.default_rng(29)
    for idx, x0 in enumerate([0.070, 0.115, 0.160]):
        pattern = rng.choice([-1, 1], size=(7, 7))
        ax.imshow(pattern, cmap="gray", extent=(x0, x0 + 0.078, 0.765 - 0.020 * idx, 0.843 - 0.020 * idx), interpolation="nearest", zorder=3)
        ax.add_patch(Rectangle((x0, 0.765 - 0.020 * idx), 0.078, 0.078, fill=False, ec=BLUE, lw=0.65, zorder=4))
    ax.add_patch(Ellipse((0.207, 0.745), 0.085, 0.135, fc="#F8FAFC", ec=GRAY, lw=1.0))
    ax.text(0.207, 0.745, r"$x$", ha="center", va="center", fontsize=10)
    ax.add_patch(Circle((0.278, 0.745), 0.035, fc=LIGHT_BLUE, ec=BLUE, lw=1.0))
    ax.text(0.278, 0.745, r"$y$", ha="center", va="center", fontsize=9, color=BLUE)
    arrow(ax, (0.200, 0.745), (0.245, 0.745), BLUE, 1.2)
    ax.text(0.176, 0.655, r"$y_i=\langle a_i,x\rangle+\epsilon_i$", ha="center", fontsize=7.7, color=BLUE)
    ax.text(0.112, 0.870, "structured\npatterns", ha="center", va="top", fontsize=7.0, color=BLUE)
    ax.text(0.278, 0.685, "bucket", ha="center", fontsize=7.0, color=BLUE)

    # (b) Measurement geometry
    ax.add_patch(Ellipse((0.500, 0.745), 0.215, 0.120, angle=-15, fc="#FBFCFD", ec=BLUE, lw=1.0))
    ax.plot([0.405, 0.590], [0.705, 0.785], color=GREEN, lw=1.9)
    ax.scatter([0.475], [0.735], s=45, color=BLUE, edgecolor="white", zorder=4)
    arrow(ax, (0.477, 0.736), (0.555, 0.770), GREEN, 1.35)
    ax.text(0.495, 0.815, r"$\mathcal{C}_y=\{x:Ax=y\}$", ha="center", fontsize=7.6, color=GREEN)
    ax.text(0.573, 0.790, r"$v\in\mathrm{Null}(A)$", fontsize=7.0, color=GREEN)
    ax.text(0.500, 0.647, "many images share same y", ha="center", fontsize=7.1, color=GRAY)

    # (c) Physical data solution
    ax.add_patch(Ellipse((0.820, 0.755), 0.205, 0.120, angle=-15, fc="#FBFCFD", ec=BLUE, lw=1.0, alpha=0.80))
    ax.plot([0.728, 0.910], [0.715, 0.792], color=GREEN, lw=1.7)
    ax.scatter([0.792], [0.742], s=60, color=BLUE, edgecolor="white", zorder=4)
    ax.text(0.792, 0.704, r"$x_{\rm data}$", ha="center", fontsize=7.5, color=BLUE)
    ax.text(0.820, 0.840, r"$x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y$", ha="center", fontsize=6.9, color=BLUE)
    ax.text(0.820, 0.645, "measured row-space\nrepresentative", ha="center", fontsize=7.1, color=BLUE)

    # (d) Neural null-space completion
    x0, y0, w, h = panels["d"][:4]
    ax.scatter([0.180], [0.265], s=58, color=BLUE, edgecolor="white", zorder=4)
    ax.text(0.180, 0.220, r"$x_{\rm data}$", ha="center", fontsize=7.5, color=BLUE)
    ax.add_patch(FancyBboxPatch((0.240, 0.238), 0.075, 0.058, boxstyle="round,pad=0.006,rounding_size=0.012", fc=LIGHT_ORANGE, ec=ORANGE, lw=1.0))
    ax.text(0.278, 0.267, r"$G_\theta$", ha="center", va="center", fontsize=8.2, color=ORANGE)
    ax.add_patch(FancyBboxPatch((0.348, 0.238), 0.070, 0.058, boxstyle="round,pad=0.006,rounding_size=0.012", fc=LIGHT_GREEN, ec=GREEN, lw=1.0))
    ax.text(0.383, 0.267, r"$P_N$", ha="center", va="center", fontsize=8.2, color=GREEN)
    arrow(ax, (0.205, 0.265), (0.240, 0.265), ORANGE, 1.2)
    arrow(ax, (0.315, 0.265), (0.348, 0.265), GREEN, 1.2)
    path = MplPath([(0.430, 0.265), (0.452, 0.307), (0.462, 0.225), (0.448, 0.250), (0.430, 0.265)])
    ax.add_patch(PathPatch(path, fc=LIGHT_ORANGE, ec=ORANGE, lw=1.0, alpha=0.9))
    ax.text(0.301, 0.158, "learned prior supplies\nmissing structure", ha="center", fontsize=7.2, color=ORANGE)
    ax.text(0.383, 0.333, "weakly observed /\nunobserved", ha="center", fontsize=6.9, color=GREEN)

    # (e) Measurement audit
    ax.add_patch(Ellipse((0.705, 0.276), 0.205, 0.120, angle=-15, fc="#FBFCFD", ec=BLUE, lw=1.0, alpha=0.85))
    ax.plot([0.612, 0.798], [0.236, 0.316], color=GREEN, lw=1.8)
    ax.scatter([0.620], [0.345], s=48, color=RED, edgecolor="white", zorder=4)
    ax.scatter([0.703], [0.275], s=60, color=GREEN, edgecolor="white", zorder=4)
    arrow(ax, (0.626, 0.337), (0.690, 0.285), GREEN, 1.5, rad=-0.15)
    ax.text(0.615, 0.370, "candidate", ha="center", fontsize=7.0, color=RED)
    ax.text(0.703, 0.228, r"$\hat{x}$", ha="center", fontsize=8.0, color=GREEN)
    ax.add_patch(FancyBboxPatch((0.790, 0.312), 0.060, 0.048, boxstyle="round,pad=0.006,rounding_size=0.012", fc=LIGHT_GREEN, ec=GREEN, lw=1.0))
    ax.text(0.820, 0.336, r"$\Pi_y$", ha="center", va="center", fontsize=8.2, color=GREEN)
    ax.text(0.820, 0.248, r"$A\hat{x}\approx y$", ha="center", fontsize=7.6, color=GREEN)
    ax.text(0.820, 0.203, "RelMeasErr", ha="center", fontsize=7.1, color=GRAY)
    ax.text(0.705, 0.158, "final image remains\nauditable", ha="center", fontsize=7.2, color=GREEN)

    # Cross-panel path
    arrow(ax, (0.326, 0.755), (0.353, 0.755), BLUE, 0.9, scale=8, alpha=0.55)
    arrow(ax, (0.646, 0.755), (0.673, 0.755), BLUE, 0.9, scale=8, alpha=0.55)
    arrow(ax, (0.815, 0.575), (0.422, 0.450), ORANGE, 0.9, rad=0.18, scale=8, alpha=0.40)
    arrow(ax, (0.475, 0.270), (0.535, 0.270), GREEN, 0.9, scale=8, alpha=0.55)
    save_figure(fig, "fig1_mechanism_final")


def fig2_primary_metrics_final() -> None:
    setup_matplotlib()
    reg = registry_by_id()
    groups = [
        (["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab"], "STL-10 5%", ["Rad-5", "Scr-5"], 20.0, 0.60),
        (["rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"], "STL-10 10%", ["Rad-10", "Scr-10"], 22.0, 0.65),
        (["mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"], "MNIST/Fashion 5%", ["MNIST-5", "Fashion-5"], 25.0, 0.80),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 4.85), sharey="row")
    for i, (mids, title, labels, psnr_thr, ssim_thr) in enumerate(groups):
        x = np.arange(len(mids))
        colors = [BLUE, GREEN] if i < 2 else [PURPLE, ORANGE]
        vals_by_row = [[as_float(reg[mid]["psnr"]) for mid in mids], [as_float(reg[mid]["ssim"]) for mid in mids]]
        for row, (vals, thr, ylabel) in enumerate(zip(vals_by_row, [psnr_thr, ssim_thr], ["PSNR (dB)", "SSIM"])):
            ax = axes[row, i]
            ax.bar(x, vals, color=colors, width=0.60)
            ax.axhline(thr, color=RED, ls="--", lw=0.95)
            top = max(vals + [thr]) + (1.25 if row == 0 else 0.08)
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
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.13, wspace=0.22, hspace=0.12)
    save_figure(fig, "fig2_primary_metrics_final")


def copy_figure_set(src_dir: Path, src_stem: str, dst_stem: str) -> None:
    ensure_dir(FIG_DIR)
    ensure_dir(PROJECT_FIG_DIR)
    for ext in ("pdf", "png", "svg"):
        src = src_dir / f"{src_stem}.{ext}"
        if not src.exists():
            continue
        dst = FIG_DIR / f"{dst_stem}.{ext}"
        shutil.copy2(src, dst)
        shutil.copy2(dst, PROJECT_FIG_DIR / dst.name)


def fig5_inference_ablation_final() -> None:
    setup_matplotlib()
    ab = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    modes = [
        ("full_model", "Full"),
        ("no_dc_project", "-MC"),
        ("no_null_project", "-Null"),
        ("stage1_only", "Stage1"),
        ("raw_weights", "Raw"),
        ("ema_weights", "EMA"),
    ]
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
    fig.text(
        0.5,
        0.012,
        "-MC removes the final measurement-consistency projection.",
        ha="center",
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.12, hspace=0.30)
    save_figure(fig, "fig5_inference_ablation_final")


def figS1_relmeaserr_final() -> None:
    setup_matplotlib()
    ab = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(STL_METHODS))
    rel = [as_float(ab[(mid, "no_dc_project")]["delta_vs_full_relmeaserr"]) for mid in STL_METHODS]
    ax.bar(x, rel, color=RED, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in STL_METHODS])
    ax.set_ylabel("RelMeasErr increase")
    ax.set_title("No-MC projection increases measurement inconsistency")
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.20)
    save_figure(fig, "figS1_relmeaserr_ablation_final", exts=("pdf", "png", "svg"))


def main() -> None:
    ensure_dir(FIG_DIR)
    ensure_dir(PROJECT_FIG_DIR)
    fig1_mechanism_final()
    fig2_primary_metrics_final()
    copy_figure_set(PHASE28_FIG_DIR, "fig3_qualitative_reconstruction_v28", "fig3_qualitative_final")
    copy_figure_set(PHASE28_FIG_DIR, "fig4_measurement_attribution_v28", "fig4_measurement_attribution_final")
    fig5_inference_ablation_final()
    figS1_relmeaserr_final()
    copy_figure_set(PHASE23_FIG_DIR, "fig6_robustness_baselines_v9", "fig6_robustness_baselines_final")
    print(
        {
            "figure_dir": str(FIG_DIR),
            "latex_figure_dir": str(PROJECT_FIG_DIR),
            "figure1": str(FIG_DIR / "fig1_mechanism_final.pdf"),
            "figure2": str(FIG_DIR / "fig2_primary_metrics_final.pdf"),
        }
    )


if __name__ == "__main__":
    main()
