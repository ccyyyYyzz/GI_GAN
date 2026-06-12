from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from .phase18_rewrite_common import METHOD_LABEL, OUT, PHASE15, PHASE16, as_float, main_results_rows, registry_by_id, table


FIG = OUT / "figures"
LATEX_FIG = OUT / "latex_project" / "figures"

BLUE = "#2F6F95"
LIGHT_BLUE = "#DCECF5"
ORANGE = "#D98C3A"
LIGHT_ORANGE = "#FAE6D1"
GREEN = "#4A9B67"
LIGHT_GREEN = "#DDEFE3"
RED = "#B94A48"
GRAY = "#4D4D4D"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_all(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        path = FIG / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        shutil.copy2(path, LATEX_FIG / f"{stem}.{ext}")
    plt.close(fig)


def panel(ax, label: str) -> None:
    ax.text(-0.04, 1.03, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def box(ax, xy, w, h, text, fc, ec, fontsize=8.5) -> None:
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.018,rounding_size=0.025", fc=fc, ec=ec, lw=1.2)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#202020")


def arrow(ax, a, b, color=GRAY, lw=1.2, rad=0.0) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def fig1_mechanism() -> None:
    setup_style()
    fig = plt.figure(figsize=(12.5, 5.2))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.0, 0.95], wspace=0.28, hspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    bottom = fig.add_subplot(gs[1, :])

    # (a) acquisition
    ax = axes[0]
    ax.axis("off")
    for i, x in enumerate([0.08, 0.17, 0.26]):
        ax.add_patch(Rectangle((x, 0.56 - 0.05 * i), 0.22, 0.26, fc=LIGHT_BLUE, ec=BLUE, lw=1))
        ax.plot([x + 0.03, x + 0.19], [0.61 - 0.05 * i, 0.76 - 0.05 * i], color=BLUE, lw=1)
        ax.plot([x + 0.19, x + 0.03], [0.61 - 0.05 * i, 0.76 - 0.05 * i], color=BLUE, lw=1)
    ax.add_patch(Ellipse((0.56, 0.62), 0.24, 0.32, fc="#ECECEC", ec=GRAY, lw=1.2))
    ax.text(0.56, 0.62, "$x$", ha="center", va="center", fontsize=13)
    ax.add_patch(Circle((0.86, 0.62), 0.085, fc=LIGHT_BLUE, ec=BLUE, lw=1.2))
    ax.text(0.86, 0.62, "$y_i$", ha="center", va="center", fontsize=10)
    arrow(ax, (0.34, 0.65), (0.44, 0.64), BLUE)
    arrow(ax, (0.68, 0.62), (0.77, 0.62), BLUE)
    ax.text(0.48, 0.18, "$y_i=\\langle a_i,x\\rangle+\\epsilon_i$", ha="center", fontsize=9)
    ax.set_title("Acquisition")
    panel(ax, "a")

    # (b) underdetermined set
    ax = axes[1]
    ax.axis("off")
    ax.add_patch(Ellipse((0.52, 0.56), 0.72, 0.38, angle=-12, fc=LIGHT_BLUE, ec=BLUE, lw=1.2))
    ax.add_patch(FancyArrowPatch((0.2, 0.35), (0.83, 0.72), arrowstyle="<->", mutation_scale=9, color=GREEN, lw=1.3))
    ax.scatter([0.42, 0.61], [0.55, 0.58], s=28, color=[BLUE, ORANGE], zorder=3)
    ax.text(0.42, 0.45, "$x_0$", ha="center")
    ax.text(0.61, 0.68, "$x_0+v$", ha="center")
    ax.text(0.52, 0.2, "$\\mathcal{C}_y=\\{x:Ax=y\\}$\n$v\\in\\mathrm{Null}(A)$", ha="center", fontsize=8.5)
    ax.set_title("Underdetermined set")
    panel(ax, "b")

    # (c) data solution + residual
    ax = axes[2]
    ax.axis("off")
    box(ax, (0.08, 0.56), 0.35, 0.22, "$x_{data}$", LIGHT_BLUE, BLUE)
    box(ax, (0.58, 0.56), 0.35, 0.22, "$G_\\theta$", LIGHT_ORANGE, ORANGE)
    box(ax, (0.34, 0.18), 0.36, 0.2, "$r_\\theta$", LIGHT_ORANGE, ORANGE)
    arrow(ax, (0.43, 0.67), (0.58, 0.67), ORANGE)
    arrow(ax, (0.75, 0.56), (0.56, 0.38), ORANGE)
    ax.text(0.5, 0.05, "predict only missing structure", ha="center", fontsize=8)
    ax.set_title("Neural residual")
    panel(ax, "c")

    # (d) projections
    ax = axes[3]
    ax.axis("off")
    box(ax, (0.08, 0.6), 0.36, 0.22, "$P_N$", LIGHT_GREEN, GREEN)
    box(ax, (0.56, 0.6), 0.36, 0.22, "$\\Pi_y$", LIGHT_GREEN, GREEN)
    box(ax, (0.3, 0.22), 0.4, 0.2, "$\\tilde{x}\\rightarrow\\hat{x}$", "#F3F3F3", GRAY)
    arrow(ax, (0.44, 0.71), (0.56, 0.71), GREEN)
    arrow(ax, (0.73, 0.6), (0.55, 0.42), GREEN)
    ax.text(0.5, 0.06, "$P_N$ completes null space;\n$\\Pi_y$ restores measurements", ha="center", fontsize=8)
    ax.set_title("Physics projections")
    panel(ax, "d")

    # (e) output
    ax = axes[4]
    ax.axis("off")
    ax.add_patch(Rectangle((0.15, 0.36), 0.34, 0.34, fc=LIGHT_BLUE, ec=BLUE, lw=1.1))
    ax.add_patch(Rectangle((0.52, 0.36), 0.34, 0.34, fc=LIGHT_GREEN, ec=GREEN, lw=1.1))
    ax.text(0.32, 0.54, "BP", ha="center", va="center", fontsize=11)
    ax.text(0.69, 0.54, "$\\hat{x}$", ha="center", va="center", fontsize=13)
    arrow(ax, (0.49, 0.53), (0.52, 0.53), GREEN)
    ax.text(0.5, 0.17, "$A\\hat{x}\\approx y$\nimage metrics use clipped output", ha="center", fontsize=8)
    ax.set_title("Reconstruction")
    panel(ax, "e")

    bottom.axis("off")
    box(bottom, (0.04, 0.42), 0.18, 0.24, "measured row space\nprotected by data solution", LIGHT_BLUE, BLUE)
    box(bottom, (0.30, 0.42), 0.18, 0.24, "unmeasured null space\ncompleted by network", LIGHT_ORANGE, ORANGE)
    box(bottom, (0.56, 0.42), 0.18, 0.24, "measurement projection\nchecks bucket signal", LIGHT_GREEN, GREEN)
    box(bottom, (0.82, 0.42), 0.14, 0.24, "inconsistency\npenalized", "#F8DDDA", RED)
    for a, b, color in [((0.22, 0.54), (0.30, 0.54), GRAY), ((0.48, 0.54), (0.56, 0.54), GRAY), ((0.74, 0.54), (0.82, 0.54), GRAY)]:
        arrow(bottom, a, b, color)
    bottom.text(0.5, 0.16, "Low-sampling GI is treated as measurement-constrained completion, not ordinary denoising.", ha="center", fontsize=10)
    save_all(fig, "fig1_mechanism")


def pattern(kind: str, n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(8)
    if kind == "rademacher":
        return rng.choice([-1.0, 1.0], size=(n, n))
    x = np.arange(n)
    y = np.arange(n)[:, None]
    if kind == "lowfreq":
        return np.cos(2 * np.pi * x / n) + np.cos(2 * np.pi * y / n)
    return (((x[None, :] * 5 + y * 9) % 23) < 11).astype(float) * 2 - 1


def fig2_measurement_attribution() -> None:
    setup_style()
    attr = [r for r in table("attribution") if r.get("method_id") in METHOD_LABEL]
    order = [m for m in ["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab", "rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab", "mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"]]
    by_id = {r["method_id"]: r for r in attr}
    attr = [by_id[m] for m in order if m in by_id]
    fig = plt.figure(figsize=(12.5, 6.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.9, 1.35], wspace=0.34, hspace=0.52)
    for idx, (kind, title) in enumerate([("rademacher", "Rademacher"), ("scrambled", "Scrambled Hadamard"), ("lowfreq", "Low-frequency Hadamard")]):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(pattern(kind), cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        panel(ax, chr(ord("a") + idx))
    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(len(attr))
    bp = [as_float(r["backproj_psnr"]) for r in attr]
    model = [as_float(r["model_psnr"]) for r in attr]
    ax.bar(x - 0.18, bp, width=0.36, label="Backprojection", color="#9DB9CA")
    ax.bar(x + 0.18, model, width=0.36, label="Model", color=BLUE)
    for xi, value in zip(x - 0.18, bp):
        ax.text(xi, value + 0.25, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    for xi, value in zip(x + 0.18, model):
        ax.text(xi, value + 0.25, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[r["method_id"]] for r in attr], rotation=32, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Physical initialization vs final model")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    panel(ax, "d")
    ax = fig.add_subplot(gs[1, 2])
    delta = [as_float(r["delta_psnr"]) for r in attr]
    ax.barh([METHOD_LABEL[r["method_id"]] for r in attr], delta, color=GREEN)
    for yi, value in enumerate(delta):
        ax.text(value + 0.25, yi, f"{value:.1f}", va="center", fontsize=7)
    ax.set_xlabel(r"$\Delta$PSNR (dB)")
    ax.set_title("Neural refinement gain")
    ax.grid(axis="x", alpha=0.25)
    panel(ax, "e")
    save_all(fig, "fig2_measurement_attribution")


def fig3_main_results() -> None:
    setup_style()
    reg = registry_by_id()
    stl5 = ["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab"]
    stl10 = ["rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"]
    simple = ["mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"]
    fig = plt.figure(figsize=(12.5, 6.0))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.08], wspace=0.32, hspace=0.55)
    groups = [(stl5, "STL-10 5%", 20.0, 0.60), (stl10, "STL-10 10%", 22.0, 0.65), (simple, "MNIST/Fashion 5%", 25.0, 0.80)]
    for i, (ids, title, psnr_thr, _ssim_thr) in enumerate(groups):
        ax = fig.add_subplot(gs[0, i])
        xs = np.arange(len(ids))
        vals = [as_float(reg[mid]["psnr"]) for mid in ids]
        ax.bar(xs, vals, color=[BLUE, GREEN][: len(ids)])
        ax.axhline(psnr_thr, color=RED, ls="--", lw=1.1, label="HQ threshold")
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.25, f"{v:.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABEL[mid] for mid in ids], rotation=25, ha="right")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if i == 0:
            ax.legend(frameon=False, fontsize=7)
        panel(ax, chr(ord("a") + i))
    ax = fig.add_subplot(gs[1, :])
    grid_path = PHASE15 / "imported_noleak" / "rademacher5_hq_noise001_colab" / "eval_samples" / "recon_grid.png"
    if grid_path.exists():
        img = plt.imread(grid_path)
        ax.imshow(img)
        ax.set_title("Representative imported no-leak reconstruction grid (Rad-5)")
        ax.axis("off")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "TODO: insert GT / Backprojection / Reconstruction / Error examples", ha="center", va="center", fontsize=12)
    panel(ax, "d")
    save_all(fig, "fig3_main_results")


def fig4_inference_ablation() -> None:
    setup_style()
    rows = table("ablation")
    methods = ["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab", "rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"]
    modes = ["full_model", "no_dc_project", "no_null_project", "stage1_only", "raw_weights", "ema_weights"]
    by = {(r["method_id"], r["ablation_mode"]): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.7), sharey=True)
    colors = [BLUE, RED, "#C0C7CD", ORANGE, "#9B8AC0", GREEN]
    for ax, mid in zip(axes.ravel(), methods):
        vals = [as_float(by[(mid, mode)]["psnr"]) for mode in modes if (mid, mode) in by]
        labels = [mode.replace("_", "\n").replace("project", "proj.") for mode in modes if (mid, mode) in by]
        xs = np.arange(len(vals))
        ax.bar(xs, vals, color=colors[: len(vals)])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.22, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(METHOD_LABEL[mid])
        ax.set_ylabel("PSNR (dB)")
        ax.grid(axis="y", alpha=0.25)
    for i, ax in enumerate(axes.ravel()):
        panel(ax, chr(ord("a") + i))
    save_all(fig, "fig4_inference_ablation")


def fig5_robustness_baselines() -> None:
    setup_style()
    noise = table("noise")
    pert = table("perturbation")
    base = table("baseline")
    stats = table("statistics")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.0), gridspec_kw={"wspace": 0.32, "hspace": 0.55})
    ax = axes[0, 0]
    for mid in ["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab", "rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"]:
        sub = sorted([r for r in noise if r["method_id"] == mid], key=lambda r: as_float(r["noise_std"]))
        ax.plot([as_float(r["noise_std"]) for r in sub], [as_float(r["psnr"]) for r in sub], marker="o", label=METHOD_LABEL[mid])
    ax.set_xlabel("Measurement noise std")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Finite noise sweep")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    panel(ax, "a")
    ax = axes[0, 1]
    severe = [r for r in pert if r.get("perturbation_mode") in {"shuffle_coefficients", "wrong_sample"}]
    xs = np.arange(len(severe))
    vals = [as_float(r["psnr_drop_from_normal"]) for r in severe]
    ax.bar(xs, vals, color=ORANGE)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL.get(r["method_id"], "") + "\n" + r["perturbation_mode"].replace("_", " ") for r in severe], rotation=30, ha="right", fontsize=6.8)
    ax.set_ylabel("PSNR drop (dB)")
    ax.set_title("Measurement perturbation")
    ax.grid(axis="y", alpha=0.25)
    panel(ax, "b")
    ax = axes[1, 0]
    model = registry_by_id()
    best = []
    for mid in ["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab", "rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab", "mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"]:
        sub = [r for r in base if r["method_id"] == mid and r["baseline"] == "tv_pgd"]
        if sub:
            tv = max(sub, key=lambda r: as_float(r["psnr"]))
            best.append((mid, as_float(model[mid]["psnr"]), as_float(tv["psnr"])))
    x = np.arange(len(best))
    ax.bar(x - 0.18, [b[1] for b in best], width=0.36, color=BLUE, label="Ours")
    ax.bar(x + 0.18, [b[2] for b in best], width=0.36, color="#9B8A4E", label="CS-TV (PGD)")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[b[0]] for b in best], rotation=30, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Learned model vs CS-TV")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    panel(ax, "c")
    ax = axes[1, 1]
    xs = np.arange(len(stats))
    means = np.array([as_float(r["mean_psnr"]) for r in stats])
    lo = np.array([as_float(r["ci95_psnr_low"]) for r in stats])
    hi = np.array([as_float(r["ci95_psnr_high"]) for r in stats])
    ax.errorbar(xs, means, yerr=np.vstack([means - lo, hi - means]), fmt="o", color=GREEN, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL.get(r["method_id"], r["method_id"]) for r in stats], rotation=30, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Bootstrap 95% CI")
    ax.grid(axis="y", alpha=0.25)
    panel(ax, "d")
    save_all(fig, "fig5_robustness_baselines")


def supplement_figures() -> None:
    mapping = {
        "figS1_dc_row_control": PHASE16 / "dc_row_control" / "dc_row_psnr.png",
        "figS2_classwise": PHASE16 / "classwise" / "classwise_psnr.png",
        "figS4_histograms": PHASE16 / "statistics" / "psnr_histograms.png",
    }
    for stem, src in mapping.items():
        if not src.exists():
            continue
        img = plt.imread(src)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.imshow(img)
        ax.axis("off")
        save_all(fig, stem)
    runtime = table("runtime")
    if runtime:
        setup_style()
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        rows = [r for r in runtime if r.get("path") == "ns_mc_gan_full_inference"]
        ax.bar([METHOD_LABEL.get(r["method_id"], r["method_id"]) for r in rows], [as_float(r["runtime_sec_per_image"]) for r in rows], color=BLUE)
        ax.set_ylabel("sec / image")
        ax.set_title("Runtime: learned inference path")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        save_all(fig, "figS3_runtime")


def main() -> None:
    fig1_mechanism()
    fig2_measurement_attribution()
    fig3_main_results()
    fig4_inference_ablation()
    fig5_robustness_baselines()
    supplement_figures()
    print({"figures": str(FIG), "latex_figures": str(LATEX_FIG)})


if __name__ == "__main__":
    main()
