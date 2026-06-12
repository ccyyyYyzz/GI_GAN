from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

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
    write_text,
)


OUT = Path("E:/ns_mc_gan_gi/outputs_phase23_top_journal_rewrite")
FIG_DIR = OUT / "figures"


def save_figure(fig: plt.Figure, stem: str, exts: tuple[str, ...] = ("pdf", "png", "svg"), dpi: int = 300) -> None:
    ensure_dir(FIG_DIR)
    for ext in exts:
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(FIG_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def arrow(ax, p0, p1, color=GRAY, lw=1.2, rad=0.0, scale=11) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=scale,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def box(ax, xy, w, h, text, fc, ec, fs=8.0, rounded=True) -> None:
    if rounded:
        patch = FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            fc=fc,
            ec=ec,
            lw=1.0,
        )
    else:
        patch = Rectangle(xy, w, h, fc=fc, ec=ec, lw=1.0)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _hadamard(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1]])
    h = _hadamard(n // 2)
    return np.block([[h, h], [h, -h]])


def fig1_concept_v9() -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.35, 5.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.975,
        "Low-sampling GI as measurement-constrained null-space completion",
        ha="center",
        va="top",
        fontsize=11.0,
        fontweight="bold",
    )

    # Layer backgrounds
    layer_specs = [
        (0.685, 0.235, LIGHT_BLUE, BLUE, "optical\nmeasure-\nment"),
        (0.372, 0.255, "#F7F7F7", GRAY, "geometry\nof y=Ax"),
        (0.085, 0.225, LIGHT_GREEN, GREEN, "algorithmic reconstruction"),
    ]
    for y, h, fc, ec, label in layer_specs:
        ax.add_patch(FancyBboxPatch((0.035, y), 0.93, h, boxstyle="round,pad=0.010,rounding_size=0.020", fc=fc, ec=ec, lw=0.9, alpha=0.72))
        ax.text(0.055, y + h - 0.035, label, ha="left", va="top", fontsize=8.2, color=ec, fontweight="bold", linespacing=0.95)

    rng = np.random.default_rng(23)
    for j, x0 in enumerate([0.260, 0.320, 0.380]):
        arr = rng.choice([-1, 1], size=(8, 8))
        ax.imshow(arr, cmap="gray", extent=(x0, x0 + 0.105, 0.775 - 0.022 * j, 0.880 - 0.022 * j), interpolation="nearest", zorder=4)
        ax.add_patch(Rectangle((x0, 0.775 - 0.022 * j), 0.105, 0.105, fill=False, ec=BLUE, lw=0.65, zorder=5))
    ax.text(0.370, 0.735, "patterns", ha="center", fontsize=7.6, color=BLUE)
    ax.add_patch(Ellipse((0.610, 0.815), 0.150, 0.190, fc="white", ec=GRAY, lw=1.1))
    ax.text(0.610, 0.815, "object\nx", ha="center", va="center", fontsize=8.0)
    ax.add_patch(Circle((0.840, 0.815), 0.062, fc="white", ec=BLUE, lw=1.2))
    ax.text(0.840, 0.815, "bucket\ny", ha="center", va="center", fontsize=8.0, color=BLUE)
    arrow(ax, (0.500, 0.815), (0.535, 0.815), BLUE, 1.2)
    arrow(ax, (0.690, 0.815), (0.770, 0.815), BLUE, 1.2)
    ax.text(0.735, 0.760, r"$y_i=\langle a_i,x\rangle$", ha="center", fontsize=8.0, color=BLUE)

    # Geometry layer: a space, an affine set, row-space and null-space directions.
    ax.add_patch(Ellipse((0.405, 0.500), 0.455, 0.205, angle=-12, fc="white", ec=BLUE, lw=1.1))
    ax.text(0.250, 0.586, "image space", fontsize=8.0, color=BLUE)
    ax.plot([0.190, 0.615], [0.432, 0.568], color=GREEN, lw=1.7)
    ax.text(0.493, 0.586, r"$\mathcal{C}_y=\{x:Ax=y\}$", fontsize=8.2, color=GREEN)
    ax.scatter([0.335], [0.478], s=70, color=BLUE, edgecolor="white", lw=0.7, zorder=5)
    ax.text(0.292, 0.438, r"$x_{\rm data}$", fontsize=8.2, color=BLUE)
    arrow(ax, (0.235, 0.472), (0.320, 0.492), BLUE, 1.0)
    ax.text(0.118, 0.454, "measured\nrow-space", ha="center", fontsize=7.6, color=BLUE)
    arrow(ax, (0.418, 0.503), (0.575, 0.553), GREEN, 1.15)
    ax.text(0.650, 0.522, "unmeasured\nnull-space", ha="center", fontsize=7.6, color=GREEN)
    ax.add_patch(Polygon([[0.742, 0.438], [0.897, 0.472], [0.875, 0.582], [0.710, 0.560]], closed=True, fc=LIGHT_ORANGE, ec=ORANGE, lw=1.0))
    ax.text(0.805, 0.511, "many plausible\nimages share y", ha="center", va="center", fontsize=7.6, color=ORANGE)
    arrow(ax, (0.630, 0.504), (0.712, 0.509), ORANGE, 1.0)

    # Algorithm layer.
    y0 = 0.145
    box(ax, (0.070, y0), 0.110, 0.075, r"$x_{\rm data}$", "white", BLUE, 8.3)
    box(ax, (0.245, y0), 0.125, 0.075, r"$G_\theta$", LIGHT_ORANGE, ORANGE, 8.5)
    box(ax, (0.435, y0), 0.110, 0.075, r"$P_N$", "white", GREEN, 8.5)
    box(ax, (0.610, y0), 0.110, 0.075, r"$\Pi_y$", "white", GREEN, 8.5)
    box(ax, (0.790, y0), 0.110, 0.075, r"$\hat{x}$", "white", BLUE, 8.5)
    arrow(ax, (0.180, y0 + 0.038), (0.245, y0 + 0.038), ORANGE)
    arrow(ax, (0.370, y0 + 0.038), (0.435, y0 + 0.038), GREEN)
    arrow(ax, (0.545, y0 + 0.038), (0.610, y0 + 0.038), GREEN)
    arrow(ax, (0.720, y0 + 0.038), (0.790, y0 + 0.038), BLUE)
    ax.text(0.307, 0.097, "neural prior", ha="center", fontsize=7.5, color=ORANGE)
    ax.text(0.490, 0.097, "null-space\ninsertion", ha="center", fontsize=7.4, color=GREEN)
    ax.text(0.665, 0.097, "measurement\naudit", ha="center", fontsize=7.4, color=GREEN)
    arrow(ax, (0.845, 0.220), (0.690, 0.300), RED, 1.0, rad=0.18)
    ax.text(0.727, 0.310, r"check $A\hat{x}\approx y$", fontsize=7.4, color=RED)
    ax.text(
        0.500,
        0.034,
        r"Final form: $\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))]$",
        ha="center",
        fontsize=8.2,
    )
    save_figure(fig, "fig1_concept_v9")


def fig2_primary_metrics_v9() -> None:
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
    save_figure(fig, "fig2_primary_metrics_v9")


def fig3_qualitative_reconstruction_v9() -> None:
    setup_matplotlib()
    selected = [
        ("rademacher5_hq_noise001_colab", 5, "aircraft-like sample"),
        ("scrambled_hadamard5_hq_noise001_colab", 2, "ship/aircraft-like sample"),
        ("rademacher10_full_noise001_colab", 3, "vehicle-like sample"),
        ("scrambled_hadamard10_full_noise001_colab", 4, "vehicle-like sample"),
    ]
    cols = ["GT", "Backprojection", "Reconstruction", "Absolute error"]
    fig = plt.figure(figsize=(7.35, 7.25))
    gs = fig.add_gridspec(len(selected), 5, width_ratios=[1, 1, 1, 1, 0.055], wspace=0.045, hspace=0.075)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(len(selected))])
    cax = fig.add_subplot(gs[:, 4])
    err_arrays: list[np.ndarray] = []
    cells_by_row = []
    warnings = []
    for mid, sample_idx, note in selected:
        try:
            cells = crop_sample(mid, sample_idx)
            gt = np.asarray(cells["GT"].convert("L"), dtype=np.float32) / 255.0
            recon = np.asarray(cells["Reconstruction"].convert("L"), dtype=np.float32) / 255.0
            err = np.abs(gt - recon)
            err_arrays.append(err)
            cells_by_row.append((cells, err))
        except Exception as exc:  # pragma: no cover - diagnostic path
            warnings.append(f"- WARNING: could not load {mid} row {sample_idx}: {exc}")
            blank = np.zeros((256, 256), dtype=np.float32)
            cells_by_row.append(({}, blank))
            err_arrays.append(blank)
        warnings.append(f"- Selected {METHOD_LABEL[mid]} row {sample_idx}: {note}.")
    vmax = max(0.065, float(np.quantile(np.concatenate([e.ravel() for e in err_arrays]), 0.985)))
    last_im = None
    for ridx, ((mid, _, _), (cells, err)) in enumerate(zip(selected, cells_by_row)):
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
                ax.set_title(col, fontsize=9.2)
            if cidx == 0:
                ax.set_ylabel(METHOD_LABEL[mid], rotation=0, labelpad=31, va="center", fontsize=8.8)
    if last_im is not None:
        fig.colorbar(last_im, cax=cax).ax.tick_params(labelsize=6.5)
    fig.text(0.5, 0.014, "Representative STL-10 samples were reselected for visually clearer object structure. Error maps use a shared 98.5th-percentile scale.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.13, right=0.965, top=0.925, bottom=0.075)
    save_figure(fig, "fig3_qualitative_reconstruction_v9")
    write_text(
        OUT / "qualitative_selection_warning.md",
        "# Qualitative Selection Audit\n\n"
        "No new evaluation was run. Samples were selected from saved evaluation grids to avoid visually indistinct main-text examples.\n\n"
        + "\n".join(warnings),
    )


def _attr_rows() -> dict[str, dict[str, str]]:
    return {r["method_id"]: r for r in table("attribution")}


def fig4_regime_map_v9() -> None:
    setup_matplotlib()
    attr = _attr_rows()
    display = [
        ("rademacher5_hq_noise001_colab", "Rad-5", ORANGE),
        ("scrambled_hadamard5_hq_noise001_colab", "Scr-5", GREEN),
        ("rademacher10_full_noise001_colab", "Rad-10", RED),
        ("scrambled_hadamard10_full_noise001_colab", "Scr-10", BLUE),
        ("stl10_hadamard5_local_medium", "Lowfreq-5", PURPLE),
        ("stl10_hadamard10_local_full", "Lowfreq-10", GRAY),
    ]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.35, 3.65), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax.axvspan(6.0, 10.2, color=LIGHT_ORANGE, alpha=0.42)
    ax.axvspan(10.2, 16.0, color=LIGHT_GREEN, alpha=0.30)
    ax.axvspan(16.0, 21.0, color=LIGHT_BLUE, alpha=0.28)
    ax.text(8.35, 17.75, "weak initialization\nlarge learned gain", ha="center", fontsize=6.6)
    ax.text(13.2, 12.0, "stronger initialization\nmoderate gain", ha="center", fontsize=6.9)
    ax.text(18.4, 6.2, "strong initialization\nsmaller gain", ha="center", fontsize=6.9)
    for mid, label, color in display:
        row = attr[mid]
        x = as_float(row["backproj_psnr"])
        y = as_float(row["delta_psnr"])
        marker = "o" if "rademacher" in mid else "s" if "scrambled" in mid else "^"
        ax.scatter([x], [y], s=66, marker=marker, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        if label == "Scr-5":
            offset = (0.25, -0.68)
        elif label == "Lowfreq-10":
            offset = (0.24, 0.58)
        elif "rademacher" in mid:
            offset = (0.12, -0.55)
        else:
            offset = (0.15, 0.30)
        ax.text(x + offset[0], y + offset[1], label, fontsize=7.0)
    ax.set_xlabel("Backprojection PSNR (dB)")
    ax.set_ylabel(r"Neural gain $\Delta$PSNR (dB)")
    ax.set_title("Regime map")
    ax.set_xlim(6.3, 20.2)
    ax.set_ylim(3.5, 18.7)
    ax.grid(alpha=0.25)
    ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    bar_mids = [
        ("rademacher5_hq_noise001_colab", "Rad-5", ORANGE),
        ("scrambled_hadamard5_hq_noise001_colab", "Scr-5", GREEN),
        ("rademacher10_full_noise001_colab", "Rad-10", RED),
        ("scrambled_hadamard10_full_noise001_colab", "Scr-10", BLUE),
    ]
    x = np.arange(len(bar_mids))
    bp = [as_float(attr[mid]["backproj_psnr"]) for mid, _, _ in bar_mids]
    model = [as_float(attr[mid]["model_psnr"]) for mid, _, _ in bar_mids]
    axb.bar(x - 0.18, bp, width=0.36, color=LIGHT_GRAY, edgecolor=GRAY, label="BP")
    axb.bar(x + 0.18, model, width=0.36, color=[c for _, _, c in bar_mids], label="Model")
    axb.set_xticks(x)
    axb.set_xticklabels([label for _, label, _ in bar_mids], rotation=22, ha="right")
    axb.set_ylabel("PSNR (dB)")
    axb.set_title("BP vs model")
    axb.grid(axis="y", alpha=0.25)
    axb.legend(frameon=False, ncol=2, fontsize=7.0)
    axb.text(-0.18, 1.05, "(b)", transform=axb.transAxes, fontsize=9, fontweight="bold")
    fig.text(0.5, 0.015, "Low-frequency Hadamard points are auxiliary regime diagnostics, not main high-quality claims.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.20, wspace=0.34)
    save_figure(fig, "fig4_regime_map_v9")


def fig5_inference_ablation_v9() -> None:
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
    fig.text(0.5, 0.012, "Removing the measurement projection is the clearest failure; -Null is reported as a limited metric change for these trained checkpoints.", ha="center", fontsize=6.8)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.12, hspace=0.30)
    save_figure(fig, "fig5_inference_ablation_v9")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(STL_METHODS))
    rel = [as_float(ab[(mid, "no_dc_project")]["delta_vs_full_relmeaserr"]) for mid in STL_METHODS]
    ax.bar(x, rel, color=RED, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in STL_METHODS])
    ax.set_ylabel("RelMeasErr increase")
    ax.set_title("No-DC projection increases measurement inconsistency")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "figS1_relmeaserr_ablation_v9", exts=("pdf", "png"))


def fig6_robustness_baselines_v9() -> None:
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
    ax.set_title("Rad/Scr-5 perturbation")
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
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids], rotation=15, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("STL-10 Ours vs CS-TV")
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
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids], rotation=15, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Bootstrap 95% CI")
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "(d)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.text(0.5, 0.012, "These diagnostics support finite-noise stability and measurement dependence; they do not imply universal robustness.", ha="center", fontsize=7.0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.15, wspace=0.30, hspace=0.48)
    save_figure(fig, "fig6_robustness_baselines_v9")


def main() -> None:
    ensure_dir(OUT)
    fig1_concept_v9()
    fig2_primary_metrics_v9()
    fig3_qualitative_reconstruction_v9()
    fig4_regime_map_v9()
    fig5_inference_ablation_v9()
    fig6_robustness_baselines_v9()
    print({"figures": str(FIG_DIR)})


if __name__ == "__main__":
    main()
