from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase33_mechanism_overhaul"
FIG_DIR = OUT / "figures"
ATTR_CSV = ROOT / "outputs_phase16" / "supplementary_experiments" / "attribution" / "attribution_final.csv"

BLUE = "#1F77B4"
LIGHT_BLUE = "#E8F2FB"
ORANGE = "#E58A1F"
LIGHT_ORANGE = "#FFF0D9"
GREEN = "#2E8B57"
LIGHT_GREEN = "#E6F4EA"
RED = "#C93C3C"
LIGHT_RED = "#FBE7E7"
GRAY = "#60646B"
LIGHT_GRAY = "#F5F6F7"
MID_GRAY = "#A9ADB5"
PURPLE = "#7A5AC8"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str, tight: bool = False) -> None:
    kwargs = {"bbox_inches": "tight"} if tight else {}
    for ext in ("pdf", "png", "svg"):
        if ext == "png":
            fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, **kwargs)
        else:
            fig.savefig(FIG_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def arrow(ax, p0, p1, color=GRAY, lw=1.3, rad=0.0, scale=12) -> None:
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


def rounded(ax, xy, w, h, text="", fc=LIGHT_GRAY, ec=GRAY, lw=1.2, fs=9, color="black") -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            fc=fc,
            ec=ec,
            lw=lw,
        )
    )
    if text:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def pattern_stack(ax, x0, y0, scale=0.105) -> None:
    rng = np.random.default_rng(33)
    for i in range(3):
        arr = rng.choice([0, 1], size=(7, 7))
        x = x0 + 0.030 * i
        y = y0 - 0.030 * i
        ax.imshow(arr, cmap="gray", interpolation="nearest", extent=(x, x + scale, y, y + scale), zorder=2)
        ax.add_patch(Rectangle((x, y), scale, scale, fill=False, ec=BLUE, lw=0.9, zorder=3))
    ax.text(x0 + 0.080, y0 + scale + 0.022, r"known patterns $a_i$", ha="center", fontsize=9, color=BLUE)


def image_icon(ax, cx, cy, w=0.13, h=0.13, label=r"$x$", ec=GRAY, fc=LIGHT_GRAY) -> None:
    rounded(ax, (cx - w / 2, cy - h / 2), w, h, fc=fc, ec=ec, lw=1.1)
    ax.add_patch(Ellipse((cx - 0.026, cy - 0.004), w * 0.45, h * 0.42, angle=-20, fc="#C7D3DD", ec="none"))
    ax.add_patch(Ellipse((cx + 0.026, cy + 0.012), w * 0.33, h * 0.30, angle=25, fc="#93A4B3", ec="none"))
    ax.plot([cx - w * 0.34, cx + w * 0.36], [cy - h * 0.30, cy - h * 0.25], color="#6B7680", lw=1.0)
    ax.text(cx, cy - h * 0.64, label, ha="center", fontsize=9, color=ec)


def rough_image(ax, cx, cy, w=0.12, h=0.12, label=r"$x_{\rm data}$") -> None:
    rng = np.random.default_rng(3)
    arr = rng.normal(size=(12, 12))
    ax.imshow(arr, cmap="gray", interpolation="nearest", extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2))
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec=BLUE, lw=1.0))
    ax.text(cx, cy - h * 0.66, label, ha="center", fontsize=9, color=BLUE)


def bucket_icon(ax, cx, cy, label=r"$y$") -> None:
    rounded(ax, (cx - 0.035, cy - 0.075), 0.070, 0.150, fc=LIGHT_BLUE, ec=BLUE, lw=1.1)
    for i, ht in enumerate([0.035, 0.060, 0.092, 0.048]):
        x = cx - 0.023 + i * 0.015
        ax.add_patch(Rectangle((x, cy - 0.055), 0.009, ht, fc=BLUE, ec="none"))
    ax.text(cx, cy - 0.105, label, ha="center", fontsize=10, color=BLUE)


def draw_pn_filter(ax, x, y, w, h, show_label: bool = True) -> None:
    pts = [(x, y + h), (x + w, y + h), (x + w * 0.64, y + h * 0.46), (x + w * 0.64, y), (x + w * 0.36, y), (x + w * 0.36, y + h * 0.46)]
    ax.add_patch(Polygon(pts, closed=True, fc=LIGHT_ORANGE, ec=ORANGE, lw=1.4))
    ax.text(x + w / 2, y + h * 0.58, r"$P_N$", ha="center", va="center", fontsize=12, color=ORANGE)
    if show_label:
        ax.text(x + w / 2, y - 0.025, "residual filter", ha="center", fontsize=9, color=ORANGE)


def draw_audit(ax, cx, cy) -> None:
    ax.add_patch(Circle((cx, cy), 0.050, fc=LIGHT_GREEN, ec=GREEN, lw=1.4))
    ax.plot([cx - 0.022, cx - 0.006, cx + 0.027], [cy, cy - 0.016, cy + 0.020], color=GREEN, lw=2.0)
    ax.text(cx, cy - 0.077, r"$\Pi_y$ final audit", ha="center", fontsize=9, color=GREEN)


def base_axes(figsize=(7.2, 5.2)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def write_storyboard() -> None:
    text = r"""# Figure 1 Storyboard

## Five-sentence mechanism for non-specialist readers

1. The bucket measurements \(y\) do not determine a unique image when \(m\ll n\).
2. The data solution \(x_{\rm data}\) is the part we can compute directly from the measured buckets.
3. A neural network proposes missing image structure, but it should not overwrite what the measurements already determine.
4. \(P_N\) filters the neural residual so that it mostly fills directions weakly seen or unseen by the measurement operator.
5. \(\Pi_y\) is the final audit: the completed image is projected back so that its simulated bucket measurements match \(y\).

## 中文解释版

- 桶测量太少，所以很多图像都可能符合。
- \(x_{\rm data}\) 是物理粗图，代表测量已经给出的部分。
- 网络只负责补缺失结构。
- \(P_N\) 限制网络不要改已测信息。
- \(\Pi_y\) 最后把整张图重新拿去和 bucket measurements 对账。
"""
    (OUT / "FIGURE1_STORYBOARD.md").write_text(text, encoding="utf-8")


def variant_a(stem: str = "fig1_variant_A_problem_solution") -> None:
    fig, ax = base_axes((7.2, 5.35))
    fig.suptitle("Low-sampling GI as measurement-constrained completion", fontsize=13.5, fontweight="bold", y=0.965)
    cols = [(0.035, 0.105, 0.290, 0.755), (0.360, 0.105, 0.280, 0.755), (0.675, 0.105, 0.292, 0.755)]
    titles = [
        "Few bucket\nmeasurements",
        "Unconstrained neural\nreconstruction can drift",
        "Measured part + missing\npart + audit",
    ]
    colors = [(BLUE, LIGHT_BLUE), (RED, LIGHT_RED), (GREEN, LIGHT_GREEN)]
    for (x, y, w, h), title, (ec, _fc), letter in zip(cols, titles, colors, "ABC"):
        rounded(ax, (x, y), w, h, fc="white", ec="#D1D5DB", lw=1.25)
        ax.text(x + 0.018, y + h - 0.058, f"({letter})", fontsize=11.5, fontweight="bold", ha="left")
        ax.text(x + w / 2 + 0.025, y + h - 0.055, title, fontsize=9.4, fontweight="bold", ha="center", color=ec)

    x, y, w, _h = cols[0]
    pattern_stack(ax, x + 0.028, y + 0.475, 0.078)
    image_icon(ax, x + 0.147, y + 0.525, w=0.085, h=0.082, label=r"object $x$")
    bucket_icon(ax, x + 0.240, y + 0.525, r"$y$")
    arrow(ax, (x + 0.190, y + 0.525), (x + 0.207, y + 0.525), BLUE, scale=11)
    ax.text(x + w / 2, y + 0.365, r"$y=Ax+\epsilon,\quad m\ll n$", ha="center", fontsize=10.3, color=BLUE)
    ax.text(x + w / 2, y + 0.260, "Many images can produce\nthe same bucket vector.", ha="center", fontsize=9.2, color=GRAY)
    ax.add_patch(Ellipse((x + w / 2, y + 0.205), 0.190, 0.052, fc="none", ec=BLUE, lw=1.0, alpha=0.8))

    x, y, w, _h = cols[1]
    rough_image(ax, x + 0.070, y + 0.545, w=0.085, h=0.085, label=r"$x_{\rm data}$")
    rounded(ax, (x + 0.124, y + 0.505), 0.070, 0.078, r"neural", fc=LIGHT_ORANGE, ec=ORANGE, fs=9.0, color=ORANGE)
    image_icon(ax, x + 0.235, y + 0.545, w=0.085, h=0.085, label=r"plausible $\hat{x}$", ec=RED, fc="#FFF8F8")
    arrow(ax, (x + 0.112, y + 0.545), (x + 0.124, y + 0.545), ORANGE, scale=10)
    arrow(ax, (x + 0.194, y + 0.545), (x + 0.192, y + 0.545), ORANGE, scale=10)
    ax.text(x + w / 2, y + 0.382, r"$A\hat{x}\not\approx y$", ha="center", fontsize=12.0, color=RED)
    ax.text(x + w / 2, y + 0.292, "visually plausible,\nbut not measurement-audited", ha="center", fontsize=9.2, color=RED)
    rounded(ax, (x + 0.055, y + 0.180), w - 0.110, 0.060, "risk: buckets can be violated", fc=LIGHT_RED, ec=RED, fs=8.8, color=RED)

    x, y, w, _h = cols[2]
    ax.text(x + 0.020, y + 0.645, "1", fontsize=10.5, fontweight="bold", color=BLUE)
    rounded(ax, (x + 0.048, y + 0.610), 0.205, 0.064, r"$x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y$", fc=LIGHT_BLUE, ec=BLUE, fs=8.7, color=BLUE)
    ax.text(x + 0.150, y + 0.579, "measured component", ha="center", fontsize=9.0, color=BLUE)

    ax.text(x + 0.020, y + 0.455, "2", fontsize=10.5, fontweight="bold", color=ORANGE)
    rough_image(ax, x + 0.060, y + 0.430, w=0.066, h=0.066, label="")
    rounded(ax, (x + 0.105, y + 0.397), 0.060, 0.060, r"$G_\theta$", fc=LIGHT_ORANGE, ec=ORANGE, fs=9.3, color=ORANGE)
    draw_pn_filter(ax, x + 0.185, y + 0.385, 0.054, 0.082, show_label=False)
    arrow(ax, (x + 0.093, y + 0.430), (x + 0.105, y + 0.430), ORANGE, scale=10)
    arrow(ax, (x + 0.165, y + 0.430), (x + 0.185, y + 0.430), ORANGE, scale=10)
    ax.text(x + 0.213, y + 0.360, "residual\nfilter", ha="center", fontsize=8.5, color=ORANGE)
    ax.text(x + 0.150, y + 0.303, "learned missing structure\nP_N avoids measured part", ha="center", fontsize=8.5, color=ORANGE)

    ax.text(x + 0.020, y + 0.210, "3", fontsize=10.5, fontweight="bold", color=GREEN)
    rounded(ax, (x + 0.052, y + 0.181), 0.130, 0.058, r"$x_{\rm data}+P_N(G_\theta)$", fc="white", ec=GRAY, fs=8.5)
    ax.add_patch(Circle((x + 0.230, y + 0.220), 0.045, fc=LIGHT_GREEN, ec=GREEN, lw=1.4))
    ax.plot([x + 0.210, x + 0.224, x + 0.252], [y + 0.221, y + 0.205, y + 0.242], color=GREEN, lw=1.9)
    ax.text(x + 0.230, y + 0.152, r"$\Pi_y$ final audit", ha="center", fontsize=9.0, color=GREEN)
    image_icon(ax, x + 0.230, y + 0.078, w=0.068, h=0.055, label=r"$\hat{x}$", ec=GREEN, fc=LIGHT_GREEN)
    ax.text(x + 0.230, y + 0.018, r"$A\hat{x}\approx y$", ha="center", fontsize=9.4, color=GREEN)
    ax.text(x + 0.150, y + 0.000, "measurement-consistent reconstruction", ha="center", fontsize=8.6, color=GREEN)
    arrow(ax, (x + 0.182, y + 0.210), (x + 0.180, y + 0.210), GREEN, scale=9)
    arrow(ax, (x + 0.230, y + 0.175), (x + 0.230, y + 0.116), GREEN, scale=9)

    ax.set_aspect("auto")
    fig.subplots_adjust(left=0, right=1, top=0.935, bottom=0.02)
    save(fig, stem)


def variant_b() -> None:
    fig, ax = base_axes((7.2, 5.0))
    fig.suptitle("Measurement geometry plus reconstruction pipeline", fontsize=15, fontweight="bold", y=0.97)
    ax.add_patch(Ellipse((0.330, 0.600), 0.450, 0.220, angle=-12, fc=LIGHT_BLUE, ec=BLUE, lw=1.5))
    ax.plot([0.120, 0.555], [0.505, 0.700], color=GREEN, lw=2.0)
    arrow(ax, (0.210, 0.545), (0.475, 0.665), GREEN, scale=16)
    ax.scatter([0.270], [0.570], s=70, color=BLUE, edgecolor="white", zorder=4)
    ax.text(0.330, 0.760, r"measured affine set $\{x:Ax=y\}$", ha="center", fontsize=11, color=BLUE)
    ax.text(0.350, 0.460, r"null / weakly seen directions", ha="center", fontsize=10, color=GREEN)
    rough_image(ax, 0.105, 0.250, w=0.115, h=0.115, label=r"$x_{\rm data}$")
    rounded(ax, (0.220, 0.210), 0.120, 0.080, r"$G_\theta$", fc=LIGHT_ORANGE, ec=ORANGE, fs=11, color=ORANGE)
    draw_pn_filter(ax, 0.380, 0.197, 0.090, 0.115)
    draw_audit(ax, 0.565, 0.250)
    image_icon(ax, 0.710, 0.250, w=0.120, h=0.100, label=r"$\hat{x}$", ec=GREEN, fc=LIGHT_GREEN)
    for p0, p1, color in [
        ((0.162, 0.250), (0.220, 0.250), ORANGE),
        ((0.340, 0.250), (0.380, 0.250), ORANGE),
        ((0.470, 0.250), (0.515, 0.250), GREEN),
        ((0.615, 0.250), (0.650, 0.250), GREEN),
    ]:
        arrow(ax, p0, p1, color)
    ax.text(0.105, 0.130, "directly computed\nfrom buckets", ha="center", fontsize=9, color=BLUE)
    ax.text(0.425, 0.120, "learned residual\nis filtered", ha="center", fontsize=9, color=ORANGE)
    ax.text(0.565, 0.120, "final audit", ha="center", fontsize=9, color=GREEN)
    save(fig, "fig1_variant_B_geometry_pipeline")


def variant_c() -> None:
    fig, ax = base_axes((7.2, 5.0))
    fig.suptitle("Equation decomposition of measurement-consistent reconstruction", fontsize=15, fontweight="bold", y=0.97)
    ax.text(0.500, 0.820, r"$\hat{x}=\Pi_y\left[x_{\rm data}+P_N\left(G_\theta(x_{\rm data},z)\right)\right]$", ha="center", fontsize=18)
    blocks = [
        (0.075, 0.535, 0.220, 0.150, r"$x_{\rm data}$", "measured component\ncomputed from y", LIGHT_BLUE, BLUE),
        (0.325, 0.535, 0.220, 0.150, r"$G_\theta(x_{\rm data},z)$", "neural proposal for\nmissing structure", LIGHT_ORANGE, ORANGE),
        (0.575, 0.535, 0.150, 0.150, r"$P_N$", "residual filter", "#FFF7EA", ORANGE),
        (0.770, 0.535, 0.150, 0.150, r"$\Pi_y$", "bucket audit", LIGHT_GREEN, GREEN),
    ]
    for x, y, w, h, formula, desc, fc, ec in blocks:
        rounded(ax, (x, y), w, h, formula, fc=fc, ec=ec, fs=12, color=ec)
        ax.text(x + w / 2, y - 0.048, desc, ha="center", fontsize=9, color=ec)
    arrow(ax, (0.295, 0.610), (0.325, 0.610), GRAY)
    arrow(ax, (0.545, 0.610), (0.575, 0.610), ORANGE)
    arrow(ax, (0.725, 0.610), (0.770, 0.610), GREEN)
    ax.text(0.500, 0.365, "The equation separates what the buckets determine, what the network proposes,\nand what the final projection audits.", ha="center", fontsize=11, color=GRAY)
    ax.text(0.500, 0.225, r"$A\hat{x}\approx y$", ha="center", fontsize=16, color=GREEN)
    save(fig, "fig1_variant_C_equation_decomposition")


def variant_comparison() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    items = [
        ("Variant A", "Problem -> risk -> solution", "Best for first-time readers:\nexplains why the structure is needed.", GREEN),
        ("Variant B", "Geometry + pipeline", "Best for technical readers:\nshows affine-set intuition.", BLUE),
        ("Variant C", "Equation decomposition", "Best for formula reference:\nclear but less narrative.", ORANGE),
    ]
    for ax, (title, subtitle, desc, color) in zip(axes, items):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        rounded(ax, (0.045, 0.140), 0.910, 0.720, fc="white", ec="#D1D5DB", lw=1.2)
        ax.text(0.500, 0.740, title, ha="center", fontsize=12, fontweight="bold", color=color)
        ax.text(0.500, 0.590, subtitle, ha="center", fontsize=9.5)
        ax.text(0.500, 0.360, desc, ha="center", fontsize=9, color=GRAY)
        if title.endswith("A"):
            ax.add_patch(Rectangle((0.190, 0.205), 0.140, 0.070, fc=LIGHT_BLUE, ec=BLUE))
            ax.add_patch(Rectangle((0.430, 0.205), 0.140, 0.070, fc=LIGHT_RED, ec=RED))
            ax.add_patch(Rectangle((0.670, 0.205), 0.140, 0.070, fc=LIGHT_GREEN, ec=GREEN))
            arrow(ax, (0.330, 0.240), (0.430, 0.240), GRAY, scale=9)
            arrow(ax, (0.570, 0.240), (0.670, 0.240), GRAY, scale=9)
        elif title.endswith("B"):
            ax.add_patch(Ellipse((0.500, 0.240), 0.500, 0.170, fc=LIGHT_BLUE, ec=BLUE))
            ax.plot([0.260, 0.740], [0.195, 0.285], color=GREEN, lw=1.7)
        else:
            ax.text(0.500, 0.235, r"$\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta)]$", ha="center", fontsize=9.5)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.92, bottom=0.08, wspace=0.080)
    save(fig, "fig1_variants_comparison")


def supplement_equation_figure() -> None:
    fig, ax = base_axes((7.2, 4.8))
    fig.suptitle("Algebraic form of the measurement-consistent reconstruction", fontsize=12.6, fontweight="bold", y=0.955)
    rows = [
        (r"$y=Ax+\epsilon$", "Bucket forward model:\nknown patterns produce scalar measurements.", BLUE),
        (r"$x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y$", "Data solution:\nthe directly computable measured component.", BLUE),
        (r"$P_N=I-A^T(AA^T+\lambda I)^{-1}A$", "Approximate null-space filter:\nkeep residuals in weakly measured directions.", ORANGE),
        (r"$\Pi_y(u)=u-A^T(AA^T+\lambda I)^{-1}(Au-y)$", "Projection audit:\npull the completed image back to the measured set.", GREEN),
        (r"$\hat{x}=\Pi_y\left[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))\right]$", "Final reconstruction:\nmeasured part plus learned missing structure, then audit.", GREEN),
    ]
    y = 0.790
    for formula, desc, color in rows:
        rounded(ax, (0.060, y - 0.050), 0.370, 0.087, formula, fc="white", ec=color, fs=9.5, color=color)
        ax.text(0.470, y - 0.006, desc, ha="left", va="center", fontsize=9.2, color=GRAY)
        y -= 0.145
    ax.set_aspect("auto")
    save(fig, "figS_mechanism_equations")


def load_attr() -> list[dict[str, str]]:
    with ATTR_CSV.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def figure4() -> None:
    rows = load_attr()
    by_id = {r["method_id"]: r for r in rows}
    primary_ids = [
        "rademacher5_hq_noise001_colab",
        "scrambled_hadamard5_hq_noise001_colab",
        "rademacher10_full_noise001_colab",
        "scrambled_hadamard10_full_noise001_colab",
    ]
    diag_ids = [
        ("stl10_hadamard5_local_medium", "Lowfreq-5"),
        ("stl10_hadamard10_local_full", "Lowfreq-10"),
    ]
    labels = ["Rad-5", "Scr-5", "Rad-10", "Scr-10"]
    colors = [ORANGE, GREEN, ORANGE, GREEN]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))

    ax = axes[0]
    x = np.arange(len(primary_ids))
    bp = [f(by_id[mid], "backproj_psnr") for mid in primary_ids]
    model = [f(by_id[mid], "model_psnr") for mid in primary_ids]
    ax.bar(x - 0.18, bp, width=0.36, color=LIGHT_BLUE, edgecolor=BLUE, label="GI/BP")
    ax.bar(x + 0.18, model, width=0.36, color=BLUE, label="Ours")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(a) Physical initialization vs final model", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for xi, yi in zip(x + 0.18, model):
        ax.text(xi, yi + 0.35, f"{yi:.1f}", ha="center", fontsize=8)

    ax = axes[1]
    ax.axvspan(0, 10.5, color=LIGHT_ORANGE, alpha=0.55)
    ax.axvspan(10.5, 17.0, color=LIGHT_GREEN, alpha=0.40)
    ax.axvspan(17.0, 23.5, color=LIGHT_GRAY, alpha=0.70)
    ax.text(6.8, 16.4, "weak BP\nlarge gain", ha="center", fontsize=7.6, color=ORANGE)
    ax.text(13.5, 13.3, "stronger BP\nmoderate gain", ha="center", fontsize=8.0, color=GREEN)
    ax.text(19.0, 6.9, "diagnostic\nlowfreq", ha="center", fontsize=8.0, color=GRAY)
    offsets = {
        "Rad-5": (0.28, 0.20),
        "Scr-5": (0.20, -0.60),
        "Rad-10": (0.35, -0.50),
        "Scr-10": (0.20, 0.20),
    }
    for mid, label, color in zip(primary_ids, labels, colors):
        xx = f(by_id[mid], "backproj_psnr")
        yy = f(by_id[mid], "delta_psnr")
        ax.scatter([xx], [yy], s=70, color=color, edgecolor="white", linewidth=0.9, zorder=4)
        dx, dy = offsets[label]
        ax.text(xx + dx, yy + dy, label, fontsize=8.4, color=color)
    diag_offsets = {"Lowfreq-5": (0.35, -0.45), "Lowfreq-10": (0.35, 0.35)}
    for mid, label in diag_ids:
        row = by_id[mid]
        xx = f(row, "backproj_psnr")
        yy = f(row, "delta_psnr")
        ax.scatter([xx], [yy], s=76, facecolors="white", edgecolors=GRAY, linewidth=1.5, zorder=5)
        dx, dy = diag_offsets[label]
        ax.text(xx + dx, yy + dy, f"{label}\n(diagnostic)", fontsize=7.4, color=GRAY)
    ax.set_xlabel("GI/BP PSNR (dB)")
    ax.set_ylabel(r"Neural gain $\Delta$PSNR (dB)")
    ax.set_title("(b) Initialization-gain regime map", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlim(5.5, 21.0)
    ax.set_ylim(3.2, 18.3)
    ax.grid(alpha=0.25)

    fig.text(
        0.5,
        0.035,
        "Final PSNR alone hides whether performance comes from physical initialization, neural refinement, or both.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.080, right=0.985, top=0.875, bottom=0.210, wspace=0.250)
    save(fig, "fig4_measurement_attribution_v33")


def write_decision() -> None:
    text = """# Figure 1 Variant Decision

## Generated variants

- Variant A: Problem -> hallucination risk -> proposed audited reconstruction.
- Variant B: Measurement geometry + reconstruction pipeline.
- Variant C: Central equation decomposition.

## Pros and cons

| Variant | Strength | Limitation | Decision |
|---|---|---|---|
| A | Best explains why the method needs data solution, residual filtering, and final audit. | Less compact than a pure equation diagram. | Selected for main Figure 1. |
| B | Gives useful geometric intuition about the measurement affine set. | More abstract for non-specialist readers. | Keep as alternative. |
| C | Makes the central equation explicit. | Too formula-heavy for first exposure. | Move the equation emphasis to the Supplement. |

## Final choice

Variant A is selected as the main Figure 1 because it makes the problem-risk-solution story explicit: low-sampling buckets are ambiguous, unconstrained neural inversion can drift, and the proposed method inserts learned missing structure while auditing the result against the measurements.
"""
    (OUT / "FIGURE1_VARIANT_DECISION.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    setup()
    write_storyboard()
    variant_a()
    variant_b()
    variant_c()
    variant_comparison()
    variant_a("fig1_mechanism_final_v33")
    supplement_equation_figure()
    figure4()
    write_decision()
    print(
        {
            "output_dir": str(OUT),
            "final_figure1": str(FIG_DIR / "fig1_mechanism_final_v33.pdf"),
            "figure4": str(FIG_DIR / "fig4_measurement_attribution_v33.pdf"),
        }
    )


if __name__ == "__main__":
    main()
