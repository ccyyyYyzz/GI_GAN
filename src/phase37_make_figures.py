from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Circle


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase37_author_guided_rewrite"
FIG_DIR = OUT / "figures"

BLUE = "#1f77b4"
GREEN = "#238b45"
ORANGE = "#d97904"
GRAY = "#5f6368"
DARK = "#202124"
LIGHT = "#f8fbff"


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=240)
    plt.close(fig)


def block(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    formula_lines: list[str],
    subtitle: str,
    color: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.8,
            edgecolor=color,
            facecolor=LIGHT,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.068,
        title,
        ha="center",
        va="top",
        fontsize=12.3,
        weight="bold",
        color=DARK,
        linespacing=0.96,
    )
    center_y = y + h * 0.54
    if len(formula_lines) == 1:
        ys = [center_y]
        fs = 14.0
    else:
        ys = [center_y + 0.040, center_y - 0.040]
        fs = 12.7
    for line, yy in zip(formula_lines, ys):
        ax.text(x + w / 2, yy, line, ha="center", va="center", fontsize=fs, color=DARK)
    ax.text(x + w / 2, y + 0.082, subtitle, ha="center", va="center", fontsize=10.6, color=color)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = GRAY, lw: float = 2.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def filter_icon(ax: plt.Axes, cx: float, cy: float, scale: float = 1.0) -> None:
    w = 0.050 * scale
    h = 0.085 * scale
    points = [
        (cx - w, cy + h),
        (cx + w, cy + h),
        (cx + w * 0.38, cy + h * 0.08),
        (cx + w * 0.20, cy - h),
        (cx - w * 0.20, cy - h),
        (cx - w * 0.38, cy + h * 0.08),
    ]
    ax.add_patch(Polygon(points, closed=True, facecolor="#fff7ed", edgecolor=ORANGE, linewidth=2.0))
    for off in (-0.022, 0.0, 0.022):
        ax.plot([cx + off, cx + off], [cy + h * 0.55, cy - h * 0.62], color=ORANGE, lw=1.4)


def audit_icon(ax: plt.Axes, cx: float, cy: float, scale: float = 1.0) -> None:
    r = 0.046 * scale
    ax.add_patch(Circle((cx, cy), r, facecolor="#effaf2", edgecolor=GREEN, linewidth=2.0))
    ax.plot([cx - r * 0.50, cx - r * 0.12, cx + r * 0.58], [cy - r * 0.02, cy - r * 0.45, cy + r * 0.50], color=GREEN, lw=3.0)


def draw_fig1() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(12.2, 4.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.965,
        "From GI correlation to measurement-audited neural completion",
        ha="center",
        va="top",
        fontsize=17.0,
        weight="bold",
        color=DARK,
    )

    y = 0.315
    h = 0.475
    w = 0.292
    xs = [0.026, 0.354, 0.682]
    block(
        ax,
        xs[0],
        y,
        w,
        h,
        "Conventional GI correlation",
        [r"$\hat{x}_{\rm GI}=A^Ty=\sum_i y_i a_i$"],
        r"raw bucket weights $y_i$",
        BLUE,
    )
    block(
        ax,
        xs[1],
        y,
        w,
        h,
        "Regularized data solution",
        [r"$q=(AA^T+\lambda I)^{-1}y$", r"$x_{\rm data}=A^Tq$"],
        r"decorrelated bucket weights $q_i$",
        BLUE,
    )
    block(
        ax,
        xs[2],
        y,
        w,
        h,
        "Measurement-audited\nneural completion",
        [r"$r_\theta=G_\theta(x_{\rm data})$", r"$\hat{x}=\Pi_y[x_{\rm data}+P_N(r_\theta)]$"],
        r"candidate residual $\rightarrow$ residual filter $\rightarrow$ bucket audit",
        GREEN,
    )

    arrow(ax, (xs[0] + w + 0.016, y + h / 2), (xs[1] - 0.016, y + h / 2))
    arrow(ax, (xs[1] + w + 0.016, y + h / 2), (xs[2] - 0.016, y + h / 2))

    filter_icon(ax, xs[2] + 0.105, y + 0.155, 0.52)
    audit_icon(ax, xs[2] + 0.182, y + 0.155, 0.52)
    arrow(ax, (xs[2] + 0.122, y + 0.155), (xs[2] + 0.160, y + 0.155), color=ORANGE, lw=1.45)

    chain_box = FancyBboxPatch(
        (0.125, 0.095),
        0.750,
        0.125,
        boxstyle="round,pad=0.018,rounding_size=0.020",
        linewidth=1.4,
        edgecolor="#c7ced8",
        facecolor="#ffffff",
    )
    ax.add_patch(chain_box)
    ax.text(
        0.5,
        0.157,
        r"$A^Ty\;\rightarrow\;A^T(AA^T+\lambda I)^{-1}y\;\rightarrow\;\Pi_y[x_{\rm data}+P_N(G_\theta)]$",
        ha="center",
        va="center",
        fontsize=15.0,
        color=DARK,
    )

    fig.subplots_adjust(left=0.012, right=0.988, top=0.960, bottom=0.055)
    save(fig, "fig1_minimal_mechanism_v37")


def main() -> None:
    draw_fig1()
    print({"figure1": str(FIG_DIR / "fig1_minimal_mechanism_v37.pdf")})


if __name__ == "__main__":
    main()
