from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from .phase18b_common import BLUE, GREEN, LIGHT_BLUE, LIGHT_GREEN, LIGHT_ORANGE, ORANGE, OUT, RED, save_figure, setup_matplotlib


def arrow(ax, p0, p1, color="#555555", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9.5, lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def box(ax, xy, w, h, text, fc, ec, fs=7.5, rounded=True):
    if rounded:
        patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.018,rounding_size=0.025", fc=fc, ec=ec, lw=1.1)
    else:
        patch = Rectangle(xy, w, h, fc=fc, ec=ec, lw=1.1)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def mini_image(kind: str, n: int = 26) -> np.ndarray:
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    if kind == "gt":
        return np.clip(0.55 + 0.35 * np.exp(-((X + 0.25) ** 2 + (Y + 0.05) ** 2) / 0.28) - 0.25 * (X > 0.35), 0, 1)
    if kind == "bp":
        rng = np.random.default_rng(5)
        return np.clip(0.35 + 0.22 * rng.standard_normal((n, n)) + 0.12 * np.sin(10 * X), 0, 1)
    return np.clip(0.50 + 0.30 * np.exp(-((X + 0.2) ** 2 + (Y + 0.05) ** 2) / 0.36) - 0.18 * (X > 0.42), 0, 1)


def main() -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.55))
    labels = ["a", "b", "c", "d", "e"]
    titles = ["Acquisition", "Underdetermined set", "Data + residual", "Physics projections", "Output"]
    for ax, lab, title in zip(axes, labels, titles):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.0, 1.03, lab, transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=8.3, pad=8)

    ax = axes[0]
    for i, x0 in enumerate([0.06, 0.12, 0.18]):
        arr = np.indices((8, 8)).sum(0) % 2 if i == 0 else np.random.default_rng(i).choice([0, 1], size=(8, 8))
        ax.imshow(arr, cmap="gray", extent=(x0, x0 + 0.22, 0.58 - i * 0.04, 0.78 - i * 0.04), interpolation="nearest", zorder=1)
        ax.add_patch(Rectangle((x0, 0.58 - i * 0.04), 0.22, 0.20, fc="none", ec=BLUE, lw=0.8, zorder=2))
    ax.imshow(mini_image("gt"), cmap="gray", extent=(0.43, 0.67, 0.55, 0.79), interpolation="bicubic")
    ax.add_patch(Rectangle((0.43, 0.55), 0.24, 0.24, fc="none", ec="#333333", lw=0.9))
    ax.add_patch(Circle((0.86, 0.67), 0.07, fc=LIGHT_BLUE, ec=BLUE, lw=1.1))
    ax.text(0.86, 0.67, "$y_i$", ha="center", va="center", fontsize=8)
    arrow(ax, (0.30, 0.68), (0.42, 0.68), BLUE)
    arrow(ax, (0.68, 0.67), (0.78, 0.67), BLUE)
    ax.text(0.5, 0.24, r"$y_i=\langle a_i,x\rangle+\epsilon_i$", ha="center", fontsize=7.3)

    ax = axes[1]
    ax.add_patch(Ellipse((0.50, 0.56), 0.72, 0.35, angle=-15, fc=LIGHT_BLUE, ec=BLUE, lw=1.1))
    ax.plot([0.18, 0.82], [0.35, 0.74], color=GREEN, lw=1.4)
    arrow(ax, (0.25, 0.39), (0.75, 0.70), GREEN)
    ax.scatter([0.40, 0.60], [0.52, 0.59], s=[22, 22], color=[BLUE, ORANGE], zorder=3)
    ax.text(0.50, 0.23, r"$\mathcal{C}_y=\{x:Ax=y\}$", ha="center", fontsize=7.2)
    ax.text(0.50, 0.13, r"$m\ll n,\quad v\in\mathrm{Null}(A)$", ha="center", fontsize=7.2)

    ax = axes[2]
    box(ax, (0.08, 0.60), 0.36, 0.20, r"$x_{data}$", LIGHT_BLUE, BLUE)
    box(ax, (0.57, 0.60), 0.34, 0.20, r"$G_\theta$", LIGHT_ORANGE, ORANGE)
    box(ax, (0.32, 0.25), 0.40, 0.18, r"$r_\theta$", LIGHT_ORANGE, ORANGE)
    arrow(ax, (0.44, 0.70), (0.57, 0.70), ORANGE)
    arrow(ax, (0.75, 0.60), (0.55, 0.43), ORANGE)
    ax.text(0.50, 0.12, r"$x_{data}=A^T(AA^T+\lambda I)^{-1}y$", ha="center", fontsize=6.5)

    ax = axes[3]
    box(ax, (0.08, 0.62), 0.34, 0.19, r"$P_N$", LIGHT_GREEN, GREEN)
    box(ax, (0.58, 0.62), 0.34, 0.19, r"$\Pi_y$", LIGHT_GREEN, GREEN)
    box(ax, (0.26, 0.29), 0.47, 0.18, r"$\tilde{x}\rightarrow\hat{x}$", "#F4F4F4", "#555555")
    arrow(ax, (0.42, 0.72), (0.58, 0.72), GREEN)
    arrow(ax, (0.75, 0.62), (0.58, 0.47), GREEN)
    arrow(ax, (0.58, 0.30), (0.83, 0.30), RED, lw=1.0)
    ax.text(0.86, 0.30, r"$A\hat{x}$ vs $y$", ha="center", va="center", fontsize=6.7, color=RED)
    ax.text(0.50, 0.12, r"$P_N$ completes; $\Pi_y$ restores consistency", ha="center", fontsize=6.7)

    ax = axes[4]
    for i, (kind, title) in enumerate([("gt", "GT"), ("bp", "BP"), ("recon", "Recon")]):
        x0 = 0.08 + i * 0.28
        ax.imshow(mini_image(kind), cmap="gray", extent=(x0, x0 + 0.22, 0.54, 0.76), interpolation="bicubic")
        ax.add_patch(Rectangle((x0, 0.54), 0.22, 0.22, fc="none", ec="#333333", lw=0.8))
        ax.text(x0 + 0.11, 0.49, title, ha="center", fontsize=6.9)
    ax.text(0.50, 0.27, r"$A\hat{x}\approx y$", ha="center", fontsize=8.2, color=GREEN)
    ax.text(0.50, 0.14, "RelMeasErr tracks physics", ha="center", fontsize=6.8)

    fig.text(
        0.5,
        0.01,
        "Physics-consistent null-space neural reconstruction: row-space data are preserved, residual structure is completed, and measurements are rechecked.",
        ha="center",
        fontsize=7.4,
        color="#444444",
    )
    fig.subplots_adjust(left=0.02, right=0.995, top=0.84, bottom=0.16, wspace=0.06)
    save_figure(fig, "fig1_mechanism")
    print({"figure": str(OUT / "figures" / "fig1_mechanism.pdf")})


if __name__ == "__main__":
    main()
