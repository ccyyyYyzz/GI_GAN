from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .phase18b_common import BLUE, GREEN, LIGHT_BLUE, LIGHT_GREEN, LIGHT_ORANGE, METHOD_LABEL, ORANGE, OUT, as_float, save_figure, setup_matplotlib, table


def pattern(kind: str, n: int = 42) -> np.ndarray:
    rng = np.random.default_rng(18)
    if kind == "rademacher":
        return rng.choice([-1.0, 1.0], size=(n, n))
    x = np.arange(n)
    y = np.arange(n)[:, None]
    if kind == "lowfreq":
        return np.cos(2 * np.pi * x / n) + np.cos(2 * np.pi * y / n)
    return (((x[None, :] * 7 + y * 11) % 29) < 14).astype(float) * 2 - 1


def box(ax, xy, w, h, text, fc, ec):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.018,rounding_size=0.04", fc=fc, ec=ec, lw=1.1)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8)


def main() -> None:
    setup_matplotlib()
    rows = {r["method_id"]: r for r in table("attribution")}
    mids = [
        "rademacher5_hq_noise001_colab",
        "scrambled_hadamard5_hq_noise001_colab",
        "rademacher10_full_noise001_colab",
        "scrambled_hadamard10_full_noise001_colab",
    ]
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.9, 1.2], width_ratios=[1, 1, 1, 1.25], hspace=0.42, wspace=0.32)

    for i, (kind, title) in enumerate([("rademacher", "Rademacher"), ("scrambled", "Scrambled\nHadamard"), ("lowfreq", "Low-frequency\nHadamard")]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(pattern(kind), cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8.2)
        ax.text(-0.12, 1.08, chr(ord("a") + i), transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.text(-0.04, 1.08, "d", transform=ax.transAxes, fontsize=10, fontweight="bold")
    box(ax, (0.04, 0.64), 0.38, 0.20, "weak BP", LIGHT_BLUE, BLUE)
    box(ax, (0.58, 0.64), 0.36, 0.20, "large\ngain", LIGHT_ORANGE, ORANGE)
    ax.add_patch(FancyArrowPatch((0.42, 0.74), (0.58, 0.74), arrowstyle="-|>", mutation_scale=10, color=ORANGE, lw=1.2))
    ax.text(0.49, 0.52, "Rademacher", ha="center", fontsize=8.2)
    box(ax, (0.04, 0.22), 0.38, 0.20, "stronger\nBP", LIGHT_GREEN, GREEN)
    box(ax, (0.58, 0.22), 0.36, 0.20, "similar\nfinal", LIGHT_BLUE, BLUE)
    ax.add_patch(FancyArrowPatch((0.42, 0.32), (0.58, 0.32), arrowstyle="-|>", mutation_scale=10, color=GREEN, lw=1.2))
    ax.text(0.49, 0.08, "Scrambled Hadamard", ha="center", fontsize=8.2)

    ax = fig.add_subplot(gs[1, :3])
    x = np.arange(len(mids))
    bp = [as_float(rows[mid]["backproj_psnr"]) for mid in mids]
    model = [as_float(rows[mid]["model_psnr"]) for mid in mids]
    ax.bar(x - 0.18, bp, width=0.36, color="#A9BFCE", label="Backprojection")
    ax.bar(x + 0.18, model, width=0.36, color=BLUE, label="Model")
    for xi, yi in zip(x - 0.18, bp):
        ax.text(xi, yi + 0.25, f"{yi:.1f}", ha="center", fontsize=7)
    for xi, yi in zip(x + 0.18, model):
        ax.text(xi, yi + 0.25, f"{yi:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids])
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Backprojection vs final model")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.10, 1.05, "e", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = fig.add_subplot(gs[1, 3])
    delta = [as_float(rows[mid]["delta_psnr"]) for mid in mids]
    ax.bar(np.arange(len(mids)), delta, color=[ORANGE, GREEN, ORANGE, GREEN])
    for xi, yi in enumerate(delta):
        ax.text(xi, yi + 0.25, f"{yi:.1f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(mids)))
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids], rotation=0)
    ax.set_ylabel(r"$\Delta$PSNR (dB)")
    ax.set_title("Neural gain")
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.22, 1.05, "f", transform=ax.transAxes, fontsize=10, fontweight="bold")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.08)
    save_figure(fig, "fig3_measurement_attribution")
    print({"figure": str(OUT / "figures" / "fig3_measurement_attribution.pdf")})


if __name__ == "__main__":
    main()
