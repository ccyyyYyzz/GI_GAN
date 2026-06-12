from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .phase18b_common import BLUE, GREEN, METHOD_LABEL, OUT, RED, as_float, registry_by_id, save_figure, setup_matplotlib


def plot_metric_pair(ax_psnr, ax_ssim, mids: list[str], title: str, psnr_thr: float, ssim_thr: float, colors: list[str]) -> None:
    reg = registry_by_id()
    x = np.arange(len(mids))
    psnr = [as_float(reg[mid]["psnr"]) for mid in mids]
    ssim = [as_float(reg[mid]["ssim"]) for mid in mids]
    ax_psnr.bar(x, psnr, color=colors[: len(mids)], width=0.62)
    ax_psnr.axhline(psnr_thr, color=RED, lw=1.0, ls="--", label="threshold")
    for xi, yi in zip(x, psnr):
        ax_psnr.text(xi, yi + 0.22, f"{yi:.2f}", ha="center", fontsize=7.4)
    ax_psnr.set_title(title)
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_xticks(x)
    ax_psnr.set_xticklabels([])
    ax_psnr.grid(axis="y", alpha=0.25)

    ax_ssim.bar(x, ssim, color=colors[: len(mids)], width=0.62)
    ax_ssim.axhline(ssim_thr, color=RED, lw=1.0, ls="--")
    for xi, yi in zip(x, ssim):
        ax_ssim.text(xi, yi + 0.013, f"{yi:.3f}", ha="center", fontsize=7.4)
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.set_ylim(0, max(1.0, max(ssim) + 0.08))
    ax_ssim.set_xticks(x)
    ax_ssim.set_xticklabels([METHOD_LABEL[mid] for mid in mids], fontsize=8)
    ax_ssim.grid(axis="y", alpha=0.25)


def main() -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), sharey="row")
    groups = [
        (["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab"], "STL-10 5%", 20.0, 0.60, [BLUE, GREEN]),
        (["rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"], "STL-10 10%", 22.0, 0.65, [BLUE, GREEN]),
        (["mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"], "Simple domains 5%", 25.0, 0.80, ["#6A8DAD", "#9678B6"]),
    ]
    for i, (mids, title, psnr_thr, ssim_thr, colors) in enumerate(groups):
        plot_metric_pair(axes[0, i], axes[1, i], mids, title, psnr_thr, ssim_thr, colors)
        axes[0, i].text(-0.18, 1.08, chr(ord("a") + i), transform=axes[0, i].transAxes, fontsize=10, fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=7.2)
    fig.text(0.5, 0.01, "Dashed lines are internal engineering thresholds, not theoretical limits.", ha="center", fontsize=7.6, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.13, wspace=0.22, hspace=0.12)
    save_figure(fig, "fig2_main_metrics")
    print({"figure": str(OUT / "figures" / "fig2_main_metrics.pdf")})


if __name__ == "__main__":
    main()
