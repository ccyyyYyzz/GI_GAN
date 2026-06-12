from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .phase18b_common import BLUE, GREEN, METHOD_LABEL, ORANGE, OUT, RED, STL_METHODS, as_float, save_figure, setup_matplotlib, table


MODES = [
    ("full_model", "Full"),
    ("no_dc_project", "-DC"),
    ("no_null_project", "-Null"),
    ("stage1_only", "Stage1"),
    ("raw_weights", "Raw"),
    ("ema_weights", "EMA"),
]


def main() -> None:
    setup_matplotlib()
    rows = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=True)
    colors = [BLUE, RED, "#B6BBC2", ORANGE, "#9A8DB8", GREEN]
    for idx, (ax, mid) in enumerate(zip(axes.ravel(), STL_METHODS)):
        vals = [as_float(rows[(mid, mode)]["psnr"]) for mode, _ in MODES]
        xs = np.arange(len(MODES))
        ax.bar(xs, vals, color=colors, width=0.68)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.28, f"{v:.1f}", ha="center", fontsize=6.8)
        ax.set_title(METHOD_LABEL[mid])
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _mode, lab in MODES])
        if idx % 2 == 0:
            ax.set_ylabel("PSNR (dB)")
        ax.grid(axis="y", alpha=0.25)
        ax.text(-0.12, 1.05, chr(ord("a") + idx), transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.text(0.5, 0.015, "The -DC setting removes measurement-consistency projection and causes the largest degradation.", ha="center", fontsize=7.4, color="#444444")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12, wspace=0.12, hspace=0.30)
    save_figure(fig, "fig5_inference_ablation")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=False)
    for idx, (ax, mid) in enumerate(zip(axes.ravel(), STL_METHODS)):
        vals = [as_float(rows[(mid, mode)]["rel_meas_err"]) for mode, _ in MODES]
        xs = np.arange(len(MODES))
        ax.bar(xs, vals, color=colors, width=0.68)
        ax.set_title(METHOD_LABEL[mid])
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _mode, lab in MODES])
        if idx % 2 == 0:
            ax.set_ylabel("RelMeasErr")
        ax.grid(axis="y", alpha=0.25)
        ax.text(-0.12, 1.05, chr(ord("a") + idx), transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.10, wspace=0.20, hspace=0.30)
    save_figure(fig, "figS_ablation_relmeaserr")
    print({"figure": str(OUT / "figures" / "fig5_inference_ablation.pdf"), "supplement": str(OUT / "figures" / "figS_ablation_relmeaserr.pdf")})


if __name__ == "__main__":
    main()
