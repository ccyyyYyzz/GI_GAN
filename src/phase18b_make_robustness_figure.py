from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .phase18b_common import BLUE, GREEN, METHOD_LABEL, METHOD_ORDER, ORANGE, OUT, STL_METHODS, as_float, registry_by_id, save_figure, setup_matplotlib, table


def main() -> None:
    setup_matplotlib()
    noise = table("noise")
    perturb = table("perturbation")
    baseline = table("baseline")
    stats = table("statistics")
    reg = registry_by_id()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

    ax = axes[0, 0]
    palette = [BLUE, GREEN, "#6A8DAD", "#7FA67B"]
    for color, mid in zip(palette, STL_METHODS):
        sub = sorted([r for r in noise if r["method_id"] == mid], key=lambda r: as_float(r["noise_std"]))
        ax.plot([as_float(r["noise_std"]) for r in sub], [as_float(r["psnr"]) for r in sub], marker="o", lw=1.4, color=color, label=METHOD_LABEL[mid])
    ax.set_xlabel("Noise std")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Finite noise sweep")
    ax.legend(frameon=False, ncol=2, fontsize=6.8)
    ax.grid(alpha=0.25)
    ax.text(-0.13, 1.06, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = axes[0, 1]
    modes = ["shuffle_coefficients", "wrong_sample"]
    mids = STL_METHODS[:2]
    x = np.arange(len(mids))
    width = 0.34
    for i, mode in enumerate(modes):
        vals = []
        for mid in mids:
            row = next(r for r in perturb if r["method_id"] == mid and r["perturbation_mode"] == mode)
            vals.append(as_float(row["psnr_drop_from_normal"]))
        ax.bar(x + (i - 0.5) * width, vals, width=width, label="Shuffle" if i == 0 else "Wrong-y", color=ORANGE if i == 0 else "#C45A56")
        for xi, yi in zip(x + (i - 0.5) * width, vals):
            ax.text(xi, yi + 0.25, f"{yi:.1f}", ha="center", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids])
    ax.set_ylabel("PSNR drop (dB)")
    ax.set_title("Measurement perturbation")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = axes[1, 0]
    best_tv = {}
    for mid in METHOD_ORDER:
        sub = [r for r in baseline if r["method_id"] == mid and r["baseline"] == "tv_pgd"]
        if sub:
            best_tv[mid] = max(sub, key=lambda r: as_float(r["psnr"]))
    mids = [mid for mid in METHOD_ORDER if mid in best_tv]
    x = np.arange(len(mids))
    ax.bar(x - 0.18, [as_float(reg[mid]["psnr"]) for mid in mids], width=0.36, color=BLUE, label="Ours")
    ax.bar(x + 0.18, [as_float(best_tv[mid]["psnr"]) for mid in mids], width=0.36, color="#9B8A4E", label="CS-TV")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in mids], fontsize=7)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Ours vs CS-TV (PGD)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "c", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = axes[1, 1]
    order = [mid for mid in METHOD_ORDER if any(r["method_id"] == mid for r in stats)]
    stat_by = {r["method_id"]: r for r in stats}
    x = np.arange(len(order))
    means = np.array([as_float(stat_by[mid]["mean_psnr"]) for mid in order])
    lo = np.array([as_float(stat_by[mid]["ci95_psnr_low"]) for mid in order])
    hi = np.array([as_float(stat_by[mid]["ci95_psnr_high"]) for mid in order])
    ax.errorbar(x, means, yerr=np.vstack([means - lo, hi - means]), fmt="o", color=GREEN, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[mid] for mid in order], fontsize=7)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Bootstrap 95% CI")
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.13, 1.06, "d", transform=ax.transAxes, fontsize=10, fontweight="bold")

    fig.text(0.5, 0.012, "CS-TV is TV-regularized compressed sensing solved by PGD on a lightweight small subset.", ha="center", fontsize=7.3, color="#444444")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.12, wspace=0.28, hspace=0.42)
    save_figure(fig, "fig6_robustness_baselines")
    print({"figure": str(OUT / "figures" / "fig6_robustness_baselines.pdf")})


if __name__ == "__main__":
    main()
