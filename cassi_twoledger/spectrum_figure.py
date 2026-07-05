# -*- coding: utf-8 -*-
"""The headline CASSI figure: the NON-UNIFORM singular spectrum sigma_j = sqrt(Phi_s) and
its per-mode audit contraction lambda/(lambda+sigma_j^2) -- the empirical face of the modal
contraction theorem that masked-Fourier MRI (all sigma_i = 1) cannot show.

(a) sigma over the detector plane [H, Ws]: a wavelength-shear envelope modulated by the coded
    aperture -> spatially structured, not flat.
(b) histogram of sigma_j (CASSI, 79k measured modes) vs MRI's single spike at sigma = 1.
(c) per-mode contraction curve lambda/(lambda+sigma^2): CASSI spreads it over 4 orders of
    magnitude; MRI collapses to one point.
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cassi_operator import CASSI, load_mask

OUT = os.path.dirname(os.path.abspath(__file__))
MASK = os.path.join(OUT, "mask.mat")
C_RED = "#c0392b"; C_BLUE = "#3a6ea5"; C_GOLD = "#e0a500"


def main(lam=1e-3):
    op = CASSI(load_mask(MASK), nC=28, step=2)
    Phi_s = op.Phi_s.cpu().numpy()
    sigma_map = np.sqrt(Phi_s)
    s = op.singular_values().cpu().numpy()               # measured modes only
    smin, smax, smed = s.min(), s.max(), np.median(s)

    fig = plt.figure(figsize=(14, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05], wspace=0.30)

    # (a) sigma over the detector plane
    axa = fig.add_subplot(gs[0])
    im = axa.imshow(sigma_map, cmap="viridis", aspect="auto")
    axa.set_title(r"(a) $\sigma$ across the detector: structured, not flat", fontsize=10)
    axa.set_xlabel("detector column (spectral shear)"); axa.set_ylabel("detector row")
    cb = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.03); cb.set_label(r"$\sigma_j=\sqrt{\Phi_{s,j}}$", fontsize=9)

    # (b) histogram of sigma vs MRI's single value
    axb = fig.add_subplot(gs[1])
    axb.hist(s, bins=80, color=C_BLUE, alpha=0.85, density=True)
    axb.axvline(1.0, color=C_RED, lw=2.2, label="MRI: every mode $\\sigma_i=1$")
    axb.axvline(smed, color=C_GOLD, lw=1.6, ls="--", label=f"CASSI median {smed:.2f}")
    axb.set_xlabel(r"singular value $\sigma_j$"); axb.set_ylabel("density")
    axb.set_title(f"(b) CASSI spectrum spans {smax/smin:.0f}$\\times$\n"
                  f"$\\sigma\\in[{smin:.2f},{smax:.2f}]$ ({s.size//1000}k modes)", fontsize=10)
    axb.legend(fontsize=8)
    for sp in ("top", "right"): axb.spines[sp].set_visible(False)

    # (c) per-mode contraction lambda/(lambda+sigma^2)
    axc = fig.add_subplot(gs[2])
    grid = np.linspace(smin, smax, 400)
    axc.plot(grid, lam / (lam + grid ** 2), color="#333", lw=2)
    # shade the actual CASSI sigma range and mark quartiles
    qs = np.percentile(s, [5, 50, 95])
    for q, c in zip(qs, [C_BLUE, C_GOLD, C_BLUE]):
        axc.plot([q], [lam / (lam + q ** 2)], "o", color=c, ms=7, zorder=5)
    axc.axvspan(smin, smax, color=C_BLUE, alpha=0.08)
    # MRI single point
    axc.plot([1.0], [lam / (lam + 1.0)], "s", color=C_RED, ms=9, zorder=6, label="MRI (single point)")
    axc.annotate("MRI: one\ncontraction", (1.0, lam / (lam + 1.0)), textcoords="offset points",
                 xytext=(12, 6), fontsize=8, color=C_RED)
    cmin, cmax = lam / (lam + smax ** 2), lam / (lam + smin ** 2)
    axc.set_yscale("log")
    axc.set_xlabel(r"singular value $\sigma_j$")
    axc.set_ylabel(r"audit contraction $\lambda/(\lambda+\sigma_j^2)$")
    axc.set_title(f"(c) per-mode contraction spans\n{cmax/cmin:.0e}$\\times$ ($\\lambda={lam:g}$)", fontsize=10)
    axc.legend(fontsize=8, loc="upper right")
    for sp in ("top", "right"): axc.spines[sp].set_visible(False)

    fig.suptitle("CASSI: a genuinely non-uniform singular spectrum makes the modal-contraction "
                 "theorem visible (MRI's is a single point)", fontsize=11.5, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "CASSI_SPECTRUM." + ext), dpi=170, bbox_inches="tight")
    print(f"wrote CASSI_SPECTRUM.* | sigma in [{smin:.3f},{smax:.3f}] median {smed:.3f} span {smax/smin:.0f}x")
    print(f"per-mode contraction at lambda={lam}: [{cmin:.2e}, {cmax:.2e}] span {cmax/cmin:.0e}x "
          f"(MRI single value {lam/(lam+1):.2e})")


if __name__ == "__main__":
    main()
