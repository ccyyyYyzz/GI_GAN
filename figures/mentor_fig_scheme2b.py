#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheme 2B (radial binary optimization) mechanism figure -- ENGLISH.

Self-contained (numpy + matplotlib only). Physics-faithful: dual-focus
target, binary ring vector, concentric binary zone-plate mask, and the
target-driven inverse-design optimization loop.

Adapted (and relabeled to English) from fig5_method_b in
make_proposal_figures.py. Parameters / formulas are reused verbatim:
    lambda = 532 nm, R = 2 mm, f0 = 150 mm, f1 = 135 mm, f2 = 165 mm.

Output (PDF + PNG): mfig_scheme2b_mechanism.{pdf,png}
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# English / mathtext only -- no CJK font needed.
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ---------------------------------------------------------------------------
# Physical parameters (mm) -- identical to canonical source.
# ---------------------------------------------------------------------------
LAM = 532e-6        # wavelength in mm (532 nm)
R = 2.0             # outer radius mm
F0 = 150.0          # nominal focal length mm
F1, F2 = 135.0, 165.0

# colour palette (colorblind-friendly, muted)
DARK = (0.16, 0.18, 0.20)
ACC = "#1f6feb"     # blue
RED = "#d1495b"     # muted red
GRN = "#2a9d5c"     # green


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name + ".pdf / .png")


def fig_scheme2b():
    rng = np.random.default_rng(7)

    # ---- axial profiles --------------------------------------------------
    z = np.linspace(120, 180, 600)
    sig = 3.5
    tgt = (np.exp(-(z - F1) ** 2 / (2 * sig**2))
           + np.exp(-(z - F2) ** 2 / (2 * sig**2)))
    tgt /= tgt.max()
    sim = (0.92 * np.exp(-(z - F1 - 0.8) ** 2 / (2 * (sig + 0.4) ** 2))
           + 0.9 * np.exp(-(z - F2 + 0.7) ** 2 / (2 * (sig + 0.6) ** 2))
           + 0.09 * np.exp(-(z - 150) ** 2 / (2 * 6.0**2)))
    sim += 0.012 * rng.standard_normal(z.size)
    sim = np.clip(sim, 0, None)
    sim /= sim.max()

    # ---- binary ring vector ---------------------------------------------
    M = 36
    m = np.arange(M)
    a = ((np.floor(np.sqrt(m + 0.5)) % 2) == 0).astype(int)
    for fl in (5, 12, 23, 30):
        a[fl] ^= 1

    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.6))
    axA, axB, axC, axD = axs.ravel()

    # =====================================================================
    # (a) target vs simulated axial intensity
    # =====================================================================
    axA.fill_between(z, 0, tgt, color=RED, alpha=0.12)
    axA.plot(z, tgt, color=RED, lw=2.3, label=r"$I_{target}(z)$")
    axA.plot(z, sim, color=ACC, lw=1.7, ls="--", label=r"simulated $I(z)$")
    for pk in (F1, F2):
        axA.axvline(pk, ls=":", lw=0.8, color="#aab")
        axA.text(pk, 1.07, f"{pk:.0f}", ha="center", fontsize=8.5,
                 color="#666")
    axA.set_xlim(120, 180)
    axA.set_ylim(0, 1.18)
    axA.set_xlabel("axial position z (mm)")
    axA.set_ylabel("normalized intensity")
    axA.set_title("(a) Target vs. simulated axial intensity",
                  fontsize=11, loc="left")
    axA.legend(fontsize=9.5, frameon=False, loc="upper center")
    axA.spines[["top", "right"]].set_visible(False)

    # =====================================================================
    # (b) binary ring variables a_m in {0,1}
    # =====================================================================
    axB.set_xlim(-0.5, M + 0.5)
    axB.set_ylim(-0.85, 1.75)
    for i, v in enumerate(a):
        axB.add_patch(Rectangle((i, 0), 1, 1,
                                facecolor=("#f1f3f5" if v else DARK),
                                edgecolor="#9aa0a8", lw=0.5))
    for i in range(0, M + 1, 6):
        axB.text(i, -0.14, str(i), ha="center", va="top", fontsize=7.5,
                 color="#666")
    axB.text(0.0, 1.40, "ring index $m$", fontsize=8.5, color="#666")
    # small legend swatches for open / blocked
    axB.add_patch(Rectangle((0.0, 1.05), 1.0, 0.22, facecolor="#f1f3f5",
                            edgecolor="#9aa0a8", lw=0.5))
    axB.text(1.4, 1.16, "1 = open", fontsize=9, color="#333", va="center")
    axB.add_patch(Rectangle((11.0, 1.05), 1.0, 0.22, facecolor=DARK,
                            edgecolor="#9aa0a8", lw=0.5))
    axB.text(12.4, 1.16, "0 = blocked", fontsize=9, color="#333",
             va="center")
    axB.set_title(r"(b) Binary ring variables $a_m\in\{0,1\}$",
                  fontsize=11, loc="left")
    axB.axis("off")

    # =====================================================================
    # (c) generated circular binary zone-plate mask (outer->inner z-order)
    # =====================================================================
    axC.set_aspect("equal")
    axC.set_xlim(-1.08, 1.08)
    axC.set_ylim(-1.08, 1.08)
    redges = np.linspace(0, 1, M + 1)
    for i in range(M - 1, -1, -1):
        col = "#f1f3f5" if a[i] else DARK
        axC.add_patch(Circle((0, 0), redges[i + 1], facecolor=col,
                             edgecolor="none", zorder=(M - i)))
    axC.add_patch(Circle((0, 0), 1.0, fill=False, ec="#111", lw=1.6,
                        zorder=M + 1))
    axC.set_title("(c) Generated binary zone-plate mask",
                  fontsize=11, loc="left")
    axC.axis("off")

    # =====================================================================
    # (d) target-driven optimization loop
    # =====================================================================
    axD.set_xlim(0, 10)
    axD.set_ylim(0, 10)
    axD.axis("off")
    axD.set_title("(d) Target-driven optimization loop",
                  fontsize=11, loc="left")
    steps = [r"target axial intensity $I_{target}(z)$",
             r"initialize binary mask $a$",
             "Fresnel / angular-spectrum propagation",
             r"evaluate objective $J(a)$",
             "flip ring / simulated annealing",
             "converged?",
             "export manufacturable mask"]
    # Compress the flow toward the top so the objective box has a clear band
    # along the bottom (well below the feedback-arrow corridor).
    ys = np.linspace(9.35, 2.95, len(steps))
    cols = ["#ffeef0"] + ["#eef4ff"] * 5 + ["#e9f7ef"]
    bw, bh = 6.0, 0.72
    cx = 4.0
    for y, t, c in zip(ys, steps, cols):
        axD.add_patch(FancyBboxPatch((cx - bw / 2, y - bh / 2), bw, bh,
                                     boxstyle="round,pad=0.03,rounding_size=0.1",
                                     lw=1.0, edgecolor=ACC, facecolor=c))
        axD.text(cx, y, t, ha="center", va="center", fontsize=8.2)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        axD.add_patch(FancyArrowPatch((cx, y0 - bh / 2), (cx, y1 + bh / 2),
                                      arrowstyle="-|>", mutation_scale=11,
                                      lw=1.1, color="#444"))
    # red "no" feedback from "converged?" (idx 5) back to propagation (idx 2),
    # routed on the right and drawn ON TOP (high zorder) of all boxes.
    rx = cx + bw / 2
    lp = FancyArrowPatch((rx, ys[5]), (rx, ys[2]),
                         connectionstyle="arc3,rad=-0.32", arrowstyle="-|>",
                         mutation_scale=11, lw=1.3, color=RED, zorder=8)
    axD.add_patch(lp)
    axD.text(rx + 1.02, (ys[5] + ys[2]) / 2, "no", color=RED, fontsize=9.5,
             rotation=90, ha="center", va="center", zorder=9)

    # objective expression (two-line band along the bottom, clear of the
    # right-side feedback corridor)
    obj = (r"objective  $J(a)=\sum_j w_j\,[\,I_a(z_j)-I_{target}(z_j)\,]^2$"
           "\n"
           r"$\qquad\quad +\,\alpha\,S_{sidelobe}+\beta\,C_{edge}"
           r"+\gamma\,C_{min}$")
    axD.text(5.0, 1.10, obj, ha="center", va="center", fontsize=7.8,
             color="#333", zorder=3,
             bbox=dict(boxstyle="round,pad=0.4", fc="#fbfbf6",
                       ec="#cccccc", lw=0.9))

    fig.tight_layout(pad=1.4)
    save(fig, "mfig_scheme2b_mechanism")


if __name__ == "__main__":
    fig_scheme2b()
