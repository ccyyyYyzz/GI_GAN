# -*- coding: utf-8 -*-
"""Three explanatory figures for the theory core of the unified paper (readability pass):
  ROADMAP.pdf        - the deductive chain as a visual map (main-thread anchor, section 1)
  WITNESS_GEOMETRY.pdf - the fiber + witness-splice construction as a 2-D cartoon (section 3)
  CERT_SEPARATION.pdf  - per-mode contraction spectrum + the quality/accountability separation (sections 4-5)
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

REPO = Path(r"E:\ns_mc_gan_gi_code_fcc_phase1")
OUTS = [REPO / "paper", REPO / "outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper"]

def save(fig, name):
    for o in OUTS:
        fig.savefig(o / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(o / f"{name}.png", dpi=200, bbox_inches="tight")
    print("wrote", name)

# ---------------------------------------------------------------- ROADMAP
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.set_xlim(0, 11); ax.set_ylim(0, 3.4); ax.axis("off")

def box(x, y, w, h, title, sub, fc, title_fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec="k", lw=1.1))
    ax.text(x + w/2, y + h - 0.38, title, ha="center", va="center", fontsize=title_fs, fontweight="bold")
    ax.text(x + w/2, y + (h - 0.55)/2 - 0.03, sub, ha="center", va="center", fontsize=8.3)

def arrow(x1, y1, x2, y2, label=None, dy=0.16):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.4, color="k"))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + dy, label, ha="center", fontsize=8.2, style="italic")

# the identity feeds three acts
box(0.15, 1.15, 2.0, 1.15, "$AP_0 = 0$", "the record $y$ fixes $P_Rx$\nand nothing else  (§2)", "#f0f0f0", 12)
box(2.95, 2.15, 2.35, 1.1, "CANNOT  (§3)", "feasible-but-wrong witness:\nnull content is unverifiable", "#f6d7d7")
box(2.95, 0.15, 2.35, 1.1, "CAN  (§4–5)", "exact per-mode certificate:\naudit the measured part only", "#d7e6f6")
box(6.1, 1.15, 2.3, 1.15, "THEREFORE (§6–7)", "governed dial $B$: inject detail\nonly where the bucket is blind,\n$A\\hat{x}_B = y$ exact for every $B$", "#dcf0dc")
box(9.0, 2.15, 1.85, 1.1, "LIMITS (§7.4)", "no GT-free per-image\nadaptation on one record", "#f0e6d2")
box(9.0, 0.15, 1.85, 1.1, "FIELD (§8.4)", "3 published pipelines:\ngains live in the null ledger", "#f0e6d2")
arrow(2.18, 1.95, 2.92, 2.55)
arrow(2.18, 1.5, 2.92, 0.85)
arrow(5.33, 2.55, 6.35, 2.05, "licenses", 0.14)
arrow(5.33, 0.85, 6.35, 1.35, "protects", -0.2)
arrow(8.43, 2.0, 8.97, 2.5)
arrow(8.43, 1.45, 8.97, 0.9)
fig.tight_layout(); save(fig, "ROADMAP"); plt.close(fig)

# ------------------------------------------------------ WITNESS GEOMETRY
fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.set_xlim(-0.4, 10.4); ax.set_ylim(-0.6, 7.4); ax.axis("off")
# the fiber = a slanted line (affine flat), row-space = perpendicular direction
t = np.linspace(-0.2, 9.4, 10)
fx, fy = t, 0.55 * t + 1.0
ax.plot(fx, fy, color="#4c72b0", lw=2.4)
ax.text(5.7, 0.55*5.7 + 0.62, "the fiber $\\{x: Ax=y_i\\} = A^{\\dagger}y_i + \\mathcal{N}(A)$",
        fontsize=10, color="#4c72b0", ha="center", rotation=22)
# row-space direction arrow (perpendicular)
ax.add_patch(FancyArrowPatch((1.6, 1.88), (0.62, 3.62), arrowstyle="-|>", mutation_scale=14, color="#777"))
ax.text(-0.3, 4.05, "row space $\\mathcal{R}(A^\\top)$:\nmoving off the line changes $y$", fontsize=8.6, color="#555", ha="left")
# points: truth x_i (on fiber up to noise), donor x_j (off fiber), witness u_ij (on fiber)
xi = (3.0, 0.55*3.0 + 1.0)
ax.plot(*xi, "o", ms=11, color="#2a2a2a"); ax.annotate("true scene $x_i$", xi, xytext=(3.5, 1.35), fontsize=10,
        arrowprops=dict(arrowstyle="->", lw=0.9))
xj = (7.6, 5.9)
ax.plot(*xj, "s", ms=10, color="#c44e52"); ax.annotate("donor $x_j$ (different class)", xj, xytext=(4.8, 6.6), fontsize=10,
        arrowprops=dict(arrowstyle="->", lw=0.9))
# projection of donor onto fiber = witness
uij = (7.6 + 0.55*(5.9 - (0.55*7.6+1.0))/(1+0.55**2)*(-0.55)*-1, 0)  # compute orth projection properly below
# orthogonal projection of xj onto line y = 0.55x + 1
d, c = 0.55, 1.0
px = (xj[0] + d*(xj[1]-c)) / (1 + d*d)
py = d*px + c
ax.plot([xj[0], px], [xj[1], py], ls=":", color="#c44e52", lw=1.6)
ax.plot(px, py, "D", ms=11, color="#c44e52", mec="k")
ax.annotate("witness $u_{ij} = x_j - A^{\\dagger}(Ax_j - y_i)$\nsame record as $x_i$ (to $10^{-15}$),\ndonor's null content",
            (px, py), xytext=(4.6, -0.35), fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="->", lw=0.9))
# noise ball around fiber at truth
circ = Circle(xi, 0.42, fc="none", ec="#2a2a2a", ls="--", lw=1.0)
ax.add_patch(circ)
ax.text(xi[0]-2.6, xi[1]-0.15, "noisy truth misses its own\nrecord by $\\sim\\sigma_\\varepsilon$", fontsize=8.6, ha="left")
ax.set_title("The converse in one picture: every point on the line reproduces the record;\n"
             "the measurement cannot prefer the truth over the splice", fontsize=10.5)
fig.tight_layout(); save(fig, "WITNESS_GEOMETRY"); plt.close(fig)

# ----------------------------------------------------- CERT + SEPARATION
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
# (a) contraction spectrum
ax = axes[0]
sig = np.linspace(0.5, 5.5, 60)
for lam, cstr in [(1e-2, "#4c72b0"), (1e-1, "#55a868"), (1.0, "#c44e52")]:
    ax.semilogy(sig, lam/(lam + sig**2), lw=2, color=cstr, label=f"$\\lambda={lam:g}$")
ax.set_xlabel("singular value $\\sigma_i$ of the measured mode")
ax.set_ylabel("residual contraction  $\\lambda/(\\lambda+\\sigma_i^2)$")
ax.set_title("(a) The certificate: each measured mode\ncontracts by a closed-form factor")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
# (b) separation: residual falls orders, PSNR flat
ax = axes[1]
arms = ["BP", "Tikhonov", "CS–TV", "learned"]
pre = [3.2e-2, 2.1e-2, 8.0e-3, 3.68e-2]
post = [2.5e-6, 2.2e-6, 1.9e-6, 1.90e-6]
x = np.arange(4)
ax.semilogy(x - 0.12, pre, "o", ms=9, color="#c44e52", label="RelMeasErr before audit")
ax.semilogy(x + 0.12, post, "o", ms=9, color="#4c72b0", label="after audit")
for xi_, a, b in zip(x, pre, post):
    ax.annotate("", (xi_ + 0.12, b*2), (xi_ - 0.12, a/2), arrowprops=dict(arrowstyle="->", lw=1.2, color="#888"))
ax2 = ax.twinx()
ax2.plot(x, [0.014, 0.02, 0.03, 0.0136], "s--", color="#55a868", ms=6, label="|ΔPSNR| (dB)")
ax2.set_ylim(0, 0.25); ax2.set_ylabel("$|\\Delta$PSNR$|$ (dB)", color="#55a868")
ax2.tick_params(axis="y", colors="#55a868")
ax.set_xticks(x); ax.set_xticklabels(arms)
ax.set_ylabel("relative measurement error")
ax.set_title("(b) The separation: accountability moves\norders of magnitude, quality does not")
ax.legend(fontsize=8, loc="center right")
ax.grid(alpha=0.3)
fig.tight_layout(); save(fig, "CERT_SEPARATION"); plt.close(fig)
