"""Pseudo-3D mechanism figure for the measurement-consistent VQGAN detail-fusion paper.

The figure renders the *geometry* of the method rather than a data-flow flowchart:

  * the affine plane  {x : A x = y} = x0 + null(A)  drawn in oblique pseudo-3D — every
    measurement-consistent image lives here;
  * the normal (row-space) direction sticking out of the plane — moving along it changes the
    bucket y, so we never leave the plane;
  * the LMMSE anchor x0, the VQAE point x_A (structure), the VQGAN point x_G (detail), and the
    true scene x* (whose null-space location y cannot reveal), all ON the plane;
  * the fusion dial: x_hat_B = x0 + P0(d_A + B (d_G - d_A)) slides along the in-plane segment
    x_A -> x_G; balanced is the validation-selected operating point;
  * a perception-distortion gradient under the dial.

Writes METHOD_DIAGRAM_3D.{png,pdf,svg} next to the other paper figures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle, FancyBboxPatch, Circle
from matplotlib.collections import LineCollection

OUT = Path("outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper")

# ----------------------------------------------------------------------------- projection
KY, JY, KZ = 0.52, 0.34, -0.14          # oblique: +Y (depth) -> up-right, +Z (normal) -> up-left


def P(X, Y, Z=0.0):
    return (X + KY * Y + KZ * Z, JY * Y + Z)


# ----------------------------------------------------------------------------- palette
C_PLANE = "#eef3f9"; C_PLANE_EDGE = "#93a9c4"; C_GRID = "#d6e0ee"
C_ANCHOR = "#3d3d3d"; C_VQAE = "#3a6ea5"; C_VQGAN = "#c0392b"
C_BAL = "#e0a500"; C_BAL_EDGE = "#6b4e00"; C_TRUTH = "#8a8a8a"
C_NORMAL = "#2f3e4e"; C_FORBID = "#c0392b"; C_FUSE = "#5f4b8b"
INK = "#1c1c1c"


# ----------------------------------------------------------------------------- stylized chips
def chip(kind, n=80):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    cx = cy = (n - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (n / 2.0)
    blob = np.clip(1.0 - r, 0, 1)
    rng = np.random.default_rng(7)
    if kind == "vqae":                                   # smooth, low-contrast structure
        img = 0.46 + 0.40 * blob
    elif kind == "bal":                                  # structure + a crisp edge ring
        ring = np.exp(-((r - 0.52) ** 2) / 0.010)
        img = np.clip(0.30 + 0.60 * blob + 0.22 * ring, 0, 1)
    else:                                                # adversarial high-frequency detail
        hi = (np.sin(xx * 1.5) * np.sin(yy * 1.5) * 0.5 + 0.5)
        tex = 0.55 * rng.random((n, n)) + 0.45 * hi
        img = np.clip(0.24 + 0.58 * blob + 0.55 * (tex - 0.5) * (blob > 0.10), 0, 1)
    return img


def place_chip(ax, cx, cy, kind, edge, label, half=0.92):
    ax.imshow(chip(kind), cmap="gray", vmin=0, vmax=1,
              extent=[cx - half, cx + half, cy - half, cy + half],
              zorder=7, interpolation="bilinear")
    ax.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half,
                           fill=False, edgecolor=edge, lw=2.2, zorder=7.1))
    ax.text(cx, cy - half - 0.18, label, ha="center", va="top",
            fontsize=8.4, color=edge, fontweight="bold", zorder=7.2)


def arrow(ax, a, b, color, lw=2.0, ls="-", ms=15, z=5, alpha=1.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 linestyle=ls, color=color, shrinkA=0, shrinkB=0,
                                 zorder=z, alpha=alpha))


def main():
    fig, ax = plt.subplots(figsize=(11.6, 7.4))
    ax.set_aspect("equal"); ax.axis("off")

    # -------------------------------------------------------------- the consistency plane
    XW, YW = 12.5, 8.2
    corners = [P(0, 0), P(XW, 0), P(XW, YW), P(0, YW)]
    ax.add_patch(Polygon(corners, closed=True, facecolor=C_PLANE,
                         edgecolor=C_PLANE_EDGE, lw=1.6, zorder=1))
    segs = []
    for x in np.arange(2, XW, 2):
        segs.append([P(x, 0), P(x, YW)])
    for y in np.arange(2, YW, 2):
        segs.append([P(0, y), P(XW, y)])
    ax.add_collection(LineCollection(segs, colors=C_GRID, linewidths=0.8, zorder=1.1))

    # name the surface, faint, in the clear back-left corner of the plane
    ntag = P(10.6, 7.3)
    ax.text(ntag[0], ntag[1], r"$\mathcal{N}(A)$", ha="center", va="center",
            fontsize=14, color="#8aa0bb", style="italic", zorder=1.4)

    # -------------------------------------------------------------- key in-plane points
    x0 = (3.0, 2.5)
    xA = (4.8, 3.9)
    xG = (9.6, 6.4)
    tB = 0.55
    xBAL = (xA[0] + tB * (xG[0] - xA[0]), xA[1] + tB * (xG[1] - xA[1]))
    xT = (9.0, 2.4)

    def pt(p, z=0.0):
        return P(p[0], p[1], z)

    # d_A, d_G null-space contribution vectors (x0 -> xA, x0 -> xG)
    arrow(ax, pt(x0), pt(xA), C_VQAE, lw=1.6, ls=(0, (5, 2)), ms=12, z=3)
    arrow(ax, pt(x0), pt(xG), C_VQGAN, lw=1.6, ls=(0, (5, 2)), ms=12, z=3)
    mAx, mAy = (pt(x0)[0] + pt(xA)[0]) / 2, (pt(x0)[1] + pt(xA)[1]) / 2
    mGx, mGy = (pt(x0)[0] + pt(xG)[0]) / 2, (pt(x0)[1] + pt(xG)[1]) / 2
    ax.text(mAx - 0.15, mAy + 0.18, r"$d_A=P_0(x_A-x_0)$", color=C_VQAE, fontsize=8.6,
            ha="right", va="bottom", rotation=0)
    ax.text(mGx + 0.1, mGy - 0.28, r"$d_G=P_0(x_G-x_0)$", color=C_VQGAN, fontsize=8.6,
            ha="left", va="top")

    # the fusion dial: solid segment xA -> xG, on the plane
    ax.add_patch(FancyArrowPatch(pt(xA), pt(xG), arrowstyle="-", lw=5.0, color=C_FUSE,
                                 alpha=0.30, zorder=3.4, shrinkA=0, shrinkB=0))
    # B tick marks along the dial
    for tval in np.linspace(0, 1, 11):
        q = P(xA[0] + tval * (xG[0] - xA[0]), xA[1] + tval * (xG[1] - xA[1]))
        ax.plot([q[0]], [q[1]], marker="|", ms=9, color=C_FUSE, mew=1.4, zorder=3.5,
                alpha=0.8)

    # endpoints + anchor + truth
    ax.plot(*pt(x0), "o", ms=11, color=C_ANCHOR, zorder=5, markeredgecolor="white", mew=1.3)
    ax.plot(*pt(xA), "o", ms=12, color=C_VQAE, zorder=5, markeredgecolor="white", mew=1.4)
    ax.plot(*pt(xG), "o", ms=12, color=C_VQGAN, zorder=5, markeredgecolor="white", mew=1.4)
    ax.plot(*pt(xBAL), marker="*", ms=26, color=C_BAL, zorder=6,
            markeredgecolor=C_BAL_EDGE, mew=1.5)
    ax.plot(*pt(xT), marker="*", ms=20, color="white", zorder=5,
            markeredgecolor=C_TRUTH, mew=1.6)

    # point labels
    ax.annotate(r"$x_0$  LMMSE anchor" "\n" r"$A x_0 = y$ (audited)",
                pt(x0), (pt(x0)[0] - 1.9, pt(x0)[1] - 1.15), fontsize=9, color=C_ANCHOR,
                ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=C_ANCHOR, lw=0.8))
    ax.text(pt(xA)[0] - 0.18, pt(xA)[1] - 0.05, r"$x_A$", color=C_VQAE, fontsize=11,
            ha="right", va="center", fontweight="bold")
    ax.text(pt(xG)[0] + 0.18, pt(xG)[1], r"$x_G$", color=C_VQGAN, fontsize=11,
            ha="left", va="center", fontweight="bold")
    ax.annotate(r"$B\!\approx\!0.55$" "\n" "balanced (selected on val)",
                pt(xBAL), (pt(xBAL)[0] + 0.15, pt(xBAL)[1] - 1.5), fontsize=9.2,
                color=C_BAL_EDGE, ha="center", va="top", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_BAL_EDGE, lw=0.9))
    ax.text(pt(xA)[0] - 0.05, pt(xA)[1] + 0.32, r"$B=0$", color=C_FUSE, fontsize=8, ha="right")
    ax.text(pt(xG)[0] + 0.05, pt(xG)[1] + 0.30, r"$B=1$", color=C_FUSE, fontsize=8, ha="left")
    ax.annotate(r"$x^{*}$  true scene" "\n" r"($A x^{*}=y$ too — but its null-space" "\n" "location is unknowable from $y$)",
                pt(xT), (pt(xT)[0] + 1.0, pt(xT)[1] - 0.4), fontsize=8.4, color=C_TRUTH,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=C_TRUTH, lw=0.8))

    # fusion-rule label above the dial
    fx, fy = (pt(xA)[0] + pt(xG)[0]) / 2, (pt(xA)[1] + pt(xG)[1]) / 2
    ax.text(fx - 0.35, fy + 0.86,
            r"fuse in the null space:   $\hat{x}_B = x_0 + P_0\,(d_A + B\,(d_G-d_A))$",
            color=C_FUSE, fontsize=10.5, ha="center", va="bottom", fontweight="bold",
            rotation=np.degrees(np.arctan2(pt(xG)[1] - pt(xA)[1], pt(xG)[0] - pt(xA)[0])))

    # -------------------------------------------------------------- normal (row space) axis
    nbase = pt(x0)
    ntip = P(x0[0], x0[1], 3.55)
    arrow(ax, nbase, ntip, C_NORMAL, lw=2.4, ms=17, z=4)
    # a forbidden off-plane ghost on the normal
    gp = P(x0[0], x0[1], 2.1)
    ax.plot(*gp, "o", ms=11, color="white", markeredgecolor=C_FORBID, mew=2.0, zorder=4.5)
    ax.text(gp[0] + 0.2, gp[1], r"$A\hat{x}\neq y$  ✗", color=C_FORBID, fontsize=8.8,
            ha="left", va="center", fontweight="bold")
    ax.text(ntip[0], ntip[1] + 0.14, r"row space $\mathcal{R}(A^{\top})$",
            color=C_NORMAL, fontsize=10, ha="center", va="bottom", fontweight="bold")
    ax.text(ntip[0], ntip[1] - 0.16, r"fixed by $y$  ($m{=}205$ dims)",
            color=C_NORMAL, fontsize=8.4, ha="center", va="top", style="italic")

    # -------------------------------------------------------------- forward-problem vignette
    vx, vy = -1.8, 6.05
    ax.add_patch(FancyBboxPatch((vx - 0.05, vy - 1.05), 3.0, 1.95,
                                boxstyle="round,pad=0.06,rounding_size=0.12",
                                facecolor="#fbfbf7", edgecolor="#b9b39c", lw=1.2, zorder=8))
    ax.text(vx + 1.45, vy + 0.66, "low-rate ghost imaging", ha="center", fontsize=8.8,
            color=INK, fontweight="bold", zorder=8.1)
    # tiny scene + operator + buckets
    sc = chip("bal", 48)
    ax.imshow(sc, cmap="gray", vmin=0, vmax=1, extent=[vx + 0.05, vx + 0.7, vy - 0.55, vy + 0.4],
              zorder=8.1, interpolation="bilinear")
    ax.text(vx + 0.375, vy - 0.72, r"$x$", ha="center", fontsize=8, zorder=8.1)
    ax.add_patch(FancyArrowPatch((vx + 0.82, vy - 0.08), (vx + 1.42, vy - 0.08),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.5,
                                 color="#444", zorder=8.1))
    ax.text(vx + 1.12, vy + 0.08, r"$A$", ha="center", va="bottom", fontsize=9, zorder=8.1)
    rng = np.random.default_rng(1)
    bars = rng.uniform(0.2, 1.0, 7)
    for i, b in enumerate(bars):
        ax.add_patch(Rectangle((vx + 1.5 + i * 0.095, vy - 0.45), 0.07, b * 0.7,
                               facecolor="#185FA5", edgecolor="none", zorder=8.1))
    ax.text(vx + 1.83, vy - 0.72, r"$y=Ax$", ha="center", fontsize=8, zorder=8.1)
    ax.text(vx + 1.45, vy + 0.40, r"$m=205 \ll n=4096$  (5%)", ha="center", fontsize=7.6,
            color="#41556d", zorder=8.1)
    # vignette -> anchor
    arrow(ax, (vx + 1.6, vy - 1.05), (pt(x0)[0] - 0.2, pt(x0)[1] + 0.45), "#9a937c",
          lw=1.4, ms=12, z=4)
    ax.text((vx + 1.6 + pt(x0)[0]) / 2 - 0.5, (vy - 1.05 + pt(x0)[1]) / 2 + 0.25,
            "anchor", fontsize=8, color="#7d7765", style="italic", ha="center")

    # -------------------------------------------------------------- chip strip (top): effect of B
    ax.text(7.4, 7.95, r"perceptual effect of the dial $B$", ha="center", va="bottom",
            fontsize=10, color=INK, fontweight="bold")
    place_chip(ax, 5.0, 7.05, "vqae", C_VQAE, r"structure  $B{=}0$", half=0.62)
    place_chip(ax, 7.4, 7.05, "bal", C_BAL_EDGE, "balanced", half=0.62)
    place_chip(ax, 9.8, 7.05, "vqgan", C_VQGAN, r"detail  $B{=}1$", half=0.62)
    arrow(ax, (5.75, 7.05), (9.05, 7.05), "#b7c0cd", lw=1.3, ms=11, z=6.6)
    # subtle tie from the balanced chip down to the balanced operating point
    ax.plot([7.4, pt(xBAL)[0]], [7.05 - 0.62, pt(xBAL)[1]], color="#cdb86a", lw=0.9,
            ls=":", zorder=4)

    # -------------------------------------------------------------- perception-distortion bar
    gy = -1.35
    gx0, gx1 = 1.0, 12.0
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=[gx0, gx1, gy - 0.28, gy + 0.28], aspect="auto",
              cmap="coolwarm", zorder=2, alpha=0.85)
    ax.add_patch(Rectangle((gx0, gy - 0.28), gx1 - gx0, 0.56, fill=False,
                           edgecolor="#888", lw=1.0, zorder=2.1))
    ax.text(gx0 - 0.15, gy, "VQAE  $B{=}0$", ha="right", va="center", fontsize=8.6,
            color=C_VQAE, fontweight="bold")
    ax.text(gx1 + 0.15, gy, "$B{=}1$  VQGAN", ha="left", va="center", fontsize=8.6,
            color=C_VQGAN, fontweight="bold")
    ax.text((gx0 + gx1) / 2, gy + 0.46, "the dial $B$ : a measurement-safe perception–distortion trade",
            ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.text(gx0 + 0.2, gy - 0.46, "best PSNR / RMSE (pixel fidelity)", ha="left", va="top",
            fontsize=7.8, color=C_VQAE)
    ax.text(gx1 - 0.2, gy - 0.46, "best LPIPS / KID (perceptual realism)", ha="right", va="top",
            fontsize=7.8, color=C_VQGAN)
    # mark balanced on the bar
    bxbar = gx0 + tB * (gx1 - gx0)
    ax.plot([bxbar], [gy], marker="*", ms=17, color=C_BAL, markeredgecolor=C_BAL_EDGE,
            mew=1.2, zorder=2.5)

    # -------------------------------------------------------------- title + subtitle + key insight
    ax.text(6.5, 8.92, "Measurement-consistent VQGAN detail fusion lives in the null space",
            ha="center", fontsize=14.5, color=INK, fontweight="bold")
    ax.text(6.5, 8.34,
            r"$\{x : A x = y\}=x_0+\mathcal{N}(A)$ :  the bucket fixes the row space ($m{=}205$);  the null space ($3891$ dims of detail) is free",
            ha="center", fontsize=10, color="#41556d")
    ax.add_patch(FancyBboxPatch((10.85, 5.32), 4.7, 1.42,
                                boxstyle="round,pad=0.12,rounding_size=0.18",
                                facecolor="#f3f0fb", edgecolor=C_FUSE, lw=1.5, zorder=8))
    ax.text(13.2, 6.32, r"$A P_0 = 0\;\Rightarrow\;A\hat{x}_B = y$",
            ha="center", fontsize=11, color=C_FUSE, fontweight="bold", zorder=8.1)
    ax.text(13.2, 5.92, r"for every $B$",
            ha="center", fontsize=9.5, color=C_FUSE, zorder=8.1)
    ax.text(13.2, 5.55, "the perceptual dial costs nothing\nin measurement fidelity — by construction",
            ha="center", va="center", fontsize=8.2, color="#4a3f6b", style="italic", zorder=8.1)

    ax.set_xlim(-2.8, 15.9)
    ax.set_ylim(-2.1, 9.4)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"METHOD_DIAGRAM_3D.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote METHOD_DIAGRAM_3D.png/pdf/svg")


if __name__ == "__main__":
    main()
