"""Core mechanism figure (concise): measurement-consistent VQGAN detail fusion.

One clean flow: bucket data fix the row space; adversarial detail is added only
through the null space (A P0 = 0), so A x_hat_B = y exactly for every dial B.

Exports CORE_MECHANISM_FIGURE.{png,pdf,svg} into the detail-fusion paper assets
(matplotlib, dpi=200, tight). Run with the py311 env.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper"

BLUE, ORANGE, PURPLE, GREEN = "#2f6fed", "#e0820c", "#9039e6", "#18a558"
INK, SUB = "#1c1e21", "#5a5f66"
W, H = 1060, 540


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.invert_yaxis(); ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, ec, label, sub=None, lab_c=None, fs=15):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                                    ec=ec, fc=to_rgba(ec, 0.10), lw=2, zorder=2))
        cy = y + h / 2 - (8 if sub else 0)
        ax.text(x + w / 2, cy, label, fontsize=fs, color=lab_c or ec, fontweight="bold",
                ha="center", va="center", zorder=4)
        if sub:
            ax.text(x + w / 2, y + h / 2 + 13, sub, fontsize=11, color=SUB, ha="center", va="center", zorder=4)

    def arrow(x1, y1, x2, y2, color=SUB, lw=2, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                                     shrinkA=2, shrinkB=2, zorder=3))

    fig.canvas.draw(); rend = fig.canvas.get_renderer()

    def row(x, y, items, fs, anchor="center"):
        widths = []
        for s, c, w in items:
            t = ax.text(0, -100, s, fontsize=fs, fontweight=w, ha="left", va="center")
            bb = t.get_window_extent(renderer=rend)
            (a, _), (b, _) = ax.transData.inverted().transform([(bb.x0, 0), (bb.x1, 0)])
            widths.append(abs(b - a)); t.remove()
        total = sum(widths)
        cx = x - total / 2 if anchor == "center" else x
        for (s, c, w), wd in zip(items, widths):
            ax.text(cx, y, s, fontsize=fs, color=c, fontweight=w, ha="left", va="center", zorder=4)
            cx += wd

    # ---- title ----
    ax.text(W / 2, 30, "Measurement-Consistent VQGAN Detail Fusion", fontsize=20,
            color=INK, fontweight="bold", ha="center", va="center")

    # ---- principle band ----
    row(W / 2, 66, [(r"$y = A\,x$  ", INK, "bold"), (r"$(m \ll n)$        ", SUB, "normal"),
                    ("x = ", INK, "normal"), (r"$P_R x$", BLUE, "bold"), ("  +  ", INK, "normal"),
                    (r"$P_0 x$", ORANGE, "bold"), ("        ", INK, "normal"),
                    (r"$A\,P_0 = 0$", GREEN, "bold")], 17)
    row(W / 2, 92, [("bucket data fix the ", SUB, "normal"), ("blue row space", BLUE, "bold"),
                    (" ; the ", SUB, "normal"), ("orange null space", ORANGE, "bold"),
                    (" is free for a prior", SUB, "normal")], 12.5)

    # ---- pipeline ----
    cy = 300
    box(40, cy - 48, 150, 96, BLUE, r"$x_0$", "audited LMMSE anchor", fs=20)
    box(250, 168, 165, 86, BLUE, r"$x_A$", "VQAE  ·  structure", fs=18)
    box(250, 346, 165, 86, PURPLE, r"$x_G$", "VQGAN  ·  detail", fs=18)
    arrow(190, cy - 22, 250, 210, SUB, 1.8, rad=-0.25)
    arrow(190, cy + 22, 250, 390, SUB, 1.8, rad=0.25)

    # dial
    dx, dy, dw, dh = 470, cy - 96, 360, 192
    ax.add_patch(FancyBboxPatch((dx, dy), dw, dh, boxstyle="round,pad=0,rounding_size=12",
                                ec=INK, fc=to_rgba(INK, 0.035), lw=1.8, zorder=2))
    row(dx + dw / 2, dy + 34, [(r"$\hat{x}_B = x_0 + P_0\,[\,$", INK, "normal"), (r"$d_A$", ORANGE, "bold"),
                               (r"$ + B\,($", INK, "normal"), (r"$d_G$", PURPLE, "bold"),
                               (r"$ - $", INK, "normal"), (r"$d_A$", ORANGE, "bold"), (r"$)\,]$", INK, "normal")], 16)
    tx0, tx1, ty = dx + 42, dx + dw - 42, dy + 104
    ax.plot([tx0, tx1], [ty, ty], color="#9aa0a8", lw=3.2, solid_capstyle="round", zorder=3)
    ax.add_patch(Circle((tx0, ty), 6, fc=BLUE, ec="none", zorder=4))
    ax.add_patch(Circle((tx1, ty), 6, fc=PURPLE, ec="none", zorder=4))
    ax.plot((tx0 + tx1) / 2, ty, marker="*", ms=26, color=GREEN, mec="#0f7a40", mew=1, zorder=5)
    ax.text(tx0, ty + 24, "B=0", fontsize=12, color=BLUE, fontweight="bold", ha="center", va="center")
    ax.text(tx0, ty + 40, "VQAE", fontsize=10.5, color=SUB, ha="center", va="center")
    ax.text(tx1, ty + 24, "B=1", fontsize=12, color=PURPLE, fontweight="bold", ha="center", va="center")
    ax.text(tx1, ty + 40, "VQGAN", fontsize=10.5, color=SUB, ha="center", va="center")
    ax.text((tx0 + tx1) / 2, ty - 22, "B ≈ 0.5  balanced", fontsize=11.5, color=GREEN, fontweight="bold", ha="center", va="center")
    ax.text(dx + dw / 2, dy + dh - 16, "one fixed scalar B : perception ↔ distortion", fontsize=11, color=SUB, ha="center", va="center")

    arrow(415, 210, 470, cy - 36, ORANGE, 1.8, rad=-0.2)
    arrow(415, 390, 470, cy + 36, ORANGE, 1.8, rad=0.2)
    arrow(dx + dw, cy, 880, cy, GREEN, 2)
    box(880, cy - 48, 150, 96, GREEN, r"$\hat{x}_B$", "fused output", lab_c=INK, fs=20)

    # ---- consistency bar ----
    ax.add_patch(FancyBboxPatch((40, 460), W - 80, 56, boxstyle="round,pad=0,rounding_size=12",
                                ec=GREEN, fc=to_rgba(GREEN, 0.10), lw=2, zorder=2))
    ax.add_patch(Circle((72, 488), 14, fc=GREEN, ec="none", zorder=4))
    ax.plot([66, 70.5, 79], [488, 493, 482], color="white", lw=2.6, solid_capstyle="round",
            solid_joinstyle="round", zorder=5)
    row(100, 482, [(r"$A\,\hat{x}_B = A x_0 + $", INK, "normal"), (r"$A P_0$", ORANGE, "bold"),
                   (r"$[\ldots] = A x_0 = y$", INK, "normal"), ("   ✓", GREEN, "bold")], 16.5, anchor="left")
    ax.text(100, 503, "measurement preserved exactly for every B — detail is added only where A is blind",
            fontsize=11.5, color=SUB, ha="left", va="center")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"CORE_MECHANISM_FIGURE.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote CORE_MECHANISM_FIGURE.{png,pdf,svg} ->", OUT)


if __name__ == "__main__":
    main()
