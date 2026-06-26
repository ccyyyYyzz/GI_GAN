"""Paper figures: Pareto (dev vs locked), method diagram, qualitative grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import vqgan_detail_fusion as vdf
import vqgan_detail_fusion_locked as vlk
import paper_assembly as pa

LOCK = vlk.LOCK
DEV = vlk.DEV
PAPER = pa.PAPER
COL = {"lmmse_anchor": "#888780", "vqae": "#5F5E5A", "fusion_balanced": "#0F6E56",
       "fusion_quality_lite": "#185FA5", "vqgan": "#993C1D"}
LBL = pa.ARM_LABEL


def pooled_dev():
    agg = {}
    import csv
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(list))
    for s in (0, 1, 2):
        with open(DEV / "canary" / f"seed{s}_dev_rows.csv", newline="") as f:
            for r in csv.DictReader(f):
                for m in ("lpips", "psnr", "full_rmse"):
                    try:
                        acc[r["method"]][m].append(float(r[m]))
                    except (TypeError, ValueError):
                        pass
    return {a: {m: float(np.mean(v)) for m, v in d.items()} for a, d in acc.items()}


def fig_pareto():
    locked = pa.pooled(LOCK / "locked_per_image_rows.csv", ["lpips", "psnr", "full_rmse"])
    dev = pooled_dev()
    order = ["lmmse_anchor", "vqae", "fusion_balanced", "fusion_quality_lite", "vqgan"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, xk, xl in [(axes[0], "psnr", "PSNR (dB)  —  higher = less distortion"),
                       (axes[1], "full_rmse", "Full RMSE  —  lower = less distortion")]:
        lx = [locked[a][xk] for a in order if a in locked]
        ly = [locked[a]["lpips"] for a in order if a in locked]
        ax.plot(lx, ly, color="gray", alpha=0.4, lw=1.4, zorder=1)
        for a in order:
            if a in locked:
                ax.scatter(locked[a][xk], locked[a]["lpips"], s=120, color=COL[a], zorder=3)
                ax.annotate(LBL[a], (locked[a][xk], locked[a]["lpips"]), fontsize=8,
                            xytext=(5, 4), textcoords="offset points")
            if a in dev:
                ax.scatter(dev[a][xk], dev[a]["lpips"], s=70, facecolors="none",
                           edgecolors=COL[a], zorder=2)
        if xk == "psnr" and "vqae" in locked:
            ax.axvline(locked["vqae"]["psnr"] - 0.5, color="#BA7517", ls="--", lw=1.2)
        ax.set_xlabel(xl)
        ax.set_ylabel("LPIPS  —  lower = better perception")
        ax.grid(alpha=0.3)
    fig.suptitle("Perception-distortion operating points (locked = filled, development = hollow)\n"
                 "LOCKED_BALANCED_VQGAN_FUSION_CONFIRMED", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PAPER / f"PARETO_FIGURE.{ext}", dpi=150)
    plt.close(fig)
    print("wrote PARETO_FIGURE.png/pdf")


def _box(ax, xy, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=9, zorder=3, color="#1a1a1a")


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.4, color="#555", zorder=1, shrinkA=2, shrinkB=2))


def fig_method_diagram():
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _box(ax, (5, 9.2), 4.4, 0.9, "Bucket measurement   y = A x   (5%, m=205)", "#E6F1FB", "#185FA5")
    _box(ax, (5, 7.6), 4.4, 0.9, "LMMSE anchor  x0    (A x0 = y)", "#F1EFE8", "#888780")
    _box(ax, (2.4, 5.7), 3.6, 1.0, "VQAE prior + refiner\nx_A  (structure)", "#E1F5EE", "#0F6E56")
    _box(ax, (7.6, 5.7), 3.6, 1.0, "VQGAN prior + refiner\nx_G  (adversarial detail)", "#FAECE7", "#D85A30")
    _box(ax, (5, 3.9), 6.4, 1.0, "Null-space residuals\nd_A = P0(x_A - x0),   d_G = P0(x_G - x0)", "#EEEDFE", "#534AB7")
    _box(ax, (5, 2.2), 6.4, 1.0, "Fixed null-space fusion (B from validation)\nd_F = d_A + B (d_G - d_A)", "#EEEDFE", "#534AB7")
    _box(ax, (5, 0.6), 5.8, 0.9, "x̂ = x0 + P0 d_F      A x̂ = y  (measurement-consistent)", "#E1F5EE", "#0F6E56")
    _arrow(ax, (5, 8.75), (5, 8.05))
    _arrow(ax, (4.0, 7.3), (2.7, 6.2)); _arrow(ax, (6.0, 7.3), (7.3, 6.2))
    _arrow(ax, (2.6, 5.2), (4.2, 4.4)); _arrow(ax, (7.4, 5.2), (5.8, 4.4))
    _arrow(ax, (5, 3.4), (5, 2.7)); _arrow(ax, (5, 1.7), (5, 1.05))
    ax.text(5, 4.75, "A P0 = 0  ⇒  any B stays measurement-consistent", ha="center",
            fontsize=8, style="italic", color="#534AB7")
    ax.set_title("Measurement-consistent VQGAN detail fusion", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(PAPER / f"METHOD_DIAGRAM.{ext}", dpi=150)
    plt.close(fig)
    print("wrote METHOD_DIAGRAM.png/pdf/svg")


def fig_qualitative_grid(seed=0, n=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = vdf.load_cfg(seed)
    measurement, projector = vdf.build_meas(cfg, device)
    bal, ql = vlk.frozen_B()
    pk = torch.load(LOCK / "cache" / f"locked_seed{seed}.pt", map_location=device)
    pre = vdf.prep_residuals(pk, measurement, projector)
    x0f, d_A, d_G, y, truth = pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], pre["truth"]
    sidx = pk["source_index"].cpu().numpy().tolist()
    cols = [("truth", truth), ("LMMSE x0", pk["x0"]),
            ("VQAE (B=0)", vdf.fuse(("scalar", 0.0), x0f, d_A, d_G, y, measurement, projector, [])),
            (f"balanced (B={bal[seed]})", vdf.fuse(("scalar", bal[seed]), x0f, d_A, d_G, y, measurement, projector, [])),
            (f"quality-lite (B={ql[seed]})", vdf.fuse(("scalar", ql[seed]), x0f, d_A, d_G, y, measurement, projector, [])),
            ("full VQGAN (B=1)", vdf.fuse(("scalar", 1.0), x0f, d_A, d_G, y, measurement, projector, []))]
    pick = np.linspace(0, truth.shape[0] - 1, n).astype(int)
    fig, axes = plt.subplots(n, len(cols), figsize=(len(cols) * 1.6, n * 1.7))
    for r, i in enumerate(pick):
        for c, (title, t) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(t[i, 0].clamp(0, 1).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8.5)
        axes[r, 0].set_ylabel(f"img {sidx[i]}", fontsize=7, rotation=0, ha="right", va="center", labelpad=18)
    fig.suptitle(f"Locked-split reconstructions (fixed samples, seed{seed}): structure → fused detail → full VQGAN",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PAPER / f"QUALITATIVE_GRID.{ext}", dpi=140)
    plt.close(fig)
    print("wrote QUALITATIVE_GRID.png/pdf")


if __name__ == "__main__":
    fig_pareto()
    fig_method_diagram()
    fig_qualitative_grid()
