"""Deliverable figures + reproducible package for the VQGAN detail-fusion LOCKED test."""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vqgan_detail_fusion as vdf
import vqgan_detail_fusion_locked as vlk

LOCK = vlk.LOCK
DEV = vlk.DEV


def pooled_means(csv_path):
    by = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            for m in ("lpips", "psnr", "full_rmse", "rapsd"):
                try:
                    by[r["method"]][m].append(float(r[m]))
                except (TypeError, ValueError):
                    pass
    return {a: {m: float(np.mean(v)) for m, v in d.items()} for a, d in by.items()}


def make_pareto():
    locked = pooled_means(LOCK / "locked_per_image_rows.csv")
    dev = pooled_means(DEV / "canary" / "seed0_dev_rows.csv")  # seed0 dev arms (vqae/vqgan/fusion_*)
    order = ["vqae", "fusion_balanced", "fusion_quality_lite", "vqgan"]
    colors = {"vqae": "#5F5E5A", "fusion_balanced": "#0F6E56",
              "fusion_quality_lite": "#185FA5", "vqgan": "#993C1D"}
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for a in order:
        if a in locked:
            ax.scatter(locked[a]["psnr"], locked[a]["lpips"], s=130, color=colors[a], zorder=3,
                       label=f"{a} (locked)")
            ax.annotate(a.replace("fusion_", "F:"), (locked[a]["psnr"], locked[a]["lpips"]),
                        fontsize=9, xytext=(5, 4), textcoords="offset points")
        if a in dev:
            ax.scatter(dev[a]["psnr"], dev[a]["lpips"], s=70, facecolors="none",
                       edgecolors=colors[a], zorder=2)
    xs = [locked[a]["psnr"] for a in order if a in locked]
    ys = [locked[a]["lpips"] for a in order if a in locked]
    ax.plot(xs, ys, color="gray", alpha=0.4, lw=1.4, zorder=1)
    vqae_psnr = locked["vqae"]["psnr"]
    ax.axvline(vqae_psnr - 0.5, color="#BA7517", ls="--", lw=1.3, label="balanced PSNR floor (vqae-0.5dB)")
    ax.set_xlabel("PSNR (dB) — higher better")
    ax.set_ylabel("LPIPS — lower better")
    ax.set_title("VQGAN detail-fusion: LOCKED (filled) vs development (hollow)\nLOCKED_BALANCED_VQGAN_FUSION_CONFIRMED")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(LOCK / "LOCKED_FUSION_PARETO.png", dpi=140)
    plt.close(fig)
    print("wrote LOCKED_FUSION_PARETO.png")


def _per_image_lpips(seed):
    out = {}
    with open(LOCK / "locked_per_image_rows.csv", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["train_seed"]) != seed:
                continue
            try:
                out.setdefault(int(r["source_index"]), {})[r["method"]] = float(r["lpips"])
            except (TypeError, ValueError):
                pass
    return out


def make_qualitative(seed=0, n=6, mode="spread", out=None):
    """mode: 'spread' = evenly spaced; 'top_improve' = largest balanced-vs-VQAE LPIPS gain."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = vdf.load_cfg(seed)
    measurement, projector = vdf.build_meas(cfg, device)
    bal, _ = vlk.frozen_B()
    pk = torch.load(LOCK / "cache" / f"locked_seed{seed}.pt", map_location=device)
    pre = vdf.prep_residuals(pk, measurement, projector)
    x0f, d_A, d_G, y, truth = pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], pre["truth"]
    x0_img = pk["x0"]
    sidx = pk["source_index"].cpu().numpy().tolist()
    vqae = vdf.fuse(("scalar", 0.0), x0f, d_A, d_G, y, measurement, projector, [])
    fbal = vdf.fuse(("scalar", bal[seed]), x0f, d_A, d_G, y, measurement, projector, [])
    vqgan = vdf.fuse(("scalar", 1.0), x0f, d_A, d_G, y, measurement, projector, [])

    dl = {}
    if mode in ("top_improve", "worst"):
        pl = _per_image_lpips(seed)
        for pos, si in enumerate(sidx):
            row = pl.get(si, {})
            if "fusion_balanced" in row and "vqae" in row:
                dl[pos] = row["fusion_balanced"] - row["vqae"]
        # most negative = most improved; most positive = fusion worst vs VQAE
        pick = sorted(dl, key=lambda p: dl[p], reverse=(mode == "worst"))[:n]
    else:
        pick = np.linspace(0, truth.shape[0] - 1, n).astype(int).tolist()

    cols = [("truth", truth), ("LMMSE x0", x0_img), ("VQAE (B=0)", vqae),
            (f"balanced (B={bal[seed]})", fbal), ("VQGAN (B=1)", vqgan)]
    fig, axes = plt.subplots(n, len(cols), figsize=(len(cols) * 1.7, n * 1.85))
    for r, i in enumerate(pick):
        for c, (title, t) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(t[i, 0].clamp(0, 1).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
        tag = f"img {sidx[i]}"
        if i in dl:
            tag += f"\nΔLPIPS={dl[i]:+.3f}"
        axes[r, 0].set_ylabel(tag, fontsize=7, rotation=0, ha="right", va="center", labelpad=22)
    sub = {"top_improve": "most-improved (balanced vs VQAE)",
           "worst": "worst / least-improved (balanced vs VQAE)"}.get(mode, "spread")
    fig.suptitle(f"Locked reconstructions seed{seed} [{sub}]: VQAE structure -> fused detail -> full VQGAN", fontsize=10)
    fig.tight_layout()
    out = out or f"LOCKED_QUALITATIVE_seed{seed}.png"
    fig.savefig(LOCK / out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def make_package():
    files = [
        LOCK / "PREREGISTRATION.json",
        LOCK / "LOCKED_GATE_REPORT.json",
        LOCK / "LOCKED_CONFIRMATION_REPORT.md",
        LOCK / "LOCKED_CLAIM_EVIDENCE_LEDGER.md",
        LOCK / "LOCKED_FUSION_PARETO.png",
        LOCK / "LOCKED_QUALITATIVE.png",
        LOCK / "locked_per_image_rows.csv",
        LOCK / "reports" / "locked_split_manifest.json",
        LOCK / "reports" / "sample_hash_audit.csv",
        LOCK / "reports" / "locked_source_indices.npy",
        vlk.ROOT / "vqgan_detail_fusion.py",
        vlk.ROOT / "vqgan_detail_fusion_locked.py",
        vlk.ROOT / "vqgan_detail_fusion_locked_figs.py",
        vlk.ROOT / "locked_bundle" / "CODEX_NEXT_GOAL_VQGAN_DETAIL_FUSION_LOCKED.md",
        vlk.ROOT / "locked_bundle" / "PROJECT_BRIEF_VQGAN_DETAIL_FUSION_LOCKED.md",
    ]
    zpath = LOCK / "VQGAN_DETAIL_FUSION_LOCKED_CONFIRMATION_PACKAGE.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.exists():
                arc = f.name if f.parent == vlk.ROOT or f.parent.name in ("locked_bundle",) else str(f.relative_to(LOCK))
                z.write(f, arcname=arc)
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest().upper()
    (LOCK / "VQGAN_DETAIL_FUSION_LOCKED_PACKAGE_SHA256.txt").write_text(sha + "\n")
    print("PACKAGE:", zpath.name, "BYTES:", zpath.stat().st_size, "SHA256:", sha)


if __name__ == "__main__":
    make_pareto()
    make_qualitative()
    make_package()
