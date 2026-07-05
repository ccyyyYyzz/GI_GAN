# -*- coding: utf-8 -*-
"""Feasible-but-wrong witness on real CASSI: two different spectral scenes share ONE coded
snapshot. u = A^dagger y_target + P_0 x_donor reproduces the target's snapshot exactly while
carrying the donor's spectral null content. Also a box-constrained ([0,1]) POCS witness, and
a per-pixel spectrum plot showing target vs witness spectra differ under an identical record.
"""
import os, sys, glob, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cassi_operator import CASSI, load_mask

OUT = os.path.dirname(os.path.abspath(__file__))
MASK = os.path.join(OUT, "mask.mat")
DTYPE = torch.float64


def load_scene(p):
    import scipy.io as sio
    return torch.from_numpy(sio.loadmat(p)["img"].astype(np.float64)).permute(2, 0, 1)  # [28,256,256]


def to_rgb(cube):
    """Render a [28,H,W] cube to sRGB-ish for display (coarse 3-band pooling)."""
    c = cube.clamp(0, None)
    b = c[0:9].mean(0); g = c[9:19].mean(0); r = c[19:28].mean(0)
    rgb = torch.stack([r, g, b], -1).cpu().numpy()
    rgb = rgb / (np.percentile(rgb, 99) + 1e-9)
    return np.clip(rgb, 0, 1)


def box_pocs(op, x_donor, y_target, iters=200):
    u = op.witness(x_donor, y_target)
    for _ in range(iters):
        u = u.clamp(0, 1)                       # spectral reflectance box
        u = u - op.A_dagger(op.A(u) - y_target)  # exact fiber projection
    return u.clamp(0, 1)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean()); return -10 * np.log10(max(mse, 1e-30))


def main(n_pairs=10, lam=1e-3):
    op = CASSI(load_mask(MASK), nC=28, step=2)
    scenes = sorted(glob.glob(os.path.join(OUT, "scenes", "scene*.mat")))
    cubes = [load_scene(p) for p in scenes]
    print(f"{len(cubes)} scenes; operator cube[28,{op.H},{op.W}] -> snapshot[{op.H},{op.Ws}] (23.1x)")

    rows = []
    gallery = None
    for k in range(min(n_pairs, len(cubes) - 1)):
        xt, xd = cubes[k], cubes[(k + 3) % len(cubes)]
        yt = op.A(xt)
        u = op.witness(xd, yt)
        r_wit = op.rel_meas_err(u, yt)
        null_match = float(torch.linalg.vector_norm(op.P_0(u) - op.P_0(xd)) /
                           (torch.linalg.vector_norm(op.P_0(xd)) + 1e-30))
        ub = box_pocs(op, xd, yt)
        r_box = op.rel_meas_err(ub, yt)
        rows.append(dict(pair=[k, (k + 3) % len(cubes)], relerr_witness=r_wit,
                         relerr_box_witness=r_box, null_match=null_match,
                         psnr_wit_to_target=psnr(u, xt), psnr_wit_to_donor=psnr(u, xd),
                         box_oob=float(((op.witness(xd, yt) < 0) | (op.witness(xd, yt) > 1)).float().mean())))
        if gallery is None:
            gallery = (xt, xd, u, ub, yt, op.A(u))
    arr = lambda key: np.array([r[key] for r in rows])
    print(f"witness record residual   median {np.median(arr('relerr_witness')):.2e}  max {arr('relerr_witness').max():.2e}")
    print(f"box-POCS record residual  median {np.median(arr('relerr_box_witness')):.2e}  max {arr('relerr_box_witness').max():.2e}")
    print(f"null-content match donor  median {np.median(arr('null_match')):.2e}")
    print(f"witness PSNR to target {np.median(arr('psnr_wit_to_target')):.1f} dB | to donor {np.median(arr('psnr_wit_to_donor')):.1f} dB")
    json.dump(rows, open(os.path.join(OUT, "witness_certificate.json"), "w"), indent=2)

    xt, xd, u, ub, yt, yu = gallery
    fig = plt.figure(figsize=(14, 3.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.15], wspace=0.22)
    for i, (im, ti) in enumerate([
            (to_rgb(xt), "target scene $x_t$"), (to_rgb(xd), "donor scene $x_d$"),
            (to_rgb(u), f"witness $u$\n$\\|Au-y_t\\|/\\|y_t\\|$={min(arr('relerr_witness')):.0e}"),
            (to_rgb(ub), f"box-legal $u$ (POCS)\nres {min(arr('relerr_box_witness')):.0e}")]):
        a = fig.add_subplot(gs[i]); a.imshow(im); a.axis("off"); a.set_title(ti, fontsize=9)
    # per-pixel spectrum: target vs witness differ under identical snapshot
    axs = fig.add_subplot(gs[4])
    py, px = 128, 128
    wl = np.linspace(450, 650, 28)
    axs.plot(wl, xt[:, py, px].cpu().numpy(), color="#3a6ea5", lw=2, label="target spectrum")
    axs.plot(wl, u[:, py, px].cpu().numpy(), color="#c0392b", lw=2, ls="--", label="witness spectrum")
    axs.set_xlabel("wavelength (nm)"); axs.set_ylabel("reflectance")
    axs.set_title(f"(e) same snapshot,\ndifferent spectra @({py},{px})", fontsize=9)
    axs.legend(fontsize=7)
    for sp in ("top", "right"): axs.spines[sp].set_visible(False)
    fig.suptitle(f"Two different spectral scenes, one identical 23$\\times$ coded snapshot "
                 f"(record residual {min(arr('relerr_witness')):.0e})", fontsize=11.5, y=1.03)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "CASSI_WITNESS." + ext), dpi=170, bbox_inches="tight")
    print("wrote CASSI_WITNESS.* + witness_certificate.json")


if __name__ == "__main__":
    main()
