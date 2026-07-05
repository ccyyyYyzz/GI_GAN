# -*- coding: utf-8 -*-
"""Feasible-but-wrong witness on REAL fastMRI single-coil knee k-space.

Two different knees share ONE undersampled record: u = A^dagger y_target + P_0 x_donor
reproduces the target's k-space record exactly (to ~1e-13) while its null-space content
is the donor's. Any record-consistency test loose enough to accept the true target
accepts u. Produces:
  (1) numeric certificate over N cross-volume pairs (record residual, null match)
  (2) a magnitude-legality (box-constrained POCS) variant
  (3) a gallery figure: target GT | donor GT | witness u | |A u - y| vs |A x_tgt - y|
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mf_operator import MaskedFourier, equispaced_mask, random_mask, center_crop
import data as D

OUT = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.complex128


def norm(t): return float(torch.linalg.vector_norm(t).item())


def psnr(a, b):
    a = a / a.max(); b = b / (b.max() + 1e-12)
    mse = float(((a - b) ** 2).mean())
    return -10 * np.log10(max(mse, 1e-30))


def box_pocs(op, x_donor, y_target, iters=200):
    """Magnitude-legality variant: alternate exact fiber projection with a magnitude
    clip (scale complex phase, clamp magnitude into the target's dynamic range)."""
    lo, hi = 0.0, None  # magnitude lower bound 0; upper set from target scale
    u = op.witness(x_donor, y_target)
    hi = float(center_crop(op.A_dagger(y_target).abs(), (320, 320)).max().item()) * 1.5
    for _ in range(iters):
        mag = u.abs().clamp(lo, hi)
        phase = torch.exp(1j * u.angle())
        u = mag * phase                       # box (magnitude) projection
        u = u - op.A_dagger(op.A(u) - y_target)  # exact fiber projection
    return u


def main(accel=8, cf=0.04, n_pairs=12, seed=0):
    vols = D.list_volumes(dominant_shape=True)
    assert len(vols) >= 2, "need val data downloaded/extracted first"
    rng = np.random.default_rng(seed)
    H, W = D.load_slice(vols[0])[0].shape
    print(f"{len(vols)} volumes at common shape {H}x{W}")
    mask = equispaced_mask(W, acceleration=accel, center_fraction=cf)
    op = MaskedFourier(mask, (H, W))
    print(f"operator: {H}x{W}, sampling {op.sampling_rate:.3f} (accel ~{1/op.sampling_rate:.1f}x)")

    idx = rng.permutation(len(vols))
    rows = []
    gallery = None
    for p in range(min(n_pairs, len(idx) // 2)):
        vt, vd = vols[idx[2 * p]], vols[idx[2 * p + 1]]
        xt, _, _ = D.load_slice(vt)
        xd, _, _ = D.load_slice(vd)
        yt = op.A(xt)
        u = op.witness(xd, yt)
        r_true = op.rel_meas_err(xt, yt)          # target's own record residual (=0 noiseless)
        r_wit = op.rel_meas_err(u, yt)            # witness record residual
        null_match = norm(op.P_0(u) - op.P_0(xd)) / (norm(op.P_0(xd)) + 1e-30)
        ub = box_pocs(op, xd, yt)
        r_box = op.rel_meas_err(ub, yt)
        # semantic distance on magnitude (cropped)
        gt_t = center_crop(xt.abs(), (320, 320))
        gt_d = center_crop(xd.abs(), (320, 320))
        wit_m = center_crop(u.abs(), (320, 320))
        box_m = center_crop(ub.abs(), (320, 320))
        rows.append(dict(pair=[os.path.basename(vt), os.path.basename(vd)],
                         relerr_true=r_true, relerr_witness=r_wit,
                         relerr_box_witness=r_box, null_match=null_match,
                         psnr_wit_to_target=psnr(wit_m, gt_t),
                         psnr_wit_to_donor=psnr(wit_m, gt_d)))
        if gallery is None:
            gallery = (gt_t, gt_d, wit_m, box_m,
                       center_crop((op.A(u) - yt).abs(), (320, 320)),
                       center_crop((op.A(xt) - yt).abs(), (320, 320)))

    arr = lambda k: np.array([r[k] for r in rows])
    print(f"\n=== witness certificate over {len(rows)} cross-volume pairs (accel {1/op.sampling_rate:.1f}x) ===")
    print(f"  target own record residual   median {np.median(arr('relerr_true')):.2e}")
    print(f"  WITNESS  record residual     median {np.median(arr('relerr_witness')):.2e}  max {arr('relerr_witness').max():.2e}")
    print(f"  box-POCS record residual     median {np.median(arr('relerr_box_witness')):.2e}  max {arr('relerr_box_witness').max():.2e}")
    print(f"  null-content match to donor  median {np.median(arr('null_match')):.2e}")
    print(f"  witness PSNR to target       median {np.median(arr('psnr_wit_to_target')):.1f} dB")
    print(f"  witness PSNR to donor        median {np.median(arr('psnr_wit_to_donor')):.1f} dB")

    json.dump(rows, open(os.path.join(OUT, f"witness_certificate_{int(round(1/op.sampling_rate))}x.json"), "w"), indent=2)

    # gallery figure
    gt_t, gt_d, wit_m, box_m, res_w, res_t = gallery
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    for a, im, ti in zip(
        ax,
        [gt_t, gt_d, wit_m, box_m],
        ["target knee $x_t$ (GT)", "donor knee $x_d$ (GT)",
         f"witness $u=A^\\dagger y_t+P_0 x_d$\n$\\|Au-y_t\\|/\\|y_t\\|$={arr('relerr_witness').min():.1e}",
         f"box-legal witness (POCS)\nresidual {arr('relerr_box_witness').min():.1e}"]):
        a.imshow(im.cpu().numpy(), cmap="gray"); a.set_title(ti, fontsize=9); a.axis("off")
    fig.suptitle(f"Two different knees, one identical {1/op.sampling_rate:.0f}$\\times$-undersampled record "
                 f"— the measurement cannot tell $x_t$ from $u$", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"WITNESS_MRI_{int(round(1/op.sampling_rate))}x.{ext}"), dpi=170, bbox_inches="tight")
    print(f"\nwrote WITNESS_MRI figure + certificate json")


if __name__ == "__main__":
    main()
