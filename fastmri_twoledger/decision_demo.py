# -*- coding: utf-8 -*-
"""Certified-vs-null decomposition of pathology findings on fastMRI+ (the clinical leg).

Honest finding: at clinical MRI acceleration the auto-calibration (ACS) lines certify the
COARSE presence of a lesion (low-frequency energy), but the fine detail that a radiologist
reads -- margins, texture, internal structure -- lives largely in the null ledger. We
quantify this over every held fastMRI+ pathology box, then show a governed null-space edit
that changes a lesion's fine appearance while the k-space record stays byte-identical.

Two deliverables:
  (1) aggregate: null-energy fraction inside pathology boxes, FULL vs FINE-DETAIL (high-pass),
      at 8x and 16x, over all annotations we hold.
  (2) illustration: true lesion vs a donor-null witness on the SAME record (A x = y to ~1e-16),
      with the box difference and a fine-detail detector that moves while the record does not.
Writes decision_demo.json + DECISION_MRI figure.
"""
import os, sys, json, csv, time
import numpy as np
import torch
import torch.nn.functional as Fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mf_operator import MaskedFourier, random_mask, center_crop
import data as D

OUT = os.path.dirname(os.path.abspath(__file__))
ANN = r"E:/GAN_FCC_WORK/data_warehouse/fastmri_knee_sc/knee_annotations.csv"


def log(*a): print(f"[decision {time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_annotations():
    rows = []
    with open(ANN, newline="") as f:
        for r in csv.DictReader(f):
            if r["x"] in ("", "-1") or r["label"] == "":
                continue
            rows.append(r)
    return rows


def _box(r):
    return int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"])


def detail(x):
    """High-pass (fine detail) = image minus a 7x7 box-blur, complex."""
    k = torch.ones(1, 1, 7, 7, dtype=torch.float64) / 49.0
    lp = (Fn.conv2d(x.real[None, None], k, padding=3)[0, 0]
          + 1j * Fn.conv2d(x.imag[None, None], k, padding=3)[0, 0])
    return x - lp


def aggregate_null_fraction(vols, accel, cf):
    have = {os.path.splitext(os.path.basename(v))[0]: v for v in vols}
    ann = [r for r in load_annotations() if r["file"] in have]
    full, det = [], []
    for r in ann:
        try:
            sl = int(r["slice"]); x0, y0, w, h = _box(r)
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        x_gt, _, _ = D.load_slice(have[r["file"]], sl=sl)
        H, W = x_gt.shape
        op = MaskedFourier(random_mask(W, acceleration=accel, center_fraction=cf, seed=7), (H, W))
        P0 = op.P_0(x_gt)
        fm = center_crop(x_gt.abs(), (320, 320)); nm = center_crop(P0.abs(), (320, 320))
        df = center_crop(detail(x_gt).abs(), (320, 320)); dn = center_crop(detail(P0).abs(), (320, 320))
        bf = float((fm[y0:y0 + h, x0:x0 + w] ** 2).sum()); bn = float((nm[y0:y0 + h, x0:x0 + w] ** 2).sum())
        gf = float((df[y0:y0 + h, x0:x0 + w] ** 2).sum()); gn = float((dn[y0:y0 + h, x0:x0 + w] ** 2).sum())
        full.append(bn / (bf + 1e-30)); det.append(gn / (gf + 1e-30))
    return np.array(full), np.array(det), len(full)


def main(accel=8, cf=0.04):
    vols = D.list_volumes()
    log(f"{len(vols)} held volumes")

    # (1) aggregate null-energy fraction in pathology boxes
    agg = {}
    for a, c in [(8, 0.04), (16, 0.02)]:
        full, det, n = aggregate_null_fraction(vols, a, c)
        agg[a] = {"n": n, "full_median": float(np.median(full)), "detail_median": float(np.median(det))}
        log(f"accel {a:2d}x over {n} pathology boxes: null-energy median "
            f"full {np.median(full)*100:.1f}%  |  fine-detail {np.median(det)*100:.1f}%")

    # (2) illustration: true lesion vs donor-null witness on the SAME record
    have = {os.path.splitext(os.path.basename(v))[0]: v for v in vols}
    ann = [r for r in load_annotations() if r["file"] in have]
    # choose an annotation whose box carries the most fine detail (clearest illustration)
    best = None
    for r in ann:
        try:
            sl = int(r["slice"]); x0, y0, w, h = _box(r)
        except ValueError:
            continue
        if w < 12 or h < 12:
            continue
        x_gt, _, _ = D.load_slice(have[r["file"]], sl=sl)
        d = center_crop(detail(x_gt).abs(), (320, 320))
        score = float((d[y0:y0 + h, x0:x0 + w] ** 2).sum())
        if best is None or score > best[0]:
            best = (score, have[r["file"]], sl, (x0, y0, w, h), r["label"])
    _, vol, sl, box, label = best
    x0, y0, w, h = box
    log(f"illustration: {os.path.basename(vol)} slice {sl} '{label}' box {box}")

    x_gt, _, _ = D.load_slice(vol, sl=sl)
    H, W = x_gt.shape
    op = MaskedFourier(random_mask(W, acceleration=accel, center_fraction=cf, seed=7), (H, W))
    y = op.A(x_gt)
    same = D.list_volumes(match_shape=(H, W))
    donor = next(p for p in same if os.path.basename(p) != os.path.basename(vol))
    x_don, _, _ = D.load_slice(donor, sl=(sl if sl < 30 else None))
    direction = op.P_0(x_don) - op.P_0(x_gt)      # exactly in ker A
    base = op.A_dagger(y) + op.P_0(x_gt)          # == x_gt

    def fine_contrast(mag):
        d = detail(mag.to(torch.complex128)).abs()
        return float(d[y0:y0 + h, x0:x0 + w].std())

    Bs = np.linspace(0.0, 1.0, 11)
    scores, resids, panels = [], [], {}
    for B in Bs:
        xB = base + B * direction
        resids.append(op.rel_meas_err(xB, y))
        magB = center_crop(xB.abs(), (320, 320))
        scores.append(fine_contrast(magB))
        if abs(B) < 1e-9: panels["true (B=0)"] = magB
        if np.isclose(B, 1.0): panels["witness (B=1)"] = magB
    res_max = float(np.max(resids))
    swing = 100 * (scores[-1] - scores[0]) / (scores[0] + 1e-30)
    log(f"record residual across all B: max {res_max:.2e}")
    log(f"fine-detail box contrast: true {scores[0]:.3e} -> witness {scores[-1]:.3e} ({swing:+.0f}% at identical record)")

    summary = {"aggregate": agg, "illustration": {
        "file": os.path.basename(vol), "slice": sl, "box": box, "label": label,
        "donor": os.path.basename(donor), "accel": 1 / op.sampling_rate,
        "record_residual_max_over_B": res_max,
        "fine_contrast_true": scores[0], "fine_contrast_witness": scores[-1],
        "fine_contrast_swing_pct": swing,
        "B_grid": [float(b) for b in Bs], "detector": [float(s) for s in scores],
        "record_residual": [float(r) for r in resids]}}
    json.dump(summary, open(os.path.join(OUT, "decision_demo.json"), "w"), indent=2)

    # figure: true | witness | box difference | aggregate null-fraction bars
    fig = plt.figure(figsize=(13.5, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2], wspace=0.28)
    t = panels["true (B=0)"]; wt = panels["witness (B=1)"]
    diff = (t - wt).abs()
    for i, (im, ti, cmap) in enumerate([
            (t, f"true '{label}'", "gray"), (wt, "witness (same record)", "gray"),
            (diff, "|difference| (null only)", "magma")]):
        a = fig.add_subplot(gs[i]); a.imshow(im.cpu().numpy(), cmap=cmap); a.axis("off")
        a.set_title(ti, fontsize=9)
        a.add_patch(mp.Rectangle((x0, y0), w, h, ec="#39d353", fc="none", lw=1.6))
    axb = fig.add_subplot(gs[3])
    accs = list(agg.keys()); xp = np.arange(len(accs))
    axb.bar(xp - 0.2, [agg[a]["full_median"] * 100 for a in accs], 0.4, color="#3a6ea5", label="full box")
    axb.bar(xp + 0.2, [agg[a]["detail_median"] * 100 for a in accs], 0.4, color="#c0392b", label="fine detail")
    axb.set_xticks(xp); axb.set_xticklabels([f"{a}x" for a in accs])
    axb.set_ylabel("null-energy in pathology box (%)")
    axb.set_title(f"(d) fine detail is null-supplied\n(n={agg[accs[0]]['n']} boxes)", fontsize=9)
    axb.legend(fontsize=7)
    for sp in ("top", "right"): axb.spines[sp].set_visible(False)
    fig.suptitle(f"The bucket certifies the coarse lesion, not its detail — witness and truth share the exact "
                 f"{1/op.sampling_rate:.0f}$\\times$ record (residual {res_max:.0e})", fontsize=11, y=1.03)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "DECISION_MRI." + ext), dpi=170, bbox_inches="tight")
    log("wrote decision_demo.json + DECISION_MRI figure")


if __name__ == "__main__":
    main()
