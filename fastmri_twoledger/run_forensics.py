# -*- coding: utf-8 -*-
"""Two-ledger forensics + GT-free record-consistency audit on real fastMRI single-coil knee.

For N validation volumes at a locked acceleration:
  * range ceiling = zero-filled magnitude PSNR (zero-filled = P_R x_gt exactly).
  * decompose each reconstructor's magnitude gain over the ceiling into
        null-supply  (A^dagger y + P_0 x_method, row held at ceiling)
        row-effect   (residual change from moving the measured component off A^dagger y)
  * GT-free audit Pi_lambda contracts each method's measured residual by lambda/(lambda+1),
    needing only (A, y, lambda) -- shown on the non-data-consistent U-Net.
Writes forensics_<accel>x.json + FORENSICS_MRI figure.
"""
import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mf_operator import MaskedFourier, equispaced_mask, random_mask, center_crop
import data as D
import reconstructors as R

OUT = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a): print(f"[forensics {time.strftime('%H:%M:%S')}]", *a, flush=True)


def main(accel=4, cf=0.08, n_vol=30, seed=0, mask_type="random"):
    vols = D.list_volumes(dominant_shape=True)
    assert len(vols) >= 2, "val data not ready"
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(vols))[:n_vol]
    H, W = D.load_slice(vols[0])[0].shape
    # fastMRI single-coil U-Net leaderboard was trained on RANDOM Cartesian masks at
    # 4x/8x; use the matching mask family so the audited model is in-distribution.
    if mask_type == "random":
        mask = random_mask(W, acceleration=accel, center_fraction=cf, seed=1234)
    else:
        mask = equispaced_mask(W, acceleration=accel, center_fraction=cf)
    op = MaskedFourier(mask, (H, W))
    log(f"operator {H}x{W}, sampling {op.sampling_rate:.3f} (accel ~{1/op.sampling_rate:.1f}x), {len(pick)} volumes")

    rec = {m: [] for m in ["ceiling", "wavelet_cs", "unet", "unet_dc"]}
    audit = {"unet_pre": [], "unet_post": [], "ratio": [], "contra_dev": []}
    null_frac = {"wavelet_cs": [], "unet": [], "unet_dc": []}
    drift = {"unet": [], "unet_dc": []}
    lam = 1e-3

    for vi, pi in enumerate(pick):
        x_gt, esc, _ = D.load_slice(vols[pi])
        gt_mag = esc if esc is not None else center_crop(x_gt.abs(), (320, 320))
        y = op.A(x_gt)

        x_zf = R.zero_filled(op, y)
        ceil = R.psnr_mag(x_zf, gt_mag)
        rec["ceiling"].append(ceil)

        # wavelet CS (data-consistent, fills null via sparsity)
        x_cs = R.wavelet_cs(op, y)
        p_cs = R.psnr_mag(x_cs, gt_mag)
        dc_cs = R.data_consistent(op, x_cs, y)
        p_cs_null = R.psnr_mag(op.A_dagger(y) + op.P_0(x_cs), gt_mag)
        rec["wavelet_cs"].append(p_cs)
        null_frac["wavelet_cs"].append((p_cs_null - ceil, p_cs - p_cs_null))

        # U-Net (magnitude post-processor; NOT data-consistent)
        _, x_un = R.unet(op, y, device=DEV)
        p_un = R.psnr_mag(x_un, gt_mag)
        x_un_null = op.A_dagger(y) + op.P_0(x_un)     # governed: U-Net null + exact record
        p_un_null = R.psnr_mag(x_un_null, gt_mag)
        rec["unet"].append(p_un)
        null_frac["unet"].append((p_un_null - ceil, p_un - p_un_null))
        drift["unet"].append(op.rel_meas_err(x_un, y))

        # data-consistent (governed) U-Net: keep its null content, restore exact record
        p_un_dc = R.psnr_mag(x_un_null, gt_mag)
        rec["unet_dc"].append(p_un_dc)
        null_frac["unet_dc"].append((p_un_dc - ceil, 0.0))   # gain is 100% null by construction
        drift["unet_dc"].append(op.rel_meas_err(x_un_null, y))

        # GT-free audit on the U-Net (data-inconsistent) reconstruction
        r_pre = op.rel_meas_err(x_un, y)
        x_un_aud = op.audit(x_un, y, lam)
        r_post = op.rel_meas_err(x_un_aud, y)
        audit["unet_pre"].append(r_pre)
        audit["unet_post"].append(r_post)
        audit["ratio"].append(r_post / r_pre if r_pre > 0 else np.nan)
        # per-mode contraction check: residual should scale by exactly lam/(lam+1)
        audit["contra_dev"].append(abs((r_post / r_pre if r_pre > 0 else 0) - lam / (lam + 1)))
        if vi % 5 == 0:
            log(f"vol {vi+1}/{len(pick)}: ceil {ceil:.2f} | CS {p_cs:.2f} | UNet {p_un:.2f} dB | "
                f"UNet record-drift {r_pre:.2e} -> {r_post:.2e}")

    def stats(a): a = np.array(a); return float(np.mean(a)), float(np.std(a))
    ceil_m, ceil_s = stats(rec["ceiling"])
    summary = {"accel": 1 / op.sampling_rate, "sampling_rate": op.sampling_rate,
               "n_vol": len(pick), "lambda": lam,
               "range_ceiling_psnr": [ceil_m, ceil_s]}
    log("\n===== FORENSICS SUMMARY =====")
    log(f"range ceiling (zero-filled = P_R x_gt): {ceil_m:.2f} +/- {ceil_s:.2f} dB")
    for m in ["wavelet_cs", "unet", "unet_dc"]:
        pm, ps = stats(rec[m])
        ns = np.array(null_frac[m])
        null_gain, row_gain = ns[:, 0].mean(), ns[:, 1].mean()
        total = pm - ceil_m
        summary[m] = {"psnr": [pm, ps], "total_gain_db": total,
                      "null_supply_db": float(null_gain), "row_effect_db": float(row_gain),
                      "null_share_pct": float(100 * null_gain / total) if total > 0 else np.nan}
        if m in drift:
            summary[m]["record_drift_median"] = float(np.median(drift[m]))
        dtxt = f" | record drift {np.median(drift[m]):.1e}" if m in drift else ""
        log(f"{m:11s}: {pm:.2f} dB | gain over ceiling {total:+.2f} dB = "
            f"null-supply {null_gain:+.2f} + row-effect {row_gain:+.2f}  "
            f"(null share {summary[m]['null_share_pct']:.0f}%){dtxt}")
    r_pre_m = np.median(audit["unet_pre"]); r_post_m = np.median(audit["unet_post"])
    summary["audit"] = {"unet_record_drift_pre_median": float(r_pre_m),
                        "unet_record_drift_post_median": float(r_post_m),
                        "contraction_ratio_median": float(np.median(audit["ratio"])),
                        "contraction_dev_max": float(np.max(audit["contra_dev"])),
                        "target_ratio": lam / (lam + 1)}
    log(f"GT-free audit on U-Net: record drift {r_pre_m:.2e} -> {r_post_m:.2e} "
        f"(ratio {np.median(audit['ratio']):.2e}, target {lam/(lam+1):.2e}, "
        f"max dev {np.max(audit['contra_dev']):.1e})")

    json.dump(summary, open(os.path.join(OUT, f"forensics_{int(round(1/op.sampling_rate))}x.json"), "w"), indent=2)

    # figure: (a) PSNR bars vs ceiling with null/row split ; (b) U-Net record-drift contraction
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    methods = ["wavelet_cs", "unet", "unet_dc"]
    xpos = np.arange(len(methods))
    nulls = [summary[m]["null_supply_db"] for m in methods]
    rows_ = [summary[m]["row_effect_db"] for m in methods]
    ax[0].axhline(0, color="#888", lw=1)
    ax[0].bar(xpos, nulls, 0.5, color="#c0392b", label="null-supply (prior)")
    ax[0].bar(xpos, rows_, 0.5, bottom=nulls, color="#3a6ea5", label="row-effect (measured)")
    ax[0].set_xticks(xpos); ax[0].set_xticklabels(["wavelet-CS", "U-Net\n(raw SOTA)", "U-Net\n(governed DC)"])
    ax[0].set_ylabel("PSNR gain over range ceiling (dB)")
    ax[0].set_title(f"(a) Headline gain is null-supplied  ({1/op.sampling_rate:.0f}$\\times$)")
    ax[0].legend(fontsize=8)
    for sp in ("top", "right"): ax[0].spines[sp].set_visible(False)

    ax[1].scatter(audit["unet_pre"], audit["unet_post"], s=16, color="#c0392b", alpha=0.6)
    lo = min(audit["unet_post"]); hi = max(audit["unet_pre"])
    xs = np.array([lo, hi])
    ax[1].plot(xs, xs * lam / (lam + 1), "--", color="#333", label=f"$\\lambda/(\\lambda+1)$={lam/(lam+1):.0e}")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("U-Net record drift $\\|A\\hat x-y\\|/\\|y\\|$ (pre-audit)")
    ax[1].set_ylabel("post-audit")
    ax[1].set_title("(b) GT-free audit contracts every mode exactly")
    ax[1].legend(fontsize=8)
    for sp in ("top", "right"): ax[1].spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"FORENSICS_MRI_{int(round(1/op.sampling_rate))}x.{ext}"), dpi=170, bbox_inches="tight")
    log("wrote forensics json + FORENSICS_MRI figure")


if __name__ == "__main__":
    main(accel=4, cf=0.08)
    main(accel=8, cf=0.04)
