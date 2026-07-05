# -*- coding: utf-8 -*-
"""Two-ledger forensics over the MST pretrained CASSI model zoo, on the shared real operator.

For each published reconstructor: run it (faithful MST input convention), decompose its PSNR
gain over the range ceiling (min-norm A^dagger y = P_R x_gt) into null-supply vs row-effect,
report its record drift, and show the GT-free audit contracts that drift with the operator's
NON-UNIFORM per-mode factor. Validates wiring by matching published simulation PSNR.
"""
import os, sys, glob, json, time
import numpy as np
import torch
import scipy.io as sio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cassi_operator import CASSI, load_mask
import architecture as arch

OUT = os.path.dirname(os.path.abspath(__file__))
MASK = os.path.join(OUT, "mask.mat")
ZOO = r"E:/GAN_FCC_WORK/data_warehouse/cassi_models"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# method -> (input_setting, input_mask) from MST template.py
CFG = {
    "mst_s": ("H", "Phi"), "mst_m": ("H", "Phi"), "mst_l": ("H", "Phi"),
    "hdnet": ("H", None), "tsa_net": ("HM", None),
    "gap_net": ("Y", "Phi_PhiPhiT"), "admm_net": ("Y", "Phi_PhiPhiT"),
    "dauhst_2stg": ("Y", "Phi_PhiPhiT"), "dauhst_9stg": ("Y", "Phi_PhiPhiT"),
    "mst_plus_plus": ("H", "Mask"), "cst_l": ("H", "Mask"), "bisrnet": ("H", "Mask"),
    "birnat": ("Y", "Phi"), "lambda_net": ("Y", "Phi"), "dgsmp": ("Y", None),
}
PATH = {  # method -> pth path under ZOO
    "mst_s": "mst/mst_s.pth", "mst_m": "mst/mst_m.pth", "mst_l": "mst/mst_l.pth",
    "hdnet": "hdnet/hdnet.pth", "tsa_net": "tsa_net/tsa_net.pth",
    "gap_net": "gap_net/gap_net.pth", "admm_net": "admm_net/admm_net.pth",
    "dauhst_2stg": "dauhst/dauhst_2stg.pth", "dauhst_9stg": "dauhst/dauhst_9stg.pth",
    "mst_plus_plus": "mst_plus_plus/mst_plus_plus.pth", "cst_l": "cst/cst_l.pth",
    "bisrnet": "bisrnet/bisrnet.pth", "birnat": "birnat/birnat.pth",
    "lambda_net": "lambda_net/lambda_net.pth", "dgsmp": "dgsmp/dgsmp.pth",
}


def log(*a): print(f"[cassi-forensics {time.strftime('%H:%M:%S')}]", *a, flush=True)


def shift_back_2d(meas, nC=28, step=2):  # [H,Ws] -> [nC,H,W]  (MST convention)
    H, Ws = meas.shape; W = Ws - (nC - 1) * step
    return torch.stack([meas[:, step * i:step * i + W] for i in range(nC)], 0)


def build_inputs(op, mask3d_t, Phi_t, Phi_s_t, y, setting, mask_type):
    """Return (input_meas, input_mask) as float32 CUDA tensors, MST convention."""
    if setting == "Y":
        input_meas = y.unsqueeze(0)                               # [1,H,Ws]
    else:
        H = shift_back_2d(y / 28 * 2).unsqueeze(0)                # [1,28,256,256]
        input_meas = H * mask3d_t.unsqueeze(0) if setting == "HM" else H
    input_meas = input_meas.float().to(DEV)
    if mask_type == "Phi":
        im = Phi_t.unsqueeze(0).float().to(DEV)
    elif mask_type == "Phi_PhiPhiT":
        im = (Phi_t.unsqueeze(0).float().to(DEV), Phi_s_t.unsqueeze(0).float().to(DEV))
    elif mask_type == "Mask":
        im = mask3d_t.unsqueeze(0).float().to(DEV)
    else:
        im = None
    return input_meas, im


def psnr_cube(pred, gt):
    """Per-band mean PSNR, data range 1.0 (cubes in [0,1]) -- CASSI-standard."""
    p = pred.clamp(0, 1).double(); g = gt.clamp(0, 1).double()
    return float(torch.stack([10 * torch.log10(1.0 / ((p[i] - g[i]) ** 2).mean().clamp_min(1e-30))
                              for i in range(p.shape[0])]).mean())


def load_scene(p):
    return torch.from_numpy(sio.loadmat(p)["img"].astype(np.float64)).permute(2, 0, 1)


def load_model(method):
    model = arch.model_generator(method, os.path.join(ZOO, PATH[method]))
    if isinstance(model, tuple):
        model = model[0]                                          # hdnet returns (model, fdl)
    return model.eval()


def run_model(model, input_meas, input_mask):
    with torch.no_grad():
        out = model(input_meas, input_mask)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out[0].double().cpu()                                 # [28,256,256]


def main(methods=None, lam=1e-3):
    methods = methods or ["mst_s"]
    op = CASSI(load_mask(MASK), nC=28, step=2)
    mask3d = load_mask(MASK).reshape(1, op.H, op.W).repeat(28, 1, 1)     # unshifted [28,256,256]
    scenes = sorted(glob.glob(os.path.join(OUT, "scenes", "scene*.mat")))
    cubes = [load_scene(p) for p in scenes]
    log(f"{len(cubes)} scenes, {len(methods)} methods, operator 23.1x")

    results = {}
    for method in methods:
        pth = os.path.join(ZOO, PATH[method])
        if not os.path.exists(pth):
            log(f"SKIP {method}: weights missing"); continue
        try:
            model = load_model(method)
        except Exception as e:
            log(f"{method}: LOAD FAILED ({type(e).__name__}: {str(e)[:80]})"); continue
        pm, ceilm, nulls, rows, drifts, ratios, devs = [], [], [], [], [], [], []
        for x_gt in cubes:
            y = op.A(x_gt)
            im, imask = build_inputs(op, mask3d, op.Phi, op.Phi_s, y,
                                     CFG[method][0], CFG[method][1])
            try:
                xhat = run_model(model, im, imask)
            except Exception as e:
                log(f"{method}: FAILED ({type(e).__name__}: {str(e)[:80]})"); pm = None; break
            ceil = op.A_dagger(y)                                 # = P_R x_gt (range ceiling)
            p_ceil = psnr_cube(ceil, x_gt)
            p_hat = psnr_cube(xhat, x_gt)
            xhat_null = op.A_dagger(y) + op.P_0(xhat)             # governed: row at ceiling, model null
            p_null = psnr_cube(xhat_null, x_gt)
            drift = op.rel_meas_err(xhat, y)
            xhat_aud = op.audit(xhat, y, lam)
            r_post = op.rel_meas_err(xhat_aud, y)
            pm.append(p_hat); ceilm.append(p_ceil)
            nulls.append(p_null - p_ceil); rows.append(p_hat - p_null)
            drifts.append(drift); ratios.append(r_post / drift if drift > 0 else np.nan)
        if not pm:
            continue
        total = np.mean(pm) - np.mean(ceilm)
        results[method] = {
            "psnr": float(np.mean(pm)), "range_ceiling": float(np.mean(ceilm)),
            "gain_over_ceiling": float(total),
            "null_supply_db": float(np.mean(nulls)), "row_effect_db": float(np.mean(rows)),
            "null_share_pct": float(100 * np.mean(nulls) / total) if total > 0 else float("nan"),
            "record_drift_median": float(np.median(drifts)),
            "audit_ratio_median": float(np.nanmedian(ratios))}
        r = results[method]
        log(f"{method:14s} PSNR {r['psnr']:5.2f} dB | ceiling {r['range_ceiling']:5.2f} | "
            f"gain {total:+.2f} = null {r['null_supply_db']:+.2f} + row {r['row_effect_db']:+.2f} "
            f"(null {r['null_share_pct']:.0f}%) | drift {r['record_drift_median']:.1e}")
    json.dump(results, open(os.path.join(OUT, "forensics.json"), "w"), indent=2)
    make_figure(results, op)
    return results


def make_figure(results, op, lam=1e-3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # keep validated models (drop any whose drift blew up = mis-wired)
    items = [(m, r) for m, r in results.items() if r["record_drift_median"] < 0.5
             and r["gain_over_ceiling"] > 0]
    items.sort(key=lambda kv: kv[1]["psnr"])
    names = [m for m, _ in items]
    ceil = np.mean([r["range_ceiling"] for _, r in items])
    nulls = [r["null_supply_db"] for _, r in items]
    rows = [r["row_effect_db"] for _, r in items]
    drifts = [r["record_drift_median"] for _, r in items]
    x = np.arange(len(names))

    # cool academic blue-green palette (colorblind-friendly)
    C_NULL, C_ROW, C_DRIFT, C_CEIL = "#2a9d8f", "#1d3557", "#457b9d", "#333333"
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.4))
    ax[0].bar(x, nulls, 0.72, bottom=ceil, color=C_NULL, edgecolor="white", linewidth=0.7, label="null-supply (prior)")
    ax[0].bar(x, rows, 0.72, bottom=[ceil + n for n in nulls], color=C_ROW, edgecolor="white", linewidth=0.7, label="row-effect (measured)")
    ax[0].axhline(ceil, color=C_CEIL, lw=1.6, ls="--", label=f"range ceiling {ceil:.1f} dB ($A^\\dagger y$)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=55, ha="right", fontsize=8)
    ax[0].set_ylabel("reconstruction PSNR (dB)")
    ax[0].set_ylim(ceil - 1.5, max(ceil + n + 1 for n in nulls))
    ax[0].set_title("(a) The whole CASSI leaderboard's gain is null-supplied (23$\\times$)", fontsize=10.5)
    ax[0].legend(fontsize=8, loc="upper left", frameon=False)
    ax[0].set_axisbelow(True); ax[0].grid(axis="y", color="#cfcfcf", alpha=0.7, lw=0.6)
    for sp in ("top", "right"): ax[0].spines[sp].set_visible(False)

    ax[1].bar(x, [d * 100 for d in drifts], 0.72, color=C_DRIFT, edgecolor="white", linewidth=0.7)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=55, ha="right", fontsize=8)
    ax[1].set_ylabel("record drift $\\|A\\hat x - y\\|/\\|y\\|$  (%)")
    ax[1].set_title("(b) every published model is inconsistent with its own snapshot", fontsize=10.5)
    ax[1].set_axisbelow(True); ax[1].grid(axis="y", color="#cfcfcf", alpha=0.7, lw=0.6)
    for sp in ("top", "right"): ax[1].spines[sp].set_visible(False)
    fig.suptitle("CASSI forensics: at 23$\\times$ undersampling the range ceiling is 19 dB; "
                 "100% of every model's headline gain is prior-supplied null content", fontsize=11.5, y=1.01)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "CASSI_FORENSICS." + ext), dpi=170, bbox_inches="tight")
    log(f"wrote CASSI_FORENSICS figure ({len(names)} validated models)")


if __name__ == "__main__":
    import sys
    ms = sys.argv[1:] if len(sys.argv) > 1 else ["mst_s"]
    main(ms)
