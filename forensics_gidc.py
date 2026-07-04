"""Forensics case study, target 2: GIDC (Wang et al., Light: Science & Applications 11, 1 (2022)).

REAL experimental data audit (GT-free by necessity -- the shipped scene has no ground truth, so per the
pre-registration its reported gains are NOT MSE-decomposable; what IS auditable without GT):
  1. The physical operator itself: 1200 measured speckle patterns (nonnegative, calibrated) + integer
     photon-count buckets. Characterize rank/spectrum/certificate at the actual shot-noise level, and the
     row-sum statistics (nonnegative speckle = DGI's proper regime, unlike our signed fusion operator).
  2. GT-free ledgers of their reconstructions (their GIDC run UNTOUCHED in TF1; 8-bit BMP outputs as
     released, quantization +-0.002 disclosed): null fraction of the output and measurement consistency
     against the actual recorded buckets.
  3. The super-resolution claim, located: 'resolution beyond the diffraction limit' content lives, by
     definition, in null(A) of the patterns actually measured -- we meter exactly how much of the GIDC
     output's energy is unverifiable from their own record.
Convention notes: their pipeline standardizes A and y (whole-tensor affine); their saved x_out uses
order='F' reshape (we test both orientations and keep the lower-residual one, documented).
"""
from __future__ import annotations
import json, time
import numpy as np
import scipy.io as sio
import torch
from PIL import Image

REPO = r"E:\ns_mc_gan_gi_code_fcc_phase1"
EXT = REPO + r"\external_audit\GIDC"
OUT = REPO + r"\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 0.1  # their demo setting


def log(*a): print(f"[forensics {time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    log("device =", DEV)
    d = sio.loadmat(EXT + r"\data.mat")
    pats = d["patterns"].astype(np.float64)            # [64,64,1200]
    y_all = d["measurements"].astype(np.float64).reshape(-1)   # photon counts
    n = 64 * 64
    m = int(round(n * SR))                             # 410, their demo
    P = pats[:, :, :m]
    A = torch.from_numpy(P.reshape(n, m).T.copy()).double().to(DEV)   # rows = C-order flattened patterns
    y = torch.from_numpy(y_all[:m]).to(DEV)

    # ---- 1. the real operator ----
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    tol = S[0] * max(A.shape) * torch.finfo(torch.float64).eps
    rank = int((S > tol).sum()); rank1e3 = int((S > 1e-3 * S[0]).sum())
    # shot-noise level: relative sigma per bucket ~ 1/sqrt(count)
    rel_sigma = float(np.mean(1.0 / np.sqrt(y_all[:m])))
    # certificate on the normalized operator: gains lambda/(lambda+sigma^2) with sigma scaled to row norms
    An = A / A.norm(dim=1, keepdim=True)
    Sn = torch.linalg.svdvals(An)
    lam = Sn ** 2
    gain = (lam / (lam + rel_sigma ** 2)).cpu().numpy()
    ones = torch.ones(n, device=DEV, dtype=torch.float64)
    R = A @ ones
    char = {"m": m, "n": n, "sampling_pct": 100.0 * m / n,
            "rank_machine": rank, "rank_rel1e-3": rank1e3,
            "condition": float(S[0] / S[S > tol].min()),
            "row_sum_mean": float(R.mean()), "row_sum_min": float(R.min()),
            "row_sum_rel_spread": float(R.std() / R.mean()),
            "photon_counts_mean": float(y_all[:m].mean()), "rel_shot_noise": rel_sigma,
            "cert_gain_ge_0.9": int((gain >= 0.9).sum()), "null_dim": int(n - rank)}
    log(f"REAL operator: m={m} ({100*m/n:.1f}%), rank={rank} (rank@1e-3={rank1e3}), cond={char['condition']:.2e}")
    log(f"  nonnegative speckle: row sums {char['row_sum_mean']:.1f}+-{char['row_sum_rel_spread']*100:.1f}% "
        f"(min {char['row_sum_min']:.1f}) -> DGI's proper regime (vs our signed operator where DGI collapses)")
    log(f"  photon counts ~{char['photon_counts_mean']:.2e} -> rel shot noise {rel_sigma:.1e}; "
        f"certificate: {char['cert_gain_ge_0.9']}/{rank} modes gain>=0.9; null dim {char['null_dim']}")

    Vr = Vh[:rank]
    def P_R(v): return (v @ Vr.T) @ Vr
    def P_0(v): return v - P_R(v)

    # ---- 2. GT-free ledgers ----
    # recompute their DGI in float (their exact in-script formula)
    cnt = 0; SI = np.zeros((64, 64)); B_av = 0.0; R_av = 0.0; RI = np.zeros((64, 64))
    for i in range(m):
        p = pats[:, :, i]; b = y_all[i]; cnt += 1
        SI = (SI * (cnt - 1) + p * b) / cnt
        B_av = (B_av * (cnt - 1) + b) / cnt
        R_av = (R_av * (cnt - 1) + p.sum()) / cnt
        RI = (RI * (cnt - 1) + p.sum() * p) / cnt
    DGI = SI - B_av / R_av * RI

    def ledger(img2d, name):
        """GT-free: null fraction + consistency (affine-fit, orientation-resolved)."""
        best = None
        for tag, xm in [("asis", img2d), ("T", img2d.T)]:
            xv = torch.from_numpy(np.ascontiguousarray(xm, dtype=np.float64)).to(DEV).reshape(-1)
            # affine-fit consistency: min_{a,b} ||a A x + b 1 - y|| / ||y - mean||  (their pipeline is affine-normalized)
            Ax = A @ xv
            M = torch.stack([Ax, torch.ones_like(Ax)], 1)
            coef = torch.linalg.lstsq(M, y.unsqueeze(1)).solution.reshape(-1)
            res = float((M @ coef - y).norm() / (y - y.mean()).norm())
            if best is None or res < best[1]:
                best = (tag, res, xv)
        tag, res, xv = best
        nf = float(P_0(xv).norm() ** 2 / (xv.norm() ** 2))
        # centered null fraction (remove mean, which is heavily measured): structure-level split
        xc = xv - xv.mean()
        nfc = float(P_0(xc).norm() ** 2 / (xc.norm() ** 2))
        log(f"  {name:16s} [{tag}] consist-relerr(affine)={res:.3f}  null-frac={nf:.3f}  null-frac(centered)={nfc:.3f}")
        return {"orientation": tag, "consist_relerr_affine": res, "null_fraction": nf, "null_fraction_centered": nfc}

    rows = {}
    rows["DGI_float"] = ledger(DGI, "DGI (recomputed)")
    for step in (0, 200):
        bmp = np.array(Image.open(EXT + rf"\results\GIDC_{m}_{step}.bmp"), dtype=np.float64) / 255.0
        rows[f"GIDC_step{step}"] = ledger(bmp, f"GIDC step {step}")

    out = {"target": "FeiWang0824/GIDC (Light Sci Appl 11, 1 (2022)) -- REAL experimental data "
                      "(1200 measured speckle patterns + photon-count buckets); their demo SR=0.1 (m=410), "
                      "GIDC run untouched in TF1; outputs are their released 8-bit BMPs (quantization ~0.002)",
           "gt_status": "NO ground truth for the experimental scene -> reported gains NOT MSE-decomposable "
                        "(pre-registered exclusion); all quantities here are GT-free",
           "operator": char, "ledgers": rows,
           "superres_note": "'Resolution beyond the diffraction limit' content lives by definition in "
                            "null(A) of the measured patterns; null_fraction_centered of the GIDC output "
                            "meters exactly how much of its structure is unverifiable from their record."}
    with open(OUT + r"\forensics_gidc.json", "w") as f:
        json.dump(out, f, indent=2)
    log("wrote forensics_gidc.json")


if __name__ == "__main__":
    main()
