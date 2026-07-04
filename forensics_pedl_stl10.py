"""Forensics case study, target 1: physics-enhanced deep learning SPI (Wang et al., Photon. Res. 2022).

Audits the RELEASED artifacts of FeiWang0824/physics-driven-fine-tuning (cloned to external_audit/):
the learned 1024-pattern operator (trained_stl10_patterns_1024_Unet_wDGI_64.mat) IS the exact sensing
matrix A, released as plain data. Their sim is noiseless with y = <pattern_i, im> then whole-vector
standardization (an affine map that leaves row(A), and hence P_R/P_0, untouched).

Stage 1 (this script):
  1. EXTRACT A from their released .mat; verify the measurement convention by reproducing their shipped
     y (stl10_sim.mat) from their shipped GT to float tolerance.  [own-operator condition: satisfied]
  2. CHARACTERIZE their operator: rank, singular spectrum, effective sampling, certificate profile
     lambda/(lambda+sigma^2), null dimension. A learned operator's rows are correlated -- the exact rank
     (vs the nominal m=1024) is itself a finding.
  3. DECOMPOSE their shipped reconstruction (dgi_r: DGI with learned patterns -- the physics baseline of
     their pipeline) through exact P_R/P_0 ON THEIR OPERATOR: row error vs null error vs the min-norm
     baseline P_R GT; Bhadra-style split of injected null content into prior-correct vs hallucinated.
Their DNN / fine-tuned outputs need their TF1 env (Stage 2); everything here uses released data only.
"""
from __future__ import annotations
import json, time
import numpy as np
import scipy.io as sio
import torch

REPO = r"E:\ns_mc_gan_gi_code_fcc_phase1"
EXT = REPO + r"\external_audit\physics-driven-fine-tuning"
OUT = REPO + r"\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOMINAL_SIGMA = 0.01


def log(*a): print(f"[forensics {time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    log("device =", DEV)
    # ---- 1. extract A and verify the convention against their shipped record ----
    pat = sio.loadmat(EXT + r"\model\trained_stl10_patterns_1024_Unet_wDGI_64.mat")["trained_patterns"]
    log("patterns shape:", pat.shape, "dtype:", pat.dtype)          # [64,64,1,1024]
    m = pat.shape[-1]; n = pat.shape[0] * pat.shape[1]
    A = torch.from_numpy(pat[:, :, 0, :].reshape(n, m).T.copy()).double().to(DEV)   # [1024,4096], C-order = conv2d order
    sim = sio.loadmat(EXT + r"\data\stl10_sim.mat")
    y_ship = torch.from_numpy(sim["y"].astype(np.float64)).reshape(-1).to(DEV)
    gt = torch.from_numpy(sim["GT"].astype(np.float64)).to(DEV)
    dgi = torch.from_numpy(sim["dgi_r"].astype(np.float64)).to(DEV)
    x = gt.reshape(-1)
    y_raw = A @ x
    y_std = (y_raw - y_raw.mean()) / y_raw.std(correction=0)         # their tf.nn.moments = population std
    conv_err = float((y_std - y_ship).abs().max())
    log(f"convention check: max|standardize(A @ GT) - shipped y| = {conv_err:.3e}  "
        f"({'OK - exact operator confirmed' if conv_err < 1e-3 else 'MISMATCH - investigate'})")

    # ---- 2. characterize their operator ----
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    tol = S[0] * max(A.shape) * torch.finfo(torch.float64).eps
    rank = int((S > tol).sum())
    # numerically effective rank at 1e-3 relative
    rank_1e3 = int((S > 1e-3 * S[0]).sum())
    lam = (S ** 2)
    gain = (lam / (lam + NOMINAL_SIGMA ** 2)).cpu().numpy()
    char = {"m_nominal": int(m), "n": int(n), "rank_machine": rank, "rank_rel1e-3": rank_1e3,
            "sv_max": float(S[0]), "sv_min_nonzero": float(S[S > tol].min()),
            "condition": float(S[0] / S[S > tol].min()),
            "nominal_sampling_pct": 100.0 * m / n, "effective_sampling_pct_rel1e-3": 100.0 * rank_1e3 / n,
            "cert_gain_ge_0.9_at_sigma0.01": int((gain >= 0.9).sum()),
            "null_dim_machine": int(n - rank)}
    log(f"operator: nominal m={m} ({100*m/n:.1f}%), machine rank={rank}, rank@1e-3={rank_1e3} "
        f"({100*rank_1e3/n:.1f}% effective), cond={char['condition']:.2e}")
    log(f"certificate profile @sigma={NOMINAL_SIGMA}: {char['cert_gain_ge_0.9_at_sigma0.01']}/{rank} modes gain>=0.9; "
        f"null dim = {char['null_dim_machine']}")

    # exact projectors from the SVD (rank-truncated)
    Vr = Vh[:rank]                                                    # [rank, n]
    def P_R(v): return (v @ Vr.T) @ Vr
    def P_0(v): return v - P_R(v)

    # ---- 3. decompose their shipped DGI-with-learned-patterns result ----
    def split(xin, name):
        e = xin.reshape(-1) - x
        er, e0 = P_R(e), P_0(e)
        # Bhadra-style: injected null content vs true null content
        z = P_0(xin.reshape(-1)); u = P_0(x)
        if float(u.norm()) > 1e-12:
            zpar = (z @ u) / (u @ u) * u; zperp = z - zpar
            par_coef = float((z @ u) / (u @ u))
        else:
            zpar = torch.zeros_like(z); zperp = z; par_coef = 0.0
        row = {"mse_total": float((e ** 2).mean()), "mse_row": float((er ** 2).mean()),
               "mse_null": float((e0 ** 2).mean()),
               "null_share_of_error": float((e0 ** 2).sum() / (e ** 2).sum().clamp(min=1e-30)),
               "psnr": float(-10 * np.log10(max(float((e ** 2).mean()), 1e-30))),
               "null_content_norm": float(z.norm()), "true_null_norm": float(u.norm()),
               "prior_correct_coef": par_coef, "hallucinated_null_norm": float(zperp.norm()),
               "range_consistency_relerr": float((A @ xin.reshape(-1) - A @ x).norm() / (A @ x).norm())}
        log(f"  {name:24s} PSNR={row['psnr']:5.2f}dB  row-MSE={row['mse_row']:.2e}  null-MSE={row['mse_null']:.2e}  "
            f"null-share={row['null_share_of_error']*100:5.1f}%  align-coef={par_coef:+.3f}  "
            f"halluc-null={row['hallucinated_null_norm']:.3f}  rel-consist={row['range_consistency_relerr']:.1e}")
        return row

    rows = {}
    xmin = P_R(x)                                                     # min-norm / range-only reconstruction
    rows["minnorm_PRx"] = split(xmin, "min-norm (P_R GT)")
    rows["dgi_learned"] = split(dgi, "their DGI (shipped)")
    # true-scene reference norms
    ref = {"gt_row_norm": float(P_R(x).norm()), "gt_null_norm": float(P_0(x).norm()),
           "gt_null_energy_share": float((P_0(x).norm() ** 2) / (x.norm() ** 2))}
    log(f"scene: ||P_R x||={ref['gt_row_norm']:.2f}  ||P_0 x||={ref['gt_null_norm']:.2f}  "
        f"(true scene has {100*ref['gt_null_energy_share']:.1f}% of its energy in THEIR null space)")

    out = {"target": "FeiWang0824/physics-driven-fine-tuning (Photon. Res. 10, 104 (2022)), stl10 64x64, "
                      "1024 learned patterns; released operator + released single-scene sim record",
           "convention_check_max_abs": conv_err, "operator": char, "scene": ref, "decompositions": rows,
           "notes": "y standardization is whole-vector affine -> row space & projectors unaffected. Sim is "
                    "noiseless so y = A GT exactly. dgi_r is their shipped physics-layer output (DGI with "
                    "learned patterns) -- the DNN and fine-tuned outputs require their TF1 env (Stage 2)."}
    with open(OUT + r"\forensics_pedl_stl10.json", "w") as f:
        json.dump(out, f, indent=2)
    log("wrote forensics_pedl_stl10.json")


if __name__ == "__main__":
    main()
