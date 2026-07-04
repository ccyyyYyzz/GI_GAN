"""Forensics GIDC Part B (labeled OUR extension): their REAL measured patterns + a KNOWN ground truth
+ Poisson noise at their photon scale, their GIDC code UNTOUCHED (sibling dir, only data.mat swapped).

Fills the missing cell: GIDC's hardware operator with a GT-decomposable scene. GT = PEDL's shipped
stl10.bmp (a released artifact of the same group), prepared with THEIR gen-script convention
(first channel, min-max to [0,1]). y ~ Poisson(A x scaled to their real count level ~1.8e7).

  build   -> writes external_audit/GIDC_partB/{code files, data.mat}
  decompose -> after running their GIDC_main.py in the TF1 env, decompose BMPs vs GT through exact projectors
"""
from __future__ import annotations
import json, shutil, sys, time
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from PIL import Image

REPO = Path(r"E:\ns_mc_gan_gi_code_fcc_phase1")
SRC = REPO / "external_audit" / "GIDC"
DST = REPO / "external_audit" / "GIDC_partB"
PEDL = REPO / "external_audit" / "physics-driven-fine-tuning"
OUT = REPO / "outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_MEAN_COUNTS = 1.8e7   # matches their real record's photon level (forensics_gidc.json)
SR = 0.1                     # their demo setting -> m=410


def log(*a): print(f"[forensics {time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_gt():
    im = np.array(Image.open(PEDL / "data/images/stl10.bmp"), dtype=np.float64)
    if im.ndim == 3: im = im[:, :, 0]
    if im.shape != (64, 64):
        im = np.array(Image.fromarray(im.astype(np.uint8)).resize((64, 64), Image.BICUBIC), dtype=np.float64)
    im = (im - im.min()) / (im.max() - im.min())
    return im


def build():
    d = sio.loadmat(SRC / "data.mat")
    pats = d["patterns"].astype(np.float64)            # [64,64,1200] real measured speckle
    gt = load_gt()
    m_all = pats.shape[-1]
    y_clean = np.einsum("hwk,hw->k", pats, gt)         # bucket = <pattern, x>
    scale = TARGET_MEAN_COUNTS / y_clean.mean()
    rng = np.random.default_rng(20260704)
    y_counts = rng.poisson(np.clip(y_clean * scale, 0, None)).astype(np.float64)
    rel = float(np.linalg.norm(y_counts - y_clean * scale) / np.linalg.norm(y_clean * scale))
    log(f"sim record: mean counts {y_counts.mean():.3e}, realized rel shot noise {rel:.2e}")
    DST.mkdir(exist_ok=True)
    for f in ["GIDC_main.py", "GIDC_model_Unet.py", "README.md"]:
        shutil.copy2(SRC / f, DST / f)                 # their code, byte-identical
    sio.savemat(DST / "data.mat", {"patterns": pats, "measurements": y_counts.reshape(-1, 1)})
    np.save(DST / "gt.npy", gt)
    log(f"built {DST} (their code byte-identical; only data.mat is ours)")


def decompose():
    d = sio.loadmat(DST / "data.mat")
    pats = d["patterns"].astype(np.float64)
    n = 64 * 64; m = int(np.round(n * SR))             # their convention: round -> 410
    y = torch.from_numpy(d["measurements"].astype(np.float64).reshape(-1)[:m]).to(DEV)
    gt = np.load(DST / "gt.npy")
    A = torch.from_numpy(pats[:, :, :m].reshape(n, m).T.copy()).double().to(DEV)
    x = torch.from_numpy(gt.reshape(-1)).to(DEV)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    tol = S[0] * max(A.shape) * torch.finfo(torch.float64).eps
    rank = int((S > tol).sum()); Vr = Vh[:rank]
    def P_R(v): return (v @ Vr.T) @ Vr
    def P_0(v): return v - P_R(v)
    u = P_0(x); un2 = float(u @ u)
    noise_rel = float((y - A @ x * (y.mean() / (A @ x).mean())).norm() / y.norm())
    xr = P_R(x)
    def psnr_of(v): return float(-10 * np.log10(max(float(((v - x) ** 2).mean()), 1e-30)))
    log(f"operator: their real patterns, m={m}, rank={rank}; realized record rel-noise ~{noise_rel:.2e}")
    log(f"range ceiling P_R x = {psnr_of(xr):.2f} dB; GT null-energy share = {un2/float(x@x):.3f}")

    def split(img2d, name):
        best = None
        for tag, xm in [("asis", img2d), ("T", img2d.T)]:
            xv = torch.from_numpy(np.ascontiguousarray(xm, dtype=np.float64)).to(DEV).reshape(-1)
            # affine-match to GT scale (their BMPs are min-max stretched): fit a,b minimizing ||a v + b - x||
            M = torch.stack([xv, torch.ones_like(xv)], 1)
            ab = torch.linalg.lstsq(M, x.unsqueeze(1)).solution.reshape(-1)
            v = M @ ab
            e = v - x
            if best is None or float((e ** 2).mean()) < best[1]:
                best = (tag, float((e ** 2).mean()), v)
        tag, _, v = best
        e = v - x; er, e0 = P_R(e), P_0(e)
        z = P_0(v); coef = float((z @ u) / un2); zperp = z - coef * u
        row = {"orientation": tag, "psnr": psnr_of(v),
               "mse_row": float((er ** 2).mean()), "mse_null": float((e0 ** 2).mean()),
               "null_share_of_error": float((e0 ** 2).sum() / (e ** 2).sum().clamp(min=1e-30)),
               "prior_correct_coef": coef, "halluc_null_norm": float(zperp.norm())}
        log(f"  {name:14s} [{tag}] PSNR={row['psnr']:5.2f}  row-MSE={row['mse_row']:.3e}  "
            f"null-MSE={row['mse_null']:.3e}  null%={100*row['null_share_of_error']:5.1f}  "
            f"align={coef:+.3f}  halluc={row['halluc_null_norm']:.3f}")
        return row

    rows = {"range_ceiling_psnr": psnr_of(xr), "gt_null_energy_share": un2 / float(x @ x),
            "record_rel_noise": noise_rel}
    # their DGI (recompute in float, their formula)
    cnt = 0; SI = np.zeros((64, 64)); B_av = 0.0; R_av = 0.0; RI = np.zeros((64, 64))
    yv = d["measurements"].reshape(-1)
    for i in range(m):
        p = pats[:, :, i]; b = float(yv[i]); cnt += 1
        SI = (SI * (cnt - 1) + p * b) / cnt
        B_av = (B_av * (cnt - 1) + b) / cnt
        R_av = (R_av * (cnt - 1) + p.sum()) / cnt
        RI = (RI * (cnt - 1) + p.sum() * p) / cnt
    rows["DGI"] = split(SI - B_av / R_av * RI, "DGI (float)")
    for step in (0, 100, 200):
        f = DST / "results" / f"GIDC_{m}_{step}.bmp"
        if f.exists():
            rows[f"GIDC_step{step}"] = split(np.array(Image.open(f), dtype=np.float64) / 255.0, f"GIDC step {step}")
    (OUT / "forensics_gidc_partB.json").write_text(json.dumps(
        {"design": "Part B (OUR extension, labeled): their real measured patterns + known GT (PEDL's shipped "
                   "stl10.bmp, their gen-script convention) + Poisson at their real count level (~1.8e7); "
                   "their GIDC code byte-identical, only data.mat swapped.", **rows}, indent=2))
    log("wrote forensics_gidc_partB.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "decompose": decompose()
    else: build()
