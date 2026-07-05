# -*- coding: utf-8 -*-
"""Reviewer must-fix #2: are the witness and the dial output legal images?

1. WITNESS diagnostics on the dial's own operator (m=205, 5%): for cross-image pairs,
   report min/max, out-of-[0,1] fraction, residual of the raw witness, residual of the
   CLIPPED witness, and PSNR-to-target.
2. BOX-CONSTRAINED WITNESS via POCS: alternate exact fiber projection (audit_flat) and
   box clipping; report whether a [0,1]-feasible wrong witness exists (residual after
   final clip, semantic wrongness).
3. DIAL output domain: out-of-range pixel fraction of x_hat_B and the residual of the
   clamped output, quantifying what display clipping costs the identity A x_hat = y.
"""
import json, time
import numpy as np
import torch
import gan_high_quality_gi as hq
import vqgan_detail_fusion as vdf

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = vdf.BASE / "detail_fusion_paper"


def log(*a): print(f"[domain {time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    log("device =", DEV)
    cfg = vdf.load_cfg(0)
    meas, proj = vdf.build_meas(cfg, DEV)
    pack = vdf.load_pack(0, "dev", DEV)
    pre = vdf.prep_residuals(pack, meas, proj)
    truth = pre["truth"].double()
    n_img = truth.shape[0]

    def relerr(vflat, y):
        return float((proj.A_forward(vflat) - y).norm() / y.norm())

    # ---------- 1+2: witnesses ----------
    rng = np.random.default_rng(0)
    pairs = [(int(i), int(j)) for i, j in zip(rng.permutation(n_img)[:16], rng.permutation(n_img)[16:32])]
    rows = []
    for (i, j) in pairs:
        xi = meas.flatten_img(truth[i:i + 1])
        xj = meas.flatten_img(truth[j:j + 1])
        yi = proj.A_forward(xi)                       # noiseless record of target (operator convention)
        u = proj.audit_flat(xj, yi)                   # witness: donor projected onto target fiber
        u_img = u.reshape(-1)
        oob = float(((u_img < 0) | (u_img > 1)).float().mean())
        umin, umax = float(u_img.min()), float(u_img.max())
        r_raw = relerr(u, yi)
        u_clip = u.clamp(0, 1)
        r_clip = relerr(u_clip, yi)
        # POCS: alternate box clip and exact fiber projection
        v = u.clone()
        for _ in range(200):
            v = proj.audit_flat(v.clamp(0, 1), yi)
        v_final = v.clamp(0, 1)                        # report the box-feasible object
        r_pocs = relerr(v_final, yi)
        oob_pocs = float((( v.reshape(-1) < -1e-6) | (v.reshape(-1) > 1 + 1e-6)).float().mean())
        def psnr(a, b):
            mse = float(((a - b) ** 2).mean())
            return -10 * np.log10(max(mse, 1e-30))
        rows.append({"pair": [i, j], "min": umin, "max": umax, "oob_frac": oob,
                     "relerr_raw": r_raw, "relerr_clipped": r_clip,
                     "relerr_pocs_boxed": r_pocs, "oob_after_pocs_preclip": oob_pocs,
                     "psnr_witness_vs_target": psnr(v_final.reshape(1, 1, 64, 64), truth[i:i + 1]),
                     "psnr_witness_vs_donor": psnr(v_final.reshape(1, 1, 64, 64), truth[j:j + 1])})
    arr = lambda k: np.array([r[k] for r in rows])
    log(f"witness raw:   range [{arr('min').min():.2f}, {arr('max').max():.2f}]  oob median {np.median(arr('oob_frac'))*100:.1f}%")
    log(f"witness residual: raw median {np.median(arr('relerr_raw')):.2e} | clipped median {np.median(arr('relerr_clipped')):.2e}")
    log(f"POCS boxed witness: residual median {np.median(arr('relerr_pocs_boxed')):.2e}  max {arr('relerr_pocs_boxed').max():.2e}")
    log(f"  semantic: PSNR to target median {np.median(arr('psnr_witness_vs_target')):.1f} dB | to donor {np.median(arr('psnr_witness_vs_donor')):.1f} dB")

    # ---------- 3: dial output domain ----------
    dial = {}
    for B in [0.0, 0.55, 1.0]:
        xh = vdf.fuse(("scalar", B), pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], meas, proj, [])
        xf = meas.flatten_img(xh).double()
        oob = float(((xf < 0) | (xf > 1)).float().mean())
        r_un = float(np.mean([relerr(xf[k:k + 1], pre["y"][k:k + 1]) for k in range(0, 64)]))
        xc = xf.clamp(0, 1)
        r_cl = float(np.mean([relerr(xc[k:k + 1], pre["y"][k:k + 1]) for k in range(0, 64)]))
        dial[f"B{B}"] = {"oob_frac": oob, "relerr_unclipped": r_un, "relerr_clipped": r_cl}
        log(f"dial B={B}: oob {oob*100:.2f}%  relerr unclipped {r_un:.2e} -> clipped {r_cl:.2e}")

    (OUT / "domain_diagnostics.json").write_text(json.dumps({"witness_pairs": rows, "dial": dial}, indent=2))
    log("wrote domain_diagnostics.json")


if __name__ == "__main__":
    main()
