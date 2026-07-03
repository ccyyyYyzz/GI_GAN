"""HQ near-term wins, really run on cached dev residuals (seed0), scored by LPIPS.
 HQ-9  2-D dial:  x_hat = x0 + P0(B1 d_A + B2 d_G), exact audit; grid vs the current scalar (B1=1).
 HQ-8  per-image B ceiling: oracle per-image argmin-LPIPS over a B grid (upper bound of any GT-free selector).
 HQ-10 endpoint interp: the scalar frontier's val-optimal point (== current balanced) -- reported as reference.
All quality stays inside P0 (A x_hat = y exactly). This is a dev-scored CEILING probe, not a locked claim.
"""
from __future__ import annotations
import json
import numpy as np
import torch
import gan_high_quality_gi as hq
import vqgan_detail_fusion as vdf
import vqgan_detail_fusion_locked as vlk

def log(*a): vdf.log(*a)

def fuse2d(x0f, dA, dG, b1, b2, y, meas, proj):
    dF = proj.null_project_flat(b1 * dA + b2 * dG)
    return meas.unflatten_img(proj.audit_flat(x0f + dF, y)).float()

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    cfg = vdf.load_cfg(seed)
    rows_np, _ = hq.build_structured_operator_rows(
        img_size=int(cfg["data"]["img_size"]), total_m=int(cfg["operator"]["total_m"]),
        dct_rows=int(cfg["operator"]["dct_rows"]), hadamard_rows=int(cfg["operator"]["hadamard_rows"]),
        random_rows=int(cfg["operator"]["random_rows"]), seed=int(cfg["operator"]["seed"]))
    meas = hq.make_measurement_operator(rows_np, img_size=int(cfg["data"]["img_size"]),
                                        device=dev, lambda_solver=float(cfg["operator"]["lambda_solver"]))
    from src.projections import get_exact_projector
    proj = get_exact_projector(meas, dtype=torch.float64, device=dev)
    pack = vdf.load_pack(seed, "dev", device=dev)
    pre = vdf.prep_residuals(pack, meas, proj)
    x0f, dA, dG, y, truth = pre["x0f"], pre["d_A"], pre["d_G"], pre["y"], pre["truth"]
    lp = hq.load_lpips(dev)
    n = truth.shape[0]
    log(f"HQ recon probe: dev n={n}")

    # ---- baseline: VQAE (B=0) and current scalar sweep ----
    vqae = vdf.fast_means(vdf.fuse(("scalar", 0.0), x0f, dA, dG, y, meas, proj, []), truth, y, meas, lp)
    grid1 = [round(b, 2) for b in np.linspace(0, 1, 21)]
    scal = []
    for b in grid1:
        m = vdf.fast_means(vdf.fuse(("scalar", b), x0f, dA, dG, y, meas, proj, []), truth, y, meas, lp)
        scal.append((b, m["lpips"], m["psnr"]))
    # best scalar under PSNR tolerance (>= VQAE - 0.5), like the gate
    ok = [(b, l, p) for (b, l, p) in scal if p >= vqae["psnr"] - 0.5]
    b_s, lp_s, ps_s = min(ok, key=lambda t: t[1]) if ok else min(scal, key=lambda t: t[1])
    log(f"  VQAE(B=0): LPIPS={vqae['lpips']:.4f} PSNR={vqae['psnr']:.2f}")
    log(f"  best SCALAR (PSNR>=VQAE-0.5): B={b_s} LPIPS={lp_s:.4f} PSNR={ps_s:.2f}")

    # ---- HQ-9: 2-D dial grid (subset for speed, verify best on full) ----
    sub = slice(0, min(160, n))
    x0s, dAs, dGs, ys, ts = x0f[sub], dA[sub], dG[sub], y[sub], truth[sub]
    g = [round(b, 2) for b in np.linspace(0, 1, 11)]
    best2d = None
    for b1 in g:
        for b2 in g:
            pred = fuse2d(x0s, dAs, dGs, b1, b2, ys, meas, proj)
            m = vdf.fast_means(pred, ts, ys, meas, lp)
            if m["psnr"] >= vqae["psnr"] - 0.5 and (best2d is None or m["lpips"] < best2d[2]):
                best2d = (b1, b2, m["lpips"], m["psnr"])
    b1, b2, _, _ = best2d
    m2 = vdf.fast_means(fuse2d(x0f, dA, dG, b1, b2, y, meas, proj), truth, y, meas, lp)  # full 512
    log(f"  HQ-9 best 2-D dial (full): (B1={b1},B2={b2}) LPIPS={m2['lpips']:.4f} PSNR={m2['psnr']:.2f}  "
        f"| delta vs scalar = {m2['lpips']-lp_s:+.4f}")

    # ---- HQ-8: oracle per-image B (ceiling of any GT-free per-image selector) ----
    per = {b: vlk.per_image_metrics(vdf.fuse(("scalar", b), x0f, dA, dG, y, meas, proj, []), truth, y, meas, lp)["lpips"]
           for b in grid1}
    L = np.stack([per[b] for b in grid1], axis=1)         # (n, 21)
    oracle = float(np.mean(np.min(L, axis=1)))
    log(f"  HQ-8 ORACLE per-image B: LPIPS={oracle:.4f}  | ceiling delta vs scalar = {oracle-lp_s:+.4f}")

    out = {"vqae_lpips": vqae["lpips"], "best_scalar": {"B": b_s, "lpips": lp_s, "psnr": ps_s},
           "hq9_2d": {"B1": b1, "B2": b2, "lpips": m2["lpips"], "psnr": m2["psnr"], "delta_vs_scalar": m2["lpips"]-lp_s},
           "hq8_oracle_per_image_lpips": oracle, "hq8_ceiling_delta": oracle-lp_s}
    (vdf.BASE / "detail_fusion_paper" / "hq_recon_probe.json").write_text(json.dumps(out, indent=2))
    log("wrote hq_recon_probe.json")

if __name__ == "__main__":
    main()
