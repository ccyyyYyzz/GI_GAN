"""IP-1 / operator-reconciliation: feasible-wrong witnesses on the SAME 64x64 m=205 operator used by
the VQGAN detail-fusion locked result. u = audit_flat(flatten(x_j), y_i) projects a different scene x_j
onto {x : A x = y_i}; it reproduces y_i to ~machine precision yet is semantically x_j. Residuals are
computed in float64 against A built directly from the operator rows (self-consistent), so the ~1e-15
number is exact. Unifies the converse (P1) and the fusion (P3) on one operator.
"""
from __future__ import annotations
import json
import numpy as np
import torch
import gan_high_quality_gi as hq
import vqgan_detail_fusion as vdf
from src.projections import get_exact_projector

def log(*a): vdf.log(*a)

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
    proj = get_exact_projector(meas, dtype=torch.float64, device=dev)
    pack = vdf.load_pack(seed, "dev", device=dev)
    truth = pack["truth"].to(dev)
    n = truth.shape[0]
    A = torch.as_tensor(rows_np, dtype=torch.float64, device=dev)          # (m, npix)
    flat = meas.flatten_img(truth).to(torch.float64)                        # (n, npix)
    Y = flat @ A.T                                                          # (n, m) exact float64 records
    log(f"operator 64x64 m={A.shape[0]}; dev images={n}; npix={A.shape[1]}")
    rng = np.random.default_rng(20260704)
    rows = []
    tries = 0
    while len(rows) < 40 and tries < 8000:
        tries += 1
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j or torch.mean((truth[i] - truth[j]) ** 2).item() < 1e-3:
            continue
        u = proj.audit_flat(flat[j:j+1], Y[i:i+1])                          # project x_j onto {x: A x = y_i}
        rel = (torch.norm(u @ A.T - Y[i:i+1]) / torch.norm(Y[i:i+1])).item()
        ui = u.reshape(truth[i:i+1].shape).float()
        psnr = -10.0 * np.log10(max(torch.mean((ui - truth[i:i+1].float()) ** 2).item(), 1e-12))
        rows.append({"i": i, "j": j, "relmeas_u_vs_yi": rel, "psnr_u_vs_xi": psnr})
    rel = np.array([r["relmeas_u_vs_yi"] for r in rows]); ps = np.array([r["psnr_u_vs_xi"] for r in rows])
    summary = {"operator": "64x64_m205_seed772001", "n_pairs": len(rows),
               "relmeas_u_median": float(np.median(rel)), "relmeas_u_max": float(np.max(rel)),
               "relmeas_u_min": float(np.min(rel)), "psnr_u_vs_target_median": float(np.median(ps))}
    log("FEASIBLE-WRONG on the fusion operator:")
    log(f"  pairs={len(rows)}  relmeas(u vs y_i): median={summary['relmeas_u_median']:.2e} "
        f"max={summary['relmeas_u_max']:.2e} min={summary['relmeas_u_min']:.2e}  |  "
        f"PSNR(u vs true x_i) median={summary['psnr_u_vs_target_median']:.2f} dB")
    out = vdf.BASE / "detail_fusion_paper" / "feasible_wrong_fusion_operator.json"
    out.write_text(json.dumps({"summary": summary, "pairs": rows}, indent=2))
    log("wrote", str(out))

if __name__ == "__main__":
    main()
