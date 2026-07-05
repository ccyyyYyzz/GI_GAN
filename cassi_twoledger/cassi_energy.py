# -*- coding: utf-8 -*-
"""Error-energy decomposition for the CASSI model zoo (supplement to the null-share table).
For each model, over the 10 KAIST scenes, compute the conserved orthogonal partition
    ||x_hat - x||^2 = ||P_R(x_hat-x)||^2 + ||P_0(x_hat-x)||^2 = E_R + E_0,
and report the null error-energy fraction E_0/(E_R+E_0). Reuses the forensics harness."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cassi_operator import CASSI, load_mask
import cassi_forensics as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MASK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask.mat")
import glob
NAMES = {"tsa_net":"TSA-Net","gap_net":"GAP-Net","dgsmp":"DGSMP","hdnet":"HDNet","mst_s":"MST-S",
         "mst_m":"MST-M","mst_l":"MST-L","dauhst_2stg":"DAUHST-2stg","cst_l":"CST-L",
         "mst_plus_plus":"MST++","birnat":"BIRNAT","lambda_net":"$\\lambda$-Net","dauhst_9stg":"DAUHST-9stg"}
ORDER = ["tsa_net","gap_net","dgsmp","lambda_net","hdnet","mst_s","mst_m","dauhst_2stg",
         "mst_l","cst_l","mst_plus_plus","birnat","dauhst_9stg"]


def main():
    op = CASSI(load_mask(MASK), nC=28, step=2)
    mask3d = load_mask(MASK).reshape(1, op.H, op.W).repeat(28, 1, 1)
    scenes = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes", "scene*.mat")))
    cubes = [F.load_scene(p) for p in scenes]
    # min-norm reference: e = A^dagger y - x = -P_0 x  ->  E_R=0, E_0=||P_0 x||^2 (100% null by construction)
    e0_ref = np.mean([float((op.P_0(x)**2).sum()) for x in cubes])
    rows = {}
    for m in ORDER:
        pth = os.path.join(F.ZOO, F.PATH[m])
        if not os.path.exists(pth):
            continue
        try:
            model = F.load_model(m)
        except Exception as e:
            print(f"skip {m}: {e}"); continue
        ERs, E0s = [], []
        for x in cubes:
            y = op.A(x)
            im, imask = F.build_inputs(op, mask3d, op.Phi, op.Phi_s, y, F.CFG[m][0], F.CFG[m][1])
            try:
                xhat = F.run_model(model, im, imask)
            except Exception as e:
                print(f"{m} failed: {str(e)[:60]}"); ERs = None; break
            e = xhat - x
            ER = float((op.P_R(e)**2).sum()); E0 = float((op.P_0(e)**2).sum())
            ERs.append(ER); E0s.append(E0)
        if not ERs:
            continue
        ER, E0 = np.mean(ERs), np.mean(E0s)
        frac = E0 / (ER + E0) * 100
        rows[m] = {"E_R": ER, "E_0": E0, "null_energy_frac_pct": frac}
        print(f"{NAMES[m]:14s}  E_R={ER:.4g}  E_0={E0:.4g}  null-energy {frac:.1f}%")
    print(f"\nmin-norm reference (A^dagger y): E_R=0, E_0={e0_ref:.4g}, null-energy 100.0% (by construction)")
    json.dump({"min_norm_E0": e0_ref, "models": rows}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "energy_split.json"), "w"), indent=2)
    # markdown table
    print("\n| Model | $E_R$ | $E_0$ | null energy $E_0/E$ |")
    print("|---|---|---|---|")
    for m in ORDER:
        if m in rows:
            r = rows[m]
            print(f"| {NAMES[m]} | ${r['E_R']:.2e}$ | ${r['E_0']:.2e}$ | ${r['null_energy_frac_pct']:.1f}\\%$ |")


if __name__ == "__main__":
    main()
