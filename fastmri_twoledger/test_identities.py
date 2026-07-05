# -*- coding: utf-8 -*-
"""Certify the exact projector identities of the masked-Fourier operator in float64
(complex128), on synthetic complex images. No real data needed. These must hold to
~1e-14 for the two-ledger accountability claims to be exact on this operator."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mf_operator import MaskedFourier, equispaced_mask, random_mask, fft2c, ifft2c

torch.manual_seed(0)
DTYPE = torch.complex128
H, W = 128, 120

def norm(t): return float(torch.linalg.vector_norm(t).item())

def run(mask_name, mask1d):
    op = MaskedFourier(mask1d, (H, W))
    x = torch.randn(2, H, W, dtype=DTYPE)           # batch of 2 complex images
    y = op.A(x)
    lam = 1e-3

    checks = {}
    checks["||A P_0 x|| (want ~0)"] = norm(op.A(op.P_0(x)))
    checks["||P_R P_R x - P_R x||"] = norm(op.P_R(op.P_R(x)) - op.P_R(x))
    checks["||P_R x + P_0 x - x||"] = norm(op.P_R(x) + op.P_0(x) - x)
    checks["||A A^dagger y - y||"] = norm(op.A(op.A_dagger(y)) - y)
    a = torch.randn(H, W, dtype=DTYPE); b = torch.randn(H, W, dtype=DTYPE)
    ip1 = torch.vdot(op.P_R(a).flatten(), b.flatten())
    ip2 = torch.vdot(a.flatten(), op.P_R(b).flatten())
    checks["|<P_R a,b> - <a,P_R b>|"] = float(torch.abs(ip1 - ip2).item())
    checks["||ifft2c fft2c x - x||"] = norm(ifft2c(fft2c(x)) - x)

    donor = torch.randn(H, W, dtype=DTYPE)
    tgt = torch.randn(H, W, dtype=DTYPE)
    yt = op.A(tgt)
    u = op.witness(donor, yt)
    checks["witness ||A u - y_t||/||y_t||"] = op.rel_meas_err(u, yt)
    checks["witness ||P_0 u - P_0 donor||"] = norm(op.P_0(u) - op.P_0(donor))

    v = op.A_dagger(yt) + 0.5 * op.P_0(donor) + 0.01 * torch.randn(H, W, dtype=DTYPE)
    r0 = op.rel_meas_err(v, yt)
    v2 = op.audit(v, yt, lam)
    r1 = op.rel_meas_err(v2, yt)
    ratio = r1 / r0 if r0 > 0 else float("nan")
    checks[f"audit resid ratio (want {lam/(lam+1):.6f})"] = ratio
    checks["||P_0 audit(v) - P_0 v||"] = norm(op.P_0(v2) - op.P_0(v))

    print(f"\n===== mask = {mask_name} | sampling rate = {op.sampling_rate:.3f} "
          f"(accel ~{1/op.sampling_rate:.1f}x) =====")
    ok = True
    target_ratio = lam / (lam + 1)
    for k, val in checks.items():
        if "ratio" in k:
            good = abs(val - target_ratio) < 5e-2
        else:
            good = val < 1e-10
        ok = ok and good
        print(f"  {'OK ' if good else '!! '}{k:42s} = {val:.3e}")
    return ok

if __name__ == "__main__":
    ok1 = run("equispaced 4x", equispaced_mask(W, acceleration=4, center_fraction=0.08))
    ok2 = run("random 8x", random_mask(W, acceleration=8, center_fraction=0.04, seed=1))
    print("\nALL IDENTITIES HOLD" if (ok1 and ok2) else "\nSOME CHECK FAILED")
