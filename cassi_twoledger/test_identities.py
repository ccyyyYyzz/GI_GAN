# -*- coding: utf-8 -*-
"""Certify the exact CASSI projector identities in float64 on the REAL coded aperture.
Because A A^T = diag(Phi_s) is exactly diagonal, all identities are closed-form and must
hold to ~1e-13. Also reports the non-uniform singular spectrum summary."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cassi_operator import CASSI, load_mask

DTYPE = torch.complex128 if False else torch.float64  # CASSI is real-valued
MASK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask.mat")


def norm(t): return float(torch.linalg.vector_norm(t).item())


def main():
    mask = load_mask(MASK, dtype=DTYPE)
    op = CASSI(mask, nC=28, step=2)
    print(f"mask {tuple(mask.shape)} fill {float(mask.mean()):.3f} | cube [28,{op.H},{op.W}] "
          f"-> snapshot [{op.H},{op.Ws}]  (undersampling {op.Ws*op.H/(28*op.H*op.W):.3f} = {28*op.H*op.W/(op.Ws*op.H):.1f}x)")
    torch.manual_seed(0)
    x = torch.rand(28, op.H, op.W, dtype=DTYPE)
    y = op.A(x)

    checks = {}
    checks["||A P_0 x|| (want ~0)"] = norm(op.A(op.P_0(x)))
    checks["||P_R P_R x - P_R x||"] = norm(op.P_R(op.P_R(x)) - op.P_R(x))
    checks["||P_R x + P_0 x - x||"] = norm(op.P_R(x) + op.P_0(x) - x)
    checks["||A A^dagger y - y|| (on support)"] = norm((op.A(op.A_dagger(y)) - y)[op.support])
    a = torch.rand(28, op.H, op.W, dtype=DTYPE); b = torch.rand(28, op.H, op.W, dtype=DTYPE)
    checks["|<P_R a,b> - <a,P_R b>|"] = abs(float((op.P_R(a) * b).sum()) - float((a * op.P_R(b)).sum()))

    donor = torch.rand(28, op.H, op.W, dtype=DTYPE)
    tgt = torch.rand(28, op.H, op.W, dtype=DTYPE); yt = op.A(tgt)
    u = op.witness(donor, yt)
    checks["witness ||A u - y_t||/||y_t||"] = op.rel_meas_err(u, yt)
    checks["witness ||P_0 u - P_0 donor||"] = norm(op.P_0(u) - op.P_0(donor))

    lam = 1e-3
    v = op.A_dagger(yt) + 0.5 * op.P_0(donor)
    r0 = op.rel_meas_err(v, yt); v2 = op.audit(v, yt, lam); r1 = op.rel_meas_err(v2, yt)
    checks["||P_0 audit(v) - P_0 v||"] = norm(op.P_0(v2) - op.P_0(v))
    # per-mode contraction is NON-uniform here: report its spread, and that audit reduces residual
    c = op.contraction(lam)
    checks["audit reduces residual (r1<r0)"] = 0.0 if r1 < r0 else 1.0

    print()
    ok = True
    for k, val in checks.items():
        good = val < 1e-9
        ok = ok and good
        print(f"  {'OK ' if good else '!! '}{k:40s} = {val:.3e}")

    s = op.singular_values()
    print(f"\nNON-UNIFORM singular spectrum sigma_j = sqrt(Phi_s):")
    print(f"  {s.numel()} measured detector modes | sigma range [{float(s.min()):.3f}, {float(s.max()):.3f}] "
          f"median {float(s.median()):.3f} | ratio max/min {float(s.max()/s.min()):.1f}x")
    print(f"  per-mode contraction lambda/(lambda+sigma^2) at lambda={lam}: "
          f"[{float(c.min()):.2e}, {float(c.max()):.2e}]  (MRI would be a single value {lam/(lam+1):.2e})")
    print("\nALL IDENTITIES HOLD" if ok else "\nSOME CHECK FAILED")


if __name__ == "__main__":
    main()
