# -*- coding: utf-8 -*-
"""Independent numeric confirmation of every theoretical claim in the paper (OPTICS_DRAFT).
General range-null identities on a full-row-rank operator; the exact-diagonal CASSI identity
and the singular spectrum on the real released coded-aperture operator. Run:
    python theory_verify.py
All checks should print [OK ] to ~1e-15 (float64)."""
import os, sys
import numpy as np
import torch
np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cassi_operator import CASSI, load_mask


def chk(name, cond, val=None):
    print(f"[{'OK ' if cond else 'FAIL'}] {name}" + ("" if val is None else f"   ({val})"))
    return bool(cond)


def main():
    n, m = 64, 13
    A = np.random.randn(m, n)                              # generic full row rank, m<n
    Apinv = A.T @ np.linalg.inv(A @ A.T)                   # right pseudoinverse A^dagger
    PR = Apinv @ A
    P0 = np.eye(n) - PR

    print("== 2.1 range-null geometry ==")
    chk("P_R = A^dagger A idempotent & symmetric", np.allclose(PR @ PR, PR) and np.allclose(PR, PR.T))
    chk("P_0 idempotent, P_R + P_0 = I", np.allclose(P0 @ P0, P0) and np.allclose(PR + P0, np.eye(n)))
    chk("A P_0 = 0", np.allclose(A @ P0, 0), val=f"max|AP_0|={np.abs(A @ P0).max():.1e}")
    chk("A P_R = A ; A A^dagger = I_m (full row rank)", np.allclose(A @ PR, A) and np.allclose(A @ Apinv, np.eye(m)))
    x = np.random.randn(n); eps = 1e-2 * np.random.randn(m); y = A @ x + eps
    chk("A^dagger y = P_R x + A^dagger eps", np.allclose(Apinv @ y, PR @ x + Apinv @ eps))

    print("\n== 2.2 feasible-but-wrong witness ==")
    xj = np.random.randn(n)
    u = xj - Apinv @ (A @ xj - y)
    chk("A u_j(y) = y", np.allclose(A @ u, y), val=f"rel={np.linalg.norm(A @ u - y) / np.linalg.norm(y):.1e}")
    chk("P_0 u_j = P_0 x_j (donor null content)", np.allclose(P0 @ u, P0 @ xj))
    chk("witness semantically != target", np.linalg.norm(u - x) > 0.5 * np.linalg.norm(x),
        val=f"||u-x||/||x||={np.linalg.norm(u - x) / np.linalg.norm(x):.2f}")

    print("\n== 2.3 measurement-consistency audit (MCA) ==")
    lam = 1e-3
    xhat = np.random.randn(n)
    M = A @ A.T
    Glam = A.T @ np.linalg.inv(M + lam * np.eye(m))
    Pi = xhat - Glam @ (A @ xhat - y)
    resid_post = A @ Pi - y
    chk("post-audit residual = lam (AA^T+lam I)^{-1}(Ax_hat - y)",
        np.allclose(resid_post, lam * np.linalg.inv(M + lam * np.eye(m)) @ (A @ xhat - y)))
    chk("correction in row space: P_0 Pi = P_0 x_hat", np.allclose(P0 @ Pi, P0 @ xhat))
    w, U = np.linalg.eigh(M); svals = np.sqrt(w)
    c_pred = lam / (lam + svals ** 2)
    r_in = U.T @ (A @ xhat - y); r_out = U.T @ resid_post
    chk("per-mode contraction c_i = lam/(lam+sigma_i^2)", np.allclose(r_out, c_pred * r_in),
        val=f"max err={np.abs(r_out - c_pred * r_in).max():.1e}")
    chk("c_i = 1 - Tikhonov filter factor", np.allclose(c_pred, 1 - svals ** 2 / (svals ** 2 + lam)))

    print("\n== 2.4 governed null-space dial ==")
    dA, dG = np.random.randn(n), np.random.randn(n)
    ok = all(np.allclose(A @ (Apinv @ y + P0 @ (dA + B * (dG - dA))), y) for B in [0.0, 0.37, 1.0, 2.5])
    chk("A x_B = y for B in {0, 0.37, 1, 2.5}", ok)

    print("\n== 3.2/S1 conserved orthogonal error-energy partition ==")
    e = xhat - x
    chk("||e||^2 = ||P_R e||^2 + ||P_0 e||^2 ; P_R e _|_ P_0 e",
        np.allclose(e @ e, (PR @ e) @ (PR @ e) + (P0 @ e) @ (P0 @ e)) and np.allclose((PR @ e) @ (P0 @ e), 0))

    print("\n== 3.3 CASSI diagonal identity + spectrum on the released operator ==")
    op = CASSI(load_mask(os.path.join(HERE, "mask.mat")), nC=28, step=2)
    Phi_s = op.Phi_s.cpu().numpy()
    sig = np.sqrt(Phi_s[Phi_s > 0])
    chk("every detector mode nonzero (rank = m)", (Phi_s > 0).all(), val=f"{(Phi_s > 0).sum()}/{Phi_s.size}")
    chk("sigma span ~ 528x", np.isclose(sig.max() / sig.min(), 528, rtol=0.02),
        val=f"[{sig.min():.4f},{sig.max():.3f}] ratio={sig.max() / sig.min():.0f}")
    c = lam / (lam + sig ** 2)
    chk("contraction span ~ 2e4", np.isclose(c.max() / c.min(), 2e4, rtol=0.15),
        val=f"[{c.min():.2e},{c.max():.3f}] ratio={c.max() / c.min():.0f}")
    vv = torch.randn(1, op.H, op.Ws, dtype=op.Phi.dtype)
    AAtV = op.A(op.At(vv.squeeze(0)))
    chk("A A^T v = Phi_s (x) v (exact diagonal)", torch.allclose(AAtV, op.Phi_s * vv.squeeze(0), atol=1e-8),
        val=f"max err={(AAtV - op.Phi_s * vv.squeeze(0)).abs().max().item():.1e}")
    print(f"    median sigma = {np.median(sig):.2f} (paper 3.51); undersampling n/m = "
          f"{28 * 256 * 256 / (Phi_s > 0).sum():.1f}x (paper 23.1x)")


if __name__ == "__main__":
    main()
