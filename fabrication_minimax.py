# -*- coding: utf-8 -*-
"""Feasibility derivation of a 'fabrication-component minimax' theory for null-space hallucination.
Rigor test: derive the closed forms, verify by Monte Carlo, and see what framework it reduces to.

Null error of a reconstruction: e_N = P_N(x_hat - x).  a = P_N x_hat (asserted null content), t = P_N x (true).
  - ERASER (min-norm, a=0):     ||e_N||^2 = ||t||^2.               [pure erasure]
  - FABRICATOR (a != 0):        ||e_N||^2 = ||a - t||^2.
Question: is the minimax/Bayes theory of the fabrication term NEW, or Bayes-linear / minimax-vs-Bayes?
"""
import numpy as np
rng = np.random.default_rng(0)
n, m = 64, 13
A = rng.standard_normal((m, n))
Apinv = A.T @ np.linalg.inv(A @ A.T)              # full row rank
PR = Apinv @ A; PN = np.eye(n) - PR
sig2 = 1e-2

# prior with null<->measured CORRELATION (natural-image-like: everything correlated)
Broot = rng.standard_normal((n, n)); Sigma = Broot @ Broot.T / n + 1e-3*np.eye(n)

# ---- closed forms ----
M = A @ Sigma @ A.T + sig2*np.eye(m)
Minv = np.linalg.inv(M)
C = Sigma - Sigma @ A.T @ Minv @ A @ Sigma        # Bayes posterior covariance
erase_null = np.trace(PN @ Sigma @ PN)            # min-norm null risk = tr(P_N Sigma P_N)
bayes_null = np.trace(PN @ C @ PN)                # Bayes null risk = tr(P_N C P_N)
benefit_direct = erase_null - bayes_null
benefit_formula = np.trace(PN @ Sigma @ A.T @ Minv @ A @ Sigma @ PN)  # predictable-null term

print("=== 1. minimax over a symmetric null-ball ||P_N x||<=rho (prior-free) ===")
rho = 1.0
# assert a with ||a||=c: worst-case null risk over the ball = (c+rho)^2; minimized at c=0 (eraser)
for c in [0.0, 0.5, 1.0]:
    print(f"  assert ||a||={c}: worst-case null risk = {(c+rho)**2:.2f}   (eraser c=0 is minimax = rho^2={rho**2:.2f})")
print("  => min-norm/eraser is the unique minimax reconstruction of null content over a symmetric class.")
print("     Any fabrication ||a||=c pays excess c^2+2*c*rho.  [this is minimax-vs-Bayes / Chebyshev-center folklore]")

print("\n=== 2. Gaussian prior (closed form): fabrication BENEFIT = null-measured correlation ===")
print(f"  eraser (min-norm) null risk  tr(P_N Sigma P_N) = {erase_null:.4f}")
print(f"  Bayes (fabricator) null risk tr(P_N C P_N)     = {bayes_null:.4f}")
print(f"  fabrication benefit  (direct)  = {benefit_direct:.4f}")
print(f"  fabrication benefit  (formula) = {benefit_formula:.4f}   [tr(P_N Sigma A^T M^-1 A Sigma P_N)]")
print(f"  match: {np.isclose(benefit_direct, benefit_formula)}")
# control: block-diagonal prior (null uncorrelated with measured) -> zero benefit
Sig_bd = PR @ Sigma @ PR + PN @ Sigma @ PN
Cb = Sig_bd - Sig_bd @ A.T @ np.linalg.inv(A@Sig_bd@A.T+sig2*np.eye(m)) @ A @ Sig_bd
print(f"  CONTROL block-diagonal prior (null _|_ measured): benefit = "
      f"{np.trace(PN@Sig_bd@PN)-np.trace(PN@Cb@PN):.2e}  (=> fabrication helps ONLY via correlation)")

print("\n=== 3. Monte Carlo confirmation ===")
K = 20000
xs = (Broot @ rng.standard_normal((n, K)))/np.sqrt(n) + rng.standard_normal((n,K))*np.sqrt(1e-3)
ys = A @ xs + np.sqrt(sig2)*rng.standard_normal((m, K))
xmn = Apinv @ ys
xbayes = Sigma @ A.T @ Minv @ ys
e_mn = PN @ (xmn - xs); e_by = PN @ (xbayes - xs)
print(f"  MC eraser null risk  = {np.mean(np.sum(e_mn**2,0)):.4f}  (closed form {erase_null:.4f})")
print(f"  MC Bayes null risk   = {np.mean(np.sum(e_by**2,0)):.4f}  (closed form {bayes_null:.4f})")

print("\n=== 4. fabrication RISK under prior misspecification (the honest cost of the bet) ===")
# fabricator trained on WRONG prior Sigma_a; truth drawn from Sigma_t
Broot2 = rng.standard_normal((n, n)); Sigma_a = Broot2 @ Broot2.T / n + 1e-3*np.eye(n)  # assumed (wrong)
Ma = A@Sigma_a@A.T+sig2*np.eye(m); Minva=np.linalg.inv(Ma)
xbayes_mis = Sigma_a @ A.T @ Minva @ ys                     # uses wrong prior on true-Sigma data
e_mis = PN @ (xbayes_mis - xs)
print(f"  eraser (prior-free) null risk        = {np.mean(np.sum(e_mn**2,0)):.4f}")
print(f"  fabricator on CORRECT prior          = {np.mean(np.sum(e_by**2,0)):.4f}  (beats eraser)")
print(f"  fabricator on WRONG prior (shift)    = {np.mean(np.sum(e_mis**2,0)):.4f}  (can EXCEED eraser -> the bet backfires)")
print("\n  VERDICT: fabrication = a null-space Bayes bet; benefit = prior's null<->measured correlation;")
print("  robust choice = eraser (min-norm), minimax over symmetric classes. This is the Gaussian/Wiener")
print("  linear-Bayes model + minimax-vs-Bayes + Chebyshev-center(Iagaru) -- NO new inequality, a specialization.")
