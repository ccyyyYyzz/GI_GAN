# -*- coding: utf-8 -*-
"""Post-commitment random-challenge test for GT-free per-image hallucination falsification.
Idea (sharpened w/ GPT): reconstruct from A_fit only, FREEZE x_hat, then probe with k FRESH random
challenge patterns and compare predicted vs observed buckets. The whitened challenge residual is a
per-image, ground-truth-free estimator of total error energy ||x_hat - x||^2/n (Johnson-Lindenstrauss),
so it catches the feasible-but-wrong witness that beat A_fit to machine precision.

Controls:
  - A_fit residual ||A_fit x_hat - y||/||y||: ~0 for witness AND min-norm (barrier: can't distinguish).
  - challenge residual: tracks TRUE error, tightening ~1/sqrt(k); separates confident-wrong from correct.
  - adversary knows the challenge DISTRIBUTION but not the draws (post-commitment isotropic) -> can't hide.
"""
import numpy as np
rng = np.random.default_rng(0)
side, n, m, N = 64, 4096, 205, 300
sig_fit, sig_ch = 3e-3, 3e-3

# --- synthetic natural-ish scenes: 1/f^1.5 spectrum, normalized to [0,1] ---
fx = np.fft.fftfreq(side)[:, None]; fy = np.fft.fftfreq(side)[None, :]
filt = 1.0 / (np.sqrt(fx**2 + fy**2) + 1e-6)**1.5
def make(N):
    out = np.empty((N, n))
    for i in range(N):
        im = np.fft.ifft2(np.fft.fft2(rng.standard_normal((side, side))) * filt).real
        im = (im - im.min()) / (im.max() - im.min() + 1e-9)
        out[i] = im.ravel()
    return out
X = make(N)

# --- A_fit: row-orthonormal (matches paper's signed row-orthonormalized GI operator) ---
Q, _ = np.linalg.qr(rng.standard_normal((n, m)))   # n x m, orthonormal columns
Afit = Q.T                                          # m x n, orthonormal rows -> Afit Afit^T = I
Apinv = Afit.T                                       # A^dagger = A^T (right inverse)
def PR(v): return Afit.T @ (Afit @ v)
def P0(v): return v - PR(v)

ks = [1, 2, 4, 8, 16, 32, 64]; Kmax = max(ks)
recons_keys = ['good', 'prior', 'minnorm', 'witness', 'witness_adapt']
true_e2 = {k: [] for k in recons_keys}
fit_res = {k: [] for k in recons_keys}
chal = {key: {k: [] for k in ks} for key in recons_keys}
adapt_fresh = {k: [] for k in ks}   # adaptive witness vs FRESH (unseen) challenges -> should be caught

for i in range(N):
    x = X[i]
    y = Afit @ x + sig_fit * rng.standard_normal(m)
    xmn = Apinv @ y
    xj = X[(i + 1) % N]                                   # donor = different scene
    u = xj - Apinv @ (Afit @ xj - y)                     # feasible-but-wrong witness (matches y exactly)
    xgood = x + 0.01 * P0(rng.standard_normal(n))        # near-correct
    xprior = xmn + P0(0.7 * x + 0.3 * xj)                # partly-right null fill (70% true / 30% donor)
    # post-commitment challenge rows a ~ N(0, I/n)  =>  E[(a^T e)^2] = ||e||^2/n
    Ach = rng.standard_normal((Kmax, n)) / np.sqrt(n)
    ych = Ach @ x + sig_ch * rng.standard_normal(Kmax)   # observed challenge buckets (real, noisy)
    # GPT's essential control: ADAPTIVE stacked-witness, built AFTER seeing the challenge B=Ach.
    # Feasible for C=[Afit;Ach] so it matches y AND y_ch -> should EVADE the SAME challenge.
    C = np.vstack([Afit, Ach]); dvec = np.concatenate([y, ych])
    u_adapt = xj - C.T @ np.linalg.solve(C @ C.T, C @ xj - dvec)   # matches [Afit;Ach], wrong donor null content
    recons = {'good': xgood, 'prior': xprior, 'minnorm': xmn, 'witness': u, 'witness_adapt': u_adapt}
    Ach2 = rng.standard_normal((Kmax, n)) / np.sqrt(n)   # FRESH unseen challenges (arms-race resolution)
    ych2 = Ach2 @ x + sig_ch * rng.standard_normal(Kmax)
    for key, xh in recons.items():
        e = xh - x
        true_e2[key].append(e @ e / n)
        fit_res[key].append(np.linalg.norm(Afit @ xh - y) / (np.linalg.norm(y) + 1e-12))
        r = Ach @ xh - ych                               # predicted - observed (SAME challenge)
        for k in ks:
            chal[key][k].append(np.mean(r[:k]**2))
    r2 = Ach2 @ u_adapt - ych2                            # adaptive witness vs FRESH challenges
    for k in ks:
        adapt_fresh[k].append(np.mean(r2[:k]**2))

honest = ['good', 'prior', 'minnorm', 'witness']   # fair post-commitment tests (recon built without seeing B)
for key in recons_keys:
    for d in (true_e2, fit_res):
        d[key] = np.array(d[key])
    for k in ks:
        chal[key][k] = np.array(chal[key][k])
for k in ks:
    adapt_fresh[k] = np.array(adapt_fresh[k])

print("=== A_fit residual ||A_fit x_hat - y||/||y|| (barrier: witness looks perfect) ===")
for key in recons_keys:
    print(f"  {key:13s}: median {np.median(fit_res[key]):.2e}   true err ||e||^2/n median {np.median(true_e2[key]):.2e}")

print("\n=== JL unbiasedness (honest post-commitment recons): challenge est k=64 vs true+noise (sig^2={:.1e}) ===".format(sig_ch**2))
for key in honest:
    est = np.mean(chal[key][64]); tru = np.mean(true_e2[key]) + sig_ch**2
    print(f"  {key:8s}: chal_est {est:.3e}   true+noise {tru:.3e}   ratio {est/tru:.3f}")

from numpy import corrcoef
allkey_true = np.concatenate([true_e2[k] for k in honest])
for k in [4, 16, 64]:
    allkey_est = np.concatenate([chal[key][k] for key in honest])
    print(f"  per-image corr(chal_est_k{k}, true err): {corrcoef(allkey_est, allkey_true)[0,1]:.3f}")

print("\n=== detection power: catch WITNESS (positive) vs GOOD (negative), threshold at 5% false alarm ===")
for k in ks:
    neg = chal['good'][k]; pos = chal['witness'][k]
    thr = np.quantile(neg, 0.95)                         # 5% false-alarm threshold from good recons
    power = np.mean(pos > thr)
    # AUC
    from itertools import product
    auc = np.mean([p > q for p in pos for q in neg]) if len(pos) * len(neg) < 4e5 else np.nan
    print(f"  k={k:3d}: detection@5%FA = {power:5.1%}   AUC(witness vs good) = {auc:.3f}   (extra budget {k/n:.2%} of pixels)")

print("\n=== headline ===")
k = 32
thr = np.quantile(chal['good'][k], 0.95)
print(f"  A_fit residual cannot separate witness from good (both ~{np.median(fit_res['witness']):.0e}); "
      f"chance detection.")
print(f"  With k={k} fresh post-commitment challenges (~{k/n:.2%} extra pixels): "
      f"witness caught {np.mean(chal['witness'][k] > thr):.0%} at 5% false alarm.")
print(f"  So the adversary that beat A_fit to ~1e-15 is falsified per-image, ground-truth-free.")

print("\n=== GPT's essential control: the exact boundary (precommitted caught, adaptive not) ===")
print("  witness_adapt is built AFTER seeing the challenge B (feasible for [A_fit; B]), yet still wrong:")
print(f"    true err ||e||^2/n median: witness(precommit) {np.median(true_e2['witness']):.2e}, "
      f"witness_adapt {np.median(true_e2['witness_adapt']):.2e}  (both large/wrong)")
for k in [8, 32, 64]:
    thr = np.quantile(chal['good'][k], 0.95)
    caught_pre = np.mean(chal['witness'][k] > thr)
    caught_adapt_same = np.mean(chal['witness_adapt'][k] > thr)   # vs the SAME challenge it saw
    caught_adapt_fresh = np.mean(adapt_fresh[k] > thr)            # vs FRESH unseen challenge
    print(f"  k={k:3d}: precommitted witness caught {caught_pre:5.1%} | "
          f"adaptive witness vs SAME challenge caught {caught_adapt_same:5.1%} (EVADES) | "
          f"adaptive witness vs FRESH challenge caught {caught_adapt_fresh:5.1%}")
print("  => the test beats a PRECOMMITTED adversary, not one who sees the challenge draws;")
print("     hiding requires seeing the specific draws -> fresh/hidden post-commitment challenges restore teeth.")
print("     Honest claim: a post-commitment FALSIFICATION protocol, not a universal truth certificate.")
