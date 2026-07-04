# Certified-Decision Sandwich — Feasibility Prototype (2026-07-04)

Direction #1 from the deep-innovation audit: certify a *nonlinear* classifier's decision over the exact
measurement fiber, exploiting geometry (not compute) to turn a normally-4096-dim certified-robustness problem
into a k≈8-dim one. Brief from the user: **small compute, big improvement, real innovation — not A+B.**

Script: `certified_decision_sandwich.py` (subcommands `selftest | prep | gate | diag | risk`). All sim-only,
RTX 4060, hand-rolled IBP+CROWN+BaB (zero external verifier dep; the GitHub `auto_LiRPA` install is sandbox-
blocked and unnecessary — the hand-rolled verifier passes its soundness selftest).

## What was validated (infrastructure — the geometry-not-compute thesis)
- **Exact affine slab.** B = top-k eigenvectors of the null-projected train covariance (from LMMSE `z_scaled`,
  economy SVD — no 4096² eigh). Audit: `||A b_j||_inf = 3.6e-16` — the whole slab is in Null(A) to float64, so
  every `x_hat + Bc` satisfies `A x' = y` exactly.
- **Sound verifier.** Hand-rolled backward CROWN + IBP + best-first input-split BaB. Selftest: CROWN lower
  bound ≤ dense-sampled empirical min margin at every width (PASS).
- **Geometry beats compute, confirmed.** Certifying a decision over the k=8 slab is **~50–80 ms/scene**
  (4 CROWN passes) — what standard certified robustness pays 4096-dim ε-ball verification for. The dimensional
  collapse is real and is the reusable asset.

## Honest negative: the naive "certify most decisions" goal is dead (a scissors)
`diag` (k=8, 64 scenes):
- **Containment:** real images' null-space spread / prior-σ ≈ 0.76–0.88 per direction (slab is well
  calibrated, not inflated). To be *prior-plausible* the box needs **w ≈ 2σ** (91% of real scenes fit); at
  w=1σ only 28% fit.
- **The scissors:** at w=0.1σ CROWN certifies 69% of decisions — but that box covers **~0%** of real image
  variation (vacuous). At w≥0.5σ (prior-plausible scales) **CDR = 0**. There is **no width where the
  certificate is both prior-plausible AND non-trivially certifiable.**
- **Meaning (a real finding):** at prior-plausible scale, a GI classifier's decision is genuinely **not
  determined by the measurement** — P1 (feasible-wrong twins) extends from pixels to *decisions*. The `gate`
  pass found in-range twins (FLIPPED_PHYSICAL) that flip the label while matching y to float64.

## The cheap win: the certified decision radius r* as a GT-free trust score
Pivot forced by the data: don't certify *most* decisions (impossible) — **rank** them by how measurement-
determined they are. Define r*(scene) = largest w whose decision is CROWN-certified (units of prior σ).
Target = decision fidelity `DF = 1[argmax f(x_hat) == argmax f(truth)]` (recon's decision matches the true
scene's decision). 128 dev scenes, **~50–80 ms/scene**.

| operating point | coverage | DF (certified) | DF (rest) | lift vs base (0.656) |
|---|:---:|:---:|:---:|:---:|
| r* ≥ 0.10 (k=8) | 0.64 | 0.768 | 0.457 | +0.112 |
| r* ≥ 0.15 (k=8) | 0.40 | **0.902** | 0.494 | **+0.246** |
| r* ≥ 0.20 (k=8) | 0.16 | 1.000 | 0.593 | +0.344 |

Spearman(r*, DF) = **+0.37** (k=8), +0.38 (k=16). Abstaining on the uncertified tail raises decision-fidelity
substantially, and the discarded scenes are demonstrably the unreliable ones (DF ≈ 0.46).

## The novelty-critical control: r* vs plain softmax confidence
Confidence-based selective prediction is old; the deep claim is that r* is **measurement-aware** and catches a
failure confidence cannot. Head-to-head (128 scenes):

| | k=8 | k=16 |
|---|:---:|:---:|
| Spearman(r*, DF) | +0.370 | +0.379 |
| Spearman(conf, DF) | +0.336 | +0.336 |
| Spearman(r*, conf) | +0.867 | +0.733 |
| selective-pred AUC: r* vs conf | 0.831 vs 0.803 | 0.797 vs 0.803 |
| **hi-conf subgroup: DF(certified) vs DF(uncertified)** | **0.900 (n=50) vs 0.500 (n=14)** | **0.926 (n=27) vs 0.730 (n=37)** |

**Two-part honest verdict:**
1. **As a global ranker, r\* ≈ confidence** — they are strongly correlated (ρ 0.73–0.87) and the selective-
   prediction AUCs are within ±0.03 (r* marginally ahead at k=8, marginally behind at k=16). We would NOT
   claim "r* beats confidence."
2. **But r\* carries a real component confidence lacks.** Among *high-confidence* decisions — the ones naive
   selective prediction trusts unconditionally — the subset r* flags as **uncertified** has DF 0.50 / 0.73,
   vs 0.90 / 0.93 for the certified subset. r* isolates **confident-but-measurement-fragile** decisions:
   confident on the specific null-fill in x_hat, yet a prior-plausible measurement-consistent twin flips them.
   That is precisely the converse (P1) operationalized — the *confidently-wrong-on-hallucinated-content*
   failure mode, which softmax confidence is structurally blind to.

## Verdict on direction #1: VIABLE, with an honest scope
- **Cheap:** yes — 50–80 ms/scene, no BaB needed for the trust score, no external deps.
- **Improvement:** yes but *targeted* — not a blowout over confidence as a ranker; the genuine, defensible
  contribution is the measurement-aware catch of confident-but-fragile decisions (the high-confidence
  subgroup split).
- **Real innovation, not A+B:** yes — a *sound* decision certificate over the exact measurement fiber is a new
  object (nonlinear-f fiber-invariance; it structurally escaped the round-1 renaming kill-pattern), and it
  yields a GT-free reliability score whose value is exactly the paper's thesis at the decision level.

## What the paper-grade version needs (next, ranked)
1. **Stronger, still-verifiable classifier** (fold eval-mode BN into conv — exact; recovers the 0.537 SmallCNN
   margins). The 0.375 net here makes DF noisy and likely *understates* the separation.
2. **Real correctness, not just DF:** run the recon pipeline on STL10 `train` (labeled, disjoint from the
   classifier's `test` training; disclose fusion-refiner leakage) and correlate r* with `pred==true_label`.
3. **The catch as the headline metric:** precision/recall of r* flagging the low-DF cases *inside* the
   high-confidence set (where it beats confidence by construction).
4. Multi-seed/operator, k sweep {8,16,32}, and the design-dual (certificate-guided vs random-axis vs
   eigen-order acquisition) — reviewer-flagged ratio ceiling = k, so headline at k=32.

All arms keep `A x_hat = y` exact; the certificate is slab-relative determination of the decision, never
correctness. Result JSONs: `outputs/.../detail_fusion_paper/sandwich/{gate,diag,risk}_k*.json` (local-only).
