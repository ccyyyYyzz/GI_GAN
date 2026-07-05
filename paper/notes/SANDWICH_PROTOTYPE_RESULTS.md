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

## Stage 2 — the paper-grade pass overturns the Stage-1 positive (rigor)
The Stage-1 "measurement-aware catch" was tested honestly with (a) a **properly trained, still-verifiable**
classifier (train with BN, fold exactly at eval — `rel|diff|=1.6e-4`, 100% argmax agreement; heldout acc
**0.651**, on-truth acc **0.605**) and (b) **real labels** on 256 **double-clean** scenes (the classifier's
own heldout STL10-`test` subset — classifier never trained on them; the fusion refiners/priors never saw
`split=test` at all). `risk2`, k∈{8,16}:

| signal (256 scenes, real classifier) | r* | softmax confidence |
|---|:---:|:---:|
| Spearman vs decision-fidelity | +0.098 | **+0.254** |
| Spearman vs label-correctness | +0.095 | **+0.235** |
| selective-pred AUC (correctness) | 0.429 | **0.523** |
| catch inside hi-conf set — recall (correctness) | 0.14 | **0.32** |

**r\* loses to free softmax confidence on every axis, including the "catch" that was the whole novel claim.**
Diagnosis: on the strong classifier the cheap CROWN-only r* is **degenerate — identically 0** at k=16 (nothing
certifies at any width in the grid). The fragility probe (k=8, w=0.05/0.1σ) resolves *why*: PGD finds **~0**
flips while CROWN certifies **0** → **GAP-dominated**. So the decisions are **not** genuinely fragile (they are
likely certifiable), but the **cheap verifier is too loose to prove it**; a non-degenerate r* would require
heavy α-CROWN + deep branch-and-bound (~tens of s/scene, still GAP-dominated at 2048 boxes) — which **breaks the
"small compute" premise outright**.

The Stage-1 positive (0.90-vs-0.50 catch) was an **artifact** of the weak 0.375 classifier (poorly-calibrated
confidence) + decision-fidelity self-agreement instead of correctness — the **same class of artifact** as C1's
global-B-shift illusion. It did not survive.

## Verdict on direction #1: NOT a cheap win — do not pursue as specified
- **Cheap?** Only while degenerate. A non-degenerate certificate on a real classifier needs heavy BaB → the
  "small compute" premise fails.
- **Improvement over the free baseline?** No. Where r* is cheaply computable (weak classifier) it merely ties
  softmax confidence; on a real classifier the cheap r* is all-zero and loses. No demonstrated advantage.
- **What genuinely survives** (worth keeping, not a paper on its own):
  1. **Geometry-not-compute infrastructure** — exact k-dim slab (`||A b_j||=3.6e-16`) + sound hand-rolled
     verifier. The dimensional collapse (4096→k) is real and reusable; it simply did not buy a winning app here.
  2. **A verifier-limited converse**: at prior-plausible scale the cheap verifier cannot certify decisions AND
     (for a strong classifier) PGD cannot easily flip them — the certifiability of GI decisions is
     **inconclusive without heavy compute**, which is itself evidence against a cheap certificate.
- **Honest lineage of the negative:** this is the third rigor-caught illusion of the session (C1 B-shift;
  deep-audit's 15 renamings; now the r* trust-score). Each cost little and prevented an overclaim.

## Recommendation
Stop investing in a new *method/quantity* for GI (the deep-innovation audit already showed the linear-Gaussian
layer is mined out; this prototype shows the one structurally-novel escape does not cash out cheaply). The
highest-value remaining work is **consolidating the existing paper**: own the lineage
(Backus–Gilbert → Barrett–Myers → MacKay → us), fold in the defensive citations, and add the free
**no-adaptation lemma** — the contribution is the converse + governance framing for DL-era GI, not another gadget.

Result JSONs: `outputs/.../detail_fusion_paper/sandwich/{gate,diag,risk,risk2}_k*.json` (local-only).
