# NN Architecture Exploration — First Real Result (2026-07-04)

Beyond GAN. Ranked 6 emerging architectures (see `NN_EXPLORATION.md`); top picks: DEQ null-space refiner
(#1), INR/SIREN (#2). Prototyped **INR/SIREN** first (fastest, training-free).

## Prototype B — INR/SIREN with exact null-space fitting (`inr_nullspace_gi.py`)
Per-scene test-time fit (DIP-style, **no training set**): min_θ ‖A f_θ − y‖² + 0.02·TV(f_θ); ~1500 Adam
steps, ~8 s/image on the RTX 4060; optional exact-projector rectify onto {x:Ax=y}. Scored on 6 seed-0 dev
STL10 images at 5% sampling vs the LMMSE anchor and the current VQGAN reconstruction.

| method | PSNR (dB) | LPIPS |
|--------|:---:|:---:|
| **INR/SIREN** | **23.78** | 0.229 |
| INR + exact audit | 23.78 | 0.228 |
| LMMSE anchor `x0` | 22.74 | 0.423 |
| VQGAN `x_G` | 20.05 | 0.178 |

## Findings (honest)
- **INR dominates the linear anchor**: +1.0 dB PSNR AND LPIPS 0.423→0.229 — a large perceptual gain, with
  **zero training and no dataset**. A novel-architecture win over the classical baseline is real.
- **INR occupies a distinct corner of the perception–distortion plane** from VQGAN: INR is **higher-fidelity**
  (+3.7 dB PSNR) but **less perceptually sharp** (LPIPS 0.229 vs 0.178). They are complementary, not competing.
- The exact projector barely moves INR (it is already measurement-consistent from the fit), confirming the
  projector-wrapping mechanic is clean and cost-free.
- Caveat: 6-image probe, single rate/seed, no val/dev firewall — a signal, not a locked claim.

## The experiment this grew into
**INR-as-structure-prior fusion.** INR (high-PSNR structure) is a training-free stand-in for / complement to
the VQAE structure branch; fuse INR-structure + VQGAN-detail through the exact null-space dial
`x̂_B = x0 + P0(d_INR + B(d_G − d_INR))` and test whether it captures INR's fidelity AND VQGAN's perception.
Next candidates: DEQ Milestone-0 (does iterating in the null space beat one-step fusion?); Prototype C
(raw-bucket-stream SSM/liquid sequence encoder + operator-row tags + exact projector — the most-novel angle).
