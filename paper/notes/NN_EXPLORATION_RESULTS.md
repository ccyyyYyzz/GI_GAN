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

---

# Follow-up experiments — all three ran (2026-07-04)

The user asked to try **all three** proposed follow-ups. Each was run for real on cached seed-0 dev STL10
(5% sampling), LPIPS-scored, PSNR-tolerance matched to the gate. JSONs live in
`outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/`
(`inr_vqgan_fusion.json`, `deq_milestone0.json`, `seq_ssm_gi.json`). **Honest verdicts below — two negatives,
one near-tie with a real sub-win.**

## ① INR-as-structure-prior × VQGAN-detail fusion (`inr_vqgan_fusion.py`)
Fit a SIREN per scene (K=64 dev images, cached to `dINR_seed0_dev_K64.pt`), form `d_INR = P0(x_INR − x0)`,
and run the exact null-space dial `x̂_B = x0 + P0(d_S + B(d_G − d_S))` for structure branch `d_S ∈ {d_INR, d_VQAE}`.
Balanced point = min-LPIPS `B` under `PSNR ≥ (branch B=0 PSNR) − 0.5`.

| branch | standalone PSNR / LPIPS | balanced `B` | balanced PSNR / LPIPS |
|--------|:---:|:---:|:---:|
| **INR structure** | **22.84 / 0.248** | 0.50 | **22.42** / 0.197 |
| VQAE structure | 22.53 / 0.293 | 0.55 | 22.11 / **0.192** |
| VQGAN detail (B=1) | 20.91 / 0.167 | — | — |

- **Real sub-win (standalone):** INR **strictly dominates** VQAE as a structure prior — better on *both* axes
  (+0.31 dB, −0.045 LPIPS) **and training-free**. Confirms the Prototype-B finding at K=64.
- **Fusion is a near-tie (honest):** once VQGAN detail is mixed in, it dominates perception, so swapping the
  structure branch barely moves the balanced point — INR-structure ends **+0.31 dB PSNR but +0.005 LPIPS**
  vs VQAE-structure. Not a quality leap.
- **Practical value:** INR is a **drop-in, training-free replacement for the trained VQAE structure branch** —
  comparable fused quality, slightly more fidelity, zero training cost. A simplification, not a breakthrough.

## ② DEQ Milestone-0 — iterate the frozen refiner in the null space (`deq_milestone0.py`)
Test the DEQ premise cheaply: does re-applying the frozen VQGAN refiner as a fixed-point map,
`x_{k+1} = audit(x0 + P0(refine(x_k) − x0))`, beat the one-step fusion?

| k | 1 | 2 | 3 | 5 | 10 | 20 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| PSNR | **20.91** | 18.76 | 17.54 | 16.44 | 15.56 | 15.04 |
| LPIPS | **0.167** | 0.194 | 0.223 | 0.255 | 0.287 | 0.300 |

- **Clean NEGATIVE:** quality **monotonically degrades** with iteration on *both* axes; **k=1 (one-step) is
  optimal**. The frozen refiner is **not a contraction** in the null space — iterating walks away from the data.
- **Scoping lesson (not a refutation of DEQ):** you cannot get DEQ benefits for free by looping a
  non-contractive frozen refiner; a real DEQ would have to be **trained** with an explicit contraction/stability
  constraint. This is a publishable rigor result — it forecloses the cheap version and names the requirement.

## ③ Liquid/sequence SSM reading the raw bucket stream (`seq_ssm_gi.py`)
The most-novel angle: an S4D-style **diagonal state-space model** consumes the measurement as a 1-D sequence
`y_t` tagged with learned per-pattern (operator-row) embeddings, trained from scratch on STL10 (1500 steps,
tiny), output pixels optionally projected onto `{x:Ax=y}`.

| variant | PSNR | LPIPS |
|---------|:---:|:---:|
| SSM raw | 12.57 | 0.746 |
| SSM + exact projector | 22.04 | 0.465 |

- **NEGATIVE at this scale:** the raw SSM is far off (12.6 dB); the **exact projector does all the work**,
  lifting it to anchor-level *fidelity* (22.04 dB) but leaving it **perceptually worst-in-class** (LPIPS 0.465
  vs LMMSE 0.423, vs fusion 0.19). The learned sequence part adds nothing the projector didn't already give.
- Consistent with the a-priori ranking that placed Liquid/SSM low (#4). The **range/null projector is
  vindicated** (it rescues an otherwise-useless net), but the "read the bucket stream with a liquid net"
  premise is **not a shortcut to quality** — it needs far more scale/training to even reach the linear anchor.
  Undertrained-probe caveat applies, but the gap to competitive is large.

## Net conclusion across the exploration
Of the emerging architectures, **INR/SIREN is the one genuine win** — a training-free structure prior that
beats the classical LMMSE anchor outright and matches the trained VQAE inside the fusion. **DEQ-by-iteration**
and **Liquid/SSM-by-sequence** are honest negatives at this scale, each yielding a useful *rigor* statement
(one-step is optimal for a non-contractive refiner; the projector — not the sequence net — carries the
reconstruction). The exact range/null projector is the common thread that survives every probe. No result
overturns the impossibility-first thesis: all quality is still bought inside `P_0` with `A x̂ = y` held exact.
