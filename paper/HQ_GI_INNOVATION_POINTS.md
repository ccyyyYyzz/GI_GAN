# HQ Ghost-Imaging Reconstruction — Ranked Innovation Deck

**Goal of this deck.** Not accountability/certification (that is the sibling paper, `INNOVATION_POINTS.md`) but **raising the reconstruction quality ceiling** of our exact null-space fusion — pushing LPIPS/perceptual realism at 2–10% sampling past the current locked `−32.6% LPIPS` operating point, ideally without paying PSNR.

**Invariant every point respects.** All edits live inside the exact null projector: `x_hat = x0 + P0(·)`, so `A x_hat = y` to ~2e-15 by construction (`A P0 = 0`). Quality is bought only by *what content we write into `Null(A)`* and *how the measured budget is spent* — never by relaxing consistency (the one exception, IP-14, is explicitly flagged as a separate soft-consistency arm for the noisy-`y` regime).

**Shared assets referenced throughout.** `vqgan_detail_fusion.py` (`fuse`, `prep_residuals`, `apply_band`, `weight_map`, `radial_masks`, `gen_specs`, `select_operating_point`, `refine`), the exact `ExactRangeNullProjector` in `src/projections.py` (`null_project_flat` / `data_anchor_flat` / `audit_flat`, `relative_measurement_error`), the measurement-audited **LMMSE anchor** `x0` (`hq.EmpiricalLMMSE`), frozen **multi-seed VQAE (`d_A`) and VQGAN (`d_G`) refiners** (`refiner_ckpt`), structured DCT+Hadamard+random operators at 2/5/10%, and the **locked val→dev multi-seed gate** (`cmd_canary` / `cmd_gate`).

---

## The 14 Innovation Points

### HQ-1 — NDS-Diff: null-space-confined diffusion over the audited anchor
**Idea.** Replace the *single, statically generated* detail residual with a short reverse-diffusion trajectory whose every `x0|t` estimate is rectified back onto the exact null space. A full generative posterior writes richer, sharper detail than any two-point blend of `d_A`/`d_G`, while `A x_hat = y` holds at every step.
**Mechanism.** Run a reverse loop (25 steps). At each step estimate `x0|t`, then `x0_hat = data_anchor_flat(y) + null_project_flat(x0|t − x0_anchor)`; because `A P0 = 0` this is exactly DDNM's rectification specialized to *our* matrix-free projector instead of an operator-specific `A†`. Optionally add DDRM-style per-mode noise on the measured modes using the LMMSE `λ/(λ+σ_i²)` contraction. Finish with one `audit_flat` against float drift.
**Builds on.** DDNM/DDNM+ (R027), DDRM per-σ (R070), MCG null-only update (R074), Score-SDE inpainting-as-null-fill (R061).
**Uses our assets.** Swaps DDNM's `A†` for `ExactRangeNullProjector`; the **frozen VQGAN decoder (`decode_embeddings`) is the `x0` predictor**, so nothing is trained; seeds the trajectory from `x_G`; scores through the same locked gate as the vqae/vqgan/fusion arms.
**First experiment.** Add a `diffusion` arm to `vqgan_detail_fusion.py`: `nds_diff(x0_anchor, y, projector, steps=25)`, route output through `fuse`→`audit_flat`→`score_method`. Success = `relmeaserr ≤ 1e-5` AND LPIPS beats `fusion_balanced` at PSNR drop ≤ 0.5 dB on seeds 0/1/2.
**Boldness.** ambitious.

### HQ-2 — B-as-guidance-scale: one governed knob that sweeps a *true* P–D frontier
**Idea.** Recast the scalar `B` from a linear blend weight into the **guidance temperature** of the HQ-1 sampler. `B=0` recovers the audited anchor (max fidelity); `B=1` recovers full DDNM detail (max perception); intermediate `B` traces a monotone perception–distortion curve rather than a chord between two frozen images — while keeping the paper's single-pre-registered-scalar governance story.
**Mechanism.** Scale the per-step null increment: `x0_hat = x0_anchor + null_project_flat( B · (x0|t − x0_anchor) )`. Sweep `B` on val, select one `B` on the locked gate, report the full `B`-swept Pareto. Reuse `radial_masks` to make `B` band-dependent (low-freq fidelity, high-freq perception) exactly as the current band fusion does.
**Builds on.** Perception–Distortion tradeoff (R052/R060), DNI continuous-effect precedent (R059/R066), DPS soft-gradient scale (R071), DiffPIR `ρ_t` balance (R075), DDNM rectification (R027).
**Uses our assets.** Directly generalizes the `gen_specs()` `scalar_B` grid and `select_operating_point()`; swaps the linear `null_blend` for the guided increment; identical val→dev firewall and `relmeaserr ≤ 1e-5` audit.
**First experiment.** Replace the scalar branch of `fuse()` with a guided increment for `B ∈ {0,0.1,…,1.0}`; regenerate the val Pareto, confirm monotone LPIPS-vs-PSNR ordering, then `cmd_gate`. Success = a `B*` beats the locked `−32.6%` at equal-or-smaller PSNR cost.
**Boldness.** near-term-win.

### HQ-3 — Codebook-lookup null-space refiner (CodeFormer-style code prediction)
**Idea.** Our VQGAN detail branch does nearest-neighbor code lookup on features from a *corrupted* (2–5% GI) input, so at the lowest rates lookup returns wrong codes and `d_G` is noisy exactly where we most want signal. CodeFormer's finding: a transformer predicting the code sequence *globally* is far more robust to heavy degradation than local lookup. A cleaner, more semantically coherent `d_G` should lift LPIPS most at the lowest rates where lookup breaks.
**Mechanism.** Train a small GPT-style transformer that takes `x0` encoded to a token grid and predicts the HQ codebook index sequence, decoded by the **existing VQGAN decoder** to give `x_G`; set `d_G = x_G − x0`. Fusion path unchanged, so consistency is preserved. Add CodeFormer's scalar-`w` Controllable Feature Transformation as a second knob orthogonal to `B`.
**Builds on.** CodeFormer (R058 code-transformer + `w`), VQGAN (R044/R055), VQ-VAE (R042), ViT-VQGAN L2-normalized codes to kill dead codes (R056), RestoreFormer (R059).
**Uses our assets.** Reuses the frozen VQGAN decoder + codebook, the LMMSE anchor, the exact `P0`, and `fuse()` verbatim; only the `d_G` *generator* changes. Refiner-checkpoint plumbing (`refiner_ckpt`/`refine`) already hosts it.
**First experiment.** Add a detail-branch variant that replaces argmin lookup with a 2-layer code predictor trained on ~2000 STL10 64×64 HQ code grids; feed `x_G` as `d_G` into `fuse()`, run `cmd_canary` at 2% seed0, compare LPIPS/PSNR against the lookup arm at matched `B`.
**Boldness.** ambitious.

### HQ-4 — Latent-space (VQ) null-space diffusion with a pixel-space consistency bridge
**Idea.** Pixel-space null-diffusion can inject perceptually-noisy, measurement-invisible high-frequency energy. Running the diffusion *in the VQGAN latent/codebook* biases every sample onto the natural-image manifold, so the null content it writes is realistic texture, not arbitrary HF noise — higher perceptual quality at the same fidelity.
**Mechanism.** Diffuse in latent `z`. Each step: `decode_embeddings(z0|t) → x0|t`, rectify `x0_hat = x0_anchor + null_project_flat(x0|t − x0_anchor)`, re-`encode` for the next step. The decode→project→encode loop keeps the trajectory on the codebook manifold while the pixel-space projection guarantees exact consistency.
**Builds on.** DDRM/DDNM spectral null-fill (R070/R027), latent-diffusion inverse solvers (survey R073), VQGAN/Taming Transformers (R042), MCG manifold-constraint rationale (R074).
**Uses our assets.** Reuses the frozen VQGAN `encode`/`decode_embeddings`/`quantize` already wired in `regen_seed()`, the anchor-initialized latent refiner as the initializer, `ExactRangeNullProjector` as the bridge, the multi-seed gate. No retraining.
**First experiment.** Extend `refine()` into a `K∈{4,8}`-step latent loop (decode→project→encode) initialized from `z` of `x_G`; score `fast_means` on seed0 val vs the single-shot refiner. Success = lower LPIPS/RAPSD at matched PSNR, `relmeaserr ≤ 1e-5` after final `audit_flat`.
**Boldness.** moonshot.

### HQ-5 — RestoreFormer cross-attention fusion (content-adaptive `β(x)` beats scalar `B`)
**Idea.** `d_A` and `d_G` are combined by an image-independent radial band schedule; but *which* prior is trustworthy varies per image and region (smooth sky → structure, texture → detail). RestoreFormer shows spatial cross-attention over an HQ dictionary beats fixed spatial-feature-transform blending. A learned, content-adaptive fusion should extract more quality from the *same two residuals*.
**Mechanism.** A lightweight cross-attention head with queries from anchor/structure features and keys/values from VQGAN HQ code embeddings produces a spatially-varying blend field `β(x)` replacing scalar `B` / the radial `weight_map`. Still route through `P0`: `x_hat = x0 + P0(d_A + β(x)·(d_G − d_A))`, so consistency holds regardless of `β`. `β ≡ B` is a special case → can only improve on the tuned baseline.
**Builds on.** RestoreFormer spatial cross-attention (R059), CodeFormer CFT gating (R058), VQ-VAE-2 top/bottom factorization mirroring our `d_A`/`d_G` (R043), VQGAN (R044).
**Uses our assets.** Generalizes `apply_band`/`weight_map`/`radial_masks` (scalar `B` and radial masks become special cases); reuses both trained refiners, anchor, projector; same locked gate.
**First experiment.** Replace static radial weights with a per-pixel `β` from a tiny CNN/attention head over `(x0, d_A, d_G)`, trained only to maximize LPIPS at fixed consistency (automatic via `P0`) on STL10 train. Compare vs best fixed-`B`/radial arm at 5% seed0 through `cmd_gate`; verify it holds on seeds 1–2.
**Boldness.** ambitious.

### HQ-6 — Null-Space MoDL: unrolled, data-consistency-free refiner replacing scalar `B`
**Idea.** A single global scalar `B` cannot adapt the perception–distortion mix per region. An unrolled MoDL/ISTA-style refiner that alternates a learned denoiser and re-projection into `Null(A)` makes the blend spatially and iteratively adaptive, pushing both LPIPS and PSNR past any fixed `B`.
**Mechanism.** Unroll `K=3–6` weight-shared stages living entirely in the null space, so data consistency is free-by-construction (MoDL needs a CG block only because it lacks an exact null projector; we own `P0`): `z_k = D_w(x_k)`, `x_{k+1} = x0 + P0(z_k − x0)`, so `A x_{k+1} = y` exactly every stage. Initialize `x_0 = x_hat_B` at the locked `B` so stage-0 reproduces the current Pareto point and can only improve. ~100k params (weight-sharing) suits the small STL10 testbed.
**Builds on.** MoDL (R038, CG-free here), Deep Null-Space Learning `L = Id + P_ker N` (R021), ISTA-Net learned per-phase step/threshold (R042), Deep Decomposition gated denoiser (R022).
**Uses our assets.** Swaps the scalar-`B` line in `fuse`; reuses `P0`, LMMSE `x0`, `d_A`/`d_G` as a two-channel `D_w` input, and the multi-seed gate. No new operator/prior training.
**First experiment.** Add a 3-stage `D_w` initialized so stage-0 = `x_hat_B` (identity denoiser); train only `D_w` on dev against LPIPS+PSNR with `x0` frozen; compare on the locked gate at 5%. Success = LPIPS below `−32.6%` at equal-or-better PSNR on ≥2/3 seeds. ~1 GPU-hour.
**Boldness.** near-term-win.

### HQ-7 — PnP-RED in the null space: VQGAN as a convergent denoiser (zero training)
**Idea.** A feed-forward fusion consults the detail prior *once*. A plug-and-play/RED fixed-point loop calls it repeatedly, letting fine detail accrete while the range stays pinned to `y`. RED supplies a convergence/optimality story a black-box refiner lacks — a paper-grade principled claim at *zero* training cost.
**Mechanism.** Projected fixed-point iteration that never leaves the feasible set: `x_{k+1} = x0 + P0( x_k − η (x_k − f(x_k)) )`, where `f` is the VQGAN refiner as the RED denoiser. IDBP guarantees progress is confined to the null space, so iterating changes exactly the invented detail. Pick `η` via RED passivity; if it fails, fall back to PnP-ADMM. 5–15 calls, inference-only.
**Builds on.** RED (R044 fixed-point/ADMM), PnP priors (R043), IDBP null-space-improvement theorem (R045), DDNM per-step rectification (R026).
**Uses our assets.** Wraps the existing `refine()` VQGAN latent refiner as `f`; reuses `P0`, `x0`; **no training** — cheapest possible quality bump and a clean ablation against one-shot fusion.
**First experiment.** Add a 10-iteration null-space RED loop around the locked refiner in `vqgan_detail_fusion_locked.py`, sweep `η ∈ {0.3,0.5,1.0}`, measure LPIPS/PSNR vs one-shot at 5% and 2%; log `‖x_{k+1}−x_k‖` for the convergence figure. Success = monotone LPIPS improvement to a stable fixed point beating one-shot at matched PSNR.
**Boldness.** near-term-win.

### HQ-8 — Consistency-gap-driven per-image `B` (residuals decide their own mixing; no training)
**Idea.** A fixed global `B` leaves quality on the table: where VQAE and VQGAN agree in the null space, any `B` is safe; where they disagree, a wrong `B` injects hallucination or blur. A **closed-form per-image `B` computed from the disagreement `d_G − d_A` itself** targets this directly. The earlier *learned* gate likely failed because it predicted `B` from features; here `B` is a closed-form function of quantities we already compute.
**Mechanism.** Per image, disagreement energy `g = ‖d_G − d_A‖² / (‖d_A‖² + ε)` and a monotone map `B_i = B_max·sigmoid(a(g − g0))`. Small disagreement → trust VQGAN (cheap detail win); large → stay near VQAE (avoid hallucination). Fuse as today. Two–three scalars `(a, g0, B_max)` tuned on dev, then frozen. Per-*image* scalar (not per-pixel) → cannot introduce seams.
**Builds on.** Perception–Distortion / RDP (R050/R051), DNI single-scalar MSE↔GAN axis (R066/R053), Hallucination limits — injected detail must be measurement-near-invisible, i.e. the null residual (R085), EI Prop 1 — MC gives no null info so disagreement is the only per-image cue (R016).
**Uses our assets.** Reuses `prep_residuals` (`d_A`,`d_G` already null-projected), exact `P0`+`audit_flat`, and `select_operating_point`'s dev/gate split verbatim. Only `fuse()` gains a `consistgap` spec.
**First experiment.** Add spec `('consistgap', (a,g0,Bmax))` in `fuse()`; compute per-image `g`, run the canary on seed0 dev, sweep ~9 `(a,g0)` pairs, compare LPIPS/PSNR vs frozen `B_star` via the locked selector. Win = ≥ `−32.6%` LPIPS at equal-or-better PSNR on the gate split.
**Boldness.** near-term-win.

### HQ-9 — Three-prior cascade with a governed 2-D dial (LMMSE → VQAE → VQGAN, each stage exact)
**Idea.** A single two-prior fusion is a chord of one interval; a cascade lets each prior fix what the previous cannot, and reliably beats any single member on the P–D plane. Because every stage edits only the null space and re-projects, the cascade stays exactly consistent end-to-end.
**Mechanism.** Stage 1: `x0` (LMMSE). Stage 2: `x1 = x0 + B1·P0(x_A − x0)`. Stage 3: `x2 = x1 + B2·P0(x_G − x1)` — detail measured against the *structured* intermediate, not `x0`, so `d_G' = P0(x_G − x1)` differs from today's `d_G`: VQGAN adds only what VQAE missed, cutting double-counting and PSNR cost. `(B1,B2)` is a 2-D governed dial; the current scalar fusion is the special case `B1=1`. Every stage ends in `null_project_flat`+`audit_flat`.
**Builds on.** DNI/ESRGAN net-interp governed multi-point dial (R066/R053), Deep Cascade of CNNs + MoDL consistency-projected refinement (R026/R027), CodeFormer `w` / VQFR staged fidelity control (R058/R057), P–D/RDP ensemble reachability (R050/R051).
**Uses our assets.** Reuses `x0`, `x_A`, `x_G`, `P0`, `audit_flat`, `select_operating_point`; only new code is `d_G'` and a 2-D `(B1,B2)` grid in place of the 1-D sweep. Refiner and anchor-init search unchanged.
**First experiment.** Add a `cascade` spec taking `(B1,B2)`; on seed0 dev sweep an 11×11 grid (all closed-form on cached residuals), select via dev→gate, compare gate LPIPS/PSNR to the best 1-D scalar arm. Success = strictly dominates the scalar Pareto point.
**Boldness.** near-term-win.

### HQ-10 — Perception-distortion-optimal endpoint interpolation (certified operating point)
**Idea.** The P–D theorem says no single reconstruction optimizes both fidelity and realism; ESRGAN/DNI interpolate a distortion-oriented and a perception-oriented model to trace the curve without artifacts. Our `B` already interpolates two *fixed* endpoints; making the endpoints themselves a **distortion-optimal** and a **perception-optimal** null-space filler and interpolating inside `P0` lets us hit any P(D) point and pick the max-perception point at a *certified* PSNR budget.
**Mechanism.** `d_MMSE` (posterior-mean/L2-optimal filler — LMMSE or distortion-trained decoder, the low-distortion endpoint) and `d_perc` (adversarial VQGAN/diffusion filler, high-perception endpoint). Fuse `x0 + P0((1−α)d_MMSE + α d_perc)` and sweep `α`; every point keeps `A x_hat = y`. Choose `α` by maximizing LPIPS subject to a PSNR-drop budget → a principled certified operating point, not a hand-tuned `B`.
**Builds on.** P–D tradeoff (R050), RDP (R051), ESRGAN/DNI net-interp (R053/R066), PULSE — MSE-optimal SR is the manifold average, justifying a distinct distortion endpoint (R065), LPIPS (R052).
**Uses our assets.** Reuses the LMMSE anchor as/near the distortion endpoint, VQGAN detail as the perception endpoint, the existing `B`-sweep + `select_operating_point`, `P0`, and the locked P–D reporting.
**First experiment.** Define `d_MMSE`/`d_perc`, sweep `α ∈ {0,.25,.5,.75,1}` through `fuse()`, plot LPIPS-vs-PSNR at 5% across seeds via the gate/report commands. Verify it dominates the single-`B` curve; pick the max-LPIPS point under a fixed PSNR budget.
**Boldness.** near-term-win.

### HQ-11 — Anchor-conditioned PiGDM in the null space (LMMSE second-order statistics steer the sampler)
**Idea.** DPS/MCG steer diffusion with an *isotropic* measurement gradient; PiGDM improves on them with the operator pseudo-inverse and a Gaussian-posterior covariance. **We already have a fitted LMMSE (covariance + anchor)**, so we can build the PiGDM guidance almost for free and confine it to `Null(A)` — sharper, better-placed detail because the sampler is steered by the GI operator's actual spectral geometry, not a generic gradient.
**Mechanism.** Each reverse step: `x0_hat = x0_anchor + P0( x0|t + Σ_null·(μ_LMMSE(y) − x0|t) )`, where `Σ_null` is the null-restricted LMMSE posterior weighting; then `audit_flat`. This is PiGDM's covariance-weighted likelihood, but the exact null projector guarantees consistency instead of PiGDM's soft pinv step.
**Builds on.** PiGDM (survey R073), DPS Tweedie posterior-mean (R071), score-based medical `A=P(Λ)T` proximal (R069), GSNR predictability `ρ_j²=c_j/(c_j+μ_j)` for weighting null modes (R003).
**Uses our assets.** Reuses `EmpiricalLMMSE` (`anchor()` and `uncertainty_map()`) as both range anchor and null covariance preconditioner; `ExactRangeNullProjector` for hard consistency; frozen VQGAN as the score. No retraining.
**First experiment.** Standalone script importing `hq` (LMMSE) and `src.projections`: on seed0 val cache compare 20-step samplers — (a) isotropic null-DDNM, (b) LMMSE-preconditioned null guidance, (c) `fusion_balanced` — via `fast_means`. Success = (b) lowers LPIPS and RAPSD vs (a) at equal PSNR.
**Boldness.** ambitious.

### HQ-12 — Perception-weighted operator design (put the row space where LPIPS lives)
**Idea.** Everything the operator *measures* is fixed ground truth; only the null space is invented (and perceptually risky). If we choose the `m` measured rows to span the directions humans actually care about, the perceptually-costly null space shrinks *in a perceptual sense* even at fixed 2–10% sampling — attacking the LPIPS floor directly rather than trading PSNR for it.
**Mechanism.** Replace DCT/Hadamard/random rows with an operator whose rows maximize captured *perceptual* variance: (1) pull VGG/LPIPS feature Jacobians on STL10 train, form the image-space Gram `G = E[JᵀJ]`; (2) take the top-`m` generalized eigenvectors (perceptual-PCA masks) as `A`'s rows, optionally binarized via straight-through for DMD realism. The measured `y` then already carries the semantically dominant content, so `P0(...)` only invents perceptually-cheap detail.
**Builds on.** Deep-learning X-ray GI PCA/SVD masks (card #91), single-pixel CS learned bases (#9/#13), Blau–Michaeli P–D bound being beaten perceptually (#50/#60), GSNR basis coverage/predictability (#2).
**Uses our assets.** `A` plugs into `get_exact_projector` in `src/projections.py`; `P_R`/`P0`/`data_anchor` and the fusion `x_hat_B` unchanged (still exact). Reuses LMMSE anchor, both refiners, and the locked gate. Operator-swap harness exists (`eval_pattern_swap.py`, `calibrate_operator_equivalence.py`).
**First experiment.** Compute the LPIPS-Jacobian Gram over a few hundred STL10 train images, take top-`m` eigenvectors as a fixed operator at 5%, run through `get_exact_projector` + existing fusion, compare LPIPS/PSNR under the locked gate vs DCT/Hadamard rows. No prior training; single eval pass.
**Boldness.** near-term-win.

### HQ-13 — Fusion-in-the-loop operator learning (co-design `A` and the dial `B` end-to-end)
**Idea.** The operator is fixed *before* the priors act, blind to what our VQAE+VQGAN fusion can and cannot hallucinate. Co-optimizing the sensing rows *for our specific reconstructor* makes the operator measure exactly the content the fusion would otherwise get wrong — a strictly better quality-vs-sampling curve than any prior-agnostic operator.
**Mechanism.** Make operator rows a learnable `θ` (straight-through binarization for DMD), unroll `y = A_θ x → LMMSE anchor → exact fusion x_hat_B → LPIPS+distortion loss`, backprop through the differentiable exact projector into `A_θ` and `B` jointly. Learned-operator/deep-unrolling co-design (Learned Primal-Dual style) but only the *sensing matrix* and *fusion dial* train; heavy priors frozen. `A P0 = 0` held analytically, so consistency is never a loss term.
**Builds on.** Learned Primal-Dual (#29), Variational Network (#28), jointly learned pattern+recon SPI (#97/#13), NPN joint null-projection+predictor (#4).
**Uses our assets.** Reuses the matrix-free differentiable projector (`row_project_flat`/`null_project_flat`), the fixed fusion as a differentiable head, learnable-pattern infrastructure (`sanity_learnable_patterns.py`, `use_learned_patterns` flag) and the straight-through binarization already present. Priors frozen; only `A_θ`,`B` train. Multi-seed gate.
**First experiment.** From `src/sanity_learnable_patterns.py` at 5%, freeze priors, add the exact-fusion head + LPIPS loss, train only `A_θ`,`B` for a few hundred steps on STL10; verify `audit_flat` stays at numerical zero; check gated LPIPS vs fixed DCT/Hadamard.
**Boldness.** ambitious.

### HQ-14 — Learned Primal-Dual two-stream refiner that also cleans the range space (noisy-`y` regime)
**Idea.** Our fusion pins the range to `y` exactly — correct when `y` is clean, *wrong* under real GI noise, where it locks in measurement noise and caps PSNR. A Learned-Primal-Dual/Deep-Decomposition architecture adds a **second learned stream** that corrects the range (noise) component while the null stream invents detail, breaking the noise ceiling a null-only fusion cannot cross. *(Runs as a separate soft-consistency arm — does not touch the locked exact-fusion line.)*
**Mechanism.** Generalize to `x_hat = P_R(x0 + F(x0,y)) + P0(G(d_A,d_G))`, where `G` is the null detail refiner and `F` a range residual net trained to remove measurement noise, bounded by `‖A F − noise‖`. Couple via 3–5 unrolled LPD steps (learned primal/dual proximals). Measurement consistency becomes *soft and noise-aware* (`A x_hat ≈ y`) — the correct choice when `y` itself is noisy.
**Builds on.** Learned Primal-Dual (R040, +6–10 dB CT), Deep Decomposition `A = P_r F + P_n G` (R022, ~4 dB over null-only), Variational/ADMM-Net (R039/R041), P–D via range cleanup toward posterior mean (R061).
**Uses our assets.** Reuses `P_R`/`P0`, LMMSE anchor as the `F`-initializer, both refiners as the `G`-stream. Requires relaxing exact consistency → runs as a **separate arm** for a noise-robustness section. Extends `gan_high_quality_gi.py`'s operator.
**First experiment.** On a noisy 5% operator (Gaussian read-noise on `y`), train a 2-stream Deep-Decomposition refiner (3 unrolled LPD steps) and compare PSNR/LPIPS vs the exact null-only fusion forced to fit noisy `y`. Success = higher PSNR at equal LPIPS under noise.
**Boldness.** ambitious.

---

## TOP 3 highest-quality-ceiling bets
1. **HQ-1 — NDS-Diff (null-confined diffusion).** The single largest ceiling lift: a full generative posterior replaces a two-point blend as the *source* of detail, with exact consistency preserved. Everything downstream (HQ-2 guidance dial, HQ-4 latent, HQ-11 PiGDM) is a refinement of this substrate — it is the highest-leverage new capability.
2. **HQ-3 — CodeFormer-style code prediction.** Attacks the *root cause* of poor detail at 2–5%: nearest-neighbor lookup on corrupted features returns wrong codes. Global code prediction is the documented fix and raises `d_G` quality exactly where the current fusion starves — the biggest single-arm quality upgrade that keeps the frozen decoder.
3. **HQ-12 — Perception-weighted operator design.** The only bet that shrinks the *perceptual* null space itself rather than filling it better. If the measured rows carry LPIPS-dominant content, every downstream method starts from a higher floor; it is orthogonal to and stacks with all the null-space fillers.

## TOP 3 near-term wins (runnable now on existing VQAE/VQGAN/projector assets)
1. **HQ-7 — PnP-RED in the null space.** Zero training: wrap the existing `refine()` in a 10-iteration null-space fixed-point loop, sweep `η`, ship a convergence figure. Cheapest possible quality bump.
2. **HQ-8 — Consistency-gap-driven per-image `B`.** Closed-form `B` from `d_G − d_A` on already-cached residuals; ~9-point sweep on seed0 dev through the existing selector. No training, no new prior.
3. **HQ-9 — Three-prior cascade with a 2-D `(B1,B2)` dial.** An 11×11 closed-form grid on cached residuals; residual-to-intermediate `d_G'` cuts double-counting. Strictly generalizes the current scalar arm (`B1=1` recovers it), so it can only match-or-beat.

## Honest marginal-gain risks (likely small deltas vs the current fusion)
- **HQ-2 (B-as-guidance-scale)** — only lifts quality *if HQ-1's sampler already wins*; on its own it is a reparameterization of the dial, not new detail. Value is mostly the cleaner governance/P–D story, not a large LPIPS jump.
- **HQ-9 (cascade)** and **HQ-10 (endpoint interpolation)** — both still interpolate the *same two frozen residuals* (plus a re-referenced `d_G'`); the reachable set barely enlarges beyond scalar `B`. Expect modest gains unless a genuinely new endpoint (e.g. HQ-1's diffusion `d_perc`) is plugged in.
- **HQ-8 (consistgap B)** — a per-*image* scalar cannot fix *where within an image* the wrong prior is used; gains are bounded by how much image-to-image (not region-to-region) `B` variation actually helps. Good cheap win, low ceiling.
- **HQ-5 (β(x)) vs HQ-6 (MoDL)** — both are spatially-adaptive learned blends of the same residuals; a prior learned-gate attempt already underperformed. Real risk that a spatial gate re-learns ≈ the scalar and buys little; MoDL's iterative denoiser is the higher-ceiling of the two because it can synthesize content, not just reweight.
- **HQ-14** — a noise-robustness arm, not a clean-`y` quality lift; on the current noiseless testbed it will not beat the exact fusion and is only justified once a noisy/real-GI regime is on the table.

*14 innovation points. Invariant held throughout: all quality is bought inside `P0` (`A x_hat = y` to ~2e-15) except HQ-14, explicitly flagged as a separate soft-consistency noisy-`y` arm.*
