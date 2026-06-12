# PROMPT FOR CODEX — Sampling-mode (GAN) track: forensics, infrastructure, G2 preflight

You are working inside the research repository of a low-sampling computational ghost imaging
paper. The paper's main line (measurement-certified reconstruction) is handled elsewhere and
is **not your concern**. Your mission is the parallel exploratory track: make the
**sampling mode of the prior** (a GAN operating strictly in the unmeasured/null subspace)
either (a) demonstrably ready to run as one controlled fine-tune, or (b) cleanly documented
as a post-mortem. You prepare everything **up to but not including** launching full training.

---

## 1. Context you must internalize

Forward model `y = A x + eps`, `A ∈ R^{m×n}`, 64×64 images (`n = 4096`), 5% sampling
(`m ≈ 205`), ensembles Rademacher ("Rad") and scrambled Hadamard ("Scr", `A A^T = I`).
Main deterministic ("mean-mode") model, published numbers on Scr-5:
**PSNR 22.2708 / SSIM 0.6317 / RelMeasErr 0.005453.**

Operators (λ from config, expected ≈ 0.02 — verify):

```
B_λ = A^T (A A^T + λ I)^(-1) ;  Π_y^λ(v) = v − B_λ(A v − y)        # audit
A† = A^T (A A^T)^(-1) ;  P_R = A† A ;  P_0 = I − P_R                # exact projectors
RelMeasErr(x̂; y) = ||A x̂ − y|| / ||y||
```

Theory the sampling mode rests on (verify the identities numerically before any data work):

```
(G-I1)  Π_y^0(v) = P_0 v + A† y   ⇒   P_0 ∘ Π_y^0 = P_0  and  A ∘ Π_y^0 ≡ y
        # adversarial gradients in the null gauge CANNOT change the certificate
(G-I2)  exact posterior sampler:  E||x̂ − x||² = 2 · MMSE  (the "3 dB law")
        # define sampling completeness  κ = nullMSE_sampling / nullMSE_mean ∈ [1, 2]
        # κ ≈ 1: mean-collapsed;  κ ≈ 2: full posterior sampler
        # consistency check: predicted ΔPSNR ≈ −10·log10(κ) in the null-dominated regime
```

**The G1 anomaly you are investigating.** An old Scr-5 GAN pilot was evaluated
(zero-training) with: PSNR 23.3448 / SSIM 0.6662 / RelMeasErr 0.005365, mean pixel std
0.0007775, null-variance ratio 0.012526, per-sample outputs not saved, LPIPS/FID/KID not
computed. This gives κ = 10^(−1.074/10) ≈ **0.78 < 1, outside the admissible range [1, 2]**
— formally impossible for a valid sampling mode. The two candidate explanations:
(a) training-budget confound (pilot is simply a longer/differently-trained deterministic
model), or (b) protocol drift / split contamination (the pilot dates from the same
infrastructure era in which a critic's AUC 0.992 later collapsed to 0.61 under strict
image-ID group split — phases named `Phase53C/D`, `Phase56`; a planned run `Phase60` was
skipped as `unsafe_to_run` citing: no main split hash, no per-sample outputs, insufficient
G1 provenance).

## 2. Ground rules (non-negotiable)

1. **You may NOT launch full G2 training.** You prepare, smoke-test, and stop. The launch
   decision is human.
2. Never modify main-pipeline training/eval code in place; add new files/utilities only.
   Never overwrite results. All outputs to `results/sampling_mode_<UTCdate>/`.
3. **Flag, don't fake.** Missing checkpoints/packages/configs are findings to report, never
   gaps to improvise over. If LPIPS/FID packages or weights are unavailable offline, say so
   — do not substitute an undeclared proxy.
4. `RUNLOG.md` for every command; fixed seeds; record exact configs.
5. Every conclusion in your reports must be traceable to a file or a command output.

## 3. Tasks

### S-1. Identity gate (≤30 min)
Verify (G-I1) to ≤1e-12 relative error on a synthetic A, and (G-I2) on a Gaussian toy with
closed-form posterior (expect sampler MSE / MMSE = 2.00 ± 0.01 over ≥2e4 trials). Save
`checks/gan_identity_gate.py`. Do not proceed until both pass.

### S0. G1 forensic post-mortem  **[decides how the supplement is written]**
Locate the pilot's checkpoints, training config, optimizer state, epoch count, loss weights,
dataset split indices, and eval script. Answer, with evidence:
1. Training budget vs the published mean-mode model (epochs/steps, lr schedule, losses).
2. Whether the pilot's eval used the same test indices as the main no-leak split
   (compare index hashes if reconstructible; this is the leakage question).
3. Whether any stochastic z was actually active at eval (mean std 7.8e-4 suggests z-collapse
   or z disabled — check the forward code path).
Classify the κ = 0.78 anomaly as: budget confound / protocol drift / leakage / mixed /
undeterminable, with the evidence chain. Output `G1_POSTMORTEM.md`. Also recompute, if the
checkpoint loads: the **certificate-invariance check** (RelMeasErr of pilot vs mean-mode,
expected ≈ equal: 0.005365 vs 0.005453) — this is the one G1 number that remains reportable.

### S1. Infrastructure repair  **[unblocks everything; shared with the main repo]**
1. **Per-sample saving:** extend the eval harness (new module, e.g. `eval_sampling.py`) to
   save K=32 stochastic outputs per test image (z-indexed, HDF5/NPZ), plus the z seeds.
2. **Split-hash utility:** `tools/split_hash.py` that loads any experiment's split indices
   and emits SHA256 of the sorted arrays; run it on the main no-leak split if locatable and
   on the pilot's split; write both into `PROVENANCE_SAMPLING.json`.
3. **Sampling metrics module** computing, from saved samples: per-image PSNR/SSIM/RelMeasErr
   of (i) each sample, (ii) the sample mean; pixelwise std maps; null-variance ratio using
   **exact P_0** (Q = orth(A^T)); κ with the mean-mode model as MMSE proxy (state the proxy
   assumption in the docstring); the Prop-12 invariance assertion (RelMeasErr identical
   pre/post adversarial weights, tolerance 1e-6 relative when the audit is on).
4. **Perceptual metrics:** detect availability of LPIPS / clean-FID / KID with local
   weights. If available, wire them in; if not, document unavailability and the exact
   install/weight requirement for the human.

### S2. G2 preflight: config + smoke test (NO full run)
Build the complete G2 configuration, then smoke-test mechanics only.
**Recipe (fixed; do not redesign):**
- Initialize G from the published Scr-5 mean-mode checkpoint (kills the budget confound by
  construction). Scr-5 only.
- Discriminator: small conv net on the **null gauge**: input channels = [P_0 x̂, x_data]
  (exact P_0), projection-style conditioning on x_data; hinge loss.
- Generator loss: L_data (existing reconstruction loss, on the sample mean over P=2–4 z's)
  + β · adversarial term (−D(P_0 x̂, x_data)) + rcGAN-style diversity regularizer
  (std reward; fallback if rcGAN unavailable: maximize E||x̂(z1) − x̂(z2)||_1 with a small
  weight, clearly labeled as the fallback).
- Audit Π_y^λ stays ON in the deliverable path; β sweep plan {0.3·β0, β0, 3·β0}.
**Smoke test:** ≤200 iterations on ≤256 training images. Pass criteria: losses finite and
moving; D not saturated (real/fake margins reported); per-sample saving works; pixel std
strictly increasing from init over the smoke run (anti-collapse signal); **certificate
invariant** (Prop-12 assertion holds at init and after 200 iters); κ computable end-to-end.
Output: `G2_CONFIG.yaml`, smoke logs, `figs/smoke_std_curve.pdf`.

### S3. Launch dossier (the stopping point)
Write `G2_READY.md` containing: exact launch command; estimated wall-clock and GPU memory
(measured from the smoke run, extrapolated); the **pre-registered acceptance band**
(κ ≥ 1.15; observed ΔPSNR consistent with −10·log10 κ within ±0.3 dB; RelMeasErr column
unchanged to 1e-6 relative; visible diversity in an 8×4 sample grid) and the
**pre-registered no-go reading** (κ ≈ 1 despite the diversity term ⇒ report "adversarial
fine-tuning reduces to the mean mode at this information budget" — an acceptable,
publishable outcome); plus the list of artifacts the run will emit. End with a one-line
status: `READY TO LAUNCH: yes/no — blockers: [...]`.

## 4. Deliverables manifest

```
results/sampling_mode_<date>/
  REPORT.md                 # executive summary; per-task status; flagged items
  G1_POSTMORTEM.md          # S0, with the anomaly classification + evidence chain
  PROVENANCE_SAMPLING.json  # split hashes (pilot + main if locatable)
  G2_CONFIG.yaml  G2_READY.md
  checks/gan_identity_gate.py
  tools/split_hash.py  eval_sampling.py  (+ metrics module)
  figs/smoke_std_curve.pdf
  RUNLOG.md
```

REPORT.md must end with **"What I could not determine and why"** and a single-table summary:
{G1 anomaly cause, certificate-invariance result, infra status, smoke-test verdict,
READY-TO-LAUNCH flag}. Assume a strict reviewer will read it; assume the human will decide
about launching G2 based solely on your dossier.
