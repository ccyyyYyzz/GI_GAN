# Round 45 — independent optics-first invention after the first GAN-essential positive result

## Role and non-negotiable scope

Act as the independent theory inventor, not as a reviewer of the local agent's prose.  The project is specifically **GAN + computational ghost imaging (GI) for higher reconstruction quality**.  The work must remain simulation-only because no physical optical experiment can be performed.  The desired contribution is structurally simple, physics-first, theoretically deep, and not a compute-scaling paper.  The held-out test split remains unopened.

Repository: `ccyyyYyzz/GI_GAN`

Branch: `codex/gan-gi-journal-poc-20260718`

Relevant implementation:

- `src/fiber_residual_phase_gan.py`
- `train_fiber_residual_phase_gan.py`
- `diagnose_fiber_residual_frequency_fusion.py`
- `src/fiber_residual_spectral_fusion.py`
- `train_fiber_residual_spectral_fusion.py`
- `src/factorial_moment_dithered_residual.py`
- `diagnose_fused_residual_physical_readout.py`

## Current inverse problem and evidence

Images are (64\times64), so (n=4096).  The fixed structured GI operator has 205 displayed rows and numerical rank 200.  All final estimates are projected exactly onto the box-constrained measurement fiber

\[
\mathcal F_y=\{x\in[0,1]^n:Ax=y\}.
\]

The stable VQAE reconstruction is the base.  A VQGAN reconstruction supplies perceptual detail but can hallucinate.  A small residual-phase generator is initialized from a supplied prior difference and is explicitly projected into `ker(A)`.  The GAN arm uses a conditional high-pass PatchGAN; the matched non-GAN arm has identical generator capacity but no adversarial discriminator and uses an independent VQAE proposal.

On 512 untouched validation images, the original VQAE is:

- PSNR 23.215669 dB
- SSIM 0.658264
- LPIPS 0.300723

The strongest matched non-GAN structural solution is:

- PSNR 23.351547 dB
- SSIM 0.665604
- LPIPS 0.289040

The best balanced GAN residual solution is:

- PSNR 23.309730 dB
- SSIM 0.663847
- LPIPS 0.266499

Thus the GAN solution has much better perceptual distance, while the non-GAN solution has stronger distortion metrics.

## First GAN-essential joint dominance

Let (c_S\in\ker A) be the learned non-GAN structural correction and (c_G\in\ker A) the learned GAN correction.  The current fixed rule is

\[
c_{\rm fuse}=P_{\ker A}\left[c_S+\alpha H_{\omega_c}(c_G-c_S)\right],
\qquad
\hat x=\Pi_{\mathcal F_y}(x_{\rm VQAE}+c_{\rm fuse}),
\]

where (H_{\omega_c}) is a smooth radial high-pass filter.  No truth is used at inference.  Hyperparameters are selected only on development/validation data, before the test split is opened.

For the fixed validation-selected balanced point (\omega_c=0.12,\alpha=0.5), the exact projected result is:

- PSNR 23.372128 dB
- SSIM 0.667365
- LPIPS 0.287987

Relative to the strongest non-GAN solution, paired mean changes and 95% bootstrap CIs are:

- PSNR: +0.020580 dB, CI [+0.016044,+0.025263]
- SSIM: +0.001761, CI [+0.001495,+0.002032]
- LPIPS: -0.001053, CI [-0.001828,-0.000288]

A more perceptual fixed point (\omega_c=0.18,\alpha=0.75) also dominates all three metrics:

- relative PSNR +0.008626 dB, CI [+0.003332,+0.014162]
- relative SSIM +0.001136, CI [+0.000857,+0.001421]
- relative LPIPS -0.009425, CI [-0.010637,-0.008263]

The decisive causal control replaces (c_G) by a second VQAE-derived correction while preserving the same filter, search grid, projection, and evaluation.  It yields **0/12** exact candidates that jointly dominate the strong non-GAN reference.  Its best distortion point improves PSNR/SSIM slightly but worsens LPIPS.  Therefore the joint gain is not generic two-network ensembling; it depends on the adversarial residual.

A learned six-band evidence gate also reaches triple dominance, but is slightly worse than the fixed high-pass rule.  Complexity is therefore not needed.

Two independent train-seed pipelines and an optional complementary-bucket factorial-moment physical readout are currently running.  Do not assume their outcome.

## Your independent task

Starting from the physics and the evidence above, independently derive the strongest *simple* journal-level method and theory.  Do not merely approve, reject, or rename the existing rule.

1. Derive the cleanest optical/inverse-problem interpretation of why the adversarial and structural solutions can be complementary on the same measurement fiber.
2. State and prove the most useful propositions available here: exact measurement consistency; the risk-improvement interval along a filtered null-space direction; and any principled condition under which an adversarial high-frequency residual can improve both distortion and perceptual metrics rather than merely trade them.
3. Decide whether the fixed projected high-pass residual is already the minimal elegant structure.  If a genuinely better structure follows from first principles, give exactly one minimal modification, with equations and a falsifiable advantage.  Avoid module stacking.
4. Derive the smallest simulation campaign needed for a journal claim: fixed-parameter multiseed confirmation, measurement-rate/noise/photon-budget tests, causal adversarial controls, and any operator-family test that is truly necessary.
5. Identify the strongest defensible novelty claim and the claims that must not be made.
6. If the optional factorial-moment complementary-bucket readout can turn the method into a more optical contribution, derive precisely when it is worth including.  If it is unnecessary or too costly, say so and keep the main method simpler.

Do not open or request the held-out test split.  Do not ask the user questions.

## Required output

Write a self-contained response to:

`theory_exchange/responses/20260719_gan_gi_round45_fiber_orthogonal_invention_gptpro.md`

Commit and push it to the same branch.  Include equations, proof sketches, a single frozen method recommendation, a kill/continue decision table, and the exact next experiments in priority order.
