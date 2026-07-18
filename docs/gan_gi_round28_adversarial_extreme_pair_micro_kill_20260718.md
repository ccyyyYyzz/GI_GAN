# Round 28: direct adversarial extreme-pair query micro-test

Date: 2026-07-18

## Question

Can the frozen PQBF discriminator regularize two maximally separated images in
the current GI fiber strongly enough that their closed-form balanced binary
separator becomes an informative next measurement?

## Protocol

- First 32 validation images; test unopened.
- Start from the equal-step zero-adversarial reconstruction.
- Optimize a symmetric null-space perturbation for 40 Adam updates.
- `no_critic`: maximize pair separation with box penalty only.
- `adversarial_plausibility`: same optimization and initialization class, with a
  hinge requiring both discriminator scores to remain no worse than the starting
  reconstruction minus 0.05.
- Convert the optimized pair difference to the equal-flux half-on/half-off DMD
  row, acquire its true simulated bucket, and assimilate it into the original
  205-row record.

## Result

Baseline: `22.987684 dB / 0.691360 / 0.156892` for PSNR / SSIM / LPIPS on this
32-image prefix.

| Pair construction | Delta PSNR (dB) | Delta SSIM | Delta LPIPS | query/error absolute cosine |
|---|---:|---:|---:|---:|
| no critic | +0.001257 | -0.0000158 | +0.000168 | 0.01261 |
| adversarial plausibility | +0.001909 | -0.0000372 | +0.000566 | 0.01509 |

Both arms are effectively orthogonal to the actual reconstruction error and
both worsen perceptual quality.  The discriminator does not supply a useful
ambiguity direction.

More seriously, the starting reconstruction has mean critic score `-1.666`,
whereas the optimized extremes reach about `+1.70`.  The pair therefore exploits
the frozen critic rather than remaining on a reliable object manifold.  A hinge
lower bound on critic plausibility cannot prevent this failure.

Assimilation residuals are adequate for this directional kill: old-fiber
relative residual below `2.48e-7`, new-bucket relative residual below
`3.48e-6`, and remaining box violation below `1.99e-5`.

## Decision

`DIRECT_PIXEL_EXTREME_PAIR_NO_GO`

Do not scale or tune direct discriminator-constrained pixel optimization.  Any
remaining version must generate pairs through a calibrated conditional latent
model and must beat the same latent architecture with adversarial weight zero.
That larger investment is justified only if the Round-27 literature/theory
adjudication leaves a genuine novelty residual.

## Artifacts

- Script: `diagnose_adversarial_extreme_pair_query.py`
- Raw summary:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round28/adversarial_extreme_pair_micro/summary.json`

