# Round 25: projected discriminator-score refinement — causal KILL

Date: 2026-07-18

## Question

Can the trained conditional GAN discriminator supply the missing natural-image
prior force directly at inference, if its input gradient is restricted to the GI
null space and every update is projected exactly back to the bounded measurement
fiber?

## Protocol

- Validation split only; the test split remains unopened.
- Start from the equal-step, zero-adversarial 750-step continuation.
- Load the discriminator from the matched 750-step conditional-GAN checkpoint.
- Take one normalized ascent step along the null-space projection of the
  discriminator input gradient.
- Sweep pixel-RMS step sizes
  `1.00e-4, 2.50e-4, 5.00e-4, 1.00e-3, 2.00e-3`.
- Apply the exact bounded-fiber projector before scoring. Maximum observed relative
  intrinsic-record error is at numerical precision (approximately `1.73e-11` or
  lower across reported batches).

## Result

| Step size | Delta PSNR (dB) | Delta SSIM | Delta LPIPS |
|---:|---:|---:|---:|
| 1.00e-4 | +0.000099 | +0.000003 | +0.000042 |
| 2.50e-4 | +0.000212 | +0.000003 | +0.000109 |
| 5.00e-4 | +0.000299 | -0.000008 | +0.000224 |
| 1.00e-3 | +0.000105 | -0.000076 | +0.000450 |
| 2.00e-3 | -0.001759 | -0.000385 | +0.000933 |

The discriminator score increases in the intended direction for every dose, yet
LPIPS worsens monotonically. The smallest PSNR/SSIM changes are numerical-scale and
do not compensate for the perceptual loss. No dose improves all three metrics.

## Decision

`KILL_FIBER_CRITIC_SCORE_REFINEMENT`

The discriminator learned a separator that is internally traversable but whose
gradient is not an image-error descent direction. Do not tune the step schedule or
regularize this critic: adversarially learned variational regularizers for inverse
problems are established prior art (Lunz, Oektem, and Schoenlieb, NeurIPS 2018),
and the present GI-specific causal pilot is negative.

## Artifacts

- Code: `diagnose_fiber_critic_score_refinement.py`
- Raw summary:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round25/fiber_critic_score_refinement/summary.json`

