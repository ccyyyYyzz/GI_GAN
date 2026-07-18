# GAN–GI journal search: baseline frequency diagnostic (2026-07-18)

## Purpose

This validation-only diagnostic separates the effect of the frozen PQBF-GAN
checkpoint from an equal-step, equal-optimizer supervised continuation.  It is
used to reject incremental proposals that merely redirect the existing image
discriminator toward spatial-frequency bands.

## Inputs

- Validation images: 128 hash-disjoint STL-10 samples from the existing PQBF
  cache; the frozen test split is not opened.
- Content checkpoint: Stage A, step 2000.
- GAN checkpoint: validation-selected Stage B, step 750.
- Control checkpoint: 750-step matched zero-adversarial continuation.
- Operator SHA-256:
  `8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628`.
- Every prediction is passed through the exact box–fiber projection before the
  diagnostic.

## Result

| Method | Low-frequency MSE | Mid-frequency MSE | High-frequency MSE | All-frequency MSE |
|---|---:|---:|---:|---:|
| LMMSE anchor | 0.0356083 | 0.0116070 | 0.000853027 | 0.00655992 |
| Stage-A content | 0.0270327 | 0.0110736 | 0.000848679 | 0.00585100 |
| Matched continuation | 0.0268492 | 0.0115017 | 0.000858623 | 0.00598744 |
| PQBF-GAN | 0.0268643 | 0.0115222 | 0.000859457 | 0.00599567 |

Relative to the matched continuation, PQBF-GAN increases Fourier-domain MSE by
0.0563% in the low band, 0.1783% in the middle band, 0.0971% in the high band,
and 0.1375% overall.  Only 16.41% of validation images have lower pixel MSE under
GAN than under the matched continuation.  The mean cosine alignment between the
GAN-specific update and the matched model's remaining ground-truth error is
`-0.06609`.

## Decision consequence

The statistically resolved LPIPS benefit of PQBF-GAN is not a hidden
frequency-wise distortion improvement.  The current adversarial update moves
away from the ground truth in pixel/Fourier MSE on most samples while improving
perceptual features.  Therefore a generic frequency discriminator, a larger
PatchGAN, or loss reweighting is not a journal-level route.  A new method must
change the physical/statistical object estimated by the adversarial game or
introduce genuinely new sample-specific information; it cannot be justified as
recovering a frequency band that the present discriminator simply overlooked.

## Reproduction

The diagnostic is implemented by `diagnose_pqbf_frequency_tradeoff.py`.  The
machine-readable output is stored outside Git under
`E:/GAN_FCC_WORK/experiments/gan_gi_journal_round22/frequency_diag_seed0/`.

