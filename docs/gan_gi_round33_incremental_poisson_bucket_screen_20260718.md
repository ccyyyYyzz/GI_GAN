# Round 33: incremental complementary-Poisson screen of the frozen one-bucket method

## Decision

`INCREMENTAL_POISSON_POSITIVE_FULL_NOISE_PENDING`

The frozen one-bucket VQGAN-disagreement method remains all-metric positive when
the added bucket is generated from two independent complementary Poisson counts
with 1% background.  At both `1.00e4` and `1.00e5` expected signal photons per
complementary pair, it beats an operator-aware fixed balanced Hadamard row on
PSNR, SSIM, and LPIPS with image-clustered 95% bootstrap intervals excluding
zero.

The scientific identity of the Round-31 method is unchanged.  This experiment
changes only the observation model for the additional bucket.

This is an incremental noise screen, not a complete photon-limited GI result:
the original 205-row record and its VQAE/VQGAN reconstructions remain noiseless,
and the noisy new bucket is enforced as a hard affine constraint.  A final
experiment must noise all measurements and use the exact Poisson/Skellam
likelihood or a justified shrinkage rule.

## Physical protocol

- Each adaptive or fixed signed row contains exactly 2048 positive and 2048
  negative pixels and is implemented by a complementary DMD mask pair.
- The original DC measurement estimates object flux; no truth value is used to
  set exposure.  Its maximum relative flux error is `7.52e-7` in this cache.
- Exposure is chosen so that each complementary pair receives the requested
  expected signal photons.  Equal background contributes 1% additional
  expected counts and cancels in the mean difference while increasing variance.
- Eight independent count pairs are simulated for each of 128 validation
  images.  Metrics are averaged within image, then 5000 paired bootstrap samples
  are drawn over images.
- The fixed baseline is selected without image or truth access: among 512
  low-sequency balanced Hadamard rows, choose the row with the largest component
  outside the original row space.  Its null-space norm is `0.998562`.

## Mean metrics

### `1.00e4` signal photons per complementary pair

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| No new row | 23.306278 | 0.664635 | 0.299304 |
| Fixed Hadamard | 23.359572 | 0.669543 | 0.285782 |
| VQAE-disagreement row | 23.463953 | **0.675864** | 0.296323 |
| **VQGAN-disagreement row** | **23.388388** | 0.671524 | **0.275882** |

VQGAN disagreement minus fixed Hadamard:

| Metric | Mean delta | 95% clustered-bootstrap CI |
|---|---:|---:|
| PSNR | +0.028816 dB | [+0.018721, +0.039564] |
| SSIM | +0.001980 | [+0.001388, +0.002579] |
| LPIPS | -0.009900 | [-0.012653, -0.007408] |

### `1.00e5` signal photons per complementary pair

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| No new row | 23.306278 | 0.664635 | 0.299304 |
| Fixed Hadamard | 23.380303 | 0.671615 | 0.292457 |
| VQAE-disagreement row | 23.502949 | **0.677868** | 0.299985 |
| **VQGAN-disagreement row** | **23.421007** | 0.673367 | **0.281940** |

VQGAN disagreement minus fixed Hadamard:

| Metric | Mean delta | 95% clustered-bootstrap CI |
|---|---:|---:|
| PSNR | +0.040704 dB | [+0.029783, +0.053238] |
| SSIM | +0.001751 | [+0.001285, +0.002278] |
| LPIPS | -0.010517 | [-0.013437, -0.007840] |

At `1.00e5` photons, the VQGAN row also improves the no-new-row VQAE start by
`+0.114728 dB`, `+0.008731` SSIM, and `-0.017364` LPIPS; all three intervals
exclude zero.  At `1.00e4` photons the corresponding gains are `+0.082110 dB`,
`+0.006889`, and `-0.023422`.

## Causal interpretation under photon noise

The VQAE-disagreement row retains its distortion advantage but is perceptually
inferior to the fixed Hadamard row at both photon levels.  The VQGAN row retains
the opposite, complementary signature: smaller distortion gains but a clear
perceptual gain.  The adversarial/non-adversarial separation therefore survives
Poisson count noise and is not an artifact of an exact floating-point bucket.

The current equal-budget margin over the fixed physical row is material for
LPIPS but still small for PSNR and SSIM.  It does not yet pass the stricter
journal gate against the strongest *adaptive* non-GAN control.  Training or
noise-aware estimation should advance only if the pending theory review finds a
novel mechanism-level residual.

## Reproducibility

- Script: `diagnose_vqgan_disagreement_poisson_bucket.py`
- Script SHA-256:
  `0fb1c555b4f9efad3c9cf9cdd3cbc849d4482869ca659cb33c6e16e1392351d9`
- Result:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round33/vq_disagreement_poisson_bucket/summary_val_128.json`
- Result SHA-256:
  `623856704048ae2de7cba5002dc103665c112ae00bdefc73f149c9201cf19c61`
- Original operator SHA-256:
  `8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628`
- Split: validation only; the held-out test remains unopened.

## Remaining physical gates

1. Apply Poisson noise and background to all 205 original complementary pairs,
   not only the added pair.
2. Replace hard equality to a noisy bucket by the exact joint-Poisson likelihood
   or a derived low-count shrinkage estimator.
3. Match total photons when reallocating one pair from, rather than appending it
   to, a fixed 206-pair acquisition.
4. Compare with the strongest adaptive non-adversarial residual/posterior query.
