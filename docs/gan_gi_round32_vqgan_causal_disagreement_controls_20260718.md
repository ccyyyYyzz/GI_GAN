# Round 32: causal controls for physical VQGAN-disagreement adjudication

## Decision

`GAN_CAUSAL_PERCEPTUAL_SIGNAL_POSITIVE_NOVELTY_UNRESOLVED`

The one-bucket disagreement experiment is repeated with matched alternative
models from three VQGAN seeds and two non-adversarial VQAE seeds.  All queries
start from the same seed-0 VQAE reconstruction, use the same original record,
consume one additional balanced signed row, and are evaluated on the same 128
validation images.  Cache identity is exact for source indices, truth, original
measurements, and LMMSE anchors.

Different VQAE seeds produce a stronger distortion-oriented query but do not
improve LPIPS.  Every VQGAN seed produces a statistically clear LPIPS gain while
retaining positive PSNR and SSIM gains.  The adversarial prior therefore has a
repeatable causal role in identifying perceptual detail, rather than merely
adding generic model diversity.

This is still not a novelty verdict.  The experiment is noiseless and the query
can be described as two-member model discrimination.  Exact photon accounting,
strong adaptive controls, and the pending hostile prior-art review remain
mandatory.

## Mean validation metrics

| Alternative defining the one-row query | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| No new row: VQAE seed 0 | 23.306278 | 0.664635 | 0.299304 |
| Fixed next-DCT row | 23.405334 | 0.673243 | 0.294768 |
| Fixed random balanced row | 23.387431 | 0.672213 | 0.294504 |
| VQAE seed 1 | 23.505895 | 0.677794 | 0.300475 |
| VQAE seed 2 | **23.512874** | **0.678319** | 0.300220 |
| VQGAN seed 0 | 23.430561 | 0.673880 | 0.283279 |
| VQGAN seed 1 | 23.424319 | 0.673657 | **0.282003** |
| VQGAN seed 2 | 23.425510 | 0.673762 | 0.283617 |

The VQAE-diversity rows improve PSNR by `+0.1996` to `+0.2066 dB` and
SSIM by `+0.01316` to `+0.01368`, but their mean LPIPS changes are
`+0.00117` and `+0.000916`.  By contrast, the three VQGAN rows improve
PSNR by `+0.1180` to `+0.1243 dB`, SSIM by `+0.00902` to `+0.00924`, and
LPIPS by `-0.01569` to `-0.01730`.

## Paired uncertainty

For VQGAN seed 0 versus the no-new-row VQAE start, 5000 paired bootstrap
replicates give:

| Metric | Mean delta | 95% bootstrap CI | Fraction of images improved |
|---|---:|---:|---:|
| PSNR | +0.124283 dB | [+0.100900, +0.149936] | 0.96094 |
| SSIM | +0.009245 | [+0.006424, +0.012514] | 0.85156 |
| LPIPS | -0.016025 | [-0.019014, -0.013271] | 0.93750 |

For VQGAN seed 0 versus the same-budget fixed next-DCT row:

| Metric | Mean delta | 95% bootstrap CI | Fraction of images improved |
|---|---:|---:|---:|
| PSNR | +0.025227 dB | [+0.013192, +0.038488] | 0.64844 |
| SSIM | +0.000637 | [+0.000076, +0.001252] | 0.57812 |
| LPIPS | -0.011490 | [-0.014341, -0.008838] | 0.85156 |

All three intervals exclude zero.  VQGAN seeds 1 and 2 also beat the DCT row
clearly in LPIPS and PSNR; their small SSIM differences have intervals crossing
zero.

The non-adversarial VQAE rows provide the complementary causal result.  Their
LPIPS differences versus the VQAE start have intervals crossing zero, while
their LPIPS differences versus the DCT row are significantly *worse*:

- VQAE seed 1 minus DCT: `+0.005706`, CI `[+0.003667,+0.007696]`;
- VQAE seed 2 minus DCT: `+0.005451`, CI `[+0.003136,+0.007610]`.

Thus ordinary reconstruction diversity does not explain the perceptual gain.

## Mechanistic interpretation

The bucket-estimated pair coordinate uses `-1` for the VQAE end and `+1` for
the VQGAN end.  Its mean is `-0.764`, `-0.773`, and `-0.806` for the three
VQGAN alternatives.  The physical record therefore accepts only roughly
`9.7%--11.8%` of the full adversarial path on average.  This explains how a
large VQGAN perceptual advantage can be used without inheriting its large
PSNR/SSIM loss.

The true bucket is bracketed by the VQAE/VQGAN candidate buckets for
`82.8%--89.8%` of images.  Only `10.2%--17.2%` of pair coordinates require
clipping.  This supports a one-dimensional physical-adjudication model, while
also exposing the residual cases that need a robust noisy estimator.

## Reproducibility

- Script: `diagnose_vqgan_causal_disagreement_controls.py`
- Script SHA-256:
  `b0ec78a69ecf560aa369dbddb97469f031c62bf7248e40deeee3d029cd40e1a3`
- Result:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round32/vq_disagreement_causal_controls/summary_val_128.json`
- Result SHA-256:
  `d02edd556eaff5e0ab4614a5b6ed4ca2d5652abf95069dcd74dcf45ad6cfd8f4`
- Original operator SHA-256:
  `8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628`
- GPU: local RTX 4060.
- Split: validation only; held-out test remains unopened.
- Truth is not used to select any query.  It is used only to simulate the new
  bucket and calculate metrics.

## Next hard gates

1. Replace the noiseless bucket with exact complementary Poisson counts,
   Skellam differencing, background, and fixed total photon/exposure budgets.
2. Compare against an established adaptive residual/posterior method, not only
   fixed DCT and random rows.
3. Test whether the GAN row remains perceptually superior when a learned
   non-adversarial residual predictor receives matched compute.
4. Tighten the box-plus-fiber projection certificate.
5. Require the effect across seeds and photon levels before any test opening.
