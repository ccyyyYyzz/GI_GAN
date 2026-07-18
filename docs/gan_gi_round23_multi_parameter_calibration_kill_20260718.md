# Multi-parameter unpaired optical calibration: causal GAN kill (2026-07-18)

## Question

Does the positive one-parameter defocus pilot remain GAN-specific when the
unknown illumination-path transfer is expanded to a compact but nontrivial
physical model?

The simulated transfer contains four parameters:

- horizontal and vertical standard deviations of a positive unit-mass
  anisotropic Gaussian point-spread function;
- horizontal and vertical coefficients of a positive, mean-normalized
  illumination field.

The true parameters are `(1.40, 0.70, 0.35, -0.25)`.  The adversarial and
non-adversarial arms start from the same nominal transfer
`(0.30, 0.30, 0, 0)`, use the same two unpaired 1024-image halves, and receive
2500 updates.  The non-adversarial control matches the per-bucket mean and
standard deviation.  Evaluation uses only the existing 128-image validation
split; the test split is not opened.

## Result

| Calibration | sigma-x | sigma-y | field-x | field-y | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nominal wrong operator | 0.05 | 0.05 | 0 | 0 | 17.20372 | 0.544952 | 0.376566 |
| Mean/variance matching | 1.39323 | 0.70240 | 0.35367 | -0.24588 | **22.31476** | **0.606801** | 0.337949 |
| Adversarial calibration | 1.49095 | 0.70051 | 0.37273 | -0.26761 | 22.24255 | 0.606110 | **0.337846** |
| Oracle transfer | 1.40 | 0.70 | 0.35 | -0.25 | 22.31635 | 0.606770 | 0.337946 |

Both calibration methods strongly improve all three metrics relative to the
wrong operator, confirming that the physical quantity is important.  However,
the moment control nearly identifies all four parameters and reaches the oracle
reconstruction.  Relative to that control, the GAN changes PSNR by
`-0.07221 dB`, SSIM by `-0.000692`, and LPIPS by only `-0.000103`.

## Decision

`KILL_SIMPLE_DETERMINISTIC_APOC_AS_JOURNAL_GAN`.

A deterministic low-dimensional optical transfer is a useful target-free
calibration problem, but this experiment shows no need for an adaptive
adversarial statistic: low-order bucket moments already construct the missing
object more accurately.  The one-parameter positive pilot must not be promoted
as evidence of a journal-level GAN contribution.

The only defensible continuation is to identify a physical distributional
object that is invisible to the relevant low-order moments but identifiable
from the full bucket-record law—for example a stochastic or multimodal
instrument state—and then compare the GAN with characteristic MMD/optimal
transport as well as moment controls.  Otherwise the physical-calibration GAN
direction is a formal no-go.

## Reproduction

- Script: `diagnose_unpaired_optical_calibration_multi.py`
- Summary:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round23/unpaired_optical_calibration_multi_seed0/summary.json`
- Runtime: 142.9 s on the local RTX 4060 Laptop GPU.

