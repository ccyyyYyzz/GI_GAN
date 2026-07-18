# Unpaired optical-transfer calibration GAN: micro pilot (2026-07-18)

## Physical question

Computational ghost imaging assumes that the masks stored by the computer are
the masks that reach the object.  Projector defocus violates this assumption:
the bucket record is produced by blurred masks, while reconstruction still uses
the nominal operator.  The unavailable classical object is therefore the
optical transfer between the modulator and object plane.

This pilot tests whether that transfer can be inferred without paired
calibration targets.  One half of the training images supplies simulated bucket
records from the defocused instrument.  The other half is an unpaired clean
image bank.  A discriminator compares real bucket-record vectors with records
obtained by passing the unpaired images through a differentiable candidate
optical transfer.  The generator contains only the physical defocus parameter;
it does not generate or post-process images.  After calibration, a fixed ridge
GI reconstructor is rebuilt with the estimated operator.

## Protocol

- Simulated projector defocus: Gaussian standard deviation `1.30` pixels.
- Calibration data: `1024` bucket records and `1024` unpaired reference images
  from disjoint halves of the existing STL-10 training cache.
- Evaluation: the existing `128`-image validation cache only; the frozen test
  split was not opened.
- Operator: the existing 205-row structured operator at 64 x 64, SHA-256
  `8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628`.
- Global flux is divided out by the DC bucket before adversarial calibration.
- Adversarial calibration: `1500` generator updates, three discriminator
  updates per generator update.
- Non-adversarial control: identical initialization and `1500` updates using
  per-bucket mean and standard-deviation matching.
- Runtime: about 52 s on the local RTX 4060 Laptop GPU, including both controls,
  four reconstructions, and LPIPS evaluation.

## Result

| Calibration used by reconstruction | Estimated defocus | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Nominal, uncalibrated | 0.05 | 20.79124 | 0.568721 | 0.399040 |
| Mean/variance matching control | 1.17181 | 22.21220 | 0.606489 | 0.356338 |
| **Adversarial unpaired calibration** | **1.33805** | **22.24696** | **0.606185** | **0.348482** |
| Oracle physical calibration | 1.30 | 22.25310 | 0.606867 | 0.350097 |

Against the uncalibrated instrument, adversarial calibration changes PSNR by
`+1.45571 dB`, SSIM by `+0.037464`, and LPIPS by `-0.050559`.  It recovers the
true defocus to `0.03805` pixel and nearly reaches the oracle reconstruction.

Against the stronger moment-matching control, the adversarial game gives
`+0.03475 dB` PSNR and `-0.007856` LPIPS but changes SSIM by `-0.000304`.
Therefore the physical-calibration direction is strongly positive, whereas the
claim that adversarial distribution matching is necessary is not yet complete.

An independent adversarial initialization repeats the result: it estimates
defocus `1.34403` and reaches PSNR `22.24515 dB`, SSIM `0.606042`, and LPIPS
`0.348235`.  The two adversarial estimates differ by only `0.00598` pixel.

## Decision

`DIRECTIONAL_GO_FOR_THEORY_ATTACK_AND_CAUSAL_EXPANSION`.

This result clears the main first-principles objection to the prior GANs: the
GAN now estimates missing instrument information rather than selecting a more
realistic point on one fixed measurement fiber.  It consequently improves all
three image-quality metrics relative to the physically mismatched baseline.

Before it can become the journal method, the next experiment must replace the
one-parameter Gaussian transfer by a constrained multi-parameter optical
transfer for which low-order moment matching is insufficient, and it must show
a stable adversarial advantage under equal-compute controls, multiple
distortions, operators, and datasets.  A closest-work audit must also separate
the method from supervised projector-defocus compensation, target-based
illumination calibration, AmbientGAN, and generic unpaired forward-model
estimation.

## Reproduction

- Script: `diagnose_unpaired_optical_calibration.py`
- Machine-readable summary:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round23/unpaired_optical_calibration_1500/summary.json`
- Independent-seed summary:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round23/unpaired_optical_calibration_1500_seed1/summary.json`
