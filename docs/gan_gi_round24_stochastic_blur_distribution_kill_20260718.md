# Round 24: stochastic blur-distribution adversary — causal KILL

Date: 2026-07-18

## Question

Can an adversarially fitted two-state optical blur distribution recover a physically
missing stochastic forward object that low-order bucket moments cannot construct,
and does that recovery improve GI reconstruction?

## Protocol

- STL-10, 64 x 64, validation split only; the test split remains unopened.
- Fixed 205-row structured single-pixel operator.
- Per-record projector blur is sampled with equal probability from
  `(sigma_low, sigma_high) = (0.45, 1.55)`.
- The calibration records and clean reference images are source-disjoint and
  unpaired.
- The adversarial estimator and a non-adversarial mean-plus-standard-deviation
  matching control use the same two-parameter ordered blur generator.
- Reconstruction uses two state-specific ridge models and a bucket-domain
  diagonal-Gaussian state classifier. An oracle with true state labels quantifies
  the available reconstruction headroom.
- The adversarial fit uses 2500 generator updates, three discriminator updates per
  generator update, and batch size 64.

## Result

| Method | Estimated sigmas | State accuracy | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| nominal single state | `(0.05, 0.05)` | 0.507812 | 21.332088 | 0.576926 | 0.380527 |
| moment mixture | `(0.446891, 1.537893)` | 0.687500 | 21.256054 | 0.575693 | 0.361306 |
| adversarial mixture | `(0.492457, 1.780631)` | 0.679688 | 21.034580 | 0.566391 | 0.361813 |
| oracle parameters, blind state | `(0.45, 1.55)` | 0.679688 | 21.238745 | 0.574950 | 0.361824 |
| oracle parameters, true state | `(0.45, 1.55)` | 1.000000 | 22.213652 | 0.601418 | 0.345896 |

Against the equal-parameterization moment control, the adversarial fit changes
PSNR by -0.221474 dB, SSIM by -0.009302, and LPIPS by +0.000508. It therefore
loses all three reconstruction metrics. The moment control nearly identifies the
true blur modes, so the adversarial objective is not necessary for this physical
family.

The oracle comparison isolates a second failure: even with the exact blur modes,
blind per-record state inference reaches only 0.679688 accuracy. Supplying the
true state raises PSNR by 0.974907 dB relative to the blind oracle and by 0.881563
dB relative to the nominal reconstruction. Thus the useful physical headroom is
real, but it is locked behind state observability rather than distribution fitting.

## Decision

`KILL_TWO_STATE_BLUR_DISTRIBUTION_GAN`

Do not tune the adversarial loss, critic, or training length. A successor must make
the nuisance state observable through a compact acquisition mechanism and must
beat a characteristic-distribution non-adversarial control, not merely low-order
moments. The GAN must supply a reconstruction-relevant object that the matched
non-adversarial estimator cannot construct.

## Artifacts

- Code: `diagnose_stochastic_blur_distribution_gan.py`
- Raw summary:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round24/stochastic_blur_distribution/summary.json`

