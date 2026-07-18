# Complementary DCT bucket GAN pilot (2026-07-18)

## Question

Can the adversarial component become more GI-specific by judging only the
unmeasured, physically interpretable DCT bucket spectrum rather than the image
itself?  The measured 5% record remains unchanged and every output is returned
to the same box--measurement fiber.

## Protocol

- Validation only: 128 hash-disjoint STL-10 images; the frozen test split was
  not opened.
- Fixed operator: 205 rows at 64 x 64, SHA-256
  `8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628`.
- Reused Stage-A content checkpoint; 750 Stage-B updates.
- Exact causal control: 750 supervised-only updates with the same initialization,
  optimizer, learning rate, batch order, and update count.  Both batch-order
  hashes are
  `f5872c85f80d9d4ef8cd756bfbc4d339c1fcc48c5a869be09c7df4264131d769`.
- The critic receives the measurement-conditioned anchor and the complement of
  the first 128 non-DC DCT buckets.  Acquired DCT buckets are masked from the
  real/fake evidence.

## Result

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| Stage-A content | 22.953881 | 0.674117 | 0.187955 |
| 750-step supervised continuation | 22.869033 | 0.676136 | 0.164820 |
| 750-step complementary-bucket GAN | 22.855598 | 0.675816 | 0.163621 |

Paired GAN-minus-matched effects:

| Metric | Mean | 95% bootstrap CI |
|---|---:|---:|
| PSNR (dB) | -0.0134355 | [-0.0150603, -0.0118317] |
| SSIM | -0.000319964 | [-0.000377417, -0.000262151] |
| LPIPS | -0.00119858 | [-0.00137125, -0.00103349] |

The adversarial game is active rather than collapsed: its tail-median hinge
loss is 1.15625 and its discriminator gap is 1.17676.  The run takes 344 s on
an RTX 4060 Laptop GPU, including cache verification, matched control, metrics,
bootstrap, and galleries.

## Decision

`NO_GO_AS_JOURNAL_PRIMARY`.

Moving the critic from image space to the physically meaningful unmeasured
bucket spectrum roughly doubles the causal LPIPS effect seen with the original
image critic, but it does not break the perception--distortion trade-off.  It
slightly worsens both PSNR and SSIM relative to the equal-step no-adversarial
control.  A publishable successor therefore needs new sample-specific
information or a different estimand; another choice of transform, critic, or
loss weighting is not enough.

Machine-readable evidence is under
`E:/GAN_FCC_WORK/experiments/gan_gi_journal_round22/complementary_dct_seed0/`.
