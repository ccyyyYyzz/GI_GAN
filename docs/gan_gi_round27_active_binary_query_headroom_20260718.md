# Round 27: balanced binary ambiguity-query headroom

Date: 2026-07-18

## Question

After the fixed 205-row GI record, can a very small number of additional
physically realizable complementary DMD queries materially improve the current
reconstruction if their directions are chosen well?  This is a mechanism
headroom test, not a reconstruction-method result.

## Physical query

A signed row is implemented by two complementary half-on/half-off DMD masks.
For a hypothesized ambiguity direction `d`, the balanced binary row places `+1`
on the largest half of the entries of `d` and `-1` on the smallest half.  By the
rearrangement inequality, this is the exact maximizer of `q^T d` among equal-flux
binary complementary rows.  Under additive Gaussian bucket noise it therefore
maximizes the pairwise measurement KL divergence for the two hypotheses whose
difference is `d`.

Each acquired bucket is assimilated while retaining the original 205-row
record.  The validation split is used; the test split remains unopened.

## Result

The equal-step zero-adversarial reconstruction is the primary starting point:

`22.869164 dB / 0.676138 / 0.164824` for PSNR / SSIM / LPIPS.

### Truth-directed headroom only

The following balanced binary row uses the sign ordering of the *true current
error*.  It is intentionally unavailable to an implementable method and is an
upper-bound diagnostic.

| Added signed rows | Delta PSNR (dB) | Delta SSIM | Delta LPIPS |
|---:|---:|---:|---:|
| 1 | +3.074478 | +0.138788 | +0.010645 |
| 2 | +6.588287 | +0.244330 | -0.097219 |
| 4 | +11.368870 | +0.300215 | -0.144229 |
| 8 | +16.122559 | +0.317981 | -0.158213 |

The first row improves distortion strongly but worsens LPIPS; two or more rows
improve all three metrics.  Equal-flux balancing costs almost nothing relative
to an unconstrained sign row.

### Fixed schedules

| Added signed rows | Schedule | Delta PSNR (dB) | Delta SSIM | Delta LPIPS |
|---:|---|---:|---:|---:|
| 8 | next low-frequency DCT rows | +0.083878 | +0.003531 | -0.002365 |
| 8 | fixed random balanced binary rows | +0.010284 | -0.000090 | +0.000461 |

The large gap between the truth-directed binary ceiling and fixed schedules
shows that query direction, rather than the mere presence of additional
measurements, can dominate the benefit at this operating point.

### First truth-free bridge using the existing GAN

The 0.025-strength GAN and its equal-step supervised control are two
same-record reconstructions.  Their balanced sign difference was used as one
new DMD query without accessing truth.  Starting from the supervised control,
the actual new bucket produced:

`+0.017973 dB / -0.000466 SSIM / +0.001239 LPIPS`.

This is a negative result: the existing GAN-supervised disagreement is not a
useful perceptual ambiguity direction and is inferior to even the next fixed DCT
row on SSIM and LPIPS.  The oracle ceiling therefore cannot be attributed to the
current GAN.

## Decision

`PHYSICAL_HEADROOM_LARGE_CURRENT_GAN_PAIR_NO_GO`

The only justified continuation is a focused novelty and feasibility attack on
an adversarial same-fiber *extreme-pair* generator: both hypotheses must remain
plausible, their balanced binary separating query is closed-form optimal, and
the true object must frequently lie along the exposed ambiguity.  Do not train
that model unless the literature subtraction leaves a genuine residual and an
equal-compute non-adversarial extreme-pair control is specified.

The current Dykstra implementation was run for 512 iterations.  Original-fiber
relative residuals are below `6.65e-7`, new-bucket relative residuals below
`4.79e-5`, and the remaining box violation is at most about `1.18e-3`.  These
numbers are adequate for the directional headroom decision but not for a final
claim; a certified augmented-fiber projection is required before any formal
experiment.

## Artifacts

- Script: `diagnose_active_binary_query_headroom.py`
- Raw summary:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round27/active_binary_query_headroom/summary.json`

