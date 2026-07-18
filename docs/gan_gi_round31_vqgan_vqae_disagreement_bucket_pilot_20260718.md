# Round 31 validation pilot: a physical bucket adjudicates VQGAN--VQAE disagreement

## Decision

`VALIDATION_PILOT_POSITIVE_NOVELTY_UNRESOLVED`

On 128 validation images, one object-adaptive balanced binary row derived only
from the disagreement between a measurement-conditioned VQAE reconstruction
and a matched VQGAN reconstruction improves PSNR, SSIM, and LPIPS over the VQAE
starting point.  Its pair-segment update also beats one fixed next-DCT row and
one fixed random balanced row on all three metrics.  This is the first
deployable all-metric-positive GAN-specific signal in the current campaign.

This result does **not** yet support a journal novelty claim.  The experiment is
noiseless, uses one seed, and adaptive query-by-committee is established prior
art.  A hostile mechanism-level literature audit, a non-adversarial
dual-autoencoder control, photon-matched Poisson/Skellam evaluation, and paired
uncertainty estimates are required before opening the held-out split.

## Mechanism tested

For the existing GI record `y = A x`, the two measurement-consistent estimates
are `x_A` (VQAE) and `x_G` (VQGAN).  The new complementary-DMD row is the
balanced binary maximizer of their null-space disagreement,

`q = argmax_{q_i in {+1,-1}/sqrt(n), sum(q)=0} q^T P_null(x_G-x_A)`.

The simulated bucket `b=q^T x` is acquired only after `q` is fixed.  Thus the
truth is not used to choose the row.  Two updates are tested:

1. the minimum-norm correction of `x_A` consistent with the old record and the
   new bucket; and
2. a bucket-determined coordinate on the segment joining `x_A` and `x_G`,
   followed by projection onto the image box and the augmented measurement
   fiber.

The second update asks a simple physical question: which amount of the
adversarially supplied alternative is supported by one real bucket value?

## Frozen validation result

The primary result uses 8192 Dykstra iterations and 128 cached validation
images.  No development/test split is opened.

| Method | PSNR (dB) | SSIM | LPIPS | Delta PSNR vs VQAE | Delta SSIM vs VQAE | Delta LPIPS vs VQAE |
|---|---:|---:|---:|---:|---:|---:|
| VQAE start | 23.306278 | 0.664635 | 0.299304 | 0 | 0 | 0 |
| VQGAN alternative | 21.725985 | 0.582677 | 0.173820 | -1.580294 | -0.081958 | -0.125484 |
| Fixed next-DCT, one row | 23.405333 | 0.673243 | 0.294769 | +0.099054 | +0.008608 | -0.004535 |
| Fixed random balanced, one row | 23.387430 | 0.672213 | 0.294503 | +0.081152 | +0.007578 | -0.004801 |
| Disagreement row, minimum update | 23.412252 | 0.672712 | 0.285569 | +0.105974 | +0.008077 | -0.013735 |
| **Disagreement row, pair segment** | **23.430561** | **0.673880** | **0.283278** | **+0.124283** | **+0.009245** | **-0.016026** |

Relative to the strongest fixed one-row control, the pair-segment method gains
at least +0.025228 dB PSNR, +0.000637 SSIM, and -0.011225 LPIPS while degrading
no reported metric.

## Mechanistic diagnostics

- Mean absolute cosine between the VQGAN--VQAE pair direction and the unresolved
  VQAE error: `0.088409`.
- Mean absolute cosine after balanced-binary compilation: `0.066683`.
- The true bucket lies between the two candidate buckets for `86.71875%` of
  images, explaining why a one-dimensional physical adjudication is often
  meaningful.
- Mean candidate bucket separation: `2.335427` in the normalized signed-row
  convention.
- Maximum old-fiber relative residual for the pair-segment output:
  `1.33e-6`; maximum new-bucket relative residual: `1.78e-5`.
- The final affine iterate has a maximum box violation `6.58e-4`; this is small
  but must be tightened or handled with a certified projection before final
  reporting.

## Reproducibility

- Code: `diagnose_vqgan_vqae_disagreement_bucket.py`
- Code SHA-256:
  `bfd569963a2c3ab142fa50f697b7eafbe85913ffd957f26aa9f41c3165b4fc52`
- Result:
  `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round31/vq_disagreement_bucket/pilot8192/summary_val_128.json`
- Result SHA-256:
  `ec341d10a68c13dffca360e0e12e02465f58fc4451f927380d29b8a0ad3c8c27`
- Original 205-row operator SHA-256:
  `8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628`
- Script declares and enforces CUDA; the run uses the local RTX 4060.
- Query selection does not access truth.  Truth is used only to simulate the
  additional bucket and to compute evaluation metrics.

## Next causal gates

1. Add paired per-image metrics and bootstrap confidence intervals.
2. Compare against disagreement between two non-adversarial autoencoders with
   matched architecture, data, compute, and seed separation.
3. Compare against a residual/error predictor and established posterior/QBC
   adaptive sensing under the same complementary exposures and photon budget.
4. Repeat under exact complementary Poisson counts and Skellam differencing.
5. Open the held-out split only if the GAN-specific control and photon-matched
   validation gates remain positive.
