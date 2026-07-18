# Round 36: AFRB proposal-support headroom

## Decision

**KILL AFRB before ratio-critic training.** The VQGAN proposal family contains a
real three-metric correction relative to the box-valid LMMSE anchor, but its
truth-assisted upper bound does not reach the matched deterministic VQAE
control. A learned density-ratio critic cannot repair missing proposal support.

The locked test split remains unopened.

## Question and fixed screen

The independent AFRB invention proposes eight stochastic VQGAN token decodes,
null-projects a `Beta(1,7)` fraction of each decode-anchor residual, applies the
same `1/255` null dither and radius-72 bound used by the theory, and learns
same-record density-ratio weights. Before training that critic, this screen asks
whether the fixed eight-particle convex hull has enough headroom.

The screen uses 128 validation images and reports three truth-only ceilings:

- the best particle by null-space MSE;
- the globally MSE-optimal simplex barycenter of the eight particles;
- uniform weights as the zero-adversarial-loss control.

Truth is used only for these diagnostic ceilings and metrics. No deployed
weight, proposal seed, temperature, or hyperparameter sees truth.

## Result

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| Box-valid LMMSE anchor | 22.9190 | 0.63446 | 0.39745 |
| VQGAN uniform barycenter | 23.0242 | 0.64084 | 0.38397 |
| VQGAN oracle nearest particle | 23.0708 | 0.64334 | 0.33824 |
| VQGAN oracle simplex barycenter | 23.1058 | 0.64556 | 0.35059 |
| Deterministic VQAE | **23.2962** | **0.66059** | **0.29918** |
| Deterministic VQGAN | 21.7285 | 0.58010 | **0.17409** |

The oracle simplex barycenter improves the anchor by `+0.1868 dB` PSNR,
`+0.01110` SSIM, and `-0.04686` LPIPS. All paired 95% intervals favor it over
the anchor. However, relative to deterministic VQAE it changes PSNR by
`-0.1904 dB` (95% CI `[-0.2653,-0.1198]`), SSIM by `-0.01503`
(`[-0.01916,-0.01116]`), and LPIPS by `+0.05141`
(`[+0.04029,+0.06300]`). All three comparisons fail in the same direction.

The oracle uses a mean effective particle count of 2.47. Its mean null MSE is
`0.005753`, versus `0.005852` for uniform weighting. Thus weighting can exploit
some within-proposal variation, but the dominant limitation is the location of
the proposal family rather than weight estimation.

## Implementation and approximation audit

The original LMMSE uncertainty map was not serialized. Loading the full
STL-10 `train+unlabeled` array caused avoidable paging on the 16 GB host, so the
screen reconstructs the fixed conditioning map from 32 cached `(x0,x_G)` pairs
without truth or quality metrics. The resulting proxy reproduces the cached
deterministic VQGAN output with mean absolute error `0.00103` after 200 updates.
This proxy is adequate for a kill screen because the oracle-to-VQAE gaps are
one to two orders of magnitude larger. An exact streaming reconstruction would
be required only after a GO.

The operator geometry is constructed by thin QR of the wide transpose plus an
SVD of its 205-by-205 factor. This reproduces the direct-SVD row and null
projectors while avoiding its large Windows workspace. The actual operator has
rank 200 from 205 stored rows; redundant rows are preserved in the operator
identity hash.

All stochastic barycenter projections converge with zero box violation and
maximum relative record residual below `5.2e-9`. The deterministic control
projections stop at 256 iterations (`1.6e-6` for VQAE and `3.8e-5` for VQGAN),
far below the observed metric gaps and irrelevant to the AFRB kill decision.

## Artifacts

- Runner: `diagnose_afrb_proposal_headroom.py`
- Core helpers: `src/fiber_ratio_barycenter.py`
- Summary: `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round36/afrb_proposal_headroom/summary_val_128.json`
- Proxy map and audit: `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round36/afrb_proposal_headroom/uncertainty_map_seed0_proxy.pt` and `uncertainty_map_seed0_proxy_report.json`
- Tests: `tests/test_fiber_ratio_barycenter.py`, `tests/test_gauge_geometry.py`
