# Round 26: PQBF adversarial-strength causal Pareto diagnostic

Date: 2026-07-18

## Question

Is the small PSNR/SSIM loss of the existing PQBF-GAN merely caused by an overly
strong adversarial gradient, such that a weaker nonzero adversarial game improves
PSNR, SSIM, and LPIPS simultaneously over an equal-step zero-adversarial
continuation?

## Protocol

- Reuse the frozen Stage-A checkpoint, split, 205-row operator, cache, loader order,
  discriminator seed, optimizer, and 750-update Stage-B budget.
- Change only the adaptive adversarial target and its initial coefficient.
- Score the 128-image validation split; the test split remains unopened.
- Compare with the previously produced equal-step zero-adversarial checkpoint.

The configured target is not the realized gradient ratio because the adaptive
coefficient is exponentially smoothed. The realized tail-median ratio is therefore
reported below.

## Result

Zero-adversarial validation reference:
`22.869164 dB / 0.676138 / 0.164824` for PSNR / SSIM / LPIPS.

| Configured target | Realized adversarial/supervised gradient ratio | Delta PSNR (dB) | Delta SSIM | Delta LPIPS |
|---:|---:|---:|---:|---:|
| 0.025 | 0.105631 | -0.001990 | -0.00000691 | -0.000251 |
| 0.050 | 0.137185 | -0.003212 | -0.00001730 | -0.000422 |
| 0.100 (existing) | 0.182797 | -0.005424 | -0.00004276 | -0.000676 |

Every nonzero adversarial strength improves LPIPS and worsens both distortion
metrics relative to the equal-step supervised continuation. Reducing the
adversarial strength moves all three causal effects smoothly toward zero rather
than revealing an all-metric optimum.

All outputs remain in `[0,1]`; maximum relative original-measurement residual is
below `1.50e-8` in the two new runs.

## Decision

`PARETO_CONFIRMED_NO_ALL_METRIC_ADVERSARIAL_SWEET_SPOT`

Do not continue scalar adversarial-weight tuning. The 0.025 target is a more
balanced completion-report operating point, but its GAN-specific effects are too
small and retain the same perception-distortion tradeoff. A journal method must
change the information role of the GAN, not its coefficient.

## Artifacts

- Configs:
  - `configs/completion_gan_round18/pilot_adv_ratio_0p025.yaml`
  - `configs/completion_gan_round18/pilot_adv_ratio_0p05.yaml`
- Raw summaries:
  - `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round26/pqbf_adv_ratio_0p025/reports/run_summary.json`
  - `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round26/pqbf_adv_ratio_0p05/reports/run_summary.json`

