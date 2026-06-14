# Exploratory Posterior And G2R Work

This file indexes posterior/diversity work that should stay out of the core evidence package unless it is rerun under a locked protocol and reviewed separately.

## Posterior Anti-Collapse / Calibration Chain

| Phase | Local folder | Status | Key signal | Citation status |
| --- | --- | --- | --- | --- |
| Phase79 posterior | `E:\ns_mc_gan_gi\outputs_phase79_posterior_anti_collapse\rad5_rowspace_diversity_diagnostic` | Completed exploratory run | Final smoke mean pixel std 0.022008; P0 variance 0.00091181; PR variance about 2.97e-12; RelMeasErr max about 3.94e-05. | Future-work only |
| Phase80 | `E:\ns_mc_gan_gi\outputs_phase80_posterior_calibration\rad5_centered_diversity_anchor` | Completed calibration repair | Final smoke mean pixel std 0.004556; P0 variance 3.69e-05; anchor RMSE 0.007015; RelMeasErr max about 2.87e-05. | Future-work only |
| Phase81 div2 | `E:\ns_mc_gan_gi\outputs_phase81_diversity_weight_scan\rad5_centered_anchor2_div2` | Completed weight scan | Final smoke mean pixel std 0.021936; P0 variance 0.0009715; anchor RMSE 0.03358; RelMeasErr max about 3.12e-05. | Future-work only |
| Phase81 div4 | `E:\ns_mc_gan_gi\outputs_phase81_diversity_weight_scan\rad5_centered_anchor2_div4` | Partial runlog only | Runlog reaches step 50 with std 0.00959 and P0 variance 0.000192845. | Do not cite |

Shared provenance anchors:

- Deterministic anchor checkpoint SHA256: `b44de4aef4c0b0a9a8d4ff7f5e59c5916e6e0cd6f5a999054016fe411fbff6e0`
- Rad-5 A float32-byte SHA256: `6b840ba6a9daad98862e2be23c215a23ac40ee626975e8e6467eca82ef118c4f`

Interpretation:

- These runs are promising for future posterior/diversity work.
- They are not part of the core measurement-certificate paper.
- They do not establish calibrated uncertainty or a deployable posterior sampler.

## G2R Pilot Gate Results

| Run | Overall gate | Certificate gate | Mean-quality gate | Diversity gate | Notes |
| --- | --- | --- | --- | --- | --- |
| `g2r_pilot_sanity` | fail | warn/pass; max RelMeasErr 1.43e-06 | fail; sample mean PSNR 20.35 vs baseline 22.04 | fail; median pixel std 0.00115 | Too little null variance and weaker quality. |
| `g2r_pilot_scr5_adv1e-2` | fail | warn/pass; max RelMeasErr 1.09e-05 | fail; sample mean PSNR 7.56 vs baseline 22.25 | fail; variance not structure-aligned | High variance but severe quality collapse. |
| `g2r_pilot_scr5_adv1e-3` | fail | warn/pass; max RelMeasErr 1.68e-05 | fail; sample mean PSNR 7.10 vs baseline 22.25 | fail; variance not structure-aligned | High variance but severe quality collapse. |
| `g2r_pilot_scr5_adv3e-3` | fail | warn/pass; max RelMeasErr 1.68e-05 | fail; sample mean PSNR 6.77 vs baseline 22.25 | fail; variance not structure-aligned | High variance but severe quality collapse. |

Safe archival statement:

- G2R pilots are useful failed-gate evidence for future posterior design.
- They should not be used as successful posterior-sampling evidence.
