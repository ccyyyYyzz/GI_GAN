# Endpoint-quotiented FOHI decision

Decision: **KEEP_FOHI_KILL_EQ_FOHI**.

The held-out test remains unopened. The decision uses exactly three frozen validation pairings and no retuning.

## Crossed seed-by-image direct contrast: EQ-FOHI minus FOHI

| Metric | Mean delta | 95% interval | Favorable direction |
|---|---:|---:|---|
| PSNR | 0.00023150 | [0.00017841, 0.00030652] | positive |
| SSIM | 0.00002473 | [0.00001896, 0.00003206] | positive |
| LPIPS | -0.00000104 | [-0.00000421, 0.00000210] | negative |

Per-seed gate: **True**. Crossed direct gate: **False**.

The endpoint quotient is not rescued by coefficient, cutoff, norm, or tangent-cone changes if this gate fails.
