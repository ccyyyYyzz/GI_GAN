# FOHI cross-rate validation

The held-out test remains unopened. All entries use frozen cutoff 0.12, transition 0.03, and alpha 0.50.

| Rate | Seed | ΔPSNR (dB) | ΔSSIM | ΔLPIPS | Triple CI | Projection |
|---:|---:|---:|---:|---:|---|---|
| 2% | 0 | +0.087339 | +0.005624 | -0.005647 | True | True |
| 2% | 1 | +0.078168 | +0.005051 | -0.019044 | True | True |
| 2% | 2 | +0.118269 | +0.007470 | +0.001404 | False | True |
| 10% | 0 | +0.225282 | +0.010751 | -0.013175 | True | True |
| 10% | 1 | +0.209157 | +0.010172 | -0.018508 | True | True |
| 10% | 2 | +0.224409 | +0.010603 | -0.018228 | True | True |

At 10%, all three seeds pass the joint PSNR/SSIM/LPIPS confidence gate. At 2%, seeds 0 and 1 pass, while seed 2 improves distortion but significantly worsens LPIPS; the ultra-low-rate limitation is retained rather than tuned away.

The main 5% three-seed freeze is reported separately in `results/gan_gi_journal_round47/FREEZE.md`.
