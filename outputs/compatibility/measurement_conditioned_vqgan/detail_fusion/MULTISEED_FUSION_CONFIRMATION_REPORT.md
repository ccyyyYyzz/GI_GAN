# Multi-Seed VQGAN Detail-Fusion Canary Report

Classification: `BALANCED_VQGAN_FUSION_CONFIRMED`
- balanced_gate_passed: `True`
- quality_lite_gate_passed: `True`
- full_vqgan_quality_passed: `True`
- seeds: [0, 1, 2] | bootstrap_reps: 2000 | radial bands: 16

Zero-training null-space fusion: x_hat = x0 + P0( d_A + W (d_G - d_A) ), exact measurement audit.
Selection on val, mechanical scoring on dev. Baseline = VQAE refiner (W=0).

## Dev method means (pooled over seeds)

| arm | LPIPS | full_rmse | centered_rmse | psnr | ssim | rapsd | relmeaserr |
|---|---|---|---|---|---|---|---|
| vqae | 0.2931 | 0.0749 | 0.0749 | 22.8688 | 0.6581 | 0.0031 | 3.57e-07 |
| vqgan | 0.1669 | 0.0913 | 0.0913 | 21.1675 | 0.5728 | 0.0020 | 3.56e-07 |
| fusion_balanced | 0.1966 | 0.0788 | 0.0788 | 22.4371 | 0.6346 | 0.0028 | 3.55e-07 |
| fusion_quality_lite | 0.1769 | 0.0833 | 0.0833 | 21.9576 | 0.6108 | 0.0025 | 3.56e-07 |
| fusion_oracle | 0.1669 | 0.0912 | 0.0912 | 21.1751 | 0.5732 | 0.0020 | 3.55e-07 |

## Fused-vs-VQAE clustered bootstrap (per arm)

### fusion_balanced
- ΔLPIPS = -0.0965 CI[-0.1002, -0.0929], relative_gain = 0.329, same-direction seeds = 3/3
- ΔPSNR = -0.432 dB | ΔRAPSD = -0.00029 | method relmeaserr mean = 3.55e-07
- conditions: {"lpips_gain_ge_5pct_ci_upper_lt0": true, "lpips_2_of_3_seeds_same_direction": true, "rapsd_same_direction": true, "psnr_drop_within_2p5db": true, "psnr_drop_within_0p5db": true, "rmse_increase_within_0p005": true, "relmeaserr_ok": true}

### fusion_quality_lite
- ΔLPIPS = -0.1162 CI[-0.1199, -0.1125], relative_gain = 0.396, same-direction seeds = 3/3
- ΔPSNR = -0.911 dB | ΔRAPSD = -0.00061 | method relmeaserr mean = 3.56e-07
- conditions: {"lpips_gain_ge_5pct_ci_upper_lt0": true, "lpips_2_of_3_seeds_same_direction": true, "rapsd_same_direction": true, "psnr_drop_within_2p5db": true, "psnr_drop_within_0p5db": false, "rmse_increase_within_0p005": false, "relmeaserr_ok": true}

### fusion_oracle
- ΔLPIPS = -0.1262 CI[-0.1305, -0.1218], relative_gain = 0.431, same-direction seeds = 3/3
- ΔPSNR = -1.694 dB | ΔRAPSD = -0.00113 | method relmeaserr mean = 3.55e-07
- conditions: {"lpips_gain_ge_5pct_ci_upper_lt0": true, "lpips_2_of_3_seeds_same_direction": true, "rapsd_same_direction": true, "psnr_drop_within_2p5db": true, "psnr_drop_within_0p5db": false, "rmse_increase_within_0p005": false, "relmeaserr_ok": true}

### vqgan
- ΔLPIPS = -0.1262 CI[-0.1305, -0.1218], relative_gain = 0.431, same-direction seeds = 3/3
- ΔPSNR = -1.701 dB | ΔRAPSD = -0.00113 | method relmeaserr mean = 3.56e-07
- conditions: {"lpips_gain_ge_5pct_ci_upper_lt0": true, "lpips_2_of_3_seeds_same_direction": true, "rapsd_same_direction": true, "psnr_drop_within_2p5db": true, "psnr_drop_within_0p5db": false, "rmse_increase_within_0p005": false, "relmeaserr_ok": true}

## Per-seed selected operating points (chosen on val)

- seed0: {"balanced": "scalar_0.55", "quality_lite": "scalar_0.75", "oracle": "lowpass_cut2"}
- seed1: {"balanced": "scalar_0.55", "quality_lite": "scalar_0.75", "oracle": "lowpass_cut2"}
- seed2: {"balanced": "scalar_0.50", "quality_lite": "scalar_0.70", "oracle": "lowpass_cut2"}