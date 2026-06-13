# Phase 3 pilot — three-arm report (scr5, K=4, 20000 steps)

## omega_adv = 1e-3 (g2r_pilot_scr5_adv1e-3)

Final: **3/6 gates** — G-CAL PASS, G-DIV FAIL, G-NVR PASS, G-MEAN FAIL, G-CERT WARN, G-PERC FAIL; collapse_detected=False; roundtrip diff 0.0

### Gate trajectory (fixed val N=128, K=8, seed-pinned)

| step | N | std_med | PSNR(mean) | PSNR(sample) | G-CAL gap | edge_rho | NVR | relmeas_med | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 128 | 0.3800 | 14.32 | 7.48 | 6.83 | 0.132 | 3.132 | 7.00e-07 | 2/6 |
| 4000 | 128 | 0.4633 | 10.61 | 6.00 | 4.61 | 0.004 | 4.347 | 8.14e-07 | 2/6 |
| 6000 | 128 | 0.4841 | 9.56 | 5.90 | 3.66 | -0.024 | 4.604 | 1.04e-06 | 2/6 |
| 8000 | 128 | 0.4841 | 8.88 | 5.83 | 3.05 | -0.028 | 4.669 | 1.27e-06 | 3/6 |
| 10000 | 128 | 0.4841 | 8.40 | 5.77 | 2.63 | -0.028 | 4.739 | 1.44e-06 | 3/6 |
| 12000 | 128 | 0.4841 | 7.95 | 5.73 | 2.22 | -0.028 | 4.789 | 1.65e-06 | 3/6 |
| 14000 | 128 | 0.4841 | 7.54 | 5.67 | 1.87 | -0.029 | 4.828 | 1.98e-06 | 3/6 |
| 16000 | 128 | 0.4841 | 7.39 | 5.55 | 1.84 | -0.027 | 4.836 | 2.30e-06 | 3/6 |
| 18000 | 128 | 0.4841 | 7.26 | 5.58 | 1.67 | -0.028 | 4.882 | 2.50e-06 | 3/6 |
| 20000 | 128 | 0.4841 | 7.14 | 5.60 | 1.54 | -0.031 | 4.889 | 2.71e-06 | 3/6 |
| 20000 | 512 | 0.4841 | 7.10 | 5.66 | 1.44 | -0.025 | 4.928 | 2.69e-06 | 3/6 |

### Discriminator real/fake margin (window-averaged hinge logits)

| steps | d_real (mean) | d_fake (mean) | margin | grad-norm anomalies |
|---:|---:|---:|---:|:---|
| 1-1000 | +0.953 | -1.553 | +2.506 | steps [1] |
| 1001-2000 | +3.987 | -7.776 | +11.763 |  |
| 2001-3000 | +4.975 | -11.703 | +16.678 |  |
| 3001-4000 | +5.162 | -13.100 | +18.262 |  |
| 4001-5000 | +5.664 | -15.282 | +20.947 |  |
| 5001-6000 | +5.902 | -18.700 | +24.602 |  |
| 6001-7000 | +6.098 | -22.093 | +28.190 |  |
| 7001-8000 | +6.130 | -25.364 | +31.494 |  |
| 8001-9000 | +6.043 | -29.059 | +35.102 |  |
| 9001-10000 | +4.308 | -42.427 | +46.735 |  |
| 10001-11000 | +4.585 | -48.388 | +52.974 |  |
| 11001-12000 | +4.791 | -54.239 | +59.030 |  |
| 12001-13000 | +4.825 | -60.326 | +65.150 |  |
| 13001-14000 | +4.835 | -66.504 | +71.339 |  |
| 14001-15000 | +4.781 | -73.228 | +78.010 | steps [14500] |
| 15001-16000 | +4.796 | -79.030 | +83.825 |  |
| 16001-17000 | +4.743 | -85.802 | +90.545 |  |
| 17001-18000 | +4.804 | -92.373 | +97.178 |  |
| 18001-19000 | +4.815 | -100.325 | +105.140 |  |
| 19001-20000 | +4.795 | -107.170 | +111.966 |  |

## omega_adv = 3e-3 (g2r_pilot_scr5_adv3e-3)

Final: **3/6 gates** — G-CAL PASS, G-DIV FAIL, G-NVR PASS, G-MEAN FAIL, G-CERT WARN, G-PERC FAIL; collapse_detected=False; roundtrip diff 0.0

### Gate trajectory (fixed val N=128, K=8, seed-pinned)

| step | N | std_med | PSNR(mean) | PSNR(sample) | G-CAL gap | edge_rho | NVR | relmeas_med | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 128 | 0.3758 | 14.53 | 7.51 | 7.02 | 0.137 | 3.104 | 7.06e-07 | 2/6 |
| 4000 | 128 | 0.4677 | 10.84 | 5.86 | 4.97 | -0.016 | 4.308 | 8.01e-07 | 2/6 |
| 6000 | 128 | 0.4841 | 9.63 | 5.86 | 3.78 | -0.036 | 4.564 | 9.97e-07 | 2/6 |
| 8000 | 128 | 0.4841 | 8.89 | 5.75 | 3.14 | -0.040 | 4.667 | 1.21e-06 | 3/6 |
| 10000 | 128 | 0.4841 | 8.18 | 5.66 | 2.51 | -0.037 | 4.722 | 1.38e-06 | 3/6 |
| 12000 | 128 | 0.4841 | 7.28 | 5.44 | 1.85 | -0.045 | 4.713 | 1.55e-06 | 3/6 |
| 14000 | 128 | 0.4841 | 7.93 | 5.77 | 2.16 | -0.039 | 4.760 | 1.83e-06 | 3/6 |
| 16000 | 128 | 0.4841 | 7.70 | 5.72 | 1.98 | -0.043 | 4.846 | 2.19e-06 | 3/6 |
| 18000 | 128 | 0.4841 | 7.21 | 5.68 | 1.53 | -0.041 | 4.816 | 2.43e-06 | 3/6 |
| 20000 | 128 | 0.4841 | 6.81 | 5.42 | 1.39 | -0.038 | 4.748 | 2.62e-06 | 3/6 |
| 20000 | 512 | 0.4841 | 6.77 | 5.51 | 1.26 | -0.029 | 4.851 | 2.61e-06 | 3/6 |

### Discriminator real/fake margin (window-averaged hinge logits)

| steps | d_real (mean) | d_fake (mean) | margin | grad-norm anomalies |
|---:|---:|---:|---:|:---|
| 1-1000 | +1.041 | -1.496 | +2.537 | steps [1] |
| 1001-2000 | +4.047 | -7.443 | +11.490 |  |
| 2001-3000 | +4.763 | -10.057 | +14.820 |  |
| 3001-4000 | +5.321 | -12.742 | +18.063 |  |
| 4001-5000 | +5.811 | -15.729 | +21.540 |  |
| 5001-6000 | +6.309 | -19.289 | +25.597 |  |
| 6001-7000 | +6.415 | -23.190 | +29.604 |  |
| 7001-8000 | +6.417 | -26.821 | +33.238 |  |
| 8001-9000 | +6.521 | -30.439 | +36.959 |  |
| 9001-10000 | +7.578 | -32.166 | +39.744 |  |
| 10001-11000 | +7.582 | -36.413 | +43.995 |  |
| 11001-12000 | +7.573 | -40.301 | +47.874 |  |
| 12001-13000 | +7.606 | -44.621 | +52.227 |  |
| 13001-14000 | +7.603 | -48.837 | +56.440 |  |
| 14001-15000 | +7.574 | -53.591 | +61.165 |  |
| 15001-16000 | +7.575 | -57.916 | +65.491 |  |
| 16001-17000 | +7.526 | -62.473 | +70.000 |  |
| 17001-18000 | +7.578 | -67.205 | +74.783 |  |
| 18001-19000 | +7.597 | -72.562 | +80.159 |  |
| 19001-20000 | +7.578 | -77.417 | +84.995 |  |

## omega_adv = 1e-2 (g2r_pilot_scr5_adv1e-2)

Final: **3/6 gates** — G-CAL PASS, G-DIV FAIL, G-NVR PASS, G-MEAN FAIL, G-CERT WARN, G-PERC FAIL; collapse_detected=False; roundtrip diff 0.0

### Gate trajectory (fixed val N=128, K=8, seed-pinned)

| step | N | std_med | PSNR(mean) | PSNR(sample) | G-CAL gap | edge_rho | NVR | relmeas_med | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 128 | 0.3868 | 14.76 | 7.47 | 7.30 | 0.120 | 3.193 | 6.94e-07 | 2/6 |
| 4000 | 128 | 0.4687 | 11.58 | 5.87 | 5.71 | -0.028 | 4.236 | 7.77e-07 | 2/6 |
| 6000 | 128 | 0.4841 | 10.32 | 5.80 | 4.52 | -0.044 | 4.504 | 8.52e-07 | 2/6 |
| 8000 | 128 | 0.4841 | 9.85 | 6.06 | 3.79 | -0.049 | 4.685 | 1.02e-06 | 2/6 |
| 10000 | 128 | 0.4841 | 9.35 | 6.08 | 3.27 | -0.046 | 4.681 | 1.23e-06 | 3/6 |
| 12000 | 128 | 0.4841 | 8.89 | 5.73 | 3.17 | -0.056 | 4.764 | 1.37e-06 | 3/6 |
| 14000 | 128 | 0.4841 | 8.31 | 5.86 | 2.45 | -0.047 | 4.766 | 1.50e-06 | 3/6 |
| 16000 | 128 | 0.4841 | 7.48 | 5.53 | 1.95 | -0.057 | 4.742 | 1.72e-06 | 3/6 |
| 18000 | 128 | 0.4841 | 7.35 | 5.53 | 1.83 | -0.049 | 4.746 | 2.03e-06 | 3/6 |
| 20000 | 128 | 0.4841 | 7.72 | 5.74 | 1.98 | -0.051 | 4.836 | 2.28e-06 | 3/6 |
| 20000 | 512 | 0.4841 | 7.56 | 5.81 | 1.75 | -0.038 | 4.848 | 2.27e-06 | 3/6 |

### Discriminator real/fake margin (window-averaged hinge logits)

| steps | d_real (mean) | d_fake (mean) | margin | grad-norm anomalies |
|---:|---:|---:|---:|:---|
| 1-1000 | +0.894 | -1.472 | +2.366 | steps [1] |
| 1001-2000 | +3.945 | -6.366 | +10.311 |  |
| 2001-3000 | +3.956 | -9.720 | +13.676 |  |
| 3001-4000 | +4.419 | -13.568 | +17.986 |  |
| 4001-5000 | +4.890 | -17.040 | +21.930 |  |
| 5001-6000 | +5.749 | -19.555 | +25.304 |  |
| 6001-7000 | +5.321 | -24.783 | +30.105 |  |
| 7001-8000 | +5.532 | -28.864 | +34.396 |  |
| 8001-9000 | +5.694 | -33.457 | +39.151 |  |
| 9001-10000 | +5.728 | -38.280 | +44.008 |  |
| 10001-11000 | +5.726 | -43.688 | +49.415 |  |
| 11001-12000 | +5.719 | -48.723 | +54.442 |  |
| 12001-13000 | +5.728 | -54.134 | +59.862 |  |
| 13001-14000 | +5.758 | -59.642 | +65.400 |  |
| 14001-15000 | +5.701 | -65.856 | +71.558 |  |
| 15001-16000 | +5.703 | -71.045 | +76.749 |  |
| 16001-17000 | +5.690 | -76.981 | +82.672 |  |
| 17001-18000 | +5.721 | -83.041 | +88.762 |  |
| 18001-19000 | +5.735 | -89.744 | +95.478 |  |
| 19001-20000 | +5.717 | -95.914 | +101.631 |  |

## Decision (pre-registered rules)

| arm (omega_adv) | gates passed | G-CAL | G-DIV | G-NVR | G-MEAN | G-CERT | G-PERC | mean PSNR drop vs base |
|---|---:|---|---|---|---|---|---|---:|
| 1e-3 | 3/6 | PASS | FAIL | PASS | **FAIL** | WARN | FAIL | -15.16 dB |
| 3e-3 | 3/6 | PASS | FAIL | PASS | **FAIL** | WARN | FAIL | -15.49 dB |
| 1e-2 | 3/6 | PASS | FAIL | PASS | **FAIL** | WARN | FAIL | -14.69 dB |

**STOP RULE FIRED:** all three arms fail G-MEAN (required >= baseline - 0.3 dB;
observed ~15 dB below). Per the pre-registered rule, we report the G-MEAN
trajectories and STOP — no unreported beta_SD tuning. No collapse snapshots
(std rose to a plateau, never declined for >4000 steps); roundtrip diff 0.0
on every arm.

### Interpretation of the omega_adv lower bound (from the D margin)

In all three arms the discriminator real/fake margin grows **without bound**
(to +85 / +112 / +102), driven almost entirely by `d_fake` collapsing toward
-80..-107 while `d_real` stays bounded near +5..+7. The discriminator was
therefore **strongly winning throughout** — adversarial pressure was never
nominal/idle in any arm, including the weakest (1e-3). This is the load-
bearing reason the three gate trajectories are nearly indistinguishable: with
D saturating, the generator's adversarial gradient (omega_adv * d/dG[-d_fake])
is swamped by the fixed-beta_SD reward, which drives per-pixel sample std to
the same output-range attractor **0.4841** in every arm by ~6000 steps,
regardless of omega_adv across the full 10x range. Thus the failure is NOT
"omega_adv too small to engage"; it is the fixed-beta_SD variance-saturation
attractor (mean PSNR collapses, edge_rho ~ 0, NVR inflated to ~4.9). The
omega_adv lower bound is therefore not identifiable from these runs (even 1e-3
produced saturated D); the controlling variable is beta_SD, which motivates
the pre-registered Round 2 closed-loop controller.

RelMeasErr stayed at the float32 floor on every arm (median 7e-7 -> 2.7e-6,
the rise tracking ||inner|| growth as variance inflates; p95 < 1e-5
throughout), so G-CERT is WARN (float32-floor flag), as designed.

### Round 2 (authorized)

The pre-registered amendment (configs/g2r/ROUND2_AMENDMENT.md) executes after
this report iff the stop rule fired — both conditions are now met. Its
omega_adv rule resolves to **1e-2** ("smallest arm that visibly contained
variance growth; if none did, 1e-2" — none did; all reached the 0.4841
plateau). One change only: fixed beta_SD -> closed-loop controller.

