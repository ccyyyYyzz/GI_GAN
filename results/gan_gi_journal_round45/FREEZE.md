# Round 45 three-seed validation freeze

The held-out test remains unopened. The method and its two scalar settings are frozen from three 512-image validation seeds before any test evaluation.

Frozen setting: cutoff = 0.18, alpha = 0.58, transition = 0.03.

Selection rule: require favorable seedwise means and favorable seedwise paired 95% confidence intervals for PSNR, SSIM, and LPIPS; among survivors, maximize the worst-seed normalized benefit.

| validation pairing | delta PSNR (95% CI) | delta SSIM (95% CI) | delta LPIPS (95% CI) | normalized benefit |
|---|---:|---:|---:|---:|
| seed0_primary_seed1_control | +0.016614 [+0.012557, +0.020736] | +0.001474 [+0.001254, +0.001698] | -0.003081 [-0.003941, -0.002238] | 4.395 |
| seed1_primary_seed2_control | +0.011061 [+0.007123, +0.015052] | +0.001245 [+0.001034, +0.001449] | -0.001616 [-0.002583, -0.000670] | 3.365 |
| seed2_primary_seed0_control | +0.004704 [+0.000349, +0.008980] | +0.001163 [+0.000924, +0.001394] | -0.005259 [-0.006339, -0.004250] | 3.613 |

Operator SHA-256: `8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628`.

The GAN proposal is compared against a matched VQAE-only residual generator. The same spectral rule, projection, and validation images are used in both arms; the only causal difference is the adversarial proposal source.

Source summary hashes:

- `seed0_primary_seed1_control`: `81f8427c3c8423cb7a611af814e9d84c9ad58ab13fa7fbcf96f114a87dba3af7`
- `seed1_primary_seed2_control`: `e2e9dfd25f2b55d3f5708b88cd7e3f4e4e202f20ef5fb29ec732eb806b659095`
- `seed2_primary_seed0_control`: `af7d07e49f2edeaa336390268a6ed17cd2110f21a6a9b05498517d179074f93b`
