# FOHI three-seed freeze

Fiber-orthogonal high-pass innovation (FOHI) is frozen before the held-out test. It removes, image by image, the component of the filtered GAN correction that is parallel to the structural correction. No norm restoration or learned gate is used.

Frozen setting: cutoff = 0.12, transition = 0.03, alpha = 0.50.

| validation pairing | removed parallel energy | delta PSNR (95% CI) | delta SSIM (95% CI) | delta LPIPS (95% CI) |
|---|---:|---:|---:|---:|
| seed0_primary_seed1_control | 14.5% | +0.026601 [+0.022431, +0.030791] | +0.001710 [+0.001457, +0.001967] | -0.004458 [-0.005180, -0.003723] |
| seed1_primary_seed2_control | 18.1% | +0.025783 [+0.021963, +0.029641] | +0.001385 [+0.001165, +0.001605] | -0.002632 [-0.003402, -0.001856] |
| seed2_primary_seed0_control | 12.4% | +0.018411 [+0.013824, +0.022926] | +0.001489 [+0.001218, +0.001761] | -0.005200 [-0.006082, -0.004352] |

Every seed passes the paired three-metric confidence-interval gate against the matched non-GAN structural control. Every exact box-fiber projection converges with zero box violation and relative record error below 1.00e-7.

This freeze supersedes the validation-only unorthogonalized setting (cutoff 0.18, alpha 0.58). The replacement is based solely on the pre-test FOHI falsification rule and introduces no new fitted parameter.

Operator SHA-256: `8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628`.

Source summary hashes:

- `seed0_primary_seed1_control`: `5cde50cacdbe0732122dff97df43cb82181681e6dc3014c86347047fb4e8140c`
- `seed1_primary_seed2_control`: `0c92eca5219764c252f2f67a60a035ba42c7c55f644c3e1c9a575ed668eecaff`
- `seed2_primary_seed0_control`: `4abf6a9e6a92b3f9f2dbd9ca449e8be3b2a815823f0f2fef041468f9f68bca8d`
