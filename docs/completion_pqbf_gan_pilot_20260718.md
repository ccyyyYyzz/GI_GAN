# PQBF-GAN completion-project pilot (2026-07-18)

## Outcome

The local RTX 4060 directional pilot is positive. A validation-only checkpoint sweep selects the Stage-B EMA at step 750. Relative to the content-only PQBF model, it reduces LPIPS by 12.67% while changing PSNR by -0.090 dB and SSIM by +0.00198. Relative to the exact bounded fiber LMMSE anchor, it improves PSNR by +0.525 dB, SSIM by +0.0683, and LPIPS by 49.88%.

The selected GAN therefore contributes a measurable perceptual improvement without replacing the physics constraint: every scored reconstruction is the certified projection onto the intersection of the measurement fiber and the image box.

## Fixed pilot protocol

- Dataset: 2048 STL-10 training images, 128 validation images, and 128 unopened test images; source hashes are disjoint.
- Measurement operator: 205 signed measurements at 64 x 64 resolution, effective rank 200, SHA-256 `8a16664e078e1ea9c3b163da24262f017ef553c50bb8012f130ab9af6589a628`.
- Generator: shared three-step projector-gated U-Net, 787107 trainable parameters.
- Discriminator: conditional spectral-normalized PatchGAN, 314977 trainable parameters.
- Stage A: 2000 supervised updates.
- Stage B: 1500 adversarial updates, with EMA checkpoints evaluated at steps 300, 500, 750, 1000, 1250, and 1500.
- Selection rule: among Stage-B checkpoints with PSNR delta >= -0.10 dB and SSIM delta >= -0.002 relative to Stage A, select the lowest validation LPIPS.

## Validation results

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| Exact bounded fiber LMMSE anchor | 22.3386 | 0.60777 | 0.32754 |
| PQBF content model | 22.9539 | 0.67412 | 0.18796 |
| PQBF-GAN, step 300 | 22.9257 | 0.67557 | 0.17611 |
| PQBF-GAN, step 500 | 22.9003 | 0.67607 | 0.17002 |
| **PQBF-GAN, step 750 (selected)** | **22.8637** | **0.67609** | **0.16415** |
| PQBF-GAN, step 1000 | 22.8262 | 0.67571 | 0.15986 |
| PQBF-GAN, step 1250 | 22.7854 | 0.67504 | 0.15634 |
| PQBF-GAN, step 1500 | 22.7472 | 0.67420 | 0.15419 |

For the selected checkpoint, the paired mean PSNR difference relative to the anchor is +0.5252 dB (95% bootstrap interval +0.4269 to +0.6272 dB). The paired mean LPIPS difference relative to the content model is -0.02381 (95% bootstrap interval -0.02625 to -0.02145), corresponding to a 12.67% relative reduction. The test split remains unopened.

## Adversarial-game evidence

After correcting the PatchGAN R1 scale to differentiate each sample's spatially averaged patch score, the tail medians are:

- discriminator hinge-plus-R1 loss: 0.6506;
- real-minus-fake discriminator gap: 2.0078;
- achieved adversarial-to-supervised tail-gradient ratio: 0.1828.

All three lie inside the preregistered healthy reference ranges. This establishes that the discriminator is load-bearing rather than nominal.

## Constraint certificates and provenance

- Maximum scored original-measurement relative residual: `1.37e-8`.
- Pixel range: exactly within `[0, 1]`, with no post-evaluation clamp.
- Selected checkpoint: `E:/GAN_FCC_WORK/experiments/completion_gan_round18/pqbf_pilot_checkpoint_sweep_seed0/checkpoints/stage_b_step000750.pt`.
- Selected checkpoint SHA-256: `1aa9272117ca7230552dd96880c14a1d6704533c60d904b8fff90d4b0803c847`.
- Run summary: `E:/GAN_FCC_WORK/experiments/completion_gan_round18/pqbf_pilot_checkpoint_sweep_seed0/reports/run_summary.json`.
- Run-summary SHA-256: `80ad44aec1324e654b55bf3f84a3a2240710902378df4f5c05bd841848f0213c`.
- Qualitative grid: `E:/GAN_FCC_WORK/experiments/completion_gan_round18/pqbf_pilot_checkpoint_sweep_seed0/reports/validation/qualitative_grid.png`.
- Configuration: `configs/completion_gan_round18/pilot_checkpoint_sweep.yaml`.

This document records a directional pilot, not the final independent test result.
