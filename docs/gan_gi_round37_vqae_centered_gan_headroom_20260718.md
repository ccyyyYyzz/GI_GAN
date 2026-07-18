# Round 37: VQAE-centred GAN residual headroom

## Decision

**GO to a learned VQAE-centred residual adapter.** Moving the proposal centre
from the clipped LMMSE anchor to the stronger VQAE reconstruction reverses the
Round-36 support failure. The current, not-yet-retuned VQGAN already contains
small null-space corrections that can beat VQAE in PSNR, SSIM, and LPIPS at the
same time.

This is a validation-only support result. Oracle weights use truth and are not a
deployable method. The locked test remains unopened.

## Evidence on 128 validation images

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| Deterministic VQAE | 23.2962 | 0.66059 | 0.29918 |
| VQAE-centred uniform VQGAN particles | 23.2996 | 0.66061 | 0.29143 |
| VQAE-centred oracle nearest particle | 23.3227 | 0.66050 | 0.28053 |
| VQAE-centred oracle simplex barycenter | **23.3417** | **0.66205** | **0.28735** |
| VQAE-to-VQGAN oracle segment | **23.3467** | **0.66251** | **0.28817** |

The oracle simplex barycenter changes PSNR by `+0.04550 dB` (95% CI
`[+0.03601,+0.05591]`), SSIM by `+0.001458`
(`[+0.001035,+0.001891]`), and LPIPS by `-0.01183`
(`[-0.01371,-0.01014]`) relative to VQAE. The direct segment oracle gives
`+0.05054 dB`, `+0.001922`, and `-0.01101`, also with three favorable paired
intervals.

The direct GAN correction is positive for 85.9% of images, with mean coefficient
0.111 and median 0.104. Thus the useful regime is a small GAN residual around
VQAE, not a full move from LMMSE toward VQGAN.

## Immediate implementation

The pilot trains a sub-million-parameter conditional gate that can only
reweight the supplied prior residual. Its output is null-projected before
addition and receives one final box/fiber projection. Spatial and radial-band
versions are trained in parallel. A matched control replaces the VQGAN residual
with an independently trained VQAE residual while keeping architecture, data,
steps, losses, and projection identical.

The local two-step smoke test also establishes a strong fixed control: a global
GAN coefficient of 0.10 on all 512 validation images changes PSNR by
`+0.02885 dB`, SSIM by `+0.001192`, and LPIPS by `-0.01036`; all paired 95%
intervals are favorable. A learned adapter must beat this constant, not merely
beat raw VQAE.

## Artifacts

- Headroom runner: `diagnose_vqae_centered_gan_headroom.py`
- Adapter runner: `train_vqae_centered_residual_adapter.py`
- Adapter module: `src/vqae_centered_residual_adapter.py`
- Headroom summary: `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round37/vqae_centered_gan_headroom/summary_val_128.json`
- Local smoke summary: `E:/GAN_FCC_WORK/experiments/gan_gi_journal_round38/local_smoke/summary.json`
