# Round 35: GAN-induced rank-coordinate screen

## Question

Does the VQGAN--VQAE old-fiber residual define a sample-specific pixel order in
which the unknown reconstruction error is concentrated in a few low-sequency
Walsh modes?  If so, a classical complementary-DMD measurement could read those
modes after the GAN supplies only the missing coordinate system.

This is a validation-only causal screen.  No test item was read.  Truth selected
no pattern; it was used only to simulate the additional bucket records and to
score reconstructions.

## Exact construction

For each image, sort the 4096 pixels by a truth-free guide residual.  Pull the
first non-DC sequency-ordered Walsh rows back through that permutation.  Every
resulting row is exactly balanced and binary and is therefore implemented by one
complementary DMD exposure pair.  Joint Dykstra projection enforces the original
205 measurements, all newly acquired signed buckets, and the image box.

The guides were:

- VQGAN minus the seed-0 VQAE anchor (adversarial guide);
- seed-1 VQAE minus the same anchor (matched non-adversarial diversity control);
- convex guide blends with VQGAN weights 0.25, 0.50, and 0.75;
- a deterministic random permutation control.

The equal-pattern comparator was the next 1, 2, 4, or 8 low-frequency DCT rows.

## Validation result (128 images, 256 projection iterations)

| guide | rows | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| fixed next DCT | 1 | 23.388594 | 0.672294 | 0.295211 |
| VQGAN rank | 1 | 23.395598 | 0.671720 | 0.286019 |
| VQAE rank | 1 | 23.434616 | 0.673591 | 0.285083 |
| blend, GAN weight 0.25 | 1 | **23.433322** | **0.673130** | **0.280835** |
| fixed next DCT | 2 | 23.412883 | 0.673456 | 0.295098 |
| blend, GAN weight 0.25 | 2 | 23.435511 | 0.673210 | 0.279036 |
| fixed next DCT | 4 | 23.440572 | 0.675451 | 0.293330 |
| blend, GAN weight 0.25 | 4 | 23.463811 | 0.675084 | 0.278796 |
| fixed next DCT | 8 | **23.495684** | **0.678896** | 0.291216 |
| blend, GAN weight 0.25 | 8 | 23.489293 | 0.676512 | **0.275612** |

The higher-accuracy 2048-iteration one-row repeat gave
`23.444476 / 0.673785 / 0.280590` for the 0.25 GAN blend and
`23.400359 / 0.672974 / 0.294910` for fixed DCT.  Paired mean differences
(blend minus DCT) were:

- PSNR `+0.044116` dB, 95% bootstrap CI `[+0.030268,+0.059682]`;
- SSIM `+0.000811`, CI `[-0.000007,+0.001669]`;
- LPIPS `-0.014320`, CI `[-0.018235,-0.010843]`.

The same one-row VQAE-only coordinate achieved
`23.445613 / 0.674239 / 0.284795`.  Adding 25% VQGAN content therefore supplies
an additional LPIPS improvement, but does not improve PSNR or SSIM over the
matched VQAE coordinate.

## Decision

**KILL as a scalable journal mechanism; retain as a mechanistic boundary and a
positive one-row engineering result.**

The first rank split is useful and the mixed guide beats a fixed DCT query on
mean PSNR, SSIM, and LPIPS.  However, the matched VQAE coordinate is at least as
good for distortion, and additional Walsh modes increasingly reproduce the
ordinary perception--distortion trade-off instead of creating a new favorable
regime.  The experiment therefore explains why the previously observed single
balanced query works, but it does not establish a load-bearing GAN coordinate
system or a new multi-measurement law.

The high-accuracy repeat was used only to check metric stability; its Dykstra
box residual remained about `1.1e-3`, so it is not a freeze-quality projection
artifact.  No photon-noise or test-split run is authorized for this killed
direction.

## Artifacts

- executable: `diagnose_gan_rank_coordinate_pilot.py`;
- broad screen: `experiments/gan_gi_journal_round35/gan_rank_coordinate/blend_screen/summary_val_128.json`;
- one-row repeat: `experiments/gan_gi_journal_round35/gan_rank_coordinate/confirm_b1/summary_val_128.json`.
