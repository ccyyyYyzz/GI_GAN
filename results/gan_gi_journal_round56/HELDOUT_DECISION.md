# VQGAN-guided FOHI one-shot held-out decision

Headline six-component gate: **PASS**.

| Rate | Delta PSNR | Delta SSIM | Delta LPIPS | Joint interval gate |
|---:|---:|---:|---:|---:|
| 5% | +0.084193 | +0.004876 | -0.010070 | PASS |
| 10% | +0.234878 | +0.011527 | -0.016685 | PASS |

All 6740 raw-hash-disjoint STL-10 test images are used once. The 1260 byte-identical development overlaps are excluded before any quality metric. No post-test method change is permitted.
