# Held-out attempt 1 invalidation

The first held-out invocation is invalidated before reconstruction or quality scoring.

- Frozen manifest SHA-256: `a1262b961cbcc929c4beec087b7a5d40da36c2386f18216ac645421cbbc9ebd0`.
- All three lanes pass code and artifact preflight, instantiate only the raw-hash audit, and stop before writing a test cache, reconstruction, metric vector, or aggregate quality value.
- The audit finds 1260 byte-identical raw images shared by STL-10 `test` and the frozen `train+unlabeled` development/prior prefix.
- No PSNR, SSIM, LPIPS, MSE, reconstruction image, subgroup result, or partial aggregate is produced or inspected.
- The recovery amendment excludes exactly those 1260 overlaps by raw SHA-256 and uses all 6740 remaining official-test images.  Method parameters, checkpoints, operators, rates, metrics, bootstrap seed, and the six-component decision gate remain unchanged.

This is a data-independence correction permitted by the predeclared feasibility/hash rule; it is not a response to reconstruction quality.
