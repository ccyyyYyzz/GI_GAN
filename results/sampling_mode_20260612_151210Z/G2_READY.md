# G2 Ready Dossier

## Exact Launch Command

Do not launch yet. After blockers are resolved, the intended launch command should use the prepared `G2_CONFIG.yaml` and write all outputs to a new timestamped directory, with audit enabled and K=32 per-sample saving.

## Estimated Wall-Clock And Memory

Not measured because smoke was not run. Any estimate would be fabricated.

## Pre-Registered Acceptance Band

- kappa >= 1.15.
- Observed delta PSNR consistent with -10 log10(kappa) within +/-0.3 dB.
- RelMeasErr unchanged to 1e-6 relative with audit on.
- Visible diversity in an 8x4 sample grid.

## Pre-Registered No-Go Reading

If kappa ~= 1 despite diversity pressure, report: adversarial fine-tuning reduces to the mean mode at this information budget. This is an acceptable publishable negative outcome.

## Artifacts A Future Run Must Emit

- train/val/test split indices and SHA256 hashes.
- all K=32 individual stochastic samples per test image plus z seeds.
- per-sample PSNR/SSIM/RelMeasErr and sample-mean metrics.
- pixelwise std maps, null variance ratio, kappa proxy.
- LPIPS/FID/KID only if packages and local weights are available.
- smoke loss curves and D real/fake margins.

READY TO LAUNCH: no - blockers: ['No saved main no-leak train/val/test split hashes are available.', 'Pilot split/eval index hashes are not available.', 'Old G1 code path appears deterministic with no explicit stochastic z.', 'Controlled G2 smoke was not run because provenance is unsafe and stochastic branch implementation has not been reviewed.']
