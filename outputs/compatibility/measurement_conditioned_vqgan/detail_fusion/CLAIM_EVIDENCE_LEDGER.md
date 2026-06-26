# Claim-Evidence Ledger — VQGAN Detail Fusion

Mechanical classification: `BALANCED_VQGAN_FUSION_CONFIRMED`.

| claim | evidence | status |
|---|---|---|
| Exact measurement consistency preserved | all arms relmeaserr mean <= 1e-5 (OK) | PASS |
| Balanced fusion (LPIPS↓, PSNR drop≤0.5dB, RMSE↑≤0.005, RAPSD not worse) | fusion_balanced conditions | CONFIRMED |
| Quality-lite fusion (LPIPS↓ with PSNR drop≤2.5dB, less distortion than full VQGAN) | fusion_quality_lite conditions | CONFIRMED |
| Selection used val only, scored on dev | val/dev firewall in canary | PASS |
| No retraining (frozen priors+refiners) | regen bit-matches frozen final_dev CSV | PASS |