# G1 Postmortem

## Classification

`mixed_protocol_drift_and_z_disabled_or_collapsed_budget_confound_possible_leakage_undeterminable`

The G1 pilot is not valid sampling-mode evidence. The checkpoint provenance matches the published Scr-5 mean-mode checkpoint at initialization, but the pilot then ran a gated optional generator update and did not save a final per-sample stochastic artifact set. The evaluation loop repeatedly calls deterministic reconstruction without an explicit z input. Split hashes for the main no-leak and pilot eval are not locatable, so leakage/protocol drift cannot be ruled out.

## Evidence Chain

- Published mean Scr-5 metrics: PSNR 22.2708, SSIM 0.6317, RelMeasErr 0.005453 from `E:\ns_mc_gan_gi\outputs_phase15\imported_noleak\scrambled_hadamard5_hq_noise001_colab\eval_metrics.json`.
- G1 pilot metrics: PSNR 23.3448, SSIM 0.6662, RelMeasErr 0.005365 from `E:\ns_mc_gan_gi\outputs_phase59_gan_sampling_mode_g1\g1_key_metric_table.csv`.
- G1 PSNR advantage: 1.0740 dB; kappa proxy 0.7809 < 1, outside [1, 2].
- Mean pixel std proxy: 0.00077753; null variance ratio proxy: 0.012526.
- Certificate invariance/reportable residual: G1 RelMeasErr 0.005365 vs mean 0.005453; relative difference 0.016032.
- Initialization checkpoint SHA match according to Phase60: `True`.
- Main split candidates: 0; pilot split candidates: 0.
- Stochastic z active in old eval code: `False`. Repeated deterministic reconstruct pattern found: `True`.
- Pilot command: `$ C:\Users\CYZ的computer\Documents\Codex\2026-06-04\files-mentioned-by-the-user-txt\ns_mc_gan_gi\src\phase53C_optional_gan_posterior.py --bundle_root E:/ns_mc_gan_gi/outputs_phase15/imported_noleak --output_dir E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import/session_24_optional_gan_and_posterior_sampling --session_name session_24_optional_gan_and_posterior_sampling --dataset_root E:/ns_mc_gan_gi/data --device cuda --tasks scr5 rad5 --limit_samples 256 --critic_epochs 8 --max_steps 180 --num_samples_per_y 8 --eval_batches 4`

## Training Budget vs Mean Mode

- Mean-mode config reports 80 epochs in `resolved_config.yaml`.
- G1 optional pilot command reports `--critic_epochs 8 --max_steps 180 --num_samples_per_y 8`.
- Because the final fine-tuned checkpoint and optimizer state were not saved as individual evidence, the budget confound cannot be fully separated from protocol drift.

## Leakage Question

No main no-leak train/val/test split hash and no pilot eval split hash were found. The file named `scrambled5_noleak_split_manifest.json` is a transfer chunk manifest, not a data split provenance file.
