# Sampling-Mode GAN Track Dossier

This dossier prepares the exploratory GAN sampling-mode track up to the safety gate. It does not modify the main pipeline, does not overwrite results, and does not launch full G2 training.

## Status By Task

- S-1 identity gate: PASS; results in `checks/identity_gate_results.json`.
- S0 G1 forensic post-mortem: completed; see `G1_POSTMORTEM.md`.
- S1 infrastructure repair: utility modules generated (`eval_sampling.py`, `sampling_metrics.py`, `tools/split_hash.py`); no main code edited.
- S2 G2 preflight: config prepared; smoke not run because provenance is unsafe and stochastic branch needs review.
- S3 launch dossier: completed with `READY TO LAUNCH: no`.

## Perceptual Metrics Availability

`{"fid_available": false, "kid_available": false, "lpips_available": false, "packages": {"cleanfid": false, "lpips": false, "torchmetrics": false}, "requirement_note": "LPIPS requires the lpips package and local backbone weights; FID/KID require clean-fid or torchmetrics with local feature weights/cache."}`

## Evidence Files

- Phase60 safety gate: `E:\ns_mc_gan_gi\outputs_phase60_gan_sampling_mode_g2\g2_safety_status.json`.
- Phase60 provenance: `E:\ns_mc_gan_gi\outputs_phase60_gan_sampling_mode_g2\phase60_provenance_status.json`.
- G1 key table: `E:\ns_mc_gan_gi\outputs_phase59_gan_sampling_mode_g1\g1_key_metric_table.csv`.
- Pilot command log: `E:\ns_mc_gan_gi\outputs_phase53C_exact_null_critic_import\session_24_optional_gan_and_posterior_sampling\command_log.txt`.
- Old pilot source: `C:\Users\CYZ的computer\Documents\Codex\2026-06-04\files-mentioned-by-the-user-txt\ns_mc_gan_gi\src\phase53C_optional_gan_posterior.py`.

## What I could not determine and why

- I could not confirm that the pilot used the same test indices as the main no-leak split because no saved data split hash/index files were found.
- I could not separate leakage from protocol drift because both main and pilot split hashes are missing.
- I could not prove active stochastic sampling in G1 because the old eval loop has no explicit z path and individual stochastic samples were not saved.
- I could not measure smoke wall-clock/memory because running G2 smoke would proceed past an unsafe provenance gate.

|G1 anomaly cause|certificate-invariance result|infra status|smoke-test verdict|READY-TO-LAUNCH flag|
|---|---|---|---|---|
|mixed_protocol_drift_and_z_disabled_or_collapsed_budget_confound_possible_leakage_undeterminable|RelMeasErr 0.005365 vs 0.005453; relative diff 0.016032|utilities generated; split hashes not locatable; per-sample saving module ready but untested on G2|skipped_unsafe_provenance|no|

## Follow-Up Gap Items

Detailed follow-up findings are appended in `FOLLOWUP_GAP_ITEMS.md`.

Summary:

- Main pipeline dataloaders use canonical `torchvision.datasets.STL10` partitions: train defaults to `split="train+unlabeled"` and eval defaults to `split="test"`. Subsampling uses `_limit_dataset(..., seed)` for train and `_limit_dataset(..., seed + 1)` for val/test. Code-level train/test disjointness is guaranteed by the canonical STL-10 partition, though exported split hashes are still missing.
- G1 pilot loaders use `get_val_dataloader`; the optional GAN update loop and later pilot eval both call `make_loader(config, device)`. Since `get_val_dataloader` defaults to `val_split="test"`, the G1 optional GAN pilot has a direct protocol-drift/test-set-adaptation risk.
- Certificate recheck loaded the published mean checkpoint and G1 `scr5/source_checkpoint.pt`; both used `generator_ema` and produced identical RelMeasErr on 256 STL-10 test samples: `0.005484469700604677`. The post-GAN pilot checkpoint was not found, so the old G1 pilot RelMeasErr `0.005365385441109538` remains only an aggregate CSV value, not recomputable from a saved checkpoint.
- G2_READY blockers verbatim: `['No saved main no-leak train/val/test split hashes are available.', 'Pilot split/eval index hashes are not available.', 'Old G1 code path appears deterministic with no explicit stochastic z.', 'Controlled G2 smoke was not run because provenance is unsafe and stochastic branch implementation has not been reviewed.']`
- The <=200-iteration smoke test was skipped because the provenance safety precondition failed: no saved main no-leak train/val/test split hashes were available. The new G1 loader finding reinforces the skip.
- Infrastructure files were created and unit-tested: `eval_sampling.py`, `tools/split_hash.py`, and `sampling_metrics.py`. Unit-test status: `pass`, recorded in `INFRA_UNIT_TEST_RESULTS.json`.
