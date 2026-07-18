# Experiment index

This index maps scientific claims to the executable entry point, evidence, and reproducibility class. The longer chronological explanation remains in `HANDOFF/01_RESEARCH_STORY.md` and `HANDOFF/05_EXPERIMENTS_AND_EVIDENCE.md`.

| ID | Question | Code / command | Result or receipt | Class |
|---|---|---|---|---|
| S1 | Does the range/null projector satisfy the exact identities? | `python -m pytest tests/test_exact_projections.py -q` | projector tests | RERUNNABLE |
| S3 | Can a post-hoc audit reduce measurement residual without changing quality? | `eval/audit.py`, `src/eval.py` | `docs/core_experiments/canonical_numbers.csv` | RERUNNABLE / external tables |
| S5 | Can a wrong cross-class image share the same record? | `results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py` | `T4_pairs.csv`, `CERTIFICATE_INVARIANCE_RECHECK.json` | EXTERNAL_WAREHOUSE |
| S6 | Does gauge equalization remove a measurement-row shortcut? | `src/phase69A_*`, `src/phase71_*`, `src/phase75_*`, `inspect_gate.py` | Gauge-AUC and shortcut-stress receipts | EXTERNAL_WAREHOUSE |
| S7 | Does latent sampling produce posterior diversity? | `src/phase79_rad5_rowspace_diversity_diagnostic.py` | `results/g2r_pilot_phase3/PHASE3_REPORT.md` | NEGATIVE/DORMANT |
| S8 | Does trained VQAE/VQGAN null-space fusion improve perceptual metrics? | `python vqgan_detail_fusion.py validate --seeds 0 1 2 --device cuda` | `outputs/.../detail_fusion_paper/` | LOCKED |
| S8-FCC | Is the FCC signal beyond scalar/artifact controls? | `python fcc_diagnostic_canary.py all --config configs/compatibility/fcc_diagnostic_canary64.yaml` | `outputs/.../fcc_diagnostic_canary64/reports/FINAL_REPORT.md` | RERUNNABLE |
| S-CASSI | Does the range/null ledger transfer to a real released CASSI operator? | `cassi_twoledger/` scripts and tests | `cassi_twoledger/RESULTS.md` | RERUNNABLE with public data |
| S-MRI | Does the ledger transfer to measured fastMRI k-space? | `fastmri_twoledger/` scripts and tests | `fastmri_twoledger/RESULTS.md` | EXTERNAL_DATA |

## Metric and split rules

- `RelMeasErr` is computed on the unclipped float64 reconstruction.
- Display clipping is allowed only for PSNR/visualization and must be labeled.
- Stage 8 uses the fixed operator seed `772001`, 64×64 grayscale STL10, 5% sampling, and raw-hash-clean splits.
- The locked VQGAN split is 512 images and is disjoint from the previously consumed raw hashes.
- Do not run `vqgan_detail_fusion_locked.py` again; use its committed receipt or the non-destructive `validate` command.
- Do not use the test split for training or selection.

