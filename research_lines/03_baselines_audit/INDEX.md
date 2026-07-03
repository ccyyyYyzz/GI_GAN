# Stage 3 — Hadamard/Rademacher Baselines + Audit Certificate (Phases 9–17)

**Research question:** Can one pluggable, reconstructor-agnostic measurement audit be attached at test time to any reconstructor and drive RelMeasErr to near zero without touching perceptual quality?

**Theory.** For y = Ax + eps and any estimate v, the residual r = Av - y lives entirely in range(A). The correction v' = v - A^T(AA^T + λI)^{-1} r projects v back onto the hyperplane {u : Au = y} while leaving the null-space component unchanged. Applying this at test time constitutes an audit certificate: it compresses RelMeasErr sharply for backprojection, Tikhonov, CS-TV, and learned reconstructors alike, while PSNR is nearly unchanged because null-space content is untouched.

Structured low-frequency Hadamard and random Rademacher patterns serve as locked measurement families. Exact-measurement audit contracts are defined per measured singular mode.

---

## Core evaluation toolkit (`eval/`)

All files live at repo root; run from repo root.

| File | Role |
|---|---|
| `eval/audit.py` | `b_lambda`, `audit_correct`, `rel_measurement_error`, `save_audit_arrow_plot` — core test-time audit correction and scatter plots |
| `eval/checker.py` | PASS/FAIL gate checker for posterior-sampler result dumps (`.npz`/`.pt`/`.pth`); emits structured `GateResult` objects and JSON |
| `eval/metrics.py` | CPU-friendly image metrics (PSNR, SSIM, LPIPS, FID/KID) for 64×64 grayscale outputs |
| `eval/scr5_convention_bridge.py` | Convention bridge: builds Scr-5 operator via `src.train.make_measurement`, injects null-space fakes, runs checker — validates audit API against the live measurement operator |
| `eval/seed_variance.py` | Seed-variance aggregation across multiple result dumps |
| `eval/visualize.py` | Per-dump visualization |
| `eval/MIGRATION.md` | Five-line usage guide; result dump format spec |

Checker usage (from repo root):

```bash
python -m eval.checker results/seed0_dump.npz --json-out results/seed0_eval.json
python -m eval.visualize results/seed0_dump.npz --out-dir results/seed0_viz
python -m eval.seed_variance results/seed0_dump.npz results/seed1_dump.npz results/seed2_dump.npz --json-out results/seed_variance.json
```

---

## Phase-by-phase source modules (`src/`)

### Phase 9 — Hadamard/Rademacher probe sweep
Config root: `configs/phase9/` (eight configs: `hadamard5_probe_noise001.yaml`, `hadamard10_probe_noise001.yaml`, `rademacher5_hq_noise001.yaml`, `scrambled_hadamard5_hq_noise001.yaml`, etc.).

Key scripts:
- `src/aggregate_phase9.py` — aggregate outputs from `E:/ns_mc_gan_gi/outputs_phase9`; records `backproj_psnr`, `model_psnr`, `model_ssim`, `model_rel_meas_err`
- `src/make_phase9_report.py` — produce phase 9 report

### Phase 10 — Noise sweep
Config root: `configs/phase10/` (`hadamard5_full_noise001.yaml`, `hadamard10_full_noise001.yaml`, `rademacher5_hq_noise001.yaml`, `scrambled_hadamard5_hq_noise001.yaml`, `lowfreq_no_dc10_control.yaml`, etc.).

Key scripts: `src/eval_phase10_noise_sweep.py`, `src/aggregate_phase10.py`, `src/make_phase10_report.py`, `src/export_phase10_examples.py`.

### Phase 11 — Attribution + adaptive continue
Config root: `configs/phase11/`.

Key scripts: `src/phase11_common.py`, `src/phase11_attribution.py`, `src/phase11_noise_sweep.py`, `src/phase11_adaptive_continue.py`, `src/run_phase11_multiseed.py`, `src/prepare_phase11_multiseed.py`, `src/aggregate_phase11.py`, `src/make_phase11_report.py`, `src/export_phase11_paper_assets.py`.

### Phase 12 — Fashion-MNIST / CIFAR monitor
Key scripts: `src/phase12_common.py`, `src/phase12_monitor_fashion.py`, `src/make_phase12_report.py`.

### Phase 13 — Post-phase-14 update
Key script: `src/update_phase13_after_phase14.py`.

### Phase 14 — Full ablation pack + Colab import
Config roots: `configs/phase14/` and `configs/phase14_colab/` (Rademacher 5%/10% Colab configs).

Core method IDs imported from Colab:
- `stl10_rademacher5_colab_full` — STL-10 Rademacher 5%, GPU: Colab L4/A100/T4
- `stl10_scrambled5_colab_full` — STL-10 Scrambled Hadamard 5%, GPU: Colab L4/A100/T4

Key scripts: `src/phase14_common.py`, `src/phase14_unified_eval.py`, `src/phase14_ablation_pack_common.py`, `src/phase14_checkpoint_ablation.py`, `src/phase14_inference_ablation.py`, `src/phase14_traditional_baselines.py`, `src/phase14_lightweight_traditional_baselines.py`, `src/phase14_noise_sweep.py`, `src/phase14_dc_control_final.py`, `src/phase14_attribution_table.py`, `src/phase14_statistics.py`, `src/import_phase14_colab_results.py`, `src/phase14_colab_summary.py`, `src/aggregate_phase14.py`, `src/make_phase14_report.py`, `src/make_phase14_ablation_pack_report.py`.

### Phase 15 — Strict no-leak registry and exact-A re-evaluation
No-leak registry: `E:/ns_mc_gan_gi/outputs_phase15/noleak_registry.csv`

Key scripts:
- `src/phase15_build_noleak_registry.py` — build registry
- `src/phase15_noleak_audit.py` — audit each method: checks `resolved_config.yaml`, `eval_before_training`, `eval_every`, `save_every`, `last.pt`, `eval_metrics.json`; fields include `paper_safe`
- `src/phase15_exactA_reeval.py` — re-evaluate Rademacher exact-A tensor against Colab metrics
- `src/phase15_import_noleak_results.py` — import Colab outputs into registry
- `src/phase15_make_final_figures.py`, `src/phase15_make_final_tables.py`, `src/phase15_update_final_claims.py`, `src/phase15_update_manuscript_pack.py`
- `src/make_phase15_final_lock_report.py` — executive summary; recommends scrambled Hadamard, MNIST, Fashion-MNIST as cleanest main evidence; Rademacher treated as conditional/supplementary until exact-A mismatch is resolved
- `src/phase15r_dataset_split_audit.py` — SHA-256 hashes of first-10 images per split; verifies no train/test contamination
- `src/phase15r_common.py`, `src/phase15r_inventory.py`, `src/phase15r_backprojection_test.py`, `src/phase15r_checkpoint_inspect.py`, `src/phase15r_inspect_exactA.py`, `src/phase15r_eval_variants.py`, `src/phase15r_make_golden_bundle_script.py`, `src/phase15r_replay_golden_bundle.py`, `src/make_phase15r_repro_report.py`

Exact-A path (Rademacher, imported from Colab):
`E:/ns_mc_gan_gi/outputs_phase15/imported_noleak/rademacher5_hq_noise001_colab/measurement_operator_exact.pt`

### Phase 16 — Supplementary experiments
Outputs root: `E:/ns_mc_gan_gi/outputs_phase16/supplementary_experiments`

Main method set (from `src/phase16_common.py`):
- STL-10 Rademacher 5%/10%, Scrambled Hadamard 5%/10%
- MNIST Hadamard 5%, Fashion-MNIST Hadamard 5%

Key scripts:
- `src/phase16_exactA_reeval_audit.py` — reproduce Rademacher Colab metrics locally; tolerance PSNR ±0.02 dB / SSIM ±0.002; status `reproduced` or `mismatch`
- `src/phase16_statistics_ci.py` — bootstrap 95% CI (1000 iterations, 500 samples) for PSNR and SSIM per method
- `src/phase16_noise_sweep.py` — noise sweep supplementary
- `src/phase16_traditional_baselines.py` — backprojection (ridge/adjoint) and TV-PGD (λ ∈ {0.001, 0.003, 0.01}, 50 iterations) for all main methods
- `src/phase16_stl10_classwise.py` — STL-10 per-class metric breakdown
- `src/phase16_runtime_complexity.py` — runtime/complexity comparison
- `src/phase16_real_inference_ablation.py`, `src/phase16_dc_row_control_final.py`, `src/phase16_measurement_perturbation.py`
- `src/phase16_attribution_final.py`, `src/phase16_update_writing_claims.py`, `src/phase16_verify_safe_exactA.py`
- `src/aggregate_phase16_supplementary.py`, `src/make_phase16_supplementary_report.py`

### Phase 17 — Submission pack and evidence index
Key scripts:
- `src/phase17_build_evidence_index.py` — maps each claim (`C1_STL10_5pct_HQ`, etc.) to metric values, source CSV, `safe_to_claim` flag, and `caveat`; output to `outputs_phase17/evidence_index/`
- `src/phase17_final_checklist.py` — FINAL_PAPER_CHECKLIST.md: 14-item checklist including no-SOTA-claim, no-GAN-main-mechanism-claim, no-binary-learned-illumination-claim, exact-A citation for Rademacher, and code/data availability statement status
- `src/phase17_reviewer_risk_register.py` — risk register for reviewer challenges
- `src/phase17_submission_pack.py` — assemble submission package
- `src/phase17_make_figure_table_pack.py`, `src/phase17_make_pdf_preview.py`
- `src/phase17_defense_pack.py`, `src/phase17_write_manuscript.py`, `src/phase17_write_supplement.py`, `src/phase17_write_chinese_report.py`
- `src/make_phase17_manifest.py` — manifest of all phase 17 outputs

---

## Publication baseline configs

Relocated from the main config tree and described in `research_lines/03_baselines_audit/PUB_BASELINES_CONFIG_REPORT.md`.

Config pairs (Windows + Colab): `configs/pub_baselines/{unet,resunet,unrolled_ista}_{rad5,scr5}_pub.yaml` and `configs/pub_baselines/colab/` equivalents. All configs set `use_adversarial: false`, `use_null_project: true`, `use_dc_project: true`, `use_final_dc_project: true`, noise_std 0.01, lambda_solver 0.001, 50k train / 2k val samples, 80 epochs.

Support scripts: `scripts/render_pub_baseline_colab_configs.py`, `scripts/validate_pub_baseline_configs.py`.

---

## Architecture ablation configs (phases 25–26)

Template of record for publication baselines: `configs/phase25_arch_ablation/` (10 configs: `current_hq_{rad5,scr5}.yaml`, `nafnet_small_{rad5,scr5}.yaml`, `resunet_{rad5,scr5}.yaml`, `unet_{rad5,scr5}.yaml`, `unrolled_ista_{rad5,scr5}.yaml`). Pilot variants: `configs/phase26_arch_pilot/` (manifest at `configs/phase26_arch_pilot/arch_pilot_config_manifest.{csv,json,md}`).

---

## Core certificate evidence

**Claim: measurement consistency is necessary but not sufficient for correctness.**

The audit drives RelMeasErr to near zero for any reconstructor (including backprojection, Tikhonov, CS-TV, and learned generators) while leaving PSNR essentially unchanged. This is because `eval/audit.py:audit_correct` only adjusts the range(A^T) component; the null-space projection P_0 x remains unchanged. Cross-class feasible wrong images (the boundary evidence) are documented in stage 5 (`research_lines/05_range_null_barrier/`), not here.

**What this stage proves:** Auditable accountability (A x_hat ≈ y, RelMeasErr ≪ 1) is achievable by any reconstructor via a pluggable test-time correction. It does NOT certify null-space correctness.

---

## Reproduction

All commands from repo root. Env: `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311` (Python 3.11).

```bash
# Run checker on a result dump
python -m eval.checker results/seed0_dump.npz --json-out results/seed0_eval.json

# Phase 9 aggregate
python -m src.aggregate_phase9

# Phase 16 statistics CI (bootstrap, 500 samples per method, 1000 resamples)
python -m src.phase16_statistics_ci

# Phase 16 traditional baselines (backprojection + TV-PGD)
python -m src.phase16_traditional_baselines

# Phase 17 evidence index
python -m src.phase17_build_evidence_index

# Phase 17 final checklist
python -m src.phase17_final_checklist
```

Rademacher exact-A outputs live under `E:/ns_mc_gan_gi/outputs_phase15/imported_noleak/` (Colab-imported; local exact-A re-evaluation tolerance ±0.02 dB PSNR / ±0.002 SSIM; status may be `mismatch` — see `phase15_final_lock_report.md` caution). Use scrambled Hadamard, MNIST, and Fashion-MNIST as the cleanest reproducible main evidence locally.

---

## Cross-references

- Stage 1 (core platform, `src/measurement.py`, `src/projections.py`): `research_lines/01_core_platform/INDEX.md`
- Stage 5 (feasible wrong images, consistency vs correctness): `research_lines/05_range_null_barrier/INDEX.md`
- Stage 6 (GAN auditable case study, Scr-5/Rad-5 gauge diagnostics): `research_lines/06_gauge_gan_rad5/INDEX.md`
- Publication baseline config detail: `research_lines/03_baselines_audit/PUB_BASELINES_CONFIG_REPORT.md`
