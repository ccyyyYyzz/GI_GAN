# 03_CODE_MAP — Importable Core: File-by-File Map

**Repo root:** `E:/ns_mc_gan_gi_code_fcc_phase1`
**Runtime env:** Python 3.11 at `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311`
**Import model:** ALL scripts are run from repo root; `from src.X import ...` and `from eval.X import ...` require cwd = repo root. NO files were physically moved.

---

## 0. Layout Principle

The repo uses flat cwd-root imports throughout:

```
import gan_high_quality_gi as hq          # root-level module
from src.measurement import ...           # src/ package
from eval.audit import ...                # eval/ package
```

Code is **never** moved into subdirectories to satisfy import paths. The `HANDOFF/research_lines/*/INDEX.md` files point to files in their original locations. Every path below is relative to repo root.

---

## 1. Core Platform Library — `src/`

### 1a. Entrypoints (run directly or via scripts)

| File | Role | Key imports FROM this file |
|---|---|---|
| `src/train.py` | Main training loop: GAN + data-consistency + measurement | `.datasets`, `.exact_measurement`, `.losses`, `.measurement`, `.models`, `.metrics`, `.utils`, `.dc_balanced`, `.projections` |
| `src/eval.py` | Evaluation loop: load checkpoint, score all metrics | `.datasets`, `.exact_measurement`, `.measurement`, `.metrics`, `.models`, `.pattern_diagnostics`, `.pattern_regularizers`, `.pattern_utils`, `.utils` |

### 1b. Library Modules (imported by others, not run directly)

**Physics / measurement:**

| File | Role |
|---|---|
| `src/measurement.py` | `GhostMeasurementOperator`, `LearnableGhostMeasurementOperator`, `LearnablePatternBank`; Hadamard/Rademacher matrix construction; forward `y = A x`, LMMSE solver `A^T(AA^T+λI)^{-1}y` |
| `src/exact_measurement.py` | `apply_measurement_override_from_config`; loads a frozen float32 A matrix from artifact; used by both train.py and eval.py |
| `src/projections.py` | `get_exact_projector`, `relative_measurement_error`, exact null-space `P0` and row-space `P_R` projectors (float64 SVD, hash-verified) |
| `src/dc_balanced.py` | `dc_row`, `dct_lowfreq_non_dc_rows`, `hadamard_lowsequency_non_dc_rows`, `random_zero_mean_rows`; operator-row construction helpers |

**Models:**

| File | Role |
|---|---|
| `src/models.py` | `build_generator` (U-Net-style), `PatchDiscriminator`; shared by train.py, eval.py, and all root-level GAN modules |

**Data:**

| File | Role |
|---|---|
| `src/datasets.py` | `get_dataloaders`, `get_val_dataloader`, `build_transform`; STL10/FashionMNIST/MNIST; hash-clean split management |
| `src/split_guard.py` | `SplitViolationError`; runtime guard that raises if any training dataloader draws from the test split |

**Losses:**

| File | Role |
|---|---|
| `src/losses.py` | `reconstruction_loss`, `data_consistency_loss`, `generator_adversarial_loss`, `discriminator_wgan_loss`, `gradient_penalty`, `frequency_loss`, `charbonnier_loss`, `sobel_edge_loss`, `gradient_difference_loss`, `total_variation_loss`, `multiscale_ssim_loss`, `differentiable_ssim_loss`, `simple_multiscale_l1` |

**Metrics:**

| File | Role |
|---|---|
| `src/metrics.py` | `psnr`, `ssim`, `batch_metrics`; CPU-friendly, used by src/ and root-level modules |

**Utilities:**

| File | Role |
|---|---|
| `src/utils.py` | `load_config`, `apply_experiment_defaults`, `ensure_dir`, `compare_metric_sets`, `format_metric_comparison`, `set_seed` |
| `src/checkpoint_utils.py` | Checkpoint save/load helpers; used by phase runners |
| `src/overnight_runner.py` | Generic overnight sweep coordinator |

**Pattern learning (phases 3–8):**

| File | Role |
|---|---|
| `src/pattern_utils.py` | `save_pattern_grid`, `save_pattern_stats_json` |
| `src/pattern_regularizers.py` | `secant_rip_loss` and related pattern regularizers |
| `src/pattern_diagnostics.py` | `compare_pattern_states`, `save_pattern_change_visualization`, `save_pattern_diagnostics_json` |
| `src/measurement_quality.py` | Pattern-quality metrics: coherence, RIP estimates |

**Compatibility / FCC / posterior-sampling support:**

| File | Role |
|---|---|
| `src/compatibility_data.py` | `SplitComponents`, `exact_data_anchor`, `exact_null_project`, `exact_row_project`; null/row decomposition for range-null FCC experiments |
| `src/compatibility_model.py` | `LightImageEncoder`, symmetric-InfoNCE critic for FCC compatibility experiments |
| `src/compatibility_eval.py` | ROC-AUC eval; imports from `.compatibility_data` |
| `src/fcc_canary.py` | Low-level FCC helper: deployable nuisance controls, mechanical classification logic; imported by root-level `fcc_diagnostic_canary.py` |
| `src/bayesian_witness.py` | `BayesianWitnessError`, Bayesian witness assimilation logic; imported by root-level `bayesian_witness_assimilation.py` |
| `src/operator_conditioned_nullspace.py` | `MatrixFreeNullProjector`, `OperatorConditionedNullspaceError`; operator-conditioned null-space network components |

**G2R posterior sampling (Stage 7, dormant):**

| File | Role |
|---|---|
| `src/g2r_modec.py` | Mode-C sampler: `x_hat(z) = x_star + P0 @ G_theta(z, x_data)`; null-space-only generator head; discriminator hard rules |
| `src/g2r_modec_train.py` | G2R training entrypoint; imports `g2r_modec` |

### 1c. Phase Script Modules in `src/`

These are numbered modules that run experiments, produce results, or assemble reports. They are **imported within the package** by later phase modules that aggregate them. Listed by research line:

**Stages 3–8 (operator/pattern exploration):**
`src/phase1_1_controls.py`, `src/phase1_4b_scoring.py`, `src/phase1_4ir_uid_safe_scoring.py`, `src/phase2_fresh_operator.py`, `src/phase2_locked_protocol.py`, `src/phase2_witness.py`, and aggregate/report scripts `src/aggregate_phase{3..8}.py`, `src/make_phase{2..8}_report.py`, `src/export_paper_assets.py`, `src/export_hq_examples.py`

**Stages 9–17 (Hadamard/Rademacher baselines + audit certificate):**
`src/phase11_common.py`, `src/phase11_adaptive_continue.py`, `src/phase11_attribution.py`, `src/phase11_noise_sweep.py`, `src/phase12_common.py`, `src/phase14_common.py`, `src/phase14_unified_eval.py`, `src/phase14_statistics.py`, `src/phase14_ablation_pack_common.py`, `src/phase15_common.py`, `src/phase15_build_noleak_registry.py`, `src/phase15_noleak_audit.py`, `src/phase16_common.py`, `src/phase16_statistics_ci.py`, `src/phase17_common.py`, and associated aggregate/report/export scripts.

**Stages 18–45 (manuscript / mechanism construction):**
`src/phase18_rewrite_common.py`, `src/phase18b_common.py`, `src/phase19_common.py`, `src/phase20_common.py`, `src/phase25_pca_oracle.py`, `src/phase26_common.py`, and paired make-figures / check / rewrite scripts through phase 45. Also: `src/make_paper_figures.py`, `src/make_paper_tables.py`, `src/build_final_result_registry.py`.

**Stages 48–60 (range-null boundary / feasible-wrong barrier):**
`src/phase48_49_common.py`, `src/phase48_49_mechanistic_probes.py`, `src/phase48_49_train_ablation.py`, `src/phase51A_*.py`, `src/phase53B_*.py`, `src/phase53C_common.py`, `src/phase53C_exact_projector.py` (builds rowspace basis via float-64 SVD), `src/phase53C_exact_null_critic_evaluator.py`, `src/phase53C_feasible_hallucination_figure.py`, `src/phase53D_*.py`, `src/phase55_*.py`, `src/phase56_*.py`, `src/phase59_g1_sampling_mode_eval.py`, `src/phase60_*.py`.

**Stage 6 (Gauge-GAN / Rad-5 case study) — CURRENT MAIN LINE:**

| File | Role | Key imports |
|---|---|---|
| `src/phase69A_gauge_gan_signal_diagnostic.py` | Diagnostic: Scr-5/Rad-5 gauge AUC; imports `.datasets`, `.models`, `.utils` | — |
| `src/phase69B_compute_lpips.py` | LPIPS computation for phase-69 pilots | — |
| `src/phase69B_controlled_gauge_cgan_pilot.py` | Controlled cGAN pilot | `.phase69A_gauge_gan_signal_diagnostic` |
| `src/phase70_gauge_gan_paper_expansion.py` | Expanded gauge-GAN sweep | — |
| `src/phase71_gauge_cgan_paired_seeds.py` | Paired-seed cGAN eval | `.phase69A_gauge_gan_signal_diagnostic` |
| `src/phase72_scr10_gauge_cgan_regime_validation.py` | Scr-10 regime validation | — |
| `src/phase73_overnight_gauge_gan_expansion.py` | Overnight expansion sweep | — |
| `src/phase74_high_tier_gauge_cgan_pack.py` | High-tier pack | — |
| `src/phase75_final_high_tier_validation.py` | Final high-tier validation | — |
| `src/phase76_high_upside_auditable_gan_exploration.py` | Upside exploration | — |
| `src/phase77_auditable_gan_paper_assembly.py` | Paper assembly (dev) | — |
| `src/phase77_final_auditable_gan_paper_assembly.py` | **Canonical paper assembly; reads canonical_results_table.csv** | — |

**Stage 7 (G2R posterior sampling, dormant):**

| File | Role |
|---|---|
| `src/phase78_96px_rad5_one_seed_probe.py` | 96px Rad-5 probe |
| `src/phase79_96px_rad5_p0_error_validation.py` | P0 error validation |
| `src/phase79_rad5_rowspace_diversity_diagnostic.py` | Row-space diversity (z_collapsed, negative result) |
| `src/phase80_rad5_centered_diversity_calibration.py` | Diversity calibration |
| `src/phase81_96px_rad5_paper_completion.py` | 96px completion |

These import `src/phase69A_gauge_gan_signal_diagnostic`, `src/phase69B_controlled_gauge_cgan_pilot`, `src/phase73_overnight_gauge_gan_expansion`, `.models`, `.utils`.

---

## 2. Eval Package — `eval/`

All files are library modules (imported by root-level scripts and by `src/` phase modules). None are entrypoints.

| File | Role | Imports |
|---|---|---|
| `eval/metrics.py` | `psnr`, `ssim`, `fid_kid`, `lpips_distance`, `flatten_images`, `to_nhw`; CPU-friendly, minimal deps | numpy only |
| `eval/checker.py` | `check_results`, `load_array_artifact`; PASS/FAIL gate for posterior-sampler dumps | `.metrics` |
| `eval/audit.py` | `audit_correct`, `rel_measurement_error`, `b_lambda`; test-time B_λ audit correction, scatter plots | `.checker`, `.metrics` |
| `eval/seed_variance.py` | `_flatten_numeric`; seed-variance aggregation across result dumps | `.checker` |
| `eval/scr5_convention_bridge.py` | Convention bridge: builds Scr-5 operator via `src.train.make_measurement`, injects null-space samples, runs `eval.checker` | `src.train`, `eval.checker` |
| `eval/__init__.py` | Empty package marker | — |
| `eval/tests/test_checker_synthetic.py` | Pytest: synthetic data checker tests | `eval.checker` |
| `eval/tests/test_metrics_slow.py` | Pytest: slow metric tests | `eval.metrics` |

---

## 3. Root-Level GAN Modules (Stage 6 — Gauge-GAN case study)

All four files stay at repo root; they are imported by name without a package prefix.

| File | Type | Key imports |
|---|---|---|
| `gan_high_quality_gi.py` | **Core library + entrypoint** | `src.dc_balanced`, `src.losses`, `src.measurement`, `src.metrics`, `src.models` |
| `gan_high_quality_gi_matched.py` | Library + entrypoint (matched-operator variant) | `gan_high_quality_gi as hq` |
| `gan_gauge_aligned_nsgan.py` | Library + entrypoint (gauge-aligned NS-GAN) | `gan_high_quality_gi as hq`, `gan_high_quality_gi_matched as matched` |
| `inspect_gate.py` | Utility script: prints gate JSON from outputs dir | stdlib only |

**Import chain:**
```
gan_gauge_aligned_nsgan → gan_high_quality_gi, gan_high_quality_gi_matched
gan_high_quality_gi_matched → gan_high_quality_gi
```

---

## 4. Root-Level VQGAN / FCC Modules (Stage 8 — VQGAN/FCC compatibility subline)

All stay at repo root; imported by name.

| File | Type | Key imports |
|---|---|---|
| `measurement_conditioned_vqgan.py` | Core VQGAN library (MC-VQGAN prior) | `gan_high_quality_gi as hq`, `gan_gauge_aligned_nsgan as ga`, `src.losses`, `src.metrics`, `src.projections` |
| `anchor_initialized_vqgan_inversion.py` | Anchor-initialized VQGAN inversion refiner | `gan_high_quality_gi as hq`, `gan_gauge_aligned_nsgan as ga`, `measurement_conditioned_vqgan as mc`, `src.losses`, `src.metrics`, `src.projections` |
| `mc_vqgan_prior_long_canary.py` | Long canary run for MC-VQGAN prior | `gan_high_quality_gi as hq`, `gan_gauge_aligned_nsgan as ga`, `measurement_conditioned_vqgan as mc` |
| `vqgan_detail_fusion.py` | **Main fusion entrypoint** (zero-training null-space fusion); subcommands: regen/validate/canary/gate/all | `gan_high_quality_gi as hq`, `anchor_initialized_vqgan_inversion as ai`, `measurement_conditioned_vqgan as mc` (via internal calls) |
| `vqgan_detail_fusion_locked.py` | **LOCKED confirmatory test** (one-shot, frozen B scalars) | `gan_high_quality_gi as hq` and fusion machinery |
| `vqgan_detail_fusion_locked_figs.py` | Figure generation for locked result | `vqgan_detail_fusion as vdf` |
| `fcc_diagnostic_canary.py` | **FCC row-null diagnostic** (subcommands: build/train/eval/classify/all) | `src.fcc_canary`, `src.projections`, `src.measurement`, `src.datasets`, `src.compatibility_model` |
| `structure_detail_fcc.py` | Structure-detail FCC variant on fusion data | `gan_high_quality_gi as hq`, `gan_gauge_aligned_nsgan as ga`, `src.losses`, `src.metrics`, `src.projections` |
| `experiments_rate_fusion.py` | Cross-rate generalization analysis (2%, 10%) | `gan_high_quality_gi as hq`, `anchor_initialized_vqgan_inversion as ai`, `vqgan_detail_fusion as vdf` |
| `experiments_local.py` | Local experiment runner (convenience wrapper) | root-level modules |
| `paper_assembly.py` | Paper assembly from locked results | `vqgan_detail_fusion as vdf`, `anchor_initialized_vqgan_inversion as ai` |
| `core_mechanism_figure.py` | Mechanism figure generator (matplotlib, no-retrain) | stdlib + matplotlib only |
| `method_diagram_3d.py` | Pseudo-3D geometry figure | stdlib + matplotlib only |
| `paper_figures.py` | Additional paper figure helpers | — |

**VQGAN/FCC import chain:**
```
experiments_rate_fusion → gan_high_quality_gi, anchor_initialized_vqgan_inversion, vqgan_detail_fusion
paper_assembly          → vqgan_detail_fusion, anchor_initialized_vqgan_inversion
anchor_initialized_vqgan_inversion → gan_high_quality_gi, gan_gauge_aligned_nsgan, measurement_conditioned_vqgan
measurement_conditioned_vqgan      → gan_high_quality_gi, gan_gauge_aligned_nsgan
vqgan_detail_fusion                → (internal: reuses frozen prior/refiner machinery)
fcc_diagnostic_canary              → src.fcc_canary, src.projections, src.compatibility_model
structure_detail_fcc               → gan_high_quality_gi, gan_gauge_aligned_nsgan, src.projections
```

---

## 5. Other Root-Level Entrypoints

| File | Role |
|---|---|
| `train_compatibility.py` | Compatibility experiment train loop (range-null critics) |
| `eval_compatibility.py` | Compatibility experiment eval loop |
| `eval_candidate_selection.py` | Candidate selection for locked evaluation |
| `bayesian_witness_assimilation.py` | Bayesian witness run; imports `src.bayesian_witness` |
| `bayesian_witness_fixed_total.py` | Fixed-total-budget variant |
| `dc_balanced_fixed_total.py` | DC-balanced operator construction run |
| `nonlinear_operator_transfer.py` | Nonlinear operator transfer exploration |
| `nonlinear_headroom.py` | Nonlinear headroom canary |
| `operator_conditioned_budget_predictor.py` | Budget-predictor variant; imports `src.operator_conditioned_nullspace` |
| `operator_conditioned_nullspace_canary.py` | Null-space canary; imports `src.operator_conditioned_nullspace` |
| `phase1_1_corrected_pipeline.py`, `phase1_2_rad5_64_pipeline.py`, `phase1_3_freeze_and_audit.py`, `phase1_3r_recovery_and_relock.py`, `phase1_4a_freeze_and_blind.py`, `phase1_4ir_incident_recovery.py`, `phase1_4v4a_blind_inference.py` | Root-level run coordination scripts for compatibility phases 1–4 |
| `phase2_fresh_operator_smoke.py`, `phase2_locked_test_preflight.py`, `phase2_locked_test_score_once.py`, `phase2_witness_pilot.py` | Phase-2 smoke/preflight/score runs |
| `run_phase1_4a_blind_final_inference.py`, `score_phase1_4b_final_once.py`, `score_phase1_4b_final_once_v2.py`, `score_phase1_4v4_final_once.py`, `score_phase1_4v4_final_once_v2.py` | Locked-score one-shot runners |
| `make_colab_detach.py` | Generates Colab detach scripts under `colab/` |
| `make_rate_configs.py` | Generates cross-rate YAML configs under `configs/compatibility/` |
| `inspect_seed0_outputs.py` | Utility: print outputs for seed 0 |
| `local_env_check.py` | Environment sanity check |
| `md_to_pdf.py` | Markdown to PDF converter |

---

## 6. Paper Assets — `paper/`

| File | Role |
|---|---|
| `paper/main.tex` | Main IEEE-TCI manuscript (LaTeX); imports macros for P_R, P_0, RelMeasErr, B_λ |
| `paper/figures/make_paper_figures.py` | Top-level figure generator: imports `paper/figures/make_figure1_feasible_geometry.py` etc. |
| `paper/figures/make_figure1_feasible_geometry.py` | Figure 1: feasibility geometry |
| `paper/figures/make_feasible_wrong_gallery.py` | Feasible-wrong-image gallery figure |
| `paper/figures/make_feasible_wrong_candidate_pool.py` | Candidate pool builder |
| `paper/figures/select_feasible_wrong_images.py` | Image selector for T4 pairs |

---

## 7. Configs — `configs/`

No Python logic; pure YAML consumed by the entrypoints above.

| Directory / File(s) | Contents |
|---|---|
| `configs/default.yaml`, `configs/debug.yaml` | Core platform defaults |
| `configs/quick_train_*.yaml`, `configs/clean_quick_*.yaml` | Short smoke/sweep configs for Stages 1–2 |
| `configs/ablation_5pct_*.yaml`, `configs/clean_ablation_5pct_*.yaml` | Ablation configs |
| `configs/phase3_*.yaml` — `configs/phase8_*.yaml` | Operator/pattern learning sweeps (Stages 3–8) |
| `configs/phase8_hq/`, `configs/phase9/`, `configs/phase10/`, `configs/phase11/` | Hadamard/Rademacher HQ and audit sweeps (Stage 3) |
| `configs/phase14/`, `configs/phase14_colab/` | Phase-14 STL10 unified eval |
| `configs/phase25_arch_ablation/`, `configs/phase26_arch_pilot/` | Architecture ablation |
| `configs/phase48_49/`, `configs/phase51A/`, `configs/phase53B/`, `configs/phase53C/` | Range-null barrier configs (Stage 5) |
| `configs/pub_baselines/` | Traditional baselines (BP, Tikhonov, CS-TV) |
| `configs/colab/` | Colab-specific versions of core configs |
| `configs/g2r/` | G2R Mode-C posterior sampling configs (Stage 7, dormant); includes `ROUND2_AMENDMENT.md` |
| `configs/compatibility/` | All compatibility / VQGAN / FCC configs; includes `fcc_diagnostic_canary64.yaml`, `structure_detail_fcc.yaml`, `anchor_vqgan_inversion_rate{02,10}_seed{0,1,2}*.yaml`, `mc_vqgan_prior_multiseed_hashclean_seed{0,1,2}*.yaml`, `gan_high_quality_gi_locked_64_5pct.yaml`, `ga_nsgan_*.yaml` |

---

## 8. Scripts — `scripts/`

Wrapper scripts that call the entrypoints above. No importable logic.

| Group | Files |
|---|---|
| Core (Stages 1–2) | `train_quick_5pct.ps1`, `eval_quick_5pct.ps1`, `train_ablation_5pct.ps1`, `eval_ablation_5pct.ps1`, `sanity_physics.ps1`, `verify_env.ps1`, `train_5pct.sh`, `eval_5pct.sh` |
| Phases 3–11 | `train_phase{3..7}_*.ps1`, `eval_phase{3..7}_*.ps1`, `aggregate_phase{3..11}.ps1`, `phase{9..11}_*.ps1`, `phase8_hq_*.ps1` |
| Phases 12–17 | `phase12_*.ps1`, `phase14_*.ps1` |
| Colab infrastructure | `colab_setup.sh`, `colab_run_task.sh`, `colab_pack_outputs.sh`, `colab_setup_local.sh`, `colab_run_local_task.sh`, `colab_pack_local_outputs.sh`, `run_colab_smoke_test.sh`, `run_colab_repo_sanity_test.sh` |
| Gauge-GAN (Stage 6) | `phase69A_gauge_gan_signal_diagnostic.ps1` through `phase77_final_auditable_gan_paper_assembly.ps1` |
| VQGAN/FCC (Stage 8) | `scripts/aggregate_vqgan_multiseed_pareto.py`, `scripts/run_vqgan_multiseed_local.py` |
| Utilities | `validate_colab_runner.py`, `validate_pub_baseline_configs.py`, `preflight_pub_baseline_run.py`, `render_pub_baseline_colab_configs.py` |

---

## 9. Colab — `colab/`

Scripts that run in Google Colab (GPU) and write outputs to Drive. Not imported locally.

| File | Role |
|---|---|
| `colab/vqgan_multiseed_colab_job_common.py` | Shared job bootstrap: unzips repo bundle, installs deps, runs subcommand |
| `colab/vqgan_multiseed_colab_job_seed{0,1,2}.py` | Per-seed Colab jobs for 5% rate VQGAN multi-seed |
| `colab/vqgan_rate02_seed{0,1,2}_job.py`, `colab/vqgan_rate02_seed{0,1,2}_detach.py` | 2%-rate per-seed Colab jobs and detached variants |
| `colab/vqgan_rate10_seed{0,1,2}_job.py`, `colab/vqgan_rate10_seed{0,1,2}_detach.py` | 10%-rate per-seed jobs |
| `colab/vqgan_rate02_smoke_seed{0,1,2}_job.py`, `colab/vqgan_rate10_smoke_seed{0,1,2}_job.py` | Smoke variants |
| `colab/run_gan_hq_from_zip.py` | Runs `gan_high_quality_gi.py` from a zipped bundle |
| `colab/status_probe.py`, `colab/_drive_check.py` | Drive/status health checks |
| `colab/check_remote_repo.py`, `colab/clear_remote_repo.py` | Remote repo inspection utilities |
| `colab/phase14_colab_5pct_hq.ipynb`, `colab/pub_baselines_colab_runner.ipynb` | Colab notebooks for phase-14 and published baselines |

---

## 10. Import Dependency Summary

```
# Root-level GAN chain
gan_high_quality_gi
  └── src.{dc_balanced, losses, measurement, metrics, models}
gan_high_quality_gi_matched
  └── gan_high_quality_gi
gan_gauge_aligned_nsgan
  └── gan_high_quality_gi, gan_high_quality_gi_matched

# VQGAN chain
measurement_conditioned_vqgan
  └── gan_high_quality_gi, gan_gauge_aligned_nsgan, src.{losses, metrics, projections}
anchor_initialized_vqgan_inversion
  └── gan_high_quality_gi, gan_gauge_aligned_nsgan, measurement_conditioned_vqgan,
      src.{losses, metrics, projections}
vqgan_detail_fusion
  └── (self-contained machinery reusing frozen priors; no live import of mc_vqgan at run time,
       references paths to artifacts built by measurement_conditioned_vqgan)
vqgan_detail_fusion_locked → (same pattern as vqgan_detail_fusion)
experiments_rate_fusion
  └── gan_high_quality_gi, anchor_initialized_vqgan_inversion, vqgan_detail_fusion
paper_assembly
  └── vqgan_detail_fusion, anchor_initialized_vqgan_inversion

# FCC chain
fcc_diagnostic_canary
  └── src.{fcc_canary, projections, measurement, datasets, compatibility_model}
structure_detail_fcc
  └── gan_high_quality_gi, gan_gauge_aligned_nsgan, src.{losses, metrics, projections}

# Eval package
eval.audit    → eval.checker, eval.metrics
eval.checker  → eval.metrics
eval.seed_variance → eval.checker
eval.scr5_convention_bridge → src.train, eval.checker

# Core platform (src/)
src.train → src.{datasets, exact_measurement, losses, measurement, models, metrics,
                  pattern_diagnostics, pattern_regularizers, pattern_utils, utils,
                  dc_balanced, projections}
src.eval  → src.{datasets, exact_measurement, measurement, metrics, models,
                  pattern_diagnostics, pattern_regularizers, pattern_utils, utils}
src.compatibility_eval → src.compatibility_data
src.compatibility_data → src.projections
src.g2r_modec_train    → src.g2r_modec
src.phase79_rad5_rowspace_diversity_diagnostic
  → src.{phase69A_gauge_gan_signal_diagnostic, phase69B_controlled_gauge_cgan_pilot,
          phase73_overnight_gauge_gan_expansion, models, utils}
```

---

## 11. What Is Not Here

- **`HANDOFF/`** — documentation only; no importable code.
- **`research_lines/`** — INDEX.md pointer files only; all code remains at its original path.
- **`_unrelated_fresnel_zone_plate/`** — quarantined; not part of the GI program.
- **`outputs/`, `results/`, `cert_package_20260612/`** — artifact directories; not source code.
- **ZIFB battery content** — deleted from this repo; belongs to `E:/zifb_final_9129_luck`.
