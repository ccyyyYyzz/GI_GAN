# Feasible Counterfactual Compatibility: Phase-1 Implementation Plan

This document records the initial repository reconnaissance and the planned
implementation sequence.  No generator loss changes are in scope for this
phase.

## 1. Current Code Reconnaissance

### 1.1 Forward operator storage and normalization

- The fixed forward operator is `GhostMeasurementOperator` in
  `src/measurement.py`.
- `A` is stored explicitly as a float32 tensor with shape `[m, n]` in
  `measurement.A`; the code never materializes an `[n, n]` null projector.
- The solver cache is `K = A A^T + lambda I`, stored as `measurement.K`, with
  an optional Cholesky factor and a float64 shadow cache for freshness checks.
- Non-Hadamard fixed patterns are built by `create_fixed_measurement_matrix`.
  For Rademacher, rows are sampled in `{ -1, +1 }` and normalized by
  `row_norm_sqrt_n_over_m`, equivalent to row norm `sqrt(n / m)` unless
  normalization is disabled.
- Hadamard-family patterns are built by `create_hadamard_measurement_matrix`.
  With `matrix_normalization: orthonormal_rows`, selected rows are scaled by
  `1 / sqrt(n)`, so `A A^T` is close to identity.
- Current Scr-5 scripts do not rely only on regenerated config-side A.  They
  load the locked cert-package matrix `results/cert_package_20260612/cache/A_scr5.npy`
  and install it through `measurement.set_A_override(...)`, rebuilding the
  solver cache.

Relevant locations:

- `src/measurement.py`: `create_hadamard_measurement_matrix`, `create_fixed_measurement_matrix`,
  `GhostMeasurementOperator`, `set_A_override`.
- `src/eval.py` and `src/train.py`: `make_measurement(...)` maps
  `lambda_solver` to `lambda_dc` and passes normalization fields.
- `src/phase69A_gauge_gan_signal_diagnostic.py`: Scr-5 locked A path and
  preflight orthonormality check.
- `src/phase73_overnight_gauge_gan_expansion.py`: regime A override for
  `scr5`, `scr10`, and `rad5`.

### 1.2 B_lambda, audit, row/null projection implementation

Current soft/ridge projection operations are already matrix-free in `n x n`:

- `data_solution(y, "ridge_pinv")`: `A^T (A A^T + lambda I)^(-1) y`.
- `null_project(v)`: `v - A^T (A A^T + lambda I)^(-1) A v`.
- `dc_project(v, y)`: `v - A^T (A A^T + lambda I)^(-1) (A v - y)`.
- `A_forward(v)`: `v @ A.T`.
- `AT_forward(y)`: `y @ A`.

These use `solve_K` and never construct dense `P0`.

There are also exact/orthonormal helper variants in phase scripts:

- `phase69A.exact_projectors`, `p0_exact`, `blambda_y` for diagnostics.
- `phase69B.p0_ortho`, `blambda_ortho`, `pi_lambda_ortho` for Scr-5
  orthonormal-row assumptions.

Phase-1 should consolidate a public utility API around the stable
`GhostMeasurementOperator` methods, while using exact row/null projection only
when explicitly requested for counterfactual construction.

### 1.3 Rad-5 and Scr-5 dataset, image size, and m

There are two relevant Rad-5 contexts:

- Source imported checkpoint config:
  `E:/ns_mc_gan_gi/outputs_phase15/imported_noleak/rademacher5_hq_noise001_colab/resolved_config.yaml`
  has `dataset_name: stl10`, `img_size: 64`, `sampling_ratio: 0.05`,
  `pattern_type: rademacher`, `matrix_normalization: legacy_sqrt_m`,
  `lambda_solver: 0.001`, and `model_type: hq_two_stage`.
  At 64x64, `n=4096` and `m=round(0.05*n)=205`.
- Current Rad-5 B/C/D paper-completion path:
  `src/phase78_96px_rad5_one_seed_probe.py` forces `IMG_SIZE=96`,
  `sampling_ratio=0.05`, `pattern_type=rademacher`, and checks `m=461`.
  It uses STL10 train+unlabeled for train/val and STL10 test for test,
  resized to 96x96 grayscale.

Scr-5 current paired/gauge path:

- Source imported checkpoint config:
  `E:/ns_mc_gan_gi/outputs_phase15/imported_noleak/scrambled_hadamard5_hq_noise001_colab/resolved_config.yaml`
  has `dataset_name: stl10`, `img_size: 64`, `sampling_ratio: 0.05`,
  `pattern_type: lowfreq_hadamard`, `matrix_normalization: orthonormal_rows`,
  `backprojection_mode: hadamard_zero_filled`, `lambda_solver: 0.001`,
  and `model_type: hq_two_stage`.
- The Scr-5 scripts override the operator with the locked `A_scr5.npy`.
  At 64x64, `n=4096` and `m=205`.
- Phase71 uses train/val/test counts 1024/256/256 from locked splits.

### 1.4 Current B/C/D checkpoint loading

Published/source checkpoint loading:

- Rad-5 96px source generator:
  `phase78.load_generator_96(...)` loads `RAD5_CHECKPOINT`, builds the
  generator from config, and prefers `generator_ema` over `generator`.
- Scr-5 source generator:
  `phase69A.load_checkpoint_and_generator(...)` and
  `phase69B.load_generator_from_checkpoint(...)` load the imported Scr-5
  checkpoint and prefer `generator_ema` over `generator`.
- Phase73 generalized regimes use `load_regime_generator(...)` and the same
  `generator_ema`-then-`generator` rule.

Fine-tuned B/C/D checkpoint loading:

- Rad-5 Phase81:
  `best_ckpt(seed, arm)` uses Phase78 seed01 B/C if present, otherwise
  Phase81 `seedXX/{B,C,D_standard}/checkpoints/best_by_val.pt`.
  Eval loads those via `phase78.load_probe_checkpoint_for_eval(...)`, which
  loads `payload["generator"]`.
- Scr-5 Phase71/74/75:
  B/C are `outputs_phase71_gauge_cgan_paired_seeds/seedXX/{B,C}/checkpoints/best_by_val.pt`.
  D_standard is in Phase74/Phase75 standard-cGAN outputs.
  Eval loaders build a fresh generator and load `payload["generator"]`.

### 1.5 Current random sampler for multiple candidates from the same y

- Deterministic forward paths use `zero_noise = torch.zeros_like(x_data)`.
  This appears in `phase78.forward_candidate`, `phase73.forward_candidate_general`,
  and the Scr-5 `phase69B.forward_candidate`.
- The stochastic same-y candidate path is currently in
  `src/phase79_rad5_rowspace_diversity_diagnostic.py`.
  `smoke_sampling(...)` repeats one measurement `y` to `K` copies, samples
  `noise = torch.randn(x_data.shape, generator=seeded_generator)`, then calls
  `forward_with_noise(...)`.
- The current generator receives the noise map as the second channel input, but
  previous diagnostics found the deterministic checkpoint mostly ignores it.
  Phase-1 candidate selection must therefore support both deterministic
  baselines and stochastic candidate sources, without changing generator loss.

## 2. New Phase-1 Modules and Files

Planned additions:

- `src/projections.py`
  - `row_project(x, operator, exact=False, lambda_=None)`
  - `null_project(x, operator, exact=False, lambda_=None)`
  - `soft_audit(x, y, operator, lambda_=None)`
  - `relative_measurement_error(x, y, operator)`
  - Handles image `[B,1,H,W]` and flat `[B,n]` inputs.
  - Reuses `operator.A_forward`, `operator.AT_forward`, and `operator.solve_K`.
  - Never constructs dense `[n,n]` projectors.

- `src/models/compatibility.py`
  - Two light convolutional encoders, `f_r` and `f_n`.
  - Output dimension 128 by default.
  - L2-normalized embeddings.
  - `score = dot(f_r, f_n) / temperature`.
  - Optional small joint MLP head.

- `src/compatibility_data.py`
  - On-the-fly matched/mismatched pair generator.
  - Optional sharded disk cache for `r_i`, `n_i`, energy, labels, and indices.
  - No GPU-wide cache by default.
  - No clipping before feasibility checks or critic input.

- `train_compatibility.py`
  - YAML config.
  - AMP, gradient clipping, resume/checkpoint, atomic checkpoint writes.
  - Symmetric InfoNCE for in-batch negatives.
  - Stage-2 semi-hard negatives.
  - Interpolated negatives with margin ranking.
  - Logs config, git commit, seeds, runtime, split hashes.

- `eval_compatibility.py`
  - Random-negative ROC-AUC.
  - Semi-hard ROC-AUC.
  - Recall@1 and Recall@5 among 32 donors.
  - Spearman correlation between score and true P0 error.
  - Score distributions.
  - Energy-matched negative metrics.
  - Same-class/cross-class metrics for evaluation only.
  - Per-image CSV and summary JSON.

- `eval_candidate_selection.py`
  - K in `{1,4,8,16,32}` candidates for one y.
  - Deterministic output, random candidate, posterior mean, critic-selected,
    oracle best-of-K.
  - All candidates use the same audit.
  - Metrics: P0 RMSE, PSNR, SSIM, LPIPS, RAPSD, RelMeasErr, optional semantic
    classifier consistency, selection regret, oracle-gain fraction, per-image
    CSV, paired bootstrap CIs.

- `configs/compatibility/rad5_pilot.yaml`
- `configs/compatibility/scr5_pilot.yaml`

- `tests/test_projections.py`
- `tests/test_feasible_counterfactuals.py`
- `tests/test_compatibility_model.py`

## 3. Projection Semantics

Phase-1 needs both exact geometry and soft audit:

- Exact row/null decomposition for counterfactual construction:
  - `P_R x = A^dagger A x`.
  - `P0 x = x - P_R x`.
  - For full-row-rank A, implement via `G = A A^T` and solves with `G`, not
    with `G + lambda I`.
  - For Scr-5 orthonormal rows, exact projection reduces to `A^T A x`.
  - Use float64 in tests and when validating feasibility.

- Soft audit for deployed consistency:
  - `Pi_y^lambda(v) = v - B_lambda (A v - y)`.
  - `B_lambda = A^T (A A^T + lambda I)^(-1)`.
  - Default lambda comes from `operator.lambda_dc` / config `lambda_solver`.

The user-facing `row_project` and `null_project` should expose an `exact`
flag.  Default training data for feasible counterfactuals should use exact
projection; deployed auditing should use soft audit.

## 4. Projection Unit Tests

For both Rad and Scrambled-Hadamard regimes:

- Float64:
  - `||A P0(v)|| / ||v|| < 1e-10`, unless conditioning forces a documented
    relaxed tolerance in the test report.
- Float32:
  - target `< 1e-5`.
- Reconstruction:
  - `PR(v) + P0(v)` reconstructs input.
- Orthogonality:
  - `<PR(v), P0(v)>` near zero.
- No dense `[n,n]` projector construction.

Rad-5 pilot should first use small batch sizes and current 96px Phase81
operator.  Scr-5 tests should use the locked `A_scr5.npy` override.

## 5. Feasible Counterfactual Data

For each train sample:

- Compute `y_i = A x_i`.
- Compute `r_i = PR x_i` using exact row projection.
- Compute `n_i = P0 x_i`.

Construct:

- Positive: `(r_i, n_i)`.
- Random negative: `(r_i, n_j)`, `j != i`, donor only from train/calibration
  pools, never test.
- Semi-hard negative:
  - donor `r_j` close to `r_i`,
  - null energy matched to `n_i`,
  - no test labels and no direct class labels for training.
- Interpolated negative:
  - `n_ij(alpha) = (1-alpha)n_i + alpha*n_j`,
  - `alpha in {0.25, 0.5, 0.75, 1.0}`.

Feasibility validation:

- `u_ij = r_i + n_j`.
- Check `||A u_ij - y_i|| / ||y_i||` at projection precision.
- Never clip `u_ij` before projection, feasibility, or critic input.
- Only visualization copies may be clipped.

## 6. Compatibility Critic Training

Model:

- `CompatibilityCritic`.
- Separate encoders for row image and null image.
- L2-normalized 128-d embeddings.
- Dot-product score divided by learned or fixed temperature.
- Optional joint MLP only after embeddings.

Training:

- Stage 1: symmetric InfoNCE on matched batch pairs; in-batch other nulls are
  negatives.
- Stage 2: add semi-hard negatives.
- Interpolated negatives: margin ranking loss.
- AMP enabled by config.
- Gradient clipping.
- Atomic checkpoints:
  - write temp file,
  - fsync/close where available,
  - rename to final.
- Resume restores model, optimizer, scaler, scheduler, epoch/step, RNG states.
- All seeds and git commit are written to output metadata.

## 7. Candidate Selection with Frozen Generator

The generator remains frozen.  No existing generator loss is modified.

Candidate sources:

- Deterministic candidate from current forward path.
- Stochastic candidates from repeated `y` plus sampled noise map, following
  `phase79.forward_with_noise`.
- Optional future candidate source can be configured, but Phase-1 does not
  fine-tune the generator.

For each `K in {1, 4, 8, 16, 32}`:

- Generate candidates for the same `y`.
- Apply the same `soft_audit` to each candidate.
- Compute row/null components for scoring.
- Select by compatibility critic without ground truth.
- Compute oracle best-of-K only offline with ground truth and never use it for
  selection or threshold tuning.

## 8. Leakage Controls

Hard constraints:

- Test ground truth only for metrics and oracle.
- Critic-selected path never reads ground truth.
- Negative donors never come from test.
- Normalization statistics only from train.
- No full-test threshold tuning.
- Keep independent train/validation/calibration/test split manifests.
- Same-class/cross-class metrics are evaluation-only and do not enter training.

Implementation controls:

- Split manifests written into each output directory.
- Donor sampler asserts donor pool split.
- Candidate selector accepts no target image in critic-selected code path.
- Evaluation code computes oracle in a separate function with an explicit
  `oracle_only=True` marker.

## 9. Gate Report

`gate_report.json` must include:

- projector tests pass/fail,
- feasible-counterfactual tests pass/fail,
- baseline reproduction differences,
- semi-hard AUC,
- Recall@1 relative to random,
- score/P0-error Spearman,
- oracle best-of-16 headroom,
- critic fraction of oracle gain.

It should recommend generator fine-tuning only if all are true:

1. semi-hard AUC >= 0.70,
2. Recall@1 >= 4 times random Recall@1,
3. score/P0-error Spearman <= -0.20,
4. oracle best-of-16 improves over deterministic,
5. critic captures at least 30% of oracle gain.

## 10. Execution Order

1. Add projection utility module and tests.
2. Add counterfactual pair dataset and feasibility tests.
3. Add compatibility model unit tests.
4. Add Rad-5 pilot YAML and training CLI.
5. Train/evaluate Rad-5 pilot.
6. Add frozen-generator candidate-selection evaluation.
7. Generate Rad-5 `gate_report.json`.
8. Only after Rad-5 pilot passes basic plumbing, run Scr-5 with locked
   `A_scr5.npy`.

Not in this phase:

- No witness split implementation.
- No generator loss changes.
- No GAN loss modification.
- No test-set tuning.

## 11. Phase-1 Implementation Update

This implementation is being performed in the isolated working copy
`E:/ns_mc_gan_gi_code_fcc_phase1`; the original repository
`E:/ns_mc_gan_gi_code` is not modified.

Changes from the initial reconnaissance plan:

- Exact geometry is now exposed with explicit names:
  `exact_row_project`, `exact_null_project`, and `exact_data_anchor`.
  The existing ridge operations remain soft/noise-aware and are not used for
  feasible counterfactual construction.
- `src/models/compatibility.py` could not be created without changing the
  existing module layout, because this repository already has a top-level
  `src/models.py` module used by generator/discriminator imports.  To avoid
  breaking baseline behavior, the compatibility critic is implemented in
  `src/compatibility_model.py`.
- The Rad-5 pilot config is `configs/compatibility/rad5_96_pilot.yaml`.
  It keeps the canonical Phase78 Rad-5/96 operator (`m=461`) but uses a short
  single-seed pilot budget.
- Candidate selection is gate-controlled.  `eval_candidate_selection.py` reads
  `gate_report_e1.json` and writes a skipped E2 report unless the E1
  compatibility gate passes.
