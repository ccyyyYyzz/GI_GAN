# NS-MC-GAN Ghost Imaging

> **🧭 New here? Start with [`HANDOFF/00_START_HERE.md`](HANDOFF/00_START_HERE.md).**
>
> This repository is one working copy of a larger **undersampled ghost-imaging range/null-space research program**. The [`HANDOFF/`](HANDOFF/) directory is the authoritative, chronological guide to the whole work — theory, code map, reproducibility, evidence, papers, and red lines — written so a fresh session or a new reader can follow the research **in order**, reuse the code, and reproduce every result.

## How this repository is organized

- **[`HANDOFF/`](HANDOFF/)** — the research handoff. Read in order:
  [`00_START_HERE.md`](HANDOFF/00_START_HERE.md) →
  [`01_RESEARCH_STORY.md`](HANDOFF/01_RESEARCH_STORY.md) (chronological spine, stages 0–8) →
  [`02_THEORY_CORE.md`](HANDOFF/02_THEORY_CORE.md) →
  [`03_CODE_MAP.md`](HANDOFF/03_CODE_MAP.md) →
  [`04_REPRODUCIBILITY_GUIDE.md`](HANDOFF/04_REPRODUCIBILITY_GUIDE.md) →
  [`05_EXPERIMENTS_AND_EVIDENCE.md`](HANDOFF/05_EXPERIMENTS_AND_EVIDENCE.md) →
  [`06_PAPERS_AND_CLAIMS.md`](HANDOFF/06_PAPERS_AND_CLAIMS.md) →
  [`07_RED_LINES_AND_WORKING_RULES.md`](HANDOFF/07_RED_LINES_AND_WORKING_RULES.md).
  [`HANDOFF/archive_gan_fcc_work/`](HANDOFF/archive_gan_fcc_work/) holds the full-program Chinese handoff copied from the master work root (`E:/GAN_FCC_WORK`).
- **[`research_lines/`](research_lines/)** — one folder per research line (`00_program_overview` … `08_vqgan_fcc`) with an `INDEX.md` that **points to** the relevant code. The code is **not** moved: it stays flat at the repository root so that `cwd = repo-root` imports keep working.
- **Importable core stays flat at root**: `src/`, `eval/`, `configs/`, the root `*.py` modules, `paper/`, `colab/`, `scripts/`, `tests/`. Always run commands from the repository root.
- **`_unrelated_fresnel_zone_plate/`** — an unrelated Fresnel-zone-plate student manuscript, quarantined out of the root during organization (it is **not** ghost-imaging science; see [`HANDOFF/08_NON_GI_CONTENT_QUARANTINE.md`](HANDOFF/08_NON_GI_CONTENT_QUARANTINE.md)).

The rest of this file is the original **per-phase reproduction manual (Phases 1–10)**, kept as-is for the early-phase details.

---

## About this project

`ns_mc_gan_gi` is a compact PyTorch research project for low-sampling-rate
computational ghost imaging / single-pixel imaging reconstruction. The input is
bucket measurements

```text
y = A x + epsilon
```

and the model reconstructs an image `x_hat`.

The main idea is to constrain the generator to measurement-null-space detail
rather than letting a GAN hallucinate a full image. The physics term anchors the
output to the observed buckets, while the WGAN-GP critic nudges reconstructions
toward natural grayscale images.

## File Tree

```text
ns_mc_gan_gi/
  README.md
  requirements.txt
  configs/
    default.yaml
    debug.yaml
    quick_train_5pct.yaml
  src/
    __init__.py
    datasets.py
    measurement.py
    models.py
    losses.py
    metrics.py
    train.py
    eval.py
    sanity_physics.py
    visualize.py
    utils.py
  scripts/
    train_5pct.sh
    eval_5pct.sh
    sweep_sampling.sh
    debug_5pct.ps1
    train_quick_5pct.ps1
    eval_quick_5pct.ps1
    sanity_physics.ps1
  outputs/
    .gitkeep
```

## Math Model

The measurement equation is

```text
y = A x + epsilon
```

The Tikhonov backprojection/data solution is

```text
x_data = A^T (A A^T + lambda I)^(-1) y
```

The null-space projection used for the generated residual is

```text
P_N(v) = v - A^T (A A^T + lambda I)^(-1) A v
```

The data-consistency projection is

```text
Pi_y(v) = v - A^T (A A^T + lambda I)^(-1) (A v - y)
```

The final reconstruction is

```text
r = G(x_data, z)
x_hat = Pi_y(x_data + P_N(r))
```

This is better suited to low-sampling ghost imaging than a plain GAN because:

- The generator only fills in measurement-null-space residual information.
- The data-consistency projection reduces measurement-violating hallucination.
- The adversarial loss supplies a natural-image prior without replacing physics.

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

Use Python 3.10 or newer. The requirements pin `numpy<2` because many common
PyTorch / torchvision / scikit-image Windows environments still ship binary
extensions compiled against NumPy 1.x.

The default config keeps generated data off the C drive:

```yaml
dataset_root: E:/ns_mc_gan_gi/data
output_dir: E:/ns_mc_gan_gi/outputs
```

Change these in `configs/default.yaml` or pass CLI overrides if you want local
project-relative storage.

## Debug Run

Runs 2 epochs with 512 train images and 128 validation images:

```bash
python -m src.train --config configs/debug.yaml --device cuda
```

On Windows PowerShell:

```powershell
.\scripts\debug_5pct.ps1
```

The PowerShell scripts first honor `NS_MCGAN_PYTHON`, then try the local
Anaconda Python path, and finally fall back to `python`.

## Physics Sanity Check

```bash
python -m src.sanity_physics --config configs/debug.yaml --device cuda
```

This writes `E:/ns_mc_gan_gi/outputs/debug_5pct/sanity_physics.json` with
random-tensor and STL-10 batch checks for null-space projection, data
consistency projection, and the data-solution backprojection.

## Training

```bash
bash scripts/train_5pct.sh
```

Equivalent direct command:

```bash
python -m src.train --config configs/default.yaml --sampling_ratio 0.05 --device cuda
```

The script writes checkpoints, TensorBoard logs, and sample grids to
`E:/ns_mc_gan_gi/outputs/sr_005`.

## Evaluation

```bash
bash scripts/eval_5pct.sh
```

Equivalent direct command:

```bash
python -m src.eval \
  --config configs/default.yaml \
  --checkpoint E:/ns_mc_gan_gi/outputs/sr_005/best_ssim.pt \
  --sampling_ratio 0.05 \
  --device cuda
```

Evaluation prints mean PSNR, SSIM, MSE, and relative measurement error. It also
saves qualitative grids to `eval_samples/`.

Phase 1.5 evaluation reports both:

- Backprojection: `x_data = A^T(AA^T + lambda I)^(-1)y`
- NS-MC-GAN: `x_hat`

It writes `eval_metrics.json` and saves a fixed 8-sample grid to
`eval_samples/recon_grid.png`.

## Sampling Sweep

Evaluate checkpoints for 1%, 2%, 5%, and 10% sampling:

```bash
bash scripts/sweep_sampling.sh
```

Train all four ratios:

```bash
bash scripts/sweep_sampling.sh train
```

## Phase 2 Workflow

Create a clean Windows conda environment when the current environment has a
NumPy ABI mismatch:

```powershell
.\scripts\create_clean_env.ps1
conda activate ns_mc_gan_gi_py311
```

Verify the active environment:

```bash
python -m src.verify_env --dataset_root E:/ns_mc_gan_gi/data --output_dir E:/ns_mc_gan_gi/outputs
```

Run the 5% quick experiment:

```bash
python -m src.sanity_physics --config configs/quick_train_5pct.yaml --device cuda
python -m src.train --config configs/quick_train_5pct.yaml --device cuda
python -m src.eval --checkpoint E:/ns_mc_gan_gi/outputs/quick_5pct/best_ssim.pt --config configs/quick_train_5pct.yaml --device cuda
```

Run the quick sampling sweep on Windows:

```powershell
.\scripts\train_sweep_quick.ps1
.\scripts\eval_sweep_quick.ps1
python -m src.aggregate_results
```

Run the 5% ablations:

```powershell
.\scripts\train_ablation_5pct.ps1
.\scripts\eval_ablation_5pct.ps1
python -m src.aggregate_ablations
```

Generate the Phase 2 report:

```bash
python -m src.make_phase2_report
```

Phase 2 aggregation writes CSV, Markdown, and figure artifacts under
`E:/ns_mc_gan_gi/outputs`.

## Phase 2.1 Clean Reproduction

Phase 2.1 keeps the previous ABI-affected outputs untouched and writes all clean
environment artifacts to:

```text
E:/ns_mc_gan_gi/outputs_clean_phase2
```

Create the clean conda environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/create_clean_env.ps1
```

The script creates the clean prefix environment on E:

```text
E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311
```

Verify the clean environment:

```powershell
$CleanEnv = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$env:PYTHONNOUSERSITE = "1"
conda run -p $CleanEnv python -s -m src.verify_env `
  --dataset_root E:/ns_mc_gan_gi/data `
  --output_dir E:/ns_mc_gan_gi/outputs_clean_phase2 `
  --report_path E:/ns_mc_gan_gi/outputs_clean_phase2/env_report_clean.json
```

The clean report should show `numpy_version < 2`, `torch_numpy_bridge: ok`,
CUDA available, the RTX 4060 Laptop GPU, and successful torchvision,
matplotlib, tensorboard, and skimage imports.

Run the clean 5% reproduction first:

```powershell
$CleanEnv = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$env:PYTHONNOUSERSITE = "1"
conda run -p $CleanEnv python -s -m src.sanity_physics --config configs/clean_quick_5pct.yaml --device cuda
conda run -p $CleanEnv python -s -m src.train --config configs/clean_quick_5pct.yaml --device cuda
conda run -p $CleanEnv python -s -m src.eval --checkpoint E:/ns_mc_gan_gi/outputs_clean_phase2/quick_5pct/best_ssim.pt --config configs/clean_quick_5pct.yaml --device cuda
conda run -p $CleanEnv python -s -m src.compare_old_vs_clean --old_dir E:/ns_mc_gan_gi/outputs/quick_5pct --clean_dir E:/ns_mc_gan_gi/outputs_clean_phase2/quick_5pct --output_dir E:/ns_mc_gan_gi/outputs_clean_phase2
```

Run the clean sampling sweep:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_clean_sweep_quick.ps1
powershell -ExecutionPolicy Bypass -File scripts/eval_clean_sweep_quick.ps1
$CleanEnv = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$env:PYTHONNOUSERSITE = "1"
conda run -p $CleanEnv python -s -m src.aggregate_results --base_dir E:/ns_mc_gan_gi/outputs_clean_phase2 --output_prefix clean_phase2
```

The clean sweep has separate configs for 1%, 2%, 5%, and 10%:

```text
configs/clean_quick_1pct.yaml
configs/clean_quick_2pct.yaml
configs/clean_quick_5pct.yaml
configs/clean_quick_10pct.yaml
```

Run the clean 5% ablations after clean 5% and at least 2% and 10% have
completed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_clean_ablation_5pct.ps1
powershell -ExecutionPolicy Bypass -File scripts/eval_clean_ablation_5pct.ps1
$CleanEnv = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$env:PYTHONNOUSERSITE = "1"
conda run -p $CleanEnv python -s -m src.aggregate_ablations --base_dir E:/ns_mc_gan_gi/outputs_clean_phase2 --output_prefix clean_phase2_ablation
```

Generate the clean Phase 2.1 report:

```powershell
$CleanEnv = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$env:PYTHONNOUSERSITE = "1"
conda run -p $CleanEnv python -s -m src.make_clean_phase2_report --base_dir E:/ns_mc_gan_gi/outputs_clean_phase2 --old_dir E:/ns_mc_gan_gi/outputs/quick_5pct
```

Important output files:

```text
E:/ns_mc_gan_gi/outputs_clean_phase2/env_report_clean.json
E:/ns_mc_gan_gi/outputs_clean_phase2/compare_old_vs_clean_5pct.json
E:/ns_mc_gan_gi/outputs_clean_phase2/compare_old_vs_clean_5pct.md
E:/ns_mc_gan_gi/outputs_clean_phase2/clean_phase2_results.csv
E:/ns_mc_gan_gi/outputs_clean_phase2/clean_phase2_results.md
E:/ns_mc_gan_gi/outputs_clean_phase2/clean_phase2_ablation_results.csv
E:/ns_mc_gan_gi/outputs_clean_phase2/clean_phase2_ablation_results.md
E:/ns_mc_gan_gi/outputs_clean_phase2/CLEAN_PHASE2_REPORT.md
```

Do not mix `E:/ns_mc_gan_gi/outputs` and
`E:/ns_mc_gan_gi/outputs_clean_phase2`; the first is the old ABI environment
result root and the second is the clean Phase 2.1 result root.

## Phase 3: Learnable Speckle Patterns

Phase 3 adds learnable physical illumination patterns for ghost imaging while
keeping the null-space residual and measurement-consistency projections.

The learned physical pattern bank stores logits `L_phi` and produces
non-negative projected patterns:

```text
P_phi = sigmoid(L_phi / tau)
```

For `learned_binary_ste`, the forward pass uses hard binary DMD/SLM-style
patterns and the backward pass uses a straight-through estimator:

```text
P_phi = stopgrad(1[P_soft > 0.5] - P_soft) + P_soft
```

For `learned_continuous`, the physical pattern is the soft value directly.
In both modes, `P_phi` stays in `[0, 1]`, while the effective measurement
matrix is centered and standardized for differential ghost imaging:

```text
A_phi = (P_phi - mean(P_phi)) / (std(P_phi) sqrt(m))
```

All Phase 3 outputs are written under:

```text
E:/ns_mc_gan_gi/outputs_phase3
```

Run the debug path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/phase3_debug.ps1
```

Run the 5% learned-pattern experiments:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_phase3_5pct.ps1
powershell -ExecutionPolicy Bypass -File scripts/eval_phase3_5pct.ps1
```

Run the learned binary 2% and 10% sweep:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_phase3_binary_sweep.ps1
powershell -ExecutionPolicy Bypass -File scripts/eval_phase3_binary_sweep.ps1
```

Run pattern-regularization ablations:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_phase3_pattern_ablation.ps1
powershell -ExecutionPolicy Bypass -File scripts/eval_phase3_pattern_ablation.ps1
```

Aggregate and report:

```powershell
$CleanEnv = "E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311"
$env:PYTHONNOUSERSITE = "1"
conda run -p $CleanEnv python -s -m src.aggregate_phase3
conda run -p $CleanEnv python -s -m src.make_phase3_report
```

Important Phase 3 artifacts:

```text
E:/ns_mc_gan_gi/outputs_phase3/phase3_main_results.csv
E:/ns_mc_gan_gi/outputs_phase3/phase3_main_results.md
E:/ns_mc_gan_gi/outputs_phase3/phase3_pattern_ablation_results.csv
E:/ns_mc_gan_gi/outputs_phase3/phase3_pattern_stats.csv
E:/ns_mc_gan_gi/outputs_phase3/PHASE3_REPORT.md
```

Learned pattern images are saved per epoch under `patterns/` and at eval time
under `eval_patterns/`, for example:

```text
E:/ns_mc_gan_gi/outputs_phase3/learned_binary_5pct/eval_patterns/final_patterns.png
E:/ns_mc_gan_gi/outputs_phase3/learned_binary_5pct/eval_samples/recon_grid.png
```

## Phase 4: Learned Pattern Optimization

Phase 4 keeps the fixed measurement path intact and adds a fairer learned-pattern tuning path. Outputs are written to `E:/ns_mc_gan_gi/outputs_phase4`.

Fixed-compatible initialization starts learned binary patterns from the same sign structure as the fixed rademacher baseline:

```text
P_0 = 1[A_fixed_sign > 0]
L_0 = alpha(2P_0 - 1)
```

Balanced binary STE enforces per-row transmission:

```text
P_soft = sigmoid(L / tau)
P_hard = TopK(P_soft, k = target_transmission * n)
P = stopgrad(P_hard - P_soft) + P_soft
```

Contrast regularization prevents continuous patterns from collapsing near 0.5 with weak physical modulation:

```text
L_contrast = mean_i (std(P_i) - target_contrast)^2
```

Typical commands from the project root:

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.sanity_learnable_patterns --config configs/phase4_debug_matched_binary_5pct.yaml --device cuda
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.train --config configs/phase4_matched_binary_5pct.yaml --device cuda
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.eval --checkpoint E:/ns_mc_gan_gi/outputs_phase4/matched_binary_5pct/best_score.pt --config configs/phase4_matched_binary_5pct.yaml --device cuda
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.aggregate_phase4
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.make_phase4_report
```

Convenience scripts:

```powershell
.\scripts\phase4_debug.ps1
.\scripts\train_phase4_tuning_5pct.ps1
.\scripts\eval_phase4_tuning_5pct.ps1
.\scripts\phase4_continuous_to_binary.ps1
.\scripts\aggregate_phase4.ps1
```

The continuous-to-binary curriculum converts a continuous learned checkpoint into balanced binary logits:

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.convert_continuous_to_binary_checkpoint --continuous_checkpoint E:/ns_mc_gan_gi/outputs_phase4/continuous_contrast_5pct/best_score.pt --output_checkpoint E:/ns_mc_gan_gi/outputs_phase4/continuous_to_binary_5pct/init_from_continuous.pt --target_mode learned_balanced_binary_ste --target_transmission 0.5 --logit_abs_scale 2.0
```

Phase 4 summary artifacts:

```text
E:/ns_mc_gan_gi/outputs_phase4/phase4_tuning_results.csv
E:/ns_mc_gan_gi/outputs_phase4/phase4_tuning_results.md
E:/ns_mc_gan_gi/outputs_phase4/phase4_pattern_stats.csv
E:/ns_mc_gan_gi/outputs_phase4/phase4_fixed_vs_phase3_vs_phase4_5pct.png
E:/ns_mc_gan_gi/outputs_phase4/PHASE4_REPORT.md
```

Interpret `PHASE4_REPORT.md` by first checking epoch0 warm-start metrics, then the 5% tuning table, then whether the best Phase 4 score beats the fixed rademacher score. Missing experiments are intentionally marked `missing`.

## Phase 5: Exact Operator Calibration and Robustness

Phase 5 targets the Phase 4 epoch0 mismatch by making the learned binary operator exactly reproduce the fixed rademacher operator at initialization. Outputs are written to `E:/ns_mc_gan_gi/outputs_phase5`.

The fixed operator uses row-normalized rademacher rows, so for binary physical illumination the exact signed mode is:

```text
A_fixed in { -c, +c }^(m x n)
P_0 = 1[A_fixed > 0]
L_0 = alpha(2P_0 - 1)
P_phi = stopgrad(P_hard - P_soft) + P_soft
A_phi = c(2P_phi - 1)
```

`effective_A_mode` controls the learned operator:

```text
centered_standardized  # Phase 3/4 behavior
signed_from_physical   # (2P - 1) / sqrt(m)
signed_exact_fixed     # fixed rademacher scale/sign matching
```

Run the exact operator calibration first:

```powershell
.\scripts\phase5_calibrate.ps1
```

Then run exact 5% tuning and evaluation:

```powershell
.\scripts\train_phase5_tuning_5pct.ps1
.\scripts\eval_phase5_tuning_5pct.ps1
.\scripts\aggregate_phase5.ps1
```

If a Phase 5 5% exact configuration qualifies, generate and run the best sweep:

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.prepare_phase5_best_sweep
.\scripts\train_phase5_best_sweep.ps1
.\scripts\eval_phase5_best_sweep.ps1
```

Run robustness eval without retraining:

```powershell
.\scripts\phase5_noise_sweep.ps1
.\scripts\aggregate_phase5.ps1
```

Phase 5 summary artifacts:

```text
E:/ns_mc_gan_gi/outputs_phase5/operator_calibration_5pct.json
E:/ns_mc_gan_gi/outputs_phase5/operator_calibration_5pct.md
E:/ns_mc_gan_gi/outputs_phase5/phase5_tuning_results.csv
E:/ns_mc_gan_gi/outputs_phase5/phase5_epoch0_equivalence.csv
E:/ns_mc_gan_gi/outputs_phase5/phase5_pattern_stats.csv
E:/ns_mc_gan_gi/outputs_phase5/phase5_noise_sweep_summary.md
E:/ns_mc_gan_gi/outputs_phase5/PHASE5_REPORT.md
```

Interpret `PHASE5_REPORT.md` by checking calibration first. `A_rel_fro_error < 1e-6` means exact operator matching succeeded; the tuning table then separates operator fairness from final reconstruction performance. Missing optional experiments remain marked `missing`.

## Phase 6: Pattern Causality Audit

Phase 6 asks whether Phase 5 gains come from learned physical illumination or from continued generator/discriminator fine-tuning after warm start.

Main additions:

- `src/pattern_diagnostics.py` captures the initial hard/soft patterns, logits, and effective operator, then reports hard flip fraction, operator drift, soft/logit drift, secant-RIP drift, and off-diagonal correlation drift.
- `freeze_patterns`, `freeze_generator_all`, `freeze_discriminator_all`, and `pattern_requires_grad` allow causal controls.
- `src/eval_paired_controls.py` evaluates multiple checkpoints on the same validation subset and writes paired bootstrap confidence intervals.
- `src/aggregate_phase6.py`, `src/make_phase6_report.py`, and `src/export_paper_assets.py` generate the Phase 6 evidence chain, report, figures, and LaTeX tables.

Typical run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/train_phase6_controls.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/eval_phase6_controls.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/phase6_paired_eval.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/aggregate_phase6.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/export_phase6_paper_assets.ps1
```

Outputs are written to `E:/ns_mc_gan_gi/outputs_phase6`. The main report is `E:/ns_mc_gan_gi/outputs_phase6/PHASE6_REPORT.md`.

Interpretation rule:

- If pattern-trainable improves over G-only and hard pattern drift is nonzero with a positive paired CI, evidence supports learned physical illumination.
- If pattern-trainable is close to G-only or hard flip fraction is zero, the improvement should be attributed mainly to generator fine-tuning under the current hard-binary setup.

## Phase 7: Flip-aware And Continuous Pattern Learning

Phase 7 extends the Phase 6 attribution audit with two stronger learned-pattern branches:

- `learned_flip_aware_binary_ste` keeps hard binary forward patterns while adding threshold noise, margin regularization, and a soft flip proxy to make hard flips possible.
- `continuous_differential` treats `P in [0,1]` as a hardware-realistic continuous illumination pattern and reports continuous contrast separately from binary claims.
- `src/eval_pattern_swap.py` evaluates `Fixed G + Fixed A`, `Learned G + Initial A`, `Learned G + Learned A`, and `Fixed G + Learned A`.
- `src/measurement_quality.py` reports measurement-only geometry: Gram spectrum, secant-RIP, bucket SNR proxy, and off-diagonal correlation.
- `src/aggregate_phase7.py`, `src/make_phase7_report.py`, and `src/export_phase7_paper_assets.py` generate Phase 7 tables, report, figures, and paper assets.

Minimum run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/train_phase7_flipaware.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/eval_phase7_flipaware.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/train_phase7_continuous.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/eval_phase7_continuous.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/phase7_pattern_swap.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/phase7_measurement_quality.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/aggregate_phase7.ps1
```

Outputs are written to `E:/ns_mc_gan_gi/outputs_phase7`. The main report is `E:/ns_mc_gan_gi/outputs_phase7/PHASE7_REPORT.md`.

Interpretation rule:

- Claim binary learned illumination only if hard flips and `A` drift are nonzero and the flip-aware run beats the G-only reference by a meaningful margin.
- Claim continuous learned illumination only if continuous trainable beats continuous G-only and swap/measurement-quality diagnostics support an `A` contribution.
- If neither branch beats G-only, report learned illumination as unresolved or negative and attribute reconstruction gains to measurement-consistent fine-tuning.

## Phase 10: Full/Near-full Hadamard HQ Validation

Phase 10 turns the Phase 9 diagnosis into full or near-full evidence. Phase 9 is important because it established that low-frequency Hadamard with DC included gives a physically meaningful backprojection, that the Hadamard operator sanity checks pass, and that short STL-10 probes can reach usable quality while random Rademacher remains a weak control. Phase 10 should not call those Phase 9 `short_train` probes full training.

All Phase 10 data and outputs stay off the C drive:

```text
E:/ns_mc_gan_gi/data
E:/ns_mc_gan_gi/outputs_phase10
E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311
```

Run the resumable overnight queue:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/phase10_overnight.ps1
```

The runner reads `configs/phase10/overnight_queue.yaml`, writes task state to `E:/ns_mc_gan_gi/outputs_phase10/overnight_status.json`, and logs stdout/stderr separately under `E:/ns_mc_gan_gi/outputs_phase10/logs`. It skips completed tasks when required outputs already exist, so the same command can be resumed after an interruption.

Manual core path:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/phase10_core_manual.ps1
```

Evaluate existing checkpoints and regenerate aggregate artifacts:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/phase10_eval_all.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/phase10_aggregate.ps1
```

Run robustness evaluation for the best 10% Hadamard checkpoint:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/phase10_noise_sweep.ps1
```

Phase 10 artifacts:

```text
E:/ns_mc_gan_gi/outputs_phase10/phase10_results.csv
E:/ns_mc_gan_gi/outputs_phase10/phase10_results.md
E:/ns_mc_gan_gi/outputs_phase10/PHASE10_REPORT.md
E:/ns_mc_gan_gi/outputs_phase10/paper_examples/
```

The internal high-quality thresholds are used only to decide which claims are allowed in `PHASE10_REPORT.md`. A 10% STL-10 claim requires the completed 10% low-frequency Hadamard row to pass the threshold. A 5% STL-10 claim requires the completed 5% row to pass its threshold. MNIST/Fashion results support only simple-domain claims. If model PSNR/SSIM barely exceeds the Hadamard backprojection, the report must attribute most quality to the low-frequency Hadamard initialization rather than to learned reconstruction.

## Key Modules

- `src/measurement.py`: builds `A`, solves `A A^T + lambda I`, implements
  `data_solution`, `null_project`, and `dc_project` without constructing an
  `n x n` projection matrix.
- `src/models.py`: implements the residual U-Net generator and PatchGAN-style
  WGAN-GP critic.
- `src/losses.py`: implements L1, data consistency, TV, WGAN losses, and
  gradient penalty.
- `src/train.py`: runs the alternating WGAN-GP training loop with validation,
  TensorBoard logging, image grids, and checkpointing.
- `src/eval.py`: loads a checkpoint and reports reconstruction metrics.

## Expected Outputs

Training produces:

```text
E:/ns_mc_gan_gi/outputs/
  sr_005/
    best_ssim.pt
    last.pt
    resolved_config.yaml
    RUN_REPORT.md
    val_metrics_latest.json
    best_metrics.json
    tb/
    samples/
    eval_samples/
```

## Possible Next Improvements

- Add multiple noise samples per measurement and ensemble the reconstructions.
- Add structured ghost-imaging patterns such as Hadamard masks.
- Use AMP mixed precision for faster CUDA training.
- Cache the fixed measurement operator per sampling ratio for reproducibility.
- Add a learned denoising prior or diffusion-style residual sampler.
