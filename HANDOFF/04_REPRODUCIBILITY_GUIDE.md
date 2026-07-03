# 04 — Reproducibility Guide

This document is the single reference for re-running any experiment in this repo. It covers the environment, the import rule, per-stage reproduction commands, critical numeric conventions, and what is deliberately excluded from git and how to regenerate it.

Cross-reference: `HANDOFF/00_START_HERE.md` (program map), `research_lines/*/INDEX.md` (per-stage detail).

---

## 0. ⚠️ Verified working setup — 2026-07-03 (env & data were relocated)

During an E-drive cleanup the **entire `E:/ns_mc_gan_gi/` working tree was moved** into
`E:/GAN_FCC_WORK/data_warehouse/ns_mc_gan_gi/` (the conda env, STL10 data, checkpoints, `outputs_*`,
and `results/cert_package_20260612`). `E:` is an **exFAT** volume, so a junction/symlink back to the
old path is **not possible** (`mklink` → *"Local NTFS volumes are required"*). Use the relocated paths
directly, move the trees back (a same-volume move is instant), or recreate the env from `requirements.txt`.

**Environment verified working (this exact interpreter):**

```
E:/GAN_FCC_WORK/data_warehouse/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311/python.exe
# py 3.11.15 | torch 2.2.1+cu121 | CUDA available | numpy 1.26.4 | lpips / skimage / yaml present
```

Python **≥ 3.10 is required** — the code uses `X | None` union annotations. (The D: `pytorch` conda
env is Python 3.9 and fails to import `src.measurement`; do not use it.)

**Relocated data** (only needed for commands that read STL10, e.g. `regen`/training): `dataset_root`
was `E:/datasets`, now at `E:/GAN_FCC_WORK/datasets/` (also mirrored under
`.../data_warehouse/ns_mc_gan_gi/data/`). Point `dataset_root` there.

**Reproductions actually run and PASSING on this setup (2026-07-03), from the repo root:**

| Stage | Check | Command (`PY` = the interpreter above) | Result |
|---|---|---|---|
| core | range/null projector + checkpoint-wiring theory | `"$PY" -m pytest tests/test_exact_projections.py tests/test_train_checkpoint_wiring.py -q` | **9 passed** |
| 8 (VQGAN) | detail-fusion headline, bit-faithful, **all 3 seeds** | `"$PY" vqgan_detail_fusion.py validate --seeds 0 1 2 --device cuda` | **FAITHFUL=True × 3/3** — 5120 (method,beta,image) rows/seed vs committed CSV; `full_rmse`/`psnr`/`rapsd` max\|Δ\| = 0, `lpips` 4.6e-4 (GPU AlexNet nondeterminism), `relmeaserr` 3.6e-7 |
| 6 (gauge-GAN) | Rad-5 gate report | `"$PY" inspect_gate.py` | all arms **status: PASS** (from committed canonical table; no retrain) |
| 5 (range-null) | feasible-wrong-image evidence | inspect `results/cert_package_20260612/tables/T4_pairs.csv` (relocated tree) | **16/16** cross-class pairs, `RelMeasErr_u_vs_yi` ≈ 2e-15, `hallucination_residual_smaller_than_truth = True` |

`validate` reads cached reconstructions + the committed reference CSV, so it needs **no dataset and no
retraining** — it is the fastest end-to-end proof that the headline result reproduces.

---

## 1. Environment

### 1.1 Canonical Python 3.11 conda env

| Field | Value |
|---|---|
| Env path (original) | `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311` — **moved** (see Section 0) |
| Env path (current, verified) | `E:/GAN_FCC_WORK/data_warehouse/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311/python.exe` |
| Python version | 3.11.15 (≥ 3.10 required) |
| Direct invocation | `"E:/GAN_FCC_WORK/data_warehouse/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311/python.exe" <script>` |

The `-s` flag prevents `.pth` injection from interfering with cwd-relative imports (see Section 2).

### 1.2 Core requirements

File: `requirements.txt` (repo root).

```
torch
torchvision
numpy<2
tqdm
matplotlib
scikit-image
PyYAML
tensorboard
```

Verified project-env versions: `torch==2.2.1+cu121`, `torchvision==0.17.1+cu121` (CUDA 12.1).

### 1.3 Evaluation requirements

File: `eval/requirements-eval.txt`.

```
numpy==1.26.4
scipy==1.13.1
scikit-image==0.24.0
matplotlib==3.9.2
# Use the project-provided torch/torchvision build.
# Verified project env: torch==2.2.1+cu121, torchvision==0.17.1+cu121.
torchmetrics==1.4.2
torch-fidelity==0.3.0
lpips==0.1.4
pytest==8.3.3
```

Install evaluation dependencies on top of the core env:

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 pip install -r eval/requirements-eval.txt
```

### 1.4 Environment sanity check

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.verify_env `
  --dataset_root E:/ns_mc_gan_gi/data `
  --output_dir E:/ns_mc_gan_gi/outputs
```

---

## 2. CWD-Root Import Rule

**Every script in this repo uses cwd-relative imports.** All importable modules (`src/`, `eval/`, root-level `*.py` modules such as `gan_high_quality_gi.py`, `vqgan_detail_fusion.py`, `fcc_diagnostic_canary.py`) live flat at the repo root and are imported as `from src... import`, `from eval...`, or `import gan_high_quality_gi`. This means:

- **Always run from repo root.** Never `cd` into a subdirectory before calling a script.
- **Never move core modules.** `src/`, `eval/`, `configs/`, `paper/`, `colab/`, `scripts/`, `tests/`, `outputs/`, `results/` must remain flat at the root.
- The `-s` flag (`python -s`) prevents user site-packages path injection that can shadow these imports.

Correct invocation pattern:

```powershell
# From E:/ns_mc_gan_gi_code_fcc_phase1 (repo root)
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s <script.py> [args]
# OR for module style:
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.train --config configs/default.yaml
```

---

## 3. Per-Stage Reproduction Commands

The eight stages map directly to `research_lines/00_program_overview/` through `research_lines/08_vqgan_fcc/INDEX.md`. Commands below are the minimal reproduction entry points; consult each `INDEX.md` for full options.

### Stage 1 — NS-MC-GAN Core Platform

Train and evaluate with the default config:

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.train `
  --config configs/default.yaml
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.eval `
  --config configs/default.yaml
```

Key source files: `src/train.py`, `src/eval.py`, `src/models.py`, `src/datasets.py`, `src/losses.py`, `src/measurement.py`, `src/metrics.py`, `src/projections.py`, `src/utils.py`.

### Stage 2 — Operator/Pattern-Learning Exploration

The sweep configs live under `configs/phase3_*.yaml` through `configs/phase8_*.yaml`. Example:

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.train `
  --config configs/phase4_best_5pct.yaml --device cuda
```

### Stage 3 — Hadamard/Rademacher Baselines + Audit Certificate

Probe configs are in `configs/phase9/`. Audit is applied via `eval/audit.py`:

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.eval `
  --config configs/phase9/hadamard5_probe_noise001.yaml
# Post-hoc audit scatter:
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m eval.audit `
  --v <recon.npy> --y <meas.npy> --A <operator.npy> --lambda 0.001 --out audit_plot.png
```

### Stage 4 — Manuscript / Mechanism Construction

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s core_mechanism_figure.py
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s method_diagram_3d.py
```

LaTeX paper: `paper/main.tex`. Figure generators: `paper/figures/*.py`.

### Stage 5 — Range-Null Boundary / Feasible-Wrong-Image Barrier

The canonical result is in `results/sampling_mode_20260612_151210Z/CERTIFICATE_INVARIANCE_RECHECK.json` (already computed). To re-derive:

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s `
  results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py
```

This script requires external checkpoint artifacts under `E:/ns_mc_gan_gi/outputs_phase15/` and `E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import/` (see Section 5 on excluded artifacts). The T4 cross-class feasible-pair evidence (16/16 pairs satisfy wrong record to ~2e-15) is sourced from `E:/ns_mc_gan_gi/results/cert_package_20260612/tables/T4_pairs.csv`, which lives outside this repo in the legacy outputs directory.

### Stage 6 — Gauge-GAN / Rad-5 Auditable Generative Case Study

The locked canonical result is already computed. Inspect it without retraining:

```powershell
# Inspect gate results (no GPU needed):
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s inspect_gate.py
# Canonical results table lives at (external, see Section 5):
# E:/ns_mc_gan_gi/outputs_phase77_auditable_gan_paper_assembly/canonical_results_table.csv
```

To rerun the paper assembly from archived outputs:

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.phase77_final_auditable_gan_paper_assembly
```

Gate config: `gates.yaml`. Gauge-AUC values (Scr-5 = 0.8466, Rad-5 = 0.8771) are in `E:/ns_mc_gan_gi/outputs_phase75_final_high_tier_validation/regime_map_final.csv`.

### Stage 7 — G2R Posterior-Sampling Anti-Collapse (DORMANT)

**Status: dormant / negative result.** Do not restart without explicit decision. Evidence: z-variation diagnostic shows pixel std ~7.19e-4 (collapsed). Scripts: `src/g2r_modec.py`, `src/g2r_modec_train.py`, `src/phase79_rad5_rowspace_diversity_diagnostic.py`.

### Stage 8 — VQGAN/FCC Measurement-Conditioned Detail Fusion

**Locked confirmatory result is final. Do NOT re-run `vqgan_detail_fusion_locked.py`.**

FCC diagnostic canary (re-runnable, ~30 min on GPU):

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s fcc_diagnostic_canary.py all `
  --config configs/compatibility/fcc_diagnostic_canary64.yaml
```

Smoke check (fast):

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s fcc_diagnostic_canary.py all `
  --config configs/compatibility/fcc_diagnostic_canary64_smoke.yaml
```

Development VQGAN detail fusion (re-runnable):

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s vqgan_detail_fusion.py all
```

Regenerate paper figures from locked outputs (no retraining):

```powershell
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s vqgan_detail_fusion_locked_figs.py
```

Paper figures and tables are in `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/`.

Colab cross-rate jobs (dispatched to Google Colab):

```bash
# 2% rate, seed 0
python colab/vqgan_rate02_seed0_job.py
# 10% rate, smoke
python colab/vqgan_rate10_smoke_seed0_job.py
```

---

## 4. Key Conventions and Fixed Constants

These values are locked across all experiments that cite them. Using different values produces incompatible results.

### 4.1 Operator seed: 772001

The mixed-measurement operator (1 DC + 128 DCT + 56 Hadamard + 20 random rows, orthonormalized, `m=205`, 64x64, 5% rate) uses `seed: 772001`. This constant appears in every compatibility-line config under `configs/compatibility/` and is the single canonical operator for Stage 8 (VQGAN) and Stage 6 (GAN). Changing this seed produces a different operator and breaks cross-experiment comparisons.

Source: `configs/compatibility/mc_vqgan_dev_64_5pct.yaml`, `configs/compatibility/gan_high_quality_gi_locked_64_5pct.yaml`, and all `configs/compatibility/anchor_vqgan_inversion_*.yaml`.

### 4.2 Regularization: `lambda_solver` and `lmmse_lambda`

Two regularization parameters control the Tikhonov backprojection and LMMSE baseline respectively.

| Parameter | Role | Value in Stage 8 configs |
|---|---|---|
| `lambda_solver` | Tikhonov ridge for `x_data = A^T(AA^T + lambda I)^{-1} y` and null-space projection | `1.0e-6` (MC-VQGAN/anchor configs) |
| `lmmse_lambda` | Ridge for the LMMSE baseline | `1.0e-3` (MC-VQGAN/anchor configs) |

Stage 6 GAN configs use `lambda_solver: 1.0e-4` and `lmmse_lambda: 1.0e-4`. The early core-platform configs (Stage 1/2) use `lambda_solver: 0.001`. Never mix these across stages.

### 4.3 RelMeasErr — unclipped float64

`RelMeasErr = ||A x_hat - y|| / ||y||` must be computed in **float64** without clipping or clamping. The implementation is in `eval/audit.py::rel_measurement_error`:

```python
v_flat  = flatten_images(v).astype(np.float64)
y_flat  = np.asarray(y, dtype=np.float64).reshape(v_flat.shape[0], -1)
residual = v_flat @ np.asarray(A, dtype=np.float64).T - y_flat
denom   = np.maximum(np.linalg.norm(y_flat, axis=1), 1e-300)
return np.linalg.norm(residual, axis=1) / denom
```

Float32 computation saturates at `k=2` modes and is not valid for modal-contraction verification (Stage 3/5). The locked Stage 8 result reports mean RelMeasErr 3.6e-7 (balanced fusion), which is at the floating-point floor — this is expected and correct.

### 4.4 Raw-SHA256 split deduplication (hash-clean splits)

Stage 8 uses `hash_clean: true` in the data config. The split builder (`src/fcc_canary.py::build_hash_clean_split`) scans raw pixel bytes, hashes each image with SHA-256, discards any hash seen before or in the exclude set, and slices the resulting unique indices into train/val/dev/locked splits. This guarantees:

- No duplicate images across any split (dedup by raw bytes, first occurrence wins).
- The locked split is disjoint by raw hash from all 60,497 previously consumed STL10 images.

The locked split: `locked_source_indices_sha256 = 103976e4...` (from `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/REPRODUCIBILITY_MANIFEST.json`). Operator row digest: `rows_sha256 = 8a16664e...` (same across all 3 reconstruction seeds).

Split sizes (Stage 8 locked): train 20,000 / val 512 / dev 512 / locked 512.

### 4.5 Dataset

STL10, 64x64 grayscale. Dataset root referenced in configs:

- Legacy core-platform configs: `E:/ns_mc_gan_gi/data`
- Compatibility-line and Stage 8 configs: `E:/datasets` (Colab); override to local path in `*_local.yaml` variants

---

## 5. Large Artifacts Excluded from Git

The `.gitignore` excludes:

```
data/
outputs/*
!outputs/.gitkeep
*.pt
*.pth
runs/
```

These directories and file types are never committed. The table below explains where each class of excluded artifact lives and how to regenerate it.

| Artifact class | Location (not in git) | How to regenerate |
|---|---|---|
| Training checkpoints (`*.pt`, `*.pth`) | Vary by stage — see per-stage config `output_dir` | Re-run training command for that stage; or unzip the corresponding result package from `outputs/compatibility/*.zip` |
| Stage 8 VQGAN/anchor checkpoints | `outputs/compatibility/measurement_conditioned_vqgan/{anchor_multiseed_hashclean_seed{0,1,2},anchor_rate02_*,anchor_rate10_*}/` (local) or Colab-artifact ZIPs | Re-run `anchor_initialized_vqgan_inversion.py` with the appropriate `*_local.yaml` config; Colab runs produce `VQGAN_MULTI_SEED_*.zip` packages in `outputs/compatibility/measurement_conditioned_vqgan/` |
| Locked paper package | `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/` (figures, CSVs, JSONs, PDFs) | Regenerate figures only: `python -s vqgan_detail_fusion_locked_figs.py`. Do NOT re-run `vqgan_detail_fusion_locked.py` (one-shot protocol) |
| FCC canary outputs | `outputs/compatibility/fcc_diagnostic_canary64/` | `python -s fcc_diagnostic_canary.py all --config configs/compatibility/fcc_diagnostic_canary64.yaml` |
| GAN canonical result table | `E:/ns_mc_gan_gi/outputs_phase77_auditable_gan_paper_assembly/canonical_results_table.csv` (outside repo) | Re-run `src/phase77_final_auditable_gan_paper_assembly.py`; requires external checkpoint at `E:/ns_mc_gan_gi/outputs_phase15/` and `E:/ns_mc_gan_gi/outputs_phase53C_*` |
| Cert-package evidence tables (T1–T7) | `E:/ns_mc_gan_gi/results/cert_package_20260612/` (outside repo) | Stored in legacy outputs dir; full backup in `GAN_FCC_WORK/data_warehouse`; do not delete |
| Gauge-AUC regime map | `E:/ns_mc_gan_gi/outputs_phase75_final_high_tier_validation/regime_map_final.csv` (outside repo) | Re-run `src/phase75_final_high_tier_validation.py`; requires Phase 73/74 checkpoints |
| Raw STL10 data | `E:/ns_mc_gan_gi/data/` or `E:/datasets/` (outside repo) | Download via `torchvision.datasets.STL10(download=True)` |

**Frozen originals backup:** `GAN_FCC_WORK/data_warehouse` (E:/GAN_FCC_WORK/) holds the complete snapshot. Never `git reset --hard` or `git clean` the worktree; the backup is not a substitute for repo history.

---

## 6. Smoke Tests

Before running any full experiment, verify the environment and imports with the quick sanity checks:

```powershell
# Physics sanity (CPU, ~5 s):
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m src.sanity_physics `
  --config configs/debug.yaml

# Stage 8 smoke (GPU, ~5 min):
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s fcc_diagnostic_canary.py all `
  --config configs/compatibility/fcc_diagnostic_canary64_smoke.yaml

conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s vqgan_detail_fusion.py smoke

# Pytest suite:
conda run -p E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311 python -s -m pytest tests/ -q
```

---

## 7. Red Lines

- Do NOT run `vqgan_detail_fusion_locked.py` again. The locked result is final.
- Do NOT train on the test split. The split guard (`src/split_guard.py`) will raise `SplitViolationError` at dataloader creation if this is attempted.
- Do NOT auto-merge the Stage 8 VQGAN positive draft into the conservative Stage 6 IEEE-TCI main claim.
- Do NOT commit `*.pt`, `*.pth`, `data/`, or `outputs/` — the `.gitignore` blocks most of these, but large result ZIPs under `outputs/compatibility/` are also not committed (covered by `outputs/*`).
- Always label supported vs. forbidden claims; RelMeasErr improvement is NOT a claim of Stage 6/8.
