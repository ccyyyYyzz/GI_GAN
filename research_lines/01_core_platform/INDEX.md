# Stage 1 — NS-MC-GAN Core Platform

**Research question:** Separate what the measurement determines (row-space) from what the prior fills in (null-space).

**Theory:** Given forward model `y = A x + eps` with `m << n`, the data solution is
`x_data = A^T (A A^T + lambda I)^-1 y`.  The residual lives in `null(A)` and is unconstrained by `y`.
This stage formalises `range(A^T)` vs `null(A)`, implements the projection/audit machinery, and provides
the shared generator/critic/loss/metric infrastructure that all later stages build on.

---

## Core Files (kept flat at repo root — do NOT move)

All imports use `cwd = repo root`. The files below are in `src/` and must remain there.

| File | Role |
|---|---|
| `src/train.py` | Main training loop: generator/critic updates, loss orchestration, checkpoint saves |
| `src/eval.py` | Evaluation loop: loads checkpoint, runs metrics, saves sample images |
| `src/models.py` | Generator (U-Net style, `ConvBlock` + GroupNorm + LeakyReLU) and discriminator definitions; `build_generator()` factory |
| `src/datasets.py` | DataLoader factories: `get_dataloaders()`, `get_val_dataloader()` |
| `src/losses.py` | All loss terms: `reconstruction_loss`, `data_consistency_loss`, `generator_adversarial_loss`, `discriminator_wgan_loss`, `gradient_penalty`, `total_variation_loss`, frequency/Charbonnier/Sobel/SSIM losses |
| `src/measurement.py` | `GhostMeasurementOperator`, `LearnableGhostMeasurementOperator`, `LearnablePatternBank`; Hadamard construction; `StaleSolverCacheError` guard |
| `src/projections.py` | `ExactProjectorInfo` (SHA-256-locked); row-space and null-space projectors; shape helpers |
| `src/utils.py` | Shared utilities (seeding, logging, checkpoint I/O, misc helpers) |

**Config:**

| File | Role |
|---|---|
| `configs/default.yaml` | Canonical hyperparameter baseline: `img_size=64`, `sampling_ratio=0.05`, `pattern_type=rademacher`, `epochs=50`, `batch_size=64`, `lr_g/lr_d=0.0002`, `lambda_l1=100.0`, `lambda_dc_loss=10.0`, `lambda_adv=0.01`, `lambda_gp=10.0` |

---

## Reproduction

Run from repo root (env: `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311`):

```bash
# Training
python -m src.train --config configs/default.yaml

# Evaluation
python -m src.eval --config configs/default.yaml
```

`src/train.py` imports from sibling modules via relative imports (e.g. `from .datasets import get_dataloaders`).
All scripts must be launched as `python -m src.<module>` from repo root to resolve the package correctly.

---

## Key Design Decisions

- **Row/null decomposition is exact:** `src/projections.py` SHA-256-locks the operator `A` so the projector cannot silently drift between train and eval.
- **Data-consistency loss is a soft audit during training:** `lambda_dc_loss=10.0` penalises `||A x_hat - y||` but does not replace the hard test-time audit (see `03_baselines_audit`).
- **No test-split contamination:** `src/datasets.py` enforces the train/val/test split guard; `limit_val_samples=1000` is the only size cap exposed in the config.
- **Operator type is pluggable:** `measurement.py` supports fixed Rademacher/Hadamard patterns and learnable pattern banks; the config key `pattern_type` selects the mode.

---

## Relationships to Other Stages

| Stage | Dependency on this platform |
|---|---|
| `02_operator_pattern_learning` | Extends `measurement.py` with learnable operators; reuses train/eval loop |
| `03_baselines_audit` | Adds `eval/audit.py` on top of `src/eval.py`; test-time certificates use `src/projections.py` |
| `06_gauge_gan_rad5` | Gauge-GAN generator is the `src/models.py` generator with Rad-5 illumination config |
| `08_vqgan_fcc` | VQGAN inversion + FCC diagnostics import `src/measurement.py` and `src/projections.py` |

See `research_lines/00_program_overview/INDEX.md` for the full stage map.
