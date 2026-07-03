# Stage 2 — Operator / Pattern-Learning Exploration (Phases 3–8)

**Status:** Historical / supporting. All experiments are completed; outputs are archived in
`E:/ns_mc_gan_gi/outputs_phase{3..8}/`. No retraining is required.

**Research question:** Do metric gains come from learning the illumination operator /
measurement patterns, or from the choice of reconstruction prior?

---

## 1. Theory

Under the forward model `y = A x + eps`, the row-space component `P_R x = A† y` is fixed by
the measurement regardless of the pattern bank used. Phases 3–8 vary **only** `A` (how patterns
are formed) while holding the GAN prior fixed. The attribution question — operator vs prior —
remained ambiguous at the end of Phase 8, which motivated the audit / certificate line
(Stage 3).

Key pattern modes explored (defined in `src/train.py` via `LearnablePatternBank`):

| Mode string | Description |
|---|---|
| `learned_binary_ste` | Binary patterns via straight-through estimator (STE) |
| `learned_flip_aware_binary_ste` | STE with flip-margin control and warm-up schedule |
| `learned_continuous` | Unconstrained continuous differential patterns |
| `fixed_rademacher` / `fixed_hadamard` | Non-trainable structured baselines |

---

## 2. Source Modules (all run from repo root)

These are **shared-platform** modules in `src/`; no dedicated per-phase `.py` file
exists for phases 3–8. The pattern-learning logic is parameterised entirely through configs.

| File | Role |
|---|---|
| `src/train.py` | Training loop; reads `use_learned_patterns`, `pattern_mode`, `flip_*` keys |
| `src/eval.py` | Evaluation; enforces binary thresholding for STE and flip-aware modes |
| `src/models.py` | Generator / discriminator architectures |
| `src/measurement.py` | Forward operator `A`, solver, data-consistency projection |
| `src/losses.py` | Adversarial + measurement-domain losses |
| `src/pattern_regularizers.py` | Energy, decorrelation, binary-penalty losses |
| `src/pattern_utils.py` | Pattern grid savers, stats JSON |
| `src/pattern_diagnostics.py` | Before/after pattern-change visualisation |
| `src/calibrate_operator_equivalence.py` | Post-hoc check that learned A matches fixed A |
| `src/convert_continuous_to_binary_checkpoint.py` | Convert a continuous checkpoint to binary |
| `src/eval_pattern_swap.py` | Swap pattern banks between checkpoints at eval time |
| `src/eval_paired_controls.py` | Paired operator vs prior ablation evaluation |

Per-phase aggregation modules (collect metrics from output dirs, emit CSV):

| File | Phases covered |
|---|---|
| `src/aggregate_phase3.py` | Phase 3 |
| `src/aggregate_phase4.py` | Phase 4 |
| `src/aggregate_phase5.py` | Phase 5 |
| `src/aggregate_phase6.py` | Phase 6 |
| `src/aggregate_phase7.py` | Phase 7 |
| `src/aggregate_phase8.py` | Phase 8 (cross-phase comparison) |
| `src/aggregate_phase8_hq.py` | Phase 8 high-quality runs |

---

## 3. Configs

All configs live in `configs/` and are consumed by `src/train.py` and `src/eval.py`.

### Phase 3 — Binary illumination baseline + ablations (9 files)
```
configs/phase3_learned_binary_{2,5,10}pct.yaml
configs/phase3_learned_continuous_5pct.yaml
configs/phase3_binary_5pct_no_decorrelation.yaml
configs/phase3_binary_5pct_no_energy.yaml
configs/phase3_binary_5pct_no_secrip.yaml
configs/phase3_debug_{binary,continuous}_5pct.yaml
```

### Phase 4 — Matched / frozen binary operator (9 files)
```
configs/phase4_matched_binary_5pct.yaml
configs/phase4_matched_binary_no_freeze_5pct.yaml
configs/phase4_matched_binary_slow_5pct.yaml
configs/phase4_continuous_{contrast,to_binary}_5pct.yaml
configs/phase4_best_{2,5,10}pct.yaml
configs/phase4_debug_matched_binary_5pct.yaml
configs/phase4_matched_binary_no_freeze_5pct_resume.yaml
```

### Phase 5 — Exact-measurement binary and extreme undersampling (9 files)
```
configs/phase5_exact_binary_{,freezeG_,slow_}5pct.yaml
configs/phase5_calibrate_exact_5pct.yaml
configs/phase5_centered_vs_exact_5pct.yaml
configs/phase5_{extreme_0p5,fixed_0p5}pct.yaml
configs/phase5_best_{1,2,5,10}pct.yaml
```

### Phase 6 — Pattern-only and generator-only fine-tuning (7 files)
```
configs/phase6_{g_only_finetune,soft_signed_train}_5pct.yaml
configs/phase6_pattern_only_alpha1_5pct.yaml
configs/phase6_pattern_trainable_alpha{0p5,1,2,6}_5pct.yaml
```

### Phase 7 — Continuous and flip-aware patterns (7 files)
```
configs/phase7_continuous_{g_only,long,pattern_only,physical}_5pct.yaml
configs/phase7_flipaware_{aggressive,alpha0p5,alpha1}_5pct.yaml
```
`phase7_flipaware_alpha1_5pct.yaml` uses `pattern_mode: learned_flip_aware_binary_ste` with
`flip_threshold: 0.5`, `flip_warmup_epochs: 3`, and balanced row flipping.

### Phase 8 — Wide architecture, MNIST domain, quality audit (8 flat + subdir)
```
configs/phase8_{fixed_wide,fixed_wide_refiner,continuous_physical_wide,continuous_g_only_wide}_5pct.yaml
configs/phase8_{direct_y_fixed,mnist_fixed,mnist_continuous}_5pct.yaml
configs/phase8_quality_audit_5pct.yaml
configs/phase8_hq/   ← 10 files: {cifar10_gray,fashion_mnist,mnist,...}_{hq_{5,10}pct}.yaml
```

---

## 4. Scripts

### PowerShell (`.ps1`) — Windows local runs
Located in `scripts/`; call `src/train.py` or `src/eval.py` via `python -m src.train --config`.

| Script | Purpose |
|---|---|
| `scripts/train_phase3_{5pct,binary_sweep,pattern_ablation}.ps1` | Phase 3 training |
| `scripts/eval_phase3_{5pct,binary_sweep,pattern_ablation}.ps1` | Phase 3 eval |
| `scripts/phase3_debug.ps1` | Quick debug run |
| `scripts/train_phase4_{best_sweep,tuning_5pct}.ps1` | Phase 4 training |
| `scripts/eval_phase4_{best_sweep,tuning_5pct}.ps1` | Phase 4 eval |
| `scripts/aggregate_phase4.ps1` | Phase 4 CSV aggregation |
| `scripts/phase4_{debug,continuous_to_binary}.ps1` | Phase 4 utilities |
| `scripts/train_phase5_{best_sweep,tuning_5pct}.ps1` | Phase 5 training |
| `scripts/eval_phase5_{best_sweep,tuning_5pct}.ps1` | Phase 5 eval |
| `scripts/aggregate_phase5.ps1` | Phase 5 CSV aggregation |
| `scripts/phase5_{calibrate,noise_sweep}.ps1` | Phase 5 utilities |
| `scripts/train_phase6_{controls,long,multiseed}.ps1` | Phase 6 training |
| `scripts/eval_phase6_controls.ps1` | Phase 6 eval |
| `scripts/aggregate_phase6.ps1` | Phase 6 CSV aggregation |
| `scripts/phase6_paired_eval.ps1` | Operator-vs-prior paired eval |
| `scripts/train_phase7_{continuous,flipaware}.ps1` | Phase 7 training |
| `scripts/eval_phase7_{continuous,flipaware}.ps1` | Phase 7 eval |
| `scripts/aggregate_phase7.ps1` | Phase 7 CSV aggregation |
| `scripts/phase7_{measurement_quality,pattern_swap}.ps1` | Phase 7 diagnostics |
| `scripts/train_phase8_strong_5pct.ps1` | Phase 8 training |
| `scripts/eval_phase8_strong_5pct.ps1` | Phase 8 eval |
| `scripts/aggregate_phase8.ps1` | Phase 8 CSV aggregation |
| `scripts/phase8_{quality_audit,mnist_sanity}.ps1` | Phase 8 utilities |
| `scripts/phase8_hq_{train,eval}_{core,domain_sanity}.ps1` | Phase 8 HQ subset |

### Shell (`.sh`) — Colab / Linux
`scripts/eval_5pct.sh`, `scripts/train_5pct.sh`, `scripts/sweep_sampling.sh` are generic
wrappers that were also used for phases 3–8 sweeps.

---

## 5. Reproduction

All scripts are **historical** — the outputs already exist. To re-run a specific phase:

```bash
# From repo root, using the canonical py311 env:
conda activate E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311

# Example: re-run phase 3 binary 5 pct
python -m src.train --config configs/phase3_learned_binary_5pct.yaml

# Example: eval + aggregate phase 7
python -m src.eval  --config configs/phase7_flipaware_alpha1_5pct.yaml
python -m scripts.aggregate_phase7
```

Do **not** retrain unless explicitly requested. Canonical archived results live in
`E:/ns_mc_gan_gi/outputs_phase{3..8}/` (outside this repo; not committed to git).

---

## 6. Key Finding and Forward Reference

No phase in the 3–8 range yielded a clear attribution of metric gains to the operator versus
the prior. The best-performing learned-binary runs (Phase 4 `matched_binary_no_freeze`,
Phase 7 `flipaware_alpha1`) showed marginal PSNR / SSIM improvements over fixed Rademacher,
but test-set measurement-error (`rel_meas_err`) was not consistently controlled.

This ambiguity motivated:

- **Stage 3** (`../03_baselines_audit/`) — Hadamard/Rademacher structured baselines with a
  pluggable test-time audit certificate that disentangles row-space accountability from
  null-space reconstruction quality.
- **Stage 1** (`../01_core_platform/`) — the shared `src/` modules that phases 3–8 all depend on.
