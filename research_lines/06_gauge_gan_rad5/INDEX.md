# Stage 6 — Gauge-GAN / Rad-5 Auditable Generative Case Study

**Phases 69–83 | CURRENT MAIN LINE**

## Research question

Under the measurement certificate, can a GAN prior improve perceptual and
spectral metrics while preserving accountability, and can a gauge diagnostic
expose the shortcut boundary?

## Theoretical position

The GAN is repositioned as an **auditable generative-prior example**, not a
quality-SOTA claim. Any reconstruction x_hat satisfies the measurement equation
A x_hat = y after a test-time audit; the prior determines only what fills the
null space P_0 x_hat. The gauge diagnostic (Scr-5 / Rad-5 AUC) tests whether
the discriminator's signal comes from image quality or from a leakable
measurement shortcut. A paired shortcut stress test verifies that gauge
equalization removes the shortcut signal while quality is preserved.

## Primary source files

All paths are relative to the repo root
(`E:/ns_mc_gan_gi_code_fcc_phase1`). Code runs with `cwd=repo root`.

| File | Role |
|------|------|
| `gan_high_quality_gi.py` | Core GAN training loop, measurement loss, audit projection, split guards |
| `gan_gauge_aligned_nsgan.py` | Gauge-aligned NS-GAN: `FeaturePatchDiscriminator`, `GaugeDiscriminatorBundle`, paired full+residual arms |
| `gan_high_quality_gi_matched.py` | Matched-pair evaluation scaffold (METHOD_NO_GAN / METHOD_GAN) |
| `inspect_gate.py` | Quick gate report reader — reads `outputs/compatibility/measurement_conditioned_vqgan/anchor_initialized_seed0_hashclean/reports/gate_report.json` |
| `gates.yaml` | Pre-registered admissibility gates (G-CAL / G-DIV / G-NVR / G-MEAN / G-CERT / G-PERC / G-PROTO) registered 2026-06-12 on branch `g2r-protocol` before any training |

## Phase modules in `src/`

| Module | Purpose |
|--------|---------|
| `src/phase69A_gauge_gan_signal_diagnostic.py` | First gauge signal diagnostic (Scr-5 checkpoint, cert cache, AUC probe) |
| `src/phase69B_controlled_gauge_cgan_pilot.py` | Controlled cGAN pilot; matched eval scaffold |
| `src/phase69B_compute_lpips.py` | LPIPS computation helper |
| `src/phase70_gauge_gan_paper_expansion.py` | Paper-expansion sweep; Phase69B repro check |
| `src/phase71_gauge_cgan_paired_seeds.py` | Paired-seed Scr-5 delta metrics |
| `src/phase72_scr10_gauge_cgan_regime_validation.py` | Scr-10 regime validation (weak-gate confirmation) |
| `src/phase73_overnight_gauge_gan_expansion.py` | Overnight Rad-5 seed sweep; `rad5_seed_metrics.csv` source |
| `src/phase74_high_tier_gauge_cgan_pack.py` | Scr-5 seed01 standard comparison |
| `src/phase75_final_high_tier_validation.py` | Canonical aggregate; gauge-AUC gate; shortcut stress test; regime map |
| `src/phase76_high_upside_auditable_gan_exploration.py` | Alpha-trust sweep; unmeasured-content map; failure-detector AUC |
| `src/phase77_auditable_gan_paper_assembly.py` | Paper assembly v1 |
| `src/phase77_final_auditable_gan_paper_assembly.py` | Final canonical paper assembly — reads Phase73 + Phase75 CSVs; writes `canonical_results_table.csv` |
| `src/phase78_96px_rad5_one_seed_probe.py` | 96 px Rad-5 one-seed probe |
| `src/phase79_96px_rad5_p0_error_validation.py` | P0 error validation at 96 px (negative result) |
| `src/phase79_rad5_rowspace_diversity_diagnostic.py` | Row-space diversity diagnostic (negative: collapsed z) |
| `src/phase80_rad5_centered_diversity_calibration.py` | Diversity calibration follow-up |
| `src/phase81_96px_rad5_paper_completion.py` | 96 px completion pass |

## Canonical result numbers

Source: `docs/core_experiments/canonical_numbers.csv` (rows tagged
`auditable GAN`). All numbers are from external output directories under
`E:/ns_mc_gan_gi/` (the training workspace; read-only from this repo).

### Gauge-AUC gate (regime map)

Source file: `outputs_phase75_final_high_tier_validation/regime_map_final.csv`

| Regime | Gauge-AUC | Decision |
|--------|-----------|----------|
| Scr-5  | **0.8466** | strong gate; proceed |
| Rad-5  | **0.8771** | strong gate; proceed |
| Scr-10 | 0.6240    | weak gate; no cGAN / stop |
| Rad-10 | 0.6396    | weak gate; no cGAN / stop |

### Shortcut stress test (Phase 75)

Source file: `outputs_phase75_final_high_tier_validation/SHORTCUT_STRESS_TEST_REPORT.md`

| Arm | Shortcut-stress row delta |
|-----|--------------------------|
| standard cGAN | 0.4767 |
| gauge cGAN | 0.0 |

Claim B2 (supported): gauge equalization removes the shortcut signal.

### Canonical GAN metrics table (3 seeds, Scr-5 and Rad-5)

Source file: `outputs_phase77_auditable_gan_paper_assembly/canonical_results_table.csv`

**Scr-5 arms (Phase75 `standard_cgan_seed_metrics.csv`)**

| Arm | PSNR | SSIM | LPIPS | RAPSD dist | RelMeasErr (unclipped) |
|-----|------|------|-------|-----------|------------------------|
| B | 22.262 | 0.6266 | 0.2349 | 5.39e-03 | 5.56e-03 |
| C_gauge | 22.257 | 0.6281 | 0.2308 | 5.21e-03 | 5.57e-03 |
| D_standard | 22.258 | 0.6281 | 0.2310 | 5.22e-03 | 5.57e-03 |

**Rad-5 arms (Phase73 `rad5_seed_metrics.csv`)**

| Arm | PSNR | SSIM | LPIPS | RAPSD dist | RelMeasErr (unclipped) |
|-----|------|------|-------|-----------|------------------------|
| A | 22.062 | 0.6247 | 0.2244 | 4.93e-03 | 4.59e-05 |
| B | 22.304 | 0.6293 | 0.2344 | 5.60e-03 | 3.49e-05 |
| C | 22.277 | 0.6314 | 0.2283 | 5.09e-03 | 3.65e-05 |

RelMeasErr convention: **unclipped float64** (see `gates.yaml` G-CERT and
`docs/core_experiments/method_conventions.md`). Clipping to [0,1] is for
display/PSNR only.

### Alpha trust knob (Phase 76)

Source: `outputs_phase76_high_upside_auditable_gan_exploration/reports/ALPHA_TRUST_SHARPNESS_REPORT.md`

| Regime | RelMeasErr span across alpha sweep |
|--------|------------------------------------|
| Rad-5  | 4.37e-09 |
| Scr-5  | 1.74e-07 |

The alpha knob controls the weight of prior-supplied detail; the certificate
(RelMeasErr) is invariant across the sweep to sub-nanoscale precision.
Claim B6 (supported).

### P0 error validation — negative result (Phase 79)

Source: `outputs_phase79_96px_rad5_p0_error_validation/PHASE79_P0_ERROR_VALIDATION_REPORT.md`

| Arm | Pooled Spearman |
|-----|-----------------|
| B (96 px Rad-5) | 0.0733 |
| C (96 px Rad-5) | 0.0719 |

The high-|P_0 x_hat| pixel-error diagnostic is not supported at 96 px.
Reported as a limitation; not used in main claims.

## Supported and forbidden claims

Supported claims (see `docs/core_experiments/supported_claims.md`):

| ID | Summary |
|----|---------|
| B1 | GAN/prior descriptively improves LPIPS/RAPSD in 5% regimes; not a quality-dominance claim |
| B2 | Gauge equalization removes shortcut (stress delta 0.4767 → 0.0) |
| B3 | Standard cGAN is comparable; gauge adds safety without quality cost |
| B4 | Gauge-AUC: 5% strong (0.8466 / 0.8771), 10% weak (0.6240 / 0.6396) |
| B5 | Unmeasured-content map visualizes prior-supplied content (h vs high-freq proxy corr ~0.53; with caveats) |
| B6 | Alpha knob changes detail while certificate stays invariant (RelMeasErr span < 2e-07) |
| B8 | Failure-detector AUC 0.6366 — weak/preliminary only (supp/future) |

Forbidden claims (see `docs/core_experiments/unsupported_forbidden_claims.md`):

| ID | Summary |
|----|---------|
| B7 | z-sampling diversity — abandoned; pixel std ~7.19e-4, collapsed (see Stage 7) |
| C1 | Discriminator certificate — forbidden; Pi_y^lambda is the certificate |
| C4 | SOTA / beats diffusion / hardware validation — forbidden; no benchmark evidence |

## Reproduction (no retrain)

All evidence is already computed. Do not retrain.

```bash
# Inspect the canonical numbers CSV (training workspace):
# E:/ns_mc_gan_gi/outputs_phase77_auditable_gan_paper_assembly/canonical_results_table.csv

# Inspect the regime map / gauge-AUC gate:
# E:/ns_mc_gan_gi/outputs_phase75_final_high_tier_validation/regime_map_final.csv

# Cross-reference with the in-repo canonical numbers:
python -c "
import csv
rows = list(csv.DictReader(open('docs/core_experiments/canonical_numbers.csv')))
gan_rows = [r for r in rows if r['domain'] == 'auditable GAN']
for r in gan_rows:
    print(r['row_label'], r['metric'], r['value'])
"
```

To verify gate report (VQGAN anchor seed 0):
```bash
python inspect_gate.py
```

Run from repo root (`E:/ns_mc_gan_gi_code_fcc_phase1`) with env
`E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311`.

## Cross-references

- **Stage 3** (`research_lines/03_baselines_audit/`): Hadamard/Rademacher baselines and the post-hoc audit certificate that this stage's GAN must satisfy.
- **Stage 5** (`research_lines/05_range_null_barrier/`): Feasible-wrong-image barrier — why consistency (A x_hat = y) does not certify null-space content.
- **Stage 7** (`research_lines/07_g2r_posterior_sampling/`): G2R posterior-sampling side branch (negative result: z collapsed, diversity not viable).
- **Stage 8** (`research_lines/08_vqgan_fcc/`): VQGAN/FCC detail-fusion subline (independent draft; do not auto-merge into this conservative IEEE-TCI claim).
- `docs/core_experiments/canonical_numbers.csv` — authoritative number registry for all stages.
- `docs/core_experiments/supported_claims.md` / `unsupported_forbidden_claims.md` — claim evidence matrix.
