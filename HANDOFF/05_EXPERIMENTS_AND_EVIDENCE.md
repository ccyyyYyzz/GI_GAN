# 05 — Experiments and Evidence Index

Per-research-line evidence index with verified file paths and canonical numbers.
All paths are relative to repo root (`E:/ns_mc_gan_gi_code_fcc_phase1`) unless noted otherwise.
Numbers are read directly from on-disk artifacts; invented numbers are forbidden.

---

## Notation

- **RelMeasErr** = `||A x_hat - y|| / ||y||`; unclipped float64 convention throughout.
- **Locked** = one-shot evaluation on a brand-new raw-hash-disjoint split (never re-run).
- **Dev** = mechanical development scoring on a separate 512-image hash-clean split.
- **cert_package** files live in the sibling repo `E:\ns_mc_gan_gi\results\cert_package_20260612\`
  and are cross-referenced here through `docs/core_experiments/canonical_numbers.csv`
  and `docs/core_experiments/claim_evidence_matrix.csv`.

---

## Line 3 — Hadamard/Rademacher Baselines + Audit Certificate (Stages 9–17)

### What was shown

A single test-time measurement audit is pluggable to any reconstructor.
After auditing, relative measurement error drops 3–4 orders of magnitude with negligible PSNR change.

### Canonical numbers (from `docs/core_experiments/canonical_numbers.csv`, rows sourced from `cert_package_20260612`)

| Reconstructor | Operator | PSNR (post-audit, dB) | RelMeasErr pre-audit | RelMeasErr post-audit |
|---|---|---|---|---|
| BP pipeline | Rad-5 5% | 7.2969 | 5.17e-05 | 3.04e-09 |
| TV-PGD best | Rad-5 5% | 8.4933 | 1.31e-02 | 8.55e-07 |
| main_rad5 (GAN, already audited) | Rad-5 5% | 22.316 | 3.78e-05 | 2.15e-09 |
| main_scr5 (GAN, already audited) | Scr-5 5% | 22.272 | 5.51e-03 | 5.50e-06 |
| main_rad10 (GAN, already audited) | Rad-10 10% | 24.781 | 5.87e-05 | 7.61e-09 |
| main_scr10 (GAN, already audited) | Scr-10 10% | 24.730 | 5.71e-03 | 5.71e-06 |

Full T1 post-hoc audit table: `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T1_posthoc_external.csv`
(18 audit rows, all pass; max |dPSNR| = 0.039 dB).

### Key scripts (repo root)

- `eval/audit.py`, `eval/checker.py`, `eval/metrics.py` — audit and metrics infrastructure
- `results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py` — invariance recheck

---

## Line 5 — Range-Null Boundary + Feasible-Wrong-Image Barrier (Stages 48–60)

### What was shown

Measurement consistency (`A x_hat = y`) does not imply semantic correctness.
Exact null-space perturbations yield cross-class images that satisfy a wrong measurement record to floating-point precision.

### Canonical numbers

| Table | Finding | Source |
|---|---|---|
| T4_pairs.csv | 16/16 cross-class pairs satisfy wrong record to ~2e-15 | `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T4_pairs.csv` |
| T3_contraction | Modal contraction matches `lambda/(lambda+sigma^2)` in float64; k=2 saturates in float32 | `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T3_contraction_summary.csv` |
| T6/T7 | Wrong-y collapse 12.2–14.8 dB; shuffle 14.5–17.0 dB; 5% A-drift destroys contraction | `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T6_dependence.csv`, `T7_adrift.csv` |

Claim evidence matrix entry A6: "Certificate boundary: feasible cross-class images can satisfy y."
Status: sufficient; no additional runs needed.
See `docs/core_experiments/claim_evidence_matrix.csv` row A6 and
`E:\ns_mc_gan_gi\results\cert_package_20260612\REPORT.md` for narrative.

### Key scripts

- `results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py`
- `results/sampling_mode_20260612_151210Z/CERTIFICATE_INVARIANCE_RECHECK.json`

---

## Line 6 — Gauge-GAN / Rad-5 Auditable Generative Case Study (Stages 69–83)

### What was shown

A GAN prior repositioned as an auditable generative example:
LPIPS/RAPSD improve at 5% sampling while measurement accountability is maintained,
and the gauge diagnostic removes a known nuisance shortcut.

### Gauge-AUC gate results (from `docs/core_experiments/canonical_numbers.csv`, rows B4)

Source artifact: `E:\ns_mc_gan_gi\outputs_phase75_final_high_tier_validation\regime_map_final.csv`

| Operator/regime | Gauge AUC | Decision |
|---|---|---|
| Scr-5 (5%) | **0.8466** | Strong gate — cGAN run |
| Rad-5 (5%) | **0.8771** | Strong gate — cGAN run |
| Scr-10 (10%) | 0.6240 | Weak gate — stopped |
| Rad-10 (10%) | 0.6396 | Weak gate — stopped |

### Shortcut stress test (claim B2)

Source: `E:\ns_mc_gan_gi\outputs_phase75_final_high_tier_validation\SHORTCUT_STRESS_TEST_REPORT.md`

- Standard row delta: **0.4767**
- Gauge row delta: **0.0**

Gauge equalization removes the nuisance shortcut entirely.

### Canonical GAN quality table (claim B1, B3)

Source: `E:\ns_mc_gan_gi\outputs_phase77_auditable_gan_paper_assembly\canonical_results_table.csv`
Cross-referenced in `docs/core_experiments/canonical_numbers.csv` rows auditable-GAN.

Selected entries (mean over 3 paired seeds):

| Row label | PSNR (dB) | SSIM | LPIPS | RAPSD dist | RelMeasErr |
|---|---|---|---|---|---|
| Scr-5 B (standard) | 22.262 | 0.6266 | 0.2349 | 0.005387 | 0.005561 |
| Scr-5 C_gauge | 22.257 | 0.6281 | 0.2308 | 0.005214 | 0.005567 |
| Scr-5 D_standard | 22.258 | 0.6281 | 0.2310 | 0.005224 | 0.005567 |
| Rad-5 A | 22.062 | 0.6247 | 0.2244 | 0.004928 | 4.59e-05 |
| Rad-5 B | 22.304 | 0.6293 | 0.2344 | 0.005597 | 3.49e-05 |
| Rad-5 C | 22.277 | 0.6314 | 0.2283 | 0.005085 | 3.65e-05 |

B/C/D quality is close on Scr-5 (claim B3: gauge offers safety without performance cost).
LPIPS and RAPSD descriptively improve relative to the audited GAN anchor (22.316 dB, RelMeasErr ~2e-09).

### Alpha trust knob (claim B6)

RelMeasErr span when sweeping alpha:
- Rad-5: **4.37e-09** (measurement invariant)
- Scr-5: **1.74e-07** (measurement invariant)

Source: `E:\ns_mc_gan_gi\outputs_phase76_high_upside_auditable_gan_exploration\reports\ALPHA_TRUST_SHARPNESS_REPORT.md`

### Key scripts (repo root)

- `gan_high_quality_gi.py` — main GAN training/eval entry
- `gan_gauge_aligned_nsgan.py` — gauge-aligned variant
- `gan_high_quality_gi_matched.py` — matched-control experiment
- `src/phase69A_gauge_gan_signal_diagnostic.py`
- `src/phase71_gauge_cgan_paired_seeds.py`
- `src/phase73_overnight_gauge_gan_expansion.py`
- `src/phase75_final_high_tier_validation.py`
- `src/phase77_final_auditable_gan_paper_assembly.py`
- `inspect_gate.py`, `gates.yaml`

Outputs: `outputs/compatibility/gan_high_quality_gi/`

---

## Line 7 — G2R Posterior Sampling Anti-Collapse Side Line (Dormant)

### What was shown

Negative result: z-variation diagnostic failed. The generator collapsed rather than producing
null-space diversity under fixed y.

### Evidence

| Metric | Value | Source |
|---|---|---|
| Pixel std (mean over z draws) | ~7.19e-04 | `docs/core_experiments/canonical_numbers.csv` row B7 |
| Gate outcome | 3/6 gates at step 20000 (G-DIV FAIL, G-MEAN FAIL, G-PERC FAIL) | `results/g2r_pilot_phase3/PHASE3_REPORT.md` |
| Decision | `z_collapsed_not_viable` | claim_evidence_matrix.csv row B7 |

Status: DORMANT / ABANDON. No further runs planned. Archived in `results/g2r_pilot_phase3/`.

### Key scripts

- `src/g2r_modec.py`, `src/g2r_modec_train.py`
- `src/phase79_rad5_rowspace_diversity_diagnostic.py`
- `scripts/g2r/`

---

## Line 8 — VQGAN/FCC Measurement-Conditioned Detail Fusion (This Repo Focus)

### 8A. VQGAN Null-Space Detail Fusion — Locked Confirmation

#### Setup

- Task: 64×64 grayscale STL10, bucket measurement `y = Ax`, `n = 4096`, `m = 205` (5.0% sampling)
- Operator seed 772001, rows SHA-256 = `8a16664e…`; noiseless acquisition
- Splits (raw-SHA256 deduplicated): train 20,000 / val 512 / dev 512 / locked 512
- Locked split overlap with 60,497 previously consumed STL10 raw hashes: **0**
- Locked source indices SHA-256: `103976e4…`
- 3 random seeds; frozen operating point B selected on val only, never on dev/locked
- Balanced B values: `{seed0: 0.55, seed1: 0.55, seed2: 0.50}`
- Quality-lite B values: `{seed0: 0.75, seed1: 0.75, seed2: 0.70}`
- Pre-registered acceptance gate: 8 conditions fixed before locked split was touched

#### Locked main result — balanced fusion vs VQAE (from `FACTS.json`)

Source: `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/FACTS.json`

| Metric | Delta (balanced - VQAE) | Interpretation |
|---|---|---|
| LPIPS | **-0.0977** (CI [-0.1016, -0.0940]) | **32.6% relative gain**; CI entirely < 0 |
| PSNR | -0.45 dB | Within pre-registered 0.5 dB tolerance |
| full-RMSE | +0.0039 | Within pre-registered 0.005 tolerance |
| RAPSD | -0.00030 | Improves (not worse) |
| RelMeasErr mean / max | 3.6e-07 / 5.7e-07 | Numerical precision; gate threshold 1e-05 |
| Seeds same direction | 3/3 | All seeds agree on LPIPS direction |

Gate result: **8/8 conditions PASS**

Dev vs locked replication:
- Dev dLPIPS = -0.0965 (32.9% gain), dPSNR = -0.43 dB
- Locked dLPIPS = -0.0977 (32.6% gain), dPSNR = -0.45 dB
- Effect sizes nearly coincide — confirmed replication on unseen data.

#### Locked perception-distortion ladder — absolute method means (512 images, mean over 3 seeds)

Source: `FACTS.json` → `locked_method_means`

| Method | LPIPS | PSNR (dB) | full-RMSE | SSIM | RAPSD |
|---|---|---|---|---|---|
| LMMSE anchor (x0) | 0.404 | 22.80 | 0.076 | 0.629 | 0.0041 |
| VQAE (B=0) | 0.300 | 23.13 | 0.073 | 0.657 | 0.0030 |
| **Balanced fusion** | **0.202** | 22.69 | 0.077 | 0.633 | 0.0027 |
| Quality-lite | 0.182 | 22.21 | 0.081 | 0.609 | 0.0024 |
| Full VQGAN (B=1) | 0.172 | 21.43 | 0.089 | 0.571 | 0.0019 |

All five operating points form a monotone perception-distortion ladder.

#### Dataset-level KID (from `locked_kid.json`)

Source: `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/locked_kid.json`

| Method | KID |
|---|---|
| VQAE | 0.1190 |
| **Balanced fusion** | **0.0433** |
| Quality-lite | 0.0322 |
| Full VQGAN | 0.0260 |

Balanced fusion improves KID 2.7× over VQAE (0.119 → 0.043).

#### Cross-rate generalization (development, not a second locked claim)

Source: `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/rate_generalization.json`

| Sampling rate | dLPIPS | Relative LPIPS gain | dPSNR (dB) | Seeds same direction |
|---|---|---|---|---|
| 2% (m=82) | -0.116 | 29.3% | -0.39 | 3/3 |
| 5% (m=205, locked) | -0.098 | 32.6% | -0.45 | 3/3 |
| 10% (m=410) | -0.076 | 34.2% | -0.43 | 3/3 |

Advantage holds at every rate from 2% to 10%; relative gain grows mildly with sampling rate.

#### Key scripts

- `vqgan_detail_fusion.py` — main fusion pipeline (`python vqgan_detail_fusion.py all`)
- `vqgan_detail_fusion_locked.py` — locked one-shot evaluation
- `measurement_conditioned_vqgan.py` — MC-VQGAN prior
- `anchor_initialized_vqgan_inversion.py` — anchor-initialized refiner
- `mc_vqgan_prior_long_canary.py` — prior canary
- `experiments_rate_fusion.py` — cross-rate experiments
- `experiments_local.py` — local development experiments
- `method_diagram_3d.py`, `core_mechanism_figure.py` — mechanism figures

#### Output artifacts

- `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md` — full draft
- `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.tex` / `.pdf`
- `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/FACTS.json` — machine-readable canonical numbers
- `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/locked_kid.json` — KID table
- `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/rate_generalization.json` — cross-rate table
- `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/CLAIM_EVIDENCE_LEDGER.md`
- `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/VQGAN_FUSION_PAPER_PACKAGE.zip` — submission bundle

Colab configs: `configs/compatibility/anchor_vqgan_inversion_rate{02,05,10}_seed{0,1,2}.yaml`

---

### 8B. FCC Row-Null Diagnostic Canary (64×64)

#### What was shown

The FCC critic achieves real-pair retrieval (Recall@1 = 1.0, 32× over random),
but deployable scalar/sum-image baselines explain the separation on nuisance-balanced negatives.
No row-null mutual information is certified beyond scalar/artifact signal.

#### Canonical classification: `ONLY_SCALAR_OR_ARTIFACT_SIGNAL`

Source: `outputs/compatibility/fcc_diagnostic_canary64/reports/FACTS.json`
(git commit `760873e8`, created 2026-06-27)

| Key value | Number |
|---|---|
| Layer A Recall@1 | 1.0000 (random = 0.0312; 32.0× over random) |
| Label-shuffle Recall@1 | 0.0000 (correctly near random) |
| FCC balanced AUC | 0.9917 |
| Best deployable balanced AUC (pair_logistic) | 0.9987 |
| FCC minus deployable balanced AUC | **-0.0070** (FCC does NOT exceed deployable) |
| Balance feature SMD max | 1.48 (target: 0.25 — negatives not fully balanced) |
| `fcc_exceeds_deployable` check | **False** |
| `structural_balanced_arm` check | **False** |
| `negatives_well_balanced` check | **False** |

Operator: Rademacher, m=205 (5%), SHA-256 = `75f9818b…`
Splits: train 2048 / val 512 / dev 512 (hash-clean, consumed hashes excluded).
Layer C (generated-candidate transfer) gated OFF per protocol.

Interpretation: real (r, n) pairs are retrievable, but separability is attributable to
naturalness/scalar nuisance statistics rather than certified row-null mutual information.
No generator selection should be justified from this result.

#### Key scripts

- `fcc_diagnostic_canary.py` — canary runner
  (`python fcc_diagnostic_canary.py all --config configs/compatibility/fcc_diagnostic_canary64.yaml`)
- `src/fcc_canary.py` — canary implementation
- `structure_detail_fcc.py` — FCC structure/detail diagnostic

#### Output artifacts

- `outputs/compatibility/fcc_diagnostic_canary64/reports/FINAL_REPORT.md`
- `outputs/compatibility/fcc_diagnostic_canary64/reports/FACTS.json`
- `outputs/compatibility/fcc_diagnostic_canary64/reports/classification.json`
- `outputs/compatibility/fcc_diagnostic_canary64/config_used.yaml`
- `configs/compatibility/fcc_diagnostic_canary64.yaml` (canonical config)

---

## Cross-line evidence registry

| Claim ID | Short claim | Status | Primary artifact |
|---|---|---|---|
| A1 | Quality and accountability are separable | sufficient | `T1_posthoc_external.csv` (cert_package) |
| A3 | Post-hoc audit reduces RelMeasErr across reconstructors | sufficient | `T1_posthoc_external.csv` |
| A4 | Modal contraction verified in float64 | sufficient | `T3_contraction_summary.csv` |
| A5 | Audit catches wrong-y / shuffle / A-drift | sufficient | `T6_dependence.csv`, `T7_adrift.csv` |
| A6 | Feasible-wrong-image barrier: 16/16 cross-class pairs satisfy wrong record to ~2e-15 | sufficient | `T4_pairs.csv` |
| A7 | Measurement consistency != semantic correctness | sufficient | `cert_package_20260612/REPORT.md` |
| B1 | GAN prior improves perceptual metrics at 5% | sufficient (descriptive; not quality dominance) | `canonical_results_table.csv` |
| B2 | Gauge equalization removes shortcut | sufficient | `SHORTCUT_STRESS_TEST_REPORT.md` |
| B3 | Standard cGAN comparable; gauge adds safety | sufficient | Phase75 aggregate |
| B4 | Gauge-AUC: Scr-5 0.8466, Rad-5 0.8771 (strong); Scr/Rad-10 ~0.62–0.64 (weak) | sufficient | `regime_map_final.csv` |
| B6 | Alpha trust knob is measurement-invariant | sufficient | `ALPHA_TRUST_SHARPNESS_REPORT.md` |
| B7 | G2R z-sampling collapsed (negative) | DORMANT / ABANDON | `results/g2r_pilot_phase3/PHASE3_REPORT.md` |
| VQGAN-main | Balanced fusion LPIPS -0.0977 / 32.6%, 8/8 gate PASS, 3/3 seeds, locked | sufficient | `FACTS.json`, `PAPER_DRAFT.md` |
| VQGAN-KID | KID 0.119→0.043 (2.7×), locked | sufficient | `locked_kid.json` |
| VQGAN-rate | Cross-rate 29–34% gain, 3/3 seeds, 2/5/10% | development only, not locked | `rate_generalization.json` |
| FCC-canary | ONLY_SCALAR_OR_ARTIFACT_SIGNAL | sufficient (canary complete) | `fcc_diagnostic_canary64/reports/FACTS.json` |

Full claim-evidence matrix: `docs/core_experiments/claim_evidence_matrix.csv`
Full canonical numbers registry: `docs/core_experiments/canonical_numbers.csv`

---

## Forbidden / out-of-scope claims

The following claims are explicitly forbidden and must not appear in any manuscript
derived from this repo (see `docs/core_experiments/claim_evidence_matrix.csv` rows C1–C4
and `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/CLAIM_EVIDENCE_LEDGER.md`):

- Discriminator certificate / exact-null critic (C1, C2): unsafe provenance.
- G1 sampling success (C3): post-GAN checkpoint missing; test-set adaptation risk.
- SOTA / beats diffusion / hardware validation (C4): no benchmark evidence.
- Null-space texture certified as real scene content: the bucket certifies `A x_hat = y`,
  not null-space truth.
- FCC certifies row-null mutual information in the VQGAN context: canary result is
  `ONLY_SCALAR_OR_ARTIFACT_SIGNAL`.
- VQGAN uniformly better than VQAE at no distortion cost: balanced mode incurs bounded
  PSNR/RMSE cost; worst-case failures concentrate on periodic/man-made structures.
