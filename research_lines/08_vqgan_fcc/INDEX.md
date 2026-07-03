# Stage 8 — VQGAN/FCC Measurement-Conditioned Detail Fusion

**Status:** LOCKED confirmatory result obtained. Independent VQGAN detail-fusion draft. Do NOT auto-merge into the conservative IEEE-TCI main claim (Stage 6).

**Related plan doc (relocated here):** `research_lines/08_vqgan_fcc/phase1_2_plan.md`

---

## 1. Research Question

Can a measurement-conditioned VQGAN/VQAE prior fuse null-space detail into low-rate ghost imaging reconstructions without ever violating the measurement constraint? Do FCC row-null / structure-detail diagnostics reveal a learnable compatibility signal beyond deployable scalar baselines?

---

## 2. Theory

The bucket measurement `y = A x` fixes only the row-space component `P_R x = A† y`; the null space `P_0 x` (where `P_0 = I - A† A`, `A P_0 = 0`) is entirely unconstrained and carries most perceptual detail.

**Fusion identity (exact, for any B):**

```
x_hat_B = x0 + P0( d_A + B (d_G - d_A) )
d_A = P0(x_A - x0),  d_G = P0(x_G - x0)
```

Because `A P_0 = 0` and `A x0 = y`, we have `A x_hat_B = y` exactly for every value of `B`. The scalar `B` is a free perceptual dial: `B=0` recovers the VQAE structure reconstruction, `B=1` recovers the full VQGAN, and intermediate `B` trades distortion for perception without any measurement penalty.

This is an **auditable** example of prior-driven null-space filling (same theoretical framing as Stage 6). Measurement consistency certifies `A x_hat ≈ y`, NOT that injected texture is the true scene.

---

## 3. Core Files

All paths are relative to the repo root. Run everything from repo root with cwd imports.
Env: `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311/python.exe`.

### 3.1 Primary scripts

| File | Role |
|---|---|
| `vqgan_detail_fusion.py` | Development pipeline: build operator, run VQAE/VQGAN branches, fuse, score dev/val splits, select B |
| `vqgan_detail_fusion_locked.py` | **One-shot locked confirmatory scoring** — do not re-run; result is archived |
| `vqgan_detail_fusion_locked_figs.py` | Generate paper figures from locked outputs |
| `measurement_conditioned_vqgan.py` | Measurement-conditioned VQGAN prior (MC-VQGAN) module |
| `anchor_initialized_vqgan_inversion.py` | Anchor-initialized latent refiner that drives decoded images to measurement consistency |
| `mc_vqgan_prior_long_canary.py` | Long-run canary for MC-VQGAN prior stability |
| `fcc_diagnostic_canary.py` | FCC row-null compatibility diagnostic (standalone, gated multi-stage) |
| `structure_detail_fcc.py` | Structure-detail FCC diagnostic variant |
| `experiments_rate_fusion.py` | Cross-rate generalization experiments (2%, 5%, 10% sampling) |
| `experiments_local.py` | Local-only experiment runner for dev/smoke |
| `method_diagram_3d.py` | Generates 3-D mechanism figure (METHOD_DIAGRAM_3D.{png,pdf,svg}) |
| `md_to_pdf.py` | Converts PAPER_DRAFT.md to PAPER_DRAFT.pdf via LaTeX |

### 3.2 Configs

`configs/compatibility/` contains all Stage 8 configs. Key entries:

| Config | Purpose |
|---|---|
| `mc_vqgan_smoke.yaml` | Quick MC-VQGAN smoke check |
| `mc_vqgan_dev_64_5pct.yaml` | MC-VQGAN 64px 5% dev run |
| `mc_vqgan_prior_multiseed_hashclean_seed{0,1,2}.yaml` | Multi-seed hash-clean prior training |
| `anchor_vqgan_inversion_multiseed_hashclean_seed{0,1,2}.yaml` | Anchor refiner (Colab) |
| `anchor_vqgan_inversion_rate02_seed{0,1,2}.yaml` | Cross-rate 2% (Colab) |
| `anchor_vqgan_inversion_rate10_seed{0,1,2}.yaml` | Cross-rate 10% (Colab) |
| `*_local.yaml` variants | Local-only mirrors of the above |
| `fcc_diagnostic_canary64.yaml` | Canonical FCC canary config (64px) |
| `structure_detail_fcc.yaml` | Structure-detail FCC config |

### 3.3 Colab scripts

`colab/` contains detached job scripts for Google Colab runs:

- `vqgan_rate02_seed{0,1,2}_job.py` / `_detach.py` — 2% rate jobs
- `vqgan_rate10_seed{0,1,2}_job.py` / `_detach.py` — 10% rate jobs
- `vqgan_rate{02,10}_smoke_seed{0,1,2}_job.py` — smoke versions
- `vqgan_rate_colab_job_common.py` — shared Colab job boilerplate

---

## 4. Paper Package (Outputs)

All paper outputs live under:
`outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/`

| Artifact | Description |
|---|---|
| `PAPER_DRAFT.md` / `.tex` / `.pdf` | Full paper draft (abstract through conclusions, 10 sections) |
| `FINAL_PROJECT_STATUS.md` | One-page canonical result summary |
| `FACTS.json` | Machine-readable numbers for all locked and dev results |
| `CLAIM_EVIDENCE_LEDGER.md` | Claim-by-claim evidence table (all 11 claims PASS) |
| `LIMITATIONS_AND_NEGATIVE_RESULTS.md` | Explicit scope bounds and negative branches |
| `REVIEWER_STRESS_TEST.md` | Pre-emptive reviewer Q&A |
| `REPRODUCIBILITY_MANIFEST.json` | Operator/split/checkpoint/script SHA-256 manifest |
| `MAIN_TABLE.csv` / `.tex` | Locked perception-distortion table |
| `PARETO_FIGURE.{png,pdf}` | B-sweep Pareto ladder figure |
| `B_CURVE.{png,pdf}` | LPIPS vs B curve |
| `METHOD_DIAGRAM.{png,pdf,svg}` | Method schematic |
| `METHOD_DIAGRAM_3D.{png,pdf,svg}` | 3-D mechanism figure |
| `CORE_MECHANISM_FIGURE.{png,pdf,svg}` | Core row/null-space mechanism figure |
| `QUALITATIVE_GRID.{png,pdf}` | Qualitative side-by-side grid |
| `NOISE_ROBUSTNESS.{png,pdf}` | Measurement-noise robustness sweep |
| `rate_generalization_figure.{png,pdf}` | Cross-rate generalization figure |
| `rate_generalization.json` | Cross-rate numeric results |
| `locked_kid.json` | Locked KID by arm |
| `b_curve.csv` | B-sweep LPIPS/PSNR/RMSE values |
| `noise_sweep.csv` | Noise-sweep LPIPS values |
| `EXTRA_RESULTS.md` | Supplementary results (noise, KID, cross-rate, ablation) |
| `VQGAN_FUSION_PAPER_PACKAGE.zip` + `_SHA256.txt` | Submission zip + hash |
| `_bundle/` | Colab smoke/run scripts for reproducing rate jobs |

---

## 5. Locked Result

**Classification: `LOCKED_BALANCED_VQGAN_FUSION_CONFIRMED`**
Gate: **8/8 conditions PASS**. No blockers.

### 5.1 Setup

- Task: low-rate ghost imaging `y = A x`, STL10 64x64 grayscale (n=4096 px)
- Operator: m=205 rows, 5.0% sampling (1 DC + 128 low-freq DCT + 56 low-sequency Hadamard + 20 random, orthonormalized), seed 772001, noiseless
- Operator digest: `rows_sha256 = 8a16664e...` (identical across all reconstruction seeds)
- 3 seeds; nothing retrained between seeds — seeds vary only the stochastic latent refiner
- Hash-clean splits (raw-SHA256 dedup): train 20,000 / val 512 / dev 512 / locked 512
- Locked split: brand-new, raw-hash disjoint from union of 60,497 previously consumed STL10 hashes (overlap = 0, intra-duplicates = 0; `locked_source_indices_sha256 = 103976e4...`)
- Frozen B (balanced): seed0=0.55, seed1=0.55, seed2=0.50 — selected on val only, never on dev/locked
- Frozen B (quality-lite): seed0=0.75, seed1=0.75, seed2=0.70

### 5.2 Locked perception-distortion ladder (absolute means, n=512 images)

| Method | LPIPS | PSNR (dB) | full-RMSE | SSIM | RAPSD | RelMeasErr |
|---|---|---|---|---|---|---|
| LMMSE anchor (x0) | 0.4036 | 22.80 | 0.07565 | 0.6292 | 0.00408 | ~1e-7 |
| VQAE (B=0) | 0.2997 | 23.13 | 0.07296 | 0.6574 | 0.00303 | ~1e-7 |
| **Balanced fusion** | **0.2020** | **22.69** | **0.07686** | **0.6328** | **0.00273** | **3.6e-7** |
| Quality-lite | 0.1822 | 22.21 | 0.08128 | 0.6088 | 0.00240 | ~1e-7 |
| Full VQGAN (B=1) | 0.1723 | 21.43 | 0.08913 | 0.5708 | 0.00189 | ~1e-7 |

Lower LPIPS/RMSE/RAPSD is better; higher PSNR/SSIM is better. Moving VQAE → balanced → full-VQGAN monotonically trades distortion for perception.

### 5.3 Balanced fusion vs VQAE (primary locked effect)

| Metric | Delta | CI / note |
|---|---|---|
| LPIPS | **-0.0977** (32.6% relative gain) | CI [-0.1016, -0.0940]; all CI upper < 0 |
| PSNR | -0.45 dB | Within 0.5 dB pre-registered tolerance |
| full-RMSE | +0.0039 | Within 0.005 pre-registered tolerance |
| RAPSD | -0.00030 | Improves (not worse) |
| Seeds same direction | **3/3** | Per-seed dLPIPS: seed0 -0.1027, seed1 -0.0936, seed2 -0.0967 |
| RelMeasErr (mean/max) | 3.6e-7 / 5.7e-7 | Floating-point floor; `A x_hat = y` exact |

Development effect (512 dev imgs): dLPIPS = -0.0965 (32.9% gain), dPSNR = -0.43 dB — nearly coincides with locked, confirming replication on unseen data.

### 5.4 Locked KID (dataset-level, lower is better)

| Arm | KID |
|---|---|
| VQAE | 0.119 |
| Balanced fusion | **0.043** (2.7x reduction) |
| Quality-lite | 0.032 |
| Full VQGAN | 0.026 |

Source: `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/locked_kid.json`

### 5.5 Cross-rate generalization (development-level, 3 seeds/rate)

Rate-agnostic priors reused unchanged; only anchor refiner retrained at each rate.

| Sampling rate | ΔLPIPS (bal − VQAE) | Rel. LPIPS gain | ΔPSNR (dB) | Seeds same-direction |
|---|---|---|---|---|
| 2% (m=82) | -0.116 | 29.3% | -0.39 | 3/3 |
| 5% (m=205, **locked**) | -0.098 | 32.6% | -0.45 | 3/3 |
| 10% (m=410) | -0.076 | 34.2% | -0.43 | 3/3 |

Source: `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/rate_generalization.json`

### 5.6 Measurement-noise robustness (locked split, mean over 3 seeds)

| noise std | VQAE LPIPS | Balanced LPIPS | VQGAN LPIPS |
|---|---|---|---|
| 0.000 | 0.300 | 0.202 | 0.172 |
| 0.005 | 0.299 | 0.199 | 0.172 |
| 0.010 | 0.297 | 0.195 | 0.176 |
| 0.020 | 0.295 | 0.197 | 0.204 |
| 0.050 | 0.304 | 0.250 | 0.293 |

Source: `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/EXTRA_RESULTS.md`

---

## 6. FCC Diagnostic Canary

The FCC (Feasibility-Consistency Compatibility) diagnostic asks whether the row-space skeleton `r = P_R x` and null-space content `n = P_0 x` carry a learnable compatibility signal that exceeds *deployable* scalar/nuisance baselines.

**Result: `ONLY_SCALAR_OR_ARTIFACT_SIGNAL`**

The canary ran to completion (build → train → eval → classify). The classification indicates real-pair retrieval exists but deployable nuisance baselines explain it, or FCC does not exceed them on balanced negatives. This is a **negative/boundary result**: FCC is exploratory (appendix-level only); it does NOT constitute a proven success. See also Stage 5 (range-null barrier) for related consistency-vs-correctness evidence.

**Repro:**
```
E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe fcc_diagnostic_canary.py all \
  --config configs/compatibility/fcc_diagnostic_canary64.yaml
```

Relevant files: `fcc_diagnostic_canary.py`, `src/fcc_canary.py`, `configs/compatibility/fcc_diagnostic_canary64.yaml`, `configs/compatibility/fcc_diagnostic_canary64_smoke.yaml`, `structure_detail_fcc.py`, `configs/compatibility/structure_detail_fcc.yaml`

---

## 7. Reproduction Commands

Run all commands from repo root. No training needed for the main result — checkpoints are frozen.

### Development pipeline (select B, score dev/val)
```
E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe vqgan_detail_fusion.py all
```

### Regenerate paper figures from locked outputs (no re-scoring)
```
E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe vqgan_detail_fusion_locked_figs.py
```

### Regenerate 3-D mechanism figure
```
E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe method_diagram_3d.py
```

### FCC canary (standalone, full pipeline)
```
E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe fcc_diagnostic_canary.py all \
  --config configs/compatibility/fcc_diagnostic_canary64.yaml
```

### Cross-rate experiments (development-level)
```
E:\ns_mc_gan_gi\conda_envs\ns_mc_gan_gi_py311\python.exe experiments_rate_fusion.py
```

### DO NOT re-run locked scoring
`vqgan_detail_fusion_locked.py` was executed once (one-shot protocol). The locked result is archived in `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/FACTS.json` and `REPRODUCIBILITY_MANIFEST.json`. Re-running would not produce new evidence and risks confusing the audit trail.

---

## 8. Checkpoint Hashes (from REPRODUCIBILITY_MANIFEST.json)

| Seed | VQAE prior | VQGAN prior | VQAE refiner | VQGAN refiner |
|---|---|---|---|---|
| seed0 | `9e0bb903...` | `0069abc9...` | `266364c3...` | `0ae40b16...` |
| seed1 | `af4e0e7f...` | `7e519b56...` | `728467f3...` | `15ec9e77...` |
| seed2 | `7c4b7815...` | `55ab20b1...` | `68cf2918...` | `205b7930...` |

Script hashes: `vqgan_detail_fusion.py` = `c964dddd...`, `vqgan_detail_fusion_locked.py` = `cb8fa91d...`

---

## 9. Scope and Red Lines

**Supported claims:**
- Balanced null-space fusion improves LPIPS by 32.6% (CI [-0.1016, -0.0940]) vs VQAE at -0.45 dB PSNR cost, 3/3 seeds, 8/8 gate PASS, on brand-new raw-hash-disjoint locked split.
- Exact measurement consistency: `A x_hat_B = y` for any B (RelMeasErr ~1e-7, floating-point floor).
- KID improves 2.7x (0.119 → 0.043) at balanced operating point.
- 29–34% relative LPIPS gain generalizes across 2–10% sampling (dev-level, 3/3 seeds each rate).
- The simplest fusion (single global scalar B) beat frequency-band and learned-gate variants in development.

**Forbidden claims / out of scope:**
- The bucket does NOT certify injected VQGAN texture as the true scene; consistency ≠ null-space truth.
- B is NOT a per-image oracle; it is a fixed validation-selected operating point.
- FCC is NOT a proven success; result is `ONLY_SCALAR_OR_ARTIFACT_SIGNAL` (exploratory/appendix only).
- VQGAN is NOT uniformly better than VQAE; it trades PSNR/RMSE for LPIPS. Failures concentrate on man-made periodic/edge structures (fences, vehicle panels, airplane): worst case +0.07/+0.04 LPIPS on 13/512 images.
- Result does NOT prove all low-sampling tasks benefit.
- Do NOT auto-merge this VQGAN positive draft into the conservative Stage 6 (IEEE-TCI) main claim.

**Next step:** Human pre-submission review only — English polish, venue formatting, read-through against `CLAIM_EVIDENCE_LEDGER.md`. No further automated experiments. Do not change B, do not re-lock, do not add ablations.

---

## 10. Sibling Index Docs

- `research_lines/00_program_overview/INDEX.md` — full program map, unifying theory
- `research_lines/05_range_null_barrier/INDEX.md` — consistency ≠ correctness; feasible-wrong-image barrier
- `research_lines/06_gauge_gan_rad5/INDEX.md` — Stage 6 gauge-GAN (conservative main claim, do not merge)
- `research_lines/03_baselines_audit/INDEX.md` — audit certificate backbone
