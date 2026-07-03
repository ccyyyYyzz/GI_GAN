# 06 Papers and Claims

Two manuscripts share the ghost-imaging research program in this repo.
They target different venues, differ in their core claim, and **must not be merged**.

---

## 1. Conservative IEEE-TCI Paper (main line)

**File:** `paper/main.tex`

**Title:** *Measurement Auditing for Learned Ghost Imaging: Certificates, Limits, and Prior-Supplied Content*

**Target venue:** IEEE Transactions on Computational Imaging (IEEEtran journal class).

**Abstract summary (from the source):** A range-null decomposition of the sensing operator separates measurement accountability from image quality. A plug-in test-time audit `Pi_y^lambda(v) = v - B_lambda(Av - y)` with `B_lambda = A^T(AA^T + lambda I)^{-1}` is attached after any reconstructor and provably contracts the measurement residual by `lambda / (lambda + sigma_i^2)` per singular mode. Consistency with the bucket record is not correctness. A GAN branch appears only as a representative auditable prior, not a quality-SOTA result.

**Core contributions (as stated in `paper/main.tex` §1):**
1. Range-null separability: the measurement `y = Ax` constrains `A(P_R x)` and never constrains `A(P_0 x)`.
2. Plug-in audit `Pi_y^lambda` with proven singular-mode contraction.
3. Certificate boundary: the audit certifies measurement agreement, not null-space correctness.
4. GAN branch as representative generative prior under the audit (not a benchmark comparison).

**Key numerical anchors** (sources: `paper/materials_inventory.md`; primary tables in `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\`):

| Regime | Learned PSNR | BP PSNR | RelMeasErr pre | RelMeasErr post |
|---|---:|---:|---:|---:|
| Rad-5 | 22.316 dB | 7.297 dB | 3.78e-05 | 2.15e-09 |
| Scr-5 | 22.271 dB | 14.310 dB | 5.51e-03 | 5.50e-06 |
| Rad-10 | 24.781 dB | 7.756 dB | 5.87e-05 | 7.61e-09 |
| Scr-10 | 24.730 dB | 14.533 dB | 5.71e-03 | 5.71e-06 |

Post-audit PSNR moves at most +0.039 dB on trained outputs (see ablation rows in `paper/materials_inventory.md` §3).
Feasible wrong images satisfy the wrong measurement record to ~2e-15 (T4_pairs.csv, 16/16 pairs).

**Source tables** (all live in `E:\ns_mc_gan_gi\`, not in this repo's working tree — read-only):
- `results/cert_package_20260612/tables/T1_posthoc_external.csv` — post-hoc audit on BP/TV/learned
- `results/cert_package_20260612/tables/T2_sweep.csv` — range-share sweep
- `results/cert_package_20260612/tables/T3_contraction_summary.csv` — modal contraction
- `results/cert_package_20260612/tables/T4_pairs.csv` — feasible wrong images
- `results/cert_package_20260612/tables/T6_dependence.csv` — wrong-y / shuffle dependence
- `results/cert_package_20260612/tables/T7_adrift.csv` — operator-drift robustness
- `outputs_phase77_auditable_gan_paper_assembly/canonical_results_table.csv` — GAN canonical table

**Scr-5 LPIPS lock:** Multiple protocol-specific values exist (Phase69B: 0.2263; Phase71 seeds: 0.229-0.231; Phase75 canonical: 0.2308). Paper must use one locked source. Current project lock points to Phase75 canonical aggregate. Requires human confirmation before final manuscript numbers are inserted. Mixing phases will introduce inconsistent LPIPS values — see `paper/materials_inventory.md` §Special Check.

---

## 2. VQGAN Positive Sibling Draft

**File:** `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md`

**Title:** *Measurement-Consistent VQGAN Detail Fusion for Low-Rate Ghost Imaging*

**Status:** Independent positive-result draft. Do NOT auto-merge into the conservative IEEE-TCI main claim.

**Core claim:** Fusing the null-space contribution of an adversarially trained VQGAN prior into a measurement-audited LMMSE anchor improves perceptual quality while satisfying `A x_hat_B = y` exactly by construction, for every fusion weight `B`. The fusion rule is `x_hat_B = x_0 + P_0(d_A + B(d_G - d_A))`. Because `A P_0 = 0` and `A x_0 = y`, measurement consistency is a theorem, not a penalty term.

**Locked primary result** (5.0% sampling, 64x64 STL10, n=4096, m=205, 512-image brand-new raw-hash-disjoint split, 3 seeds):

| Method | LPIPS | PSNR (dB) | Full-RMSE | SSIM | RAPSD |
|---|---:|---:|---:|---:|---:|
| LMMSE anchor (x_0) | 0.404 | 22.80 | 0.076 | 0.629 | 0.0041 |
| VQAE (B=0) | 0.300 | 23.13 | 0.073 | 0.657 | 0.0030 |
| **Balanced fusion** | **0.202** | 22.69 | 0.077 | 0.633 | 0.0027 |
| Quality-lite | 0.182 | 22.21 | 0.081 | 0.609 | 0.0024 |
| Full VQGAN (B=1) | 0.172 | 21.43 | 0.089 | 0.571 | 0.0019 |

- Delta LPIPS balanced vs VQAE: -0.0977 (CI [-0.1016, -0.0940]), a **32.6% relative gain**.
- PSNR cost: -0.45 dB (within 0.5 dB pre-registered tolerance).
- RMSE cost: +0.0039 (within 0.005 pre-registered tolerance).
- RAPSD improves (spectral fidelity not degraded).
- 3/3 seeds agree in direction; 8/8 pre-registered gate conditions PASS.
- RelMeasErr: mean 3.6e-07, max 5.7e-07 (numerical-precision measurement consistency).
- KID: VQAE 0.119 -> balanced 0.043 (2.7x reduction).

**Cross-rate generalization** (development level only, not locked):

| Rate | Delta LPIPS | Relative gain | Delta PSNR | Seeds |
|---|---:|---:|---:|---:|
| 2% (m=82) | -0.116 | 29.3% | -0.39 dB | 3/3 |
| 5% (m=205, locked) | -0.098 | 32.6% | -0.45 dB | 3/3 |
| 10% (m=410) | -0.076 | 34.2% | -0.43 dB | 3/3 |

**Operator identity:** rows_sha256 = 8a16664e..., seed 772001, 1 DC + 128 DCT + 56 Hadamard + 20 random, orthonormalized.

**Locked split provenance:** raw-hash disjoint from 60,497 previously consumed STL10 hashes; overlap = 0, intra-duplicates = 0; `locked_source_indices_sha256 = 103976e4...`.

---

## 3. Supported Claims

Sources: `docs/core_experiments/supported_claims.md`; `paper/materials_inventory.md`.

| ID | Claim | Status | Key numbers | Evidence |
|---|---|---|---|---|
| A1 | Image quality and bucket accountability are separable. | main paper | 18/18 audit rows pass; max abs dPSNR 0.039 dB; RelMeasErr 3-4 orders lower post-audit | T1_posthoc_external.csv |
| A2 | Range-share law explains when audit affects PSNR. | main paper | PSNR flat over lambda grid; Rad/Scr contraction differs by spectrum/range scaling; formula DeltaPSNR_max = -10 log10(1-s) | T2_sweep.csv, T5_rho.csv |
| A3 | Post-hoc audit reduces RelMeasErr across reconstructors (BP, Tikhonov, CS-TV, learned). | main paper | BP/TV/learned all reduce residual by orders of magnitude; dPSNR ≤ 0.039 dB | T1_posthoc_external.csv |
| A4 | Modal contraction verified: float64 k=1 matches lambda/(lambda+sigma^2). | main / supp | float64 k=1: max mode dev 1.044e-10 (Rad-5), 2.286e-12 (Scr-5). f32 k=2 saturates. | T3_contraction_summary.csv |
| A5 | Audit catches wrong-y, shuffle, A-drift accountability failures. | main / supp | Wrong-y PSNR drop 12.2-14.8 dB; shuffle 14.5-17.0 dB; 5% A-drift destroys contraction | T6_dependence.csv, T7_adrift.csv |
| A6 | Certificate boundary: feasible cross-class images can satisfy y. | main paper | 16/16 cross-class pairs satisfy wrong record to ~2e-15; constructed image RelMeasErr 2.16e-15 to 4.00e-15 on Rad rows, 0.000 on Scr rows | T4_pairs.csv |
| A7 | Measurement consistency is not semantic correctness. | main paper | Feasible wrong images + POCS/box caveat | T4_pairs.csv, REPORT.md |
| B1 | GAN/prior improves perceptual details in 5% regimes. | main / supp | Scr-5 and Rad-5 LPIPS/RAPSD improve descriptively. Not a quality-dominance claim. No RelMeasErr improvement claim. | canonical_results_table.csv |
| B2 | Gauge equalization removes measurement-row shortcut. | main / supp | standard row delta 0.4767; gauge row delta 0.0 | SHORTCUT_STRESS_TEST_REPORT.md |
| B3 | Standard cGAN is comparable; gauge offers safety without performance cost. | main paper | B/C/D Scr-5 quality close in Phase75 aggregate. Do not claim gauge quality dominance. | standard_vs_gauge_decision.md |
| B4 | Gauge-AUC diagnostic: Scr-5 0.8466, Rad-5 0.8771 (strong); Scr-10 0.6240, Rad-10 0.6396 (weak, stop). | main / supp | 5% strong gate; 10% weak gate explains run/stop decisions | regime_map_final.csv |
| B5 | Unmeasured-content map visualizes prior-supplied content. | main / supp with caveats | Scr-5 h vs high-frequency proxy corr ~0.53. NOT proof of false hallucination. Pixelwise error localization is negative (see forbidden). | UNMEASURED_CONTENT_MAP_REPORT.md |
| B6 | Alpha trust knob changes prior detail while certificate stays invariant. | main paper | RelMeasErr span Rad-5 4.37e-09, Scr-5 1.74e-07 | ALPHA_TRUST_SHARPNESS_REPORT.md |
| B8 | Failure detector is weak/preliminary. | supp / future | Best AUC 0.6366 on artificial labels. Not deployable OOD detector. | FAILURE_DETECTOR_AUC_REPORT.md |
| V1 | VQGAN null-space fusion improves LPIPS 32.6% (locked) with 8/8 gate PASS. | VQGAN draft | Delta LPIPS -0.0977 CI [-0.1016, -0.0940]; 3/3 seeds; 8/8 gate; RelMeasErr ~3.6e-07 | PAPER_DRAFT.md §5 |
| V2 | Fusion satisfies A x_hat_B = y exactly for every B (theorem + numerical confirmation). | VQGAN draft | Proof: A x_0 = y and A P_0 = 0; empirical max RelMeasErr 5.7e-07 | PAPER_DRAFT.md §3.4 |
| V3 | KID improves 2.7x: VQAE 0.119 -> balanced 0.043. | VQGAN draft | Locked split (512 images) | PAPER_DRAFT.md §5 |
| V4 | Cross-rate 29-34% relative LPIPS gain, 3/3 seeds, 2-10% rates. | VQGAN draft (dev only) | Development-level; not a second locked claim | PAPER_DRAFT.md §8.1 |

---

## 4. Forbidden / Unsupported Claims

Sources: `docs/core_experiments/unsupported_forbidden_claims.md`.

| ID | Claim | Status | Reason |
|---|---|---|---|
| B7 | z-sampling gives posterior diversity / uncertainty. | Abandon | pixel std mean ~7.19e-04; decision z_collapsed_not_viable. G2R/Mode-C is a NEGATIVE result. Source: Z_VARIATION_DIAGNOSTIC_REPORT.md |
| C1 | The discriminator is the certificate. | Abandon / Forbidden | Pi_y^lambda is the certificate. The discriminator is not a measurement-accountability certificate. Source: PHASE53C_FAILED_CLAIMS.md |
| C2 | Exact-null critic succeeded. | Abandon / Archive | AUC values archived only; unsafe provenance/semantics. Source: PHASE53C_AGGREGATE_REPORT.md |
| C3 | G1 sampling succeeded. | Abandon / Archive | Post-GAN checkpoint missing; test-set adaptation risk. Source: results/sampling_mode_.../REPORT.md |
| C4 | SOTA / beats diffusion models / hardware validation. | Forbidden | No broad benchmark, no diffusion-model comparison, no hardware evidence exists in this repo. Source: FORBIDDEN_CLAIMS_FINAL.md |
| — | Bucket measurement certifies null-space texture is real. | Forbidden | Core claim of the work is the opposite. A P_0 = 0 means null-space content is invisible to the bucket. Consistency != correctness (T4_pairs.csv; PAPER_DRAFT.md §9). |
| — | Per-image oracle B. | Forbidden | B is a fixed validation-selected operating point, not a per-image or learned quantity. Per-image gating did not beat the global scalar in development (PAPER_DRAFT.md §8.4). |
| — | Unmeasured-content map is a pixelwise error locator. | Forbidden | Phase79 validation negative: top-10% |P0 x_hat| pixels have LOWER actual error than rest-90 for arms B/C. Source: p0_error_correlation_summary.csv. B5 above is supported only for the image-level correlation claim, not pixel-level. |
| — | Merge VQGAN positive draft into conservative IEEE-TCI main claim. | Forbidden | The two manuscripts target different claims and different venues. The VQGAN draft is an independent positive result; merging would violate the conservative peer-review standard of the IEEE-TCI submission. |

---

## 5. Claim Scope Boundary (Summary Table)

| Question | What we can say | What we cannot say |
|---|---|---|
| Does the audit certify image quality? | No. Audit certifies `A x_hat = y`. PSNR is unchanged ≤ 0.039 dB post-audit. | That high-PSNR implies measurement accountability. |
| Does measurement consistency certify correctness? | No. T4_pairs: 16/16 feasible wrong images satisfy wrong record to ~2e-15. | That a consistent reconstruction is semantically true. |
| Does the GAN improve measurable quality? | LPIPS and RAPSD improve descriptively in 5% regimes (B1). | That GAN beats diffusion / is SOTA. |
| Does the VQGAN fusion improve perceptual quality? | Yes (V1, locked). Delta LPIPS -0.0977, 32.6%, 8/8 gate PASS, 3/3 seeds. | That null-space texture is the true scene, or that B is an oracle. |
| Does the gauge remove shortcut? | Yes (B2). Gauge row delta 0.0 vs standard 0.4767. | That gauge quality dominates standard. |
| Can diverse posterior samples be drawn under fixed y? | No. z_collapsed_not_viable (B7, negative). | Any positive posterior-sampling diversity claim. |
| Is cross-rate VQGAN generalization locked? | No. Development-level only (V4). | A locked cross-rate claim. |

---

## 6. Path Cross-Reference

| Asset | Path |
|---|---|
| IEEE-TCI manuscript | `paper/main.tex` |
| VQGAN paper draft | `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md` |
| Supported claims ledger | `docs/core_experiments/supported_claims.md` |
| Unsupported/forbidden claims ledger | `docs/core_experiments/unsupported_forbidden_claims.md` |
| Materials inventory (numbers + source tables) | `paper/materials_inventory.md` |
| Canonical numbers reference | `docs/core_experiments/canonical_numbers.md` |
| Conflicting numbers archive | `docs/core_experiments/conflicting_numbers_archive.md` |
| Gap analysis | `docs/core_experiments/gap_analysis.md` |
| Paper-ready bundle manifest | `docs/core_experiments/paper_ready_bundle_manifest.md` |
| GAN canonical results table | `E:\ns_mc_gan_gi\outputs_phase77_auditable_gan_paper_assembly\canonical_results_table.csv` (read-only) |
| Cert package tables | `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\` (read-only) |
| Shortcut stress report | `E:\ns_mc_gan_gi\outputs_phase75_final_high_tier_validation\SHORTCUT_STRESS_TEST_REPORT.md` (read-only) |
| Regime map | `E:\ns_mc_gan_gi\outputs_phase75_final_high_tier_validation\regime_map_final.csv` (read-only) |

**Sibling handoff docs:** `HANDOFF/01_RESEARCH_STORY.md` (chronological spine) and the per-line indexes `research_lines/01_core_platform/INDEX.md` (shared platform), `research_lines/03_baselines_audit/INDEX.md` (audit certificates), `research_lines/05_range_null_barrier/INDEX.md` (feasibility boundary), `research_lines/06_gauge_gan_rad5/INDEX.md` (GAN case study), `research_lines/08_vqgan_fcc/INDEX.md` (VQGAN/FCC line).

---

## 7. Reproduction Notes

- All importable modules stay flat at repo root (`src/`, `eval/`, root `*.py`). Run everything from `E:/ns_mc_gan_gi_code_fcc_phase1/` (repo root). Never move core modules.
- Active Python environment: `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311` (Python 3.11).
- Source tables referenced in §2 above live in `E:\ns_mc_gan_gi\` (GAN_FCC_WORK read-only mirror) — view/copy only.
- The Scr-5 LPIPS lock decision must be confirmed by the author before final manuscript numbers are written. Use Phase75 canonical aggregate (`outputs_phase75_final_high_tier_validation/standard_cgan_seed_metrics.csv`) and never mix with Phase69B/70 or Phase71 rows.
