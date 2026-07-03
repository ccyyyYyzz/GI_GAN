# Research Story: Chronological Spine

**Repo:** `E:/ns_mc_gan_gi_code_fcc_phase1`
**Companion (Chinese):** `HANDOFF/archive_gan_fcc_work/01_RESEARCH_STORY.md`
**Env:** Python 3.11 at `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311`
**Run convention:** all commands from repo root; imports are flat (`from src...`, `from eval...`).

---

## Unifying Theory

All stages operate on the same linear inverse problem:

    y = A x + eps,   m << n

The measurement constrains only the **row-space component** of x:
`P_R x = A^dagger y`  (data solution).
The **null-space component** `P_0 x`, where `P_0 = I - A^dagger A` and `A P_0 = 0`, is
unconstrained by y. It carries most perceptual detail and is where every prior makes
its uncertified bet.

A **test-time measurement audit** (`A x_hat ~= y`) enforces accountability but cannot
certify correctness: feasible-wrong images exist (consistency != correctness). The two
critical turning points of the program are:

- **(a) Quality -> Accountability:** the audit contract formalised in Stage 3.
- **(b) Consistency -> Correctness:** the feasible-wrong-image barrier proved in Stage 5.

GAN, VQGAN, and FCC are auditable case studies of how different priors fill the null
space, not quality-SOTA claims.

---

## Stage 0 — Precursor PCCN-GI / cGAN

**Phase:** pre-phase3 (legacy; code is NOT in this repo)

**Scientific question:** Can a conditional cGAN or U-Net improve undersampled ghost
imaging quality?

**Turn / insight:** Conditional reconstruction (CGI + U-Net / PCCN-GI) with L1 and
measurement-domain physics consistency worked well, but the range/null decomposition
had not yet been formalised. GAN and SSIM results were used as quality signals before
it was clear that quality and accountability are separate axes.

**Theory:** Supervised conditional generation; L1 + physics-consistency loss; GAN and
SSIM as ablation objectives. Pre-range-null formalism.

**This-repo files:** None — source archived at
`GAN_FCC_WORK/project_sources/pc_cgan_gi_precursor_sourceonly` (read-only history).

**Repro:** N/A (archaeological record only).

**Evidence:** Precursor README keeps GAN/SSIM as historical references. No canonical
numbers are claimed from this stage.

**Status:** Historical precursor. Not a source of main-paper numbers.

---

## Stage 1 — NS-MC-GAN Measurement-Consistent Core Platform

**Phase:** Core platform (active)

**Scientific question:** Can we formally separate what the measurement determines from
what the prior fills in, and build a platform that enforces measurement consistency at
test time?

**Turn / insight:** Formalising `range(A^T)` vs `null(A)` unlocked a clean split: the
data solution `A^T(AA^T + lambda I)^-1 y` recovers the row-space component; the prior
supplies null-space content; and a pluggable projection/audit operator can enforce or
verify `A x_hat ~= y` without retraining. This platform became the backbone for every
subsequent stage.

**Theory:**
- Forward model `y = Ax + eps`.
- Row-space recovery: `x_R = A^dagger y` (exact or regularised).
- Null-space residual: `x_0 = x_hat - x_R`.
- Data-consistency projection: `Pi_y^lambda(x) = x + A^T(AA^T + lambda I)^-1(y - Ax)`.
- Measurement audit: `RelMeasErr = ||A x_hat - y|| / ||y||`.

**This-repo files (all verified to exist):**

| File | Purpose |
|---|---|
| `src/train.py` | Training entry point |
| `src/eval.py` | Evaluation entry point |
| `src/models.py` | Generator / critic architectures |
| `src/datasets.py` | STL10 loaders, split guards |
| `src/losses.py` | Adversarial + measurement losses |
| `src/measurement.py` | Operator construction and application |
| `src/metrics.py` | PSNR, SSIM, LPIPS, RelMeasErr, RAPSD |
| `src/projections.py` | Row-space / null-space projection utilities |
| `src/utils.py` | Shared helpers |
| `configs/default.yaml` | Default training config |

**Repro:**

    python -m src.train --config configs/default.yaml
    python -m src.eval  --config configs/default.yaml

**Evidence:** Shared platform — measurement operator, dataset loaders with test-split
guards, generator/critic architectures, losses, metrics, and exact/soft audit
operators. Used by every downstream stage.

**Status:** Shared foundation; actively maintained.

---

## Stage 2 — Operator / Pattern-Learning Exploration

**Phase:** Phases 3–8

**Scientific question:** Do reconstruction gains come from learning the illumination
operator / patterns, or from learning a better prior?

**Turn / insight:** Exploring binary STE operators, continuous differential patterns,
and operator calibration showed the attribution was ambiguous — gains could reflect
operator adaptation or prior quality, and the two could not be cleanly separated. This
motivated a harder test: a pluggable, reconstructor-agnostic audit that would work
regardless of operator design.

**Theory:** Binary and continuous illumination patterns; flip-aware binary
straight-through estimator (STE); differential operator learning; calibration to
measured patterns; operator vs. prior attribution analysis.

**This-repo files:**

| File | Purpose |
|---|---|
| `src/` (phase3–8 modules) | Operator learning experiments |
| `configs/` (phase3–8 sweeps) | Sweep configurations |
| `scripts/*.ps1`, `scripts/*.sh` | Aggregation and run scripts |

**Repro:**

    python -m src.train --config configs/<phase3-8 sweep>.yaml

**Evidence:** Early controls established the platform; attribution remained ambiguous
across binary vs. continuous operators.

**Status:** Historical exploration; supporting context for the audit line.

---

## Stage 3 — Hadamard/Rademacher Baselines + Audit Certificate

**Phase:** Phases 9–17

**Scientific question:** Can ONE pluggable, reconstructor-agnostic measurement audit be
attached at test time to any reconstructor and drive RelMeasErr down without degrading
PSNR?

**>> TURNING POINT (a): Quality -> Accountability <<**

**Turn / insight:** The audit contract is independent of what the reconstructor does in
the null space. Applying `Pi_y^lambda` post hoc to any output — whether from
back-projection, CS-TV, or a learned network — brings `A x_hat ~= y` to near-machine
precision. PSNR changes by at most hundredths of a dB. This proved accountability and
quality are separable: a reconstructor can be made measurement-consistent without
retraining.

**Theory:**
- Structured operators: low-frequency Hadamard rows + Rademacher rows.
- Locked measurement certificates: exact singular-mode contracts.
- Post-hoc audit: `Pi_y^lambda` applied to any pre-trained output.
- `RelMeasErr` = relative measurement error (unclipped float64).

Canonical numbers from `docs/core_experiments/canonical_numbers.csv` (source:
`E:\ns_mc_gan_gi\results\cert_package_20260612\`):

| Reconstructor | Regime | PSNR pre-audit | RelMeasErr pre | RelMeasErr post |
|---|---|---|---|---|
| bp_pipeline | Rad-5 5% | 7.30 dB | 5.17e-05 | 3.04e-09 |
| tv_pgd_best | Rad-5 5% | 8.49 dB | 1.31e-02 | 8.55e-07 |
| main_rad5 (ref, already audited) | Rad-5 5% | 22.316 dB | 3.78e-05 | 2.15e-09 |
| bp_pipeline | Scr-5 5% | 14.31 dB | 1.05e-07 | 1.05e-10 |
| tv_pgd_best | Scr-5 5% | 15.84 dB | 3.08e-03 | 3.07e-06 |
| main_scr5 (ref, already audited) | Scr-5 5% | 22.271 dB | 5.51e-03 | 5.50e-06 |

The audit drives RelMeasErr down by 3–4 orders of magnitude while PSNR changes at most
~0.05 dB (dPSNR column in canonical_numbers.csv is 0.0000–0.0136 for trained models).

**This-repo files:**

| File | Purpose |
|---|---|
| `eval/audit.py` | Test-time measurement audit |
| `eval/checker.py` | Gate checker |
| `eval/metrics.py` | Eval-side metrics |
| `eval/scr5_convention_bridge.py` | Scr-5 / Rad-5 naming bridge |
| `src/` (phase9–17 modules) | Baseline and cert experiments |
| `configs/` (phase9–16 configs) | Sweep configs |

**Repro:**

    python -m src.eval --config configs/<phase9-17>.yaml   # audit via eval/audit.py

**Evidence:** For BP / Tikhonov / CS-TV / learned reconstructors the same post-hoc
audit brings RelMeasErr to float64 floor (~1e-9) while PSNR is unchanged. Certificate
is reconstructor-agnostic.

**Status:** Core certificate backbone; active main-paper support.

---

## Stage 4 — Manuscript / Mechanism Construction

**Phase:** Phases 18–45

**Scientific question:** Can the accumulated evidence be assembled into publication
assets — mechanism figures, tables, provenance decompositions, conventional-GI anchors?

**Turn / insight:** Building the paper forced a precise statement of what is and is not
claimed. The conventional-GI anchor (PSNR, SSIM from published algorithms) and the
provenance decomposition (which dB come from the operator, which from the prior)
clarified the scope: gains from the audit are accountability gains, not quality gains.

**Theory:** LaTeX manuscript construction; figure/table automation; mechanism diagrams;
conventional-GI anchors; provenance decomposition.

**This-repo files:**

| File | Purpose |
|---|---|
| `paper/main.tex` | Main manuscript |
| `paper/materials_inventory.md` | Asset inventory |
| `paper/figures/` | Figure source scripts |
| `src/make_phase12_report.py` | Phase-12 result report builder |
| `core_mechanism_figure.py` | Core mechanism diagram (root-level) |
| `method_diagram_3d.py` | 3-D method schematic (root-level) |

**Repro:**

    python core_mechanism_figure.py
    python method_diagram_3d.py

**Evidence:** Mechanism figures, LaTeX tables, conventional-GI anchors, provenance
decomposition tables. Output in `paper/` and `paper/figures/`.

**Status:** Manuscript assets; supports main paper.

---

## Stage 5 — Range-Null Boundary and Feasible-Wrong-Image Barrier

**Phase:** Phases 48–60 + paper1 phases 67+

**Scientific question:** Is measurement consistency (`A x_hat ~= y`) sufficient to
certify that `x_hat` is a correct reconstruction?

**>> TURNING POINT (b): Consistency -> Correctness <<**

**Turn / insight:** For any `z` in `null(A)`, `A(x + z) = Ax`, so arbitrarily many
measurement-consistent images exist. The certificate can be satisfied by images from
entirely wrong semantic classes. This is not a failure of the audit — it is the
fundamental information-theoretic limit of undersampled measurement. It sharpens the
claim: the audit certifies **accountability** (the reconstruction reproduces the
measured buckets), not **correctness** (the null-space content is the true scene).

**Theory:**
- Null-space structure: `A P_0 = 0`, so `A(x + z) = Ax` for all `z` in `null(A)`.
- Exact null pairs: constructed explicitly by adding null-space vectors to images.
- Feasible cross-class counterfactuals: a wrong-class image `x'` can satisfy
  `||A x' - y|| ~= 0` to machine precision.
- Morozov / noise-floor audit: certificates are anchored to the noise floor of the
  measurement, not to semantic correctness.

**Evidence (verified via `docs/core_experiments/claim_evidence_matrix.csv` and
`docs/core_experiments/supported_claims.md`):**

16/16 cross-class image pairs satisfy the wrong measurement record to ~2e-15 relative
error (claim A6, evidence in
`E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T4_pairs.csv`; REPORT and figs
at `E:\ns_mc_gan_gi\results\cert_package_20260612\`).

Key figure assets in `paper/` (verified to exist):
`paper/feasible_hallucination_pair.pdf`,
`paper/figure1_feasible_geometry.pdf`.

**This-repo files:**

| File | Purpose |
|---|---|
| `results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py` | Recheck script |
| `results/sampling_mode_20260612_151210Z/REPORT.md` | Sampling-mode report |
| `results/sampling_mode_20260612_151210Z/CERTIFICATE_INVARIANCE_RECHECK.json` | Invariance recheck results |
| `docs/core_experiments/claim_evidence_matrix.csv` | Full claim/evidence matrix |
| `docs/core_experiments/supported_claims.md` | Supported claims register |

Note: `cert_package_20260612/` tables are in the sibling GAN_FCC repo
(`E:\ns_mc_gan_gi\results\cert_package_20260612\`), not in this working copy. The
canonical numbers are mirrored to `docs/core_experiments/canonical_numbers.csv` in
this repo.

**Repro:**

    python results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py

**Status:** Core boundary evidence; active main-paper support.
Written rule: do NOT claim the audit certifies null-space content as real.

---

## Stage 6 — Gauge-GAN / Rad-5 Auditable Generative Case Study

**Phase:** Phases 69–83 (current main line)

**Scientific question:** Under the measurement certificate, can a GAN prior improve
perceptual and spectral metrics while preserving accountability, and can the
gauge-AUC diagnostic explain which measurement regimes support adversarial fine-tuning?

**Turn / insight:** Repositioning the GAN as an *auditable generative prior example*
(not a quality-SOTA claim) let the paper make a crisp three-way statement: (i) GAN
detail measurably improves LPIPS/RAPSD; (ii) exact null-space fusion keeps
`A x_hat ~= y` to machine precision; (iii) the gauge-AUC gate predicts which regimes
benefit. The shortcut stress test (standard non-gauge cGAN: gauge-AUC 0.4767 vs
gauge-GAN: 0.0) showed the diagnostic is not trivially satisfied.

**Theory:**
- Gauge discriminator on `P_0 x_hat` (null-space component only).
- Gauge-AUC gate: adversarial fine-tuning is deployed only when gauge-AUC > threshold.
- Paired seeds: 3 seeds per regime; RelMeasErr controlled at machine precision.
- Scr-5 (scrambled Hadamard, 5%): main controlled regime.
- Rad-5 (Rademacher, 5%): robustness/regime check.
- Scr-10 / Rad-10: weak-gate regimes (gauge-AUC < 0.65); cGAN not deployed.

**Canonical numbers (from `docs/core_experiments/canonical_numbers.csv`, sourced from
`E:\ns_mc_gan_gi\outputs_phase75_final_high_tier_validation\regime_map_final.csv` and
`outputs_phase77_auditable_gan_paper_assembly\canonical_results_table.csv`):**

Gauge-AUC:

| Regime | Gauge-AUC | Decision |
|---|---|---|
| Scr-5 | **0.8466** | Positive (3/3 seeds) |
| Rad-5 | **0.8771** | Positive (3/3 seeds) |
| Scr-10 | 0.6240 | Weak gate; no cGAN |
| Rad-10 | 0.6396 | Weak gate; no cGAN |

Shortcut stress: standard non-gauge cGAN gauge-AUC **0.4767** vs gauge cGAN **0.0**
(Phase74 standard_cgan_baseline_scr5.csv).

Canonical metric table (Phase75 Scr-5 locked; Phase73 Rad-5 locked; 3 seeds):

| Arm | Regime | PSNR (dB) | SSIM | LPIPS | RAPSD | RelMeasErr |
|---|---|---|---|---|---|---|
| B (cGAN) | Scr-5 | 22.262 | 0.6266 | 0.2349 | 0.00539 | 5.56e-03 |
| C_gauge | Scr-5 | 22.257 | 0.6281 | 0.2308 | 0.00521 | 5.57e-03 |
| D_standard | Scr-5 | 22.258 | 0.6281 | 0.2310 | 0.00522 | 5.57e-03 |
| A | Rad-5 | 22.062 | 0.6247 | 0.2244 | 0.00493 | 4.59e-05 |
| B | Rad-5 | 22.304 | 0.6293 | 0.2344 | 0.00560 | 3.49e-05 |
| C | Rad-5 | 22.277 | 0.6314 | 0.2283 | 0.00509 | 3.65e-05 |

Alpha knob (measurement-invariant): RelMeasErr span across alpha sweep is 4.4e-09
(Rad-5) and 1.7e-07 (Scr-5) — effectively zero.

**This-repo files:**

| File | Purpose |
|---|---|
| `gan_high_quality_gi.py` | High-quality GI pipeline |
| `gan_gauge_aligned_nsgan.py` | Gauge-aligned NS-GAN |
| `gan_high_quality_gi_matched.py` | Matched GI pipeline |
| `inspect_gate.py` | Gate inspection utility |
| `gates.yaml` | Gate thresholds |
| `src/phase69A_gauge_gan_signal_diagnostic.py` | Gauge signal diagnostic |
| `src/phase71_gauge_cgan_paired_seeds.py` | Paired seeds experiment |
| `src/phase73_overnight_gauge_gan_expansion.py` | Rad-5 expansion |
| `src/phase75_final_high_tier_validation.py` | Final regime validation |
| `src/phase77_final_auditable_gan_paper_assembly.py` | Paper assembly |

**Repro (no retrain required):**

    python inspect_gate.py
    # Inspect: src/phase77_final_auditable_gan_paper_assembly.py -> canonical_results_table.csv
    # (canonical_results_table.csv is in E:\ns_mc_gan_gi\outputs_phase77_auditable_gan_paper_assembly\)

**Status:** Current active paper case study.

---

## Stage 7 — G2R / Posterior-Sampling Anti-Collapse Side Line

**Phase:** Side line (dormant)

**Scientific question:** Under fixed `y`, can we draw multiple measurement-consistent
posterior samples with real null-space diversity, where the row-space component
`P_R x_hat` is fixed and only `P_0 x_hat` varies?

**Turn / insight:** Stop-rule fired across all three omega_adv arms (1e-3, 3e-3, 1e-2).
The discriminator saturated (real/fake margin grew to +85 / +112 / +102, driven by
d_fake collapsing to -80..-107 while d_real stayed near +5..+7). The fixed beta_SD
reward drove per-pixel sample std to the same attractor (**0.4841**) in every arm by
~6000 steps, causing mean PSNR to collapse ~15 dB below baseline regardless of
omega_adv. G-MEAN failed in all arms (required: baseline - 0.3 dB; observed: ~15 dB
below). The controlling variable is beta_SD (variance-saturation attractor), not
omega_adv. Round 2 (closed-loop beta_SD controller) authorised but not yet executed.

**Evidence:** `results/g2r_pilot_phase3/PHASE3_REPORT.md` — STOP RULE FIRED; all 3
arms 3/6 gates (G-CAL PASS, G-DIV FAIL, G-NVR PASS, G-MEAN FAIL, G-CERT WARN, G-PERC
FAIL); roundtrip diff 0.0 on all arms; RelMeasErr median 7e-7 -> 2.7e-6 (float32
floor; CERT is WARN by design).

**This-repo files:**

| File | Purpose |
|---|---|
| `src/g2r_modec.py` | G2R Mode-C sampler |
| `src/g2r_modec_train.py` | Training wrapper |
| `src/phase79_rad5_rowspace_diversity_diagnostic.py` | Row-space diversity diagnostic |
| `configs/g2r/` | G2R pilot configs (8 configs + ROUND2_AMENDMENT.md) |
| `results/g2r_pilot_phase3/` | Pilot results and stop-rule record |

**Repro:**

    python -m src.g2r_modec_train --config configs/g2r/<run>.yaml

**Status:** DORMANT high-risk side branch. Negative result on record. Round 2 authorised
(closed-loop beta_SD controller) but pending; do not merge into main paper.

---

## Stage 8 — VQGAN / FCC Measurement-Conditioned Detail Fusion

**Phase:** FCC phase1 branch (this repo's focus)

**Scientific question:** Can a measurement-conditioned VQGAN/VQAE prior fuse null-space
detail into low-rate ghost imaging without violating the measurement, and what do FCC
row-null / structure-detail diagnostics reveal about the compatibility signal?

**Turn / insight (two sub-findings):**

1. **Null-space fusion is exact and controllable.** The formula
   `x_hat_B = x0 + P0(d_A + B(d_G - d_A))` satisfies `A x_hat_B = y` exactly for any
   scalar `B`, because `A P0 = 0` and `A x0 = y`. A single validation-selected `B`
   traces a smooth, monotone perception-distortion ladder.

2. **FCC row-null compatibility is nuisance-driven.** The FCC diagnostic canary shows
   that the apparent row-null compatibility signal is attributable to nuisance /
   naturalness statistics (best deployable scalar baseline AUC 0.9987 on balanced
   negatives exceeds or matches FCC AUC 0.9917), not to certified row-null mutual
   information. Classification: `ONLY_SCALAR_OR_ARTIFACT_SIGNAL`.

**Theory:**
- `x_hat_B = x0 + P0(d_A + B(d_G - d_A))`, with `d_A = P0(x_A - x0)`,
  `d_G = P0(x_G - x0)`, `P0 = I - A^dagger A`.
- Because `A P0 = 0` and `A x0 = y`: `A x_hat_B = y` exactly for all `B`.
- Two matched VQ priors per seed: VQAE (reconstruction-structured) and VQGAN
  (adversarial), both followed by anchor-initialized latent refiners.
- Fusion weight `B` is selected on val only, then frozen.
- FCC (Feature Compatibility Critic) row-null diagnostic: dual-encoder InfoNCE on
  (row-component, null-component) pairs; balanced nuisance controls.

**Canonical numbers (from `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md`):**

Setup: 5.0% sampling, 64x64 grayscale STL10, n=4096, m=205, 3 seeds, raw-hash-disjoint
locked test split (512 images).

Locked-split main result (balanced fusion, B~0.55):

| Quantity | Value |
|---|---|
| Delta LPIPS vs VQAE | **-0.0977** (CI [-0.1016, -0.0940]; **32.6% relative gain**) |
| Delta PSNR | -0.45 dB (within 0.5 dB pre-registered tolerance) |
| Delta RMSE | +0.0039 (within 0.005 tolerance) |
| Delta RAPSD | -0.00030 (spectral fidelity improves) |
| Seeds agreeing | 3/3 |
| Gate | **8/8 pre-registered conditions PASS** |
| RelMeasErr mean | 3.6e-07, max 5.7e-07 |

Dev vs. locked replication: dev Delta LPIPS -0.0965 (32.9%), locked -0.0977 (32.6%) —
effect sizes nearly coincide.

KID (dataset-level): VQAE 0.119 -> balanced fusion 0.043 (2.7x reduction).

Locked-split absolute means (5.0%, B~0.55):
- LMMSE anchor: LPIPS 0.404, PSNR 22.80 dB
- VQAE (B=0): LPIPS 0.300, PSNR 23.13 dB
- Balanced fusion (B~0.55): LPIPS 0.202, PSNR ~22.68 dB
- Quality-lite (B~0.72): LPIPS 0.182, PSNR 22.21 dB
- Full VQGAN (B=1): LPIPS 0.172, PSNR 21.43 dB

Cross-rate generalization (development split only; not re-locked):
- 2% sampling: Delta LPIPS -0.116, 29.3% relative gain, 3/3 seeds
- 10% sampling: Delta LPIPS -0.076, 34.2% relative gain, 3/3 seeds

FCC canary (64x64, Rad-5 5%, `outputs/compatibility/fcc_diagnostic_canary64/`):
- Real-pair Recall@1 = 1.0000 (hard negatives); label-shuffle Recall@1 = 0.0000.
- FCC critic AUC: random 1.0000, nuisance-balanced **0.9917**.
- Best deployable baseline balanced AUC: **0.9987** (pair_logistic).
- Classification: **`ONLY_SCALAR_OR_ARTIFACT_SIGNAL`**.
  The apparent compatibility is attributable to nuisance statistics; no generator
  selection is justified from this result.

**This-repo files:**

| File | Purpose |
|---|---|
| `vqgan_detail_fusion.py` | Main fusion pipeline |
| `vqgan_detail_fusion_locked.py` | Locked confirmatory run |
| `measurement_conditioned_vqgan.py` | Measurement-conditioned VQGAN |
| `anchor_initialized_vqgan_inversion.py` | Anchor-initialized latent refiner |
| `mc_vqgan_prior_long_canary.py` | MC-VQGAN prior long canary |
| `fcc_diagnostic_canary.py` | FCC row-null compatibility diagnostic |
| `structure_detail_fcc.py` | Structure-detail FCC |
| `experiments_rate_fusion.py` | Cross-rate fusion experiments |
| `experiments_local.py` | Local experiment runner |
| `method_diagram_3d.py` | 3-D method diagram |
| `configs/compatibility/` | Per-seed compatibility configs |
| `outputs/compatibility/fcc_diagnostic_canary64/` | FCC canary outputs |
| `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/` | VQGAN paper assets |
| `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md` | Full draft |

**Repro:**

    python vqgan_detail_fusion.py all
    python fcc_diagnostic_canary.py all --config configs/compatibility/fcc_diagnostic_canary64.yaml

**Status:** Related compatibility subline / independent VQGAN detail-fusion draft.
**Do NOT auto-merge** into the conservative IEEE-TCI main claim. The VQGAN draft is a
separate positive result; the FCC canary is a separate negative/null result.
Both are clearly labelled in `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/LIMITATIONS_AND_NEGATIVE_RESULTS.md`.

---

## Summary Table

| Stage | Key claim | Turning point | Status |
|---|---|---|---|
| 0 Precursor | cGAN improves GI quality | — | Historical |
| 1 Core platform | Range/null split; audit contract | — | Active |
| 2 Operator/pattern | Attribution ambiguous | — | Supporting |
| 3 Audit certificate | **(a)** Accountability separable from quality; audit drives RelMeasErr to ~1e-9 | Quality -> Accountability | Core; active |
| 4 Manuscript | Publication assets | — | Manuscript assets |
| 5 Feasible-wrong | **(b)** Consistency != correctness; 16/16 cross-class pairs satisfy wrong record to ~2e-15 | Consistency -> Correctness | Core; active |
| 6 Gauge-GAN | GAN is auditable prior; gauge-AUC 0.8466/0.8771; LPIPS/RAPSD improve; RelMeasErr ~1e-5 | — | Current main line |
| 7 G2R | Negative: posterior diversity collapsed (stop rule fired; PSNR -15 dB) | — | Dormant |
| 8 VQGAN/FCC | VQGAN fusion: 8/8 gate PASS, LPIPS -32.6%, KID 0.119->0.043; FCC: ONLY_SCALAR_OR_ARTIFACT_SIGNAL | — | Compatibility subline |

---

## Red Lines

- Never `git reset --hard` / `git clean` the worktree.
- Never train on the test split.
- Do NOT auto-merge the VQGAN positive draft into the conservative IEEE-TCI main claim.
- Label all claims as supported or forbidden; cite only canonical numbers from
  `docs/core_experiments/canonical_numbers.csv` or the verified source files above.
- Compute only when required (no retrain for Stage 6 inspection).
- `E:/ns_mc_gan_gi_code` (GAN_FCC source) is read-only; view/copy/run only.
