# Stage 5 — Range-Null Barrier and Feasible-Wrong-Image Evidence

**Phases 48–60 + paper1 phase 67**
**Research question:** Is measurement consistency sufficient for correctness?

---

## 1. Core Claim

For any null-space vector z satisfying A z = 0, the perturbed image x + z is
measurement-identical to x: A(x + z) = A x. A reconstructor that satisfies the audit
A x_hat ≈ y (RelMeasErr near zero) may nonetheless produce an image that differs
structurally from the true scene. This stage constructs, verifies, and hardens that
"feasible-wrong" barrier as formal evidence in the main paper.

The boundary is not a failure of the audit; it is an intrinsic property of the undersampled
forward operator. Consistency ≠ correctness; feasible wrong images exist.

---

## 2. Key Evidence Artefacts

### 2.1 cert_package_20260612 (sibling repo)

Located at `E:/ns_mc_gan_gi/results/cert_package_20260612/` (the canonical GAN training
repo, read-only from this working copy). Referenced by:

- `paper/figures/make_figure1_feasible_geometry.py` — loads `cache/main_rad5.npz`,
  `cache/A_rad5.npy`, `cache/split_eval_indices_stl10_test.npy`; target image index
  I_TARGET = 1789, donor index J_DONOR = 935.
- `T4_pairs.csv` (inside `cert_package_20260612/tables/`) — 16/16 cross-class pairs
  satisfy the WRONG record to ~2e-15 RelMeasErr.
- `REPORT.md` (inside `cert_package_20260612/`) — summary of the certificate invariance
  finding.

> These files are NOT in this working copy. To reproduce: check out
> `E:/ns_mc_gan_gi` and run the figure builder from repo root.

### 2.2 Certificate Invariance Recheck (this repo)

`results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py`

Verifies that the post-GAN pilot checkpoint does not break the measurement certificate.
Run from repo root:

```
python results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py
```

Output JSON written to `results/sampling_mode_20260612_151210Z/CERTIFICATE_INVARIANCE_RECHECK.json`.

**Confirmed numbers (CERTIFICATE_INVARIANCE_RECHECK.json, 2026-06-12):**

| Checkpoint | RelMeasErr mean | RelMeasErr std | n |
|---|---|---|---|
| Phase 15 mean (scrambled_hadamard5_hq_noise001_colab/last.pt) | 0.005484 | 0.002216 | 256 |
| G1 source checkpoint (scr5/source_checkpoint.pt) | 0.005484 | 0.002216 | 256 |
| source_minus_mean_relmeas | 0.0 | — | — |

Both checkpoints use `generator_ema`. Post-GAN pilot checkpoint: `not_found` — the
optional GAN pilot updated the generator in memory but did not save a post-GAN checkpoint;
only `source_checkpoint.pt` is present for Scr-5.

Eval set: `torchvision.datasets.STL10 split="test"`, limit 256 samples, seed 43.

---

## 3. Phase-Level Source Files

### Phases 48–49 — Mechanistic probes and ablation notebooks

**Colab preparation report (relocated to this directory):**
`research_lines/05_range_null_barrier/PHASE48_49_COLAB_READY_REPORT.md`

Five Colab sessions prepared:

| Session | Variant | Train? |
|---|---|---|
| session_01_eval_probes | mechanistic probes | no |
| session_02_rad5_no_gate | Rad-5 no gate ablation | yes (overnight) |
| session_03_rad5_no_final_audit | Rad-5 no final audit | yes (overnight) |
| session_04_scr5_no_gate | Scr-5 no gate ablation | yes (overnight) |
| session_05_scr5_no_final_audit | Scr-5 no final audit | yes (overnight) |

Core commands:
```
python -m src.phase48_49_mechanistic_probes           # session 01
python -m src.phase48_49_train_ablation --task rad5 --variant no_gate
python -m src.phase48_49_train_ablation --task rad5 --variant no_final_audit
python -m src.phase48_49_train_ablation --task scr5 --variant no_gate
python -m src.phase48_49_train_ablation --task scr5 --variant no_final_audit
```

Bundle preparation: `.\scripts\phase48_49\phase48_49_prepare_upload_bundle.ps1`
Import after Colab: `.\scripts\phase48_49\phase48_49_import_colab_outputs.ps1`
Default import target: `E:/ns_mc_gan_gi/outputs_phase48_49_colab_import`

### Phase 50 — Second-wave plan (not yet approved)

`research_lines/05_range_null_barrier/PHASE50_SECOND_WAVE_PLAN.md`

Candidate experiments queued but held pending Phase 48/49 Session 01 inspection:
Rad/Scr sampling-ratio scaling (2.5%, 7.5%, 15%, 20%), combined no-gate+no-final-audit,
stronger CS-TV/FISTA-TV baselines, mixed Rad/Scr model. **Do not launch without explicit
user approval.**

### Phases 51A, 53B–D, 55–56 — Feasible-hallucination construction and hardening

Source modules (all importable from `src/`):

| File | Role |
|---|---|
| `src/phase51A_train_ablation.py` | Standardised ablation runner |
| `src/phase53B_feasible_hallucination.py` | Session 22 feasible-hallucination dataset |
| `src/phase53B_shortcut_audit.py` | Session 21 shortcut audit (blind + full critics) |
| `src/phase53C_feasible_hallucination_figure.py` | Session 22 feasible-hallucination figure |
| `src/phase53C_exact_projector.py` | Exact P0 / row-space projector |
| `src/phase53D_feasible_hallucination.py` | Phase53D variant |
| `src/phase53D_posthoc_certificate_sweep.py` | Post-hoc certificate sweep on no-audit ablations |
| `src/phase55_cross_audit.py` | Cross-audit between Phase53C and 53D evidence |
| `src/phase55_make_claims.py` | Claim collation |
| `src/phase56_build_exact_null_pairs.py` | Exact P0 / anchor feature cache and pair metadata |
| `src/phase56_harden_feasible_hallucination.py` | Hardening/selection of feasible pairs |
| `src/phase56_build_group_splits.py` | Group split construction |
| `src/phase56_memorization_audit.py` | Memorisation audit |

Configs:

```
configs/phase53B/session_22_feasible_hallucination_dataset.yaml
configs/phase53C/session_22_feasible_hallucination_figure.yaml
```

Colab notebooks:
```
colab/phase53B/session_22_feasible_hallucination_dataset.ipynb
colab/phase53C/session_22_feasible_hallucination_figure.ipynb
```

Phase 56 hardening selection criterion: `cross_relmeas <= 0.005 OR cross_over_ours <= 2.0`.
Caption language for passing pairs: "measurement-consistent feasible alternative".
Caption language for near-threshold pairs: "near-feasible cross solution".

### Phases 59–60 — G1/G2 sampling mode (boundary probe)

| File | Role |
|---|---|
| `src/phase59_g1_sampling_mode_eval.py` | G1 optional GAN + posterior sampling eval |
| `results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py` | Certificate recheck |
| `results/sampling_mode_20260612_151210Z/CERTIFICATE_INVARIANCE_RECHECK.json` | Recheck output |

G2 launch is blocked: no saved main no-leak train/val/test split hashes; old G1 eval loop
has no explicit stochastic z path; smoke was skipped (unsafe provenance gate).
See `results/sampling_mode_20260612_151210Z/FOLLOWUP_GAP_ITEMS.md` for full blocker list.

### Feasible-pair unit test (regression guard)

`tests/test_feasible_counterfactuals.py`

Runs `make_derangement`, `make_semihard_donors`, `verify_feasible_pairs` on 24-sample toy
operators (Rademacher and scrambled Hadamard). Asserts `pass_float32_proxy`. Run from
repo root:

```
python -m pytest tests/test_feasible_counterfactuals.py -v
```

---

## 4. Paper Figure Assets

| File | Content |
|---|---|
| `paper/figures/make_figure1_feasible_geometry.py` | Geometry diagram (range/null decomposition, feasible-wrong locus) |
| `paper/figures/figure1_feasible_geometry.pdf` | Built figure (PDF) |
| `paper/figures/figure1_feasible_geometry.png` | Built figure (PNG) |
| `paper/figures/feasible_hallucination_pair.pdf` | Example feasible-wrong image pair |
| `paper/figures/feasible_hallucination_pair.png` | Example feasible-wrong image pair (PNG) |
| `paper/figures/make_feasible_wrong_gallery.py` | Gallery builder |
| `paper/figures/make_feasible_wrong_candidate_pool.py` | Candidate pool selector |
| `paper/figures/select_feasible_wrong_images.py` | Final image selector |

Rebuild geometry figure from repo root:
```
python paper/figures/make_figure1_feasible_geometry.py
```
Requires the cert_package cache at `E:/ns_mc_gan_gi/results/cert_package_20260612/cache/`.

Eval and audit infrastructure: `eval/audit.py` (audit-correction scatter and RelMeasErr
computation), `eval/checker.py`, `eval/metrics.py`.

---

## 5. Measurement Infrastructure

Row-space / null-space decomposition lives in `src/projections.py` and
`src/compatibility_data.py` (functions `decompose_split`, `make_derangement`,
`make_semihard_donors`, `verify_feasible_pairs`, `exact_null_project`,
`exact_row_project`, `get_exact_projector`).

The forward operator is `src/measurement.py` (`GhostMeasurementOperator`). Exact-A
override is `src/exact_measurement.py` (`apply_measurement_override_from_config`).

---

## 6. Repro Summary

| Target | Command |
|---|---|
| Certificate invariance recheck | `python results/sampling_mode_20260612_151210Z/certificate_invariance_recheck.py` |
| Feasible-pair unit test | `python -m pytest tests/test_feasible_counterfactuals.py -v` |
| Geometry figure (requires sibling repo cache) | `python paper/figures/make_figure1_feasible_geometry.py` |
| Phase 48/49 Colab bundle | `.\scripts\phase48_49\phase48_49_prepare_upload_bundle.ps1` |

All commands run from `E:/ns_mc_gan_gi_code_fcc_phase1` (repo root). Environment:
Python 3.11 at `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311`.

---

## 7. Cross-References

- Stage 3 (audit certificate backbone): `research_lines/03_baselines_audit/INDEX.md`
- Stage 6 (GAN auditable case study): `research_lines/06_gauge_gan_rad5/INDEX.md`
- Stage 7 (G2R posterior sampling, negative result): `research_lines/07_g2r_posterior_sampling/INDEX.md`
- Shared projection + measurement platform: `src/projections.py`, `src/measurement.py`
- cert_package_20260612 (T4_pairs.csv, REPORT.md): `E:/ns_mc_gan_gi/results/cert_package_20260612/` (sibling repo, read-only)
