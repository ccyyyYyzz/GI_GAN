# Program Overview — Research Lines Index

**Repo:** `E:\ns_mc_gan_gi_code_fcc_phase1` (GitHub remote: `ccyyyYyzz/GI_GAN`, branch `codex/vqgan-multiseed-handoff`)

---

## Start Here

The primary entry point for any new agent or collaborator is the **HANDOFF/** directory at repo root.
The full-program Chinese handoff (copied from `GAN_FCC_WORK`) lives at:

```
HANDOFF/archive_gan_fcc_work/
```

Read in this order:

1. `HANDOFF/archive_gan_fcc_work/00_START_HERE_FOR_NEW_AGENT.md` — one-line project orientation and reading order
2. `HANDOFF/archive_gan_fcc_work/01_RESEARCH_STORY.md` — complete research trajectory, precursor to current
3. `HANDOFF/archive_gan_fcc_work/02_THEORY_CORE.md` — range-null decomposition, audit certificate, boundary
4. `HANDOFF/archive_gan_fcc_work/03_CODE_MAP.md` — which code is main-line, historical, side branch, evidence-only
5. `HANDOFF/archive_gan_fcc_work/05_EXPERIMENTS_AND_EVIDENCE.md` — per-experiment folders, reports, manifests
6. `HANDOFF/archive_gan_fcc_work/04_REPRODUCIBILITY_GUIDE.md` — safe repro rules (split guards, Colab/GPU budget)
7. `HANDOFF/archive_gan_fcc_work/06_PAPERS_AND_CLAIMS.md` — two papers, VQGAN/FCC draft, supported vs forbidden claims
8. `HANDOFF/archive_gan_fcc_work/07_RED_LINES_AND_WORKING_RULES.md` — inviolable constraints
9. `HANDOFF/archive_gan_fcc_work/09_PAPERS_INDEX.md` — all manuscript locations including VQGAN detail-fusion draft

---

## Relocated Documents in This Folder

The files below were generated during the `codex/vqgan-multiseed-handoff` session and are kept here as program-level references. They **point to** code; they do not contain it.

| File | What it covers |
|---|---|
| `PROJECT_MAP_AND_SESSION_SUMMARY.md` | GitHub repo inventory (as of 2026-06-29), three local workspaces (A/B/C), session output summary |
| `CLAUDE_CODE_HANDOFF.md` | Claude Code handoff for VQGAN multi-seed Pareto confirmation; branch/commit/env/next-steps |
| `COLAB_RUNNER_REPORT.md` | Colab runner creation report; files changed: `colab/pub_baselines_colab_runner.ipynb`, `colab/README_COLAB_RUNNER.md`, `scripts/validate_colab_runner.py`, `scripts/collect_colab_artifacts.py` |
| `implementation_plan.md` | Feasible-counterfactual compatibility Phase-1 implementation plan; `GhostMeasurementOperator`, null-projector design |
| `BACKUP_MANIFEST.json` | Backup record: source `E:\ns_mc_gan_gi_code` → `E:\ns_mc_gan_gi_code_backups\ns_mc_gan_gi_code_20260624_143957`; created 2026-06-24; source was not modified |
| `manuscript_audit/` | Round-1 manuscript view identity audit (`ROUND1_MANUSCRIPT_VIEW_IDENTITY_AUDIT.csv`, `ROUND1_MANUSCRIPT_VIEW_IDENTITY_AUDIT_CN.md`) |

---

## 30-Second Map of the 8 Research Lines

All code stays **flat at repo root** (`src/`, `eval/`, `configs/`, root `*.py`).
Run everything from repo root. Environment: Python 3.11 at `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311`.

| # | Folder | Status | One-line question |
|---|---|---|---|
| 0 | *(precursor PCCN-GI/cGAN — not in this repo)* | historical | Can conditional cGAN/U-Net make undersampled GI look better? |
| 1 | [`01_core_platform/INDEX.md`](../01_core_platform/INDEX.md) | active shared foundation | What does the measurement determine vs what does the prior fill in? |
| 2 | [`02_operator_pattern_learning/`](../02_operator_pattern_learning/) | historical supporting | Do gains come from learning the operator/illumination or from the prior? |
| 3 | [`03_baselines_audit/`](../03_baselines_audit/) | active core certificate | Can ONE pluggable test-time audit attach to any reconstructor? |
| 4 | [`04_manuscript_mechanism/`](../04_manuscript_mechanism/) | manuscript assets | Turn accumulated evidence into publication figures, tables, and mechanism diagrams |
| 5 | [`05_range_null_barrier/`](../05_range_null_barrier/) | active boundary evidence | Is measurement consistency sufficient for correctness? |
| 6 | [`06_gauge_gan_rad5/`](../06_gauge_gan_rad5/) | current active paper case study | Under the certificate, can a GAN prior improve perceptual metrics while preserving accountability? |
| 7 | [`07_g2r_posterior_sampling/`](../07_g2r_posterior_sampling/) | dormant / negative result | Can we draw multiple measurement-consistent posterior samples with real null-space diversity? |
| 8 | [`08_vqgan_fcc/`](../08_vqgan_fcc/) | related compatibility subline | Can measurement-conditioned VQGAN/VQAE fuse null-space detail, and how do FCC row-null diagnostics behave? |

### Unifying theory (one paragraph)

Undersampled ghost imaging: `y = A x + eps`, `m << n`. The measurement fixes only the row-space component `P_R x = A† y`; the null-space component `P_0 x` (`P_0 = I - A† A`, `A P_0 = 0`) is unconstrained by `y` and carries most perceptual detail. A test-time measurement audit can pull any reconstructor back to `A x_hat ≈ y` (accountability), but **cannot** certify null-space content is correct — consistency ≠ correctness; feasible-wrong images exist. GAN/VQGAN/FCC are auditable examples of how different priors fill the null space; they are **not** quality-SOTA claims.

### Key evidence numbers (verified)

- **Gauge-GAN (line 6):** LPIPS/RAPSD improve; RelMeasErr controlled; gauge-AUC Scr-5 0.8466 / Rad-5 0.8771; shortcut stress standard 0.4767 vs gauge 0.0 — see `src/phase77_final_auditable_gan_paper_assembly.py` canonical results table.
- **Range-null barrier (line 5):** 16/16 cross-class pairs satisfy the wrong measurement record to ~2e-15 — see `cert_package_20260612/tables/T4_pairs.csv`.
- **VQGAN fusion (line 8):** balanced fusion B≈0.55 gives LPIPS −0.0977 (−32.6%) at −0.45 dB PSNR; KID 0.119→0.043; 3/3 seeds; cross-rate 2–10% generalizes — see `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md`.
- **FCC canary (line 8):** ONLY_SCALAR_OR_ARTIFACT_SIGNAL — see `fcc_diagnostic_canary.py` + `src/fcc_canary.py`.

### Red lines

- Never `git reset --hard` or `clean` the worktree.
- Never train on the test split.
- Do NOT auto-merge the VQGAN positive draft into the conservative IEEE-TCI main claim.
- `E:\ns_mc_gan_gi_code` is read-only — view/copy/run only.
- All importable core stays flat at repo root; do not move `src/`, `eval/`, or root `*.py` modules.
- Frozen-original backup: `E:\ns_mc_gan_gi_code_backups\ns_mc_gan_gi_code_20260624_143957`.
