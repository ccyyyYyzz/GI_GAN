# 00 — Start Here

**Repo:** `E:/ns_mc_gan_gi_code_fcc_phase1` (GitHub-connected, branch `codex/vqgan-multiseed-handoff`)
**Role:** One working copy of a larger ghost-imaging range-null program. The VQGAN/FCC phase-1 compatibility subline lives here. The authoritative full-program Chinese handoff (00–10 docs + CSV index) is archived under `HANDOFF/archive_gan_fcc_work/`.

---

## Theory (one paragraph)

Undersampled ghost imaging acquires `y = A x + eps` with `m << n`. The measurement pins exactly the row-space component `P_R x = A† y`; the null-space component `P_0 x` (`P_0 = I − A†A`, `A P_0 = 0`) is unconstrained by `y` and carries most perceptual detail. A test-time measurement audit can pull any reconstructor back to `A x_hat = y` (accountability), but cannot certify null-space content as correct — consistency is not correctness; measurement-identical-but-wrong "feasible wrong images" exist (documented in stage 5). GAN, VQGAN, and FCC are auditable examples of how different priors fill the null space; none are quality-SOTA claims.

---

## Stage Map

| # | Label | Research question (short) | Status | Key files (repo-root-relative) | Index / doc |
|---|---|---|---|---|---|
| 0 | Precursor PCCN-GI/cGAN | Can conditional cGAN/U-Net improve undersampled GI? | Historical precursor; not in this repo | `GAN_FCC_WORK/project_sources/pc_cgan_gi_precursor_sourceonly` | `archive_gan_fcc_work/01_RESEARCH_STORY.md` |
| 1 | NS-MC-GAN core platform | Separate what the measurement determines vs what the prior fills in | Active shared foundation | `src/train.py`, `src/eval.py`, `src/models.py`, `src/datasets.py`, `src/losses.py`, `src/measurement.py`, `src/metrics.py`, `src/projections.py`, `src/utils.py`, `configs/default.yaml` | `research_lines/01_core_platform/INDEX.md` |
| 2 | Operator / pattern-learning exploration | Do gains come from learning the operator or from the prior? | Historical exploration; supporting | `src/` phases 3–8 modules; `configs/` phases 3–8 sweeps; `scripts/*.ps1`, `scripts/*.sh` | `archive_gan_fcc_work/03_CODE_MAP.md` |
| 3 | Hadamard/Rademacher baselines + audit certificate | Can one pluggable test-time measurement audit attach to any reconstructor? | CORE certificate / active main-paper support | `eval/audit.py`, `eval/checker.py`, `eval/metrics.py`, `eval/scr5_convention_bridge.py`, `configs/` phases 9–16 | `research_lines/03_baselines_audit/PUB_BASELINES_CONFIG_REPORT.md` |
| 4 | Manuscript / mechanism construction | Turn accumulated evidence into publication assets | Manuscript-assets; supports main paper | `paper/main.tex`, `paper/materials_inventory.md`, `src/make_phase12_report.py`, `core_mechanism_figure.py`, `method_diagram_3d.py` | `archive_gan_fcc_work/06_PAPERS_AND_CLAIMS.md` |
| 5 | Range-null boundary + feasible-wrong-image barrier | Is measurement consistency sufficient for correctness? | CORE boundary evidence; active main-paper support | `docs/core_experiments/supported_claims.md` (cites `cert_package_20260612/tables/T4_pairs.csv` in `E:/ns_mc_gan_gi`); `results/sampling_mode_20260612_151210Z/` | `archive_gan_fcc_work/02_THEORY_CORE.md` |
| 6 | Gauge-GAN / Rad-5 auditable generative case study | Can a GAN prior improve perceptual/spectral metrics while preserving accountability and exposing the audit boundary? | CURRENT active paper case study | `gan_high_quality_gi.py`, `gan_gauge_aligned_nsgan.py`, `gan_high_quality_gi_matched.py`, `src/phase69A_gauge_gan_signal_diagnostic.py`, `src/phase71_gauge_cgan_paired_seeds.py`, `src/phase73_overnight_gauge_gan_expansion.py`, `src/phase75_final_high_tier_validation.py`, `src/phase77_final_auditable_gan_paper_assembly.py`, `inspect_gate.py`, `gates.yaml` | `archive_gan_fcc_work/05_EXPERIMENTS_AND_EVIDENCE.md` |
| 7 | G2R / posterior-sampling anti-collapse side line | Can we draw multiple measurement-consistent posterior samples with real null-space diversity instead of collapse? | DORMANT; negative result | `src/g2r_modec.py`, `src/g2r_modec_train.py`, `scripts/g2r/`, `src/phase79_rad5_rowspace_diversity_diagnostic.py`, `configs/g2r/` | `archive_gan_fcc_work/08_NEXT_AGENT_CHECKLIST.md` |
| 8 | VQGAN/FCC measurement-conditioned detail-fusion | Can a measurement-conditioned VQGAN/VQAE prior fuse null-space detail into low-rate GI, and how do FCC row-null / structure-detail diagnostics behave? | Related compatibility subline; independent draft (do NOT auto-merge into conservative IEEE-TCI main claim) | `vqgan_detail_fusion.py`, `vqgan_detail_fusion_locked.py`, `measurement_conditioned_vqgan.py`, `anchor_initialized_vqgan_inversion.py`, `mc_vqgan_prior_long_canary.py`, `fcc_diagnostic_canary.py`, `structure_detail_fcc.py`, `experiments_rate_fusion.py`, `experiments_local.py`, `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md` | `research_lines/08_vqgan_fcc/phase1_2_plan.md` |

---

## Key Evidence Numbers (verified from source files)

**Stage 6 — Gauge-GAN (Rad-5):** Scr-5 gauge-AUC 0.8466 / Rad-5 gauge-AUC 0.8771; shortcut stress standard 0.4767 vs gauge 0.0. Canonical results table: `src/phase77_final_auditable_gan_paper_assembly.py` writes `canonical_results_table.csv`; inspect with `python inspect_gate.py`.

**Stage 7 — G2R (negative):** z-variation diagnostic: pixel std ~7.19e-4; `z_collapsed_not_viable`. Stop-rule failures recorded in `src/phase79_rad5_rowspace_diversity_diagnostic.py`.

**Stage 8 — VQGAN detail fusion (locked):** 5.0% sampling, 64x64 grayscale STL10, n=4096, m=205, 3 seeds, raw-hash-disjoint locked split (512 images). Balanced fusion (B≈0.55): LPIPS −0.0977 (CI [−0.1016, −0.0940]; 32.6% relative gain) vs VQAE branch; PSNR cost −0.45 dB; RAPSD improves; 3/3 seeds agree; 8/8 pre-registered gate PASS. KID: 0.119 (VQAE) → 0.043 (balanced fusion), 2.7× reduction. RelMeasErr mean 3.6×10⁻⁷ (numerical precision). Cross-rate: 29–34% relative LPIPS gain holds at 2%, 5%, 10% sampling. FCC canary64: `ONLY_SCALAR_OR_ARTIFACT_SIGNAL`.

**Stage 5 — Feasible-wrong-image barrier:** 16/16 cross-class pairs satisfy the wrong measurement record to ~2e-15 (data in `E:/ns_mc_gan_gi/results/cert_package_20260612/tables/T4_pairs.csv`; claim indexed at `docs/core_experiments/supported_claims.md` claim A6).

---

## Import / cwd Warning

**All code uses `cwd = repo root` imports.** Scripts do `from src... import`, `from eval... import`, `import gan_high_quality_gi`, etc. The importable core stays flat at root (`src/`, `eval/`, `configs/`, root `*.py` modules, `paper/`, `colab/`, `scripts/`, `tests/`, `outputs/`, `results/`). Never move these. Always launch from repo root:

```
# correct
cd E:/ns_mc_gan_gi_code_fcc_phase1
python vqgan_detail_fusion.py all

# also correct for package modules
python -m src.train --config configs/default.yaml
```

---

## Quick Repro (3 commands)

```bash
# 1. Activate environment
conda activate ns_mc_gan_gi_py311
# (env prefix: E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311)

# 2. VQGAN detail-fusion repro (stage 8 — runs all seeds, locked + dev splits)
cd E:/ns_mc_gan_gi_code_fcc_phase1
python vqgan_detail_fusion.py all

# 3. FCC row-null canary (stage 8 diagnostic)
python fcc_diagnostic_canary.py all --config configs/compatibility/fcc_diagnostic_canary64.yaml
```

No retraining is needed. The locked test split is raw-hash-disjoint from all consumed data; run the locked evaluation exactly once.

---

## Archive — Full-Program Handoff

The complete Chinese-language full-program handoff (00–10 docs, HANDOFF_INDEX.csv, RESEARCH_LINEAGE.md) was copied from `E:/GAN_FCC_WORK/handoff/` and lives at:

```
HANDOFF/archive_gan_fcc_work/
  00_START_HERE_FOR_NEW_AGENT.md   ← original entry point for that package
  01_RESEARCH_STORY.md
  02_THEORY_CORE.md
  03_CODE_MAP.md
  04_REPRODUCIBILITY_GUIDE.md
  05_EXPERIMENTS_AND_EVIDENCE.md
  06_PAPERS_AND_CLAIMS.md
  07_RED_LINES_AND_WORKING_RULES.md
  08_NEXT_AGENT_CHECKLIST.md
  09_PAPERS_INDEX.md
  10_FILE_BY_FILE_CURATION.md
  HANDOFF_INDEX.csv
  RESEARCH_LINEAGE.md
```

Paths inside that archive reference `E:/GAN_FCC_WORK/` (the original work root), which is a sibling directory, not this repo. The data warehouse backing the frozen-original snapshot is at `E:/GAN_FCC_WORK/data_warehouse/`.

---

## Red Lines (summary)

- Never `git reset --hard` or `git clean` the worktree.
- Never train on the test split; never re-run the locked evaluation more than once.
- Do NOT auto-merge the VQGAN/FCC positive draft (stage 8) into the conservative IEEE-TCI main claim (stage 6). They are independent papers.
- `E:/ns_mc_gan_gi_code` (the GAN_FCC source) is read-only from this repo's perspective — view/copy/run only.
- Label supported vs forbidden claims; see `docs/core_experiments/supported_claims.md`.

Full red lines and working rules: `HANDOFF/archive_gan_fcc_work/07_RED_LINES_AND_WORKING_RULES.md`.
