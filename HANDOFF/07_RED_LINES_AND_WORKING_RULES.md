# 07 Red Lines And Working Rules

These rules come from choices made during the project that must not be silently reversed. A new agent should follow them first and raise any disagreement explicitly before changing them.

---

## 1. Frozen Original and Backup Location

The GAN_FCC working copy that was in use before this repo was reorganized is preserved as a read-only backup:

- **Backup root:** `E:\ns_mc_gan_gi_code_backups\ns_mc_gan_gi_code_20260624_143957`
- **Backup manifest:** `research_lines/00_program_overview/BACKUP_MANIFEST.json`
- The manifest records source `E:\ns_mc_gan_gi_code`, robocopy exit code 1 (files copied), and confirms the source was not modified during backup.

Do not treat the backup as a mutable workspace. Scan or read from it, but do not write back to it.

---

## 2. Filesystem Red Lines

1. **Never `git reset --hard` or `git clean` the working tree.** The worktree is intentionally dirty with untracked files and staged outputs. Losing them is irreversible.

2. **Never use `git checkout -- .` or `git restore .`** to bulk-revert tracked files without explicit user approval.

3. **Never move or rename core code at the repo root.** The following flat-root modules are importable by scripts that rely on `cwd = repo root`:
   - `gan_high_quality_gi.py`, `gan_gauge_aligned_nsgan.py`, `gan_high_quality_gi_matched.py`
   - `vqgan_detail_fusion.py`, `vqgan_detail_fusion_locked.py`
   - `measurement_conditioned_vqgan.py`, `anchor_initialized_vqgan_inversion.py`
   - `mc_vqgan_prior_long_canary.py`, `fcc_diagnostic_canary.py`, `structure_detail_fcc.py`
   - `experiments_rate_fusion.py`, `experiments_local.py`, `method_diagram_3d.py`
   - All of `src/`, `eval/`, `configs/`, `paper/`, `colab/`, `scripts/`, `tests/`, `outputs/`, `results/`

   Confirmed import patterns (e.g. `vqgan_detail_fusion.py` line 32: `import gan_high_quality_gi as hq`; `fcc_diagnostic_canary.py` line 58: `from src import fcc_canary as fc`). Moving any of these breaks all downstream scripts.

4. **Run everything from repo root.** Do not `cd` into a subdirectory before calling any module. All `python -m src.train`, `python vqgan_detail_fusion.py`, `python fcc_diagnostic_canary.py` invocations assume `cwd = E:\ns_mc_gan_gi_code_fcc_phase1`.

5. **Python environment:** `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311` (Python 3.11). Do not silently switch to a system Python.

---

## 3. Experiment Red Lines

1. **Never train or tune hyperparameters on the test split.** The split guard is enforced in `src/split_guard.py` and tested in `tests/test_split_guard.py`. If a script lacks a split guard, add one before running.

2. **Never retroactively change a success criterion after observing results.** Pre-registered gates are recorded in `gates.yaml` (G2R posterior-sampling series, version 1, registered 2026-06-12) and in the VQGAN 8-condition gate documented in `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md`. Do not amend these after results are known.

3. **Never report a posterior-sampling claim without per-sample output support.** Any claim that a sampler produces diverse posterior samples must be backed by saved per-sample reconstructions and the following minimum thresholds (from `gates.yaml`, gate G-DIV and G-NVR):
   - Median per-pixel sample std >= 1e-2 (images in [0, 1])
   - Null-variance ratio (null-space fraction of sample variance) >= 0.1
   - Std map edge correlation (Spearman r, std vs Sobel) >= 0.2
   - All criteria computed on the canonical eval subset, not a hand-picked subset.

4. **Never claim the G2R posterior-sampling branch is viable.** The G2R Mode-C pilot (`results/g2r_pilot_phase3/g2r_pilot_scr5_adv1e-2/`) produced a definitive negative result: G-MEAN FAIL (-14.69 dB sample-mean PSNR vs baseline), G-PERC FAIL (LPIPS 0.452 vs baseline 0.232), G-DIV FAIL (std-vs-edge Spearman r = -0.038, below 0.2 threshold), and G-CERT WARN (median RelMeasErr 2.3e-6, above float64 threshold). The branch is dormant. Do not re-enable it without a new protocol review.

5. **Multi-seed count is pre-registered.** The VQGAN detail-fusion paper uses exactly 3 seeds (seed0, seed1, seed2). The GAN Gauge-GAN case study uses 3 paired seeds. Do not add seeds because individual results look ambiguous; do not drop seeds to clean up a result.

---

## 4. Do Not Auto-Merge the VQGAN Draft into the IEEE-TCI Main Claim

The repo contains two independent paper lines:

| Line | Files | Claim scope |
|------|-------|-------------|
| IEEE-TCI main (conservative) | `paper/main.tex`, `paper/materials_inventory.md`, `src/phase77_final_auditable_gan_paper_assembly.py` | GAN as auditable generative prior; Gauge-AUC Scr-5 0.8466 / Rad-5 0.8771; shortcut standard-D 0.4767 vs gauge-D 0.0; 3 seeds |
| VQGAN detail-fusion draft (compatible but separate) | `vqgan_detail_fusion_locked.py`, `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md` | Zero-training null-space fusion; locked 8/8 gate PASS; LPIPS -0.0977 (32.6%) at -0.45 dB PSNR; 3 seeds; KID 0.119->0.043 |

These are complementary. The VQGAN result is a positive separate draft. **Do not silently incorporate VQGAN numbers into `paper/main.tex`** or vice versa. Any merger must be an explicit authorial decision.

---

## 5. Commit Sources, Not Large Artifacts

Default policy: commit source code, configs, small CSVs, and manuscript `.tex` files. Do not commit:

- Large model checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- Large image batches or per-image NPZ dumps
- Colab-only generated PDFs or PNGs (unless they are final locked figures)

Exception: small locked evidence tables (e.g. `docs/core_experiments/canonical_numbers.csv`, `outputs/compatibility/.../PAPER_DRAFT.md`) are intentionally tracked because they are the canonical source of truth for claim numbers. Do not `.gitignore` them.

---

## 6. Paper Claims: Supported vs Forbidden

**Supported (grounded in existing locked results):**

- Measurement consistency: `A x_hat = y` exactly (RelMeasErr mean 3.6e-7, max 5.7e-7 on locked split).
- VQGAN null-space fusion perceptual gain: LPIPS -0.0977, CI [-0.1016, -0.0940], 3/3 seeds, locked 8/8 gate PASS.
- Range-null separability: improving RelMeasErr while PSNR is unchanged is expected, not a failure.
- Gauge-AUC evidence of auditable GAN signal: Scr-5 0.8466 / Rad-5 0.8771.
- Shortcut safety: standard-D sensitivity 0.4767; gauge-D sensitivity 0.0 (Phase75 shortcut stress test).
- FCC canary: `ONLY_SCALAR_OR_ARTIFACT_SIGNAL` (see `research_lines/08_vqgan_fcc/INDEX.md`).

**Forbidden (not supported by evidence in this repo):**

- Do not claim SOTA over diffusion models or any modern learned reconstructor.
- Do not claim measurement consistency implies semantic correctness. `A x_hat = y` does not certify null-space content as the true scene. Feasible wrong images exist: any `z` in `null(A)` satisfies `A(x + z) = y`. The range-null barrier evidence (`results/sampling_mode_20260612_151210Z/REPORT.md`, T4_pairs from stage 5) demonstrates 16/16 cross-class pairs satisfy the wrong measurement record to ~2e-15.
- Do not claim the G2R sampler produces real null-space diversity (negative result above).
- Do not claim the VQGAN draft's LPIPS numbers support the IEEE-TCI main paper without explicit authorial decision.
- Do not report a metric without specifying whether RelMeasErr is computed on clipped or unclipped reconstructions. The binding rule (from `gates.yaml` header): RelMeasErr is always on the UNCLIPPED vector; clipping to [0,1] is for display/PSNR only.

---

## 7. Compute-Only-When-Required

1. Colab/GPU is not a default action. Launch compute only when the task explicitly requires new training or a long inference run.

2. Before any training run, confirm and record:
   - Checkpoint or initialization path and hash
   - Data split and split hash
   - Data order (shuffle seed)
   - Optimizer and learning-rate schedule
   - Budget (epochs, wall-clock limit)
   - Selection rule (how a checkpoint is chosen)
   - What is the single variable changing from the previous run

3. After any training or inference run, save:
   - Config file
   - Checkpoint hash
   - Split hash
   - Per-sample outputs (not just aggregate metrics)
   - Metrics CSV
   - Code hash or dirty-diff summary

---

## 8. Interpretation Rules

These are binding interpretive constraints, not stylistic preferences:

- **Low RelMeasErr means measurement-consistent, not correct.** Do not write "the reconstruction is correct" when you mean "it satisfies the measurement."
- **PSNR unchanged while RelMeasErr improves is expected** from range-null separability; it is not a failure or a paradox.
- **Null-space diversity that looks like white noise is not real posterior diversity.** Check G-DIV (std-edge correlation) before claiming the sampler works.
- **Perceptual improvement on LPIPS/KID does not imply the injected texture is from the true scene.** The audit certifies `A x_hat = y`; it cannot certify what is in the null space.

---

## 9. Research-Line Quarantine

- `_unrelated_fresnel_zone_plate/` contains student FZP manuscripts unrelated to ghost imaging. Do not use its content as evidence or cite it.
- ZIFB battery content (`E:\zifb_final_9129_luck`) is a separate research program. It has been removed from this repo. Do not re-import it here.
- The GAN_FCC Rad-5 source at `E:\ns_mc_gan_gi_code` is read-only by policy (see `HANDOFF/archive_gan_fcc_work/` note on GAN_FCC read-only source). New artifacts go to authorized output directories, not back to that source tree.

---

Cross-reference: `HANDOFF/archive_gan_fcc_work/07_RED_LINES_AND_WORKING_RULES.md` (Chinese-language predecessor, covers GAN_FCC/GAN_GI program-wide rules). `research_lines/06_gauge_gan_rad5/INDEX.md` (gauge-AUC and shortcut numbers). `research_lines/08_vqgan_fcc/INDEX.md` (VQGAN fusion draft provenance). `gates.yaml` (pre-registered G2R gates). `paper/materials_inventory.md` (canonical source for all main-paper numbers).
