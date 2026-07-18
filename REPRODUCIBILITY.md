# Reproducibility entry point

This file is the shortest path from a fresh clone to the code, evidence, and manuscripts in this repository. It is an index, not a second implementation. The importable source tree remains flat by design; moving `src/`, `eval/`, or the root-level Python modules breaks cwd-relative imports.

## Repository identity

- Repository: `ccyyyYyzz/GI_GAN`
- Base snapshot documented here: `main` at `e3deaa68c35d46d4054756c8d86488580612b78f`
- Full program map: [`HANDOFF/00_START_HERE.md`](HANDOFF/00_START_HERE.md)
- Detailed reproduction commands: [`HANDOFF/04_REPRODUCIBILITY_GUIDE.md`](HANDOFF/04_REPRODUCIBILITY_GUIDE.md)
- Claim/evidence ledger: [`docs/core_experiments/claim_evidence_matrix.csv`](docs/core_experiments/claim_evidence_matrix.csv)
- Machine-readable manifest: [`repro/REPRODUCIBILITY_MANIFEST.json`](repro/REPRODUCIBILITY_MANIFEST.json)

Use the exact commit named by the experiment record. `main` is the integrated code and manuscript snapshot; `codex/vqgan-multiseed-handoff` is the matching VQGAN handoff snapshot; older branches are historical and must not be mixed with current numbers.

## Five-minute verification

Run every command from the repository root with Python 3.10 or newer. The verified project environment is Python 3.11.15, PyTorch 2.2.1+cu121, torchvision 0.17.1+cu121, NumPy 1.26.4.

```powershell
python repro/verify_reproducibility.py
python -m pytest tests/test_exact_projections.py tests/test_train_checkpoint_wiring.py -q
python vqgan_detail_fusion.py validate --seeds 0 1 2 --device cuda
python inspect_gate.py
python paper/build_variant.py OPTICS_DRAFT
```

The first command is dependency-free and checks the repository layout. The projection tests validate the range/null identities. The VQGAN validation reads the committed cached reconstructions and reference tables; it does not retrain and does not reopen the locked split. `inspect_gate.py` reads the canonical Gauge-GAN table. Building the paper requires XeLaTeX and the tracked figure assets.

## What is reproducible from this repository

| Line | Entry point | Status | Evidence location |
|---|---|---|---|
| Range/null projector and audit | `src/`, `eval/`, `tests/` | rerunnable | `docs/core_experiments/` and `results/sampling_mode_20260612_151210Z/` |
| Feasible-wrong-image barrier | `results/sampling_mode_20260612_151210Z/` | result committed; full regeneration needs the external warehouse | `T4_pairs.csv` and `CERTIFICATE_INVARIANCE_RECHECK.json` |
| Gauge-GAN case study | `gan_high_quality_gi.py`, `gan_gauge_aligned_nsgan.py`, `src/phase75_*`, `src/phase77_*` | canonical table inspectable; retraining requires checkpoints | `inspect_gate.py`, `gates.yaml`, external Phase 75/77 warehouse |
| VQAE/VQGAN detail fusion | `vqgan_detail_fusion.py`, `vqgan_detail_fusion_locked.py` | locked result is committed; do not rerun the locked scorer | `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/` |
| FCC canary | `fcc_diagnostic_canary.py` | rerunnable canary | `outputs/compatibility/fcc_diagnostic_canary64/` |
| CASSI two-ledger | `cassi_twoledger/` | rerunnable with released TSA-Net data/mask | `cassi_twoledger/RESULTS.md` |
| fastMRI two-ledger | `fastmri_twoledger/` | rerunnable only after obtaining the declared fastMRI files/weights | `fastmri_twoledger/RESULTS.md` |
| G2R posterior sampling | `src/g2r_*`, `results/g2r_pilot_phase3/` | negative and dormant | `results/g2r_pilot_phase3/PHASE3_REPORT.md` |

The VQAE and VQGAN networks are trained models. The positive Stage 8 result evaluates a test-time null-space fusion of their outputs; it is not evidence that the models were untrained. The locked result reports the fusion effect, not a new optical sensor.

## Manuscripts and their evidence

- **Optics Express target:** [`paper/OPTICS_DRAFT.tex`](paper/OPTICS_DRAFT.tex). This is the current measurement-accountability manuscript.
- **Unified draft:** [`paper/UNIFIED_PAPER_DRAFT.tex`](paper/UNIFIED_PAPER_DRAFT.tex). Broad GI/MRI/CASSI narrative.
- **Earlier conservative draft:** [`paper/main.tex`](paper/main.tex).
- **Independent positive VQGAN draft:** `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.tex`.
- **Figure and source inventory:** [`paper/README.md`](paper/README.md) and [`paper/materials_inventory.md`](paper/materials_inventory.md).

The VQGAN draft and the conservative accountability manuscript have different claims and must not be silently merged.

## External data and large artifacts

Git intentionally excludes raw STL10 data, training checkpoints, and some historical result warehouses. The canonical local layout used by the verified runs is:

```text
E:/GAN_FCC_WORK/datasets/
E:/GAN_FCC_WORK/data_warehouse/ns_mc_gan_gi/
```

The external warehouse contains the Phase 15/53/75/77 checkpoints and the cert package used by the canonical GAN and range-null tables. A fresh clone can verify the committed code, tests, figures, paper, VQGAN cached tables, and CASSI/MRI scripts; it cannot recreate excluded checkpoints without separately restoring that warehouse or rerunning training.

## Reproduction classes

- `LOCKED`: one-shot evaluation already consumed; inspect or validate its stored receipt, never rerun the scorer.
- `RERUNNABLE`: code, configuration, and required data are available for a new run.
- `EXTERNAL_WAREHOUSE`: the canonical result is committed or referenced, but checkpoints/results live outside Git.
- `DEVELOPMENT`: useful diagnostic evidence, not a confirmatory claim.
- `NEGATIVE/DORMANT`: a registered failure that should remain visible and should not be rebranded.

The authoritative claim boundary is [`HANDOFF/07_RED_LINES_AND_WORKING_RULES.md`](HANDOFF/07_RED_LINES_AND_WORKING_RULES.md). In particular, measurement consistency never certifies null-space correctness, and the G2R branch remains negative.

