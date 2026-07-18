# Data, checkpoints, and artifact inventory

This file makes the boundary between a fresh public clone and the local evidence warehouse explicit. It prevents a reader from mistaking a missing large file for a missing code path, or from silently substituting a different split.

## Publicly tracked in this repository

The repository contains source code, tests, configuration files, figures, paper sources, result summaries, canonical VQGAN cached tables, and the small receipts needed to validate the reported tables. Start with `HANDOFF/04_REPRODUCIBILITY_GUIDE.md` for the exact command and environment matrix.

## Restored outside Git

The verified local layout uses:

```text
E:/GAN_FCC_WORK/datasets/
E:/GAN_FCC_WORK/data_warehouse/ns_mc_gan_gi/
```

The warehouse may contain raw STL-10 files, trained checkpoints, phase-specific GAN artifacts, and certificate packages excluded by `.gitignore` because of size or licensing. A fresh clone cannot reproduce a checkpoint-dependent table until the same files are restored or training is repeated. Do not replace them with an unrecorded download.

| Artifact family | Public path / entry point | External requirement | Reproduction class |
|---|---|---|---|
| Range/null projector | `src/`, `eval/`, `tests/` | none for unit identities | RERUNNABLE |
| STL-10 image data | `datasets/` or declared warehouse root | raw data and split manifest | EXTERNAL_DATA |
| Phase 15/53/75/77 GAN checkpoints | `outputs/`, phase scripts and handoff tables | matching checkpoint warehouse | EXTERNAL_WAREHOUSE |
| VQAE/VQGAN detail fusion | `vqgan_detail_fusion.py`, `outputs/compatibility/measurement_conditioned_vqgan/` | committed cached tables; training weights for a fresh train | LOCKED / EXTERNAL_WAREHOUSE |
| CASSI | `cassi_twoledger/` | released TSA-Net data and mask | RERUNNABLE_WITH_PUBLIC_DATA |
| fastMRI | `fastmri_twoledger/` | declared fastMRI files and weights | EXTERNAL_DATA |

## Identity checks before a run

1. Run from the repository root; the project intentionally uses flat imports.
2. Read the experiment row in `repro/EXPERIMENT_INDEX.md` and the matching handoff section.
3. Record the Git commit, dataset root, split manifest/hash, model/checkpoint identifiers, device, and seed in the run directory.
4. Do not rerun one-shot locked scorers. Use the stored receipt or the non-destructive validation command.
5. Keep raw arrays separate from display-clipped images. `RelMeasErr` uses the unclipped float64 reconstruction; clipping is for display/PSNR only and must be labeled.

## Environment fingerprint

The reference run used Python 3.11.15, PyTorch 2.2.1+cu121, torchvision 0.17.1+cu121, and NumPy 1.26.4. The exact environment path and package pinning are in `HANDOFF/04_REPRODUCIBILITY_GUIDE.md`; the path is local-machine metadata, not a requirement that a new user has the same drive letter.

