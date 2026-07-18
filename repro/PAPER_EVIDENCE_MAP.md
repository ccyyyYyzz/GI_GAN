# Manuscript-to-evidence map

The repository contains several manuscripts because they answer different questions. This map prevents a reader from combining a result from one draft with the claims of another.

| Manuscript | Scope | Primary evidence | Build / inspect |
|---|---|---|---|
| `paper/OPTICS_DRAFT.tex` | Measurement-accountability and range/null claims | `docs/core_experiments/`, Gauge-GAN tables, certificate receipts | `python paper/build_variant.py OPTICS_DRAFT` |
| `paper/UNIFIED_PAPER_DRAFT.tex` | Broad GI/MRI/CASSI ledger narrative | `cassi_twoledger/`, `fastmri_twoledger/`, core ledger tables | `paper/README.md` |
| `paper/main.tex` | Earlier conservative draft | earlier tracked figures and tables | `paper/README.md` |
| `outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.tex` | VQAE/VQGAN test-time detail-fusion result | committed cached VQGAN tables and figure assets | inspect the local compatibility README |

## Claim boundary

The VQAE and VQGAN models are trained models. The relevant result is a test-time fusion of their outputs in a declared null-space direction; it is not a claim that the networks were untrained and it is not a claim that a new optical sensor was built. The accountability manuscript and the VQGAN manuscript must retain separate result tables and limitations.

## Figure and table provenance

Use `paper/README.md` and `paper/materials_inventory.md` to locate each figure source. Every final figure should be traceable to a script or a committed input array, and every numerical paragraph should point to a row in `docs/core_experiments/claim_evidence_matrix.csv` or the relevant result report. Do not replace a stored table with a screenshot or an unlabelled recomputation.

## Fresh-clone expectation

From a fresh clone a reader can run the structural checks, projector tests, paper assembly checks, and non-destructive VQGAN validation. Checkpoint-dependent GAN tables require the declared external warehouse; this is a documented reproducibility condition, not a reason to invent a substitute result.

