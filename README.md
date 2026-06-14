# GI_GAN

Ghost imaging and auditable GAN experiment workspace.

## Core Paper Experiments

The curated paper-facing evidence package is in [`docs/core_experiments/`](docs/core_experiments/).

Start with:

- [`docs/core_experiments/README.md`](docs/core_experiments/README.md)
- [`docs/core_experiments/canonical_numbers.md`](docs/core_experiments/canonical_numbers.md)
- [`docs/core_experiments/claim_evidence_matrix.csv`](docs/core_experiments/claim_evidence_matrix.csv)
- [`docs/core_experiments/source_scripts_manifest.md`](docs/core_experiments/source_scripts_manifest.md)

Large checkpoints and raw arrays are intentionally not stored in GitHub. The GitHub package indexes the canonical evidence, scripts, claims, figures, and release gaps; heavyweight artifacts should be published separately with hashes.

## Auxiliary Experiment Archive

Non-core experiments are indexed separately in [`docs/auxiliary_experiments/`](docs/auxiliary_experiments/).

Use that directory for pilots, negative results, invalidated branches, manuscript/figure history, and exploratory posterior/G2R work. Do not use it as a source for canonical paper numbers unless an item is explicitly re-audited and promoted into `docs/core_experiments/`.
