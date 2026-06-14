# Auxiliary Experiment Archive

This directory indexes non-core experiments for the ghost-imaging / auditable-GAN project.

It is deliberately separate from `docs/core_experiments/`. Files here are useful for provenance, negative evidence, abandoned branches, reviewer-risk analysis, and future-work planning, but they are not canonical paper tables.

## Use This Directory For

- Finding historical pilot experiments and why they were superseded.
- Preserving negative results so they do not get rediscovered and accidentally promoted.
- Tracking exploratory posterior/diversity/G2R work that is not yet paper-ready.
- Locating manuscript, figure, and ablation history without mixing it into the core evidence package.

## Do Not Use This Directory For

- Main paper quantitative claims.
- Canonical Scr-5/Rad-5/Scr-10/Rad-10 tables.
- Claims that GANs improve RelMeasErr.
- Claims of SOTA, diffusion superiority, hardware validation, semantic certificates, or hallucination proof.

## Start Here

- [Experiment register](experiment_register.md)
- [Archive and negative results](archive_and_negative_results.md)
- [Exploratory posterior and G2R work](exploratory_posterior_and_g2r.md)
- [Manuscript and figure history](manuscript_and_figure_history.md)
- [Local artifact manifest](local_artifact_manifest.csv)

Large checkpoints, raw arrays, per-sample images, and long training logs remain outside GitHub. This archive only records what exists, why it matters, and whether it is safe to cite.
