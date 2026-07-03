# Core Experiment Evidence Package

This directory is the GitHub-facing index for the NS-MC-GAN / measurement-certified GI / auditable GAN paper assets.

It is intentionally a lightweight evidence package, not a raw data dump. It collects the canonical experiment tables, claim-evidence map, figure plan, reproducibility notes, and paper outline produced by the Phase79 full paper asset audit. Large checkpoints, `.npz` per-sample outputs, raw arrays, and exhaustive figure inventories stay outside GitHub and are referenced through manifests.

## Canonical Story

The paper should be written as one integrated audit-and-boundary paper:

1. Low-sampling ghost imaging needs both visual quality and bucket-signal accountability.
2. `Pi_y^lambda` provides a post-hoc measurement audit/certificate.
3. The audit can reduce measurement residuals by orders of magnitude while leaving PSNR nearly unchanged.
4. The certificate has a boundary: measurement consistency is not semantic correctness.
5. A GAN prior is a case study for prior-supplied content, not the main certificate.
6. `P0 xhat` maps unmeasured/prior-supplied content, and the alpha knob changes prior detail while preserving the measurement certificate.

## Start Here

- [Phase79 final report](phase79_final_report.md)
- [Canonical numbers](canonical_numbers.md)
- [Canonical numbers CSV](canonical_numbers.csv)
- [Claim-evidence matrix](claim_evidence_matrix.csv)
- [Supported claims](supported_claims.md)
- [Unsupported and forbidden claims](unsupported_forbidden_claims.md)
- [Main figure plan](main_figure_plan.md)
- [Reproducibility audit](reproducibility_audit.md)
- [Method conventions](method_conventions.md)
- [One-paper outline](one_paper_outline.md)
- [Core source scripts manifest](source_scripts_manifest.md)

## Canonical Sources

First-paper measurement-certified GI evidence is locked to `cert_package_20260612` and Phase67 zero-training checks. GAN/prior evidence is locked to the Phase77 canonical table, which consolidates Phase73 Rad-5 and Phase75 Scr-5 results.

Main text should not mix in Phase53C, G1/G2, Phase69B/70 pilot rows, old clipped RelMeasErr conventions, or Phase78 one-seed 96px exploratory rows.

## What Is Not Uploaded Here

- Checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- Large per-sample arrays (`*.npz`, `*.npy`)
- Exhaustive figure inventory CSVs larger than normal GitHub review size
- Raw local output folders

Those assets are indexed by the Phase79 manifests and should be released through a data artifact channel, not as ordinary source files.
