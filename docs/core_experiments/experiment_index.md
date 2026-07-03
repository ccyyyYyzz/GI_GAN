# Core Experiment Index

## Measurement-Certified GI

| Area | Canonical source | Main claim | Status |
| --- | --- | --- | --- |
| Exact reproduction and provenance | `cert_package_20260612/PROVENANCE.md` | Splits, exact-A, checkpoint hashes, lambda, and metric conventions are locked. | Main |
| Post-hoc audit | `cert_package_20260612/tables/T1_posthoc_external.csv` | Audit reduces RelMeasErr by 3.0-4.3 orders with at most 0.039 dB PSNR movement. | Main |
| Lambda frontier | `cert_package_20260612/tables/T2_sweep.csv` | PSNR remains nearly flat while measurement residual follows the audit strength. | Main |
| Modal contraction | `cert_package_20260612/tables/T3_contraction_summary.csv` | Float64 contraction follows `lambda/(lambda+sigma^2)`; f32 k=2 saturates. | Main with caveat |
| Certificate boundary | `cert_package_20260612/tables/T4_pairs.csv` | Cross-class feasible images can satisfy the same measurement record. | Main |
| Wrong-y and A-drift | `cert_package_20260612/tables/T6_dependence.csv`, `T7_adrift.csv` | Audit/accountability depends on the correct record and operator. | Main/supplement |

## Auditable GAN / Prior-Supplied Content

| Area | Canonical source | Main claim | Status |
| --- | --- | --- | --- |
| Gauge signal | Phase69A, Phase73, Phase75 regime map | 5% regimes have useful gauge signal; 10% regimes are weak and stopped. | Main/supplement |
| Scr-5 B/C/standard | Phase77 canonical table from Phase75 aggregate | Gauge cGAN and standard cGAN are comparable; gauge provides shortcut safety without quality dominance. | Main |
| Rad-5 robustness | Phase77 canonical table from Phase73 aggregate | Rad-5 paired seeds support robustness of the prior-detail result. | Main/supplement |
| Shortcut stress | Phase75 shortcut stress | Gauge equalization removes measured-row shortcut sensitivity. | Main/supplement |
| Unmeasured-content map | Phase76/77 maps and metrics | `P0 xhat` visualizes prior-supplied content; not proof of false hallucination. | Main with caveat |
| Alpha knob | Phase76 alpha sweep | Alpha changes prior detail while RelMeasErr stays invariant. | Main |

## Archived Or Forbidden

| Source | Reason |
| --- | --- |
| Phase53C AUC 0.992 | Split/provenance and claim semantics are unsafe for main evidence. |
| G1 GAN pilot | kappa/std collapse, missing post-GAN checkpoint/per-sample outputs, test-set adaptation risk. |
| G2 skipped | No positive result exists. |
| Phase69B seed0 | Excluded from paired-seed statistics after stricter paired-rule audit. |
| Phase78 96px Rad-5 | One-seed exploratory only; not a canonical paper table. |
| Phase79 96px P0-error validation | Negative result for high-`|P0 xhat|` pixel-error diagnostic. |

## Release Advice

Keep source code, configs, small CSVs, and paper-facing manifests in GitHub. Put checkpoints and large arrays in a separate release artifact or data repository with hashes copied back into the GitHub manifest.

