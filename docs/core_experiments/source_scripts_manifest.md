# Core Source Scripts Manifest

These are the small source files that should accompany the core experiment evidence on GitHub. They are source entry points and orchestration wrappers only; large checkpoints and array outputs are intentionally excluded.

## Python Experiment Scripts

| Path | Role |
| --- | --- |
| `src/phase69A_gauge_gan_signal_diagnostic.py` | Gauge-equalized discriminator signal diagnostic. |
| `src/phase69B_controlled_gauge_cgan_pilot.py` | Controlled Scr-5 gauge cGAN pilot. Archived from main statistics. |
| `src/phase69B_compute_lpips.py` | LPIPS post-processing helper for Phase69B. |
| `src/phase70_gauge_gan_paper_expansion.py` | Early paper expansion and pilot consolidation. |
| `src/phase71_gauge_cgan_paired_seeds.py` | Strict Scr-5 paired-seed validation. |
| `src/phase72_scr10_gauge_cgan_regime_validation.py` | Scr-10 weak-gate regime validation. |
| `src/phase73_overnight_gauge_gan_expansion.py` | Rad-5 robustness and expanded figure package. |
| `src/phase74_high_tier_gauge_cgan_pack.py` | Standard cGAN comparison and high-tier evidence pack. |
| `src/phase75_final_high_tier_validation.py` | Final standard/gauge/shortcut/regime validation. |
| `src/phase76_high_upside_auditable_gan_exploration.py` | Alpha knob, unmeasured-content maps, weak failure detector, z diagnostic. |
| `src/phase77_final_auditable_gan_paper_assembly.py` | Auditable-GAN paper assembly and canonical table consolidation. |
| `src/phase78_96px_rad5_one_seed_probe.py` | Exploratory 96px Rad-5 one-seed probe; not canonical. |
| `src/phase79_96px_rad5_p0_error_validation.py` | Negative pixel-level P0-error validation on 96px Rad-5 outputs. |
| `src/phase79_rad5_rowspace_diversity_diagnostic.py` | Posterior anti-collapse / row-space diversity diagnostic; not main paper evidence. |

## PowerShell Wrappers

| Path | Role |
| --- | --- |
| `scripts/phase69A_gauge_gan_signal_diagnostic.ps1` | Runs Phase69A. |
| `scripts/phase69B_controlled_gauge_cgan_pilot.ps1` | Runs Phase69B. |
| `scripts/phase70_gauge_gan_paper_expansion.ps1` | Runs Phase70. |
| `scripts/phase71_gauge_cgan_paired_seeds.ps1` | Runs Phase71. |
| `scripts/phase72_scr10_gauge_cgan_regime_validation.ps1` | Runs Phase72. |
| `scripts/phase73_overnight_gauge_gan_expansion.ps1` | Runs Phase73. |
| `scripts/phase74_high_tier_gauge_cgan_pack.ps1` | Runs Phase74. |
| `scripts/phase75_final_high_tier_validation.ps1` | Runs Phase75. |
| `scripts/phase76_high_upside_auditable_gan_exploration.ps1` | Runs Phase76. |
| `scripts/phase77_final_auditable_gan_paper_assembly.ps1` | Runs Phase77 assembly. |

## Canonicality Notes

- Phase69B and Phase70 are included for provenance, but their Scr-5 numbers are archived only.
- Phase71, Phase73, Phase75, Phase76, and Phase77 are the core GAN/prior evidence path.
- Phase78 and Phase79 96px outputs are exploratory or negative-result evidence and must not be merged into canonical paper tables.
- The certificate side is primarily represented by the Phase79 docs and the `cert_package_20260612` reports/manifests, not by retraining scripts.

