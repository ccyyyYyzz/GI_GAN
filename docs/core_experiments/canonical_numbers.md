# Canonical Numbers

Generated: 2026-06-14 23:28:55

## Decisions

1. Scr-5 GAN numbers are canonical only from `outputs_phase77_auditable_gan_paper_assembly/canonical_results_table.csv`, which locks the Phase75 Scr-5 B/C/standard-D aggregate and Phase73 Rad-5 aggregate.
2. Phase69B seed0 is excluded from paired-seed statistics and remains archive/provenance only.
3. RelMeasErr convention is unclipped float64 against recorded `y`; PSNR/SSIM are clipped display-image metrics.
4. First-paper main text should use the cert package tables `T1`-`T7` plus PROVENANCE. GAN support should use the Phase77 canonical table, Phase75 regime/shortcut, and Phase76 alpha/unmeasured-content evidence.
5. Old Phase53C, G1/G2, Phase69B/70 pilot numbers, old clipped RelMeasErr, and Phase78 one-seed 96px rows should not be cited as main evidence.

## First-Paper Canonical Anchors

- Rad-5: PSNR 22.3160, RelMeasErr 0.006991.
- Scr-5: PSNR 22.2708, RelMeasErr 0.005452.
- Rad-10: PSNR 24.7812, RelMeasErr 0.006405.
- Scr-10: PSNR 24.7301, RelMeasErr 0.005994.
- Post-hoc audit: 18/18 rows pass; RelMeasErr reduction 3.0-4.3 orders; max |dPSNR| 0.039 dB.

## GAN Canonical Table

| regime | arm | seeds | n_per_seed | psnr_mean | ssim_mean | lpips_mean | rapsd_distance_mean | relmeaserr_unclipped_float64_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scr-5 | B | 3 | 256 | 22.26204344174215 | 0.6265835547898294 | 0.2348977884103078 | 0.005386925042957899 | 0.005561496671927267 |
| Scr-5 | C_gauge | 3 | 256 | 22.25658032211213 | 0.6281304470339605 | 0.23076087010364663 | 0.0052143194116456995 | 0.005567117915891334 |
| Scr-5 | D_standard | 3 | 256 | 22.257859148133274 | 0.6281001879275925 | 0.2309550932792869 | 0.005224457756023467 | 0.005567188304869168 |
| Rad-5 | A | 3 | 256 | 22.061882994400325 | 0.6246857264270635 | 0.22441272700962137 | 0.0049278881746018 | 4.58638395153755e-05 |
| Rad-5 | B | 3 | 256 | 22.304291465698256 | 0.6293254234590531 | 0.23438980771364484 | 0.005597302308268833 | 3.493286041857996e-05 |
| Rad-5 | C | 3 | 256 | 22.27672180148927 | 0.631377923872302 | 0.22833316350685587 | 0.005085400020814733 | 3.654031726462565e-05 |

## Regime Gate

| regime | gauge_auc | auc_ci | outcome | decision |
| --- | --- | --- | --- | --- |
| Scr-5 | 0.8466 | Phase69A | 3 paired seeds positive | train/evidence positive |
| Rad-5 | 0.8771 | 0.8446-0.9072 | 3 paired seeds positive | train/evidence positive |
| Scr-10 | 0.6240 | 0.5791-0.6700 | weak gate; no cGAN | stop |
| Rad-10 | 0.6396 | 0.5900-0.6774 | weak gate; no cGAN | stop |

Full machine-readable table: `canonical_numbers.csv`.
