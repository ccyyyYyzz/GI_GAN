# Auxiliary Experiment Register

| Area | Phases / folders | Status | What it is safe to say | Do not say |
| --- | --- | --- | --- | --- |
| Early reconstruction/operator sweeps | Phase3-14 | Historical exploration | These runs explored measurement operators, ablations, and early reconstruction behavior before the certificate package was locked. | Do not mix their metrics with the Phase67/cert-package convention. |
| Supplementary ablations | Phase15-16, Phase25-26 | Archive / optional supplement | These runs contain debug, no-leak imports, supplementary perturbations, and architecture pilots. | Do not use them as primary paper anchors unless re-audited. |
| Manuscript and figure iterations | Phase17-30, Phase37-50 | Manuscript history | These phases preserve writing, figure, and submission-package iterations. | Do not treat old manuscript tables as canonical if they conflict with Phase79. |
| Null-critic / adversarial-certificate attempts | Phase53B/C/D, Phase55-56 | Invalid/archive plus boundary lessons | These phases are useful history for why the certificate must be `Pi_y^lambda`, not a discriminator. | Do not claim adversarial certificate, exact-null critic success, or Phase53C AUC 0.992 as main evidence. |
| GAN sampling mode G1/G2 | Phase59-60 | Post-mortem / skipped follow-up | G1 is post-mortem only; G2 was blocked by safety/provenance gates. | Do not cite G1 as positive sampling evidence or claim controlled G2 success. |
| Gauge-GAN pilots before canonical aggregation | Phase69A/B, Phase70-72 | Mechanism / pilot / weak-regime archive | Phase69A supports gauge signal and shortcut-risk reasoning; Phase69B/70 are historical pilots; Phase72 is weak Scr-10 gate evidence. | Do not cite Phase69B seed0 as canonical multi-seed Scr-5 evidence. |
| 96px and P0-error probes | Phase78-79 | Exploratory / negative result | Phase78 is one-seed 96px feasibility; Phase79 P0-error validation is negative for a pixel-error claim. | Do not merge 96px one-seed rows into paper tables or claim high `|P0 xhat|` predicts pixel error. |
| Posterior anti-collapse / calibration | Phase79 posterior, Phase80, Phase81 | Exploratory future work | These runs explore null-space diversity, centered calibration, and diversity-weight scans. | Do not make main paper posterior uncertainty claims. |
| G2R pilots | `outputs_g2r` | Exploratory failed-gate archive | G2R runs tested posterior/gate criteria and mostly failed overall gates despite some certificate warnings passing. | Do not present them as successful posterior sampler evidence. |

## Canonical Boundary

If a result is in both this directory and `docs/core_experiments/`, the core directory wins for paper claims. This auxiliary archive exists to explain history and non-core outcomes, not to broaden the canonical evidence set.
