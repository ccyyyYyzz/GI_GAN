# Archive And Negative Results

## Null-Critic And Adversarial-Certificate Attempts

Phase53C produced strong-looking exact-null critic numbers, including a reported max AUC of 0.992, but the Phase79 audit marks those results as unsafe for main evidence because of provenance and claim-semantics problems.

Safe archival statement:

- Exact-null critic experiments are part of the project history.
- They helped rule out discriminator-as-certificate framing.
- The measurement audit/certificate path is `Pi_y^lambda`.

Forbidden promotion:

- Do not claim adversarial certificate.
- Do not claim the discriminator proves measurement consistency.
- Do not claim Phase53C AUC 0.992 as a reliable main result.

## G1 / G2 Sampling Mode

Phase59/60 investigated optional GAN sampling mode.

- G1 is post-mortem only. The apparent PSNR advantage is not comparable.
- G1 kappa proxy was 0.7809010160897641, below the valid sampling-mode range.
- G1 mean pixel std proxy was 0.0007775262411087169.
- G1 null variance ratio proxy was 0.012526112579507753.
- G2 controlled training was skipped as unsafe because saved train/val/test split hash files were missing.

Safe archival statement:

- Sampling-mode evidence was investigated and rejected for the current paper.
- No controlled G2 evidence exists.

## Phase69B / Phase70 Pilot Rows

Phase69B and Phase70 are historical Scr-5 pilots. They motivated stricter paired-seed runs, but Phase79 excludes Phase69B seed0 and Phase70 pilot rows from canonical statistics.

Safe archival statement:

- Pilot C vs B LPIPS gain was 0.00868, with PSNR -0.0029 dB and RelMeasErr slightly worse by 5.13e-06.
- These are provenance-only pilot numbers.

Forbidden promotion:

- Do not cite these as multi-seed Scr-5 evidence.
- Do not mix them with Phase75/77 canonical Scr-5 rows.

## Phase72 Weak Scr-10 Gate

Phase72 stopped at weak gauge signal, with no B/C cGAN training.

- Scr-10 gauge AUC: 0.6240, CI 0.5791-0.6700.
- Residual shortcut AUC: 0.9077.

Safe archival statement:

- Scr-10 is weak-gate evidence and supports the regime boundary.

Forbidden promotion:

- Do not claim Scr-10 adversarial improvement.

## Phase78 / Phase79 96px Probes

Phase78 was a one-seed 96px Rad-5 feasibility probe. Phase79 tested whether high `|P0 xhat|` predicts pixel error and found negative evidence.

- Phase78 one-seed C vs B RAPSD gain: 0.000712.
- Phase78 one-seed LPIPS gain: 0.00655.
- Phase79 pooled Spearman for P0-error validation: B about 0.073, C about 0.072.
- Top-10 `|P0|` pixel error was lower than rest-90 for B/C in the tested setting.

Safe archival statement:

- The 96px path is future-work/exploratory.
- `P0 xhat` should be treated as prior-content accountability, not a direct pixel-error predictor.

Forbidden promotion:

- Do not merge Phase78 rows into canonical paper tables.
- Do not claim high `|P0 xhat|` systematically predicts higher pixel error.
