# Stage 7 — G2R Posterior Sampling (DORMANT / negative result)

**Status: DORMANT — z-collapse confirmed; no active development.**

This line asked whether a trained GAN prior could draw multiple
measurement-consistent posterior samples with genuine null-space diversity
instead of collapsing to a single reconstruction. The answer was no: z-variation
produced pixel std ~7.19e-4, the sampler was flagged `z_collapsed_not_viable`,
and all stop-rule evaluations failed.

---

## Research question

Under a fixed measurement y = Ax, can a Mode-C posterior sampler produce
K samples {x_hat^(k)} such that:
- A x_hat^(k) = y exactly for all k (measurement consistent), and
- the P0 components P0 x_hat^(k) are genuinely diverse (not degenerate)?

The null-space projector P0 = I - A^+ A satisfies A P0 = 0, so P0 content
is invisible to the measurement. Diversity there cannot be constrained by y
and must be driven by the prior alone.

---

## Core files

All paths are relative to repo root. Run everything from repo root with the
canonical environment `E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311`.

| File | Role |
|---|---|
| `src/g2r_modec.py` | Mode-C sampler components: P0 artifact loader, z-injected `_ZFuse` refiner, `ModeCSampler`, `CondPatchGAN`, hinge/R1 losses, exact-consistency audit |
| `src/g2r_modec_train.py` | Trainer: guarded TRAIN loader (`train_split=unlabeled`), held-out val (`val_split=train`), `CheckpointManager`, all metrics on unclipped tensors, run IDs `g2r_*` |
| `src/phase79_rad5_rowspace_diversity_diagnostic.py` | Rad-5 diagnostic: row-space-only reconstruction loss + hinge diversity on P0 pairwise L1; smoke sampler reports `smoke_mean_pixel_std`, `smoke_p0_variance`, `smoke_pr_variance`; outputs to `E:/ns_mc_gan_gi/outputs_phase79_posterior_anti_collapse/rad5_rowspace_diversity_diagnostic/` |
| `scripts/g2r/build_p0.py` | Builds and verifies float32 P0 projector artifacts (float64 SVD, checks A P0 v / A v <= 1e-12 and idempotence <= 1e-10) for tasks {rad5, scr5, rad10, scr10}; artifacts are ~64 MB each |
| `scripts/g2r/check_gates.py` | Loads `gates.yaml`; evaluates gates G-CAL, G-DIV, G-NVR against a metrics JSON; exits 1 on any FAIL |
| `scripts/g2r/build_phase3_report.py` | Assembles Phase 3 three-arm report from `gate_trajectory.csv` + `train_log.json`; writes `results/g2r_pilot_phase3/PHASE3_REPORT.md` |

---

## Configs

| Config | Run ID | Regime | Steps | Notes |
|---|---|---|---|---|
| `configs/g2r/g2r_modec_smoke_scr5.yaml` | `g2r_modec_smoke` | scr5 | 500 | Smoke test, K=2 |
| `configs/g2r/g2r_pilot_scr5_adv1e-3.yaml` | — | scr5 | Phase 3 arm | omega_adv = 1e-3 |
| `configs/g2r/g2r_pilot_scr5_adv3e-3.yaml` | — | scr5 | Phase 3 arm | omega_adv = 3e-3 |
| `configs/g2r/g2r_pilot_scr5_adv1e-2.yaml` | — | scr5 | Phase 3 arm | omega_adv = 1e-2 |
| `configs/g2r/g2r_pilot_sanity.yaml` | — | — | Sanity | — |
| `configs/g2r/g2r_r2_scr5_betactrl.yaml` | `g2r_r2_scr5_betactrl` | scr5 | 12000 | Round 2: closed-loop beta_SD controller |
| `configs/g2r/g2r_r2_sanity.yaml` | — | — | Sanity | — |

Round 2 was authorized by `configs/g2r/ROUND2_AMENDMENT.md` (pre-registered
2026-06-13, before execution), which changed exactly one thing versus the
Phase 3 pilot arms: replaced fixed `beta_SD` with a closed-loop controller
updating every 500 steps via `beta_SD <- beta_SD * exp(0.1 * (r_t - s_t) /
max(r_t, 1e-8))`, clamped to [0.01, 10]. The amendment fired because all
three Phase 3 arms failed G-MEAN; the adopted diagnosis was a
variance-saturation attractor (arm 1: pixel std pinned at 0.4841 near output
range bound, mean PSNR collapsed, edge_rho ~ 0).

---

## Negative result record

**z-collapse.** The smoke sampler (`smoke_sampling` in
`src/phase79_rad5_rowspace_diversity_diagnostic.py`) draws K=16 samples from
a single measurement by varying the input noise z. Logged `smoke_mean_pixel_std`
was ~7.19e-4 — the generator ignored the noise input and produced essentially
the same output regardless of z. This is the canonical collapse mode, logged
with tag `z_collapsed_not_viable`.

**Stop-rule firings.** All three Phase 3 pilot arms (omega_adv 1e-3 / 3e-3 /
1e-2) failed the G-MEAN gate at final evaluation. Round 2 (closed-loop
beta_SD, 12000 steps) was executed per the pre-registered amendment; final
gate outcome: stop-rules failed.

**Per-sample evidence rule.** Every claim above is grounded in logged outputs
from actual runs (diagnostic_protocol.json, training_log.csv,
diagnostic_summary.json under the output dirs named in the configs, and
RUNLOG.md under the phase79 output dir). No claim is inferred from model
design alone.

---

## Design constraints (for re-readers)

- **Discriminator hard rule** (enforced in `src/g2r_modec.py` docstring): D
  receives exactly `concat(candidate, x_data)` and nothing else — no Av-y,
  no RelMeasErr, no residual-derived feature in any form.
- **P0 artifact** is loaded from a verified float32 file; it is never
  rebuilt inside the sampler. SHA-256 is checked on load when
  `expected_sha256` is supplied.
- **Split guard**: TRAIN loader uses `train_split=unlabeled`; val uses
  `val_split=train` (verified disjoint); test split is never evaluated during
  training.
- **Certificate semantics**: the exact (lambda=0) x_star audit fits the
  *recorded* noisy y by design (consistency with the recorded measurement,
  not with the unknown clean signal); lambda is logged in the certificate
  tuple.

---

## Relationship to other stages

- Stage 6 (`06_gauge_gan_rad5/`) is the active main case study. Its Rad-5
  checkpoint is the warm-start source for `src/phase79_rad5_rowspace_diversity_diagnostic.py`
  (see `p73.REGIMES["rad5"]["checkpoint"]`).
- Stage 5 (`05_range_null_barrier/`) establishes the theoretical boundary
  this line tried to exploit: measurement-consistent images can be wrong, so
  diversity in null space is meaningful — but the sampler could not produce it.
- Stage 8 (`08_vqgan_fcc/`) is an independent compatibility subline; do not
  conflate with G2R results.
