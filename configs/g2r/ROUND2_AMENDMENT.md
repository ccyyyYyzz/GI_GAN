# Round 2 Amendment — pre-registered 2026-06-13 (before execution)

Execute ONLY after the three-arm Phase 3 report, and ONLY if the registered
stop rule fired (all arms fail G-MEAN).

Adopted diagnosis: fixed beta_SD has a variance-saturation attractor
(arm 1: std pinned at 0.4841 ~ output-range bound, mean PSNR collapsed,
edge_rho ~ 0); the principled rcGAN mechanism is closed-loop calibration.

Change exactly ONE thing — replace fixed beta_SD with a controller:

- Every 500 steps, on a fixed seed-pinned val batch, measure:
  r_t = per-pixel mean |residual| of the K-sample mean (calibration target),
  s_t = median per-pixel sample std.
- Update: beta_SD <- beta_SD * exp(0.1 * (r_t - s_t) / max(r_t, 1e-8)),
  clamped to [0.01, 10]; log beta_SD at every update.
- omega_adv: the smallest round-1 arm that visibly contained variance
  growth; if none did, 1e-2. (To be resolved from the three-arm report.)
- Everything else unchanged: K=4, TTUR, gates, guards, exact_x_star_audit,
  run IDs g2r_r2_*. 12000 steps, trajectory every 1000, scr5, one seed.
- Success path: if G-MEAN and G-DIV pass at final eval, add 2 seeds.

Report: gate table + the beta_SD trajectory (it should settle to a finite
band) + std-vs-r_t tracking curve.
