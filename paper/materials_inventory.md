# Materials Inventory For IEEE TCI Ghost-Imaging Paper

Generated: 2026-06-15

Scope: inventory only. This file lists experimental numbers and source files found under
`E:\ns_mc_gan_gi` and `E:\ns_mc_gan_gi_code`. It does not draft manuscript prose.

Code state used for inventory:

- repo: `E:\ns_mc_gan_gi_code`
- branch: `pub-colab-runner`
- commit: `688f48ecb61f489c8d5849b9913c25335fc36896`
- note: working tree has many untracked experiment scripts/configs; no result files were modified by this inventory pass.

## Overall Status

- Items 1-4, 6-14: data located.
- Item 5: range-share table located; exact phrase/table label "flat corner / enhancement corner" not found as a literal label. Supporting data for row-space share, BP anchors, and learned flat PSNR movement is listed.
- Special Scr-5 conflict check: protocol-specific inconsistent LPIPS values found and recorded. Phase77 already locks canonical Scr-5 numbers to Phase75/Phase74-family aggregate, but the paper should use only one chosen table.

## 1. Main Reconstruction Table

Source table:

- `E:\ns_mc_gan_gi\outputs_phase30_submission_package\latex_project_submission\tables\table1_primary_results.csv`
- RelMeasErr reference rows:
  `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T1_posthoc_external.csv`

Primary STL-10 rows:

| regime | PSNR | SSIM | BP PSNR | Delta PSNR | RelMeasErr pre | RelMeasErr post | clamped pre | clamped post |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rad-5 | 22.316 | 0.635 | 7.297 | 15.019 | 3.775437e-05 | 2.148557e-09 | 6.991007e-03 | 6.967022e-03 |
| Scr-5 | 22.271 | 0.632 | 14.310 | 7.961 | 5.509281e-03 | 5.503777e-06 | 5.452261e-03 | 2.277363e-03 |
| Rad-10 | 24.781 | 0.747 | 7.756 | 17.025 | 5.871699e-05 | 7.612091e-09 | 6.405161e-03 | 6.369322e-03 |
| Scr-10 | 24.730 | 0.746 | 14.533 | 10.197 | 5.712462e-03 | 5.706755e-06 | 5.993919e-03 | 2.354499e-03 |

Higher-precision provenance source:

- `E:\ns_mc_gan_gi\results\cert_package_20260612\PROVENANCE.json`

Higher-precision model values in provenance:

- Rad-5: PSNR 22.31601301645809, SSIM 0.6345534668934895, BP PSNR 7.296873635085412
- Scr-5: PSNR 22.27078430026309, SSIM 0.6316654165614202, BP PSNR 14.309992579950329
- Rad-10: PSNR 24.781177393278735, SSIM 0.7472222633219942, BP PSNR 7.756178314639594
- Scr-10: PSNR 24.73013209707308, SSIM 0.7456648372923276, BP PSNR 14.532684229566472

## 2. Ablation: Gate/Audit/Measurement-Loss Removal

PSNR source:

- `E:\ns_mc_gan_gi\outputs_phase30_submission_package\latex_project_submission\tables\table3_ablation_summary.csv`

Post-hoc RelMeasErr source:

- `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T1_posthoc_external.csv`

Published ablation PSNR summary:

| method | Full | -MC | -Null | Stage1 | Raw | EMA |
|---|---:|---:|---:|---:|---:|---:|
| Rad-5 | 22.202 | 19.399 | 22.202 | 21.736 | 22.065 | 22.202 |
| Scr-5 | 22.155 | 6.352 | 22.154 | 21.294 | 22.050 | 22.155 |
| Rad-10 | 24.676 | 20.106 | 24.676 | 23.598 | 24.539 | 24.676 |
| Scr-10 | 24.608 | 6.352 | 24.607 | 22.492 | 24.518 | 24.608 |

Audit degradation rows, 5% ablations:

| row | PSNR pre | PSNR post | dPSNR | RelMeasErr pre | RelMeasErr post | contraction |
|---|---:|---:|---:|---:|---:|---:|
| rad5_no_final_audit | 22.1923 | 22.2059 | 0.0136 | 3.683409e-02 | 1.895822e-06 | 5.1469e-05 |
| rad5_no_gate_no_final_audit | 22.1859 | 22.1995 | 0.0136 | 3.686602e-02 | 1.898175e-06 | 5.1488e-05 |
| rad5_no_final_audit_no_meas_loss | 22.1803 | 22.2186 | 0.0384 | 7.304620e-02 | 3.761122e-06 | 5.1490e-05 |
| scr5_no_final_audit | 22.1459 | 22.1846 | 0.0387 | 1.803904e-02 | 1.802102e-05 | 9.9900e-04 |
| scr5_no_gate_no_final_audit | 22.1461 | 22.1846 | 0.0385 | 1.801574e-02 | 1.799774e-05 | 9.9900e-04 |
| scr5_no_final_audit_no_meas_loss | 22.1384 | 22.1705 | 0.0322 | 2.092735e-02 | 2.090644e-05 | 9.9900e-04 |

Hard-audit small-n anchor check:

- `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T1_anchor_check_phase53D.csv`

## 3. Post-Hoc Audit On BP / Tikhonov / CS-TV / Learned

Main post-hoc source:

- `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T1_posthoc_external.csv`

TV grid source:

- `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T1_tv_grid.csv`

Tikhonov and expanded frontier source:

- `E:\ns_mc_gan_gi\outputs_phase67_required_zero_training_checks\frontier_morozov.csv`

Classical baseline caveat:

- `E:\ns_mc_gan_gi\outputs_phase67_required_zero_training_checks\TABLE4_SAFE_CAPTION.md`
- `E:\ns_mc_gan_gi\outputs_phase67_required_zero_training_checks\EXTERNAL_BASELINE_AUDIT_FINAL.md`

Main T1 selected rows:

| method | ens | PSNR pre | PSNR post | dPSNR | RelMeasErr pre | RelMeasErr post | contraction |
|---|---|---:|---:|---:|---:|---:|---:|
| bp_pipeline | rad5 | 7.2969 | 7.2969 | 0.0000 | 5.168450e-05 | 3.038277e-09 | 5.8785e-05 |
| tv_pgd_best | rad5 | 8.4932 | 8.4933 | 0.0001 | 1.308185e-02 | 8.548859e-07 | 6.5349e-05 |
| rad5_no_final_audit | rad5 | 22.1923 | 22.2059 | 0.0136 | 3.683409e-02 | 1.895822e-06 | 5.1469e-05 |
| bp_pipeline | scr5 | 14.3100 | 14.3100 | 0.0000 | 1.049725e-07 | 1.048677e-10 | 9.9900e-04 |
| tv_pgd_best | scr5 | 15.8433 | 15.8436 | 0.0003 | 3.077967e-03 | 3.074892e-06 | 9.9900e-04 |
| scr5_no_final_audit | scr5 | 22.1459 | 22.1846 | 0.0387 | 1.803904e-02 | 1.802102e-05 | 9.9900e-04 |
| bp_pipeline | rad10 | 7.7562 | 7.7563 | 0.0001 | 1.048204e-04 | 1.383711e-08 | 1.3201e-04 |
| tv_pgd_best | rad10 | 9.4058 | 9.4060 | 0.0002 | 1.143788e-02 | 2.007673e-06 | 1.7553e-04 |
| bp_pipeline | scr10 | 14.5327 | 14.5327 | 0.0000 | 1.347325e-07 | 1.345979e-10 | 9.9900e-04 |
| tv_pgd_best | scr10 | 18.5814 | 18.5818 | 0.0004 | 3.565890e-03 | 3.562328e-06 | 9.9900e-04 |

Tikhonov rows at lambda=1e-3 from `frontier_morozov.csv`:

| method_id | n | PSNR before | PSNR after | dPSNR | RelMeasErr median before | RelMeasErr median after |
|---|---:|---:|---:|---:|---:|---:|
| full_rad5 | 2000 | 6.667125658741695 | 6.667125658962761 | 2.210653334833168e-10 | 5.1753801641843866e-05 | 3.0479544141944038e-09 |
| full_scr5 | 2000 | 14.309962231858565 | 14.309989951649078 | 2.771979051150275e-05 | 9.989393084509756e-04 | 9.979413670838919e-07 |
| full_rad10 | 2000 | 6.9120516172592215 | 6.912051619624914 | 2.365692359451721e-09 | 1.0488405198203114e-04 | 1.3847036501536443e-08 |
| full_scr10 | 2000 | 14.532654492410002 | 14.532683712809977 | 2.9220399974366983e-05 | 9.989409514109973e-04 | 9.97943008402595e-07 |

CS-TV rows at lambda=1e-3 from `frontier_morozov.csv` are n=8 small-subset evidence:

| method_id | n | PSNR before | PSNR after | dPSNR | RelMeasErr median before | RelMeasErr median after |
|---|---:|---:|---:|---:|---:|---:|
| full_rad5 | 8 | 8.872236372457376 | 8.872237084409928 | 7.119525509136082e-07 | 1.1113120686602647e-03 | 6.378245379800889e-08 |
| full_scr5 | 8 | 13.189632989640765 | 13.189886976345349 | 2.5398670458431916e-04 | 3.196906184593583e-03 | 3.1937124721214624e-06 |
| full_rad10 | 8 | 9.0117098438613 | 9.011716175487159 | 6.331625859945511e-06 | 2.1070200588977814e-03 | 2.634284969481863e-07 |
| full_scr10 | 8 | 13.612052321932822 | 13.612469281423767 | 4.1695949094533447e-04 | 3.948005399332543e-03 | 3.944061337994549e-06 |

Important caveat: CS-TV rows are explicitly documented as a small-subset iterative sanity check, not a full-scale baseline claim.

## 4. Shrink Theorem / Modal Contraction

Sources:

- `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T3_contraction_summary.csv`
- `E:\ns_mc_gan_gi\outputs_phase67_required_zero_training_checks\modal_contraction_results.csv`

T3 summary:

| ens | variant | k | n_images | max mode dev pipeline f32 | max mode dev float64 | within 5pct | sigma min | sigma max |
|---|---|---:|---:|---:|---:|---|---:|---:|
| rad5 | abl_rad5_no_final_audit | 1 | 256 | 1.087e-01 | 1.044e-10 | True | 3.4757 | 5.4543 |
| rad5 | abl_rad5_no_final_audit | 2 | 256 | 1.051e+04 | 2.071e-06 | False | 3.4757 | 5.4543 |
| scr5 | abl_scr5_no_final_audit | 1 | 256 | 7.743e-03 | 2.286e-12 | True | 1.0000 | 1.0000 |
| scr5 | abl_scr5_no_final_audit | 2 | 256 | 4.495e+01 | 1.403e-08 | False | 1.0000 | 1.0000 |

Interpretation stored in source report: float64 k=1 follows the lambda/(lambda+sigma^2) contraction, while pipeline float32 hits a solver floor for repeated k.

## 5. Range-Share Law

Source:

- `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T5_rho.csv`
- interpretive report: `E:\ns_mc_gan_gi\results\cert_package_20260612\REPORT.md`

Formula inventoried: `DeltaPSNR_max = -10 log10(1-s)`.

Rows:

| ens | convention | rho mean | rho std | implied ceiling dB | measured PRx PSNR | pipeline BP PSNR |
|---|---|---:|---:|---:|---:|---:|
| rad5 | raw pipeline | 0.05017 | 0.00240 | 6.667 | 7.297 | 7.297 |
| rad5 | per-image mean removed | 0.05018 | 0.00453 | 14.304 | 14.304 | 7.297 |
| scr5 | raw pipeline | 0.80401 | 0.11030 | 14.311 | 14.311 | 14.310 |
| scr5 | per-image mean removed | 0.05174 | 0.00492 | 14.311 | 14.311 | 14.310 |
| rad10 | raw pipeline | 0.10224 | 0.00522 | 6.912 | 7.756 | 7.756 |
| rad10 | per-image mean removed | 0.10062 | 0.00638 | 14.541 | 14.541 | 7.756 |
| scr10 | raw pipeline | 0.81380 | 0.10488 | 14.534 | 14.534 | 14.533 |
| scr10 | per-image mean removed | 0.09929 | 0.00649 | 14.534 | 14.534 | 14.533 |

Supporting data for "learned flat" audit movement is in Item 3: learned rows move PSNR only +0.013 to +0.039 dB under audit, while RelMeasErr collapses orders of magnitude.

[MISSING literal label] I did not find a table or report line literally named "trained flat corner / BP enhancement corner"; use the source data above if that concept is needed.

## 6. Feasible Hallucination

Source:

- `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T4_pairs.csv`

Summary:

- 16/16 rows have `hallucination_residual_smaller_than_truth=True`.
- Rad-5 constructed wrong-image residuals `RelMeasErr_u_vs_yi` range from 2.160e-15 to 4.003e-15.
- Scr-5 constructed wrong-image residuals are 0.000e+00 in all 8 listed rows.
- The truth's own residual is much larger, e.g. Rad rows 3.453e-03 to 7.464e-03; Scr rows 4.047e-03 to 7.616e-03.
- PSNR of feasible wrong image versus target is low, e.g. Rad 7.697 to 11.378 dB; Scr 7.420 to 14.434 dB.

This supports "measurement consistency is not truth." It does not certify semantic correctness.

## 7. A-Drift

Source:

- `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T7_adrift.csv`

Rows:

| ens | drift | RelMeasErr post trueA | PSNR post | dPSNR |
|---|---:|---:|---:|---:|
| rad5 | 0.000 | 1.895822e-06 | 22.2059 | 0.0136 |
| rad5 | 0.005 | 4.893121e-03 | 22.2056 | 0.0133 |
| rad5 | 0.010 | 9.785160e-03 | 22.2048 | 0.0124 |
| rad5 | 0.020 | 1.956309e-02 | 22.2013 | 0.0090 |
| rad5 | 0.050 | 4.879559e-02 | 22.1777 | -0.0146 |
| scr5 | 0.000 | 1.802102e-05 | 22.1846 | 0.0387 |
| scr5 | 0.005 | 1.264876e-03 | 22.1843 | 0.0384 |
| scr5 | 0.010 | 2.529694e-03 | 22.1834 | 0.0374 |
| scr5 | 0.020 | 5.058768e-03 | 22.1798 | 0.0338 |
| scr5 | 0.050 | 1.262806e-02 | 22.1549 | 0.0090 |

Audit residual rises by orders as operator drift increases, while PSNR is nearly flat.

## 8. Wrong-y / Shuffled-y Dependence

Source:

- `E:\ns_mc_gan_gi\results\cert_package_20260612\tables\T6_dependence.csv`

Rows:

| model | clean PSNR | wrong-y PSNR | wrong-y drop | shuffled-y PSNR | shuffled-y drop |
|---|---:|---:|---:|---:|---:|
| main_rad5 | 22.2024 | 9.9619 | 12.2405 | 7.4813 | 14.7212 |
| main_scr5 | 22.1602 | 9.9862 | 12.1740 | 7.6237 | 14.5366 |
| main_rad10 | 24.6783 | 9.8857 | 14.7926 | 7.6522 | 17.0260 |
| main_scr10 | 24.6036 | 9.8866 | 14.7170 | 7.5965 | 17.0071 |

## 9. Gauge Diagnostic

Sources:

- `E:\ns_mc_gan_gi\outputs_phase69A_gauge_gan_signal_diagnostic\critic_auc_results.csv`
- `E:\ns_mc_gan_gi\outputs_phase69A_gauge_gan_signal_diagnostic\shortcut_control_results.csv`
- `E:\ns_mc_gan_gi\outputs_phase75_final_high_tier_validation\regime_map_final.csv`

Phase69A diagnostic rows:

| model | AUC | CI low | CI high |
|---|---:|---:|---:|
| patchgan_unconditional_gauge | 0.8466262817382812 | 0.823528738092153 | 0.8735512157156857 |
| patchgan_conditional_xdata_gauge | 0.7305564880371094 | 0.6959395492020976 | 0.7583097134268217 |
| simple_cnn_gauge | 0.7696533203125 | 0.7395926980770687 | 0.7983764521695713 |
| residual_features_logistic_raw | 0.9802017211914062 | 0.9671463025813203 | 0.990356648907759 |
| patchgan_conditional_shuffled_xdata_gauge | 0.6640815734863281 | 0.6342699000373376 | 0.6936641800540779 |
| simple_cnn_adagger_gauge_subset | 0.7820167541503906 | 0.7518586780695238 | 0.8080048989895295 |

Regime map:

| regime | gauge AUC | CI / source | outcome | decision |
|---|---:|---|---|---|
| Scr-5 | 0.8466 | Phase69A | 3 paired seeds positive | train/evidence positive |
| Rad-5 | 0.8771 | 0.8446-0.9072 | 3 paired seeds positive | train/evidence positive |
| Scr-10 | 0.6240 | 0.5791-0.6700 | weak gate; no cGAN | stop |
| Rad-10 | 0.6396 | 0.5900-0.6774 | weak gate; no cGAN | stop |

## 10. Shortcut Stress

Source:

- `E:\ns_mc_gan_gi\outputs_phase75_final_high_tier_validation\shortcut_stress_summary.csv`
- report: `E:\ns_mc_gan_gi\outputs_phase75_final_high_tier_validation\SHORTCUT_STRESS_TEST_REPORT.md`

Key alpha=0.1 rows:

| score | base | perturb | mean abs delta vs alpha0 | mean RelMeasErr |
|---|---|---|---:|---:|
| standard_D_score | fake_mean | row | 0.4766808192653116 | 0.11185452935751528 |
| gauge_D_score | fake_mean | row | 0.0 | 0.11185452935751528 |
| gauge_D_score | fake_mean | null | 0.7392268102703383 | 0.005576414199822466 |
| standard_D_score | real | row | 0.3836784531304147 | 0.11294381169136614 |
| gauge_D_score | real | row | 0.0 | 0.11294381169136614 |
| gauge_D_score | real | null | 0.5402423227787949 | 0.005553482866162085 |

## 11. 96px Rad-5 Controlled GAN, Three Paired Seeds

Source:

- `E:\ns_mc_gan_gi\outputs_phase81_96px_rad5_paper_completion\task2_seed_signs.csv`
- report: `E:\ns_mc_gan_gi\outputs_phase81_96px_rad5_paper_completion\PHASE81_96PX_RAD5_COMPLETION_REPORT.md`
- hashes: `E:\ns_mc_gan_gi\outputs_phase81_96px_rad5_paper_completion\all_seed_artifact_hashes.csv`

Rows:

| seed | C-B LPIPS | C-B RAPSD distance | LPIPS C better | RAPSD C better |
|---:|---:|---:|---|---|
| 1 | -0.00654092658078298 | -0.0007121817633988654 | True | True |
| 2 | -0.003782035957556218 | -0.0009262826746460825 | True | True |
| 3 | -0.00426504819188267 | -0.0006804119398238556 | True | True |

Decision in report: Task2 3/3 same-sign decision is True.

Representative checkpoint/per-sample hashes:

- seed1 B checkpoint sha256 `60b02eced5370be9fccb79faf35e6b9ef49a250811f2a6153b865b200700e9ea`
- seed1 C checkpoint sha256 `064a63d0d318c8839265a9b45f71ed12e97dfa32db0de8d242b2ed44d95cf600`
- seed1 B per-sample outputs sha256 `d126d4d306534c0a2aed97449acea167bf5923e8364303f8f0d68fe820b36685`
- seed1 C per-sample outputs sha256 `eba48f072f2640c751c8de8161b4806a96adc5804bc6e9f94c86c7e1034ce791`
- full hash table contains all seeds/arms.

## 12. 96px Standard cGAN Control

Source:

- `E:\ns_mc_gan_gi\outputs_phase81_96px_rad5_paper_completion\task3_standard_vs_gauge_focus.csv`
- report: `E:\ns_mc_gan_gi\outputs_phase81_96px_rad5_paper_completion\PHASE81_96PX_RAD5_COMPLETION_REPORT.md`
- per-seed metrics: `E:\ns_mc_gan_gi\outputs_phase81_96px_rad5_paper_completion\all_seed_evaluation_metrics.csv`

C vs D_standard by seed:

| seed | metric | mean C | mean D standard | D-C | D better by mean |
|---:|---|---:|---:|---:|---|
| 1 | psnr | 19.78938226828819 | 19.787659619736974 | -0.00172264855121438 | False |
| 1 | ssim | 0.4579080610836096 | 0.4578483025467942 | -5.9758536815355764e-05 | False |
| 1 | rapsd_distance | 0.008899136414463039 | 0.008900477991518479 | 1.3415770554391674e-06 | False |
| 1 | lpips | 0.4111089642974548 | 0.4112642565742135 | 0.0001552922767587006 | False |
| 2 | psnr | 19.787651350448904 | 19.788435184655576 | 0.0007838342066705598 | True |
| 2 | ssim | 0.4581340811587213 | 0.45853315604077827 | 0.0003990748820569929 | True |
| 2 | rapsd_distance | 0.008875702377099032 | 0.008903902687419413 | 2.8200310320379854e-05 | False |
| 2 | lpips | 0.4146302882581949 | 0.4146672905771993 | 3.7002319004386663e-05 | False |
| 3 | psnr | 19.78825087245496 | 19.787836149882978 | -0.00041472257197982393 | False |
| 3 | ssim | 0.4585175297508554 | 0.4585310051949587 | 1.3475444103329729e-05 | True |
| 3 | rapsd_distance | 0.008687830214989891 | 0.008683662991273882 | -4.167223716008539e-06 | True |
| 3 | lpips | 0.4126690298435278 | 0.4126583163160831 | -1.071352744475007e-05 | True |

Inventory conclusion: C and D_standard are comparable/mixed, with seed-level wins in both directions. This is not a dominance result.

## 13. Unmeasured-Content Map Quantitative Validation, Negative Result

Source:

- `E:\ns_mc_gan_gi\outputs_phase79_96px_rad5_p0_error_validation\p0_error_correlation_summary.csv`
- report: `E:\ns_mc_gan_gi\outputs_phase79_96px_rad5_p0_error_validation\PHASE79_P0_ERROR_VALIDATION_REPORT.md`
- per-sample outputs: `E:\ns_mc_gan_gi\outputs_phase79_96px_rad5_p0_error_validation\per_sample_pixel_outputs`

Rows:

| arm | pixel Spearman abs P0 vs abs error, unclipped | pixel Pearson | top10 abs P0 error, unclipped | rest90 error, unclipped |
|---|---:|---:|---:|---:|
| A | -0.07925436191691487 | -0.0689764193807897 | 0.25910601019859314 | 0.24179792404174805 |
| B | 0.07326845845045213 | 0.061416217215255475 | 0.07056641578674316 | 0.07944381237030029 |
| C | 0.07185937144058595 | 0.06082814669359347 | 0.07079800218343735 | 0.07941748201847076 |

Negative result: for B/C, top10% `|P0 xhat|` pixels have lower actual error than rest90, so this does not validate `|P0 xhat|` as a pixelwise error map.

## 14. Posterior Calibration Mechanism, Line A

Sources:

- gate table: `E:\ns_mc_gan_gi\outputs_phase82_lineA_closure\lineA_gate_table.csv`
- artifact audit: `E:\ns_mc_gan_gi\outputs_phase82_lineA_closure\lineA_artifact_audit.json`
- scan summary: `E:\ns_mc_gan_gi\outputs_phase81_diversity_weight_scan\scan_summary.csv`
- narrative: `E:\ns_mc_gan_gi\outputs_phase82_lineA_closure\lineA_mechanistic_narrative.md`

Gate values:

| gate | name | key values | status |
|---:|---|---|---|
| 1 | deterministic Rad-5 collapse baseline | std 0.0009831821080297232; P0 var 1.08769949795578e-06; PR var 6.7677122994447035e-15; relmeas max 3.8393187880258514e-05 | FAIL anti-collapse; PASS measurement consistency |
| 2 | patched null/range criterion | same std/P0/PR as gate 1 | baseline cleanly classified as collapse |
| 3 | row-space-only reconstruction diagnostic | std 0.03354055806994438; P0 var 0.001278143240480694; PR var 5.733857513888001e-12; relmeas max 5.605900465501583e-05 | PASS anti-collapse criteria |
| 4 | structured non-white null variation | P0 variation slope -2.6224391447055404; high_to_low 0.0053709204268584795 | PASS spectral sanity |
| 5 | Phase79 calibration audit | pixel_cov_90 0.3809686279296875; P0_cov_90 0.4069453125; kappa_det 1.6154627576277758; P0 mean offset 0.05486070931073664 | FAIL calibration; kappa admissible |
| 6 | P0 mean anchor + centered diversity | std 0.006856838706880808; P0 var 5.189623721070667e-05; pixel_cov_90 0.10560400390625; P0_cov_90 0.1021875; offset 0.017980045462012286 | center improved; spread too narrow |
| 7 | anchor/diversity scan | pixel_cov_90 0.452839599609375; P0_cov_90 0.478072265625; kappa_det 1.3273211406260825; offset 0.03307102203891198; slope -2.1805084142297826 | coverage ceiling |
| 8 | base null-space accuracy bottleneck | base P0 det-to-GT RMSE 0.08096860725645309; best P0_cov_90 0.478072265625; best pixel_cov_90 0.452839599609375 | stop line A expansion |

Best scan row, `div2`:

| lambda_anchor | lambda_diversity | fixed-y std | P0 variance | PR variance | pixel cov 50 | pixel cov 90 | pixel cov 95 | P0 cov 50 | P0 cov 90 | P0 cov 95 | kappa sample mean | kappa deterministic | P0 mean-to-GT RMSE | P0 det-to-GT RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.0 | 2.0 | 0.03487974777817726 | 0.0013928864136743522 | 6.55442587418697e-12 | 0.2027049560546875 | 0.452839599609375 | 0.51262841796875 | 0.215427734375 | 0.478072265625 | 0.53898828125 | 1.1441964218705039 | 1.3273211406260825 | 0.08772128379741215 | 0.08096860725645326 |

LineA checkpoint and sample-bank hashes from `scan_summary.csv`:

- div2 checkpoint: `E:\ns_mc_gan_gi\outputs_phase81_diversity_weight_scan\rad5_centered_anchor2_div2\checkpoints\final.pt`
- div2 checkpoint sha256: `36272cac79610bd302d65a8cf842e7887bc62e85d7934c6bea101bd62043d6ac`
- div2 shard sample hashes: `9354bdb6eea777c336c1c9935729a68459493521df0550bd41ffe9527f5f6861;9352189e5a574558895a716067da99537d9824f4c0eadb5a44f1a5c901c2db91;4b2dd0827d20bf795f6cfab3dd7e7aebed511d93365d0c70a9985866070d41f3;a4ac9e52f082fb5cc3ca294ff3853fb1e9522cf294a245b3c3029fd209909eb0`

## Special Check: Inconsistent Scr-5 Numbers

Sources:

- canonical decision: `E:\ns_mc_gan_gi\outputs_phase77_auditable_gan_paper_assembly\CANONICAL_RUN_DECISION.md`
- canonical table: `E:\ns_mc_gan_gi\outputs_phase77_auditable_gan_paper_assembly\canonical_results_table.csv`
- old-number archive: `E:\ns_mc_gan_gi\outputs_phase77_auditable_gan_paper_assembly\old_numbers_archive.csv`
- Phase70 reproduction: `E:\ns_mc_gan_gi\outputs_phase70_gauge_gan_paper_expansion\phase69B_repro_metrics.csv`
- Phase71 seed table: `E:\ns_mc_gan_gi\outputs_phase71_gauge_cgan_paired_seeds\scr5_seed_metrics.csv`
- Phase75 canonical source: `E:\ns_mc_gan_gi\outputs_phase75_final_high_tier_validation\standard_cgan_seed_metrics.csv`

Found protocol-specific Scr-5 LPIPS values:

| source | arm | seeds | n per seed | LPIPS mean | status |
|---|---|---:|---:|---:|---|
| Phase70 Phase69B reproduction | C | 1/protocol aggregate | 256 | 0.2262847395177232 | archive/support only |
| Phase71 paired seed table | C seed01 | 1 | 256 | 0.22997693179058842 | paired-seed evidence, later archived by Phase77 |
| Phase71 paired seed table | C seed02 | 1 | 256 | 0.23099671996897087 | paired-seed evidence, later archived by Phase77 |
| Phase71 paired seed table | C seed03 | 1 | 256 | 0.23130907566519454 | paired-seed evidence, later archived by Phase77 |
| Phase75 canonical aggregate | C_gauge | 3 | 256 | 0.23076087010364663 | canonical controlled 5% result |
| Phase75 canonical aggregate | D_standard | 3 | 256 | 0.2309550932792869 | canonical controlled 5% result |

Phase77 decision text says:

- Use Phase75/Phase74-family Scr-5 numbers as canonical controlled 5% B/C/standard-D result.
- Phase69B and Phase70 Scr-5 numbers are archived earlier protocol outputs and are not main-text numbers.
- "Older LPIPS/RAPSD rows still exist in the archive for provenance, but they are protocol-specific and should not be mixed with the canonical table."

Inventory conclusion: yes, inconsistent Scr-5 LPIPS numbers exist across runs/protocols, including the 0.226 vs 0.230 pattern. The paper must use one locked source. Existing project lock points to Phase75 canonical aggregate, but this still needs human confirmation before manuscript writing.

## Missing / Caution List

- [MISSING literal label] I found the range-share law table and related audit data, but not a file/table literally named "flat corner / enhancement corner".
- [CAUTION] CS-TV rows are small-subset n=8 evidence, not full-scale baseline evidence.
- [CAUTION] Tikhonov rows are in `frontier_morozov.csv`, not in `T1_posthoc_external.csv`; cite the correct table if used.
- [CAUTION] Scr-5 64px LPIPS has multiple protocol-specific values. Do not mix Phase69B/70, Phase71, and Phase75 numbers.
- [CAUTION] 96px standard cGAN control is mixed/comparable, not a dominance claim.
- [CAUTION] unmeasured-content map quantitative validation is negative for pixelwise error localization.

