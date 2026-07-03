# Research Lineage Reconstructed From E Drive

This is the reconstructed research route from the E-drive code, README files, git state, phase scripts, and generated catalogs.

## 0. Legacy PCCN-GI Precursor

Source: `E:\GAN_FCC_WORK\project_sources\pc_cgan_gi_precursor_sourceonly`

Original: `E:\pc_cgan_gi`

This was the early synthetic ghost-imaging scaffold: random speckle/synthetic grayscale data, CGI low-quality baseline, conditional U-Net/PCCN-GI reconstruction, L1 + measurement-domain physics consistency, and optional GAN/SSIM losses as ablations. The README explicitly says GAN and SSIM were retained as references but were not the final recommended model. This is historical context, not the current paper line.

## 1. NS-MC-GAN Measurement-Consistent Reconstruction Core

Source: `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code`

The next line moved from generic conditional enhancement to the range/null-space idea. The core model uses `y = A x + epsilon`, a data solution `A^T(AA^T + lambda I)^{-1}y`, null-space residual generation, and a data-consistency projection/audit. Core modules include measurement construction, datasets, generators/critics, losses, metrics, train/eval, split guards, and exact/soft measurement operations.

## 2. Operator And Pattern-Learning Exploration, Phases 3-8

Source: active repo phase files and configs.

This line explored binary/continuous illumination patterns, operator calibration, learned physical illumination, pattern causality controls, flip-aware binary STE, and continuous differential patterns. The recurring question was whether gains came from learned measurement operators or from generator fine-tuning.

## 3. Hadamard/Rademacher Baselines And Audit Locking, Phases 9-17

Source: active repo phase files and `E:\GAN_FCC_WORK\data_warehouse\ns_mc_gan_gi` outputs.

This line established structured low-frequency Hadamard and Rademacher regimes, clean imports, no-leak checks, locked certificates, exact measurement audit behavior, paper tables/figures, and attribution. It created the measurement-audit backbone later reused by the paper.

## 4. Manuscript/Mechanism Construction, Phases 18-45

Source: active repo and paper-building phase scripts.

This line converted the evidence into manuscript assets: LaTeX builders, figure/table generators, mechanism figures, conventional GI anchors, provenance decomposition, submission package checks, and reproducibility/math rewrites. It is a paper-construction line, not a new model family.

## 5. Range-Null Boundary And Feasible Counterfactual Evidence, Phases 48-60 plus Paper1 Phase67

Sources:

- `research_lines\03_range_null_counterfactual_barrier_phases_48_60`
- `project_sources\paper1_publication_evidence_sourceonly`

This line sharpened the mathematical boundary: measurement constrains row-space content, while null-space content can vary without changing buckets. It includes exact null pairs, feasible counterfactuals, cross-audits, sampling-mode/provenance audits, hallucination POCs, Morozov/noise-floor audits, and phase67 zero-training checks in the paper1 branch. This is the origin of the paper's central "certificate has limits" argument.

## 6. G2R / Posterior-Sampling Side Line

Source: `research_lines\05_g2r_and_posterior_sampling`

This side line attempted posterior/generator-to-reconstruction protocols and gating. The local bare repo log records a G2R Round 2 / Mode-C branch with stop-rule failures. Treat it as a diagnostic side branch unless explicitly revived.

## 7. Gauge-GAN / Rad-5 Auditable Generative Case Study, Phases 69-83

Source: `research_lines\04_gauge_gan_rad5_case_study_phases_69_83`

This is the current active paper case study. It includes Scr-5/Rad-5 gauge GAN diagnostics, controlled gauge cGAN pilots, paired seeds, high-tier validation, 96px Rad-5 paper completion, posterior sampling/anti-collapse diagnostics, LPIPS/RAPSD/PSNR/RelMeasErr reporting, and phase83 seeds 4-10. The scientific role is a generative example under audit, not a claim of image-quality SOTA.

## 8. FCC / VQGAN Compatibility Branch

Source: `project_sources\fcc_phase1_vqgan_fcc_extract_sourceonly`

Original frozen/mixed location: `E:\ns_mc_gan_gi_code_fcc_phase1`

This branch contains measurement-conditioned VQGAN / VQAE prior transfer, anchor-initialized VQGAN inversion, VQGAN multi-seed Pareto confirmation, FCC row-null/structure-detail compatibility diagnostics, Bayesian witness experiments, and Colab wrappers. It is scientifically related but contaminated by unrelated ZIFB and Fresnel manuscript files in the original folder, so only the source extract should be used.

## Current Practical Rule

Use `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code` for new work. Use `project_sources` and `research_lines` for archaeology/reference. Keep `E:\GAN_FCC_WORK\data_warehouse\ns_mc_gan_gi` as the large output backing store, and do not promote any archive/extract file into active code without an explicit review step.
