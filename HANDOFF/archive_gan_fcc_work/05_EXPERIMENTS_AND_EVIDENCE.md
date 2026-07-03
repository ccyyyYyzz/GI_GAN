# Experiments And Evidence

本文件把“实验分类”和“论文证据”对应起来。原则：active repo 是代码，experiments 是导航，data warehouse 是大证据库，paper/materials_inventory 是论文数字入口。

## Experiment Layer

总入口：

`E:\GAN_FCC_WORK\experiments`

| 实验目录 | 角色 | 使用方式 |
|---|---|---|
| `00_core_platform_shared_measurement_models` | 共享测量/模型平台 | 查测量、audit、split、训练基础设施 |
| `01_operator_pattern_learning_phases03_08` | 早期 operator/pattern 探索 | 历史参考，不作为当前主 claim |
| `02_baselines_measurement_audit_phases09_17` | measurement audit certificate backbone | 主论文 Section 3/4 的关键证据来源之一 |
| `03_manuscript_mechanism_phases18_45` | 论文图表和机制构建 | 查旧图表、caption、LaTeX 资产 |
| `04_range_null_counterfactual_barrier_phases48_60` | feasible wrong image / audit 边界 | 主论文“certificate limits”的核心证据 |
| `05_paper1_publication_evidence_phase67` | paper1/zero-training certificate branch | 旧稿与补充证据 |
| `06_gauge_gan_rad5_current_phases69_83` | 当前 Gauge-GAN/Rad-5 case study | 当前主线生成式范例 |
| `07_g2r_posterior_sampling_side_line` | posterior/G2R diagnostic side branch | 只在明确任务下恢复 |
| `08_vqgan_fcc_phase1_compatibility` | VQGAN/FCC full GI-related extraction | 相关论文/支线证据 |
| `09_colab_publication_ops` | Colab/packaging/publication ops | 算力与打包支持 |
| `99_legacy_misc_review_before_use` | 未完全归类旧物 | 先审计再使用 |

每个目录的 `MANIFEST.csv` 指向真实 payload。不要把 experiment folder 当成源代码副本。

## Current Paper Evidence

当前主论文：

```text
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.tex
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.pdf
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\materials_inventory.md
```

`materials_inventory.md` 是论文数字的权威入口。写作时不能凭记忆填数；缺失就标 `[DATA MISSING]`。

## Audit / Certificate Evidence

主要位置：

```text
E:\GAN_FCC_WORK\experiments\02_baselines_measurement_audit_phases09_17
E:\GAN_FCC_WORK\data_warehouse\ns_mc_gan_gi
```

用于支持：

- post-hoc audit 对不同重建器都能降低 measurement residual；
- PSNR 与 RelMeasErr 的分离；
- BP 与 trained network 在 row-space error fraction 上的差异解释；
- soft audit / Morozov noise-floor 的诚实讨论。

## Boundary / Feasible Wrong Image Evidence

主要位置：

```text
E:\GAN_FCC_WORK\experiments\04_range_null_counterfactual_barrier_phases48_60
```

用于支持：

- 同一 measurement 下存在不同可行图像；
- audit consistency 不等价于 semantic correctness；
- 零空间内容的量、分布和正确性不能被 measurement residual 单独认证。

已整理的 feasible wrong image 子集曾被选中：

```text
rank011_u_i1789_j1947_cat_clipped
rank118_u_i1789_j567_dog_clipped
rank057_u_i1789_j1953_bird_clipped
rank106_u_i1789_j452_airplane_clipped
rank031_u_i1789_j1050_ship_clipped
```

若要写图注，必须回到对应文件夹查图像和生成记录。

## Gauge-GAN / Rad-5 Evidence

主要位置：

```text
E:\GAN_FCC_WORK\experiments\06_gauge_gan_rad5_current_phases69_83
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\scripts
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\src
```

用于支持：

- GAN 作为 auditable generative prior 的 case study；
- LPIPS/RAPSD 这类感知/谱指标改善；
- RelMeasErr 在 audit 后保持受控；
- PSNR 差异只能谨慎措辞，不作为卖点。

## VQGAN/FCC Evidence

复制版主入口：

```text
E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703
```

关键证据：

```text
CLAUDE_CODE_HANDOFF.md
locked_bundle\PROJECT_BRIEF_VQGAN_DETAIL_FUSION_LOCKED.md
outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.pdf
outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.tex
outputs\compatibility\measurement_conditioned_vqgan
outputs\compatibility\measurement_conditioned_vqgan\VQGAN_MULTI_SEED_PARETO_CONFIRMATION_PACKAGE.zip
outputs\compatibility\multiseed_pareto_confirmation
outputs\compatibility\fcc_diagnostic_canary64
outputs\compatibility\structure_detail_fcc
```

用途：

- 证明 VQGAN/FCC 有一篇完整 detail-fusion 草稿，而不只是零散证据；
- 作为后续 paper 或 supplement 候选；
- 不自动并入当前 IEEE TCI 主 claim。

## Machine-Readable Inventories

关键 inventory：

```text
E:\GAN_FCC_WORK\inventory\phase_step_catalog_active_repo.csv
E:\GAN_FCC_WORK\inventory\phase_step_catalog_summary.csv
E:\GAN_FCC_WORK\inventory\work_root_category_catalog.csv
E:\GAN_FCC_WORK\inventory\fcc_phase1_gi_related_included_files_20260703.csv
E:\GAN_FCC_WORK\inventory\fcc_phase1_gi_related_excluded_files_20260703.csv
E:\GAN_FCC_WORK\inventory\fcc_phase1_gi_related_copy_verify_payload_after_readme_20260703.csv
```

这些文件用于审计和定位，不等于论文结果本身。
