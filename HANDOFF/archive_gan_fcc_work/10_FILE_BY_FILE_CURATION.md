# File-By-File Curation Report

这份报告记录 2026-07-03 的逐文件整理结果。它不是粗略目录判断，而是把 `E:\GAN_FCC_WORK` 和 frozen `E:\ns_mc_gan_gi_code_fcc_phase1` 的每个文件都写入 CSV，再按规则给出研究线、文件族、论文标签和处理建议。

## Scope

已逐文件入账：

```text
E:\GAN_FCC_WORK
E:\ns_mc_gan_gi_code_fcc_phase1   # read-only frozen source
```

未纳入：

```text
E:\GAN_FCC_BACKUP
```

原因：backup root 是重复/备份/归档层，和工作根混扫会放大重复文件。需要整理 backup 时应另开 backup-only 审计。

## Main Inventories

第一版全量逐文件清单：

```text
E:\GAN_FCC_WORK\inventory\file_by_file_curation_20260703.csv
```

第二版精分清单：

```text
E:\GAN_FCC_WORK\inventory\file_by_file_curation_v2_20260703.csv
```

第三版精分清单（当前推荐使用）：

```text
E:\GAN_FCC_WORK\inventory\file_by_file_curation_v3_20260703.csv
```

第二版汇总：

```text
E:\GAN_FCC_WORK\inventory\file_by_file_curation_v2_summary_20260703.csv
```

第三版汇总（当前推荐使用）：

```text
E:\GAN_FCC_WORK\inventory\file_by_file_curation_v3_summary_20260703.csv
```

论文/稿件子集：

```text
E:\GAN_FCC_WORK\inventory\file_by_file_paper_manuscript_subset_20260703.csv
E:\GAN_FCC_WORK\inventory\main_tex_pdf_versions_20260703.csv
E:\GAN_FCC_WORK\inventory\main_tex_pdf_versions_annotated_20260703.csv
E:\GAN_FCC_WORK\inventory\paper_locations_validation_20260703.csv
```

待复核/清理候选：

```text
E:\GAN_FCC_WORK\inventory\review_before_use_files_v2_20260703.csv
E:\GAN_FCC_WORK\inventory\review_before_use_files_v3_20260703.csv
E:\GAN_FCC_WORK\inventory\cleanup_candidates_cache_generated_20260703.csv
```

## Counts

全量逐文件行数：

```text
92,461 files
```

其中：

```text
WORK_ROOT                  84,764
FROZEN_FCC_PHASE1_READONLY  7,697
```

第三版主要研究线分配：

```text
10_runtime_environment_cache             36,812
02_baselines_audit_phases09_17           18,079
01_operator_pattern_phases03_08          15,381
08_vqgan_fcc_phase1                       9,343
03_manuscript_mechanism_phases18_45       4,264
00_git_internal                           2,012
06_gauge_gan_rad5_phases69_83             1,778
04_range_null_counterfactual_phases48_60    991
00_core_platform                            918
PAPER_B_publication_evidence_old            816
09_colab_ops_packaging                      518
99_review_needed                            373
PAPER_A_range_null_audit_current            287
00_early_phase2_clean_or_locked_runs        204
07_g2r_posterior_sampling                   200
PAPER_C_vqgan_detail_fusion                 106
00_dataset                                   95
03_manuscript_claim_support                  70
MIXED_NON_GI_fresnel_manuscript              62
05_paper1_publication_evidence               62
03_manuscript_history_iterations             48
00_early_quick_runs_reviewed                 42
```

第二版处理建议：

```text
KEEP_IN_CURRENT_ORGANIZED_LOCATION       44,042
KEEP_RUNTIME_CACHE_OR_REBUILD_EXPLICITLY 36,812
READ_ONLY_SOURCE_DO_NOT_EDIT              7,697
REVIEW_BEFORE_USE_V2                        951
DO_NOT_TOUCH_GIT_INTERNAL                   876
KEEP_PAPER_B_REFERENCE_BRANCH               816
CLEAN_OR_ARCHIVE_AFTER_CONFIRM_SAFE         806
KEEP_PRIMARY_PAPER_A_OR_SUPPORT             287
KEEP_PACKAGE_OR_HASH_THEN_BACKUP             73
KEEP_PAPER_C_DRAFT_AND_SUPPORT               53
KEEP_AS_HISTORICAL_MANUSCRIPT_ITERATION      48
```

## Papers Found

The canonical paper index is:

```text
E:\GAN_FCC_WORK\handoff\09_PAPERS_INDEX.md
```

Paper A: current range-null audit main paper.

```text
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.tex
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.pdf
```

Paper B: old paper1/publication evidence draft.

```text
E:\GAN_FCC_WORK\project_sources\paper1_publication_evidence_sourceonly\paper1_analysis\paper1_draft_PR.tex
E:\GAN_FCC_WORK\project_sources\paper1_publication_evidence_sourceonly\paper1_analysis\paper1_draft_PR.pdf
```

Paper C: VQGAN Detail Fusion full draft.

```text
E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.tex
E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.pdf
E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.md
```

Frozen original of Paper C remains read-only:

```text
E:\ns_mc_gan_gi_code_fcc_phase1\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.pdf
```

## Main.tex / Main.pdf Audit

The project contains many `main.tex` / `main.pdf` files, but they are not all active papers.

Annotated list:

```text
E:\GAN_FCC_WORK\inventory\main_tex_pdf_versions_annotated_20260703.csv
```

Result:

```text
50 historical manuscript iterations
2 Paper A copies from fcc_phase1
2 mixed Fresnel root manuscripts
2 table assets, not manuscripts
1 review item: source_package\main.tex from phase30 submission package
```

The mixed Fresnel files are not GAN/GI/FCC papers:

```text
E:\ns_mc_gan_gi_code_fcc_phase1\main.tex
E:\ns_mc_gan_gi_code_fcc_phase1\main.pdf
E:\ns_mc_gan_gi_code_fcc_phase1\_manuscript_view\*.pdf
```

They relate to:

```text
菲涅耳波带片的目标光场编码设计：稳妥基准方案与目标光场编码创新方案
```

## What Is Still Not Fully Curated

Third-pass remaining review set:

```text
E:\GAN_FCC_WORK\inventory\review_before_use_files_v3_20260703.csv
```

This contains 373 files. These are not deleted and not promoted. They should be reviewed before use or before any physical move. They include files whose path still does not clearly encode a research line, mostly unlabelled paper figures, small scripts, reports, and a few generated cache files.

## Cleanup Candidates

Potential generated/cache cleanup:

```text
E:\GAN_FCC_WORK\inventory\cleanup_candidates_cache_generated_20260703.csv
```

Second-pass cleanup recommendation count:

```text
806 files
```

Do not delete automatically. First confirm they are generated artifacts and not evidence. The safe cleanup rule is:

1. hash or list the candidates;
2. confirm they are regenerated caches;
3. move to backup/quarantine or delete only after explicit approval.

## Handling Rules

- `READ_ONLY_SOURCE_DO_NOT_EDIT`: never edit/move/delete; this includes frozen `fcc_phase1`.
- `KEEP_PRIMARY_PAPER_A_OR_SUPPORT`: current active paper assets.
- `KEEP_PAPER_B_REFERENCE_BRANCH`: old paper1 evidence branch.
- `KEEP_PAPER_C_DRAFT_AND_SUPPORT`: VQGAN Detail Fusion paper assets.
- `KEEP_RUNTIME_CACHE_OR_REBUILD_EXPLICITLY`: environment/runtime files; do not confuse with scientific evidence.
- `KEEP_AS_HISTORICAL_MANUSCRIPT_ITERATION`: old paper build versions; keep for provenance unless explicitly archived.
- `REVIEW_BEFORE_USE_V2`: do not cite or delete before manual inspection.

## Next Physical Organization Step

Do not move files based only on this report. The safe next step is to choose one subset and act:

1. Paper C promotion: copy `detail_fusion_paper` into a proper `paper_work\vqgan_detail_fusion` workspace.
2. Cache cleanup: review the 806 generated candidates and archive/delete after approval.
3. Historical manuscript archive: move old phase18-45 LaTeX iterations to a clearer archive folder after hash manifest.
4. Review set reduction: manually inspect the 373 `review_before_use_files_v3` files and assign each to a line or archive.
