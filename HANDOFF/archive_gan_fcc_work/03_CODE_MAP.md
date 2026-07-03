# Code Map

本文件回答一个实际问题：新 agent 应该在哪儿改代码、在哪儿找证据、哪些东西只读。

## Active Repo

主代码库：

`E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code`

这是后续新工作默认进入的位置。当前分支是 `pub-colab-runner`，最近记录的 HEAD 是 `688f48e`。这个 repo 目前是 dirty worktree，先读再改，不要先清理。

## Active Repo 内部结构

| 路径 | 用途 | 接手规则 |
|---|---|---|
| `src` | 核心 Python 代码、phase 脚本、模型/评估逻辑 | 新实验主要改这里 |
| `scripts` | PowerShell/Bash 包装脚本、Colab/import/build wrappers | 运行前先读脚本头和输出路径 |
| `configs` | 训练/评估配置 | 不要静默改历史配置；新配置另存 |
| `tests` | split guard、checkpoint wiring、protocol 等测试 | 改共享逻辑后优先跑 |
| `paper` | 当前主论文 LaTeX、PDF、材料清点 | 写 paper 时只用 `materials_inventory.md` 中数字 |
| `results` / `outputs` / `artifacts` | active repo 内的结果与审计资产 | 当证据用，移动前必须建 manifest |
| `colab` | Colab runner 文档/辅助 | 只有需要远程算力时使用 |
| `eval` | 评估和 convention bridge | 指标变动要小心，会影响论文表格 |

## Experiment Navigation

实验层入口：

`E:\GAN_FCC_WORK\experiments`

这里不复制代码，而是把 active repo、warehouse、source-only 分支按科学实验分组。每个子目录都有 `README.md` 和 `MANIFEST.csv`。

最常用：

- 当前 Gauge-GAN/Rad-5：`E:\GAN_FCC_WORK\experiments\06_gauge_gan_rad5_current_phases69_83`
- range-null / feasible wrong image：`E:\GAN_FCC_WORK\experiments\04_range_null_counterfactual_barrier_phases48_60`
- VQGAN/FCC：`E:\GAN_FCC_WORK\experiments\08_vqgan_fcc_phase1_compatibility`
- measurement audit baselines：`E:\GAN_FCC_WORK\experiments\02_baselines_measurement_audit_phases09_17`

## Research-Line Navigation

研究线入口：

`E:\GAN_FCC_WORK\research_lines`

它适合考古：想知道某一段 phase 为什么存在，读这里的 README 和 FILES。不要在这里直接改源代码；真实文件仍在 active repo 或 source-only copy 中。

## Data Warehouse

大型输出与复制 payload：

`E:\GAN_FCC_WORK\data_warehouse`

关键子目录：

- `E:\GAN_FCC_WORK\data_warehouse\ns_mc_gan_gi`：主线大型实验输出。
- `E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703`：从 frozen FCC phase1 中复制出的 GI/FCC/VQGAN 相关全量 payload。

## Current Paper Files

当前 IEEE TCI 主稿：

- `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.tex`
- `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.pdf`
- `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\materials_inventory.md`

写作规则：所有数值必须来自 `materials_inventory.md` 或明确标 `[DATA MISSING]`。

## VQGAN/FCC Copy

使用复制版：

`E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703`

重要入口：

- `CLAUDE_CODE_HANDOFF.md`
- `locked_bundle\PROJECT_BRIEF_VQGAN_DETAIL_FUSION_LOCKED.md`
- `outputs\compatibility\measurement_conditioned_vqgan`
- `outputs\compatibility\measurement_conditioned_vqgan\VQGAN_MULTI_SEED_PARETO_CONFIRMATION_PACKAGE.zip`
- `outputs\compatibility\multiseed_pareto_confirmation`
- `outputs\compatibility\fcc_diagnostic_canary64`
- `outputs\compatibility\structure_detail_fcc`

不要直接改 `E:\ns_mc_gan_gi_code_fcc_phase1`。

## Backups

备份根：

`E:\GAN_FCC_BACKUP`

active repo bare backup remote：

`E:\GAN_FCC_BACKUP\repos\ns_mc_gan_gi.git`

如果要做大规模清理，先确认备份和 manifest。不要把临时清理当作科研证据。

