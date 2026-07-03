# Reproducibility Guide

这个指南优先保证“不会误跑、不会误改、不会误信”。很多历史实验成本高，不应该新 agent 一上来就重跑。

## 0. 进入 active repo

```powershell
cd E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code
git status --short --branch
git rev-parse --short HEAD
git remote -v
```

预期背景：

- branch: `pub-colab-runner`
- recorded HEAD: `688f48e`
- backup remote: `E:\GAN_FCC_BACKUP\repos\ns_mc_gan_gi.git`
- worktree dirty 是已知状态，不要把它当作错误自动清理。

## 1. 环境

Python requirements 在：

`E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\requirements.txt`

当前记录的核心依赖：

```text
torch
torchvision
numpy<2
tqdm
matplotlib
scikit-image
PyYAML
tensorboard
```

不要为了看文档或查证据创建新环境。只有要跑测试或训练时再配置。

## 2. 低成本 sanity checks

在 active repo 中：

```powershell
python -m pytest tests -q
```

若环境缺依赖，先报告缺失，不要为了一个 smoke check 大规模改环境。

优先关注这些测试：

- `tests\test_split_guard.py`
- `tests\test_run_protocol.py`
- `tests\test_train_checkpoint_wiring.py`
- `tests\test_p0_builder.py`
- `tests\test_exact_a_cache.py`
- `tests\test_gates.py`

## 3. 当前论文复现顺序

1. 读材料清点：

   `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\materials_inventory.md`

2. 打开当前主稿：

   `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.tex`

3. 打开当前 PDF：

   `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.pdf`

4. 若需要编译，先检查本机 TeX 工具；不要为了编译论文改实验文件。

论文写作硬规则：没有出现在 `materials_inventory.md` 的数字，不能写成真实结果；应标 `[DATA MISSING]`。

## 4. 当前 Gauge-GAN/Rad-5 线

实验导航：

`E:\GAN_FCC_WORK\experiments\06_gauge_gan_rad5_current_phases69_83`

相关 active scripts 包括：

- `scripts\phase69A_gauge_gan_signal_diagnostic.ps1`
- `scripts\phase69B_controlled_gauge_cgan_pilot.ps1`
- `scripts\phase70_gauge_gan_paper_expansion.ps1`
- `scripts\phase71_gauge_cgan_paired_seeds.ps1`
- `scripts\phase72_scr10_gauge_cgan_regime_validation.ps1`
- `scripts\phase73_overnight_gauge_gan_expansion.ps1`
- `scripts\phase74_high_tier_gauge_cgan_pack.ps1`
- `scripts\phase75_final_high_tier_validation.ps1`
- `scripts\phase76_high_upside_auditable_gan_exploration.ps1`
- `scripts\phase77_auditable_gan_paper_assembly.ps1`
- `scripts\phase83_run_96px_rad5_seeds_4_10.ps1`

不要直接重跑 `phase83` 或高成本脚本，除非当前任务明确要求。重跑前必须确认：

- same init；
- same split；
- same optimizer；
- same data order；
- same budget；
- checkpoint selection rule；
- 唯一改动是什么。

## 5. Posterior / Anti-Collapse 诊断线

这条线曾注册过严格判据：

- `mean pixel std > 0.01`
- `P0 variance > 1e-4`
- `P0/PR variance ratio > 5`
- 每样本 `RelMeasErr < 1e-2`
- 额外报告 `P0` 径向功率谱，防止白噪声假阳性。

不要事后改判据。不要报告没有 per-sample 输出支撑的采样结论。

## 6. VQGAN/FCC 线

使用复制版：

`E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703`

先读：

```text
CLAUDE_CODE_HANDOFF.md
locked_bundle\PROJECT_BRIEF_VQGAN_DETAIL_FUSION_LOCKED.md
```

再看 evidence outputs：

```text
outputs\compatibility\measurement_conditioned_vqgan
outputs\compatibility\multiseed_pareto_confirmation
outputs\compatibility\fcc_diagnostic_canary64
outputs\compatibility\structure_detail_fcc
```

这条线来自 frozen mixed source，不要把它和当前 active repo 混写。若要迁移进 active repo，先做 source review，再新建分支/manifest。

## 7. Colab / GPU

历史环境信息：

- WSL path 曾用：`/mnt/e/ns_mc_gan_gi_code`
- Colab CLI 曾用：`/var/tmp/codex-colab-tools/colab-cli-venv/bin/colab`
- pro accounts: `pro1`, `pro2`
- 本地 GPU: NVIDIA 4060

这些是算力入口，不是默认复现步骤。新 agent 只有在任务明确要求训练时才连接 Colab。

## 8. 保存证据的最低标准

每次新实验结束，至少保存：

- config；
- split hash；
- code commit/hash 或 dirty diff 摘要；
- checkpoint hash；
- per-sample outputs；
- metrics JSON/CSV；
- 评估脚本版本；
- README 说明该实验回答的问题和是否成功。

没有这些，不要把结果写进论文主 claim。

