# New-Agent Handoff: GAN + Ghost Imaging + FCC/VQGAN

这个目录是 `E:\GAN_FCC_WORK` 的唯一接手入口。目标不是再建一个索引堆，而是让新的会话/agent 按同一条研究思路继续工作：先理解问题，再定位代码，再核验证据，再决定是否跑实验。

编码说明：这些文档是 UTF-8。若在 Windows PowerShell 里查看中文，请用 `Get-Content -Encoding UTF8`。

## 一句话项目定位

这个项目研究欠采样鬼成像中“测量决定的内容”和“先验补出来的内容”如何分离、认证和限界。核心数学语言是 range-null 分解；核心工程工具是可插拔 measurement audit；GAN/VQGAN/FCC 是生成式先验的实验载体，不是论文卖点本身。

## 第一阅读顺序

1. `01_RESEARCH_STORY.md`：从早期 cGAN/GI 到当前 Gauge-GAN/Rad-5 和 VQGAN/FCC 支线的完整研究路线。
2. `02_THEORY_CORE.md`：全文统一的 range-null 数学主线、audit certificate、质量/问责 separability、边界。
3. `03_CODE_MAP.md`：哪些代码是主线，哪些是历史，哪些是支线，哪些只当证据库。
4. `05_EXPERIMENTS_AND_EVIDENCE.md`：每类实验对应哪些文件夹、报告、manifest 和论文证据。
5. `04_REPRODUCIBILITY_GUIDE.md`：如何安全复现，不误烧 Colab/本地 GPU，不误用 split。
6. `06_PAPERS_AND_CLAIMS.md`：两篇 paper 与 VQGAN/FCC 写作状态，哪些 claim 可以说、哪些不能说。
7. `07_RED_LINES_AND_WORKING_RULES.md`：不可碰红线。
8. `09_PAPERS_INDEX.md`：所有已发现论文/稿件的准确位置，包括 VQGAN Detail Fusion 草稿。
9. `10_FILE_BY_FILE_CURATION.md`：逐文件整理清单、剩余复核集和清理候选。
10. `08_NEXT_AGENT_CHECKLIST.md`：新会话真正开始做事前的 checklist。

## 权威工作位置

- 当前工作根：`E:\GAN_FCC_WORK`
- 当前 active repo：`E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code`
- 当前主论文：`E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.tex`
- 主论文 PDF：`E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.pdf`
- 主论文数字来源：`E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\materials_inventory.md`
- VQGAN Detail Fusion 草稿复制版：`E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.pdf`
- 大型输出/证据仓库：`E:\GAN_FCC_WORK\data_warehouse\ns_mc_gan_gi`
- 实验导航层：`E:\GAN_FCC_WORK\experiments`
- 研究线导航层：`E:\GAN_FCC_WORK\research_lines`
- 备份根：`E:\GAN_FCC_BACKUP`

## 当前代码状态

- active repo 分支：`pub-colab-runner`
- 最近记录的 HEAD：`688f48e`
- 备份 remote：`E:\GAN_FCC_BACKUP\repos\ns_mc_gan_gi.git`
- 注意：active repo 目前有大量未提交/未跟踪文件，包含当前论文、脚本、结果和整理痕迹。新 agent 不要先清理；要先读 `07_RED_LINES_AND_WORKING_RULES.md`。

## Frozen Exception

`E:\ns_mc_gan_gi_code_fcc_phase1` 是 frozen 原始混合目录。可以只读扫描，可以复制，但不要移动、删除、改写或清理它。

已经从 frozen 目录复制出 GI/FCC/VQGAN 相关 payload 到：

`E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703`

使用复制版，不要直接在 frozen 原件上工作。
