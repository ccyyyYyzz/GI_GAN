# Research Story

这条研究线不是一开始就长成现在的 range-null/audit 论文。它经历了从“让鬼成像重建更好看”到“哪些内容由测量负责、哪些内容只是先验补全”的转向。下面按科学问题，而不是按文件夹，复原研究路线。

## 0. 早期 PCCN-GI / cGAN 前身

早期项目在 `E:\GAN_FCC_WORK\project_sources\pc_cgan_gi_precursor_sourceonly`。它是一个条件重建框架：随机 speckle / 合成灰度数据、CGI baseline、conditional U-Net/PCCN-GI、L1 与 measurement-domain physics consistency，GAN/SSIM 作为辅助或 ablation。

这一阶段的核心问题还是“能不能把图像重建得更清楚”。它是历史前身，不是当前论文的主证据。

## 1. NS-MC-GAN：从重建质量转向测量一致性

当前 active repo 是：

`E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code`

这一阶段建立了当前平台：前向模型 `y = A x + epsilon`，测量矩阵/illumination pattern，训练与评估脚本，split guard，projection/audit，图像质量指标，以及 GAN/重建器相关模块。

关键思想开始成形：欠采样不是普通去噪或超分问题。测量只看见 `range(A^T)` 里的分量，看不见 `null(A)` 里的分量。因此重建器输出里有两类东西：测量约束的行空间内容，以及先验填充的零空间内容。

## 2. Phases 3-8：operator/pattern learning 探索

这一段在 `experiments\01_operator_pattern_learning_phases03_08` 和对应 `research_lines` 中。问题是：提升来自学习 illumination/operator，还是来自 generator/reconstruction prior？

包含二值/连续 pattern、operator calibration、flip-aware/physical pattern controls 等。它提供了平台和早期对照，但不是当前论文主线的最强论据。

## 3. Phases 9-17：baselines 与 audit certificate 锁定

这一段在 `experiments\02_baselines_measurement_audit_phases09_17`。它把核心证据从“模型表现”转向“measurement accountability”。

核心结论：对 BP、Tikhonov、CS-TV、learned reconstructor 等不同重建器，都可以在 test time 后接同一个 measurement audit，让 `RelMeasErr` 大幅下降，而 PSNR 通常基本不变。这个结果让论文有了“可插拔、与重建器无关”的 certificate 叙事。

## 4. Phases 18-45：机制图、论文资产和可复现组织

这一段在 `experiments\03_manuscript_mechanism_phases18_45`。它主要是把前面的实验整理成论文图、表、LaTeX、mechanism figures、conventional GI anchors、provenance decomposition 等。

这一段的重要性在写作和证据组织，不在于提出新模型。

## 5. Phases 48-60 + Phase67：range-null 边界与 feasible wrong images

这一段在 `experiments\04_range_null_counterfactual_barrier_phases48_60`，并和 paper1 branch `experiments\05_paper1_publication_evidence_phase67` 有关。

这里完成了研究路线的第二次转向：measurement consistency 是必要但不充分的。因为任意 `z in null(A)` 都满足 `A(x+z)=Ax`，所以可以构造测量完全一致但视觉/语义错误的 feasible wrong images。

这不是一个负结果，而是论文的核心边界：audit 能认证“是否符合测量”，不能认证“先验补出来的零空间内容是否正确”。

## 6. Phases 69-83：Gauge-GAN / Rad-5 当前生成式 case study

当前主线在：

`E:\GAN_FCC_WORK\experiments\06_gauge_gan_rad5_current_phases69_83`

对应 active repo 中 `src\phase69...phase81...` 和 `scripts\phase69...phase83...`。

这条线把 GAN 放回论文，但角色已经变了：GAN 不是 SOTA 质量卖点，而是一个“被 audit 约束的生成式先验范例”。重点是 Gauge-GAN/Rad-5 在 certificate 下能否改善感知指标、保持 measurement accountability，并暴露 audit 的边界。

当前需要谨慎表达：可以说 certificate、LPIPS/RAPSD、range-null boundary；不要夸张成击败 diffusion 或质量 SOTA。

## 7. Posterior sampling / anti-collapse 支线

曾经尝试在固定 `y` 下采样多个 `z`，希望得到多个 measurement-consistent posterior samples。已有诊断显示旧 checkpoint 基本忽略 `z`，pixel std 接近 collapse。后续推断认为 reconstruction loss 可能压平了零空间多样性，因此需要 row-space-only recon loss 加 diversity/对抗驱动来验证。

这条线是高风险探索线，不是当前论文已成立主结论。任何 claim 必须有 per-sample 输出、P0/PR 方差和测量误差支持。

## 8. FCC / VQGAN 兼容性分支

frozen 原始目录：

`E:\ns_mc_gan_gi_code_fcc_phase1`

已复制出的 GI/FCC/VQGAN 相关全量 payload：

`E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703`

这一支包含 measurement-conditioned VQGAN/VQAE、anchor-initialized VQGAN inversion、multi-seed Pareto confirmation、FCC row-null/structure-detail compatibility diagnostics、Bayesian witness 等。它不是没有文章，只是没有整合进当前 IEEE TCI 主稿；它更像一个可独立发展或作为 supplement/后续 paper 的兼容性证据线。

## 当前总论点

欠采样鬼成像里的可验证内容和不可验证内容可以用同一个 range-null 几何分开：`P_R x` 是测量可问责部分，`P_0 x` 是先验补充部分。Audit 可以把任何重建器拉回 measurement consistency，但它不能证明零空间内容正确。GAN/VQGAN/FCC 的价值在于展示不同先验如何填充零空间，而不是替代这个几何事实。

