# Papers And Claims

这里把写作状态和 claim 边界分开，防止新会话把工程结果写成过度论文结论。

## Paper A: 当前 IEEE TCI 主稿

位置：

```text
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.tex
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.pdf
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\materials_inventory.md
```

暂定题名：

```text
Measured and Unmeasured: Range-Null Separability, Certification,
and the Limits of Verifying Prior-Supplied Content in Ghost Imaging
```

允许主 claim：

- 欠采样成像中测量决定内容与先验补充内容可由 range-null 分解正交分离。
- 对任意重建器可附加 test-time measurement audit。
- audit 有 contraction certificate。
- image quality 与 measurement accountability 可分离。
- measurement consistency 不等于 image correctness。
- GAN 是生成式先验范例，在 audit 下展示可控感知改善与边界。

禁止或谨慎 claim：

- 不 claim 图像质量 SOTA。
- 不 claim 击败 diffusion。
- 不 claim audit 能验证零空间语义正确性。
- 不用 test split 训练或调参。
- 不写任何 `materials_inventory.md` 里没有的数字；缺失写 `[DATA MISSING]`。

## Paper B: paper1 / publication evidence branch

位置：

```text
E:\GAN_FCC_WORK\project_sources\paper1_publication_evidence_sourceonly\paper1_analysis\paper1_draft_PR.tex
E:\GAN_FCC_WORK\project_sources\paper1_publication_evidence_sourceonly\paper1_analysis\paper1_draft_PR.pdf
```

角色：

- 旧论文/证据分支；
- phase67 zero-training / classical GI / Morozov/noise-floor 等检查的参考；
- 不等同于当前 TCI 主稿。

## Paper C: VQGAN Detail Fusion 完整草稿

这条线不是只有 evidence bundle；`fcc_phase1` 里有完整 paper draft。

题名：

```text
Measurement-Consistent VQGAN Detail Fusion for Low-Rate Ghost Imaging
```

frozen 原件：

```text
E:\ns_mc_gan_gi_code_fcc_phase1\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.tex
E:\ns_mc_gan_gi_code_fcc_phase1\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.pdf
E:\ns_mc_gan_gi_code_fcc_phase1\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.md
```

工作区复制版：

```text
E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.tex
E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.pdf
E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.md
```

可先读的配套说明：

```text
CLAUDE_CODE_HANDOFF.md
locked_bundle\PROJECT_BRIEF_VQGAN_DETAIL_FUSION_LOCKED.md
outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\CLAIM_EVIDENCE_LEDGER.md
outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\REPRODUCIBILITY_MANIFEST.json
```

这篇稿件是独立的 VQGAN null-space detail fusion paper；不能再只当作“证据支线”描述。后续编辑应使用工作区复制版或迁移到 active repo 后编辑，frozen 原件保持只读。

更多位置和哈希见：

```text
E:\GAN_FCC_WORK\handoff\09_PAPERS_INDEX.md
```

## 当前写作总原则

论文卖的是几何事实、certificate 和 certificate 的边界。GAN/VQGAN/FCC 只是展示不同 prior 如何填充 `P_0` 的实验载体。

最稳妥的措辞是：

- “measurement-accountable reconstruction”
- “auditable prior-supplied content”
- “range-null separability”
- “measurement consistency does not certify correctness”
- “GAN as a generative example”

最危险的措辞是：

- “solves ghost imaging”
- “guarantees correct reconstruction”
- “posterior sampling solved” without per-sample evidence
- “SOTA” without a registered benchmark and fair comparison
- “diffusion beaten” without direct evidence
