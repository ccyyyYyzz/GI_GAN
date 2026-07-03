# Papers Index

这个文件专门列论文/稿件位置。它修正了之前遗漏：`fcc_phase1` 里确实有一篇完整 VQGAN detail fusion paper draft。

## Paper A: 当前 IEEE TCI / Range-Null Audit 主稿

主题：

```text
Measurement Auditing for Learned Ghost Imaging:
Certificates, Limits, and Prior-Supplied Content
```

工作区主位置：

```text
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.tex
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\main.pdf
E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper\materials_inventory.md
```

`fcc_phase1` 中也有一份同哈希副本：

```text
E:\ns_mc_gan_gi_code_fcc_phase1\paper\main.tex
E:\ns_mc_gan_gi_code_fcc_phase1\paper\main.pdf
```

PDF SHA256:

```text
74A85D4A74E6DD7C7C6F0632E1793CAECF4A6FB23913408AED115FA3AC768548
```

处理规则：以 `E:\GAN_FCC_WORK\active_code\ns_mc_gan_gi_code\paper` 为工作稿；不要直接改 frozen/旧位置里的副本。

## Paper B: Paper1 / Publication Evidence 旧稿

位置：

```text
E:\GAN_FCC_WORK\project_sources\paper1_publication_evidence_sourceonly\paper1_analysis\paper1_draft_PR.tex
E:\GAN_FCC_WORK\project_sources\paper1_publication_evidence_sourceonly\paper1_analysis\paper1_draft_PR.pdf
```

PDF SHA256:

```text
E7D7406FD085ECC2DC344700C8DE6ADECCB0D8E31C5DC83B2EA74421EB367AA5
```

处理规则：这是旧 publication evidence branch，查证据和旧论证可以用；不要自动并入当前 TCI 主稿。

## Paper C: VQGAN Detail Fusion 完整草稿

主题：

```text
Measurement-Consistent VQGAN Detail Fusion for Low-Rate Ghost Imaging
```

frozen 原件位置：

```text
E:\ns_mc_gan_gi_code_fcc_phase1\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.tex
E:\ns_mc_gan_gi_code_fcc_phase1\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.pdf
E:\ns_mc_gan_gi_code_fcc_phase1\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.md
```

工作区复制版位置：

```text
E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.tex
E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.pdf
E:\GAN_FCC_WORK\data_warehouse\fcc_phase1_gi_related_fullcopy_20260703\outputs\compatibility\measurement_conditioned_vqgan\detail_fusion_paper\PAPER_DRAFT.md
```

PDF SHA256:

```text
2C3CB0AB772FB2A910461FB1292E9C98B42746133E0CBE10903F5F1AD1050BAC
```

Supporting assets in the same folder include:

```text
FACTS.json
MAIN_TABLE.csv
MAIN_TABLE.tex
PARETO_FIGURE.pdf/png
METHOD_DIAGRAM.pdf/png/svg
METHOD_DIAGRAM_3D.pdf/png/svg
QUALITATIVE_GRID.pdf/png
CORE_MECHANISM_FIGURE.pdf/png/svg
CLAIM_EVIDENCE_LEDGER.md
REPRODUCIBILITY_MANIFEST.json
REVIEWER_STRESS_TEST.md
LIMITATIONS_AND_NEGATIVE_RESULTS.md
FINAL_PROJECT_STATUS.md
VQGAN_FUSION_PAPER_PACKAGE.zip
VQGAN_FUSION_PAPER_PACKAGE_SHA256.txt
```

处理规则：这是一篇真实完整草稿，不应再只标成“支线证据”。后续编辑应在工作区复制版上另建工作稿或迁移到 active repo，原 frozen 目录只读。

## Non-GI / Mixed Manuscripts Found In frozen fcc_phase1

`E:\ns_mc_gan_gi_code_fcc_phase1\main.tex` 的标题是：

```text
菲涅耳波带片的目标光场编码设计：稳妥基准方案与目标光场编码创新方案
```

`_manuscript_view` 下的 `main.pdf` / `main_scichina.pdf` / `main_scts.pdf` / `SI.pdf` 与该菲涅耳波带片稿件有关，不是 GAN/GI/FCC 主线论文。它们解释了为什么 frozen 原目录是 mixed source，不能整体当作 GI 项目根。

## Scan Record

重新扫描清单：

```text
E:\GAN_FCC_WORK\inventory\fcc_phase1_paper_candidates_rescan_20260703.csv
```

