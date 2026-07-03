# Round1 `_manuscript_view/main.pdf` 身份审计

审计对象：`file:///E:/ns_mc_gan_gi_code_fcc_phase1/_manuscript_view/main.pdf`

结论先行：这个文件不是 `ns_mc_gan_gi_code_fcc_phase1` 项目的 ghost-imaging 稿件，而是上一项目 ZIFB 论文的主 PDF。继续逐句、逐图审这个 file URL 会把项目带偏。

## 1. P0 发现：目标 PDF 是 ZIFB 错稿

目标文件：

- `E:/ns_mc_gan_gi_code_fcc_phase1/_manuscript_view/main.pdf`
- SHA256：与 `E:/zifb_final_9129_luck/manuscript/main.pdf` 完全相同
- 大小：4,166,975 bytes
- 页数：19
- 首页标题：`Retained iodine is not blocking iodine: a dimensionless theory of dissolution-limited passivation in the zinc-iodine flow battery positive electrode`

这与当前项目名 `ns_mc_gan_gi_code_fcc_phase1`、当前项目源码和 ghost-imaging 论文主题完全不匹配。

同一 preview 目录里的 `SI.pdf` 也不是本项目文件：

- `E:/ns_mc_gan_gi_code_fcc_phase1/_manuscript_view/SI.pdf`
- SHA256：与 `E:/zifb_final_9129_luck/manuscript/SI.pdf` 完全相同

因此，`_manuscript_view` 目录当前整体是从 ZIFB 项目复制或渲染过来的错误预览。

## 2. 证据文件

机器可读对照表：

- `E:/ns_mc_gan_gi_code_fcc_phase1/manuscript_audit/ROUND1_MANUSCRIPT_VIEW_IDENTITY_AUDIT.csv`

该表记录了以下文件的 hash、页数、PDF metadata 和首页抽取文本：

- `_manuscript_view/main.pdf`
- `_manuscript_view/SI.pdf`
- `E:/zifb_final_9129_luck/manuscript/main.pdf`
- `E:/zifb_final_9129_luck/manuscript/SI.pdf`
- `E:/ns_mc_gan_gi_code_fcc_phase1/main.pdf`
- `E:/ns_mc_gan_gi_code_fcc_phase1/paper/main.pdf`

## 3. 当前项目里的真实候选稿件

本项目目录下至少有两个可能被误认为“主稿”的 PDF。

### A. 根目录 `main.pdf`

- 路径：`E:/ns_mc_gan_gi_code_fcc_phase1/main.pdf`
- 对应源码：`E:/ns_mc_gan_gi_code_fcc_phase1/main.tex`
- 主题：中文“菲涅耳波带片的目标光场编码设计”
- 页数：25
- 说明：它是另一个中文物理实验/方案稿，不匹配 `ns_mc_gan_gi_code_fcc_phase1` 这个 ghost-imaging 项目名。

### B. `paper/main.pdf`

- 路径：`E:/ns_mc_gan_gi_code_fcc_phase1/paper/main.pdf`
- 对应源码：`E:/ns_mc_gan_gi_code_fcc_phase1/paper/main.tex`
- 主题：`Measurement Auditing for Learned Ghost Imaging: Certificates, Limits, and Prior-Supplied Content`
- 页数：12
- 说明：这份最像当前项目真正要审的 learned ghost imaging / measurement auditing 论文。

## 4. 对 `paper/main.pdf` 的初步风险提示

如果下一步改审 `paper/main.pdf`，已经能看到几个高优先级问题：

- `paper/main.tex` 中仍有多处 `[CITE]` 占位引用。
- Related Work 多个段落仍以 `[CITE]` 结束，不能投稿。
- `paper/main.pdf` 编译时间为 2026-06-15；项目根目录和多个脚本在 2026-06-24 至 2026-06-29 仍有大量更新，需确认这份 PDF 是否是最新论文。
- `paper/main.tex` 与 `_manuscript_view/main.pdf` 完全不是同一稿件；任何基于 `_manuscript_view` 的视觉页图都无效。

## 5. 最小修复动作

1. 不要继续把 `file:///E:/ns_mc_gan_gi_code_fcc_phase1/_manuscript_view/main.pdf` 当作本项目稿件审。
2. 先删除或重建 `_manuscript_view`，用正确的 PDF 重新生成页面 PNG。
3. 若当前项目目标是 ghost-imaging 论文，优先审：
   - `E:/ns_mc_gan_gi_code_fcc_phase1/paper/main.pdf`
   - `E:/ns_mc_gan_gi_code_fcc_phase1/paper/main.tex`
4. 若当前目标是中文菲涅耳波带片方案，则审：
   - `E:/ns_mc_gan_gi_code_fcc_phase1/main.pdf`
   - `E:/ns_mc_gan_gi_code_fcc_phase1/main.tex`
5. 重新生成 `_manuscript_view` 后，再做逐句、逐图、逐数据点审计。

## 6. 当前可执行判断

当前 file URL 的审计结论不是“正文还有哪些小问题”，而是“预览目标文件本身错误”。这是 submission-blocking / workflow-blocking 级别问题。
