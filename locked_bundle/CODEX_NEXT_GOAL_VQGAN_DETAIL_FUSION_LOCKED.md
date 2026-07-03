# Codex 目标模式：VQGAN Detail Fusion 的全新 locked 验证

你是本项目研究负责人。先审阅仓库、detail-fusion development输出和 `PROJECT_BRIEF_VQGAN_DETAIL_FUSION_LOCKED.md`，再自主完成从审计、预注册、freeze 到一次性 locked scoring 的完整链路。不要训练新模型，不要学习gate，不要再调B；当前development最强信号来自零训练全局标量融合，系统越简单越好。

## 当前判断

Development 三seed已得到理想信号：

- `fusion_balanced` 相对 VQAE：LPIPS约改善32.9%，CI不跨0；
- PSNR约下降0.43dB，在0.5dB容忍内；
- RMSE约增加0.0039，在容忍内；
- RAPSD改善；
- 3/3 seeds同向；
- RelMeasErr约1e-7量级；
- 全局标量B≈0.5优于复杂频带/门控。

这是 development 证据，不是确认性结果。下一步只做全新 locked test。

## 核心方法

已有同一anchor \(x_0\)、VQAE输出 \(x_A\)、VQGAN输出 \(x_G\)。定义：

\[
d_A=P_0(x_A-x_0),\qquad d_G=P_0(x_G-x_0).
\]

全局融合：

\[
\hat x_B=x_0+P_0\{d_A+B(d_G-d_A)\}.
\]

因 \(AP_0=0\)，所有 \(B\) 都保持measurement consistency。Locked中必须使用validation已冻结的精确B值/规则；不得根据dev或locked结果重新选择。

## 必做审计

1. 读取 development artifacts，确认B是由validation选择，dev只评分；
2. 精确记录每seed的balanced和quality-lite B，不使用“≈0.5”近似；
3. 核验VQAE/VQGAN端点与先前multi-seed确认一致；
4. 核验融合只做null-space线性运算，没有重训练、没有真值oracle；
5. 审计所有已消费集合：final-v4、Phase2 locked、旧GAN locked、VQGAN dev/val、prior训练选择集等；
6. raw/transformed SHA256去重。

若发现B由dev调过，分类为 `INVALID_DEV_SELECTION_LEAKAGE`，不得locked。

## Locked协议

只有审计通过，创建全新hash-clean locked split。排除所有历史已消费样本与exact duplicates。冻结：

- VQAE/VQGAN three-seed prior/refiner checkpoints；
- anchor/LMMSE/operator artifacts；
- balanced与quality-lite B选择规则；
- primary/secondary metrics；
- hierarchical statistics；
- sample manifest；
- one-shot runner；
- source和artifact hashes。

执行一次且仅一次locked scoring。

## Primary与Secondary

Primary：Balanced mode LPIPS vs VQAE。

Balanced成功条件：

- LPIPS相对VQAE改善≥5%；
- paired/hierarchical bootstrap CI upper<0；
- PSNR drop≤0.5dB；
- RMSE increase≤0.005；
- RAPSD不劣于VQAE；
- 至少2/3 seeds同向；
- RelMeasErr达标；
- no leakage / no duplicate / hash audit PASS。

Secondary：Quality-lite mode，full VQGAN，VQAE，LMMSE anchor，teacher upper bound（若可得）。报告PSNR、SSIM、full/centered RMSE、LPIPS、KID、RAPSD、projection norm、RelMeasErr、频带误差和固定定性图。

统计不能把 image×seed 简单当独立样本。推荐先对每image聚合seed delta，再bootstrap images；同时报告每seed效应。

## 机械结论

- `LOCKED_BALANCED_VQGAN_FUSION_CONFIRMED`
- `LOCKED_QUALITY_LITE_ONLY`
- `LOCKED_VQGAN_TRADEOFF_ONLY`
- `LOCKED_DEV_NOT_REPLICATED`
- `INVALID_LOCKED_EXPERIMENT`

若locked通过，准备论文主线：LMMSE/VQAE保结构，VQGAN提供GAN细节，全局null-space融合得到高质量且测量一致重建。若locked失败，不改B、不重跑，回到development诊断。

你可自由实现runner、hash审计、图表、报告和打包，但不得改变冻结方法。最终交付：审计、预注册、locked结果、Pareto图、定性图、claim–evidence ledger、可复现包和唯一下一步。
