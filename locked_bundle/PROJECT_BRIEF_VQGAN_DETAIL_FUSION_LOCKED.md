# 项目说明：为什么下一步是 Detail Fusion Locked Test

## 1. 当前结果是什么

三seed development已经确认，完整VQGAN注入带来巨大感知增益但失真较大；而全局标量融合 \(B\approx0.5\) 取得了更好的折中：

- 获得约76%的full-VQGAN LPIPS增益；
- 只付出约25%的PSNR代价；
- PSNR下降约0.43dB，仍在0.5dB容忍内；
- RMSE与RAPSD也满足balanced gate；
- 三个seed方向一致；
- 不需要学习gate或复杂频带权重。

这说明目前最强方案不是更复杂网络，而是简单、可解释的null-space GAN detail blending。

## 2. 方法为何自然

VQAE与VQGAN来自同构matched priors：

- VQAE保真度更好；
- VQGAN感知质量更强。

二者共享同一个LMMSE anchor \(x_0\)。把输出写成：

\[
x_A=x_0+d_A,\qquad x_G=x_0+d_G,
\]

其中残差都被投到最终算子null space。融合为：

\[
x_B=x_0+d_A+B(d_G-d_A).
\]

当 \(B=0\) 是VQAE；当 \(B=1\) 是VQGAN。中间值是显式的perception–distortion旋钮。

因为最终残差在 \(P_0\) 中：

\[
Ax_B=Ax_0=y.
\]

所以这种融合不破坏bucket measurement。

## 3. 为什么现在不训练gate

Development中复杂16频带/低通切点没有跑赢全局标量，说明数据当前不支持更复杂的选择机制。学习gate会增加过拟合和审稿风险。既然全局B已经通过balanced gate，就应锁定这个简单系统做confirmatory test。

## 4. Locked test必须验证什么

Locked test不再问“能不能找到更好B”，而只问：

> validation选出的固定B，在从未使用过的新样本上，是否仍显著优于VQAE，并满足失真容忍？

因此locked中禁止：

- 重新选B；
- 调整PSNR/RMSE容忍；
- 训练gate；
- 改变checkpoints；
- 选择展示样本后改结论；
- 使用已消费final-v4、Phase2 locked或旧GAN locked样本。

## 5. 主要比较

### VQAE

结构保真基线。

### Full VQGAN

感知上限和失真代价参照。

### Balanced Fusion

主方法。

### Quality-lite Fusion

次要高质量模式。

### LMMSE Anchor

物理/线性基线。

### Teacher-code upper bound

如果现有artifacts允许，可作为上限，不作为主比较。

## 6. 统计

对于每个图像和seed：

\[
\Delta_{i,s}=m_{\text{fusion}}(i,s)-m_{\text{VQAE}}(i,s).
\]

主统计建议：

\[
\bar\Delta_i=\frac1S\sum_s\Delta_{i,s}
\]

然后对图像bootstrap。这样不会把同一图像在不同seed下的重复评价当作独立样本。也要报告每seed均值和方向。

## 7. 可能结果

### Balanced locked confirmed

最理想：GAN细节在几乎无失真代价下真实泛化。这就是项目最自然主线。

### Quality-lite only

说明可获得强感知质量，但balanced约束过严。仍可作为高质量模式，但主张要诚实。

### Tradeoff only

VQGAN有用但必须接受较大PSNR/RMSE代价。

### Dev not replicated

保留development结果，不进入论文主结论；不要事后改B或locked split。

## 8. 论文表述

若locked通过，主故事可写为：

> A measurement-consistent VQGAN detail-fusion reconstructor: LMMSE/VQAE supplies structure, adversarial VQGAN supplies natural detail, and a fixed null-space blend preserves bucket consistency while improving perceptual quality.

允许说：

- GAN细节改善LPIPS/KID/RAPSD；
- exact null-space fusion保持measurement consistency；
- balanced模式控制了PSNR/RMSE代价。

不允许说：

- 原bucket测量认证了所有GAN纹理；
- B是每图oracle选择；
- fixed-total或witness相关旧结论被证明。
