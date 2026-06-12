from __future__ import annotations

from .phase17_common import (
    DO_NOT_CLAIM,
    PHASE17,
    TITLE,
    main_result_rows,
    markdown_table,
    write_text,
)


OUT = PHASE17 / "chinese_report"


def main() -> None:
    rows = main_result_rows()
    limitations = "\n".join(f"- 不主张：{item}" for item in DO_NOT_CLAIM)
    text = f"""# 中文论文式报告草稿

英文题目建议：**{TITLE}**

## 1. 研究背景

鬼成像和单像素成像用一组照明图案和桶探测值恢复图像。它们的优势是可以在非传统探测条件下成像，但在低采样率下，测量数量远小于图像自由度，重建问题高度欠定。此时，仅依赖神经网络直接生成图像容易得到视觉上合理、但不满足测量数据的结果。

## 2. 鬼成像和单像素成像前向模型

本文使用线性桶测量模型：

$$
y = A x + \\epsilon.
$$

其中，$x$ 是向量化图像，$A$ 是测量矩阵，$y$ 是桶探测测量值，$\\epsilon$ 是噪声。物理含义是：每一个桶测量都是图像和一个照明图案的内积。

## 3. 低采样重建难点

当采样率为 5% 或 10% 时，测量维度远小于图像维度。单纯反投影会丢失大量高频细节；单纯神经网络又可能脱离测量值。本文的核心问题是：如何让神经网络补充缺失结构，同时仍然尊重真实桶测量。

## 4. 方法：测量一致零空间神经重建

先计算数据一致的初始解：

$$
x_{{data}} = A^T(AA^T + \\lambda I)^{{-1}}y.
$$

再把网络输出限制到测量零空间方向：

$$
P_N(v) = v - A^T(AA^T + \\lambda I)^{{-1}} A v,
$$

$$
\\tilde{{x}} = x_{{data}} + P_N(G_\\theta(x_{{data}}, z)).
$$

最后再做测量一致投影：

$$
\\hat{{x}} = \\Pi_y(\\tilde{{x}}) =
\\tilde{{x}} - A^T(AA^T + \\lambda I)^{{-1}}(A\\tilde{{x}} - y).
$$

这个结构的物理意义是：已由测量决定的部分由 $x_{{data}}$ 保持，网络主要补充欠定零空间中的结构，最终结果再被拉回测量一致集合。

## 5. 测量模式

本文区分三类测量模式。Rademacher 是随机符号测量，反投影很弱，但学习模型可以显著补偿；Scrambled Hadamard 有更强的物理初始化；Lowfreq Hadamard 用于简单域 sanity 和 DC row 控制，但不能把 STL-10 的 lowfreq Hadamard 5% 写成 high-quality。

Rademacher 结果必须只引用 exact A cache-rebuilt 路径，因为 Phase15R/Phase16 已经修复并复现了旧本地复评中的缓存问题。

## 6. 网络结构与损失

本文最终写作时应把网络表述为“测量一致框架中的 HQ reconstructor”，而不是把 GAN 作为最终主贡献。训练损失可以服务于重建质量和测量一致性，但论文主线应是物理约束、零空间修正和可验证归因。

## 7. 实验协议

Phase17 不进行训练，不新增实验。所有数字来自 Phase15 strict no-leak registry 和 Phase16 supplementary tables。Colab 导入模型应写成 imported strict no-leak checkpoints，不能写成本地训练结果。pre-fix Rademacher mismatch 不进入主文。

## 8. 主结果

{markdown_table(rows, ["method", "dataset", "sampling", "family", "psnr", "ssim", "bp_psnr", "delta_psnr"])}

这些结果支持：STL-10 5% 和 10% 在 Rademacher 与 scrambled Hadamard 下达到项目定义的 high-quality 阈值；MNIST/Fashion-MNIST 5% 作为简单域 sanity 也达到 high-quality。这里不主张 strict SOTA。

## 9. 消融实验

Phase16 的 inference ablation 说明：去掉 DC/data-consistency 投影会显著降低指标；stage1-only 低于完整 refiner；EMA 权重相对 raw weights 有小幅增益。Attribution 表说明 learned refinement 对弱反投影有明显贡献。

## 10. 鲁棒性与传统 baseline

Noise sweep 只支持有限噪声水平下的 robustness diagnostic，不支持任意对抗鲁棒性。Measurement perturbation 表明模型依赖桶测量，而不是完全生成式幻觉。TV-PGD 是 small-subset lightweight baseline，不能写成 exhaustive optimized baseline。

## 11. 局限性

{limitations}

此外，如果没有硬件实验，必须明确当前结果主要是仿真协议下的证据；class-wise 结果只能作为诊断，不应过度解释。

## 12. 总结

当前证据足以进入论文人工精修阶段。建议停止大规模新增实验，转向：引用核验、图表排版、语言压缩、补充材料组织，以及针对审稿风险的问题式回应。
"""
    write_text(OUT / "chinese_report_draft.md", text)
    print({"output": str(OUT / "chinese_report_draft.md")})


if __name__ == "__main__":
    main()
