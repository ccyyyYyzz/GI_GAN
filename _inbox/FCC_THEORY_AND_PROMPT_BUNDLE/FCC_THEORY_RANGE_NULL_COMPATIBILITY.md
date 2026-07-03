# FCC 理论说明：Feasible Counterfactual Compatibility

## 0. 一句话

FCC 把 ghost imaging 的 range-null 分解变成一个自监督对比学习问题：

> 真实图像提供正确配对 \((r,n)\sim p(r,n)\)，可行反事实提供测量等价但配对错误的 \((r,n')\sim p(r)p(n)\)。  
> FCC critic 学习的是 row-space 骨架 \(r\) 与 null-space 内容 \(n\) 的统计相容性，而不是普通图像自然度。

---

## 1. 物理与线性代数背景

低采样 ghost imaging / single-pixel imaging 写成：

\[
y=Ax,\qquad A\in\mathbb R^{m\times n},\quad m\ll n.
\]

定义：

\[
P_R=A^\dagger A,\qquad P_0=I-A^\dagger A.
\]

任意图像可唯一分解为：

\[
x=P_Rx+P_0x.
\]

记：

\[
r=P_Rx,\qquad n=P_0x.
\]

其中：

- \(r\)：row-space，测量可见部分；
- \(n\)：null-space，测量不可见部分。

因为：

\[
AP_0=0,
\]

所以：

\[
A(r+n)=Ar.
\]

也就是说，bucket measurement 只能约束 \(r\)，不能直接认证 \(n\)。

---

## 2. Feasible counterfactual 构造

对训练图像 \(x_i\)：

\[
x_i=r_i+n_i.
\]

真实配对：

\[
(r_i,n_i).
\]

从另一张图像 \(x_j\) 取 null 内容 \(n_j\)，构造：

\[
u_{ij}=r_i+n_j.
\]

由于：

\[
An_j=0,
\]

所以：

\[
Au_{ij}=A(r_i+n_j)=Ar_i=Ax_i.
\]

因此 \(u_{ij}\) 与 \(x_i\) 有相同 bucket measurement，但 null-space 内容来自另一张图。它是：

\[
\boxed{\text{measurement-feasible but paired-wrong}}
\]

这就是 FCC 的反事实训练信号。

---

## 3. FCC 要学习的函数

定义打分器：

\[
s_\phi(r,n).
\]

希望：

\[
s_\phi(r_i,n_i)>s_\phi(r_i,n_j).
\]

若正样本来自联合分布：

\[
(r,n)\sim p(r,n),
\]

错配来自边缘乘积：

\[
(r,n)\sim p(r)p(n),
\]

则最优判别器满足：

\[
D^*(r,n)=\frac{p(r,n)}{p(r,n)+p(r)p(n)}.
\]

其 logit 为：

\[
\log\frac{D^*}{1-D^*}
=
\log\frac{p(r,n)}{p(r)p(n)}
=
\log p(n\mid r)-\log p(n).
\]

因此 FCC 理想上估计：

\[
\boxed{s(r,n)\approx \operatorname{PMI}(r,n)}
\]

即 pointwise mutual information。

直觉：

> \(n\) 单独看可能自然；\(r+n\) 整体也可能自然；但 FCC 问的是：这个 \(n\) 是否特别适合这个 \(r\)。

所以 FCC 不是普通 GAN discriminator。

普通 GAN 问：

\[
D(x): x\text{ 是否自然？}
\]

FCC 问：

\[
D(r,n): n\text{ 是否是 }r\text{ 的真实搭档？}
\]

---

## 4. 信息论上限

FCC 的上限取决于：

\[
I(R;N).
\]

若：

\[
p(n\mid r)\approx p(n),
\]

则：

\[
I(R;N)\approx0.
\]

此时任何 FCC 都不可能强。FCC 不能创造测量中不存在的信息，它只能利用自然图像分布中 \(r\) 与 \(n\) 的统计相关性。

因此第一个研究问题应是：

\[
\boxed{\text{\(R\) 与 \(N\) 之间到底有没有可学习互信息？}}
\]

而不是一开始就要求提高最终重建指标。

---

## 5. 推荐模型：dual-encoder retrieval

输入：

\[
r\in\mathbb R^{H\times W},\qquad n\in\mathbb R^{H\times W}.
\]

编码：

\[
z_r=f_R(r),\qquad z_n=f_N(n).
\]

L2 normalize：

\[
\tilde z_r=\frac{z_r}{\|z_r\|},\qquad
\tilde z_n=\frac{z_n}{\|z_n\|}.
\]

打分：

\[
s(r,n)=\frac{\tilde z_r^\top \tilde z_n}{\tau}.
\]

一个 batch 中有 \(B\) 个真实配对：

\[
(r_1,n_1),\ldots,(r_B,n_B).
\]

score matrix：

\[
S_{ij}=s(r_i,n_j).
\]

InfoNCE：

\[
\mathcal L_{r\to n}
=
-\frac1B
\sum_i
\log
\frac{\exp(S_{ii})}{\sum_j\exp(S_{ij})}.
\]

对称项：

\[
\mathcal L_{n\to r}
=
-\frac1B
\sum_j
\log
\frac{\exp(S_{jj})}{\sum_i\exp(S_{ij})}.
\]

总损失：

\[
\mathcal L_{\rm FCC}
=
\mathcal L_{r\to n}+\mathcal L_{n\to r}.
\]

核心评价：

- Recall@1 among 32；
- Recall@5；
- MRR；
- median rank；
- paired margin；
- score 与 true \(P_0\) error 的 Spearman。

---

## 6. 负样本设计

### 6.1 Random derangement

\[
(r_i,n_{\pi(i)}),\qquad \pi(i)\neq i.
\]

要求 \(\pi\) 是 derangement，并记录 donor index。

### 6.2 Semi-hard negative

选择 \(r_j\) 与 \(r_i\) 接近但非同一图的 donor：

\[
j\in\operatorname{NN}_k(r_i).
\]

用于避免 random negatives 太容易。

### 6.3 Interpolated feasible counterfactual

\[
n_{ij}(\alpha)=n_i+\alpha(n_j-n_i),\qquad \alpha\in[0,1].
\]

因为 null-space 是线性空间：

\[
n_{ij}(\alpha)\in\ker A.
\]

所以：

\[
u_{ij}(\alpha)=r_i+n_{ij}(\alpha)
\]

始终满足：

\[
Au_{ij}(\alpha)=Ax_i.
\]

可训练排序：

\[
s(r_i,n_i)>s(r_i,n_{ij}(\alpha)).
\]

---

## 7. 为什么必须做 nuisance-balanced control

Raw counterfactual \(r_i+n_j\) 可能带来明显伪影：

- 像素越界；
- TV 异常；
- gradient RMS 异常；
- 低频/高频能量不匹配；
- range violation；
- mean/std mismatch；
- 频谱异常。

如果 FCC 只学会这些，它不是 compatibility，而是 artifact detector。

需要构造 nuisance features：

\[
\phi(r,n)
\]

包括：

- \(\|r\|_2,\|n\|_2\)；
- mean/std/RMS；
- min/max；
- range violation；
- TV；
- gradient RMS；
- low/mid/high frequency energy；
- spectral centroid；
- \(r+n\) 的 range/TV/frequency。

可做 assignment：

\[
\min_{\pi}
\sum_i
\|\phi(r_i,n_{\pi(i)})-\phi(r_i,n_i)\|^2,\qquad \pi(i)\neq i.
\]

得到 nuisance-balanced derangement。若 scalar baseline 已能解释 FCC，说明没有学到结构相容性。

---

## 8. 三层实验逻辑

### 层 A：真实配对检索

只用真实 \((r_i,n_i)\)。问：

> 给定 \(r_i\)，能否在 32 个 \(n\) 中找回 \(n_i\)？

随机 Recall@1：

\[
1/32\approx3.125\%.
\]

若 FCC 远高于随机且高于 scalar baseline，说明 \(I(R;N)\) 有可用信号。

### 层 B：nuisance-balanced counterfactual

问：

> 排除能量、range、TV、频谱等 shortcut 后，FCC 是否还能区分真实配对与错配？

若失败，旧 FCC 可能只是 artifact classifier。

### 层 C：生成候选迁移

对同一测量 \(y_i\) 生成候选：

\[
\hat x_{i1},\ldots,\hat x_{iK}.
\]

使用共享 anchor：

\[
r_i=A^\dagger y_i.
\]

候选 null：

\[
\hat n_{ik}=P_0\hat x_{ik}.
\]

打分：

\[
s_{ik}=s(r_i,\hat n_{ik}).
\]

选择：

\[
k^*=\arg\max_k s_{ik}.
\]

评价：

\[
\|P_0(\hat x_{ik^*}-x_i)\|.
\]

对比：

- random candidate expectation；
- posterior mean；
- scalar selector；
- FCC selector；
- oracle best-of-\(K\)。

---

## 9. FCC 的概率候选融合

不必硬选单候选。可定义：

\[
\pi_k=
\frac{\exp(\beta s(r,n_k))}
{\sum_j\exp(\beta s(r,n_j))}.
\]

输出：

\[
\hat x=r+\sum_k\pi_k n_k.
\]

这比 hard selection 更适合 RMSE。若未来有 witness：

\[
\pi_k\propto
\exp(\beta s(r,n_k))
\exp\left[
-\frac{\|W(r+n_k)-y_w\|^2}{2\sigma^2}
\right].
\]

但纯 FCC 诊断应先不加 witness。

---

## 10. 必须避免的错误

- 不要用 clipped counterfactual 训练 feasibility；
- 不要构造 dense \(P_0\)；
- 不要把 \(P_R\hat x_k\) 当每候选 anchor，默认使用同一个 \(r_y=A^\dagger y\)；
- 不要只报告 random-negative AUC；
- 不要跳过 scalar baseline；
- 不要用 test truth 调 threshold；
- 不要把 high FCC score 写成 measurement-certified truth；
- 不要把 FCC 一开始包装成最终重建器。

---

## 11. 结论分类

### `STRUCTURAL_COMPATIBILITY_CONFIRMED`

真实配对检索强、nuisance-balanced 仍强、超过 scalar baseline、能迁移到生成候选。

### `REAL_PAIR_SIGNAL_BUT_NO_GENERATED_TRANSFER`

真实 \((r,n)\) 有互信息，但无法排序生成候选。

### `ONLY_SCALAR_OR_ARTIFACT_SIGNAL`

random negatives 有效，但 nuisance-balanced 或 scalar control 消除收益。

### `NO_COMPATIBILITY_SIGNAL`

真实配对检索接近随机。

### `INVALID_EXPERIMENT`

projection、split、donor、clipping、baseline 或 oracle headroom 不合规。

---

## 12. 最浓缩理论描述

FCC treats range-null decomposition as a contrastive density-ratio problem. Real images provide matched pairs \((r,n)\sim p(r,n)\); feasible counterfactuals provide mismatched but measurement-equivalent pairs \((r,n')\sim p(r)p(n)\). The critic estimates

\[
\log\frac{p(r,n)}{p(r)p(n)}
=
\log p(n\mid r)-\log p(n),
\]

which measures how compatible a null-space completion is with the measured row-space skeleton. Its possible value is limited by \(I(R;N)\), and must be validated against nuisance-balanced negatives, scalar shortcuts, and generated-candidate transfer.
