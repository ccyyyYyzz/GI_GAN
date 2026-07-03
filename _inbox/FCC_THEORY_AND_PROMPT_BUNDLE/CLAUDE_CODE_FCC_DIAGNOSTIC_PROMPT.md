# Claude Code 目标模式：重新诊断 Feasible Counterfactual Compatibility

你是研究工程负责人。阅读仓库后，按 `FCC_THEORY_RANGE_NULL_COMPATIBILITY.md` 的理论重新实现 FCC。目标不是立刻提高重建质量，而是判断 row-space 骨架 \(r\) 与 null-space 内容 \(n\) 之间是否存在可学习、超过简单统计量的 compatibility signal。

## 核心定义

低采样 GI：

\[
y=Ax,\quad P_R=A^\dagger A,\quad P_0=I-A^\dagger A.
\]

对每张图：

\[
r=P_Rx,\quad n=P_0x,\quad x=r+n.
\]

真实配对：

\[
(r_i,n_i).
\]

可行反事实：

\[
u_{ij}=r_i+n_j,\quad j\neq i.
\]

因为 \(An_j=0\)，所以：

\[
Au_{ij}=Ax_i.
\]

FCC 学习 score：

\[
s(r,n)\approx\log p(n\mid r)-\log p(n).
\]

它不是普通自然度判别器，而是 row-null compatibility critic。

## 任务 A：exact projection 与数据构造

实现 matrix-free exact row/null projection：

\[
P_Rv=A^\top(AA^\top)^{-1}Av,\quad P_0v=v-P_Rv.
\]

禁止构造 dense \(P_0\)。float64 测试：

- \(AP_0v\approx0\)；
- \(P_Rv+P_0v\approx v\)；
- \(r_i+n_i\approx x_i\)；
- \(A(r_i+n_j)\approx Ax_i\)。

projection、训练输入、feasibility check 前禁止 clipping。

## 任务 B：dual-encoder FCC

实现：

\[
z_r=f_R(r),\quad z_n=f_N(n),
\quad s(r,n)=\frac{\tilde z_r^\top \tilde z_n}{\tau}.
\]

使用 symmetric InfoNCE。batch 内正配对为对角线。支持 random derangement、semi-hard、interpolated negatives。

输出：

- Recall@1/5 among 32；
- MRR；
- median rank；
- paired margin；
- score distribution；
- per-image CSV。

## 任务 C：nuisance controls

实现 deployable scalar baselines：

- energy；
- mean/std/RMS；
- min/max；
- range violation；
- TV；
- gradient RMS；
- low/mid/high frequency；
- spectral centroid；
- \(r+n\) 统计量。

实现：

- row-only；
- null-only；
- scalar logistic/GBDT/MLP；
- label shuffle；
- duplicate audit；
- nuisance-balanced derangement。

若 scalar baseline 或 artifact features 解释了 FCC，必须判为 artifact/scalar signal。

## 任务 D：generated candidate transfer

使用冻结候选生成器，为每个 \(y_i\) 生成 \(K=16/32\) 候选。默认共享：

\[
r_y=A^\dagger y.
\]

候选 null：

\[
n_k=P_0\hat x_k.
\]

比较：

- random candidate expectation；
- posterior mean；
- scalar selector；
- FCC selector；
- soft-FCC weighted mean；
- oracle best-of-K。

主指标：

- P0 RMSE；
- full RMSE；
- LPIPS；
- oracle-gain fraction；
- score-error Spearman；
- RelMeasErr。

如果 oracle headroom 不存在，不要责怪 FCC。若 FCC 不能超过 posterior mean，不要硬说重建成功。

## 任务 E：结论分类

输出一个机械分类：

1. `STRUCTURAL_COMPATIBILITY_CONFIRMED`
2. `REAL_PAIR_SIGNAL_BUT_NO_GENERATED_TRANSFER`
3. `ONLY_SCALAR_OR_ARTIFACT_SIGNAL`
4. `NO_COMPATIBILITY_SIGNAL`
5. `INVALID_EXPERIMENT`

## 纪律

- 不使用 final-v4、VQGAN locked、Phase2 locked 等已消费集合；
- train/val/dev hash-clean；
- 所有 donor index 可追踪；
- normalization statistics 只来自 train；
- 保存 A hash、sample hash、config、checkpoint、per-image CSV；
- 跑 pytest；
- 生成 claim-evidence ledger；
- 不因结果不好而改阈值；
- 不把 FCC score 称为 measurement-certified truth。

先做 64×64 小 canary：2048 train / 512 val / 512 dev，固定一个 operator，batch 128，embedding 128，最多 20 epochs。若层 A/B 不过，不进入 generated transfer。
