# Theory Core: Range-Null Geometry

这篇工作的统一数学语言是 range-null 分解。所有结论都应该从这一件事推出，而不是分别讲成松散技巧。

## Forward Model

设

```text
y = A x + epsilon,    A in R^{m x n},    m << n.
```

对 `A` 做紧 SVD：

```text
A = U_r Sigma_r V_r^T,
```

其中 `r = rank(A)`，`U_r` 和 `V_r` 的列正交，`Sigma_r` 对角且奇异值为正。Moore-Penrose 伪逆为

```text
A^\dagger = V_r Sigma_r^{-1} U_r^T.
```

定义行空间投影和零空间投影：

```text
P_R = A^\dagger A = V_r V_r^T,
P_0 = I - A^\dagger A = I - P_R.
```

因为 `P_R` 是正交投影，所以

```text
P_R^2 = P_R,    P_R^T = P_R.
```

又因为 `P_0 = I - P_R`，

```text
P_0^2 = P_0,    P_0^T = P_0,    P_R P_0 = 0.
```

因此任意图像都有唯一正交分解：

```text
x = P_R x + P_0 x,    <P_R x, P_0 x> = 0.
```

白话：图像可以被唯一拆成“测量看得见的部分”和“测量看不见的部分”。

## D1. Measurement Only Constrains Row-Space Content

由定义，

```text
A P_0 = A(I - A^\dagger A)
      = A - A A^\dagger A
      = A - A
      = 0.
```

同时

```text
A P_R = A A^\dagger A = A.
```

所以

```text
A x = A(P_R x + P_0 x)
    = A P_R x + A P_0 x
    = A P_R x.
```

在无噪声情形 `y = A x` 下，

```text
P_R x = A^\dagger A x = A^\dagger y.
```

因此给定 `y` 后，`P_R x` 被 `A^\dagger y` 唯一确定；而 `P_0 x` 不进入测量。

白话：桶测量只负责行空间；零空间可以改变而不改变任何测量值，这就是欠定性的本质。

## D2. Audit Certificate As Singular-Mode Contraction

定义 test-time audit：

```text
Pi_y^lambda(v) = v - B_lambda (A v - y),
B_lambda = A^T (A A^T + lambda I)^{-1}.
```

令残差

```text
r = A v - y.
```

则 audit 后的测量残差为

```text
A Pi_y^lambda(v) - y
= A(v - B_lambda r) - y
= A v - A B_lambda r - y
= r - A A^T (A A^T + lambda I)^{-1} r.
```

利用恒等式

```text
I - M(M + lambda I)^{-1} = lambda (M + lambda I)^{-1},
```

取 `M = A A^T`，得到

```text
A Pi_y^lambda(v) - y
= lambda (A A^T + lambda I)^{-1} r.
```

再用 SVD：

```text
A A^T = U_r Sigma_r^2 U_r^T.
```

每个被测奇异模的残差被乘以

```text
lambda / (sigma_i^2 + lambda).
```

若 `lambda = 0` 且 `A A^T` 在测量子空间可逆，残差可被精确清零；若 `lambda > 0`，audit 是 soft contraction，会保留与噪声 floor 相容的残差。

白话：audit 不是玄学后处理，而是对每个测量模都有可写下来的收缩因子。

## D3. Image Quality And Measurement Accountability Are Separable

令重建误差

```text
e = x_hat - x.
```

由于 `P_R + P_0 = I`，

```text
e = P_R e + P_0 e.
```

由于 `P_R P_0 = 0` 且二者是正交投影，

```text
||e||_2^2 = ||P_R e||_2^2 + ||P_0 e||_2^2.
```

定义行空间误差能量占比

```text
s = ||P_R e||_2^2 / ||e||_2^2.
```

假设一次理想 audit 只移除了行空间误差，零空间误差保持不变，则新误差能量为

```text
||e'||_2^2 = ||P_0 e||_2^2
           = ||e||_2^2 - ||P_R e||_2^2
           = (1 - s) ||e||_2^2.
```

PSNR 定义为

```text
PSNR = 10 log10(MAX^2 / MSE).
```

因此最大 PSNR 增益为

```text
Delta PSNR_max
= 10 log10(MSE / MSE')
= 10 log10(1 / (1 - s))
= -10 log10(1 - s).
```

白话：如果一个 trained network 的误差主要在零空间，audit 能显著改善测量问责，却买不到多少 PSNR；如果 BP 的误差有大量行空间成分，audit 同时能改善 PSNR。

## Boundary: Consistency Is Not Correctness

若 `z in null(A)`，则

```text
A(x + z) = A x + A z = A x.
```

所以 `x` 与 `x+z` 有完全相同的测量。只要 `z` 在视觉上足够改变图像，就能得到 feasible wrong image：测量一致，但内容错误。

白话：certificate 能回答“它是否符合测量”，不能回答“看不见的内容是否真的是原图”。

## Posterior-Sampling Implication

固定 `y` 时，一个健康的 measurement-consistent posterior sampler 应该保持 `P_R x_hat` 基本固定，同时在 `P_0 x_hat` 中产生有意义多样性。

因此评估不能只看总 pixel std。至少要同时看：

- `mean pixel std` 是否足够大；
- `P_0 variance` 的绝对值是否有意义；
- `P_0 variance / P_R variance` 是否足够大；
- 每个 sample 的 `RelMeasErr` 是否仍受控；
- `P_0` 内容谱是否像自然图像，而不是白噪声。

白话：真正的 posterior diversity 应该住在测量看不见的方向里，但不能只是往零空间塞噪声。

