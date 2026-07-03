## 2. The Range–Null Geometry

Everything that follows — the converse of Section 3, the certificate of Section 4, and the governed dial of Section 6 — is a consequence of a single decomposition of the measurement operator. We state it once here, in operator-agnostic form, and reuse it throughout.

### 2.1 Forward model and compact SVD

We consider the standard linear ghost-imaging model
$$
y = A x + \varepsilon,
$$
where $x \in \mathbb{R}^n$ is the vectorized image, $y \in \mathbb{R}^m$ is the bucket-measurement vector, $A \in \mathbb{R}^{m \times n}$ is the sensing operator with $m \ll n$, and $\varepsilon$ is measurement noise. Write the compact singular value decomposition of $A$ as
$$
A = U_r \Sigma_r V_r^\top, \qquad r = \operatorname{rank}(A),
$$
where $U_r \in \mathbb{R}^{m \times r}$ and $V_r \in \mathbb{R}^{n \times r}$ have orthonormal columns ($U_r^\top U_r = V_r^\top V_r = I_r$) and $\Sigma_r = \operatorname{diag}(\sigma_1, \dots, \sigma_r)$ with every $\sigma_i > 0$. The Moore–Penrose pseudo-inverse is then $A^\dagger = V_r \Sigma_r^{-1} U_r^\top$.

### 2.2 The two projectors

Define the row-space projector $P_R = A^\dagger A$ and the null-space projector $P_0 = I - A^\dagger A$. Substituting the SVD collapses $P_R$ to a form that involves only the right singular vectors,
$$
P_R = A^\dagger A = V_r \Sigma_r^{-1} U_r^\top U_r \Sigma_r V_r^\top = V_r V_r^\top,
\qquad
P_0 = I - V_r V_r^\top.
$$
Because $V_r^\top V_r = I_r$, each is symmetric and idempotent, and the two are complementary and mutually annihilating:
$$
P_R^\top = P_R, \quad P_R^2 = P_R, \qquad
P_0^\top = P_0, \quad P_0^2 = P_0, \qquad
P_R + P_0 = I, \quad P_R P_0 = 0 .
$$
They are therefore the pair of orthogonal projectors onto the two orthogonal subspaces
$$
\mathbb{R}^n = \mathcal{R}(A^\top) \oplus \mathcal{N}(A), \qquad \mathcal{R}(A^\top) \perp \mathcal{N}(A),
$$
the row space (range of $A^\top$) and the null space (kernel of $A$). Every image splits uniquely and orthogonally,
$$
x = P_R x + P_0 x, \qquad \langle P_R x, \, P_0 x \rangle = x^\top P_R P_0\, x = 0,
$$
so the measured and unmeasured parts share no energy.

### 2.3 The Lemma: measurements constrain only the row space

The single fact that arms every later section is that $A$ annihilates its own null-space projector.

**Lemma (measurement geometry).** For any $A \in \mathbb{R}^{m \times n}$,
$$
A P_0 = 0 \qquad\text{and}\qquad A P_R = A .
$$

*Proof.* Using the Moore–Penrose identity $A A^\dagger A = A$,
$$
A P_0 = A\,(I - A^\dagger A) = A - A A^\dagger A = A - A = 0,
$$
and correspondingly $A P_R = A(A^\dagger A) = A A^\dagger A = A$. $\quad\blacksquare$

The consequence is sharp. Decomposing any candidate image and applying $A$,
$$
A x = A P_R x + A P_0 x = A P_R x ,
$$
so the record $y$ is a function of $P_R x$ alone; the null-space component $P_0 x$ is exactly invisible to the bucket. Two images that agree in the row space but differ arbitrarily in the null space produce identical measurements.

In the noiseless case $A x = y$ this pins the measured component exactly: $P_R x = A^\dagger A x = A^\dagger y$. Every feasible image with the same $y$ shares this row-space component, while any $z_0 \in \mathcal{N}(A)$ can be added freely, since $A(A^\dagger y + z_0) = A A^\dagger y + A z_0 = y$ whenever $y \in \mathcal{R}(A)$. The noiseless feasible set is therefore the null-space-translated affine flat
$$
\{\, x : A x = y \,\} = A^\dagger y + \mathcal{N}(A) .
$$
With noise, $A^\dagger y$ is the least-squares row-space estimate induced by the record, and the null component remains unobserved.

### 2.4 What can and cannot be certified

The Lemma partitions the $n$ image coordinates into two disjoint accountabilities, one settled by the data and one settled only by a prior:

- **Certifiable (the $m$ row coordinates).** The component $P_R x = A^\dagger y$ is fixed by the record: any measurement-consistent reconstruction must have $P_R \hat{x} = A^\dagger y$, and re-measuring it reproduces $y$. These $m$ coordinates are pinned down and can be audited without ground truth (Section 4).
- **Not certifiable (the $n - m$ null coordinates).** The component $P_0 x$ is unconstrained by $y$. The measurement supplies no evidence about what lives there; any content placed in the null space is a modeling choice, not a verified fact. This is the barrier of Section 5 and, read constructively, the safe channel for the governed injection of Section 6.

Concretely, in the locked setting of this paper — $64 \times 64$ grayscale images, $n = 4096$ pixels, at a $5.0\%$ sampling rate with $m = 205$ measurements — the bucket certifies exactly the $205$ row-space coordinates and leaves the remaining $3891$ null-space coordinates to a prior. Most of what the eye reads as detail lives in those $3891$ directions.

### 2.5 The rows-orthonormal fusion case

The fusion construction (Section 6) uses an operator whose rows are orthonormalized, giving $A A^\top = I_m$. This does not alter the geometry above; it only simplifies the algebra. When $A A^\top = I_m$ the singular values are all unity, so the pseudo-inverse reduces to the transpose,
$$
A^\dagger = A^\top, \qquad P_R = A^\top A, \qquad P_0 = I - A^\top A,
$$
and the Lemma reads directly as $A P_0 = A(I - A^\top A) = A - (A A^\top) A = A - A = 0$. This is the identity the exact-consistency guarantee $A \hat{x}_B = y$ rests on: any update routed through $P_0$ is invisible to $A$, so it can carry injected detail without perturbing the record. The general SVD form of Section 2.2 is what the test-time audit of Section 4 uses, where the per-mode singular values $\sigma_i$ reappear explicitly in the contraction factor $\lambda / (\lambda + \sigma_i^2)$; the two settings are the same geometry viewed at two levels of operator conditioning.
