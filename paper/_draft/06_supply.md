## 6. Governed Supply: Injecting Detail Where the Measurement Is Blind

Sections 3–5 establish an asymmetry that is exact rather than practical. The measurement $y = Ax$ fixes the row-space component $P_R x$ and constrains nothing else, because $A P_0 = 0$; the audit of Section 3 certifies that a reconstruction reproduces $P_R x$, and the barrier of Section 5 shows that no test, however sharp, can decide the null-space component $P_0 x$ from $y$ alone. That barrier is usually read as a prohibition. Here we read the *same identity* the other way: since $A P_0 = 0$, any content placed in the null space is invisible to $A$, so editing $P_0 x$ cannot move $Ax$ and cannot break the measurement record. The geometry that forbids certifying invented detail simultaneously guarantees that supplying it is measurement-safe. This section makes that reading constructive.

### 6.1 An audited anchor and two matched priors

We anchor at a measurement-audited estimate $x_0$: an empirical LMMSE map from bucket data to image, projected so that $A x_0 = y$ holds to numerical precision. The anchor carries the low-frequency, data-determined structure of the scene and serves as the origin from which every null-space edit is measured; because $A x_0 = y$, the anchor already sits on the feasible set $\{x : Ax = y\}$, and all subsequent supply will move it *within* that set.

Against this anchor we run two matched vector-quantized priors per random seed. The first is a reconstruction-trained autoencoder (VQAE), whose refined output $x_A$ is faithful and stable but perceptually flat — it spends the null space on smooth, low-risk content. The second is an architecturally identical prior trained adversarially (VQGAN), whose refined output $x_G$ carries sharper edges and richer texture. The two branches share operator, anchor, and refinement protocol per seed, differing only in whether the prior was trained for reconstruction or for adversarial realism, so their difference isolates *detail* rather than any confound of architecture or conditioning.

### 6.2 Null-space fusion by a single scalar

We do not select one reconstruction over the other. We fuse their **null-space contributions** relative to the anchor. Define
$$d_A = P_0\,(x_A - x_0), \qquad d_G = P_0\,(x_G - x_0),$$
each a pure null-space vector ($P_0 d_A = d_A$, $P_0 d_G = d_G$). A single global scalar $B$ interpolates between them, and the projector is re-applied before the update is added to the anchor:
$$\hat{x}_B = x_0 + P_0\big(d_A + B\,(d_G - d_A)\big).$$
The scalar has a clean reading: $B = 0$ returns the VQAE structure reconstruction $x_0 + P_0 d_A$, $B = 1$ returns the VQGAN detail reconstruction $x_0 + P_0 d_G$, and intermediate $B$ trades structure fidelity for adversarial detail. Consistent with the program's non-claims, $B$ is a *fixed, validation-selected* operating point — pre-registered and frozen per seed, not adapted per image and not an oracle that knows the best weight for each scene.

### 6.3 The consistency theorem: exact, for every $B$

**Theorem.** For every $B \in \mathbb{R}$, the fused reconstruction satisfies $A\hat{x}_B = y$ exactly.

*Proof.* Apply $A$ and use linearity:
$$A\hat{x}_B = A x_0 + A\,P_0\big(d_A + B(d_G - d_A)\big).$$
The audited anchor gives $A x_0 = y$. The null-space identity $A P_0 = 0$ (Section 2) annihilates the second term regardless of the bracketed vector and regardless of $B$, so
$$A\hat{x}_B = y + 0 = y. \qquad \blacksquare$$

This is the constructive counterpart of the verifiability barrier. In Section 5 the identity $A P_0 = 0$ produced a feasible-but-wrong witness that satisfied the record to $\sim\!2\times10^{-15}$ — content that is provably unverifiable. Here the *same* identity guarantees that no matter how much adversarial detail the dial injects, the reconstruction never leaves the feasible set. The audit and the fusion are two consequences of one fact: what the bucket cannot certify, editing cannot corrupt.

### 6.4 The identity holds numerically

The theorem is exact in exact arithmetic; in floating point the residual is at round-off. Across the locked split (5.0% sampling, $64\times64$ grayscale STL10, $n = 4096$, $m = 205$, 3 seeds), the relative measurement error $\mathrm{RelMeasErr} = \lVert A\hat{x}_B - y\rVert / \lVert y \rVert$ has mean $3.6\times10^{-7}$ and max $5.7\times10^{-7}$ — the same round-off scale for every operating point on the dial, confirming that raising $B$ buys perceptual detail at no measurement cost. This is the achievability that the impossibility result licenses: a supply of null-space detail that is, by construction and in measurement, free of any consistency penalty.

We restate the scope deliberately, because the firewall of this program depends on it. $A\hat{x}_B = y$ certifies that the reconstruction *reproduces the buckets* — nothing more. The injected texture lives entirely in the null space, where the measurement furnishes no evidence; it is a prior-supplied hypothesis about plausible content, never a measurement-certified fact. The dial governs *how much* invented detail to supply; it never converts that detail into truth. Section 7 characterizes where along this dial the useful operating points lie, and Section 9 returns to why the accountable row space and the responsibly supplied null space must never be read in the same currency.
