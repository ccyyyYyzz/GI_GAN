# Certify What You Measure, Govern What You Cannot: A Range–Null Account of Accountability and Governed Detail in Undersampled Ghost Imaging

*Author placeholder — Author One, Author Two, Author Three. Affiliation placeholder. Corresponding author: [email placeholder].*

---

## Abstract

In undersampled ghost imaging the bucket record $y = A x$ from an operator $A \in \mathbb{R}^{m\times n}$ with $m \ll n$ fixes only the row-space component $P_R x = A^\dagger A\,x$ of the scene; the null-space component $P_0 x = (I - A^\dagger A)\,x$, which carries most of the perceptual detail, is annihilated by the operator since $A P_0 = 0$. This geometry forces a converse: for any measured image we construct an explicit feasible-but-wrong witness, a semantically different scene that reproduces the *same* record to $\sim\!2\times10^{-15}$ relative error — tighter than the noisy truth's own residual of $\sim\!10^{-3}$ — so consistency with the measurement is not correctness of null-space content, and that content is provably unverifiable. Against this barrier we give a ground-truth-free, test-time audit $\Pi_y^\lambda$ that separates image quality from measurement accountability: it leaves the null space untouched and contracts each measured singular mode by exactly $\lambda/(\lambda+\sigma_i^2)$ (verified in float64 to $\sim\!10^{-10}$), driving the relative measurement residual of learned and classical reconstructors down by three to four orders of magnitude at a perceptual cost of $|\Delta\text{PSNR}| \le 0.039$ dB. Licensed by this limit, we then supply detail only where it is invisible to the bucket: anchoring at a measurement-audited LMMSE estimate $x_0$ ($A x_0 = y$) and fusing the null-space contributions of a reconstruction prior (VQAE) and an adversarial prior (VQGAN) through a single scalar $B$, $\hat{x}_B = x_0 + P_0\big(d_A + B(d_G - d_A)\big)$, which satisfies $A\hat{x}_B = y$ exactly for every $B$ (relative measurement error mean $3.6\times10^{-7}$). On a raw-hash-disjoint locked split at 5% sampling, this pre-registered honesty dial improves LPIPS by $-0.0977$ (a 32.6% relative gain; CI $[-0.1016, -0.0940]$) over the VQAE branch at a bounded cost of $-0.45$ dB PSNR, with 3/3 seeds agreeing, an 8/8 acceptance gate passing, and dataset-level KID falling from $0.119$ to $0.043$. Throughout we hold one scope line: $A\hat{x} = y$ certifies that a reconstruction reproduces the measurements, never that its invented texture is the true scene.

---

## 1. Introduction

A learned reconstruction of an undersampled scene can reproduce every recorded measurement to numerical precision and still be semantically wrong. Consider two images that agree on the measured coordinates but differ arbitrarily elsewhere: they produce the *identical* measurement record, yet at most one of them is the scene that was imaged. In low-rate ghost imaging this is not a pathological corner case but the generic situation. A single-pixel detector acquires a small number of correlated bucket values, $y = A x$ with $A \in \mathbb{R}^{m \times n}$ and $m \ll n$; here $n = 4096$ ($64\times64$ pixels) and $m = 205$, a 5.0% sampling rate. The measurement therefore pins down $205$ numbers, while the image carries $4096$, and modern generative priors are perfectly capable of filling the remaining $3891$ degrees of freedom with plausible, sharp, natural-looking detail. The scientific question a reconstruction pipeline must answer is not merely whether the output looks good. It is: **which content is measured, and which is invented?**

Standard image-quality metrics cannot answer this. PSNR and SSIM require a ground-truth image, which is unavailable in deployment, and they conflate measured error with unmeasured error. A high-PSNR reconstruction can carry a large measurement residual, and a measurement-consistent reconstruction can be arbitrarily wrong in its unmeasured component. The gap is not a deficiency of any particular metric or algorithm; it is a property of the sensing operator.

### 1.1 The governing fact

The entire paper is a consequence of one elementary identity. Let $A^\dagger$ be the Moore–Penrose pseudoinverse and split $\mathbb{R}^n$ into the orthogonal row-space and null-space of $A$ through the projectors $P_R = A^\dagger A$ and $P_0 = I - A^\dagger A$, so that $x = P_R x + P_0 x$ uniquely. Then

$$A P_0 = A\big(I - A^\dagger A\big) = A - (A A^\dagger A) = A - A = 0.$$

Consequently $A x = A P_R x$: the measurement record is a function of $P_R x$ **only**, and $P_0 x$ is exactly invisible to the bucket. This single fact is elementary — it is textbook linear algebra of a rank-deficient operator, and we claim no novelty in the identity itself. The contribution is what follows from taking it seriously: the empirical sharpness of the barrier it imposes, and a complete rational response to that barrier.

### 1.2 The deductive chain

We organize the paper as an impossibility result and its disciplined consequences — a converse, an achievability, and a governed use — each a corollary of $A P_0 = 0$.

**Converse (the barrier, §3).** Because $y$ fixes exactly $P_R x$ and nothing else, null-space content is *provably* unverifiable from the measurement. We make this constructive rather than abstract: for any two images $x_i, x_j$ the spliced image $u_{ij} = x_j - A^\dagger(A x_j - y_i)$ carries the measured component of $x_i$ but the null-space content of $x_j$, and satisfies the record of $x_i$ to floating-point precision. Empirically these feasible-but-wrong witnesses reach $\mathrm{RelMeasErr} \sim 2\times10^{-15}$ (16/16 constructed pairs, range $2.16\times10^{-15}$ to $4.00\times10^{-15}$) — *tighter* than the noisy truth itself (residual $3.45\times10^{-3}$ to $7.46\times10^{-3}$), while remaining semantically wrong (PSNR-to-target $7.70$ to $11.38$ dB). Consistency is not correctness.

**Achievability (the certificate, §4).** The same geometry that forbids certifying $P_0 x$ *does* permit a ground-truth-free, test-time audit of the measured component. Attaching $\Pi_y^\lambda(v) = v - A^\top(A A^\top + \lambda I)^{-1}(A v - y)$ to any reconstructor contracts each measured singular mode by exactly $\lambda/(\lambda + \sigma_i^2)$ and leaves $P_0 v$ untouched. This audit reduces the measurement residual by three to four orders of magnitude for backprojection, Tikhonov, CS-TV, and learned outputs while moving PSNR by at most $0.039$ dB (18/18 audited rows), cleanly separating image quality from measurement accountability. It exactly matches the closed-form contraction in float64 arithmetic (deviation $\sim 10^{-10}$ to $10^{-12}$).

**Governed use (the dial, §6–§8).** Since no measurement can certify the null space, the honest response is not to hide prior-supplied content but to inject it precisely where it is provably invisible to the bucket, and to meter it. Anchoring at a measurement-audited LMMSE estimate $x_0$ (so $A x_0 = y$) and fusing the null-space differences of a reconstruction prior (VQAE, contribution $d_A$) and an adversarial prior (VQGAN, contribution $d_G$) through a single scalar $B$,

$$\hat{x}_B = x_0 + P_0\big(d_A + B(d_G - d_A)\big),$$

gives $A \hat{x}_B = y$ *exactly* for every $B$ (locked $\mathrm{RelMeasErr}$ mean $3.6\times10^{-7}$). At the balanced operating point ($B \approx 0.55$) this improves LPIPS by $-0.0977$ (CI $[-0.1016, -0.0940]$; a 32.6% relative gain) over the VQAE branch at a bounded cost of $-0.45$ dB PSNR, with 3/3 seeds agreeing and an 8/8 pre-registered acceptance gate passing on a raw-hash-disjoint locked test split.

### 1.3 Figure 1

Figure 1 renders this chain as three panels — *cannot / can / therefore*. The left panel (**cannot**) shows a feasible-but-wrong pair: a true target and its spliced counterpart, sharing the measured row space and differing only in the null space, whose bucket residual ($2.93\times10^{-15}$) is smaller than the truth's own noisy residual ($5.36\times10^{-3}$) — a measurement-consistent, semantically wrong, non-natural image. The middle panel (**can**) shows the exact per-mode certificate: the audit contracting each measured singular mode by $\lambda/(\lambda+\sigma_i^2)$, driving residuals to the noise floor while PSNR barely moves. The right panel (**therefore**) shows metered injection at $B = 0.55$, with the fused reconstruction stamped $A\hat{x} = y$ — invented detail placed only where the bucket is blind, without ever breaking the measurement.

### 1.4 Contributions

1. **A constructive converse.** We turn "consistency is not correctness" from an abstract statement into a per-instance, feasible-but-wrong witness that matches the recorded bucket to $\sim 2\times10^{-15}$ — tighter than the noisy truth — establishing empirically that null-space content is unverifiable.
2. **A ground-truth-free measurement certificate.** A plug-in test-time audit $\Pi_y^\lambda$ that contracts each measured mode by exactly $\lambda/(\lambda + \sigma_i^2)$, applies uniformly across BP, Tikhonov, CS-TV, and learned reconstructors, and separates image quality from measurement accountability (residuals drop orders of magnitude at $|\Delta\mathrm{PSNR}| \le 0.039$ dB).
3. **A governed, measurement-safe injection dial.** A single scalar $B$ fusing VQAE structure and VQGAN detail strictly inside $P_0$, keeping $A\hat{x}_B = y$ exact for every $B$, delivering a locked balanced LPIPS gain of $-0.0977$ (32.6%; KID $0.119 \to 0.043$) behind a pre-registered gate, and tracing a monotone perception–distortion ladder (LMMSE $0.404 \to$ VQAE $0.300 \to$ balanced $0.202 \to$ quality-lite $0.182 \to$ VQGAN $0.172$).
4. **A no-adaptation lemma.** We prove that any ground-truth-free selection rule is constant on each measurement fiber, so per-image adaptation of null-space content is a global prior in disguise — a direct corollary of the converse. A feature-based per-image selector confirms it, recovering the per-image oracle only through a global shift ($\le 0.002$ LPIPS beyond a matched-PSNR constant; feature–oracle rank correlations $|\rho|\le0.24$). The honest dial is a scalar by necessity, not by choice.

### 1.5 Non-claims

We state the boundaries of the work up front and honor them throughout. This is **not** a state-of-the-art result; we do **not** claim to beat diffusion-based solvers, and we report **no** hardware experiments. The fusion weight $B$ is a fixed, validation-selected operating point, **not** a per-image oracle. Posterior null-space $z$-diversity was pursued and is reported only as an abandoned negative result, never as a positive claim. The cross-rate (2%, 10%) and measurement-noise studies are **development-level** evidence, not locked claims. The observable magnitude $|P_0 \hat{x}|$ is **not** a pixelwise error locator. And most importantly: measurement consistency means $A\hat{x} = y$ and nothing more — the bucket **never** certifies that the injected texture is the true scene.

---

## 2. The Range–Null Geometry

Everything that follows — the converse of §3, the certificate of §4, and the governed dial of §6 — is a consequence of a single decomposition of the measurement operator. We state it once here, in operator-agnostic form, and reuse it throughout.

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

- **Certifiable (the $m$ row coordinates).** The component $P_R x = A^\dagger y$ is fixed by the record: any measurement-consistent reconstruction must have $P_R \hat{x} = A^\dagger y$, and re-measuring it reproduces $y$. These $m$ coordinates are pinned down and can be audited without ground truth (§4).
- **Not certifiable (the $n - m$ null coordinates).** The component $P_0 x$ is unconstrained by $y$. The measurement supplies no evidence about what lives there; any content placed in the null space is a modeling choice, not a verified fact. This is the barrier of §3 and, read constructively, the safe channel for the governed injection of §6.

Concretely, in the locked setting of this paper — $64 \times 64$ grayscale images, $n = 4096$ pixels, at a $5.0\%$ sampling rate with $m = 205$ measurements — the bucket certifies exactly the $205$ row-space coordinates and leaves the remaining $3891$ null-space coordinates to a prior. Most of what the eye reads as detail lives in those $3891$ directions.

### 2.5 The rows-orthonormal fusion case

The fusion construction (§6) uses an operator whose rows are orthonormalized, giving $A A^\top = I_m$. This does not alter the geometry above; it only simplifies the algebra. When $A A^\top = I_m$ the singular values are all unity, so the pseudo-inverse reduces to the transpose,
$$
A^\dagger = A^\top, \qquad P_R = A^\top A, \qquad P_0 = I - A^\top A,
$$
and the Lemma reads directly as $A P_0 = A(I - A^\top A) = A - (A A^\top) A = A - A = 0$. This is the identity the exact-consistency guarantee $A \hat{x}_B = y$ rests on: any update routed through $P_0$ is invisible to $A$, so it can carry injected detail without perturbing the record. The general SVD form of §2.2 is what the test-time audit of §4 uses, where the per-mode singular values $\sigma_i$ reappear explicitly in the contraction factor $\lambda / (\lambda + \sigma_i^2)$; the two settings are the same geometry viewed at two levels of operator conditioning.

---

## 3. The Verifiability Barrier (Converse)

The geometry of §2 is deductive but silent about severity: the lemma $A P_0 = 0$ shows that $y$ fixes exactly $P_R x$ and constrains nothing in the null space, but it does not by itself say how badly a null-space error can hide behind a clean measurement. This section supplies the converse. We construct, for any pair of images, an explicit reconstruction that reproduces a target's bucket record to machine precision while carrying an entirely different scene's unmeasured content. The construction is per-instance and exact, not asymptotic or distributional: it exhibits a concrete feasible-but-wrong witness for every cross-class pair we test. Its consequence is the load-bearing claim of the paper — measurement consistency is certifiable but the correctness of null-space content is not, so **consistency is not correctness** — and it is placed first, before any certificate or injection dial, because everything downstream is subordinate to it.

### 3.1 A per-instance feasible-but-wrong witness

Fix a target image $x_i$ with bucket record $y_i = A x_i$ and a *donor* image $x_j$ from a different semantic class. We want a single image that carries the target's measurement but the donor's unmeasured detail. Define

$$u_{ij} = x_j - A^\dagger\big(A x_j - y_i\big).$$

The witness is measurement-consistent with the target to first principles. Because $y_i \in \mathcal{R}(A)$ and $A x_j \in \mathcal{R}(A)$, their difference lies in $\mathcal{R}(A)$, and $A A^\dagger$ is the orthogonal projector onto $\mathcal{R}(A)$, so

$$A u_{ij} = A x_j - A A^\dagger\big(A x_j - y_i\big) = A x_j - (A x_j - y_i) = y_i.$$

Its two projections separate cleanly. The row-space component is pinned to the target,

$$P_R\, u_{ij} = A^\dagger A u_{ij} = A^\dagger y_i = P_R\, x_i,$$

while the null-space component is inherited *whole* from the donor,

$$P_0\, u_{ij} = P_0\, x_j - P_0\, A^\dagger\big(A x_j - y_i\big) = P_0\, x_j,$$

where the second term vanishes because $A^\dagger$ maps into the row space and $P_0$ annihilates it ($P_0 A^\dagger = 0$). In words, $u_{ij}$ splices the measured part of $x_i$ onto the unmeasured part of $x_j$ and passes the measurement audit of $x_i$ exactly. It is the constructive imaging analogue of observational equivalence in identification theory (Koopmans and Reiersøl 1950): two structures generate the identical record, so no amount of that record can tell them apart.

### 3.2 The record is matched more tightly than the truth

The witness is not merely feasible in principle; numerically it satisfies the record more tightly than the true scene does. On the Rad-5 operator ($m = 205$, $96\times 96$), across a set of $8$ cross-class feasible pairs, the relative measurement error of the witness against the target record lies in the range $\mathrm{RelMeasErr}(u_{ij}, y_i) \in [2.16\times10^{-15},\, 4.00\times10^{-15}]$, i.e. at floating-point round-off. Over the paired families the construction holds for $16/16$ cross-class pairs at the same $\sim\!2\times10^{-15}$ scale. The *true* image, by contrast, only satisfies the (noisy) record to $[3.45\times10^{-3},\, 7.46\times10^{-3}]$. The gap is roughly twelve orders of magnitude: **the wrong image reproduces the buckets more exactly than the ground truth.** A featured car-versus-horse pair makes the point concretely — the target car has $\mathrm{RelMeasErr}(x_i, y_i) = 5.36\times10^{-3}$, while the constructed witness follows the horse donor visually and yet attains $\mathrm{RelMeasErr}(u_{ij}, y_i) = 2.93\times10^{-15}$. The witnesses remain semantically wrong with respect to the target, with $\mathrm{PSNR}(u_{ij}, x_i)$ in the range $7.70$–$11.38$ dB. Crucially, the construction is not tied to that operator: on the *same* $64\times64$ STL10 operator at $5\%$ sampling ($m=205$) used for the governed dial of §7, 40 feasible-wrong witnesses reproduce their target records at $\mathrm{RelMeasErr}(u_{ij}, y_i)$ between $1.7\times10^{-14}$ and $5.6\times10^{-13}$ (median $9.9\times10^{-14}$) while remaining semantically wrong (median $\mathrm{PSNR}(u_{ij}, x_i) = 19.5$ dB). The converse of this section and the constructive dial of §7 are therefore exhibited on one and the same operator; the residual is a few orders of magnitude above the Rad-5 figure only because the mixed-basis operator is applied through a ridge-regularized ($\lambda_{\text{solver}}=10^{-6}$) audit rather than an exact orthonormal projector, and it remains eleven orders below the $\sim\!10^{-3}$ noise floor.

The immediate corollary is that no residual-based test, however tight its threshold, can separate the witness from the truth: any threshold that admits the true image (residual $\sim\!10^{-3}$) admits the witness (residual $\sim\!10^{-15}$) a fortiori. This is precisely the failure mode PULSE exhibits for super-resolution (Menon et al. 2020) — visually distinct realistic reconstructions collapsing to the same low-dimensional observation — but here it is exact, per-instance, and derived from the operator's own geometry rather than sampled from a generator.

### 3.3 Position relative to the limits literature

The existence of consistency-preserving hallucinations, and the impossibility of assessing them without ground truth, is established prior art. Iagaru et al. (2026) prove necessary-and-sufficient conditions for detail-transfer hallucination in (possibly nonlinear) inverse problems — consistent decoders can only hallucinate details almost invisible in measurement space — and give forward-model-only, ground-truth-free algorithms that *bound* hallucination magnitude via feasible-set diameters. Gottschling et al. (2025) situate the same phenomenon within the broader account of instabilities and fundamental limits of learned reconstruction, and Bhadra et al. (2021) formalize null-space hallucinations by SVD projection, showing they are attributable solely to the prior and cannot be assessed without the truth. We take these as the anchor for the barrier itself and do not reclaim their ground: that null content is unverifiable, and that assessment must be ground-truth-free, is theirs.

Our delta is constructive and per-instance. Where the limits line establishes *existence* through necessary-and-sufficient conditions, or supplies *bounds* over feasible sets (a supremum that requires paired data to evaluate), $u_{ij}$ is an explicit machine-precision witness for each individual instance — an object one can render, measure, and display, matching the record tighter than the noisy truth. It converts "consistency is not correctness" from a theorem about maps and an SVD diagnostic into a concrete pixelwise construction. This is the engine on which the rest of the paper turns: because the barrier is real and exact, the certificate of §4 can only ever certify the row space, and the detail injection of §6 is licensed to act only where the barrier proves the measurement is blind.

### 3.4 The unmeasured-content magnitude is not an error map

A natural but mistaken hope is that the observable magnitude $|P_0 \hat{x}|$ — how much unmeasured content a reconstruction carries, pixel by pixel — could serve as a self-contained error map, flagging where a reconstruction has invented detail. It cannot, and the same decomposition explains why. For any audited reconstruction $\hat{x}$ the null-space error is

$$P_0(\hat{x} - x) = P_0 \hat{x} - P_0 x,$$

and the observable term $P_0 \hat{x}$ contains no direct information about the unknown truth term $P_0 x$. The magnitude is a property of the reconstruction, not a certificate of its error: a large value may be accurate texture, inaccurate texture, or a harmless choice among feasible completions. We therefore call $|P_0 \hat{x}|$ a **prior-supplied-content map** — it marks where a prior placed unmeasured content, not where the reconstruction is wrong.

The decoupling is empirical, not merely formal. On the $96\times 96$ Rad-5 validation, per-pixel $|P_0 \hat{x}|$ and actual null-space error are essentially uncorrelated: for the LMMSE arm the Spearman and Pearson correlations are $-0.079$ and $-0.069$, and for the two learned arms they are $\approx 0.07$ and $\approx 0.06$. The ordering is if anything inverted for the learned arms — the top-$10\%$ highest-$|P_0 \hat{x}|$ pixels carry *lower* actual null-space error ($0.071$) than the remaining $90\%$ ($0.079$). The map therefore has, at most, image-level diagnostic value (an aggregate indication that a reconstruction leans on its prior); it is not a pixelwise error locator, and we do not use it as one anywhere in this paper. This negative result closes the section on the same theme it opened: the null space is where the prior speaks, and nothing the bucket records — not the residual, not the magnitude of the supplied content — can adjudicate whether that speech is true.

---

## 4. The Accountability Certificate (Achievability)

Section 3 established a converse: the record $y$ fixes $P_R x$ and nothing else, so any two feasible reconstructions may differ arbitrarily in the null space while producing the same measurement. That result is destructive — it says what cannot be verified. This section is its constructive counterpart. If the measurement cannot certify null-space content, it should at least certify, exactly and without ground truth, the one thing it does constrain: the measured component. We give a test-time operator that does this, prove that it contracts each measured mode by a known factor determined only by $A$ and $\lambda$, and show that it can be applied after any reconstructor as a post-hoc audit.

### 4.1 A plug-in test-time audit

Let $\hat{x}$ be any reconstruction — analytic, variational, or learned — with residual $r(\hat{x}) = A\hat{x} - y$. Define the audit operator

$$\Pi_y^\lambda(v) \;=\; v - A^\top\!\big(AA^\top + \lambda I\big)^{-1}\big(Av - y\big), \qquad \lambda > 0,$$

which we write compactly as $\Pi_y^\lambda(v) = v - B_\lambda(Av - y)$ with $B_\lambda = A^\top(AA^\top + \lambda I)^{-1}$. The audit requires only the operator $A$, the record $y$, and a single scalar $\lambda$; it needs no ground truth and no access to how $\hat{x}$ was produced. It is therefore a *plug-in* audit: it wraps an existing pipeline rather than replacing it.

**The update touches only the row space.** The correction $-B_\lambda(Av - y)$ lies in $\operatorname{range}(A^\top)$, which is exactly the row space. Consequently the null-space component is left untouched:

$$P_0\,\Pi_y^\lambda(v) \;=\; P_0 v .$$

This is the achievability side of the same geometry $A P_0 = 0$ that drove the converse. The audit cannot, and does not, adjudicate the null space; it operates strictly within the subspace the measurement is accountable for. Whatever a prior has placed in $P_0$ passes through the audit unchanged — a property we will rely on in §6, where the injected detail must survive the audit intact.

### 4.2 Exact per-mode contraction

The audit does not merely reduce the measurement residual; it reduces it by an amount we can write down in closed form. Applying $A$ to the audited estimate and simplifying,

$$A\,\Pi_y^\lambda(v) - y \;=\; \big[I - AA^\top(AA^\top + \lambda I)^{-1}\big]\,r(v) \;=\; \lambda\,(AA^\top + \lambda I)^{-1}\,r(v).$$

Diagonalizing $AA^\top = U_r \Sigma_r^2 U_r^\top$ and expanding the residual in the left singular vectors $u_i$, each measured mode is scaled independently:

$$c_i(\lambda) \;=\; \frac{\lambda}{\lambda + \sigma_i^2}.$$

The interpretation is exact rather than asymptotic. After one audit, the residual along the $i$-th measured mode is multiplied by $\lambda/(\lambda + \sigma_i^2)$ — a factor fixed entirely by the operator's singular value $\sigma_i$ and the chosen $\lambda$, independent of the image and of the reconstructor. As $\lambda \to 0$ the contraction becomes a hard projection onto $\{Av = y\}$; for $\lambda > 0$ it retains a controlled residual, appropriate when $y$ is noisy and should not be over-interpreted as exact truth. This closed-form modal spectrum is the object we certify.

**Float64 verification.** The contraction formula is confirmed in double precision. For the Rad-5 operator ($\sigma_{\min} = 3.476$, $\sigma_{\max} = 5.454$) the maximum deviation between the measured contraction and $\lambda/(\lambda + \sigma_i^2)$ is $1.04\times10^{-10}$; for the Scr-5 operator ($\sigma_{\min} = \sigma_{\max} = 1.000$) it is $2.29\times10^{-12}$. The identity therefore holds to the floating-point floor. (Repeated float32 pipeline audits saturate at a solver floor and are not evidence for the modal identity; the certificate is a float64 statement.)

### 4.3 Post-hoc audit across reconstructor families

Because $\Pi_y^\lambda$ depends only on $(A, y, \lambda)$, it can be applied uniformly to reconstructions from unrelated methods. We audit backprojection (BP), Tikhonov, a small-subset CS–TV sanity check, and a learned reconstructor, on both the Rad-5 and Scr-5 operators. Across these rows the pattern is consistent: the relative measurement error $\mathrm{RelMeasErr} = \lVert A\hat{x} - y\rVert / \lVert y\rVert$ drops by three to four orders of magnitude while the PSNR barely moves. For the learned Rad-5 output, $\mathrm{RelMeasErr}$ falls from $3.68\times10^{-2}$ to $1.90\times10^{-6}$ for a PSNR change of $+0.0136$ dB; for learned Scr-5 it falls from $1.80\times10^{-2}$ to $1.80\times10^{-5}$ at $+0.0387$ dB. Aggregated over all audited rows, the residual reductions of three to four orders come at $|\Delta\mathrm{PSNR}| \le 0.039$ dB, and the sign of the certificate is stable across the audited conditions (18/18). The audit thus buys measurement accountability essentially for free in image quality — the separation of these two axes is the subject of §5.

### 4.4 Position: exact modal spectrum, not a feasible-set bound

The fundamental-limits line closest to this result is Iagaru et al. (2026), who prove necessary-and-sufficient conditions for detail-transfer hallucination and give forward-model-only, ground-truth-free algorithms that *bound* hallucination magnitude via feasible-set diameters (worst-case kernel size). Our contribution is not the ground-truth-free stance, which is theirs, but the form of the guarantee. Where their assessment yields a bound over a feasible set — a supremum requiring paired data to instantiate — we give an *exact* per-mode contraction $c_i(\lambda) = \lambda/(\lambda + \sigma_i^2)$ read directly from the operator's SVD and realized as a plug-in operator applied after BP, Tikhonov, CS–TV, and learned reconstructors alike. The distinction is exact-versus-bound and operator-versus-existence: a closed-form modal spectrum applied at test time, rather than a feasible-set-diameter estimate. This certifies precisely what the measurement is accountable for. It says nothing about the null space, and we do not read it as licensing the invented texture §6 will introduce — $A\hat{x} = y$ is not certification that the null-space content is the true scene.

A second, closer antecedent is Bayesian. MacKay's (1992) *number of well-determined parameters*, $\gamma = \sum_i \lambda_i/(\lambda_i + \alpha)$, sums precisely our per-mode factor over the spectrum: the quantity $\lambda_i/(\lambda_i + \sigma_i^2)$ is, term for term, his per-mode contribution to the effective parameter count. The formula is the same; its epistemic role is inverted. MacKay reads $\gamma$ as an aggregate scalar consumed during *fitting* — for evidence maximization and regularization-strength selection — whereas we read the identical modal contraction as a per-mode, per-record *audit* applied after any reconstructor, paired with a constructive converse establishing that the complementary $n - m$ modes are unaccountable (§3) and a dial that governs them (§6). Same modal spectrum, opposite direction of use: a fitting diagnostic becomes a test-time certificate. We flag the coincidence explicitly so the certificate is not mistaken for a new scalar; its novelty is the use, not the number.

*Verification.* All numbers in this section are drawn from the audit experiments: float64 mode deviations $1.04\times10^{-10}$ (Rad-5) and $2.29\times10^{-12}$ (Scr-5); learned-output residual drops $3.68\times10^{-2}\!\to\!1.90\times10^{-6}$ (Rad-5) and $1.80\times10^{-2}\!\to\!1.80\times10^{-5}$ (Scr-5) at $|\Delta\mathrm{PSNR}| \le 0.039$ dB.

---

## 5. Quality Is Not Accountability: The Separation Law

The certificate of §4 contracts each measured mode by exactly $\lambda/(\lambda+\sigma_i^2)$. Section 3 established that the measurement pins down only $P_R x$. This section states the consequence as a formal separation: image quality, as scored by pixel metrics such as PSNR, and measurement accountability, as scored by the audit residual, are coupled *only* through the row-space part of the error. Where the residual error lives in the null space, the two quantities move independently. This is not a weakness of any particular metric; it is a geometric identity, and it is exactly what an accountability audit is for — it catches the failures a quality metric is structurally blind to.

### 5.1 The orthogonal error split

Let $\hat{x}$ be any reconstruction of a scene $x$, and let $e = \hat{x} - x$ be its error. Using the projectors of §2, decompose $e = P_R e + P_0 e$. Because $P_R P_0 = 0$, the two components are orthogonal, and the Pythagorean identity holds exactly:
$$\|e\|_2^2 = \|P_R e\|_2^2 + \|P_0 e\|_2^2.$$
The measurement sees only the first term: $A e = A P_R e = A(\hat{x} - x) = A\hat{x} - y$, so the audit residual $\|A\hat{x}-y\|$ is a function of $P_R e$ alone and is entirely insensitive to $P_0 e$. Pixel error, in contrast, integrates *both* terms. Define the pre-audit row-error share
$$s = \frac{\|P_R e\|_2^2}{\|e\|_2^2},$$
so that $\|P_0 e\|_2^2 = (1-s)\|e\|_2^2$. The scalar $s$ measures how much of a reconstruction's error the measurement is even able to reach.

### 5.2 The PSNR ceiling

An idealized hard audit — the limit $\lambda \to 0$ of the certificate — removes all row-space error and leaves the null-space error untouched. Its mean-squared error ratio is therefore
$$\frac{\mathrm{MSE}_{\mathrm{post}}}{\mathrm{MSE}_{\mathrm{pre}}} = \frac{\|P_0 e\|_2^2}{\|e\|_2^2} = 1 - s,$$
and the maximum PSNR gain available to *any* measurement-consistency correction of a given reconstruction is
$$\Delta\mathrm{PSNR}_{\max} = 10\log_{10}\!\left(\frac{\mathrm{MSE}_{\mathrm{pre}}}{\mathrm{MSE}_{\mathrm{post}}}\right) = -10\log_{10}(1 - s).$$

**Theorem (PSNR ceiling).** *For a reconstruction with pre-audit row-error share $s$, no measurement-consistency projection can raise its PSNR by more than $-10\log_{10}(1-s)$, and the soft audit with $\lambda>0$ realizes a strictly smaller gain whenever any visible row-space error survives the mode contraction $\lambda/(\lambda+\sigma_i^2)$.*

The interpretation is the separation law in one line: **auditing can only buy image-quality improvement from the part of the error that lies in the measurement-visible row space.** A reconstruction whose remaining error is mostly null-space prior content has $s \approx 0$, so its PSNR ceiling is near zero — the audit can drive the measurement residual to machine precision while barely touching PSNR. Trained networks sit in exactly this corner: they already have small row-error share, so re-imposing consistency is nearly free in PSNR but decisive in accountability. The two axes are decoupled precisely because $A P_0 = 0$.

This decoupling is visible in the audited networks of §4. The learned Rad-5 output moves from $22.192$ to $22.206$ dB — a change of $0.0136$ dB — while its relative measurement error falls from $3.68\times10^{-2}$ to $1.90\times10^{-6}$; the Scr-5 output moves $22.146 \to 22.185$ dB ($0.0387$ dB) while the residual falls $1.80\times10^{-2} \to 1.80\times10^{-5}$. Accountability changes by orders of magnitude; PSNR changes in the third decimal.

### 5.3 The range-share law tracks the sampling rate

The row-error share $s$ governs a *reconstruction's* PSNR headroom. A companion quantity, the row-space energy share $\rho = \|P_R x\|_2^2 / \|x\|_2^2$ of the *image itself*, governs the anchor ceilings that any measurement-consistent estimator can reach. The two use the same range–null orthogonality but answer different questions, and we keep them distinct: $\rho$ explains how much of the scene's energy is measurable at a given operator, while $s$ controls the audit's PSNR budget for a specific reconstruction.

Under the per-image mean-removed convention, $\rho$ tracks the sampling rate almost exactly: $\rho = 0.050$ at Rad-5 and $0.052$ at Scr-5 (5% sampling), rising to $0.101$ at Rad-10 and $0.099$ at Scr-10 (10% sampling). The row-space share is, to leading order, the fraction of dimensions the operator measures. The corresponding row-space PSNR ceilings ($14.304$, $14.311$, $14.541$, $14.534$ dB) are met by scrambled-Hadamard back-projection but not by Rademacher back-projection, whose DC/global-mean coverage differs. The lesson is that $\rho$ is set by the operator's sampling geometry, not by the reconstruction — reinforcing that the null space, which carries the remaining $1-\rho$ of the energy, is where a prior must act and where the measurement cannot follow.

### 5.4 What accountability catches that PSNR cannot

The separation law is not merely a bookkeeping identity; it is the reason a quality metric can pass while the reconstruction has quietly stopped depending on the data. We probe three such failures. In every case PSNR is nearly flat while the accountability residual moves by orders of magnitude — the operational signature of $s \approx 0$.

**Wrong measurements.** To test whether a trained reconstructor uses the recorded bucket data or merely emits a plausible prior sample, we feed each image another image's measurement vector $y$ (a batch roll) or shuffle the measurement coordinates, on 500-image probes. Across Rad-5, Scr-5, Rad-10, and Scr-10, wrong-$y$ inputs reduce PSNR by $12.174$–$14.793$ dB and shuffled-$y$ inputs by $14.537$–$17.026$ dB. The large drops confirm the network genuinely conditions its output on the recorded measurement rather than acting as a pure dataset prior. (This dependence is not a contradiction of the row-null geometry: the linear operator constrains only $P_R x$, but a *trained conditional* reconstructor uses $y$ to choose *which* null-space completion to emit; the collapse shows the conditioning is real, not that the geometry constrains the null space.)

**Coordinate shuffle.** The shuffled-$y$ arm above is a stronger perturbation than the batch roll, and the larger PSNR drops ($14.5$–$17.0$ dB versus $12.2$–$14.8$ dB) confirm that destroying the coordinate structure of the measurement degrades the reconstruction more than substituting a coherent but wrong record.

**Operator drift.** The most incisive test attacks accountability while leaving quality nearly untouched. In a simulation-scoped calibration-mismatch probe, the audit is performed with a *drifted* operator and the residual is then evaluated against the true operator. As the relative drift grows from $0$ to $0.05$, the Rad-5 post-audit PSNR moves only from $22.206$ to $22.178$ dB — a change of $0.028$ dB — while the residual against the true operator rises from $1.90\times10^{-6}$ to $4.88\times10^{-2}$, more than four orders of magnitude. The Scr-5 case behaves identically: PSNR $22.185 \to 22.155$ dB while the true-operator residual rises $1.80\times10^{-5} \to 1.26\times10^{-2}$. Drift silently destroys the contraction that the certificate certifies, yet PSNR barely registers it. A quality-only pipeline would report success; the accountability audit reports the mismatch.

### 5.5 Statement of the separation law

Taken together, Sections 5.1–5.4 establish the **separation law**: because $\|e\|_2^2 = \|P_R e\|_2^2 + \|P_0 e\|_2^2$ and $A P_0 = 0$, image quality and measurement accountability are coupled *only* through the row-space error share $s$, and are otherwise free to move independently. Quality metrics are blind to everything in the null space and to any accountability failure that does not first show up as row-space error — precisely the failures (wrong measurements, coordinate shuffles, operator drift) that the certificate is built to catch. This is why a reconstruction can score well and still be unaccountable, and why the two ledgers must be reported side by side: a high PSNR never implies the measurement was honored, and honoring the measurement never implies high PSNR. The separation is the empirical face of the geometry, and it sets up the sharper question of §6 — whether, having certified the row space and quarantined the null space, we can now supply detail into that null space safely and by rule.

---

## 6. Governed Supply: Injecting Detail Where the Measurement Is Blind

Sections 3–5 establish an asymmetry that is exact rather than practical. The measurement $y = Ax$ fixes the row-space component $P_R x$ and constrains nothing else, because $A P_0 = 0$; the audit of §4 certifies that a reconstruction reproduces $P_R x$, and the barrier of §3 shows that no test, however sharp, can decide the null-space component $P_0 x$ from $y$ alone. That barrier is usually read as a prohibition. Here we read the *same identity* the other way: since $A P_0 = 0$, any content placed in the null space is invisible to $A$, so editing $P_0 x$ cannot move $Ax$ and cannot break the measurement record. The geometry that forbids certifying invented detail simultaneously guarantees that supplying it is measurement-safe. This section makes that reading constructive.

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
The audited anchor gives $A x_0 = y$. The null-space identity $A P_0 = 0$ (§2) annihilates the second term regardless of the bracketed vector and regardless of $B$, so
$$A\hat{x}_B = y + 0 = y. \qquad \blacksquare$$

This is the constructive counterpart of the verifiability barrier. In §3 the identity $A P_0 = 0$ produced a feasible-but-wrong witness that satisfied the record to $\sim\!2\times10^{-15}$ — content that is provably unverifiable. Here the *same* identity guarantees that no matter how much adversarial detail the dial injects, the reconstruction never leaves the feasible set. The audit and the fusion are two consequences of one fact: what the bucket cannot certify, editing cannot corrupt.

### 6.4 The identity holds numerically

The theorem is exact in exact arithmetic; in floating point the residual is at round-off. Across the locked split (5.0% sampling, $64\times64$ grayscale STL10, $n = 4096$, $m = 205$, 3 seeds), the relative measurement error $\mathrm{RelMeasErr} = \lVert A\hat{x}_B - y\rVert / \lVert y \rVert$ has mean $3.6\times10^{-7}$ and max $5.7\times10^{-7}$ — the same round-off scale for every operating point on the dial, confirming that raising $B$ buys perceptual detail at no measurement cost. This is the achievability that the impossibility result licenses: a supply of null-space detail that is, by construction and in measurement, free of any consistency penalty.

We restate the scope deliberately, because the firewall of this program depends on it. $A\hat{x}_B = y$ certifies that the reconstruction *reproduces the buckets* — nothing more. The injected texture lives entirely in the null space, where the measurement furnishes no evidence; it is a prior-supplied hypothesis about plausible content, never a measurement-certified fact. The dial governs *how much* invented detail to supply; it never converts that detail into truth. Section 7 characterizes where along this dial the useful operating points lie, and §9 returns to why the accountable row space and the responsibly supplied null space must never be read in the same currency.

---

## 7. The Honesty Dial

The consistency theorem of §6 licenses a construction but does not, by itself, decide how much invented detail to admit. The fusion weight $B$ is a free scalar: every value produces a reconstruction that reproduces the record exactly, so the measurement offers no guidance on where along the family $\hat{x}_B = x_0 + P_0\big(d_A + B(d_G - d_A)\big)$ to stand. This is the point of the construction, not a defect of it — the geometry $A P_0 = 0$ that makes null-space content unverifiable (the converse of §3) is the same geometry that makes injecting it measurement-safe (the consistency theorem of §6). What remains is to choose $B$ honestly, report the trade it buys, and pre-commit to the choice before touching the confirmatory data. We call the resulting object an *honesty dial*: a single scalar that meters null-space detail, moves along a perception–distortion frontier, and never converts the perceptual gain it produces into a measurement claim.

### 7.1 A metered perception–distortion ladder

Sweeping $B$ from $0$ (the VQAE structure branch, $\hat{x}_0 = x_0 + P_0 d_A$) to $1$ (the VQGAN detail branch, $\hat{x}_1 = x_0 + P_0 d_G$) traces a one-parameter family. Read as operating points on the perception–distortion plane of Blau and Michaeli (2018), these reconstructions form a monotone ladder: as $B$ rises, perceptual and spectral quality improve and pixel fidelity degrades, with no interior optimum and no free lunch. Table 1 collects the seed-averaged locked-split means at five named points.

**Table 1. Locked-split absolute means (512 images, mean over 3 seeds).** Lower is better for LPIPS, RMSE, RAPSD; higher is better for PSNR, SSIM. Measurement fidelity is $\mathrm{RelMeasErr}\sim10^{-7}$ for every row.

| Operating point | LPIPS $\downarrow$ | PSNR (dB) $\uparrow$ | full-RMSE $\downarrow$ | SSIM $\uparrow$ | RAPSD $\downarrow$ |
|---|---|---|---|---|---|
| LMMSE anchor $x_0$ | 0.404 | 22.80 | 0.076 | 0.629 | 0.0041 |
| VQAE ($B=0$) | 0.300 | 23.13 | 0.073 | 0.657 | 0.0030 |
| **Balanced** ($B\!\approx\!0.55$) | **0.202** | 22.69 | 0.077 | 0.633 | 0.0027 |
| Quality-lite ($B\!\approx\!0.72$) | 0.182 | 22.21 | 0.081 | 0.609 | 0.0024 |
| Full VQGAN ($B=1$) | 0.172 | 21.43 | 0.089 | 0.571 | 0.0019 |

The ordering is strictly monotone in every column. LPIPS falls from $0.404$ (LMMSE) through $0.300$ (VQAE) to $0.172$ (full VQGAN), and RAPSD — a radially-averaged power-spectral measure of how closely the reconstruction's spectrum matches natural-image statistics — improves in lockstep from $0.0041$ to $0.0019$; over the same range PSNR declines from $23.13$ dB (VQAE, the pixel-fidelity optimum) to $21.43$ dB and SSIM from $0.657$ to $0.571$. This is the geometric content of the consistency theorem made empirical: because all null-space content is confined to $P_0$, each increment of $B$ trades measured pixel fidelity for perceptual realism along a smooth curve, and does so at fixed measurement fidelity. Every row of Table 1 satisfies $A\hat{x} = y$ to $\mathrm{RelMeasErr}\sim10^{-7}$; the ladder moves entirely within the null space the bucket cannot see.

The balanced point sits at a deliberately chosen interior of this ladder. It captures roughly two-thirds of the LPIPS improvement available between VQAE ($0.300$) and full VQGAN ($0.172$) — landing at $0.202$ — and the bulk of the RAPSD gain, while holding PSNR within half a decibel of the reconstruction-optimal VQAE. It is the operating point we pre-register and confirm below; quality-lite is offered as a second registered point for applications that tolerate a larger distortion cost in exchange for further perceptual gain.

### 7.2 A pre-registered, one-shot locked confirmation

Because $B$ is a free knob and LPIPS is the metric it is chosen to improve, a naive report would risk selecting the operating point on the same data used to evaluate it. The dial is therefore *locked*: the protocol fixes every degree of freedom before the confirmatory data are touched, and the confirmatory run is executed exactly once.

**Protocol.** The task is low-rate ghost imaging on $64\times64$ grayscale STL10 ($n = 4096$) at a single fixed operator: $m = 205$ rows (5.0% sampling), composed of 1 DC term, 128 low-frequency DCT rows, 56 low-sequency Hadamard rows, and 20 random rows, orthonormalized and generated from seed 772001. Data are split by raw-SHA256 deduplication into train (20,000, for the anchor and priors), validation (512, used *only* to select $B$), development (512, mechanical scoring), and a locked set (512, one-shot confirmation). The locked set is raw-hash disjoint from the union of 60,497 previously consumed STL10 hashes — overlap $0$, intra-duplicates $0$ — so the confirmation is on genuinely unseen images rather than a re-scored development set. The balanced weight is frozen per seed at $B = \{0.55, 0.55, 0.50\}$ on validation, before the locked split is read. An 8-condition acceptance gate is fixed in advance: LPIPS improves with its bootstrap confidence interval strictly below zero; the PSNR cost stays within $0.5$ dB; the RMSE cost within $0.005$; RAPSD does not worsen; all three seeds agree in direction; and measurement fidelity is maintained. Confidence intervals are formed by averaging the per-image fusion-minus-VQAE deltas over seeds and bootstrapping over images (2000 resamples).

**The locked result.** Against the VQAE structure branch, balanced fusion improves perceptual quality by
$$\Delta\mathrm{LPIPS} = -0.0977,\qquad \text{CI }[-0.1016,\,-0.0940],$$
a $32.6\%$ relative reduction whose interval lies entirely below zero. The cost is bounded and pre-registered: $\Delta\mathrm{PSNR} = -0.45$ dB (within the $0.5$ dB tolerance) and $\Delta\mathrm{RMSE} = +0.0039$ (within the $0.005$ tolerance), while spectral realism *improves* ($\Delta\mathrm{RAPSD} = -0.00030$, a move toward the natural-image spectrum). All 3/3 seeds agree in direction, and measurement fidelity is held at numerical precision (mean $\mathrm{RelMeasErr} = 3.6\times10^{-7}$, max $5.7\times10^{-7}$), so the perceptual gain costs nothing in measurement accountability. All 8/8 gate conditions pass.

**Replication on the same-drawn development set.** The development split, scored mechanically before the locked run, gives $\Delta\mathrm{LPIPS} = -0.0965$ ($32.9\%$) and $\Delta\mathrm{PSNR} = -0.43$ dB. The development and locked effect sizes nearly coincide ($-0.0965$ versus $-0.0977$; $32.9\%$ versus $32.6\%$), so the confirmation reflects a stable property of the method rather than selection on the evaluation set.

**Distribution-level corroboration.** Per-image LPIPS measures distance to each image's own reference; kernel Inception distance (KID) instead compares each method's *distribution* of reconstructions to the natural-image distribution. KID follows the same ladder and confirms the gain at the distributional level: balanced fusion improves locked-split KID from $0.119$ (VQAE) to $0.043$, a $2.7\times$ reduction, with quality-lite and full VQGAN closer still to the natural-image manifold. The injected null-space detail therefore makes the reconstructions distributionally more natural, not merely better on a single per-image metric.

### 7.3 The dial is a fixed scalar, not a per-image oracle

Two properties keep the dial honest, and both are matters of construction rather than tuning.

First, the frontier is well-behaved. A dense 21-point sweep of $B$ (step $0.05$) on the development split is smooth and monotone with no interior optimum: LPIPS falls steadily from $0.293$ at $B = 0$ to $0.167$ at $B = 1$, while PSNR declines monotonically from $22.87$ dB to $21.17$ dB and full-RMSE rises from $0.075$ to $0.091$. There is no discontinuity and no hidden sweet spot that a single validation-selected $B$ could miss — the dial is a stable interpolation, not a fragile hyperparameter.

Second, the *simplest* dial is the confirmed one. We evaluated two richer parameterizations in development — a 16-band radial-frequency weighting (a separate weight per spatial-frequency band) and a learned per-image gate — and neither beat the single global scalar under the registered validation-selection rule. Both tolerance-bounded rules (balanced and quality-lite) select the plain scalar across all three seeds; only an *unconstrained* oracle that ignores the PSNR and RMSE tolerances prefers a low-pass-cutoff variant, and its margin over the scalar is marginal. The confirmed result is thus the minimal one: a single measurement-safe scalar, not a learned gate or a frequency-dependent weighting.

We are deliberate about what $B$ is not. It is a fixed, validation-selected operating point held constant per seed — not adapted per image, and not an oracle that knows the best weight for each scene. A per-image adaptive weight might improve the trade, but it lies outside this frozen-system result, and the learned per-image gate we tried did not beat the global scalar. The dial meters how much prior-supplied detail enters the null space; it does not, and cannot, certify that the detail it admits is the true scene. That boundary is the subject of the two-ledger synthesis (§9): the perceptual win reported here lives entirely on the quality axis, and the exact consistency $A\hat{x}_B = y$ that accompanies it certifies reproduction of the record, never the correctness of what the record cannot see.

**A no-adaptation lemma.** The failure of the learned per-image gate is not a tuning artifact; it is forced by the same geometry as the converse. Any ground-truth-free rule — a map $g$ that chooses the fusion weight, a gate, or any null-space content from the record alone — is by construction a function of $y$, so the reconstruction is $\hat{x} = x_0(y) + P_0\,g(y)$. Two scenes on the same fiber, $Ax = Ax' = y$, therefore receive the *identical* reconstruction and the identical weight; no rule that sees only $y$ can distinguish them or adapt to which is the true scene. The per-image *oracle* weight $B^\star(x)$ depends on the true null component $P_0 x$, which $y$ does not observe, so the oracle gap is unclosable by any admissible rule to exactly the extent that $B^\star$ depends on the unmeasured content. This is a corollary of the converse (§3): the feasible-but-wrong twins that defeat verification are the same twins that defeat ground-truth-free adaptation. It is empirically sharp. A feature-based per-image selector — null energies $\|d_A\|,\|d_G\|$, the chord and cosine between the two residuals, anchor texture statistics, and cross-arm perceptual distances $\mathrm{LPIPS}(x_0,x_G)$ etc. — trained on validation and tested on development, recovers the oracle only through a *global* shift of $B$: against a global scalar matched to its own mean PSNR, its per-image excess is $\le 0.002$ LPIPS, and a deliberately constant predictor reproduces it exactly. The features barely rank-correlate with the oracle weight ($|\rho| \le 0.24$). The dial is a scalar not because we declined to adapt it, but because no honest rule can — per-image adaptation from $y$ alone is a global prior in disguise.

The scalar $B$ is the closest interface in this literature to CodeFormer's fidelity–quality knob $w$ (Zhou et al. 2022) and ESRGAN's network-interpolation weight $\alpha$ (Wang et al. 2018); the difference is that those knobs act on entangled features, so higher quality genuinely moves the reconstruction off the measurement manifold, whereas $B$ acts only within $P_0$ and holds $A\hat{x}_B = y$ exact at every setting. The honesty dial is thus a perception–distortion control (Blau and Michaeli 2018) whose distortion is paid entirely in null-space coordinates the measurement never certified.

---

## 8. Robustness and Reach

The locked claim of §7 fixes one operating point: the balanced dial $B$, at a single $5.0\%$ operator, on one $64\times64$ resolution, under noiseless acquisition. This section asks how far that operating point reaches — across sampling rate, across measurement noise, and across the image population. All results here are **development-level**: they reuse the frozen system, re-select nothing on the locked split, and constitute supplementary evidence rather than a second pre-registered claim. We label them as such throughout and treat none of them as certified. The measurement-consistency guarantee, however, is not development-level: because $A P_0 = 0$ holds for every operator by construction, $A\hat{x}_B = y$ remains exact at every rate and every noise level below.

### 8.1 Cross-rate generalization (development-level)

To test whether the balanced-fusion advantage is specific to the $5\%$ operator, we reuse the rate-agnostic priors unchanged and retrain only the lightweight anchor refiner at $2\%$ ($m=82$) and $10\%$ ($m=410$), then run the identical pipeline: select the global scalar $B$ on validation under the same tolerance rule, and score on the held-out development split (3 seeds per rate). This does not re-touch the frozen $5\%$ locked result, which appears only as the anchor point.

The advantage holds at every rate. Balanced fusion lowers LPIPS relative to the VQAE branch by $-0.116$ ($29.3\%$) at $2\%$, $-0.098$ ($32.6\%$) at $5\%$, and $-0.076$ ($34.2\%$) at $10\%$, with all $3/3$ seeds agreeing in direction at each rate and a PSNR cost of $-0.39$, $-0.45$, and $-0.43$ dB respectively — below the pre-registered $0.5$ dB validation tolerance throughout. The relative gain grows mildly with sampling rate: at higher rates more genuine detail is recoverable for the fusion to exploit, which is consistent with the fusion mechanism rather than an artifact of a single operator.

| Sampling rate | $\Delta$LPIPS (balanced $-$ VQAE) | Relative gain | $\Delta$PSNR (dB) | Seeds same-direction |
|---|---|---|---|---|
| $2\%$ ($m=82$) | $-0.116$ | $29.3\%$ | $-0.39$ | $3/3$ |
| $5\%$ ($m=205$, *locked*) | $-0.098$ | $32.6\%$ | $-0.45$ | $3/3$ |
| $10\%$ ($m=410$) | $-0.076$ | $34.2\%$ | $-0.43$ | $3/3$ |

### 8.2 Noise robustness (development-level)

The locked result is noiseless. To probe robustness we add i.i.d. Gaussian noise of standard deviation $\sigma$ to the bucket measurements and re-run the frozen system with the balanced $B$ unchanged (3 seeds, locked split). Two behaviors emerge. First, VQAE is the most noise-stable branch but always the least perceptual. Second, and more instructive, full VQGAN degrades sharply as noise grows — its fine synthesized detail amplifies the measurement noise — so that balanced fusion, which the noiseless ladder places *above* full VQGAN, **overtakes it at $\sigma = 0.02$** (LPIPS $0.197$ for balanced versus $0.204$ for full VQGAN) and beats it decisively at $\sigma = 0.05$ ($0.250$ versus $0.293$).

| Bucket noise $\sigma$ | VQAE | Balanced | Full VQGAN |
|---|---|---|---|
| $0.000$ | $0.300$ | $0.202$ | $0.172$ |
| $0.005$ | $0.299$ | $0.199$ | $0.172$ |
| $0.010$ | $0.297$ | $0.195$ | $0.176$ |
| $0.020$ | $0.295$ | $0.197$ | $0.204$ |
| $0.050$ | $0.304$ | $0.250$ | $0.293$ |

This is the behavior a controlled interior operating point should have: balanced fusion keeps most of VQGAN's perceptual benefit at low noise while degrading far more gracefully as the measurement becomes unreliable. The crossover is not evidence that noise certifies anything about the null space — the guarantee $A\hat{x}_B = y$ concerns only the row space at every $\sigma$ — but it does show that the recommended dial position is also the robust one.

### 8.3 Breadth of improvement and its failure mode

The gain is broad, not anecdotal. Balanced fusion improves LPIPS on $97.5\%$ of locked images for one seed and $99.2\%$ for another, worsening it on at most $13$ of $512$ images, with a worst-case regression of only $+0.07$ LPIPS and most regressions far smaller. The failures are not random: they concentrate on **man-made periodic and edge structure** — fences, vehicle body panels, airplane fuselages — where the natural-image VQGAN prior is a mismatch and synthesizes texture that conflicts with the regular geometry the scene actually contains. This is the expected and interpretable failure mode of a natural-image prior applied to structured man-made content, and it delimits where the dial should be advanced with care: the reach of responsible injection is bounded by the support of the prior, not by the measurement.

---

## 9. One Geometry, Two Guarantees

The preceding sections established two results that appear, at first reading, to belong to different projects. Act 1 produced a certificate: a ground-truth-free, test-time audit $\Pi_y^\lambda$ that contracts each measured singular mode by exactly $\lambda/(\lambda+\sigma_i^2)$ and separates measurement accountability from image quality. Act 2 produced a dial: a null-space fusion $\hat{x}_B = x_0 + P_0\big(d_A + B(d_G - d_A)\big)$ that injects VQAE structure and VQGAN detail while keeping $A\hat{x}_B = y$ exactly for every $B$. This section states the single fact from which both follow, and draws the one consequence that neither act can state alone.

### 9.1 The same identity is the guarantee and the limit

Everything rests on $A P_0 = 0$. Because the row and null projectors are orthogonal and complementary, $P_R + P_0 = I$ with $P_R P_0 = 0$, and the measurement of any candidate image reduces to $A x = A P_R x$: the record $y$ fixes exactly $P_R x$ and says nothing about $P_0 x$.

Read forward, this identity is the certificate. The audit's contraction acts only on the measured modes; the null-space content passes through untouched, $P_0\,\Pi_y^\lambda(v) = P_0 v$, so $\Pi_y^\lambda$ can certify agreement with the record to numerical precision (RelMeasErr falling three to four orders of magnitude — for the learned Rad-5 reconstructor from $3.68\times10^{-2}$ to $1.90\times10^{-6}$ — at $|\Delta\text{PSNR}| \le 0.039$ dB across the 18 audited rows) without ever needing the ground truth.

Read the other way, the same identity is the limit. Because $A P_0 = 0$, the null component is invisible to the record, and no test on $y$ can distinguish among the images that share $P_R x$. The converse witness $u_{ij} = x_j - A^\dagger(A x_j - y_i)$ is measurement-consistent with $y_i$ while carrying the wrong null content $P_0 x_j$; it matches the record to $2.16\times10^{-15}$–$4.00\times10^{-15}$ — tighter than the noisy truth's own residual of $3.45\times10^{-3}$–$7.46\times10^{-3}$ — yet is a different scene (PSNR to target $7.70$–$11.38$ dB). Consistency is therefore not correctness. The property that lets the audit certify the row space is exactly what makes the null space uncertifiable.

The two guarantees are thus not two facts but one fact read in two directions. The audit **certifies only $P_R$**: it is a guarantee about the measured coordinates and is silent, by construction, about the rest. The dial **lives only in $P_0$**: its content is confined to the coordinates the record cannot see, which is precisely why raising $B$ costs nothing in measurement fidelity (RelMeasErr mean $3.6\times10^{-7}$ across the locked split). Guarantee and limit are the same equation, $A P_0 = 0$, applied to complementary subspaces.

### 9.2 A prior can raise naturalness without reducing distance to the true null content

The construction forces a consequence that is easy to overlook and important to state plainly. Take any two audited-and-fused reconstructions sharing an operator and anchor — for instance the balanced and the full-VQGAN points on the ladder. Both satisfy $A\hat{x} = y$, so both have the identical row-space component $P_R\hat{x} = A^\dagger y$. Their entire difference lives in the null space:
$$
\hat{x}_{B_1} - \hat{x}_{B_2} = P_0\big[(B_1 - B_2)(d_G - d_A)\big] \in \mathcal{N}(A).
$$
Whatever a stronger prior buys — the balanced point improves LPIPS by $-0.0977$ (a 32.6% relative gain, CI $[-0.1016,-0.0940]$) over the VQAE branch, and dataset-level KID falls from $0.119$ to $0.043$ — it buys entirely in $P_0$, moving along the perception–distortion ladder (LMMSE $0.404 \to$ VQAE $0.300 \to$ balanced $0.202 \to$ quality-lite $0.182 \to$ full VQGAN $0.172$).

The null-space error, however, is $P_0(\hat{x} - x) = P_0\hat{x} - P_0 x$, and its second term $P_0 x$ is the true scene's unmeasured content — the one quantity the record cannot reveal. A prior that increases naturalness changes $P_0\hat{x}$; it carries **no guarantee** of reducing the distance to $P_0 x$. A more realistic reconstruction may be nearer to, or farther from, the true null content than a flatter one, and the measurement provides no way to tell which. This is the same decoupling the observable map $|P_0\hat{x}|$ exhibits directly: on the $96\times96$ Rad-5 validation its Spearman and Pearson correlations with actual null-space error are near zero, and for the learned arms the top-10% highest-magnitude pixels carry *lower* error than the rest. The map is a prior-supplied-content map, not a pixelwise error locator. Perceptual gain is real and is measured on the quality axis; it must not be read as evidence about correctness on the null axis, because $A P_0 = 0$ severs the two.

### 9.3 Prescription: report both axes, and label provenance

The discipline that follows is procedural, and we adopt it throughout. First, **report measurement accountability and image quality as separate axes**, never one as a proxy for the other. RelMeasErr (audited to $\sim10^{-7}$–$10^{-15}$) certifies agreement with the record; LPIPS, KID, and RAPSD describe perceptual quality; PSNR and RMSE describe pixel distortion. A high value on either axis licenses no inference about the other: PSNR moves under $0.04$ dB while residuals fall orders of magnitude (Act 1), and a $32.6\%$ LPIPS gain coexists with a bounded $-0.45$ dB PSNR cost (Act 2). The audit and the perceptual metric answer different questions and are reported side by side.

Second, **label every reconstructed coordinate by provenance**. The $m$ row-space coordinates are *measured* — pinned by $y$ and certifiable by the audit. The null-space coordinates carrying the injected detail are *prior-supplied* — a modeling choice, exact against the record by $A P_0 = 0$, and openly attributed to the prior rather than the bucket. And the true null content $P_0 x$ is *unverifiable* — no test on $y$ can certify it, which is why the honest move is to inject prior detail exactly where it is provably invisible, and to say so.

These three labels are the whole synthesis. The audit certifies what the measurement determines; the dial governs what a prior supplies; and $A P_0 = 0$ guarantees, in the same stroke, that governed injection cannot break the certificate and that the certificate cannot vouch for the injection. We state the resulting boundary without hedging and without exception: $A\hat{x} = y$ certifies that a reconstruction reproduces the buckets; it never certifies that the injected texture is the true scene.

---

## 10. Related Work

Our work sits at the intersection of computational ghost imaging, the range–null geometry of undersampled linear inverse problems, generative image priors, and the recent literature on hallucination, fundamental limits, and trustworthy reconstruction. What organizes the discussion below is a single distinction: almost every prior line uses the range–null decomposition, or a prior over the unmeasured content, to **reconstruct** — to produce one image that looks right or scores well. We instead use the same geometry to **certify, bound, and meter**: to prove what the bucket measurement can and cannot vouch for, to audit that accountability at test time without ground truth, and to inject admittedly invented detail only where the measurement is provably blind. Each subsection first summarizes the line, then states our delta.

### 10.1 Ghost imaging and single-pixel imaging

Ghost imaging originated as a quantum-optical curiosity: Pittman et al. (1995) recovered a magnified aperture image purely from photon-coincidence correlations, with no image present in either detector's individual counts. Gatti et al. (2004) then proved analytically that classically correlated thermal beams reproduce all the imaging features of entangled ghost imaging, and Magatti et al. (2004) demonstrated both ghost image and ghost diffraction from a single classical thermal source by altering only the reference arm. Shapiro (2008) and Bromberg et al. (2009) collapsed the modality to its computational core: with the illumination patterns known, a single bucket detector suffices, so ghost imaging is classical coherence propagation and the reconstruction is a correlation $C(\rho) = \sum_m \phi_m(\rho)\, y[m]$ against known patterns — precisely our forward operator $y = Ax$. In parallel, the single-pixel camera (Duarte et al., 2008) and compressive ghost imaging (Katz et al., 2009) established the undersampled regime $y = \Phi x$ with $m \ll n$, recovered by $\ell_1$/TV minimization, and stated the governing fact plainly: since $m < n$, infinitely many images satisfy the same measurements, and recovery is possible only under a prior. Gibson et al. (2020) survey this progression from correlation through compressed sensing to learned reconstruction; Song et al. (2025) update it and flag interpretability and generalization as unresolved.

**Our delta.** These works establish the physical instantiation of our operator $A$ and the undersampled regime that makes its null space non-trivial. We take the "infinitely many $x$ satisfy the same $y$" observation — usually stated in passing to motivate a sparsity prior — and elevate it into the load-bearing object of the paper: a constructive, per-instance feasible-but-wrong witness and a test-time certificate over exactly this measurement model.

Learned ghost-imaging reconstructors sharpen the stakes. Lyu et al. (2017) restored recognizable images at sampling ratios as low as $\beta = 0.05$, where compressed sensing collapses, and candidly noted the outputs "do not resemble exactly the ground truth"; the CNN pipeline of Wang et al. (2018) and the U-Net denoiser of Shimobaba et al. (2017) buy large PSNR/SSIM gains with no enforced measurement consistency; and a recent X-ray study (2024) argues rigorously that reducing the number of measurements does not itself add information, so any sub-sampling-limit detail a learned reconstructor supplies originates from the prior, not the measurement.

**Our delta.** This GI-native line is the empirical hook for our converse: at low $\beta$ the network fills the null space with trained-in structure the measurement cannot certify. Prior work optimizes quality (MSE, recognizability, PSNR); we separate quality from accountability, audit the learned reconstructor alongside classical ones, and never claim the bucket certifies the invented texture.

### 10.2 Range–null and data-consistency reconstruction

A large body of work exploits the range–null decomposition $x = P_R x + P_0 x$ with $P_R = A^\dagger A$, $P_0 = I - A^\dagger A$, and $A P_0 = 0$. The origin of the "edit only the null space, keep the measurement exact" idea is IDBP (Tirer & Giryes, 2018), which alternates denoising with a backward projection onto $\{Hx = y\}$ so that all restoration dynamics live strictly in the null space. Schwab et al. (2019) formalized the null-space network $L = \mathrm{Id} + (\mathrm{Id} - A^+A)N$, proving it is a convergent regularization that edits only $\ker(A)$ while preserving data consistency exactly, and Chen & Davies (2020) trained separate range- and null-space networks over the same $P_R = H^\dagger H$, $P_0 = I - H^\dagger H$ split. In imaging practice, hard data-consistency layers (Schlemper et al., 2018), model-based unrolling with an explicit $(A^H A + \lambda I)^{-1}$ solve (Aggarwal et al., 2019), variational networks (Hammernik et al., 2018), learned primal–dual (Adler & Öktem, 2018), ADMM-Net (Yang et al., 2016), and ISTA-Net (Zhang & Ghanem, 2018) all restore or preserve the measured component while a learned prior supplies the rest. The same construction anchors the most recent null-space methods: RND-SCI (Wang et al., 2023) reconstructs hyperspectral snapshots as $x = \Phi^\dagger y + (I - \Phi^\dagger \Phi)q$ with a conditional network generating only the null term; NPN (Jacome et al., 2025) trains a network to predict a low-dimensional null-space projection $Sx^*$ from $y$, explicitly noting that data fidelity leaves the null space uncontrolled; and GSNR (Gualdron-Hurtado et al., 2026) builds a graph-smooth basis for $\mathrm{Null}(H)$ via the null-restricted Laplacian $T = P_n L P_n$ and proves minimax coverage/predictability bounds for the invisible directions, gaining up to 4.3 dB.

**Our delta — the null-space reconstruction cluster (position against).** This cluster is our closest prior art and our sharpest point of departure. Every one of these methods uses $A P_0 = 0$ to install the *correct* content in the null space, and treats consistency-by-construction as a fidelity virtue — a proxy for correctness. RND-SCI generates a monolithic null term for speed and PSNR; NPN and GSNR go further and argue the null content *is* predictable, learning it from $y$ and enforcing only soft fidelity, with GSNR's coverage/predictability curves computed at design time from dataset statistics. We invert the entire premise. The same $A P_0 = 0$ that lets these methods edit the null space without breaking consistency is exactly why measurement consistency **cannot certify** what they put there. We contribute (i) a constructive feasible-but-wrong witness, matching the record to machine precision, showing the construction is a barrier rather than a guarantee; (ii) a ground-truth-free, test-time certificate that contracts each measured mode by exactly $\lambda/(\lambda+\sigma_i^2)$, a closed-form modal spectrum rather than a soft penalty or a design-time diagnostic; and (iii) an exact-consistency dial that keeps $A\hat{x}_B = y$ for **every** setting $B$ behind a pre-registered locked gate. Schwab et al. (2019), Chen & Davies (2020), and IDBP (Tirer & Giryes, 2018) supply the theoretical scaffolding we build on, but none poses a converse, audits accountability, or meters the invention.

Plug-and-play priors (Venkatakrishnan et al., 2013) and Regularization by Denoising (Romano et al., 2017) decouple a forward-model solve from a black-box denoiser prior, the conceptual ancestor of separating measured fidelity from supplied content.

**Our delta.** PnP and RED let the denoiser alter measured modes too and offer no test-time measurement-accountability certificate; our injection is confined to the null space so $A\hat{x}_B = y$ holds exactly, and our audit reads the per-mode contraction regardless of which prior is plugged in.

### 10.3 Generative and VQ priors for inverse problems

Generative priors replace sparsity with a learned low-dimensional manifold. Bora et al. (2017) showed $O(k \log L)$ measurements suffice to recover signals near a generative model's range under the Set-Restricted Eigenvalue Condition — a condition that requires differences of natural signals to lie *away* from $\mathrm{null}(A)$. Shah & Hegde (2018) give a projected-gradient GAN-prior solver with linear-convergence guarantees under the same S-REC; Asim et al. (2020) use invertible (flow) priors with zero representation error and prove recovery error is governed by the smallest $n-m$ singular values; Lunz et al. (2018) learn an adversarial regularizer that approximates distance to the data manifold; and Yeh et al. (2017) hallucinate plausible content strictly inside a missing region while pinning observed pixels. Untrained priors — Deep Image Prior (Ulyanov et al., 2018) and the Deep Decoder (Heckel & Hand, 2019) — show that architecture alone supplies image statistics, with the Deep Decoder making the reachable set an explicit capacity knob. The vector-quantized line runs from VQ-VAE (van den Oord et al., 2017) through VQ-VAE-2 (Razavi et al., 2019), which factorizes an image into a global-structure code and a local-detail code, to VQGAN (Esser et al., 2021), whose patch discriminator synthesizes crisp texture that plain VQ-VAE cannot recover. Codebook priors dominate blind face restoration — VQFR (Gu et al., 2022), CodeFormer (Zhou et al., 2022), and RestoreFormer (Wang et al., 2022) — with CodeFormer exposing a single scalar $w \in [0,1]$ that trades fidelity for quality at test time.

**Our delta.** We build our supply mechanism directly on this lineage: VQ-VAE-2's structure/detail factorization motivates our structure (VQAE) plus detail (VQGAN) split, and CodeFormer's scalar $w$ is the closest interface to our dial $B$. But every one of these controls acts on *entangled* features, so higher quality genuinely costs measurement fidelity ($A\hat{x} \neq y$), and S-REC-style guarantees explicitly assume away the regime — null-space differences that are uncontrolled — where we operate. Our dial injects the same VQ-generated content **only** in $P_0$, so $A\hat{x}_B = y$ holds exactly for every $B$, and we treat the resulting texture, however realistic, as content the measurement provably cannot certify. Asim et al.'s (2020) smallest-$(n-m)$ singular-value bound is the quantitative cousin of our per-mode $\lambda/(\lambda+\sigma_i^2)$ contraction; we make that spectrum exact and auditable.

### 10.4 Perception–distortion tradeoffs and perceptual metrics

Blau & Michaeli (2018) proved a fundamental, distortion-measure-agnostic perception–distortion tradeoff: lowering distortion forces the reconstruction distribution away from natural-image statistics, so "matches the reference" and "looks real" are formally at odds; their rate-distortion-perception extension (2019) adds that perceptual realism must be paid for in rate or distortion. SRGAN (Ledig et al., 2017), ESRGAN (Wang et al., 2018) with its scalar network-interpolation dial, and the PIRM challenge (Blau et al., 2018) operationalized this plane for super-resolution, and LPIPS (Zhang et al., 2018) became the standard perceptual metric — the one we use to report our locked improvement. PULSE (Menon et al., 2020) is the canonical exhibit that consistency is not correctness in super-resolution: visually distinct realistic faces all downscale to the identical low-resolution input.

**Our delta (position against).** Blau & Michaeli's tradeoff is *statistical* — a divergence between the reconstruction distribution and natural-image statistics — and its GAN Lagrange knob is the conceptual ancestor of our dial. Our converse is *geometric and exact*: feasible-but-wrong images matching $y$ to machine precision make the null content unverifiable regardless of any distributional statistics. ESRGAN's $\alpha$ and CodeFormer's $w$ trade quality for fidelity along a soft curve; our dial is null-space-confined, so the measurement stays exact at every point rather than at one tradeoff position. We use LPIPS to quantify the perceptual win and keep it strictly on the quality ledger, never conflating it with accountability. PULSE supplies feasible-but-wrong images but presents one as "the" answer and audits nothing; we govern and certify.

### 10.5 Diffusion inverse-problem solvers

Modern training-free solvers use a pretrained diffusion model as a prior. The score-SDE framework (Song et al., 2021) conditions an unconditional model on the observation; Song et al. (2022) factor the operator as $A = P(\Lambda)T$ and keep measured transform coefficients while synthesizing the rest; DDRM (Kawar et al., 2022) diffuses in the SVD spectral space, synthesizing zero-singular-value (null) directions and conditioning measured ones with a per-mode blend weight; MCG (Chung et al., 2022) writes $A = I - P^\top P$ (our $P_0$) to update only the orthogonal complement while pinning the measured subspace; DPS (Chung et al., 2023) adds a soft measurement-gradient step for general noisy problems; and DiffPIR (Zhu et al., 2023) embeds half-quadratic-splitting into the sampling loop. DDNM (Wang et al., 2023) is the most direct methodological cousin of our dial: it fixes the range to $A^\dagger y$ and lets a diffusion model synthesize only the null-space content $(I - A^\dagger A) \bar{x}$, so $A\hat{x} = y$ holds exactly. The survey of Daras et al. (2024) taxonomizes the field and gives the canonical noiseless data-consistency update $(I - A^\dagger A)\, \mathbb{E}[X_0 \mid x_t] + A^\dagger y$.

**Our delta.** DDRM's per-singular-value treatment and DDNM's exact null-space fill are the spectral and structural skeletons of our framework, and MCG's $A = I - P^\top P$ is literally our $P_0$. The difference is direction and purpose. DPS, DiffPIR, DDRM ($\eta_b < 1$), and the score-based solvers enforce consistency softly, so $A\hat{x} \neq y$ in general and prior content leaks into measured modes; DDNM and MCG keep the null-space move exact but chase SOTA restoration realism. We do not compete on restoration quality and do not claim to beat diffusion. Instead, our certificate is a ground-truth-free *audit* — not a sampler — that quantifies, per measured mode, how much of any such solver's output the measurement can vouch for; and our dial governs the null-space content these methods fill ungoverned, with an exact-consistency guarantee and a locked gate they do not provide.

### 10.6 Hallucination, instability, and fundamental limits

A sobering literature documents that consistency and visual quality decouple from correctness. Antun et al. (2020) showed state-of-the-art learned MRI/CT reconstructors are unstable to near-invisible perturbations and can erase small structural detail while stable compressed-sensing baselines are not; Genzel et al. (2020) offered the balancing counterpoint that, under matched adversarial testing, standard networks match a robust TV benchmark and much instability traces to noiseless-training "inverse crimes." Buday et al. (2026) inserted diagnostically misleading anatomical hallucinations via imperceptible k-space perturbations that PSNR/NRMSE/SSIM cannot detect. Bhadra et al. (2021) formalized hallucinations by projecting a reconstruction onto measurement and null spaces, showing null-space hallucinations are attributable solely to the prior and cannot be assessed without the ground truth; DynamicDPS (Kim et al., 2025) adopts the same intrinsic (data-inconsistent) versus extrinsic (null-space) split and suppresses both with a diffusion prior plus data consistency; and HalluGen (Kim et al., 2025) fabricates controllable hallucinations by gradient ascent to train reference-free detectors, noting that consistency-preserving errors are the hardest to catch. Learned primal–dual (Adler & Öktem, 2018) even internally observes that a true feature is indistinguishable from a false feature of the same size and contrast. The fundamental-limits side is anchored by Iagaru et al. (2026), who prove necessary-and-sufficient conditions for detail-transfer hallucination in (possibly nonlinear) inverse problems — consistent decoders can only hallucinate details that are almost invisible in measurement space ($\lVert f(x + x_{\det}) - f(x)\rVert \le 2\epsilon$) — and give ground-truth-free, forward-model-only algorithms that bound and assess hallucination magnitude via feasible-set diameters (worst-case kernel size).

**Our delta — the limits-and-assessment line (position against).** This is the line we must not overclaim against, and we cite it as the anchor for the barrier itself. Antun et al. (2020), Buday et al. (2026), and Bhadra et al. (2021) establish empirically and by SVD decomposition that consistency and quality metrics do not certify correctness; Iagaru et al. (2026) prove the fundamental limit and occupy the generic slot of forward-model-only, ground-truth-free hallucination *assessment*. We do not claim "null content is unverifiable" or "ground-truth-free assessment" as novel — that ground is theirs. Our contribution is on four axes they do not cover. First, where Iagaru et al. give feasible-set-diameter *bounds* (a sup over feasible sets, requiring paired data), we give an **exact** per-mode contraction $\lambda/(\lambda+\sigma_i^2)$ from the operator's SVD, applied as a test-time plug-in audit across BP, Tikhonov, CS-TV, and learned reconstructors. Second, where they establish *existence* via necessary-and-sufficient conditions, we give a **constructive** per-instance feasible-but-wrong witness that matches the record to machine precision. Third, we frame and demonstrate a **quality-versus-accountability separation** (PSNR essentially flat while measurement residuals drop orders of magnitude) as an explicit protocol. Fourth, and entirely uncontested, we move from *assessment* to *governance*: a constructive, metered null-space injection dial with $A\hat{x}_B = y$ exact for every $B$ and a pre-registered locked gate. The clean axis is assessment versus governance, exact versus bound, constructive versus existential. Bhadra et al.'s (2021) null-space hallucination map requires the ground truth — exactly the quantity our converse proves unverifiable — so we cite it to say the null-space error is real but cannot be certified at test time.

### 10.7 Uncertainty, conformal prediction, and trust

The standard response to unreliable reconstruction is uncertainty quantification. Kendall & Gal (2017) established the aleatoric/epistemic taxonomy; Narnhofer et al. (2021) produce Bayesian epistemic-uncertainty maps for variational MRI. On the distribution-free side, Angelopoulos et al. (2022) endow any image-to-image regressor with per-pixel risk-controlling prediction intervals to flag hallucinations, conformal risk control (Angelopoulos et al., 2022) guarantees $\mathbb{E}[\mathrm{loss}] \le \alpha$ for bounded monotone losses, Wen et al. (2024) conformalize the downstream task output rather than per-pixel maps, and Lu et al. (2022) apply ordinal conformal sets to disease severity rating. Wen et al. (2024) in particular argue, as we do, that per-pixel uncertainty maps are the wrong object because they miss many-pixel hallucinated structure.

**Our delta.** These guarantees are either heuristic (Bayesian std maps) or ground-truth-dependent at calibration (RCPS, conformal risk control, task-conformal), and they operate in output/pixel/label space. Our certificate is deterministic, ground-truth-free at test time, and lives in the operator's spectral geometry: it contracts each *measured* mode by exactly $\lambda/(\lambda+\sigma_i^2)$ and audits measurement accountability rather than pixel error or a labeled risk. Crucially, using Kendall & Gal's own taxonomy, our accountability is neither aleatoric nor epistemic — it is a third, geometry-determined quantity: null-space unverifiability persists even with a perfect model in the noiseless limit, because $A P_0 = 0$ makes null content invisible to $y$ regardless of noise or model confidence. No calibration set can fix what the converse proves.

### 10.8 Cross-field framings: identifiability, information theory, and Bayesian inversion

Our thesis has deep provenance outside imaging. Econometric identification theory (Koopmans & Reiersøl, 1950) established that observationally equivalent structures generate the identical distribution, so a characteristic is knowable only if invariant across all such structures — identification precedes estimation. Rothenberg (1971) tied local identifiability to nonsingularity of the Fisher information matrix, making non-identifiability a rank deficiency / null space in information geometry. Manski (2003) founded partial identification — when data do not point-identify a parameter they confine it to a sharp set, and honest inference reports that set — and Lewbel (2019) formalized set identification and *normalizations*: restrictions that select within the identified set without altering any meaningful quantity. On the imaging side, Landau (1967) fixed the sampling-density threshold below which degrees of freedom exceed measurements and a non-trivial null space is forced by necessity, Candès, Romberg & Tao (2004) proved exact CS recovery under sparsity while noting that without the prior distinct signals share identical partial measurements, and Stuart (2010) cast inverse problems as Bayesian posteriors in which, when data dimension is below the unknown, the prior remains decisive even in the zero-noise limit. Within the imaging sciences proper the lineage is older and more direct: Backus & Gilbert (1968–1970) characterized exactly which linear functionals of an underdetermined model are resolved by finite data — a functional is determined iff its representer lies in the row space, our certifiable/uncertifiable split for linear observables — and Barrett & Myers (*Foundations of Image Science*, 2004) built the estimability and null-function calculus, and task-based image-quality assessment, on the identical range/null decomposition. MacKay (1992) is the Bayesian antecedent of the certificate itself: his effective number of well-determined parameters $\gamma = \sum_i \lambda_i/(\lambda_i+\alpha)$ sums our per-mode contraction over the spectrum (§4.4).

**Our delta.** These fields supply the rigorous vocabulary we import and instantiate, and we own the lineage rather than reinvent it. Our feasible-but-wrong images are the imaging analogue of Koopmans & Reiersøl's observational equivalence and Rothenberg's singular information direction; our null space $P_0 = I - A^\dagger A$ is Manski's identification region, Landau's forced kernel, and Barrett & Myers's null-function subspace; our certifiable-functional split is Backus–Gilbert resolvability; and, most sharply, our governed dial is Lewbel's *normalization* — injecting null content is "without loss of generality" for the measurement, since $A\hat{x}_B = y$ exactly for every $B$, leaving the identified quantity untouched. Candès–Romberg–Tao's converse is the CS-side twin of the econometric one, and Stuart's "prior stays decisive" is the Bayesian mirror of our per-mode contraction, whose factor $\lambda/(\lambda+\sigma_i^2)$ is exactly a Gaussian-posterior variance reduction along each singular direction — the same term MacKay aggregates into $\gamma$, which we instead read per record as a test-time audit. Our contribution is not a new scalar on $(A,\sigma)$: on the linear-Gaussian layer these objects are settled, and any quantity defined there is a renaming of one of them. It is the *constructive converse*, the repurposing of the modal contraction as a ground-truth-free per-record certificate that separates quality from accountability, and the governed injection dial — an operational stack, unified for undersampled ghost imaging, that these prior framings describe in pieces but never assemble as a governance instrument. The one exactly-new formal object is a corollary we make explicit: the **no-adaptation lemma** (§7.3), that any ground-truth-free selection rule is constant on each measurement fiber, so per-image adaptation of null content is a global prior in disguise — a direct consequence of the converse that the identifiability literature implies but does not state for reconstruction.

### 10.9 Novelty statement

Every line above uses the range–null geometry, or a prior over the unmeasured content, to **reconstruct**: install the "correct" prior in the null space for fidelity (Wang et al., 2023 AAAI; RND-SCI; DDNM; MCG), learn to predict null content from $y$ (NPN; GSNR; Chen & Davies, 2020), fabricate null-like error to benchmark detectors (HalluGen), quantify output uncertainty against calibration data (RCPS; conformal risk control; Wen et al., 2024), or prove abstractly that the kernel forces hallucination (Iagaru et al., 2026; Bhadra et al., 2021; Antun et al., 2020). We use the identical geometry to **certify, bound, and meter**, and our contribution rests on four airtight, uncontested hooks. **First**, we make the barrier *constructive*: an explicit feasible-but-wrong witness that matches the same bucket record to machine precision — tighter than the noisy truth — turning "consistency is not correctness" from a theorem about maps (Iagaru et al., 2026) and an SVD diagnostic (Bhadra et al., 2021) into a per-instance object. **Second**, we give a ground-truth-free, reconstructor-agnostic **test-time certificate** that contracts each measured mode by *exactly* $\lambda/(\lambda+\sigma_i^2)$ — a closed-form modal spectrum, not a feasible-set-diameter bound (Iagaru et al., 2026), a soft penalty (NPN; GSNR), a design-time predictability curve (GSNR), or a calibrated statistical interval (Angelopoulos et al., 2022) — auditing BP, Tikhonov, CS-TV, and learned reconstructors uniformly and **separating image quality from measurement accountability**. **Third**, that separation is demonstrated as an explicit protocol (perceptual/PSNR quality essentially unchanged while measurement residuals fall orders of magnitude), an axis none of the reconstruction or UQ lines report. **Fourth**, we convert null-space injection from an all-or-nothing fill into a single-scalar, pre-registered **honesty dial** that fuses VQAE structure and VQGAN detail and keeps $A\hat{x}_B = y$ *exact* for every $B$ behind a locked gate — moving the field from *assessment* (does a reconstruction hallucinate?) to *governance* (how much invented detail do we add, where the bucket is provably blind, without ever breaking or laundering the measurement?). We explicitly do **not** claim SOTA, do not beat diffusion, and never claim the measurement certifies invented texture — the point is that it provably cannot. That inversion — **reconstruct, becomes certify / bound / meter** — is the slot this paper occupies.

---

## 11. Discussion, Scope, and Limitations

The preceding sections establish a converse (§3, §5), an achievability certificate (§4), and a governed injection dial (§6–§8). Each rests on a single geometric identity, $A P_0 = 0$, and each is stated at exactly the strength the evidence supports. This section collects the boundaries of those claims in one place. We treat these boundaries as load-bearing rather than as hedging: the same facts that limit the result are the facts that generate it. The converse is what licenses the dial; the null space that cannot be certified is the only place the dial is allowed to act; and the negative results below are what fix the scope within which the positive claims are exact.

### 11.1 Simulation-only, single measurement problem

All results are computational. We simulate the ghost-imaging forward map $y = Ax$ and evaluate reconstructions against known ground truth; no optical hardware, photon statistics, or physical bucket detector is involved. The range–null geometry of §2 is operator-agnostic and holds for any linear $A$ with $m < n$, but every quantitative claim — the exact per-mode contraction $\lambda/(\lambda+\sigma_i^2)$, the feasible-but-wrong witnesses, the locked fusion result — is instantiated on one class of undersampled ghost-imaging operators. We do not claim the numerical effect sizes transfer unchanged to a physical instrument, where detector noise, pattern miscalibration, and non-idealities in $A$ would perturb both the certificate and the anchor. Hardware validation is future work.

### 11.2 The locked achievability claim is one operating point

The confirmatory Act-2 result is deliberately narrow. It is locked at a single $5.0\%$ operator (1 DC $+$ 128 DCT $+$ 56 Hadamard $+$ 20 random rows, seed 772001) on a single resolution ($64\times64$, $n=4096$, $m=205$), with a single frozen balanced operating point ($B\approx0.55$), scored once on a brand-new, raw-hash-disjoint locked split. Within that locked scope the claim is exact and pre-registered: balanced fusion improves LPIPS by $-0.0977$ (CI $[-0.1016,-0.0940]$, a $32.6\%$ relative gain) at a bounded cost of $-0.45$ dB PSNR and $+0.0039$ RMSE, with RAPSD improving, $3/3$ seeds agreeing in direction, and $8/8$ gate conditions passing; development and locked effect sizes nearly coincide ($-0.0965$ vs. $-0.0977$).

Everything beyond that one point is **development-level** and is reported as such. The cross-rate study (a $-0.116$/$29.3\%$ LPIPS gain at $2\%$, $-0.098$/$32.6\%$ at $5\%$, $-0.076$/$34.2\%$ at $10\%$, $3/3$ seeds each) and the measurement-noise sweep (balanced fusion overtaking full VQGAN at $\sigma=0.02$, $0.197$ vs. $0.204$, and beating it at $\sigma=0.05$, $0.250$ vs. $0.293$) are exploratory: $B$ is selected on validation under the same tolerance rule but the locked split is not re-touched, no gate is fixed in advance, and no confirmatory claim is made. We have not locked any rate other than $5\%$, not locked the noisy-acquisition result, and not tested other operator designs or resolutions. The result does **not** establish that every low-rate task benefits.

### 11.3 The GAN branch is a representative prior, not a state-of-the-art claim

The detail branch uses a VQGAN because its adversarial patch objective supplies exactly the kind of unverifiable high-frequency null-space content the framework is built to govern. It is a controlled, representative instance of an adversarial prior under a measurement certificate — not a claim that this VQGAN, or GANs generally, are the best generative model for the problem. We do not compete on restoration quality and do not claim to beat diffusion. The paired-seed audit of the earlier gauge-GAN case study makes the modesty of the adversarial contribution explicit: the gauge branch minus its control is positive in PSNR across ten seeds but by only $+0.0148$ dB ($t$-interval $[+0.0107,+0.0189]$ dB), and a matched standard-cGAN control gives mixed, comparable D$-$C differences across seeds and metrics rather than a dominance result. The adversarial prior earns its place as a *governable source of detail*, not as a performance headline.

### 11.4 Abandoned negative result: posterior $z$-sampling diversity

We investigated whether stochastic $z$-sampling could turn the null-space prior into a calibrated posterior over $P_0 x$, and we report it only as a coverage limitation, not as a positive contribution. A deterministic checkpoint collapses: fixed-$y$ pixel std $\approx 10^{-3}$ and $P_0$ variance $1.09\times10^{-6}$ (gate 1). Removing the row-space reconstruction pressure lifts the collapse — the anti-collapse diagnostic reaches $P_0$ variance $1.28\times10^{-3}$ (gate 2) — confirming that the full reconstruction loss, not the architecture, was suppressing feasible null-space variation. But restored spread is not calibrated coverage. The best anchor/diversity scan still reaches only $\approx 45\%$ pixel and $\approx 48\%$ $P_0$ coverage at the nominal $90\%$ level, because the deterministic base estimate is offset from ground truth in the null space (base $P_0$ det-to-GT RMSE $\approx 0.081$): diversity can widen the sample cloud while leaving it centered in the wrong place. Sampling quality is therefore bounded by the base estimator's null-space accuracy, not only by the diversity mechanism. We make **no** posterior-diversity or calibrated-uncertainty claim; the line is retained solely to bound what the dial does and does not deliver.

### 11.5 The failure detector is weak, and the certificate does not localize error

Two negative results delimit what accountability can and cannot flag.

First, adversarial signal does not make a reliable failure detector. Under matched gauge-equalized evaluation the discriminator-based diagnostic is strong only where a genuine class-consistent signal exists (Rad-5 gauge AUC $0.877$, CI $[0.845,0.907]$; Scr-5 patch-GAN gauge $0.847$), and it degrades to weak, near-chance gates elsewhere — Rad-10 $0.640$ and Scr-10 $0.624$, both flagged "weak gate; stop" — with a shuffled-conditioning control at $0.664$. We therefore do not offer the adversarial branch as a general test-time failure detector; it is a bounded diagnostic, positive only in the regimes we report it in.

Second, the null-space magnitude $|P_0\hat{x}|$ is **not** a pixelwise error locator. It measures how much prior-supplied content a reconstruction carries, not where that content is wrong: its per-pixel correlation with the true null-space error is near zero, and in the gauge case study the top-decile-magnitude pixels have *lower*, not higher, error. $|P_0\hat{x}|$ is interpretable only as an image-level indicator of prior-supplied content, never as a per-pixel map of mistakes. This is a direct consequence of the converse — the unknown $P_0 x$ never enters the residual — and we state it wherever the map appears.

### 11.6 The fusion weight is a locked dial, not a per-image oracle

$B$ is selected once on a validation split and frozen per seed; it is neither adapted per image nor an oracle that knows the best weight for each scene. This is a deliberate honesty constraint, not an oversight. In development a $16$-band radial-frequency weighting and a learned per-image gate were evaluated under the same tolerance rule and did not beat the single global scalar: the band and cutoff variants collapsed onto the full-VQGAN endpoint, and only an unconstrained oracle that ignores the PSNR/RMSE tolerances preferred a low-pass-cutoff variant, and then only marginally. The gap between such a *ground-truth* oracle and any deployable rule is not a matter of better engineering: by the no-adaptation lemma (§7.3) every ground-truth-free selection rule is constant on each measurement fiber, so a deployable per-image weight can capture the oracle only through its dependence on $y$ — empirically a global shift worth $\le 0.002$ LPIPS beyond a matched-PSNR constant. Reporting the per-image oracle as an achievable operating point would therefore conflate "what the dial can achieve when locked" with "what an unconstrained search *with access to the truth* could find" — precisely the conflation the pre-registered protocol, and the lemma, exist to prevent.

### 11.7 The measurement never certifies invented texture

The single boundary that governs all the others: $A\hat{x}_B = y$ certifies that the reconstruction reproduces the recorded buckets, and nothing more. It does not certify that the injected VQGAN texture is the true scene. The measurement provides no evidence about the null-space component $P_0 x$, which is exactly where the injected detail lives, so that detail is a prior-driven hypothesis about plausible natural-image content, not a measurement-verified fact. A reconstruction that looks realistic — even one that lowers LPIPS, KID (from $0.119$ to $0.043$), and the RAPSD spectral distance — is not thereby certified to match the scene's true fine structure. Two fused reconstructions can differ almost entirely in $P_0$ and share the identical certified record; a feasible-but-wrong witness matches the same buckets to $\sim2\times10^{-15}$, tighter than the noisy truth itself. Consistency is not correctness. The certificate and the perceptual metric belong to two different accounts, and this paper never lets a gain on the second be read as a guarantee on the first.

---

## 12. Conclusion

A single linear-algebraic fact organizes everything above. For an undersampled sensing operator $A \in \mathbb{R}^{m\times n}$ with $m \ll n$, the orthogonal projectors $P_R = A^\dagger A$ and $P_0 = I - A^\dagger A$ split each image uniquely, and $A P_0 = 0$. The bucket record $y = A x$ therefore fixes exactly the measured component $P_R x$ and constrains nothing in the $n-m$ unmeasured coordinates. From this one geometry the paper derived a converse, an achievability result, and a governed construction, and the conclusion is best stated in those terms.

**Consistency is certifiable; correctness of unmeasured content is not.** The converse is exact and unforgiving: an explicitly constructed feasible-but-wrong witness $u$ reproduces the record to floating-point scale — $\mathrm{RelMeasErr}$ of $2.16\times10^{-15}$ to $4.00\times10^{-15}$ against $y$, tighter than the true image's own noisy residual of $3.45\times10^{-3}$ to $7.46\times10^{-3}$ — while remaining $7.70$ to $11.38$ dB from the target it impersonates. No test evaluated on $y$ alone can prefer the truth over such a witness, because both share the same $P_R$ and differ only in the null space where the measurement is silent. Against this barrier, the achievability result is equally exact: the ground-truth-free audit $\Pi_y^\lambda$ contracts each measured singular mode by precisely $\lambda/(\lambda+\sigma_i^2)$, confirmed in float64 to a maximum modal deviation of $1.04\times10^{-10}$ (Rad-5) and $2.29\times10^{-12}$ (Scr-5), and drives residuals down three to four orders of magnitude across post-hoc reconstructors while displacing image quality by $|\Delta\mathrm{PSNR}| \le 0.039$ dB on $18/18$ rows. What the record can be held to, and only that, is certifiable; the null space is provably not. Consistency is not correctness.

**Therefore inject prior detail safely and controllably.** The same $A P_0 = 0$ that forbids certifying null-space content guarantees that editing it cannot break the certificate. Anchoring at a measurement-audited LMMSE estimate $x_0$ with $A x_0 = y$, and fusing the null-space contributions of a reconstruction prior (VQAE) and an adversarial prior (VQGAN) through a single scalar $B$,
$$\hat{x}_B = x_0 + P_0\big(d_A + B(d_G - d_A)\big),$$
yields $A\hat{x}_B = y$ exactly for every $B$ — realized to a mean $\mathrm{RelMeasErr}$ of $3.6\times10^{-7}$ on the locked split. Detail injection becomes *safe* because it lives entirely where the bucket is blind, and *controllable* because a single pre-registered scalar traces a monotone perception–distortion ladder (LMMSE LPIPS $0.404 \to$ VQAE $0.300 \to$ balanced $0.202 \to$ quality-lite $0.182 \to$ full VQGAN $0.172$). At the locked balanced operating point the fusion improves LPIPS by $-0.0977$ (a $32.6\%$ relative gain, CI $[-0.1016, -0.0940]$) at a bounded cost of $-0.45$ dB PSNR, with distribution-level KID falling from $0.119$ to $0.043$, $3/3$ seeds agreeing in direction, and all $8/8$ pre-registered gate conditions passing on a brand-new, raw-hash-disjoint test split. The dial is bounded in cost and locked before the confirmatory run: a single scalar, not a per-image oracle.

**Certify what you measure, govern what you cannot.** The audit certifies the measured component and nothing beyond it; the fusion governs the unmeasured component openly, supplying prior detail exactly where no measurement can vouch for it, and never reading measurement consistency as evidence about the true scene. $A\hat{x}_B = y$ certifies that the reconstruction reproduces the buckets; it never certifies that the injected texture is real. The two operations occupy disjoint subspaces by construction, so certified accountability is never spent to license invented detail, and the perceptual gain never borrows the certificate's authority.

**Next steps.** Four extensions follow directly. First, hardware: the entire construction is simulation-only, and a physical ghost-imaging bench would test whether the row/null geometry and the exact-consistency guarantee survive real optical and detector nonidealities. Second, locking the development-level studies: the cross-sampling-rate advantage (a $29$–$34\%$ relative LPIPS gain at $2\%$, $5\%$, and $10\%$) and the graceful degradation under measurement noise are established only at development level and warrant their own pre-registered locked confirmations. Third, other operators and resolutions: the geometry is operator-agnostic, but the achievability and governance results are locked at one $5\%$ operator and $64\times64$ resolution, and should be re-established for other operator designs, sampling patterns, and image sizes. Fourth, calibrated null-space uncertainty: the audit certifies the measured component but issues no honest error bar on the unmeasured component, and $|P_0\hat{x}|$ is a prior-content map rather than a pixelwise error locator. A calibrated posterior over the null space — one that reports how far the injected detail may be from the truth without claiming to certify it — is the principled way to turn the honest non-claim of this paper into a usable uncertainty statement.
