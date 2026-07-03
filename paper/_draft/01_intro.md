# 1. Introduction

A learned reconstruction of an undersampled scene can reproduce every recorded measurement to numerical precision and still be semantically wrong. Consider two images that agree on the measured coordinates but differ arbitrarily elsewhere: they produce the *identical* measurement record, yet at most one of them is the scene that was imaged. In low-rate ghost imaging this is not a pathological corner case but the generic situation. A single-pixel detector acquires a small number of correlated bucket values, $y = A x$ with $A \in \mathbb{R}^{m \times n}$ and $m \ll n$; here $n = 4096$ ($64\times64$ pixels) and $m = 205$, a 5.0% sampling rate. The measurement therefore pins down $205$ numbers, while the image carries $4096$, and modern generative priors are perfectly capable of filling the remaining $3891$ degrees of freedom with plausible, sharp, natural-looking detail. The scientific question a reconstruction pipeline must answer is not merely whether the output looks good. It is: **which content is measured, and which is invented?**

Standard image-quality metrics cannot answer this. PSNR and SSIM require a ground-truth image, which is unavailable in deployment, and they conflate measured error with unmeasured error. A high-PSNR reconstruction can carry a large measurement residual, and a measurement-consistent reconstruction can be arbitrarily wrong in its unmeasured component. The gap is not a deficiency of any particular metric or algorithm; it is a property of the sensing operator.

## The governing fact

The entire paper is a consequence of one elementary identity. Let $A^\dagger$ be the Moore–Penrose pseudoinverse and split $\mathbb{R}^n$ into the orthogonal row-space and null-space of $A$ through the projectors $P_R = A^\dagger A$ and $P_0 = I - A^\dagger A$, so that $x = P_R x + P_0 x$ uniquely. Then

$$A P_0 = A\big(I - A^\dagger A\big) = A - (A A^\dagger A) = A - A = 0.$$

Consequently $A x = A P_R x$: the measurement record is a function of $P_R x$ **only**, and $P_0 x$ is exactly invisible to the bucket. This single fact is elementary — it is textbook linear algebra of a rank-deficient operator, and we claim no novelty in the identity itself. The contribution is what follows from taking it seriously: the empirical sharpness of the barrier it imposes, and a complete rational response to that barrier.

## The deductive chain

We organize the paper as an impossibility result and its disciplined consequences — a converse, an achievability, and a governed use — each a corollary of $A P_0 = 0$.

**Converse (the barrier).** Because $y$ fixes exactly $P_R x$ and nothing else, null-space content is *provably* unverifiable from the measurement. We make this constructive rather than abstract: for any two images $x_i, x_j$ the spliced image $u_{ij} = x_j - A^\dagger(A x_j - y_i)$ carries the measured component of $x_i$ but the null-space content of $x_j$, and satisfies the record of $x_i$ to floating-point precision. Empirically these feasible-but-wrong witnesses reach $\mathrm{RelMeasErr} \sim 2\times10^{-15}$ (16/16 constructed pairs, range $2.16\times10^{-15}$ to $4.00\times10^{-15}$) — *tighter* than the noisy truth itself (residual $3.45\times10^{-3}$ to $7.46\times10^{-3}$), while remaining semantically wrong (PSNR-to-target $7.70$ to $11.38$ dB). Consistency is not correctness.

**Achievability (the certificate).** The same geometry that forbids certifying $P_0 x$ *does* permit a ground-truth-free, test-time audit of the measured component. Attaching $\Pi_y^\lambda(v) = v - A^\top(A A^\top + \lambda I)^{-1}(A v - y)$ to any reconstructor contracts each measured singular mode by exactly $\lambda/(\lambda + \sigma_i^2)$ and leaves $P_0 v$ untouched. This audit reduces the measurement residual by three to four orders of magnitude for backprojection, Tikhonov, CS-TV, and learned outputs while moving PSNR by at most $0.039$ dB (18/18 audited rows), cleanly separating image quality from measurement accountability. It exactly matches the closed-form contraction in float64 arithmetic (deviation $\sim 10^{-10}$ to $10^{-12}$).

**Governed use (the dial).** Since no measurement can certify the null space, the honest response is not to hide prior-supplied content but to inject it precisely where it is provably invisible to the bucket, and to meter it. Anchoring at a measurement-audited LMMSE estimate $x_0$ (so $A x_0 = y$) and fusing the null-space differences of a reconstruction prior (VQAE, contribution $d_A$) and an adversarial prior (VQGAN, contribution $d_G$) through a single scalar $B$,

$$\hat{x}_B = x_0 + P_0\big(d_A + B(d_G - d_A)\big),$$

gives $A \hat{x}_B = y$ *exactly* for every $B$ (locked $\mathrm{RelMeasErr}$ mean $3.6\times10^{-7}$). At the balanced operating point ($B \approx 0.55$) this improves LPIPS by $-0.0977$ (CI $[-0.1016, -0.0940]$; a 32.6% relative gain) over the VQAE branch at a bounded cost of $-0.45$ dB PSNR, with 3/3 seeds agreeing and an 8/8 pre-registered acceptance gate passing on a raw-hash-disjoint locked test split.

## Figure 1

Figure 1 renders this chain as three panels — *cannot / can / therefore*. The left panel (**cannot**) shows a feasible-but-wrong pair: a true target and its spliced counterpart, sharing the measured row space and differing only in the null space, whose bucket residual ($2.93\times10^{-15}$) is smaller than the truth's own noisy residual ($5.36\times10^{-3}$) — a measurement-consistent, semantically wrong, non-natural image. The middle panel (**can**) shows the exact per-mode certificate: the audit contracting each measured singular mode by $\lambda/(\lambda+\sigma_i^2)$, driving residuals to the noise floor while PSNR barely moves. The right panel (**therefore**) shows metered injection at $B = 0.55$, with the fused reconstruction stamped $A\hat{x} = y$ — invented detail placed only where the bucket is blind, without ever breaking the measurement.

## Contributions

1. **A constructive converse.** We turn "consistency is not correctness" from an abstract statement into a per-instance, feasible-but-wrong witness that matches the recorded bucket to $\sim 2\times10^{-15}$ — tighter than the noisy truth — establishing empirically that null-space content is unverifiable.
2. **A ground-truth-free measurement certificate.** A plug-in test-time audit $\Pi_y^\lambda$ that contracts each measured mode by exactly $\lambda/(\lambda + \sigma_i^2)$, applies uniformly across BP, Tikhonov, CS-TV, and learned reconstructors, and separates image quality from measurement accountability (residuals drop orders of magnitude at $|\Delta\mathrm{PSNR}| \le 0.039$ dB).
3. **A governed, measurement-safe injection dial.** A single scalar $B$ fusing VQAE structure and VQGAN detail strictly inside $P_0$, keeping $A\hat{x}_B = y$ exact for every $B$, delivering a locked balanced LPIPS gain of $-0.0977$ (32.6%; KID $0.119 \to 0.043$) behind a pre-registered gate, and tracing a monotone perception–distortion ladder (LMMSE $0.404 \to$ VQAE $0.300 \to$ balanced $0.202 \to$ quality-lite $0.182 \to$ VQGAN $0.172$).

## Non-claims

We state the boundaries of the work up front and honor them throughout. This is **not** a state-of-the-art result; we do **not** claim to beat diffusion-based solvers, and we report **no** hardware experiments. The fusion weight $B$ is a fixed, validation-selected operating point, **not** a per-image oracle. Posterior null-space $z$-diversity was pursued and is reported only as an abandoned negative result, never as a positive claim. The cross-rate (2%, 10%) and measurement-noise studies are **development-level** evidence, not locked claims. The observable magnitude $|P_0 \hat{x}|$ is **not** a pixelwise error locator. And most importantly: measurement consistency means $A\hat{x} = y$ and nothing more — the bucket **never** certifies that the injected texture is the true scene.
