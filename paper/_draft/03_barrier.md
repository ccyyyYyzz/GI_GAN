## 3. The Verifiability Barrier (Converse)

The geometry of Section 2 is deductive but silent about severity: the lemma $A P_0 = 0$ shows that $y$ fixes exactly $P_R x$ and constrains nothing in the null space, but it does not by itself say how badly a null-space error can hide behind a clean measurement. This section supplies the converse. We construct, for any pair of images, an explicit reconstruction that reproduces a target's bucket record to machine precision while carrying an entirely different scene's unmeasured content. The construction is per-instance and exact, not asymptotic or distributional: it exhibits a concrete feasible-but-wrong witness for every cross-class pair we test. Its consequence is the load-bearing claim of the paper — measurement consistency is certifiable but the correctness of null-space content is not, so **consistency is not correctness** — and it is placed first, before any certificate or injection dial, because everything downstream is subordinate to it.

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

The witness is not merely feasible in principle; numerically it satisfies the record more tightly than the true scene does. On the Rad-5 operator ($m = 205$, $96\times 96$), across a set of $8$ cross-class feasible pairs, the relative measurement error of the witness against the target record lies in the range $\mathrm{RelMeasErr}(u_{ij}, y_i) \in [2.16\times10^{-15},\, 4.00\times10^{-15}]$, i.e. at floating-point round-off. Over the paired families the construction holds for $16/16$ cross-class pairs at the same $\sim\!2\times10^{-15}$ scale. The *true* image, by contrast, only satisfies the (noisy) record to $[3.45\times10^{-3},\, 7.46\times10^{-3}]$. The gap is roughly twelve orders of magnitude: **the wrong image reproduces the buckets more exactly than the ground truth.** A featured car-versus-horse pair makes the point concretely — the target car has $\mathrm{RelMeasErr}(x_i, y_i) = 5.36\times10^{-3}$, while the constructed witness follows the horse donor visually and yet attains $\mathrm{RelMeasErr}(u_{ij}, y_i) = 2.93\times10^{-15}$. The witnesses remain semantically wrong with respect to the target, with $\mathrm{PSNR}(u_{ij}, x_i)$ in the range $7.70$–$11.38$ dB.

The immediate corollary is that no residual-based test, however tight its threshold, can separate the witness from the truth: any threshold that admits the true image (residual $\sim\!10^{-3}$) admits the witness (residual $\sim\!10^{-15}$) a fortiori. This is precisely the failure mode PULSE exhibits for super-resolution (Menon et al. 2020) — visually distinct realistic reconstructions collapsing to the same low-dimensional observation — but here it is exact, per-instance, and derived from the operator's own geometry rather than sampled from a generator.

### 3.3 Position relative to the limits literature

The existence of consistency-preserving hallucinations, and the impossibility of assessing them without ground truth, is established prior art. Iagaru et al. (2026) prove necessary-and-sufficient conditions for detail-transfer hallucination in (possibly nonlinear) inverse problems — consistent decoders can only hallucinate details almost invisible in measurement space — and give forward-model-only, ground-truth-free algorithms that *bound* hallucination magnitude via feasible-set diameters. Gottschling et al. (2025) situate the same phenomenon within the broader account of instabilities and fundamental limits of learned reconstruction, and Bhadra et al. (2021) formalize null-space hallucinations by SVD projection, showing they are attributable solely to the prior and cannot be assessed without the truth. We take these as the anchor for the barrier itself and do not reclaim their ground: that null content is unverifiable, and that assessment must be ground-truth-free, is theirs.

Our delta is constructive and per-instance. Where the limits line establishes *existence* through necessary-and-sufficient conditions, or supplies *bounds* over feasible sets (a supremum that requires paired data to evaluate), $u_{ij}$ is an explicit machine-precision witness for each individual instance — an object one can render, measure, and display, matching the record tighter than the noisy truth. It converts "consistency is not correctness" from a theorem about maps and an SVD diagnostic into a concrete pixelwise construction. This is the engine on which the rest of the paper turns: because the barrier is real and exact, the certificate of Section 4 can only ever certify the row space, and the detail injection of Section 6 is licensed to act only where the barrier proves the measurement is blind.

### 3.4 The unmeasured-content magnitude is not an error map

A natural but mistaken hope is that the observable magnitude $|P_0 \hat{x}|$ — how much unmeasured content a reconstruction carries, pixel by pixel — could serve as a self-contained error map, flagging where a reconstruction has invented detail. It cannot, and the same decomposition explains why. For any audited reconstruction $\hat{x}$ the null-space error is

$$P_0(\hat{x} - x) = P_0 \hat{x} - P_0 x,$$

and the observable term $P_0 \hat{x}$ contains no direct information about the unknown truth term $P_0 x$. The magnitude is a property of the reconstruction, not a certificate of its error: a large value may be accurate texture, inaccurate texture, or a harmless choice among feasible completions. We therefore call $|P_0 \hat{x}|$ a **prior-supplied-content map** — it marks where a prior placed unmeasured content, not where the reconstruction is wrong.

The decoupling is empirical, not merely formal. On the $96\times 96$ Rad-5 validation, per-pixel $|P_0 \hat{x}|$ and actual null-space error are essentially uncorrelated: for the LMMSE arm the Spearman and Pearson correlations are $-0.079$ and $-0.069$, and for the two learned arms they are $\approx 0.07$ and $\approx 0.06$. The ordering is if anything inverted for the learned arms — the top-$10\%$ highest-$|P_0 \hat{x}|$ pixels carry *lower* actual null-space error ($0.071$) than the remaining $90\%$ ($0.079$). The map therefore has, at most, image-level diagnostic value (an aggregate indication that a reconstruction leans on its prior); it is not a pixelwise error locator, and we do not use it as one anywhere in this paper. This negative result closes the section on the same theme it opened: the null space is where the prior speaks, and nothing the bucket records — not the residual, not the magnitude of the supplied content — can adjudicate whether that speech is true.
