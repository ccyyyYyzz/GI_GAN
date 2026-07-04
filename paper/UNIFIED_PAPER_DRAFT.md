# Certify What You Measure, Govern What You Cannot: A Range–Null Account of Accountability and Governed Detail in Undersampled Ghost Imaging

*Author placeholder — Author One, Author Two, Author Three. Affiliation placeholder. Corresponding author: [email placeholder].*

---

## Abstract

In low-rate ghost imaging, measurement consistency is not evidence of image correctness. The reason is geometric: the record $y = Ax + \varepsilon$ fixes only the row-space component $P_R x$, while $P_0 x \in \ker A$ is invisible to the bucket. We turn this fact into an accountability framework. First, we construct feasible-but-wrong witnesses that match a target record to floating-point precision, around $10^{-15}$ — more tightly than the noisy truth matches its own record, around $10^{-3}$ — while remaining semantically far from the target. Second, we give a test-time row-space audit: applied after any reconstructor, it contracts the $i$-th measured residual mode by exactly $\lambda/(\lambda+\sigma_i^2)$, certifying agreement with the recorded bucket without ground truth. Third, we use the same boundary as a governance rule: prior detail is injected only through $P_0$, so $A\hat{x}_B = y$ holds for every dial setting $B$. On locked $64\times64$ STL10 ghost imaging at 5% sampling, the governed dial reduces LPIPS by 32.6% at a 0.45 dB PSNR cost while preserving exact measurement consistency, and a projector audit of three published deep-learning pipelines on their own operators finds the same structure. The result is a two-ledger account: certify what the bucket measured; label and govern what the prior invented.

---

## 1. Introduction

At 5% sampling, a $64\times64$ ghost-imaging system records 205 bucket values for a 4096-pixel scene. The missing 3891 coordinates are not merely difficult to recover; they are invisible to the measurement operator. A learned reconstructor can therefore produce a sharp, natural-looking image, reproduce every recorded bucket, and still invent most of the visible detail. This paper asks how such a reconstruction should be audited: what did the bucket actually certify, and what did the prior merely supply?

The cause is one identity. With $A^\dagger$ the pseudoinverse and $P_R = A^\dagger A$, $P_0 = I - A^\dagger A$ the projectors onto the row space and null space of $A$,

$$A P_0 = 0,$$

so the record $y = Ax + \varepsilon$ is a function of $P_R x$ alone. Two images that agree on the measured component produce the identical record no matter how much they differ elsewhere. Every quality metric integrates both components; the measurement sees one. Image quality and measurement accountability are therefore different quantities, and a pipeline can score well on the first while saying nothing about the second.

The paper has one geometry and three consequences. Because $A P_0 = 0$, the bucket cannot distinguish images that share a row-space component; we make this failure concrete with feasible-but-wrong witnesses that match a target's record around $10^{-15}$ — more tightly than the noisy truth matches its own record (§2). The same geometry gives a positive result: the measured modes can be audited exactly, with a known contraction factor per mode, after any reconstructor (§3). And since the null space cannot be certified, we do not hide prior detail there; we govern it with a scalar dial that changes only $P_0$ and preserves the recorded bucket exactly (§4). The locked confirmation of the dial, its limits, and a projector audit of three published pipelines on their own operators complete the account (§4–5). The result is a two-ledger reading of low-rate ghost imaging: the **row ledger** is measured and certifiable; the **null ledger** is prior-supplied and must be governed rather than laundered as measurement evidence.

[FIG: METHOD_FIG1.pdf | 1.0 | **The two-ledger argument.** Left (*cannot*): a feasible-but-wrong image matches the target bucket record to numerical precision, so consistency cannot certify null content. Middle (*can*): the audit contracts only measured modes, with factor $\lambda/(\lambda+\sigma_i^2)$. Right (*therefore*): the dial injects prior detail only through $P_0$, preserving $A\hat{x}_B = y$ while moving along the perception–distortion curve.]

**Contributions.**

1. **Constructive impossibility.** For any target record and donor image, we construct a feasible-but-wrong witness that preserves the target bucket record while importing the donor's null-space content (§2).
2. **Exact row-space certificate.** A post-hoc audit whose residual contraction is exactly $\lambda/(\lambda+\sigma_i^2)$ in each measured singular mode, independent of the reconstructor (§3).
3. **Governed null-space supply.** A scalar dial that fuses prior detail strictly inside $\ker A$, preserving $A\hat{x}_B = y$ for every setting and improving locked LPIPS by 32.6% at a bounded distortion cost (§4).
4. **Limits and external validity.** A no-adaptation lemma for ground-truth-free dial selection, and a projector audit of three published DL-GI pipelines on their own released or measured operators (§4.5, §5).

> **Scope of all certificates.** In this paper, "certified" always means certified agreement with the recorded bucket $y$. It never means certified agreement with the unknown scene. All content in $P_0$ is prior-supplied unless additional measurements are acquired. Claims are tiered: *locked* results follow a pre-registered one-shot protocol; *development-level* results reuse the frozen system without a pre-registered gate. This box governs every section below.

The study is simulation-based, on $64\times64$ grayscale STL10 with a fixed 5% operator for the locked claim; hardware validation is future work (§7).

---

## 2. Geometry and Constructive Impossibility

**Setup.** The forward model is $y = Ax + \varepsilon$ with $x \in \mathbb{R}^n$, $y \in \mathbb{R}^m$, $m \ll n$; in the witness and audit experiments $\varepsilon$ is i.i.d. Gaussian bucket noise with $\sigma_\varepsilon = 0.01$. The relative measurement error of a candidate $v$ against a record $y$ is $\mathrm{RelMeasErr}(v;y) = \lVert Av - y\rVert_2/\lVert y\rVert_2$.

**Theorem 1 (measurement geometry).** *Let $A = U_r \Sigma_r V_r^\top$ be the compact SVD and $P_R = A^\dagger A = V_r V_r^\top$, $P_0 = I - P_R$. Then $P_R, P_0$ are orthogonal projectors splitting $\mathbb{R}^n = \mathcal{R}(A^\top) \oplus \mathcal{N}(A)$, and*

$$A P_0 = 0, \qquad A P_R = A.$$

*Consequently $Ax = A P_R x$: the record is a function of the measured component alone, and the noiseless feasible set is the affine flat $\{x: Ax = y\} = A^\dagger y + \mathcal{N}(A)$.* (Proof and the rows-orthonormal special case: Appendix A.)

In the locked setting — $n = 4096$, $m = 205$ — the bucket certifies 205 coordinates and leaves 3891 to a prior. Most of what the eye reads as detail lives in those 3891 directions.

### 2.1 The witness construction

Fix a target $x_i$ with recorded buckets $y_i$ and a donor $x_j$ from a different semantic class. Define

$$u_{ij} = x_j - A^\dagger\big(A x_j - y_i\big).$$

Because $A$ has full row rank, $A A^\dagger = I_m$, so $A u_{ij} = y_i$ for any record: the witness is exactly measurement-consistent with the target. Its two components separate cleanly — $P_R u_{ij} = A^\dagger y_i$ (pinned to the record) while $P_0 u_{ij} = P_0 x_j$ (inherited whole from the donor, since $P_0 A^\dagger = 0$). The witness splices the target's measured part onto the donor's unmeasured part. Figure 2 draws the construction.

[FIG: WITNESS_GEOMETRY.pdf | 0.8 | **The converse in one picture.** Every point on the line (the fiber $\{x: Ax = y_i\}$) reproduces the record; moving off the line is the only thing the measurement can see. The witness is the donor projected onto the fiber: it matches the record to $10^{-15}$ — more tightly than the noisy truth (dashed circle, radius $\sim\sigma_\varepsilon$) matches its own record — while carrying the donor's null content.]

### 2.2 A wrong image passes the measurement test more tightly than the truth

In our 5% instance the constructed witnesses match the target record at $10^{-15}$ to $10^{-13}$ relative error, while the noisy truth matches its own record only at $10^{-3}$. The witnesses remain visually and metrically wrong. Therefore no residual threshold that accepts the truth can reject the witness.

The numbers behind this statement: on the Rademacher operator, all 16 constructed cross-class pairs land in $[2.16, 4.00]\times10^{-15}$ against truth residuals of $[3.45, 7.46]\times10^{-3}$, with witness-to-target PSNR of only $7.7$–$11.4$ dB; a featured car-versus-horse pair reads $2.93\times10^{-15}$ against the truth's $5.36\times10^{-3}$. On the same mixed-basis operator that carries the dial of §4, 40 witnesses reproduce their records at a median of $9.9\times10^{-14}$ while remaining semantically wrong (median PSNR $19.5$ dB). The two numbers measure different things: the witness's $10^{-15}$ is geometry (it is constructed on the record), while the truth's $10^{-3}$ is its own bucket noise. Operator-level details are in Appendix B.

### 2.3 The observable null magnitude is a content map, not an error map

For any reconstruction, $P_0(\hat{x} - x) = P_0\hat{x} - P_0 x$, and the record carries no information about the second term. The observable magnitude $|P_0\hat{x}|$ therefore marks where a prior placed unmeasured content, not where the reconstruction is wrong; empirically its per-pixel correlation with actual null error is near zero (Appendix F). We use it as a provenance map and never as an error locator.

---

## 3. Certifying the Measured Ledger

The null ledger cannot be certified; the row ledger can, exactly. This section gives the audit and what it buys.

### 3.1 The audit operator

For any reconstruction $\hat{x}$, define

$$\Pi_y^\lambda(v) = v - G_\lambda\,(Av - y), \qquad G_\lambda = A^\top(AA^\top + \lambda I)^{-1}, \quad \lambda > 0.$$

The audit needs only $(A, y, \lambda)$ — no ground truth, no access to the reconstructor. Its correction lies in $\mathcal{R}(A^\top)$, so $P_0\,\Pi_y^\lambda(v) = P_0 v$: whatever a prior placed in the null space passes through untouched. The audit operates strictly inside the subspace the measurement is accountable for.

### 3.2 Exact modal contraction

Applying $A$ and diagonalizing $AA^\top$, each measured mode's residual is scaled independently:

$$c_i(\lambda) = \frac{\lambda}{\lambda + \sigma_i^2}.$$

The contraction is exact and image-independent. In float64 the measured contraction matches this formula to $1.04\times10^{-10}$ (Rademacher operator) and $2.29\times10^{-12}$ (scrambled Hadamard). As $\lambda \to 0$ the audit becomes a hard projection onto $\{Av = y\}$. The certificate is evaluated against the recorded noisy bucket $y$, not the noiseless $Ax$: a small residual certifies record reproduction. The factor $c_i$ is the complement of the classical Tikhonov filter factor (Engl, Hanke & Neubauer 1996; Hansen 1998) and, summed over modes, MacKay's (1992) effective-parameter count; the delta here is the role — a per-record, post-hoc, reconstructor-agnostic audit paired with the converse of §2 (positioning: §6, Appendix D).

[FIG: CERT_SEPARATION.pdf | 1.0 | **The certificate and the separation.** *(a)* Each measured mode contracts by the closed-form factor $\lambda/(\lambda+\sigma_i^2)$. *(b)* Applied post-hoc to four unrelated reconstructors, the audit moves measurement accountability by orders of magnitude while PSNR barely moves: the two axes decouple because $AP_0 = 0$.]

### 3.3 Quality and accountability separate

Decompose any error $e = \hat{x} - x$ as $P_R e + P_0 e$; then $\|e\|^2 = \|P_R e\|^2 + \|P_0 e\|^2$ and the audit residual depends on $P_R e$ alone. Define the row-error share $s = \|P_R e\|^2/\|e\|^2$.

**Theorem 2 (PSNR ceiling for row-space corrections).** *Any correction $T$ with $T(v) - v \in \mathcal{R}(A^\top)$ raises PSNR by at most $-10\log_{10}(1-s)$; the bound is attained by the hard projection under noiseless $y$ and not attained under noise. Corrections that edit the null space are outside the hypothesis — which is the content: only null-space injection can exceed the row budget, and the measurement certifies none of it.*

Trained networks sit near $s \approx 0$: re-imposing consistency is nearly free in PSNR and decisive in accountability. The audited learned reconstructor moves from $\mathrm{RelMeasErr} = 3.68\times10^{-2}$ to $1.90\times10^{-6}$ while PSNR changes by $0.014$ dB; across all 18 audited rows the residual falls three to four orders at $|\Delta\mathrm{PSNR}| \le 0.039$ dB. The image-level counterpart of $s$ is the row-energy share $\rho = \|P_R x\|^2/\|x\|^2$, which tracks the sampling rate almost exactly ($\rho \approx 0.05$ at 5%, $0.10$ at 10%): the operator decides how much of a scene is measurable, and the prior owns the rest.

### 3.4 What the audit catches that quality metrics cannot

The decisive probe is operator drift. Auditing with a miscalibrated operator and evaluating against the true one, PSNR moves by $0.03$ dB while the true-operator residual rises from $1.90\times10^{-6}$ to $4.88\times10^{-2}$ — four orders of magnitude. A quality-only pipeline reports success; the audit reports the mismatch. (Wrong-record and coordinate-shuffle probes confirm the trained reconstructor genuinely conditions on $y$; details in Appendix B.)

---

## 4. The Governed Null-Space Dial

The dial is not a fusion architecture. It is a provenance constraint: whatever prior is used, its contribution is projected into $\ker A$, so perceptual detail is supplied without borrowing measurement authority. We call the resulting control *governed* — and, once, *honest* — because it changes only what the measurement cannot certify and reports that change as prior-supplied detail.

### 4.1 Construction

We instantiate the rule with two matched vector-quantized priors per seed: a reconstruction-trained autoencoder (VQAE, faithful but flat, contribution $d_A$) and an architecturally identical adversarially-trained prior (VQGAN, sharp detail, contribution $d_G$), both taken relative to a measurement-audited LMMSE anchor $x_0$ with $Ax_0 = y$:

$$d_A = P_0(x_A - x_0), \qquad d_G = P_0(x_G - x_0), \qquad \hat{x}_B = x_0 + P_0\big(d_A + B\,(d_G - d_A)\big).$$

**Theorem 3 (exact consistency for every $B$).** $A\hat{x}_B = A x_0 + A P_0(\cdot) = y$ *for all* $B \in \mathbb{R}$. The identity that makes null content unverifiable (§2) is the identity that makes injecting it measurement-safe. Numerically the locked splits hold $\mathrm{RelMeasErr}$ at $3.6\times10^{-7}$ (float32 pipeline scale) for every operating point.

[FIG: METHOD_DIAGRAM_3D.pdf | 0.92 | **The mechanism.** All measurement-consistent images lie on the flat $x_0 + \mathcal{N}(A)$; the dial slides the reconstruction along the flat from the VQAE point ($B{=}0$) to the VQGAN point ($B{=}1$). Every setting reproduces the record exactly; the true scene is also on the flat, at an unknowable null-space location.]

### 4.2 The perception–distortion ladder

Sweeping $B$ traces a monotone ladder on the perception–distortion plane (Blau & Michaeli 2018):

**Table 1. Locked-split means (512 images, 3 seeds).**

| Operating point | LPIPS $\downarrow$ | PSNR (dB) $\uparrow$ | RMSE $\downarrow$ | SSIM $\uparrow$ | RAPSD $\downarrow$ |
|---|---|---|---|---|---|
| LMMSE anchor $x_0$ | 0.404 | 22.80 | 0.076 | 0.629 | 0.0041 |
| VQAE ($B=0$) | 0.300 | 23.13 | 0.073 | 0.657 | 0.0030 |
| **Balanced** ($B\!\approx\!0.55$) | **0.202** | 22.69 | 0.077 | 0.633 | 0.0027 |
| Quality-lite ($B\!\approx\!0.72$) | 0.182 | 22.21 | 0.081 | 0.609 | 0.0024 |
| Full VQGAN ($B=1$) | 0.172 | 21.43 | 0.089 | 0.571 | 0.0019 |

From VQAE onward every column is monotone: each increment of $B$ trades pixel fidelity for perceptual and spectral realism at fixed measurement fidelity. The balanced point captures roughly three-quarters of the available LPIPS range while holding PSNR within half a decibel of VQAE. A dense 21-point sweep is smooth, with strictly monotone LPIPS and no interior perceptual optimum (Appendix C).

[FIG: PARETO_FIGURE.pdf | 0.9 | **The ladder on the perception–distortion plane.** Named operating points of Table 1; every point satisfies $A\hat{x} = y$, so the trade moves entirely within the null ledger.]

### 4.3 Locked confirmation

> **Locked protocol.** Dataset: STL10 grayscale $64\times64$; operator: 5% ($m=205$), fixed, seed 772001. Splits by raw-SHA256 deduplication: train 20,000 / validation 512 (selects $B$) / development 512 / locked 512, raw-hash disjoint from all previously consumed hashes. $B$ frozen per seed $\{0.55, 0.55, 0.50\}$ before the locked read. Gate (8 pre-registered conditions): relative LPIPS gain $\ge 5\%$; LPIPS CI $<0$; seed agreement; PSNR within $0.5$ dB; RMSE within $0.005$; RAPSD not worse; $\mathrm{RelMeasErr} \le 10^{-5}$; hash-disjointness audit. Evaluation: one shot. Full details: Appendix C.

The gate passes 8/8. Balanced fusion improves LPIPS by $-0.0977$ (CI $[-0.1016, -0.0940]$; $32.6\%$) at $-0.45$ dB PSNR and $+0.0039$ RMSE, with spectral realism improving and all three seeds agreeing; $A\hat{x}_B = y$ holds at numerical precision throughout. The development split, scored before the locked run, gives $-0.0965$ ($32.9\%$) — the effect is stable, not selected. Post-hoc, distribution-level KID falls from $0.119$ to $0.043$.

[FIG: QUALITATIVE_GRID.pdf | 1.0 | **Locked-split reconstructions across the ladder.** Moving right adds perceptual detail; every column reproduces the identical bucket record. The added detail lives in the null ledger.]

### 4.4 Robustness of the operating point

The advantage travels. At 2%, 5%, and 10% sampling the balanced dial improves LPIPS by $29$–$34\%$ relative to VQAE with all seeds agreeing (development-level; Table 2). Under bucket noise the balanced point degrades gracefully and overtakes full VQGAN at $\sigma_\varepsilon = 0.02$ — the recommended setting is also the robust one (Table 3). The gain is broad: LPIPS improves on $99.0\%$, $97.5\%$, and $99.2\%$ of locked images across seeds, and the rare regressions concentrate on man-made periodic structure where a natural-image prior is a known mismatch.

**Table 2. Cross-rate generalization (development-level, 3 seeds/rate).**

| Rate | $\Delta$LPIPS vs VQAE | Relative | $\Delta$PSNR (dB) |
|---|---|---|---|
| 2% ($m=82$) | $-0.116$ | $29.3\%$ | $-0.39$ |
| 5% ($m=205$, locked) | $-0.098$ | $32.6\%$ | $-0.45$ |
| 10% ($m=410$) | $-0.076$ | $34.2\%$ | $-0.43$ |

**Table 3. LPIPS under bucket noise (development-level, $B$ frozen).**

| $\sigma_\varepsilon$ | VQAE | Balanced | Full VQGAN |
|---|---|---|---|
| 0.00 | 0.300 | 0.202 | 0.172 |
| 0.01 | 0.297 | 0.195 | 0.176 |
| 0.02 | 0.295 | **0.197** | 0.204 |
| 0.05 | 0.304 | **0.250** | 0.293 |

### 4.5 Why the dial is one scalar: the no-adaptation lemma

**Lemma.** *Any selection rule $g(A, y)$ computable from the record alone is constant on each fiber $\{x: Ax = y\}$; the closable part of the per-image oracle gap is limited to the $y$-predictable component of the oracle weight.* (Proof: two fiber-mates share $y$, hence share $g$; the oracle depends on the unobserved $P_0 x$. $\blacksquare$)

The lemma is elementary; its value is the pairing with measurement. Empirically the $y$-predictable component is worth essentially nothing here [development-level]: a feature-based per-image selector recovers the oracle only through a global shift of $B$, with excess $\le 0.002$ LPIPS over a matched-PSNR constant and feature–oracle correlations $|\rho| \le 0.24$. Richer parameterizations (16-band frequency weights, a learned gate) also lose to the scalar under the registered rule. Frameworks that do recover null information import structure beyond one record — group equivariance or operator diversity (Chen et al. 2021; Tachella et al. 2023). Held-out measurements can likewise break the fiber and estimate null error directly; that is classical compressed-sensing cross-validation (Ward 2009), not a contribution of this paper.

---

## 5. External Check: Projector Forensics of Published Pipelines

Does the two-ledger structure describe the field or only our construction? We audited three published DL-GI pipelines — pretrained+fine-tuned (Wang et al. 2022a), untrained DIP on real measured speckle (GIDC; Wang et al. 2022b), and self-supervised on a genuinely noisy record (Noise2Ghost; Manni et al. 2025) — using **each paper's own released or measured operator and its own code**, with exact orthogonal attribution of the error.

**Table 4. Forensics summary (own operator, own code).**

| Pipeline | Headline vs own range ceiling | Gain split row/null | Terminal null share |
|---|---|---|---|
| PEDL (fine-tuned) | $25.1$ vs $17.0$ dB ($+8.1$) | $61/39$ | $96.9\%$ |
| GIDC (untrained; known-GT scene) | $19.5$ vs $15.0$ dB ($+4.5$) | $60/40$ | $95.4\%$ |
| Noise2Ghost (self-supervised) | $18.4$ vs $15.3$ dB ($+3.1$) | $5/95$ | $92.7\%$ |

[FIG: FORENSICS_CROSS_TARGET.pdf | 1.0 | **Three paradigms, three operators, one structure.** *(a)* The fine-tuning trajectory decomposed per step: PSNR rises past the operator's range ceiling — everything above it is paid from the null ledger. *(b)* Attribution of each method's improvement: row repair vs null injection. *(c)* Every headline stands $+3.1$ to $+8.1$ dB above its own range ceiling with terminal error $93$–$97\%$ null.]

In all three, row-space gains saturate — the methods repair the measured ledger, then stall exactly when it is exhausted — and headline quality above the range ceiling is paid from the null ledger. The direction of this finding is forced by the sampling rate (most scene energy is null energy at these rates; base rates and caveats in Appendix E); the quantitative splits and their agreement across unrelated optimizations are the finding. Two further observations: the noisy-record case shows real in-range denoising ($42\%$ below the noisy pseudo-inverse) that nonetheless accounts for only ${\sim}5\%$ of the total gain, and GIDC's far-field super-resolution claim locates precisely in the null ledger — $43.8\%$ of the published reconstruction's structure lies in the null space of the patterns actually measured. This is a location of gains, not an accusation: the null content these methods supply is often substantially truth-aligned; the record simply cannot certify it. Full per-pipeline decompositions: Appendix E.

---

## 6. Related Work

| Prior line | What it owns | What this paper adds |
|---|---|---|
| Backus–Gilbert; Barrett–Myers | Linear estimability, null functions | Per-instance witness; GI governance protocol |
| MacKay; Tikhonov filter factors | Modal factors, effective parameters | Per-record post-hoc audit role |
| Null-space networks; DDNM; NPN/GSNR | Data-consistent null filling | Consistency read as non-certification; metered prior content |
| Bhadra; Gottschling; Antun; Iagaru | Hallucination limits, kernel instability | Constructive record-tight witness; governed dial; field audit |
| Perception–distortion; CodeFormer/ESRGAN knobs | Fidelity–quality controls | Knob acts strictly in $P_0$, preserving the record exactly |

**Ghost imaging.** The modality began with entangled-photon correlation imaging (Pittman et al. 1995); thermal-light and computational variants (Gatti et al. 2004; Ferri et al. 2005; Shapiro 2008; Bromberg et al. 2009; Duarte et al. 2008; Edgar et al. 2019) established the single-pixel operator this paper audits. Learned GI reconstructors (Lyu et al. 2017; He et al. 2018; the three pipelines of §5) buy large quality gains whose provenance is exactly what the two-ledger account makes explicit.

**Nearest prior art.** Bhadra et al. (2021) decompose reconstruction error through the same projectors and attribute null hallucinations to the prior; Iagaru et al. (2026) bound hallucination via feasible-set diameters; Gottschling et al. (2025) prove kernel-aware no-free-lunch theorems with destabilizing kernel constructions. Our witness differs in kind: cross-class, constructed on the governed dial's own operator, and matching the record more tightly than the noisy truth. Null-space networks (Schwab et al. 2019), DDNM, and NPN/GSNR use the same decomposition to reconstruct — they fill $\ker A$ with learned content; we meter that filling and refuse it measurement authority. The certificate's factor is classical (filter factors; MacKay's $\gamma$); its per-record audit role, paired with a converse, is not. The dial's nearest interfaces, CodeFormer's $w$ and ESRGAN's $\alpha$, act on entangled features and move reconstructions off the measurement manifold; $B$ cannot. The extended review — hallucination benchmarks, instability, diffusion solvers, conformal UQ, identification theory across fields — is Appendix D.

---

## 7. Discussion and Limitations

**The synthesis.** Report both axes and label provenance: the $m$ row coordinates are *measured* (certifiable by the audit); the injected null coordinates are *prior-supplied* (exact against the record by construction, and openly attributed); the true null content is *unverifiable* from the record. These three labels are the whole discipline. $A\hat{x} = y$ certifies that a reconstruction reproduces the buckets; it never certifies that invented texture is the true scene.

**Hard limitations.** The study is simulation-only, at one resolution, with locked claims on one 5% operator; there is no photon-level noise model and no hardware validation. The true null content remains uncertifiable by any within-record method — that is the theorem, not an engineering gap.

**Method limitations.** The VQGAN is a representative adversarial prior, not a state-of-the-art generator, and we make no SOTA claim; the paired-seed evidence for the adversarial branch's margin over a matched control is small. $B$ is a fixed, validation-selected scalar — the no-adaptation lemma and the selector experiment show per-record adaptation cannot do better from $y$ alone. Posterior latent sampling failed to produce calibrated null-space diversity and is reported as a negative result (Appendix F). $|P_0\hat{x}|$ is a provenance map, not an error map (§2.3).

**Evidence tiers.** Cross-rate and noise results are development-level. The forensics depend on the audited papers' released operators and calibration quality; attribution is stated relative to the min-norm reference (Appendix E).

---

## 8. Conclusion

Measurement consistency in low-rate ghost imaging certifies the row ledger only: the record fixes $P_R x$, and a wrong image can reproduce it more tightly than the truth. We built the two instruments this fact demands — an exact per-mode audit for the measured component, and a governed dial that supplies prior detail strictly inside the null space, with $A\hat{x}_B = y$ at every setting and a locked 32.6% perceptual gain at bounded distortion cost.

The account generalizes beyond our construction: on their own operators, three published deep-learning pipelines repair the row ledger, saturate it, and pay their headline quality from the null ledger.

What this changes is reporting. Reconstructions in undersampled imaging should state measurement provenance, not only quality: which content the bucket certified, and which the prior supplied. In undersampled ghost imaging, the right standard is not to forbid priors — it is to stop laundering prior detail as measured evidence.

---

## References

- Adler, J. & Öktem, O. Learned Primal-Dual Reconstruction. IEEE Transactions on Medical Imaging 37(6), 1322–1332 (2018). arXiv:1707.06474.
- Aggarwal, H. K., Mani, M. P. & Jacob, M. MoDL: Model-Based Deep Learning Architecture for Inverse Problems. IEEE Transactions on Medical Imaging 38(2), 394–405 (2019). arXiv:1712.02862.
- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., Schuster, T. Conformal Risk Control. International Conference on Learning Representations (ICLR) (2024). arXiv:2208.02814 (2022).
- Angelopoulos, A. N., Kohli, A. P., Bates, S., Jordan, M. I., Malik, J., Alshaabi, T., Upadhyayula, S., Romano, Y. Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging. Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR 162, 717-730 (2022). arXiv:2202.05265.
- Antun, V., Renna, F., Poon, C., Adcock, B. & Hansen, A. C. On instabilities of deep learning in image reconstruction and the potential costs of AI. Proceedings of the National Academy of Sciences 117(48), 30088-30095 (2020). arXiv:1902.05300.
- Asim, M., Daniels, M., Leong, O., Ahmed, A. & Hand, P. Invertible generative models for inverse problems: mitigating representation error and dataset bias. Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 119, 399-409 (2020). arXiv:1905.11672.
- Backus, G. & Gilbert, F. The Resolving Power of Gross Earth Data. Geophysical Journal of the Royal Astronomical Society 16(2), 169-205 (1968); Backus, G. & Gilbert, F. Uniqueness in the Inversion of Inaccurate Gross Earth Data. Philosophical Transactions of the Royal Society of London A 266(1173), 123-192 (1970).
- Barrett, H. H. & Myers, K. J. Foundations of Image Science. Wiley-Interscience, Hoboken, NJ, 1540 pp. (2004).
- Bhadra, S., Kelkar, V. A., Brooks, F. J., Anastasio, M. A. On Hallucinations in Tomographic Image Reconstruction. IEEE Transactions on Medical Imaging 40(11), 3249-3260 (2021). arXiv:2012.00646.
- Bińkowski, M., Sutherland, D. J., Arbel, M., Gretton, A. Demystifying MMD GANs. International Conference on Learning Representations (ICLR) (2018). arXiv:1801.01401.
- Blau, Y., Mechrez, R., Timofte, R., Michaeli, T. & Zelnik-Manor, L. The 2018 PIRM Challenge on Perceptual Image Super-Resolution. In Computer Vision – ECCV 2018 Workshops, Lecture Notes in Computer Science 11133, 334–355 (Springer, 2018). arXiv:1809.07517.
- Blau, Y., Michaeli, T. Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff. In Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97, 675-685 (2019). arXiv:1901.07821.
- Blau, Y., Michaeli, T. The Perception-Distortion Tradeoff. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 6228-6237 (2018). arXiv:1711.06077.
- Bora, A., Jalal, A., Price, E. & Dimakis, A. G. Compressed sensing using generative models. Proceedings of the 34th International Conference on Machine Learning (ICML), PMLR 70, 537-546 (2017). arXiv:1703.03208.
- Boufounos, P., Duarte, M. F., Baraniuk, R. G. Sparse Signal Reconstruction from Noisy Compressive Measurements using Cross Validation. Proc. 2007 IEEE/SP 14th Workshop on Statistical Signal Processing (SSP), Madison, WI, 299-303 (2007).
- Bromberg, Y., Katz, O. & Silberberg, Y. Ghost imaging with a single detector. Physical Review A 79, 053840 (2009). arXiv:0812.2633.
- Buğday, S., Saeys, Y., Peck, J. Triggering hallucinations in model-based MRI reconstruction via adversarial perturbations. arXiv:2602.18536 (2026).
- Candès, E. J., Romberg, J. & Tao, T. Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information. IEEE Transactions on Information Theory 52(2), 489-509 (2006). arXiv:math/0409186 (2004).
- Chen, D. & Davies, M. E. Deep Decomposition Learning for Inverse Imaging Problems. In Computer Vision – ECCV 2020, Lecture Notes in Computer Science 12373, 510–526 (Springer, 2020). arXiv:1911.11028.
- Chen, D., Tachella, J. & Davies, M. E. Equivariant Imaging: Learning Beyond the Range Space. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 4379-4388 (2021). arXiv:2103.14756.
- Chung, H., Kim, J., McCann, M. T., Klasky, M. L. & Ye, J. C. Diffusion Posterior Sampling for General Noisy Inverse Problems. The Eleventh International Conference on Learning Representations (ICLR) (2023). arXiv:2209.14687.
- Chung, H., Sim, B., Ryu, D. & Ye, J. C. Improving Diffusion Models for Inverse Problems using Manifold Constraints. Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 25683-25696 (2022). arXiv:2206.00941.
- Coates, A., Ng, A. Y., Lee, H. An Analysis of Single-Layer Networks in Unsupervised Feature Learning. Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 15, 215-223 (2011).
- Daras, G., Chung, H., Lai, C.-H., Mitsufuji, Y., Ye, J. C., Milanfar, P., Dimakis, A. G. & Delbracio, M. A Survey on Diffusion Models for Inverse Problems. arXiv preprint arXiv:2410.00083 (2024).
- Duarte, M. F., Davenport, M. A., Takhar, D., Laska, J. N., Sun, T., Kelly, K. F. & Baraniuk, R. G. Single-pixel imaging via compressive sampling. IEEE Signal Processing Magazine 25(2), 83-91 (2008).
- Durall, R., Keuper, M. & Keuper, J. Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 7890-7899 (2020). arXiv:2003.01826.
- Edgar, M. P., Gibson, G. M. & Padgett, M. J. Principles and prospects for single-pixel imaging. Nature Photonics 13, 13-20 (2019).
- Engl, H. W., Hanke, M. & Neubauer, A. Regularization of Inverse Problems. Kluwer Academic Publishers, Dordrecht (1996).
- Esser, P., Rombach, R. & Ommer, B. Taming Transformers for High-Resolution Image Synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 12873-12883 (2021). arXiv:2012.09841.
- Ferri, F., Magatti, D., Gatti, A., Bache, M., Brambilla, E., Lugiato, L. A. High-Resolution Ghost Image and Ghost Diffraction Experiments with Thermal Light. Physical Review Letters 94, 183602 (2005). (Preprint: Magatti, D., Ferri, F., Gatti, A., Bache, M., Brambilla, E., Lugiato, L. A. Experimental evidence of high-resolution ghost imaging and ghost diffraction with classical thermal light. arXiv:quant-ph/0408021, 2004.).
- Gatti, A., Brambilla, E., Bache, M., Lugiato, L. A. Ghost Imaging with Thermal Light: Comparing Entanglement and Classical Correlation. Physical Review Letters 93, 093602 (2004). arXiv:quant-ph/0307187.
- Genzel, M., Macdonald, J., März, M. Solving Inverse Problems With Deep Neural Networks – Robustness Included? IEEE Transactions on Pattern Analysis and Machine Intelligence 45(1), 1119–1134 (2023). arXiv:2011.04268.
- Gerchberg, R. W. Super-resolution through error energy reduction. Optica Acta 21(9), 709-720 (1974).
- Gibson, G. M., Johnson, S. D. & Padgett, M. J. Single-pixel imaging 12 years on: a review. Optics Express 28(19), 28190-28208 (2020).
- Gottschling, N. M., Antun, V., Hansen, A. C., Adcock, B. The Troublesome Kernel: On Hallucinations, No Free Lunches, and the Accuracy-Stability Tradeoff in Inverse Problems. SIAM Review 67(1), 73-104 (2025). arXiv:2001.01258.
- Gu, Y., Wang, X., Xie, L., Dong, C., Li, G., Shan, Y. & Cheng, M.-M. VQFR: Blind Face Restoration with Vector-Quantized Dictionary and Parallel Decoder. In European Conference on Computer Vision (ECCV), LNCS 13678, 126-143 (2022). arXiv:2205.06803.
- Gualdrón-Hurtado, R., Jacome, R., Suárez, R. S. & Arguello, H. GSNR: Graph Smooth Null-Space Representation for Inverse Problems. Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR) (2026). arXiv:2602.20328.
- Hammernik, K., Klatzer, T., Kobler, E., Recht, M. P., Sodickson, D. K., Pock, T. & Knoll, F. Learning a Variational Network for Reconstruction of Accelerated MRI Data. Magnetic Resonance in Medicine 79(6), 3055–3071 (2018). arXiv:1704.00447.
- Hansen, P. C. Rank-Deficient and Discrete Ill-Posed Problems: Numerical Aspects of Linear Inversion. SIAM, Philadelphia (1998).
- He, Y., Wang, G., Dong, G., Zhu, S., Chen, H., Zhang, A. & Xu, Z. Ghost imaging based on deep learning. Scientific Reports 8, 6469 (2018).
- Heckel, R. & Hand, P. Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks. In 7th International Conference on Learning Representations (ICLR) (2019). arXiv:1810.03982.
- Huang, Y., Lösel, P. D., Paganin, D. M. & Kingston, A. M. Deep learning in classical x-ray ghost imaging for dose reduction. Physical Review A 110, 063512 (2024). arXiv:2411.06340.
- Iagaru, D., Gottschling, N. M., Hansen, A. C. & Garnier, J. On Hallucinations in Inverse Problems: Fundamental Limits and Provable Assessment Methods. arXiv:2605.13146 (2026).
- Isola, P., Zhu, J.-Y., Zhou, T., Efros, A. A. Image-to-Image Translation with Conditional Adversarial Networks. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1125-1134 (2017). arXiv:1611.07004.
- Jacome, R., Gualdrón-Hurtado, R., Suarez, L. & Arguello, H. NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems. Advances in Neural Information Processing Systems (NeurIPS 2025) (2025). arXiv:2510.01608.
- Katz, O., Bromberg, Y. & Silberberg, Y. Compressive ghost imaging. Applied Physics Letters 95, 131110 (2009). arXiv:0905.0321.
- Kawar, B., Elad, M., Ermon, S. & Song, J. Denoising Diffusion Restoration Models. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 23593–23606 (2022). arXiv:2201.11793.
- Kendall, A., Gal, Y. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? Advances in Neural Information Processing Systems (NeurIPS/NIPS) 30, 5580–5590 (2017). arXiv:1703.04977.
- Kim, S., Tregidgo, H. F. J., Figini, M., Jin, C., Joshi, S., Alexander, D. C. Tackling Hallucination from Conditional Models for Medical Image Reconstruction with DynamicDPS. Medical Image Computing and Computer Assisted Intervention – MICCAI 2025, LNCS (Springer, 2025). arXiv:2503.01075.
- Kim, S., Tregidgo, H. F. J., Jin, C., Figini, M., Alexander, D. C. HalluGen: Synthesizing Realistic and Controllable Hallucinations for Evaluating Image Restoration. arXiv:2512.03345 (2025).
- Koopmans, T. C. & Reiersøl, O. The Identification of Structural Characteristics. Annals of Mathematical Statistics 21, 165-181 (1950).
- Landau, H. J. Necessary Density Conditions for Sampling and Interpolation of Certain Entire Functions. Acta Mathematica 117, 37-52 (1967).
- Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., Aitken, A., Tejani, A., Totz, J., Wang, Z. & Shi, W. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4681–4690 (2017). arXiv:1609.04802.
- Lewbel, A. The Identification Zoo: Meanings of Identification in Econometrics. Journal of Economic Literature 57(4), 835-903 (2019).
- Lu, C., Angelopoulos, A. N., Pomerantz, S. Improving Trustworthiness of AI Disease Severity Rating in Medical Imaging with Ordinal Conformal Prediction Sets. Medical Image Computing and Computer Assisted Intervention - MICCAI 2022, Lecture Notes in Computer Science 13438, 545-554 (Springer, 2022). arXiv:2207.02238.
- Lunz, S., Öktem, O. & Schönlieb, C.-B. Adversarial regularizers in inverse problems. Advances in Neural Information Processing Systems 31 (NeurIPS), 8507-8516 (2018). arXiv:1805.11572.
- Lyu, M., Wang, W., Wang, H., Wang, H., Li, G., Chen, N. & Situ, G. Deep-learning-based ghost imaging. Scientific Reports 7, 17865 (2017).
- MacKay, D. J. C. Bayesian Interpolation. Neural Computation 4(3), 415-447 (1992).
- Manni, M., Karpov, D., Batenburg, K. J., Shwartz, S. & Viganò, N. Noise2Ghost: Self-supervised deep convolutional reconstruction for ghost imaging. arXiv:2504.10288 (2025).
- Manski, C. F. Partial Identification of Probability Distributions. Springer Series in Statistics (Springer-Verlag, New York, 2003).
- Menon, S., Damian, A., Hu, S., Ravi, N. & Rudin, C. PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2437-2445 (2020). arXiv:2003.03808.
- Narnhofer, D., Effland, A., Kobler, E., Hammernik, K., Knoll, F., Pock, T. Bayesian Uncertainty Estimation of Learned Variational MRI Reconstruction. IEEE Transactions on Medical Imaging 41(2), 279–291 (2022). arXiv:2102.06665.
- Papoulis, A. A new algorithm in spectral analysis and band-limited extrapolation. IEEE Transactions on Circuits and Systems 22(9), 735-742 (1975).
- Pittman, T. B., Shih, Y. H., Strekalov, D. V., Sergienko, A. V. Optical imaging by means of two-photon quantum entanglement. Physical Review A 52, R3429 (1995).
- Razavi, A., van den Oord, A. & Vinyals, O. Generating Diverse High-Fidelity Images with VQ-VAE-2. Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 14837-14847 (2019). arXiv:1906.00446.
- Romano, Y., Elad, M. & Milanfar, P. The Little Engine That Could: Regularization by Denoising (RED). SIAM Journal on Imaging Sciences 10, 1804-1844 (2017). arXiv:1611.02862.
- Ronneberger, O., Fischer, P., Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015), LNCS 9351, 234-241 (2015). arXiv:1505.04597.
- Rothenberg, T. J. Identification in Parametric Models. Econometrica 39(3), 577-591 (1971).
- Rudin, L. I., Osher, S. & Fatemi, E. Nonlinear total variation based noise removal algorithms. Physica D 60, 259-268 (1992).
- Schlemper, J., Caballero, J., Hajnal, J. V., Price, A. N. & Rueckert, D. A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction. IEEE Transactions on Medical Imaging 37(2), 491–503 (2018). arXiv:1704.02422.
- Schwab, J., Antholzer, S. & Haltmeier, M. Deep null space learning for inverse problems: convergence analysis and rates. Inverse Problems 35, 025008 (2019). arXiv:1806.06137.
- Shah, V. & Hegde, C. Solving linear inverse problems using GAN priors: an algorithm with provable guarantees. Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 4609-4613 (2018). arXiv:1802.08406.
- Shapiro, J. H. Computational ghost imaging. Physical Review A 78, 061802(R) (2008). arXiv:0807.2614.
- Shimobaba, T., Endo, Y., Nishitsuji, T., Takahashi, T., Nagahama, Y., Hasegawa, S., Sano, M., Hirayama, R., Kakue, T., Shiraki, A. & Ito, T. Computational ghost imaging using deep learning. Optics Communications 413, 147-151 (2018). arXiv:1710.08343.
- Song, K., Bian, Y., Wang, D., Li, R., Wu, K., Liu, H., Qin, C., Hu, J. & Xiao, L. Advances and challenges of single-pixel imaging based on deep learning. Laser & Photonics Reviews 19(7), 2401397 (2025).
- Song, Y., Shen, L., Xing, L. & Ermon, S. Solving Inverse Problems in Medical Imaging with Score-Based Generative Models. In International Conference on Learning Representations (ICLR) (2022). arXiv:2111.08005.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S. & Poole, B. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations (ICLR) (2021). arXiv:2011.13456.
- Stuart, A. M. Inverse Problems: A Bayesian Perspective. Acta Numerica 19, 451-559 (2010).
- Tachella, J., Chen, D. & Davies, M. Sensing Theorems for Unsupervised Learning in Linear Inverse Problems. Journal of Machine Learning Research 24(39), 1-45 (2023). arXiv:2203.12513.
- Tikhonov, A. N. Solution of incorrectly formulated problems and the regularization method. Soviet Mathematics Doklady 4, 1035-1038 (1963).
- Tirer, T. & Giryes, R. Image restoration by iterative denoising and backward projections. IEEE Transactions on Image Processing 28, 1220-1234 (2019). arXiv:1710.06647.
- Ulyanov, D., Vedaldi, A. & Lempitsky, V. Deep image prior. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 9446-9454 (2018). arXiv:1711.10925.
- van den Oord, A., Vinyals, O. & Kavukcuoglu, K. Neural Discrete Representation Learning. Advances in Neural Information Processing Systems 30 (NIPS 2017), 6306-6315 (2017). arXiv:1711.00937.
- Venkatakrishnan, S. V., Bouman, C. A. & Wohlberg, B. Plug-and-Play Priors for Model Based Reconstruction. IEEE Global Conference on Signal and Information Processing (GlobalSIP), 945-948 (2013).
- Wang, F., Wang, C., Chen, M., Gong, W., Zhang, Y., Han, S. & Situ, G. Far-field super-resolution ghost imaging with a deep neural network constraint. Light: Science & Applications 11, 1 (2022). [Cited as Wang et al. 2022b.]
- Wang, F., Wang, C., Deng, C., Han, S. & Situ, G. Single-pixel imaging using physics enhanced deep learning. Photonics Research 10, 104-110 (2022). [Cited as Wang et al. 2022a.]
- Wang, J., Wang, S., Zhang, R., Zheng, Z., Liu, W. & Wang, X. A Range-Null Space Decomposition Approach for Fast and Flexible Spectral Compressive Imaging. arXiv:2305.09746 (2023).
- Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., Loy, C. C. ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks. Computer Vision - ECCV 2018 Workshops, Lecture Notes in Computer Science 11133, 63-79 (2019). arXiv:1809.00219.
- Wang, Y., Hu, Y., Yu, J., Zhang, J. GAN Prior Based Null-Space Learning for Consistent Super-resolution. Proceedings of the AAAI Conference on Artificial Intelligence 37(3), 2724-2732 (2023). arXiv:2211.13524.
- Wang, Y., Yu, J. & Zhang, J. Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model. The Eleventh International Conference on Learning Representations (ICLR) (2023). arXiv:2212.00490.
- Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing 13(4), 600-612 (2004).
- Wang, Z., Zhang, J., Chen, R., Wang, W. & Luo, P. RestoreFormer: High-Quality Blind Face Restoration from Undegraded Key-Value Pairs. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17512-17521 (2022). arXiv:2201.06374.
- Ward, R. Compressed Sensing With Cross Validation. IEEE Transactions on Information Theory 55(12), 5773-5782 (2009). arXiv:0803.1845.
- Wen, J., Ahmad, R., Schniter, P. Task-Driven Uncertainty Quantification in Inverse Problems via Conformal Prediction. Computer Vision - ECCV 2024, Lecture Notes in Computer Science 15118, 182-199 (Springer, 2025). arXiv:2405.18527.
- Yang, Y., Sun, J., Li, H. & Xu, Z. Deep ADMM-Net for Compressive Sensing MRI. Advances in Neural Information Processing Systems 29 (NIPS 2016), 10–18 (2016).
- Yeh, R. A., Chen, C., Lim, T.-Y., Schwing, A. G., Hasegawa-Johnson, M. & Do, M. N. Semantic image inpainting with deep generative models. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 6882-6890 (2017). arXiv:1607.07539.
- Youla, D. C. & Webb, H. Image Restoration by the Method of Convex Projections: Part 1 — Theory. IEEE Transactions on Medical Imaging 1(2), 81-94 (1982).
- Zhang, J. & Ghanem, B. ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing. Proc. IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR), 1828-1837 (2018). arXiv:1706.07929.
- Zhang, R., Isola, P., Efros, A. A., Shechtman, E. & Wang, O. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 586–595 (2018). arXiv:1801.03924.
- Zhou, S., Chan, K. C. K., Li, C., Loy, C. C. Towards Robust Blind Face Restoration with Codebook Lookup Transformer. Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 30599-30611 (2022). arXiv:2206.11253.
- Zhu, Y., Zhang, K., Liang, J., Cao, J., Wen, B., Timofte, R. & Van Gool, L. Denoising Diffusion Models for Plug-and-Play Image Restoration. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW, NTIRE), 1219-1229 (2023). arXiv:2305.08995.

---

# Appendices

*(The appendices preserve the long-form material of the working draft; internal section numbers refer to its original structure where they appear.)*

## Appendix A. Geometry: derivations and special cases

Everything that follows — the converse of §3, the certificate of §4, and the governed dial of §6 — is a consequence of a single decomposition of the measurement operator. We state it once here, in operator-agnostic form, and reuse it throughout.

### 2.1 Forward model and compact SVD

We consider the standard linear ghost-imaging model
$$
y = A x + \varepsilon,
$$
where $x \in \mathbb{R}^n$ is the vectorized image, $y \in \mathbb{R}^m$ is the bucket-measurement vector, $A \in \mathbb{R}^{m \times n}$ is the sensing operator with $m \ll n$, and $\varepsilon$ is measurement noise. Throughout, the relative measurement error of a candidate $v$ against a record $y$ is $\mathrm{RelMeasErr}(v; y) = \lVert A v - y \rVert_2 / \lVert y \rVert_2$. Write the compact singular value decomposition of $A$ as
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

## Appendix B. The converse and the audit: operator-level detail


The geometry of §2 is deductive but silent about severity: the lemma $A P_0 = 0$ shows that $y$ fixes exactly $P_R x$ and constrains nothing in the null space, but it does not by itself say how badly a null-space error can hide behind a clean measurement. This section supplies the converse. We construct, for any pair of images, an explicit reconstruction that reproduces a target's bucket record to machine precision while carrying an entirely different scene's unmeasured content. The construction is per-instance and exact, not asymptotic or distributional: it exhibits a concrete feasible-but-wrong witness for every cross-class pair we test. Its consequence is the load-bearing claim of the paper — measurement consistency is certifiable but the correctness of null-space content is not, so **consistency is not correctness** — and it is placed first, before any certificate or injection dial, because everything downstream is subordinate to it.

### 3.1 A per-instance feasible-but-wrong witness

Fix a target image $x_i$ with recorded buckets $y_i = A x_i + \varepsilon_i$ — throughout the witness experiments $\varepsilon$ is i.i.d. Gaussian bucket noise with $\sigma_\varepsilon = 0.01$ — and a *donor* image $x_j$ from a different semantic class. We want a single image that carries the target's measurement but the donor's unmeasured detail. Define

$$u_{ij} = x_j - A^\dagger\big(A x_j - y_i\big).$$

The witness is measurement-consistent with the *record* to first principles. Because $A$ has full row rank here, $\mathcal{R}(A) = \mathbb{R}^m$, so $A A^\dagger = I_m$ and for *any* recorded $y_i$ (noisy or not),

$$A u_{ij} = A x_j - A A^\dagger\big(A x_j - y_i\big) = A x_j - (A x_j - y_i) = y_i.$$

Its two projections separate cleanly. The row-space component is pinned to the target,

$$P_R\, u_{ij} = A^\dagger A u_{ij} = A^\dagger y_i = P_R\, x_i + A^\dagger \varepsilon_i,$$

equal to $P_R x_i$ exactly in the noiseless case and pinned to the *noisy record* — not to the clean target — otherwise,

while the null-space component is inherited *whole* from the donor,

$$P_0\, u_{ij} = P_0\, x_j - P_0\, A^\dagger\big(A x_j - y_i\big) = P_0\, x_j,$$

where the second term vanishes because $A^\dagger$ maps into the row space and $P_0$ annihilates it ($P_0 A^\dagger = 0$). In words, $u_{ij}$ splices the measured part of $x_i$ onto the unmeasured part of $x_j$ and passes the measurement audit of $x_i$ exactly. Figure 3 draws the construction: the feasible set is a line (an affine flat), the truth sits on it only up to its own noise, and the witness is simply the donor dropped orthogonally onto that line — which is why no test computed from the record can prefer the truth to the splice.


### 3.2 The record is matched more tightly than the truth

The witness is not merely feasible in principle; numerically it satisfies the record more tightly than the true scene does. On the Rad-5 operator ($m = 205$, $64\times 64$), across a set of $8$ cross-class feasible pairs, the relative measurement error of the witness against the target record lies in the range $\mathrm{RelMeasErr}(u_{ij}, y_i) \in [2.16\times10^{-15},\, 4.00\times10^{-15}]$, i.e. at floating-point round-off. Over the paired families the construction holds for $16/16$ cross-class pairs at the same $\sim\!2\times10^{-15}$ scale. The *true* image, by contrast, only satisfies the (noisy) record to $[3.45\times10^{-3},\, 7.46\times10^{-3}]$. The gap is roughly twelve orders of magnitude: **the wrong image reproduces the buckets more exactly than the ground truth.** The two numbers measure different things, and saying so is the point: the witness's $10^{-15}$ is pure geometry (it is *constructed* on the record), while the truth's $10^{-3}$ is exactly its own bucket noise ($\sigma_\varepsilon = 0.01$) — so the twelve orders quantify how far *any* residual test is from adjudicating truth: every threshold loose enough to admit the noisy truth admits the exact-fitting impostor a fortiori. 
A featured car-versus-horse pair makes the point concretely — the target car has $\mathrm{RelMeasErr}(x_i, y_i) = 5.36\times10^{-3}$, while the constructed witness follows the horse donor visually and yet attains $\mathrm{RelMeasErr}(u_{ij}, y_i) = 2.93\times10^{-15}$. The witnesses remain semantically wrong with respect to the target, with $\mathrm{PSNR}(u_{ij}, x_i)$ in the range $7.70$–$11.38$ dB.

Crucially, the construction is not tied to that operator: on the *same* $64\times64$ STL10 operator at $5\%$ sampling ($m=205$) used for the governed dial of §7, 40 feasible-wrong witnesses reproduce their target records at $\mathrm{RelMeasErr}(u_{ij}, y_i)$ between $1.7\times10^{-14}$ and $5.6\times10^{-13}$ (median $9.9\times10^{-14}$) while remaining semantically wrong (median $\mathrm{PSNR}(u_{ij}, x_i) = 19.5$ dB). The converse of this section and the constructive dial of §7 are therefore exhibited on one and the same operator; the residual is a few orders of magnitude above the Rad-5 figure only because the mixed-basis operator is applied through a ridge-regularized ($\lambda_{\text{solver}}=10^{-6}$) audit rather than an exact orthonormal projector, and it remains eleven orders below the $\sim\!10^{-3}$ noise floor.

The immediate corollary is that no residual-based test, however tight its threshold, can separate the witness from the truth: any threshold that admits the true image (residual $\sim\!10^{-3}$) admits the witness (residual $\sim\!10^{-15}$) a fortiori. This is precisely the failure mode PULSE exhibits for super-resolution (Menon et al. 2020) — visually distinct realistic reconstructions collapsing to the same low-dimensional observation — but here it is exact, per-instance, and derived from the operator's own geometry rather than sampled from a generator.

### 3.3 Position relative to the limits literature

The existence of consistency-preserving hallucinations, and the impossibility of assessing them without ground truth, is established prior art. Iagaru et al. (2026) prove necessary-and-sufficient conditions for detail-transfer hallucination in (possibly nonlinear) inverse problems — consistent decoders can only hallucinate details almost invisible in measurement space — and give forward-model-only, ground-truth-free algorithms that *bound* hallucination magnitude via feasible-set diameters. Gottschling et al. (2025) situate the same phenomenon within the broader account of instabilities and fundamental limits of learned reconstruction, and Bhadra et al. (2021) formalize null-space hallucinations by SVD projection, showing they are attributable solely to the prior and cannot be assessed without the truth. We take these as the anchor for the barrier itself and do not reclaim their ground: that null content is unverifiable, and that assessment must be ground-truth-free, is theirs.

Our delta is constructive and per-instance. Where the limits line establishes *existence* through necessary-and-sufficient conditions, or supplies *bounds* over feasible sets (a supremum that requires paired data to evaluate), $u_{ij}$ is an explicit machine-precision witness for each individual instance — an object one can render, measure, and display, matching the record tighter than the noisy truth. It converts "consistency is not correctness" from a theorem about maps and an SVD diagnostic into a concrete pixelwise construction. This is the engine on which the rest of the paper turns: because the barrier is real and exact, the certificate of §4 can only ever certify the row space, and the detail injection of §6 is licensed to act only where the barrier proves the measurement is blind.

### 3.4 The unmeasured-content magnitude is not an error map

A natural but mistaken hope is that the observable magnitude $|P_0 \hat{x}|$ — how much unmeasured content a reconstruction carries, pixel by pixel — could serve as a self-contained error map, flagging where a reconstruction has invented detail. It cannot, and the same decomposition explains why. For any audited reconstruction $\hat{x}$ the null-space error is

$$P_0(\hat{x} - x) = P_0 \hat{x} - P_0 x,$$

and the observable term $P_0 \hat{x}$ contains no direct information about the unknown truth term $P_0 x$. The magnitude is a property of the reconstruction, not a certificate of its error: a large value may be accurate texture, inaccurate texture, or a harmless choice among feasible completions. We therefore call $|P_0 \hat{x}|$ a **prior-supplied-content map** — it marks where a prior placed unmeasured content, not where the reconstruction is wrong.

The decoupling is empirical, not merely formal. On the $64\times 64$ Rad-5 validation, per-pixel $|P_0 \hat{x}|$ and actual null-space error are essentially uncorrelated: for the LMMSE arm the Spearman and Pearson correlations are $-0.079$ and $-0.069$, and for the two learned arms they are $\approx 0.07$ and $\approx 0.06$. The ordering is if anything inverted for the learned arms — the top-$10\%$ highest-$|P_0 \hat{x}|$ pixels carry *lower* actual null-space error ($0.071$) than the remaining $90\%$ ($0.079$). The map therefore has, at most, image-level diagnostic value (an aggregate indication that a reconstruction leans on its prior); it is not a pixelwise error locator, and we do not use it as one anywhere in this paper. This negative result closes the section on the same theme it opened: the null space is where the prior speaks, and nothing the bucket records — not the residual, not the magnitude of the supplied content — can adjudicate whether that speech is true.

---

### The audit in full


Section 3 established a converse: the record $y$ fixes $P_R x$ and nothing else, so any two feasible reconstructions may differ arbitrarily in the null space while producing the same measurement. That result is destructive — it says what cannot be verified. This section is its constructive counterpart. If the measurement cannot certify null-space content, it should at least certify, exactly and without ground truth, the one thing it does constrain: the measured component. We give a test-time operator that does this, prove that it contracts each measured mode by a known factor determined only by $A$ and $\lambda$, and show that it can be applied after any reconstructor as a post-hoc audit.

### 4.1 A plug-in test-time audit

Let $\hat{x}$ be any reconstruction — analytic, variational, or learned — with residual $r(\hat{x}) = A\hat{x} - y$. Define the audit operator

$$\Pi_y^\lambda(v) \;=\; v - A^\top\!\big(AA^\top + \lambda I\big)^{-1}\big(Av - y\big), \qquad \lambda > 0,$$

which we write compactly as $\Pi_y^\lambda(v) = v - G_\lambda(Av - y)$ with $G_\lambda = A^\top(AA^\top + \lambda I)^{-1}$. The audit requires only the operator $A$, the record $y$, and a single scalar $\lambda$; it needs no ground truth and no access to how $\hat{x}$ was produced. It is therefore a *plug-in* audit: it wraps an existing pipeline rather than replacing it.

**The update touches only the row space.** The correction $-G_\lambda(Av - y)$ lies in $\operatorname{range}(A^\top)$, which is exactly the row space. Consequently the null-space component is left untouched:

$$P_0\,\Pi_y^\lambda(v) \;=\; P_0 v .$$

This is the achievability side of the same geometry $A P_0 = 0$ that drove the converse. The audit cannot, and does not, adjudicate the null space; it operates strictly within the subspace the measurement is accountable for. Whatever a prior has placed in $P_0$ passes through the audit unchanged — a property we will rely on in §6, where the injected detail must survive the audit intact.

### 4.2 Exact per-mode contraction

The audit does not merely reduce the measurement residual; it reduces it by an amount we can write down in closed form. Applying $A$ to the audited estimate and simplifying,

$$A\,\Pi_y^\lambda(v) - y \;=\; \big[I - AA^\top(AA^\top + \lambda I)^{-1}\big]\,r(v) \;=\; \lambda\,(AA^\top + \lambda I)^{-1}\,r(v).$$

Diagonalizing $AA^\top = U_r \Sigma_r^2 U_r^\top$ and expanding the residual in the left singular vectors $u_i$, each measured mode is scaled independently:

$$c_i(\lambda) \;=\; \frac{\lambda}{\lambda + \sigma_i^2}.$$

The interpretation is exact rather than asymptotic. After one audit, the residual along the $i$-th measured mode is multiplied by $\lambda/(\lambda + \sigma_i^2)$ — a factor fixed entirely by the operator's singular value $\sigma_i$ and the chosen $\lambda$, independent of the image and of the reconstructor. As $\lambda \to 0$ the contraction becomes a hard projection onto $\{Av = y\}$; for $\lambda > 0$ it retains a controlled residual, appropriate when $y$ is noisy and should not be over-interpreted as exact truth. This closed-form modal spectrum is the object we certify. One scope sentence governs all audit numbers below: the certificate certifies agreement with the *recorded* $y$ (which carries $\sigma_\varepsilon = 0.01$ bucket noise in these experiments), never agreement with the noiseless $A x$; driving the residual far below the noise floor therefore reproduces the record, noise realization included — a deliberate choice we flag rather than a virtue, and the reason a deployment would set $\lambda$ by a discrepancy criterion instead of as small as possible.


**Float64 verification.** The contraction formula is confirmed in double precision. For the Rad-5 operator ($\sigma_{\min} = 3.476$, $\sigma_{\max} = 5.454$) the maximum deviation between the measured contraction and $\lambda/(\lambda + \sigma_i^2)$ is $1.04\times10^{-10}$; for the Scr-5 operator ($\sigma_{\min} = \sigma_{\max} = 1.000$) it is $2.29\times10^{-12}$. The identity therefore holds to the floating-point floor. (Repeated float32 pipeline audits saturate at a solver floor and are not evidence for the modal identity; the certificate is a float64 statement.)

### 4.3 Post-hoc audit across reconstructor families

Because $\Pi_y^\lambda$ depends only on $(A, y, \lambda)$, it can be applied uniformly to reconstructions from unrelated methods. We audit backprojection (BP), Tikhonov, a small-subset CS–TV sanity check, and a learned reconstructor, on both the Rad-5 and Scr-5 operators. Across these rows the pattern is consistent: the relative measurement error $\mathrm{RelMeasErr} = \lVert A\hat{x} - y\rVert / \lVert y\rVert$ drops by three to four orders of magnitude while the PSNR barely moves. For the learned Rad-5 output, $\mathrm{RelMeasErr}$ falls from $3.68\times10^{-2}$ to $1.90\times10^{-6}$ for a PSNR change of $+0.0136$ dB; for learned Scr-5 it falls from $1.80\times10^{-2}$ to $1.80\times10^{-5}$ at $+0.0387$ dB. Aggregated over all audited rows, the residual reductions of three to four orders come at $|\Delta\mathrm{PSNR}| \le 0.039$ dB, and the sign of the certificate is stable across the audited conditions (18/18). The audit thus buys measurement accountability essentially for free in image quality — the separation of these two axes is the subject of §5.

### 4.4 Position: exact modal spectrum, not a feasible-set bound

The fundamental-limits line closest to this result is Iagaru et al. (2026), who prove necessary-and-sufficient conditions for detail-transfer hallucination and give forward-model-only, ground-truth-free algorithms that *bound* hallucination magnitude via feasible-set diameters (worst-case kernel size). Our contribution is not the ground-truth-free stance, which is theirs, but the form of the guarantee. Where their assessment yields a bound over a feasible set — a supremum requiring paired data to instantiate — we give an *exact* per-mode contraction $c_i(\lambda) = \lambda/(\lambda + \sigma_i^2)$ read directly from the operator's SVD and realized as a plug-in operator applied after BP, Tikhonov, CS–TV, and learned reconstructors alike. The distinction is exact-versus-bound and operator-versus-existence: a closed-form modal spectrum applied at test time, rather than a feasible-set-diameter estimate. This certifies precisely what the measurement is accountable for. It says nothing about the null space, and we do not read it as licensing the invented texture §6 will introduce — $A\hat{x} = y$ is not certification that the null-space content is the true scene.

A second, closer antecedent is Bayesian. MacKay's (1992) *number of well-determined parameters* is, in his symbols, $\gamma = \sum_i \lambda_i/(\lambda_i + \alpha)$ with $\lambda_i$ the eigenvalues of the (Gram) curvature and $\alpha$ the regularizer weight; under the correspondence $\lambda_i \leftrightarrow \sigma_i^2$ and $\alpha \leftrightarrow \lambda$, each summand is exactly the complement of our contraction factor, $1 - c_i(\lambda) = \sigma_i^2/(\sigma_i^2+\lambda)$. The formula is the same; its epistemic role is inverted. MacKay reads $\gamma$ as an aggregate scalar consumed during *fitting* — for evidence maximization and regularization-strength selection — whereas we read the identical modal contraction as a per-mode, per-record *audit* applied after any reconstructor, paired with a constructive converse establishing that the complementary $n - m$ modes are unaccountable (§3) and a dial that governs them (§6). Same modal spectrum, opposite direction of use: a fitting diagnostic becomes a test-time certificate. The same disclosure applies one field over: $c_i(\lambda) = \lambda/(\lambda+\sigma_i^2) = 1 - \varphi_i(\lambda)$ where $\varphi_i$ are the classical Tikhonov filter factors of regularization theory (Engl, Hanke & Neubauer 1996; Hansen 1998) — the residual factor of one ridge-regularized consistency step is textbook. We flag both coincidences explicitly so the certificate is not mistaken for a new scalar; the delta is entirely in the role — a per-record, ground-truth-free, reconstructor-agnostic *audit primitive* attached after arbitrary (including learned) pipelines, paired with the converse that fixes what it can never certify.

*Verification.* All numbers in this section are drawn from the audit experiments: float64 mode deviations $1.04\times10^{-10}$ (Rad-5) and $2.29\times10^{-12}$ (Scr-5); learned-output residual drops $3.68\times10^{-2}\!\to\!1.90\times10^{-6}$ (Rad-5) and $1.80\times10^{-2}\!\to\!1.80\times10^{-5}$ (Scr-5) at $|\Delta\mathrm{PSNR}| \le 0.039$ dB.

---

### The separation law in full


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

**Theorem (PSNR ceiling for row-space corrections).** *Let $T$ be any correction whose update lies in the row space — $T(v) - v \in \mathcal{R}(A^\top)$, equivalently $P_0 T(v) = P_0 v$, as holds for $\Pi_y^\lambda$. Then for a reconstruction with pre-correction row-error share $s$, $T$ cannot raise PSNR by more than $-10\log_{10}(1-s)$; the bound is attained by the hard projection under noiseless $y$, is not attained under noisy $y$ (the projection then imports $\|A^\dagger\varepsilon\|^2$ of row error), and the soft audit with $\lambda>0$ realizes a strictly smaller gain whenever visible row-space error survives the contraction.* Corrections that *edit the null space* while restoring consistency — DDNM-style updates, the dial of §6, and the audited pipelines of §8.4 that end $3$–$8$ dB above their range ceilings — are excluded by the hypothesis, and that exclusion is the content: only null-space injection can exceed the row-space budget, and nothing about it is certified by the measurement.

The interpretation is the separation law in one line: **auditing can only buy image-quality improvement from the part of the error that lies in the measurement-visible row space.** A reconstruction whose remaining error is mostly null-space prior content has $s \approx 0$, so its PSNR ceiling is near zero — the audit can drive the measurement residual to machine precision while barely touching PSNR. Trained networks sit in exactly this corner: they already have small row-error share, so re-imposing consistency is nearly free in PSNR but decisive in accountability. The two axes are decoupled precisely because $A P_0 = 0$.

This decoupling is visible in the audited networks of §4. The learned Rad-5 output moves from $22.192$ to $22.206$ dB — a change of $0.0136$ dB — while its relative measurement error falls from $3.68\times10^{-2}$ to $1.90\times10^{-6}$; the Scr-5 output moves $22.146 \to 22.185$ dB ($0.0387$ dB) while the residual falls $1.80\times10^{-2} \to 1.80\times10^{-5}$. Accountability changes by orders of magnitude; PSNR changes in the third decimal.

### 5.3 The range-share law tracks the sampling rate

The row-error share $s$ governs a *reconstruction's* PSNR headroom. A companion quantity, the row-space energy share $\rho = \|P_R x\|_2^2 / \|x\|_2^2$ of the *image itself*, governs the anchor ceilings that any measurement-consistent estimator can reach. The two use the same range–null orthogonality but answer different questions, and we keep them distinct: $\rho$ explains how much of the scene's energy is measurable at a given operator, while $s$ controls the audit's PSNR budget for a specific reconstruction.

Under the per-image mean-removed convention, $\rho$ tracks the sampling rate almost exactly: $\rho = 0.050$ at Rad-5 and $0.052$ at Scr-5 (5% sampling), rising to $0.101$ at Rad-10 and $0.099$ at Scr-10 (10% sampling). The row-space share is, to leading order, the fraction of dimensions the operator measures. The corresponding row-space PSNR ceilings ($14.304$, $14.311$, $14.541$, $14.534$ dB) are met by scrambled-Hadamard back-projection but not by Rademacher back-projection, whose DC/global-mean coverage differs. The lesson is that $\rho$ is set by the operator's sampling geometry, not by the reconstruction — reinforcing that the null space, which carries the remaining $1-\rho$ of the energy, is where a prior must act and where the measurement cannot follow.

### 5.4 What accountability catches that PSNR cannot

The separation law is not merely a bookkeeping identity; it is the reason a quality metric can pass while the reconstruction has quietly stopped honoring the data. We run three probes with two distinct roles. The first two (wrong measurements, coordinate shuffle) are *prerequisites*: they verify the trained reconstructor genuinely conditions on the record — there the large PSNR drops are the point. Only the third (operator drift) is the separation exhibit proper: quality passes while accountability fails.

**Wrong measurements.** To test whether a trained reconstructor uses the recorded bucket data or merely emits a plausible prior sample, we feed each image another image's measurement vector $y$ (a batch roll) or shuffle the measurement coordinates, on 500-image probes. Across Rad-5, Scr-5, Rad-10, and Scr-10, wrong-$y$ inputs reduce PSNR by $12.174$–$14.793$ dB and shuffled-$y$ inputs by $14.537$–$17.026$ dB. The large drops confirm the network genuinely conditions its output on the recorded measurement rather than acting as a pure dataset prior. (This dependence is not a contradiction of the row-null geometry: the linear operator constrains only $P_R x$, but a *trained conditional* reconstructor uses $y$ to choose *which* null-space completion to emit; the collapse shows the conditioning is real, not that the geometry constrains the null space.)

**Coordinate shuffle.** The shuffled-$y$ arm above is a stronger perturbation than the batch roll, and the larger PSNR drops ($14.5$–$17.0$ dB versus $12.2$–$14.8$ dB) confirm that destroying the coordinate structure of the measurement degrades the reconstruction more than substituting a coherent but wrong record.

**Operator drift.** The most incisive test attacks accountability while leaving quality nearly untouched. In a simulation-scoped calibration-mismatch probe, the audit is performed with a *drifted* operator and the residual is then evaluated against the true operator. As the relative drift grows from $0$ to $0.05$, the Rad-5 post-audit PSNR moves only from $22.206$ to $22.178$ dB — a change of $0.028$ dB — while the residual against the true operator rises from $1.90\times10^{-6}$ to $4.88\times10^{-2}$, more than four orders of magnitude. The Scr-5 case behaves identically: PSNR $22.185 \to 22.155$ dB while the true-operator residual rises $1.80\times10^{-5} \to 1.26\times10^{-2}$. Drift silently destroys the contraction that the certificate certifies, yet PSNR barely registers it. A quality-only pipeline would report success; the accountability audit reports the mismatch.

### 5.5 Statement of the separation law

Taken together, Sections 5.1–5.4 establish the **separation law**: because $\|e\|_2^2 = \|P_R e\|_2^2 + \|P_0 e\|_2^2$ and $A P_0 = 0$, image quality and measurement accountability are coupled *only* through the row-space error share $s$, and are otherwise free to move independently. Quality metrics are blind to everything in the null space and to any accountability failure that does not first show up as row-space error — precisely the failures (wrong measurements, coordinate shuffles, operator drift) that the certificate is built to catch. This is why a reconstruction can score well and still be unaccountable, and why the two ledgers must be reported side by side: a high PSNR never implies the measurement was honored, and honoring the measurement never implies high PSNR. The separation is the empirical face of the geometry, and it sets up the sharper question of §6 — whether, having certified the row space and quarantined the null space, we can now supply detail into that null space safely and by rule.

---

## Appendix C. The dial: protocol, sweep, and lemma detail


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

From the VQAE branch onward the ordering is strictly monotone in every column (the LMMSE anchor, which precedes the dial, is dominated by VQAE on every axis). LPIPS falls from $0.404$ (LMMSE) through $0.300$ (VQAE) to $0.172$ (full VQGAN), and RAPSD — a radially-averaged power-spectral measure of how closely the reconstruction's spectrum matches natural-image statistics (cf. Durall et al. 2020) — improves in lockstep from $0.0041$ to $0.0019$; over the same range PSNR declines from $23.13$ dB (VQAE, the pixel-fidelity optimum) to $21.43$ dB and SSIM from $0.657$ to $0.571$. This is the geometric content of the consistency theorem made empirical: because all null-space content is confined to $P_0$, each increment of $B$ trades measured pixel fidelity for perceptual realism along a smooth curve, and does so at fixed measurement fidelity. Every row of Table 1 satisfies $A\hat{x} = y$ to $\mathrm{RelMeasErr}\sim10^{-7}$; the ladder moves entirely within the null space the bucket cannot see.

The balanced point sits at a deliberately chosen interior of this ladder. It captures roughly three-quarters of the LPIPS improvement available between VQAE ($0.300$) and full VQGAN ($0.172$) — landing at $0.202$ — and the bulk of the RAPSD gain, while holding PSNR within half a decibel of the reconstruction-optimal VQAE. It is the operating point we pre-register and confirm below; quality-lite is offered as a second registered point for applications that tolerate a larger distortion cost in exchange for further perceptual gain. Figure 6 places the named operating points on the perception–distortion plane, and Figure 7 shows the full 21-point sweep.



### 7.2 A pre-registered, one-shot locked confirmation

Because $B$ is a free knob and LPIPS is the metric it is chosen to improve, a naive report would risk selecting the operating point on the same data used to evaluate it. The dial is therefore *locked*: the protocol fixes every degree of freedom before the confirmatory data are touched, and the confirmatory run is executed exactly once.

**Protocol.** The task is low-rate ghost imaging on $64\times64$ grayscale STL10 ($n = 4096$) at a single fixed operator: $m = 205$ rows (5.0% sampling), composed of 1 DC term, 128 low-frequency DCT rows, 56 low-sequency Hadamard rows, and 20 random rows, orthonormalized and generated from seed 772001.

Data are split by raw-SHA256 deduplication into train (20,000, for the anchor and priors), validation (512, used *only* to select $B$), development (512, mechanical scoring), and a locked set (512, one-shot confirmation). The locked set is raw-hash disjoint from the union of 60,497 previously consumed STL10 hashes — overlap $0$, intra-duplicates $0$ — so the confirmation is on genuinely unseen images rather than a re-scored development set. The balanced weight is frozen per seed at $B = \{0.55, 0.55, 0.50\}$ on validation, before the locked split is read.

An 8-condition acceptance gate is fixed in advance: (1) relative LPIPS gain $\ge 5\%$; (2) the LPIPS bootstrap confidence interval strictly below zero; (3) at least 2 of 3 seeds agree in direction (achieved: 3/3); (4) PSNR cost within $0.5$ dB; (5) RMSE cost within $0.005$; (6) RAPSD not worse; (7) mean $\mathrm{RelMeasErr} \le 10^{-5}$; (8) the raw-hash disjointness audit passes. Confidence intervals are formed by averaging the per-image fusion-minus-VQAE deltas over seeds and bootstrapping over images (2000 resamples).

The one-shot property attaches to the pre-registered *gate evaluation*: the eight conditions were evaluated exactly once on the locked split. Later development-level analyses (the noise sweep of §8.2 and the breadth analysis of §8.3) re-read the same locked images with the dial frozen and *no further selection of any kind*; they postdate the gate, change nothing upstream of it, and are labeled development-level precisely because the split's one-shot statistical guarantee does not extend to them.

**The locked result.** Against the VQAE structure branch, balanced fusion improves perceptual quality by
$$\Delta\mathrm{LPIPS} = -0.0977,\qquad \text{CI }[-0.1016,\,-0.0940],$$
a $32.6\%$ relative reduction whose interval lies entirely below zero. The cost is bounded and pre-registered: $\Delta\mathrm{PSNR} = -0.45$ dB (within the $0.5$ dB tolerance) and $\Delta\mathrm{RMSE} = +0.0039$ (within the $0.005$ tolerance), while spectral realism *improves* ($\Delta\mathrm{RAPSD} = -0.00030$, a move toward the natural-image spectrum). All 3/3 seeds agree in direction, and measurement fidelity is held at numerical precision (mean $\mathrm{RelMeasErr} = 3.6\times10^{-7}$, max $5.7\times10^{-7}$), so the perceptual gain costs nothing in measurement accountability. All 8/8 gate conditions pass.

**Replication on the same-drawn development set.** The development split, scored mechanically before the locked run, gives $\Delta\mathrm{LPIPS} = -0.0965$ ($32.9\%$) and $\Delta\mathrm{PSNR} = -0.43$ dB. The development and locked effect sizes nearly coincide ($-0.0965$ versus $-0.0977$; $32.9\%$ versus $32.6\%$), so the confirmation reflects a stable property of the method rather than selection on the evaluation set.

**Distribution-level corroboration (post-hoc; not among the eight gate conditions).** Per-image LPIPS measures distance to each image's own reference; kernel Inception distance (KID) instead compares each method's *distribution* of reconstructions to the natural-image distribution. KID follows the same ladder and confirms the gain at the distributional level: balanced fusion improves locked-split KID from $0.119$ (VQAE) to $0.043$, a $2.7\times$ reduction, with quality-lite and full VQGAN closer still to the natural-image manifold. The injected null-space detail therefore makes the reconstructions distributionally more natural, not merely better on a single per-image metric. Figure 8 shows fixed locked-split samples across the ladder.


### 7.3 The dial is a fixed scalar, not a per-image oracle

Two properties keep the dial honest, and both are matters of construction rather than tuning.

First, the frontier is well-behaved. A dense 21-point sweep of $B$ (step $0.05$) on the development split is smooth, with LPIPS *strictly* monotone: it falls steadily from $0.293$ at $B = 0$ to $0.167$ at $B = 1$, with no discontinuity and no interior perceptual optimum a single validation-selected $B$ could miss. PSNR declines from $22.87$ dB to $21.17$ dB and full-RMSE rises from $0.075$ to $0.091$ over the same sweep, near-monotonically — PSNR exhibits one shallow interior maximum ($22.90$ dB at $B = 0.10$, a $+0.03$ dB bump over the $B=0$ endpoint) before its steady decline, a negligible non-monotonicity we report for completeness. The dial is a stable interpolation, not a fragile hyperparameter.

Second, the *simplest* dial is the confirmed one. We evaluated two richer parameterizations in development — a 16-band radial-frequency weighting (a separate weight per spatial-frequency band) and a learned per-image gate — and neither beat the single global scalar under the registered validation-selection rule. Both tolerance-bounded rules (balanced and quality-lite) select the plain scalar across all three seeds; only an *unconstrained* oracle that ignores the PSNR and RMSE tolerances prefers a low-pass-cutoff variant, and its margin over the scalar is marginal. The confirmed result is thus the minimal one: a single measurement-safe scalar, not a learned gate or a frequency-dependent weighting.

We are deliberate about what $B$ is not. It is a fixed, validation-selected operating point held constant per seed — not adapted per image, and not an oracle that knows the best weight for each scene. A per-image adaptive weight might improve the trade, but it lies outside this frozen-system result, and the learned per-image gate we tried did not beat the global scalar. The dial meters how much prior-supplied detail enters the null space; it does not, and cannot, certify that the detail it admits is the true scene. That boundary is the subject of the two-ledger synthesis (§9): the perceptual win reported here lives entirely on the quality axis, and the exact consistency $A\hat{x}_B = y$ that accompanies it certifies reproduction of the record, never the correctness of what the record cannot see.

The scalar $B$ is the closest interface in this literature to CodeFormer's fidelity–quality knob $w$ (Zhou et al. 2022) and ESRGAN's network-interpolation weight $\alpha$ (Wang et al. 2018); the difference is that those knobs act on entangled features, so higher quality genuinely moves the reconstruction off the measurement manifold, whereas $B$ acts only within $P_0$ and holds $A\hat{x}_B = y$ exact at every setting. The honesty dial is thus a perception–distortion control (Blau and Michaeli 2018) whose distortion is paid entirely in null-space coordinates the measurement never certified.

### 7.4 The no-adaptation lemma and the audit tax

**Lemma (no ground-truth-free per-record adaptation).** *Let $g$ be any selection rule computable from the operator and the record alone — a fusion weight, a gate, or arbitrary null-space content, $g:(A, y) \mapsto g(A,y)$. Then $g$ is constant on each measurement fiber $\{x : Ax = y\}$; consequently the reconstruction $\hat{x} = x_0(y) + P_0\, g(A,y)$ is identical for every scene on the fiber, and the closable part of the per-image oracle gap is limited to the component of the oracle weight $B^\star(x)$ that is predictable from $y$.*

*Proof.* $g$ depends on $x$ only through $y = Ax$; two scenes on one fiber produce the same $y$, hence the same $g$ and the same reconstruction. The oracle weight depends on the unobserved $P_0 x$; a rule matching it beyond its $y$-predictable component would distinguish fiber-mates, contradicting constancy. $\blacksquare$

The lemma is elementary — the converse of §3 restated for selection rules — and we claim it as a clarification, not as mathematical depth: it is the measurability fact the per-image-adaptation literature keeps paying for ignoring.

What it does *not* rule out is a $y$-adaptive dial $B(y)$ capturing whatever of $B^\star$ correlates with the measured component. Whether that component is worth anything is an empirical question, and on this benchmark the answer is no [development-level]: a feature-based per-image selector — null energies $\|d_A\|, \|d_G\|$, the chord and cosine between the residuals, anchor texture statistics, cross-arm perceptual distances — trained on validation and tested on development recovers the oracle only through a *global* shift of $B$: its excess over a global scalar matched to its own mean PSNR is $\le 0.002$ LPIPS, a deliberately constant predictor reproduces it exactly, and the features barely rank-correlate with the oracle weight ($|\rho| \le 0.24$). The scalar dial is thus honest twice over — exactly, because nothing computable from one record can adapt across a fiber; empirically, because the $y$-predictable component of the oracle is worth essentially nothing here.

We scope the lemma explicitly: it concerns ground-truth-free selection on one fixed operator and one record. Frameworks that do recover null-space information import exactly what the lemma says is missing — structure beyond the single record — as in equivariant imaging and its sensing theorems (Chen et al. 2021; Tachella et al. 2023), where group actions or operator diversity convert the null content of one record into the range content of another.

**Remark (the audit tax).** The lemma also prices its own escape: change the fiber. Hold out $q$ extra bucket rows, never used for reconstruction; the holdout residual estimates the entire $(n-m)$-dimensional null error of any measurement-consistent candidate, and $q = O(\varepsilon^{-2}\log|\mathcal{K}|)$ rows suffice to rank $|\mathcal{K}|$ candidates. This is compressed-sensing cross-validation, verbatim (Ward 2009; Boufounos et al. 2007) — we run no new experiment and claim no novelty for the estimator. The allocational reading is the point: spent on sensing, $q = 32$ rows shrink this null space by ${\sim}0.8\%$; spent on auditing, they estimate the whole null ledger. Testing is cheaper than sensing — but by the oracle ceilings above, the tax buys *falsification* of a hypothesized completion, not quality: even a ground-truth per-image dial choice under the registered tolerance is worth only ${\sim}0.004$ LPIPS here.

---

[FIG: B_CURVE.pdf | 0.85 | The dense 21-point sweep of the dial: LPIPS strictly monotone from $0.293$ to $0.167$; PSNR declines from $22.87$ to $21.17$ dB with one shallow interior maximum ($+0.03$ dB at $B=0.10$).]

### Robustness studies in full


The locked claim of §7 fixes one operating point: the balanced dial $B$, at a single $5.0\%$ operator, on one $64\times64$ resolution, under noiseless acquisition. This section asks how far that operating point reaches — across sampling rate, across measurement noise, and across the image population. All results here are **development-level**: they reuse the frozen system, re-select nothing on the locked split, and constitute supplementary evidence rather than a second pre-registered claim. We label them as such throughout and treat none of them as certified. The measurement-consistency guarantee, however, is not development-level: because $A P_0 = 0$ holds for every operator by construction, $A\hat{x}_B = y$ remains exact at every rate and every noise level below.

### 8.1 Cross-rate generalization (development-level)

To test whether the balanced-fusion advantage is specific to the $5\%$ operator, we reuse the rate-agnostic priors unchanged and retrain only the lightweight anchor refiner at $2\%$ ($m=82$) and $10\%$ ($m=410$), then run the identical pipeline: select the global scalar $B$ on validation under the same tolerance rule, and score on the held-out development split (3 seeds per rate). This does not re-touch the frozen $5\%$ locked result, which appears only as the anchor point.

The advantage holds at every rate. Balanced fusion lowers LPIPS relative to the VQAE branch by $-0.116$ ($29.3\%$) at $2\%$, $-0.098$ ($32.6\%$) at $5\%$, and $-0.076$ ($34.2\%$) at $10\%$, with all $3/3$ seeds agreeing in direction at each rate and a PSNR cost of $-0.39$, $-0.45$, and $-0.43$ dB respectively — below the pre-registered $0.5$ dB validation tolerance throughout. The relative gain grows mildly with sampling rate: at higher rates more genuine detail is recoverable for the fusion to exploit, which is consistent with the fusion mechanism rather than an artifact of a single operator.

**Table 2. Cross-rate generalization (development-level, 3 seeds per rate).**

| Sampling rate | $\Delta$LPIPS (balanced $-$ VQAE) | Relative gain | $\Delta$PSNR (dB) | Seeds same-direction |
|---|---|---|---|---|
| $2\%$ ($m=82$) | $-0.116$ | $29.3\%$ | $-0.39$ | $3/3$ |
| $5\%$ ($m=205$, *locked*) | $-0.098$ | $32.6\%$ | $-0.45$ | $3/3$ |
| $10\%$ ($m=410$) | $-0.076$ | $34.2\%$ | $-0.43$ | $3/3$ |


### 8.2 Noise robustness (development-level)

The locked result is noiseless. To probe robustness we add i.i.d. Gaussian noise of standard deviation $\sigma_\varepsilon$ to the bucket measurements and re-run the frozen system with the balanced $B$ unchanged (3 seeds, locked split). Two behaviors emerge. First, VQAE is the most noise-stable branch but always the least perceptual. Second, and more instructive, full VQGAN degrades sharply as noise grows — its fine synthesized detail amplifies the measurement noise — so that balanced fusion, which the noiseless ladder places *above* full VQGAN, **overtakes it at $\sigma_\varepsilon = 0.02$** (LPIPS $0.197$ for balanced versus $0.204$ for full VQGAN) and beats it decisively at $\sigma_\varepsilon = 0.05$ ($0.250$ versus $0.293$).

**Table 3. Noise robustness, LPIPS by bucket-noise level (development-level, 3 seeds, locked split, $B$ frozen).**

| Bucket noise $\sigma_\varepsilon$ | VQAE | Balanced | Full VQGAN |
|---|---|---|---|
| $0.000$ | $0.300$ | $0.202$ | $0.172$ |
| $0.005$ | $0.299$ | $0.199$ | $0.172$ |
| $0.010$ | $0.297$ | $0.195$ | $0.176$ |
| $0.020$ | $0.295$ | $0.197$ | $0.204$ |
| $0.050$ | $0.304$ | $0.250$ | $0.293$ |


This is the behavior a controlled interior operating point should have: balanced fusion keeps most of VQGAN's perceptual benefit at low noise while degrading far more gracefully as the measurement becomes unreliable. The crossover is not evidence that noise certifies anything about the null space — the guarantee $A\hat{x}_B = y$ concerns only the row space at every $\sigma_\varepsilon$ — but it does show that the recommended dial position is also the robust one.

### 8.3 Breadth of improvement and its failure mode

The gain is broad, not anecdotal. Balanced fusion improves LPIPS on $99.0\%$, $97.5\%$, and $99.2\%$ of locked images for the three seeds respectively, worsening it on at most $13$ of $512$ images, with a worst-case regression of only $+0.07$ LPIPS and most regressions far smaller. The failures are not random: they concentrate on **man-made periodic and edge structure** — fences, vehicle body panels, airplane fuselages — where the natural-image VQGAN prior is a mismatch and synthesizes texture that conflicts with the regular geometry the scene actually contains. This is the expected and interpretable failure mode of a natural-image prior applied to structured man-made content, and it delimits where the dial should be advanced with care: the reach of responsible injection is bounded by the support of the prior, not by the measurement.

[FIG: rate_generalization_figure.pdf | 0.85 | Cross-rate generalization (development-level): the balanced dial's relative LPIPS gain at 2%, 5% (locked), and 10% sampling.]

[FIG: NOISE_ROBUSTNESS.pdf | 0.85 | Noise robustness (development-level): balanced fusion degrades gracefully and overtakes full VQGAN at $\sigma_\varepsilon = 0.02$.]

## Appendix D. Extended related work

Our work sits at the intersection of computational ghost imaging, the range–null geometry of undersampled linear inverse problems, generative image priors, and the recent literature on hallucination, fundamental limits, and trustworthy reconstruction. What organizes the discussion below is a single distinction: almost every prior line uses the range–null decomposition, or a prior over the unmeasured content, to **reconstruct** — to produce one image that looks right or scores well. We instead use the same geometry to **certify, bound, and meter**: to prove what the bucket measurement can and cannot vouch for, to audit that accountability at test time without ground truth, and to inject admittedly invented detail only where the measurement is provably blind. Each subsection first summarizes the line, then states our delta.

### 10.1 Ghost imaging and single-pixel imaging

Ghost imaging originated as a quantum-optical curiosity: Pittman et al. (1995) recovered a magnified aperture image purely from photon-coincidence correlations, with no image present in either detector's individual counts. Gatti et al. (2004) then proved analytically that classically correlated thermal beams reproduce all the imaging features of entangled ghost imaging, and Ferri et al. (2004) demonstrated both ghost image and ghost diffraction from a single classical thermal source by altering only the reference arm. Shapiro (2008) and Bromberg et al. (2009) collapsed the modality to its computational core: with the illumination patterns known, a single bucket detector suffices, so ghost imaging is classical coherence propagation and the reconstruction is a correlation $C(\rho) = \sum_m \phi_m(\rho)\, y[m]$ against known patterns — precisely our forward operator $y = Ax$. In parallel, the single-pixel camera (Duarte et al., 2008) and compressive ghost imaging (Katz et al., 2009) established the undersampled regime $y = \Phi x$ with $m \ll n$, recovered by $\ell_1$/TV minimization, and stated the governing fact plainly: since $m < n$, infinitely many images satisfy the same measurements, and recovery is possible only under a prior. Gibson et al. (2020) survey this progression from correlation through compressed sensing to learned reconstruction; Song et al. (2025) update it and flag interpretability and generalization as unresolved.

**Our delta.** These works establish the physical instantiation of our operator $A$ and the undersampled regime that makes its null space non-trivial. We take the "infinitely many $x$ satisfy the same $y$" observation — usually stated in passing to motivate a sparsity prior — and elevate it into the load-bearing object of the paper: a constructive, per-instance feasible-but-wrong witness and a test-time certificate over exactly this measurement model.

Learned ghost-imaging reconstructors sharpen the stakes. Lyu et al. (2017) restored recognizable images at sampling ratios as low as $\beta = 0.05$, where compressed sensing collapses, and candidly noted the outputs "do not resemble exactly the ground truth"; the CNN pipeline of He et al. (2018) and the U-Net denoiser of Shimobaba et al. (2017) buy large PSNR/SSIM gains with no enforced measurement consistency; and the X-ray study of Huang et al. (2024) argues rigorously that reducing the number of measurements does not itself add information, so any sub-sampling-limit detail a learned reconstructor supplies originates from the prior, not the measurement.

**Our delta.** This GI-native line is the empirical hook for our converse: at low $\beta$ the network fills the null space with trained-in structure the measurement cannot certify. Prior work optimizes quality (MSE, recognizability, PSNR); we separate quality from accountability, audit the learned reconstructor alongside classical ones, and never claim the bucket certifies the invented texture.

For breadth of the modality itself we point the reader to Edgar, Gibson & Padgett's (2019) survey of single-pixel imaging, whose operator abstraction is the one this paper audits.

### 10.2 Range–null and data-consistency reconstruction

A large body of work exploits the range–null decomposition $x = P_R x + P_0 x$ with $P_R = A^\dagger A$, $P_0 = I - A^\dagger A$, and $A P_0 = 0$. The classical antecedents of "edit freely, keep the measurement exact" are the alternating-projection extrapolators — Gerchberg (1974) and Papoulis (1975) for band-limited completion, generalized by Youla & Webb (1982) — which already alternate a prior step with a hard projection onto $\{Hx = y\}$. Its modern deep-learning-era formulation is IDBP (Tirer & Giryes, 2018), which alternates denoising with a backward projection so that all restoration dynamics live strictly in the null space. Schwab et al. (2019) formalized the null-space network $L = \mathrm{Id} + (\mathrm{Id} - A^+A)N$, proving it is a convergent regularization that edits only $\ker(A)$ while preserving data consistency exactly, and Chen & Davies (2020) trained separate range- and null-space networks over the same $P_R = H^\dagger H$, $P_0 = I - H^\dagger H$ split. In imaging practice, hard data-consistency layers (Schlemper et al., 2018), model-based unrolling with an explicit $(A^H A + \lambda I)^{-1}$ solve (Aggarwal et al., 2019), variational networks (Hammernik et al., 2018), learned primal–dual (Adler & Öktem, 2018), ADMM-Net (Yang et al., 2016), and ISTA-Net (Zhang & Ghanem, 2018) all restore or preserve the measured component while a learned prior supplies the rest. The same construction anchors the most recent null-space methods: RND-SCI (Wang et al., 2023) reconstructs hyperspectral snapshots as $x = \Phi^\dagger y + (I - \Phi^\dagger \Phi)q$ with a conditional network generating only the null term; NPN (Jacome et al., 2025) trains a network to predict a low-dimensional null-space projection $Sx^*$ from $y$, explicitly noting that data fidelity leaves the null space uncontrolled; and GSNR (Gualdron-Hurtado et al., 2026) builds a graph-smooth basis for $\mathrm{Null}(H)$ via the null-restricted Laplacian $T = P_n L P_n$ and proves minimax coverage/predictability bounds for the invisible directions, gaining up to 4.3 dB.

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

A sobering literature documents that consistency and visual quality decouple from correctness. Antun et al. (2020) showed state-of-the-art learned MRI/CT reconstructors are unstable to near-invisible perturbations and can erase small structural detail while stable compressed-sensing baselines are not; Genzel et al. (2020) offered the balancing counterpoint that, under matched adversarial testing, standard networks match a robust TV benchmark and much instability traces to noiseless-training "inverse crimes." Buğday et al. (2026) inserted diagnostically misleading anatomical hallucinations via imperceptible k-space perturbations that PSNR/NRMSE/SSIM cannot detect. Bhadra et al. (2021) formalized hallucinations by projecting a reconstruction onto measurement and null spaces, showing null-space hallucinations are attributable solely to the prior and cannot be assessed without the ground truth; Gottschling et al. (2025) prove kernel-aware no-free-lunch theorems and construct explicit kernel-element perturbations that destabilize learned maps — the nearest formal ancestors of our witnesses, which differ in kind (cross-class semantic splices on the governed dial's own operator, matching the record tighter than the noisy truth) but not in geometry; DynamicDPS (Kim et al., 2025) adopts the same intrinsic (data-inconsistent) versus extrinsic (null-space) split and suppresses both with a diffusion prior plus data consistency; and HalluGen (Kim et al., 2025) fabricates controllable hallucinations by gradient ascent to train reference-free detectors, noting that consistency-preserving errors are the hardest to catch. Learned primal–dual (Adler & Öktem, 2018) even internally observes that a true feature is indistinguishable from a false feature of the same size and contrast. The fundamental-limits side is anchored by Iagaru et al. (2026), who prove necessary-and-sufficient conditions for detail-transfer hallucination in (possibly nonlinear) inverse problems — consistent decoders can only hallucinate details that are almost invisible in measurement space ($\lVert f(x + x_{\det}) - f(x)\rVert \le 2\epsilon$) — and give ground-truth-free, forward-model-only algorithms that bound and assess hallucination magnitude via feasible-set diameters (worst-case kernel size).

**Our delta — the limits-and-assessment line (position against).** This is the line we must not overclaim against, and we cite it as the anchor for the barrier itself. Antun et al. (2020), Buğday et al. (2026), and Bhadra et al. (2021) establish empirically and by SVD decomposition that consistency and quality metrics do not certify correctness; Iagaru et al. (2026) prove the fundamental limit and occupy the generic slot of forward-model-only, ground-truth-free hallucination *assessment*. We do not claim "null content is unverifiable" or "ground-truth-free assessment" as novel — that ground is theirs. Our contribution is on four axes they do not cover. First, where Iagaru et al. give feasible-set-diameter *bounds* (a sup over feasible sets, requiring paired data), we give an **exact** per-mode contraction $\lambda/(\lambda+\sigma_i^2)$ from the operator's SVD, applied as a test-time plug-in audit across BP, Tikhonov, CS-TV, and learned reconstructors. Second, where they establish *existence* via necessary-and-sufficient conditions, we give a **constructive** per-instance feasible-but-wrong witness that matches the record to machine precision. Third, we frame and demonstrate a **quality-versus-accountability separation** (PSNR essentially flat while measurement residuals drop orders of magnitude) as an explicit protocol. Fourth, and entirely uncontested, we move from *assessment* to *governance*: a constructive, metered null-space injection dial with $A\hat{x}_B = y$ exact for every $B$ and a pre-registered locked gate. The clean axis is assessment versus governance, exact versus bound, constructive versus existential. Bhadra et al.'s (2021) null-space hallucination map requires the ground truth — exactly the quantity our converse proves unverifiable — so we cite it to say the null-space error is real but cannot be certified at test time.

### 10.7 Uncertainty, conformal prediction, and trust

The standard response to unreliable reconstruction is uncertainty quantification. Kendall & Gal (2017) established the aleatoric/epistemic taxonomy; Narnhofer et al. (2021) produce Bayesian epistemic-uncertainty maps for variational MRI. On the distribution-free side, Angelopoulos et al. (2022) endow any image-to-image regressor with per-pixel risk-controlling prediction intervals to flag hallucinations, conformal risk control (Angelopoulos et al., 2022) guarantees $\mathbb{E}[\mathrm{loss}] \le \alpha$ for bounded monotone losses, Wen et al. (2024) conformalize the downstream task output rather than per-pixel maps, and Lu et al. (2022) apply ordinal conformal sets to disease severity rating. Wen et al. (2024) in particular argue, as we do, that per-pixel uncertainty maps are the wrong object because they miss many-pixel hallucinated structure.

**Our delta.** These guarantees are either heuristic (Bayesian std maps) or ground-truth-dependent at calibration (RCPS, conformal risk control, task-conformal), and they operate in output/pixel/label space. Our certificate is deterministic, ground-truth-free at test time, and lives in the operator's spectral geometry: it contracts each *measured* mode by exactly $\lambda/(\lambda+\sigma_i^2)$ and audits measurement accountability rather than pixel error or a labeled risk. Crucially, using Kendall & Gal's own taxonomy, our accountability is neither aleatoric nor epistemic — it is a third, geometry-determined quantity: null-space unverifiability persists even with a perfect model in the noiseless limit, because $A P_0 = 0$ makes null content invisible to $y$ regardless of noise or model confidence. No calibration set can fix what the converse proves.

### 10.8 Cross-field framings: identifiability, information theory, and Bayesian inversion

Our thesis has deep provenance outside imaging. Econometric identification theory (Koopmans & Reiersøl, 1950) established that observationally equivalent structures generate the identical distribution, so a characteristic is knowable only if invariant across all such structures — identification precedes estimation. Rothenberg (1971) tied local identifiability to nonsingularity of the Fisher information matrix, making non-identifiability a rank deficiency / null space in information geometry. Manski (2003) founded partial identification — when data do not point-identify a parameter they confine it to a sharp set, and honest inference reports that set — and Lewbel (2019) formalized set identification and *normalizations*: restrictions that select within the identified set without altering any meaningful quantity. On the imaging side, Landau (1967) fixed the sampling-density threshold below which degrees of freedom exceed measurements and a non-trivial null space is forced by necessity, Candès, Romberg & Tao (2004) proved exact CS recovery under sparsity while noting that without the prior distinct signals share identical partial measurements, and Stuart (2010) cast inverse problems as Bayesian posteriors in which, when data dimension is below the unknown, the prior remains decisive even in the zero-noise limit. Within the imaging sciences proper the lineage is older and more direct: Backus & Gilbert (1968–1970) characterized exactly which linear functionals of an underdetermined model are resolved by finite data — a functional is determined iff its representer lies in the row space, our certifiable/uncertifiable split for linear observables — and Barrett & Myers (*Foundations of Image Science*, 2004) built the estimability and null-function calculus, and task-based image-quality assessment, on the identical range/null decomposition. MacKay (1992) is the Bayesian antecedent of the certificate itself: his effective number of well-determined parameters $\gamma = \sum_i \lambda_i/(\lambda_i+\alpha)$ sums our per-mode contraction over the spectrum (§4.4).

**Our delta.** These fields supply the rigorous vocabulary we import and instantiate, and we own the lineage rather than reinvent it. Our feasible-but-wrong images are the imaging analogue of Koopmans & Reiersøl's observational equivalence and Rothenberg's singular information direction; our null space $P_0 = I - A^\dagger A$ is Manski's identification region, Landau's forced kernel, and Barrett & Myers's null-function subspace; our certifiable-functional split is Backus–Gilbert resolvability; and, most sharply, our governed dial is Lewbel's *normalization* — injecting null content is "without loss of generality" for the measurement, since $A\hat{x}_B = y$ exactly for every $B$, leaving the identified quantity untouched. Candès–Romberg–Tao's converse is the CS-side twin of the econometric one, and Stuart's "prior stays decisive" is the Bayesian mirror of our per-mode contraction, whose factor $\lambda/(\lambda+\sigma_i^2)$ is exactly a Gaussian-posterior variance reduction along each singular direction — the same term MacKay aggregates into $\gamma$, which we instead read per record as a test-time audit. Our contribution is not a new scalar on $(A,\sigma)$: on the linear-Gaussian layer these objects are settled, and any quantity defined there is a renaming of one of them. It is the *constructive converse*, the repurposing of the modal contraction as a ground-truth-free per-record certificate that separates quality from accountability, and the governed injection dial — an operational stack, unified for undersampled ghost imaging, that these prior framings describe in pieces but never assemble as a governance instrument. The **no-adaptation lemma** (§7.4) is, as mathematics, an elementary measurability fact; we state it formally because the reconstruction literature keeps paying for its absence — per-image adaptation schemes that cannot, even in principle, out-adapt their own global prior on a fixed record — and because its value lies in the pairing with the empirical finding that the $y$-predictable remainder is worth essentially nothing here.

### 10.9 Novelty statement

Every line above uses the range–null geometry, or a prior over the unmeasured content, to **reconstruct**: install the "correct" prior in the null space for fidelity (Wang et al., 2023 AAAI; RND-SCI; DDNM; MCG), learn to predict null content from $y$ (NPN; GSNR; Chen & Davies, 2020), fabricate null-like error to benchmark detectors (HalluGen), quantify output uncertainty against calibration data (RCPS; conformal risk control; Wen et al., 2024), or prove abstractly that the kernel forces hallucination (Iagaru et al., 2026; Bhadra et al., 2021; Antun et al., 2020). We use the identical geometry to **certify, bound, and meter**, and our contribution rests on four airtight, uncontested hooks. **First**, we make the barrier *constructive*: an explicit feasible-but-wrong witness that matches the same bucket record to machine precision — tighter than the noisy truth — turning "consistency is not correctness" from a theorem about maps (Iagaru et al., 2026) and an SVD diagnostic (Bhadra et al., 2021) into a per-instance object — and where Gottschling et al.'s (2025) kernel constructions establish destabilizing existence, ours are cross-class, record-tight, and displayed.

**Second**, we give a ground-truth-free, reconstructor-agnostic **test-time certificate** that contracts each measured mode by *exactly* $\lambda/(\lambda+\sigma_i^2)$ — a closed-form modal spectrum, not a feasible-set-diameter bound (Iagaru et al., 2026), a soft penalty (NPN; GSNR), a design-time predictability curve (GSNR), or a calibrated statistical interval (Angelopoulos et al., 2022) — auditing BP, Tikhonov, CS-TV, and learned reconstructors uniformly — including, in §8.4, three *published* DL-GI pipelines decomposed on their own released or measured operators — and **separating image quality from measurement accountability**.

**Third**, that separation is demonstrated as an explicit protocol (perceptual/PSNR quality essentially unchanged while measurement residuals fall orders of magnitude), an axis none of the reconstruction or UQ lines report.

**Fourth**, we convert null-space injection from an all-or-nothing fill into a single-scalar, pre-registered **honesty dial** that fuses VQAE structure and VQGAN detail and keeps $A\hat{x}_B = y$ *exact* for every $B$ behind a locked gate — moving the field from *assessment* (does a reconstruction hallucinate?) to *governance* (how much invented detail do we add, where the bucket is provably blind, without ever breaking or laundering the measurement?).

We explicitly do **not** claim SOTA, do not beat diffusion, and never claim the measurement certifies invented texture — the point is that it provably cannot. That inversion — **reconstruct, becomes certify / bound / meter** — is the slot this paper occupies.

---

## Appendix E. Projector forensics: full decompositions

### 8.4 External validity: projector forensics of published pipelines, on their own operators

Everything above is demonstrated on our operator. To test whether the two-ledger reading describes the field rather than our construction, we audited three published DL-GI pipelines — three different learning paradigms on three different operators — using **each paper's own released or measured operator and each paper's own code**, with exact orthogonal attribution $\|\hat{x}-x\|^2 = \|P_R(\hat{x}-x)\|^2 + \|P_0(\hat{x}-x)\|^2$ computed from the SVD of *their* $A$ (Figure 11).


**Physics-enhanced deep learning SPI** (Wang et al. 2022a, *Photon. Res.*; pretrained U-Net + measurement-loss fine-tuning; released learned patterns verified against their shipped record to $1.2\times10^{-7}$). Their fine-tuning trajectory, decomposed per step, is a two-phase object: of the $+4.36$ dB the 300 steps buy, $61.4\%$ of the MSE improvement is row-space repair — exactly what a measurement loss is entitled to fix — and $38.6\%$ is null-space spillover through the network's coupling. By step ${\sim}200$ the row ledger is exhausted ($96.9\%$ of the remaining error is null) and PSNR plateaus. The final $25.12$ dB stands $8.1$ dB above the operator's range ceiling of $16.99$ dB: everything above the ceiling is prior-supplied null content, truth-aligned on the audited scene (alignment ${\approx}0.85$) but, by the converse, unverifiable from the record.

**GIDC** (Wang et al. 2022b, *Light Sci. Appl.*; untrained DIP-style network; $410$ *physically measured* speckle patterns and photon-count buckets). On the experimental scene (no ground truth), $43.8\%$ of the published reconstruction's centered structure lies in the null space of the patterns actually measured — a precise ledger location for the paper's far-field super-resolution claim, since the diffraction-limited speckle bounds the measured subspace and super-resolved content is definitionally null-space content. With a known scene substituted onto the same real patterns at their photon level (their code byte-identical), the gain attribution is $60.3\%$ row / $39.7\%$ null — nearly identical to the fine-tuned pipeline's split — with the untrained prior's null content genuinely moving toward the truth (alignment $0.07 \to 0.65$: GIDC's real merit, quantified), terminating $4.5$ dB above the ceiling at $95.4\%$-null residual error.

**Noise2Ghost** (Manni et al., 2025; self-supervised, and the one genuinely *noisy* record, $\mathrm{rel}\,\varepsilon = 1.2\times10^{-2}$ — the regime where range saturation is non-tautological because in-range denoising is possible). It is real: N2G's row-space error lands $42\%$ *below* the noisy pseudo-inverse, an effect noiseless simulations cannot exhibit. And it is small: in-range denoising accounts for ${\approx}4.9\%$ of the total MSE improvement; $95.1\%$ is null-ledger (null error halved at alignment $+0.44$), with the terminal error $92.7\%$ null.

**Table 4. Forensics summary — each pipeline on its own operator, with its own code.**

| pipeline | operator | headline vs own range ceiling | gain split (row/null) | terminal null share | final consistency |
|---|---|---|---|---|---|
| PEDL, fine-tuned (Wang et al. 2022a) | learned patterns, 25% | $25.12$ vs $16.99$ dB ($+8.1$) | $61/39$ | $96.9\%$ | $2.3\times10^{-2}$ |
| GIDC, untrained (Wang et al. 2022b); known-GT scene | real measured speckle, 10% | $19.54$ vs $15.04$ dB ($+4.5$) | $60/40$ | $95.4\%$ | $0.134$ (experimental scene) |
| Noise2Ghost, self-supervised (Manni et al. 2025) | sim masks, 10%, noisy record | $18.40$ vs $15.26$ dB ($+3.1$) | $5/95$ | $92.7\%$ | noise floor |

**The cross-target regularity, with its base rates.** The *qualitative direction* of these findings is forced by the sampling rate and must be said plainly: at $6$–$25\%$ sampling, most of any scene's energy is null-space energy to begin with (the true scene carries $10.1\%$ of its energy in the learned operator's null space, $17.2\%$ in GIDC's, and the pseudo-inverse reference point has a $100\%$-null terminal error by construction), so "most residual error is null" is partially dimension counting.

What is *not* forced, and constitutes the findings, are the quantitative splits and their agreement: the $61/39$ and $60/40$ row/null attributions of two entirely different optimizations, the $5/95$ split of the self-supervised method whose in-range denoising is real ($42\%$ below the noisy pseudo-inverse) yet small, the truth-alignment trajectories ($0.07 \to 0.65$ for GIDC), and the fact that all three terminate with the row ledger *repaired* — the methods reduce the null share of error below the base rate, and stall exactly when the certifiable ledger is exhausted.

Two caveats bound the audit's own accountability: the GIDC null attribution inherits any calibration error in the released patterns (the operator is measured, not exact), and the attribution is stated relative to the min-norm reference. None of the three pipelines applies an exact consistency projection (residuals $2.3\times10^{-2}$, $0.134$, and noise-floor respectively), so the audit step of §4 is available to each at zero cost.

We stress the reading: this is not an accusation of error — the null content these methods supply is often substantially truth-aligned — it is the *location* of their gains in the two-ledger account, measured on their own instruments. The separation law of §5 describes the field, not just our operator.

---

## Appendix F. Negative results

### 11.4 Abandoned negative result: posterior $z$-sampling diversity

We investigated whether stochastic $z$-sampling could turn the null-space prior into a calibrated posterior over $P_0 x$, and we report it only as a coverage limitation, not as a positive contribution. A deterministic checkpoint collapses: fixed-$y$ pixel std $\approx 10^{-3}$ and $P_0$ variance $1.09\times10^{-6}$ (gate 1). Removing the row-space reconstruction pressure lifts the collapse — the anti-collapse diagnostic reaches $P_0$ variance $1.28\times10^{-3}$ (gate 2) — confirming that the full reconstruction loss, not the architecture, was suppressing feasible null-space variation. But restored spread is not calibrated coverage. The best anchor/diversity scan still reaches only $\approx 45\%$ pixel and $\approx 48\%$ $P_0$ coverage at the nominal $90\%$ level, because the deterministic base estimate is offset from ground truth in the null space (base $P_0$ det-to-GT RMSE $\approx 0.081$): diversity can widen the sample cloud while leaving it centered in the wrong place. Sampling quality is therefore bounded by the base estimator's null-space accuracy, not only by the diversity mechanism. We make **no** posterior-diversity or calibrated-uncertainty claim; the line is retained solely to bound what the dial does and does not deliver.

The prior-content map in detail: on the $64\times64$ Rademacher validation, per-pixel $|P_0\hat{x}|$ and actual null-space error are essentially uncorrelated (Spearman $-0.079$, Pearson $-0.069$ for the LMMSE arm; $\approx 0.07$ for learned arms), and the top-10% highest-magnitude pixels carry *lower* error ($0.071$) than the rest ($0.079$). The map is provenance, not error.
