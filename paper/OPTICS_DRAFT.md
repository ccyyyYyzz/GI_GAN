# A Ground-Truth-Free Measurement-Consistency Audit for Learned Ghost and Single-Pixel Imaging at Low Sampling Ratios

*Author placeholder — Author One, Author Two, Author Three. Affiliation placeholder. Corresponding author: [email placeholder].*

---

## Abstract

Ghost imaging (GI) and single-pixel imaging (SPI) reconstruct a scene from few bucket measurements, attractive for low-light, broadband, and computational imaging. At low sampling ratios, learned reconstructors produce sharp, natural-looking images. However, which content is supported by the measurements and which is supplied by the prior remains unclear, and cannot be assessed when the ground truth is unknown. Here, we report a ground-truth-free measurement-consistency audit (MCA) that, from the sensing operator and recorded signals alone, quantifies the measurement-constrained component of any reconstruction and attributes the remainder to the prior, per record and independent of the reconstructor. Because most degrees of freedom are unmeasured, any record-consistent reconstruction must supply most detail from the unmeasured (null) subspace. Using the audit forensically, we locate the reported gains of published deep-learning pipelines there: null shares of $39$–$95\%$ for three ghost-imaging pipelines and about $100\%$ for twelve of thirteen spectral-imaging models. A feasible-but-wrong reconstruction can match the record to ${\sim}10^{-15}$, far below the ${\sim}10^{-3}$ noise floor, while depicting a different object. Demonstrations are simulation- or declared-operator-level, without hardware validation; the audit reports consistency with the declared operator, not object truth. Within that scope, MCA opens an avenue for accountable GI/SPI reconstruction at low sampling ratios.

---

## 1. Introduction

Ghost imaging and single-pixel imaging retrieve a two-dimensional scene from a sequence of structured illuminations and the corresponding single-pixel (bucket) intensities, without a spatially resolved sensor [1–6]. Since the first correlation-based demonstrations [1,2] and the computational and single-pixel variants that followed [3–6], the modality has become an attractive tool for computational imaging and has been applied in a range of settings, e.g., low-light imaging, imaging through scattering media, multispectral and three-dimensional sensing, and optical security [5–9]. A recurring practical constraint is that high image quality is difficult to obtain at low sampling ratios, where the number of bucket measurements is far smaller than the number of image pixels and the reconstruction problem is deeply underdetermined.

Deep learning has become a powerful tool for this regime, and learned reconstructors routinely report large improvements in image quality over conventional correlation, back-projection, and compressed-sensing methods [10–14]. However, a learned network fills the unmeasured part of the image with content learned from a training prior. At low sampling ratios most of the visible detail is prior-supplied, and a reconstruction can be sharp, natural-looking, and fully consistent with the recorded bucket signals while depicting structure the measurement never constrained. The accuracy of such a reconstruction cannot be assessed in practical applications where the ground truth is unknown [12]; existing reliability tools estimate a statistical uncertainty from a Bayesian network [12], but do not separate, from the measurement itself, which content is constrained by the measurement and which is supplied by the learned prior.

Several families of methods address image quality at low sampling ratios. Optimized illumination bases and deep priors raise reconstruction quality from few measurements [10,11,13,14]; however, they optimize fidelity to a ground-truth reference during training and are silent about which of the recovered detail the measurement itself constrains. Compressed-sensing regularization exploits sparsity to invert the underdetermined system [15,16]; however, it substitutes a hand-chosen prior for the missing measurements without stating how much of the result that prior supplies. Range–null-space reconstruction installs learned content directly in the operator's kernel to preserve data consistency [17–19]; however, it is a reconstruction strategy that trusts the injected null-space content rather than an audit that meters it. In each case, fidelity is scored against a reference image, and none provides a ground-truth-free statement of what the bucket record actually certifies. Therefore, it is desirable to certify, without any reference image, which part of a reconstruction is supported by the measurement and which part is supplied by the prior, and to make this certificate exact and independent of the reconstructor.

In this paper, we report a ground-truth-free measurement-consistency audit (MCA) for learned ghost and single-pixel imaging, built on the range–null geometry of the sensing operator. The contributions are summarized as follows. Firstly, we show that measurement consistency cannot certify image content, and we construct a feasible-but-wrong reconstruction that matches a target bucket record to floating-point precision — more tightly than the true scene matches its own noisy record — while depicting a different object. Secondly, we develop the proposed audit, a post-hoc operator that audits the measurement-constrained content of any reconstruction with an exact per-mode contraction $\lambda/(\lambda+\sigma_i^2)$, using only the sensing operator, the recorded bucket signals, and one regularization parameter; the operator itself is the classical Tikhonov filter-factor complement, and the novelty is its ground-truth-free, per-record, reconstructor-independent use. Thirdly, the audit is used as a forensic tool to decompose the reported gains of published pipelines through each operator's own projectors — to the best of our knowledge, the first ground-truth-free attribution of published pipelines through their own sensing operators — and it is demonstrated on a snapshot spectral operator that the per-mode contraction resolves into a $2\times10^4$-fold graded profile that a flat-spectrum operator cannot exhibit. Finally, the same boundary is turned into a governed dial that supplies prior detail strictly in the unmeasured subspace, preserving the recorded bucket signals analytically. The numerical demonstrations audit declared forward models — a simulated GI operator and the released physical coded-aperture mask under a declared integer-shift coded-aperture snapshot spectral imaging (CASSI) operator — with no physical optical capture; unmodeled hardware error is outside the present validation. The principle is presented in Section 2, results and discussion in Section 3, and conclusions in Section 4.

---

## 2. Principle

### 2.1 Range–null geometry of the sensing operator

The bucket measurements of a GI/SPI system can be described by

$$y = A x + \varepsilon,$$

where $x \in \mathbb{R}^{n}$ denotes the vectorized scene, $y \in \mathbb{R}^{m}$ denotes the bucket-measurement vector with $m \ll n$, $A \in \mathbb{R}^{m\times n}$ denotes the sensing operator whose rows are the illumination patterns, and $\varepsilon$ denotes the measurement noise. Let $A^{\dagger}$ denote the Moore–Penrose pseudoinverse and let $P_R = A^{\dagger}A$ and $P_0 = I - A^{\dagger}A$ denote the orthogonal projectors onto the row space (the measured subspace) and the null space (the unmeasured subspace) of $A$, respectively. These projectors satisfy

$$A P_0 = 0, \qquad A P_R = A,$$

so that $A x = A P_R x$: the recorded bucket signals are a function of the measured component $P_R x = A^{\dagger}y$ alone (its noisy counterpart $A^{\dagger}y = P_R x + A^{\dagger}\varepsilon$ under noise), while the null-space component $P_0 x$ is invisible to the sensor. Any two scenes that share the measured component therefore produce the same bucket record no matter how much they differ in the null space. In the locked GI setting used below — $n = 4096$, $m = 205$ (a $5\%$ sampling ratio) — the measured subspace has dimension $205$ and the unmeasured subspace has dimension $3891$; most of what the eye reads as detail is carried by the unmeasured subspace. We take $A$ to have full row rank (numerical rank determined at a threshold of $10^{-10}$ relative to the largest singular value), so that $\mathcal{R}(A) = \mathbb{R}^m$ and any record $y$ — including a noisy one — lies in the range of $A$; measurement noise then enters the row-space estimate as $A^{\dagger}y = P_R x + A^{\dagger}\varepsilon$, leaving the projector identities unchanged. A genuinely rank-deficient operator would instead require the least-squares projection of $y$ onto $\mathcal{R}(A)$, a case we do not treat here.

### 2.2 Measurement consistency does not certify image content

Because $A P_0 = 0$, measurement consistency is a statement about the measured subspace only. This is made concrete by a construction. For any target record $y$ and any donor scene $x_j$, a feasible-but-wrong reconstruction can be obtained by

$$u_j(y) = x_j - A^{\dagger}\big(A x_j - y\big)$$

where $x_j$ denotes the donor scene, $A^{\dagger}$ denotes the Moore–Penrose pseudoinverse, and $y$ denotes the target bucket record. This reconstruction satisfies $A\,u_j(y) = y$ and $P_0\,u_j(y) = P_0\,x_j$: it reproduces the target bucket record exactly while carrying the donor's null-space content. The identity $A u_j(y) = y$ holds exactly when $A$ has full row rank (so that $AA^{\dagger} = I_m$); for a noisy target record, the construction lands on the recorded fiber $\{v : Av = y\}$, whose distance from the clean scene is the noise level. It is demonstrated that such a reconstruction matches the record to a relative residual $\|Au_j-y\|/\|y\|\approx10^{-15}$ (unconstrained, float64) — far below the ${\sim}10^{-3}$ noise floor at which the true scene matches its own noisy record — while remaining semantically far from the target (Fig. 1, left; Fig. 2). That an exact projection onto the record fiber beats a noisy-record residual is expected; the substantive point is the semantic divergence at equal-or-better consistency. Alternating the exact projection onto the record with a clip to the physical range $[0,1]$ yields a box-legal feasible-but-wrong reconstruction whose residual remains well below the noise floor; the barrier is therefore not an artifact of unconstrained linear algebra. Consistency is not correctness, and a reliability indicator based on the record alone must certify only what the measurement constrained.

[FIG: METHOD_FIG1.pdf | 1.0 | Fig. 1. Range–null accountability for undersampled ghost imaging. Left, a feasible-but-wrong reconstruction $u$ that matches the target bucket record to numerical precision, so measurement consistency cannot certify the unmeasured content ($\|Au-y\|/\|y\|\approx10^{-13}$ for the pair shown, ${\sim}10^{-15}$ across pairs, versus a $10^{-3}$ noise floor). Middle, the proposed audit (MCA) contracts only the measured modes, with per-mode factor $\lambda/(\lambda+\sigma_i^2)$. Right, the governed dial injects prior detail only through $P_0$, preserving $A\hat{x}_B=y$ analytically (numerically to the reported residual) while moving along the perception–distortion curve (locked: LPIPS $-32.6\%$ at $-0.45$ dB, $8/8$ gate). GI: ghost imaging; MCA: measurement-consistency audit; $u$: feasible-but-wrong reconstruction; $A$: sensing operator; $y$: bucket-measurement vector; $P_0$: projector onto the unmeasured (null) subspace; $\sigma_i$: singular value; $\lambda$: regularization parameter; $B$: dial weight; LPIPS: learned perceptual image patch similarity.]

[FIG: WITNESS_GEOMETRY.pdf | 0.78 | Fig. 2. Geometry of the feasible-but-wrong reconstruction. Every point on the line (the record fiber $\{x: Ax=y\}$) reproduces the bucket record; only motion off the line is visible to the measurement. The constructed reconstruction is the donor scene projected onto the fiber: it matches the record to ${\sim}10^{-15}$, tighter than the true scene (dashed circle, radius ${\sim}\sigma_\varepsilon$), while carrying the donor's null-space content. $A$: sensing operator; $y$: bucket record; $\{x: Ax=y\}$: record fiber; $A^{\dagger}$: Moore–Penrose pseudoinverse; $\sigma_\varepsilon$: standard deviation of the bucket noise.]

### 2.3 The proposed measurement-consistency audit (MCA)

For any reconstruction $\hat{x}$, the proposed audit (MCA) can be described by

$$\Pi_y^{\lambda}(\hat{x}) = \hat{x} - G_\lambda\,(A\hat{x} - y), \qquad G_\lambda = A^{\top}(AA^{\top} + \lambda I)^{-1},$$

where $\lambda > 0$ denotes a regularization parameter, fixed at $\lambda = 1.0\times10^{-3}$ in all experiments below (when the noise level is known, the Morozov discrepancy principle — matching $\|A\hat{x}-y\|$ to the noise level — is the principled selection rule), and the correction $G_\lambda(A\hat{x}-y)$ lies in the row space so that $P_0\,\Pi_y^{\lambda}(\hat{x}) = P_0\,\hat{x}$: whatever a prior placed in the null space passes through untouched. The post-audit record residual is $A\,\Pi_y^{\lambda}(\hat{x}) - y = \lambda(AA^{\top} + \lambda I)^{-1}(A\hat{x} - y)$, which makes the per-mode contraction transparent. The proposed audit needs only $(A, y, \lambda)$ — no ground truth and no access to the reconstructor. Diagonalizing $AA^{\top}$, the residual in the $i$-th measured singular mode of $A$ (singular value $\sigma_i$) is scaled by

$$c_i(\lambda) = \frac{\lambda}{\lambda + \sigma_i^2},$$

which is exact, image-independent, and reconstructor-independent; as $\lambda \to 0$ the audit becomes a hard projection onto the record fiber $\{x: Ax = y\}$. The factor $c_i(\lambda)$ is the complement of the Tikhonov filter factor [20] (that is, $c_i(\lambda) = 1 - \sigma_i^2/(\sigma_i^2+\lambda)$, so the record residual of a well-measured mode with $\sigma_i^2 \gg \lambda$ is almost fully removed, $c_i \to 0$, while that of a weakly measured mode with $\sigma_i^2 \ll \lambda$ is left almost untouched, $c_i \to 1$); its use here is a post-hoc, per-record reliability indicator applied after arbitrary reconstructors. In float64 the measured contraction matches the closed form to $1.0\times10^{-10}$, so the proposed audit reports, mode by mode, how far a reconstruction stands from the recorded bucket signals. The location-of-content decomposition below depends only on the projectors $P_R, P_0$ and is independent of $\lambda$; $\lambda$ enters solely the proposed audit's contraction, where $c_i(\lambda)$ varies smoothly and monotonically, and all reported audit results use $\lambda = 10^{-3}$.

### 2.4 The governed null-space dial

Because the null space cannot be certified, prior detail is not hidden there but metered. For classical and generative endpoints $d_A$ and $d_G$ of a reconstruction, the governed dial can be obtained by

$$\hat{x}_B = A^{\dagger}y + P_0\big(d_A + B(d_G - d_A)\big),$$

where $B$ denotes a scalar dial weight and the injected detail is confined to the null space, so that $A\hat{x}_B = A A^{\dagger}y = y$ holds analytically for every $B$ (numerically to the reported residual). Algebraically, this null-space data-consistency form coincides with the range–null decomposition used by denoising diffusion null-space models [18] and by range–null-space spectral reconstruction [19]; what is new here is not the form but its use — the proposed audit as a post-hoc, reconstructor-agnostic certificate, and the governed dial as a governance instrument that meters, rather than trusts, the injected content. The dial traces a perception–distortion curve while never converting the perceptual gain it produces into a measurement claim.

---

## 3. Results and discussion

### 3.1 The governed dial on locked ghost imaging

The proposed dial is evaluated on $64\times64$ grayscale STL10 scenes at a $5.0\%$ sampling ratio ($n = 4096$, $m = 205$), using a signed, row-orthonormalized computational GI operator, with a representative generative (vector-quantized adversarial) prior for $d_G$, a classical linear-minimum-mean-square-error (Wiener) reconstruction for $d_A$, and a regularization parameter $\lambda = 1.0\times10^{-3}$. The signed, row-orthonormalized operator is an algebraic benchmark rather than a physical illumination scheme — real GI patterns are non-negative, often differential, and DMD-quantized (§3.5) — and LPIPS is computed with the standard AlexNet backbone on the grayscale images replicated to three channels. Under a pre-specified, one-shot protocol with a frozen dial weight, the balanced setting reduces the perceptual distance (LPIPS) by $32.6\%$ relative to the projected classical endpoint at distortion costs of $0.45$ dB and $0.024$ in peak signal-to-noise ratio (PSNR) and structural similarity (SSIM, from $0.657$ to $0.633$), respectively (all figures are means over the same three independent training seeds; the seed-to-seed standard deviations are $0.9\%$, below $0.01$ dB, and $0.0005$ for the LPIPS reduction, the PSNR cost, and the SSIM cost, respectively; Supplement S2), and passes all eight pre-specified acceptance conditions (a frozen gate on the LPIPS gain, the PSNR cost, the record-consistency residual, and per-seed sign and monotonicity of the trade; $8/8$ passed). The consistency $A\hat{x}_B = y$ is analytically exact and numerically preserved to a relative measurement error of $3.6\times10^{-7}$ (the float32 pipeline scale) at every operating point (Fig. 3). It is worth noting that the perceptual gain lives entirely on the quality axis; the analytically exact consistency that accompanies it certifies reproduction of the record, not the correctness of what the record cannot see.

[FIG: PARETO_FIGURE_OPTICS.pdf | 1.0 | Fig. 3. Perception–distortion trade of the governed null-space dial at exact record consistency. As the dial weight $B$ increases from $0$ to $1$, prior detail is injected only through the unmeasured subspace $P_0$, moving the reconstruction along the LPIPS–PSNR curve while $A\hat{x}_B = y$ holds analytically (numerically to a relative measurement error of $3.6\times10^{-7}$) at every point. The balanced operating point (highlighted) reduces LPIPS by $32.6\%$ at a PSNR cost of $0.45$ dB. $B$: dial weight; $A$: sensing operator; $\hat{x}_B$: dial reconstruction; $y$: bucket-measurement vector; $P_0$: projector onto the unmeasured (null) subspace; LPIPS: learned perceptual image patch similarity; PSNR: peak signal-to-noise ratio.]

### 3.2 Forensic decomposition of published ghost-imaging pipelines

Subsequently, the proposed audit is used as a forensic tool. At these sampling ratios most degrees of freedom are unmeasured, so any record-consistent reconstruction must supply most visible detail from the null space; the decomposition below therefore quantifies *where* each pipeline's content is located — often close to the truth — not whether it is wrong. For three published deep-learning ghost-imaging pipelines — a pretrained-and-fine-tuned reconstructor (PEDL), an untrained deep-image-prior method on a real measured speckle operator (GIDC), and a self-supervised method on a genuinely noisy record (Noise2Ghost) — the reported improvement is decomposed, through each pipeline's own released or measured operator, into row-space repair and null-space supply. The split is grounded in the conserved orthogonal error-energy partition $\|\hat{x}-x\|^2 = \|P_R(\hat{x}-x)\|^2 + \|P_0(\hat{x}-x)\|^2$: concretely, a pipeline's *null share* is the fraction of its PSNR improvement over the range ceiling that survives when the row component is reset to $A^{\dagger}y$ (i.e., attributable to $P_0\hat{x}$), a monotone function of the null error-energy fraction $\|P_0(\hat{x}-x)\|^2/\|\hat{x}-x\|^2$. The PSNR percentages reported are this re-expression, taken relative to a range ceiling (the PSNR of the oracle row-space (measured) image $P_R x$, which requires ground truth to compute and is used only as an analysis reference, never by the audit itself). The results are shown in Fig. 4. Row-space repair saturates early, and the reported PSNR above the range ceiling is supplied from the null space; the null shares of the reported gain are $39\%$, $40\%$, and $95\%$ for PEDL, GIDC, and Noise2Ghost, respectively (Noise2Ghost, a near-pure null-supplier, is an outlier), while the terminal residual error is separately $93$–$97\%$ null in every case (the underlying error-energy fractions $\|P_0(\hat{x}-x)\|^2/\|\hat{x}-x\|^2$ are $96.8\%$ and $92.7\%$ for the two pipelines with available ground truth; the real-scene pipeline has no ground truth and is excluded from this split, Supplement S1). On the real measured speckle operator, the audit drives the record residual from $3.0\times10^{-3}$ to $3.0\times10^{-15}$ at a PSNR change of $+0.20$ dB, and the measured per-mode contraction matches $\lambda/(\lambda+\sigma_i^2)$ to $3.5\times10^{-11}$ in float64, so the closed form holds on a physically measured operator, not only on designed ones.

**Table 1. Projector forensics of three published ghost-imaging pipelines on their own operators.** *Null share of gain* is the null-located fraction of the reported PSNR gain over the operator's range ceiling; the terminal residual error is $93$–$97\%$ null for all three. PEDL: physics-enhanced deep learning [13]; GIDC: ghost-imaging deep-image-prior on measured speckle [14]; Noise2Ghost: self-supervised on a noisy record.

| Pipeline | Operator source | Null share of gain |
|---|---|---|
| PEDL (pretrained + fine-tuned) | released learned patterns (simulated) | $39\%$ |
| GIDC (untrained deep-image-prior) | real measured speckle | $40\%$ |
| Noise2Ghost (self-supervised) | released masks, noisy record | $95\%$ |

[FIG: FORENSICS_CROSS_TARGET.pdf | 1.0 | Fig. 4. Forensic decomposition of three published ghost-imaging pipelines on their own operators. (a) A fine-tuning trajectory decomposed per step: the PSNR rises past the operator's range ceiling, and everything above it is supplied from the null space. (b) Attribution of each method's improvement into row-space repair and null-space supply. (c) Every reported result stands $+3.1$ to $+8.1$ dB above its own range ceiling, with terminal error $93$–$97\%$ null. GI: ghost imaging; $P_R x$: oracle row-space (measured) image; $P_0$: unmeasured subspace; PSNR: peak signal-to-noise ratio.]

### 3.3 A non-uniform per-mode spectrum on the released coded-aperture mask

The per-mode structure of the proposed audit is made visible on a single-disperser CASSI operator, which shifts, masks, and sums a $28$-band spectral cube into a $256\times310$ coded snapshot — a $23.1\times$ undersampling — through the released coded-aperture mask under the declared integer-shift operator (fill $0.563$). Because, in the declared integer-shift model, each voxel reaches exactly one detector pixel, $AA^{\top} = \operatorname{diag}(\Phi_s)$ is exactly diagonal, so the projectors and the entire singular spectrum are closed-form; a real single-disperser capture has PSF blur, cross-talk, and non-integer shifts that would break exact diagonality and smooth the spectrum, so the numbers below are operator-specific to that declared model. The measured singular values $\sigma_j = \sqrt{\Phi_s}$ (obtained in closed form as $\Phi_s = \sum_\lambda M_\lambda^2$ over the wavelength-sheared coded mask $M$, normalized to unit peak transmittance) over $79{,}360$ detector modes span $[0.0093, 4.907]$ (median $3.51$, a strongly left-skewed distribution), a $528\times$ ratio; no detector mode is exactly zero because every column of the sheared mask receives at least one spectral band, and the per-mode contraction $\lambda/(\lambda+\sigma_j^2)$ at $\lambda=10^{-3}$ therefore spans $[4.15\times10^{-5}, 9.21\times10^{-1}]$, a $2\times10^4$ spread (Fig. 5). This is the mode-resolved face of the same closed form that collapses to a single value for a flat-spectrum operator: for a masked-Fourier operator, as used in accelerated magnetic-resonance imaging (included here only as an inverse-problem control, not an optics result), all measured singular values equal one and the contraction is a single number $9.99\times10^{-4}$; the projector identities there hold in float64 to ${\sim}10^{-17}$. The two operators bracket the audit, from a flat spectrum to a strongly non-uniform one.

[FIG: CASSI_SPECTRUM_OPTICS.pdf | 1.0 | Fig. 5. The per-mode contraction on the released coded-aperture mask (declared integer-shift operator). Left, the singular value $\sigma_j=\sqrt{\Phi_s}$ across the detector plane. Middle, the distribution of $\sigma_j$ over $79{,}360$ measured modes spans a $528\times$ range; the red line marks the single value ($\sigma=1$) of a flat-spectrum masked-Fourier operator. Right, the per-mode audit factor $\lambda/(\lambda+\sigma_j^2)$ spreads across four orders of magnitude, where a flat-spectrum operator collapses to a single point. CASSI: coded-aperture snapshot spectral imaging; $\sigma_j$: singular value of the $j$-th detector mode; $\Phi_s$: diagonal of $AA^{\top}$; $\lambda$: regularization parameter.]

### 3.4 A decade of learned spectral-imaging gains in the unmeasured subspace

The forensic decomposition is applied across thirteen published spectral-imaging models on the shared coded aperture, with the wiring validated by reproducing each model's published simulation PSNR (e.g., $34.38$ versus $34.26$ dB, and $38.18$ versus $38.36$ dB). The min-norm reconstruction $A^{\dagger}y$ scores $19.04$ dB (the range ceiling). It is demonstrated that the models supply essentially all of their PSNR gain over the ceiling from the null space; a slightly negative row-space effect — the models mildly perturb the measured component rather than repairing it — makes the null share of twelve of the thirteen exceed $100\%$, at $100$–$101\%$, with GAP-Net the sole exception at $93\%$ (GAP-Net also shows the largest reproduction gap in Table 2, so we draw no comparative conclusion from it). The null share of the PSNR *gain* can exceed $100\%$ because the row-space effect is negative; the underlying raw error-energy fraction $E_0/E = \|P_0(\hat{x}-x)\|^2/\|\hat{x}-x\|^2$ is bounded in $[0,1]$ and, computed directly on the same reconstructions, is $94.8$–$99.5\%$ across the twelve consistent models (GAP-Net $82.6\%$), tightening toward $100\%$ as the models improve — the measured component is pinned by the data, so almost all residual error, and increasingly so for the best models, lives in the unmeasured subspace (Supplement S1). The entire decade of reported progress ($+6.5$ dB across the field) is thus located wholly in the unmeasured subspace, and the models are $0.9$–$12\%$ inconsistent with their own coded snapshot (Fig. 6). As a flat-spectrum control, the same decomposition on a masked-Fourier operator (used in accelerated magnetic-resonance imaging, where all measured singular values equal one) gives $91$–$92\%$ null-located gain for an official single-coil reconstruction network, with a governed data-consistent variant retaining nearly all of that gain at exact record consistency — confirming that the ledger structure is not specific to the non-uniform CASSI spectrum. It should be emphasized that, at these sampling ratios, the null-dominance of a record-consistent estimator's error is expected by construction and is not the finding; the finding is the contingent content that undersampling does not entail — the measured record-inconsistency of published pipelines ($0.9$–$12\%$), the negative row-space effect, the specific $528\times$ non-uniform spectrum, and the agreement of the split across an entire model zoo. The ledger is an accounting of where each pipeline's quality is manufactured, not an accusation of error; the null content these models supply is often close to the truth, but its location — entirely in the uncertifiable subspace — is what the proposed audit makes explicit.

**Table 2. Attribution over thirteen published spectral-imaging models on the shared coded aperture** (min-norm range ceiling $19.04$ dB). Our reproduced PSNR matches the published value within ${\sim}0.2$ dB for most models; the larger gaps for GAP-Net, $\lambda$-Net, and DAUHST-2stg reflect input-convention differences in our unified harness, and for GAP-Net this correlates with its outlier null share and record residual. *Null share* is the null-located fraction of the PSNR gain over the ceiling (exceeding $100\%$ where the row-space effect is negative); *record residual* is $\|A\hat{x}-y\|/\|y\|$.

| Model | PSNR (ours / pub.) | Null share | Record residual |
|---|---|---|---|
| TSA-Net | $31.67$ / $31.46$ | $101\%$ | $5.3\%$ |
| GAP-Net | $29.57$ / $33.03$ | $93\%$ | $11.9\%$ |
| DGSMP | $32.68$ / $32.63$ | $101\%$ | $5.9\%$ |
| $\lambda$-Net | $28.65$ / $31.00$ | $101\%$ | $5.3\%$ |
| HDNet | $35.06$ / $34.97$ | $101\%$ | $3.2\%$ |
| MST-S | $34.38$ / $34.26$ | $101\%$ | $3.8\%$ |
| MST-M | $35.04$ / $34.94$ | $101\%$ | $3.1\%$ |
| DAUHST-2stg | $35.31$ / $36.34$ | $101\%$ | $2.8\%$ |
| MST-L | $35.33$ / $35.18$ | $101\%$ | $2.9\%$ |
| CST-L | $35.94$ / $36.12$ | $101\%$ | $2.6\%$ |
| MST++ | $36.10$ / $35.99$ | $101\%$ | $2.6\%$ |
| BIRNAT | $37.73$ / $37.58$ | $100\%$ | $1.3\%$ |
| DAUHST-9stg | $38.18$ / $38.36$ | $100\%$ | $0.9\%$ |

[FIG: CASSI_FORENSICS.pdf | 1.0 | Fig. 6. Reported spectral-imaging gains decomposed on the shared released coded-aperture mask (declared integer-shift operator). (a) Thirteen published models against the $19.04$ dB min-norm range ceiling; twelve of the thirteen gains are $100$–$101\%$ null-supplied (one exception at $93\%$), and the row-space effect is slightly negative. (b) Each model's record inconsistency $\|A\hat{x}-y\|/\|y\|$ ranges from $0.9\%$ to $12\%$. $A^{\dagger}y$: min-norm (range-ceiling) reconstruction; $\hat{x}$: a model reconstruction; $P_0$: unmeasured subspace; PSNR: peak signal-to-noise ratio.]

### 3.5 Scope

All operators in this work are simulation- or declared-operator-level: the GI operator is simulated; the masked-Fourier leg uses an emulated single-coil operator; and the CASSI leg uses the declared integer-shift operator on the real released aperture rather than a physical dispersive capture. The measurement noise is modeled as additive Gaussian, a first-order proxy for the signal-dependent Poisson statistics of photon counting, and no hardware validation is performed; the certificates therefore concern the declared model of the sensor. The projectors $P_R, P_0$ and the range–null geometry are noise-independent, so the constructive impossibility, the governed dial, and the location-of-content ledger are unchanged under any noise model. The audit's quantitative calibration, however, is not: under signal-dependent Poisson (shot) statistics the per-mode residual variance becomes heteroscedastic, the noise floor becomes spatially varying, the discrepancy-principle $\lambda$ changes, and the witness and record margins loosen from the ${\sim}10^{-15}$ and ${\sim}10^{-3}$ values reported here; a measured mode whose $\sigma_i^2$ is comparable to its per-mode photon-noise variance is effectively uncertified even where $c_i(\lambda)$ is small. Real GI illumination patterns are also non-negative, often differential, and DMD-quantized, whereas the locked operator here is a signed, row-orthonormalized matrix; the range–null conclusions are pattern-agnostic, while the specific spectral numbers ($528\times$, $2\times10^4$) are operator-specific, and Section 3.2 already shows the closed form surviving one real measured speckle operator. Shot-limited records and physically captured operators are therefore the primary next step. The unmeasured content remains uncertifiable by any within-record method — this is a property of the geometry, not an engineering gap — and the proposed audit is precisely a statement of that boundary. Finally, the audit certifies consistency with the *declared* operator: it cannot detect errors in the declared operator itself (miscalibration, an incorrect pattern model, or a wrong forward map), which must be established separately.

---

## 4. Conclusion

We have reported a ground-truth-free measurement-consistency audit for learned ghost and single-pixel imaging, together with a constructive impossibility, a forensic decomposition of published pipelines, and a governed null-space dial. It is demonstrated that measurement consistency cannot certify image content, that the proposed audit quantifies the measurement-constrained content of any reconstruction with an exact per-mode contraction $\lambda/(\lambda+\sigma_i^2)$ from the record alone, and that the reported gains of published deep-learning pipelines are largely located in the unmeasured subspace — null shares of $39\%$, $40\%$, and $95\%$ for three ghost-imaging pipelines, and $100$–$101\%$ for twelve of thirteen spectral-imaging models (with $93\%$ for the exception). On the released coded-aperture mask under the declared integer-shift operator the per-mode contraction resolves into a $2\times10^4$-fold graded profile that a flat-spectrum operator cannot exhibit, and the governed dial supplies prior detail only where the measurement is blind, preserving the recorded bucket signals analytically. It should be emphasized that the proposed audit provides a model-level accountability layer that reports measurement consistency separately from image quality, pending validation on physically captured operators with photon-limited statistics; within that scope it could open up an avenue for accountable GI/SPI reconstruction in complex, low-sampling-ratio imaging conditions.

---

## Funding

Funding information to be added.

## Acknowledgments

Acknowledgment information to be added.

## Disclosures

The authors declare no conflicts of interest.

## Data availability

Data underlying the results presented in this paper are available in Refs. [21–23] (the released MST/TSA-Net coded aperture and spectral scenes [21,22] and the fastMRI control data [23]); the pretrained reconstruction models are the cited authors' released checkpoints. Code implementing the operator constructions, the projector and audit computations, the feasible-but-wrong witness, and the figure- and table-generation is available from the authors upon reasonable request.

---

## Supplementary material

### S1. Error-energy decomposition of the forensic ledgers

The null shares reported in Tables 1 and 2 are re-expressions of PSNR improvement over the range ceiling. Here we report the underlying quantity directly: the conserved orthogonal partition of the reconstruction error $e = \hat{x} - x$,
$$\|e\|^2 = \underbrace{\|P_R e\|^2}_{E_R} + \underbrace{\|P_0 e\|^2}_{E_0},$$
into a measured (row-space) energy $E_R$ and an unmeasured (null-space) energy $E_0$, with null energy fraction $E_0/(E_R+E_0)$. This fraction is bounded in $[0,1]$ by construction and requires ground truth to compute (it is an analysis quantity, never used by the audit). Because $E_R, E_0$ are reported in each problem's native units — per-pixel MSE for the $64\times64$ ghost-imaging reconstructions, summed squared error over the $28$-band cube averaged across ten KAIST scenes for CASSI — the dimensionless fraction is the comparable column across ledgers.

**Table S1. Error-energy split $E_R$ (measured) and $E_0$ (unmeasured) with the null energy fraction $E_0/E$.** For the min-norm reconstruction $A^{\dagger}y = P_R x$, $E_R = 0$ by construction, giving a $100\%$ reference. The null fraction tightens toward $100\%$ as models improve, because the measured component is increasingly pinned by the data while the residual concentrates in the unmeasured subspace.

*Ghost-imaging pipelines (Table 1):*

| Pipeline | Regime | $E_R$ | $E_0$ | Null energy $E_0/E$ |
|---|---|---|---|---|
| PEDL [13] | simulated, noiseless | $9.92\times10^{-5}$ | $2.98\times10^{-3}$ | $96.8\%$ |
| Noise2Ghost | simulated, noisy | $1.06\times10^{-3}$ | $1.34\times10^{-2}$ | $92.7\%$ |
| GIDC [14] | real scene, no ground truth | — | — | GT-free$^{\dagger}$ |

$^{\dagger}$GIDC uses a real measured speckle record with no ground truth, so the error is not MSE-decomposable (pre-registered exclusion). Its ground-truth-free surrogate — the null-space fraction of the reconstruction itself, $\|P_0\hat{x}\|^2/\|\hat{x}\|^2$ on the mean-removed image — is $43.8\%$, metering how much of the output structure is unconstrained by the record.

*Spectral-imaging models (Table 2), min-norm reference $E_0 = 2.85\times10^{4}$:*

| Model | $E_R$ | $E_0$ | Null energy $E_0/E$ |
|---|---|---|---|
| TSA-Net | $5.7\times10^{1}$ | $1.45\times10^{3}$ | $96.2\%$ |
| GAP-Net | $9.6\times10^{2}$ | $4.59\times10^{3}$ | $82.6\%$ |
| DGSMP | $6.7\times10^{1}$ | $1.21\times10^{3}$ | $94.8\%$ |
| $\lambda$-Net | $6.9\times10^{1}$ | $2.99\times10^{3}$ | $97.8\%$ |
| HDNet | $2.1\times10^{1}$ | $6.97\times10^{2}$ | $97.1\%$ |
| MST-S | $2.8\times10^{1}$ | $8.00\times10^{2}$ | $96.7\%$ |
| MST-M | $1.8\times10^{1}$ | $7.09\times10^{2}$ | $97.5\%$ |
| DAUHST-2stg | $1.9\times10^{1}$ | $6.51\times10^{2}$ | $97.2\%$ |
| MST-L | $1.6\times10^{1}$ | $6.52\times10^{2}$ | $97.6\%$ |
| CST-L | $1.5\times10^{1}$ | $5.67\times10^{2}$ | $97.5\%$ |
| MST++ | $1.4\times10^{1}$ | $5.53\times10^{2}$ | $97.5\%$ |
| BIRNAT | $3.4\times10^{0}$ | $4.35\times10^{2}$ | $99.2\%$ |
| DAUHST-9stg | $1.9\times10^{0}$ | $3.61\times10^{2}$ | $99.5\%$ |

Twelve of the thirteen models place $94.8$–$99.5\%$ of their error energy in the unmeasured subspace (DGSMP the lowest at $94.8\%$ and the remaining eleven at $96.2$–$99.5\%$); GAP-Net, the sole outlier at $82.6\%$, is also the model with the largest reproduction gap in Table 2, and no comparative conclusion is drawn from it. Consistent with the main text, $E_R$ falls monotonically as the models improve (from $9.6\times10^{2}$ for GAP-Net to $1.9\times10^{0}$ for DAUHST-9stg), so the residual error of the strongest models is almost purely null-space.

### S2. Seed-level spread of the governed-dial tradeoff

The governed-dial operating point of Section 3.1 was retrained under three independent seeds. Aggregating per seed over the locked test split, the balanced setting reduces LPIPS relative to the projected classical (VQAE) endpoint by a mean of $32.6\%$ (per-seed values $31.6\%$, $32.4\%$, $33.8\%$; population standard deviation $0.9\%$ over the $n=3$ seeds, or $1.1\%$ as a sample standard deviation) at a PSNR cost of $-0.45$ dB (per-seed values $-0.45$, $-0.45$, $-0.44$ dB; standard deviation below $0.01$ dB either way) and an SSIM cost of $0.024$ (SSIM $0.657$ at the projected classical endpoint versus $0.633$ at the balanced setting; population standard deviation $0.0005$ over the three seeds). The record-consistency residual is $3.6\times10^{-7}$ at every seed and operating point (the float32 pipeline scale). The tradeoff is therefore stable across retraining, not an artifact of a single run.

---

## References

1. T. B. Pittman, Y. H. Shih, D. V. Strekalov, and A. V. Sergienko, "Optical imaging by means of two-photon quantum entanglement," Phys. Rev. A **52**, R3429 (1995).
2. A. Gatti, E. Brambilla, M. Bache, and L. A. Lugiato, "Ghost imaging with thermal light: comparing entanglement and classical correlation," Phys. Rev. Lett. **93**, 093602 (2004).
3. J. H. Shapiro, "Computational ghost imaging," Phys. Rev. A **78**, 061802(R) (2008).
4. Y. Bromberg, O. Katz, and Y. Silberberg, "Ghost imaging with a single detector," Phys. Rev. A **79**, 053840 (2009).
5. M. F. Duarte, M. A. Davenport, D. Takhar, J. N. Laska, T. Sun, K. F. Kelly, and R. G. Baraniuk, "Single-pixel imaging via compressive sampling," IEEE Signal Process. Mag. **25**(2), 83–91 (2008).
6. M. P. Edgar, G. M. Gibson, and M. J. Padgett, "Principles and prospects for single-pixel imaging," Nat. Photonics **13**, 13–20 (2019).
7. P. Clemente, V. Durán, V. Torres-Company, E. Tajahuerce, and J. Lancis, "Optical encryption based on computational ghost imaging," Opt. Lett. **35**, 2391–2393 (2010).
8. W. Chen, "Ghost identification based on single-pixel imaging in big data environment," Opt. Express **25**, 16509–16516 (2017).
9. W. Jiao et al., "Single-pixel imaging: principles, methods, algorithms, and applications," Adv. Imaging (2026), doi:10.3788/AI.2026.20002.
10. M. Lyu, W. Wang, H. Wang, H. Wang, G. Li, N. Chen, and G. Situ, "Deep-learning-based ghost imaging," Sci. Rep. **7**, 17865 (2017).
11. Y. He, G. Wang, G. Dong, S. Zhu, H. Chen, A. Zhang, and Z. Xu, "Ghost imaging based on deep learning," Sci. Rep. **8**, 6469 (2018).
12. R. Shang, M. A. O'Brien, F. Wang, G. Situ, and G. P. Luke, "Approximating the uncertainty of deep learning reconstruction predictions in single-pixel imaging," Commun. Eng. **2**, 53 (2023).
13. F. Wang, C. Wang, C. Deng, S. Han, and G. Situ, "Single-pixel imaging using physics enhanced deep learning," Photonics Res. **10**, 104–110 (2022).
14. F. Wang, C. Wang, M. Chen, W. Gong, Y. Zhang, S. Han, and G. Situ, "Far-field super-resolution ghost imaging with a deep neural network constraint," Light Sci. Appl. **11**, 1 (2022).
15. E. J. Candès, J. Romberg, and T. Tao, "Robust uncertainty principles: exact signal reconstruction from highly incomplete frequency information," IEEE Trans. Inf. Theory **52**, 489–509 (2006).
16. J. Zhang and B. Ghanem, "ISTA-Net: interpretable optimization-inspired deep network for image compressive sensing," in Proc. IEEE/CVF CVPR (2018), pp. 1828–1837.
17. J. Schwab, S. Antholzer, and M. Haltmeier, "Deep null space learning for inverse problems: convergence analysis and rates," Inverse Probl. **35**, 025008 (2019).
18. Y. Wang, J. Yu, and J. Zhang, "Zero-shot image restoration using denoising diffusion null-space model," in Proc. ICLR (2023), arXiv:2212.00490.
19. J. Wang, S. Wang, R. Zhang, Z. Zheng, W. Liu, and X. Wang, "A range-null space decomposition approach for fast and flexible spectral compressive imaging," arXiv:2305.09746 (2023).
20. P. C. Hansen, *Rank-Deficient and Discrete Ill-Posed Problems* (SIAM, 1998).
21. Y. Cai, J. Lin, X. Hu, H. Wang, X. Yuan, Y. Zhang, R. Timofte, and L. Van Gool, "Mask-guided spectral-wise transformer for efficient hyperspectral image reconstruction," in Proc. IEEE/CVF CVPR (2022), pp. 17502–17511.
22. Z. Meng, J. Ma, and X. Yuan, "End-to-end low cost compressive spectral imaging with spatial-spectral self-attention," in Proc. ECCV, LNCS 12368 (Springer, 2020), pp. 187–204.
23. J. Zbontar, F. Knoll, A. Sriram, et al., "fastMRI: an open dataset and benchmarks for accelerated MRI," arXiv:1811.08839 (2018).
24. D. J. C. MacKay, "Bayesian interpolation," Neural Comput. **4**, 415–447 (1992).
25. R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proc. IEEE/CVF CVPR (2018), pp. 586–595.
