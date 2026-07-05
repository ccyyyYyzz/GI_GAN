# Certify What You Measure: A Ground-Truth-Free Range–Null Audit for Learned Ghost and Single-Pixel Imaging

*Author placeholder — Author One, Author Two, Author Three. Affiliation placeholder. Corresponding author: [email placeholder].*

---

## Abstract

Ghost imaging (GI) and single-pixel imaging (SPI) reconstruct a scene from a small number of bucket measurements and have become attractive for low-light, broadband, and computational imaging. At low sampling ratios, learned reconstructors produce sharp, natural-looking images; however, it remains unclear which image content is supported by the recorded measurements and which is supplied by the prior, and this fidelity cannot be assessed when the ground truth is unknown. Here, we report a ground-truth-free measurement-consistency audit (MCA) that, from the sensing operator and the recorded bucket signals alone, certifies the measured content of any reconstruction and attributes the remainder to the prior. It is shown that the audit contracts the $i$-th measured mode of the record residual by exactly $\lambda/(\lambda+\sigma_i^2)$, independent of the reconstructor. On a snapshot spectral (CASSI) operator whose singular spectrum spans $528\times$, this contraction is demonstrated to vary over $2\times10^4$, the mode-resolved face of the same closed form that collapses to a single value for a flat-spectrum operator. Using the proposed audit as a forensic tool, we decompose the reported gains of published deep-learning pipelines through each operator's own projectors: $39$–$95\%$ of three ghost-imaging pipelines' gains, and $93$–$101\%$ of thirteen published spectral-imaging models' gains, are supplied by the unmeasured null space rather than by the measurement. The same range–null boundary is turned into a governed dial that injects prior detail only where the measurement is blind, so that $A\hat{x}=y$ is preserved exactly at every setting. It is further demonstrated that a feasible-but-wrong reconstruction can match the bucket record more tightly than the true scene itself (to ${\sim}10^{-16}$) while depicting a different object. The proposed audit provides a ground-truth-free reliability indicator that opens up an avenue for accountable GI/SPI reconstruction at low sampling ratios.

---

## 1. Introduction

Ghost imaging and single-pixel imaging retrieve a two-dimensional scene from a sequence of structured illuminations and the corresponding single-pixel (bucket) intensities, without a spatially resolved sensor [1–6]. Since the first correlation-based demonstrations [1,2] and the computational and single-pixel variants that followed [3–6], the modality has been applied across low-light imaging, imaging through scattering media, multispectral and three-dimensional sensing, and optical security [5–9]. A recurring practical constraint is that high image quality is difficult to obtain at low sampling ratios, where the number of bucket measurements is far smaller than the number of image pixels and the reconstruction problem is deeply underdetermined.

Deep learning has become a powerful tool for this regime, and learned reconstructors routinely report large improvements in image quality over conventional correlation, back-projection, and compressed-sensing methods [10–14]. However, a learned network fills the unmeasured part of the image with content learned from a training prior. At low sampling ratios most of the visible detail is prior-supplied, and a reconstruction can be sharp, natural-looking, and fully consistent with the recorded bucket signals while depicting structure the measurement never constrained. The accuracy of such a reconstruction cannot be assessed in practical applications where the ground truth is unknown [12]; existing reliability tools estimate a statistical uncertainty from a Bayesian network [12], but do not separate, from the measurement itself, which content is measured from which is invented.

Several routes address image quality at low sampling ratios — optimized illumination bases and deep priors [10,11,13,14], compressed-sensing regularization [15,16], and range–null-space reconstruction that installs learned content in the operator's kernel [17–19] — yet each optimizes fidelity to a ground-truth reference and none provides a ground-truth-free statement of what the bucket record actually certifies. Therefore, it is desirable to certify, without any reference image, which part of a reconstruction is supported by the measurement and which part is supplied by the prior, and to make this certificate exact and independent of the reconstructor.

In this paper, we report a ground-truth-free measurement-consistency audit (MCA) for learned ghost and single-pixel imaging, built on the range–null geometry of the sensing operator. The contributions are summarized as follows. Firstly, we show that measurement consistency cannot certify image content, and we construct a feasible-but-wrong reconstruction that matches a target bucket record to floating-point precision — more tightly than the true scene matches its own noisy record — while depicting a different object. Secondly, we develop the proposed audit, a post-hoc operator that certifies the measured content of any reconstruction with an exact per-mode contraction $\lambda/(\lambda+\sigma_i^2)$, using only the sensing operator, the recorded bucket signals, and one regularization parameter. Thirdly, the audit is used as a forensic tool to decompose the reported gains of published pipelines through each operator's own projectors, and it is demonstrated on a snapshot spectral operator that the per-mode contraction resolves into a $2\times10^4$-fold graded profile that a flat-spectrum operator cannot exhibit. Finally, the same boundary is turned into a governed dial that supplies prior detail strictly in the unmeasured subspace, preserving the recorded bucket signals exactly. The principle is presented in Section 2, results and discussion in Section 3, and conclusions in Section 4.

---

## 2. Principle

### 2.1 Range–null geometry of the sensing operator

The bucket measurements of a GI/SPI system can be described by

$$y = A x + \varepsilon,$$

where $x \in \mathbb{R}^{n}$ denotes the vectorized scene, $y \in \mathbb{R}^{m}$ denotes the bucket-measurement vector with $m \ll n$, $A \in \mathbb{R}^{m\times n}$ denotes the sensing operator whose rows are the illumination patterns, and $\varepsilon$ denotes the measurement noise. Let $A^{\dagger}$ denote the Moore–Penrose pseudoinverse and let $P_R = A^{\dagger}A$ and $P_0 = I - A^{\dagger}A$ denote the orthogonal projectors onto the row space (the measured subspace) and the null space (the unmeasured subspace) of $A$, respectively. These projectors satisfy

$$A P_0 = 0, \qquad A P_R = A,$$

so that $A x = A P_R x$: the recorded bucket signals are a function of the measured component $P_R x = A^{\dagger}y$ alone (its noisy counterpart $A^{\dagger}y = P_R x + A^{\dagger}\varepsilon$ under noise), while the null-space component $P_0 x$ is invisible to the sensor. Any two scenes that share the measured component therefore produce the same bucket record no matter how much they differ in the null space. In the locked GI setting used below — $n = 4096$, $m = 205$ (a $5\%$ sampling ratio) — the measured subspace has dimension $205$ and the unmeasured subspace has dimension $3891$; most of what the eye reads as detail is carried by the unmeasured subspace.

### 2.2 Measurement consistency does not certify image content

Because $A P_0 = 0$, measurement consistency is a statement about the measured subspace only. This is made concrete by a construction. For any target record $y$ and any donor scene $x_j$, the feasible-but-wrong reconstruction

$$u_j(y) = x_j - A^{\dagger}\big(A x_j - y\big)$$

satisfies $A\,u_j(y) = y$ and $P_0\,u_j(y) = P_0\,x_j$: it reproduces the target bucket record exactly while carrying the donor's null-space content. It is demonstrated that such a reconstruction matches the record to ${\sim}10^{-15}$ — more tightly than the true scene matches its own noisy record, which sits at the ${\sim}10^{-3}$ noise floor — while remaining semantically far from the target (Fig. 1, left; Fig. 2). Alternating the exact projection onto the record with a clip to the physical range $[0,1]$ yields a box-legal feasible-but-wrong reconstruction that still matches the record to ${\sim}10^{-12}$; the barrier is therefore not an artifact of unconstrained linear algebra. Consistency is not correctness, and a reliability indicator based on the record alone must certify only what the measurement constrained.

[FIG: METHOD_FIG1.pdf | 1.0 | Fig. 1. Range–null accountability for undersampled ghost imaging. Left, a feasible-but-wrong reconstruction $u$ that matches the target bucket record to numerical precision, so measurement consistency cannot certify the unmeasured content ($\|Au-y\|/\|y\|\approx10^{-13}$, versus a $10^{-3}$ noise floor). Middle, the proposed audit (MCA) contracts only the measured modes, with per-mode factor $\lambda/(\lambda+\sigma_i^2)$. Right, the governed dial injects prior detail only through $P_0$, preserving $A\hat{x}_B=y$ while moving along the perception–distortion curve (locked: LPIPS $-32.6\%$ at $-0.45$ dB, $8/8$ gate).]

[FIG: WITNESS_GEOMETRY.pdf | 0.78 | Fig. 2. Geometry of the feasible-but-wrong reconstruction. Every point on the line (the record fiber $\{x: Ax=y\}$) reproduces the bucket record; only motion off the line is visible to the measurement. The constructed reconstruction is the donor scene projected onto the fiber: it matches the record to ${\sim}10^{-15}$, tighter than the true scene (dashed circle, radius ${\sim}\sigma_\varepsilon$), while carrying the donor's null-space content.]

### 2.3 The proposed measurement-consistency audit

For any reconstruction $\hat{x}$, the proposed audit can be described by

$$\Pi_y^{\lambda}(\hat{x}) = \hat{x} - G_\lambda\,(A\hat{x} - y), \qquad G_\lambda = A^{\top}(AA^{\top} + \lambda I)^{-1},$$

where $\lambda > 0$ denotes a regularization parameter chosen by a discrepancy principle, and the correction $G_\lambda(A\hat{x}-y)$ lies in the row space so that $P_0\,\Pi_y^{\lambda}(\hat{x}) = P_0\,\hat{x}$: whatever a prior placed in the null space passes through untouched. The audit needs only $(A, y, \lambda)$ — no ground truth and no access to the reconstructor. Diagonalizing $AA^{\top}$, the residual in the $i$-th measured singular mode of $A$ (singular value $\sigma_i$) is scaled by

$$c_i(\lambda) = \frac{\lambda}{\lambda + \sigma_i^2},$$

which is exact, image-independent, and reconstructor-independent; as $\lambda \to 0$ the audit becomes a hard projection onto the record fiber $\{x: Ax = y\}$. The factor $c_i(\lambda)$ is the complement of the Tikhonov filter factor [20]; its use here is a post-hoc, per-record reliability indicator applied after arbitrary reconstructors. In float64 the measured contraction matches the closed form to $1.0\times10^{-10}$, so the audit reports, mode by mode, how far a reconstruction stands from the recorded bucket signals.

### 2.4 The governed null-space dial

Because the null space cannot be certified, prior detail is not hidden there but metered. For classical and generative endpoints $d_A$ and $d_G$ of a reconstruction, the governed dial can be obtained by

$$\hat{x}_B = A^{\dagger}y + P_0\big(d_A + B(d_G - d_A)\big),$$

where $B$ denotes a scalar dial weight and the injected detail is confined to the null space, so that $A\hat{x}_B = A A^{\dagger}y = y$ holds exactly for every $B$. The dial meters how much prior-supplied detail enters the unmeasured subspace and traces a perception–distortion curve, while never converting the perceptual gain it produces into a measurement claim.

---

## 3. Results and discussion

### 3.1 The governed dial on locked ghost imaging

The proposed dial is evaluated on $64\times64$ grayscale STL10 scenes at a $5\%$ sampling ratio, using a signed, row-orthonormalized computational GI operator ($m=205$), with a representative generative (vector-quantized adversarial) prior for $d_G$ and a classical linear-minimum-mean-square-error reconstruction for $d_A$. Under a pre-specified, one-shot protocol with a frozen dial weight, the balanced setting reduces the perceptual distance (LPIPS) by $32.6\%$ relative to the projected classical endpoint at a peak-signal-to-noise-ratio (PSNR) cost of $0.45$ dB, and passes all eight pre-specified acceptance conditions ($8/8$ gate). The exact consistency $A\hat{x}_B = y$ is preserved to a relative measurement error of $3.6\times10^{-7}$ (the float32 pipeline scale) at every operating point. It is worth noting that the perceptual gain lives entirely on the quality axis; the exact consistency that accompanies it certifies reproduction of the record, not the correctness of what the record cannot see.

### 3.2 Forensic decomposition of published ghost-imaging pipelines

The proposed audit is next used as a forensic tool. For three published deep-learning ghost-imaging pipelines — a pretrained-and-fine-tuned reconstructor, an untrained deep-image-prior method on a real measured speckle operator, and a self-supervised method on a genuinely noisy record — the reported improvement is decomposed, through each pipeline's own released or measured operator, into row-space repair and null-space supply relative to a range ceiling (the PSNR of the oracle row-space image $P_R x$, an accounting reference for that operator). The results are shown in Fig. 4. Row-space repair saturates early, and the reported PSNR above the range ceiling is supplied from the null space; the row/null split is $61/39$, $60/40$, and $5/95$, respectively, and the terminal error is $93$–$97\%$ null in every case. On the real measured speckle operator, the audit drives the record residual from $3.0\times10^{-3}$ to $3.0\times10^{-15}$ at a PSNR change of $+0.20$ dB, and the measured per-mode contraction matches $\lambda/(\lambda+\sigma_i^2)$ to $3.5\times10^{-11}$ in float64, so the closed form holds on a physically measured operator, not only on designed ones.

[FIG: FORENSICS_CROSS_TARGET.pdf | 1.0 | Fig. 4. Forensic decomposition of three published ghost-imaging pipelines on their own operators. (a) A fine-tuning trajectory decomposed per step: the PSNR rises past the operator's range ceiling, and everything above it is supplied from the null space. (b) Attribution of each method's improvement into row-space repair and null-space supply. (c) Every reported result stands $+3.1$ to $+8.1$ dB above its own range ceiling, with terminal error $93$–$97\%$ null. GI: ghost imaging.]

### 3.3 A non-uniform per-mode spectrum on a real coded aperture

The per-mode structure of the audit is made visible on a snapshot compressive spectral imaging (single-disperser CASSI) operator, which shifts, masks, and sums a $28$-band spectral cube into a $256\times310$ coded snapshot — a $23.1\times$ undersampling — through the real released coded aperture (fill $0.563$). Because each voxel reaches exactly one detector pixel, $AA^{\top} = \operatorname{diag}(\Phi_s)$ is exactly diagonal, so the projectors and the entire singular spectrum are closed-form. The measured singular values $\sigma_j = \sqrt{\Phi_s}$ over $79{,}360$ detector modes span $[0.009, 4.907]$ (median $3.51$), a $528\times$ ratio, and the per-mode contraction $\lambda/(\lambda+\sigma_j^2)$ at $\lambda=10^{-3}$ therefore spans $[4.15\times10^{-5}, 9.21\times10^{-1}]$, a $2\times10^4$ spread (Fig. 5). This is the mode-resolved face of the same closed form that collapses to a single value for a flat-spectrum operator: for a masked-Fourier operator, as used in accelerated magnetic-resonance imaging on real data, all measured singular values equal one and the contraction is a single number $9.99\times10^{-4}$; the projector identities there hold in float64 to ${\sim}10^{-17}$. The two operators bracket the audit, from a flat spectrum to a strongly non-uniform one.

[FIG: CASSI_SPECTRUM.pdf | 1.0 | Fig. 5. The per-mode contraction on a real coded aperture. Left, the singular value $\sigma_j=\sqrt{\Phi_s}$ across the detector plane. Middle, the distribution of $\sigma_j$ over $79{,}360$ measured modes spans a $528\times$ range; the red line marks the single value ($\sigma=1$) of a flat-spectrum masked-Fourier operator. Right, the per-mode audit factor $\lambda/(\lambda+\sigma_j^2)$ spreads across four orders of magnitude, where a flat-spectrum operator collapses to a single point.]

### 3.4 A decade of learned spectral-imaging gains in the unmeasured subspace

The forensic decomposition is applied across thirteen published spectral-imaging models on the shared coded aperture, with the wiring validated by reproducing each model's published simulation PSNR (e.g., $34.38$ versus $34.26$ dB, and $38.18$ versus $38.36$ dB). The min-norm reconstruction $A^{\dagger}y$ scores $19.04$ dB (the range ceiling). It is demonstrated that twelve of the thirteen models supply $100$–$101\%$ of their PSNR gain over the ceiling from the null space, with a slightly negative row-space effect (the models mildly perturb the measured component rather than repairing it); one model is the sole exception at $93\%$. The entire decade of reported progress ($+6.5$ dB across the field) occurs wholly in the unmeasured subspace, and the models are $0.9$–$12\%$ inconsistent with their own coded snapshot (Fig. 6). On a real masked-Fourier operator the same decomposition gives $91$–$92\%$ null-supplied for an official single-coil reconstruction network, with the network $3$–$6\%$ inconsistent with its own record; a governed data-consistent variant retains nearly all of the reported gain while restoring exact record consistency. It should be emphasized that, at these sampling ratios, the null-dominance of a record-consistent estimator's error is expected by construction and is not the finding; the finding is the contingent content that undersampling does not entail — the measured record-inconsistency of published pipelines ($0.9$–$12\%$), the negative row-space effect, the specific $528\times$ non-uniform spectrum, and the agreement of the split across an entire model zoo. The ledger is an accounting of where each pipeline's quality is manufactured, not an accusation of error; the null content these models supply is often close to the truth, but its location — entirely in the uncertifiable subspace — is what the proposed audit makes explicit.

[FIG: CASSI_FORENSICS.pdf | 1.0 | Fig. 6. Reported spectral-imaging gains decomposed on the shared real coded aperture. (a) Thirteen published models against the $19.04$ dB min-norm range ceiling; twelve of the thirteen gains are $100$–$101\%$ null-supplied (one exception at $93\%$), and the row-space effect is slightly negative. (b) Each model's record inconsistency $\|A\hat{x}-y\|/\|y\|$ ranges from $0.9\%$ to $12\%$.]

### 3.5 Scope

All operators in this work are simulation- or declared-operator-level: the GI operator is simulated; the masked-Fourier leg uses an emulated single-coil operator; and the CASSI leg uses the declared integer-shift operator on the real released aperture rather than a physical dispersive capture. The measurement noise is modeled as additive Gaussian, a first-order proxy for the signal-dependent Poisson statistics of photon counting, and no hardware validation is performed; the certificates therefore concern the declared model of the sensor. The unmeasured content remains uncertifiable by any within-record method — this is a property of the geometry, not an engineering gap — and the proposed audit is precisely a statement of that boundary.

---

## 4. Conclusion

We have reported a ground-truth-free measurement-consistency audit for learned ghost and single-pixel imaging, together with a constructive impossibility, a forensic decomposition of published pipelines, and a governed null-space dial. It is demonstrated that measurement consistency cannot certify image content, that the proposed audit certifies the measured content of any reconstruction with an exact per-mode contraction $\lambda/(\lambda+\sigma_i^2)$ from the record alone, and that the reported gains of published deep-learning pipelines are largely supplied by the unmeasured subspace — $39$–$95\%$ for three ghost-imaging pipelines and $93$–$101\%$ for thirteen spectral-imaging models. On a real coded aperture the per-mode contraction resolves into a $2\times10^4$-fold graded profile that a flat-spectrum operator cannot exhibit, and the governed dial supplies prior detail only where the measurement is blind, preserving the recorded bucket signals exactly. The proposed audit provides a ground-truth-free reliability indicator that reports measurement accountability separately from image quality, and it could open up an avenue for accountable GI/SPI reconstruction in low-sampling-ratio and complex imaging conditions. Extending the audit to physically captured operators with photon-limited statistics is left to future work.

---

## Funding

Funding information to be added.

## Disclosures

The authors declare no conflicts of interest.

## Data availability

The data and code underlying the results are available from the corresponding author upon reasonable request.

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
12. R. Shang, M. A. O'Brien, and G. P. Luke, "Deep-learning-driven reliable single-pixel imaging with uncertainty approximation," arXiv:2107.11678 (2021).
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
