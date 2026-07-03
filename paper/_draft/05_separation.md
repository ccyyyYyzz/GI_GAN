# 5. Quality Is Not Accountability: The Separation Law

The certificate of Section 4 contracts each measured mode by exactly $\lambda/(\lambda+\sigma_i^2)$. Section 3 established that the measurement pins down only $P_R x$. This section states the consequence as a formal separation: image quality, as scored by pixel metrics such as PSNR, and measurement accountability, as scored by the audit residual, are coupled *only* through the row-space part of the error. Where the residual error lives in the null space, the two quantities move independently. This is not a weakness of any particular metric; it is a geometric identity, and it is exactly what an accountability audit is for — it catches the failures a quality metric is structurally blind to.

## 5.1 The orthogonal error split

Let $\hat{x}$ be any reconstruction of a scene $x$, and let $e = \hat{x} - x$ be its error. Using the projectors of Section 3, decompose $e = P_R e + P_0 e$. Because $P_R P_0 = 0$, the two components are orthogonal, and the Pythagorean identity holds exactly:
$$\|e\|_2^2 = \|P_R e\|_2^2 + \|P_0 e\|_2^2.$$
The measurement sees only the first term: $A e = A P_R e = A(\hat{x} - x) = A\hat{x} - y$, so the audit residual $\|A\hat{x}-y\|$ is a function of $P_R e$ alone and is entirely insensitive to $P_0 e$. Pixel error, in contrast, integrates *both* terms. Define the pre-audit row-error share
$$s = \frac{\|P_R e\|_2^2}{\|e\|_2^2},$$
so that $\|P_0 e\|_2^2 = (1-s)\|e\|_2^2$. The scalar $s$ measures how much of a reconstruction's error the measurement is even able to reach.

## 5.2 The PSNR ceiling

An idealized hard audit — the limit $\lambda \to 0$ of the certificate — removes all row-space error and leaves the null-space error untouched. Its mean-squared error ratio is therefore
$$\frac{\mathrm{MSE}_{\mathrm{post}}}{\mathrm{MSE}_{\mathrm{pre}}} = \frac{\|P_0 e\|_2^2}{\|e\|_2^2} = 1 - s,$$
and the maximum PSNR gain available to *any* measurement-consistency correction of a given reconstruction is
$$\Delta\mathrm{PSNR}_{\max} = 10\log_{10}\!\left(\frac{\mathrm{MSE}_{\mathrm{pre}}}{\mathrm{MSE}_{\mathrm{post}}}\right) = -10\log_{10}(1 - s).$$

**Theorem (PSNR ceiling).** *For a reconstruction with pre-audit row-error share $s$, no measurement-consistency projection can raise its PSNR by more than $-10\log_{10}(1-s)$, and the soft audit with $\lambda>0$ realizes a strictly smaller gain whenever any visible row-space error survives the mode contraction $\lambda/(\lambda+\sigma_i^2)$.*

The interpretation is the separation law in one line: **auditing can only buy image-quality improvement from the part of the error that lies in the measurement-visible row space.** A reconstruction whose remaining error is mostly null-space prior content has $s \approx 0$, so its PSNR ceiling is near zero — the audit can drive the measurement residual to machine precision while barely touching PSNR. Trained networks sit in exactly this corner: they already have small row-error share, so re-imposing consistency is nearly free in PSNR but decisive in accountability. The two axes are decoupled precisely because $A P_0 = 0$.

This decoupling is visible in the audited networks of Section 4. The learned Rad-5 output moves from $22.192$ to $22.206$ dB — a change of $0.0136$ dB — while its relative measurement error falls from $3.68\times10^{-2}$ to $1.90\times10^{-6}$; the Scr-5 output moves $22.146 \to 22.185$ dB ($0.0387$ dB) while the residual falls $1.80\times10^{-2} \to 1.80\times10^{-5}$. Accountability changes by orders of magnitude; PSNR changes in the third decimal.

## 5.3 The range-share law tracks the sampling rate

The row-error share $s$ governs a *reconstruction's* PSNR headroom. A companion quantity, the row-space energy share $\rho = \|P_R x\|_2^2 / \|x\|_2^2$ of the *image itself*, governs the anchor ceilings that any measurement-consistent estimator can reach. The two use the same range–null orthogonality but answer different questions, and we keep them distinct: $\rho$ explains how much of the scene's energy is measurable at a given operator, while $s$ controls the audit's PSNR budget for a specific reconstruction.

Under the per-image mean-removed convention, $\rho$ tracks the sampling rate almost exactly: $\rho = 0.050$ at Rad-5 and $0.052$ at Scr-5 (5% sampling), rising to $0.101$ at Rad-10 and $0.099$ at Scr-10 (10% sampling). The row-space share is, to leading order, the fraction of dimensions the operator measures. The corresponding row-space PSNR ceilings ($14.304$, $14.311$, $14.541$, $14.534$ dB) are met by scrambled-Hadamard back-projection but not by Rademacher back-projection, whose DC/global-mean coverage differs. The lesson is that $\rho$ is set by the operator's sampling geometry, not by the reconstruction — reinforcing that the null space, which carries the remaining $1-\rho$ of the energy, is where a prior must act and where the measurement cannot follow.

## 5.4 What accountability catches that PSNR cannot

The separation law is not merely a bookkeeping identity; it is the reason a quality metric can pass while the reconstruction has quietly stopped depending on the data. We probe three such failures. In every case PSNR is nearly flat while the accountability residual moves by orders of magnitude — the operational signature of $s \approx 0$.

**Wrong measurements.** To test whether a trained reconstructor uses the recorded bucket data or merely emits a plausible prior sample, we feed each image another image's measurement vector $y$ (a batch roll) or shuffle the measurement coordinates, on 500-image probes. Across Rad-5, Scr-5, Rad-10, and Scr-10, wrong-$y$ inputs reduce PSNR by $12.174$–$14.793$ dB and shuffled-$y$ inputs by $14.537$–$17.026$ dB. The large drops confirm the network genuinely conditions its output on the recorded measurement rather than acting as a pure dataset prior. (This dependence is not a contradiction of the row-null geometry: the linear operator constrains only $P_R x$, but a *trained conditional* reconstructor uses $y$ to choose *which* null-space completion to emit; the collapse shows the conditioning is real, not that the geometry constrains the null space.)

**Coordinate shuffle.** The shuffled-$y$ arm above is a stronger perturbation than the batch roll, and the larger PSNR drops ($14.5$–$17.0$ dB versus $12.2$–$14.8$ dB) confirm that destroying the coordinate structure of the measurement degrades the reconstruction more than substituting a coherent but wrong record.

**Operator drift.** The most incisive test attacks accountability while leaving quality nearly untouched. In a simulation-scoped calibration-mismatch probe, the audit is performed with a *drifted* operator and the residual is then evaluated against the true operator. As the relative drift grows from $0$ to $0.05$, the Rad-5 post-audit PSNR moves only from $22.206$ to $22.178$ dB — a change of $0.028$ dB — while the residual against the true operator rises from $1.90\times10^{-6}$ to $4.88\times10^{-2}$, more than four orders of magnitude. The Scr-5 case behaves identically: PSNR $22.185 \to 22.155$ dB while the true-operator residual rises $1.80\times10^{-5} \to 1.26\times10^{-2}$. Drift silently destroys the contraction that the certificate certifies, yet PSNR barely registers it. A quality-only pipeline would report success; the accountability audit reports the mismatch.

## 5.5 Statement of the separation law

Taken together, Sections 5.1–5.4 establish the **separation law**: because $\|e\|_2^2 = \|P_R e\|_2^2 + \|P_0 e\|_2^2$ and $A P_0 = 0$, image quality and measurement accountability are coupled *only* through the row-space error share $s$, and are otherwise free to move independently. Quality metrics are blind to everything in the null space and to any accountability failure that does not first show up as row-space error — precisely the failures (wrong measurements, coordinate shuffles, operator drift) that the certificate is built to catch. This is why a reconstruction can score well and still be unaccountable, and why the two ledgers must be reported side by side: a high PSNR never implies the measurement was honored, and honoring the measurement never implies high PSNR. The separation is the empirical face of the geometry, and it sets up the sharper question of Section 6 — whether measurement consistency can certify *correctness* at all.
