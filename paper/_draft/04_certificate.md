## 4. The Accountability Certificate (Achievability)

Section 3 established a converse: the record $y$ fixes $P_R x$ and nothing else, so any two feasible reconstructions may differ arbitrarily in the null space while producing the same measurement. That result is destructive — it says what cannot be verified. This section is its constructive counterpart. If the measurement cannot certify null-space content, it should at least certify, exactly and without ground truth, the one thing it does constrain: the measured component. We give a test-time operator that does this, prove that it contracts each measured mode by a known factor determined only by $A$ and $\lambda$, and show that it can be applied after any reconstructor as a post-hoc audit.

### 4.1 A plug-in test-time audit

Let $\hat{x}$ be any reconstruction — analytic, variational, or learned — with residual $r(\hat{x}) = A\hat{x} - y$. Define the audit operator

$$\Pi_y^\lambda(v) \;=\; v - A^\top\!\big(AA^\top + \lambda I\big)^{-1}\big(Av - y\big), \qquad \lambda > 0,$$

which we write compactly as $\Pi_y^\lambda(v) = v - B_\lambda(Av - y)$ with $B_\lambda = A^\top(AA^\top + \lambda I)^{-1}$. The audit requires only the operator $A$, the record $y$, and a single scalar $\lambda$; it needs no ground truth and no access to how $\hat{x}$ was produced. It is therefore a *plug-in* audit: it wraps an existing pipeline rather than replacing it.

**The update touches only the row space.** The correction $-B_\lambda(Av - y)$ lies in $\operatorname{range}(A^\top)$, which is exactly the row space. Consequently the null-space component is left untouched:

$$P_0\,\Pi_y^\lambda(v) \;=\; P_0 v .$$

This is the achievability side of the same geometry $A P_0 = 0$ that drove the converse. The audit cannot, and does not, adjudicate the null space; it operates strictly within the subspace the measurement is accountable for. Whatever a prior has placed in $P_0$ passes through the audit unchanged — a property we will rely on in Section 6, where the injected detail must survive the audit intact.

### 4.2 Exact per-mode contraction

The audit does not merely reduce the measurement residual; it reduces it by an amount we can write down in closed form. Applying $A$ to the audited estimate and simplifying,

$$A\,\Pi_y^\lambda(v) - y \;=\; \big[I - AA^\top(AA^\top + \lambda I)^{-1}\big]\,r(v) \;=\; \lambda\,(AA^\top + \lambda I)^{-1}\,r(v).$$

Diagonalizing $AA^\top = U_r \Sigma_r^2 U_r^\top$ and expanding the residual in the left singular vectors $u_i$, each measured mode is scaled independently:

$$c_i(\lambda) \;=\; \frac{\lambda}{\lambda + \sigma_i^2}.$$

The interpretation is exact rather than asymptotic. After one audit, the residual along the $i$-th measured mode is multiplied by $\lambda/(\lambda + \sigma_i^2)$ — a factor fixed entirely by the operator's singular value $\sigma_i$ and the chosen $\lambda$, independent of the image and of the reconstructor. As $\lambda \to 0$ the contraction becomes a hard projection onto $\{Av = y\}$; for $\lambda > 0$ it retains a controlled residual, appropriate when $y$ is noisy and should not be over-interpreted as exact truth. This closed-form modal spectrum is the object we certify.

**Float64 verification.** The contraction formula is confirmed in double precision. For the Rad-5 operator ($\sigma_{\min} = 3.476$, $\sigma_{\max} = 5.454$) the maximum deviation between the measured contraction and $\lambda/(\lambda + \sigma_i^2)$ is $1.04\times10^{-10}$; for the Scr-5 operator ($\sigma_{\min} = \sigma_{\max} = 1.000$) it is $2.29\times10^{-12}$. The identity therefore holds to the floating-point floor. (Repeated float32 pipeline audits saturate at a solver floor and are not evidence for the modal identity; the certificate is a float64 statement.)

### 4.3 Post-hoc audit across reconstructor families

Because $\Pi_y^\lambda$ depends only on $(A, y, \lambda)$, it can be applied uniformly to reconstructions from unrelated methods. We audit backprojection (BP), Tikhonov, a small-subset CS–TV sanity check, and a learned reconstructor, on both the Rad-5 and Scr-5 operators. Across these rows the pattern is consistent: the relative measurement error $\mathrm{RelMeasErr} = \lVert A\hat{x} - y\rVert / \lVert y\rVert$ drops by three to four orders of magnitude while the PSNR barely moves. For the learned Rad-5 output, $\mathrm{RelMeasErr}$ falls from $3.68\times10^{-2}$ to $1.90\times10^{-6}$ for a PSNR change of $+0.0136$ dB; for learned Scr-5 it falls from $1.80\times10^{-2}$ to $1.80\times10^{-5}$ at $+0.0387$ dB. Aggregated over all audited rows, the residual reductions of three to four orders come at $|\Delta\mathrm{PSNR}| \le 0.039$ dB, and the sign of the certificate is stable across the audited conditions (18/18). The audit thus buys measurement accountability essentially for free in image quality — the separation of these two axes is the subject of Section 5.

### 4.4 Position: exact modal spectrum, not a feasible-set bound

The fundamental-limits line closest to this result is Iagaru et al. (2026), who prove necessary-and-sufficient conditions for detail-transfer hallucination and give forward-model-only, ground-truth-free algorithms that *bound* hallucination magnitude via feasible-set diameters (worst-case kernel size). Our contribution is not the ground-truth-free stance, which is theirs, but the form of the guarantee. Where their assessment yields a bound over a feasible set — a supremum requiring paired data to instantiate — we give an *exact* per-mode contraction $c_i(\lambda) = \lambda/(\lambda + \sigma_i^2)$ read directly from the operator's SVD and realized as a plug-in operator applied after BP, Tikhonov, CS–TV, and learned reconstructors alike. The distinction is exact-versus-bound and operator-versus-existence: a closed-form modal spectrum applied at test time, rather than a feasible-set-diameter estimate. This certifies precisely what the measurement is accountable for. It says nothing about the null space, and we do not read it as licensing the invented texture Section 6 will introduce — $A\hat{x} = y$ is not certification that the null-space content is the true scene.

*Verification.* All numbers in this section are drawn from the audit experiments in `main.tex` (the contraction certificate and post-hoc audit tables): float64 mode deviations $1.04\times10^{-10}$ (Rad-5) and $2.29\times10^{-12}$ (Scr-5); learned-output residual drops $3.68\times10^{-2}\!\to\!1.90\times10^{-6}$ (Rad-5) and $1.80\times10^{-2}\!\to\!1.80\times10^{-5}$ (Scr-5) at $|\Delta\mathrm{PSNR}| \le 0.039$ dB.
