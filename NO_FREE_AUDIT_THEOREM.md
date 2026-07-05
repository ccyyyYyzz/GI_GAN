# The No-Free-Audit Theorem and the Impossibility Quadrilateral

*Draft theory note (2026-07-05). The clean negative result + the minimal honest escape, tying together the barrier paper's feasible-but-wrong witness, the no-adaptation lemma, Iagaru's feasible-set diameter, Ward's cross-validation, and the post-commitment challenge experiment (`challenge_falsification_poc.py`). Converged on by three independent adversarial passes + an independent GPT peer critique.*

## Setup

Undersampled linear imaging: an operator $A \in \mathbb{R}^{m\times n}$ with $m < n$ (ghost imaging, single-pixel, CASSI, MRI). The recorded data is
$$ y = Ax + \eta, $$
where $x\in\mathbb{R}^n$ is the scene and $\eta$ is measurement noise. Write $P_R = A^\dagger A$ and $P_0 = I - A^\dagger A$ for the orthogonal projectors onto the measured (row) and unmeasured (null) subspaces. "Hallucination" is content a reconstruction $\hat{x}$ supplies in $N(A)$, i.e. the null component $P_0\hat{x}$, which the record does not constrain.

**Definition 1 (measurement fiber).** The *fiber* of a scene $x$ is the affine set of all scenes producing the same clean record,
$$ \mathcal{F}(x) \;=\; \{\,x' : Ax' = Ax\,\} \;=\; x + N(A). $$

**Definition 2 (ground-truth-free audit).** A *GT-free audit* is any (possibly randomized) measurable map
$$ S \;=\; s(A,\,y;\,\omega), $$
whose only inputs are the declared operator $A$, the record $y$, and internal randomness $\omega$ independent of the scene. The output $S$ lives in any measurable space (a score, an interval, a set, a flag). Crucially, $s$ does **not** take $x$.

**Assumption (physical realizability).** The law of the noise $\eta$ may depend on $A$ and on the clean measurement $Ax$ (e.g. Poisson shot noise has rate $Ax$), but not on the null component $P_0 x$. Equivalently: *the physical record depends on the scene only through $Ax$.* This holds for additive noise independent of $x$, for signal-dependent Poisson noise, and for any noise whose law is a function of $Ax$.

## Theorem 1 (fiber-invariance of GT-free audits)

*For any two scenes $x, x'$ in the same fiber ($Ax = Ax'$), the audit output $s(A,y;\omega)$ has the **same distribution** whether the true scene is $x$ or $x'$.*

**Proof.** If the scene is $x$, then $y = Ax + \eta$. If it is $x' = x+h$ with $h\in N(A)$, then $y' = Ax' + \eta' = Ax + \eta'$, since $Ah=0$. By the realizability assumption $\eta$ and $\eta'$ have the same law (both determined by $Ax = Ax'$), so $y \stackrel{d}{=} y'$. As $\omega$ is independent of the scene, $s(A,y;\omega) \stackrel{d}{=} s(A,y';\omega)$. $\qquad\blacksquare$

In words: **a GT-free audit is a function of the fiber, never of the scene within it.**

## Corollary 1 (no per-image null-space bound)

*Let $g:\mathbb{R}^n\to\mathbb{R}$ be non-constant on some fiber — e.g. the hallucination error $g(x)=\lVert P_0(\hat{x}-x)\rVert$ for a fixed reconstruction $\hat{x}$, which varies as $P_0 x$ moves in $N(A)$. Then no GT-free audit can report a per-image bound on $g$ that is simultaneously valid (holds with probability $\ge 1-\alpha$) and non-vacuous.*

**Proof.** By Theorem 1 the audit produces the same output distribution for every $x\in\mathcal{F}$, yet $g$ takes different values across $\mathcal{F}$. Any bound valid for the record $y$ must therefore hold for **all** $x\in\mathcal{F}$ at once, so the tightest valid upper bound is $\ge \sup_{x\in\mathcal{F}} g(x)$. For the null-space error this supremum is $+\infty$ unless $\mathcal{F}$ is externally constrained (a bounded prior/support). The tightest legitimate quantity is precisely the constrained feasible-set diameter of Iagaru et al. — a worst-case, not a per-image, statement. $\qquad\blacksquare$

## Corollary 2 (no-adaptation lemma)

*A deterministic GT-free rule $s(A,y)$ is constant on each fiber. Hence any rule that appears to "adapt per image" to null-space content is a fixed function of the fiber alone — a disguised global prior, not per-image adaptation.*

This is the earlier no-adaptation lemma, recovered as the deterministic case of Theorem 1.

## The impossibility quadrilateral

The hypotheses of Theorem 1 are exactly three: the audit's inputs are only $(A,y,\omega)$ **(GT-free)**; the operator $A$ is fixed **(no extra measurement)**; and $s$ is a genuine observable. An idea is **elegant** in the target aesthetic precisely when it is *a one-line invariant of $(A,y)$* — but any such invariant is fiber-invariant, hence null-blind by Theorem 1. Therefore the four desiderata

$$ \{\text{elegant}\}\ \wedge\ \{\text{genuinely novel}\}\ \wedge\ \{\text{ground-truth-free}\}\ \wedge\ \{\text{no extra measurement / template / prior}\} $$

are **jointly unsatisfiable** for linear undersampled imaging. One may have any three:

| Escape (break one hypothesis) | Gains | Loses |
|---|---|---|
| clever invariant of the record | elegant, GT-free, no-extra-info | null-blind (Thm 1) |
| stored template / reference | per-image teeth | not GT-free |
| learned prior / distribution | teeth in expectation | only distribution-level, not per-image |
| extra measured rows (challenge) | per-image teeth, GT-free | not "no extra measurement" |
| nonlinear-in-object physical measurement | teeth | it **is** a new sensor/channel |

Every one of the twelve elegant candidates we adversarially killed is a corollary of Theorem 1: each computed a function of $\{\langle P_i,x\rangle\} = $ the record (so, fiber-invariant), or silently added measured rows (→ challenge), or needed a template (→ not GT-free). The failures are not independent; they are one theorem.

## Proposition (the minimal honest escape = mechanism design)

The least-committal escape is the extra-measurement one, executed with maximal honesty. **Draw $k$ isotropic rows $B$ ($b_i\sim\mathcal N(0,I/n)$) *after* the reconstructor commits to $\hat{x}$**, measure $y_B = Bx + \eta_B$ with $\eta_B\sim\mathcal N(0,\sigma^2 I)$, and test
$$ T \;=\; \lVert B\hat{x} - y_B\rVert^2/\sigma^2. $$
The fiber contracts from $x+N(A)$ to $x+N\big(\begin{smallmatrix}A\\B\end{smallmatrix}\big)$: $B$ injects exactly $k$ new linear facts about the object. For a precommitted $\hat{x}$ with error $e=\hat{x}-x$, $T\sim\chi^2_k(\lambda)$ with non-centrality $\lambda = \lVert Be\rVert^2/\sigma^2 \gtrsim (1-\varepsilon)\,k\lVert e\rVert^2/(n\sigma^2)$, giving the detection-power lower bound
$$ \Pr(\text{detect}) \;\ge\; 1-\delta - F_{\chi^2_k(\lambda_{\min})}\!\big(q_{k,1-\alpha}\big),\qquad \lambda_{\min}=(1-\varepsilon)\tfrac{k\lVert e\rVert^2}{n\sigma^2}, $$
by Johnson–Lindenstrauss / $\chi^2$ concentration (Rachel Ward, *Compressed Sensing with Cross Validation*, 2008). Unbiased error-energy estimate: $\widehat{\lVert e\rVert^2} = n\big(\tfrac1k\lVert B\hat x - y_B\rVert^2 - \sigma^2\big)$.

**This provably catches the barrier's own feasible-but-wrong witness** $u = x_j - A^\dagger(A x_j - y)$ (which matches $A$ to machine precision): our simulation catches it $100\%$ at $k\ge4$ with $<1\%$ extra measurement budget.

**Honest scope (what is *not* claimed).**
1. It does **not** beat the worst-case diameter: $N(\big[\begin{smallmatrix}A\\B\end{smallmatrix}\big])\ne\{0\}$ since $m+k<n$, so an **adaptive** adversary who sees $B$ before committing can hide in the stacked null space (our sim: the adaptive stacked-witness $u_{\text{stack}}=x_j - C^\dagger(Cx_j-d)$, $C=[A;B]$, evades the *same* $B$ $0\%$ caught, but a *fresh* $B$ catches it $100\%$). It beats a **precommitted** adversary.
2. It is *not* a new invariant — it is **mechanism design**. It equals ordinary held-out cross-validation *unless* commitment and non-leakage are enforced (if $B$ is drawn before training, reused, used for model selection, or known to the reconstructor, it collapses to standard validation / stacked inversion). The novelty is the **post-commitment, hidden challenge** protocol applied to null-space hallucination, not the concentration bound.

## The one sentence

> The barrier says no same-record statistic can see null-space hallucination; the challenge buys exactly $k$ new random linear facts about the object *after* the reconstruction has committed, converting an unfalsifiable null-space error into a $\chi^2$-detectable measurement residual.

Negative theorem (elegant, minimal, a clean impossibility) $\;\oplus\;$ the minimal information-injecting escape (mechanism design, honestly scoped). A positive per-image *gadget* that is also GT-free, elegant, and adds no information does not exist in the linear-Gaussian layer (Cor. 1); obtaining one requires leaving the layer — a genuinely nonlinear-in-object physical measurement, i.e. new hardware.

## Numerical confirmation

On a row-orthonormal $A$ ($n=4096$, $m=205$), a scene $x$ and a fiber-sibling $x' = x + h$ ($h\in N(A)$):

```
Thm 1:  ||A x - A x'|| = 5.4e-14            (record is fiber-invariant)
        GT-free audit output on x vs x': diff = 0.0e+00   (identical, Thm 1)
Cor 1:  ||P0 x - P0 x'|| = 62.95            (null content differs arbitrarily -> no per-image bound)
Escape: k=4  challenge rows: ||B(x)-B(x')|| = 1.656   (facts injected -> now visible)
        k=32 challenge rows: ||B(x)-B(x')|| = 6.239
```

The audit literally cannot separate $x$ from $x'$ (output difference exactly $0$), while their hallucination content differs by $63$; only extra measured rows make it visible. Full detection-power and adaptive-adversary boundary in `challenge_falsification_poc.py`.

## Appendix: the "fabrication component" reduces to null-space Wiener–Bayes (no new inequality)

The one avenue a careful skeptic leaves open is a *minimax theory of the fabricated component specifically* (not total error): decompose the null error and lower-bound only the "fabrication." Worked out rigorously (`fabrication_minimax.py`, closed form + Monte Carlo), it collapses to known frameworks. Write $a=P_N\hat{x}$ (asserted null content), $t=P_N x$ (true), null error $e_N=a-t$.

1. **Minimax over a symmetric null-ball $\{\lVert P_N x\rVert\le\rho\}$ (prior-free):** asserting $\lVert a\rVert=c$ gives worst-case null risk $\sup_{\lVert t\rVert\le\rho}\lVert a-t\rVert^2=(c+\rho)^2$, minimized at $c=0$. So **min-norm ("erasure") is the unique minimax reconstruction of null content; any fabrication pays excess $c^2+2c\rho$.** This is minimax-vs-Bayes / Chebyshev-center folklore (and the Chebyshev radius over a structured class is Iagaru's diameter).

2. **Gaussian prior $x\sim\mathcal N(0,\Sigma)$, closed form:** eraser null risk $=\operatorname{tr}(P_N\Sigma P_N)$; Bayes (fabricating) null risk $=\operatorname{tr}(P_N C P_N)$ with posterior covariance $C=\Sigma-\Sigma A^\top(A\Sigma A^\top+\sigma^2 I)^{-1}A\Sigma$. The **fabrication benefit is exactly**
$$ \operatorname{tr}(P_N\Sigma P_N)-\operatorname{tr}(P_N C P_N)=\operatorname{tr}\!\big(P_N\Sigma A^\top(A\Sigma A^\top+\sigma^2 I)^{-1}A\Sigma P_N\big), $$
the null content *predictable from the measured content via the prior's null↔measured cross-correlation*. Verified numerically ($49.94\to39.59$, benefit $10.35$ by both direct and formula; Monte-Carlo $49.90/39.56$). **Control:** a block-diagonal prior (null $\perp$ measured) gives benefit $\approx 0$ — fabrication helps *only* through correlation. This is the **Gaussian/Wiener linear-Bayes model** (posterior covariance), a known object.

3. **Cost of the bet (misspecification):** a fabricator trained on the wrong prior incurs null risk that *exceeds* the eraser under distribution shift (numerically $49.90$ eraser vs $39.56$ correct-prior vs $\mathbf{62.39}$ shifted-prior). This is the standard robustness of the minimax estimator.

**Verdict.** Hallucination $=$ a *null-space Bayes bet*: its benefit is the prior's null↔measured correlation (an exact trace formula), its robust alternative is erasure (min-norm, minimax over symmetric classes), and its failure mode under shift is quantified. Every piece is Wiener–Bayes + minimax-vs-Bayes + Chebyshev-center(Iagaru). **There is no new inequality here.** What is genuinely worth keeping is the *characterization* — it upgrades the empirical forensic ledger ($E_R,E_0$, null-share) from "accounting" to "the measured instance of a precise decision-theoretic tradeoff": each model's null-share is where it sits on the erasure↔fabrication↔correct spectrum, and fabrication is optimal iff the test scene's null content matches the training prior. That framing (rigorous, ledger-anchored), plus the impossibility and the constructive witness/challenge methodology, is the honest contribution — not a new theorem.
