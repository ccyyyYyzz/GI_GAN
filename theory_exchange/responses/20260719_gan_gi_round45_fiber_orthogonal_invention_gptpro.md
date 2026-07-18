# Round 45 — Fiber-Orthogonal High-Pass Innovation for GAN-assisted computational ghost imaging

## Decision

The strongest simple journal-level method is an exact-fiber fusion of the distortion-favorable structural residual and only the genuinely new high-frequency part of the adversarial residual. The existing smooth high-pass rule has the correct physical skeleton, but it mixes two effects: it adds a new GAN direction and it silently rescales the structural correction. The only necessary modification is therefore a per-image rank-one orthogonalization that removes the second effect.

I recommend **fiber-orthogonal high-pass innovation (FOHI)** and freeze all previously selected values:

$$
\omega_c=0.12,\qquad \tau=0.03,\qquad \alpha=0.5.
$$

For the stable VQAE base x_B and null-space corrections c_S (strong non-GAN structural arm) and c_G (GAN arm), define

$$
d=c_G-c_S,
\qquad
u=P_N H_{0.12,0.03} P_N d,
$$

$$
\beta_S=
\begin{cases}
\langle u,c_S\rangle/\|c_S\|_2^2,&\|c_S\|_2>0,\\
0,&\|c_S\|_2=0,
\end{cases}
\qquad
v=u-\beta_S c_S,
$$

and return

$$
\boxed{\hat x_{\rm FOHI}=\Pi_{\mathcal F_y}(x_B+c_S+0.5v).}
$$

Do not renormalize v after orthogonalization. Do not add a learned gate or per-image coefficient model. Keep the exact box-fiber projection. The held-out test remains unopened.

## 1. Optics-first interpretation

Let A be the GI operator, Q have orthonormal rows spanning row(A), and

$$
P_R=Q^TQ,\qquad P_N=I-P_R.
$$

Here n=4096 and rank(A)=200, so before active box constraints the measurement fiber has 3896 unmeasured dimensions:

$$
\mathcal F_y=\{x\in[0,1]^n:Ax=y\}=\{x\in[0,1]^n:Qx=z\}.
$$

The row space contains optical modes visible to the displayed patterns and bucket detector. The null space contains object perturbations producing no change in the ideal measurement record. Therefore c_S and c_G do not represent competing explanations of different photons. Once projected into ker(A), they are two prior-selected coordinates on the same equivalence class.

The structural solution is approximately a conservative conditional-mean choice: low variance, good morphology, and strong PSNR/SSIM, but attenuated uncertain detail. The adversarial solution is encouraged toward high-density natural-image feature statistics: sharper high-frequency content and lower LPIPS, but a greater risk of choosing the wrong conditional mode. Their complementarity is physically meaningful only inside the same measurement fiber. The GAN supplies no new measurement; it supplies a candidate tangent direction within the unmeasured part of the inverse problem.

The smooth radial high-pass H is already the minimal comparative-advantage prior. Its null-space compression

$$
K=P_NHP_N
$$

is self-adjoint, positive semidefinite, nonexpansive, maps into ker(A), and annihilates the measured row space. Indeed, for any q,

$$
q^TKq=(P_Nq)^TH(P_Nq)\ge0,\qquad \|K\|\le1,
$$

and P_NK=K while KP_R=0. This is cleaner than the learned six-band gate: the fixed operator adds a physical inductive bias but no sample-dependent capacity.

## 2. Why orthogonalization is the unique necessary improvement

The current filtered rule uses c_S+αu. Decompose u into the part parallel to c_S and the remainder:

$$
u=\beta_Sc_S+v,\qquad \langle v,c_S\rangle=0.
$$

Then

$$
c_S+\alpha u=(1+\alpha\beta_S)c_S+\alpha v.
$$

Thus the GAN branch currently performs two operations: it changes the amplitude of the already selected structural solution, and it adds a new direction. The first operation is not GAN-specific innovation; it is an unadvertised retuning of the strongest non-GAN correction. FOHI fixes the structural coefficient at one and attributes any causal gain to the second operation alone.

### Proposition 1 — nearest genuine innovation

Let T_S=ker(A)∩c_S^⊥. Then v is the unique solution of

$$
v=\arg\min_{w\in T_S}\|w-u\|_2^2.
$$

**Proof sketch.** Since c_S∈ker(A), the orthogonal projector onto T_S is

$$
P_{T_S}=P_N-\frac{c_Sc_S^T}{\|c_S\|_2^2}
$$

for nonzero c_S. Since u∈ker(A), P_{T_S}u=u-β_Sc_S=v. Orthogonal projection onto a closed subspace is the unique Euclidean nearest point. The zero-correction case gives v=u. ∎

This is a rank-one operation, introduces no trainable parameter, and is falsifiable. Define the removed parallel-energy fraction

$$
\eta_{\parallel}=\frac{\|\beta_Sc_S\|_2^2}{\|u\|_2^2+\epsilon}.
$$

If its mean is below a preregistered descriptive threshold of 1%, FOHI is practically inert and the existing rule should remain the main method. If it is nontrivial, FOHI predicts preserved LPIPS gain with improved or less seed-sensitive PSNR/SSIM. If frozen FOHI loses triple dominance, kill the modification without retuning.

## 3. Guarantees

### Proposition 2 — exact measurement consistency

Assume x_B∈F_y and c_S,v∈ker(A). For every scalar α,

$$
A(x_B+c_S+\alpha v)=y.
$$

If the proposal lies in [0,1]^n it is already feasible. Otherwise its Euclidean projection onto the nonempty closed convex set F_y exists uniquely and satisfies

$$
\hat x_\alpha=\Pi_{F_y}(x_B+c_S+\alpha v)\in F_y.
$$

For every truth x*∈F_y,

$$
\|\hat x_\alpha-x^*\|_2^2
\le
\|x_B+c_S+\alpha v-x^*\|_2^2
-
\|x_B+c_S+\alpha v-\hat x_\alpha\|_2^2.
$$

**Proof sketch.** The affine equality follows from Ac_S=Av=0. The final claims are the existence, uniqueness, and variational inequality of Euclidean projection onto a closed convex set. ∎

The intermediate clamp in the generator path is not an exact measurement projection. Only the final exact box-fiber solver supports the consistency claim. Report convergence, maximum intrinsic-record residual, box violation, and stationarity/proximal certificates.

### Proposition 3 — exact risk-improvement interval

Let x_S=x_B+c_S and let w∈ker(A), w≠0. For normalized squared error

$$
D_w(\alpha)=\frac1n\|x_S+\alpha w-x^*\|_2^2,
$$

define

$$
a_w=\langle x^*-x_S,w\rangle,\qquad b_w=\|w\|_2^2.
$$

Then

$$
D_w(\alpha)-D_w(0)=\frac{\alpha^2b_w-2\alpha a_w}{n}.
$$

For α>0, strict MSE improvement occurs if and only if

$$
a_w>0,\qquad 0<\alpha<\frac{2a_w}{b_w}.
$$

The line-optimal coefficient is α*=a_w/b_w. This follows by direct expansion of the square. If the proposal leaves the box, Proposition 2 shows that exact projection cannot destroy a pre-projection Euclidean-risk improvement.

For FOHI, ⟨c_S,v⟩=0, so with r*=x*−x_B,

$$
a_v=\langle x^*-x_B-c_S,v\rangle=\langle r^*,v\rangle.
$$

The condition is therefore especially clean: the new GAN coordinate improves distortion exactly when it aligns with the true missing null residual, and the frozen α lies inside its quadratic improvement interval. In population form, with A_v=E⟨r*,v⟩ and B_v=E||v||²,

$$
R_D(\alpha)-R_D(0)=\frac{\alpha^2B_v-2\alpha A_v}{n},
$$

so a fixed α improves expected MSE when A_v>0 and 0<α<2A_v/B_v. A generic second-network direction with zero expected alignment gives a strictly positive α² energy term and worsens expected MSE; diversity alone is insufficient.

### Proposition 4 — when distortion and perceptual losses improve together

Let ℓ_D be distortion, ℓ_P a perceptual loss, and ℓ_S=1−SSIM. Along x(α)=x_S+αv, assume each loss has directional smoothness

$$
\ell_j(x_S+tv)\le \ell_j(x_S)+t g_j+\frac{L_j}{2}t^2\|v\|_2^2,
$$

where g_j=⟨∇_xℓ_j(x_S,x*),v⟩. If

$$
g_D<0,\qquad g_P<0,\qquad g_S<0,
$$

then all three losses improve for every positive α smaller than

$$
\min_j\frac{-2g_j}{L_j\|v\|_2^2}
$$

and smaller than the positive box-safe line endpoint. The proof applies the quadratic upper bound to each loss and intersects the three descent intervals.

For a local LPIPS-style feature distance ||W(φ(x)−φ(x*))||², a concrete first-order sufficient condition is

$$
\langle WJ_\phi v,\;W(\phi(x^*)-\phi(x_S))\rangle>0.
$$

Thus joint improvement is possible when v is simultaneously aligned with the missing pixel residual and the missing feature residual. It is not guaranteed by high frequency alone. The adversarial discriminator can bias the proposal toward this shared descent cone, while the MSE/L1/SSIM/LPIPS generator losses restrain hallucination; paired validation and causal controls must establish that the learned direction actually has the required alignment. Euclidean projection preserves the MSE guarantee, but LPIPS and SSIM must be remeasured after projection.

## 4. Frozen method and minimal implementation

Use the already selected structural and balanced GAN checkpoints. For each image:

```python
d = c_g - c_s
u = geometry.null_project_flat(
    smooth_radial_high_pass(d, cutoff=0.12, transition=0.03).flatten(1)
)
s = geometry.null_project_flat(c_s.flatten(1))
den = (s * s).sum(1, keepdim=True)
num = (u * s).sum(1, keepdim=True)
beta = torch.where(den > 0, num / den, torch.zeros_like(num))
v = u - beta * s
proposal = x_base.flatten(1) + s + 0.5 * v
x_hat, audit = exact_box_fiber_project(proposal, intrinsic_record)
```

Use a dtype-scaled zero test in code, not a tunable ridge. Audit |⟨v,c_S⟩|/(||v||||c_S||+ε), η_parallel, and the exact projection certificate. Freeze cutoff, transition, α, architectures, losses, split hashes, checkpoint-selection rule, and metrics. No retuning is allowed by seed, rate, noise, operator, or test result.

## 5. Smallest simulation campaign, in priority order

A failed early gate stops later compute. All selection stays on development/validation data.

1. **Zero-training FOHI diagnostic on all 512 untouched validation images.** Compare the strongest structural reference, the current fixed fusion, and frozen FOHI. Use 10,000 paired bootstrap resamples. The primary conjunction requires the 95% CI for ΔPSNR and ΔSSIM versus structural to be entirely positive and the CI for ΔLPIPS entirely negative. Also report MSE, β_S, η_parallel, ||v||, validation-only oracle alignment ⟨r*,v⟩, the fraction for which α=0.5 is inside the exact MSE interval, box-safe line statistics, and pre/post projection audits. Do not choose between methods with a new weighted score.

2. **Causal controls under the identical frozen operation.** (A) same proposal source, generator architecture, training split, parameter count, LPIPS term, seed schedule, and checkpoint rule, but discriminator weight exactly zero; (B) replace c_G with the independent second-VQAE correction; (C) replace H with I−H as a frozen low-pass negative control, without energy matching or retuning. “GAN-essential” survives only if adversarial FOHI passes the triple criterion and both non-adversarial controls fail it. Otherwise narrow the claim to multi-prior fusion.

3. **Fixed-parameter multiseed confirmation.** Use exactly three complete training pipelines: the current seed and the two independently initialized pipelines already running. Do not assume their outcomes and do not cherry-pick checkpoints. Report each seed and a crossed seed-by-image bootstrap. Continue only if every seed mean has the correct sign in all three metrics and pooled intervals establish triple dominance.

4. **Measurement-rate test.** Build nested structured operators from one master pattern bank at numerical ranks {100, 200, 400}, preserving row-family proportions. Regenerate measurements and reconstructions. Use one preregistered training seed for this screen and no FOHI retuning. Claim rate robustness only for rates that pass.

5. **Noise and photon budgets.** Perturb measurements before every reconstruction. Screen normalized bucket SNR {infinity, 30 dB, 20 dB}. For complementary Poisson simulations use signal photons per original pattern pair {10^4,10^5,10^6}, 1% background, and eight count realizations per image. Regenerate base, structural, and GAN outputs from each noisy record. Report both absolute quality and paired FOHI-minus-structural gain.

   A critical qualification is that under noisy y_obs, the clean truth generally does not lie in Ax=y_obs. For a physical noise-robustness claim, all arms must use the same noise-aware data set, for example

   $$
   \mathcal F_{y,\delta}=\{x\in[0,1]^n:\|\Sigma_y^{-1/2}(Ax-y_{obs})\|_2^2\le\chi^2_{r,0.95}\}
   $$

   for Gaussian noise, or a fixed Poisson likelihood-deviance confidence set. If the implementation instead enforces exact equality to noisy y_obs, label it an algorithmic stress test rather than statistically correct noisy consistency.

6. **One genuinely different operator family.** Use balanced i.i.d. Rademacher/complementary masks at rank 200, normalized to the same row energy and photon accounting, with three preregistered operator seeds and one fixed training seed per operator. This is sufficient. If it fails, keep a fixed-structured-operator paper and remove operator-universal language.

7. **Optional physical readout only after gates 1–6.** It is not required for the main claim.

Publish per-image metric vectors, operator/split hashes, checkpoint manifests, and projection audits. Across multiple stress conditions use Holm correction or label them secondary/descriptive.

## 6. Kill/continue decisions

| Gate | Continue | Kill, omit, or narrow |
|---|---|---|
| Feasibility | Every final image is in [0,1]^n; projection converges; max relative intrinsic-record error ≤10^−7; orthogonality residual is numerical. | Kill any result that changes the record or lacks a projection certificate. |
| FOHI necessity | Mean η_parallel is at least the preregistered descriptive 1% level and frozen FOHI retains triple dominance. | If η_parallel<1%, retain the simpler current high-pass rule. If triple dominance is lost, kill FOHI without retuning. |
| GAN causality | Adversarial FOHI passes; discriminator-zero and second-VQAE controls do not; direct adversarial-minus-control contrasts favor GAN. | If a control also robustly triple-dominates, remove “GAN-essential.” |
| Frequency mechanism | High-pass succeeds and frozen low-pass does not. | If low-pass is comparable, remove high-frequency-specific claims. |
| Seed stability | Correct signs in each of three seeds and crossed-bootstrap triple dominance. | A sign reversal in any primary metric makes the journal claim unstable; do not hide it by adding seeds. |
| Rate robustness | Correct signs at ranks 100, 200, 400. | Restrict claims to passing rates; failure on regenerated rank 200 kills the main result. |
| Noise/photon | Paired gain remains positive in declared regimes using common noise-aware consistency. | State a minimum regime; do not infer physical robustness from noiseless caches. |
| Operator family | Pooled Rademacher results preserve all three directions. | Restrict novelty to the fixed structured operator. |
| Held-out test | Method, code, parameters, and claims are frozen before one separately authorized evaluation. | Any post-test tuning invalidates confirmatory status. The test is not opened in Round 45. |

## 7. Optional factorial-moment complementary-bucket readout

Let p=v/||v||, crest factor κ=sqrt(n)||p||_infinity, and dithered binary rows q_k∈{−1/sqrt(n),+1/sqrt(n)}^n with

$$
E[q_k]=\gamma p,\qquad \gamma=\rho/\kappa.
$$

For the structural anchor x_S define the missing coefficient β=⟨p,x*−x_S⟩. After subtracting the known anchor response, a complementary bucket difference Z has E[Z]=γβ. With K dither pairs,

$$
\hat\beta=\frac{1}{\gamma K}\sum_{k=1}^K Z_k,
$$

and its variance scales as κ²/(Kρ²) times dither nuisance plus shot variance. High-crest directions are therefore optically inefficient.

For any measured or shrunk coefficient beta_tilde, x_read=x_S+beta_tilde p gives

$$
E[\Delta SSE]=E[(\widetilde\beta-\beta)^2]-\beta^2.
$$

Hence the exact inclusion criterion is

$$
\boxed{MSE(\widetilde\beta;\beta)<\beta^2.}
$$

For an unbiased estimator this becomes Var(beta_hat)<β², i.e. coefficient SNR greater than one. Positive-part shrinkage and the significance gate are justified only as attempts to satisfy this inequality.

Use the existing fixed screen K=16, ρ=0.75, photons {10^4,10^5,10^6}, eight replicates, and 1% background, but correct two weaknesses before any optical claim: simulate noise in the original measurements and therefore in the structural anchor (the current diagnostic explicitly says `old_anchor_noise_simulated=False`), and match total photons/exposures or state the exact extra cost (16 complementary pairs normally add 32 binary exposures). Include the readout only if the bootstrap upper 95% CI of E[(beta_tilde−β)²]/E[β²] is below one, reconstruction metrics improve after the line/box restriction, matched-budget accounting passes, and crest statistics are acceptable. Otherwise omit it; FOHI is the cleaner main contribution.

## 8. Novelty and claim boundaries

The strongest defensible novelty claim, conditional on the gates above, is:

> A distortion-optimized and an adversarial reconstruction can be decomposed on the same exact GI measurement fiber into a structural coordinate and a high-frequency adversarial innovation. Orthogonally projecting that innovation onto ker(A)∩c_S^⊥ yields a parameter-free, exactly measurement-consistent fusion with a provable MSE-improvement interval and a shared pixel/feature descent condition. Matched adversarial controls test whether the gain is adversarial alignment rather than generic two-network averaging.

The novelty is not “GAN plus projection” alone. It is the combination of exact box-fiber consistency, the compressed filter P_NHP_N, the unique rank-one orthogonal innovation, the risk/descent theory, and causal controls.

Do **not** claim that the GAN recovers additional measured information; exact consistency makes the inverse unique; null-space membership proves correctness; the perception-distortion tradeoff is universally defeated; gains hold sample by sample; current evidence generalizes to arbitrary rates, noise, datasets, or operators; 0/12 second-VQAE candidates alone isolates the discriminator; this is a physical experiment; the factorial readout works before matched-photon and anchor-noise tests; the six-band gate is necessary; or the held-out test supports any statement before a separately authorized one-time opening.

## Final frozen recommendation

Use only

$$
d=c_G-c_S,\quad
u=P_NH_{0.12,0.03}P_Nd,\quad
v=u-\frac{\langle u,c_S\rangle}{\|c_S\|_2^2}c_S,\quad
\hat x=\Pi_{\mathcal F_y}(x_B+c_S+0.5v),
$$

with v=u when c_S=0. Run the zero-training validation diagnostic first. If it fails, retain the existing fixed high-pass fusion and add no module. If it passes, proceed through the causal, three-seed, rate, noise/photon, and one-operator-family gates. Keep factorial-moment readout secondary and optional.
