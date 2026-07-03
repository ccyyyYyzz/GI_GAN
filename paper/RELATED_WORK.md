# Related Work

Our work sits at the intersection of computational ghost imaging, the range–null
geometry of undersampled linear inverse problems, generative image priors, and the
recent literature on hallucination, fundamental limits, and trustworthy
reconstruction. What organizes the discussion below is a single distinction: almost
every prior line uses the range–null decomposition, or a prior over the unmeasured
content, to **reconstruct** — to produce one image that looks right or scores well.
We instead use the same geometry to **certify, bound, and meter**: to prove what the
bucket measurement can and cannot vouch for, to audit that accountability at test
time without ground truth, and to inject admittedly invented detail only where the
measurement is provably blind. Each subsection first summarizes the line, then states
our delta.

## Ghost imaging and single-pixel imaging

Ghost imaging originated as a quantum-optical curiosity: Pittman et al. (1995)
recovered a magnified aperture image purely from photon-coincidence correlations,
with no image present in either detector's individual counts. Gatti et al. (2004)
then proved analytically that classically correlated thermal beams reproduce all the
imaging features of entangled ghost imaging, and Magatti et al. (2004) demonstrated
both ghost image and ghost diffraction from a single classical thermal source by
altering only the reference arm. Shapiro (2008) and Bromberg et al. (2009) collapsed
the modality to its computational core: with the illumination patterns known, a single
bucket detector suffices, so ghost imaging is classical coherence propagation and the
reconstruction is a correlation `C(rho) = sum_m phi_m(rho) y[m]` against known
patterns — precisely our forward operator `y = Ax`. In parallel, the single-pixel
camera (Duarte et al., 2008) and compressive ghost imaging (Katz et al., 2009)
established the undersampled regime `y = Phi x` with `m << n`, recovered by
`l1`/TV minimization, and stated the governing fact plainly: since `m < n`, infinitely
many images satisfy the same measurements, and recovery is possible only under a prior.
Gibson et al. (2020) survey this progression from correlation through compressed
sensing to learned reconstruction; Song et al. (2025) update it and flag interpretability
and generalization as unresolved.

**Our delta.** These works establish the physical instantiation of our operator `A`
and the undersampled regime that makes its null space non-trivial. We take the
"infinitely many `x` satisfy the same `y`" observation — usually stated in passing to
motivate a sparsity prior — and elevate it into the load-bearing object of the paper:
a constructive, per-instance feasible-but-wrong witness and a test-time certificate
over exactly this measurement model.

Learned ghost-imaging reconstructors sharpen the stakes. Lyu et al. (2017) restored
recognizable images at sampling ratios as low as `beta = 0.05`, where compressed
sensing collapses, and candidly noted the outputs "do not resemble exactly the ground
truth"; the CNN pipeline of Wang et al. (2018) and the U-Net denoiser of Shimobaba et
al. (2017) buy large PSNR/SSIM gains with no enforced measurement consistency; and a
recent X-ray study (2024) argues rigorously that reducing the number of measurements
does not itself add information, so any sub-sampling-limit detail a learned
reconstructor supplies originates from the prior, not the measurement.

**Our delta.** This GI-native line is the empirical hook for our converse: at low
`beta` the network fills the null space with trained-in structure the measurement
cannot certify. Prior work optimizes quality (MSE, recognizability, PSNR); we separate
quality from accountability, audit the learned reconstructor alongside classical ones,
and never claim the bucket certifies the invented texture.

## Range–null and data-consistency reconstruction

A large body of work exploits the range–null decomposition `x = P_R x + P_0 x` with
`P_R = A^dagger A`, `P_0 = I - A^dagger A`, and `A P_0 = 0`. The origin of the "edit
only the null space, keep the measurement exact" idea is IDBP (Tirer & Giryes, 2018),
which alternates denoising with a backward projection onto `{Hx = y}` so that all
restoration dynamics live strictly in the null space. Schwab et al. (2019) formalized
the null-space network `L = Id + (Id - A^+A)N`, proving it is a convergent
regularization that edits only `ker(A)` while preserving data consistency exactly, and
Chen & Davies (2020) trained separate range- and null-space networks over the same
`P_R = H^dagger H`, `P_0 = I - H^dagger H` split. In imaging practice, hard
data-consistency layers (Schlemper et al., 2018), model-based unrolling with an
explicit `(A^H A + lambda I)^{-1}` solve (Aggarwal et al., 2019), variational networks
(Hammernik et al., 2018), learned primal–dual (Adler & Öktem, 2018), ADMM-Net (Yang et
al., 2016), and ISTA-Net (Zhang & Ghanem, 2018) all restore or preserve the measured
component while a learned prior supplies the rest. The same construction anchors the
most recent null-space methods: RND-SCI (Wang et al., 2023) reconstructs hyperspectral
snapshots as `x = Phi^dagger y + (I - Phi^dagger Phi)q` with a conditional network
generating only the null term; NPN (Jacome et al., 2025) trains a network to predict a
low-dimensional null-space projection `Sx*` from `y`, explicitly noting that data
fidelity leaves the null space uncontrolled; and GSNR (Gualdron-Hurtado et al., 2026)
builds a graph-smooth basis for `Null(H)` via the null-restricted Laplacian
`T = P_n L P_n` and proves minimax coverage/predictability bounds for the invisible
directions, gaining up to 4.3 dB.

**Our delta — the null-space reconstruction cluster (position against).** This
cluster is our closest prior art and our sharpest point of departure. Every one of
these methods uses `A P_0 = 0` to install the *correct* content in the null space, and
treats consistency-by-construction as a fidelity virtue — a proxy for correctness.
RND-SCI generates a monolithic null term for speed and PSNR; NPN and GSNR go further
and argue the null content *is* predictable, learning it from `y` and enforcing only
soft fidelity, with GSNR's coverage/predictability curves computed at design time from
dataset statistics. We invert the entire premise. The same `A P_0 = 0` that lets these
methods edit the null space without breaking consistency is exactly why measurement
consistency **cannot certify** what they put there. We contribute (i) a constructive
feasible-but-wrong witness, matching the record to machine precision, showing the
construction is a barrier rather than a guarantee; (ii) a ground-truth-free, test-time
certificate that contracts each measured mode by exactly `lambda/(lambda+sigma_i^2)`,
a closed-form modal spectrum rather than a soft penalty or a design-time diagnostic;
and (iii) an exact-consistency dial that keeps `A x_hat_B = y` for **every** setting `B`
behind a pre-registered locked gate. Schwab et al. (2019), Chen & Davies (2020), and
IDBP (Tirer & Giryes, 2018) supply the theoretical scaffolding we build on, but none
poses a converse, audits accountability, or meters the invention.

Plug-and-play priors (Venkatakrishnan et al., 2013) and Regularization by Denoising
(Romano et al., 2017) decouple a forward-model solve from a black-box denoiser prior,
the conceptual ancestor of separating measured fidelity from supplied content.

**Our delta.** PnP and RED let the denoiser alter measured modes too and offer no
test-time measurement-accountability certificate; our injection is confined to the
null space so `A x_hat_B = y` holds exactly, and our audit reads the per-mode
contraction regardless of which prior is plugged in.

## Generative and VQ priors for inverse problems

Generative priors replace sparsity with a learned low-dimensional manifold. Bora et
al. (2017) showed `O(k log L)` measurements suffice to recover signals near a generative
model's range under the Set-Restricted Eigenvalue Condition — a condition that
requires differences of natural signals to lie *away* from `null(A)`. Shah & Hegde
(2018) give a projected-gradient GAN-prior solver with linear-convergence guarantees
under the same S-REC; Asim et al. (2020) use invertible (flow) priors with zero
representation error and prove recovery error is governed by the smallest `n-m`
singular values; Lunz et al. (2018) learn an adversarial regularizer that approximates
distance to the data manifold; and Yeh et al. (2017) hallucinate plausible content
strictly inside a missing region while pinning observed pixels. Untrained priors —
Deep Image Prior (Ulyanov et al., 2018) and the Deep Decoder (Heckel & Hand, 2019) —
show that architecture alone supplies image statistics, with the Deep Decoder making
the reachable set an explicit capacity knob. The vector-quantized line runs from
VQ-VAE (van den Oord et al., 2017) through VQ-VAE-2 (Razavi et al., 2019), which
factorizes an image into a global-structure code and a local-detail code, to VQGAN
(Esser et al., 2021), whose patch discriminator synthesizes crisp texture that plain
VQ-VAE cannot recover. Codebook priors dominate blind face restoration — VQFR (Gu et
al., 2022), CodeFormer (Zhou et al., 2022), and RestoreFormer (Wang et al., 2022) —
with CodeFormer exposing a single scalar `w in [0,1]` that trades fidelity for quality
at test time.

**Our delta.** We build our supply mechanism directly on this lineage: VQ-VAE-2's
structure/detail factorization motivates our structure (VQAE) plus detail (VQGAN)
split, and CodeFormer's scalar `w` is the closest interface to our dial `B`. But every
one of these controls acts on *entangled* features, so higher quality genuinely costs
measurement fidelity (`A x_hat != y`), and S-REC-style guarantees explicitly assume
away the regime — null-space differences that are uncontrolled — where we operate. Our
dial injects the same VQ-generated content **only** in `P_0`, so `A x_hat_B = y` holds
exactly for every `B`, and we treat the resulting texture, however realistic, as
content the measurement provably cannot certify. Asim et al.'s (2020) smallest-`n-m`
singular-value bound is the quantitative cousin of our per-mode
`lambda/(lambda+sigma_i^2)` contraction; we make that spectrum exact and auditable.

## Perception–distortion tradeoffs and perceptual metrics

Blau & Michaeli (2018) proved a fundamental, distortion-measure-agnostic
perception–distortion tradeoff: lowering distortion forces the reconstruction
distribution away from natural-image statistics, so "matches the reference" and
"looks real" are formally at odds; their rate-distortion-perception extension (2019)
adds that perceptual realism must be paid for in rate or distortion. SRGAN (Ledig et
al., 2017), ESRGAN (Wang et al., 2018) with its scalar network-interpolation dial, and
the PIRM challenge (Blau et al., 2018) operationalized this plane for super-resolution,
and LPIPS (Zhang et al., 2018) became the standard perceptual metric — the one we use
to report our locked improvement. PULSE (Menon et al., 2020) is the canonical exhibit
that consistency is not correctness in super-resolution: visually distinct realistic
faces all downscale to the identical low-resolution input.

**Our delta (position against).** Blau & Michaeli's tradeoff is *statistical* — a
divergence between the reconstruction distribution and natural-image statistics — and
its GAN Lagrange knob is the conceptual ancestor of our dial. Our converse is
*geometric and exact*: feasible-but-wrong images matching `y` to machine precision make
the null content unverifiable regardless of any distributional statistics. ESRGAN's
`alpha` and CodeFormer's `w` trade quality for fidelity along a soft curve; our dial is
null-space-confined, so the measurement stays exact at every point rather than at one
tradeoff position. We use LPIPS to quantify the perceptual win and keep it strictly on
the quality ledger, never conflating it with accountability. PULSE supplies
feasible-but-wrong images but presents one as "the" answer and audits nothing; we
govern and certify.

## Diffusion inverse-problem solvers

Modern training-free solvers use a pretrained diffusion model as a prior. The
score-SDE framework (Song et al., 2021) conditions an unconditional model on the
observation; Song et al. (2022) factor the operator as `A = P(Lambda)T` and keep
measured transform coefficients while synthesizing the rest; DDRM (Kawar et al., 2022)
diffuses in the SVD spectral space, synthesizing zero-singular-value (null) directions
and conditioning measured ones with a per-mode blend weight; MCG (Chung et al., 2022)
writes `A = I - P^T P` (our `P_0`) to update only the orthogonal complement while
pinning the measured subspace; DPS (Chung et al., 2023) adds a soft measurement-gradient
step for general noisy problems; and DiffPIR (Zhu et al., 2023) embeds
half-quadratic-splitting into the sampling loop. DDNM (Wang et al., 2023) is the most
direct methodological cousin of our dial: it fixes the range to `A^dagger y` and lets a
diffusion model synthesize only the null-space content `(I - A^dagger A) x_bar`, so
`A x_hat = y` holds exactly. The survey of Daras et al. (2024) taxonomizes the field and
gives the canonical noiseless data-consistency update
`(I - A^dagger A) E[X_0|x_t] + A^dagger y`.

**Our delta.** DDRM's per-singular-value treatment and DDNM's exact null-space fill are
the spectral and structural skeletons of our framework, and MCG's `A = I - P^T P` is
literally our `P_0`. The difference is direction and purpose. DPS, DiffPIR, DDRM
(`eta_b < 1`), and the score-based solvers enforce consistency softly, so
`A x_hat != y` in general and prior content leaks into measured modes; DDNM and MCG
keep the null-space move exact but chase SOTA restoration realism. We do not compete on
restoration quality and do not claim to beat diffusion. Instead, our certificate is a
ground-truth-free *audit* — not a sampler — that quantifies, per measured mode, how much
of any such solver's output the measurement can vouch for; and our dial governs the
null-space content these methods fill ungoverned, with an exact-consistency guarantee
and a locked gate they do not provide.

## Hallucination, instability, and fundamental limits

A sobering literature documents that consistency and visual quality decouple from
correctness. Antun et al. (2020) showed state-of-the-art learned MRI/CT reconstructors
are unstable to near-invisible perturbations and can erase small structural detail
while stable compressed-sensing baselines are not; Genzel et al. (2020) offered the
balancing counterpoint that, under matched adversarial testing, standard networks match
a robust TV benchmark and much instability traces to noiseless-training "inverse
crimes." Buday et al. (2026) inserted diagnostically misleading anatomical hallucinations
via imperceptible k-space perturbations that PSNR/NRMSE/SSIM cannot detect. Bhadra et
al. (2021) formalized hallucinations by projecting a reconstruction onto measurement and
null spaces, showing null-space hallucinations are attributable solely to the prior and
cannot be assessed without the ground truth; DynamicDPS (Kim et al., 2025) adopts the
same intrinsic (data-inconsistent) versus extrinsic (null-space) split and suppresses
both with a diffusion prior plus data consistency; and HalluGen (Kim et al., 2025)
fabricates controllable hallucinations by gradient ascent to train reference-free
detectors, noting that consistency-preserving errors are the hardest to catch. Learned
primal–dual (Adler & Öktem, 2018) even internally observes that a true feature is
indistinguishable from a false feature of the same size and contrast. The fundamental-
limits side is anchored by Iagaru et al. (2026), who prove necessary-and-sufficient
conditions for detail-transfer hallucination in (possibly nonlinear) inverse problems —
consistent decoders can only hallucinate details that are almost invisible in
measurement space (`|||f(x + x_det) - f(x)||| <= 2 epsilon`) — and give ground-truth-free,
forward-model-only algorithms that bound and assess hallucination magnitude via
feasible-set diameters (worst-case kernel size).

**Our delta — the limits-and-assessment line (position against).** This is the line
we must not overclaim against, and we cite it as the anchor for the barrier itself.
Antun et al. (2020), Buday et al. (2026), and Bhadra et al. (2021) establish
empirically and by SVD decomposition that consistency and quality metrics do not
certify correctness; Iagaru et al. (2026) prove the fundamental limit and occupy the
generic slot of forward-model-only, ground-truth-free hallucination *assessment*. We do
not claim "null content is unverifiable" or "ground-truth-free assessment" as novel —
that ground is theirs. Our contribution is on four axes they do not cover. First, where
Iagaru et al. give feasible-set-diameter *bounds* (a sup over feasible sets, requiring
paired data), we give an **exact** per-mode contraction `lambda/(lambda+sigma_i^2)` from
the operator's SVD, applied as a test-time plug-in audit across BP, Tikhonov, CS-TV, and
learned reconstructors. Second, where they establish *existence* via
necessary-and-sufficient conditions, we give a **constructive** per-instance
feasible-but-wrong witness that matches the record to machine precision. Third, we frame
and demonstrate a **quality-versus-accountability separation** (PSNR essentially flat
while measurement residuals drop orders of magnitude) as an explicit protocol. Fourth,
and entirely uncontested, we move from *assessment* to *governance*: a constructive,
metered null-space injection dial with `A x_hat_B = y` exact for every `B` and a
pre-registered locked gate. The clean axis is assessment versus governance, exact versus
bound, constructive versus existential. Bhadra et al.'s (2021) null-space hallucination
map requires the ground truth — exactly the quantity our converse proves unverifiable —
so we cite it to say the null-space error is real but cannot be certified at test time.

## Uncertainty, conformal prediction, and trust

The standard response to unreliable reconstruction is uncertainty quantification.
Kendall & Gal (2017) established the aleatoric/epistemic taxonomy; Narnhofer et al.
(2021) produce Bayesian epistemic-uncertainty maps for variational MRI. On the
distribution-free side, Angelopoulos et al. (2022) endow any image-to-image regressor
with per-pixel risk-controlling prediction intervals to flag hallucinations, conformal
risk control (Angelopoulos et al., 2022) guarantees `E[loss] <= alpha` for bounded
monotone losses, Wen et al. (2024) conformalize the downstream task output rather than
per-pixel maps, and Lu et al. (2022) apply ordinal conformal sets to disease severity
rating. Wen et al. (2024) in particular argue, as we do, that per-pixel uncertainty maps
are the wrong object because they miss many-pixel hallucinated structure.

**Our delta.** These guarantees are either heuristic (Bayesian std maps) or
ground-truth-dependent at calibration (RCPS, conformal risk control, task-conformal),
and they operate in output/pixel/label space. Our certificate is deterministic,
ground-truth-free at test time, and lives in the operator's spectral geometry: it
contracts each *measured* mode by exactly `lambda/(lambda+sigma_i^2)` and audits
measurement accountability rather than pixel error or a labeled risk. Crucially, using
Kendall & Gal's own taxonomy, our accountability is neither aleatoric nor epistemic —
it is a third, geometry-determined quantity: null-space unverifiability persists even
with a perfect model in the noiseless limit, because `A P_0 = 0` makes null content
invisible to `y` regardless of noise or model confidence. No calibration set can fix
what the converse proves.

## Cross-field framings: identifiability, information theory, and Bayesian inversion

Our thesis has deep provenance outside imaging. Econometric identification theory
(Koopmans & Reiersøl, 1950) established that observationally equivalent structures
generate the identical distribution, so a characteristic is knowable only if invariant
across all such structures — identification precedes estimation. Rothenberg (1971) tied
local identifiability to nonsingularity of the Fisher information matrix, making
non-identifiability a rank deficiency / null space in information geometry. Manski (2003)
founded partial identification — when data do not point-identify a parameter they confine
it to a sharp set, and honest inference reports that set — and Lewbel (2019) formalized
set identification and *normalizations*: restrictions that select within the identified
set without altering any meaningful quantity. On the imaging side, Landau (1967) fixed
the sampling-density threshold below which degrees of freedom exceed measurements and a
non-trivial null space is forced by necessity, Candès, Romberg & Tao (2004) proved exact
CS recovery under sparsity while noting that without the prior distinct signals share
identical partial measurements, and Stuart (2010) cast inverse problems as Bayesian
posteriors in which, when data dimension is below the unknown, the prior remains decisive
even in the zero-noise limit.

**Our delta.** These fields supply the rigorous vocabulary we import and instantiate.
Our feasible-but-wrong images are the imaging analogue of Koopmans & Reiersøl's
observational equivalence and Rothenberg's singular information direction; our null space
`P_0 = I - A^dagger A` is Manski's identification region and Landau's forced kernel; and,
most sharply, our governed dial is Lewbel's *normalization* — injecting null content is
"without loss of generality" for the measurement, since `A x_hat_B = y` exactly for every
`B`, leaving the identified quantity untouched. Candès–Romberg–Tao's converse is the
CS-side twin of the econometric one, and Stuart's "prior stays decisive" is the Bayesian
mirror of our per-mode contraction, whose factor `lambda/(lambda+sigma_i^2)` is exactly a
Gaussian-posterior variance reduction along each singular direction. We are the first, to
our knowledge, to unify these into an operational, ground-truth-free certificate plus a
governed injection dial for undersampled ghost imaging.

## Novelty statement

Every line above uses the range–null geometry, or a prior over the unmeasured content,
to **reconstruct**: install the "correct" prior in the null space for fidelity (Wang et
al., 2023 AAAI; RND-SCI; DDNM; MCG), learn to predict null content from `y` (NPN; GSNR;
Chen & Davies, 2020), fabricate null-like error to benchmark detectors (HalluGen), quantify
output uncertainty against calibration data (RCPS; conformal risk control; Wen et al.,
2024), or prove abstractly that the kernel forces hallucination (Iagaru et al., 2026;
Bhadra et al., 2021; Antun et al., 2020). We use the identical geometry to **certify,
bound, and meter**, and our contribution rests on four airtight, uncontested hooks.
**First**, we make the barrier *constructive*: an explicit feasible-but-wrong witness that
matches the same bucket record to machine precision — tighter than the noisy truth —
turning "consistency is not correctness" from a theorem about maps (Iagaru et al., 2026)
and an SVD diagnostic (Bhadra et al., 2021) into a per-instance object. **Second**, we
give a ground-truth-free, reconstructor-agnostic **test-time certificate** that contracts
each measured mode by *exactly* `lambda/(lambda+sigma_i^2)` — a closed-form modal spectrum,
not a feasible-set-diameter bound (Iagaru et al., 2026), a soft penalty (NPN; GSNR), a
design-time predictability curve (GSNR), or a calibrated statistical interval (Angelopoulos
et al., 2022) — auditing BP, Tikhonov, CS-TV, and learned reconstructors uniformly and
**separating image quality from measurement accountability**. **Third**, that separation is
demonstrated as an explicit protocol (perceptual/PSNR quality essentially unchanged while
measurement residuals fall orders of magnitude), an axis none of the reconstruction or UQ
lines report. **Fourth**, we convert null-space injection from an all-or-nothing fill into
a single-scalar, pre-registered **honesty dial** that fuses VQAE structure and VQGAN detail
and keeps `A x_hat_B = y` *exact* for every `B` behind a locked gate — moving the field from
*assessment* (does a reconstruction hallucinate?) to *governance* (how much invented detail
do we add, where the bucket is provably blind, without ever breaking or laundering the
measurement?). We explicitly do **not** claim SOTA, do not beat diffusion, and never claim
the measurement certifies invented texture — the point is that it provably cannot. That
inversion — **reconstruct, becomes certify / bound / meter** — is the slot this paper occupies.
