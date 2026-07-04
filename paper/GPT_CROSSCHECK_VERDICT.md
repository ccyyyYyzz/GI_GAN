# GPT Cross-Check — Adversarial Verification of the 3 GPT Survivors (2026-07-04)

Process: GPT (given our graveyard prompt) proposed 5 candidates, self-killed 2, advanced 3. We then ran 6
hostile referees (novelty + depth per candidate), seeded with our own sharpened attacks (Antun/Gottschling
Jacobian-instability lineage for C1; the Ward-2009 collapse for C2). **Verdict: 1 killed, 2 survive with
strict conditions — and the survivors converge with our own earlier audit.**

## C2 — Null-Only Audit Probes: **KILLED (both referees)**
The collapse is total, and worse than we thought:
- For fiber-consistent candidates, `x̃_k − x = A†e + P₀Δ_k` with `A†e` **identical for all k** — by
  orthogonality the vanilla Ward-2009 holdout statistic equals null error + a candidate-independent
  constant. **The ranking is verbatim Ward even with noisy y**; the "row-space subtraction" is only a
  Rao–Blackwellization (absolute calibration, not ranking).
- The killer from our own paper: §11.6's oracle ceiling — per-image dial selection buys ≤0.002 LPIPS even
  with ground truth — so a perfect per-image null-error estimate **has nothing to select for**.
- **Salvage (worth doing, cheap):** a ~1-page subsection of the existing paper, "the audit tax", explicitly
  citing Ward 2009 / Boufounos et al. 2007: q held-out rows buy an unbiased, ranking-invariant estimate of
  the entire ~3891-dim null error (vs shrinking the null space by ~0.8% if used for reconstruction) —
  pricing the *escape* from the no-adaptation lemma. Honest, small, fits the governance frame.
- Revival as standalone would require genuinely new theory: fixed-budget reconstruction-vs-audit allocation
  theorem (with a phase transition), concentration for physical probe ensembles (nonnegative correlated
  Poisson DMD masks where JL fails), or anytime e-value audits with optional stopping.

## C1 — Null-Steering Spectrum: **survives, but only as an INSTRUMENT SECTION, not a theory paper**
Both referees: not refuted, but graded A+B. The "theorem" `P₀J_f = J_g` is a chain-rule remark on
Schwab–Antholzer–Haltmeier's null-space-network *architecture definition* — presenting it as a theorem
"invites referee contempt". What has **no prior occupant** (adversarially searched): the spectrum/effective
rank of the null-projected data-Jacobian of a learned reconstructor, the LMMSE-alignment control
(α_f vs C₀y(Cyy+σ²)⁻¹ — deliberately the killed linear layer), and the governance read of the top singular
vectors of S_f as *monitorable measurement directions that maximally steer unverifiable content*.
**Mandatory survival conditions:**
1. Section of the existing paper; zero theorems; the contribution is the measurement instrument.
2. Demote the identity to a one-sentence remark crediting Schwab et al. (1806.06137).
3. Pre-register the **bias-vs-steering confound**: hallucination may be dominated by the zeroth-order bias
   term (what the prior paints in regardless of y), which a first-order Jacobian cannot see. Report
   zeroth-order `||P₀(f(y)−x)||` and first-order steering energy separately; α_f must add predictive power
   for held-out per-image oracle null MSE **beyond** each of: unprojected ‖J_f‖ (Antun instability),
   tr(J_f) (SURE/GDF), Bhadra value-level map energy, static null fraction. If it beats none → dead.
4. BP/Tikhonov are exact-zero *controls* (they output in range(Aᵀ)), not methods to distinguish.
5. Non-smooth models need definition-first: Gaussian-smoothed Jacobian with δ-sensitivity curves for VQ
   (straight-through codebook jumps); fixed seeds / mean-map for per-scene INR.
6. Position against the Tweedie/posterior-covariance line (Manor–Michaeli ICLR 2024; Nehme NeurIPS 2023):
   for Bayes-optimal f, S_f *is* the posterior null–range cross-covariance block; our delta is diagnosing a
   deterministic trained estimator vs that optimum. Cite or be scooped-adjacent.
7. Known-outcome risk: denoiser≈MMSE folklore predicts well-trained nets align with LMMSE — an
   "everything aligns" result reduces to confirming folklore and the section should then be dropped.

## C3 — Projector Forensics: **survives (depth: SOLID) — the strongest verdict, do this first**
Both independent models converged on this direction (it is our earlier audit's #2), and the depth referee
**verified the repos live**:
- `FeiWang0824/physics-driven-fine-tuning`: pretrained models via live Google-Drive link; the optimized
  sampling patterns **ship inside the model** → the exact operator A is released; GT-evaluated sim numbers
  (STL10 64×64 / CelebA 128×128, 1024 patterns, β=6.25%) to reproduce. Cleanest target.
- `FeiWang0824/GIDC`: real measured patterns+measurements in `data.mat` (exact own operator), but untrained/
  per-scene and the headline demo is experimental with **no GT** → only its sim numbers are auditable.
- `Noise2Ghost` (CEA, arXiv 2504.10288): modern pip-installable PyTorch, runnable on the 4060; its natively
  **noisy** regime is the one place the range-saturation claim is non-tautological.
- DLGI (feedforward) as a 4th target, ideal for measuring data-consistency violations.
**Conditions:** frame as a **pre-registered forensic case study** (2–3 clean + 1–2 conditional targets), NOT
a "literature audit" (n≥4 gate will not be met in weeks); two cleanly separated claims — Part A: decompose
each paper's *reported, GT-evaluated* number at its own operating point (headline = Bhadra prior-correct vs
hallucinated null split + range-consistency violations; at low noise "gains are null" is *expected*, the
split is the finding); Part B (labeled as our extension, not their claim): noise-dial saturation curves.
Learned-pattern methods (operator co-designed with prior) reported as their own category; GIDC experimental
claims listed as NOT auditable rather than proxied.

## Final consensus plan (two independent models + hostile review agree)
1. **Now:** C3 forensic case study — SOLID, feasibility verified repo-by-repo, evidence-type contribution
   the audit referees left open. First deliverable: reproduce physics-driven-fine-tuning's STL10 number,
   decompose it through its own released operator.
2. **Alongside/after:** C1 as an instrument section of the unified paper, executed under conditions 1–7
   (the ablation table is the go/no-go gate).
3. **Cheap add now:** C2's salvage — the 1-page "audit tax" subsection citing Ward 2009, pricing the escape
   from the no-adaptation lemma.
4. C2 standalone: dead unless one of the three named theorems is actually proven.
