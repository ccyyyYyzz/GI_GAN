# Deep-Innovation Audit — What Is Actually Left (2026-07-04)

Trigger: the user's critique that recent innovation points are "A+B gluing." Method: 5 literature miners over
the GI99 cards → 6 orientation ideators (12 formal candidates) → 24 hostile referees (novelty + depth per
candidate) → 3 salvage candidates → 6 more referees. **41 agents, 15 candidates, 0 survived in original form.**
That outcome is itself the most valuable finding of the session.

## 1. The meta-finding: the linear-Gaussian layer is mined out

Every "new quantity" definable on `(A, σ, Gaussian prior)` was killed as a renaming of something with a
50-year-old name:

| our candidate | killed by (prior name) |
|---|---|
| Task Identifiability Index | **Backus–Gilbert 1968** (identified functionals from finite linear data); **Barrett–Myers** estimability / null functions |
| Photon-certifiable dimension / dose→accountability | **MacKay** well-determined parameters γ=Σλ/(λ+α); **Landau** degrees of freedom; **Harwit–Sloane 1979** (multiplex advantage reversal under Poisson) |
| Certificate-profile achievable region (operator design) | linear-Gaussian **water-filling / experimental design** (Elfving; Chaloner–Verdinelli 1995) |
| Certificate half-life under operator misknowledge | **Wedin/Davis–Kahan sin-θ**; Herman–Strohmer perturbed CS; Chi–Scharf basis mismatch; Eldar robust minimax |
| Observable algebra / gauge-variance of metrics | Backus–Gilbert again + identification theory (Koopmans–Reiersøl, Manski) |
| Calibration of P(x_null\|x_range) | **Gneiting–Balabdaoui–Raftery 2007** (PIT calibration); it *is* the Bayesian posterior on the fiber |
| Certified adaptive stopping | in linear-Gaussian models design criteria are **data-independent** (Lindley 1956; Chaloner–Verdinelli) — adaptivity has provably zero value there |

**Why everything feels like A+B:** on this mathematical layer only combinations remain. The feeling is
accurate, and it is a property of the layer, not of the researcher. Escapes must change layer: (i) empirical
findings about the *literature* (metrology), (ii) **nonlinear** functionals (where Backus–Gilbert stops),
(iii) **physically constrained** design (binary/nonnegative DMD patterns + Poisson budgets, where closed forms
genuinely do not exist).

## 2. URGENT defensive citations for the current paper

The referees found these instantly; real reviewers will too. The draft must cite and differentiate:
- **MacKay (1992)** — our per-mode certificate λ/(λ+σ²) is term-for-term his well-determined-parameter
  fraction. Differentiation: we use it as a *per-record audit primitive* with exact projectors and a converse,
  not as a model-selection heuristic.
- **Backus & Gilbert (1968–70)**; **Barrett & Myers, Foundations of Image Science** (estimability, null
  functions, task-based assessment). Differentiation: they cover linear functionals and statistical observers;
  our converse/dial/governance framing and the DL-era instantiation are the delta — but the lineage must be owned.
- **Landau (1967/1975)** degrees of freedom; **Bhadra et al., IEEE TMI 2021** (already card R031 —
  hallucination maps via the same P_R/P_0 decomposition); **Gottschling et al., "The Troublesome Kernel"**;
  **Shimron et al., PNAS 2022 "data crimes"** (the metrology genre template).

## 3. A free theorem the audit surfaced (state it, it is ours)

**No-adaptation lemma.** Any ground-truth-free selection rule s(y) is constant on the fiber
{x : Ax = y} — hence per-image null-space adaptation from y alone is *provably* a global prior in disguise.
One line from A·P₀ = 0, but stated nowhere in the library. It converts our C1 empirical negative
(GT-free selector = global B-shift artifact) into a **theorem confirmed by experiment**, and retroactively
explains the prior learned-gate failure. Add to §7.3/§11.6.

## 4. The three survivable narrowed forms (referee-prescribed, ranked for this lab)

### #1 — Certified-decision sandwich over the prior slab (from S2; novelty referee: not refuted, SOLID)
The only direction that **structurally** escapes the kill pattern: for a *nonlinear* classifier f,
fiber-invariance of the decision depends on f's decision geometry — not a functional of (A, σ, prior), so
not renamable to Backus–Gilbert/MacKay. Prior work has only the **attack half** (Antun/Gottschling
kernel-aware perturbations); no complete verification, no design dual.
- **Kill fixed:** whole-fiber certification is vacuous (our own P1 guarantees flippable twins → CDR ≈ 0)
  and 3891-dim CROWN is intractable. **Narrowed object:** certify over the fiber ∩ span(top-k null prior
  modes, k≈8–20) with ±3σ coefficient bounds — α,β-CROWN/MILP tractable on the 4060.
- **Deliverable:** a **sound/complete sandwich on P1**: twins = attack lower bound; verification = certified
  upper bound; per-scene gap = "certified fraction of fiber prior mass."
- **Gameability neutralized:** audit the *deployed* recon+classifier pipeline; report the y-domain-classifier
  baseline (CDR=1 by construction) and its accuracy cost explicitly.
- Must state the linear reduction (linear f → classical estimability) and claim only the nonlinear regime.

### #2 — The real literature audit, on the papers' own operators (from S1; novelty referee: not refuted, SOLID)
The *study* does not exist: no one has decomposed the DL-GI literature's **reported gains** through exact
projectors. Referee-fixed protocol:
- Only papers with **released code/weights**, each run on **its own published operator** (GI operators are
  explicit matrices → exact P_R/P_0 per paper). Reproduce each paper's reported number within tolerance
  *before* decomposing (else it collapses to "our reimplementations"). Accept n≈4–8 papers; frame as a
  metrology case study (genre: Shimron data-crimes).
- Attribution **only** via the exact MSE orthogonal split ‖x̂−x‖² = ‖P_R(x̂−x)‖² + ‖P₀(x̂−x)‖²; perceptual
  metrics reported separately with the range-only-hybrid confound explicitly controlled (matched-blur baseline).
- Split null content **prior-correct vs hallucinated** (Bhadra's convention: compare P₀x̂ to P₀x_true) —
  "unverifiable" ≠ "wrong."
- The defensible headline is the **saturation curve** (range-component contribution of every method family
  saturates at the linear/MMSE level) — with the noise-level dial pre-registered, since at near-noiseless GI
  SNRs the claim risks tautology. Deliverable includes the reporting standard: range-PSNR + null-fraction +
  hallucinated-null-fraction alongside image metrics.

### #3 — Physically constrained verifiability design (from S3's fix; hardest, deepest if it lands)
The epistemic-asymmetry narrative and "OOD censorship theorem" are killed (PCA-optimality tautology +
Fowler/Candès–Tao universality of random projections). What survives is where closed forms genuinely stop:
**binary/nonnegative DMD patterns + Poisson photon budget** — neither Bayesian D-optimal design nor
task-specific-information theory has solutions there. Prove/measure the Pareto frontier: certified
in-distribution fidelity vs worst-case anomaly range-energy, under the physical constraint set. Alternatively
fold the one-figure version (certificate coverage vs anomaly range-energy across mask families) into the
current paper as a safety corollary of P2/P3.

## 5. What NOT to pursue (killed with no survivable form)
Photon-Landau ceilings, certificate-profile achievable regions, gauge/observable algebras, null-posterior
calibration theory, certified adaptive stopping, operator-misknowledge certificate half-life — all renamings;
any revival must first differentiate against the named prior work above, and none of the referees could see how.

## 6. Recommendation
1. **Immediately** (this week, zero risk): defensive citations (§2) + the no-adaptation lemma (§3) into the
   draft. These protect the existing paper.
2. **Next experiment** (2–4 weeks, one 4060): the **certified-decision sandwich** (#1) — it is the only
   candidate whose novelty a hostile referee could not refute, it reuses the exact-projector machinery, it
   two-sides our own P1, and its first experiment (k≈10 prior-slab CROWN on an STL10/MNIST classifier) is
   concretely executable.
3. **In parallel, low intensity**: scope the literature audit (#2) by counting how many DL-GI papers actually
   release code + operator; if ≥4, it is a paper.
4. The current paper's true position, post-audit: its deep contribution was never a new mathematical quantity —
   it is the **converse + exact-audit instantiation + governance stance** for DL-era GI. Own the lineage
   (Backus–Gilbert → Barrett–Myers → us), and the "A+B feeling" dissolves: the paper is the bridge that field
   never built, and the two follow-ups (#1, #2) are the first bricks that are genuinely ours.
