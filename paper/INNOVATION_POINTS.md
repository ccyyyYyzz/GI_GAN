# INNOVATION POINTS — "Certify What You Measure, Govern What You Cannot"

Synthesized from the 99 GI reading cards (`GI99_reading_cards.json`, `method_core` + `innovation_spark`) and the verified threat differentiation (`NOVELTY_THREAT_DIFFERENTIATION.md`). The ~99 raw sparks cluster into the **13 strongest, de-duplicated innovation points** below.

**Framing invariant (the axis vs the field).** Every reconstruction paper in the corpus uses the range/null geometry (`P_R = A†A`, `P_0 = I − A†A`, `A P_0 = 0`) to *fill* the null space and call it restoration. We use the *same* geometry to **certify what the measurement accounts for, bound what it provably cannot, and meter the invented remainder** — reconstruct → certify/bound/meter.

**Honesty red line (applies to every point).** The measurement never certifies invented content. The certificate audits the *measured (range)* ledger; the dial governs the *null* ledger and is labelled invented by construction. No point below claims otherwise, and several are engineered specifically to make that separation visible.

---

## The 13 Innovation Points

### IP-1 — The Feasible-Wrong Twin: a constructive machine-precision witness
**Idea.** Turn the abstract "consistency ≠ correctness" barrier into a per-instance *object*: build an explicit image `x + P_0 v` that matches the same record `y` to ~2e-15 — tighter than the noisy ground truth — yet is visibly wrong. This upgrades an existence theorem into a demonstrated witness anyone can reproduce on a fixed operator.
**Strengthens.** P1 Converse (headline).
**Smallest first move.** Constrain the Antun-style gradient-ascent attack to the null space (`Ar = 0`) to produce a "zero-measurement-change adversary," then report `‖A x̂ − y‖ ≈ 2e-15` for both twins side by side.
**Boldness.** near-term-win.
**Cards.** 0, 20, 85/92 (2605.13146), 11, 91.

### IP-2 — Exact per-mode certificate `λ/(λ+σ_i²)` vs worst-case bounds
**Idea.** The certificate reports, for every singular mode, the *exact closed-form* contraction `λ/(λ+σ_i²)` — a full modal spectrum, not a soft penalty or a single worst-case kernel-size number. This is the sharpest, uncontested delta against the fundamental-limits anchor (2605.13146/Gottschling), which stops at a scalar `diam(F_y)` bound.
**Strengthens.** P2 Certificate (headline delta).
**Smallest first move.** One figure: our mode-wise `λ/(λ+σ_i²)` spectrum overlaid on the single global `diam(F_y)` bound — "we localize *which* modes are accountable, they give one number."
**Boldness.** near-term-win.
**Cards.** 2 (GSNR predictability `ρ² ≤ c/(c+μ)`), 21 (Anastasio measurement-space map as a sharp-truncation special case), 85/92, 98 (Stuart Bayesian posterior-variance pedigree).

### IP-3 — Ground-truth-free, reconstructor-agnostic quality/accountability separation
**Idea.** Run the same test-time audit across BP / Tikhonov / CS-TV / learned reconstructors and show PSNR/LPIPS stays flat while RelMeasErr drops 3–4 orders — proving the certificate measures the *operator's* accountability, not the reconstructor's quality. This is the "two ledgers are orthogonal" empirical spine.
**Strengthens.** P2 Certificate → Separation section.
**Smallest first move.** Reuse the fastMRI/GI operators already in the repo; produce two reconstructions with identical SSIM but very different (exactly quantified) null content, in the community's own metric.
**Boldness.** near-term-win.
**Cards.** 22 (fastMRI pixelwise-metric warning), 27 (MoDL DC block is `(AᴴA+λI)⁻¹`, certificate invariant to swapping `D_w`), 30 (ADMM-Net stage sweep), 67 (score-SDE sampler-invariance), 87/94 (GIDL vs CS-GI on identical measurements).

### IP-4 — The Governed Dial: single scalar B, `A x̂_B = y` exact for every B
**Idea.** Convert null-space injection from an all-or-nothing fill into a single pre-registered scalar `B` fusing VQAE structure + VQGAN detail, with measurement consistency held to ~2e-15 for *every* setting, behind a locked 8/8 gate. This is the constructive-governance hook the assessment-only literature entirely lacks.
**Strengthens.** P3 Governed Dial (the actual method — clear space).
**Smallest first move.** A one-figure ablation: `‖A x̂ − y‖` vs the dial for our-B (flat at 2e-15) beside CodeFormer-w / ESRGAN-α / DNI-α (drifting residual).
**Boldness.** ambitious.
**Cards.** 14 (DNSL `ALx=Ax`), 15 (DDN gated form), 53/63/66 (weight-space dials drift), 58 (CodeFormer w-knob), 44/55 (VQGAN adaptive λ), 72 (MCG null-only), 73 (DiffPIR ρ_t).

### IP-5 — Same geometry, opposite epistemic stance (the inversion catalogue)
**Idea.** Assemble a compact Related-Work table showing that the *identical* `x = A†y + P_0 q` construction appears in Wang-AAAI, RND-SCI, NPN, GSNR, DDNM, DNSL, DDN — all to *reconstruct* — while we alone use it to *certify/bound/meter*. The unoccupied slot is not a new operator; it is the inverted purpose.
**Strengthens.** P4 Stance + Related-Work (novelty defense).
**Smallest first move.** Quote proponents against themselves: NPN's "any null component may be arbitrarily modified" and IDBP's Thm 2 are our converse written by null-space advocates.
**Boldness.** near-term-win.
**Cards.** 0, 3, 4, 2, 19, 14, 15, 34 (IDBP), 26 (DC layer).

### IP-6 — Cross-field identifiability unification (the null space renamed four ways)
**Idea.** Give `P_0` a semantics beyond linear algebra: it is Koopmans–Reiersol/Manski's *observational-equivalence set*, Rothenberg's *singular-information direction*, and Landau's *below-density lost modes* — four names for one object. Casts the certificate as reporting the *identified* component and the dial as a *free normalization* within the identification region.
**Strengthens.** HINGE / Geometry section (gravitas + differentiation).
**Smallest first move.** A four-row unifying table (Rothenberg / Manski / Landau / our `A P_0 = 0`) plus Manski's "Law of Decreasing Credibility" as an epigraph.
**Boldness.** ambitious.
**Cards.** 82, 83, 88/95, 89/96, 86/93 (Landau).

### IP-7 — Impossibility trigger: single-operator regime makes null content provably non-identifiable
**Idea.** Use the equivariant-imaging rank conditions (`m|G| ≥ n`, `m > k + n/G`) to state *exactly when* the null space becomes identifiable — and show ghost imaging (`G=1`, `m ≪ n`) never reaches it. This is the formal switch from "certify recovery" to "govern the invented content."
**Strengthens.** P1 Converse / HINGE (formal trigger).
**Smallest first move.** A single-operator-vs-invariance contrast table; construct a feasible-wrong pair that *also respects* an assumed symmetry group to show even equivariance smuggles unverifiable content.
**Boldness.** ambitious.
**Cards.** 16 (EI), 17 (REI), 18 (Tachella `m > k + n/G`), 81 (equivariant review).

### IP-8 — Two-ledgers instrument: exact measurement guarantee ⊕ statistical task guarantee
**Idea.** Cleanly separate the two guarantees that the UQ literature conflates: our `A x̂_B = y` holds for *every* B with *no data*; wrap it in conformal risk control to *additionally* certify a distribution-free bound on a downstream task loss (e.g. detection FNR) as B injects detail. Two ledgers, two guarantee types, one honest statement.
**Strengthens.** P2/P3 → Two-Ledgers synthesis + Discussion.
**Smallest first move.** Add a CRC layer over the B-dial on a labeled calibration set and plot task-risk vs B while the measurement residual stays pinned at 2e-15.
**Boldness.** ambitious.
**Cards.** 74/75 (RCPS im2im-UQ), 80 (conformal risk control monotone knob), 76 (task-UQ acquisition), 79 (aleatoric/epistemic + the missing accountability axis).

### IP-9 — Re-axed perception–distortion plane: accountability is an orthogonal coordinate
**Idea.** The perception–distortion / rate–distortion–perception tradeoffs are soft and probabilistic; our dial collapses their curve onto a vertical segment because null-only edits leave the measured subspace fixed. Add a third, ground-truth-free *measurement-accountability* axis the PIRM plane never had.
**Strengthens.** P3 Dial + Separation (framing upgrade).
**Smallest first move.** Re-plot a PIRM-style plane with the residual `‖A x̂ − y‖` as a third axis: our dial sweeps perception pinned at 2e-15; GAN-SR drifts on both.
**Boldness.** near-term-win.
**Cards.** 50/60 (P–D tradeoff), 51/61 (R–D–P), 54/64 (PIRM), 52/62 (LPIPS certifies nothing about measurement).

### IP-10 — LPIPS-up / certificate-flat demonstration (perceptual gain ≠ certified content)
**Idea.** A single dramatizing experiment: sweep B so LPIPS improves −32.6% while the per-mode certificate stays flat — visual proof that a moving perceptual metric certifies nothing about the measurement. Directly neutralizes the reviewer instinct that "better LPIPS = better recovery."
**Strengthens.** P2 Separation (the money figure).
**Smallest first move.** Already buildable from existing VQAE+VQGAN fusion assets: LPIPS-vs-B curve beside the flat certificate spectrum.
**Boldness.** near-term-win.
**Cards.** 1 (HalluGen: LPIPS rewards hallucination), 38/45 (SRGAN), 52/62 (LPIPS), 42 (VQ-VAE blur = null content).

### IP-11 — GI-native demonstrations (the paper speaks the readers' modality)
**Idea.** Ground every abstract claim in real ghost-imaging / single-pixel testbeds rather than only MRI: the "digit 2 reconstructed as 3 / fabricated stripes" failures are P1's converse in a modality the audience already trusts, and reconstruct-then-refine pipelines let the certificate drop in between the two steps.
**Strengthens.** HINGE + Separation (credibility to the target community).
**Smallest first move.** Re-run GIDL and CS-GI on identical GI measurements through the audit; show identical measured-mode contraction, wildly different null content/error.
**Boldness.** near-term-win.
**Cards.** 87/94 (GIDL), 90 (GICNN), 91 (X-ray GI dose, fabricated stripes), 97 (SPI review: field asks "quality/speed", not "certifiable"), 5/6/7/8 (physical GI antecedents of the range/null split).

### IP-12 — Live "consistency ≠ correctness" demos on the strongest baselines (DDNM/DPS/DDRM)
**Idea.** Rather than argue against diffusion in the abstract, run the certificate *on* DDNM/DPS/DDRM/PULSE outputs: their measured modes are perfectly accounted for while their diffusion-hallucinated null content is provably unverifiable — turning the strongest reconstructors into live P1/P4 exhibits, and swapping their null-fill for our governed B-fusion yields an exactly-consistent variant.
**Strengthens.** P1 + P4 Stance (turns threats into demonstrations).
**Smallest first move.** Certificate on a DDNM reconstruction (already exact-consistent in range) → show the null texture is uncertified; then A/B its diffusion null-fill vs our governed fill under the same `P_0`.
**Boldness.** ambitious.
**Cards.** 19 (DDNM), 69 (DDRM per-σ), 70 (DPS soft), 65 (PULSE multi-init variety → accountability), 71/72/73 (diffusion survey/MCG/DiffPIR), 68 (score-medical `P(Λ)T` special case).

### IP-13 — Two-knob null-space governance (structure vs detail), separately audited
**Idea.** Generalize the single scalar `B` into two governed knobs — one for injected *structure* (VQAE / top-level codes), one for injected *detail* (VQGAN / bottom-level codes) — mirroring VQ-VAE-2's structure/detail factorization, with the certificate reporting separately how much of each was invented. A finer honesty ledger without breaking exact consistency.
**Strengthens.** P3 Dial (extension) + Two-Ledgers synthesis.
**Smallest first move.** Split the fusion codebook into structure/detail channels routed through `P_0`; report per-channel null energy at fixed residual 2e-15.
**Boldness.** moonshot.
**Cards.** 43 (VQ-VAE-2 top/bottom), 56 (ViT-VQGAN factorized codes), 57 (VQFR parallel decoder / contamination), 59 (RestoreFormer two-source fusion).

---

## TOP 3 that most raise the paper's ceiling
1. **IP-2 — Exact per-mode certificate `λ/(λ+σ_i²)`.** This is the single uncontested technical delta against the #1 threat (2605.13146 gives only worst-case bounds). Owning "exact modal spectrum vs one global number" is what makes P2 defensible as *novel*, not just *ground-truth-free*.
2. **IP-6 — Cross-field identifiability unification.** Reframing `P_0` as the observational-equivalence set (Koopmans/Manski/Rothenberg/Landau) elevates the work from "a CS trick" to a genuine cross-disciplinary accountability framework — the ceiling-raiser reviewers reward as depth, and it hands us the "why should imaging people care" answer with a 1950 pedigree.
3. **IP-4 — The Governed Dial with exact `A x̂_B = y` ∀B.** Pillar 3 is entirely clear space (the assessment-only literature has nothing constructive). This is the paper's *method* and its most differentiated contribution; the assessment-vs-governance axis is where we stand alone.

## TOP 3 near-term wins buildable from existing assets
1. **IP-10 — LPIPS-up / certificate-flat figure.** Directly buildable from the existing VQAE+VQGAN fusion; one curve pair that dramatizes the whole quality/accountability separation.
2. **IP-1 — Feasible-wrong twin at ~2e-15.** A short null-constrained perturbation on an operator already in the repo yields the headline P1 witness with a reproducible residual number.
3. **IP-11 — GI-native GIDL vs CS-GI on identical measurements.** Uses established, cheap GI baselines to make "certify what you measure" reconstructor-agnostic *in the readers' own modality* — a Separation-table column and credibility anchor at once.

---

*13 innovation points. Honesty red line held throughout: the certificate audits the measured ledger; the dial governs — and labels as invented — the null ledger; the measurement is never claimed to certify invented content.*
