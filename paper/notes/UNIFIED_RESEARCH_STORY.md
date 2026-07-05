# Unified Research Story: Accountable Row Space, Responsible Null Space

> A blueprint for merging the two ghost-imaging manuscripts —
> **Paper 1** (`paper/main.tex`, conservative audit: *"Measurement Auditing for Learned Ghost Imaging: Certificates, Limits, and Prior-Supplied Content"*) and
> **Paper 2** (`outputs/compatibility/measurement_conditioned_vqgan/detail_fusion_paper/PAPER_DRAFT.md`, positive VQGAN detail fusion) —
> into **one** research story. Produced 2026-07-03. **This is a design blueprint for a NEW third artifact; it does not edit either original.**

## Chosen framing
A hybrid: **framing "accountability vs responsibility"** as the organizing spine, disciplined by **certificate-first ordering**. The dichotomy is the only candidate whose thesis is a single symmetric concept mapping one-to-one onto the shared geometry `A P_0 = 0`, and it doubles as a structural firewall against the program's strongest forbidden claim. Certificate-first ordering (the conservative spine is load-bearing and comes first; fusion is introduced only after the limit is proven) prevents the misread a dial-first opening would maximize.

## Title options
1. Accountable Row Space, Responsible Null Space: Certificates, Limits, and Measurement-Safe Detail Injection in Undersampled Ghost Imaging
2. Certify What You Measure, Own What You Invent: A Range-Null Theory of Accountable and Responsible Ghost-Imaging Reconstruction
3. The Bucket's Ledger and the Prior's Debt: What Undersampled Ghost Imaging Can Certify and What It Must Only Own
4. Two Ledgers, One Geometry: A Measurement-Accountability Certificate and a Measurement-Safe Detail Dial for Ghost Imaging
5. From Audit to Injection: Accountability, Its Verifiability Barrier, and Responsible Null-Space Fusion in Low-Rate Ghost Imaging

## Unified thesis
A single linear-algebra fact governs every undersampled ghost-imaging reconstruction: for sensing operator `A` with `m << n`, the orthogonal projectors `P_R = A^dagger A` and `P_0 = I - A^dagger A` split each image uniquely, and because `A P_0 = 0` the bucket measurement `y = Ax` constrains **only** the row-space component `P_R x`. This one fact defines two disjoint responsibilities:

- The **bucket is ACCOUNTABLE for the row space**: a plug-in, ground-truth-free audit `Pi_y^lambda` certifies measurement consistency by contracting each measured singular mode by exactly `lambda/(lambda+sigma_i^2)`, separates image quality from measurement fidelity, and reaches a hard **verifiability barrier** — feasible-but-wrong images satisfy the record to ~2e-15, so consistency is certifiable but correctness of null-space content is not.
- A **prior is RESPONSIBLE for the null space**: because no measurement can certify `P_0 x`, the honest move is not to hide prior-supplied content but to inject it exactly where it is provably invisible to the bucket — anchoring at a measurement-audited LMMSE estimate and fusing a reconstruction (VQAE) and adversarial (VQGAN) prior's null-space differences through a single scalar `B`, so that `A x_hat_B = y` holds exactly for every `B` and "how much invented detail" becomes a controlled, measurement-safe perception–distortion dial.

Accountability and responsibility are the two faces of `A P_0 = 0`: the audit certifies the ledger the bucket can be held to, and a responsible prior openly owns — safely and controllably — the debt the bucket can never pay. The paper's discipline is that these two ledgers **never trade currencies**: certified accountability is never used to launder invented detail as truth, and the perceptual win never claims certification.

## Arc
Every undersampled reconstruction lives in two disjoint linear-algebraic worlds fixed by one fact. **Act 1 (Accountability)**: because the measurement constrains only `P_R x`, a plug-in audit contracts each measured mode by `lambda/(lambda+sigma_i^2)`, separates quality from measurement fidelity (PSNR moves ≤0.04 dB while residuals drop orders of magnitude), catches wrong-y/shuffle/drift failures PSNR is blind to, and hits a hard barrier — feasible cross-class images satisfy the wrong record to ~2e-15. The same `A P_0 = 0` that lets the audit certify the row space is exactly what makes the null space uncertifiable. That is the **hinge** into **Act 2 (Responsibility)**: since no measurement can certify `P_0 x`, inject prior detail only where it is invisible to the bucket — `x_hat_B = x0 + P_0(d_A + B(d_G - d_A))`, giving `A x_hat_B = y` exactly for every `B` (RelMeasErr ~3.6e-7) and turning detail injection into a certificate-safe dial (locked LPIPS −32.6%, 8/8 pre-registered gate, 3/3 seeds, bounded PSNR cost). The synthesis: one geometric fact enables the dial, certifies its measurement fidelity, and bounds its epistemic claim — certify what you measure, own what you invent, and never read null-space detail as verified truth.

## Section-by-section outline (evidence mapping)

**Abstract** — *both*. States the dichotomy, both acts, the firewall; leads with certificate/limit, presents fusion as the licensed payoff. Evidence: `P_R/P_0`, `A P_0 = 0`; mode contraction `lambda/(lambda+sigma^2)`; RelMeasErr drops 3–4 orders at |ΔPSNR|≤0.039 dB; feasible wrong images ~2e-15 (16/16); `x_hat_B = x0 + P_0(d_A + B(d_G − d_A))`, `A x_hat_B = y`; locked LPIPS −0.0977 (32.6%, CI [−0.1016,−0.0940]), 8/8 gate; scope line "A x̂ = y is not certification."

**1. Introduction: Two Questions the Bucket Cannot Answer the Same Way** — *both*. Motivates accountability (measured vs prior-supplied) vs responsibility (what a prior owns); both reduce to `A P_0 = 0`; two-act, certificate-first structure; contributions + up-front non-claims (not SOTA/diffusion/hardware; no null certification; `B` not an oracle). Evidence: m=205 pins data, ~3891 null dims carry detail.

**2. One Geometry: The Range-Null Decomposition** — *both*. The single shared trunk. Evidence: compact SVD, `P_R = V_r V_r^T`, `P_0 = I − V_r V_r^T`, `A P_0 = 0`, `A P_R = A`, feasible set `A^dagger y + null(A)`; CAN certify m=205 row coords, CANNOT certify ~3891 null coords; rows-orthonormal fusion case `A A^T = I_m`, `A^dagger = A^T`.

**3. [ACT 1] The Accountability Certificate: Plug-In Audit and Its Contraction** — *paper 1*. Evidence: `Pi_y^lambda(v) = v − B_lambda(Av − y)`, `B_lambda = A^T(AA^T + lambda I)^-1`; `P_0 Pi(v) = P_0 v`; `c_i = lambda/(lambda+sigma_i^2)`; float64 1.04e-10 (Rad-5)/2.29e-12 (Scr-5); post-hoc BP/Tikhonov/CS-TV(n=8)/learned, RelMeasErr drops orders (learned Rad-5 3.68e-2 → 1.90e-6) at |ΔPSNR|≤0.039 dB, 18/18 rows (A1/A3/A4).

**4. [ACT 1] Accountability Is Not Quality: The Separation Law** — *paper 1*. Evidence: `||e||^2 = ||P_R e||^2 + ||P_0 e||^2`; Theorem 1 `MSE_post/MSE_pre = 1−s`, `ΔPSNR_max = −10 log10(1−s)`; range-share `rho` tracks sampling (Rad-5 0.050, Scr-5 0.052, Rad-10 0.101, Scr-10 0.099), `rho` distinct from `s`; wrong-y drop 12.2–14.8 dB, shuffle 14.5–17.0 dB, 5% A-drift residual 1.90e-6 → 4.88e-2 (A2/A5).

**5. [HINGE] The Verifiability Barrier: Where Accountability Ends** — *paper 1*. The pivot. Evidence: `u_ij = x_j − A^dagger(Ax_j − y_i)`, `A u_ij = y_i` ~2e-15 (16/16, 2.16e-15–4.00e-15 vs noisy truth 3.45e-3–7.46e-3), PSNR-to-target 7.70–11.38, car-vs-horse fig (A6/A7); `|P_0 x̂|` is a prior-content map NOT a pixelwise error map (Spearman/Pearson ~0; top-10% LOWER error for B/C) — image-level only (B5). Unified boundary: consistency ≠ correctness because `A P_0 = 0`.

**6. [ACT 2] Responsible Supply: Injecting Detail Where the Bucket Is Blind** — *paper 2 + new bridge*. Constructive hinge. Evidence: `x0` = measurement-audited LMMSE (ridge 1e-3, `A x0 = y`); matched VQAE `x_A` + adversarial VQGAN `x_G`; `d_A = P_0(x_A − x0)`, `d_G = P_0(x_G − x0)`; `x_hat_B = x0 + P_0(d_A + B(d_G − d_A))`; theorem `A x_hat_B = y+0 = y` for every `B`, RelMeasErr mean 3.6e-7/max 5.7e-7 (V2). New bridge: the same `A P_0 = 0` that forbids certifying null content guarantees editing it cannot break the certificate.

**7. [ACT 2] A Controllable Responsibility Dial: The Perception–Distortion Ladder** — *paper 2*. Evidence: protocol (64×64 STL10, n=4096, m=205, 1 DC+128 DCT+56 Hadamard+20 random, seed 772001, 3 seeds, raw-hash-disjoint locked 512, `B` frozen on val, gate once); balanced LPIPS −0.0977 (32.6%, CI [−0.1016,−0.0940]), PSNR −0.45 dB, RMSE +0.0039, RAPSD improves, 3/3, 8/8 PASS; KID 0.119 → 0.043 (2.7×); dev −0.0965 ≈ locked −0.0977 (V1/V3); ladder LMMSE 0.404 → VQAE 0.300 → Balanced 0.202 → Quality-lite 0.182 → Full VQGAN 0.172; dense 21-pt sweep monotone; scalar beats 16-band + learned gate (`B` fixed, NOT per-image oracle).

**8. Robustness and Reach of Responsible Injection** — *paper 2*. Evidence: cross-rate DEVELOPMENT-LEVEL ONLY — 2% −0.116 (29.3%), 5% locked −0.098 (32.6%), 10% −0.076 (34.2%), 3/3 (V4); noise — balanced overtakes full VQGAN at σ=0.02 (0.197 vs 0.204), beats at 0.05 (0.250 vs 0.293); breadth 97.5%/99.2% images improved, worst +0.07 LPIPS, failures on man-made periodic structure.

**9. Two Ledgers, One Geometry: Why Accountability and Responsibility Must Not Trade Currencies** — *new*. The unique synthesis neither source paper can state alone. Evidence: audit certifies only `P_R` (Act 1), fusion lives only in `P_0` (Act 2); `A P_0 = 0` is simultaneously guarantee and limit; two fused recons differ almost entirely in `P_0`, so a prior raises naturalness without guaranteed reduction in distance to true `P_0 x`. Prescription: report BOTH audit AND quality metrics; label content measured / prior-supplied / unverifiable.

**10. Related Work** — *both*. Range-null/DDNM (decomposition → here a test-time audit + contraction certificate + barrier + dial); adversarial regularization (prior under a certificate; gauge shortcut 0.4767 → 0.0, B2/B3/B4); deep null-space learning (VQAE branch); data-consistency layers (audit is a soft DC layer, novelty = mode contraction + separation + hallucination boundary); explicit non-comparison to DPS/DDRM/PiGDM — certified only in `P_R` (C4).

**11. Discussion, Scope, and Limitations** — *both*. Simulation-only, single GI problem; Act 2 locked at one 5% operator / 64×64 (2%/10%/noise development-level); GAN branch representative not SOTA, standard-cGAN mixed, +0.0148 dB tiny; z-sampling diversity NEGATIVE/abandoned (std ~7.19e-4) — cited only as coverage limit; failure detector weak (~0.64); `B` fixed not oracle; `|P_0 x̂|` not a pixelwise locator; explicit: `A x̂ = y` never certifies texture is the true scene.

**12. Conclusion** — *both*. Consistency certifiable; correctness of prior-supplied content not; therefore inject prior content safely (`A x_hat_B = y`) and controllably (single `B`, bounded cost, 8/8 gate). Next: hardware, locking cross-rate/noise, other operators/resolutions, calibrated null-space uncertainty.

**Appendix A: Contraction and Calibration Diagnostics** — *paper 1*. Float64 modal contraction (f32 k=2 saturates); T6/T7; posterior-calibration gates as a LIMIT only (collapse P_0 var 1.09e-6 → anti-collapse 1.28e-3, slope −2.62, coverage ~45–48% at 90%, base P_0 RMSE ~0.081) — no diversity claim (B7).

**Appendix B: Fusion Protocol, Provenance, and Ablations** — *paper 2*. rows_sha256=8a16664e…, locked_source_indices_sha256=103976e4…, overlap 0/intra-dup 0 vs 60,497 hashes; frozen B {0.55,0.55,0.50}/{0.75,0.75,0.70}; 8-condition gate; bootstrap (2000); Table 5 ablation (scalar beats band-weighting + learned gate).

## Claim scope
**Makes (supported):** A1–A7 (separability; range-share/ceiling; post-hoc audit across BP/Tikhonov/CS-TV/learned; float64 modal contraction; wrong-y/shuffle/drift catches; feasible wrong images 16/16 ~2e-15; consistency ≠ correctness); B1–B6, B8 (GAN improves detail *descriptively only*; gauge removes row shortcut 0.4767→0.0; standard cGAN comparable; gauge-AUC diagnostic; unmeasured-content map at image-level correlation only; alpha knob with invariant certificate; weak failure detector); V1–V4 (locked LPIPS/gate/RelMeasErr; exact `A x_hat_B = y`; KID 2.7×; cross-rate development-level).

**Avoids (forbidden, firewalled):** "bucket certifies null-space texture is real" (Sections 5 & 9 assert the opposite); posterior z-diversity (B7, excluded); discriminator-is-the-certificate (C1); exact-null-critic / G1-sampling success (C2/C3); SOTA / beats diffusion / hardware (C4); per-image oracle `B`; `|P_0 x̂|` as a pixelwise error locator; a LOCKED cross-rate claim; high-PSNR implies accountability / consistency implies semantic truth.

**Red line addressed head-on:** the claim ledger forbids folding the positive VQGAN draft INTO the conservative IEEE-TCI submission and diluting its peer-review standard. This deliverable is a **NEW THIRD artifact for a NEW venue**, not an edit to either live submission — both originals remain untouched. Merging is now sound because (a) it is a NARRATIVE unification, not a claim-merge — the accountability/responsibility dichotomy is itself the firewall, and Section 9 forbids trading currencies; (b) the ordering is certificate-first, so fusion is subordinated to the proven limit; (c) the perceptual gain is confined to `P_0`, which the certificate PROVABLY cannot certify, and the paper says so explicitly. The conservative standard is protected structurally, not merely by disclaimer.

## What changes from each source
**Paper 1 (audit):** KEEP the decomposition, audit + contraction certificate, separation law, and verifiability barrier as the load-bearing spine. REFRAME the barrier as the *license* for Act 2 (premise, not just limitation), renaming the territories accountability/responsibility. DEMOTE the GAN case study (+0.0148 dB, 96×96) — the VQGAN fusion is now the constructive payoff; keep gauge results only in Related Work + Appendix A. CUT z-sampling as any positive claim; retain only as a coverage limit.

**Paper 2 (fusion):** KEEP the full constructive engine (two-branch method, fusion formula, exact-consistency theorem, locked result, ladder, reproducibility). REFRAME as "responsible supply" that Act 1's barrier licenses; its `A x_hat_B = y` guarantee becomes a corollary of the audit paper's own `A P_0 = 0`. SOFTEN cross-rate/noise to development-level throughout.

**Both:** RECONCILE the differing operators/resolutions (Rad-5/Scr-5 96×96 vs 64×64 STL10 5%) by stating Sec 2 geometry as operator-agnostic and compartmentalizing the two settings; DISAMBIGUATE the test-time audit `Pi_y^lambda` from the training/anchor LMMSE audit.

## Open decisions (for the author)
1. **GO/NO-GO on the merge:** confirm this is a THIRD artifact for a NEW venue, not a replacement/edit of either live submission (touches the ledger red line).
2. **Target venue** that tolerates both a limits/certificate contribution and a positive perceptual result; accept 12 sections + 2 appendices breadth.
3. **Operator/resolution reconciliation:** operator-agnostic Sec 2 + compartmentalize, or re-run one act on the other's operator for a single unified setup.
4. **Prominence of Paper 1's GAN case study:** demote to Related Work + Appendix A (recommended) vs a short main-text paragraph.
5. **Naming of the two "audits"** (`Pi_y^lambda` vs anchor LMMSE) to prevent reviewer conflation.
6. **anti-defensive-writing on boundary/limitations:** caveats are load-bearing (they generate the constructive idea) — decide how hard to trim without weakening the firewall.
7. **Depth of Section 9** ("two ledgers, one geometry"): the novel contribution — how far to push vs keep tight.
8. **Title selection**, and whether to foreground accountability/responsibility (thesis-carrying) or certificate/dial (more literal).
