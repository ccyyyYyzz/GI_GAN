## Research Director's Reading Guide
### Cross-field literature to strengthen the WRITING of the GI+GAN imaging paper

Your paper already has a clean four-pillar skeleton: (1) a blind subspace, (2) consistency ≠ correctness, (3) a ground-truth-free certificate, and (4) governed injection. The job of outside reading is **not** to add machinery — it is to borrow *established vocabulary and rhetorical scaffolds* so that each pillar reads to a skeptical reviewer as "a known result correctly applied here" rather than "a novel optics claim we are asserting." Below are six themed directions, then a triage on what to read first, ready-to-adapt sentences, and an honesty red-line note.

---

## 1. Theme: **The Identifiability Firewall**
*Maps to pillars (1) blind subspace and (2) consistency ≠ correctness.*

**Why it maps.** Three fields say the same thing your null space says, in three vocabularies a reviewer already trusts. Econometric **identification** is literally the theory of pre-data equivalence classes: a parameter is *not identified* when distinct values are observationally equivalent — exactly your `A P_0 x = 0` leaving an affine family `x + null(A)` that produces the same `y` to machine precision. Classical **regularization** gives it as a textbook theorem: any least-squares solution splits into a minimum-norm row-space part plus an undetermined null-space part (Moore-Penrose), and non-uniqueness is the failure of Hadamard's *second* well-posedness condition. **Duhem-Quine underdetermination** is the philosophy-of-science version: evidence never uniquely fixes a hypothesis; the residual is settled by non-empirical commitments (your prior).

**What to borrow.** The triad *observational equivalence / identified set vs point identification / identifying restrictions* (Koopmans-Reiersol's doctrine that identification is settled BEFORE estimation — use it as your Intro's logical spine). Manski's *partial identification / report the set, not a false point* precedent legitimizes reporting a governed family over fusion weight B. From regularization, name the split `P_R x + P_0 x` a *cited standard result*, not a new claim. From philosophy, the phrase *empirical equivalence class* reframes "we hallucinate" into "we honestly resolve an unavoidable underdetermination, labeling which pixels are evidence and which are chosen prior."

**Where it helps.** Intro hook + Theory framing + Discussion/limitations.

**Seed anchors.** Koopmans & Reiersol (1950); Rothenberg (1971, *Econometrica*); Manski, *Partial Identification* (2003); Lewbel (2019, *JEL* "identification zoo"); Engl-Hanke-Neubauer (1996); Hansen (1998); Laudan & Leplin (1991); SEP "Underdetermination."
**Search terms.** `observational equivalence identification` · `identified set partial identification Manski` · `null space Moore-Penrose minimum norm solution` · `Hadamard well-posedness three conditions` · `empirical equivalence underdetermination`.

---

## 2. Theme: **The Converse-First Spine**
*Maps to pillar (2) — the impossibility proven by a same-record impostor.*

**Why it maps.** Your headline — feasible-wrong images matching the record to ~2e-15, *tighter than the noisy truth* — is a **converse/impossibility** statement of exactly the type these fields formalize. Compressed sensing is built on an achievability boundary (RIP / **Null-Space Property** ⇒ recovery) and its converse (below threshold, or without a sparsity restriction, recovery is provably impossible because multiple feasible signals share `y`). Shannon's coding theorems give the two-part *converse-then-achievability* skeleton: your operator `A` is a deterministic degraded channel whose null space is "information the channel erased." **Fano's inequality** is the proof engine: a large equivocation set forces an error floor independent of the algorithm — the rigorous version of "no better solver fixes this."

**What to borrow.** The *achievability-vs-converse* scaffold as your Theory section's organizing frame (state the converse FIRST). The named, citable term **Null-Space Property** for the blind subspace. Fano's *equivocation / indistinguishable set / genie-aided* vocabulary so the 2e-15 tie reads as a measure-of-equivocation demonstration, not a numerical curiosity. The Donoho-Tanner *phase-transition* figure as a template for "measurement-accountability vs sampling fraction." And the Goodhart / adversarial-example framing ("the objective is met, the intent is not — provably") to arm the Discussion against "isn't tighter data-fit always better?"

**Where it helps.** Theory framing (the converse) + Intro hook + a phase-boundary figure + reviewer-anticipation.

**Seed anchors.** Candès-Romberg-Tao (2006); Cohen-Dahmen-DeVore (2009, NSP); Donoho & Tanner (2009); Foucart & Rauhut (2013); Shannon (1948); Fano (1961); Cover & Thomas; Szegedy et al. (2014); Geirhos et al. (2020).
**Search terms.** `null space property compressed sensing` · `converse lower bound recovery` · `Fano inequality equivocation error lower bound` · `erasure channel information loss` · `Goodhart law evaluation gaming`.

---

## 3. Theme: **The Ground-Truth-Free Certificate**
*Maps to pillar (3) — the audit contracting each measured mode by `λ/(λ+σ²)`.*

**Why it maps.** Your contraction factor is *not ad hoc* — it is the same object in four independent literatures, which is a gift: cite any two and the certificate reads as canonical. It is the **Wiener/MMSE** per-mode gain in a Gaussian channel (I-MMSE ties it to delivered mutual information). It is the **Tikhonov filter factor** `f_i = λ_i/(λ_i+σ²)` from SVD regularization (with the Picard condition and Morozov's discrepancy principle as companions). It is the eigenvalue of the **likelihood-informed subspace** generalized eigenproblem in Bayesian inverse problems — and there, the theorem *posterior = prior in the data-null subspace* is the exact justification that null-space-only injection is Bayes-consistent for every B. **Sampling theory** (Landau density, Slepian's prolate eigenvalue *cliff*) quantifies how few degrees of freedom a ~5% measurement can fix, so "the null space carries most of the detail" becomes a cited bound, not a hand-wave.

**What to borrow.** Name the factor a *filter factor* / *MMSE shrinkage* / *posterior-contraction* so reviewers recognize its pedigree. Borrow **metrology's** *per-mode uncertainty budget*, *traceability*, and the GUM distinction between a *result* and its *stated uncertainty* — this earns a Nature/PNAS-register claim ("we bring metrological accountability to reconstructed images"). Borrow **conformal prediction's** *validity without ground truth / the guarantee is about the procedure, not the instance* to pre-empt "how can you certify anything without the truth?" Borrow **certified robustness's** practice of reporting *certified vs empirical* metrics side by side as the direct template for "separate measurement-accountability from image quality," and the *sound-but-conservative* posture to answer "why isn't your bound tight?"

**Where it helps.** Certificate derivation + a mode-spectrum / water-filling figure + reviewer-anticipation.

**Seed anchors.** Stuart (2010, *Acta Numerica*); Spantini et al. (2015); Cui et al. (2014, LIS); Guo-Shamai-Verdú (2005, I-MMSE); Hansen (1998, filter factors); Landau (1967); Slepian & Pollak (1961); JCGM 100:2008 (GUM); Angelopoulos & Bates (2021); Cohen-Rosenfeld-Kolter (2019).
**Search terms.** `likelihood informed subspace inverse problem` · `posterior equals prior data null subspace` · `Tikhonov filter factors SVD` · `Slepian prolate eigenvalue cliff degrees of freedom` · `conformal prediction distribution-free coverage` · `certified vs empirical robustness`.

---

## 4. Theme: **Governed Injection as Decoder Side-Information**
*Maps to pillar (4) — prior fills the null space while `A x_hat = y` stays untouched for every B.*

**Why it maps.** The sharpest reviewer objection is "you're just adding information the sensor never had." **Wyner-Ziv** answers it rigorously: side information available only at the *decoder* legitimately supplies detail the encoded rate never carried, *without changing the encoding*. That is exactly your prior filling null-space coordinates while the measurement "bin" (`y`) is preserved. **Rate-distortion-perception** (Blau-Michaeli) upgrades your fusion dial from heuristic to a principled operating curve — with a novel twist an IT reviewer will like: the usual perception-distortion tradeoff is unconstrained, whereas yours traces the frontier along a *fixed data-consistency isocontour*, so perception is bought free of distortion in the certifiable coordinates. **Set-membership estimation** names the non-singleton *feasible set*, formalizing "the reconstruction is an arbitrary selection from the coset — governed injection is a principled selection rule." **Selective prediction** casts null-space governance as *abstention with a risk-coverage curve*. **Differential privacy** is the precedent for collapsing an unbounded anxiety into one auditable scalar with a semantic guarantee — the template for B.

**What to borrow.** "The prior is decoder side-information, not extra measurement." The RDP figure archetype, *redrawn* as perception-vs-distortion at fixed measurement-consistency, with B sweeping the frontier. "We replace point estimation with governed exploration of the feasible set." The *risk-coverage* curve as the fusion-dial figure. DP's move: "you don't need to enumerate every hallucination; you bound one null-space injection scalar," plus composition to argue B behaves predictably under repeated fusion.

**Where it helps.** Discussion (where does the detail come from?) + the perception-distortion / fusion-dial figure + Related Work.

**Seed anchors.** Wyner & Ziv (1976); Slepian & Wolf (1973); Blau & Michaeli (2018 CVPR; 2019 ICML); Theis & Wagner (2021); Milanese & Vicino (1991); Combettes (1993); Geifman & El-Yaniv (2017); Dwork & Roth (2014).
**Search terms.** `Wyner-Ziv side information decoder` · `rate distortion perception tradeoff` · `set membership estimation feasible set` · `selective prediction risk coverage tradeoff` · `differential privacy epsilon single scalar guarantee`.

---

## 5. Theme: **Hallucination Made Provable**
*Maps to pillars (2) and (4) — extrinsic hallucination given its first operational definition.*

**Why it maps.** The LLM-hallucination field's hardest lesson is structurally your pillar (2): a generation can be perfectly *plausible* (sharp reconstruction) yet *unfaithful* to the evidence. Its **intrinsic vs extrinsic** taxonomy maps one-to-one onto your row-space (verifiable) vs null-space (unverifiable-by-construction) split — *extrinsic* hallucination IS `P_0 x`, content that neither contradicts nor is supported by the record. This lets you claim the *first operational, provable definition* of extrinsic hallucination rather than a post-hoc detector. **Epistemic vs aleatoric uncertainty** sharpens why after-the-fact detection fails on the null space: there is nothing to calibrate against — *calibration measures confidence, not competence*. **Scientific-image-integrity norms** (Rossner-Yamada) draw the exact ethics line you satisfy by construction: whole-record enhancement is permissible, fabricating unsupported content is not, and `A x_hat = y` to 2e-15 is a machine-checkable version of "you may enhance presentation but not alter the evidence." **Confirmation theory** supplies "the likelihood is flat in the null space, so the output there is prior, not evidence," and the honest reporting device of a *prior-sensitivity analysis*.

**What to borrow.** The *faithfulness-vs-plausibility* and *intrinsic/extrinsic* vocabulary for the Intro. The *epistemic (structurally unidentifiable) not merely under-quantified* framing to pre-empt "why not just UQ/calibrate the generator?" The integrity norm cited to *preempt* the fabrication charge. A *prior-sensitivity* section as the honest way to report null-space content ("expected and localized prior dependence, now made explicit and bounded").

**Where it helps.** Intro hook + Discussion/limitations (hallucination reframing) + Significance.

**Seed anchors.** Ji et al. (2023, *ACM Comput. Surv.*); Maynez et al. (2020, ACL); Kendall & Gal (2017); Hüllermeier & Waegeman (2021); Rossner & Yamada (2004, *JCB*); Howson & Urbach, *Scientific Reasoning*.
**Search terms.** `intrinsic vs extrinsic hallucination` · `faithfulness versus plausibility generation` · `epistemic aleatoric uncertainty decomposition` · `scientific image integrity manipulation guidelines` · `likelihood flat prior dominates posterior`.

---

## 6. Theme: **An Accountability Contract for Images**
*Maps to pillars (3) and (4) — the exact-consistency invariant as a machine-checkable provenance guarantee.*

**Why it maps.** This theme buys you the *big-picture significance* register. **Cryptographic commitments / ZK proofs** are the exact epistemic shape of your certificate: a commitment is *binding* on the row space (you cannot open it two ways) and *hiding* on the null space (many openings are consistent — a feasible-wrong image is literally a second valid opening); a ZK proof "proves a property while revealing nothing more," which is precisely soundness-without-completeness. **C2PA / content provenance** turns the invariant into a *content credential for reconstructions*: the row space is the tamper-evident payload, the null space is the declared edit. **Proof-carrying code** gives the shipping contract: an artifact travels *with* a machine-checkable certificate of one property, and the consumer verifies only that property (`A x_hat = y`), trusting nothing else about the producer. **Oreskes' verification/validation/confirmation** supplies the highest-register closing line: models of open systems can be verified and confirmed but never validated — your certificate *verifies and confirms* the measured modes while explicitly declining to *validate* the null space.

**What to borrow.** The *binding/hiding* and *soundness/completeness* pairs as precise language for consistency ≠ correctness. "Proves a property without claiming more" as an Intro hook. The *content-credential / measurement-backed vs model-authored* provenance vocabulary for Significance. The proof-carrying *trusted-computing-base* boundary (measured modes = trusted; null space = untrusted-but-fenced). Oreskes as the sentence that elevates a local optics result to a general statement about the limits of what fitting data can certify.

**Where it helps.** Significance + Intro societal hook + Discussion.

**Seed anchors.** Goldwasser-Micali-Rackoff (1985); Goldreich (2001, commitments); C2PA specification / Content Authenticity Initiative; Necula (1997, proof-carrying code); Oreskes-Shrader-Frechette-Belitz (1994, *Science*).
**Search terms.** `zero-knowledge proof soundness completeness` · `commitment scheme binding hiding` · `content provenance C2PA content credentials` · `proof-carrying code certificate check one property` · `verification validation confirmation numerical models Oreskes`.

---

## Read these THREE first (biggest writing payoff for least reading)

1. **Classical regularization of ill-posed inverse problems** (Hansen 1998; Engl-Hanke-Neubauer 1996). One textbook simultaneously gives you Hadamard's uniqueness condition (pillar 1), the `P_R x + P_0 x` decomposition as a *cited theorem* (pillar 1), and the *filter factor* `λ/(λ+σ²)` (pillar 3). This is your native math home and the cheapest way to make three claims look standard.
2. **Compressed-sensing converse + the Null-Space Property** (Cohen-Dahmen-DeVore 2009; Foucart & Rauhut 2013), with a Fano assist. This hands you the *named object* ("null-space property") and the *converse-first* narrative genre for pillar 2 — the single move that reframes your 2e-15 result from "curiosity" to "impossibility theorem."
3. **Bayesian inverse problems / likelihood-informed subspace** (Stuart 2010; Spantini 2015; Cui 2014). Gives the certificate a *second, independent* (statistics) pedigree and, crucially, the theorem *posterior = prior in the data-null subspace* — the rigorous justification that governed injection is measurement-safe for every B (pillar 4).

Together these three cover all four pillars, come from fields reviewers cannot dismiss, and are the highest-density vocabulary per page.

### One ready-to-adapt sentence for each of the top 3

- **Regularization (Theory/Intro):** "Ghost imaging under heavy undersampling violates Hadamard's uniqueness condition: the forward operator `A` admits a non-trivial null space, so every reconstruction decomposes into a data-fixed minimum-norm (row-space) component and a measurement-invisible component in `null(A)`, the latter being precisely the classical filter-factor limit `λ/(λ+σ²) → 0`."
- **CS converse (Theory/Related):** "Because our sampling regime enforces no sparsity restriction, the null-space property fails by construction; this is not a solver weakness but a converse — analogous to compressed-sensing lower bounds, no estimator can distinguish the observationally equivalent family from the truth given only `y`."
- **Bayesian LIS (Discussion/limitations):** "In the data-null subspace the Gaussian posterior equals the prior; our generative injection therefore acts *only* where the likelihood is flat, so it is Bayes-consistent and leaves `A x_hat = y` invariant for every fusion weight — while remaining, by the same token, unfalsifiable by the data, which we state as the honest boundary of the method."

---

## Honesty red-line: borrowings that risk overclaiming, and how to cite them safely

The invariant rule: **the measurement certifies only the row-space component; nothing licenses a claim that invented null-space content is real.** Three borrowings can quietly cross that line if imported loosely.

1. **Wyner-Ziv / "side information supplies detail."** Safe as a *mechanism* analogy (detail enters at decode without altering the encoding). **Overclaim risk:** Wyner-Ziv side information is *correlated with the source* and yields quantifiable distortion gains; your generative prior is **not** a measurement of the true `x` and gives **no** fidelity guarantee on the injected content. Cite it strictly for *"the encoding is preserved,"* never for *"therefore the detail is accurate."* Add: "unlike Wyner-Ziv side information, our prior carries no measured correlation with the specific ground truth, so it certifies structure preservation, not detail correctness."

2. **ZK "proves a property" / commitments / content-provenance.** Safe for the *shape* (binding on row space, hiding on null space; certify one property, reveal nothing more). **Overclaim risk:** cryptographic soundness is adversarial and probabilistic; your certificate is a numerical/statistical accountability statement, not a cryptographic proof. Say "in the epistemic sense of ZK" or "structurally analogous to a commitment," and never imply the null-space content is authenticated — the commitment is deliberately *hiding* there, i.e., *un*-vouched.

3. **Conformal / certified-robustness / metrology "guarantee."** Safe as *validity-not-correctness* and *certified-vs-empirical* framing. **Overclaim risk:** each supplies a guarantee *about a defined quantity under stated assumptions* (marginal coverage, a certified radius, a stated uncertainty). Scope every use to the **measured modes only**, and state the assumption (Gaussian noise model, known `σ`). Pair every "certificate" sentence with its explicit silence: "the certificate makes no claim about `P_0 x`."

General safe-citation posture, adaptable verbatim: *"We borrow these frameworks for vocabulary and epistemic structure, not as transferred guarantees; each is invoked to name a property our measurement genuinely has (row-space accountability) and to make explicit the property it structurally lacks (null-space verification)."* Following Oreskes, keep the verbs disciplined — **verify** and **confirm** the measured subspace; never **validate** the injected detail.