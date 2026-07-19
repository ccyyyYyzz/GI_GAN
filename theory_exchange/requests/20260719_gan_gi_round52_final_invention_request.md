# Round 52 — Final invention pass after causal, rate, and independent-operator evidence

## Your role

Act as the **theory inventor and method architect**, not as a reviewer of Codex's proposal.  Continue the same Round 45/48 line of reasoning, but now use the completed falsification evidence below to decide what the final journal method actually is.  The desired result is a compact GAN-assisted ghost-imaging idea with physical meaning, structural elegance, and a theorem package that explains the observed effect.  Do not build a module stack or hide a weak idea behind notation.

Read branch `codex/gan-gi-journal-poc-20260718` of `ccyyyYyzz/GI_GAN`, starting at commit `74d782f`.  The held-out STL-10 test split is still unopened.

Write the answer and commit it to:

`theory_exchange/responses/20260719_gan_gi_round52_final_invention_gptpro.md`

## Frozen incumbent

The surviving method is the original raw-correction **fiber-orthogonal high-pass innovation (FOHI)**, not EQ-FOHI.  For the measurement-consistent base `x_B`, structural null correction `c_S`, and VQGAN-source null correction `c_G`, it uses

$$
d=c_G-c_S,\qquad u=P_NH_{0.12,0.03}P_Nd,
$$

$$
v=u-\frac{\langle u,c_S\rangle}{\|c_S\|_2^2}c_S,
\qquad
\widehat x=\Pi_{\mathcal F_y}(x_B+c_S+0.5v).
$$

The coefficient, cutoff, transition, architectures, checkpoint rule, and exact box-fiber projection are frozen.  There is no truth-dependent inference and no per-image gate.

The selected proposal arm is now the **discriminator-off adapter fed by a pretrained VQGAN reconstruction**.  This means the small residual adapter does not need its own adversarial loss, but the proposal still comes from the project's adversarially trained VQGAN prior.  The matched structural reference and second-network control use VQAE sources.

## Evidence that must constrain the invention

### 1. Original 5% multiseed validation

Three independent 512-image validation pairings all pass the joint confidence-interval gate against their matched structural arms:

| Pairing | Delta PSNR | Delta SSIM | Delta LPIPS |
|---|---:|---:|---:|
| seed 0 | +0.026601 | +0.001710 | -0.004458 |
| seed 1 | +0.025783 | +0.001385 | -0.002632 |
| seed 2 | +0.018411 | +0.001489 | -0.005200 |

Evidence: `results/gan_gi_journal_round47/`.

### 2. Endpoint quotient was a clean negative

EQ-FOHI was slightly better in mean PSNR and SSIM, but its direct LPIPS confidence interval against FOHI crossed zero.  It failed the hard replacement rule and was killed without rescue.  The old alpha-zero anchor shift was only about `1e-6 dB`, so the positive FOHI result is not a material clipping/projection baseline artifact.

Evidence: `results/gan_gi_journal_round48/eq_fohi/`.

### 3. Four-cell causal matrix

Cells used identical deterministic geometry:

- A: VQGAN source, adapter adversarial weight `0.0015`, high-pass;
- B: VQGAN source, adapter adversarial weight `0`, high-pass;
- C: independent second VQAE source, high-pass;
- D: VQGAN source, adapter adversarial weight `0.0015`, low-pass.

Outcomes across three pairings:

- A: 3/3 triple-positive;
- B: 3/3 triple-positive;
- C: 0/3 because LPIPS worsens in every seed;
- D: 3/3 small triple-positive.

A has no significant advantage over B, so parsimony selects B.  A directly and strongly beats C:

$$
\Delta_{A-C}=+0.01374\ {\rm dB},\quad +0.000831\ {\rm SSIM},\quad -0.007492\ {\rm LPIPS},
$$

with all crossed confidence intervals strictly favorable.  A also directly beats D in all three metrics.  Thus the evidence supports **VQGAN proposal alignment**, not an essential adversarial loss in the added adapter; high-pass is dominant, not exclusive.

Evidence: `results/gan_gi_journal_round49/`.

### 4. Selected B arm across measurement rates and bands

For the selected discriminator-off VQGAN-source FOHI:

- at 10% sampling: 3/3 seeds pass, with average deltas about `+0.227 dB`, `+0.0108 SSIM`, and `-0.0164 LPIPS`;
- at 5% sampling: 3/3 seeds pass;
- at 2% sampling: 2/3 pass; the failed seed improves PSNR and SSIM but worsens LPIPS with an entirely unfavorable LPIPS interval.

At 5%, low-pass also gives a small 3/3 benefit, but the direct high-pass-minus-low-pass crossed intervals are strictly favorable in all three metrics:

$$
+0.01927\ {\rm dB},\quad +0.001267\ {\rm SSIM},\quad -0.004614\ {\rm LPIPS}.
$$

Therefore 2% is a declared operating boundary and the spectral claim is dominance, not exclusivity.

Evidence: `results/gan_gi_journal_round50/`.

### 5. Independent measurement operators

Three newly generated, independently seeded 5% GI operators have distinct matrix hashes and all pass the joint CI gate.  The seed-by-image hierarchical result is:

$$
\Delta {\rm PSNR}=+0.08298\ [0.07906,0.08682]\ {\rm dB},
$$

$$
\Delta {\rm SSIM}=+0.004974\ [0.004757,0.005190],
$$

$$
\Delta {\rm LPIPS}=-0.009940\ [-0.010576,-0.009333].
$$

All exact projection certificates pass.  No test image was accessed.

Evidence: `results/gan_gi_journal_round51/`.

## Questions you must solve as an inventor

1. **Choose and articulate the final method.**  Is the surviving FOHI already the cleanest journal-level construction once B, C, D, rate, and operator evidence are known?  If yes, freeze it and give it the strongest accurate name and one-sentence contribution.  If not, invent at most **one** further transform, but only if it is structurally inevitable, parameter-free or uses no new selected parameter, and can be falsified in one validation experiment without retuning.  Do not revive EQ-FOHI, a learned gate, tangent-cone stacking, norm restoration, or a band search.

2. **Explain the actual GAN mechanism.**  The added adapter discriminator is unnecessary, yet a VQGAN-source proposal succeeds and the matched second-VQAE source fails.  Formulate precisely what the pretrained adversarial prior contributes inside the unmeasured GI fiber.  The answer must distinguish “GAN-generated proposal alignment” from “the adapter uses an adversarial loss.”

3. **Build the minimal theorem package.**  Derive only the propositions needed to make the method intellectually sharp: physical fiber consistency, uniqueness/minimality of the rank-one innovation, and a risk or joint-descent condition that connects proposal alignment to simultaneous PSNR/SSIM/LPIPS improvement.  Clearly label standard ingredients versus the genuinely new combined principle.  Avoid theorem inflation.

4. **Start from optics, not modules.**  Explain why a GI record determines a measured row-space component but leaves a large unmeasured fiber; why VQAE and VQGAN select different coordinates on that fiber; and why the useful VQGAN displacement is predominantly high-frequency yet not exclusively so.  The method must read as a physical inverse-problem construction, not generic feature fusion.

5. **Decide the final validation boundary.**  State exactly what can be claimed at 5% and 10%, what the 2% failure means, and whether three independent operators are sufficient for an operator-robust simulation claim.  The project cannot perform a real optical experiment.  A statistically correct noisy extension may use a relaxed likelihood/confidence fiber, but decide whether it is required before the held-out test or belongs as a separate secondary experiment/future extension.

6. **Freeze a one-shot held-out protocol.**  Specify the exact fixed arm, rates, operator seed policy, metrics, familywise decision rule, and allowed claims before opening any held-out data.  No post-test modification is permitted.

7. **Give an author-facing decision table.**  End with a short table: final method, killed alternatives, primary causal claim, claim boundaries, whether one more validation experiment is necessary, and the exact next command-level action.

## Hard constraints

- GAN must remain substantively used: the proposal comes from the pretrained VQGAN prior.
- The aim is improved GI reconstruction quality, not LPIPS alone.
- No real optical experiment is possible.
- No compute-stacking paper, no architecture zoo, no truth-dependent selection, no hidden tuning after seeing the held-out test.
- Do not merely score novelty.  Invent and commit the final theory/method decision.
- The held-out test remains closed while you work.
