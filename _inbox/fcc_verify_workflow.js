export const meta = {
  name: 'fcc-canary-verify',
  description: 'Adversarially verify the FCC diagnostic canary for discipline, control validity, classification honesty, and numeric accuracy',
  phases: [
    { title: 'Review' },
    { title: 'Synthesize' },
  ],
}

const OUT = 'E:/ns_mc_gan_gi_code_fcc_phase1/outputs/compatibility/fcc_diagnostic_canary64'
const CODE = 'E:/ns_mc_gan_gi_code_fcc_phase1'

const FINDING_SCHEMA = {
  type: 'object',
  properties: {
    dimension: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'ok'] },
          claim: { type: 'string' },
          evidence: { type: 'string' },
        },
        required: ['severity', 'claim', 'evidence'],
      },
    },
    verdict: { type: 'string' },
  },
  required: ['dimension', 'findings', 'verdict'],
}

const DIMENSIONS = [
  {
    key: 'discipline',
    prompt: `You are auditing research DISCIPLINE compliance of an FCC row-null diagnostic.
Read these files:
- ${CODE}/src/fcc_canary.py
- ${CODE}/fcc_diagnostic_canary.py
- ${OUT}/reports/build_manifest.json
- ${CODE}/_inbox/FCC_THEORY_AND_PROMPT_BUNDLE/CLAUDE_CODE_FCC_DIAGNOSTIC_PROMPT.md (the "纪律"/discipline section)

Verify each, citing exact evidence (file:line or JSON value):
1. NO clipping is applied before projection / feasibility checks (search for clamp/clip on images pre-projection; ToTensor [0,1] is fine, an explicit clip is NOT).
2. NO dense P0 (n x n) is ever constructed; projections go through matrix-free exact_row_project/exact_null_project (A^T (A A^T)^-1 A v).
3. Splits are hash-clean (dedup by raw sha256) AND exclude consumed hashes; cross_split_raw_index_overlap is all zero in build_manifest.json; exclusion_pool_size is large (tens of thousands).
4. Normalization statistics come ONLY from train (compute_train_normalization on splits['train']).
5. Donor indices are traceable and A-hash / sample-hash audit / split manifest are saved.
6. float64 geometry checks actually pass (geometry_checks.pass == true, A_P0_rel_max < 1e-9).
Return a structured finding per the schema. Be adversarial: if something is only partially compliant, flag it.`,
  },
  {
    key: 'control_validity',
    prompt: `You are auditing CONTROL VALIDITY of an FCC row-null diagnostic. The whole point is that prior phases failed because a NON-DEPLOYABLE oracle control (using the TRUE matched null energy: positives scored exactly 0 -> AUC 1.0 for any feature) was mistaken for a real control.
Read:
- ${CODE}/src/fcc_canary.py (DeployableClassifier, fit_deployable_baselines, baseline_pair_auc, baseline_score_rows)
- ${CODE}/fcc_diagnostic_canary.py (cmd_eval: how baselines are fit/scored, how the oracle control is flagged)
- ${OUT}/reports/eval_summary.json

Verify, citing evidence:
1. Deployable baselines are TRUTH-BLIND: a classifier is fit on TRAIN (positives vs random-derangement negatives) over phi(r,n) and scored on DEV; it never receives the true matched null of the test pair.
2. The true-null-energy control is explicitly marked non_deployable=true AND excluded_from_gate=true, and is NOT used in classification.
3. The nuisance-balanced derangement is one-to-one with no fixed points (balance report fixed_points==0, donor_unique_fraction==1.0).
4. Deployable baselines include row-only, null-only, sum-image, and full pair feature modes (so a shortcut in any is caught).
5. FCC and baselines are compared on the SAME negatives (random AND nuisance-balanced) and the SAME fixed-32 candidate manifests.
Be adversarial: any way the "deployable" baseline secretly sees the truth, or the oracle leaks into the gate, is a BLOCKER.`,
  },
  {
    key: 'classification_honesty',
    prompt: `You are auditing CLASSIFICATION HONESTY of an FCC row-null diagnostic against its theory.
Read:
- ${CODE}/_inbox/FCC_THEORY_AND_PROMPT_BUNDLE/FCC_THEORY_RANGE_NULL_COMPATIBILITY.md (section 11: the 5 classifications)
- ${CODE}/src/fcc_canary.py (classify_fcc)
- ${OUT}/reports/classification.json
- ${OUT}/reports/FINAL_REPORT.md
- ${OUT}/reports/eval_summary.json

Verify, citing evidence:
1. The classify_fcc gate logic faithfully maps to the 5 theory classifications (STRUCTURAL_COMPATIBILITY_CONFIRMED / REAL_PAIR_SIGNAL_BUT_NO_GENERATED_TRANSFER / ONLY_SCALAR_OR_ARTIFACT_SIGNAL / NO_COMPATIBILITY_SIGNAL / INVALID_EXPERIMENT).
2. "Structural" REQUIRES FCC to EXCEED deployable baselines on nuisance-balanced negatives AND those baselines to be neutralised (<=0.60) AND negatives actually balanced. There is NO path to a structural claim while a deployable baseline still explains the signal.
3. The emitted classification actually follows from the numbers in eval_summary.json (recompute the checks by hand).
4. Layer C (generated transfer / Task D) is gated OFF and not claimed.
5. No over-claim: FCC score is not called measurement-certified truth anywhere.
Be adversarial: if the thresholds could let a nuisance-explained result be called structural, that is a BLOCKER.`,
  },
  {
    key: 'numeric_accuracy',
    prompt: `You are auditing NUMERIC ACCURACY of an FCC row-null diagnostic's reports.
Read:
- ${OUT}/reports/eval_summary.json (source of truth)
- ${OUT}/reports/build_manifest.json
- ${OUT}/reports/FACTS.json
- ${OUT}/reports/CLAIM_EVIDENCE_LEDGER.md
- ${OUT}/reports/FINAL_REPORT.md

Cross-check EVERY number that appears in FACTS.json, the ledger, and the final report against eval_summary.json / build_manifest.json. List any mismatch (value, file, expected vs printed). Specifically verify: recall_at_1, random_recall_at_1, fcc balanced AUC, best deployable balanced AUC + its key, balance feature SMD max, geometry A_P0_rel_max, feasibility u_rel_max, operator m and A sha256. A fabricated or mismatched number is a BLOCKER.`,
  },
]

phase('Review')
const reviews = await parallel(DIMENSIONS.map((d) => () =>
  agent(d.prompt, { label: `review:${d.key}`, phase: 'Review', schema: FINDING_SCHEMA })
))

phase('Synthesize')
const valid = reviews.filter(Boolean)
const synthesis = await agent(
  `You are the lead reviewer. Synthesize these ${valid.length} dimension reviews of the FCC diagnostic canary into a final verdict.
Reviews (JSON):
${JSON.stringify(valid, null, 2)}

Produce: (1) overall PASS/PASS_WITH_NITS/FAIL, (2) every blocker and major finding with its evidence, (3) a short statement of whether the emitted classification is trustworthy and disciplined, (4) a prioritized fix list (empty if none). Be concise and concrete.`,
  { label: 'synthesis', phase: 'Synthesize' }
)

return { synthesis, reviews: valid }
