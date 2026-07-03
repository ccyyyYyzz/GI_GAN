export const meta = {
  name: 'structure-detail-fcc-verify',
  description: 'Adversarially verify the Structure-Detail FCC diagnostic (feasibility, truth-blind controls, classification honesty, numeric accuracy)',
  phases: [{ title: 'Review' }, { title: 'Synthesize' }],
}

const CODE = 'E:/ns_mc_gan_gi_code_fcc_phase1'
const OUT0 = 'E:/ns_mc_gan_gi_code_fcc_phase1/outputs/compatibility/structure_detail_fcc/seed0'

const SCHEMA = {
  type: 'object',
  properties: {
    dimension: { type: 'string' },
    findings: { type: 'array', items: { type: 'object', properties: {
      severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'ok'] },
      claim: { type: 'string' }, evidence: { type: 'string' },
    }, required: ['severity', 'claim', 'evidence'] } },
    verdict: { type: 'string' },
  },
  required: ['dimension', 'findings', 'verdict'],
}

const DIMS = [
  { key: 'feasibility_and_pairing', prompt: `Audit the Structure-Detail FCC PAIRING and FEASIBILITY.
Read ${CODE}/structure_detail_fcc.py (make_components, sd_geometry, sd_feasibility, build_seed) and ${OUT0}/reports/build_manifest.json.
Verify with evidence:
1. structure s = x_A and detail d = P0(x_G - x_A) are constructed correctly; d uses the EXACT float64 null projector (projector.null_project_flat), not the float32 measurement.A_forward.
2. The detail d is in the null space to high precision (build_manifest geometry float64_A_P0_rel_max ~1e-9) so swapping detail preserves the measurement: A(s_i + d_j) = A(s_i + d_i). Confirm the feasibility metric (donor_null_rel_max, u_rel_max) is ~1e-9 and the pass threshold (1e-6) is justified for float32-sourced cached recons.
3. The note that A·x_A != y (~17% struct_consistency) is correctly deemed IRRELEVANT to feasibility (only detail-null matters). Confirm this reasoning is sound (the shared measurement of true & counterfactual is A·x_A, and detail swap preserves it).
4. Splits are firewalled by STRUCTURE source-index: cross_split_structure_source_index_overlap all zero; train=val-split structures, val/dev=dev-split structures (disjoint). Single fusion seed per run (no detail-realization confound). Operator rows_sha256 matches the frozen 8a16664e... manifest.
Be adversarial: any way the counterfactual is NOT measurement-equivalent, or splits leak structures, is a blocker.` },
  { key: 'controls_and_classification', prompt: `Audit CONTROL VALIDITY and CLASSIFICATION HONESTY of the Structure-Detail FCC.
Read ${CODE}/src/fcc_canary.py (classify_fcc, fit_deployable_baselines, baseline_pair_auc), ${OUT0}/reports/eval_summary.json, ${OUT0}/reports/classification.json, ${OUT0}/reports/FINAL_REPORT.md.
Context: critic recall@1~0.10 (weak), FCC balanced AUC~0.63, but deployable pair_logistic balanced AUC~0.97; null-only & row-only = 0.5; classification = ONLY_SCALAR_OR_ARTIFACT_SIGNAL (3/3 seeds).
Verify with evidence:
1. Deployable baselines are truth-blind (fit on TRAIN pos vs random-derangement neg over phi(s,d), scored on DEV); never see the test pair's true detail.
2. The classification logic is honest: a WEAK critic plus a STRONG deployable baseline must NOT be mislabelled NO_COMPATIBILITY_SIGNAL. Confirm classify_fcc uses 'critic OR deployable separates' for real_pair_signal, but requires the CRITIC to exceed neutralised deployable baselines for STRUCTURAL. Confirm ONLY_SCALAR_OR_ARTIFACT_SIGNAL is the correct label here (deployable not neutered: 0.97 on balanced; critic does not exceed it).
3. Hand-recompute the gate from eval_summary.json and confirm the emitted classification.
4. No over-claim: the report does not claim structural compatibility or measurement-certified truth; it notes the critic underperforms the deployable baseline.
Be adversarial: if the gate could call this STRUCTURAL, or if ONLY_SCALAR hides a real "no signal", flag it.` },
  { key: 'numeric_accuracy', prompt: `Audit NUMERIC ACCURACY across the 3 seeds of the Structure-Detail FCC.
Read ${CODE}/outputs/compatibility/structure_detail_fcc/MULTISEED_CLASSIFICATION_SUMMARY.json and for each seed in {0,1,2}: outputs/compatibility/structure_detail_fcc/seed{N}/reports/eval_summary.json + classification.json + FINAL_REPORT.md.
Verify: (1) all 3 seeds classify ONLY_SCALAR_OR_ARTIFACT_SIGNAL; (2) every number quoted in each FINAL_REPORT.md matches its eval_summary.json (recall@1, FCC balanced AUC, best deployable balanced AUC + key, balance smd_max, label-perm); (3) null-only and row-only baselines are exactly 0.5 across seeds; (4) the deployable pair AUC clearly exceeds the FCC critic AUC on balanced negs in every seed. List any mismatch as blocker.` },
]

phase('Review')
const reviews = (await parallel(DIMS.map((d) => () => agent(d.prompt, { label: `verify:${d.key}`, phase: 'Review', schema: SCHEMA })))).filter(Boolean)

phase('Synthesize')
const synthesis = await agent(
  `Synthesize these ${reviews.length} adversarial reviews of the Structure-Detail FCC diagnostic into a final verdict.
Reviews JSON:\n${JSON.stringify(reviews, null, 2)}
Give: (1) overall PASS / PASS_WITH_NITS / FAIL; (2) every blocker/major with evidence; (3) one sentence on whether ONLY_SCALAR_OR_ARTIFACT_SIGNAL is the trustworthy, honest call for structure-detail; (4) prioritized fixes (empty if none). Concise.`,
  { label: 'synthesis', phase: 'Synthesize' })

return { synthesis, reviews }
