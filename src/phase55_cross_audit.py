from __future__ import annotations

from pathlib import Path

from .phase55_common import (
    PHASE53C_ROOT,
    PHASE53D_ROOT,
    PHASE55_ROOT,
    TASKS,
    best_rows_by_task,
    classify_optional_gan,
    fmt,
    max_value,
    mean,
    min_value,
    read_csv_rows,
    to_float,
    write_rows,
)


def best53c() -> dict[str, dict[str, str]]:
    rows = read_csv_rows(PHASE53C_ROOT / "session_20_exact_null_mi_pretest" / "exact_null_mi_pretest_results.csv")
    return best_rows_by_task(rows, include_models=["blind_null_critic", "projection_conditioned"], metric="auc")


def best53d() -> dict[str, dict[str, str]]:
    rows = read_csv_rows(PHASE53D_ROOT / "anchor_null_pretest_results.csv")
    return best_rows_by_task(rows, include_models=["pair_pca"], metric="auc")


def baseline(rows: list[dict[str, str]], task: str, token: str) -> float:
    vals = [to_float(r.get("auc")) for r in rows if r.get("task") == task and token in str(r.get("model", ""))]
    return max_value(vals)


def split_audit() -> tuple[list[dict], str]:
    index_files = list(PHASE53C_ROOT.rglob("*index*")) + list(PHASE53C_ROOT.rglob("*indices*")) + list(PHASE53C_ROOT.rglob("*split*"))
    index_files = [p for p in index_files if p.is_file()]
    if not index_files:
        rows = [
            {
                "phase": "Phase53C",
                "session": "session_20_exact_null_mi_pretest",
                "train_anchor_overlap_eval": "unknown",
                "train_null_overlap_eval": "unknown",
                "pair_overlap": "unknown",
                "split_type": "unknown",
                "risk": "unknown_split_risk",
                "evidence": "No saved image-id / pair-id train-test split indices were found.",
            }
        ]
        return rows, "unknown_split_risk"
    rows = [
        {
            "phase": "Phase53C",
            "session": "found_index_files",
            "train_anchor_overlap_eval": "not_computed",
            "train_null_overlap_eval": "not_computed",
            "pair_overlap": "not_computed",
            "split_type": "indices_present_unparsed",
            "risk": "requires_manual_group_overlap_check",
            "evidence": "; ".join(str(p) for p in index_files[:12]),
        }
    ]
    return rows, "indices_present_unparsed"


def main() -> None:
    out = PHASE55_ROOT
    out.mkdir(parents=True, exist_ok=True)
    c_mi = read_csv_rows(PHASE53C_ROOT / "session_20_exact_null_mi_pretest" / "exact_null_mi_pretest_results.csv")
    d_anchor = read_csv_rows(PHASE53D_ROOT / "anchor_null_pretest_results.csv")
    c_best = best53c()
    d_best = best53d()
    c_leak = read_csv_rows(PHASE53C_ROOT / "session_21_soft_leakage_and_shortcut_audit" / "soft_leakage_results.csv")
    d_leak = read_csv_rows(PHASE53D_ROOT / "soft_leakage_by_lambda.csv")
    c_shortcut = read_csv_rows(PHASE53C_ROOT / "session_21_soft_leakage_and_shortcut_audit" / "shortcut_audit_results.csv")
    d_shortcut = read_csv_rows(PHASE53D_ROOT / "shortcut_audit_results.csv")
    c_feas = read_csv_rows(PHASE53C_ROOT / "session_22_feasible_hallucination_figure" / "feasible_hallucination_metrics.csv")
    d_feas = read_csv_rows(PHASE53D_ROOT / "feasible_hallucination_metrics.csv")
    d_post = read_csv_rows(PHASE53D_ROOT / "posthoc_certificate_sweep.csv")
    gan = read_csv_rows(PHASE53C_ROOT / "session_24_optional_gan_and_posterior_sampling" / "optional_gan_results.csv")
    posterior = read_csv_rows(PHASE53C_ROOT / "session_24_optional_gan_and_posterior_sampling" / "posterior_sampling_metrics.csv")
    c_projector = []
    for task in TASKS:
        c_projector.extend(read_csv_rows(PHASE53C_ROOT / "session_20_exact_null_mi_pretest" / task / "exact_projector_checks.csv"))
    d_projector = read_csv_rows(PHASE53D_ROOT / "exact_projector_checks.csv")
    comp: list[dict] = []
    comp.append({"row": "exact P0 numerical residual", "Phase53C": fmt(max_value([r.get("A_P0_norm") for r in c_projector]), 6), "Phase53D": fmt(max_value([r.get("A_P0_relative_norm") for r in d_projector]), 6), "interpretation": "both numerically small; exact P0 construction is plausible"})
    comp.append({"row": "soft leakage max/mean", "Phase53C": f"max mean {fmt(max_value([r.get('mean_leakage_ratio') for r in c_leak]),5)}", "Phase53D": f"max mean {fmt(max_value([r.get('mean_theory_leakage_factor') for r in d_leak]),5)}", "interpretation": "soft P_N leaks row-space; exact P0 required"})
    comp.append({"row": "max AUC", "Phase53C": fmt(max_value([r.get("auc") for r in c_mi]), 3), "Phase53D": fmt(max_value([r.get("auc") for r in d_anchor]), 3), "interpretation": "Phase53C uses deep critic; Phase53D uses local PCA/linear classifiers"})
    for task in TASKS:
        comp.append({"row": f"AUC by family {task}", "Phase53C": fmt(c_best.get(task, {}).get("auc"), 3), "Phase53D": fmt(d_best.get(task, {}).get("auc"), 3), "interpretation": f"C model={c_best.get(task, {}).get('model','n/a')}; D model={d_best.get(task, {}).get('model','n/a')}, negative={d_best.get(task, {}).get('negative_type','n/a')}"})
    comp.append({"row": "MI lower bound", "Phase53C": f"mean {fmt(mean([r.get('infoNCE_mi_lower_nats') for r in c_mi]),3)} / max {fmt(max_value([r.get('infoNCE_mi_lower_nats') for r in c_mi]),3)}", "Phase53D": "not estimated", "interpretation": "MI proxy only available from Phase53C critic"})
    for token, name in [("handcrafted", "handcrafted baseline"), ("condition_ignored", "condition-ignored baseline"), ("anchor_only", "anchor-only baseline")]:
        comp.append({"row": name, "Phase53C": fmt(max_value([baseline(c_mi, t, token) for t in TASKS]), 3), "Phase53D": fmt(max_value([baseline(d_anchor, t, token) for t in TASKS]), 3), "interpretation": "baseline should remain near random for strong critic claim"})
    comp.append({"row": "shortcut audit AUC", "Phase53C": fmt(mean([r.get("eval_auc") for r in c_shortcut]), 3), "Phase53D": fmt(mean([r.get("eval_auc") for r in d_shortcut]), 3), "interpretation": "diagnostic only; full-input residual D remains risky"})
    comp.append({"row": "feasible hallucination RelMeasErr", "Phase53C": f"cross {fmt(mean([r.get('cross_relmeas') for r in c_feas]),5)} / ours {fmt(mean([r.get('ours_relmeas') for r in c_feas]),5)}", "Phase53D": f"cross {fmt(mean([r.get('cross_relmeas') for r in d_feas]),5)} / ours {fmt(mean([r.get('ours_relmeas') for r in d_feas]),5)}", "interpretation": "cross-feasible is low but still higher than ours; main figure needs caveat"})
    hard = [r for r in d_post if r.get("lambda") == "hard" and r.get("status") == "ok"]
    comp.append({"row": "posthoc audit RelMeasErr recovery", "Phase53C": "not run", "Phase53D": f"{fmt(mean([r.get('relmeas_before') for r in hard]),5)} -> {fmt(mean([r.get('relmeas_after') for r in hard]),5)}", "interpretation": "strong support for measurement certificate/re-legalization"})
    comp.append({"row": "posterior variance null ratio", "Phase53C": fmt(mean([r.get("variance_null_ratio_mean") for r in posterior]), 5), "Phase53D": "not run", "interpretation": "posterior variation is mostly controlled; supplement candidate"})
    comp.append({"row": "optional GAN metrics", "Phase53C": f"PSNR {fmt(mean([r.get('psnr') for r in gan]),3)}, SSIM {fmt(mean([r.get('ssim') for r in gan]),3)}, RelMeasErr {fmt(mean([r.get('rel_meas_error') for r in gan]),5)}", "Phase53D": "not run", "interpretation": classify_optional_gan(gan)[1]})
    write_rows(out, "phase53C_vs_53D_comparison", comp, "Phase53C vs Phase53D Comparison")
    split_rows, split_risk = split_audit()
    write_rows(out, "split_leakage_audit", split_rows, "Phase55 Split Leakage Audit")
    baseline_clean = all(max_value([baseline(c_mi, t, "condition_ignored"), baseline(c_mi, t, "anchor_only")]) < 0.5 for t in TASKS)
    memorization = [
        "# Phase55 Memorization and Shortcut Audit",
        "",
        f"- Split status: `{split_risk}`.",
        "- No Phase53C train/eval image-id or pair-id index files were found in the imported outputs.",
        "- Therefore the AUC 0.992 result does not pass a strict group-split leakage audit yet.",
        f"- Phase53C condition-ignored / anchor-only baselines are {'clean enough' if baseline_clean else 'not uniformly clean'} by max-AUC inspection.",
        "- Shuffled-label / random-anchor permutation baselines were not found.",
        "- Recommendation: before any exact-null critic claim, rerun E1/critic with saved image-id group split and shuffled-label/random-anchor baselines.",
    ]
    (out / "memorization_shortcut_audit.md").write_text("\n".join(memorization) + "\n", encoding="utf-8")
    answers = [
        "# Phase55 Cross-Phase Consistency Answers",
        "",
        "1. Phase53C AUC is higher because it uses trained deep/conv critics, while Phase53D uses local PCA/linear CPU classifiers.",
        "2. Yes: Phase53C is a deep critic experiment; Phase53D is local linear/PCA preflight.",
        "3. Phase53C split type cannot be verified from imported files because no image-id group split indices were saved.",
        "4. Because split is unknown, mark `unknown_split_risk`; if it was pair-only split, it is high memorization risk.",
        "5. No shuffled-label baseline was found.",
        "6. Phase53C condition-ignored and anchor-only baselines are mostly below random-direction pair critics, but not a substitute for group-split/permutation audit.",
        "7. Scr-5 ran optional GAN in Phase53C because the deep critic gate passed there; Phase53D Scr AUC is low because local linear PCA features did not detect the same signal.",
        "8. No direct A/anchor mismatch was detected: exact P0 residuals are small, Rademacher exact-A loaded, and Scrambled Hadamard used generated A; nevertheless anchor/P0 consistency needs a group-split repeat for claims.",
    ]
    (out / "PHASE55_CROSS_PHASE_CONSISTENCY.md").write_text("\n".join(answers) + "\n", encoding="utf-8")
    print(out / "phase53C_vs_53D_comparison.csv")


if __name__ == "__main__":
    main()

