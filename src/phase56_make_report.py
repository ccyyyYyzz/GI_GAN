from __future__ import annotations

import argparse
from pathlib import Path

from .phase56_common import add_args, finalize, fmt, max_value, mean, read_csv_rows, to_float, write_rows
from .utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase56 final report and claim decision.")
    add_args(parser)
    return parser.parse_args()


def best_strict(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out = {}
    priority = ["deep_exact_null_critic", "ridge_classifier", "pca_logistic", "pca_linear_svm"]
    for task in sorted({r.get("task", "") for r in rows}):
        task_rows = [r for r in rows if r.get("task") == task and r.get("split_mode") == "strict_both_group_split" and r.get("status") == "ok"]
        best = None
        for model in priority:
            cand = [r for r in task_rows if r.get("model") == model and r.get("negative_type") == "random"]
            if cand:
                best = max(cand, key=lambda r: to_float(r.get("auc_abs")))
                break
        if best is None and task_rows:
            best = max(task_rows, key=lambda r: to_float(r.get("auc_abs")))
        if best:
            out[task] = best
    return out


def decision_for_auc(auc: float, ci_low: float, clean: bool) -> str:
    if clean and auc >= 0.85 and ci_low > 0.75:
        return "strong"
    if clean and auc >= 0.75 and ci_low > 0.65:
        return "supplement-worthy"
    if auc < 0.65:
        return "weak"
    return "moderate_or_baseline_limited"


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_dir)
    results = read_csv_rows(root / "group_split_critic_results.csv")
    summary = read_csv_rows(root / "group_split_critic_summary.csv")
    overlaps = read_csv_rows(root / "group_split_overlap_audit.csv")
    mem = read_csv_rows(root / "memorization_leakage_diagnostics.csv")
    scr = read_csv_rows(root / "scr_sanity_audit.csv")
    feas = read_csv_rows(root / "feasible_hallucination_hardened_metrics.csv")
    best = best_strict(results)
    baseline_clean_by_task = {}
    for task in best:
        baselines = [
            r
            for r in results
            if r.get("task") == task
            and r.get("split_mode") == "strict_both_group_split"
            and r.get("model") in {"anchor_only_baseline", "p0_only_condition_ignored", "shuffled_label_baseline", "random_anchor_baseline"}
            and r.get("status") == "ok"
        ]
        baseline_clean_by_task[task] = max_value([r.get("auc_abs") for r in baselines]) < 0.65 if baselines else False
    strict_rows = []
    for task, row in best.items():
        clean = baseline_clean_by_task.get(task, False)
        auc_abs = to_float(row.get("auc_abs"))
        ci_low = to_float(row.get("auc_ci_low"))
        strict_rows.append(
            {
                "task": task,
                "family": row.get("family", ""),
                "model": row.get("model", ""),
                "negative_type": row.get("negative_type", ""),
                "auc": row.get("auc", ""),
                "auc_abs": auc_abs,
                "auc_ci_low": ci_low,
                "auc_ci_high": row.get("auc_ci_high", ""),
                "baseline_clean": clean,
                "decision": decision_for_auc(auc_abs, ci_low, clean),
            }
        )
    write_rows(root, "phase56_strict_auc_decision_table", strict_rows, "Phase56 Strict AUC Decision Table")
    survived = any(r["decision"] in {"strong", "supplement-worthy"} for r in strict_rows)
    strong = any(r["decision"] == "strong" for r in strict_rows)
    if strong:
        claim = "A. strict group-split AUC high: exact-null critic can be cited as evaluator/diagnostic, possibly main-text diagnostic; never certificate."
        next_action = "include exact-null critic in main text as diagnostic only"
    elif survived:
        claim = "C. group-split moderate/family-dependent: write as family-dependent supplement diagnostic; do not generalize."
        next_action = "include exact-null critic in supplement only"
    else:
        pair_high = any(to_float(r.get("pair_split_auc_mean")) >= 0.85 for r in summary)
        if pair_high:
            claim = "B. only pair-split/old AUC high while strict split is weak: Phase53C AUC 0.992 is not valid group-split evidence."
        else:
            claim = "B. old AUC 0.992 is not reproduced under strict group split or corrected pair-level reproduction; it is not valid evidence."
        next_action = "do not continue critic/GAN"
    feasible_selected = sum(1 for r in feas if str(r.get("selected_for_main")) in {"True", "true", "1"})
    scr_status = "; ".join(f"{r.get('task')}:{r.get('status')}" for r in scr)
    report = [
        "# Phase56 Group-Split Exact-Null Critic Report",
        "",
        "Scope: exact-null critic/classifier repeat with image-ID group split. No reconstruction generator, GAN generator, or main reconstruction network was trained.",
        "",
        "## Group Split Status",
        "",
        "- Strict split ID CSVs and split manifests are saved under `splits/`.",
        "- `group_split_overlap_audit.csv` reports strict train/test image overlap as pass/fail.",
        "",
        "## Strict Group-Split AUC by Family",
    ]
    for row in strict_rows:
        report.append(f"- {row['task']}: AUC_abs={fmt(row['auc_abs'],3)}, original AUC={fmt(row['auc'],3)}, model={row['model']}, decision={row['decision']}.")
    report.extend(
        [
            "",
            "## Main Decision",
            "",
            f"- {claim}",
            f"- AUC 0.992 survived group split? {'yes, conditionally' if survived else 'no'}",
            f"- Scr sanity: {scr_status}",
            f"- Feasible hallucination hardened rows selected: {feasible_selected}",
            "- GAN continuation: no. Default rule forbids GAN unless strict group-split AUC is strong and perceptual metrics are later measured.",
        ]
    )
    (root / "PHASE56_GROUP_SPLIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    claim_lines = [
        "# Phase56 Claim Decision",
        "",
        f"Decision: {claim}",
        "",
        "Thresholds:",
        "- AUC >= 0.85 with CI lower > 0.75 and clean baselines: strong.",
        "- AUC >= 0.75 with CI lower > 0.65: supplement-worthy.",
        "- AUC < 0.65: weak, do not claim.",
        "- Shuffled-label / anchor-only / P0-only baselines must be near random.",
        "",
        "Never call the critic a certificate; it is a critic/evaluator only.",
    ]
    (root / "PHASE56_CLAIM_DECISION.md").write_text("\n".join(claim_lines) + "\n", encoding="utf-8")
    next_lines = [
        "# Phase56 Next Action",
        "",
        f"Recommended next action: **{next_action}**.",
        "",
        "- Do not continue GAN from Phase56.",
        "- Do not change the main result table.",
        "- Use posthoc certificate and accountability separation as safer main claims.",
    ]
    (root / "PHASE56_NEXT_ACTION.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")
    save_json(
        {
            "phase": 56,
            "output_dir": str(root),
            "no_reconstruction_network_training": True,
            "no_gan_training": True,
            "best_strict": strict_rows,
            "claim": claim,
            "next_action": next_action,
        },
        root / "PHASE56_MANIFEST.json",
    )
    finalize(root, {"phase": 56, "status": "complete", "no_reconstruction_network_training": True, "no_gan_training": True})
    print(root / "PHASE56_GROUP_SPLIT_REPORT.md")


if __name__ == "__main__":
    main()
