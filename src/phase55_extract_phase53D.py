from __future__ import annotations

from .phase55_common import PHASE53D_ROOT, PHASE55_ROOT, add_metric, mean, read_csv_rows, read_text, to_float, write_rows


def main() -> None:
    root = PHASE53D_ROOT
    out = PHASE55_ROOT
    rows: list[dict] = []
    for report_name in ["PHASE53D_LOCAL_PREFLIGHT_REPORT.md", "PHASE53D_GO_NO_GO_DECISION.md"]:
        add_metric(rows, phase="Phase53D", group="report", metric=f"{report_name}_present", value=bool((root / report_name).exists()), source=str(root / report_name))
    exact = read_csv_rows(root / "exact_projector_checks.csv")
    for r in exact:
        for metric in ["A_P0_relative_norm", "P0_idempotence_relative_norm", "PR_idempotence_relative_norm", "row_rank", "singular_min", "singular_max"]:
            add_metric(rows, phase="Phase53D", group="exact_projector", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), source="exact_projector_checks.csv")
    soft = read_csv_rows(root / "soft_leakage_by_lambda.csv")
    for r in soft:
        for metric in ["mean_theory_leakage_factor", "max_theory_leakage_factor"]:
            add_metric(rows, phase="Phase53D", group="soft_leakage", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), source="soft_leakage_by_lambda.csv", note=f"lambda={r.get('lambda','')}")
    anchor = read_csv_rows(root / "anchor_null_pretest_results.csv")
    for r in anchor:
        for metric in ["auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "bp_psnr", "model_psnr", "rel_meas_err"]:
            add_metric(rows, phase="Phase53D", group="anchor_null_pretest", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), model=r.get("model", ""), source="anchor_null_pretest_results.csv", note=r.get("negative_type", ""))
        model = r.get("model", "")
        if "handcrafted" in model:
            add_metric(rows, phase="Phase53D", group="baseline", metric="handcrafted_baseline_auc", value=r.get("auc"), task=r.get("task", ""), family=r.get("family", ""), model=model, source="anchor_null_pretest_results.csv")
        if "condition_ignored" in model:
            add_metric(rows, phase="Phase53D", group="baseline", metric="condition_ignored_auc", value=r.get("auc"), task=r.get("task", ""), family=r.get("family", ""), model=model, source="anchor_null_pretest_results.csv")
        if "anchor_only" in model:
            add_metric(rows, phase="Phase53D", group="baseline", metric="anchor_only_auc", value=r.get("auc"), task=r.get("task", ""), family=r.get("family", ""), model=model, source="anchor_null_pretest_results.csv")
    feasible = read_csv_rows(root / "feasible_hallucination_metrics.csv")
    for r in feasible:
        for metric in ["gt_relmeas", "ours_relmeas", "cross_relmeas", "ours_psnr", "cross_psnr_vs_gt", "cross_ssim_vs_gt"]:
            add_metric(rows, phase="Phase53D", group="feasible_hallucination", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), source="feasible_hallucination_metrics.csv")
    shortcut = read_csv_rows(root / "shortcut_audit_results.csv")
    for r in shortcut:
        for metric in ["train_auc", "eval_auc", "eval_accuracy", "eval_balanced_accuracy"]:
            add_metric(rows, phase="Phase53D", group="shortcut_audit", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), model=f"{r.get('feature_set','')}:{r.get('classifier','')}", source="shortcut_audit_results.csv", note=r.get("test", ""))
    posthoc = read_csv_rows(root / "posthoc_certificate_sweep.csv")
    for r in posthoc:
        for metric in ["psnr_before", "psnr_after", "relmeas_before", "relmeas_after", "psnr_change_after_minus_before", "relmeas_change_after_minus_before"]:
            add_metric(rows, phase="Phase53D", group="posthoc_certificate", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("variant", ""), model=r.get("lambda", ""), source="posthoc_certificate_sweep.csv", note=r.get("status", ""))
    write_rows(out, "phase53D_extracted_summary", rows, "Phase53D Extracted Summary")
    report = [
        "# Phase53D Extracted Summary",
        "",
        f"- exact projector rows: {len(exact)}",
        f"- anchor-null pretest rows: {len(anchor)}",
        f"- shortcut rows: {len(shortcut)}",
        f"- posthoc rows: {len(posthoc)}",
        f"- local report present: {bool(read_text(root / 'PHASE53D_LOCAL_PREFLIGHT_REPORT.md'))}",
        f"- max local E1 AUC: {max([to_float(r.get('auc')) for r in anchor] or [float('nan')]):.3f}",
        f"- mean hard posthoc PSNR change: {mean([r.get('psnr_change_after_minus_before') for r in posthoc if r.get('lambda') == 'hard']):.3f}",
    ]
    (out / "PHASE53D_EXTRACTION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out / "phase53D_extracted_summary.csv")


if __name__ == "__main__":
    main()

