from __future__ import annotations

from .phase55_common import (
    PHASE53C_ROOT,
    PHASE55_ROOT,
    TASKS,
    add_metric,
    mean,
    read_csv_rows,
    read_text,
    to_float,
    write_rows,
)


def main() -> None:
    root = PHASE53C_ROOT
    out = PHASE55_ROOT
    rows: list[dict] = []
    add_metric(
        rows,
        phase="Phase53C",
        group="report",
        metric="aggregate_report_present",
        value=bool((root / "PHASE53C_AGGREGATE_REPORT.md").exists()),
        source=str(root / "PHASE53C_AGGREGATE_REPORT.md"),
    )
    mi_rows = read_csv_rows(root / "session_20_exact_null_mi_pretest" / "exact_null_mi_pretest_results.csv")
    for r in mi_rows:
        task = r.get("task", "")
        family = r.get("family", "")
        model = r.get("model", "")
        source = "session_20_exact_null_mi_pretest/exact_null_mi_pretest_results.csv"
        for metric in ["auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "infoNCE_mi_lower_nats"]:
            if metric in r and r.get(metric) != "":
                add_metric(rows, phase="Phase53C", group="exact_null_mi_pretest", metric=metric, value=r.get(metric), task=task, family=family, model=model, source=source, note=r.get("negative_mode", ""))
        if "handcrafted" in model:
            add_metric(rows, phase="Phase53C", group="baseline", metric="handcrafted_baseline_auc", value=r.get("auc"), task=task, family=family, model=model, source=source)
        if "condition_ignored" in model:
            add_metric(rows, phase="Phase53C", group="baseline", metric="condition_ignored_auc", value=r.get("auc"), task=task, family=family, model=model, source=source)
        if "anchor_only" in model:
            add_metric(rows, phase="Phase53C", group="baseline", metric="anchor_only_auc", value=r.get("auc"), task=task, family=family, model=model, source=source)
    for task in TASKS:
        checks = read_csv_rows(root / "session_20_exact_null_mi_pretest" / task / "exact_projector_checks.csv")
        for r in checks:
            for metric in ["A_P0_norm", "qtq_minus_I_norm", "P0_idempotence_norm", "row_rank", "singular_min", "singular_max"]:
                if metric in r:
                    add_metric(rows, phase="Phase53C", group="exact_projector", metric=metric, value=r.get(metric), task=task, family=r.get("family", ""), source=f"session_20_exact_null_mi_pretest/{task}/exact_projector_checks.csv")
    leakage = read_csv_rows(root / "session_21_soft_leakage_and_shortcut_audit" / "soft_leakage_results.csv")
    for r in leakage:
        for metric in ["mean_leakage_ratio", "max_leakage_ratio", "recover_Au_R2", "A_P0_norm"]:
            if metric in r:
                add_metric(rows, phase="Phase53C", group="soft_leakage", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), model=r.get("projection", ""), source="session_21_soft_leakage_and_shortcut_audit/soft_leakage_results.csv", note=f"lambda={r.get('lambda','')}")
    shortcut = read_csv_rows(root / "session_21_soft_leakage_and_shortcut_audit" / "shortcut_audit_results.csv")
    for r in shortcut:
        for metric in ["train_auc", "eval_auc", "eval_accuracy"]:
            add_metric(rows, phase="Phase53C", group="shortcut_audit", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), model=r.get("model", ""), source="session_21_soft_leakage_and_shortcut_audit/shortcut_audit_results.csv", note=r.get("test", ""))
    feasible = read_csv_rows(root / "session_22_feasible_hallucination_figure" / "feasible_hallucination_metrics.csv")
    for r in feasible:
        for metric in ["gt_relmeas", "ours_relmeas", "cross_relmeas", "ours_psnr", "cross_psnr_vs_gt", "cross_ssim_vs_gt"]:
            add_metric(rows, phase="Phase53C", group="feasible_hallucination", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), source="session_22_feasible_hallucination_figure/feasible_hallucination_metrics.csv")
    critic_eval = read_csv_rows(root / "session_23_exact_null_critic_evaluator" / "critic_evaluator_scores.csv")
    for r in critic_eval:
        for metric in ["critic_score_mean", "rel_meas_err", "psnr", "ssim"]:
            if metric in r:
                add_metric(rows, phase="Phase53C", group="critic_evaluator", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), model=r.get("method", ""), source="session_23_exact_null_critic_evaluator/critic_evaluator_scores.csv")
    gan = read_csv_rows(root / "session_24_optional_gan_and_posterior_sampling" / "optional_gan_results.csv")
    for r in gan:
        for metric in ["psnr", "ssim", "rel_meas_err", "beta"]:
            if metric in r:
                add_metric(rows, phase="Phase53C", group="optional_gan", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), model=r.get("status", ""), source="session_24_optional_gan_and_posterior_sampling/optional_gan_results.csv", note=f"beta={r.get('beta','')}")
        add_metric(rows, phase="Phase53C", group="optional_gan", metric="status", value=r.get("status", ""), task=r.get("task", ""), family=r.get("family", ""), source="session_24_optional_gan_and_posterior_sampling/optional_gan_results.csv")
    posterior = read_csv_rows(root / "session_24_optional_gan_and_posterior_sampling" / "posterior_sampling_metrics.csv")
    for r in posterior:
        for metric in ["variance_null_ratio_mean", "variance_row_ratio_mean", "rel_meas_err_mean", "psnr_mean", "ssim_mean"]:
            if metric in r:
                add_metric(rows, phase="Phase53C", group="posterior_sampling", metric=metric, value=r.get(metric), task=r.get("task", ""), family=r.get("family", ""), source="session_24_optional_gan_and_posterior_sampling/posterior_sampling_metrics.csv")
    write_rows(out, "phase53C_extracted_summary", rows, "Phase53C Extracted Summary")
    report = [
        "# Phase53C Extracted Summary",
        "",
        f"- exact-null MI/pretest rows: {len(mi_rows)}",
        f"- shortcut rows: {len(shortcut)}",
        f"- feasible hallucination rows: {len(feasible)}",
        f"- optional GAN rows: {len(gan)}",
        f"- posterior rows: {len(posterior)}",
        f"- aggregate report present: {bool(read_text(root / 'PHASE53C_AGGREGATE_REPORT.md'))}",
        f"- max AUC: {max([to_float(r.get('auc')) for r in mi_rows] or [float('nan')]):.3f}",
        f"- mean InfoNCE lower bound: {mean([r.get('infoNCE_mi_lower_nats') for r in mi_rows]):.3f}",
    ]
    (out / "PHASE53C_EXTRACTION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out / "phase53C_extracted_summary.csv")


if __name__ == "__main__":
    main()

