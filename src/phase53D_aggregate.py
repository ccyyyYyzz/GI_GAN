from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .phase53D_common import (
    DEFAULT_OUTPUT_ROOT,
    add_phase53d_args,
    fmt,
    read_csv_rows,
    save_bar,
    save_scatter,
    to_float,
    write_rows,
)
from .utils import ensure_dir


TASK_LABELS = {
    ("Rad-5", "full"): "Full Rad-5",
    ("Rad-5", "no_gate"): "Rad-5 no_gate",
    ("Rad-5", "no_final_audit"): "Rad-5 no_final_audit",
    ("Rad-5", "no_gate_no_final_audit"): "Rad-5 no_gate_no_final_audit",
    ("Rad-5", "no_final_audit_no_meas_loss"): "Rad-5 no_final_audit_no_meas_loss",
    ("Scr-5", "full"): "Full Scr-5",
    ("Scr-5", "no_gate"): "Scr-5 no_gate",
    ("Scr-5", "no_final_audit"): "Scr-5 no_final_audit",
    ("Scr-5", "no_gate_no_final_audit"): "Scr-5 no_gate_no_final_audit",
    ("Scr-5", "no_final_audit_no_meas_loss"): "Scr-5 no_final_audit_no_meas_loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase53D local preflight outputs.")
    add_phase53d_args(parser)
    return parser.parse_args()


def best_auc_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if "pair_pca" not in str(row.get("model", "")):
            continue
        task = str(row.get("task", ""))
        if not task:
            continue
        if task not in best or to_float(row.get("auc")) > to_float(best[task].get("auc")):
            best[task] = row
    return best


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [to_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def maxval(rows: list[dict[str, Any]], key: str) -> float:
    vals = [to_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v == v]
    return max(vals) if vals else float("nan")


def build_ablation_matrix(args, out: Path) -> list[dict[str, Any]]:
    combined = read_csv_rows(Path(args.phase51A_root) / "phase51A_ablation_matrix.csv")
    if not combined:
        combined = read_csv_rows(Path(args.phase48_root) / "phase48_49_summary.csv")
    rows: list[dict[str, Any]] = []
    for source in combined:
        family = str(source.get("family", ""))
        variant = str(source.get("variant", ""))
        label = TASK_LABELS.get((family, variant))
        if not label:
            continue
        full = next((r for r in combined if r.get("family") == family and r.get("variant") == "full"), {})
        full_psnr = to_float(full.get("psnr"))
        full_rel = to_float(full.get("rel_meas_err"))
        psnr = to_float(source.get("psnr"))
        rel = to_float(source.get("rel_meas_err"))
        rel_ratio = rel / full_rel if full_rel and full_rel == full_rel else float("nan")
        if variant == "full":
            interp = "strict no-leak full reference"
        elif "no_final_audit_no_meas_loss" in variant:
            interp = "measurement accountability degrades most; PSNR remains close"
        elif "no_final_audit" in variant:
            interp = "PSNR mostly survives, RelMeasErr rises; final audit mainly certifies measurements"
        elif variant == "no_gate":
            interp = "P_N/gate is not the current train-time PSNR driver"
        else:
            interp = "diagnostic ablation"
        rows.append(
            {
                "row": label,
                "PSNR": psnr,
                "SSIM": to_float(source.get("ssim")),
                "RelMeasErr": rel,
                "PSNR drop vs Full": full_psnr - psnr if full_psnr == full_psnr else to_float(source.get("psnr_drop_vs_full")),
                "RelMeasErr ratio": rel_ratio,
                "status": source.get("status", "imported"),
                "interpretation": interp,
            }
        )
    order = {label: i for i, label in enumerate(TASK_LABELS.values())}
    rows.sort(key=lambda r: order.get(str(r["row"]), 999))
    write_rows(out, "phase48_51A_ablation_matrix", rows, "Phase48/49 + Phase51A Ablation Matrix")
    return rows


def build_anchor_proxy(out: Path, anchor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = best_auc_rows(anchor_rows)
    rows = []
    for task, row in sorted(best.items()):
        bp = to_float(row.get("bp_psnr"))
        model = to_float(row.get("model_psnr"))
        auc = to_float(row.get("auc"))
        rows.append(
            {
                "task": task,
                "family": row.get("family", ""),
                "BP PSNR / anchor PSNR": bp,
                "model PSNR": model,
                "Delta PSNR": model - bp if bp == bp and model == model else "",
                "RelMeasErr full": row.get("rel_meas_err", ""),
                "Task 2 AUC": auc,
                "Task 2 AUC CI": f"{fmt(row.get('auc_ci_low'), 3)}-{fmt(row.get('auc_ci_high'), 3)}",
                "Task 2 InfoNCE MI estimate": "",
                "provenance measured indicator": "not tested in Phase53D",
                "learned/null indicator": "local CPU pair classifier",
            }
        )
    write_rows(out, "anchor_information_proxy", rows, "Phase53D Anchor Information Proxy")
    save_scatter(out / "auc_vs_bp_psnr.png", rows, "BP PSNR / anchor PSNR", "Task 2 AUC", "AUC vs BP PSNR", "BP PSNR", "AUC")
    save_scatter(out / "mi_vs_bp_psnr.png", rows, "BP PSNR / anchor PSNR", "Task 2 AUC", "AUC proxy vs BP PSNR", "BP PSNR", "AUC proxy")
    save_scatter(out / "delta_psnr_vs_auc.png", rows, "Task 2 AUC", "Delta PSNR", "Delta PSNR vs AUC", "AUC", "Delta PSNR")
    save_bar(out / "anchor_strength_summary.png", rows, "task", "BP PSNR / anchor PSNR", "Anchor strength summary", "BP PSNR")
    return rows


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    ablation = build_ablation_matrix(args, out)
    exact = read_csv_rows(out / "exact_projector_checks.csv")
    soft = read_csv_rows(out / "soft_leakage_by_lambda.csv")
    anchor = read_csv_rows(out / "anchor_null_pretest_results.csv")
    feasible = read_csv_rows(out / "feasible_hallucination_metrics.csv")
    shortcut = read_csv_rows(out / "shortcut_audit_results.csv")
    posthoc = read_csv_rows(out / "posthoc_certificate_sweep.csv")
    proxy = build_anchor_proxy(out, anchor)
    best = best_auc_rows(anchor)
    max_auc = max((to_float(r.get("auc")) for r in best.values()), default=float("nan"))
    best_task = max(best.values(), key=lambda r: to_float(r.get("auc")), default={})
    max_ap0 = maxval(exact, "A_P0_relative_norm")
    max_soft = maxval(soft, "mean_theory_leakage_factor")
    cross_rel = mean(feasible, "cross_relmeas")
    ours_rel = mean(feasible, "ours_relmeas")
    hard_posthoc = [r for r in posthoc if str(r.get("lambda")) == "hard" and r.get("status") == "ok"]
    rel_before = mean(hard_posthoc, "relmeas_before")
    rel_after = mean(hard_posthoc, "relmeas_after")
    psnr_delta = mean(hard_posthoc, "psnr_change_after_minus_before")
    residual_wrong = [r for r in shortcut if r.get("feature_set") == "residual_features" and r.get("test") == "train_wrong_y_test_feasible"]
    residual_eval_auc = mean(residual_wrong, "eval_auc")
    exact_feas = [r for r in shortcut if r.get("feature_set") == "exact_null_features" and "feasible" in str(r.get("test"))]
    exact_feas_auc = maxval(exact_feas, "eval_auc")
    all_auc_under = all(to_float(r.get("auc")) < 0.6 for r in best.values()) if best else True
    scr_positive = any(
        str(r.get("task")) in {"scr5", "scr10"} and to_float(r.get("auc")) >= 0.70 and to_float(r.get("auc_ci_low")) > 0.60
        for r in best.values()
    )
    go = "continue Colab exact-null critic evaluator/pretest for positive Scr families" if scr_positive else "no GAN; keep projector theory and feasible hallucination diagnostics"
    if all_auc_under:
        go = "no GAN; E1-mini did not pass AUC 0.60"
    report = [
        "# Phase53D Local Preflight Report",
        "",
        "Scope: local preflight / diagnostic only. No full neural training and no main reconstruction-network training were started.",
        "",
        "## Required Answers",
        "",
        f"1. exact P0 numerical correctness: max A_P0_relative_norm across tasks = {max_ap0:.3e}.",
        f"2. soft P_N^lambda row-space leakage: max mean theoretical leakage factor = {fmt(max_soft, 5)}; leakage follows lambda/(lambda+sigma_i^2).",
        f"3. E1-mini anchor-null separability: max best-task AUC = {fmt(max_auc, 3)}.",
        f"4. Most separable family: {best_task.get('task', 'n/a')} ({best_task.get('family', 'n/a')}) with AUC {fmt(best_task.get('auc'), 3)}.",
        f"5. Rad-5 weak-anchor consistency: inspect Rad-5 AUC vs BP PSNR in `anchor_information_proxy.csv`; low Rad-5 AUC is consistent with weak anchor.",
        "6. Learned classifier vs handcrafted baseline: compare `ridge/logistic/linear_svm_pair_pca` rows against `handcrafted_abs_cosine_baseline` in `anchor_null_pretest_results.csv`.",
        f"7. Feasible hallucination success: cross-feasible mean RelMeasErr = {fmt(cross_rel, 5)} vs ours mean RelMeasErr = {fmt(ours_rel, 5)}.",
        f"8. Residual shortcut: residual-feature train-wrong-y/test-feasible mean AUC = {fmt(residual_eval_auc, 3)}; exact-null feasible max AUC = {fmt(exact_feas_auc, 3)}.",
        f"9. Posthoc certificate: hard audit mean RelMeasErr {fmt(rel_before, 5)} -> {fmt(rel_after, 5)}, mean PSNR change {fmt(psnr_delta, 3)} dB.",
        f"10. Colab Phase53C recommendation: {go}.",
        "11. Main-text candidates: projector theory and feasible hallucination figure if visually clear.",
        "12. Supplement candidates: E1-mini AUC tables, soft-leakage checks, shortcut audit, posthoc certificate sweep.",
        "13. GAN fine-tune recommendation: do not run GAN unless a family passes E1; prefer critic-as-evaluator.",
        "14. Sampling scaling: useful next only after positive E1/critic signal; otherwise not mandatory.",
        "",
        "## Ablation Interpretation",
        "",
        "- P_N/gate is not the current train-time PSNR driver.",
        "- final audit is not a retrained PSNR necessary condition.",
        "- final audit / measurement loss primarily affects RelMeasErr / accountability.",
        "- PSNR and measurement accountability can separate.",
    ]
    (out / "PHASE53D_LOCAL_PREFLIGHT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision = [
        "# Phase53D Go/No-Go Decision",
        "",
        f"Decision: {go}.",
        "",
        "Rules applied:",
        "- If all exact-null AUC < 0.6: no GAN.",
        "- If Scr-5 or Scr-10 AUC >= 0.70 and CI lower > 0.60: continue Colab exact-null critic evaluator.",
        "- Never recommend full MCAC with residual inputs.",
    ]
    (out / "PHASE53D_GO_NO_GO_DECISION.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    supported = [
        "# Phase53D Supported Local Claims",
        "",
        "- Exact P0 is the correct row-blind input for null-space plausibility diagnostics when projector checks pass.",
        "- Soft P_N^lambda leaks row-space information and should not be used as an exact-null critic input.",
        "- Analytic Pi_y can serve as a post-hoc measurement certificate / re-legalization operator.",
        "- Feasible hallucination illustrates that measurement consistency alone does not imply perceptual correctness.",
        "- Phase48/51A ablations support separating PSNR from measurement accountability.",
    ]
    (out / "PHASE53D_SUPPORTED_LOCAL_CLAIMS.md").write_text("\n".join(supported) + "\n", encoding="utf-8")
    failed = [
        "# Phase53D Failed Or Risky Claims",
        "",
        "- Do not claim GAN improvement or SOTA from Phase53D.",
        "- Do not claim hardware effects.",
        "- Do not claim P_N or Pi_y is a train-time PSNR-essential module.",
        "- Do not use full-input residual discriminators as proof of null-space plausibility.",
        "- Do not change the main no-leak result table based on this diagnostic phase.",
    ]
    (out / "PHASE53D_FAILED_OR_RISKY_CLAIMS.md").write_text("\n".join(failed) + "\n", encoding="utf-8")
    print(out / "PHASE53D_LOCAL_PREFLIGHT_REPORT.md")


if __name__ == "__main__":
    main()
