from __future__ import annotations

import argparse

from .phase56_common import add_args, mean, read_csv_rows, save_bar, save_scatter, to_float, write_command_log, write_rows
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase56 memorization and leakage diagnostics.")
    add_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_dir)
    write_command_log(root)
    rows = [r for r in read_csv_rows(root / "group_split_critic_results.csv") if r.get("status") == "ok"]
    out = []
    for task in sorted({r.get("task", "") for r in rows}):
        for model in ["deep_exact_null_critic", "ridge_classifier", "pca_logistic", "pca_linear_svm"]:
            strict = [r for r in rows if r.get("task") == task and r.get("model") == model and r.get("split_mode") == "strict_both_group_split" and r.get("negative_type") == "random"]
            pair = [r for r in rows if r.get("task") == task and r.get("model") == model and r.get("split_mode") == "pair_split_reproduction" and r.get("negative_type") == "random"]
            if not strict and not pair:
                continue
            strict_auc = mean([r.get("auc") for r in strict])
            pair_auc = mean([r.get("auc") for r in pair])
            out.append(
                {
                    "task": task,
                    "model": model,
                    "strict_random_auc_mean": strict_auc,
                    "pair_random_auc_mean": pair_auc,
                    "pair_minus_strict": pair_auc - strict_auc,
                    "diagnosis": "pair_split_artifact_risk" if pair_auc - strict_auc > 0.15 and strict_auc < 0.65 else "no_large_pair_split_gap",
                }
            )
        for baseline in ["anchor_only_baseline", "p0_only_condition_ignored", "shuffled_label_baseline", "random_anchor_baseline", "handcrafted_baseline"]:
            strict = [r for r in rows if r.get("task") == task and r.get("model") == baseline and r.get("split_mode") == "strict_both_group_split"]
            auc = mean([r.get("auc_abs") for r in strict])
            out.append(
                {
                    "task": task,
                    "model": baseline,
                    "strict_random_auc_mean": mean([r.get("auc") for r in strict if r.get("negative_type") == "random"]),
                    "pair_random_auc_mean": "",
                    "pair_minus_strict": "",
                    "diagnosis": "baseline_not_random_risk" if auc > 0.65 else "baseline_near_random_or_inverted",
                }
            )
    write_rows(root, "memorization_leakage_diagnostics", out, "Phase56 Memorization Leakage Diagnostics")
    save_bar(root / "baseline_failure_checks.png", [r for r in out if "baseline" in r.get("model", "")], "model", "strict_random_auc_mean", "Baseline failure checks", "strict random AUC")
    save_bar(root / "id_overlap_heatmap.png", read_csv_rows(root / "group_split_overlap_audit.csv"), "split_mode", "any_image_overlap_count", "ID overlap audit", "overlap count")
    lines = [
        "# Phase56 Memorization Leakage Diagnostics",
        "",
        "- Pair split vs group split is summarized in `memorization_leakage_diagnostics.csv`.",
        "- If pair split is high while strict split is low, Phase53C AUC 0.992 should be treated as pair-split artifact risk.",
        "- Anchor-only, P0-only, shuffled-label, random-anchor, and handcrafted baselines are checked as leakage controls.",
    ]
    (root / "MEMORIZATION_LEAKAGE_DIAGNOSTICS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "memorization_leakage_diagnostics.csv")


if __name__ == "__main__":
    main()

