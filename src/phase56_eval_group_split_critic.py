from __future__ import annotations

import argparse

from .phase56_common import add_args, max_value, mean, read_csv_rows, save_bar, to_float, write_command_log, write_rows
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase56 aggregate/eval group-split critic results.")
    add_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_dir)
    write_command_log(root)
    rows = [r for r in read_csv_rows(root / "group_split_critic_results.csv") if r.get("status") == "ok"]
    summary = []
    for task in sorted({r.get("task", "") for r in rows}):
        for model in ["deep_exact_null_critic", "ridge_classifier", "pca_logistic", "pca_linear_svm", "anchor_only_baseline", "p0_only_condition_ignored", "shuffled_label_baseline", "random_anchor_baseline", "handcrafted_baseline"]:
            strict = [r for r in rows if r.get("task") == task and r.get("model") == model and r.get("split_mode") == "strict_both_group_split"]
            pair = [r for r in rows if r.get("task") == task and r.get("model") == model and r.get("split_mode") == "pair_split_reproduction"]
            if not strict and not pair:
                continue
            summary.append(
                {
                    "task": task,
                    "family": (strict or pair)[0].get("family", ""),
                    "model": model,
                    "strict_auc_mean": mean([r.get("auc") for r in strict]),
                    "strict_auc_abs_max": max_value([r.get("auc_abs") for r in strict]),
                    "strict_ci_low_min": min([to_float(r.get("auc_ci_low")) for r in strict], default=float("nan")),
                    "pair_split_auc_mean": mean([r.get("auc") for r in pair]),
                    "pair_minus_strict_auc": mean([r.get("auc") for r in pair]) - mean([r.get("auc") for r in strict]),
                    "strict_rows": len(strict),
                    "pair_rows": len(pair),
                }
            )
    write_rows(root, "group_split_critic_summary", summary, "Phase56 Group Split Critic Summary")
    save_bar(root / "pair_split_vs_group_split.png", [r for r in summary if r.get("model") in {"deep_exact_null_critic", "ridge_classifier"}], "task", "pair_minus_strict_auc", "Pair split minus strict group split", "AUC gap")
    print(root / "group_split_critic_summary.csv")


if __name__ == "__main__":
    main()

