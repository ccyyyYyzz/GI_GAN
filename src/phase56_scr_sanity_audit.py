from __future__ import annotations

import argparse

from .phase56_common import add_args, max_value, mean, read_csv_rows, save_bar, to_float, write_command_log, write_rows
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase56 Scrambled Hadamard sanity audit.")
    add_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_dir)
    write_command_log(root)
    consistency = [r for r in read_csv_rows(root / "projector_anchor_consistency.csv") if r.get("task") in {"scr5", "scr10"}]
    results = read_csv_rows(root / "group_split_critic_results.csv")
    rows = []
    for r in consistency:
        task = r.get("task", "")
        strict = [x for x in results if x.get("task") == task and x.get("split_mode") == "strict_both_group_split" and x.get("model") in {"deep_exact_null_critic", "ridge_classifier"} and x.get("status") == "ok"]
        auc = max_value([x.get("auc_abs") for x in strict])
        axdata = to_float(r.get("Axdata_minus_y_rel_mean"))
        row_gram = to_float(r.get("row_gram_minus_I"))
        status = "ok"
        if axdata > 1e-3:
            status = "high_risk_scr_anchor_mismatch"
        rows.append(
            {
                "task": task,
                "family": r.get("family", ""),
                "hadamard_normalization_row_gram_minus_I": row_gram,
                "Axdata_minus_y_rel_mean": axdata,
                "A_P0_probe_relative": r.get("A_P0_probe_relative", ""),
                "P0_xdata_energy_ratio_mean": r.get("P0_xdata_energy_ratio_mean", ""),
                "strict_auc_abs_max": auc,
                "label_inversion_auc_abs": auc,
                "status": status,
                "phase53D_scr_near_random_possible_issue": "unlikely_anchor_A_mismatch" if status == "ok" else "possible_anchor_A_mismatch",
                "capacity_explanation": "If deep strict AUC is high while local linear is low, nonlinear critic capacity is plausible.",
            }
        )
    write_rows(root, "scr_sanity_audit", rows, "Phase56 Scr Sanity Audit")
    save_bar(root / "scr_anchor_projector_consistency.png", rows, "task", "Axdata_minus_y_rel_mean", "Scr anchor/projector consistency", "Axdata-y rel")
    save_bar(root / "scr_auc_direction_check.png", rows, "task", "strict_auc_abs_max", "Scr AUC direction check", "max(AUC,1-AUC)")
    report = [
        "# Phase56 Scr Sanity Audit",
        "",
        "- Scrambled Hadamard anchor/projector consistency is checked using `Ax_data ~= y`, row normalization, exact P0 residual, and AUC direction.",
        "- Phase53D Scr near-random is treated as capacity/feature limitation if consistency checks pass and group-split deep critic is high.",
        "- If group-split deep Scr is low, Phase53C Scr GAN pilot should not be cited.",
    ]
    (root / "SCR_SANITY_AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(root / "scr_sanity_audit.csv")


if __name__ == "__main__":
    main()

