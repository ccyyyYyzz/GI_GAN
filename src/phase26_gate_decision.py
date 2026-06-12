from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from .phase26_common import (
    best_by_metric,
    drive_root,
    fmt,
    main_results_from_drive,
    markdown_table,
    output_root,
    read_csv,
    safe_float,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 26 full-training gate decision.")
    parser.add_argument("--drive_root", default=None)
    return parser.parse_args()


def best_pca_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners = []
    for method_id in sorted({row.get("method_id", "") for row in rows}):
        subset = [row for row in rows if row.get("method_id") == method_id and row.get("status") == "ok"]
        if subset:
            winners.append(sorted(subset, key=lambda row: safe_float(row.get("pca_psnr")), reverse=True)[0])
    return winners


def pilot_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return best_by_metric([row for row in rows if row.get("status") == "complete"], "psnr", "family")


def current_hq_by_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        if row.get("status") == "complete" and row.get("config_name", "").startswith("current_hq_"):
            out[row.get("family", "")] = row
    return out


def compare_pilots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current = current_hq_by_family(rows)
    decisions = []
    for row in rows:
        if row.get("status") != "complete" or row.get("config_name", "").startswith("current_hq_"):
            continue
        base = current.get(row.get("family", ""))
        if not base:
            decisions.append({**row, "decision": "pending", "reason": "current_hq pilot missing"})
            continue
        d_psnr = safe_float(row.get("psnr")) - safe_float(base.get("psnr"))
        d_ssim = safe_float(row.get("ssim")) - safe_float(base.get("ssim"))
        params_ratio = safe_float(row.get("params")) / max(1.0, safe_float(base.get("params")))
        if d_psnr >= 0.3 or d_ssim >= 0.01:
            decision = "recommend_full_training"
            reason = f"beats current_hq by {d_psnr:.3f} dB / {d_ssim:.4f} SSIM"
        elif d_psnr >= -0.2 and params_ratio < 0.5:
            decision = "optional_efficiency_full_training"
            reason = f"within 0.2 dB and parameter ratio is {params_ratio:.3f}"
        else:
            decision = "do_not_full_train"
            reason = f"does not beat gate; delta={d_psnr:.3f} dB / {d_ssim:.4f} SSIM"
        decisions.append({**row, "delta_psnr_vs_current_hq": d_psnr, "delta_ssim_vs_current_hq": d_ssim, "decision": decision, "reason": reason})
    return decisions


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def main() -> None:
    args = parse_args()
    root = drive_root(args.drive_root)
    out = output_root(root)
    pca_rows = read_csv(out / "pca_oracle_full" / "pca_oracle_full_results.csv")
    pilot_rows = read_csv(out / "arch_pilot_results.csv")
    current = main_results_from_drive(root)
    pca_best = best_pca_by_method(pca_rows)
    pilot_best = pilot_winners(pilot_rows)
    pilot_decisions = compare_pilots(pilot_rows)

    pca_gap_rows = []
    for row in pca_best:
        gap = safe_float(row.get("gap_to_current_psnr"))
        pca_gap_rows.append(
            {
                "method_id": row.get("method_id", ""),
                "best_k": row.get("k", ""),
                "pca_psnr": fmt(row.get("pca_psnr"), 3),
                "pca_ssim": fmt(row.get("pca_ssim"), 3),
                "current_model_psnr": fmt(row.get("current_model_psnr"), 3),
                "gap_to_current_psnr": fmt(gap, 3),
                "interpretation": "far_below_current" if math.isfinite(gap) and gap >= 1.0 else "near_current",
            }
        )

    recommend_full = [row for row in pilot_decisions if row.get("decision") == "recommend_full_training"]
    optional_full = [row for row in pilot_decisions if row.get("decision") == "optional_efficiency_full_training"]
    any_complete_pilot = any(row.get("status") == "complete" for row in pilot_rows)
    all_expected_pilots_complete = len([row for row in pilot_rows if row.get("status") == "complete"]) >= 6
    pca_far_below = bool(pca_gap_rows) and all(safe_float(row["gap_to_current_psnr"]) >= 1.0 for row in pca_gap_rows)

    decision_rows = []
    for row in pilot_decisions:
        item = {
            "config_name": row.get("config_name", ""),
            "family": row.get("family", ""),
            "model_type": row.get("model_type", ""),
            "psnr": fmt(row.get("psnr"), 3),
            "ssim": fmt(row.get("ssim"), 3),
            "delta_psnr_vs_current_hq": fmt(row.get("delta_psnr_vs_current_hq"), 3),
            "delta_ssim_vs_current_hq": fmt(row.get("delta_ssim_vs_current_hq"), 4),
            "decision": row.get("decision", ""),
            "reason": row.get("reason", ""),
        }
        decision_rows.append(item)

    text = f"""# Phase 26 Gate Decision

## Inputs

- PCA full results: `{out / "pca_oracle_full" / "pca_oracle_full_results.csv"}`
- Architecture pilot results: `{out / "arch_pilot_results.csv"}`
- Current strict no-leak main results: `{root / "outputs_phase16" / "supplementary_experiments" / "attribution" / "attribution_final.csv"}`

## Required Questions

1. Is the PCA oracle clearly below the current model?

{yes_no(pca_far_below)}. Current best PCA gaps are:

{markdown_table(pca_gap_rows, ["method_id", "best_k", "pca_psnr", "pca_ssim", "current_model_psnr", "gap_to_current_psnr", "interpretation"])}

2. Can the linear prior explain the current 22-25 dB results?

{"No, not by itself, if the above gaps stay >=1 dB." if pca_far_below else "Possibly partially; inspect the method-wise gaps before making a strong claim."}

3. Which architecture pilot is strongest?

{markdown_table(pilot_best, ["config_name", "family", "model_type", "psnr", "ssim", "params", "status"])}

4. Does NAFNet-small beat current_hq pilot?

{yes_no(any(row.get("model_type") == "nafnet_small" and row.get("decision") == "recommend_full_training" for row in pilot_decisions))}

5. Does Unrolled-ISTA beat current_hq pilot?

{yes_no(any(row.get("model_type") == "unrolled_ista" and row.get("decision") == "recommend_full_training" for row in pilot_decisions))}

6. Is full 80-epoch training necessary?

{"Yes, for the recommended rows below." if recommend_full else ("Optional only for efficiency rows below." if optional_full else "No full architecture training is recommended from completed pilots.")}

7. If full training is needed, which configs?

{markdown_table(recommend_full or optional_full, ["config_name", "family", "model_type", "decision", "reason"])}

8. If no clear improvement exists, should architecture exploration stop?

{"Yes for now; keep architecture pilot as future/planning evidence." if not recommend_full and any_complete_pilot else "Not decidable until medium pilots finish."}

## Pilot Gate Table

{markdown_table(decision_rows, ["config_name", "family", "model_type", "psnr", "ssim", "delta_psnr_vs_current_hq", "delta_ssim_vs_current_hq", "decision", "reason"])}

## Guardrails

- Medium pilot numbers must not be mixed with the strict no-leak 80-epoch main table.
- Do not claim strict SOTA from this gate.
- Do not claim GAN is the final dominant mechanism.
- Architecture pilot is a planning signal unless full no-leak training is explicitly approved.
- Gate status: `{"complete" if pca_rows and all_expected_pilots_complete else "partial_or_pending"}`.
"""
    write_text(out / "PHASE26_GATE_DECISION.md", text)
    write_json(
        out / "PHASE26_GATE_DECISION.json",
        {
            "pca_best": pca_gap_rows,
            "pilot_best": pilot_best,
            "pilot_decisions": pilot_decisions,
            "recommend_full": recommend_full,
            "optional_full": optional_full,
            "status": "complete" if pca_rows and all_expected_pilots_complete else "partial_or_pending",
        },
    )
    print({"gate_report": str(out / "PHASE26_GATE_DECISION.md"), "recommend_full": len(recommend_full)})


if __name__ == "__main__":
    main()
