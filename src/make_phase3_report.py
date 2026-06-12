from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


PHASE3_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase3")
PHASE2_CLEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_clean_phase2")
ENV_PATH = Path("E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Phase 3 report.")
    parser.add_argument("--phase3_dir", default=str(PHASE3_ROOT))
    parser.add_argument("--phase2_clean_dir", default=str(PHASE2_CLEAN_ROOT))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value) -> str:
    if value in ("", None):
        return "missing"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def markdown_table(rows: list[dict], cols: list[str]) -> list[str]:
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    if not rows:
        lines.append("|" + "|".join(["missing"] * len(cols)) + "|")
        return lines
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    return lines


def find_row(rows: list[dict], method: str, ratio: float) -> dict | None:
    for row in rows:
        try:
            same_ratio = abs(float(row.get("sampling_ratio", -1)) - ratio) < 1e-9
        except Exception:
            same_ratio = False
        if row.get("method") == method and same_ratio:
            return row
    return None


def compare_5pct(rows: list[dict]) -> dict:
    fixed = find_row(rows, "Fixed Rademacher", 0.05) or {}
    learned = find_row(rows, "Learned Binary STE", 0.05) or {}
    if fixed.get("status") != "ok" or learned.get("status") != "ok":
        return {"status": "missing"}
    return {
        "status": "ok",
        "fixed_psnr": float(fixed["model_psnr"]),
        "learned_psnr": float(learned["model_psnr"]),
        "delta_psnr": float(learned["model_psnr"]) - float(fixed["model_psnr"]),
        "fixed_ssim": float(fixed["model_ssim"]),
        "learned_ssim": float(learned["model_ssim"]),
        "delta_ssim": float(learned["model_ssim"]) - float(fixed["model_ssim"]),
    }


def conclusion(rows: list[dict], sanity: dict | None) -> str:
    comp = compare_5pct(rows)
    sanity_ok = bool((sanity or {}).get("passed", False))
    learned = find_row(rows, "Learned Binary STE", 0.05) or {}
    if not sanity_ok:
        return "Phase 3 minimum sanity check is not complete yet, so learned-pattern claims should wait."
    if learned.get("status") != "ok":
        return "Learnable patterns pass sanity, but learned binary 5% train/eval is not complete yet."
    if comp.get("status") != "ok":
        return "Learned binary 5% is complete, but fixed-vs-learned comparison is missing."
    if comp["delta_psnr"] > 0 or comp["delta_ssim"] > 0:
        return (
            "Phase 3 minimum chain is complete. Learned binary 5% improves at least one "
            f"primary metric over fixed rademacher: delta_PSNR={comp['delta_psnr']:.6f}, "
            f"delta_SSIM={comp['delta_ssim']:.6f}."
        )
    return (
        "Phase 3 minimum chain is complete, but learned binary 5% does not yet beat the "
        f"fixed rademacher baseline: delta_PSNR={comp['delta_psnr']:.6f}, "
        f"delta_SSIM={comp['delta_ssim']:.6f}."
    )


def phase4_suggestion(rows: list[dict], sanity: dict | None) -> str:
    comp = compare_5pct(rows)
    learned = find_row(rows, "Learned Binary STE", 0.05) or {}
    minimum_ok = bool((sanity or {}).get("passed", False)) and learned.get("status") == "ok"
    if not minimum_ok:
        return "Do not start Phase 4 yet; finish sanity and learned binary 5% first."
    if comp.get("status") == "ok" and (comp["delta_psnr"] > 0 or comp["delta_ssim"] > 0):
        return "Phase 4 is reasonable: the minimum learned-pattern chain is complete and improves at least one 5% metric."
    return "Phase 4 can be scoped as optimization/tuning, but not as a stronger learned-pattern claim yet."


def main() -> None:
    args = parse_args()
    phase3_dir = Path(args.phase3_dir)
    phase2_clean_dir = Path(args.phase2_clean_dir)
    main_rows = read_csv(phase3_dir / "phase3_main_results.csv")
    ablation_rows = read_csv(phase3_dir / "phase3_pattern_ablation_results.csv")
    stats_rows = read_csv(phase3_dir / "phase3_pattern_stats.csv")
    sanity = read_json(phase3_dir / "debug_binary_5pct" / "sanity_learnable_patterns.json")

    comp = compare_5pct(main_rows)
    lines = [
        "# Phase 3 Report",
        "",
        f"- Experiment report time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Clean environment path: {ENV_PATH}",
        "- Dataset path: E:/ns_mc_gan_gi/data",
        f"- Phase 3 output path: {phase3_dir}",
        f"- Phase 2 clean baseline path: {phase2_clean_dir}",
        f"- sanity_learnable_patterns passed: {bool((sanity or {}).get('passed', False))}",
        "",
        "## Phase 2 Fixed Baseline Summary",
        "",
        "|sampling_ratio|fixed_model_psnr|fixed_model_ssim|fixed_model_mse|status|",
        "|---|---|---|---|---|",
    ]
    for ratio in [0.02, 0.05, 0.10]:
        row = find_row(main_rows, "Fixed Rademacher", ratio) or {}
        lines.append(
            "|"
            + "|".join(
                [
                    fmt(ratio),
                    fmt(row.get("model_psnr", "")),
                    fmt(row.get("model_ssim", "")),
                    fmt(row.get("model_mse", "")),
                    fmt(row.get("status", "missing")),
                ]
            )
            + "|"
        )

    lines.extend(
        [
            "",
            "## Phase 3 Learned Pattern Model",
            "",
            "Physical non-negative pattern:",
            "",
            "```text",
            "P_phi = sigmoid(L_phi / tau)",
            "```",
            "",
            "Binary straight-through estimator:",
            "",
            "```text",
            "P_phi = stopgrad(1[P_soft > 0.5] - P_soft) + P_soft",
            "```",
            "",
            "Centered differential measurement matrix:",
            "",
            "```text",
            "A_phi = (P_phi - mean(P_phi)) / (std(P_phi) sqrt(m))",
            "```",
            "",
            "Ghost measurement:",
            "",
            "```text",
            "y = A_phi x + epsilon",
            "```",
            "",
            "Data-consistent initialization:",
            "",
            "```text",
            "x_data = A_phi^T (A_phi A_phi^T + lambda I)^(-1) y",
            "```",
            "",
            "Null-space projection:",
            "",
            "```text",
            "P_N(v) = v - A_phi^T (A_phi A_phi^T + lambda I)^(-1) A_phi v",
            "```",
            "",
            "Measurement-consistency projection:",
            "",
            "```text",
            "Pi_y(v) = v - A_phi^T (A_phi A_phi^T + lambda I)^(-1)(A_phi v - y)",
            "```",
            "",
            "Secant-RIP proxy:",
            "",
            "```text",
            "L_secRIP = E_d (||A_phi d||_2^2 - 1)^2",
            "```",
            "",
            "## Fixed 5% vs Learned Binary 5%",
            "",
        ]
    )
    if comp.get("status") == "ok":
        lines.extend(
            markdown_table(
                [
                    {
                        "fixed_psnr": comp["fixed_psnr"],
                        "learned_psnr": comp["learned_psnr"],
                        "delta_psnr": comp["delta_psnr"],
                        "fixed_ssim": comp["fixed_ssim"],
                        "learned_ssim": comp["learned_ssim"],
                        "delta_ssim": comp["delta_ssim"],
                    }
                ],
                ["fixed_psnr", "learned_psnr", "delta_psnr", "fixed_ssim", "learned_ssim", "delta_ssim"],
            )
        )
    else:
        lines.append("missing")

    lines.extend(
        [
            "",
            "## Learned Main Results",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            main_rows,
            [
                "method",
                "sampling_ratio",
                "model_psnr",
                "model_ssim",
                "model_rel_meas_err",
                "pattern_mean",
                "mean_abs_offdiag_corr",
                "status",
            ],
        )
    )
    lines.extend(["", "## Pattern Regularization Ablation", ""])
    lines.extend(
        markdown_table(
            ablation_rows,
            [
                "method",
                "model_psnr",
                "model_ssim",
                "pattern_mean",
                "mean_abs_offdiag_corr",
                "secant_rip_eval_loss",
                "status",
            ],
        )
    )
    lines.extend(["", "## Pattern Stats", ""])
    lines.extend(
        markdown_table(
            stats_rows,
            [
                "method",
                "sampling_ratio",
                "pattern_mode",
                "pattern_mean",
                "pattern_std",
                "binary_fraction_005_095",
                "mean_abs_offdiag_corr",
                "secant_rip_eval_loss",
                "status",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            f"- Main CSV: {phase3_dir / 'phase3_main_results.csv'}",
            f"- Main Markdown: {phase3_dir / 'phase3_main_results.md'}",
            f"- Pattern ablation CSV: {phase3_dir / 'phase3_pattern_ablation_results.csv'}",
            f"- Pattern stats CSV: {phase3_dir / 'phase3_pattern_stats.csv'}",
            f"- Fixed vs learned PSNR figure: {phase3_dir / 'phase3_fixed_vs_learned_psnr.png'}",
            f"- Fixed vs learned SSIM figure: {phase3_dir / 'phase3_fixed_vs_learned_ssim.png'}",
            f"- Pattern ablation PSNR figure: {phase3_dir / 'phase3_pattern_ablation_psnr.png'}",
            f"- Learned binary pattern image: {phase3_dir / 'learned_binary_5pct' / 'eval_patterns' / 'final_patterns.png'}",
            f"- Learned binary reconstruction image: {phase3_dir / 'learned_binary_5pct' / 'eval_samples' / 'recon_grid.png'}",
            "",
            "## Current Conclusion",
            "",
            conclusion(main_rows, sanity),
            "",
            "## Current Limitations",
            "",
            "- Learned patterns are optimized on the configured STL-10 subset and short Phase 3 schedules.",
            "- Joint optimization can be sensitive to tau, lr_patterns, and the relative strength of Secant-RIP.",
            "- Missing rows in the aggregation are intentionally kept as missing and are not counted as completed experiments.",
            "",
            "## Phase 4 Suggestion",
            "",
            phase4_suggestion(main_rows, sanity),
        ]
    )
    report_path = phase3_dir / "PHASE3_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Phase 3 report to: {report_path}")


if __name__ == "__main__":
    main()

