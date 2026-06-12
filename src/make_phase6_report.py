from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .utils import ensure_dir


PHASE6_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase6")
ENV_PATH = Path("E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311")
DATASET_PATH = Path("E:/ns_mc_gan_gi/data")
PHASE5_BEST = Path("E:/ns_mc_gan_gi/outputs_phase5/exact_binary_slow_5pct/eval_metrics.json")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value) -> str:
    if value in ("", None):
        return "missing"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def table(rows: list[dict], cols: list[str]) -> list[str]:
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    return lines


def by_method(rows: list[dict], name: str) -> dict | None:
    for row in rows:
        if row.get("method") == name and row.get("status") == "ok":
            return row
    return None


def best_pattern_row(rows: list[dict]) -> dict | None:
    candidates = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("method", "").startswith("Pattern Trainable")
        and row.get("score") not in ("", None)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["score"]))


def paired_for(paired: list[dict], method_a: str, method_b: str) -> dict | None:
    for row in paired:
        if row.get("method_a") == method_a and row.get("method_b") == method_b:
            return row
    return None


def conclusion(rows: list[dict], paired: list[dict]) -> str:
    g_only = by_method(rows, "G-only Fine-tune")
    best_pattern = best_pattern_row(rows)
    if not g_only or not best_pattern:
        return "Phase 6 minimum controls are incomplete; attribution remains unresolved."
    g_score = float(g_only.get("score", 0.0))
    p_score = float(best_pattern.get("score", 0.0))
    hard_flip = float(best_pattern.get("hard_flip_fraction") or 0.0)
    pair = paired_for(paired, "G-only Fine-tune", best_pattern["method"])
    ci_supports = False
    if pair and pair.get("status") == "ok":
        try:
            ci_supports = float(pair.get("score_ci_low", 0.0)) > 0.0
        except Exception:
            ci_supports = False
    if p_score > g_score and hard_flip > 0.0 and ci_supports:
        return "Evidence supports learned physical illumination."
    if abs(p_score - g_score) < 0.05 or hard_flip <= 1e-6:
        return "Current improvement is mainly attributable to generator fine-tuning; binary pattern did not materially change."
    return "Pattern-trainable differs from G-only, but evidence is not yet strong enough for a paper claim."


def phase7_suggestion(rows: list[dict], paired: list[dict]) -> str:
    text = conclusion(rows, paired)
    if text.startswith("Evidence supports"):
        return "Yes. Phase 7 should broaden seeds and longer schedules for paper-ready confirmation."
    if "generator fine-tuning" in text:
        return "Yes, but Phase 7 should focus on soft relaxations, flip-aware regularization, or hardware-realistic continuous modulation."
    return "Not yet. Finish minimum controls and paired bootstrap first."


def main() -> None:
    phase6 = ensure_dir(PHASE6_ROOT)
    rows = read_csv(phase6 / "phase6_control_results.csv")
    diag_rows = read_csv(phase6 / "phase6_pattern_diagnostics.csv")
    paired = read_csv(phase6 / "paired_5pct" / "paired_summary.csv")
    best_pattern = best_pattern_row(rows)
    g_only = by_method(rows, "G-only Fine-tune")
    pattern_only = by_method(rows, "Pattern-only Alpha1")

    lines = [
        "# Phase 6 Report",
        "",
        f"- Experiment report time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Clean environment path: {ENV_PATH}",
        f"- Dataset path: {DATASET_PATH}",
        f"- Phase 6 output path: {phase6}",
        f"- Phase 5 best metrics path: {PHASE5_BEST}",
        "",
        "## Core Question",
        "",
        "Does the improvement come from learned physical illumination or generator fine-tuning?",
        "",
        "## Pattern Causality Audit Method",
        "",
        "```text",
        "flip = mean(1[P_final_hard != P_initial_hard])",
        "Delta_A = ||A_final - A_initial||_F / ||A_initial||_F",
        "G-only control: pattern frozen, G/D trainable",
        "Pattern-trainable control: pattern, G, D all trainable",
        "Pattern-only control: G frozen, pattern trainable",
        "",
        "If pattern-trainable improves over G-only and pattern drift > 0:",
        "  evidence supports learned physical illumination.",
        "If pattern-trainable improves over fixed but not over G-only:",
        "  improvement is likely generator fine-tuning.",
        "If hard_flip_fraction = 0 and A_rel_fro_delta = 0:",
        "  no hard physical pattern change; cannot claim binary illumination learning.",
        "```",
        "",
        "## 5% Control Table",
        "",
    ]
    lines.extend(
        table(
            rows,
            [
                "method",
                "model_psnr",
                "model_ssim",
                "score",
                "hard_flip_fraction",
                "A_rel_fro_delta",
                "status",
            ],
        )
    )
    lines.extend(["", "## G-only vs Pattern-trainable", ""])
    lines.extend(
        table(
            [row for row in [g_only, best_pattern] if row],
            ["method", "score", "hard_flip_fraction", "A_rel_fro_delta", "attribution_note"],
        )
    )
    lines.extend(["", "## Pattern-only", ""])
    lines.extend(
        table(
            [pattern_only] if pattern_only else [{"method": "Pattern-only Alpha1", "status": "missing"}],
            ["method", "model_psnr", "model_ssim", "score", "hard_flip_fraction", "status"],
        )
    )
    lines.extend(["", "## Pattern Flip / A Drift / Secant-RIP Drift", ""])
    lines.extend(
        table(
            diag_rows,
            ["method", "hard_flip_fraction", "A_rel_fro_delta", "secant_rip_delta", "offdiag_corr_delta", "status"],
        )
    )
    lines.extend(["", "## Paired Bootstrap", ""])
    if paired:
        lines.extend(
            table(
                paired,
                [
                    "method_a",
                    "method_b",
                    "score_mean_delta",
                    "score_ci_low",
                    "score_ci_high",
                    "score_p_gt_0",
                    "status",
                ],
            )
        )
    else:
        lines.append("Paired bootstrap not completed.")

    lines.extend(
        [
            "",
            "## Paper Assets",
            "",
            f"- Figure/table output directory: {phase6 / 'paper_assets'}",
            f"- Control LaTeX table: {phase6 / 'phase6_latex_table_controls.tex'}",
            f"- Paired LaTeX table: {phase6 / 'phase6_latex_table_paired.tex'}",
            "",
            "## Artifact Paths",
            "",
            f"- Control CSV: {phase6 / 'phase6_control_results.csv'}",
            f"- Pattern diagnostics CSV: {phase6 / 'phase6_pattern_diagnostics.csv'}",
            f"- Paired summary: {phase6 / 'phase6_paired_summary.md'}",
            f"- PHASE6_REPORT.md: {phase6 / 'PHASE6_REPORT.md'}",
            "",
            "## Current Conclusion",
            "",
            conclusion(rows, paired),
            "",
            "## Current Limitations",
            "",
            "- Runs use quick STL-10 subsets and short schedules.",
            "- Phase 6 attribution depends on hard pattern drift and paired CIs, not only aggregate score.",
            "- Missing optional long-run or multi-seed experiments remain marked missing.",
            "",
            "## Phase 7 Suggestion",
            "",
            phase7_suggestion(rows, paired),
        ]
    )
    path = phase6 / "PHASE6_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Phase 6 report to: {path}")


if __name__ == "__main__":
    main()
