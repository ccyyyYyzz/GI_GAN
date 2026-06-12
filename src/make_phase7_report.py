from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .utils import ensure_dir


PHASE7_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase7")
ENV_PATH = Path("E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311")
DATASET_PATH = Path("E:/ns_mc_gan_gi/data")


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


def as_float(value, default=None):
    try:
        if value in ("", None, "missing"):
            return default
        return float(value)
    except Exception:
        return default


def best_row(rows: list[dict], prefix: str) -> dict | None:
    candidates = [r for r in rows if r.get("method", "").startswith(prefix) and r.get("status") == "ok"]
    candidates = [r for r in candidates if as_float(r.get("score")) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: as_float(r["score"], -1e9))


def by_method(rows: list[dict], method: str) -> dict | None:
    for row in rows:
        if row.get("method") == method and row.get("status") == "ok":
            return row
    return None


def binary_supported(rows: list[dict]) -> bool:
    row = best_row(rows, "Flip-aware")
    if not row:
        return False
    return (
        as_float(row.get("hard_flip_fraction"), 0.0) > 0.0
        and as_float(row.get("A_rel_fro_delta"), 0.0) > 0.0
        and as_float(row.get("pattern_trainable_minus_g_only"), -1e9) > 0.02
    )


def continuous_supported(rows: list[dict], swap_rows: list[dict]) -> bool:
    physical = by_method(rows, "Continuous Physical")
    g_only = by_method(rows, "Continuous G-only")
    if not physical or not g_only:
        return False
    score_gain = as_float(physical.get("score"), -1e9) - as_float(g_only.get("score"), 1e9)
    learned_minus_initial = None
    for row in swap_rows:
        if row.get("swap_experiment") == "continuous_physical_5pct":
            if row.get("method") == "Learned G + Learned A":
                learned = as_float(row.get("score"))
            elif row.get("method") == "Learned G + Initial A":
                initial = as_float(row.get("score"))
    try:
        learned_minus_initial = learned - initial
    except Exception:
        learned_minus_initial = None
    return score_gain > 0.02 and (learned_minus_initial is None or learned_minus_initial > 0.0)


def current_conclusion(rows: list[dict], swap_rows: list[dict]) -> str:
    if binary_supported(rows):
        return "Evidence supports binary learned physical illumination."
    if continuous_supported(rows, swap_rows):
        return "Evidence supports continuous learned physical illumination."
    return (
        "Current evidence does not support learned physical illumination; use fixed operator "
        "+ fine-tuning as the main result, and report learned pattern as a negative finding."
    )


def paper_claims(rows: list[dict], swap_rows: list[dict]) -> list[dict]:
    return [
        {
            "claim": "Exact operator matching works",
            "supported": "yes",
            "evidence": "Phase 5/6 fixed exact operator and G-only controls improve reconstruction.",
            "caveat": "This is not learned illumination.",
        },
        {
            "claim": "Generator fine-tuning improves reconstruction",
            "supported": "yes",
            "evidence": "Phase 6 G-only fine-tune exceeded Phase 5 best.",
            "caveat": "Attribution is to G unless pattern controls beat it.",
        },
        {
            "claim": "Hard binary learned illumination improves reconstruction",
            "supported": "yes" if binary_supported(rows) else "no",
            "evidence": "Requires hard flips, A drift, and score over G-only.",
            "caveat": "If hard flips remain zero, binary pattern learning remains unresolved.",
        },
        {
            "claim": "Continuous physical illumination improves reconstruction",
            "supported": "yes" if continuous_supported(rows, swap_rows) else "no",
            "evidence": "Requires continuous trainable to beat continuous G-only and swap support.",
            "caveat": "Continuous is not a binary DMD claim.",
        },
    ]


def claims_table(rows: list[dict]) -> list[str]:
    cols = ["claim", "supported", "evidence", "caveat"]
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(col, "")) for col in cols) + "|")
    return lines


def main() -> None:
    phase7 = ensure_dir(PHASE7_ROOT)
    rows = read_csv(phase7 / "phase7_results.csv")
    swap_rows = read_csv(phase7 / "phase7_pattern_swap_results.csv")
    mq_rows = read_csv(phase7 / "phase7_measurement_quality.csv")
    lines = [
        "# Phase 7 Report",
        "",
        f"- Experiment report time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Clean environment path: {ENV_PATH}",
        f"- Dataset path: {DATASET_PATH}",
        f"- Phase 7 output path: {phase7}",
        "",
        "## Phase 6 Summary",
        "",
        "Hard binary pattern did not materially change; improvements were mainly generator fine-tuning.",
        "",
        "## Phase 7 Question",
        "",
        "Can flip-aware or continuous physical pattern learning produce measurable pattern changes and reconstruction gains?",
        "",
        "## Flip-aware Binary Method",
        "",
        "```text",
        "P_soft = sigmoid((L + epsilon) / tau)",
        "P_hard = TopK(P_soft, k) or 1[P_soft > t]",
        "P = stopgrad(P_hard - P_soft) + P_soft",
        "L_softflip = (mean(|P_soft - P_initial|) - delta_target)^2",
        "```",
        "",
        "## Continuous Physical Modulation Method",
        "",
        "```text",
        "P in [0, 1]",
        "A_eff = row_standardize(P - mean_row(P))",
        "L_contrast = mean_i (std(P_i) - target_contrast)^2",
        "```",
        "",
        "## Main Result Table",
        "",
    ]
    lines.extend(
        table(
            rows,
            [
                "method",
                "physical_pattern_type",
                "model_psnr",
                "model_ssim",
                "score",
                "hard_flip_fraction",
                "A_rel_fro_delta",
                "pattern_trainable_minus_g_only",
                "status",
            ],
        )
    )
    lines.extend(["", "## Flip-aware Binary Results", ""])
    lines.extend(
        table(
            [r for r in rows if r.get("method", "").startswith("Flip-aware")],
            ["method", "score", "hard_flip_fraction", "A_rel_fro_delta", "pattern_trainable_minus_g_only", "status"],
        )
    )
    lines.extend(["", "## Continuous Results", ""])
    lines.extend(
        table(
            [r for r in rows if r.get("method", "").startswith("Continuous")],
            ["method", "score", "continuous_contrast", "pattern_trainable_minus_g_only", "pattern_only_gain", "status"],
        )
    )
    lines.extend(["", "## Pattern Swap Test", ""])
    lines.extend(
        table(
            swap_rows,
            ["swap_experiment", "method", "model_psnr", "model_ssim", "score", "A_rel_fro_delta", "status"],
        )
    )
    lines.extend(["", "## Measurement-only Diagnostics", ""])
    lines.extend(
        table(
            mq_rows,
            ["method", "secant_rip_loss", "mean_abs_offdiag_corr", "gram_condition_number", "bucket_snr_proxy", "status"],
        )
    )
    claims = paper_claims(rows, swap_rows)
    lines.extend(["", "## What Can Be Claimed In A Paper", ""])
    lines.extend(claims_table(claims))
    lines.extend(
        [
            "",
            "## Current Conclusion",
            "",
            current_conclusion(rows, swap_rows),
            "",
            "## What Cannot Be Claimed",
            "",
            "- Do not claim learned binary physical illumination unless hard flips and G-only controls support it.",
            "- Do not describe continuous illumination as binary DMD learning.",
            "- Do not attribute generator-only gains to learned patterns.",
            "",
            "## Recommended Phase 8",
            "",
            "If Phase 7 supports continuous learning, expand seeds and longer schedules around continuous physical modulation. If neither branch beats G-only, pivot the paper claim to fixed measurement plus measurement-consistent fine-tuning and report learned illumination as a negative result.",
            "",
            "## Artifact Paths",
            "",
            f"- phase7_results.csv: {phase7 / 'phase7_results.csv'}",
            f"- pattern swap CSV: {phase7 / 'phase7_pattern_swap_results.csv'}",
            f"- measurement quality CSV: {phase7 / 'phase7_measurement_quality.csv'}",
            f"- paper assets: {phase7 / 'paper_assets'}",
        ]
    )
    path = phase7 / "PHASE7_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Phase 7 report to: {path}")


if __name__ == "__main__":
    main()
