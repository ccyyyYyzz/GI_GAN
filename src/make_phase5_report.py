from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from .utils import ensure_dir


PHASE5_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase5")
PHASE4_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase4")
PHASE2_CLEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_clean_phase2")
ENV_PATH = Path("E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311")
DATASET_PATH = Path("E:/ns_mc_gan_gi/data")
FIXED_5_SCORE = 22.315117
FIXED_5_PSNR = 18.287900
FIXED_5_SSIM = 0.402722
PHASE4_BEST_SCORE = 22.789400


def parse_args():
    parser = argparse.ArgumentParser(description="Write Phase 5 report.")
    parser.add_argument("--phase5_dir", default=str(PHASE5_ROOT))
    parser.add_argument("--phase4_dir", default=str(PHASE4_ROOT))
    parser.add_argument("--phase2_clean_dir", default=str(PHASE2_CLEAN_ROOT))
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def md_table(rows: list[dict], cols: list[str]) -> list[str]:
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    return lines


def row_from_eval(method: str, sampling_ratio: float, run_dir: Path) -> dict:
    metrics = read_json(run_dir / "eval_metrics.json")
    if not metrics:
        return {"method": method, "sampling_ratio": sampling_ratio, "status": "missing"}
    model = metrics.get("model", {})
    psnr = model.get("psnr", "")
    ssim = model.get("ssim", "")
    score = ""
    if psnr != "" and ssim != "":
        score = float(psnr) + 10.0 * float(ssim)
    return {
        "method": method,
        "sampling_ratio": sampling_ratio,
        "model_psnr": psnr,
        "model_ssim": ssim,
        "score": score,
        "status": "ok",
    }


def best_phase5_row(rows: list[dict]) -> dict | None:
    ok = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("method", "").startswith("Phase 5")
        and row.get("score") not in ("", None)
    ]
    if not ok:
        return None
    return max(ok, key=lambda row: float(row["score"]))


def current_conclusion(calibration: dict, best_row: dict | None) -> str:
    exact_solved = (
        calibration.get("status") == "exact_match_passed"
        and float(calibration.get("A_rel_fro_error", 1.0)) < 1e-6
    )
    if best_row is None:
        if exact_solved:
            return "operator mismatch solved, but Phase 5 training results are not complete yet."
        return "exact matching is not yet verified; fixed rademacher remains the reference."
    score = float(best_row.get("score", 0.0))
    if score > FIXED_5_SCORE:
        return "learned physical illumination shows benefit under calibrated operator."
    if exact_solved:
        return "exact matching improved fairness but not final performance."
    return "fixed rademacher remains strong under current setting."


def phase6_suggestion(calibration: dict, best_row: dict | None, noise_done: bool) -> str:
    exact_solved = calibration.get("status") == "exact_match_passed"
    if best_row and float(best_row.get("score", 0.0)) > PHASE4_BEST_SCORE and noise_done:
        return "Yes. Phase 6 can focus on paper-ready broader robustness and longer schedules."
    if exact_solved:
        return "Yes, but frame Phase 6 around why exact matching changes performance versus centered preconditioning."
    return "Not yet. First resolve exact operator mismatch."


def main() -> None:
    args = parse_args()
    phase5_dir = ensure_dir(args.phase5_dir)
    calibration = read_json(phase5_dir / "operator_calibration_5pct.json") or {}
    tuning_rows = read_csv(phase5_dir / "phase5_tuning_results.csv")
    sweep_rows = read_csv(phase5_dir / "phase5_best_sweep_results.csv")
    extreme_rows = [
        row_from_eval("Fixed Rademacher 0.5%", 0.005, phase5_dir / "fixed_0p5pct"),
        row_from_eval("Phase 5 Exact 0.5%", 0.005, phase5_dir / "extreme_0p5pct"),
    ]
    pattern_rows = read_csv(phase5_dir / "phase5_pattern_stats.csv")
    noise_summary = phase5_dir / "phase5_noise_sweep_summary.md"
    noise_done = noise_summary.exists() and "No noise sweep" not in noise_summary.read_text(
        encoding="utf-8"
    )
    best_row = best_phase5_row(tuning_rows)

    best_checkpoint = best_row.get("checkpoint_best_score", "") if best_row else ""
    pattern_image = best_row.get("pattern_image", "") if best_row else ""
    sample_image = best_row.get("sample_image", "") if best_row else ""

    lines = [
        "# Phase 5 Report",
        "",
        f"- Experiment report time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Clean environment path: {ENV_PATH}",
        f"- Dataset path: {DATASET_PATH}",
        f"- Phase 5 output path: {phase5_dir}",
        f"- Phase 4 output path: {args.phase4_dir}",
        f"- Phase 2 clean baseline path: {args.phase2_clean_dir}",
        f"- Fixed 5% baseline: PSNR={FIXED_5_PSNR:.6f}, SSIM={FIXED_5_SSIM:.6f}, score={FIXED_5_SCORE:.6f}",
        f"- Phase 4 best score: {PHASE4_BEST_SCORE:.6f}",
        "",
        "## Core Problem",
        "",
        "Phase 4 fixed-compatible initialization did not make epoch0 match the fixed baseline, so Phase 5 tests whether exact fixed-operator matching removes that fairness issue.",
        "",
        "## Exact Operator Matching",
        "",
        "```text",
        "A_fixed in { -c, +c }^(m x n)",
        "P_0 = 1[A_fixed > 0]",
        "L_0 = alpha(2P_0 - 1)",
        "P_phi = stopgrad(P_hard - P_soft) + P_soft",
        "A_phi = c(2P_phi - 1)",
        "```",
        "",
        "## Operator Calibration",
        "",
        f"- status: {calibration.get('status', 'missing')}",
        f"- A_rel_fro_error: {fmt(calibration.get('A_rel_fro_error'))}",
        f"- A_max_abs_error: {fmt(calibration.get('A_max_abs_error'))}",
        f"- A_cosine: {fmt(calibration.get('A_cosine'))}",
        f"- y_rel_error: {fmt(calibration.get('y_rel_error'))}",
        f"- x_data_rel_error: {fmt(calibration.get('x_data_rel_error'))}",
    ]
    generator = calibration.get("generator", {})
    if generator:
        lines.extend(
            [
                f"- epoch0 psnr_fixed: {fmt(generator.get('psnr_fixed'))}",
                f"- epoch0 ssim_fixed: {fmt(generator.get('ssim_fixed'))}",
                f"- epoch0 psnr_learned: {fmt(generator.get('psnr_learned'))}",
                f"- epoch0 ssim_learned: {fmt(generator.get('ssim_learned'))}",
            ]
        )

    lines.extend(["", "## Phase 5 Tuning", ""])
    lines.extend(
        md_table(
            tuning_rows,
            [
                "method",
                "model_psnr",
                "model_ssim",
                "score",
                "epoch0_psnr",
                "A_rel_fro_error",
                "status",
            ],
        )
    )

    fixed_phase4_phase5 = [
        row
        for row in tuning_rows
        if row.get("method")
        in {"Fixed Rademacher", "Phase 4 Best", (best_row or {}).get("method", "")}
    ]
    lines.extend(["", "## Fixed Vs Phase 4 Vs Phase 5", ""])
    lines.extend(md_table(fixed_phase4_phase5, ["method", "model_psnr", "model_ssim", "score", "status"]))

    lines.extend(["", "## Best Sweep", ""])
    lines.extend(md_table(sweep_rows, ["method", "sampling_ratio", "model_psnr", "model_ssim", "score", "status"]))

    lines.extend(["", "## Extreme 0.5% Sampling", ""])
    lines.extend(md_table(extreme_rows, ["method", "sampling_ratio", "model_psnr", "model_ssim", "score", "status"]))

    lines.extend(["", "## Noise Sweep", ""])
    if noise_summary.exists():
        lines.append(f"- Summary path: {noise_summary}")
    else:
        lines.append("- Noise sweep not completed.")

    lines.extend(["", "## Pattern Stats", ""])
    lines.extend(
        md_table(
            pattern_rows,
            [
                "method",
                "effective_A_mode",
                "pattern_mean",
                "pattern_std",
                "row_std_mean",
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
            f"- Calibration JSON: {phase5_dir / 'operator_calibration_5pct.json'}",
            f"- Calibration Markdown: {phase5_dir / 'operator_calibration_5pct.md'}",
            f"- Tuning CSV: {phase5_dir / 'phase5_tuning_results.csv'}",
            f"- Epoch0 equivalence CSV: {phase5_dir / 'phase5_epoch0_equivalence.csv'}",
            f"- Pattern stats CSV: {phase5_dir / 'phase5_pattern_stats.csv'}",
            f"- Best sweep CSV: {phase5_dir / 'phase5_best_sweep_results.csv'}",
            f"- Best checkpoint: {best_checkpoint or 'missing'}",
            f"- Pattern image: {pattern_image or 'missing'}",
            f"- Reconstruction image: {sample_image or 'missing'}",
            "",
            "## Current Conclusion",
            "",
            current_conclusion(calibration, best_row),
            "",
            "## Current Limitations",
            "",
            "- Runs use STL-10 quick subsets and short schedules.",
            "- Exact matching can solve operator fairness while still changing optimization behavior.",
            "- Missing experiments remain marked as missing in aggregation.",
            "",
            "## Phase 6 Suggestion",
            "",
            phase6_suggestion(calibration, best_row, noise_done),
        ]
    )

    report_path = phase5_dir / "PHASE5_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Phase 5 report to: {report_path}")


if __name__ == "__main__":
    main()
