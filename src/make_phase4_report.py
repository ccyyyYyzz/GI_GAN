from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


PHASE4_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase4")
PHASE3_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase3")
PHASE2_CLEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_clean_phase2")
ENV_PATH = Path("E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Phase 4 report.")
    parser.add_argument("--phase4_dir", default=str(PHASE4_ROOT))
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


def table(rows: list[dict], cols: list[str]) -> list[str]:
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    if not rows:
        lines.append("|" + "|".join(["missing"] * len(cols)) + "|")
        return lines
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    return lines


def ok_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("status") == "ok"]


def find_row(rows: list[dict], method: str) -> dict:
    for row in rows:
        if row.get("method") == method and row.get("status") == "ok":
            return row
    return {}


def best_phase4_row(rows: list[dict]) -> dict:
    candidates = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("method", "").startswith("Phase 4")
    ]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: float(row.get("score", "-inf") or "-inf"))


def conclusion(rows: list[dict]) -> str:
    fixed = find_row(rows, "Fixed Rademacher")
    phase3 = find_row(rows, "Phase 3 Binary STE")
    best = best_phase4_row(rows)
    if not fixed or not best:
        return "Phase 4 minimum report is generated, but required fixed or Phase 4 metrics are missing."
    fixed_score = float(fixed["score"])
    best_score = float(best["score"])
    fixed_psnr = float(fixed["model_psnr"])
    fixed_ssim = float(fixed["model_ssim"])
    best_psnr = float(best["model_psnr"])
    best_ssim = float(best["model_ssim"])
    if best_psnr > fixed_psnr or (
        best_ssim > fixed_ssim and best_psnr >= fixed_psnr - 0.5
    ) or best_score > fixed_score:
        return (
            "learned pattern optimization produced an improvement: "
            f"best={best['method']}, PSNR={best_psnr:.6f}, SSIM={best_ssim:.6f}, "
            f"score={best_score:.6f}."
        )
    if phase3 and best_score > float(phase3.get("score", "-inf")):
        return (
            "optimization improved learned pattern but fixed rademacher remains stronger: "
            f"best={best['method']}, score={best_score:.6f}, fixed_score={fixed_score:.6f}."
        )
    return (
        "Phase 4 did not improve over the current learned-pattern baseline. "
        "The likely causes are short schedules, sensitivity to A_eff normalization, tau, "
        "and the small STL-10 quick subset."
    )


def phase5_suggestion(rows: list[dict]) -> str:
    text = conclusion(rows)
    if text.startswith("learned pattern optimization produced an improvement"):
        return "Yes. Phase 5 can run a broader sweep, because Phase 4 found a 5% setting that challenges or beats fixed."
    return (
        "Not as a stronger-claim phase yet. Phase 5 should first be longer training, larger data, "
        "lower noise, or class-adaptive pattern tuning."
    )


def main() -> None:
    args = parse_args()
    phase4_dir = Path(args.phase4_dir)
    phase3_dir = Path(args.phase3_dir)
    phase2_dir = Path(args.phase2_clean_dir)
    tuning_rows = read_csv(phase4_dir / "phase4_tuning_results.csv")
    sweep_rows = read_csv(phase4_dir / "phase4_best_sweep_results.csv")
    stats_rows = read_csv(phase4_dir / "phase4_pattern_stats.csv")
    epoch0 = read_json(phase4_dir / "matched_binary_5pct" / "eval_epoch000_metrics.json")
    debug_sanity = read_json(phase4_dir / "debug_matched_binary_5pct" / "sanity_learnable_patterns.json")

    best = best_phase4_row(tuning_rows)
    continuous = find_row(tuning_rows, "Phase 4 Continuous Contrast")
    curriculum = find_row(tuning_rows, "Phase 4 Continuous To Binary")

    lines = [
        "# Phase 4 Report",
        "",
        f"- Experiment report time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Clean environment path: {ENV_PATH}",
        "- Dataset path: E:/ns_mc_gan_gi/data",
        f"- Phase 4 output path: {phase4_dir}",
        f"- Phase 3 output path: {phase3_dir}",
        f"- Phase 2 clean baseline path: {phase2_dir}",
        f"- sanity_learnable_patterns passed: {bool((debug_sanity or {}).get('passed', False))}",
        "",
        "## Optimization Strategy",
        "",
        "Fixed-compatible initialization:",
        "",
        "```text",
        "P_0 = 1[A_fixed_sign > 0]",
        "L_0 = alpha(2P_0 - 1)",
        "```",
        "",
        "Balanced binary STE:",
        "",
        "```text",
        "P_soft = sigmoid(L / tau)",
        "P_hard = TopK(P_soft, k = target_transmission * n)",
        "P = stopgrad(P_hard - P_soft) + P_soft",
        "```",
        "",
        "Contrast regularization:",
        "",
        "```text",
        "L_contrast = mean_i (std(P_i) - target_contrast)^2",
        "```",
        "",
        "Warm-start objective: initialize G and D from the fixed NS-MC-GAN checkpoint, then jointly fine-tune A_phi and G.",
        "",
        "## Epoch 0 Analysis",
        "",
    ]
    if epoch0:
        model = epoch0.get("model", {})
        lines.extend(
            [
                f"- matched_binary_5pct epoch0 PSNR: {fmt(model.get('psnr'))}",
                f"- matched_binary_5pct epoch0 SSIM: {fmt(model.get('ssim'))}",
                "- If epoch0 is not close to fixed, likely A_eff normalization is not fully identical to fixed A.",
            ]
        )
    else:
        lines.append("- matched_binary_5pct epoch0 metrics missing.")

    lines.extend(["", "## 5% Tuning Results", ""])
    lines.extend(
        table(
            tuning_rows,
            ["method", "model_psnr", "model_ssim", "score", "pattern_std", "row_std_mean", "status"],
        )
    )
    lines.extend(["", "## Fixed Vs Phase 3 Vs Phase 4", ""])
    lines.extend(
        table(
            [
                row
                for row in tuning_rows
                if row.get("method") in {"Fixed Rademacher", "Phase 3 Binary STE", "Phase 3 Continuous"}
                or row == best
            ],
            ["method", "model_psnr", "model_ssim", "score", "status"],
        )
    )
    lines.extend(
        [
            "",
            "## Continuous Contrast",
            "",
            (
                f"- Continuous contrast row_std_mean: {fmt(continuous.get('row_std_mean'))}; "
                f"PSNR: {fmt(continuous.get('model_psnr'))}; SSIM: {fmt(continuous.get('model_ssim'))}."
                if continuous
                else "- Continuous contrast experiment missing."
            ),
            "",
            "## Continuous To Binary",
            "",
            (
                f"- Curriculum PSNR: {fmt(curriculum.get('model_psnr'))}; "
                f"SSIM: {fmt(curriculum.get('model_ssim'))}; score: {fmt(curriculum.get('score'))}."
                if curriculum
                else "- Continuous-to-binary curriculum missing."
            ),
            "",
            "## Best Sweep",
            "",
        ]
    )
    lines.extend(
        table(sweep_rows, ["method", "sampling_ratio", "model_psnr", "model_ssim", "score", "status"])
    )
    lines.extend(["", "## Pattern Stats", ""])
    lines.extend(
        table(
            stats_rows,
            [
                "method",
                "pattern_mode",
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
            f"- Tuning CSV: {phase4_dir / 'phase4_tuning_results.csv'}",
            f"- Pattern stats CSV: {phase4_dir / 'phase4_pattern_stats.csv'}",
            f"- Best sweep CSV: {phase4_dir / 'phase4_best_sweep_results.csv'}",
            f"- 5% comparison figure: {phase4_dir / 'phase4_fixed_vs_phase3_vs_phase4_5pct.png'}",
        ]
    )
    if best:
        lines.extend(
            [
                f"- Best checkpoint: {best.get('checkpoint_best_score') or best.get('checkpoint_best_ssim')}",
                f"- Pattern image: {best.get('pattern_image')}",
                f"- Reconstruction image: {best.get('sample_image')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Current Conclusion",
            "",
            conclusion(tuning_rows),
            "",
            "## Current Limitations",
            "",
            "- The current runs use STL-10 quick subsets and short schedules.",
            "- Fixed-compatible initialization matches sign structure, but learned A_eff row standardization is not identical to the fixed operator scale.",
            "- Missing experiments are preserved as missing in aggregation.",
            "",
            "## Phase 5 Suggestion",
            "",
            phase5_suggestion(tuning_rows),
        ]
    )
    out = phase4_dir / "PHASE4_REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Phase 4 report to: {out}")


if __name__ == "__main__":
    main()
