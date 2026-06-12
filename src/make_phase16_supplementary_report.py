from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .phase15_common import read_csv, read_json, write_json
from .phase16_common import PHASE16, as_float, copy_to_legacy, ensure_dir, registry_rows


REPORT_DIR = PHASE16 / "_report"
REPORT = REPORT_DIR / "PHASE16_SUPPLEMENTARY_REPORT.md"


def fmt(value: Any, digits: int = 3) -> str:
    val = as_float(value)
    if val == val:
        return f"{val:.{digits}f}"
    if value is None:
        return ""
    return str(value)


def load_rows(rel: str) -> list[dict[str, str]]:
    return read_csv(PHASE16 / rel)


def compact_table(rows: list[dict[str, Any]], fields: list[str], limit: int | None = None) -> list[str]:
    shown = rows if limit is None else rows[:limit]
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in shown:
        lines.append("|" + "|".join(fmt(row.get(field, "")) for field in fields) + "|")
    if limit is not None and len(rows) > limit:
        lines.append("|...|" + "|".join([""] * (len(fields) - 1)) + "|")
    return lines


def status_value(rows: list[dict[str, str]], key: str) -> str:
    for row in rows:
        if row.get("check") == key:
            return row.get("status", "")
    return ""


def best_tv(rows: list[dict[str, str]], method_id: str) -> dict[str, str] | None:
    sub = [r for r in rows if r.get("method_id") == method_id and r.get("baseline") == "tv_pgd"]
    if not sub:
        return None
    return max(sub, key=lambda r: as_float(r.get("psnr")))


def method_short(method_id: str) -> str:
    return (
        method_id.replace("_hq_noise001_colab", "")
        .replace("_full_noise001_colab", "")
        .replace("_full_colab", "")
        .replace("scrambled_hadamard", "scrambled")
        .replace("rademacher", "rad")
    )


def main() -> None:
    ensure_dir(REPORT_DIR)
    registry = registry_rows()
    safe = read_json(PHASE16 / "safe_exactA_verification" / "safe_exactA_verification.json")
    exact_audit = load_rows("exactA_reeval/exactA_reeval_results.csv")
    attribution = load_rows("attribution/attribution_final.csv")
    ablation = load_rows("inference_ablation/real_inference_ablation_results.csv")
    noise = load_rows("noise_sweep/noise_sweep_results.csv")
    baselines = load_rows("traditional_baselines/tv_pgd_baseline_results.csv")
    dc = load_rows("dc_row_control/dc_row_final.csv")
    stats = load_rows("statistics/statistics_ci.csv")
    classwise = load_rows("classwise/classwise_stl10_metrics.csv")
    perturb = load_rows("measurement_perturbation/measurement_perturbation.csv")
    runtime = load_rows("runtime_complexity/runtime_complexity.csv")
    aggregate = load_rows("_aggregate/phase16_experiment_status.csv")

    registry_main = [
        {
            "method": method_short(r.get("method_id", "")),
            "psnr": r.get("psnr", ""),
            "ssim": r.get("ssim", ""),
            "bp_psnr": r.get("backproj_psnr", ""),
            "delta_psnr": r.get("delta_psnr", ""),
        }
        for r in registry
    ]

    rademacher_audit_ok = exact_audit and all(r.get("status") in {"reproduced", "within_tolerance"} for r in exact_audit)
    safe_ok = safe.get("status") == "pass" if isinstance(safe, dict) else False
    completed = [r for r in aggregate if r.get("status") == "completed"]
    missing = [r for r in aggregate if r.get("status") != "completed"]

    lines: list[str] = [
        "# Phase16 supplementary reviewer-defense report",
        "",
        "This report uses imported strict no-leak checkpoints and local re-evaluation only. No new training is performed here.",
        "",
        "## Executive summary",
        "",
        f"- Safe exact-A verification: {'pass' if safe_ok else 'needs attention'}.",
        f"- Rademacher exact-A re-evaluation: {'reproduced' if rademacher_audit_ok else 'needs attention'}.",
        f"- Supplementary experiments completed: {len(completed)}/{len(aggregate)}.",
        f"- Missing or skipped modules: {', '.join(r.get('experiment', '') for r in missing) if missing else 'none'}.",
        "",
        "## Main no-leak result snapshot",
        "",
    ]
    lines.extend(compact_table(registry_main, ["method", "psnr", "ssim", "bp_psnr", "delta_psnr"]))

    lines.extend(
        [
            "",
            "## Seventeen direct answers",
            "",
            "### 1. Is the no-leak evaluation path valid?",
            "",
            f"Yes, subject to the safe exact-A audit. The verification result is `{safe_ok}` and records exact-A loading, solver-cache rebuilding, and file scan status in `safe_exactA_verification.json`.",
            "",
            "### 2. Why did the old Rademacher checkpoint re-evaluation disagree?",
            "",
            "The earlier mismatch came from replacing `A/K` without rebuilding the cached normal-equation solver. Phase15R added a safe override path that rebuilds the Cholesky cache after exact-A injection. Phase16 reuses that path.",
            "",
            "### 3. Do Rademacher results still count?",
            "",
            f"{'Yes' if rademacher_audit_ok else 'Not yet'}: exact-A re-evaluation rows are listed below. They should be cited only with the safe exact-A cache-rebuilt code path.",
            "",
        ]
    )
    lines.extend(compact_table(exact_audit, ["method_id", "original_psnr", "reeval_psnr", "abs_diff_psnr", "status"]))

    lines.extend(
        [
            "",
            "### 4. Is performance mainly backprojection or learned model?",
            "",
            "The attribution table separates physical backprojection from the learned reconstruction. Positive `delta_psnr` and `delta_ssim` mean the learned inverse adds information beyond the initialization.",
            "",
        ]
    )
    lines.extend(compact_table(attribution, ["method_id", "backproj_psnr", "model_psnr", "delta_psnr", "classification"], limit=10))

    lines.extend(
        [
            "",
            "### 5. Rademacher vs scrambled Hadamard under the same protocol",
            "",
            "Rademacher has much weaker backprojection but reaches a similar final reconstruction level after learned refinement; scrambled Hadamard starts from a stronger structured initialization. This supports reporting both as complementary sensing regimes, not as interchangeable baselines.",
            "",
            "### 6. Which inference components matter?",
            "",
            "The ablation table compares DC projection, null-space projection, refiner, raw weights, and EMA weights under real checkpoint inference.",
            "",
        ]
    )
    lines.extend(compact_table(ablation, ["method_id", "ablation_mode", "psnr", "ssim", "delta_vs_full_psnr", "status"], limit=20))

    lines.extend(
        [
            "",
            "### 7. Noise robustness",
            "",
            "The noise sweep evaluates the same checkpoints at multiple measurement-noise levels. It is a robustness check, not a retraining result.",
            "",
        ]
    )
    lines.extend(compact_table(noise, ["method_id", "noise_std", "psnr", "ssim", "rel_meas_err"], limit=20))

    lines.extend(
        [
            "",
            "### 8. Traditional baselines",
            "",
            "Backprojection/adjoint are full-set linear controls. TV-PGD is included as a small-subset iterative baseline, because it is slow and is intended here as reviewer-defense evidence.",
            "",
        ]
    )
    base_comp = []
    for row in registry:
        method_id = row.get("method_id", "")
        tv = best_tv(baselines, method_id)
        if tv:
            base_comp.append({"method_id": method_id, "model_psnr": row.get("psnr", ""), "best_tv_psnr": tv.get("psnr", ""), "best_tv_lambda": tv.get("lambda_tv", "")})
    lines.extend(compact_table(base_comp, ["method_id", "model_psnr", "best_tv_psnr", "best_tv_lambda"], limit=12))

    lines.extend(
        [
            "",
            "### 9. DC-row control",
            "",
            "The low-frequency Hadamard DC-row control checks whether including the DC row alone explains reconstruction quality. It is a diagnostic around measurement design rather than a primary model result.",
            "",
        ]
    )
    lines.extend(compact_table(dc, ["sampling_ratio", "hadamard_include_dc", "hadamard_skip_dc", "backproj_psnr", "backproj_ssim"], limit=10))

    lines.extend(
        [
            "",
            "### 10. Confidence intervals and distributions",
            "",
            "The statistics module reports per-sample PSNR/SSIM distributions and bootstrap confidence intervals for the no-leak checkpoints.",
            "",
        ]
    )
    lines.extend(compact_table(stats, ["method_id", "mean_psnr", "ci95_psnr_low", "ci95_psnr_high", "mean_ssim", "ci95_ssim_low", "ci95_ssim_high"]))

    lines.extend(
        [
            "",
            "### 11. STL-10 class-wise behavior",
            "",
            "Class-wise results are diagnostic only. They identify whether a method is consistently weak on specific categories, but they should not be over-interpreted without more samples per class.",
            "",
        ]
    )
    lines.extend(compact_table(classwise, ["method_id", "class_id", "class_name", "mean_psnr", "mean_ssim", "num_samples"], limit=20))

    lines.extend(
        [
            "",
            "### 12. Measurement perturbation",
            "",
            "Perturbing measurements should reduce quality; a large drop for shuffled or wrong-sample measurements is evidence that the model uses the measurements rather than generating generic images.",
            "",
        ]
    )
    lines.extend(compact_table(perturb, ["method_id", "perturbation_mode", "psnr", "psnr_drop_from_normal", "rel_meas_err"], limit=24))

    lines.extend(
        [
            "",
            "### 13. Runtime and complexity",
            "",
            "Runtime is measured locally on a subset and should be reported as approximate hardware-specific evidence.",
            "",
        ]
    )
    lines.extend(compact_table(runtime, ["method_id", "path", "runtime_sec_per_image", "model_params_m", "peak_cuda_mem_mb"], limit=20))

    lines.extend(
        [
            "",
            "### 14. Are MNIST/Fashion results useful?",
            "",
            "Yes, as simple-domain sanity checks. They should not be the paper's main novelty, but they show the pipeline works in easy regimes.",
            "",
            "### 15. What claims are supported?",
            "",
            "Supported: strict no-leak evaluation, exact-A Rademacher re-evaluation with cache rebuild, learned refinement over backprojection, robustness/noise diagnostics, and baseline comparisons. Use precise wording and cite the generated tables.",
            "",
            "### 16. What should not be claimed?",
            "",
            "Do not claim SOTA, universal robustness, or that low-frequency 5% Hadamard is high-quality on STL-10. TV-PGD rows are small-subset controls, not exhaustive optimized baselines.",
            "",
            "### 17. Can we stop here?",
            "",
            "If all aggregate rows are completed and the safe exact-A/reeval audits pass, this is enough for a reviewer-defense supplement. Further work should be driven by a specific reviewer concern or a writing gap, not by fear of missing another run.",
            "",
            "## Output map",
            "",
            f"- Phase16 root: `{PHASE16}`",
            "- Aggregated status: `_aggregate/phase16_experiment_status.csv`",
            "- File manifest: `_aggregate/phase16_file_manifest.csv`",
            "- Final supported-claims sheet: `_report/PHASE16_SUPPORTED_CLAIMS.md` after running the claims updater",
            "",
        ]
    )

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    write_json(REPORT_DIR / "PHASE16_SUPPLEMENTARY_REPORT_META.json", {"safe_ok": safe_ok, "rademacher_audit_ok": rademacher_audit_ok, "completed": len(completed), "total": len(aggregate)})
    legacy = copy_to_legacy(REPORT)
    print(json.dumps({"report": str(REPORT), "legacy_copy": str(legacy)}, indent=2))


if __name__ == "__main__":
    main()
