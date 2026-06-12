from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .phase11_common import ROOT9, ROOT10, ROOT11, as_float, read_csv_rows, read_json, write_md_table
from .utils import ensure_dir


def status_counts() -> dict[str, int]:
    status = read_json(ROOT10 / "overnight_status.json")
    counts: dict[str, int] = {}
    for task in status.get("tasks", []):
        key = str(task.get("status", "missing"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def task_lines() -> list[str]:
    status = read_json(ROOT10 / "overnight_status.json")
    tasks = status.get("tasks", [])
    if not tasks:
        return ["- overnight status missing"]
    return [f"- {t.get('task_name')}: {t.get('status')} rc={t.get('return_code')}" for t in tasks]


def best_row(rows: list[dict]) -> dict | None:
    completed = [row for row in rows if row.get("status") == "completed" and as_float(row.get("hq_score")) is not None]
    return max(completed, key=lambda row: float(row["hq_score"]), default=None)


def rows_for(rows: list[dict], names: list[str]) -> list[dict]:
    return [row for row in rows if row.get("method") in names]


def threshold_bool(rows: list[dict], key: str) -> bool:
    return any(row.get("status") == "completed" and str(row.get(key)).lower() == "true" for row in rows)


def closest_gap(rows: list[dict]) -> list[str]:
    candidates = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        dataset = row.get("dataset_name")
        ratio = as_float(row.get("sampling_ratio")) or 0.0
        psnr = as_float(row.get("model_psnr"))
        ssim = as_float(row.get("model_ssim"))
        if psnr is None or ssim is None:
            continue
        if dataset == "stl10" and ratio >= 0.099:
            candidates.append((abs(22.0 - psnr) + abs(0.65 - ssim) * 20.0, row, "STL-10 10%"))
        if dataset == "stl10" and ratio <= 0.051:
            candidates.append((abs(20.0 - psnr) + abs(0.60 - ssim) * 20.0, row, "STL-10 5%"))
        if dataset in {"mnist", "fashion_mnist"}:
            candidates.append((abs(25.0 - psnr) + abs(0.80 - ssim) * 20.0, row, "simple-domain 5%"))
    if not candidates:
        return ["- closest_config: missing", "- gap: unknown", "- bottleneck: no completed long-run metrics"]
    _, row, target = min(candidates, key=lambda item: item[0])
    return [
        f"- closest_config: {row.get('method')} ({target})",
        f"- psnr: {row.get('model_psnr')}",
        f"- ssim: {row.get('model_ssim')}",
        "- bottleneck: compare threshold gaps and convergence curves before claiming HQ.",
    ]


def conclusions(rows: list[dict]) -> tuple[list[str], list[str]]:
    supported = [
        "Phase 9 calibration supports DC-included low-frequency orthogonal Hadamard as a strong physically meaningful initialization.",
        "Random Rademacher 10% backprojection is much weaker than low-frequency Hadamard in the Phase 9 reference.",
    ]
    unsupported = [
        "Do not claim STL-10 5% high-quality unless a completed 5% row reaches PSNR >= 20 and SSIM >= 0.60.",
        "Do not call Phase 9 short_train evidence full training.",
        "Do not claim network-only quality dominance when attribution is missing or backprojection dominated.",
    ]
    if threshold_bool(rows, "reaches_stl10_10pct_hq"):
        supported.append("STL-10 64x64 10% lowfreq Hadamard achieves the internal HQ threshold under full/near-full training.")
    else:
        unsupported.append("Only short-train or missing evidence exists for STL-10 10% full HQ until hadamard10_full_noise001 completes and passes.")
    if threshold_bool(rows, "reaches_stl10_5pct_hq"):
        supported.append("STL-10 64x64 5% lowfreq Hadamard reaches the internal HQ threshold.")
    if threshold_bool(rows, "reaches_simple_domain_hq"):
        supported.append("High-quality reconstruction is supported for structured simple targets at 5%.")
    return supported, unsupported


def main() -> None:
    ensure_dir(ROOT11)
    rows = read_csv_rows(ROOT11 / "phase11_summary.csv")
    attribution = read_csv_rows(ROOT11 / "attribution_results.csv")
    noise = read_csv_rows(ROOT11 / "noise_sweep_summary.csv")
    multiseed = read_csv_rows(ROOT11 / "multiseed_summary.csv")
    best = best_row(rows)
    supported, unsupported = conclusions(rows)
    counts = status_counts()
    completed_long = [row for row in rows if row.get("status") == "completed" and str(row.get("phase")) in {"phase10", "phase11"}]
    lines = [
        "# Phase 11 Report",
        "",
        "## 1. Experiment Date Time",
        datetime.now().isoformat(timespec="seconds"),
        "",
        "## 2. Environment Paths",
        "- conda_env: E:/ns_mc_gan_gi/conda_envs/ns_mc_gan_gi_py311",
        "- phase10_outputs: E:/ns_mc_gan_gi/outputs_phase10",
        "- phase11_outputs: E:/ns_mc_gan_gi/outputs_phase11",
        "",
        "## 3. Dataset Path",
        "E:/ns_mc_gan_gi/data",
        "",
        "## 4. Phase 9 Calibration Summary",
        "- full sampling rel_error: 4.325507e-07",
        "- 10% lowfreq Hadamard include DC backprojection: PSNR 21.8253 / SSIM 0.6420",
        "- 10% skip DC backprojection: PSNR 6.7351 / SSIM 0.1498",
        "- 10% Rademacher ridge backprojection: PSNR 6.0037 / SSIM 0.0337",
        "",
        "## 5. Phase 10 Overnight Status",
        f"- completed: {counts.get('completed', 0)}",
        f"- failed: {counts.get('failed', 0)}",
        f"- missing/pending: {counts.get('pending', 0)}",
        f"- skipped_existing: {counts.get('skipped_existing', 0)}",
        *task_lines(),
        "",
        "## 6. Completed Long Training Runs",
        *(write_table_inline(completed_long, ["phase", "method", "run_scale", "epochs_actual", "model_psnr", "model_ssim", "status"]) if completed_long else ["No completed Phase 10/11 long training rows yet."]),
        "",
        "## 7. Hadamard10 Full Noise001 Result",
        *(write_table_inline(rows_for(rows, ["hadamard10_full_noise001"]), ["method", "model_psnr", "model_ssim", "backproj_psnr", "hq_score", "status"]) or ["missing"]),
        "",
        "## 8. Hadamard5 Medium/Full/Push Result",
        *(write_table_inline(rows_for(rows, ["hadamard5_medium_noise001", "hadamard5_full_noise001", "hadamard5_push_hq"]), ["method", "model_psnr", "model_ssim", "hq_score", "status"]) or ["missing"]),
        "",
        "## 9. Rademacher10 Control",
        *(write_table_inline(rows_for(rows, ["rademacher10_full_noise001"]), ["method", "model_psnr", "model_ssim", "backproj_psnr", "status"]) or ["missing"]),
        "",
        "## 10. Scrambled Hadamard10 Control",
        *(write_table_inline(rows_for(rows, ["scrambled_hadamard10_full_noise001"]), ["method", "model_psnr", "model_ssim", "backproj_psnr", "status"]) or ["missing"]),
        "",
        "## 11. MNIST/Fashion Sanity",
        *(write_table_inline(rows_for(rows, ["mnist_hadamard5_full", "fashion_hadamard5_full"]), ["method", "dataset_name", "model_psnr", "model_ssim", "status"]) or ["missing"]),
        "",
        "## 12. Multiseed Result",
        *(write_table_inline(multiseed, ["mean_psnr", "std_psnr", "mean_ssim", "std_ssim", "success_count", "status"]) if multiseed else ["missing"]),
        "",
        "## 13. Noise Robustness Result",
        *(write_table_inline(noise, ["method", "best_noise_std", "best_model_psnr", "best_model_ssim", "status"]) if noise else ["missing"]),
        "",
        "## 14. Attribution: Backprojection vs Model",
        *(write_table_inline(attribution, ["method", "delta_psnr_model_minus_backproj", "delta_ssim_model_minus_backproj", "classification", "status"]) if attribution else ["missing"]),
        "",
        "## 15. Current Best Method",
        str(best.get("method") if best else "missing"),
        "",
        "## 16. Current Best PSNR",
        str(best.get("model_psnr") if best else "missing"),
        "",
        "## 17. Current Best SSIM",
        str(best.get("model_ssim") if best else "missing"),
        "",
        "## 18. Current Best HQ Score",
        str(best.get("hq_score") if best else "missing"),
        "",
        "## 19. STL-10 10% High-quality",
        str(threshold_bool(rows, "reaches_stl10_10pct_hq")),
        "",
        "## 20. STL-10 5% High-quality",
        str(threshold_bool(rows, "reaches_stl10_5pct_hq")),
        "",
        "## 21. Simple-domain High-quality",
        str(threshold_bool(rows, "reaches_simple_domain_hq")),
        "",
        "## 22. Threshold-passing Rows",
        *(write_table_inline([row for row in rows if row.get("status") == "completed" and (str(row.get("reaches_stl10_10pct_hq")).lower() == "true" or str(row.get("reaches_stl10_5pct_hq")).lower() == "true" or str(row.get("reaches_simple_domain_hq")).lower() == "true")], ["dataset_name", "sampling_ratio", "pattern_type", "checkpoint", "sample_image"]) or ["none"]),
        "",
        "## 23. Closest Configuration And Gap",
        *closest_gap(rows),
        "",
        "## 24. Continue Training Recommendation",
        *(write_table_inline([row for row in rows if row.get("convergence_continue_recommended") not in {"", None}], ["method", "convergence_continue_recommended", "status"]) or ["No completed convergence recommendation yet."]),
        "",
        "## 25. Conclusions Allowed In Paper",
        *[f"- {item}" for item in supported],
        "",
        "## 26. Conclusions Not Allowed In Paper",
        *[f"- {item}" for item in unsupported],
        "",
        "## 27. Next Steps",
        "- Complete the Phase 10 overnight queue, prioritizing hadamard10_full_noise001 and hadamard5_medium_noise001.",
        "- Regenerate Phase 10/11 aggregation after completed checkpoints appear.",
        "- Run multiseed only after hadamard10_full_noise001 reaches the 10% threshold.",
        "",
    ]
    report = ROOT11 / "PHASE11_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Phase 11 report written to: {report}")


def write_table_inline(rows: list[dict], fields: list[str]) -> list[str]:
    if not rows:
        return []
    out = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        out.append("|" + "|".join(str(row.get(field, "")).replace("|", "/") for field in fields) + "|")
    return out


if __name__ == "__main__":
    main()
