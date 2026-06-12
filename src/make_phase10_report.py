from __future__ import annotations

import csv
import json
from pathlib import Path

from .utils import ensure_dir


ROOT = Path("E:/ns_mc_gan_gi/outputs_phase10")
PHASE9_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase9")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def as_float(value):
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def yes(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def best_row(rows: list[dict]) -> dict | None:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("phase") == "phase10"]
    if not completed:
        completed = [row for row in rows if row.get("status") == "completed"]
    best = None
    best_value = float("-inf")
    for row in completed:
        value = as_float(row.get("hq_score"))
        if value is None:
            psnr = as_float(row.get("model_psnr"))
            ssim = as_float(row.get("model_ssim"))
            value = None if psnr is None or ssim is None else psnr + 20.0 * ssim
        if value is not None and value > best_value:
            best = row
            best_value = value
    return best


def phase9_artifacts() -> list[str]:
    checks = [
        PHASE9_ROOT / "sanity_hadamard",
        PHASE9_ROOT / "hadamard_sanity",
        PHASE9_ROOT / "PHASE9_REPORT.md",
        PHASE9_ROOT / "phase9_results.csv",
        PHASE9_ROOT / "hadamard10_probe_noise001" / "eval_metrics.json",
        PHASE9_ROOT / "hadamard10_probe_nonoise" / "eval_metrics.json",
        PHASE9_ROOT / "rademacher10_probe_noise001" / "eval_metrics.json",
        PHASE9_ROOT / "overfit_hadamard_10pct" / "overfit_metrics.csv",
    ]
    lines = []
    for path in checks:
        lines.append(f"- {path}: {'present' if path.exists() else 'missing'}")
    return lines


def task_summary() -> list[str]:
    status = read_json(ROOT / "overnight_status.json")
    tasks = status.get("tasks", [])
    if not tasks:
        return ["- No overnight status was generated yet."]
    lines = []
    for task in tasks:
        lines.append(
            f"- {task.get('task_name')}: {task.get('status')} "
            f"(rc={task.get('return_code')}, log={task.get('stdout_log')})"
        )
    return lines


def conclusion_lines(rows: list[dict]) -> list[str]:
    phase10 = [row for row in rows if row.get("phase") == "phase10" and row.get("status") == "completed"]
    had10 = next((row for row in phase10 if row.get("method") == "hadamard10_full_noise001"), None)
    had5 = next((row for row in phase10 if row.get("method") in {"hadamard5_full_noise001", "hadamard5_medium_noise001"} and yes(row.get("reaches_stl10_5pct_hq"))), None)
    simple = next((row for row in phase10 if yes(row.get("reaches_simple_domain_hq"))), None)
    conclusions = []
    if had10 and yes(had10.get("reaches_stl10_10pct_hq")):
        conclusions.append(
            "STL-10 64x64 10% lowfreq Hadamard achieves the internal high-quality threshold."
        )
    if had5:
        conclusions.append(
            "STL-10 64x64 5% lowfreq Hadamard reaches the internal high-quality threshold."
        )
    if had10 and yes(had10.get("reaches_stl10_10pct_hq")) and not had5:
        conclusions.append(
            "High-quality reconstruction is currently supported at 10% sampling; 5% remains challenging."
        )
    if simple and not (had10 and yes(had10.get("reaches_stl10_10pct_hq"))):
        conclusions.append(
            "Simple targets reach the quality threshold, while STL-10 does not yet support a full high-quality claim."
        )
    best = best_row(rows)
    if best:
        delta = as_float(best.get("delta_model_vs_backproj_psnr"))
        if delta is not None and delta < 0.5:
            conclusions.append(
                "Much of high quality comes from lowfreq Hadamard backprojection; learned reconstruction contributes marginal refinement."
            )
        elif delta is not None and delta >= 0.5:
            conclusions.append(
                "The HQ reconstructor improves over the physically meaningful lowfreq Hadamard initialization."
            )
    if not conclusions:
        conclusions.append("No completed Phase 10 run reaches an internal high-quality threshold yet.")
    return conclusions


def table(rows: list[dict], fields: list[str]) -> list[str]:
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(field, "")).replace("|", "/") for field in fields) + "|")
    return lines


def main() -> None:
    ensure_dir(ROOT)
    rows = read_rows(ROOT / "phase10_results.csv")
    best = best_row(rows)
    conclusions = conclusion_lines(rows)
    completed = [row for row in rows if row.get("status") == "completed"]
    missing = [row for row in rows if row.get("status") != "completed"]
    lines = [
        "# Phase 10 Report",
        "",
        "## 1. Phase 9 Inputs",
        *phase9_artifacts(),
        "",
        "## 2. Phase 10 Objective",
        "Validate whether full or near-full low-frequency Hadamard ghost-imaging runs can support high-quality reconstruction claims.",
        "",
        "## 3. Overnight Runner Status",
        *task_summary(),
        "",
        "## 4. Completed Runs",
        *(table(completed, ["phase", "method", "dataset_name", "sampling_ratio", "model_psnr", "model_ssim", "hq_score", "status"]) if completed else ["No completed runs yet."]),
        "",
        "## 5. Missing Or Failed Runs",
        *(table(missing, ["phase", "method", "dataset_name", "sampling_ratio", "status"]) if missing else ["No missing runs."]),
        "",
        "## 6. Best Method",
        f"Best method: {best.get('method') if best else 'missing'}",
        f"Best checkpoint: {best.get('checkpoint') if best else 'missing'}",
        "",
        "## 7. Best Metrics",
        f"PSNR: {best.get('model_psnr') if best else 'missing'}",
        f"SSIM: {best.get('model_ssim') if best else 'missing'}",
        f"HQ score: {best.get('hq_score') if best else 'missing'}",
        "",
        "## 8. 10 Percent STL-10 Threshold",
        str(any(row.get("phase") == "phase10" and yes(row.get("reaches_stl10_10pct_hq")) for row in rows)),
        "",
        "## 9. 5 Percent STL-10 Threshold",
        str(any(row.get("phase") == "phase10" and yes(row.get("reaches_stl10_5pct_hq")) for row in rows)),
        "",
        "## 10. Simple Domain Threshold",
        str(any(row.get("phase") == "phase10" and yes(row.get("reaches_simple_domain_hq")) for row in rows)),
        "",
        "## 11. Backprojection Versus Model",
        "See phase10_backproj_vs_model.png and delta_model_vs_backproj_psnr in the CSV.",
        "",
        "## 12. Hadamard 10 Percent Noise 0.01",
        *(table([row for row in rows if row.get("method") == "hadamard10_full_noise001"], ["method", "model_psnr", "model_ssim", "backproj_psnr", "delta_model_vs_backproj_psnr", "status"]) or ["missing"]),
        "",
        "## 13. Hadamard 10 Percent No Noise",
        *(table([row for row in rows if row.get("method") == "hadamard10_full_nonoise"], ["method", "model_psnr", "model_ssim", "status"]) or ["missing"]),
        "",
        "## 14. Hadamard 5 Percent",
        *(table([row for row in rows if "hadamard5" in row.get("method", "")], ["method", "model_psnr", "model_ssim", "status"]) or ["missing"]),
        "",
        "## 15. Rademacher Control",
        *(table([row for row in rows if "rademacher10_full" in row.get("method", "")], ["method", "model_psnr", "model_ssim", "backproj_psnr", "status"]) or ["missing"]),
        "",
        "## 16. Scrambled Hadamard Control",
        *(table([row for row in rows if "scrambled_hadamard10_full" in row.get("method", "")], ["method", "model_psnr", "model_ssim", "backproj_psnr", "status"]) or ["missing"]),
        "",
        "## 17. No-DC Control",
        *(table([row for row in rows if row.get("method") == "lowfreq_no_dc10_control"], ["method", "model_psnr", "model_ssim", "backproj_psnr", "status"]) or ["missing"]),
        "",
        "## 18. MNIST And Fashion",
        *(table([row for row in rows if row.get("dataset_name") in {"mnist", "fashion_mnist"}], ["method", "dataset_name", "model_psnr", "model_ssim", "status"]) or ["missing"]),
        "",
        "## 19. CIFAR10 Gray",
        *(table([row for row in rows if row.get("dataset_name") == "cifar10_gray"], ["method", "model_psnr", "model_ssim", "status"]) or ["missing"]),
        "",
        "## 20. Continuous Physical Run",
        *(table([row for row in rows if row.get("method") == "continuous_physical_hq10_full"], ["method", "model_psnr", "model_ssim", "status"]) or ["optional run missing"]),
        "",
        "## 21. Convergence",
        *(table([row for row in rows if row.get("phase") == "phase10"], ["method", "convergence_summary", "continue_training_recommended", "status"]) or ["missing"]),
        "",
        "## 22. Paper Examples",
        f"Examples directory: {ROOT / 'paper_examples'}",
        "",
        "## 23. Allowed Conclusions",
        *[f"- {line}" for line in conclusions],
        "",
        "## 24. Not Allowed Conclusions",
        "- Do not claim full Phase 10 completion for any row whose status is missing or failed.",
        "- Do not claim learned reconstruction is the main source of quality unless it clearly beats backprojection.",
        "- Do not call short Phase 9 probe runs full training.",
        "- Do not make STL-10 5% high-quality claims unless the 5% threshold row is completed and passing.",
        "",
        "## 25. Artifacts",
        f"- results_csv: {ROOT / 'phase10_results.csv'}",
        f"- results_md: {ROOT / 'phase10_results.md'}",
        f"- report: {ROOT / 'PHASE10_REPORT.md'}",
        f"- psnr_plot: {ROOT / 'phase10_psnr.png'}",
        f"- ssim_plot: {ROOT / 'phase10_ssim.png'}",
        f"- hq_plot: {ROOT / 'phase10_hq_score.png'}",
        f"- relmeas_plot: {ROOT / 'phase10_relmeaserr.png'}",
        f"- backproj_vs_model_plot: {ROOT / 'phase10_backproj_vs_model.png'}",
        "",
    ]
    report = ROOT / "PHASE10_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Phase 10 report written to: {report}")


if __name__ == "__main__":
    main()
