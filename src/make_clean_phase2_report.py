from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


OUTPUT_ROOT = Path("E:/ns_mc_gan_gi/outputs_clean_phase2")
ENV_REPORT = OUTPUT_ROOT / "env_report_clean.json"
OLD_METRICS = Path("E:/ns_mc_gan_gi/outputs/quick_5pct/eval_metrics.json")
COMPARE_REPORT = OUTPUT_ROOT / "compare_old_vs_clean_5pct.json"
SAMPLING_RESULTS = OUTPUT_ROOT / "clean_phase2_results.csv"
ABLATION_RESULTS = OUTPUT_ROOT / "clean_phase2_ablation_results.csv"


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value) -> str:
    if value in (None, ""):
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


def quick_row(rows: list[dict]) -> dict | None:
    for row in rows:
        if ratio_key(row.get("sampling_ratio")) == "0.05":
            return row
    return None


def ratio_key(value) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def status_for_run(run_dir: Path, eval_config: str) -> dict:
    metrics = run_dir / "eval_metrics.json"
    checkpoint = run_dir / "best_ssim.pt"
    report = run_dir / "RUN_REPORT.md"
    if metrics.exists() and checkpoint.exists():
        return {"status": "completed", "reason": "ok", "next_command": "none"}
    if checkpoint.exists() and not metrics.exists():
        return {
            "status": "missing",
            "reason": "checkpoint exists but eval_metrics.json is missing",
            "next_command": (
                f"python -m src.eval --checkpoint {checkpoint.as_posix()} "
                f"--config {eval_config} --device cuda"
            ),
        }
    if report.exists():
        return {
            "status": "failed",
            "reason": "RUN_REPORT.md exists but best checkpoint or eval metrics are incomplete",
            "next_command": f"python -m src.train --config {eval_config} --device cuda",
        }
    return {
        "status": "missing",
        "reason": "run directory or required artifacts are missing",
        "next_command": f"python -m src.train --config {eval_config} --device cuda",
    }


def sampling_completion() -> list[dict]:
    specs = [
        ("1%", OUTPUT_ROOT / "quick_1pct", "configs/clean_quick_1pct.yaml"),
        ("2%", OUTPUT_ROOT / "quick_2pct", "configs/clean_quick_2pct.yaml"),
        ("5%", OUTPUT_ROOT / "quick_5pct", "configs/clean_quick_5pct.yaml"),
        ("10%", OUTPUT_ROOT / "quick_10pct", "configs/clean_quick_10pct.yaml"),
    ]
    rows = []
    for label, run_dir, config in specs:
        row = status_for_run(run_dir, config)
        row["sampling_ratio"] = label
        rows.append(row)
    return rows


def ablation_completion() -> list[dict]:
    specs = [
        ("No DC Projection", OUTPUT_ROOT / "ablation_5pct_no_dc", "configs/clean_ablation_5pct_no_dc.yaml"),
        ("No Null Projection", OUTPUT_ROOT / "ablation_5pct_no_null", "configs/clean_ablation_5pct_no_null.yaml"),
        ("No Adversarial", OUTPUT_ROOT / "ablation_5pct_no_adv", "configs/clean_ablation_5pct_no_adv.yaml"),
    ]
    rows = []
    for label, run_dir, config in specs:
        row = status_for_run(run_dir, config)
        row["method"] = label
        rows.append(row)
    return rows


def sanity_rows() -> list[dict]:
    rows = []
    for label, run_dir in [
        ("1%", OUTPUT_ROOT / "quick_1pct"),
        ("2%", OUTPUT_ROOT / "quick_2pct"),
        ("5%", OUTPUT_ROOT / "quick_5pct"),
        ("10%", OUTPUT_ROOT / "quick_10pct"),
    ]:
        sanity = read_json(run_dir / "sanity_physics.json") or {}
        random_tensor = sanity.get("random_tensor") or {}
        stl10_batch = sanity.get("stl10_batch") or {}
        rows.append(
            {
                "sampling_ratio": label,
                "random_null_error": random_tensor.get("null_error", ""),
                "random_dc_error": random_tensor.get("dc_error", ""),
                "stl10_null_error": stl10_batch.get("null_error", ""),
                "stl10_dc_error": stl10_batch.get("dc_error", ""),
                "status": "ok" if sanity else "missing",
            }
        )
    return rows


def artifact_rows(sampling_rows: list[dict]) -> list[dict]:
    by_ratio = {ratio_key(row.get("sampling_ratio")): row for row in sampling_rows}
    rows = []
    for ratio, label in [("0.01", "1%"), ("0.02", "2%"), ("0.05", "5%"), ("0.10", "10%")]:
        row = by_ratio.get(ratio, {})
        rows.append(
            {
                "sampling_ratio": label,
                "checkpoint": row.get("checkpoint", "missing"),
                "sample_image": row.get("sample_image", "missing"),
                "run_report": row.get("run_report", "missing"),
            }
        )
    return rows


def conclusion(sampling_rows: list[dict], ablation_rows: list[dict]) -> str:
    row = quick_row(sampling_rows)
    if not row or row.get("status") != "ok":
        return "Clean 5% reproduction is not complete yet, so Phase 2.1 cannot claim a clean reproduction result."
    delta_psnr = fmt(row.get("delta_psnr"))
    delta_ssim = fmt(row.get("delta_ssim"))
    completed_sweep = sum(1 for item in sampling_rows if item.get("status") == "ok")
    completed_ablation = sum(1 for item in ablation_rows if item.get("status") == "ok")
    return (
        f"Clean 5% reproduction is complete. NS-MC-GAN improves over backprojection by "
        f"{delta_psnr} dB PSNR and {delta_ssim} SSIM on the evaluated STL-10 subset. "
        f"Completed sampling rows: {completed_sweep}/4; completed ablations: {completed_ablation}/4 including the full model row."
    )


def phase3_guidance(completion_rows: list[dict], ablation_completion_rows: list[dict]) -> str:
    required_sampling = {"2%", "5%", "10%"}
    done_sampling = {row["sampling_ratio"] for row in completion_rows if row["status"] == "completed"}
    no_dc_done = any(
        row["method"] == "No DC Projection" and row["status"] == "completed"
        for row in ablation_completion_rows
    )
    if required_sampling.issubset(done_sampling) and no_dc_done:
        return "Phase 3 can start from the clean Phase 2.1 baseline because 5%, 2%, 10%, and the no_dc ablation are complete."
    return "Phase 3 should wait until clean 5%, 2%, 10%, and at least the no_dc ablation are complete."


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = read_json(ENV_REPORT) or {}
    old = read_json(OLD_METRICS) or {}
    compare = read_json(COMPARE_REPORT) or {}
    sampling_rows = read_csv(SAMPLING_RESULTS)
    ablation_rows = read_csv(ABLATION_RESULTS)
    sampling_status_rows = sampling_completion()
    ablation_status_rows = ablation_completion()
    quick = quick_row(sampling_rows) or {}

    clean_5pct_done = "yes" if quick.get("status") == "ok" else "missing"
    sweep_done = "yes" if sampling_rows and all(row.get("status") == "ok" for row in sampling_rows) else "missing/partial"
    ablation_done = "yes" if ablation_rows and all(row.get("status") == "ok" for row in ablation_rows) else "missing/partial"

    lines = [
        "# Clean Phase 2.1 Report",
        "",
        f"- Experiment report time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Clean environment report path: {ENV_REPORT if ENV_REPORT.exists() else 'missing'}",
        f"- Python version: {env.get('python_version', 'missing')}",
        f"- PyTorch version: {env.get('torch_version', 'missing')}",
        f"- torchvision version: {env.get('torchvision_version', 'missing')}",
        f"- NumPy version: {env.get('numpy_version', 'missing')}",
        f"- torch_numpy_bridge: {env.get('torch_numpy_bridge', 'missing')}",
        f"- CUDA available: {env.get('cuda_available', 'missing')}",
        f"- GPU name: {env.get('gpu_name', 'missing')}",
        f"- Dataset path: {env.get('dataset_root', 'E:/ns_mc_gan_gi/data')}",
        f"- Output path: {OUTPUT_ROOT}",
        f"- Clean 5% reproduction completed: {clean_5pct_done}",
        f"- Sampling sweep completed: {sweep_done}",
        f"- Ablation completed: {ablation_done}",
        "",
        "## Clean 5% Metrics",
        "",
    ]
    lines.extend(
        markdown_table(
            [quick],
            [
                "sampling_ratio",
                "m",
                "backproj_psnr",
                "model_psnr",
                "delta_psnr",
                "backproj_ssim",
                "model_ssim",
                "delta_ssim",
                "status",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Old ABI 5% Metrics",
            "",
        ]
    )
    old_model = old.get("model", {})
    old_back = old.get("backprojection", {})
    lines.extend(
        markdown_table(
            [
                {
                    "old_env_model_psnr": old_model.get("psnr", ""),
                    "old_env_model_ssim": old_model.get("ssim", ""),
                    "old_env_model_mse": old_model.get("mse", ""),
                    "old_env_backproj_psnr": old_back.get("psnr", ""),
                    "status": "ok" if old else "missing",
                }
            ],
            ["old_env_model_psnr", "old_env_model_ssim", "old_env_model_mse", "old_env_backproj_psnr", "status"],
        )
    )
    lines.extend(
        [
            "",
            "## Clean vs Old 5%",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [compare],
            [
                "old_env_model_psnr",
                "clean_env_model_psnr",
                "clean_minus_old_psnr",
                "old_env_model_ssim",
                "clean_env_model_ssim",
                "clean_minus_old_ssim",
                "note",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Sampling Sweep Main Table",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            sampling_rows,
            [
                "sampling_ratio",
                "m",
                "backproj_psnr",
                "model_psnr",
                "delta_psnr",
                "backproj_ssim",
                "model_ssim",
                "delta_ssim",
                "status",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Sampling Sweep Completion",
            "",
        ]
    )
    lines.extend(markdown_table(sampling_status_rows, ["sampling_ratio", "status", "reason", "next_command"]))
    lines.extend(
        [
            "",
            "## Sampling Curve Figures",
            "",
            f"- PSNR: {OUTPUT_ROOT / 'clean_phase2_psnr_vs_sampling.png'}",
            f"- SSIM: {OUTPUT_ROOT / 'clean_phase2_ssim_vs_sampling.png'}",
            f"- MSE: {OUTPUT_ROOT / 'clean_phase2_mse_vs_sampling.png'}",
            f"- RelMeasErr: {OUTPUT_ROOT / 'clean_phase2_relmeaserr_vs_sampling.png'}",
            "",
            "## Ablation Main Table",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ablation_rows,
            ["method", "model_psnr", "model_ssim", "model_rel_meas_err", "delta_psnr", "delta_ssim", "status"],
        )
    )
    lines.extend(
        [
            "",
            "## Ablation Completion",
            "",
        ]
    )
    lines.extend(markdown_table(ablation_status_rows, ["method", "status", "reason", "next_command"]))
    lines.extend(
        [
            "",
            "## Ablation Figures",
            "",
            f"- PSNR bar: {OUTPUT_ROOT / 'clean_phase2_ablation_bar_psnr.png'}",
            f"- SSIM bar: {OUTPUT_ROOT / 'clean_phase2_ablation_bar_ssim.png'}",
            f"- RelMeasErr bar: {OUTPUT_ROOT / 'clean_phase2_ablation_bar_relmeaserr.png'}",
            "",
            "## Sanity Physics Key Results",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            sanity_rows(),
            ["sampling_ratio", "random_null_error", "random_dc_error", "stl10_null_error", "stl10_dc_error", "status"],
        )
    )
    lines.extend(
        [
            "",
            "## Checkpoints And Sample Images",
            "",
        ]
    )
    lines.extend(markdown_table(artifact_rows(sampling_rows), ["sampling_ratio", "checkpoint", "sample_image", "run_report"]))
    lines.extend(
        [
            "",
            "## Current Conclusion",
            "",
            conclusion(sampling_rows, ablation_rows),
            "",
            "## Current Limitations",
            "",
            "- Missing rows are explicit missing/partial status and are not counted as completed experiments.",
            "- Clean runs use the STL-10 subset configured by limit_train_samples and limit_val_samples.",
            "- If OOM, worker crash, STL-10 download, CUDA, or NumPy ABI problems occur, the exact failure should stay visible in the run output and completion table.",
            "",
            "## Phase 3 Suggestion",
            "",
            phase3_guidance(sampling_status_rows, ablation_status_rows),
        ]
    )

    report_path = OUTPUT_ROOT / "CLEAN_PHASE2_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Clean Phase 2.1 Summary",
        "",
        f"- clean environment report: {ENV_REPORT if ENV_REPORT.exists() else 'missing'}",
        f"- clean 5% reproduction: {clean_5pct_done}",
        f"- torch_numpy_bridge: {env.get('torch_numpy_bridge', 'missing')}",
        f"- sampling sweep: {sweep_done}",
        f"- ablation: {ablation_done}",
        f"- conclusion: {conclusion(sampling_rows, ablation_rows)}",
        f"- phase 3: {phase3_guidance(sampling_status_rows, ablation_status_rows)}",
        f"- report: {report_path}",
    ]
    (OUTPUT_ROOT / "clean_phase2_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote clean Phase 2.1 report to: {report_path}")


if __name__ == "__main__":
    main()
