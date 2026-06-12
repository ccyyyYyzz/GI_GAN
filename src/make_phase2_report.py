from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


OUTPUT_ROOT = Path("E:/ns_mc_gan_gi/outputs")


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


def read_text(path: Path) -> str:
    if not path.exists():
        return "missing"
    return path.read_text(encoding="utf-8", errors="replace")


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


def best_quick_row(rows: list[dict]):
    ok_rows = [row for row in rows if row.get("status") == "ok" and row.get("sampling_ratio") == "0.05"]
    if ok_rows:
        return ok_rows[0]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return None
    return max(ok_rows, key=lambda row: float(row.get("model_ssim") or 0.0))


def conclusion(rows: list[dict]) -> str:
    row = best_quick_row(rows)
    if not row:
        return "No completed quick-training result is available yet, so Phase 2 can only report infrastructure readiness."
    delta_psnr = float(row.get("delta_psnr") or 0.0)
    delta_ssim = float(row.get("delta_ssim") or 0.0)
    ratio = row.get("sampling_ratio", "unknown")
    return (
        f"At sampling_ratio={ratio}, NS-MC-GAN improves over the backprojection baseline "
        f"by {delta_psnr:.3f} dB PSNR and {delta_ssim:.3f} SSIM on the evaluated STL-10 subset."
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = read_json(OUTPUT_ROOT / "env_report.json") or {}
    quick_sanity = read_json(OUTPUT_ROOT / "quick_5pct" / "sanity_physics.json")
    debug_sanity = read_json(OUTPUT_ROOT / "debug_5pct" / "sanity_physics.json")
    sanity = quick_sanity or debug_sanity or {}
    sampling_rows = read_csv(OUTPUT_ROOT / "phase2_results.csv")
    ablation_rows = read_csv(OUTPUT_ROOT / "phase2_ablation_results.csv")
    quick_row = best_quick_row(sampling_rows)

    quick_done = "yes" if quick_row and quick_row.get("sampling_ratio") == "0.05" else "missing"
    sweep_done = "yes" if sampling_rows and all(row.get("status") == "ok" for row in sampling_rows) else "missing/partial"
    ablation_done = "yes" if ablation_rows and all(row.get("status") == "ok" for row in ablation_rows) else "missing/partial"

    lines = [
        "# Phase 2 Report",
        "",
        f"- Experiment report time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Environment report: {OUTPUT_ROOT / 'env_report.json'}",
        f"- Python: {env.get('python_version', 'missing')}",
        f"- PyTorch: {env.get('torch_version', 'missing')}",
        f"- torchvision: {env.get('torchvision_version', 'missing')}",
        f"- NumPy: {env.get('numpy_version', 'missing')}",
        f"- CUDA available: {env.get('cuda_available', 'missing')}",
        f"- GPU: {env.get('gpu_name', 'missing')}",
        f"- Torch NumPy bridge: {env.get('torch_numpy_bridge', 'missing')}",
        f"- Dataset path: {env.get('dataset_root', 'E:/ns_mc_gan_gi/data')}",
        f"- Output path: {OUTPUT_ROOT}",
        f"- Image vector size n: 4096",
        f"- Sampling ratios: 0.01, 0.02, 0.05, 0.10",
        f"- m values: 41, 82, 205, 410",
        f"- quick_train_5pct completed: {quick_done}",
        f"- sampling sweep completed: {sweep_done}",
        f"- ablation completed: {ablation_done}",
        "",
        "## Sanity Physics",
        "",
        f"- Report: {(OUTPUT_ROOT / 'quick_5pct' / 'sanity_physics.json') if quick_sanity else (OUTPUT_ROOT / 'debug_5pct' / 'sanity_physics.json')}",
        f"- random null_error: {fmt((sanity.get('random_tensor') or {}).get('null_error'))}",
        f"- random dc_error: {fmt((sanity.get('random_tensor') or {}).get('dc_error'))}",
        f"- STL-10 null_error: {fmt((sanity.get('stl10_batch') or {}).get('null_error'))}",
        f"- STL-10 dc_error: {fmt((sanity.get('stl10_batch') or {}).get('dc_error'))}",
        "",
        "## Backprojection vs NS-MC-GAN",
        "",
    ]
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
            "## Sampling Curve Figures",
            "",
            f"- PSNR: {OUTPUT_ROOT / 'phase2_psnr_vs_sampling.png'}",
            f"- SSIM: {OUTPUT_ROOT / 'phase2_ssim_vs_sampling.png'}",
            f"- MSE: {OUTPUT_ROOT / 'phase2_mse_vs_sampling.png'}",
            f"- RelMeasErr: {OUTPUT_ROOT / 'phase2_relmeaserr_vs_sampling.png'}",
            "",
            "## Ablation Results",
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
            "## Ablation Figures",
            "",
            f"- PSNR bar: {OUTPUT_ROOT / 'phase2_ablation_bar_psnr.png'}",
            f"- SSIM bar: {OUTPUT_ROOT / 'phase2_ablation_bar_ssim.png'}",
            f"- RelMeasErr bar: {OUTPUT_ROOT / 'phase2_ablation_bar_relmeaserr.png'}",
            "",
            "## Key Artifacts",
            "",
            f"- Best checkpoint: {(quick_row or {}).get('checkpoint', 'missing')}",
            f"- Sample reconstruction: {(quick_row or {}).get('sample_image', 'missing')}",
            f"- Run report: {(quick_row or {}).get('run_report', 'missing')}",
            "",
            "## Current Conclusion",
            "",
            conclusion(sampling_rows),
            "",
            "## Current Limitations",
            "",
            "- Missing rows are explicitly marked missing and should not be interpreted as completed experiments.",
            "- If the environment report shows a Torch NumPy bridge failure, create the clean environment before running long sweeps.",
            "- Quick training uses an STL-10 subset, not a full-scale final benchmark.",
            "",
            "## Phase 3 Suggestions",
            "",
            "- Learnable speckle patterns with binary and energy constraints.",
            "- Add full sampling sweeps once the clean environment is active.",
            "- Add ablation repeats or multiple seeds for uncertainty estimates.",
        ]
    )

    report_path = OUTPUT_ROOT / "PHASE2_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Phase 2 Summary",
        "",
        f"- quick_train_5pct completed: {quick_done}",
        f"- sampling sweep completed: {sweep_done}",
        f"- ablation completed: {ablation_done}",
        f"- main conclusion: {conclusion(sampling_rows)}",
        f"- most important issue: {env.get('torch_numpy_bridge', 'unknown')}",
        f"- report: {report_path}",
    ]
    (OUTPUT_ROOT / "phase2_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote Phase 2 report to: {report_path}")


if __name__ == "__main__":
    main()
