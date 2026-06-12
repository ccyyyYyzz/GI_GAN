from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .phase14_ablation_pack_common import load_eval_targets, merge_existing_rows, out_dir, plot_bar, read_json, write_rows


MODES = {
    "full_model": {},
    "no_dc_project_inference": {"--use_dc_project": "false"},
    "no_null_project_inference": {"--use_null_project": "false"},
    "clamp_after_dc": {"--output_range_mode": "clamp_after_dc"},
    "clamp_eval_only": {"--output_range_mode": "clamp_eval_only"},
    "stage1_only": {"--enable_refiner": "false"},
    "raw_generator_no_ema": {"--use_ema": "false"},
    "ema_generator": {"--use_ema": "true"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 14 inference-time ablations.")
    parser.add_argument("--limit_val_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only_method_id", default="", help="Only run one method_id and merge it back into existing results.")
    return parser.parse_args()


def model_metrics(metrics: dict) -> dict:
    return metrics.get("model", {}) if metrics else {}


def main() -> None:
    args = parse_args()
    root = out_dir() / "inference_eval"
    rows = []
    targets = load_eval_targets(include_phase14=True)
    if args.only_method_id:
        targets = [target for target in targets if target["method_id"] == args.only_method_id]
    for target in targets:
        for mode, flags in MODES.items():
            eval_dir = root / target["slug"] / mode
            metrics_path = eval_dir / "eval_metrics.json"
            status = "completed"
            reason = ""
            if args.force or not metrics_path.exists():
                cmd = [
                    sys.executable,
                    "-m",
                    "src.eval",
                    "--config",
                    target["config"],
                    "--checkpoint",
                    target["checkpoint"],
                    "--output_dir",
                    str(eval_dir),
                    "--dataset_root",
                    "E:/ns_mc_gan_gi/data",
                    "--device",
                    args.device,
                    "--limit_val_samples",
                    str(args.limit_val_samples),
                    "--batch_size",
                    str(args.batch_size),
                ]
                for key, value in flags.items():
                    cmd.extend([key, value])
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    status = "failed"
                    reason = f"eval failed with exit code {exc.returncode}"
            metrics = read_json(metrics_path)
            model = model_metrics(metrics)
            rows.append(
                {
                    "method": target["method"],
                    "method_id": target["method_id"],
                    "dataset": target["dataset"],
                    "sampling_ratio": target["sampling_ratio"],
                    "ablation_mode": mode,
                    "psnr": model.get("psnr", ""),
                    "ssim": model.get("ssim", ""),
                    "rel_meas_err": model.get("rel_meas_error", ""),
                    "checkpoint": target["checkpoint"],
                    "eval_dir": str(eval_dir),
                    "status": status if metrics else ("failed" if status == "failed" else "missing_metrics"),
                    "reason": reason,
                }
            )
    fields = [
        "method",
        "method_id",
        "dataset",
        "sampling_ratio",
        "ablation_mode",
        "psnr",
        "ssim",
        "rel_meas_err",
        "checkpoint",
        "eval_dir",
        "status",
        "reason",
    ]
    rows = merge_existing_rows(
        "inference_ablation_results",
        rows,
        {args.only_method_id} if args.only_method_id else None,
    )
    write_rows("inference_ablation_results", rows, fields)
    completed = [r for r in rows if r["status"] == "completed"]
    plot_bar(completed, "psnr", out_dir() / "inference_ablation_psnr.png", "Inference ablation PSNR", "PSNR")
    plot_bar(completed, "ssim", out_dir() / "inference_ablation_ssim.png", "Inference ablation SSIM", "SSIM")
    plot_bar(
        completed,
        "rel_meas_err",
        out_dir() / "inference_ablation_relmeaserr.png",
        "Inference ablation relative measurement error",
        "RelMeasErr",
    )
    print(f"Wrote inference ablation with {len(rows)} rows")


if __name__ == "__main__":
    main()
