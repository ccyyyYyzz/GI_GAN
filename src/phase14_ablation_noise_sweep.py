from __future__ import annotations

import argparse
import subprocess
import sys

from .phase14_ablation_pack_common import load_eval_targets, merge_existing_rows, out_dir, plot_lines, read_json, write_rows


NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05]
STL10_METHOD_IDS = {
    "stl10_rademacher10_colab_full",
    "stl10_scrambled10_colab_full",
    "stl10_hadamard10_local_full",
    "stl10_hadamard5_local_medium",
    "stl10_rademacher5_colab_full",
    "stl10_scrambled5_colab_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 14 eval-only noise sweep.")
    parser.add_argument("--limit_val_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only_method_id", default="", help="Only run one method_id and merge it back into existing results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    root = out_dir() / "noise_sweep_eval"
    targets = [t for t in load_eval_targets(include_phase14=True) if t["method_id"] in STL10_METHOD_IDS]
    if args.only_method_id:
        targets = [target for target in targets if target["method_id"] == args.only_method_id]
    for target in targets:
        for noise in NOISE_LEVELS:
            noise_label = str(noise).replace(".", "p")
            eval_dir = root / target["slug"] / f"noise_{noise_label}"
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
                    "--noise_std",
                    str(noise),
                    "--limit_val_samples",
                    str(args.limit_val_samples),
                    "--batch_size",
                    str(args.batch_size),
                ]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    status = "failed"
                    reason = f"eval failed with exit code {exc.returncode}"
            metrics = read_json(metrics_path)
            model = metrics.get("model", {}) if metrics else {}
            rows.append(
                {
                    "method": target["method"],
                    "method_id": target["method_id"],
                    "dataset": target["dataset"],
                    "sampling_ratio": target["sampling_ratio"],
                    "pattern_type": target["pattern_type"],
                    "noise_std": noise,
                    "psnr": model.get("psnr", ""),
                    "ssim": model.get("ssim", ""),
                    "rel_meas_err": model.get("rel_meas_error", ""),
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
        "pattern_type",
        "noise_std",
        "psnr",
        "ssim",
        "rel_meas_err",
        "eval_dir",
        "status",
        "reason",
    ]
    rows = merge_existing_rows(
        "noise_sweep_results",
        rows,
        {args.only_method_id} if args.only_method_id else None,
    )
    write_rows("noise_sweep_results", rows, fields)
    completed = [r for r in rows if r["status"] == "completed"]
    plot_lines(completed, "noise_std", "psnr", out_dir() / "noise_sweep_psnr.png", "Noise sweep PSNR", "PSNR")
    plot_lines(completed, "noise_std", "ssim", out_dir() / "noise_sweep_ssim.png", "Noise sweep SSIM", "SSIM")
    plot_lines(
        completed,
        "noise_std",
        "rel_meas_err",
        out_dir() / "noise_sweep_relmeaserr.png",
        "Noise sweep RelMeasErr",
        "RelMeasErr",
    )
    print(f"Wrote noise sweep status with {len(rows)} rows")


if __name__ == "__main__":
    main()
