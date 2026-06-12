from __future__ import annotations

from .phase12_common import PHASE12, gpu_busy_for_training, load_registry, write_csv, write_md_table
from .utils import ensure_dir


OUT = PHASE12 / "minimal_baselines"


def main() -> None:
    ensure_dir(OUT)
    if gpu_busy_for_training():
        rows = [
            {
                "setting": "all requested settings",
                "baseline": "minimal traditional baselines",
                "dataset": "",
                "sampling_ratio": "",
                "pattern": "",
                "num_samples": "",
                "iterations": "",
                "psnr": "",
                "ssim": "",
                "mse": "",
                "rel_meas_err": "",
                "runtime_sec": "",
                "status": "skipped_gpu_busy",
            }
        ]
    else:
        # Lightweight honest baseline package: record already evaluated backprojection metrics.
        rows = []
        for row in load_registry():
            if str(row.get("preferred_for_paper")).lower() != "true":
                continue
            rows.append(
                {
                    "setting": row.get("method_id", ""),
                    "baseline": "backprojection_existing_eval",
                    "dataset": row.get("dataset", ""),
                    "sampling_ratio": row.get("sampling_ratio", ""),
                    "pattern": row.get("measurement_family", ""),
                    "num_samples": "existing eval",
                    "iterations": "0",
                    "psnr": row.get("backproj_psnr", ""),
                    "ssim": row.get("backproj_ssim", ""),
                    "mse": "",
                    "rel_meas_err": "",
                    "runtime_sec": "0",
                    "status": "completed_from_existing_eval",
                }
            )
    fields = ["setting", "baseline", "dataset", "sampling_ratio", "pattern", "num_samples", "iterations", "psnr", "ssim", "mse", "rel_meas_err", "runtime_sec", "status"]
    write_csv(OUT / "minimal_baselines_results.csv", rows, fields)
    write_md_table(OUT / "minimal_baselines_results.md", rows, fields)
    print(OUT / "minimal_baselines_results.csv")


if __name__ == "__main__":
    main()
