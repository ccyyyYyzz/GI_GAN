from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from .phase16_common import ALL_MAIN_METHODS, PHASE16, bootstrap_ci, ensure_dir, evaluate_model, write_all


OUT = PHASE16 / "statistics"
FIELDS = ["method_id", "dataset", "sampling_ratio", "measurement_family", "mean_psnr", "std_psnr", "median_psnr", "ci95_psnr_low", "ci95_psnr_high", "mean_ssim", "std_ssim", "median_ssim", "ci95_ssim_low", "ci95_ssim_high", "num_samples", "source", "status"]


def main() -> None:
    rows = []
    per_method = {}
    for method_id in ALL_MAIN_METHODS:
        summary, samples = evaluate_model(method_id, limit=500, collect_per_sample=True, noise_map_mode="fixed")
        psnr = [float(r["psnr"]) for r in samples]
        ssim = [float(r["ssim"]) for r in samples]
        psnr_low, psnr_high = bootstrap_ci(psnr, 1000)
        ssim_low, ssim_high = bootstrap_ci(ssim, 1000)
        per_method[method_id] = {"psnr": psnr, "ssim": ssim}
        rows.append(
            {
                "method_id": method_id,
                "dataset": summary["dataset"],
                "sampling_ratio": summary["sampling_ratio"],
                "measurement_family": summary["measurement_family"],
                "mean_psnr": float(np.mean(psnr)),
                "std_psnr": float(np.std(psnr)),
                "median_psnr": float(np.median(psnr)),
                "ci95_psnr_low": psnr_low,
                "ci95_psnr_high": psnr_high,
                "mean_ssim": float(np.mean(ssim)),
                "std_ssim": float(np.std(ssim)),
                "median_ssim": float(np.median(ssim)),
                "ci95_ssim_low": ssim_low,
                "ci95_ssim_high": ssim_high,
                "num_samples": len(samples),
                "source": "local re-eval 500 samples",
                "status": "completed",
            }
        )
    write_all(OUT / "statistics_ci", rows, FIELDS)
    ensure_dir(OUT / "best_median_worst_examples")
    for metric, path in [("psnr", OUT / "psnr_histograms.png"), ("ssim", OUT / "ssim_histograms.png")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method_id, data in per_method.items():
            ax.hist(data[metric], bins=24, alpha=0.35, label=method_id)
        ax.set_title(f"{metric.upper()} per-sample distributions")
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("count")
        ax.legend(fontsize=6, frameon=False)
        plt.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
