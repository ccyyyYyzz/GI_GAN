from __future__ import annotations

import json

from .phase16_common import CORE_STL_METHODS, PHASE16, evaluate_model, save_line_plot, write_all


OUT = PHASE16 / "noise_sweep"
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05]
FIELDS = ["method_id", "noise_std", "dataset", "sampling_ratio", "measurement_family", "psnr", "ssim", "mse", "rel_meas_err", "status", "notes"]


def main() -> None:
    rows = []
    for method_id in CORE_STL_METHODS:
        for noise in NOISE_LEVELS:
            try:
                summary, _ = evaluate_model(method_id, limit=500, noise_std=noise, noise_map_mode="fixed")
                rows.append(
                    {
                        "method_id": method_id,
                        "noise_std": noise,
                        "dataset": summary["dataset"],
                        "sampling_ratio": summary["sampling_ratio"],
                        "measurement_family": summary["measurement_family"],
                        "psnr": summary["psnr"],
                        "ssim": summary["ssim"],
                        "mse": summary["mse"],
                        "rel_meas_err": summary["rel_meas_err"],
                        "status": "completed",
                        "notes": "limit_eval_samples=500",
                    }
                )
            except Exception as exc:
                rows.append({"method_id": method_id, "noise_std": noise, "status": "failed", "notes": f"{type(exc).__name__}: {exc}"})
    write_all(OUT / "noise_sweep_results", rows, FIELDS)
    save_line_plot(rows, OUT / "noise_sweep_psnr.png", "noise_std", "psnr", title="Noise sweep PSNR", ylabel="PSNR")
    save_line_plot(rows, OUT / "noise_sweep_ssim.png", "noise_std", "ssim", title="Noise sweep SSIM", ylabel="SSIM")
    save_line_plot(rows, OUT / "noise_sweep_relmeaserr.png", "noise_std", "rel_meas_err", title="Noise sweep RelMeasErr", ylabel="RelMeasErr")
    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
