from __future__ import annotations

import json

from .phase16_common import CORE_STL_METHODS, PHASE16, evaluate_model, save_bar_plot, write_all


OUT = PHASE16 / "inference_ablation"
FIELDS = [
    "method_id",
    "ablation_mode",
    "dataset",
    "sampling_ratio",
    "measurement_family",
    "psnr",
    "ssim",
    "mse",
    "rel_meas_err",
    "rel_meas_err_unclamped",
    "rel_meas_err_clamped",
    "delta_vs_full_psnr",
    "delta_vs_full_ssim",
    "delta_vs_full_relmeaserr",
    "status",
    "notes",
]

MODES = {
    "full_model": {"use_dc_project": True, "use_null_project": True, "enable_refiner": True, "state_mode": "ema"},
    "no_dc_project": {"use_dc_project": False, "use_null_project": True, "enable_refiner": True, "state_mode": "ema"},
    "no_null_project": {"use_dc_project": True, "use_null_project": False, "enable_refiner": True, "state_mode": "ema"},
    "no_dc_no_null": {"use_dc_project": False, "use_null_project": False, "enable_refiner": True, "state_mode": "ema"},
    "stage1_only": {"use_dc_project": True, "use_null_project": True, "enable_refiner": False, "state_mode": "ema"},
    "stage1_plus_refiner": {"use_dc_project": True, "use_null_project": True, "enable_refiner": True, "state_mode": "ema"},
    "raw_weights": {"use_dc_project": True, "use_null_project": True, "enable_refiner": True, "state_mode": "raw"},
    "ema_weights": {"use_dc_project": True, "use_null_project": True, "enable_refiner": True, "state_mode": "ema"},
}


def main() -> None:
    rows = []
    full_by_method = {}
    for method_id in CORE_STL_METHODS:
        for mode, opts in MODES.items():
            try:
                summary, _ = evaluate_model(method_id, limit=500, noise_map_mode="fixed", **opts)
                if mode == "full_model":
                    full_by_method[method_id] = summary
                full = full_by_method.get(method_id, summary)
                rows.append(
                    {
                        "method_id": method_id,
                        "ablation_mode": mode,
                        "dataset": summary["dataset"],
                        "sampling_ratio": summary["sampling_ratio"],
                        "measurement_family": summary["measurement_family"],
                        "psnr": summary["psnr"],
                        "ssim": summary["ssim"],
                        "mse": summary["mse"],
                        "rel_meas_err": summary["rel_meas_err"],
                        "rel_meas_err_unclamped": summary["rel_meas_err_unclamped"],
                        "rel_meas_err_clamped": summary["rel_meas_err_clamped"],
                        "delta_vs_full_psnr": summary["psnr"] - full["psnr"],
                        "delta_vs_full_ssim": summary["ssim"] - full["ssim"],
                        "delta_vs_full_relmeaserr": summary["rel_meas_err"] - full["rel_meas_err"],
                        "status": "completed",
                        "notes": "limit_eval_samples=500",
                    }
                )
            except Exception as exc:
                rows.append({"method_id": method_id, "ablation_mode": mode, "status": "unsupported", "notes": f"{type(exc).__name__}: {exc}"})
    write_all(OUT / "real_inference_ablation_results", rows, FIELDS)
    save_bar_plot([r for r in rows if r.get("ablation_mode") == "full_model"], OUT / "real_inference_ablation_psnr.png", "psnr", title="Full model PSNR", ylabel="PSNR")
    save_bar_plot([r for r in rows if r.get("ablation_mode") == "full_model"], OUT / "real_inference_ablation_ssim.png", "ssim", title="Full model SSIM", ylabel="SSIM")
    save_bar_plot([r for r in rows if r.get("ablation_mode") == "no_dc_project"], OUT / "real_inference_ablation_relmeaserr.png", "rel_meas_err", title="No-DC measurement error", ylabel="RelMeasErr")
    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
