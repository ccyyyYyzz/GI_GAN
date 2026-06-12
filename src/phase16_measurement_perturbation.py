from __future__ import annotations

import json

from .phase16_common import CORE_STL_METHODS, PHASE16, as_float, evaluate_model, save_bar_plot, save_line_plot, write_all


OUT = PHASE16 / "measurement_perturbation"
PERTURBATION_MODES = [
    "normal",
    "gaussian_rel_0.01",
    "gaussian_rel_0.03",
    "gaussian_rel_0.05",
    "shuffle_coefficients",
    "wrong_sample",
]
FIELDS = [
    "method_id",
    "dataset",
    "sampling_ratio",
    "measurement_family",
    "perturbation_mode",
    "perturbation_level",
    "psnr",
    "ssim",
    "mse",
    "rel_meas_err",
    "psnr_drop_from_normal",
    "ssim_drop_from_normal",
    "num_samples",
    "status",
    "notes",
]


def main() -> None:
    rows = []
    baseline: dict[str, dict[str, float]] = {}
    for method_id in CORE_STL_METHODS:
        for mode in PERTURBATION_MODES:
            summary, _ = evaluate_model(method_id, limit=200, y_mode=mode, noise_map_mode="fixed")
            if mode == "normal":
                baseline[method_id] = {"psnr": as_float(summary["psnr"]), "ssim": as_float(summary["ssim"])}
            base = baseline.get(method_id, {"psnr": float("nan"), "ssim": float("nan")})
            rows.append(
                {
                    "method_id": method_id,
                    "dataset": summary["dataset"],
                    "sampling_ratio": summary["sampling_ratio"],
                    "measurement_family": summary["measurement_family"],
                    "perturbation_mode": mode,
                    "perturbation_level": float(mode.split("_")[-1]) if mode.startswith("gaussian_rel_") else 0.0,
                    "psnr": summary["psnr"],
                    "ssim": summary["ssim"],
                    "mse": summary["mse"],
                    "rel_meas_err": summary["rel_meas_err"],
                    "psnr_drop_from_normal": base["psnr"] - as_float(summary["psnr"]),
                    "ssim_drop_from_normal": base["ssim"] - as_float(summary["ssim"]),
                    "num_samples": summary["num_samples"],
                    "status": "completed",
                    "notes": "subset_200; perturbations are diagnostic controls, not new training",
                }
            )

    write_all(OUT / "measurement_perturbation", rows, FIELDS)
    numeric_rows = [r for r in rows if str(r["perturbation_mode"]).startswith("gaussian_rel_")]
    save_line_plot(numeric_rows, OUT / "perturbation_psnr.png", "perturbation_level", "psnr", title="Gaussian measurement perturbation PSNR", ylabel="PSNR")
    save_bar_plot([r for r in rows if r["perturbation_mode"] != "normal"], OUT / "perturbation_psnr_drop.png", "psnr_drop_from_normal", "perturbation_mode", title="Perturbation PSNR drop", ylabel="PSNR drop")
    save_bar_plot([r for r in rows if r["perturbation_mode"] != "normal"], OUT / "perturbation_ssim_drop.png", "ssim_drop_from_normal", "perturbation_mode", title="Perturbation SSIM drop", ylabel="SSIM drop")
    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
