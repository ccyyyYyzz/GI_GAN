from __future__ import annotations

import json
from pathlib import Path

from .phase15_common import read_csv
from .phase16_common import PHASE16, as_float, registry_rows, save_bar_plot, write_all


OUT = PHASE16 / "attribution"
FIELDS = [
    "method_id",
    "dataset",
    "sampling_ratio",
    "measurement_family",
    "backproj_psnr",
    "backproj_ssim",
    "model_psnr",
    "model_ssim",
    "delta_psnr",
    "delta_ssim",
    "mse",
    "rel_meas_err",
    "classification",
    "interpretation",
]


def classify(delta_psnr: float, delta_ssim: float, model_psnr: float, bp_psnr: float) -> str:
    if model_psnr < bp_psnr - 0.1:
        return "model_degrades_backprojection"
    if delta_psnr < 0.3 and delta_ssim < 0.03:
        return "backprojection_dominated"
    return "model_refinement_helpful"


def rescue_rows() -> list[dict]:
    rows = []
    path = Path("E:/ns_mc_gan_gi/outputs_phase14/noleak_rescue_audit/NOLEAK_RESCUE_AUDIT.csv")
    for row in read_csv(path):
        method_id = row.get("method_id") or row.get('\ufeff"method_id"') or row.get("\ufeffmethod_id")
        if not method_id or "hadamard" not in method_id:
            continue
        psnr = as_float(row.get("last_eval_psnr"))
        ssim = as_float(row.get("last_eval_ssim"))
        if psnr != psnr:
            continue
        ratio = 0.10 if "10" in method_id else 0.05
        bp = 18.98 if ratio == 0.05 else 14.53
        rows.append(
            {
                "method_id": method_id,
                "dataset": "STL-10",
                "sampling_ratio": ratio,
                "measurement_family": "lowfreq_hadamard",
                "backproj_psnr": bp,
                "backproj_ssim": "",
                "model_psnr": psnr,
                "model_ssim": ssim,
                "delta_psnr": psnr - bp,
                "delta_ssim": "",
                "mse": "",
                "rel_meas_err": "",
                "classification": "negative_or_auxiliary",
                "interpretation": "Lowfreq Hadamard rescue row; do not label 5% as high-quality.",
            }
        )
    return rows


def main() -> None:
    rows = []
    for row in registry_rows():
        method_id = row.get("method_id", "")
        bp = as_float(row.get("backproj_psnr"))
        model = as_float(row.get("psnr"))
        dpsnr = as_float(row.get("delta_psnr"))
        dssim = as_float(row.get("delta_ssim"))
        family = row.get("measurement_family", "")
        if family == "rademacher":
            interp = "Backprojection is weak, but learned inverse mapping recovers strong final quality."
        elif family == "scrambled_hadamard":
            interp = "Structured measurement gives stronger physical initialization and similar final quality."
        else:
            interp = "Simple-domain or low-frequency sanity measurement."
        rows.append(
            {
                "method_id": method_id,
                "dataset": row.get("dataset", ""),
                "sampling_ratio": row.get("sampling_ratio", ""),
                "measurement_family": family,
                "backproj_psnr": bp,
                "backproj_ssim": row.get("backproj_ssim", ""),
                "model_psnr": model,
                "model_ssim": row.get("ssim", ""),
                "delta_psnr": dpsnr,
                "delta_ssim": dssim,
                "mse": row.get("mse", ""),
                "rel_meas_err": row.get("rel_meas_err", ""),
                "classification": classify(dpsnr, dssim, model, bp),
                "interpretation": interp,
            }
        )
    rows.extend(rescue_rows())
    write_all(OUT / "attribution_final", rows, FIELDS)
    save_bar_plot(rows, OUT / "attribution_delta_psnr.png", "delta_psnr", title="Model gain over backprojection", ylabel="Delta PSNR")
    save_bar_plot(rows, OUT / "attribution_delta_ssim.png", "delta_ssim", title="Model SSIM gain", ylabel="Delta SSIM")
    save_bar_plot(rows, OUT / "bp_vs_model_psnr.png", "model_psnr", title="Final model PSNR", ylabel="PSNR")
    save_bar_plot(rows, OUT / "bp_vs_model_ssim.png", "model_ssim", title="Final model SSIM", ylabel="SSIM")
    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
