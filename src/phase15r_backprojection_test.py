from __future__ import annotations

import json

import torch
from tqdm import tqdm

from .metrics import batch_metrics
from .phase15_common import read_json
from .phase15r_common import (
    RADEMACHER_METHODS,
    REPRO_DEBUG,
    apply_A_override,
    base_config_for,
    get_loader,
    load_exact_A,
    make_measurement,
    method_dir,
    tensor_sha256,
    write_rows_all_formats,
)
from .utils import mean_dict, set_seed


FIELDS = [
    "method_id",
    "override_mode",
    "colab_backproj_psnr",
    "local_backproj_psnr",
    "diff_backproj_psnr",
    "colab_backproj_ssim",
    "local_backproj_ssim",
    "diff_backproj_ssim",
    "colab_backproj_mse",
    "local_backproj_mse",
    "rel_meas_err",
    "y_norm_mean",
    "y_norm_std",
    "x_data_min",
    "x_data_max",
    "x_data_mean",
    "x_data_std",
    "A_sha",
    "cache_rebuilt",
    "status",
    "likely_issue",
]


def run_one(method_id: str, override_mode: str) -> dict:
    output_dir = method_dir(method_id)
    metrics_colab = read_json(output_dir / "eval_metrics.json")
    config = base_config_for(method_id, output_dir / "last.pt")
    device = torch.device(config["device"])
    set_seed(int(config["seed"]))
    measurement = make_measurement(config, device)
    A = load_exact_A(method_id, device)
    override_info = apply_A_override(measurement, A, override_mode)
    loader = get_loader(config, "test", device)
    bp_metrics = []
    y_norms = []
    x_stats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"BP {method_id}:{override_mode}", leave=False):
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            x_data_flat = measurement.data_solution(y.float(), mode=config.get("backprojection_mode", "ridge_pinv"))
            x_data = measurement.unflatten_img(x_data_flat)
            bp_metrics.append(batch_metrics(x_data, x, measurement, y))
            y_norms.append(y.norm(dim=1).detach().cpu())
            x_stats.append(
                torch.tensor(
                    [float(x_data.min()), float(x_data.max()), float(x_data.mean()), float(x_data.std(unbiased=False))]
                )
            )
    local = mean_dict(bp_metrics)
    y_all = torch.cat(y_norms)
    x_stat = torch.stack(x_stats).mean(dim=0)
    c_psnr = float(metrics_colab["backprojection"]["psnr"])
    c_ssim = float(metrics_colab["backprojection"]["ssim"])
    c_mse = float(metrics_colab["backprojection"]["mse"])
    diff_psnr = float(local["psnr"] - c_psnr)
    diff_ssim = float(local["ssim"] - c_ssim)
    status = "A_dataset_backproj_reproduced" if abs(diff_psnr) <= 0.05 else "A_or_dataset_or_backproj_mismatch"
    likely = "model loading / EMA / refiner / forward" if status.endswith("reproduced") else "A scaling / split / transform / K cache"
    return {
        "method_id": method_id,
        "override_mode": override_mode,
        "colab_backproj_psnr": c_psnr,
        "local_backproj_psnr": float(local["psnr"]),
        "diff_backproj_psnr": diff_psnr,
        "colab_backproj_ssim": c_ssim,
        "local_backproj_ssim": float(local["ssim"]),
        "diff_backproj_ssim": diff_ssim,
        "colab_backproj_mse": c_mse,
        "local_backproj_mse": float(local["mse"]),
        "rel_meas_err": float(local.get("rel_meas_error", float("nan"))),
        "y_norm_mean": float(y_all.mean()),
        "y_norm_std": float(y_all.std(unbiased=False)),
        "x_data_min": float(x_stat[0]),
        "x_data_max": float(x_stat[1]),
        "x_data_mean": float(x_stat[2]),
        "x_data_std": float(x_stat[3]),
        "A_sha": tensor_sha256(A),
        "cache_rebuilt": override_info.get("cache_rebuilt", ""),
        "status": status,
        "likely_issue": likely,
    }


def main() -> None:
    rows = []
    for method in RADEMACHER_METHODS:
        rows.append(run_one(method["method_id"], "unsafe_old_chol"))
        rows.append(run_one(method["method_id"], "safe_rebuild"))
    write_rows_all_formats(REPRO_DEBUG / "backprojection_test", rows, FIELDS)
    print(json.dumps({"rows": len(rows), "output": str(REPRO_DEBUG / "backprojection_test.csv")}, indent=2))


if __name__ == "__main__":
    main()
