from __future__ import annotations

import json

import torch

from .datasets import get_val_dataloader
from .eval import make_measurement
from .metrics import batch_metrics
from .phase16_common import DATA_ROOT, PHASE16, save_bar_plot, write_all
from .utils import mean_dict, set_seed


OUT = PHASE16 / "dc_row_control"
FIELDS = ["method_id", "sampling_ratio", "hadamard_include_dc", "hadamard_skip_dc", "backproj_psnr", "backproj_ssim", "backproj_mse", "status", "notes"]


def eval_row(ratio: float, include_dc: bool, skip_dc: bool) -> dict:
    config = {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "img_size": 64,
        "dataset_root": str(DATA_ROOT),
        "dataset_name": "stl10",
        "batch_size": 8,
        "num_workers": 0,
        "limit_val_samples": 500,
        "sampling_ratio": ratio,
        "pattern_type": "lowfreq_hadamard",
        "matrix_normalization": "legacy_sqrt_m",
        "hadamard_include_dc": include_dc,
        "hadamard_skip_dc": skip_dc,
        "hadamard_row_order": "sequency",
        "noise_std": 0.01,
        "lambda_solver": 0.001,
        "backprojection_mode": "ridge_pinv",
        "use_learned_patterns": False,
    }
    device = torch.device(config["device"])
    set_seed(42)
    measurement = make_measurement(config, device)
    loader = get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=64,
        batch_size=8,
        num_workers=0,
        limit_val_samples=500,
        seed=42,
        val_split="test",
        pin_memory=device.type == "cuda",
        dataset_name="stl10",
    )
    metrics = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = measurement.measure(x)
            x_data = measurement.unflatten_img(measurement.data_solution(y)).clamp(0, 1)
            metrics.append(batch_metrics(x_data, x, measurement, y))
    out = mean_dict(metrics)
    return {
        "method_id": f"stl10_lowfreq_hadamard_{int(ratio*100)}pct_{'include_dc' if include_dc and not skip_dc else 'skip_dc'}",
        "sampling_ratio": ratio,
        "hadamard_include_dc": include_dc,
        "hadamard_skip_dc": skip_dc,
        "backproj_psnr": out["psnr"],
        "backproj_ssim": out["ssim"],
        "backproj_mse": out["mse"],
        "status": "completed",
        "notes": "Backprojection-only DC-row control; no model training.",
    }


def main() -> None:
    rows = [eval_row(0.10, True, False), eval_row(0.10, False, True), eval_row(0.05, True, False), eval_row(0.05, False, True)]
    write_all(OUT / "dc_row_final", rows, FIELDS)
    save_bar_plot(rows, OUT / "dc_row_psnr.png", "backproj_psnr", title="DC row control PSNR", ylabel="PSNR")
    save_bar_plot(rows, OUT / "dc_row_ssim.png", "backproj_ssim", title="DC row control SSIM", ylabel="SSIM")
    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
