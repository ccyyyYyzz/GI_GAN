from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .datasets import get_val_dataloader
from .eval import make_measurement
from .metrics import batch_metrics
from .phase14_ablation_pack_common import load_eval_targets, write_rows
from .utils import apply_experiment_defaults, load_config, mean_dict, resolve_device, set_seed


BASELINE_METHOD_IDS = {
    "stl10_rademacher10_colab_full",
    "stl10_scrambled10_colab_full",
    "stl10_hadamard10_local_full",
    "stl10_hadamard5_local_medium",
    "mnist_hadamard5_colab_full",
    "fashion_hadamard5_colab_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight traditional baselines for Phase 14.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tv_iters", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()


def tv_pgd_reconstruct(measurement, y: torch.Tensor, x0: torch.Tensor, iters: int, lr: float = 0.08, tv_weight: float = 0.002) -> torch.Tensor:
    x_var = x0.detach().clone().clamp(0.0, 1.0).requires_grad_(True)
    opt = torch.optim.Adam([x_var], lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        pred = measurement.A_forward(measurement.flatten_img(x_var))
        fidelity = torch.mean((pred - y) ** 2)
        loss = fidelity + tv_weight * tv_loss(x_var)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x_var.clamp_(0.0, 1.0)
    return x_var.detach()


def run_target(target: dict, args: argparse.Namespace) -> list[dict]:
    config = apply_experiment_defaults(load_config(target["config"]))
    config["dataset_root"] = "E:/ns_mc_gan_gi/data"
    config["device"] = args.device
    config["batch_size"] = args.batch_size
    config["num_workers"] = 0
    limit = 500 if config.get("dataset_name") in {"mnist", "fashion_mnist"} else 200
    config["limit_val_samples"] = limit
    device = resolve_device(args.device)
    set_seed(int(config["seed"]))
    measurement = make_measurement(config, device)
    loader = get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        num_workers=0,
        limit_val_samples=limit,
        seed=config["seed"],
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )
    metric_lists = {
        "backprojection_zero_filled_or_pinv": [],
        "adjoint_dgi_like_correlation": [],
        "tv_pgd_lightweight": [],
    }
    runtimes = {key: 0.0 for key in metric_lists}
    with torch.no_grad():
        cached_batches = []
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            cached_batches.append((x, y))
            t0 = time.perf_counter()
            bp = measurement.unflatten_img(measurement.data_solution(y, mode=config.get("backprojection_mode", "ridge_pinv"))).clamp(0.0, 1.0)
            runtimes["backprojection_zero_filled_or_pinv"] += time.perf_counter() - t0
            metric_lists["backprojection_zero_filled_or_pinv"].append(batch_metrics(bp, x, measurement, y))

            t0 = time.perf_counter()
            adj = measurement.unflatten_img(measurement.AT_forward(y)).clamp(0.0, 1.0)
            runtimes["adjoint_dgi_like_correlation"] += time.perf_counter() - t0
            metric_lists["adjoint_dgi_like_correlation"].append(batch_metrics(adj, x, measurement, y))
    for x, y in cached_batches:
        with torch.no_grad():
            x0 = measurement.unflatten_img(measurement.data_solution(y, mode=config.get("backprojection_mode", "ridge_pinv"))).clamp(0.0, 1.0)
        t0 = time.perf_counter()
        tv = tv_pgd_reconstruct(measurement, y, x0, args.tv_iters)
        runtimes["tv_pgd_lightweight"] += time.perf_counter() - t0
        metric_lists["tv_pgd_lightweight"].append(batch_metrics(tv, x, measurement, y))

    rows = []
    for baseline, metrics in metric_lists.items():
        mean = mean_dict(metrics)
        rows.append(
            {
                "setting": target["method"],
                "method_id": target["method_id"],
                "baseline": baseline,
                "dataset": config.get("dataset_name", ""),
                "sampling_ratio": config.get("sampling_ratio", ""),
                "pattern": config.get("pattern_type", ""),
                "num_samples": limit,
                "iterations": args.tv_iters if baseline == "tv_pgd_lightweight" else 0,
                "psnr": mean.get("psnr", ""),
                "ssim": mean.get("ssim", ""),
                "mse": mean.get("mse", ""),
                "rel_meas_err": mean.get("rel_meas_error", ""),
                "runtime_sec": round(runtimes[baseline], 4),
                "status": "completed",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    for target in load_eval_targets(include_phase14=False):
        if target["method_id"] not in BASELINE_METHOD_IDS:
            continue
        print(f"Running baselines for {target['method_id']}")
        rows.extend(run_target(target, args))
    fields = [
        "setting",
        "method_id",
        "baseline",
        "dataset",
        "sampling_ratio",
        "pattern",
        "num_samples",
        "iterations",
        "psnr",
        "ssim",
        "mse",
        "rel_meas_err",
        "runtime_sec",
        "status",
    ]
    write_rows("traditional_baselines", rows, fields)
    print(f"Wrote lightweight traditional baselines with {len(rows)} rows")


if __name__ == "__main__":
    main()
