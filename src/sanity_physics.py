from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .datasets import get_val_dataloader
from .measurement import GhostMeasurementOperator
from .utils import ensure_dir, load_config, resolve_device, save_json, set_seed, update_config_from_args


def parse_args():
    parser = argparse.ArgumentParser(description="Check measurement physics identities.")
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit_val_samples", type=int, default=None)
    return parser.parse_args()


def make_measurement(config: dict, device: torch.device) -> GhostMeasurementOperator:
    return GhostMeasurementOperator(
        img_size=config["img_size"],
        sampling_ratio=config["sampling_ratio"],
        pattern_type=config["pattern_type"],
        noise_std=0.0,
        lambda_dc=config["lambda_solver"],
        device=device,
        seed=config["seed"],
    )


def vector_rel_norm(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-12) -> float:
    numer_norm = torch.linalg.norm(numer.reshape(numer.shape[0], -1), dim=1)
    denom_norm = torch.linalg.norm(denom.reshape(denom.shape[0], -1), dim=1).clamp_min(eps)
    return (numer_norm / denom_norm).mean().item()


def check_batch(measurement: GhostMeasurementOperator, x: torch.Tensor) -> dict[str, float]:
    y = measurement.measure(x)
    v = torch.randn(x.shape[0], measurement.n, device=x.device)
    v_ns = measurement.null_project(v)
    x_dc = measurement.dc_project(v, y)
    x_data = measurement.data_solution(y)

    null_error = vector_rel_norm(measurement.A_forward(v_ns), v)
    dc_error = vector_rel_norm(measurement.A_forward(x_dc) - y, y)
    backproj_dc_error = vector_rel_norm(measurement.A_forward(x_data) - y, y)
    return {
        "null_error": null_error,
        "dc_error": dc_error,
        "backproj_dc_error": backproj_dc_error,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = update_config_from_args(
        config,
        args,
        ["device", "dataset_root", "output_dir", "batch_size", "limit_val_samples"],
    )
    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])
    output_dir = ensure_dir(config["output_dir"])
    measurement = make_measurement(config, device)

    print(
        "Sanity physics setup: "
        f"device={device}, img_size={measurement.img_size}, n={measurement.n}, "
        f"m={measurement.m}, sampling_ratio={measurement.sampling_ratio}, "
        f"pattern_type={measurement.pattern_type}, noise_std=0.0, "
        f"lambda_solver={measurement.lambda_dc}"
    )

    random_x = torch.rand(
        min(int(config.get("batch_size", 16)), 16),
        1,
        int(config["img_size"]),
        int(config["img_size"]),
        device=device,
    )
    random_results = check_batch(measurement, random_x)
    print("Random tensor checks:")
    for key, value in random_results.items():
        print(f"  {key}: {value:.8f}")

    stl_results = None
    stl_status = "not_run"
    stl_error = None
    try:
        loader = get_val_dataloader(
            dataset_root=config["dataset_root"],
            img_size=config["img_size"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            limit_val_samples=config["limit_val_samples"],
            seed=config["seed"],
            pin_memory=device.type == "cuda",
        )
        batch = next(iter(loader))[0].to(device, non_blocking=True)
        stl_results = check_batch(measurement, batch)
        stl_status = "ok"
        print("STL-10 batch checks:")
        for key, value in stl_results.items():
            print(f"  {key}: {value:.8f}")
    except Exception as exc:
        stl_status = "failed"
        stl_error = str(exc)
        print(f"STL-10 batch checks failed: {stl_error}")

    report = {
        "config": config,
        "device": str(device),
        "n": measurement.n,
        "m": measurement.m,
        "sampling_ratio": measurement.sampling_ratio,
        "pattern_type": measurement.pattern_type,
        "noise_std_used_for_check": 0.0,
        "lambda_solver": measurement.lambda_dc,
        "note": (
            "lambda_solver regularizes K = A A^T + lambda I, so projection "
            "errors are approximate rather than exact zero."
        ),
        "random_tensor": random_results,
        "stl10_batch": stl_results,
        "stl10_status": stl_status,
        "stl10_error": stl_error,
        "manual_stl10_note": (
            "If automatic download fails, place STL-10 files under "
            "E:/ns_mc_gan_gi/data/stl10_binary or rerun with network access."
        ),
    }
    out_path = save_json(report, Path(output_dir) / "sanity_physics.json")
    print(f"Saved sanity report to: {out_path}")


if __name__ == "__main__":
    main()
