from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

import torch

from .phase15_common import read_csv
from .phase15r_common import controlled_reconstruct
from .phase16_common import CORE_STL_METHODS, PHASE16, cpu_info, dataloader_for, gpu_name, setup_method, write_all


OUT = PHASE16 / "runtime_complexity"
FIELDS = [
    "method_id",
    "path",
    "dataset",
    "sampling_ratio",
    "measurement_family",
    "num_samples",
    "batch_size",
    "total_runtime_sec",
    "runtime_sec_per_image",
    "model_params_m",
    "model_param_mb",
    "peak_cuda_mem_mb",
    "device",
    "cpu",
    "status",
    "notes",
]


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def model_size(generator: torch.nn.Module) -> tuple[float, float]:
    params = sum(p.numel() for p in generator.parameters())
    mb = sum(p.numel() * p.element_size() for p in generator.parameters()) / 1024**2
    return params / 1e6, mb


def time_backprojection(method_id: str, limit: int = 64) -> dict[str, Any]:
    _, measurement, config, _ = setup_method(method_id, limit=limit, batch_size=8)
    loader = dataloader_for(config, "test")
    total = 0
    elapsed = 0.0
    peak = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(measurement.device, non_blocking=True)
            y = measurement.measure(x)
            sync()
            start = time.perf_counter()
            flat = measurement.data_solution(y, mode=config.get("backprojection_mode", "ridge_pinv"))
            _ = measurement.unflatten_img(flat).clamp(0, 1)
            sync()
            elapsed += time.perf_counter() - start
            total += int(x.shape[0])
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "method_id": method_id,
        "path": "backprojection",
        "dataset": config.get("dataset_name", ""),
        "sampling_ratio": config.get("sampling_ratio", ""),
        "measurement_family": config.get("pattern_type", ""),
        "num_samples": total,
        "batch_size": 8,
        "total_runtime_sec": elapsed,
        "runtime_sec_per_image": elapsed / max(total, 1),
        "model_params_m": 0.0,
        "model_param_mb": 0.0,
        "peak_cuda_mem_mb": peak,
        "device": gpu_name(),
        "cpu": cpu_info(),
        "status": "completed",
        "notes": "timed local subset",
    }


def time_model(method_id: str, limit: int = 64) -> dict[str, Any]:
    generator, measurement, config, _ = setup_method(method_id, limit=limit, batch_size=8)
    loader = dataloader_for(config, "test")
    params_m, param_mb = model_size(generator)
    total = 0
    elapsed = 0.0
    peak = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch[0].to(measurement.device, non_blocking=True)
            y = measurement.measure(x)
            sync()
            start = time.perf_counter()
            controlled_reconstruct(
                generator,
                measurement,
                y,
                use_null_project=True,
                use_dc_project=True,
                backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=True,
                output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                noise_map_mode="fixed",
                batch_idx=batch_idx,
                seed=int(config["seed"]),
            )
            sync()
            elapsed += time.perf_counter() - start
            total += int(x.shape[0])
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "method_id": method_id,
        "path": "ns_mc_gan_full_inference",
        "dataset": config.get("dataset_name", ""),
        "sampling_ratio": config.get("sampling_ratio", ""),
        "measurement_family": config.get("pattern_type", ""),
        "num_samples": total,
        "batch_size": 8,
        "total_runtime_sec": elapsed,
        "runtime_sec_per_image": elapsed / max(total, 1),
        "model_params_m": params_m,
        "model_param_mb": param_mb,
        "peak_cuda_mem_mb": peak,
        "device": gpu_name(),
        "cpu": cpu_info(),
        "status": "completed",
        "notes": "timed local subset",
    }


def tv_rows_from_baseline() -> list[dict[str, Any]]:
    path = PHASE16 / "traditional_baselines" / "tv_pgd_baseline_results.csv"
    rows = read_csv(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("baseline") == "tv_pgd":
            grouped[row.get("method_id", "")].append(row)
    out = []
    for method_id, sub in grouped.items():
        best = max(sub, key=lambda r: float(r.get("psnr") or 0.0))
        runtime = float(best.get("runtime_sec") or 0.0)
        n = int(float(best.get("num_samples") or 1))
        out.append(
            {
                "method_id": method_id,
                "path": "tv_pgd_best_observed",
                "dataset": best.get("dataset", ""),
                "sampling_ratio": best.get("sampling_ratio", ""),
                "measurement_family": best.get("measurement_family", ""),
                "num_samples": n,
                "batch_size": 4,
                "total_runtime_sec": runtime,
                "runtime_sec_per_image": runtime / max(n, 1),
                "model_params_m": 0.0,
                "model_param_mb": 0.0,
                "peak_cuda_mem_mb": "",
                "device": gpu_name(),
                "cpu": cpu_info(),
                "status": "completed",
                "notes": f"from traditional_baselines; lambda_tv={best.get('lambda_tv', '')}; iterations={best.get('iterations', '')}",
            }
        )
    return out


def main() -> None:
    rows = []
    for method_id in CORE_STL_METHODS:
        rows.append(time_backprojection(method_id))
        rows.append(time_model(method_id))
    rows.extend(tv_rows_from_baseline())
    write_all(OUT / "runtime_complexity", rows, FIELDS)
    print(json.dumps({"rows": len(rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
