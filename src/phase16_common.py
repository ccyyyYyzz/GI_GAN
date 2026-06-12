from __future__ import annotations

import csv
import json
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .metrics import batch_metrics
from .phase15_common import PHASE15, ensure_dir, read_csv, read_json, sha256_file, write_csv, write_json, write_md_table
from .phase15r_common import (
    apply_A_override,
    base_config_for,
    controlled_reconstruct,
    load_exact_A,
    load_generator_for_eval,
    make_measurement,
    method_dir,
    primary_checkpoint,
    tensor_sha256,
)
from .utils import mean_dict, set_seed


PHASE16 = Path("E:/ns_mc_gan_gi/outputs_phase16/supplementary_experiments")
PHASE15_SUPP = PHASE15 / "supplementary_experiments"
DATA_ROOT = Path("E:/ns_mc_gan_gi/data")
REGISTRY = PHASE15 / "noleak_registry.csv"

METHODS: dict[str, dict[str, Any]] = {
    "rademacher5_hq_noise001_colab": {
        "dataset": "STL-10",
        "dataset_name": "stl10",
        "sampling_ratio": 0.05,
        "measurement_family": "rademacher",
        "rademacher": True,
    },
    "scrambled_hadamard5_hq_noise001_colab": {
        "dataset": "STL-10",
        "dataset_name": "stl10",
        "sampling_ratio": 0.05,
        "measurement_family": "scrambled_hadamard",
        "rademacher": False,
    },
    "rademacher10_full_noise001_colab": {
        "dataset": "STL-10",
        "dataset_name": "stl10",
        "sampling_ratio": 0.10,
        "measurement_family": "rademacher",
        "rademacher": True,
    },
    "scrambled_hadamard10_full_noise001_colab": {
        "dataset": "STL-10",
        "dataset_name": "stl10",
        "sampling_ratio": 0.10,
        "measurement_family": "scrambled_hadamard",
        "rademacher": False,
    },
    "mnist_hadamard5_full_colab": {
        "dataset": "MNIST",
        "dataset_name": "mnist",
        "sampling_ratio": 0.05,
        "measurement_family": "lowfreq_hadamard",
        "rademacher": False,
    },
    "fashion_hadamard5_full_colab": {
        "dataset": "Fashion-MNIST",
        "dataset_name": "fashion_mnist",
        "sampling_ratio": 0.05,
        "measurement_family": "lowfreq_hadamard",
        "rademacher": False,
    },
}

CORE_STL_METHODS = [
    "rademacher5_hq_noise001_colab",
    "scrambled_hadamard5_hq_noise001_colab",
    "rademacher10_full_noise001_colab",
    "scrambled_hadamard10_full_noise001_colab",
]
SIMPLE_METHODS = ["mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"]
ALL_MAIN_METHODS = CORE_STL_METHODS + SIMPLE_METHODS


def registry_rows() -> list[dict[str, str]]:
    return read_csv(REGISTRY)


def registry_by_id() -> dict[str, dict[str, str]]:
    return {row.get("method_id", ""): row for row in registry_rows()}


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_all(path_base: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path_base.parent)
    write_csv(path_base.with_suffix(".csv"), rows, fields)
    write_md_table(path_base.with_suffix(".md"), rows, fields)
    write_json(path_base.with_suffix(".json"), rows)


def copy_to_legacy(src: Path) -> Path:
    dst = PHASE15_SUPP / src.relative_to(PHASE16)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def method_config(method_id: str, *, limit: int | None = None, noise_std: float | None = None, batch_size: int | None = None) -> dict[str, Any]:
    checkpoint = primary_checkpoint(method_dir(method_id), "last.pt")
    config = base_config_for(method_id, checkpoint)
    config["dataset_root"] = str(DATA_ROOT)
    config["output_dir"] = str(method_dir(method_id))
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if limit is not None:
        config["limit_val_samples"] = int(limit)
    if noise_std is not None:
        config["noise_std"] = float(noise_std)
    if batch_size is not None:
        config["batch_size"] = int(batch_size)
    return config


def setup_method(
    method_id: str,
    *,
    limit: int | None = None,
    noise_std: float | None = None,
    state_mode: str = "ema",
    batch_size: int | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    config = method_config(method_id, limit=limit, noise_std=noise_std, batch_size=batch_size)
    device = torch.device(config["device"])
    set_seed(int(config["seed"]))
    measurement = make_measurement(config, device)
    meta = METHODS.get(method_id, {})
    override_info: dict[str, Any] = {"exact_A_loaded": False, "cache_rebuilt": ""}
    if meta.get("rademacher"):
        A = load_exact_A(method_id, device)
        override_info = apply_A_override(measurement, A, "safe_rebuild")
        override_info["exact_A_loaded"] = True
        override_info["exact_A_sha256"] = sha256_file(method_dir(method_id) / "measurement_operator_exact.pt")
        override_info["exact_A_tensor_sha256"] = tensor_sha256(A)
    checkpoint = primary_checkpoint(method_dir(method_id), "last.pt")
    generator, _loaded_config, load_info = load_generator_for_eval(method_id, checkpoint, measurement, state_mode, device)
    override_info.update(load_info)
    return generator, measurement, config, override_info


def dataloader_for(config: dict[str, Any], split: str = "test"):
    from .datasets import get_val_dataloader

    return get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=0,
        limit_val_samples=config.get("limit_val_samples"),
        seed=int(config["seed"]),
        val_split=split,
        pin_memory=str(config["device"]).startswith("cuda"),
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )


def per_sample_metric_rows(x_hat: torch.Tensor, x: torch.Tensor, measurement: Any, y: torch.Tensor, labels=None) -> list[dict[str, Any]]:
    rows = []
    for idx in range(x.shape[0]):
        m = batch_metrics(x_hat[idx : idx + 1], x[idx : idx + 1], measurement, y[idx : idx + 1])
        row = {"psnr": m["psnr"], "ssim": m["ssim"], "mse": m["mse"], "rel_meas_err": m.get("rel_meas_error", "")}
        if labels is not None:
            row["label"] = int(labels[idx])
        rows.append(row)
    return rows


def evaluate_model(
    method_id: str,
    *,
    limit: int = 500,
    noise_std: float | None = None,
    state_mode: str = "ema",
    use_null_project: bool = True,
    use_dc_project: bool = True,
    enable_refiner: bool = True,
    noise_map_mode: str = "fixed",
    split: str = "test",
    collect_per_sample: bool = False,
    y_mode: str = "normal",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generator, measurement, config, info = setup_method(method_id, limit=limit, noise_std=noise_std, state_mode=state_mode)
    loader = dataloader_for(config, split)
    model_metrics = []
    backproj_metrics = []
    per_sample = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch[0].to(measurement.device, non_blocking=True)
            labels = batch[1] if len(batch) > 1 else None
            y = measurement.measure(x)
            if y_mode.startswith("gaussian_rel_"):
                rel = float(y_mode.split("_")[-1])
                gen = torch.Generator(device=y.device).manual_seed(int(config["seed"]) + batch_idx + 1000)
                perturb = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=gen)
                scale = y.norm(dim=1, keepdim=True).clamp_min(1e-12) / math.sqrt(y.shape[1])
                y = y + rel * scale * perturb
            elif y_mode == "shuffle_coefficients":
                idx = torch.randperm(y.shape[1], device=y.device)
                y = y[:, idx]
            elif y_mode == "wrong_sample":
                y = torch.roll(y, shifts=1, dims=0)
            x_hat, x_data, extras = controlled_reconstruct(
                generator,
                measurement,
                y,
                use_null_project=use_null_project,
                use_dc_project=use_dc_project,
                backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=enable_refiner,
                output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                noise_map_mode=noise_map_mode,
                batch_idx=batch_idx,
                seed=int(config["seed"]),
            )
            backproj_metrics.append(batch_metrics(x_data, x, measurement, y))
            mm = batch_metrics(x_hat, x, measurement, y)
            mm["rel_meas_err_clamped"] = mm.get("rel_meas_error", float("nan"))
            mm["rel_meas_err_unclamped"] = batch_metrics(extras["x_hat_unclamped"], x, measurement, y).get("rel_meas_error", float("nan"))
            model_metrics.append(mm)
            if collect_per_sample:
                per_sample.extend(per_sample_metric_rows(x_hat, x, measurement, y, labels=labels))
    model = mean_dict(model_metrics)
    bp = mean_dict(backproj_metrics)
    summary = {
        "method_id": method_id,
        "dataset": METHODS[method_id]["dataset"],
        "sampling_ratio": METHODS[method_id]["sampling_ratio"],
        "measurement_family": METHODS[method_id]["measurement_family"],
        "psnr": model["psnr"],
        "ssim": model["ssim"],
        "mse": model["mse"],
        "rel_meas_err": model.get("rel_meas_error", float("nan")),
        "rel_meas_err_clamped": model.get("rel_meas_err_clamped", float("nan")),
        "rel_meas_err_unclamped": model.get("rel_meas_err_unclamped", float("nan")),
        "backproj_psnr": bp["psnr"],
        "backproj_ssim": bp["ssim"],
        "backproj_mse": bp["mse"],
        "num_samples": int(config.get("limit_val_samples") or 0),
        "exact_A_loaded": info.get("exact_A_loaded", False),
        "cache_rebuilt": info.get("cache_rebuilt", ""),
        "state_mode": state_mode,
        "use_null_project": use_null_project,
        "use_dc_project": use_dc_project,
        "enable_refiner": enable_refiner,
        "noise_std": config.get("noise_std", ""),
        "y_mode": y_mode,
        "status": "completed",
        "notes": "",
    }
    return summary, per_sample


def evaluate_backprojection(method_id: str, *, limit: int = 300, noise_std: float | None = None, mode: str = "ridge_pinv") -> dict[str, Any]:
    _, measurement, config, info = setup_method(method_id, limit=limit, noise_std=noise_std)
    loader = dataloader_for(config, "test")
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(measurement.device, non_blocking=True)
            y = measurement.measure(x)
            if mode == "adjoint":
                flat = measurement.AT_forward(y)
            else:
                flat = measurement.data_solution(y, mode=config.get("backprojection_mode", "ridge_pinv"))
            x_hat = measurement.unflatten_img(flat).clamp(0.0, 1.0)
            rows.append(batch_metrics(x_hat, x, measurement, y))
    out = mean_dict(rows)
    return {
        "method_id": method_id,
        "baseline": "adjoint" if mode == "adjoint" else "backprojection",
        "dataset": METHODS[method_id]["dataset"],
        "sampling_ratio": METHODS[method_id]["sampling_ratio"],
        "measurement_family": METHODS[method_id]["measurement_family"],
        "num_samples": limit,
        "iterations": 0,
        "lambda_tv": "",
        "psnr": out["psnr"],
        "ssim": out["ssim"],
        "mse": out["mse"],
        "rel_meas_err": out.get("rel_meas_error", float("nan")),
        "runtime_sec": "",
        "status": "completed",
        "notes": "linear baseline",
    }


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def evaluate_tv_pgd(
    method_id: str,
    *,
    limit: int = 24,
    iterations: int = 50,
    lambda_tv: float = 0.003,
    noise_std: float | None = None,
) -> dict[str, Any]:
    _, measurement, config, _ = setup_method(method_id, limit=limit, noise_std=noise_std, batch_size=4)
    loader = dataloader_for(config, "test")
    metrics = []
    start = time.perf_counter()
    for batch in loader:
        x = batch[0].to(measurement.device, non_blocking=True)
        y = measurement.measure(x)
        init = measurement.unflatten_img(measurement.data_solution(y, mode=config.get("backprojection_mode", "ridge_pinv"))).clamp(0, 1)
        z = init.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=0.05)
        for _ in range(iterations):
            opt.zero_grad(set_to_none=True)
            pred_y = measurement.A_forward(measurement.flatten_img(z))
            fidelity = 0.5 * torch.mean((pred_y - y) ** 2)
            loss = fidelity + float(lambda_tv) * tv_loss(z)
            loss.backward()
            opt.step()
            with torch.no_grad():
                z.clamp_(0.0, 1.0)
        metrics.append(batch_metrics(z.detach(), x, measurement, y))
    elapsed = time.perf_counter() - start
    out = mean_dict(metrics)
    return {
        "method_id": method_id,
        "baseline": "tv_pgd",
        "dataset": METHODS[method_id]["dataset"],
        "sampling_ratio": METHODS[method_id]["sampling_ratio"],
        "measurement_family": METHODS[method_id]["measurement_family"],
        "num_samples": limit,
        "iterations": iterations,
        "lambda_tv": lambda_tv,
        "psnr": out["psnr"],
        "ssim": out["ssim"],
        "mse": out["mse"],
        "rel_meas_err": out.get("rel_meas_error", float("nan")),
        "runtime_sec": elapsed,
        "status": "completed",
        "notes": "small_subset" if limit < 100 else "",
    }


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    for i in range(n):
        means[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def save_bar_plot(rows: list[dict[str, Any]], path: Path, value_key: str, group_key: str = "method_id", title: str = "", ylabel: str = "") -> None:
    ensure_dir(path.parent)
    labels = [str(row.get(group_key, "")) for row in rows]
    values = [as_float(row.get(value_key)) for row in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.9), 3.8))
    bars = ax.bar(labels, values, color="#397097", edgecolor="#222", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel or value_key)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_line_plot(rows: list[dict[str, Any]], path: Path, x_key: str, y_key: str, series_key: str = "method_id", title: str = "", ylabel: str = "") -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    series = sorted({row.get(series_key, "") for row in rows})
    for s in series:
        sub = [row for row in rows if row.get(series_key, "") == s and np.isfinite(as_float(row.get(x_key))) and np.isfinite(as_float(row.get(y_key)))]
        sub = sorted(sub, key=lambda row: as_float(row.get(x_key)))
        if not sub:
            continue
        ax.plot([as_float(r[x_key]) for r in sub], [as_float(r[y_key]) for r in sub], marker="o", label=str(s))
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel or y_key)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def cpu_info() -> str:
    return platform.processor() or platform.machine()
