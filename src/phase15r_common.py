from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from .phase15_common import PHASE15, ensure_dir, read_json, sha256_file, write_csv, write_json, write_md_table


REPRO_DEBUG = PHASE15 / "repro_debug"
IMPORTED = PHASE15 / "imported_noleak"
DATA_ROOT = Path("E:/ns_mc_gan_gi/data")

RADEMACHER_METHODS = [
    {
        "method_id": "rademacher5_hq_noise001_colab",
        "short": "rademacher5",
        "sampling_ratio": 0.05,
        "expected_m": 205,
        "expected_n": 4096,
    },
    {
        "method_id": "rademacher10_full_noise001_colab",
        "short": "rademacher10",
        "sampling_ratio": 0.10,
        "expected_m": 410,
        "expected_n": 4096,
    },
]


def method_dir(method_id: str) -> Path:
    return IMPORTED / method_id


def torch_load(path: Path, map_location: torch.device | str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def tensor_from_exact_payload(payload: Any) -> torch.Tensor:
    if isinstance(payload, dict):
        for key in ["A", "matrix", "measurement_matrix"]:
            value = payload.get(key)
            if torch.is_tensor(value):
                return value
        for value in payload.values():
            if torch.is_tensor(value) and value.ndim == 2:
                return value
    if torch.is_tensor(payload):
        return payload
    raise TypeError(f"Could not find 2-D tensor in exact-A payload of type {type(payload).__name__}.")


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def finite_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def metric(metrics: dict[str, Any], section: str, key: str) -> float:
    return finite_float((metrics.get(section) or {}).get(key))


def infer_rademacher_normalization(A: torch.Tensor) -> str:
    A_cpu = A.detach().cpu().float()
    abs_nonzero = A_cpu.abs()[A_cpu.abs() > 1e-12]
    if abs_nonzero.numel() == 0:
        return "unknown"
    value = float(abs_nonzero.median().item())
    m, n = A_cpu.shape
    checks = {
        "legacy_sqrt_m_or_row_norm_sqrt_n_over_m": 1.0 / math.sqrt(m),
        "orthonormal_rows_or_pm1_over_sqrt_n": 1.0 / math.sqrt(n),
        "raw_pm1": 1.0,
    }
    best_name = "unknown"
    best_diff = float("inf")
    for name, expected in checks.items():
        diff = abs(value - expected)
        if diff < best_diff:
            best_name = name
            best_diff = diff
    return best_name if best_diff <= max(1e-4, 0.02 * value) else "unknown"


def exact_A_path(method_id: str) -> Path:
    return method_dir(method_id) / "measurement_operator_exact.pt"


def load_exact_A(method_id: str, device: torch.device | str = "cpu") -> torch.Tensor:
    payload = torch_load(exact_A_path(method_id), map_location=device)
    return tensor_from_exact_payload(payload).to(device=device, dtype=torch.float32)


def checkpoint_candidates(output_dir: Path) -> list[Path]:
    names = ["best_hq.pt", "best_score.pt", "best_ssim.pt", "best_psnr.pt", "last.pt"]
    return [output_dir / name for name in names if (output_dir / name).exists()]


def primary_checkpoint(output_dir: Path, preferred: str = "best_hq.pt") -> Path:
    preferred_path = output_dir / preferred
    if preferred_path.exists():
        return preferred_path
    last = output_dir / "last.pt"
    if last.exists():
        return last
    candidates = checkpoint_candidates(output_dir)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No checkpoint found in {output_dir}")


def base_config_for(method_id: str, checkpoint: Path | None = None) -> dict[str, Any]:
    from .utils import apply_experiment_defaults

    output_dir = method_dir(method_id)
    config = read_yaml(output_dir / "resolved_config.yaml")
    config = apply_experiment_defaults(config)
    if checkpoint is not None and checkpoint.exists():
        payload = torch_load(checkpoint, "cpu")
        if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
            merged = dict(config)
            merged.update(payload["config"])
            config = apply_experiment_defaults(merged)
    config["dataset_root"] = str(DATA_ROOT)
    config["output_dir"] = str(output_dir)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def make_measurement(config: dict[str, Any], device: torch.device):
    from .eval import make_measurement as eval_make_measurement

    return eval_make_measurement(config, device)


def apply_A_override(measurement: Any, A: torch.Tensor, mode: str = "safe_rebuild") -> dict[str, Any]:
    mode = str(mode)
    metadata = {"override_mode": mode, "tensor_sha256": tensor_sha256(A)}
    if mode == "safe_rebuild" and hasattr(measurement, "set_A_override"):
        stats = measurement.set_A_override(A, metadata=metadata, rebuild_cache=True)
        return dict(stats) | {"override_mode": mode}
    A = A.to(device=measurement.device, dtype=torch.float32).contiguous()
    measurement.A = A
    measurement.m = int(A.shape[0])
    measurement.n = int(A.shape[1])
    measurement.sampling_ratio = float(measurement.m / measurement.n)
    eye = torch.eye(measurement.m, device=measurement.device, dtype=A.dtype)
    measurement.K = A @ A.T + float(measurement.lambda_dc) * eye
    if mode == "unsafe_old_chol":
        # Deliberately preserve the previous _chol to reproduce the old failure mode.
        pass
    else:
        measurement._chol = None
        measurement._use_cholesky = True
        try:
            measurement._chol = torch.linalg.cholesky(measurement.K)
        except RuntimeError:
            measurement._use_cholesky = False
    measurement.measurement_metadata = metadata
    return {
        "override_mode": mode,
        "m": measurement.m,
        "n": measurement.n,
        "sampling_ratio": measurement.sampling_ratio,
        "cache_rebuilt": mode != "unsafe_old_chol",
        "uses_cholesky": bool(getattr(measurement, "_use_cholesky", False)),
    }


def load_generator_for_eval(
    method_id: str,
    checkpoint_path: Path,
    measurement: Any,
    state_mode: str = "ema",
    device: torch.device | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from .models import build_generator

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = base_config_for(method_id, checkpoint_path)
    checkpoint = torch_load(checkpoint_path, device)
    generator = build_generator(config, measurement=measurement).to(device)
    state_key = "generator"
    if isinstance(checkpoint, dict):
        if state_mode == "ema" and checkpoint.get("generator_ema") is not None:
            state_key = "generator_ema"
        state = checkpoint.get(state_key) if checkpoint.get(state_key) is not None else checkpoint.get("generator")
    else:
        state = checkpoint
    load_result = generator.load_state_dict(state, strict=False)
    generator.eval()
    info = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "state_mode": state_mode,
        "state_key_used": state_key,
        "missing_keys": ";".join(load_result.missing_keys),
        "unexpected_keys": ";".join(load_result.unexpected_keys),
        "missing_key_count": len(load_result.missing_keys),
        "unexpected_key_count": len(load_result.unexpected_keys),
        "has_refine": hasattr(generator, "refine"),
    }
    return generator, config, info


def controlled_reconstruct(
    generator: Any,
    measurement: Any,
    y: torch.Tensor,
    *,
    use_null_project: bool,
    use_dc_project: bool,
    backprojection_mode: str | None,
    enable_refiner: bool,
    output_range_mode: str,
    noise_map_mode: str = "default",
    batch_idx: int = 0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    def finalize_image(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_unclamped = measurement.unflatten_img(flat)
        mode = str(output_range_mode or "clamp_eval_only").lower()
        if mode in {"clamp_eval_only", "clamp_after_dc"}:
            return image_unclamped, torch.clamp(image_unclamped, 0.0, 1.0)
        if mode == "sigmoid_before_dc":
            return image_unclamped, torch.sigmoid(image_unclamped)
        raise ValueError(f"Unsupported output_range_mode: {output_range_mode}")

    with torch.cuda.amp.autocast(enabled=False):
        y_fp32 = y.float()
        x_data_flat = measurement.data_solution(y_fp32, mode=backprojection_mode)
        x_data = measurement.unflatten_img(x_data_flat)
    if str(noise_map_mode) == "zero":
        noise_map = torch.zeros_like(x_data)
    elif str(noise_map_mode) == "fixed":
        gen = torch.Generator(device=x_data.device).manual_seed(int(seed) + int(batch_idx))
        noise_map = torch.randn(x_data.shape, device=x_data.device, dtype=x_data.dtype, generator=gen)
    else:
        noise_map = torch.randn_like(x_data)
    extras: dict[str, Any] = {"x_data_flat": x_data_flat, "noise_map_mode": noise_map_mode}
    residual = generator(x_data, noise_map, y=y)
    with torch.cuda.amp.autocast(enabled=False):
        residual_flat = measurement.flatten_img(residual.float())
        residual_ns_flat = measurement.null_project(residual_flat) if use_null_project else residual_flat
        x_tilde_flat = x_data_flat + residual_ns_flat
        if str(output_range_mode or "").lower() == "sigmoid_before_dc":
            x_tilde_flat = measurement.flatten_img(torch.sigmoid(measurement.unflatten_img(x_tilde_flat)))
        x_stage1_flat = measurement.dc_project(x_tilde_flat, y_fp32) if use_dc_project else x_tilde_flat
        x_stage1_unclamped, x_stage1 = finalize_image(x_stage1_flat)
    extras["residual"] = residual
    extras["x_stage1_unclamped"] = x_stage1_unclamped
    extras["x_stage1"] = x_stage1
    if enable_refiner and hasattr(generator, "refine"):
        refine_residual = generator.refine(x_data, x_stage1)
        with torch.cuda.amp.autocast(enabled=False):
            refine_flat = measurement.flatten_img(refine_residual.float())
            x_refine_tilde = x_stage1_flat + refine_flat
            if str(output_range_mode or "").lower() == "sigmoid_before_dc":
                x_refine_tilde = measurement.flatten_img(torch.sigmoid(measurement.unflatten_img(x_refine_tilde)))
            x_hat_flat = measurement.dc_project(x_refine_tilde, y_fp32) if use_dc_project else x_refine_tilde
            x_hat_unclamped, x_hat = finalize_image(x_hat_flat)
        extras["refine_residual"] = refine_residual
    else:
        x_hat_unclamped = x_stage1_unclamped
        x_hat = x_stage1
    extras["x_hat_unclamped"] = x_hat_unclamped
    extras["x_hat_metric"] = x_hat
    return x_hat, x_data, extras


def get_loader(config: dict[str, Any], split: str, device: torch.device):
    from .datasets import get_val_dataloader

    return get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=0,
        limit_val_samples=config.get("limit_val_samples"),
        seed=int(config["seed"]),
        val_split=split,
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )


def evaluate_variant(method_id: str, variant: dict[str, Any]) -> dict[str, Any]:
    from tqdm import tqdm

    from .metrics import batch_metrics
    from .utils import compare_metric_sets, mean_dict, set_seed

    output_dir = method_dir(method_id)
    metrics_colab = read_json(output_dir / "eval_metrics.json")
    checkpoint_name = variant.get("checkpoint", "best_hq.pt")
    checkpoint_path = primary_checkpoint(output_dir, checkpoint_name)
    config = base_config_for(method_id, checkpoint_path)
    config.update(variant.get("config_overrides", {}))
    device = torch.device(config["device"])
    set_seed(int(config["seed"]))
    measurement = make_measurement(config, device)
    a_source = variant.get("a_source", "exact")
    a_sha = ""
    override_info: dict[str, Any] = {}
    if a_source == "exact":
        A = load_exact_A(method_id, device)
        a_sha = tensor_sha256(A)
        override_info = apply_A_override(measurement, A, str(variant.get("a_override_mode", "safe_rebuild")))
    else:
        A = measurement.get_current_A()
        a_sha = tensor_sha256(A)
    generator, config, load_info = load_generator_for_eval(
        method_id,
        checkpoint_path,
        measurement,
        str(variant.get("state_mode", "ema")),
        device,
    )
    split = str(variant.get("split", "test"))
    loader = get_loader(config, split, device)
    backprojection_metrics = []
    model_metrics = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{method_id}:{variant['variant']}", leave=False)):
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            x_hat, x_data, extras = controlled_reconstruct(
                generator,
                measurement,
                y,
                use_null_project=bool(config["use_null_project"]),
                use_dc_project=bool(config["use_dc_project"]),
                backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=bool(variant.get("enable_refiner", True)),
                output_range_mode=str(variant.get("output_range_mode", config.get("output_range_mode", "clamp_eval_only"))),
                noise_map_mode=str(variant.get("noise_map_mode", "default")),
                batch_idx=batch_idx,
                seed=int(config["seed"]),
            )
            backprojection_metrics.append(batch_metrics(x_data, x, measurement, y))
            model_batch = batch_metrics(x_hat, x, measurement, y)
            model_batch["rel_meas_err_clamped"] = model_batch.get("rel_meas_error", float("nan"))
            model_batch["rel_meas_err_unclamped"] = batch_metrics(
                extras["x_hat_unclamped"], x, measurement, y
            ).get("rel_meas_error", float("nan"))
            model_metrics.append(model_batch)
    metrics = compare_metric_sets(mean_dict(backprojection_metrics), mean_dict(model_metrics))
    psnr = metric(metrics, "model", "psnr")
    ssim = metric(metrics, "model", "ssim")
    colab_psnr = metric(metrics_colab, "model", "psnr")
    colab_ssim = metric(metrics_colab, "model", "ssim")
    match = abs(psnr - colab_psnr) <= 0.05 and abs(ssim - colab_ssim) <= 0.005
    return {
        "method_id": method_id,
        "variant": variant["variant"],
        "psnr": psnr,
        "ssim": ssim,
        "mse": metric(metrics, "model", "mse"),
        "backproj_psnr": metric(metrics, "backprojection", "psnr"),
        "backproj_ssim": metric(metrics, "backprojection", "ssim"),
        "rel_meas_err": metric(metrics, "model", "rel_meas_error"),
        "colab_psnr": colab_psnr,
        "colab_ssim": colab_ssim,
        "diff_psnr": psnr - colab_psnr,
        "diff_ssim": ssim - colab_ssim,
        "checkpoint_used": checkpoint_path.name,
        "A_source": a_source,
        "A_sha": a_sha,
        "model_mode": variant.get("state_mode", "ema"),
        "refiner": bool(variant.get("enable_refiner", True)),
        "split": split,
        "noise_map_mode": variant.get("noise_map_mode", "default"),
        "output_range_mode": variant.get("output_range_mode", config.get("output_range_mode", "clamp_eval_only")),
        "status": "reproduced_variant" if match else "mismatch",
        "missing_key_count": load_info.get("missing_key_count", ""),
        "unexpected_key_count": load_info.get("unexpected_key_count", ""),
        "override_mode": override_info.get("override_mode", ""),
        "cache_rebuilt": override_info.get("cache_rebuilt", ""),
    }


def write_rows_all_formats(path_base: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path_base.parent)
    write_csv(path_base.with_suffix(".csv"), rows, fields)
    write_md_table(path_base.with_suffix(".md"), rows, fields)
    write_json(path_base.with_suffix(".json"), rows)


def summarize_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
