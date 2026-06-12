from __future__ import annotations

import json
import os
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str | os.PathLike[str]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_json(data: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2)
    return path


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def update_config_from_args(config: dict[str, Any], args, keys: list[str]) -> dict[str, Any]:
    updated = dict(config)
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            updated[key] = value
    return updated


def reconstruct_from_measurements(
    generator,
    measurement,
    y: torch.Tensor,
    use_null_project: bool = True,
    use_dc_project: bool = True,
    use_final_dc_project: bool | None = None,
    backprojection_mode: str | None = None,
    enable_refiner: bool = True,
    output_range_mode: str = "clamp_eval_only",
    return_extras: bool = False,
):
    def finalize_image(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_unclamped = measurement.unflatten_img(flat)
        mode = str(output_range_mode or "clamp_eval_only").lower()
        if mode in {"clamp_eval_only", "clamp_after_dc"}:
            return image_unclamped, torch.clamp(image_unclamped, 0.0, 1.0)
        if mode == "sigmoid_before_dc":
            return image_unclamped, torch.sigmoid(image_unclamped)
        raise ValueError(
            "output_range_mode must be one of: clamp_eval_only, clamp_after_dc, sigmoid_before_dc."
        )

    with torch.cuda.amp.autocast(enabled=False):
        y_fp32 = y.float()
        x_data_flat = measurement.data_solution(y_fp32, mode=backprojection_mode)
        x_data = measurement.unflatten_img(x_data_flat)
    final_dc_project = use_dc_project if use_final_dc_project is None else bool(use_final_dc_project)
    if str(backprojection_mode or "").lower() == "learned_backprojection" and hasattr(
        generator, "backprojection_enhancer"
    ):
        x_data = generator.backprojection_enhancer(x_data)
        x_data_flat = measurement.flatten_img(x_data)
    noise_map = torch.randn_like(x_data)
    extras: dict[str, Any] = {"x_data_flat": x_data_flat}

    output_kind = getattr(generator, "output_kind", "residual")
    if output_kind == "direct_image":
        x_direct = generator(x_data, noise_map, y=y)
        with torch.cuda.amp.autocast(enabled=False):
            x_tilde_flat = measurement.flatten_img(x_direct.float())
            if str(output_range_mode or "").lower() == "sigmoid_before_dc":
                x_tilde_flat = measurement.flatten_img(torch.sigmoid(x_direct.float()))
            x_hat_flat = measurement.dc_project(x_tilde_flat, y_fp32) if final_dc_project else x_tilde_flat
            x_hat_unclamped, x_hat = finalize_image(x_hat_flat)
        extras["x_direct"] = x_direct
        extras["pre_final_audit_flat"] = x_tilde_flat
        extras["pre_final_audit"] = measurement.unflatten_img(x_tilde_flat)
        extras["measurement_residual_pre_final"] = measurement.A_forward(x_tilde_flat) - y_fp32
        extras["final_dc_project"] = bool(final_dc_project)
        extras["x_hat_unclamped"] = x_hat_unclamped
        extras["x_hat_metric"] = x_hat
        extras["output_range_mode"] = str(output_range_mode or "clamp_eval_only")
        return (x_hat, x_data, extras) if return_extras else (x_hat, x_data)

    residual = generator(x_data, noise_map, y=y)
    with torch.cuda.amp.autocast(enabled=False):
        residual_flat = measurement.flatten_img(residual.float())
        residual_ns_flat = measurement.null_project(residual_flat) if use_null_project else residual_flat
        x_tilde_flat = x_data_flat + residual_ns_flat
        if str(output_range_mode or "").lower() == "sigmoid_before_dc":
            x_tilde_flat = measurement.flatten_img(torch.sigmoid(measurement.unflatten_img(x_tilde_flat)))
        x_stage1_flat = measurement.dc_project(x_tilde_flat, y_fp32) if use_dc_project else x_tilde_flat
        x_stage1_unclamped, x_stage1 = finalize_image(x_stage1_flat)
    extras["raw_residual"] = residual
    extras["raw_residual_flat"] = residual_flat
    extras["filtered_residual_flat"] = residual_ns_flat
    extras["filtered_residual"] = measurement.unflatten_img(residual_ns_flat)
    extras["pre_audit_flat"] = x_tilde_flat
    extras["pre_audit"] = measurement.unflatten_img(x_tilde_flat)
    extras["measurement_residual_pre"] = measurement.A_forward(x_tilde_flat) - y_fp32
    extras["measurement_residual_stage1"] = measurement.A_forward(x_stage1_flat) - y_fp32
    extras["x_stage1_unclamped"] = x_stage1_unclamped
    extras["x_stage1"] = x_stage1

    if enable_refiner and hasattr(generator, "refine"):
        refine_residual = generator.refine(x_data, x_stage1)
        with torch.cuda.amp.autocast(enabled=False):
            refine_flat = measurement.flatten_img(refine_residual.float())
            x_refine_tilde = x_stage1_flat + refine_flat
            if str(output_range_mode or "").lower() == "sigmoid_before_dc":
                x_refine_tilde = measurement.flatten_img(
                    torch.sigmoid(measurement.unflatten_img(x_refine_tilde))
                )
            x_hat_flat = measurement.dc_project(x_refine_tilde, y_fp32) if final_dc_project else x_refine_tilde
            x_hat_unclamped, x_hat = finalize_image(x_hat_flat)
        extras["refine_residual"] = refine_residual
        extras["pre_final_audit_flat"] = x_refine_tilde
        extras["pre_final_audit"] = measurement.unflatten_img(x_refine_tilde)
        extras["measurement_residual_pre_final"] = measurement.A_forward(x_refine_tilde) - y_fp32
    else:
        x_hat_unclamped = x_stage1_unclamped
        x_hat = x_stage1
        extras["pre_final_audit_flat"] = x_stage1_flat
        extras["pre_final_audit"] = x_stage1_unclamped
        extras["measurement_residual_pre_final"] = extras["measurement_residual_stage1"]
    extras["final_dc_project"] = bool(final_dc_project)
    extras["measurement_residual_post"] = measurement.A_forward(measurement.flatten_img(x_hat_unclamped.float())) - y_fp32
    extras["x_hat_unclamped"] = x_hat_unclamped
    extras["x_hat_metric"] = x_hat
    extras["output_range_mode"] = str(output_range_mode or "clamp_eval_only")
    return (x_hat, x_data, extras) if return_extras else (x_hat, x_data)


def apply_experiment_defaults(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    updated.setdefault("use_null_project", True)
    updated.setdefault("use_dc_project", True)
    updated.setdefault("use_final_dc_project", updated["use_dc_project"])
    updated.setdefault("use_adversarial", True)
    updated.setdefault("num_eval_samples_to_save", 8)
    updated.setdefault("num_individual_eval_samples", updated["num_eval_samples_to_save"])
    updated.setdefault("dataset_name", "stl10")
    updated.setdefault("class_filter", None)
    updated.setdefault("use_augmentation", False)
    updated.setdefault("model_type", "residual_unet_small")
    updated.setdefault("base_channels", 64)
    updated.setdefault("backprojection_mode", "ridge_pinv")
    updated.setdefault("matrix_normalization", "orthonormal_rows")
    updated.setdefault("hadamard_include_dc", True)
    updated.setdefault("hadamard_row_order", "sequency")
    updated.setdefault("hadamard_skip_dc", False)
    updated.setdefault("hadamard_random_column_permutation", False)
    updated.setdefault("hadamard_random_row_permutation", False)
    updated.setdefault("output_range_mode", "clamp_eval_only")
    updated.setdefault("hybrid_lowfreq_fraction", 0.7)
    updated.setdefault("use_learned_patterns", False)
    updated.setdefault("pattern_mode", "learned_binary_ste")
    updated.setdefault("pattern_init", "bernoulli")
    updated.setdefault("pattern_tau", 1.0)
    updated.setdefault("pattern_tau_final", updated["pattern_tau"])
    updated.setdefault("effective_A_mode", "centered_standardized")
    updated.setdefault("fixed_reference_pattern_type", "rademacher")
    updated.setdefault("fixed_reference_normalization", "row_norm_sqrt_n_over_m")
    updated.setdefault("target_transmission", 0.5)
    updated.setdefault("balanced_target_transmission", updated["target_transmission"])
    updated.setdefault("pattern_logit_abs_init", 2.0)
    updated.setdefault("flip_threshold", 0.5)
    updated.setdefault("flip_margin_target", 0.05)
    updated.setdefault("flip_warmup_epochs", 3)
    updated.setdefault("flip_noise_std", 0.0)
    updated.setdefault("flip_topk_fraction", None)
    updated.setdefault("flip_balance_rows", True)
    updated.setdefault("flip_commit_epoch", None)
    updated.setdefault("min_flip_fraction_target", 0.001)
    updated.setdefault("max_flip_fraction_target", 0.05)
    updated.setdefault("target_soft_flip_delta", 0.01)
    updated.setdefault("flip_margin_decay_epochs", 5)
    updated.setdefault("continuous_A_normalization", "row_standardized")
    updated.setdefault("continuous_min_contrast", 0.05)
    updated.setdefault("continuous_target_contrast", 0.25)
    updated.setdefault("continuous_max_contrast", 0.5)
    updated.setdefault("lr_patterns", 5e-5)
    updated.setdefault("train_patterns_after_epoch", 0)
    updated.setdefault("freeze_patterns_after_epoch", None)
    updated.setdefault("lambda_pattern_energy", 1.0)
    updated.setdefault("lambda_pattern_decorrelation", 0.1)
    updated.setdefault("lambda_pattern_binary", 0.01)
    updated.setdefault("lambda_pattern_secrip", 0.1)
    updated.setdefault("lambda_pattern_contrast", 0.0)
    updated.setdefault("lambda_pattern_bounded_contrast", 0.0)
    updated.setdefault("lambda_pattern_smoothness", 0.0)
    updated.setdefault("lambda_pattern_flip_margin", 0.0)
    updated.setdefault("lambda_pattern_soft_flip", 0.0)
    updated.setdefault("target_contrast", 0.45)
    updated.setdefault("save_patterns_every", 1)
    updated.setdefault("load_generator_checkpoint", None)
    updated.setdefault("load_discriminator_checkpoint", None)
    updated.setdefault("load_pattern_checkpoint", None)
    updated.setdefault("load_generator_strict", False)
    updated.setdefault("load_discriminator_strict", False)
    updated.setdefault("load_pattern_strict", False)
    updated.setdefault("reset_optimizers_after_load", True)
    updated.setdefault("freeze_generator_epochs", 0)
    updated.setdefault("freeze_discriminator_epochs", 0)
    updated.setdefault("pattern_update_every", 1)
    updated.setdefault("pattern_grad_clip_norm", 1.0)
    updated.setdefault("generator_grad_clip_norm", None)
    updated.setdefault("discriminator_grad_clip_norm", None)
    updated.setdefault("adv_warmup_epochs", 0)
    updated.setdefault("lambda_adv_final", None)
    updated.setdefault("score_ssim_weight", 10.0)
    updated.setdefault("score_relmeas_weight", 0.0)
    updated.setdefault("checkpoint_metric_mode", "score")
    updated.setdefault("lambda_charbonnier", 0.0)
    updated.setdefault("lambda_edge", 0.0)
    updated.setdefault("lambda_ms_l1", 0.0)
    updated.setdefault("lambda_ssim", 0.0)
    updated.setdefault("lambda_ms_ssim", 0.0)
    updated.setdefault("lambda_gradient", 0.0)
    updated.setdefault("lambda_frequency", 0.0)
    updated.setdefault("lambda_perceptual", 0.0)
    updated.setdefault("use_vgg_perceptual", False)
    updated.setdefault("lambda_stage1_aux", 0.0)
    updated.setdefault("use_amp", False)
    updated.setdefault("use_ema", False)
    updated.setdefault("ema_decay", 0.999)
    updated.setdefault("early_stop_patience", None)
    updated.setdefault("min_epochs", 0)
    updated.setdefault("save_top_k", 3)
    updated.setdefault("training_stage", {})
    updated.setdefault("eval_before_training", False)
    updated.setdefault("save_pattern_diagnostics", False)
    updated.setdefault("freeze_patterns", False)
    updated.setdefault("freeze_generator_all", False)
    updated.setdefault("freeze_discriminator_all", False)
    updated.setdefault("pattern_requires_grad", True)
    updated.setdefault("resume_checkpoint", None)
    updated.setdefault("resume_mode", "full")
    updated.setdefault("resume_strict", True)
    return updated


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def mean_dict(dicts: list[dict[str, float]]) -> dict[str, float]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {key: float(np.mean([d[key] for d in dicts if key in d])) for key in keys}


def compare_metric_sets(
    backprojection: dict[str, float], model: dict[str, float]
) -> dict[str, dict[str, float]]:
    improvement = {
        "delta_psnr": model.get("psnr", 0.0) - backprojection.get("psnr", 0.0),
        "delta_ssim": model.get("ssim", 0.0) - backprojection.get("ssim", 0.0),
        "delta_mse": backprojection.get("mse", 0.0) - model.get("mse", 0.0),
    }
    return {
        "backprojection": backprojection,
        "model": model,
        "improvement": improvement,
    }


def format_metric_comparison(metrics: dict[str, dict[str, float]]) -> str:
    back = metrics.get("backprojection", {})
    model = metrics.get("model", {})
    improve = metrics.get("improvement", {})
    return "\n".join(
        [
            "Backprojection:",
            f"  MSE: {back.get('mse', float('nan')):.6f}",
            f"  PSNR: {back.get('psnr', float('nan')):.6f}",
            f"  SSIM: {back.get('ssim', float('nan')):.6f}",
            f"  RelMeasErr: {back.get('rel_meas_error', float('nan')):.6f}",
            "",
            "NS-MC-GAN:",
            f"  MSE: {model.get('mse', float('nan')):.6f}",
            f"  PSNR: {model.get('psnr', float('nan')):.6f}",
            f"  SSIM: {model.get('ssim', float('nan')):.6f}",
            f"  RelMeasErr: {model.get('rel_meas_error', float('nan')):.6f}",
            "",
            "Improvement:",
            f"  delta_PSNR: {improve.get('delta_psnr', float('nan')):.6f}",
            f"  delta_SSIM: {improve.get('delta_ssim', float('nan')):.6f}",
            f"  delta_MSE: {improve.get('delta_mse', float('nan')):.6f}",
        ]
    )


def runtime_info(device: torch.device | None = None) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_name = "unknown"
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(device if device is not None else 0)
        except Exception:
            gpu_name = "unknown"
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def write_run_report(
    output_dir: str | os.PathLike[str],
    config: dict[str, Any],
    measurement,
    start_time: datetime,
    end_time: datetime,
    best_epoch: int | None,
    best_metrics: dict[str, dict[str, float]] | None,
    checkpoint_path: str | os.PathLike[str] | None,
    sample_path: str | os.PathLike[str] | None,
    sanity_path: str | os.PathLike[str] | None,
    device: torch.device,
    notes: list[str] | None = None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    notes = notes or []
    info = runtime_info(device)
    back = (best_metrics or {}).get("backprojection", {})
    model = (best_metrics or {}).get("model", {})
    improve = (best_metrics or {}).get("improvement", {})

    def metric_value(metrics: dict[str, float], key: str) -> str:
        value = metrics.get(key)
        if value is None:
            return "unknown"
        return f"{value:.6f}"

    lines = [
        "# Run Report",
        "",
        f"- Run start: {start_time.isoformat(timespec='seconds')}",
        f"- Run end: {end_time.isoformat(timespec='seconds')}",
        f"- Python version: {info['python_version']}",
        f"- PyTorch version: {info['pytorch_version']}",
        f"- CUDA available: {info['cuda_available']}",
        f"- GPU name: {info['gpu_name']}",
        f"- Dataset path: {config.get('dataset_root', 'unknown')}",
        f"- Output path: {config.get('output_dir', output_dir)}",
        f"- img_size: {config.get('img_size', 'unknown')}",
        f"- sampling_ratio: {config.get('sampling_ratio', 'unknown')}",
        f"- n: {getattr(measurement, 'n', 'unknown')}",
        f"- m: {getattr(measurement, 'm', 'unknown')}",
        f"- pattern_type: {config.get('pattern_type', 'unknown')}",
        f"- matrix_normalization: {config.get('matrix_normalization', 'unknown')}",
        f"- hadamard_include_dc: {config.get('hadamard_include_dc', 'unknown')}",
        f"- hadamard_row_order: {config.get('hadamard_row_order', 'unknown')}",
        f"- backprojection_mode: {config.get('backprojection_mode', 'unknown')}",
        f"- output_range_mode: {config.get('output_range_mode', 'unknown')}",
        f"- noise_std: {config.get('noise_std', 'unknown')}",
        f"- lambda_solver: {config.get('lambda_solver', 'unknown')}",
        f"- training epochs: {config.get('epochs', 'unknown')}",
        f"- resumed_from: {config.get('resume_checkpoint') or 'none'}",
        f"- resume_mode: {config.get('resume_mode', 'none')}",
        f"- best epoch: {best_epoch if best_epoch is not None else 'unknown'}",
        f"- best SSIM: {metric_value(model, 'ssim')}",
        f"- corresponding PSNR: {metric_value(model, 'psnr')}",
        "",
        "## Backprojection Baseline",
        "",
        f"- MSE: {metric_value(back, 'mse')}",
        f"- PSNR: {metric_value(back, 'psnr')}",
        f"- SSIM: {metric_value(back, 'ssim')}",
        f"- RelMeasErr: {metric_value(back, 'rel_meas_error')}",
        "",
        "## NS-MC-GAN",
        "",
        f"- MSE: {metric_value(model, 'mse')}",
        f"- PSNR: {metric_value(model, 'psnr')}",
        f"- SSIM: {metric_value(model, 'ssim')}",
        f"- RelMeasErr: {metric_value(model, 'rel_meas_error')}",
        "",
        "## Improvement",
        "",
        f"- delta_PSNR: {metric_value(improve, 'delta_psnr')}",
        f"- delta_SSIM: {metric_value(improve, 'delta_ssim')}",
        f"- delta_MSE: {metric_value(improve, 'delta_mse')}",
        "",
        "## Measurement Metadata",
        "",
    ]
    measurement_metadata = getattr(measurement, "measurement_metadata", None) or {}
    if measurement_metadata:
        for key, value in measurement_metadata.items():
            if key == "selected_rows":
                rows = value if isinstance(value, list) else []
                lines.append(f"- selected_rows_count: {len(rows)}")
                lines.append(f"- selected_rows_first_32: {rows[:32]}")
            elif key == "col_perm":
                lines.append(f"- col_perm: {'present' if value is not None else 'none'}")
            else:
                lines.append(f"- {key}: {value}")
    else:
        lines.append("- unavailable")
    lines.extend([
        "",
        f"- sanity_physics.json path: {sanity_path or 'unknown'}",
        f"- sample reconstruction path: {sample_path or 'unknown'}",
        f"- checkpoint path: {checkpoint_path or 'unknown'}",
        "",
        "## Notes And Fixes",
        "",
    ])
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- unknown")

    report_path = output_dir / "RUN_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value
