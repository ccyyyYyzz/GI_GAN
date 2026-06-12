from __future__ import annotations

import argparse
import copy
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from .datasets import get_dataloaders
from .exact_measurement import apply_measurement_override_from_config
from .losses import (
    charbonnier_loss,
    data_consistency_loss,
    differentiable_ssim_loss,
    discriminator_wgan_loss,
    frequency_loss,
    generator_adversarial_loss,
    gradient_difference_loss,
    gradient_penalty,
    multiscale_ssim_loss,
    reconstruction_loss,
    simple_multiscale_l1,
    sobel_edge_loss,
    total_variation_loss,
)
from .measurement import (
    GhostMeasurementOperator,
    LearnableGhostMeasurementOperator,
    LearnablePatternBank,
)
from .metrics import batch_metrics
from .models import PatchDiscriminator, build_generator
from .pattern_diagnostics import (
    capture_initial_pattern_state,
    compare_pattern_states,
    save_pattern_change_visualization,
    save_pattern_diagnostics_json,
)
from .pattern_regularizers import total_pattern_loss
from .pattern_utils import save_pattern_grid, save_pattern_stats_json
from .utils import (
    apply_experiment_defaults,
    compare_metric_sets,
    ensure_dir,
    format_metric_comparison,
    load_config,
    mean_dict,
    reconstruct_from_measurements,
    resolve_device,
    save_json,
    save_config,
    set_seed,
    update_config_from_args,
    write_run_report,
)
from .visualize import save_recon_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Train NS-MC-GAN for ghost imaging.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sampling_ratio", type=float, default=None)
    parser.add_argument("--pattern_type", default=None)
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--resume_mode", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def make_measurement(config: dict, device: torch.device) -> GhostMeasurementOperator:
    if bool(config.get("use_learned_patterns", False)):
        pattern_bank = LearnablePatternBank(
            img_size=config["img_size"],
            sampling_ratio=config["sampling_ratio"],
            pattern_mode=config.get("pattern_mode", "learned_binary_ste"),
            init_type=config.get("pattern_init", "bernoulli"),
            tau=config.get("pattern_tau", 1.0),
            target_transmission=config.get("target_transmission", 0.5),
            pattern_logit_abs_init=config.get("pattern_logit_abs_init", 2.0),
            balanced_target_transmission=(
                config.get("flip_topk_fraction")
                if config.get("flip_topk_fraction") is not None
                else config.get("balanced_target_transmission", config.get("target_transmission", 0.5))
            ),
            effective_A_mode=config.get("effective_A_mode", "centered_standardized"),
            fixed_reference_pattern_type=config.get("fixed_reference_pattern_type", "rademacher"),
            fixed_reference_normalization=config.get(
                "fixed_reference_normalization", "row_norm_sqrt_n_over_m"
            ),
            flip_threshold=config.get("flip_threshold", 0.5),
            flip_noise_std=config.get("flip_noise_std", 0.0),
            flip_balance_rows=config.get("flip_balance_rows", True),
            continuous_A_normalization=config.get("continuous_A_normalization", "row_standardized"),
            continuous_min_contrast=config.get("continuous_min_contrast", 0.05),
            continuous_target_contrast=config.get("continuous_target_contrast", 0.25),
            continuous_max_contrast=config.get("continuous_max_contrast", 0.5),
            device=device,
            seed=config["seed"],
        ).to(device)
        return LearnableGhostMeasurementOperator(
            pattern_bank=pattern_bank,
            noise_std=config["noise_std"],
            lambda_dc=config["lambda_solver"],
            backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
            device=device,
        )
    return GhostMeasurementOperator(
        img_size=config["img_size"],
        sampling_ratio=config["sampling_ratio"],
        pattern_type=config["pattern_type"],
        noise_std=config["noise_std"],
        lambda_dc=config["lambda_solver"],
        backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
        matrix_normalization=config.get("matrix_normalization", "orthonormal_rows"),
        hadamard_include_dc=config.get("hadamard_include_dc", True),
        hadamard_row_order=config.get("hadamard_row_order", "sequency"),
        hadamard_skip_dc=config.get("hadamard_skip_dc", False),
        hadamard_random_column_permutation=config.get("hadamard_random_column_permutation", False),
        hadamard_random_row_permutation=config.get("hadamard_random_row_permutation", False),
        hybrid_lowfreq_fraction=config.get("hybrid_lowfreq_fraction", 0.7),
        device=device,
        seed=config["seed"],
    )


@torch.no_grad()
def evaluate(
    generator,
    val_loader,
    measurement,
    device: torch.device,
    config: dict,
    sample_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    generator.eval()
    pattern_bank = getattr(measurement, "pattern_bank", None)
    pattern_was_training = bool(pattern_bank.training) if pattern_bank is not None else False
    if pattern_bank is not None:
        pattern_bank.eval()
    backprojection_metrics = []
    model_metrics = []
    for batch_idx, batch in enumerate(val_loader):
        x = batch[0].to(device, non_blocking=True)
        y = measurement.measure(x)
        x_hat, x_data, extras = reconstruct_from_measurements(
            generator,
            measurement,
            y,
            use_null_project=bool(config["use_null_project"]),
            use_dc_project=bool(config["use_dc_project"]),
            use_final_dc_project=bool(config.get("use_final_dc_project", config["use_dc_project"])),
            backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
            enable_refiner=True,
            output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
            return_extras=True,
        )
        backprojection_metrics.append(batch_metrics(x_data, x, measurement, y))
        current_model_metrics = batch_metrics(x_hat, x, measurement, y)
        current_model_metrics["rel_meas_err_clamped"] = current_model_metrics.get(
            "rel_meas_error", float("nan")
        )
        current_model_metrics["rel_meas_err_unclamped"] = batch_metrics(
            extras["x_hat_unclamped"], x, measurement, y
        ).get("rel_meas_error", float("nan"))
        model_metrics.append(current_model_metrics)
        if batch_idx == 0 and sample_path is not None:
            title = (
                f"sampling={measurement.sampling_ratio:.2%} | "
                f"PSNR={current_model_metrics['psnr']:.3f} | "
                f"SSIM={current_model_metrics['ssim']:.3f} | "
                f"RelMeasErr={current_model_metrics['rel_meas_error']:.3g}"
            )
            save_recon_grid(
                x,
                x_data,
                x_hat,
                sample_path,
                max_items=int(config.get("num_eval_samples_to_save", 8)),
                title=title,
            )
    generator.train()
    if pattern_bank is not None and pattern_was_training:
        pattern_bank.train()
    metrics = compare_metric_sets(mean_dict(backprojection_metrics), mean_dict(model_metrics))
    if hasattr(measurement, "get_pattern_stats") and bool(config.get("use_learned_patterns", False)):
        metrics["pattern"] = measurement.get_pattern_stats()
    return metrics


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(value)


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        set_requires_grad(self.module, False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            src = model_state[key].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                value.copy_(src)


def make_summary_writer(log_dir: Path):
    if not torch_numpy_bridge_available():
        print("TensorBoard is unavailable; Torch NumPy bridge is not initialized.")
        return NullSummaryWriter()
    try:
        import os

        os.environ.setdefault("TENSORBOARD_NO_TENSORFLOW", "1")
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:
        print(f"TensorBoard is unavailable; continuing without TB logs. Reason: {exc}")
        return NullSummaryWriter()


def torch_numpy_bridge_available() -> bool:
    try:
        torch.zeros(1).cpu().numpy()
        return True
    except Exception:
        return False


def current_tau(config: dict, epoch: int) -> float:
    start = float(config.get("pattern_tau", 1.0))
    final = float(config.get("pattern_tau_final", start))
    total_epochs = max(1, int(config.get("epochs", 1)) - 1)
    if total_epochs <= 1:
        return final
    t = min(1.0, max(0.0, (epoch - 1) / total_epochs))
    return start + (final - start) * t


def pattern_updates_enabled(config: dict, epoch: int) -> bool:
    if not bool(config.get("use_learned_patterns", False)):
        return False
    if bool(config.get("freeze_patterns", False)):
        return False
    if not bool(config.get("pattern_requires_grad", True)):
        return False
    if float(config.get("lr_patterns", 0.0)) <= 0.0:
        return False
    if epoch < int(config.get("train_patterns_after_epoch", 0)):
        return False
    freeze_after = config.get("freeze_patterns_after_epoch")
    if freeze_after is not None and epoch >= int(freeze_after):
        return False
    return True


def current_lambda_adv(config: dict, epoch: int) -> float:
    if not bool(config.get("use_adversarial", True)):
        return 0.0
    stage = config.get("training_stage") or {}
    if isinstance(stage, dict):
        adv_start = int(stage.get("adversarial_start_epoch", 0) or 0)
        if adv_start > 0 and epoch < adv_start:
            return 0.0
    base = float(config.get("lambda_adv", 0.0))
    final = config.get("lambda_adv_final")
    if final is not None:
        total_epochs = max(1, int(config.get("epochs", 1)) - 1)
        progress = min(1.0, max(0.0, (epoch - 1) / total_epochs))
        base = float(final) * progress
    warmup = int(config.get("adv_warmup_epochs", 0))
    if warmup > 0 and epoch <= warmup:
        return 0.0
    return base


def current_lambda_flip_margin(config: dict, epoch: int) -> float:
    base = float(config.get("lambda_pattern_flip_margin", 0.0))
    decay_epochs = int(config.get("flip_margin_decay_epochs", 0) or 0)
    if decay_epochs <= 0:
        return base
    progress = min(1.0, max(0.0, (epoch - 1) / max(1, decay_epochs)))
    return base * (1.0 - progress)


def metric_score(metrics: dict, ssim_weight: float) -> float:
    model = metrics.get("model", {})
    return float(model.get("psnr", float("-inf"))) + float(ssim_weight) * float(
        model.get("ssim", float("-inf"))
    )


def metric_hq_score(metrics: dict, config: dict) -> float:
    model = metrics.get("model", {})
    return (
        float(model.get("psnr", float("-inf")))
        + float(config.get("score_ssim_weight", 20.0)) * float(model.get("ssim", float("-inf")))
        - float(config.get("score_relmeas_weight", 0.0))
        * float(model.get("rel_meas_error", 0.0))
    )


def checkpoint_metric_values(metrics: dict, config: dict) -> dict[str, float]:
    model = metrics.get("model", {})
    return {
        "ssim": float(model.get("ssim", float("-inf"))),
        "psnr": float(model.get("psnr", float("-inf"))),
        "mse": float(model.get("mse", float("inf"))),
        "score": metric_score(metrics, float(config.get("score_ssim_weight", 10.0))),
        "hq": metric_hq_score(metrics, config),
    }


def is_better_checkpoint(name: str, value: float, current: float) -> bool:
    if name == "mse":
        return value < current
    return value > current


def compute_grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().data.norm(2).item()
        total += grad_norm * grad_norm
    return total ** 0.5


def clip_gradients(module: torch.nn.Module, max_norm) -> tuple[float | None, float | None]:
    if max_norm is None:
        return None, None
    params = [param for param in module.parameters() if param.grad is not None]
    if not params:
        return 0.0, 0.0
    before = float(torch.nn.utils.clip_grad_norm_(params, float(max_norm)))
    after = compute_grad_norm(params)
    return before, after


def load_state_from_checkpoint(
    module: torch.nn.Module,
    checkpoint_path: str | None,
    state_key: str,
    strict: bool,
    device: torch.device,
) -> list[str]:
    if not checkpoint_path:
        return []
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or state_key not in checkpoint:
        raise RuntimeError(f"{checkpoint_path} does not contain '{state_key}' state_dict.")
    result = module.load_state_dict(checkpoint[state_key], strict=bool(strict))
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    return [
        f"Loaded {state_key} from {checkpoint_path}",
        f"{state_key} missing keys: {missing if missing else 'none'}",
        f"{state_key} unexpected keys: {unexpected if unexpected else 'none'}",
    ]


def save_pattern_artifacts(pattern_bank, output_dir: Path, epoch: int, config: dict) -> dict:
    pattern_dir = ensure_dir(output_dir / "patterns")
    stats = pattern_bank.get_pattern_stats()
    max_patterns = int(config.get("num_eval_samples_to_save", 8)) * 4
    with torch.no_grad():
        P = pattern_bank.get_physical_patterns()
        save_pattern_grid(
            P,
            pattern_dir / f"epoch_{epoch:03d}_patterns.png",
            pattern_bank.img_size,
            max_patterns=max_patterns,
        )
        if pattern_bank.pattern_mode in {
            "learned_binary_ste",
            "learned_balanced_binary_ste",
            "learned_flip_aware_binary_ste",
        }:
            P_soft = pattern_bank.get_soft_patterns()
            save_pattern_grid(
                P_soft,
                pattern_dir / f"epoch_{epoch:03d}_patterns_soft.png",
                pattern_bank.img_size,
                max_patterns=max_patterns,
            )
            save_pattern_grid(
                pattern_bank.get_hard_patterns(),
                pattern_dir / f"epoch_{epoch:03d}_patterns_hard.png",
                pattern_bank.img_size,
                max_patterns=max_patterns,
            )
    save_pattern_stats_json(stats, pattern_dir / f"epoch_{epoch:03d}_pattern_stats.json")
    return stats


def save_pattern_diagnostics_artifacts(
    pattern_bank,
    output_dir: Path,
    initial_pattern_state: dict | None,
    config: dict,
    *,
    epoch: int | None,
    secant_batch: torch.Tensor | None,
) -> dict:
    diagnostics = compare_pattern_states(
        pattern_bank,
        initial_pattern_state,
        secant_batch=secant_batch,
        config=config,
    )
    diag_dir = ensure_dir(output_dir / "pattern_diagnostics")
    if epoch is not None:
        save_pattern_diagnostics_json(
            diagnostics,
            diag_dir / f"epoch_{epoch:03d}_pattern_diagnostics.json",
        )
    save_pattern_diagnostics_json(diagnostics, diag_dir / "pattern_diagnostics.json")
    if initial_pattern_state and diagnostics.get("status") == "ok":
        with torch.no_grad():
            save_pattern_change_visualization(
                initial_pattern_state["P_initial_hard"],
                pattern_bank.get_hard_patterns().detach().cpu(),
                diag_dir / "pattern_change_grid.png",
                pattern_bank.img_size,
                max_patterns=int(config.get("pattern_diagnostics_max_patterns", 32)),
            )
    return diagnostics


def save_checkpoint(
    path: Path,
    generator,
    discriminator,
    opt_g,
    opt_d,
    epoch,
    config,
    metrics,
    pattern_bank=None,
    opt_patterns=None,
    initial_pattern_state=None,
    generator_ema=None,
    scaler=None,
):
    payload = {
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
    }
    if generator_ema is not None:
        payload["generator_ema"] = generator_ema.state_dict()
    if hasattr(generator, "refiner"):
        payload["refiner"] = generator.refiner.state_dict()
    if pattern_bank is not None:
        payload["pattern_bank"] = pattern_bank.state_dict()
        payload["pattern_stats"] = pattern_bank.get_pattern_stats()
        if initial_pattern_state is not None:
            payload["initial_pattern_state"] = initial_pattern_state
    if opt_patterns is not None:
        payload["optimizer_patterns"] = opt_patterns.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_optimizer_state(optimizer, checkpoint: dict, key: str, strict: bool) -> list[str]:
    if optimizer is None:
        return []
    if key not in checkpoint:
        if strict:
            raise RuntimeError(f"Resume checkpoint does not contain '{key}'.")
        return [f"Resume checkpoint is missing {key}; optimizer state was not restored."]
    optimizer.load_state_dict(checkpoint[key])
    return [f"Loaded {key} from resume checkpoint."]


def load_resume_checkpoint(
    checkpoint_path: str | None,
    generator,
    discriminator,
    opt_g,
    opt_d,
    pattern_bank,
    opt_patterns,
    scaler,
    device: torch.device,
    strict: bool,
    resume_mode: str = "full",
) -> tuple[int, dict[str, str], list[str]]:
    if not checkpoint_path:
        return 1, {}, []
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{checkpoint_path} is not a training checkpoint.")
    mode = str(resume_mode or "full").lower()
    if mode not in {"full", "weights_only", "ema_only"}:
        raise ValueError("resume_mode must be one of: full, weights_only, ema_only.")
    required = ["epoch"]
    if mode == "ema_only":
        required.append("generator_ema")
    else:
        required.append("generator")
    if mode == "full":
        required.append("discriminator")
    for key in required:
        if key not in checkpoint:
            raise RuntimeError(f"{checkpoint_path} does not contain '{key}'.")

    if mode == "ema_only":
        generator.load_state_dict(checkpoint["generator_ema"], strict=bool(strict))
    else:
        generator.load_state_dict(checkpoint["generator"], strict=bool(strict))
    if mode == "full":
        discriminator.load_state_dict(checkpoint["discriminator"], strict=bool(strict))
    if pattern_bank is not None:
        if "pattern_bank" not in checkpoint and mode == "full":
            raise RuntimeError(f"{checkpoint_path} does not contain 'pattern_bank'.")
        if "pattern_bank" in checkpoint:
            pattern_bank.load_state_dict(checkpoint["pattern_bank"], strict=bool(strict))
    notes = [
        f"Resumed from {checkpoint_path} with resume_mode={mode}.",
    ]
    if mode == "full":
        notes.append(
            f"Resumed epoch={checkpoint['epoch']}; training will continue at epoch={int(checkpoint['epoch']) + 1}."
        )
        notes.extend(load_optimizer_state(opt_g, checkpoint, "optimizer_g", bool(strict)))
        notes.extend(load_optimizer_state(opt_d, checkpoint, "optimizer_d", bool(strict)))
        if opt_patterns is not None:
            notes.extend(
                load_optimizer_state(opt_patterns, checkpoint, "optimizer_patterns", bool(strict))
            )
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
            notes.append("Loaded AMP scaler from resume checkpoint.")
        elif scaler is not None:
            notes.append("Resume checkpoint has no AMP scaler; scaler was reinitialized.")
        return int(checkpoint["epoch"]) + 1, checkpoint, notes

    notes.append("Loaded weights only; optimizer state and epoch cursor were not restored.")
    return 1, checkpoint, notes


def update_best_checkpoints(
    metrics: dict,
    epoch: int,
    config: dict,
    output_dir: Path,
    generator,
    discriminator,
    opt_g,
    opt_d,
    pattern_bank,
    opt_patterns,
    best_values: dict[str, float],
    best_records: dict[str, dict],
    sample_path: Path | None,
    initial_pattern_state=None,
    generator_ema=None,
    scaler=None,
) -> None:
    values = checkpoint_metric_values(metrics, config)
    for name, value in values.items():
        if is_better_checkpoint(name, value, best_values[name]):
            best_values[name] = value
            ckpt_path = output_dir / f"best_{name}.pt"
            save_checkpoint(
                ckpt_path,
                generator,
                discriminator,
                opt_g,
                opt_d,
                epoch,
                config,
                metrics,
                pattern_bank=pattern_bank,
                opt_patterns=opt_patterns,
                initial_pattern_state=initial_pattern_state,
                generator_ema=generator_ema,
                scaler=scaler,
            )
            best_records[name] = {
                "epoch": epoch,
                "metrics": metrics,
                "path": ckpt_path,
                "sample": sample_path,
            }
            save_json(metrics, output_dir / f"best_{name}_metrics.json")
            if name == "ssim":
                save_json(metrics, output_dir / "best_metrics.json")


def append_eval_history(
    output_dir: Path,
    epoch: int,
    metrics: dict,
    train_g: float | None = None,
    train_d: float | None = None,
) -> None:
    path = output_dir / "eval_history.csv"
    back = metrics.get("backprojection", {})
    model = metrics.get("model", {})
    improve = metrics.get("improvement", {})
    row = {
        "epoch": epoch,
        "train_g_loss": "" if train_g is None else float(train_g),
        "train_d_loss": "" if train_d is None else float(train_d),
        "backproj_mse": back.get("mse", ""),
        "backproj_psnr": back.get("psnr", ""),
        "backproj_ssim": back.get("ssim", ""),
        "backproj_rel_meas_error": back.get("rel_meas_error", ""),
        "model_mse": model.get("mse", ""),
        "model_psnr": model.get("psnr", ""),
        "model_ssim": model.get("ssim", ""),
        "model_rel_meas_error": model.get("rel_meas_error", ""),
        "rel_meas_err_unclamped": model.get("rel_meas_err_unclamped", ""),
        "rel_meas_err_clamped": model.get("rel_meas_err_clamped", ""),
        "delta_psnr": improve.get("delta_psnr", ""),
        "delta_ssim": improve.get("delta_ssim", ""),
        "delta_mse": improve.get("delta_mse", ""),
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def mean_or_blank(values: list[float]) -> float | str:
    return "" if not values else float(sum(values) / len(values))


def append_per_epoch_metrics(
    output_dir: Path,
    epoch: int,
    metrics: dict,
    train_parts: dict[str, list[float]],
    config: dict,
    best_records: dict[str, dict],
    elapsed_sec: float,
    train_d: float | None = None,
) -> None:
    path = output_dir / "per_epoch_metrics.csv"
    back = metrics.get("backprojection", {}) if metrics else {}
    model = metrics.get("model", {}) if metrics else {}
    hq_score = ""
    if model:
        hq_score = (
            float(model.get("psnr", 0.0))
            + float(config.get("score_ssim_weight", 20.0)) * float(model.get("ssim", 0.0))
            - float(config.get("score_relmeas_weight", 0.0))
            * float(model.get("rel_meas_error", 0.0))
        )
    row = {
        "epoch": epoch,
        "train_total_loss": mean_or_blank(train_parts.get("total", [])),
        "train_l1": mean_or_blank(train_parts.get("l1", [])),
        "train_dc_loss": mean_or_blank(train_parts.get("dc", [])),
        "train_ssim_loss": mean_or_blank(train_parts.get("ssim", [])),
        "train_ms_ssim_loss": mean_or_blank(train_parts.get("ms_ssim", [])),
        "train_edge_loss": mean_or_blank(train_parts.get("edge", [])),
        "train_frequency_loss": mean_or_blank(train_parts.get("frequency", [])),
        "train_d_loss": "" if train_d is None else float(train_d),
        "val_model_psnr": model.get("psnr", ""),
        "val_model_ssim": model.get("ssim", ""),
        "val_model_mse": model.get("mse", ""),
        "val_model_rel_meas_err": model.get("rel_meas_error", ""),
        "rel_meas_err_unclamped": model.get("rel_meas_err_unclamped", ""),
        "rel_meas_err_clamped": model.get("rel_meas_err_clamped", ""),
        "val_backproj_psnr": back.get("psnr", ""),
        "val_backproj_ssim": back.get("ssim", ""),
        "val_backproj_mse": back.get("mse", ""),
        "hq_score": hq_score,
        "checkpoint_best_hq": best_records.get("hq", {}).get("path", ""),
        "time_elapsed_sec": float(elapsed_sec),
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_existing_best_records(
    output_dir: Path,
    config: dict,
    device: torch.device,
    best_values: dict[str, float],
    best_records: dict[str, dict],
    strict: bool,
) -> list[str]:
    notes = []
    for name in ["ssim", "psnr", "mse", "score", "hq"]:
        ckpt_path = output_dir / f"best_{name}.pt"
        if not ckpt_path.exists():
            if strict:
                raise RuntimeError(f"Expected existing best checkpoint for resume: {ckpt_path}")
            notes.append(f"Missing existing best checkpoint for {name}: {ckpt_path}")
            continue
        checkpoint = torch.load(ckpt_path, map_location=device)
        if not isinstance(checkpoint, dict) or "metrics" not in checkpoint:
            if strict:
                raise RuntimeError(f"{ckpt_path} does not contain checkpoint metrics.")
            notes.append(f"Could not read metrics from {ckpt_path}")
            continue
        metrics = checkpoint.get("metrics") or {}
        value = checkpoint_metric_values(metrics, config)[name]
        epoch = checkpoint.get("epoch")
        sample = None
        if isinstance(epoch, int):
            candidate = output_dir / "samples" / f"epoch_{epoch:03d}.png"
            if candidate.exists():
                sample = candidate
        best_values[name] = value
        best_records[name] = {
            "epoch": epoch,
            "metrics": metrics,
            "path": ckpt_path,
            "sample": sample,
        }
        notes.append(
            f"Recovered best {name} from {ckpt_path}: epoch={epoch}, value={value}"
        )
    return notes


def main() -> None:
    start_time = datetime.now()
    train_start_perf = time.time()
    args = parse_args()
    config = load_config(args.config)
    config = apply_experiment_defaults(config)
    config = update_config_from_args(
        config,
        args,
        [
            "device",
            "sampling_ratio",
            "pattern_type",
            "noise_std",
            "dataset_root",
            "output_dir",
            "epochs",
            "batch_size",
            "resume_checkpoint",
            "resume_mode",
        ],
    )
    config = apply_experiment_defaults(config)
    if args.debug:
        config["epochs"] = 2
        config["batch_size"] = min(int(config["batch_size"]), 16)
        config["num_workers"] = 0
        config["limit_train_samples"] = 200
        config["limit_val_samples"] = 20

    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])
    output_dir = ensure_dir(config["output_dir"])
    sample_dir = ensure_dir(output_dir / "samples")
    log_dir = ensure_dir(output_dir / "tb")
    save_config(config, output_dir / "resolved_config.yaml")
    run_notes = [
        "Phase 2 uses baseline validation, physics sanity checks, ablation switches, and run reporting.",
        "TensorBoard, skimage, and matplotlib imports are guarded so local package issues do not stop core training.",
        (
            "Ablation flags: "
            f"use_null_project={config['use_null_project']}, "
            f"use_dc_project={config['use_dc_project']}, "
            f"use_final_dc_project={config.get('use_final_dc_project', config['use_dc_project'])}, "
            f"use_adversarial={config['use_adversarial']}."
        ),
    ]
    if str(config["device"]) != str(device):
        run_notes.append(f"Requested device {config['device']} resolved to {device}.")
    adjustment_notes = config.get("phase2_adjustment_notes", config.get("phase2_adjustment_note"))
    if isinstance(adjustment_notes, list):
        run_notes.extend(str(note) for note in adjustment_notes)
    elif adjustment_notes:
        run_notes.append(str(adjustment_notes))

    measurement = make_measurement(config, device)
    exact_a_info = apply_measurement_override_from_config(config, measurement, device)
    if exact_a_info.get("exact_A_loaded"):
        run_notes.append(
            "Loaded exact measurement operator with safe cache rebuild: "
            f"{exact_a_info.get('exact_A_path')}"
        )
    elif exact_a_info.get("exact_A_required"):
        run_notes.append("Exact-A was required but not loaded.")
    pattern_bank = getattr(measurement, "pattern_bank", None)
    if pattern_bank is not None:
        run_notes.append(
            "Phase 3 learned physical patterns are optimized through centered, standardized A_eff."
        )
        run_notes.append(
            "Null-space projection and measurement-consistency projection remain enabled unless explicitly ablated."
        )
    print(
        "Training setup: "
        f"device={device}, img_size={measurement.img_size}, n={measurement.n}, "
        f"m={measurement.m}, sampling_ratio={measurement.sampling_ratio}, "
        f"pattern_type={measurement.pattern_type}, noise_std={measurement.noise_std}"
    )

    pin_memory = device.type == "cuda"
    train_loader, val_loader = get_dataloaders(
        dataset_root=config["dataset_root"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        limit_train_samples=config["limit_train_samples"],
        limit_val_samples=config["limit_val_samples"],
        seed=config["seed"],
        pin_memory=pin_memory,
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
        use_augmentation=bool(config.get("use_augmentation", False)),
    )

    generator = build_generator(config, measurement=measurement).to(device)
    discriminator = PatchDiscriminator().to(device)
    load_notes = []
    load_notes.extend(
        load_state_from_checkpoint(
            generator,
            config.get("load_generator_checkpoint"),
            "generator",
            bool(config.get("load_generator_strict", False)),
            device,
        )
    )
    load_notes.extend(
        load_state_from_checkpoint(
            discriminator,
            config.get("load_discriminator_checkpoint"),
            "discriminator",
            bool(config.get("load_discriminator_strict", False)),
            device,
        )
    )
    if pattern_bank is not None and config.get("load_pattern_checkpoint"):
        load_notes.extend(
            load_state_from_checkpoint(
                pattern_bank,
                config.get("load_pattern_checkpoint"),
                "pattern_bank",
                bool(config.get("load_pattern_strict", False)),
                device,
            )
        )
    elif config.get("load_pattern_checkpoint"):
        raise RuntimeError("load_pattern_checkpoint was provided but use_learned_patterns=false.")
    if load_notes:
        print("\n".join(load_notes))
        run_notes.extend(load_notes)

    initial_pattern_state = (
        capture_initial_pattern_state(pattern_bank)
        if pattern_bank is not None and bool(config.get("save_pattern_diagnostics", False))
        else None
    )
    patterns_frozen_all = bool(config.get("freeze_patterns", False)) or not bool(
        config.get("pattern_requires_grad", True)
    )
    if pattern_bank is not None and patterns_frozen_all:
        set_requires_grad(pattern_bank, False)
        note = "Pattern bank is frozen for the full run."
        print(note)
        run_notes.append(note)

    betas = tuple(config["betas"])
    opt_g = optim.Adam(generator.parameters(), lr=config["lr_g"], betas=betas)
    opt_d = optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=betas)
    opt_patterns = None
    if pattern_bank is not None and not patterns_frozen_all and float(config.get("lr_patterns", 0.0)) > 0.0:
        opt_patterns = optim.Adam(
            pattern_bank.parameters(),
            lr=float(config.get("lr_patterns", 5e-5)),
            betas=betas,
        )
    if not bool(config.get("reset_optimizers_after_load", True)):
        for opt, opt_key, ckpt_path in [
            (opt_g, "optimizer_g", config.get("load_generator_checkpoint")),
            (opt_d, "optimizer_d", config.get("load_discriminator_checkpoint")),
            (opt_patterns, "optimizer_patterns", config.get("load_pattern_checkpoint")),
        ]:
            if opt is None or not ckpt_path:
                continue
            checkpoint = torch.load(ckpt_path, map_location=device)
            if isinstance(checkpoint, dict) and opt_key in checkpoint:
                opt.load_state_dict(checkpoint[opt_key])
                note = f"Loaded {opt_key} from {ckpt_path}"
                print(note)
                run_notes.append(note)
    use_amp = bool(config.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if bool(config.get("use_amp", False)) and not use_amp:
        run_notes.append("AMP requested but disabled because CUDA is unavailable.")
    ema = ModelEMA(generator, decay=float(config.get("ema_decay", 0.999))) if bool(config.get("use_ema", False)) else None
    if ema is not None:
        run_notes.append(f"EMA enabled with decay={float(config.get('ema_decay', 0.999))}.")
    best_values = {
        "ssim": float("-inf"),
        "psnr": float("-inf"),
        "mse": float("inf"),
        "score": float("-inf"),
        "hq": float("-inf"),
    }
    best_records = {
        name: {"epoch": None, "metrics": None, "path": output_dir / f"best_{name}.pt", "sample": None}
        for name in best_values
    }
    global_step = 0
    start_epoch_index = 1

    if config.get("resume_checkpoint"):
        start_epoch_index, resume_checkpoint, resume_notes = load_resume_checkpoint(
            config.get("resume_checkpoint"),
            generator,
            discriminator,
            opt_g,
            opt_d,
            pattern_bank,
            opt_patterns,
            scaler,
            device,
            bool(config.get("resume_strict", True)),
            config.get("resume_mode", "full"),
        )
        run_notes.extend(resume_notes)
        if ema is not None:
            if isinstance(resume_checkpoint, dict) and "generator_ema" in resume_checkpoint:
                ema.module.load_state_dict(resume_checkpoint["generator_ema"], strict=bool(config.get("resume_strict", True)))
                run_notes.append("Resumed generator EMA state.")
            else:
                ema = ModelEMA(generator, decay=float(config.get("ema_decay", 0.999)))
                run_notes.append("Resume checkpoint had no EMA state; initialized EMA from generator.")
        global_step = max(0, (start_epoch_index - 1) * len(train_loader))
        if str(config.get("resume_mode", "full")).lower() == "full":
            recovered_notes = load_existing_best_records(
                output_dir,
                config,
                device,
                best_values,
                best_records,
                bool(config.get("resume_strict", True)),
            )
            run_notes.extend(recovered_notes)
    run_notes.append(f"start_epoch: {start_epoch_index}")
    run_notes.append(f"total_epochs_target: {config['epochs']}")

    writer = make_summary_writer(log_dir)
    pattern_audit_batch = None
    if pattern_bank is not None and bool(config.get("save_pattern_diagnostics", False)):
        try:
            pattern_audit_batch = next(iter(val_loader))[0].to(device, non_blocking=True)
        except StopIteration:
            pattern_audit_batch = None
        diagnostics0 = save_pattern_diagnostics_artifacts(
            pattern_bank,
            output_dir,
            initial_pattern_state,
            config,
            epoch=0,
            secant_batch=pattern_audit_batch,
        )
        run_notes.append(f"Initial pattern diagnostics: {diagnostics0.get('pattern_attribution_note')}")

    if bool(config.get("eval_before_training", False)) and start_epoch_index <= 1:
        epoch0_sample_path = sample_dir / "epoch_000.png"
        eval_generator = ema.module if ema is not None else generator
        metrics0 = evaluate(
            eval_generator,
            val_loader,
            measurement,
            device,
            config,
            sample_path=epoch0_sample_path,
        )
        print("Epoch 0 validation comparison:")
        print(format_metric_comparison(metrics0))
        save_json(metrics0, output_dir / "eval_epoch000_metrics.json")
        append_eval_history(output_dir, 0, metrics0)
        if pattern_bank is not None:
            save_pattern_artifacts(pattern_bank, output_dir, 0, config)
            if initial_pattern_state is not None:
                metrics0["pattern_diagnostics"] = compare_pattern_states(
                    pattern_bank,
                    initial_pattern_state,
                    secant_batch=pattern_audit_batch,
                    config=config,
                )
        update_best_checkpoints(
            metrics0,
            0,
            config,
            output_dir,
            generator,
            discriminator,
            opt_g,
            opt_d,
            pattern_bank,
            opt_patterns,
            best_values,
            best_records,
            epoch0_sample_path,
            initial_pattern_state=initial_pattern_state,
            generator_ema=ema.module if ema is not None else None,
            scaler=scaler,
        )

    for epoch in range(start_epoch_index, int(config["epochs"]) + 1):
        generator.train()
        discriminator.train()
        pattern_epoch_stats = None
        pattern_train_epoch_enabled = pattern_updates_enabled(config, epoch)
        lambda_adv_epoch = current_lambda_adv(config, epoch)
        use_adv = bool(config.get("use_adversarial", True)) and lambda_adv_epoch > 0
        generator_frozen = bool(config.get("freeze_generator_all", False)) or epoch <= int(
            config.get("freeze_generator_epochs", 0)
        )
        discriminator_frozen = bool(config.get("freeze_discriminator_all", False)) or epoch <= int(
            config.get("freeze_discriminator_epochs", 0)
        )
        set_requires_grad(generator, not generator_frozen)
        set_requires_grad(discriminator, not discriminator_frozen)
        stage = config.get("training_stage") or {}
        refiner_start_epoch = int(stage.get("refiner_start_epoch", 0) or 0) if isinstance(stage, dict) else 0
        refiner_enabled = not (hasattr(generator, "refiner") and refiner_start_epoch > 0 and epoch < refiner_start_epoch)
        if hasattr(generator, "refiner"):
            set_requires_grad(generator.refiner, (not generator_frozen) and refiner_enabled)
        if pattern_bank is not None:
            pattern_bank.set_tau(current_tau(config, epoch))
        epoch_g = []
        epoch_d = []
        epoch_train_parts = {
            "total": [],
            "l1": [],
            "dc": [],
            "ssim": [],
            "ms_ssim": [],
            "edge": [],
            "frequency": [],
        }
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")

        for batch in progress:
            x = batch[0].to(device, non_blocking=True)
            pattern_update_this_step = (
                pattern_bank is not None
                and opt_patterns is not None
                and pattern_train_epoch_enabled
                and global_step % max(1, int(config.get("pattern_update_every", 1))) == 0
            )
            if pattern_bank is not None:
                set_requires_grad(pattern_bank, pattern_update_this_step)
            y = measurement.measure(x)

            if use_adv and not discriminator_frozen:
                set_requires_grad(discriminator, True)
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    x_hat_detached, x_data_detached = reconstruct_from_measurements(
                        generator,
                        measurement,
                        y,
                        use_null_project=bool(config["use_null_project"]),
                        use_dc_project=bool(config["use_dc_project"]),
                        use_final_dc_project=bool(config.get("use_final_dc_project", config["use_dc_project"])),
                        backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                        enable_refiner=refiner_enabled,
                        output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                    )
                with torch.cuda.amp.autocast(enabled=use_amp):
                    real_scores = discriminator(x)
                    fake_scores = discriminator(x_hat_detached.detach())
                    gp = gradient_penalty(discriminator, x, x_hat_detached.detach(), device)
                    d_loss = (
                        discriminator_wgan_loss(real_scores, fake_scores)
                        + float(config["lambda_gp"]) * gp
                    )
                scaler.scale(d_loss).backward()
                scaler.unscale_(opt_d)
                clip_gradients(discriminator, config.get("discriminator_grad_clip_norm"))
                scaler.step(opt_d)
                scaler.update()
                d_loss_value = d_loss.item()
                gp_value = gp.item()
            else:
                d_loss_value = 0.0
                gp_value = 0.0
            epoch_d.append(d_loss_value)

            g_loss_value = None
            should_optimize_g_block = (not generator_frozen) or pattern_update_this_step
            if global_step % int(config["n_critic"]) == 0 and should_optimize_g_block:
                set_requires_grad(discriminator, False)
                opt_g.zero_grad(set_to_none=True)
                if opt_patterns is not None:
                    opt_patterns.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    x_hat, x_data, recon_extras = reconstruct_from_measurements(
                        generator,
                        measurement,
                        y,
                        use_null_project=bool(config["use_null_project"]),
                        use_dc_project=bool(config["use_dc_project"]),
                        use_final_dc_project=bool(config.get("use_final_dc_project", config["use_dc_project"])),
                        backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                        enable_refiner=refiner_enabled,
                        output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                        return_extras=True,
                    )
                    dc_target = (
                        recon_extras.get("x_hat_unclamped", x_hat)
                        if str(config.get("output_range_mode", "clamp_eval_only")).lower() == "clamp_eval_only"
                        else x_hat
                    )
                    l1 = reconstruction_loss(x_hat, x)
                    dc = data_consistency_loss(measurement, dc_target, y)
                    tv = total_variation_loss(x_hat)
                    charb = charbonnier_loss(x_hat, x)
                    edge = sobel_edge_loss(x_hat, x)
                    ms_l1 = simple_multiscale_l1(x_hat, x)
                    ssim_l = differentiable_ssim_loss(x_hat, x)
                    ms_ssim_l = multiscale_ssim_loss(x_hat, x)
                    grad_l = gradient_difference_loss(x_hat, x)
                    freq_l = frequency_loss(x_hat, x)
                    stage1_aux = torch.zeros((), device=device)
                    if "x_stage1" in recon_extras and recon_extras["x_stage1"] is not x_hat:
                        stage1_aux = reconstruction_loss(recon_extras["x_stage1"], x)
                    if use_adv:
                        fake_scores_for_g = discriminator(x_hat)
                        adv = generator_adversarial_loss(fake_scores_for_g)
                        adv_term = lambda_adv_epoch * adv
                        adv_value = adv.item()
                    else:
                        adv = torch.zeros((), device=device)
                        adv_term = adv
                        adv_value = 0.0
                if pattern_bank is not None and pattern_update_this_step:
                    pattern_total, pattern_details = total_pattern_loss(
                        pattern_bank,
                        x,
                        lambda_energy=float(config.get("lambda_pattern_energy", 1.0)),
                        lambda_decorrelation=float(
                            config.get("lambda_pattern_decorrelation", 0.1)
                        ),
                        lambda_binary=float(config.get("lambda_pattern_binary", 0.01)),
                        lambda_secrip=float(config.get("lambda_pattern_secrip", 0.1)),
                        lambda_contrast=float(config.get("lambda_pattern_contrast", 0.0)),
                        target_contrast=float(config.get("target_contrast", 0.45)),
                        lambda_bounded_contrast=float(
                            config.get("lambda_pattern_bounded_contrast", 0.0)
                        ),
                        continuous_min_contrast=float(config.get("continuous_min_contrast", 0.05)),
                        continuous_max_contrast=float(config.get("continuous_max_contrast", 0.5)),
                        lambda_smoothness=float(config.get("lambda_pattern_smoothness", 0.0)),
                        lambda_flip_margin=current_lambda_flip_margin(config, epoch),
                        flip_margin_target=float(config.get("flip_margin_target", 0.05)),
                        lambda_soft_flip=float(config.get("lambda_pattern_soft_flip", 0.0)),
                        target_soft_flip_delta=float(config.get("target_soft_flip_delta", 0.01)),
                        initial_pattern_state=initial_pattern_state,
                        min_flip_fraction_target=float(
                            config.get("min_flip_fraction_target", 0.001)
                        ),
                        max_flip_fraction_target=float(
                            config.get("max_flip_fraction_target", 0.05)
                        ),
                    )
                else:
                    pattern_total = torch.zeros((), device=device)
                    pattern_details = {}
                g_loss = (
                    float(config["lambda_l1"]) * l1
                    + float(config["lambda_dc_loss"]) * dc
                    + float(config["lambda_tv"]) * tv
                    + float(config.get("lambda_charbonnier", 0.0)) * charb
                    + float(config.get("lambda_edge", 0.0)) * edge
                    + float(config.get("lambda_ms_l1", 0.0)) * ms_l1
                    + float(config.get("lambda_ssim", 0.0)) * ssim_l
                    + float(config.get("lambda_ms_ssim", 0.0)) * ms_ssim_l
                    + float(config.get("lambda_gradient", 0.0)) * grad_l
                    + float(config.get("lambda_frequency", 0.0)) * freq_l
                    + float(config.get("lambda_stage1_aux", 0.0)) * stage1_aux
                    + adv_term
                    + pattern_total
                )
                scaler.scale(g_loss).backward()
                if not generator_frozen:
                    scaler.unscale_(opt_g)
                if opt_patterns is not None and pattern_update_this_step:
                    scaler.unscale_(opt_patterns)
                gen_clip_before, gen_clip_after = (
                    clip_gradients(generator, config.get("generator_grad_clip_norm"))
                    if not generator_frozen
                    else (None, None)
                )
                pat_clip_before, pat_clip_after = (None, None)
                if pattern_bank is not None and pattern_update_this_step:
                    pat_clip_before, pat_clip_after = clip_gradients(
                        pattern_bank, config.get("pattern_grad_clip_norm")
                    )
                if not generator_frozen:
                    scaler.step(opt_g)
                if opt_patterns is not None and pattern_update_this_step:
                    scaler.step(opt_patterns)
                scaler.update()
                if ema is not None and not generator_frozen:
                    ema.update(generator)
                g_loss_value = g_loss.item()
                epoch_g.append(g_loss_value)
                epoch_train_parts["total"].append(float(g_loss_value))
                epoch_train_parts["l1"].append(float(l1.item()))
                epoch_train_parts["dc"].append(float(dc.item()))
                epoch_train_parts["ssim"].append(float(ssim_l.item()))
                epoch_train_parts["ms_ssim"].append(float(ms_ssim_l.item()))
                epoch_train_parts["edge"].append(float(edge.item()))
                epoch_train_parts["frequency"].append(float(freq_l.item()))

                writer.add_scalar("train/g_loss", g_loss_value, global_step)
                writer.add_scalar("train/l1", l1.item(), global_step)
                writer.add_scalar("train/data_consistency", dc.item(), global_step)
                writer.add_scalar("train/tv", tv.item(), global_step)
                writer.add_scalar("train/charbonnier", charb.item(), global_step)
                writer.add_scalar("train/edge", edge.item(), global_step)
                writer.add_scalar("train/ms_l1", ms_l1.item(), global_step)
                writer.add_scalar("train/ssim_loss", ssim_l.item(), global_step)
                writer.add_scalar("train/ms_ssim_loss", ms_ssim_l.item(), global_step)
                writer.add_scalar("train/gradient_loss", grad_l.item(), global_step)
                writer.add_scalar("train/frequency_loss", freq_l.item(), global_step)
                writer.add_scalar("train/stage1_aux", stage1_aux.item(), global_step)
                writer.add_scalar("train/adv", adv_value, global_step)
                writer.add_scalar("train/lambda_adv", lambda_adv_epoch, global_step)
                if gen_clip_before is not None:
                    writer.add_scalar("train/generator_grad_norm_before_clip", gen_clip_before, global_step)
                    writer.add_scalar("train/generator_grad_norm_after_clip", gen_clip_after, global_step)
                if pat_clip_before is not None:
                    writer.add_scalar("train/pattern_grad_norm_before_clip", pat_clip_before, global_step)
                    writer.add_scalar("train/pattern_grad_norm_after_clip", pat_clip_after, global_step)
                for name, value in pattern_details.items():
                    writer.add_scalar(f"train/{name}", value.item(), global_step)

                if global_step % int(config["save_image_every"]) == 0:
                    save_recon_grid(
                        x,
                        x_data,
                        x_hat,
                        sample_dir / f"step_{global_step:07d}.png",
                    )

            writer.add_scalar("train/d_loss", d_loss_value, global_step)
            writer.add_scalar("train/gradient_penalty", gp_value, global_step)
            progress.set_postfix(
                d_loss=f"{d_loss_value:.3f}",
                g_loss="-" if g_loss_value is None else f"{g_loss_value:.3f}",
            )
            global_step += 1

        train_g = sum(epoch_g) / max(1, len(epoch_g))
        train_d = sum(epoch_d) / max(1, len(epoch_d))
        print(f"Epoch {epoch}: train_g={train_g:.4f}, train_d={train_d:.4f}")

        metrics = {}
        wrote_per_epoch_metrics = False
        if epoch % int(config["eval_every"]) == 0:
            epoch_sample_path = sample_dir / f"epoch_{epoch:03d}.png"
            eval_generator = ema.module if ema is not None else generator
            metrics = evaluate(
                eval_generator,
                val_loader,
                measurement,
                device,
                config,
                sample_path=epoch_sample_path,
            )
            print("Validation comparison:")
            print(format_metric_comparison(metrics))
            save_json(metrics, output_dir / "val_metrics_latest.json")
            append_eval_history(output_dir, epoch, metrics, train_g=train_g, train_d=train_d)

            back = metrics.get("backprojection", {})
            model = metrics.get("model", {})
            improve = metrics.get("improvement", {})
            writer.add_scalar("val/backproj_psnr", back.get("psnr", 0.0), epoch)
            writer.add_scalar("val/backproj_ssim", back.get("ssim", 0.0), epoch)
            writer.add_scalar("val/backproj_mse", back.get("mse", 0.0), epoch)
            writer.add_scalar(
                "val/backproj_rel_meas_err", back.get("rel_meas_error", 0.0), epoch
            )
            writer.add_scalar("val/model_psnr", model.get("psnr", 0.0), epoch)
            writer.add_scalar("val/model_ssim", model.get("ssim", 0.0), epoch)
            writer.add_scalar("val/model_mse", model.get("mse", 0.0), epoch)
            writer.add_scalar(
                "val/model_rel_meas_err", model.get("rel_meas_error", 0.0), epoch
            )
            writer.add_scalar("val/improve_psnr", improve.get("delta_psnr", 0.0), epoch)
            writer.add_scalar("val/improve_ssim", improve.get("delta_ssim", 0.0), epoch)
            writer.add_scalar("val/improve_mse", improve.get("delta_mse", 0.0), epoch)
            pattern = metrics.get("pattern", {})
            if pattern:
                writer.add_scalar("pattern/mean", pattern.get("mean", 0.0), epoch)
                writer.add_scalar("pattern/std", pattern.get("std", 0.0), epoch)
                writer.add_scalar(
                    "pattern/binary_fraction_005_095",
                    pattern.get("binary_fraction_005_095", 0.0),
                    epoch,
                )
                writer.add_scalar(
                    "pattern/mean_abs_offdiag_corr",
                    pattern.get("mean_abs_offdiag_corr", 0.0),
                    epoch,
                )
                writer.add_scalar("pattern/row_std_min", pattern.get("row_std_min", 0.0), epoch)
                writer.add_scalar("pattern/row_std_mean", pattern.get("row_std_mean", 0.0), epoch)
                writer.add_scalar("pattern/row_std_max", pattern.get("row_std_max", 0.0), epoch)

            if pattern_bank is not None and initial_pattern_state is not None:
                diagnostics = save_pattern_diagnostics_artifacts(
                    pattern_bank,
                    output_dir,
                    initial_pattern_state,
                    config,
                    epoch=epoch,
                    secant_batch=pattern_audit_batch,
                )
                metrics["pattern_diagnostics"] = diagnostics
                writer.add_scalar(
                    "pattern_diagnostics/hard_flip_fraction",
                    float(diagnostics.get("hard_flip_fraction", 0.0) or 0.0),
                    epoch,
                )
                writer.add_scalar(
                    "pattern_diagnostics/A_rel_fro_delta",
                    float(diagnostics.get("A_rel_fro_delta", 0.0) or 0.0),
                    epoch,
                )
                secant_delta = diagnostics.get("secant_rip_delta", 0.0)
                if isinstance(secant_delta, (int, float)):
                    writer.add_scalar("pattern_diagnostics/secant_rip_delta", float(secant_delta), epoch)
                soft_l2_delta = diagnostics.get("soft_l2_delta", 0.0)
                if isinstance(soft_l2_delta, (int, float)):
                    writer.add_scalar("pattern_diagnostics/soft_l2_delta", float(soft_l2_delta), epoch)
                near_005 = diagnostics.get("near_threshold_fraction_0p05", 0.0)
                if isinstance(near_005, (int, float)):
                    writer.add_scalar("pattern/near_threshold_fraction_0p05", float(near_005), epoch)
                save_json(metrics, output_dir / "val_metrics_latest.json")

            update_best_checkpoints(
                metrics,
                epoch,
                config,
                output_dir,
                generator,
                discriminator,
                opt_g,
                opt_d,
                pattern_bank,
                opt_patterns,
                best_values,
                best_records,
                epoch_sample_path,
                initial_pattern_state=initial_pattern_state,
                generator_ema=ema.module if ema is not None else None,
                scaler=scaler,
            )
            append_per_epoch_metrics(
                output_dir,
                epoch,
                metrics,
                epoch_train_parts,
                config,
                best_records,
                time.time() - train_start_perf,
                train_d=train_d,
            )
            wrote_per_epoch_metrics = True

        if not wrote_per_epoch_metrics:
            append_per_epoch_metrics(
                output_dir,
                epoch,
                metrics,
                epoch_train_parts,
                config,
                best_records,
                time.time() - train_start_perf,
                train_d=train_d,
            )

        if pattern_bank is not None and epoch % int(config.get("save_patterns_every", 1)) == 0:
            pattern_epoch_stats = save_pattern_artifacts(pattern_bank, output_dir, epoch, config)

        save_checkpoint(
            output_dir / "last.pt",
            generator,
            discriminator,
            opt_g,
            opt_d,
            epoch,
            config,
            metrics,
            pattern_bank=pattern_bank,
            opt_patterns=opt_patterns,
            initial_pattern_state=initial_pattern_state,
            generator_ema=ema.module if ema is not None else None,
            scaler=scaler,
        )
        if epoch % int(config["save_every"]) == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                generator,
                discriminator,
                opt_g,
                opt_d,
                epoch,
                config,
                metrics,
                pattern_bank=pattern_bank,
                opt_patterns=opt_patterns,
                initial_pattern_state=initial_pattern_state,
                generator_ema=ema.module if ema is not None else None,
                scaler=scaler,
            )

    if pattern_bank is not None and initial_pattern_state is not None:
        final_diagnostics = save_pattern_diagnostics_artifacts(
            pattern_bank,
            output_dir,
            initial_pattern_state,
            config,
            epoch=None,
            secant_batch=pattern_audit_batch,
        )
        run_notes.extend(
            [
                f"hard_flip_fraction: {final_diagnostics.get('hard_flip_fraction', 'missing')}",
                f"A_rel_fro_delta: {final_diagnostics.get('A_rel_fro_delta', 'missing')}",
                f"soft_l2_delta: {final_diagnostics.get('soft_l2_delta', 'missing')}",
                f"secant_rip_delta: {final_diagnostics.get('secant_rip_delta', 'missing')}",
                f"offdiag_corr_delta: {final_diagnostics.get('offdiag_corr_delta', 'missing')}",
                f"freeze_patterns: {bool(config.get('freeze_patterns', False))}",
                f"freeze_generator_all: {bool(config.get('freeze_generator_all', False))}",
                f"pattern_trainable: {not patterns_frozen_all}",
                f"pattern attribution: {final_diagnostics.get('pattern_attribution_note', 'missing')}",
            ]
        )

    writer.close()
    for name in ["psnr", "ssim", "mse", "score", "hq"]:
        record = best_records[name]
        metrics_for_note = record.get("metrics") or {}
        model_for_note = metrics_for_note.get("model", {})
        run_notes.append(
            "Best "
            f"{name}: epoch={record.get('epoch')}, "
            f"psnr={model_for_note.get('psnr', 'missing')}, "
            f"ssim={model_for_note.get('ssim', 'missing')}, "
            f"mse={model_for_note.get('mse', 'missing')}, "
            f"path={record.get('path')}"
        )
    primary = str(config.get("checkpoint_metric_mode", "score")).lower()
    if primary not in best_records:
        primary = "score"
    primary_record = best_records[primary]
    report_path = write_run_report(
        output_dir=output_dir,
        config=config,
        measurement=measurement,
        start_time=start_time,
        end_time=datetime.now(),
        best_epoch=primary_record.get("epoch"),
        best_metrics=primary_record.get("metrics"),
        checkpoint_path=primary_record.get("path") or (output_dir / "last.pt"),
        sample_path=primary_record.get("sample"),
        sanity_path=output_dir / "sanity_physics.json",
        device=device,
        notes=run_notes,
    )
    print(f"Run report written to: {report_path}")
    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
