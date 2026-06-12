from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .datasets import get_val_dataloader
from .exact_measurement import apply_measurement_override_from_config
from .measurement import (
    GhostMeasurementOperator,
    LearnableGhostMeasurementOperator,
    LearnablePatternBank,
)
from .metrics import batch_metrics
from .models import build_generator
from .pattern_diagnostics import (
    compare_pattern_states,
    save_pattern_change_visualization,
    save_pattern_diagnostics_json,
)
from .pattern_regularizers import secant_rip_loss
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
    set_seed,
    update_config_from_args,
)
from .sample_export import save_eval_samples_individual, write_per_sample_csv
from .visualize import save_recon_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NS-MC-GAN checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sampling_ratio", type=float, default=None)
    parser.add_argument("--pattern_type", default=None)
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit_val_samples", type=int, default=None)
    parser.add_argument("--use_null_project", choices=["true", "false"], default=None)
    parser.add_argument("--use_dc_project", choices=["true", "false"], default=None)
    parser.add_argument("--use_final_dc_project", choices=["true", "false"], default=None)
    parser.add_argument("--enable_refiner", choices=["true", "false"], default=None)
    parser.add_argument(
        "--output_range_mode",
        choices=["clamp_eval_only", "clamp_after_dc", "sigmoid_before_dc"],
        default=None,
    )
    parser.add_argument("--use_ema", choices=["true", "false"], default=None)
    return parser.parse_args()


def arg_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() == "true"


def make_measurement(config: dict, device: torch.device) -> GhostMeasurementOperator:
    if bool(config.get("use_learned_patterns", False)):
        pattern_bank = LearnablePatternBank(
            img_size=config["img_size"],
            sampling_ratio=config["sampling_ratio"],
            pattern_mode=config.get("pattern_mode", "learned_binary_ste"),
            init_type=config.get("pattern_init", "bernoulli"),
            tau=config.get("pattern_tau_final", config.get("pattern_tau", 1.0)),
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_experiment_defaults(config)
    device = resolve_device(args.device or config["device"])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        merged = dict(config)
        merged.update(checkpoint["config"])
        config = apply_experiment_defaults(merged)
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
            "batch_size",
            "limit_val_samples",
            "output_range_mode",
        ],
    )
    use_null_project_arg = arg_bool(args.use_null_project)
    use_dc_project_arg = arg_bool(args.use_dc_project)
    use_final_dc_project_arg = arg_bool(args.use_final_dc_project)
    enable_refiner_arg = arg_bool(args.enable_refiner)
    use_ema_arg = arg_bool(args.use_ema)
    if use_null_project_arg is not None:
        config["use_null_project"] = use_null_project_arg
    if use_dc_project_arg is not None:
        config["use_dc_project"] = use_dc_project_arg
    if use_final_dc_project_arg is not None:
        config["use_final_dc_project"] = use_final_dc_project_arg
    config = apply_experiment_defaults(config)
    device = resolve_device(config["device"])
    set_seed(int(config["seed"]))

    output_dir = ensure_dir(args.output_dir or config["output_dir"] or Path(args.checkpoint).parent)
    sample_dir = ensure_dir(output_dir / "eval_samples")
    measurement = make_measurement(config, device)
    exact_a_info = apply_measurement_override_from_config(config, measurement, device)
    if exact_a_info.get("exact_A_loaded"):
        print(f"Loaded exact-A with safe cache rebuild: {exact_a_info.get('exact_A_path')}")
    pattern_bank = getattr(measurement, "pattern_bank", None)
    initial_pattern_state = None
    if bool(config.get("use_learned_patterns", False)):
        if not isinstance(checkpoint, dict) or "pattern_bank" not in checkpoint:
            raise RuntimeError(
                "config.use_learned_patterns=true but checkpoint has no pattern_bank "
                "state_dict; refusing to silently evaluate random learned patterns."
            )
        pattern_bank.load_state_dict(checkpoint["pattern_bank"])
        pattern_bank.set_tau(float(config.get("pattern_tau_final", config.get("pattern_tau", 1.0))))
        pattern_bank.eval()
        initial_pattern_state = checkpoint.get("initial_pattern_state") if isinstance(checkpoint, dict) else None

    print(
        "Eval setup: "
        f"device={device}, img_size={measurement.img_size}, n={measurement.n}, "
        f"m={measurement.m}, sampling_ratio={measurement.sampling_ratio}, "
        f"pattern_type={measurement.pattern_type}, noise_std={measurement.noise_std}"
    )

    generator = build_generator(config, measurement=measurement).to(device)
    if isinstance(checkpoint, dict):
        if use_ema_arg is False:
            state = checkpoint["generator"]
        elif checkpoint.get("generator_ema") is not None:
            state = checkpoint["generator_ema"]
        else:
            state = checkpoint["generator"]
    else:
        state = checkpoint
    generator.load_state_dict(state)
    generator.eval()

    val_loader = get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        limit_val_samples=config["limit_val_samples"],
        seed=config["seed"],
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )

    backprojection_metrics = []
    model_metrics = []
    secant_values = []
    audit_batch = None
    individual_rows = []
    individual_limit = int(config.get("num_individual_eval_samples", config.get("num_eval_samples_to_save", 8)))
    individual_seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            x = batch[0].to(device, non_blocking=True)
            if batch_idx == 0:
                audit_batch = x.detach()
            y = measurement.measure(x)
            x_hat, x_data, extras = reconstruct_from_measurements(
                generator,
                measurement,
                y,
                use_null_project=bool(config["use_null_project"]),
                use_dc_project=bool(config["use_dc_project"]),
                use_final_dc_project=bool(config.get("use_final_dc_project", config["use_dc_project"])),
                backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=True if enable_refiner_arg is None else enable_refiner_arg,
                output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                return_extras=True,
            )
            backprojection_metrics.append(batch_metrics(x_data, x, measurement, y))
            model_batch = batch_metrics(x_hat, x, measurement, y)
            model_batch["rel_meas_err_clamped"] = model_batch.get("rel_meas_error", float("nan"))
            model_batch["rel_meas_err_unclamped"] = batch_metrics(
                extras["x_hat_unclamped"], x, measurement, y
            ).get("rel_meas_error", float("nan"))
            model_metrics.append(model_batch)
            if pattern_bank is not None:
                secant_values.append(secant_rip_loss(measurement.get_current_A(), x).item())
            if batch_idx == 0:
                preview_metrics = batch_metrics(x_hat, x, measurement, y)
                title = (
                    f"sampling={measurement.sampling_ratio:.2%} | "
                    f"PSNR={preview_metrics['psnr']:.3f} | "
                    f"SSIM={preview_metrics['ssim']:.3f} | "
                    f"RelMeasErr={preview_metrics['rel_meas_error']:.3g}"
                )
                save_recon_grid(
                    x,
                    x_data,
                    x_hat,
                    sample_dir / "recon_grid.png",
                    max_items=int(config.get("num_eval_samples_to_save", 8)),
                    title=title,
                )
            if individual_seen < individual_limit:
                remaining = individual_limit - individual_seen
                rows = save_eval_samples_individual(
                    output_dir,
                    x,
                    x_data,
                    x_hat,
                    measurement,
                    y,
                    start_index=individual_seen,
                    max_items=remaining,
                )
                individual_rows.extend(rows)
                individual_seen += len(rows)

    metrics = compare_metric_sets(
        mean_dict(backprojection_metrics),
        mean_dict(model_metrics),
    )
    if pattern_bank is not None:
        pattern_stats = pattern_bank.get_pattern_stats()
        pattern_stats["secant_rip_eval_loss"] = (
            float(sum(secant_values) / max(1, len(secant_values))) if secant_values else ""
        )
        metrics["pattern"] = pattern_stats
        diag_dir = ensure_dir(output_dir / "eval_pattern_diagnostics")
        diagnostics = compare_pattern_states(
            pattern_bank,
            initial_pattern_state,
            secant_batch=audit_batch,
            config=config,
        )
        save_pattern_diagnostics_json(diagnostics, diag_dir / "pattern_diagnostics.json")
        if initial_pattern_state and diagnostics.get("status") == "ok":
            save_pattern_change_visualization(
                initial_pattern_state["P_initial_hard"],
                pattern_bank.get_hard_patterns().detach().cpu(),
                diag_dir / "pattern_change_grid.png",
                pattern_bank.img_size,
            )
        metrics["pattern_diagnostics"] = diagnostics
        metrics["pattern_hard_flip_fraction"] = diagnostics.get("hard_flip_fraction", "missing")
        metrics["pattern_A_rel_fro_delta"] = diagnostics.get("A_rel_fro_delta", "missing")
        metrics["pattern_soft_l2_delta"] = diagnostics.get("soft_l2_delta", "missing")
        metrics["pattern_soft_flip_delta"] = diagnostics.get("soft_flip_delta", "missing")
        metrics["pattern_logits_l2_delta"] = diagnostics.get("logits_l2_delta", "missing")
        metrics["pattern_secant_rip_initial"] = diagnostics.get("secant_rip_initial", "missing")
        metrics["pattern_secant_rip_final"] = diagnostics.get("secant_rip_final", "missing")
        metrics["pattern_secant_rip_delta"] = diagnostics.get("secant_rip_delta", "missing")
        metrics["pattern_offdiag_corr_initial"] = diagnostics.get("offdiag_corr_initial", "missing")
        metrics["pattern_offdiag_corr_final"] = diagnostics.get("offdiag_corr_final", "missing")
        metrics["pattern_offdiag_corr_delta"] = diagnostics.get("offdiag_corr_delta", "missing")
        metrics["pattern_attribution_note"] = diagnostics.get("pattern_attribution_note", "missing")
        metrics["pattern_physical_type"] = pattern_stats.get("pattern_physical_type", "missing")
        metrics["pattern_continuous_contrast"] = pattern_stats.get("continuous_contrast", "missing")
        metrics["pattern_mean_abs_offdiag_corr"] = pattern_stats.get("mean_abs_offdiag_corr", "missing")
        metrics["pattern_near_threshold_fraction_0p05"] = pattern_stats.get(
            "near_threshold_fraction_0p05", "missing"
        )
        pattern_dir = ensure_dir(output_dir / "eval_patterns")
        P = pattern_bank.get_physical_patterns()
        save_pattern_grid(P, pattern_dir / "final_patterns.png", pattern_bank.img_size)
        if pattern_bank.pattern_mode in {
            "learned_binary_ste",
            "learned_balanced_binary_ste",
            "learned_flip_aware_binary_ste",
        }:
            P_soft = pattern_bank.get_soft_patterns()
            save_pattern_grid(P_soft, pattern_dir / "final_patterns_soft.png", pattern_bank.img_size)
            save_pattern_grid(
                pattern_bank.get_hard_patterns(),
                pattern_dir / "final_patterns_hard.png",
                pattern_bank.img_size,
            )
        save_pattern_stats_json(pattern_stats, pattern_dir / "final_pattern_stats.json")
    print(format_metric_comparison(metrics))
    if individual_rows:
        per_sample_path = write_per_sample_csv(output_dir, individual_rows)
        metrics["per_sample_metrics"] = str(per_sample_path)
    metrics_path = save_json(metrics, output_dir / "eval_metrics.json")
    print(f"Saved metrics JSON to: {metrics_path}")
    print(f"Saved samples to: {Path(sample_dir)}")


if __name__ == "__main__":
    main()
