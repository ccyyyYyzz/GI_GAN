from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .datasets import get_val_dataloader
from .measurement import (
    GhostMeasurementOperator,
    LearnableGhostMeasurementOperator,
    LearnablePatternBank,
)
from .metrics import batch_metrics
from .models import ResidualUNetGenerator
from .utils import (
    apply_experiment_defaults,
    ensure_dir,
    load_config,
    reconstruct_from_measurements,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate fixed vs learnable operators.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fixed_pattern_type", default="rademacher")
    parser.add_argument(
        "--report_path",
        default="E:/ns_mc_gan_gi/outputs_phase5/operator_calibration_5pct.json",
    )
    return parser.parse_args()


def rel_norm(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-12) -> float:
    return float(numer.norm().detach().cpu() / denom.norm().clamp_min(eps).detach().cpu())


def make_learned_operator(config: dict, device: torch.device) -> LearnableGhostMeasurementOperator:
    pattern_bank = LearnablePatternBank(
        img_size=config["img_size"],
        sampling_ratio=config["sampling_ratio"],
        pattern_mode=config.get("pattern_mode", "learned_binary_ste"),
        init_type=config.get("pattern_init", "fixed_rademacher_match"),
        tau=config.get("pattern_tau", 1.0),
        target_transmission=config.get("target_transmission", 0.5),
        pattern_logit_abs_init=config.get("pattern_logit_abs_init", 6.0),
        balanced_target_transmission=config.get(
            "balanced_target_transmission", config.get("target_transmission", 0.5)
        ),
        effective_A_mode=config.get("effective_A_mode", "signed_exact_fixed"),
        fixed_reference_pattern_type=config.get("fixed_reference_pattern_type", "rademacher"),
        fixed_reference_normalization=config.get(
            "fixed_reference_normalization", "row_norm_sqrt_n_over_m"
        ),
        device=device,
        seed=config["seed"],
    ).to(device)
    pattern_bank.eval()
    return LearnableGhostMeasurementOperator(
        pattern_bank=pattern_bank,
        noise_std=0.0,
        lambda_dc=config["lambda_solver"],
        device=device,
    )


def load_generator(config: dict, device: torch.device):
    checkpoint_path = config.get("load_generator_checkpoint")
    if not checkpoint_path:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
        raise RuntimeError(f"{checkpoint_path} does not contain a generator state_dict.")
    generator = ResidualUNetGenerator().to(device)
    generator.load_state_dict(checkpoint["generator"], strict=bool(config.get("load_generator_strict", False)))
    generator.eval()
    return generator


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Operator Calibration",
        "",
        f"- status: {report['status']}",
        f"- effective_A_mode: {report['config'].get('effective_A_mode')}",
        f"- A_rel_fro_error: {report['A_rel_fro_error']:.12g}",
        f"- A_max_abs_error: {report['A_max_abs_error']:.12g}",
        f"- A_cosine: {report['A_cosine']:.12g}",
        f"- y_rel_error: {report['y_rel_error']:.12g}",
        f"- x_data_rel_error: {report['x_data_rel_error']:.12g}",
    ]
    if "generator" in report:
        gen = report["generator"]
        lines.extend(
            [
                "",
                "## Warm Start Generator",
                "",
                f"- x_hat_rel_error: {gen['x_hat_rel_error']:.12g}",
                f"- psnr_fixed: {gen['psnr_fixed']:.6f}",
                f"- ssim_fixed: {gen['ssim_fixed']:.6f}",
                f"- psnr_learned: {gen['psnr_learned']:.6f}",
                f"- ssim_learned: {gen['ssim_learned']:.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            report["note"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = apply_experiment_defaults(load_config(args.config))
    if args.device:
        config["device"] = args.device
    device = resolve_device(config["device"])
    set_seed(int(config["seed"]))

    output_path = Path(args.report_path)
    ensure_dir(output_path.parent)

    fixed = GhostMeasurementOperator(
        img_size=config["img_size"],
        sampling_ratio=config["sampling_ratio"],
        pattern_type=args.fixed_pattern_type,
        noise_std=0.0,
        lambda_dc=config["lambda_solver"],
        device=device,
        seed=config["seed"],
    )
    learned = make_learned_operator(config, device)
    A_fixed = fixed.get_current_A()
    A_learned = learned.get_current_A()

    diff = A_fixed - A_learned
    A_rel_fro_error = rel_norm(diff, A_fixed)
    A_max_abs_error = float(diff.abs().max().detach().cpu())
    A_cosine = float(
        torch.sum(A_fixed * A_learned).detach().cpu()
        / (A_fixed.norm() * A_learned.norm()).clamp_min(1e-12).detach().cpu()
    )

    loader = get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        limit_val_samples=config.get("limit_val_samples", config["batch_size"]),
        seed=config["seed"],
        pin_memory=device.type == "cuda",
    )
    x = next(iter(loader))[0].to(device, non_blocking=True)
    y_fixed = fixed.A_forward(fixed.flatten_img(x))
    y_learned = learned.A_forward(learned.flatten_img(x))
    y_rel_error = rel_norm(y_fixed - y_learned, y_fixed)

    x_data_fixed_flat = fixed.data_solution(y_fixed)
    x_data_learned_flat = learned.data_solution(y_learned)
    x_data_rel_error = rel_norm(x_data_fixed_flat - x_data_learned_flat, x_data_fixed_flat)

    exact_mode = config.get("effective_A_mode") == "signed_exact_fixed"
    exact_pass = A_rel_fro_error < 1e-6
    if exact_mode and exact_pass:
        status = "exact_match_passed"
        note = "signed_exact_fixed reproduced the fixed rademacher operator within tolerance."
    elif exact_mode:
        status = "exact_match_failed"
        note = (
            "signed_exact_fixed did not meet the 1e-6 Frobenius tolerance; inspect "
            "reference scaling and binary initialization."
        )
    else:
        status = "expected_mismatch"
        note = "centered_standardized is expected to differ from the fixed operator."

    report = {
        "config": config,
        "status": status,
        "A_rel_fro_error": A_rel_fro_error,
        "A_max_abs_error": A_max_abs_error,
        "A_cosine": A_cosine,
        "y_rel_error": y_rel_error,
        "x_data_rel_error": x_data_rel_error,
        "fixed_pattern_type": args.fixed_pattern_type,
        "fixed_A_shape": list(A_fixed.shape),
        "learned_A_shape": list(A_learned.shape),
        "fixed_pattern_stats": fixed.get_pattern_stats(),
        "learned_pattern_stats": learned.get_pattern_stats(),
        "note": note,
    }

    generator = load_generator(config, device)
    if generator is not None:
        torch.manual_seed(int(config["seed"]))
        x_hat_fixed, x_data_fixed = reconstruct_from_measurements(
            generator,
            fixed,
            y_fixed,
            use_null_project=bool(config["use_null_project"]),
            use_dc_project=bool(config["use_dc_project"]),
        )
        torch.manual_seed(int(config["seed"]))
        x_hat_learned, x_data_learned = reconstruct_from_measurements(
            generator,
            learned,
            y_learned,
            use_null_project=bool(config["use_null_project"]),
            use_dc_project=bool(config["use_dc_project"]),
        )
        fixed_metrics = batch_metrics(x_hat_fixed, x, fixed, y_fixed)
        learned_metrics = batch_metrics(x_hat_learned, x, learned, y_learned)
        report["generator"] = {
            "x_hat_rel_error": rel_norm(x_hat_fixed - x_hat_learned, x_hat_fixed),
            "x_data_rel_error": rel_norm(x_data_fixed - x_data_learned, x_data_fixed),
            "psnr_fixed": fixed_metrics["psnr"],
            "ssim_fixed": fixed_metrics["ssim"],
            "psnr_learned": learned_metrics["psnr"],
            "ssim_learned": learned_metrics["ssim"],
            "fixed_metrics": fixed_metrics,
            "learned_metrics": learned_metrics,
        }

    save_json(report, output_path)
    md_path = output_path.with_suffix(".md")
    write_markdown(report, md_path)
    print(f"Wrote calibration JSON to: {output_path}")
    print(f"Wrote calibration Markdown to: {md_path}")
    print(f"status={status}, A_rel_fro_error={A_rel_fro_error:.12g}")


if __name__ == "__main__":
    main()
