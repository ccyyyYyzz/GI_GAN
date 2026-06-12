from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .datasets import get_val_dataloader
from .measurement import LearnableGhostMeasurementOperator, LearnablePatternBank
from .models import ResidualUNetGenerator
from .pattern_regularizers import total_pattern_loss
from .pattern_utils import save_pattern_grid, save_pattern_stats_json
from .utils import (
    apply_experiment_defaults,
    ensure_dir,
    load_config,
    reconstruct_from_measurements,
    resolve_device,
    save_json,
    set_seed,
    update_config_from_args,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check learnable pattern bank.")
    parser.add_argument("--config", default="configs/phase3_debug_binary_5pct.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit_val_samples", type=int, default=None)
    return parser.parse_args()


def vector_rel_norm(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-12) -> float:
    numer_norm = torch.linalg.norm(numer.reshape(numer.shape[0], -1), dim=1)
    denom_norm = torch.linalg.norm(denom.reshape(denom.shape[0], -1), dim=1).clamp_min(eps)
    return (numer_norm / denom_norm).mean().item()


def main() -> None:
    args = parse_args()
    config = apply_experiment_defaults(load_config(args.config))
    config = update_config_from_args(
        config,
        args,
        ["device", "dataset_root", "output_dir", "batch_size", "limit_val_samples"],
    )
    config = apply_experiment_defaults(config)
    if not bool(config.get("use_learned_patterns", False)):
        raise ValueError("sanity_learnable_patterns requires use_learned_patterns=true.")

    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])
    output_dir = ensure_dir(config["output_dir"])

    pattern_bank = LearnablePatternBank(
        img_size=config["img_size"],
        sampling_ratio=config["sampling_ratio"],
        pattern_mode=config["pattern_mode"],
        init_type=config["pattern_init"],
        tau=config["pattern_tau"],
        target_transmission=config["target_transmission"],
        pattern_logit_abs_init=config.get("pattern_logit_abs_init", 2.0),
        balanced_target_transmission=config.get(
            "balanced_target_transmission", config.get("target_transmission", 0.5)
        ),
        effective_A_mode=config.get("effective_A_mode", "centered_standardized"),
        fixed_reference_pattern_type=config.get("fixed_reference_pattern_type", "rademacher"),
        fixed_reference_normalization=config.get(
            "fixed_reference_normalization", "row_norm_sqrt_n_over_m"
        ),
        device=device,
        seed=config["seed"],
    ).to(device)
    measurement = LearnableGhostMeasurementOperator(
        pattern_bank=pattern_bank,
        noise_std=float(config["noise_std"]),
        lambda_dc=float(config["lambda_solver"]),
        device=device,
    )
    generator = ResidualUNetGenerator().to(device)

    P = pattern_bank.get_physical_patterns()
    P_soft = pattern_bank.get_soft_patterns()
    A_eff = pattern_bank.get_effective_A()
    shape_checks = {
        "P_shape": list(P.shape),
        "P_soft_shape": list(P_soft.shape),
        "A_eff_shape": list(A_eff.shape),
        "expected_m": measurement.m,
        "expected_n": measurement.n,
        "P_in_0_1": bool((P.detach() >= 0).all().item() and (P.detach() <= 1).all().item()),
    }

    random_x = torch.rand(
        min(int(config.get("batch_size", 4)), 8),
        1,
        int(config["img_size"]),
        int(config["img_size"]),
        device=device,
    )
    y = measurement.measure(random_x)
    v = torch.randn(random_x.shape[0], measurement.n, device=device)
    v_ns = measurement.null_project(v)
    x_dc = measurement.dc_project(v, y)
    x_data = measurement.data_solution(y)
    physics = {
        "null_error": vector_rel_norm(measurement.A_forward(v_ns), v),
        "dc_error": vector_rel_norm(measurement.A_forward(x_dc) - y, y),
        "backproj_dc_error": vector_rel_norm(measurement.A_forward(x_data) - y, y),
    }

    stl_status = "not_run"
    stl_error = None
    grad = {
        "grad_exists": False,
        "grad_isfinite": False,
        "grad_norm": 0.0,
    }
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
        x = next(iter(loader))[0].to(device, non_blocking=True)
        y = measurement.measure(x)
        x_hat, _ = reconstruct_from_measurements(
            generator,
            measurement,
            y,
            use_null_project=bool(config["use_null_project"]),
            use_dc_project=bool(config["use_dc_project"]),
        )
        pattern_total, pattern_details = total_pattern_loss(
            pattern_bank,
            x,
            lambda_energy=float(config["lambda_pattern_energy"]),
            lambda_decorrelation=float(config["lambda_pattern_decorrelation"]),
            lambda_binary=float(config["lambda_pattern_binary"]),
            lambda_secrip=float(config["lambda_pattern_secrip"]),
            lambda_contrast=float(config.get("lambda_pattern_contrast", 0.0)),
            target_contrast=float(config.get("target_contrast", 0.45)),
        )
        loss = F.l1_loss(x_hat, x) + pattern_total
        loss.backward()
        logits_grad = pattern_bank.logits.grad
        if logits_grad is not None:
            grad = {
                "grad_exists": True,
                "grad_isfinite": bool(torch.isfinite(logits_grad).all().item()),
                "grad_norm": float(logits_grad.norm().detach().cpu()),
            }
        stl_status = "ok"
    except Exception as exc:
        stl_status = "failed"
        stl_error = str(exc)
        pattern_details = {}

    pattern_stats = pattern_bank.get_pattern_stats()
    pattern_dir = ensure_dir(output_dir / "patterns")
    save_pattern_grid(P, pattern_dir / "sanity_patterns.png", pattern_bank.img_size)
    save_pattern_grid(P_soft, pattern_dir / "sanity_patterns_soft.png", pattern_bank.img_size)
    if pattern_bank.pattern_mode in {"learned_binary_ste", "learned_balanced_binary_ste"}:
        save_pattern_grid(
            P_soft,
            pattern_dir / "sanity_patterns_hard.png",
            pattern_bank.img_size,
            binarize_for_display=True,
        )
    save_pattern_stats_json(pattern_stats, pattern_dir / "sanity_pattern_stats.json")

    report = {
        "config": config,
        "device": str(device),
        "m": measurement.m,
        "n": measurement.n,
        "shape_checks": shape_checks,
        "physics": physics,
        "gradient_check": grad,
        "pattern_stats": pattern_stats,
        "pattern_loss": {k: float(v.detach().cpu()) for k, v in pattern_details.items()},
        "stl10_status": stl_status,
        "stl10_error": stl_error,
        "passed": (
            shape_checks["P_in_0_1"]
            and stl_status == "ok"
            and grad["grad_exists"]
            and grad["grad_isfinite"]
            and grad["grad_norm"] > 0
        ),
    }
    out_path = save_json(report, Path(output_dir) / "sanity_learnable_patterns.json")
    print(f"Saved learnable-pattern sanity report to: {out_path}")
    print(f"Gradient check: {grad}")
    if not report["passed"]:
        raise RuntimeError("Learnable-pattern sanity check failed; see JSON report.")


if __name__ == "__main__":
    main()
