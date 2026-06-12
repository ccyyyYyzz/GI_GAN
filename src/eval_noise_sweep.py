from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

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
    compare_metric_sets,
    ensure_dir,
    load_config,
    mean_dict,
    reconstruct_from_measurements,
    resolve_device,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint over noise levels.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise_levels", default="0.0,0.005,0.01,0.02,0.05")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def parse_noise_levels(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def make_measurement(config: dict, device: torch.device):
    if bool(config.get("use_learned_patterns", False)):
        pattern_bank = LearnablePatternBank(
            img_size=config["img_size"],
            sampling_ratio=config["sampling_ratio"],
            pattern_mode=config.get("pattern_mode", "learned_binary_ste"),
            init_type=config.get("pattern_init", "bernoulli"),
            tau=config.get("pattern_tau_final", config.get("pattern_tau", 1.0)),
            target_transmission=config.get("target_transmission", 0.5),
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
        return LearnableGhostMeasurementOperator(
            pattern_bank=pattern_bank,
            noise_std=config["noise_std"],
            lambda_dc=config["lambda_solver"],
            device=device,
        )
    return GhostMeasurementOperator(
        img_size=config["img_size"],
        sampling_ratio=config["sampling_ratio"],
        pattern_type=config["pattern_type"],
        noise_std=config["noise_std"],
        lambda_dc=config["lambda_solver"],
        device=device,
        seed=config["seed"],
    )


def metric_score(metrics: dict, config: dict) -> float:
    model = metrics.get("model", {})
    return float(model.get("psnr", 0.0)) + float(config.get("score_ssim_weight", 10.0)) * float(
        model.get("ssim", 0.0)
    )


def evaluate_level(generator, checkpoint: dict, config: dict, device: torch.device) -> dict:
    measurement = make_measurement(config, device)
    pattern_bank = getattr(measurement, "pattern_bank", None)
    if pattern_bank is not None:
        if "pattern_bank" not in checkpoint:
            raise RuntimeError("Learned noise sweep checkpoint has no pattern_bank state.")
        pattern_bank.load_state_dict(checkpoint["pattern_bank"])
        pattern_bank.set_tau(float(config.get("pattern_tau_final", config.get("pattern_tau", 1.0))))
        pattern_bank.eval()

    loader = get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        limit_val_samples=config["limit_val_samples"],
        seed=config["seed"],
        pin_memory=device.type == "cuda",
    )
    backprojection_metrics = []
    model_metrics = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"noise={config['noise_std']}"):
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            x_hat, x_data = reconstruct_from_measurements(
                generator,
                measurement,
                y,
                use_null_project=bool(config["use_null_project"]),
                use_dc_project=bool(config["use_dc_project"]),
            )
            backprojection_metrics.append(batch_metrics(x_data, x, measurement, y))
            model_metrics.append(batch_metrics(x_hat, x, measurement, y))
    metrics = compare_metric_sets(mean_dict(backprojection_metrics), mean_dict(model_metrics))
    if pattern_bank is not None:
        metrics["pattern"] = pattern_bank.get_pattern_stats()
    return metrics


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "noise_std",
        "model_psnr",
        "model_ssim",
        "model_mse",
        "model_rel_meas_err",
        "backproj_psnr",
        "backproj_ssim",
        "backproj_mse",
        "backproj_rel_meas_err",
        "score",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def write_markdown(rows: list[dict], path: Path) -> None:
    cols = ["noise_std", "model_psnr", "model_ssim", "model_rel_meas_err", "score", "status"]
    lines = ["# Noise Sweep", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric(rows: list[dict], metric: str, ylabel: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        ok_rows = [row for row in rows if row.get("status") == "ok"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            [float(row["noise_std"]) for row in ok_rows],
            [float(row[metric]) for row in ok_rows],
            marker="o",
        )
        ax.set_xlabel("noise_std")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"Plot unavailable: {exc}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = apply_experiment_defaults(load_config(args.config))
    device = resolve_device(args.device or config["device"])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        merged = dict(config)
        merged.update(checkpoint["config"])
        config = apply_experiment_defaults(merged)
    config["device"] = str(device)
    set_seed(int(config["seed"]))

    generator = ResidualUNetGenerator().to(device)
    state = checkpoint["generator"] if isinstance(checkpoint, dict) else checkpoint
    generator.load_state_dict(state)
    generator.eval()

    output_dir = ensure_dir(args.output_dir)
    rows = []
    for noise_std in parse_noise_levels(args.noise_levels):
        level_config = dict(config)
        level_config["noise_std"] = float(noise_std)
        try:
            metrics = evaluate_level(generator, checkpoint, level_config, device)
            back = metrics.get("backprojection", {})
            model = metrics.get("model", {})
            rows.append(
                {
                    "noise_std": noise_std,
                    "model_psnr": model.get("psnr", ""),
                    "model_ssim": model.get("ssim", ""),
                    "model_mse": model.get("mse", ""),
                    "model_rel_meas_err": model.get("rel_meas_error", ""),
                    "backproj_psnr": back.get("psnr", ""),
                    "backproj_ssim": back.get("ssim", ""),
                    "backproj_mse": back.get("mse", ""),
                    "backproj_rel_meas_err": back.get("rel_meas_error", ""),
                    "score": metric_score(metrics, level_config),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"noise_std": noise_std, "status": f"failed: {exc}"})

    write_csv(rows, output_dir / "noise_sweep_metrics.csv")
    write_markdown(rows, output_dir / "noise_sweep_metrics.md")
    plot_metric(rows, "model_psnr", "PSNR", output_dir / "noise_sweep_psnr.png")
    plot_metric(rows, "model_ssim", "SSIM", output_dir / "noise_sweep_ssim.png")
    plot_metric(rows, "model_rel_meas_err", "RelMeasErr", output_dir / "noise_sweep_relmeaserr.png")
    print(f"Wrote noise sweep to: {output_dir}")


if __name__ == "__main__":
    main()
