from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from .datasets import get_dataloaders
from .losses import (
    charbonnier_loss,
    data_consistency_loss,
    differentiable_ssim_loss,
    frequency_loss,
    gradient_difference_loss,
    reconstruction_loss,
    sobel_edge_loss,
    total_variation_loss,
)
from .metrics import batch_metrics
from .models import build_generator
from .train import make_measurement
from .utils import apply_experiment_defaults, ensure_dir, load_config, reconstruct_from_measurements, resolve_device, save_config, set_seed
from .visualize import save_recon_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Small-set overfit proof for Phase 9 HQ models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(rows: list[dict], key: str, path: Path, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt

        xs = [int(row["step"]) for row in rows]
        ys = [float(row[key]) for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"Could not render plot: {exc}\n", encoding="utf-8")


@torch.no_grad()
def evaluate_train_set(generator, loader, measurement, device: torch.device, config: dict, sample_path: Path | None = None) -> dict:
    generator.eval()
    metrics_rows = []
    first_payload = None
    for batch_idx, batch in enumerate(loader):
        x = batch[0].to(device, non_blocking=True)
        y = measurement.measure(x)
        x_hat, x_data, extras = reconstruct_from_measurements(
            generator,
            measurement,
            y,
            use_null_project=bool(config["use_null_project"]),
            use_dc_project=bool(config["use_dc_project"]),
            backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
            enable_refiner=True,
            output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
            return_extras=True,
        )
        row = batch_metrics(x_hat, x, measurement, y)
        row["rel_meas_err_clamped"] = row.get("rel_meas_error", float("nan"))
        row["rel_meas_err_unclamped"] = batch_metrics(
            extras["x_hat_unclamped"], x, measurement, y
        ).get("rel_meas_error", float("nan"))
        metrics_rows.append(row)
        if batch_idx == 0:
            first_payload = (x.detach(), x_data.detach(), x_hat.detach())
    generator.train()
    result = {}
    if metrics_rows:
        for key in metrics_rows[0]:
            result[key] = float(sum(float(row[key]) for row in metrics_rows) / len(metrics_rows))
    if sample_path is not None and first_payload is not None:
        x, x_data, x_hat = first_payload
        save_recon_grid(x, x_data, x_hat, sample_path, max_items=int(config.get("num_eval_samples_to_save", 16)))
    return result


def main() -> None:
    args = parse_args()
    config = apply_experiment_defaults(load_config(args.config))
    if args.device is not None:
        config["device"] = args.device
    device = resolve_device(config["device"])
    set_seed(int(config["seed"]))
    output_dir = ensure_dir(config["output_dir"])
    save_config(config, output_dir / "resolved_config.yaml")

    train_loader, _ = get_dataloaders(
        dataset_root=config["dataset_root"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        limit_train_samples=config["limit_train_samples"],
        limit_val_samples=config["limit_val_samples"],
        seed=config["seed"],
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
        use_augmentation=bool(config.get("use_augmentation", False)),
    )
    measurement = make_measurement(config, device)
    generator = build_generator(config, measurement=measurement).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=float(config["lr_g"]), betas=tuple(config["betas"]))
    use_amp = bool(config.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    max_steps = int(config.get("max_steps", int(config["epochs"]) * max(1, len(train_loader))))
    eval_every = max(1, int(config.get("eval_every", 10)))
    best_psnr = float("-inf")
    best_metrics = {}
    rows = []
    step = 0
    success = False

    generator.train()
    for epoch in range(1, int(config["epochs"]) + 1):
        progress = tqdm(train_loader, desc=f"Overfit epoch {epoch}/{config['epochs']}")
        for batch in progress:
            step += 1
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                x_hat, _x_data, extras = reconstruct_from_measurements(
                    generator,
                    measurement,
                    y,
                    use_null_project=bool(config["use_null_project"]),
                    use_dc_project=bool(config["use_dc_project"]),
                    backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                    enable_refiner=True,
                    output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                    return_extras=True,
                )
                dc_target = extras.get("x_hat_unclamped", x_hat)
                l1 = reconstruction_loss(x_hat, x)
                charb = charbonnier_loss(x_hat, x)
                ssim_l = differentiable_ssim_loss(x_hat, x)
                edge = sobel_edge_loss(x_hat, x)
                grad_l = gradient_difference_loss(x_hat, x)
                freq_l = frequency_loss(x_hat, x)
                tv = total_variation_loss(x_hat)
                dc = data_consistency_loss(measurement, dc_target, y)
                loss = (
                    float(config.get("lambda_l1", 0.0)) * l1
                    + float(config.get("lambda_charbonnier", 0.0)) * charb
                    + float(config.get("lambda_ssim", 0.0)) * ssim_l
                    + float(config.get("lambda_edge", 0.0)) * edge
                    + float(config.get("lambda_gradient", 0.0)) * grad_l
                    + float(config.get("lambda_frequency", 0.0)) * freq_l
                    + float(config.get("lambda_tv", 0.0)) * tv
                    + float(config.get("lambda_dc_loss", 0.0)) * dc
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

            if step == 1 or step % eval_every == 0 or step >= max_steps:
                metrics = evaluate_train_set(
                    generator,
                    train_loader,
                    measurement,
                    device,
                    config,
                    sample_path=output_dir / "overfit_examples_last.png",
                )
                row = {
                    "epoch": epoch,
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "train_mse": metrics.get("mse", float("nan")),
                    "train_psnr": metrics.get("psnr", float("nan")),
                    "train_ssim": metrics.get("ssim", float("nan")),
                    "rel_meas_error": metrics.get("rel_meas_error", float("nan")),
                    "rel_meas_err_unclamped": metrics.get("rel_meas_err_unclamped", float("nan")),
                    "rel_meas_err_clamped": metrics.get("rel_meas_err_clamped", float("nan")),
                }
                rows.append(row)
                if float(row["train_psnr"]) > best_psnr:
                    best_psnr = float(row["train_psnr"])
                    best_metrics = row
                    evaluate_train_set(
                        generator,
                        train_loader,
                        measurement,
                        device,
                        config,
                        sample_path=output_dir / "overfit_examples_best.png",
                    )
                    torch.save(
                        {"generator": generator.state_dict(), "config": config, "metrics": row, "epoch": epoch},
                        output_dir / "best_overfit.pt",
                    )
                success = float(row["train_psnr"]) >= 30.0 and float(row["train_ssim"]) >= 0.90
                if success:
                    break
            if step >= max_steps:
                break
        if success or step >= max_steps:
            break

    write_csv(rows, output_dir / "overfit_metrics.csv")
    plot_curve(rows, "train_psnr", output_dir / "overfit_curve_psnr.png", "train PSNR")
    plot_curve(rows, "train_ssim", output_dir / "overfit_curve_ssim.png", "train SSIM")

    final = rows[-1] if rows else {}
    reasons = []
    if not success:
        reasons = [
            "measurement/backprojection scaling",
            "DC projection / clamp conflict",
            "loss weight imbalance",
            "model capacity",
            "optimizer not updating refiner",
            "AMP issue",
        ]
    lines = [
        "# Phase 9 Overfit HQ",
        "",
        f"- overfit_failed: {not success}",
        f"- success_threshold: train PSNR >= 30 and SSIM >= 0.90",
        f"- best_train_psnr: {best_metrics.get('train_psnr', float('nan'))}",
        f"- best_train_ssim: {best_metrics.get('train_ssim', float('nan'))}",
        f"- final_train_psnr: {final.get('train_psnr', float('nan'))}",
        f"- final_train_ssim: {final.get('train_ssim', float('nan'))}",
        f"- steps_actual: {step}",
        f"- best_sample: {output_dir / 'overfit_examples_best.png'}",
        f"- last_sample: {output_dir / 'overfit_examples_last.png'}",
        "",
        "## Possible Reasons If Failed",
        "",
    ]
    lines.extend([f"- {reason}" for reason in reasons] or ["- none"])
    (output_dir / "overfit_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"overfit_success={success}")
    print(f"best_train_psnr={best_metrics.get('train_psnr', float('nan'))}")
    print(f"best_train_ssim={best_metrics.get('train_ssim', float('nan'))}")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
