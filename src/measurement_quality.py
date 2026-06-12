from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .datasets import get_val_dataloader
from .eval import make_measurement
from .utils import apply_experiment_defaults, ensure_dir, load_config, resolve_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Measurement-only quality diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_measurement(config: dict, checkpoint_path: str | None, device: torch.device):
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            merged = dict(config)
            merged.update(checkpoint["config"])
            config = apply_experiment_defaults(merged)
    measurement = make_measurement(config, device)
    pattern_bank = getattr(measurement, "pattern_bank", None)
    if pattern_bank is not None:
        if checkpoint is None or not isinstance(checkpoint, dict) or "pattern_bank" not in checkpoint:
            raise RuntimeError("Learned-pattern measurement quality requires a checkpoint with pattern_bank.")
        pattern_bank.load_state_dict(checkpoint["pattern_bank"])
        pattern_bank.set_tau(float(config.get("pattern_tau_final", config.get("pattern_tau", 1.0))))
        pattern_bank.eval()
    return measurement, config


def gram_diagnostics(A: torch.Tensor) -> tuple[dict, torch.Tensor]:
    A_norm = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gram = A_norm @ A_norm.T
    offdiag = gram - torch.diag_embed(torch.diagonal(gram))
    eig = torch.linalg.eigvalsh(gram).real.clamp_min(0.0)
    eig_min = eig.min()
    eig_max = eig.max()
    positive = eig[eig > 1e-8]
    cond = eig_max / positive.min().clamp_min(1e-12) if positive.numel() else torch.tensor(float("inf"))
    denom = max(1, A.shape[0] * (A.shape[0] - 1))
    return (
        {
            "mean_abs_offdiag_corr": float(offdiag.abs().sum().detach().cpu() / denom),
            "max_abs_offdiag_corr": float(offdiag.abs().max().detach().cpu()),
            "gram_eigen_min": float(eig_min.detach().cpu()),
            "gram_eigen_max": float(eig_max.detach().cpu()),
            "gram_condition_number": float(cond.detach().cpu()),
        },
        eig.detach().cpu(),
    )


def secant_and_bucket_diagnostics(A: torch.Tensor, loader, device: torch.device, num_batches: int, noise_std: float):
    energies = []
    bucket_values = []
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            X = x.reshape(x.shape[0], -1)
            y = X @ A.T
            bucket_values.append(y.detach().reshape(-1).cpu())
            if X.shape[0] >= 2:
                d = X - torch.roll(X, shifts=1, dims=0)
                d = torch.nn.functional.normalize(d, p=2, dim=1, eps=1e-12)
                e = torch.sum((d @ A.T) ** 2, dim=1)
                energies.append(e.detach().cpu())
            batches += 1
            if batches >= int(num_batches):
                break
    energy = torch.cat(energies) if energies else torch.zeros(1)
    buckets = torch.cat(bucket_values) if bucket_values else torch.zeros(1)
    bucket_std = buckets.std(unbiased=False)
    snr = bucket_std / max(float(noise_std), 1e-12)
    return (
        {
            "secant_energy_mean": float(energy.mean()),
            "secant_energy_std": float(energy.std(unbiased=False)),
            "secant_energy_min": float(energy.min()),
            "secant_energy_max": float(energy.max()),
            "secant_rip_loss": float(torch.mean((energy - 1.0) ** 2)),
            "secant_energy_q05": float(torch.quantile(energy, 0.05)),
            "secant_energy_q95": float(torch.quantile(energy, 0.95)),
            "bucket_mean_abs": float(buckets.abs().mean()),
            "bucket_std": float(bucket_std),
            "bucket_dynamic_range": float(buckets.max() - buckets.min()),
            "bucket_snr_proxy": float(snr),
            "class_centroid_status": "skipped",
        },
        energy,
    )


def write_md(metrics: dict, path: Path) -> None:
    lines = ["# Measurement Quality", "", "|metric|value|", "|---|---|"]
    for key, value in metrics.items():
        if isinstance(value, float):
            value = f"{value:.6f}"
        lines.append(f"|{key}|{value}|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(eig: torch.Tensor, energy: torch.Tensor, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(eig.numpy(), marker=".", linewidth=1)
    ax.set_xlabel("Eigen index")
    ax.set_ylabel("Gram eigenvalue")
    fig.tight_layout()
    fig.savefig(output_dir / "measurement_quality_gram_spectrum.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(energy.numpy(), bins=30)
    ax.set_xlabel("Secant energy ||A d||^2")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "measurement_quality_secant_hist.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = apply_experiment_defaults(load_config(args.config))
    device = resolve_device(args.device or config.get("device", "cuda"))
    set_seed(int(config.get("seed", 42)))
    measurement, config = load_measurement(config, args.checkpoint, device)
    measurement.pattern_bank.eval() if hasattr(measurement, "pattern_bank") else None
    A = measurement.get_current_A().detach()
    loader = get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        limit_val_samples=config.get("limit_val_samples", 500),
        seed=config["seed"],
        pin_memory=device.type == "cuda",
    )
    gram, eig = gram_diagnostics(A)
    secant_bucket, energy = secant_and_bucket_diagnostics(
        A,
        loader,
        device,
        args.num_batches,
        float(config.get("noise_std", 0.0)),
    )
    pattern_stats = measurement.get_pattern_stats()
    metrics = {
        "status": "ok",
        "config": args.config,
        "checkpoint": args.checkpoint or "",
        "pattern_mode": pattern_stats.get("pattern_mode", "fixed"),
        "pattern_physical_type": pattern_stats.get("pattern_physical_type", "fixed"),
        **gram,
        **secant_bucket,
    }
    output_dir = ensure_dir(args.output_dir)
    save_json(metrics, output_dir / "measurement_quality.json")
    write_md(metrics, output_dir / "measurement_quality.md")
    save_plots(eig, energy, output_dir)
    print(f"Wrote measurement quality diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
