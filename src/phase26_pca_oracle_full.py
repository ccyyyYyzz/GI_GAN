from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .datasets import get_dataloaders, get_val_dataloader
from .eval import make_measurement
from .exact_measurement import apply_measurement_override_from_config
from .metrics import batch_metrics
from .phase26_common import (
    REPO_ROOT,
    drive_root,
    ensure_dir,
    fmt,
    main_results_from_drive,
    markdown_table,
    output_root,
    safe_float,
    write_csv,
    write_json,
    write_text,
)
from .utils import apply_experiment_defaults, load_config, set_seed


METHODS = {
    "rademacher5": {
        "config": REPO_ROOT / "configs" / "phase14_colab" / "rademacher5_hq_noise001_colab.yaml",
        "method_id": "rademacher5_hq_noise001_colab",
        "family": "rademacher",
        "sampling_ratio": 0.05,
        "exact_A_required": True,
    },
    "scrambled5": {
        "config": REPO_ROOT / "configs" / "phase14_colab" / "scrambled_hadamard5_hq_noise001_colab.yaml",
        "method_id": "scrambled_hadamard5_hq_noise001_colab",
        "family": "scrambled_hadamard",
        "sampling_ratio": 0.05,
        "exact_A_required": False,
    },
    "rademacher10": {
        "config": REPO_ROOT / "configs" / "colab" / "rademacher10_full_noise001_colab.yaml",
        "method_id": "rademacher10_full_noise001_colab",
        "family": "rademacher",
        "sampling_ratio": 0.10,
        "exact_A_required": True,
    },
    "scrambled10": {
        "config": REPO_ROOT / "configs" / "colab" / "scrambled_hadamard10_full_noise001_colab.yaml",
        "method_id": "scrambled_hadamard10_full_noise001_colab",
        "family": "scrambled_hadamard",
        "sampling_ratio": 0.10,
        "exact_A_required": False,
    },
}


FIELDS = [
    "method_id",
    "family",
    "sampling_ratio",
    "k",
    "effective_k",
    "train_samples",
    "eval_samples",
    "pca_psnr",
    "pca_ssim",
    "pca_mse",
    "pca_rel_meas_err",
    "backproj_psnr",
    "backproj_ssim",
    "current_model_psnr",
    "current_model_ssim",
    "gap_to_current_psnr",
    "exact_A_loaded",
    "status",
    "seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 26 full PCA linear-prior oracle.")
    parser.add_argument("--drive_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k_values", default="16,32,64,128,256,512,1024")
    parser.add_argument("--methods", default="rademacher5,scrambled5,rademacher10,scrambled10")
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--pca_oversample", type=int, default=32)
    return parser.parse_args()


def csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def method_config(method_key: str, root: Path, device: torch.device, batch_size: int, eval_samples: int) -> dict[str, Any]:
    meta = METHODS[method_key]
    config = apply_experiment_defaults(load_config(meta["config"]))
    config["dataset_root"] = str(root / "data")
    config["device"] = str(device)
    config["batch_size"] = int(batch_size)
    config["num_workers"] = 0
    config["limit_val_samples"] = int(eval_samples)
    config["exact_A_required"] = bool(meta["exact_A_required"])
    if meta["exact_A_required"]:
        config["measurement_operator_exact_path"] = str(
            root / "outputs_phase15" / "imported_noleak" / meta["method_id"] / "measurement_operator_exact.pt"
        )
    return config


def collect_training_matrix(config: dict[str, Any], train_samples: int, batch_size: int) -> torch.Tensor:
    train_loader, _ = get_dataloaders(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(batch_size),
        num_workers=0,
        limit_train_samples=int(train_samples),
        limit_val_samples=1,
        seed=int(config["seed"]),
        train_split=config.get("train_split", "train+unlabeled"),
        val_split=config.get("val_split", "test"),
        pin_memory=False,
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
        use_augmentation=False,
    )
    chunks = []
    seen = 0
    for batch in train_loader:
        x = batch[0].float().reshape(batch[0].shape[0], -1)
        chunks.append(x)
        seen += int(x.shape[0])
        if seen >= train_samples:
            break
    if not chunks:
        raise RuntimeError("No training samples were loaded for PCA.")
    return torch.cat(chunks, dim=0)[:train_samples].contiguous()


def compute_pca(train_matrix: torch.Tensor, max_k: int, device: torch.device, oversample: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    start = time.time()
    mean = train_matrix.mean(dim=0)
    centered = (train_matrix - mean).to(device=device, dtype=torch.float32)
    rank_cap = min(int(centered.shape[0]), int(centered.shape[1]))
    q = min(rank_cap, int(max_k) + int(oversample))
    _u, s, v = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    basis = v[:, : min(max_k, v.shape[1])].T.contiguous().cpu()
    energy = s.square().detach().cpu()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-12)
    info = {
        "train_samples": int(train_matrix.shape[0]),
        "n": int(train_matrix.shape[1]),
        "q": int(q),
        "basis_rows": int(basis.shape[0]),
        "seconds": time.time() - start,
        "explained_energy": {
            str(k): float(cumulative[min(k, cumulative.numel()) - 1].item())
            for k in [16, 32, 64, 128, 256, 512, 1024]
            if cumulative.numel() >= 1
        },
    }
    return mean.cpu(), basis, info


def make_method_measurement(method_key: str, config: dict[str, Any], device: torch.device):
    measurement = make_measurement(config, device)
    info = apply_measurement_override_from_config(config, measurement, device)
    return measurement, info


def pca_reconstruct(measurement, y: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor, k: int, ridge: float) -> tuple[torch.Tensor, int]:
    effective_k = min(int(k), int(basis.shape[0]))
    U = basis[:effective_k].to(device=y.device, dtype=torch.float32)
    mean = mean.to(device=y.device, dtype=torch.float32)
    A = measurement.A.to(device=y.device, dtype=torch.float32)
    AU = A @ U.T
    lhs = AU.T @ AU + float(ridge) * torch.eye(effective_k, device=y.device, dtype=torch.float32)
    rhs = (y.float() - (A @ mean).unsqueeze(0)) @ AU
    z = torch.linalg.solve(lhs, rhs.T).T
    recon_flat = mean.unsqueeze(0) + z @ U
    return measurement.unflatten_img(recon_flat).clamp(0.0, 1.0), effective_k


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    out = {}
    for key in keys:
        values = [safe_float(item.get(key)) for item in items]
        values = [value for value in values if torch.isfinite(torch.tensor(value))]
        out[key] = float(sum(values) / len(values)) if values else float("nan")
    return out


@torch.no_grad()
def evaluate_method(
    method_key: str,
    root: Path,
    device: torch.device,
    mean: torch.Tensor,
    basis: torch.Tensor,
    k_values: list[int],
    args: argparse.Namespace,
    current: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = METHODS[method_key]
    config = method_config(method_key, root, device, args.batch_size, args.eval_samples)
    set_seed(int(config["seed"]))
    measurement, exact_info = make_method_measurement(method_key, config, device)
    loader = get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(args.batch_size),
        num_workers=0,
        limit_val_samples=int(args.eval_samples),
        seed=int(config["seed"]),
        val_split=config.get("val_split", "test"),
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )
    accum: dict[int, list[dict[str, float]]] = {k: [] for k in k_values}
    bp_metrics = []
    seen = 0
    start = time.time()
    for batch in loader:
        x = batch[0].to(device, non_blocking=True).float()
        y = measurement.measure(x)
        x_data = measurement.unflatten_img(measurement.data_solution(y, mode=config.get("backprojection_mode", "ridge_pinv")))
        bp_metrics.append(batch_metrics(x_data, x, measurement, y))
        for k in k_values:
            recon, effective_k = pca_reconstruct(measurement, y, mean, basis, k, float(args.ridge))
            metrics = batch_metrics(recon, x, measurement, y)
            metrics["effective_k"] = float(effective_k)
            accum[k].append(metrics)
        seen += int(x.shape[0])
    bp = mean_metrics(bp_metrics)
    cur = current.get(meta["method_id"], {})
    rows = []
    for k in k_values:
        metrics = mean_metrics(accum[k])
        pca_psnr = metrics.get("psnr", float("nan"))
        current_psnr = safe_float(cur.get("current_model_psnr"))
        rows.append(
            {
                "method_id": meta["method_id"],
                "family": meta["family"],
                "sampling_ratio": meta["sampling_ratio"],
                "k": k,
                "effective_k": int(round(metrics.get("effective_k", 0.0))),
                "train_samples": int(args.train_samples),
                "eval_samples": int(seen),
                "pca_psnr": pca_psnr,
                "pca_ssim": metrics.get("ssim", float("nan")),
                "pca_mse": metrics.get("mse", float("nan")),
                "pca_rel_meas_err": metrics.get("rel_meas_error", float("nan")),
                "backproj_psnr": bp.get("psnr", float("nan")),
                "backproj_ssim": bp.get("ssim", float("nan")),
                "current_model_psnr": current_psnr,
                "current_model_ssim": safe_float(cur.get("current_model_ssim")),
                "gap_to_current_psnr": current_psnr - pca_psnr,
                "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                "status": "ok",
                "seconds": time.time() - start,
            }
        )
    return rows, {"method": method_key, "eval_samples": seen, "exact_A_info": exact_info}


def write_results_md(path: Path, rows: list[dict[str, Any]], pca_info: dict[str, Any]) -> None:
    pretty = []
    for row in rows:
        item = dict(row)
        for key in ["sampling_ratio", "pca_psnr", "pca_ssim", "pca_mse", "pca_rel_meas_err", "backproj_psnr", "backproj_ssim", "current_model_psnr", "current_model_ssim", "gap_to_current_psnr", "seconds"]:
            item[key] = fmt(item.get(key), 4)
        pretty.append(item)
    text = f"""# Phase 26 Full PCA Oracle Results

This is a linear-prior oracle / baseline, not a deployable final reconstruction method.

PCA metadata:

```json
{pca_info}
```

{markdown_table(pretty, FIELDS)}
"""
    write_text(path, text)


def plot_metric(rows: list[dict[str, Any]], path: Path, metric: str, ylabel: str) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    labels = sorted({row["method_id"] for row in rows})
    for label in labels:
        subset = sorted([row for row in rows if row["method_id"] == label], key=lambda row: safe_float(row["k"]))
        ax.plot([safe_float(row["k"]) for row in subset], [safe_float(row[metric]) for row in subset], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("PCA subspace dimension k")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_vs_current(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    best = []
    for method_id in sorted({row["method_id"] for row in rows}):
        subset = [row for row in rows if row["method_id"] == method_id]
        best.append(sorted(subset, key=lambda row: safe_float(row["pca_psnr"]), reverse=True)[0])
    x = list(range(len(best)))
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=160)
    ax.bar([v - 0.18 for v in x], [safe_float(row["pca_psnr"]) for row in best], width=0.36, label="best PCA oracle")
    ax.bar([v + 0.18 for v in x], [safe_float(row["current_model_psnr"]) for row in best], width=0.36, label="current model")
    ax.set_xticks(x)
    ax.set_xticklabels([row["method_id"].replace("_colab", "") for row in best], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = drive_root(args.drive_root)
    out = Path(args.output_dir) if args.output_dir else output_root(root) / "pca_oracle_full"
    ensure_dir(out)
    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    methods = csv_strings(args.methods)
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    k_values = csv_ints(args.k_values)
    current = main_results_from_drive(root)

    base_config = method_config(methods[0], root, device, args.batch_size, args.eval_samples)
    train_matrix = collect_training_matrix(base_config, int(args.train_samples), int(args.batch_size))
    mean, basis, pca_info = compute_pca(train_matrix, max(k_values), device, int(args.pca_oversample))
    all_rows = []
    method_info = []
    for method in methods:
        rows, info = evaluate_method(method, root, device, mean, basis, k_values, args, current)
        all_rows.extend(rows)
        method_info.append(info)
        write_csv(out / "pca_oracle_full_results.partial.csv", all_rows, FIELDS)

    write_csv(out / "pca_oracle_full_results.csv", all_rows, FIELDS)
    write_results_md(out / "pca_oracle_full_results.md", all_rows, pca_info)
    plot_metric(all_rows, out / "pca_psnr_vs_k.png", "pca_psnr", "PCA oracle PSNR (dB)")
    plot_metric(all_rows, out / "pca_ssim_vs_k.png", "pca_ssim", "PCA oracle SSIM")
    plot_vs_current(all_rows, out / "pca_vs_current_model.png")
    manifest = {
        "phase": 26,
        "root": str(root),
        "output_dir": str(out),
        "device": str(device),
        "train_samples": int(args.train_samples),
        "eval_samples": int(args.eval_samples),
        "k_values": k_values,
        "pca_info": pca_info,
        "method_info": method_info,
        "results_csv": str(out / "pca_oracle_full_results.csv"),
        "results_md": str(out / "pca_oracle_full_results.md"),
    }
    write_json(out / "pca_oracle_full_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
