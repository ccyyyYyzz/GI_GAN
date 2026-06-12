from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .datasets import get_dataloaders, get_val_dataloader
from .eval import make_measurement
from .metrics import batch_metrics
from .phase15r_common import apply_A_override, load_exact_A
from .utils import apply_experiment_defaults, load_config, set_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
E_ROOT = Path("E:/ns_mc_gan_gi")
PHASE25 = E_ROOT / "outputs_phase25"

METHODS = {
    "rademacher5": {
        "config": REPO_ROOT / "configs" / "phase14_colab" / "rademacher5_hq_noise001_colab.yaml",
        "method_id": "rademacher5_hq_noise001_colab",
        "family": "rademacher",
        "sampling_ratio": 0.05,
        "exact_A": True,
    },
    "scrambled5": {
        "config": REPO_ROOT / "configs" / "phase14_colab" / "scrambled_hadamard5_hq_noise001_colab.yaml",
        "method_id": "scrambled_hadamard5_hq_noise001_colab",
        "family": "scrambled_hadamard",
        "sampling_ratio": 0.05,
        "exact_A": False,
    },
    "rademacher10": {
        "config": REPO_ROOT / "configs" / "colab" / "rademacher10_full_noise001_colab.yaml",
        "method_id": "rademacher10_full_noise001_colab",
        "family": "rademacher",
        "sampling_ratio": 0.10,
        "exact_A": True,
    },
    "scrambled10": {
        "config": REPO_ROOT / "configs" / "colab" / "scrambled_hadamard10_full_noise001_colab.yaml",
        "method_id": "scrambled_hadamard10_full_noise001_colab",
        "family": "scrambled_hadamard",
        "sampling_ratio": 0.10,
        "exact_A": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 25 PCA linear-prior oracle.")
    parser.add_argument("--output_dir", default=str(PHASE25 / "limit_analysis" / "pca_oracle"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", default="rademacher5,scrambled5,rademacher10,scrambled10")
    parser.add_argument("--k_list", default="32,64,128,256")
    parser.add_argument("--max_train_samples", type=int, default=1024)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    return ""


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(field, "")).replace("|", "/") for field in fields) + "|")
    return "\n".join(lines)


def load_method_config(method_key: str, device: torch.device, batch_size: int, max_eval_samples: int) -> dict[str, Any]:
    meta = METHODS[method_key]
    config = apply_experiment_defaults(load_config(meta["config"]))
    config["dataset_root"] = str(E_ROOT / "data")
    config["device"] = str(device)
    config["batch_size"] = int(batch_size)
    config["num_workers"] = 0
    config["limit_val_samples"] = int(max_eval_samples)
    return config


def make_method_measurement(method_key: str, config: dict[str, Any], device: torch.device) -> tuple[Any, dict[str, Any]]:
    meta = METHODS[method_key]
    measurement = make_measurement(config, device)
    override = {"exact_A_loaded": False, "exact_A_required": bool(meta["exact_A"])}
    if meta["exact_A"]:
        A = load_exact_A(meta["method_id"], device)
        override = apply_A_override(measurement, A, "safe_rebuild")
        override["exact_A_loaded"] = True
    return measurement, override


def collect_training_matrix(config: dict[str, Any], max_train_samples: int, batch_size: int) -> torch.Tensor:
    train_loader, _val_loader = get_dataloaders(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(batch_size),
        num_workers=0,
        limit_train_samples=int(max_train_samples),
        limit_val_samples=1,
        seed=int(config["seed"]),
        train_split=config.get("train_split", "train+unlabeled"),
        val_split=config.get("val_split", "test"),
        pin_memory=False,
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
        use_augmentation=False,
    )
    rows = []
    total = 0
    for batch in train_loader:
        x = batch[0].float()
        rows.append(x.reshape(x.shape[0], -1))
        total += int(x.shape[0])
        if total >= max_train_samples:
            break
    if not rows:
        raise RuntimeError("No training samples were loaded for PCA.")
    return torch.cat(rows, dim=0)[:max_train_samples].contiguous()


def compute_pca_basis(train_matrix: torch.Tensor, max_k: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    start = time.time()
    train_matrix = train_matrix.float()
    mean = train_matrix.mean(dim=0)
    centered = train_matrix - mean
    _u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    max_rank = min(int(max_k), int(vh.shape[0]))
    basis = vh[:max_rank].contiguous()
    energy = s.square()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-12)
    info = {
        "num_train_samples": int(train_matrix.shape[0]),
        "n": int(train_matrix.shape[1]),
        "rank_available": int(vh.shape[0]),
        "basis_rows": int(basis.shape[0]),
        "seconds": time.time() - start,
        "explained_energy": {
            str(k): float(cumulative[min(k, cumulative.numel()) - 1].item())
            for k in [32, 64, 128, 256]
            if cumulative.numel() >= 1
        },
    }
    return mean, basis, info


def pca_reconstruct(
    measurement,
    y: torch.Tensor,
    mean: torch.Tensor,
    basis: torch.Tensor,
    requested_k: int,
    ridge: float,
) -> tuple[torch.Tensor, int]:
    effective_k = min(int(requested_k), int(basis.shape[0]))
    U = basis[:effective_k].to(device=y.device, dtype=torch.float32)
    mean = mean.to(device=y.device, dtype=torch.float32)
    A = measurement.A.to(device=y.device, dtype=torch.float32)
    AU = A @ U.T
    lhs = AU.T @ AU + float(ridge) * torch.eye(effective_k, device=y.device, dtype=torch.float32)
    rhs = (y.float() - (A @ mean).unsqueeze(0)) @ AU
    z = torch.linalg.solve(lhs, rhs.T).T
    recon_flat = mean.unsqueeze(0) + z @ U
    return measurement.unflatten_img(recon_flat).clamp(0.0, 1.0), effective_k


@torch.no_grad()
def evaluate_method(
    method_key: str,
    mean: torch.Tensor,
    basis: torch.Tensor,
    k_values: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = load_method_config(method_key, device, args.batch_size, args.max_eval_samples)
    set_seed(int(config["seed"]))
    measurement, override = make_method_measurement(method_key, config, device)
    loader = get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(args.batch_size),
        num_workers=0,
        limit_val_samples=int(args.max_eval_samples),
        seed=int(config["seed"]),
        val_split=config.get("val_split", "test"),
        pin_memory=False,
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )
    accum: dict[int, list[dict[str, float]]] = {k: [] for k in k_values}
    bp_metrics = []
    count = 0
    start = time.time()
    for batch in loader:
        x = batch[0].to(device).float()
        y = measurement.measure(x)
        x_data = measurement.unflatten_img(measurement.data_solution(y, mode=config.get("backprojection_mode", "ridge_pinv")))
        bp_metrics.append(batch_metrics(x_data, x, measurement, y))
        for k in k_values:
            recon, effective_k = pca_reconstruct(measurement, y, mean, basis, k, float(args.ridge))
            metrics = batch_metrics(recon, x, measurement, y)
            metrics["effective_k"] = float(effective_k)
            accum[k].append(metrics)
        count += int(x.shape[0])
    rows = []
    bp = mean_metrics(bp_metrics)
    for k in k_values:
        metrics = mean_metrics(accum[k])
        rows.append(
            {
                "method": method_key,
                "method_id": METHODS[method_key]["method_id"],
                "family": METHODS[method_key]["family"],
                "sampling_ratio": METHODS[method_key]["sampling_ratio"],
                "requested_k": k,
                "effective_k": int(round(metrics.get("effective_k", 0.0))),
                "eval_samples": count,
                "pca_mse": metrics.get("mse", float("nan")),
                "pca_psnr": metrics.get("psnr", float("nan")),
                "pca_ssim": metrics.get("ssim", float("nan")),
                "pca_rel_meas_err": metrics.get("rel_meas_error", float("nan")),
                "backproj_psnr": bp.get("psnr", float("nan")),
                "backproj_ssim": bp.get("ssim", float("nan")),
                "exact_A_loaded": override.get("exact_A_loaded", False),
                "seconds": time.time() - start,
            }
        )
    return rows, {"method": method_key, "override": override, "eval_samples": count}


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    out = {}
    for key in keys:
        values = [safe_float(item.get(key)) for item in items]
        values = [value for value in values if math.isfinite(value)]
        out[key] = float(sum(values) / len(values)) if values else float("nan")
    return out


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_train_samples = min(args.max_train_samples, 96)
        args.max_eval_samples = min(args.max_eval_samples, 8)
        args.batch_size = min(args.batch_size, 8)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    methods = parse_csv_strings(args.methods)
    k_values = parse_csv_ints(args.k_list)
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"Unknown method keys: {invalid}")
    base_config = load_method_config(methods[0], device, args.batch_size, args.max_eval_samples)
    train_matrix = collect_training_matrix(base_config, args.max_train_samples, args.batch_size)
    mean, basis, pca_info = compute_pca_basis(train_matrix, max(k_values))
    rows = []
    method_info = []
    for method in methods:
        method_rows, info = evaluate_method(method, mean, basis, k_values, args, device)
        rows.extend(method_rows)
        method_info.append(info)

    fields = [
        "method",
        "method_id",
        "family",
        "sampling_ratio",
        "requested_k",
        "effective_k",
        "eval_samples",
        "pca_mse",
        "pca_psnr",
        "pca_ssim",
        "pca_rel_meas_err",
        "backproj_psnr",
        "backproj_ssim",
        "exact_A_loaded",
        "seconds",
    ]
    write_csv_rows(output_dir / "pca_oracle_results.csv", rows, fields)
    pretty_rows = []
    for row in rows:
        pretty = dict(row)
        for key in ["sampling_ratio", "pca_mse", "pca_psnr", "pca_ssim", "pca_rel_meas_err", "backproj_psnr", "backproj_ssim", "seconds"]:
            pretty[key] = fmt(pretty.get(key), 4)
        pretty_rows.append(pretty)
    write_text(
        output_dir / "pca_oracle_results.md",
        "# Phase 25 PCA Linear-Prior Oracle Results\n\n"
        + ("Smoke run: small sample budget, not final evidence.\n\n" if args.smoke else "")
        + markdown_table(pretty_rows, fields),
    )
    manifest = {
        "phase": 25,
        "smoke": bool(args.smoke),
        "device": str(device),
        "output_dir": str(output_dir),
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "k_values": k_values,
        "pca_info": pca_info,
        "method_info": method_info,
        "results_csv": str(output_dir / "pca_oracle_results.csv"),
        "results_md": str(output_dir / "pca_oracle_results.md"),
    }
    write_json(output_dir / "pca_oracle_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
