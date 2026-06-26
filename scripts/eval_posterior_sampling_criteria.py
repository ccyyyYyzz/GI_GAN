from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval import make_measurement
from src.models import build_generator
from src.utils import apply_experiment_defaults, load_config, reconstruct_from_measurements, set_seed


DEFAULT_DATA_ROOT = Path("/mnt/e/ns_mc_gan_gi")
DEFAULT_OUT = DEFAULT_DATA_ROOT / "outputs_phase79_posterior_anti_collapse" / "baseline_rad5_collapse_eval"
DEFAULT_RAD5_ROOT = DEFAULT_DATA_ROOT / "outputs_phase15" / "imported_noleak" / "rademacher5_hq_noise001_colab"
DEFAULT_CACHE_ROOT = DEFAULT_DATA_ROOT / "results" / "cert_package_20260612" / "cache"


def resolve_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}:
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        if platform.system().lower() == "linux":
            return Path(f"/mnt/{drive}/{rest}")
    return Path(text)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_np(arr: np.ndarray, *, sort_int64: bool = False) -> str:
    x = np.asarray(arr)
    if sort_int64:
        x = np.sort(x.astype(np.int64))
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def p0_exact(v: torch.Tensor, a64: torch.Tensor, gram64: torch.Tensor) -> torch.Tensor:
    v64 = v.to(torch.float64)
    av = v64 @ a64.T
    sol = torch.linalg.solve(gram64, av.T).T
    return v64 - sol @ a64


def relmeas(x_flat: torch.Tensor, y: torch.Tensor, a64: torch.Tensor) -> torch.Tensor:
    x64 = x_flat.to(torch.float64)
    y64 = y.to(torch.float64)
    pred = x64 @ a64.T
    return torch.linalg.norm(pred - y64, dim=1) / torch.linalg.norm(y64, dim=1).clamp_min(1e-12)


def radial_power(images: np.ndarray, bins: int = 32) -> np.ndarray:
    arr = np.asarray(images, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[:, 0]
    h, w = arr.shape[-2:]
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    edges = np.linspace(0.0, float(rr.max()) + 1e-6, int(bins) + 1)
    profiles = []
    for img in arr.reshape(-1, h, w):
        power = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
        prof = np.zeros(int(bins), dtype=np.float64)
        for i in range(int(bins)):
            mask = (rr >= edges[i]) & (rr < edges[i + 1])
            prof[i] = float(power[mask].mean()) if np.any(mask) else 0.0
        total = prof.sum()
        profiles.append(prof / max(total, 1e-30))
    return np.mean(np.stack(profiles, axis=0), axis=0)


def spectrum_summary(profile: np.ndarray) -> dict[str, float]:
    p = np.asarray(profile, dtype=np.float64)
    eps = 1e-30
    lo = float(p[1 : max(2, len(p) // 6)].mean()) if len(p) > 3 else float(p[0])
    hi = float(p[max(1, (2 * len(p)) // 3) :].mean())
    xs = np.arange(1, len(p), dtype=np.float64)
    ys = np.log(p[1:] + eps)
    slope = float(np.polyfit(np.log(xs + 1.0), ys, deg=1)[0]) if len(xs) >= 2 else float("nan")
    return {
        "low_band_power_mean": lo,
        "high_band_power_mean": hi,
        "high_to_low_power_ratio": hi / max(lo, eps),
        "loglog_spectral_slope": slope,
        "radial_profile_cv": float(p.std() / max(p.mean(), eps)),
    }


def save_spectrum_artifacts(
    out_dir: Path,
    p0_samples: np.ndarray,
    p0_variations: np.ndarray,
    p0_ground_truth: np.ndarray,
    bins: int = 32,
) -> dict[str, Any]:
    sample_profile = radial_power(p0_samples, bins=bins)
    variation_profile = radial_power(p0_variations, bins=bins)
    gt_profile = radial_power(p0_ground_truth, bins=bins)
    rows = []
    for i in range(int(bins)):
        rows.append(
            {
                "bin": i,
                "p0_xhat_power": float(sample_profile[i]),
                "p0_variation_power": float(variation_profile[i]),
                "p0_ground_truth_power": float(gt_profile[i]),
            }
        )
    csv_path = out_dir / "p0_radial_power_spectrum.csv"
    png_path = out_dir / "p0_radial_power_spectrum.png"
    write_csv(csv_path, rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(int(bins))
    ax.plot(xs, sample_profile, marker="o", markersize=3, label="P0 xhat")
    ax.plot(xs, variation_profile, marker="o", markersize=3, label="P0 variation")
    ax.plot(xs, gt_profile, marker="o", markersize=3, label="P0 ground truth")
    ax.set_yscale("log")
    ax.set_xlabel("radial frequency bin")
    ax.set_ylabel("normalized mean power")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    summary = {
        "bins": int(bins),
        "csv": str(csv_path),
        "png": str(png_path),
        "p0_xhat": spectrum_summary(sample_profile),
        "p0_variation": spectrum_summary(variation_profile),
        "p0_ground_truth": spectrum_summary(gt_profile),
        "interpretation_hint": "White noise has a flatter radial profile and log-log slope near 0; natural-image-like content should have stronger low-frequency power and a negative slope.",
    }
    save_json(out_dir / "p0_spectrum_summary.json", summary)
    return summary


def load_generator(
    checkpoint_path: Path,
    config: dict[str, Any],
    measurement,
    device: torch.device,
    state_key: str,
    output_dir: Path,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    merged = dict(config)
    if isinstance(checkpoint, dict) and checkpoint.get("config"):
        merged.update(checkpoint["config"])
    merged["device"] = str(device)
    merged["dataset_root"] = str(resolve_path(merged.get("dataset_root", DEFAULT_DATA_ROOT)) or DEFAULT_DATA_ROOT)
    merged["output_dir"] = str(output_dir)
    merged["num_workers"] = 0
    merged["use_augmentation"] = False
    merged["use_final_dc_project"] = True
    merged["output_range_mode"] = "clamp_eval_only"
    merged = apply_experiment_defaults(merged)
    generator = build_generator(merged, measurement=measurement).to(device)
    if state_key == "auto":
        state = checkpoint.get("generator_ema") or checkpoint.get("generator")
        actual_key = "generator_ema" if checkpoint.get("generator_ema") is not None else "generator"
    else:
        state = checkpoint.get(state_key)
        actual_key = state_key
    if state is None:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no state key {state_key!r}.")
    generator.load_state_dict(state, strict=True)
    generator.eval()
    return generator, merged, actual_key, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-y posterior sampling anti-collapse criteria."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_RAD5_ROOT / "last.pt"))
    parser.add_argument("--config", default=str(DEFAULT_RAD5_ROOT / "resolved_config.yaml"))
    parser.add_argument("--A", default=str(DEFAULT_CACHE_ROOT / "A_rad5.npy"))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_ROOT / "main_rad5.npz"))
    parser.add_argument("--split_train", default=str(DEFAULT_CACHE_ROOT / "split_train_indices_stl10_train_unlabeled.npy"))
    parser.add_argument("--split_eval", default=str(DEFAULT_CACHE_ROOT / "split_eval_indices_stl10_test.npy"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state_key", default="auto", choices=["auto", "generator", "generator_ema"])
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--num_y", type=int, default=1)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--std_threshold", type=float, default=0.01)
    parser.add_argument(
        "--p0_var_threshold",
        type=float,
        default=None,
        help="Absolute P0 variance threshold for criterion2. Defaults to std_threshold^2.",
    )
    parser.add_argument("--null_range_ratio_threshold", type=float, default=5.0)
    parser.add_argument("--relmeas_threshold", type=float, default=1e-2)
    parser.add_argument("--spectrum_bins", type=int, default=32)
    parser.add_argument("--regime", default="rad5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    config_path = resolve_path(args.config)
    a_path = resolve_path(args.A)
    cache_path = resolve_path(args.cache)
    split_train_path = resolve_path(args.split_train)
    split_eval_path = resolve_path(args.split_eval)
    out_dir = resolve_path(args.output_dir)
    assert checkpoint_path is not None
    assert config_path is not None
    assert a_path is not None
    assert cache_path is not None
    assert out_dir is not None
    p0_var_threshold = (
        float(args.std_threshold) ** 2
        if args.p0_var_threshold is None
        else float(args.p0_var_threshold)
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    set_seed(int(args.seed))
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    config = apply_experiment_defaults(config)
    config["dataset_root"] = str(resolve_path(config.get("dataset_root", DEFAULT_DATA_ROOT)) or DEFAULT_DATA_ROOT)
    config["output_dir"] = str(out_dir)
    config["device"] = str(device)
    config["num_workers"] = 0
    config["use_augmentation"] = False
    config["use_final_dc_project"] = True
    config["output_range_mode"] = "clamp_eval_only"

    measurement = make_measurement(config, device)
    a_np = np.load(a_path).astype(np.float32)
    a_tensor = torch.from_numpy(a_np).to(device)
    override_stats = measurement.set_A_override(
        a_tensor,
        metadata={"source": str(a_path), "regime": str(args.regime), "sha256_float32_bytes": sha256_np(a_np)},
        rebuild_cache=True,
    )
    generator, merged_config, actual_state_key, checkpoint = load_generator(
        checkpoint_path, config, measurement, device, str(args.state_key), out_dir
    )

    cache = np.load(cache_path, allow_pickle=False)
    total = int(cache["y"].shape[0])
    start = int(args.sample_offset)
    stop = start + int(args.num_y)
    if start < 0 or stop > total:
        raise ValueError(f"Requested y slice [{start}:{stop}] exceeds cache size {total}.")
    img_size = int(merged_config.get("img_size", measurement.img_size))
    x_np = cache["x"][start:stop].reshape(int(args.num_y), 1, img_size, img_size).astype(np.float32)
    y_np = cache["y"][start:stop].astype(np.float32)
    labels_np = cache["labels"][start:stop].astype(np.int64) if "labels" in cache.files else np.full((args.num_y,), -1)
    indices_np = (
        cache["indices"][start:stop].astype(np.int64)
        if "indices" in cache.files
        else np.arange(start, stop, dtype=np.int64)
    )

    a64 = torch.from_numpy(a_np.astype(np.float64)).to(device)
    gram64 = a64 @ a64.T
    gram_eigs = torch.linalg.eigvalsh(gram64).detach().cpu().numpy()

    all_samples_unclipped: list[np.ndarray] = []
    all_samples_clipped: list[np.ndarray] = []
    all_p0_samples: list[np.ndarray] = []
    all_p0_variations: list[np.ndarray] = []
    all_p0_gt: list[np.ndarray] = []
    per_y_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for local_i in range(int(args.num_y)):
            y_one = torch.from_numpy(y_np[local_i : local_i + 1]).to(device)
            y_rep = y_one.repeat(int(args.K), 1)
            x_hat, _x_data, extras = reconstruct_from_measurements(
                generator,
                measurement,
                y_rep,
                use_null_project=bool(merged_config.get("use_null_project", True)),
                use_dc_project=bool(merged_config.get("use_dc_project", True)),
                use_final_dc_project=True,
                backprojection_mode=merged_config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=True,
                output_range_mode=merged_config.get("output_range_mode", "clamp_eval_only"),
                return_extras=True,
            )
            xhat_unclipped = extras["x_hat_unclamped"].float()
            xhat_flat = measurement.flatten_img(xhat_unclipped)
            samples = xhat_flat.detach()
            centered = samples - samples.mean(dim=0, keepdim=True)
            p0 = p0_exact(centered, a64, gram64)
            p0_samples = p0_exact(samples, a64, gram64)
            x_gt_flat = torch.from_numpy(x_np[local_i : local_i + 1]).to(device).reshape(1, -1)
            p0_gt = p0_exact(x_gt_flat, a64, gram64)
            pr = centered.to(torch.float64) - p0
            rel = relmeas(samples, y_rep, a64)
            mean_pixel_std = float(samples.std(dim=0, unbiased=False).mean().detach().cpu())
            null_variance_mean = float((p0 * p0).mean().detach().cpu())
            range_variance_mean = float((pr * pr).mean().detach().cpu())
            ratio = null_variance_mean / max(range_variance_mean, 1e-30)
            row = {
                "regime": str(args.regime),
                "cache_ordinal": start + local_i,
                "sample_index": int(indices_np[local_i]),
                "label": int(labels_np[local_i]),
                "K": int(args.K),
                "mean_pixel_std_unclipped": mean_pixel_std,
                "null_variance_mean": null_variance_mean,
                "range_variance_mean": range_variance_mean,
                "null_to_range_variance_ratio": float(ratio),
                "relmeaserr_mean": float(rel.mean().detach().cpu()),
                "relmeaserr_p95": float(torch.quantile(rel.detach().cpu(), 0.95)),
                "relmeaserr_max": float(rel.max().detach().cpu()),
                "criterion1_std_gt_threshold": bool(mean_pixel_std > float(args.std_threshold)),
                "criterion2_p0_var_gt_threshold": bool(null_variance_mean > p0_var_threshold),
                "criterion2_null_range_ratio_gt_threshold": bool(ratio > float(args.null_range_ratio_threshold)),
                "criterion3_relmeaserr_max_lt_threshold": bool(float(rel.max().detach().cpu()) < float(args.relmeas_threshold)),
            }
            row["criterion2_p0_var_and_ratio_pass"] = bool(
                row["criterion2_p0_var_gt_threshold"]
                and row["criterion2_null_range_ratio_gt_threshold"]
            )
            row["all_criteria_pass"] = bool(
                row["criterion1_std_gt_threshold"]
                and row["criterion2_p0_var_and_ratio_pass"]
                and row["criterion3_relmeaserr_max_lt_threshold"]
            )
            per_y_rows.append(row)
            all_samples_unclipped.append(xhat_unclipped.detach().cpu().numpy().astype(np.float32))
            all_samples_clipped.append(x_hat.detach().cpu().numpy().astype(np.float32))
            all_p0_samples.append(p0_samples.detach().cpu().numpy().astype(np.float32).reshape(int(args.K), 1, img_size, img_size))
            all_p0_variations.append(p0.detach().cpu().numpy().astype(np.float32).reshape(int(args.K), 1, img_size, img_size))
            all_p0_gt.append(p0_gt.detach().cpu().numpy().astype(np.float32).reshape(1, 1, img_size, img_size))

    samples_unclipped = np.stack(all_samples_unclipped, axis=0)
    samples_clipped = np.stack(all_samples_clipped, axis=0)
    p0_samples_arr = np.stack(all_p0_samples, axis=0)
    p0_variations_arr = np.stack(all_p0_variations, axis=0)
    p0_gt_arr = np.stack(all_p0_gt, axis=0)
    spectrum = save_spectrum_artifacts(
        out_dir,
        p0_samples_arr.reshape(-1, 1, img_size, img_size),
        p0_variations_arr.reshape(-1, 1, img_size, img_size),
        p0_gt_arr.reshape(-1, 1, img_size, img_size),
        bins=int(args.spectrum_bins),
    )
    np.savez_compressed(
        out_dir / "per_sample_outputs.npz",
        samples_unclipped=samples_unclipped,
        samples_clipped=samples_clipped,
        p0_samples=p0_samples_arr,
        p0_variations=p0_variations_arr,
        p0_ground_truth=p0_gt_arr,
        y=y_np,
        x=x_np,
        labels=labels_np,
        indices=indices_np,
        A=a_np,
    )

    split_hashes: dict[str, Any] = {}
    if split_train_path is not None and split_train_path.exists():
        train_idx = np.load(split_train_path).astype(np.int64)
        split_hashes["split_train_path"] = str(split_train_path)
        split_hashes["split_train_sha256_loader_order"] = sha256_np(train_idx)
        split_hashes["split_train_sha256_sorted_int64"] = sha256_np(train_idx, sort_int64=True)
    if split_eval_path is not None and split_eval_path.exists():
        eval_idx = np.load(split_eval_path).astype(np.int64)
        split_hashes["split_eval_path"] = str(split_eval_path)
        split_hashes["split_eval_sha256_loader_order"] = sha256_np(eval_idx)
        split_hashes["split_eval_sha256_sorted_int64"] = sha256_np(eval_idx, sort_int64=True)

    summary = {
        "script": str(Path(__file__).resolve()),
        "status": "success",
        "device_requested": str(args.device),
        "device_used": str(device),
        "regime": str(args.regime),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_state_key": actual_state_key,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "A_path": str(a_path),
        "A_sha256_float32_bytes": sha256_np(a_np),
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "output_dir": str(out_dir),
        "per_sample_outputs": str(out_dir / "per_sample_outputs.npz"),
        "thresholds": {
            "mean_pixel_std_unclipped_gt": float(args.std_threshold),
            "p0_variance_mean_gt": p0_var_threshold,
            "null_to_range_variance_ratio_gt": float(args.null_range_ratio_threshold),
            "relmeaserr_max_lt": float(args.relmeas_threshold),
        },
        "K": int(args.K),
        "num_y": int(args.num_y),
        "sample_offset": int(args.sample_offset),
        "seed": int(args.seed),
        "measurement": {
            "img_size": int(measurement.img_size),
            "n": int(measurement.n),
            "m": int(measurement.m),
            "sampling_ratio": float(measurement.sampling_ratio),
            "lambda_solver": float(merged_config.get("lambda_solver", getattr(measurement, "lambda_dc", float("nan")))),
            "pattern_type": str(merged_config.get("pattern_type", "")),
            "matrix_normalization": str(merged_config.get("matrix_normalization", "")),
            "override_stats": override_stats,
            "gram_eig_min": float(np.min(gram_eigs)),
            "gram_eig_max": float(np.max(gram_eigs)),
            "gram_condition": float(np.max(gram_eigs) / max(np.min(gram_eigs), 1e-30)),
        },
        "split_hashes": split_hashes,
        "p0_spectrum": spectrum,
        "per_y": per_y_rows,
        "overall": {
            "all_y_all_criteria_pass": bool(all(row["all_criteria_pass"] for row in per_y_rows)),
            "criterion1_all_y_pass": bool(all(row["criterion1_std_gt_threshold"] for row in per_y_rows)),
            "criterion2_all_y_pass": bool(all(row["criterion2_p0_var_and_ratio_pass"] for row in per_y_rows)),
            "criterion2_p0_var_all_y_pass": bool(all(row["criterion2_p0_var_gt_threshold"] for row in per_y_rows)),
            "criterion2_ratio_all_y_pass": bool(all(row["criterion2_null_range_ratio_gt_threshold"] for row in per_y_rows)),
            "criterion3_all_y_pass": bool(all(row["criterion3_relmeaserr_max_lt_threshold"] for row in per_y_rows)),
            "mean_pixel_std_unclipped_mean_over_y": float(np.mean([row["mean_pixel_std_unclipped"] for row in per_y_rows])),
            "p0_variance_mean_over_y": float(np.mean([row["null_variance_mean"] for row in per_y_rows])),
            "pr_variance_mean_over_y": float(np.mean([row["range_variance_mean"] for row in per_y_rows])),
            "null_to_range_variance_ratio_mean_over_y": float(np.mean([row["null_to_range_variance_ratio"] for row in per_y_rows])),
            "relmeaserr_max_over_all": float(np.max([row["relmeaserr_max"] for row in per_y_rows])),
        },
    }
    save_json(out_dir / "criteria_summary.json", summary)
    print(json.dumps(json_safe(summary["overall"]), indent=2))
    print(f"summary={out_dir / 'criteria_summary.json'}")
    print(f"per_sample_outputs={out_dir / 'per_sample_outputs.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
