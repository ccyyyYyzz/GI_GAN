from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import phase69A_gauge_gan_signal_diagnostic as p69a
from src.eval import make_measurement
from src.models import build_generator
from src.utils import apply_experiment_defaults, load_config, set_seed


DEFAULT_DATA_ROOT = Path("E:/ns_mc_gan_gi")
DEFAULT_PHASE79 = DEFAULT_DATA_ROOT / "outputs_phase79_posterior_anti_collapse" / "rad5_rowspace_diversity_diagnostic"
DEFAULT_CHECKPOINT = DEFAULT_PHASE79 / "checkpoints" / "final.pt"
DEFAULT_RAD5_ROOT = DEFAULT_DATA_ROOT / "outputs_phase15" / "imported_noleak" / "rademacher5_hq_noise001_colab"
DEFAULT_CACHE_ROOT = DEFAULT_DATA_ROOT / "results" / "cert_package_20260612" / "cache"
DEFAULT_OUTPUT = DEFAULT_PHASE79 / "calibration_validation"


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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_np(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


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
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
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


def pr_exact(v: torch.Tensor, a64: torch.Tensor, gram64: torch.Tensor) -> torch.Tensor:
    return v.to(torch.float64) - p0_exact(v, a64, gram64)


def load_env(args: argparse.Namespace, device: torch.device):
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    a_path = resolve_path(args.A)
    assert config_path is not None and checkpoint_path is not None and a_path is not None
    config = apply_experiment_defaults(load_config(config_path))
    config["device"] = str(device)
    config["dataset_root"] = str(DEFAULT_DATA_ROOT / "data")
    config["output_dir"] = str(resolve_path(args.output_dir) or DEFAULT_OUTPUT)
    config["num_workers"] = 0
    config["use_augmentation"] = False
    config["use_final_dc_project"] = True
    config["output_range_mode"] = "clamp_eval_only"

    measurement = make_measurement(config, device)
    a_np = np.load(a_path).astype(np.float32)
    measurement.set_A_override(
        torch.from_numpy(a_np).to(device),
        metadata={"source": str(a_path), "sha256_float32_bytes": sha256_np(a_np)},
        rebuild_cache=True,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    merged = dict(config)
    if isinstance(ckpt, dict) and ckpt.get("config"):
        merged.update(ckpt["config"])
    merged["device"] = str(device)
    merged["dataset_root"] = str(DEFAULT_DATA_ROOT / "data")
    merged["output_dir"] = config["output_dir"]
    merged["num_workers"] = 0
    merged["use_augmentation"] = False
    merged["use_final_dc_project"] = True
    merged["output_range_mode"] = "clamp_eval_only"
    merged = apply_experiment_defaults(merged)
    generator = build_generator(merged, measurement=measurement).to(device)
    state = ckpt.get("generator_ema") or ckpt.get("generator")
    if state is None:
        raise RuntimeError(f"No generator state in {checkpoint_path}")
    generator.load_state_dict(state, strict=True)
    generator.eval()
    return config, merged, measurement, generator, a_np, checkpoint_path, config_path, a_path


def forward_with_noise(generator, measurement, y: torch.Tensor, noise: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    y32 = y.float()
    x_data_flat = p69a.data_solution_safe(measurement, y32, config.get("backprojection_mode", "ridge_pinv"))
    x_data = measurement.unflatten_img(x_data_flat)
    residual = generator(x_data, noise, y=y32)
    residual_flat = measurement.flatten_img(residual.float())
    residual_ns = measurement.null_project(residual_flat) if bool(config.get("use_null_project", True)) else residual_flat
    v_stage0 = x_data_flat + residual_ns
    x_stage1 = measurement.dc_project(v_stage0, y32) if bool(config.get("use_dc_project", True)) else v_stage0
    if hasattr(generator, "refine"):
        refine = generator.refine(x_data, measurement.unflatten_img(x_stage1))
        v_pre = x_stage1 + measurement.flatten_img(refine.float())
    else:
        v_pre = x_stage1
    return measurement.dc_project(v_pre, y32) if bool(config.get("use_final_dc_project", True)) else v_pre


def bank_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "samples": out_dir / "sample_bank_unclipped_float16.npy",
        "sample_mean": out_dir / "sample_mean_unclipped_float32.npy",
        "x": out_dir / "x_gt_float32.npy",
        "y": out_dir / "y_float32.npy",
        "deterministic": out_dir / "deterministic_rad5_xhat_unclipped_float32.npy",
        "labels": out_dir / "labels_int64.npy",
        "directions": out_dir / "p0_random_directions_float32.npy",
        "manifest": out_dir / "sample_bank_manifest.json",
    }


def generate_bank(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(resolve_path(args.output_dir) or DEFAULT_OUTPUT)
    paths = bank_paths(out_dir)
    device = torch.device(args.device if not (args.device.startswith("cuda") and not torch.cuda.is_available()) else "cpu")
    set_seed(int(args.seed))
    config, merged, measurement, generator, a_np, checkpoint_path, config_path, a_path = load_env(args, device)
    cache_path = resolve_path(args.cache)
    assert cache_path is not None
    cache = np.load(cache_path, allow_pickle=False)
    n_total = int(cache["x"].shape[0])
    sample_offset = int(args.sample_offset)
    if sample_offset < 0 or sample_offset >= n_total:
        raise ValueError(f"sample_offset {sample_offset} outside cache size {n_total}.")
    n_available = n_total - sample_offset
    n_images = n_available if int(args.num_y) <= 0 else min(int(args.num_y), n_available)
    k = int(args.K)
    n = int(measurement.n)
    img_size = int(measurement.img_size)
    samples = np.lib.format.open_memmap(paths["samples"], mode="w+", dtype=np.float16, shape=(n_images, k, n))
    sample_mean = np.lib.format.open_memmap(paths["sample_mean"], mode="w+", dtype=np.float32, shape=(n_images, n))
    cache_stop = sample_offset + n_images
    np.save(paths["x"], cache["x"][sample_offset:cache_stop].reshape(n_images, n).astype(np.float32))
    np.save(paths["y"], cache["y"][sample_offset:cache_stop].astype(np.float32))
    np.save(paths["deterministic"], cache["x_hat_unclamped"][sample_offset:cache_stop].reshape(n_images, n).astype(np.float32))
    np.save(paths["labels"], cache["labels"][sample_offset:cache_stop].astype(np.int64))

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 1009)
    chunk = int(args.chunk_y)
    rel_max = 0.0
    a64 = torch.from_numpy(a_np.astype(np.float64)).to(device)
    with torch.no_grad():
        for start in range(0, n_images, chunk):
            stop = min(start + chunk, n_images)
            src_start = sample_offset + start
            src_stop = sample_offset + stop
            y = torch.from_numpy(cache["y"][src_start:src_stop].astype(np.float32)).to(device)
            b = stop - start
            y_rep = y[:, None, :].repeat(1, k, 1).reshape(b * k, -1)
            x_data_flat = p69a.data_solution_safe(measurement, y_rep, merged.get("backprojection_mode", "ridge_pinv"))
            x_data = measurement.unflatten_img(x_data_flat)
            noise = torch.randn(x_data.shape, device=device, dtype=x_data.dtype, generator=gen)
            x_hat_flat = forward_with_noise(generator, measurement, y_rep, noise, merged)
            rel = torch.linalg.norm(x_hat_flat.to(torch.float64) @ a64.T - y_rep.to(torch.float64), dim=1) / torch.linalg.norm(y_rep.to(torch.float64), dim=1).clamp_min(1e-12)
            rel_max = max(rel_max, float(rel.max().detach().cpu()))
            arr = x_hat_flat.detach().cpu().numpy().astype(np.float32).reshape(b, k, n)
            samples[start:stop] = arr.astype(np.float16)
            sample_mean[start:stop] = arr.mean(axis=1).astype(np.float32)
            samples.flush()
            sample_mean.flush()
            print(f"generated {stop}/{n_images}", flush=True)

    manifest = {
        "mode": "sample_bank",
        "N": n_images,
        "sample_offset": sample_offset,
        "sample_stop": cache_stop,
        "K": k,
        "n": n,
        "img_size": img_size,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "A": str(a_path),
        "A_sha256_float32_bytes": sha256_np(a_np),
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "seed": int(args.seed),
        "relmeas_max_generated": rel_max,
        "samples_path": str(paths["samples"]),
        "samples_sha256": sha256_file(paths["samples"]),
        "sample_mean_path": str(paths["sample_mean"]),
        "sample_mean_sha256": sha256_file(paths["sample_mean"]),
    }
    save_json(paths["manifest"], manifest)


def load_bank(out_dir: Path) -> tuple[dict[str, Any], np.memmap, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = bank_paths(out_dir)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    samples = np.load(paths["samples"], mmap_mode="r")
    x = np.load(paths["x"], mmap_mode="r")
    y = np.load(paths["y"], mmap_mode="r")
    det = np.load(paths["deterministic"], mmap_mode="r")
    mean = np.load(paths["sample_mean"], mmap_mode="r")
    return manifest, samples, x, y, det, mean


def make_p0_directions(args: argparse.Namespace, out_dir: Path, a_np: np.ndarray, device: torch.device) -> np.ndarray:
    paths = bank_paths(out_dir)
    if paths["directions"].exists():
        return np.load(paths["directions"], mmap_mode="r")
    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed) + 2027)
    a64 = torch.from_numpy(a_np.astype(np.float64)).to(device)
    gram64 = a64 @ a64.T
    dirs = []
    remaining = int(args.num_p0_dirs)
    while remaining > 0:
        b = min(remaining * 2, 128)
        raw = torch.randn((b, a64.shape[1]), device=device, dtype=torch.float64, generator=rng)
        p0 = p0_exact(raw, a64, gram64)
        for row in p0:
            norm = torch.linalg.norm(row)
            if float(norm.detach().cpu()) > 1e-8:
                dirs.append((row / norm).detach().cpu().numpy().astype(np.float32))
                remaining -= 1
                if remaining <= 0:
                    break
    arr = np.stack(dirs, axis=0)
    np.save(paths["directions"], arr)
    return arr


def interval_coverage(values: np.ndarray, truth: np.ndarray, levels: list[float]) -> list[dict[str, Any]]:
    rows = []
    for level in levels:
        lo_q = (1.0 - float(level)) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(values, lo_q, axis=1)
        hi = np.quantile(values, hi_q, axis=1)
        inside = (truth >= lo) & (truth <= hi)
        rows.append(
            {
                "level": float(level),
                "covered_count": int(inside.sum()),
                "total_count": int(inside.size),
                "empirical_coverage": float(inside.mean()),
                "coverage_minus_nominal": float(inside.mean() - float(level)),
            }
        )
    return rows


def coverage_eval(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(resolve_path(args.output_dir) or DEFAULT_OUTPUT)
    session = ensure_dir(out_dir / "session_01_coverage")
    manifest, samples, x, _y, _det, _mean = load_bank(out_dir)
    device = torch.device(args.device if not (args.device.startswith("cuda") and not torch.cuda.is_available()) else "cpu")
    a_np = np.load(resolve_path(args.A)).astype(np.float32)
    dirs = make_p0_directions(args, out_dir, a_np, device)
    dirs_t = torch.from_numpy(np.asarray(dirs, dtype=np.float32).T).to(device)
    levels = [0.50, 0.90, 0.95]
    pixel_counts = {level: [0, 0] for level in levels}
    pixel_per_image = []
    p0_counts = {level: [0, 0] for level in levels}
    p0_per_image = []
    sample_offset = int(manifest.get("sample_offset", 0))
    chunk = int(args.metric_chunk_y)
    for start in range(0, int(manifest["N"]), chunk):
        stop = min(start + chunk, int(manifest["N"]))
        s = np.asarray(samples[start:stop], dtype=np.float32)
        truth = np.asarray(x[start:stop], dtype=np.float32)
        for row in interval_coverage(s, truth, levels):
            pixel_counts[row["level"]][0] += row["covered_count"]
            pixel_counts[row["level"]][1] += row["total_count"]
        for i in range(s.shape[0]):
            for row in interval_coverage(s[i : i + 1], truth[i : i + 1], levels):
                pixel_per_image.append({"image_ordinal": sample_offset + start + i, "space": "pixel", **row})
        with torch.no_grad():
            coeff = (
                torch.from_numpy(s.reshape(-1, s.shape[-1]))
                .to(device)
                .matmul(dirs_t)
                .detach()
                .cpu()
                .numpy()
                .reshape(s.shape[0], s.shape[1], -1)
            )
        truth_coeff = truth @ dirs.T
        for row in interval_coverage(coeff, truth_coeff, levels):
            p0_counts[row["level"]][0] += row["covered_count"]
            p0_counts[row["level"]][1] += row["total_count"]
        for i in range(s.shape[0]):
            for row in interval_coverage(coeff[i : i + 1], truth_coeff[i : i + 1], levels):
                p0_per_image.append({"image_ordinal": sample_offset + start + i, "space": "p0_random_direction", **row})
        print(f"coverage {stop}/{manifest['N']}", flush=True)
    rows = []
    for space, counts in [("pixel", pixel_counts), ("p0_random_direction", p0_counts)]:
        for level in levels:
            c, t = counts[level]
            cov = c / max(t, 1)
            rows.append({"space": space, "level": level, "covered_count": c, "total_count": t, "empirical_coverage": cov, "coverage_minus_nominal": cov - level})
    write_csv(session / "coverage_curve.csv", rows)
    write_csv(session / "coverage_per_image.csv", pixel_per_image + p0_per_image)
    summary = {
        "session": "coverage",
        "sample_bank": manifest,
        "num_p0_dirs": int(args.num_p0_dirs),
        "p0_directions_path": str(bank_paths(out_dir)["directions"]),
        "p0_directions_sha256": sha256_file(bank_paths(out_dir)["directions"]),
        "coverage_curve": rows,
    }
    save_json(session / "coverage_summary.json", summary)


def kappa_eval(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(resolve_path(args.output_dir) or DEFAULT_OUTPUT)
    session = ensure_dir(out_dir / "session_02_kappa")
    manifest, samples, x, _y, det, mean = load_bank(out_dir)
    device = torch.device(args.device if not (args.device.startswith("cuda") and not torch.cuda.is_available()) else "cpu")
    a_np = np.load(resolve_path(args.A)).astype(np.float32)
    a64 = torch.from_numpy(a_np.astype(np.float64)).to(device)
    gram64 = a64 @ a64.T
    n = int(manifest["n"])
    total_sample_mse = 0.0
    total_sample_count = 0
    total_mean_mse = 0.0
    total_det_mse = 0.0
    total_width = 0.0
    per_image = []
    sample_offset = int(manifest.get("sample_offset", 0))
    chunk = int(args.metric_chunk_y)
    for start in range(0, int(manifest["N"]), chunk):
        stop = min(start + chunk, int(manifest["N"]))
        s_np = np.asarray(samples[start:stop], dtype=np.float32)
        b, k, _n = s_np.shape
        s = torch.from_numpy(s_np.reshape(b * k, n)).to(device)
        gt = torch.from_numpy(np.asarray(x[start:stop], dtype=np.float32)).to(device)
        det_t = torch.from_numpy(np.asarray(det[start:stop], dtype=np.float32)).to(device)
        mean_t = torch.from_numpy(np.asarray(mean[start:stop], dtype=np.float32)).to(device)
        p0_s = p0_exact(s, a64, gram64).reshape(b, k, n)
        p0_gt = p0_exact(gt, a64, gram64)
        p0_det = p0_exact(det_t, a64, gram64)
        p0_mean = p0_exact(mean_t, a64, gram64)
        sample_mse_i = ((p0_s - p0_gt[:, None, :]) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
        mean_mse_i = ((p0_mean - p0_gt) ** 2).mean(dim=1).detach().cpu().numpy()
        det_mse_i = ((p0_det - p0_gt) ** 2).mean(dim=1).detach().cpu().numpy()
        width_i = ((p0_s - p0_mean[:, None, :]) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
        total_sample_mse += float(sample_mse_i.sum())
        total_sample_count += int(b)
        total_mean_mse += float(mean_mse_i.sum())
        total_det_mse += float(det_mse_i.sum())
        total_width += float(width_i.sum())
        for j in range(b):
            per_image.append(
                {
                    "image_ordinal": sample_offset + start + j,
                    "null_mse_sample_to_gt": float(sample_mse_i[j]),
                    "null_mse_sample_mean_to_gt": float(mean_mse_i[j]),
                    "null_mse_deterministic_to_gt": float(det_mse_i[j]),
                    "null_sample_width": float(width_i[j]),
                    "kappa_vs_sample_mean": float(sample_mse_i[j] / max(mean_mse_i[j], 1e-30)),
                    "kappa_vs_deterministic": float(sample_mse_i[j] / max(det_mse_i[j], 1e-30)),
                }
            )
        print(f"kappa {stop}/{manifest['N']}", flush=True)
    sample_mse = total_sample_mse / max(total_sample_count, 1)
    mean_mse = total_mean_mse / max(total_sample_count, 1)
    det_mse = total_det_mse / max(total_sample_count, 1)
    width = total_width / max(total_sample_count, 1)
    summary = {
        "session": "kappa",
        "sample_bank": manifest,
        "null_mse_sample_to_gt": sample_mse,
        "null_mse_sample_mean_to_gt": mean_mse,
        "null_mse_deterministic_to_gt": det_mse,
        "null_sample_width": width,
        "kappa_vs_sample_mean": sample_mse / max(mean_mse, 1e-30),
        "kappa_vs_deterministic": sample_mse / max(det_mse, 1e-30),
        "admissible_interval": [1.0, 2.0],
    }
    write_csv(session / "kappa_per_image.csv", per_image)
    save_json(session / "kappa_summary.json", summary)


def mean_shift_eval(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(resolve_path(args.output_dir) or DEFAULT_OUTPUT)
    session = ensure_dir(out_dir / "session_03_mean_shift")
    manifest, _samples, x, _y, det, mean = load_bank(out_dir)
    device = torch.device(args.device if not (args.device.startswith("cuda") and not torch.cuda.is_available()) else "cpu")
    a_np = np.load(resolve_path(args.A)).astype(np.float32)
    a64 = torch.from_numpy(a_np.astype(np.float64)).to(device)
    gram64 = a64 @ a64.T
    rows = []
    sums: dict[str, float] = {}
    sample_offset = int(manifest.get("sample_offset", 0))
    chunk = int(args.metric_chunk_y)
    for start in range(0, int(manifest["N"]), chunk):
        stop = min(start + chunk, int(manifest["N"]))
        m = torch.from_numpy(np.asarray(mean[start:stop], dtype=np.float32)).to(device)
        d = torch.from_numpy(np.asarray(det[start:stop], dtype=np.float32)).to(device)
        gt = torch.from_numpy(np.asarray(x[start:stop], dtype=np.float32)).to(device)
        diff = m - d
        p0_diff = p0_exact(diff, a64, gram64)
        pr_diff = diff.to(torch.float64) - p0_diff
        p0_m_gt = p0_exact(m - gt, a64, gram64)
        p0_d_gt = p0_exact(d - gt, a64, gram64)
        vals = {
            "mean_vs_det_rmse": torch.sqrt((diff * diff).mean(dim=1)),
            "mean_vs_det_mae": diff.abs().mean(dim=1),
            "mean_vs_det_p0_rmse": torch.sqrt((p0_diff * p0_diff).mean(dim=1)),
            "mean_vs_det_pr_rmse": torch.sqrt((pr_diff * pr_diff).mean(dim=1)),
            "mean_vs_det_rel_l2": torch.linalg.norm(diff, dim=1) / torch.linalg.norm(d, dim=1).clamp_min(1e-12),
            "p0_mean_to_gt_rmse": torch.sqrt((p0_m_gt * p0_m_gt).mean(dim=1)),
            "p0_det_to_gt_rmse": torch.sqrt((p0_d_gt * p0_d_gt).mean(dim=1)),
        }
        vals_np = {k: v.detach().cpu().numpy().astype(float) for k, v in vals.items()}
        for j in range(stop - start):
            row = {"image_ordinal": sample_offset + start + j}
            for k2, arr in vals_np.items():
                row[k2] = float(arr[j])
                sums[k2] = sums.get(k2, 0.0) + float(arr[j])
            rows.append(row)
        print(f"mean_shift {stop}/{manifest['N']}", flush=True)
    n_img = max(int(manifest["N"]), 1)
    summary = {
        "session": "mean_shift",
        "sample_bank": manifest,
        "means": {k: v / n_img for k, v in sums.items()},
    }
    write_csv(session / "mean_shift_per_image.csv", rows)
    save_json(session / "mean_shift_summary.json", summary)


def manifest_eval(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(resolve_path(args.output_dir) or DEFAULT_OUTPUT)
    files = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".npy", ".png"}:
            files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    save_json(out_dir / "calibration_artifact_manifest.json", {"root": str(out_dir), "files": files})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure posterior calibration validation for Phase79 Rad-5 final checkpoint.")
    parser.add_argument("--mode", choices=["generate_bank", "coverage", "kappa", "mean_shift", "manifest"], required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_RAD5_ROOT / "resolved_config.yaml"))
    parser.add_argument("--A", default=str(DEFAULT_CACHE_ROOT / "A_rad5.npy"))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_ROOT / "main_rad5.npz"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--num_y", type=int, default=0, help="0 means all frozen cache images.")
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--chunk_y", type=int, default=8)
    parser.add_argument("--metric_chunk_y", type=int, default=32)
    parser.add_argument("--num_p0_dirs", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "generate_bank":
        generate_bank(args)
    elif args.mode == "coverage":
        coverage_eval(args)
    elif args.mode == "kappa":
        kappa_eval(args)
    elif args.mode == "mean_shift":
        mean_shift_eval(args)
    elif args.mode == "manifest":
        manifest_eval(args)
    else:
        raise ValueError(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
