"""PASS/FAIL gate checker for ghost-imaging posterior-sampler result dumps."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import fid_kid, flatten_images, lpips_distance, psnr, ssim, to_nhw


ALIASES = {
    "x": ("x", "x_true", "gt", "ground_truth", "target"),
    "samples": ("samples", "x_samples", "sample"),
    "sample_mean": ("sample_mean", "mean", "x_mean", "posterior_mean"),
    "baseline": ("baseline", "baseline_output", "deterministic", "x_baseline"),
    "y": ("y", "measurements"),
    "ref_x": ("ref_x", "reference_x", "reference_images", "x_ref"),
    "samples_unclipped": ("samples_unclipped", "x_samples_unclipped", "unclipped_samples"),
    "lambda": ("lambda", "lambdas", "lambda_values"),
    "k": ("k", "sample_index", "sample_indices"),
    "A_path": ("A_path", "a_path", "measurement_matrix_path"),
    "P0_path": ("P0_path", "p0_path", "null_projector_path"),
}


@dataclass
class GateResult:
    name: str
    passed: bool
    values: dict[str, Any]
    message: str
    status: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "status": self.status or ("PASS" if self.passed else "FAIL"),
            "values": _jsonable(self.values),
            "message": self.message,
        }


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def load_array_artifact(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if len(data.files) == 1:
                return np.asarray(data[data.files[0]])
            for key in ("A", "P0", "arr_0"):
                if key in data:
                    return np.asarray(data[key])
            raise KeyError(f"{path} contains multiple arrays; expected A, P0, or arr_0")
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix in {".pt", ".pth"}:
        import torch

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if len(obj) == 1:
                obj = next(iter(obj.values()))
            else:
                for key in ("A", "P0", "arr_0"):
                    if key in obj:
                        obj = obj[key]
                        break
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu().numpy()
        return np.asarray(obj)
    raise ValueError(f"Unsupported artifact type: {path}")


def load_dump(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key].item() if data[key].shape == () else data[key] for key in data.files}
    if path.suffix in {".pt", ".pth"}:
        import torch

        obj = torch.load(path, map_location="cpu")
        return {key: (val.detach().cpu().numpy() if hasattr(val, "detach") else val) for key, val in obj.items()}
    raise ValueError(f"Unsupported result dump type: {path}")


def _pick(data: dict[str, Any], canonical: str, required: bool = True, default=None):
    for key in ALIASES[canonical]:
        if key in data:
            return data[key]
    if required:
        raise KeyError(f"Missing required result field {canonical}; tried {ALIASES[canonical]}")
    return default


def _as_path(value, base_dir: Path) -> Path:
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else str(value)
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _samples_to_flat(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        return arr.reshape(arr.shape[0], arr.shape[1], -1)
    if arr.ndim == 5 and arr.shape[2] == 1:
        return arr[:, :, 0].reshape(arr.shape[0], arr.shape[1], -1)
    raise ValueError(f"Expected samples as (N,K,n) or image tensor, got {arr.shape}")


def _repeat_targets(x_flat: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(x_flat[:, None, :], k, axis=1).reshape(-1, x_flat.shape[1])


def _project_rows(values: np.ndarray, projector: np.ndarray) -> np.ndarray:
    return values @ projector.T


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    ra = _rankdata_average(a[mask])
    rb = _rankdata_average(b[mask])
    ra -= np.mean(ra)
    rb -= np.mean(rb)
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(np.dot(ra, rb) / denom) if denom > 0 else float("nan")


def _sobel_magnitude(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    gx = (
        -padded[:-2, :-2]
        - 2 * padded[1:-1, :-2]
        - padded[2:, :-2]
        + padded[:-2, 2:]
        + 2 * padded[1:-1, 2:]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return np.hypot(gx, gy)


def _edge_rho(std_maps: np.ndarray, x_images: np.ndarray) -> float:
    rhos = []
    for std_map, x in zip(std_maps, x_images):
        edge = _sobel_magnitude(x)
        rho = _spearmanr(std_map.ravel(), edge.ravel())
        if np.isfinite(rho):
            rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def _rel_measurement_errors(samples_flat: np.ndarray, y: np.ndarray, A: np.ndarray) -> np.ndarray:
    n_img, k, _ = samples_flat.shape
    y64 = np.asarray(y, dtype=np.float64).reshape(n_img, -1)
    A64 = np.asarray(A, dtype=np.float64)
    errs = np.empty((n_img, k), dtype=np.float64)
    for i in range(n_img):
        denom = max(float(np.linalg.norm(y64[i])), 1e-300)
        residual = samples_flat[i] @ A64.T - y64[i][None, :]
        errs[i] = np.linalg.norm(residual, axis=1) / denom
    return errs


def check_results(
    dump_path: str | Path,
    *,
    A_path: str | Path | None = None,
    P0_path: str | Path | None = None,
    perceptual_backend: str = "lpips",
    compute_distributional: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    dump_path = Path(dump_path)
    data = load_dump(dump_path)
    base_dir = dump_path.parent

    x = np.asarray(_pick(data, "x"), dtype=np.float64)
    samples = np.asarray(_pick(data, "samples"), dtype=np.float64)
    sample_mean = np.asarray(_pick(data, "sample_mean"), dtype=np.float64)
    baseline = np.asarray(_pick(data, "baseline"), dtype=np.float64)
    y = np.asarray(_pick(data, "y"), dtype=np.float64)
    ref_x = np.asarray(_pick(data, "ref_x", required=False, default=x), dtype=np.float64)
    samples_unclipped = np.asarray(_pick(data, "samples_unclipped", required=False, default=samples), dtype=np.float64)

    A_path = Path(A_path) if A_path is not None else _as_path(_pick(data, "A_path"), base_dir)
    P0_path = Path(P0_path) if P0_path is not None else _as_path(_pick(data, "P0_path"), base_dir)
    A = np.asarray(load_array_artifact(A_path), dtype=np.float64)
    P0 = np.asarray(load_array_artifact(P0_path), dtype=np.float64)

    x_flat = flatten_images(x).astype(np.float64)
    mean_flat = flatten_images(sample_mean).astype(np.float64)
    baseline_flat = flatten_images(baseline).astype(np.float64)
    samples_flat = _samples_to_flat(samples)
    unclipped_flat = _samples_to_flat(samples_unclipped)
    ref_flat = flatten_images(ref_x).astype(np.float64)
    n_img, k, n_pixels = samples_flat.shape
    image_shape = (int(round(np.sqrt(n_pixels))), int(round(np.sqrt(n_pixels))))
    if image_shape[0] * image_shape[1] != n_pixels:
        raise ValueError(f"Expected square images, got {n_pixels} pixels")

    repeated_x = _repeat_targets(x_flat, k)
    sample_psnr = psnr(samples_flat.reshape(-1, n_pixels), repeated_x)
    sample_mean_psnr = psnr(mean_flat, x_flat)
    baseline_psnr = psnr(baseline_flat, x_flat)
    sample_mean_ssim = ssim(mean_flat, x_flat)
    baseline_ssim = ssim(baseline_flat, x_flat)

    sample_psnr_avg = float(np.mean(sample_psnr))
    mean_psnr_avg = float(np.mean(sample_mean_psnr))
    baseline_psnr_avg = float(np.mean(baseline_psnr))
    gates: dict[str, GateResult] = {}

    gcal_low = mean_psnr_avg - 3.5
    gcal_high = mean_psnr_avg - 1.0
    gcal_gap = mean_psnr_avg - sample_psnr_avg
    gates["G-CAL"] = GateResult(
        "G-CAL",
        gcal_low <= sample_psnr_avg <= gcal_high,
        {
            "avg_sample_psnr_db": sample_psnr_avg,
            "avg_sample_mean_psnr_db": mean_psnr_avg,
            "sample_mean_minus_sample_gap_db": gcal_gap,
            "required_gap_range_db": [1.0, 3.5],
            "required_range_db": [gcal_low, gcal_high],
        },
        "Average per-sample PSNR is calibrated relative to sample-mean PSNR.",
    )

    std_flat = np.std(samples_flat, axis=1, ddof=0)
    std_maps = std_flat.reshape(n_img, *image_shape)
    median_pixel_std = float(np.median(std_flat))
    rho = _edge_rho(std_maps, to_nhw(x_flat, image_shape))
    gates["G-DIV"] = GateResult(
        "G-DIV",
        median_pixel_std >= 1e-2 and rho > 0.2,
        {
            "median_pixel_std": median_pixel_std,
            "median_pixel_std_threshold": 1e-2,
            "spearman_std_vs_gt_sobel": rho,
            "spearman_threshold": 0.2,
        },
        "Sampler has non-trivial variance and variance concentrates on image structure.",
    )

    projected_samples = _project_rows(samples_flat.reshape(-1, n_pixels), P0).reshape(n_img, k, n_pixels)
    numerator = float(np.mean(np.var(projected_samples, axis=1, ddof=0)))
    projected_ref = _project_rows(ref_flat, P0)
    # G-NVR denominator formula:
    #   denom = (1/n) * sum_j Var_i[(P0 x_ref_i)_j]
    # numerator is the analogous average over per-image sample variance:
    #   numer = (1/(N*n)) * sum_i sum_j Var_z[(P0 x_hat_i(z))_j]
    denominator = float(np.mean(np.var(projected_ref, axis=0, ddof=0)))
    nvr = float(numerator / denominator) if denominator > 0 else float("inf")
    gates["G-NVR"] = GateResult(
        "G-NVR",
        nvr >= 0.1,
        {
            "null_variance_ratio": nvr,
            "null_variance_ratio_threshold": 0.1,
            "sample_null_variance_mean": numerator,
            "reference_null_dataset_variance_mean": denominator,
        },
        "Null-space sample variance is large enough relative to reference null-space dataset variance.",
    )

    gates["G-MEAN"] = GateResult(
        "G-MEAN",
        mean_psnr_avg >= baseline_psnr_avg - 0.3,
        {
            "avg_sample_mean_psnr_db": mean_psnr_avg,
            "avg_baseline_psnr_db": baseline_psnr_avg,
            "sample_mean_minus_baseline_psnr_db": mean_psnr_avg - baseline_psnr_avg,
            "allowed_drop_db": 0.3,
            "required_min_db": baseline_psnr_avg - 0.3,
        },
        "Posterior mean is not materially worse than the deterministic baseline.",
    )

    rel_err = _rel_measurement_errors(unclipped_flat, y, A)
    max_rel = float(np.max(rel_err))
    median_rel = float(np.median(rel_err))
    frac_above_threshold = float(np.mean(rel_err > 1e-10))
    float32_floor_flag = (
        frac_above_threshold >= 0.5
        and median_rel > 1e-10
        and median_rel <= 1e-5
        and float(np.percentile(rel_err, 95)) <= 1e-5
    )
    cert_pass = max_rel <= 1e-10 or float32_floor_flag
    gates["G-CERT"] = GateResult(
        "G-CERT",
        cert_pass,
        {
            "max_rel_measurement_error": max_rel,
            "median_rel_measurement_error": median_rel,
            "threshold": 1e-10,
            "float32_floor_flag": bool(float32_floor_flag),
            "fraction_above_threshold": frac_above_threshold,
            "float32_floor_warning_rule": "WARN only when at least half of samples exceed 1e-10 but median and p95 are <= 1e-5",
        },
        "Measurement consistency recomputed in float64 on unclipped samples.",
        status="WARN" if float32_floor_flag else None,
    )

    sample_lpips = lpips_distance(
        samples_flat.reshape(-1, n_pixels),
        repeated_x,
        image_shape=image_shape,
        device=device,
        backend=perceptual_backend,
    )
    baseline_lpips = lpips_distance(
        baseline_flat,
        x_flat,
        image_shape=image_shape,
        device=device,
        backend=perceptual_backend,
    )
    sample_lpips_mean = float(np.mean(sample_lpips))
    baseline_lpips_mean = float(np.mean(baseline_lpips))
    gperc_values = {
        "mean_sample_lpips": sample_lpips_mean,
        "mean_baseline_lpips": baseline_lpips_mean,
        "baseline_minus_sample_lpips": baseline_lpips_mean - sample_lpips_mean,
        "perceptual_backend": perceptual_backend,
    }
    if compute_distributional:
        dist = fid_kid(samples_flat.reshape(-1, n_pixels), repeated_x, image_shape=image_shape, device=device)
        gperc_values.update({"fid_samples_vs_gt": dist.fid, "kid_mean_samples_vs_gt": dist.kid_mean, "kid_std_samples_vs_gt": dist.kid_std})
        if dist.warning:
            gperc_values["distributional_warning"] = dist.warning
    gates["G-PERC"] = GateResult(
        "G-PERC",
        sample_lpips_mean < baseline_lpips_mean,
        gperc_values,
        "Average sample perceptual distance is better than deterministic baseline; FID/KID are reported only.",
    )

    report = {
        "dump_path": str(dump_path),
        "A_path": str(A_path),
        "P0_path": str(P0_path),
        "n_images": n_img,
        "k_samples": k,
        "n_pixels": n_pixels,
        "image_shape": list(image_shape),
        "overall_passed": all(g.passed for g in gates.values()),
        "gates": {name: gate.to_json() for name, gate in gates.items()},
        "metrics": _jsonable(
            {
                "sample_psnr_db_mean": sample_psnr_avg,
                "sample_mean_psnr_db_mean": mean_psnr_avg,
                "baseline_psnr_db_mean": baseline_psnr_avg,
                "sample_mean_ssim_mean": float(np.mean(sample_mean_ssim)),
                "baseline_ssim_mean": float(np.mean(baseline_ssim)),
            }
        ),
    }
    return report


def pass_fail_table(report: dict[str, Any]) -> str:
    lines = ["Gate    Status  Key values"]
    for name, gate in report["gates"].items():
        status = gate["status"]
        values = gate["values"]
        compact = ", ".join(f"{k}={v:.4g}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in values.items() if k not in {"required_range_db"})
        lines.append(f"{name:<7} {status:<6} {compact}")
    lines.append(f"OVERALL {'PASS' if report['overall_passed'] else 'FAIL'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="Result dump (.npz/.pt)")
    parser.add_argument("--A", dest="A_path", help="Override measurement matrix artifact path")
    parser.add_argument("--P0", dest="P0_path", help="Override null-space projector artifact path")
    parser.add_argument("--perceptual-backend", choices=("lpips", "mse", "edge_mse"), default="lpips")
    parser.add_argument("--no-fid-kid", action="store_true", help="Skip FID/KID computation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", help="Write JSON report to this path")
    args = parser.parse_args(argv)
    report = check_results(
        args.dump,
        A_path=args.A_path,
        P0_path=args.P0_path,
        perceptual_backend=args.perceptual_backend,
        compute_distributional=not args.no_fid_kid,
        device=args.device,
    )
    print(pass_fail_table(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
