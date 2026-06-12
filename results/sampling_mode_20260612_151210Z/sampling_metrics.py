from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np


def psnr(x: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((x.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim_global(x: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    x = x.astype(np.float64).reshape(-1)
    y = target.astype(np.float64).reshape(-1)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mux, muy = float(x.mean()), float(y.mean())
    vx, vy = float(x.var()), float(y.var())
    cxy = float(((x - mux) * (y - muy)).mean())
    return ((2 * mux * muy + c1) * (2 * cxy + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2))


def rel_meas_error(A: np.ndarray, x_flat: np.ndarray, y: np.ndarray) -> float:
    err = A @ x_flat - y
    return float(np.linalg.norm(err) / max(np.linalg.norm(y), 1e-12))


def rowspace_basis_from_A(A: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(A.T)
    return q[:, : A.shape[0]]


def null_project(flat: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return flat - (flat @ Q) @ Q.T


def null_variance_ratio(A: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    # samples: K x N, already flattened.
    mean = samples.mean(axis=0, keepdims=True)
    diffs = samples - mean
    denom = np.linalg.norm(diffs, axis=1) + 1e-12
    numer = np.linalg.norm(diffs @ A.T, axis=1)
    ratios = numer / denom
    return {"mean": float(ratios.mean()), "median": float(np.median(ratios)), "max": float(ratios.max())}


def kappa_proxy(samples: np.ndarray, gt: np.ndarray, mean_mode: np.ndarray, Q: np.ndarray) -> float:
    # Proxy assumption: the deterministic mean-mode null MSE is used as an MMSE proxy.
    sample_mean = samples.mean(axis=0)
    sample_null_mse = float(np.mean((null_project(sample_mean, Q) - null_project(gt, Q)) ** 2))
    mean_null_mse = float(np.mean((null_project(mean_mode, Q) - null_project(gt, Q)) ** 2))
    return sample_null_mse / max(mean_null_mse, 1e-12)


def optional_perceptual_availability() -> dict[str, Any]:
    packages = {
        "lpips": importlib.util.find_spec("lpips") is not None,
        "cleanfid": importlib.util.find_spec("cleanfid") is not None,
        "torchmetrics": importlib.util.find_spec("torchmetrics") is not None,
    }
    return {
        "packages": packages,
        "lpips_available": packages["lpips"],
        "fid_available": packages["cleanfid"] or packages["torchmetrics"],
        "kid_available": packages["torchmetrics"],
        "requirement_note": "LPIPS requires the lpips package and local backbone weights; FID/KID require clean-fid or torchmetrics with local feature weights/cache.",
    }


def summarize_saved_samples(samples: np.ndarray, gt: np.ndarray, mean_mode: np.ndarray, A: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    flat_samples = samples.reshape(samples.shape[0], -1)
    flat_gt = gt.reshape(-1)
    flat_mean = mean_mode.reshape(-1)
    Q = rowspace_basis_from_A(A)
    sample_metrics = []
    for sample in flat_samples:
        sample_metrics.append(
            {
                "psnr": psnr(sample, flat_gt),
                "ssim_global": ssim_global(sample, flat_gt),
                "rel_meas_error": rel_meas_error(A, sample, y),
            }
        )
    sample_mean = flat_samples.mean(axis=0)
    return {
        "per_sample": sample_metrics,
        "sample_mean": {
            "psnr": psnr(sample_mean, flat_gt),
            "ssim_global": ssim_global(sample_mean, flat_gt),
            "rel_meas_error": rel_meas_error(A, sample_mean, y),
        },
        "pixel_std_mean": float(flat_samples.std(axis=0).mean()),
        "null_variance_ratio": null_variance_ratio(A, flat_samples),
        "kappa_proxy": kappa_proxy(flat_samples, flat_gt, flat_mean, Q),
        "prop12_certificate_invariance_requires_audit": True,
    }
