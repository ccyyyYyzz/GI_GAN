"""CPU-friendly image metrics for 64x64 grayscale ghost-imaging outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def as_numpy(value):
    """Convert numpy/torch-like arrays to numpy without importing torch eagerly."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def infer_hw(n_pixels: int) -> tuple[int, int]:
    side = int(round(np.sqrt(n_pixels)))
    if side * side != n_pixels:
        raise ValueError(f"Cannot infer square image shape from {n_pixels} pixels")
    return side, side


def flatten_images(images: np.ndarray) -> np.ndarray:
    arr = as_numpy(images)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def to_nhw(images: np.ndarray, image_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return images as (N, H, W), accepting flattened, NHW, or NCHW grayscale."""
    arr = as_numpy(images).astype(np.float64, copy=False)
    if arr.ndim == 1:
        if image_shape is None:
            image_shape = infer_hw(arr.size)
        return arr.reshape(1, *image_shape)
    if arr.ndim == 2:
        if image_shape is None:
            image_shape = infer_hw(arr.shape[1])
        return arr.reshape(arr.shape[0], *image_shape)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Expected flattened or grayscale images, got shape {arr.shape}")


def to_nchw3(images: np.ndarray, image_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return grayscale images replicated to (N, 3, H, W) and clipped to [0, 1]."""
    nhw = np.clip(to_nhw(images, image_shape), 0.0, 1.0)
    return np.repeat(nhw[:, None, :, :], 3, axis=1).astype(np.float32)


def mse(pred: np.ndarray, target: np.ndarray, image_shape: tuple[int, int] | None = None) -> np.ndarray:
    pred_f = flatten_images(to_nhw(pred, image_shape))
    target_f = flatten_images(to_nhw(target, image_shape))
    if pred_f.shape != target_f.shape:
        raise ValueError(f"MSE shape mismatch: {pred_f.shape} vs {target_f.shape}")
    return np.mean((pred_f - target_f) ** 2, axis=1)


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    vals = mse(pred, target)
    return 10.0 * np.log10((data_range**2) / np.maximum(vals, 1e-300))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    pred_nhw = to_nhw(pred)
    target_nhw = to_nhw(target, pred_nhw.shape[1:])
    try:
        from skimage.metrics import structural_similarity
    except Exception:  # pragma: no cover - exercised only in minimal envs
        # Global SSIM fallback for lightweight test environments. The pinned
        # eval requirements install scikit-image, whose windowed SSIM is used
        # automatically when available.
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        out = []
        for p, t in zip(pred_nhw, target_nhw):
            mux, muy = np.mean(p), np.mean(t)
            vx, vy = np.var(p), np.var(t)
            cxy = np.mean((p - mux) * (t - muy))
            out.append(((2 * mux * muy + c1) * (2 * cxy + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2)))
        return np.asarray(out, dtype=np.float64)

    out = []
    for p, t in zip(pred_nhw, target_nhw):
        win_size = min(7, p.shape[0], p.shape[1])
        if win_size % 2 == 0:
            win_size -= 1
        out.append(structural_similarity(t, p, data_range=data_range, win_size=max(win_size, 3)))
    return np.asarray(out, dtype=np.float64)


def _batched(iterable: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def lpips_distance(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    image_shape: tuple[int, int] | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    backend: str = "lpips",
) -> np.ndarray:
    """Compute LPIPS distances, or an MSE proxy when backend='mse' for tiny tests."""
    if backend == "mse":
        return mse(pred, target, image_shape=image_shape)
    if backend == "edge_mse":
        pred_nhw = to_nhw(pred, image_shape)
        target_nhw = to_nhw(target, image_shape)
        vals = []
        for p, t in zip(pred_nhw, target_nhw):
            vals.append(np.mean((_sobel_magnitude(p) - _sobel_magnitude(t)) ** 2))
        return np.asarray(vals, dtype=np.float64)
    try:
        import torch
        import lpips
    except Exception as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "LPIPS requires torch and lpips. Install eval/requirements-eval.txt "
            "or pass perceptual_backend='mse' for synthetic tests."
        ) from exc

    pred_arr = to_nchw3(pred, image_shape)
    target_arr = to_nchw3(target, image_shape)
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    vals = []
    with torch.no_grad():
        for p_b, t_b in zip(_batched(pred_arr, batch_size), _batched(target_arr, batch_size)):
            p = torch.from_numpy(p_b).to(device) * 2.0 - 1.0
            t = torch.from_numpy(t_b).to(device) * 2.0 - 1.0
            vals.append(loss_fn(p, t).reshape(-1).cpu().numpy())
    return np.concatenate(vals).astype(np.float64)


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


@dataclass(frozen=True)
class FidKidResult:
    fid: float | None
    kid_mean: float | None
    kid_std: float | None
    warning: str | None = None


def fid_kid(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    image_shape: tuple[int, int] | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> FidKidResult:
    """Compute FID/KID after grayscale-to-RGB replication.

    These metrics are distributional and have high variance on tiny eval sets;
    callers should report them without pass/fail thresholds.
    """
    try:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception as exc:  # pragma: no cover - depends on optional env
        return FidKidResult(None, None, None, f"FID/KID unavailable: {exc}")

    pred_arr = to_nchw3(pred, image_shape)
    target_arr = to_nchw3(target, image_shape)
    fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)
    kid = KernelInceptionDistance(feature=64, normalize=True, subsets=20, subset_size=min(50, len(pred_arr))).to(device)
    with torch.no_grad():
        for batch in _batched(target_arr, batch_size):
            fid.update(torch.from_numpy(batch).to(device), real=True)
            kid.update(torch.from_numpy(batch).to(device), real=True)
        for batch in _batched(pred_arr, batch_size):
            fid.update(torch.from_numpy(batch).to(device), real=False)
            kid.update(torch.from_numpy(batch).to(device), real=False)
        fid_val = float(fid.compute().cpu().item())
        kid_mean, kid_std = kid.compute()
    return FidKidResult(fid_val, float(kid_mean.cpu().item()), float(kid_std.cpu().item()))
