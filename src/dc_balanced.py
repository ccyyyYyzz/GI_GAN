from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np


class DCBalancedError(RuntimeError):
    """Raised for invalid DC-balanced row construction or metric inputs."""


def normalize_row(row: np.ndarray, *, zero_mean: bool) -> np.ndarray:
    r = np.asarray(row, dtype=np.float64).reshape(-1)
    if zero_mean:
        r = r - float(np.mean(r))
    norm = float(np.linalg.norm(r))
    if norm <= 1e-12:
        raise DCBalancedError("ZERO_NORM_ROW_AFTER_NORMALIZATION")
    return (r / norm).astype(np.float32)


def dc_row(dim: int) -> np.ndarray:
    return np.full((int(dim),), 1.0 / math.sqrt(float(dim)), dtype=np.float32)


def random_zero_mean_rows(num_rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    rows = []
    for _ in range(int(num_rows)):
        raw = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=int(dim))
        rows.append(normalize_row(raw, zero_mean=True))
    return np.stack(rows, axis=0).astype(np.float32)


def dct2_basis_row(size: int, u: int, v: int) -> np.ndarray:
    yy = np.arange(int(size), dtype=np.float64)
    xx = np.arange(int(size), dtype=np.float64)
    au = math.sqrt(1.0 / size) if int(u) == 0 else math.sqrt(2.0 / size)
    av = math.sqrt(1.0 / size) if int(v) == 0 else math.sqrt(2.0 / size)
    by = au * np.cos(np.pi * (2.0 * yy + 1.0) * int(u) / (2.0 * int(size)))
    bx = av * np.cos(np.pi * (2.0 * xx + 1.0) * int(v) / (2.0 * int(size)))
    return np.outer(by, bx).reshape(-1)


def dct_lowfreq_non_dc_rows(num_rows: int, img_size: int) -> np.ndarray:
    size = int(img_size)
    coords = [(u, v) for u in range(size) for v in range(size) if not (u == 0 and v == 0)]
    coords.sort(key=lambda uv: (uv[0] * uv[0] + uv[1] * uv[1], uv[0] + uv[1], uv[0], uv[1]))
    rows = [normalize_row(dct2_basis_row(size, u, v), zero_mean=True) for u, v in coords[: int(num_rows)]]
    return np.stack(rows, axis=0).astype(np.float32)


@lru_cache(maxsize=2)
def _hadamard_matrix(dim: int) -> np.ndarray:
    from scipy.linalg import hadamard

    return hadamard(int(dim), dtype=np.int8)


def hadamard_lowsequency_non_dc_rows(num_rows: int, dim: int) -> np.ndarray:
    h = _hadamard_matrix(int(dim))
    changes = np.sum(h[:, 1:] != h[:, :-1], axis=1)
    order = np.argsort(changes, kind="stable")
    rows = []
    for idx in order:
        row = np.asarray(h[int(idx)], dtype=np.float64)
        if abs(float(np.mean(row))) > 1e-12:
            continue
        rows.append(normalize_row(row, zero_mean=True))
        if len(rows) >= int(num_rows):
            break
    if len(rows) < int(num_rows):
        raise DCBalancedError(f"NOT_ENOUGH_NON_DC_HADAMARD_ROWS:{len(rows)}:{num_rows}")
    return np.stack(rows, axis=0).astype(np.float32)


def build_dc_balanced_rows(kind: str, num_non_dc: int, *, dim: int, img_size: int, seed: int = 0) -> np.ndarray:
    kind_norm = str(kind).lower()
    if int(num_non_dc) < 0:
        raise DCBalancedError(f"NEGATIVE_ROW_COUNT:{num_non_dc}")
    if kind_norm in {"random", "rademacher", "balanced_random"}:
        non_dc = random_zero_mean_rows(int(num_non_dc), int(dim), int(seed))
    elif kind_norm in {"dct", "dct2", "lowfreq_dct", "non_dc_lowfreq_dct"}:
        non_dc = dct_lowfreq_non_dc_rows(int(num_non_dc), int(img_size))
    elif kind_norm in {"hadamard", "lowsequency_hadamard", "non_dc_lowsequency_hadamard"}:
        non_dc = hadamard_lowsequency_non_dc_rows(int(num_non_dc), int(dim))
    else:
        raise DCBalancedError(f"UNKNOWN_DC_BALANCED_ROW_KIND:{kind}")
    if int(num_non_dc) == 0:
        return dc_row(dim)[None, :]
    return np.concatenate([dc_row(dim)[None, :], non_dc], axis=0).astype(np.float32)


def row_audit(rows: np.ndarray, *, name: str, dc: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2:
        raise DCBalancedError(f"ROWS_MUST_BE_2D:{arr.shape}")
    dc_vec = dc_row(arr.shape[1]).astype(np.float64) if dc is None else np.asarray(dc, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    means = np.mean(arr, axis=1)
    dc_dot = arr @ dc_vec
    pos = np.sum(np.maximum(arr, 0.0), axis=1)
    neg = np.sum(np.maximum(-arr, 0.0), axis=1)
    gram = arr @ arr.T
    off = gram - np.diag(np.diag(gram))
    return {
        "name": str(name),
        "shape": list(arr.shape),
        "row_norm_min": float(norms.min()) if norms.size else None,
        "row_norm_max": float(norms.max()) if norms.size else None,
        "row_norm_mean": float(norms.mean()) if norms.size else None,
        "row_mean_max_abs": float(np.max(np.abs(means))) if means.size else None,
        "non_dc_row_mean_max_abs": float(np.max(np.abs(means[1:]))) if arr.shape[0] > 1 else 0.0,
        "dc_dot_first_row": float(dc_dot[0]) if dc_dot.size else None,
        "non_dc_dc_dot_max_abs": float(np.max(np.abs(dc_dot[1:]))) if arr.shape[0] > 1 else 0.0,
        "positive_exposure_mean": float(pos.mean()) if pos.size else None,
        "negative_exposure_mean": float(neg.mean()) if neg.size else None,
        "signed_exposure_imbalance_max_abs": float(np.max(np.abs(pos - neg))) if pos.size else None,
        "mean_abs_offdiag_gram": float(np.mean(np.abs(off))) if off.size else 0.0,
        "max_abs_offdiag_gram": float(np.max(np.abs(off))) if off.size else 0.0,
        "dc_balanced_pass": bool(
            arr.shape[0] >= 1
            and abs(float(dc_dot[0]) - 1.0) < 1e-5
            and (arr.shape[0] == 1 or float(np.max(np.abs(means[1:]))) < 1e-6)
            and (arr.shape[0] == 1 or float(np.max(np.abs(dc_dot[1:]))) < 1e-5)
            and float(np.max(np.abs(norms - 1.0))) < 1e-5
        ),
    }


def centered_rmse(x_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
    pred = np.asarray(x_hat, dtype=np.float64)
    truth = np.asarray(x, dtype=np.float64)
    err = pred - truth
    centered = err - np.mean(err, axis=1, keepdims=True)
    return np.sqrt(np.mean(centered * centered, axis=1))


def full_rmse(x_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
    err = np.asarray(x_hat, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    return np.sqrt(np.mean(err * err, axis=1))


def mean_abs_error(x_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.abs(np.mean(np.asarray(x_hat, dtype=np.float64) - np.asarray(x, dtype=np.float64), axis=1))


def dct_band_rmse(x_hat: np.ndarray, x: np.ndarray, *, img_size: int, low_count: int = 40, mid_count: int = 512) -> dict[str, np.ndarray]:
    from scipy.fft import dctn

    pred = np.asarray(x_hat, dtype=np.float64).reshape(-1, int(img_size), int(img_size))
    truth = np.asarray(x, dtype=np.float64).reshape(-1, int(img_size), int(img_size))
    diff = dctn(pred - truth, axes=(1, 2), norm="ortho").reshape(pred.shape[0], -1)
    coords = [(u, v) for u in range(int(img_size)) for v in range(int(img_size)) if not (u == 0 and v == 0)]
    coords.sort(key=lambda uv: (uv[0] * uv[0] + uv[1] * uv[1], uv[0] + uv[1], uv[0], uv[1]))
    order = np.asarray([u * int(img_size) + v for u, v in coords], dtype=np.int64)
    low = order[: int(low_count)]
    mid = order[int(low_count) : int(low_count) + int(mid_count)]
    high = order[int(low_count) + int(mid_count) :]

    def rmse(cols: np.ndarray) -> np.ndarray:
        if cols.size == 0:
            return np.zeros(pred.shape[0], dtype=np.float64)
        vals = diff[:, cols]
        return np.sqrt(np.mean(vals * vals, axis=1))

    return {
        "dct_non_dc_low_rmse": rmse(low),
        "dct_mid_rmse": rmse(mid),
        "dct_high_rmse": rmse(high),
    }


def wp0_diagnostics(witness_rows: np.ndarray, projector, *, lambda_: float) -> dict[str, Any]:
    import torch

    w = np.asarray(witness_rows, dtype=np.float64)
    if w.size == 0:
        return {"rank_wp0": 0, "cond_wp0wt": None, "shape": [0, int(projector.n)]}
    with torch.no_grad():
        wt = torch.from_numpy(w).to(device=projector.device, dtype=projector.dtype)
        z = projector.null_project_flat(wt).detach().cpu().numpy().astype(np.float64)
    s = w @ z.T
    s = 0.5 * (s + s.T)
    eig = np.linalg.eigvalsh(s)
    rank = int(np.sum(eig > max(1e-10, 1e-8 * max(float(np.max(eig)), 1.0))))
    cond = None
    if s.size:
        vals = eig[eig > 1e-12]
        cond = None if vals.size == 0 else float(np.max(vals) / np.min(vals))
    reg = s + float(lambda_) * np.eye(s.shape[0], dtype=np.float64)
    return {
        "shape": list(w.shape),
        "rank_wp0": rank,
        "cond_wp0wt": cond,
        "cond_wp0wt_plus_lambda": float(np.linalg.cond(reg)) if reg.size else None,
        "eig_min": float(np.min(eig)) if eig.size else None,
        "eig_max": float(np.max(eig)) if eig.size else None,
        "lambda": float(lambda_),
    }


def bias_variance_risk(eigenvalues: Sequence[float], *, lambda_: float, noise_variance: float) -> dict[str, float]:
    vals = np.asarray(list(eigenvalues), dtype=np.float64)
    lam = float(lambda_)
    shrink = lam / np.maximum(vals + lam, 1e-300)
    variance_gain = vals / np.maximum((vals + lam) ** 2, 1e-300)
    return {
        "lambda": lam,
        "noise_variance": float(noise_variance),
        "mean_squared_bias_shrink_factor": float(np.mean(shrink * shrink)) if vals.size else 0.0,
        "noise_variance_trace_gain": float(float(noise_variance) * np.sum(variance_gain)) if vals.size else 0.0,
        "max_noise_gain": float(np.max(variance_gain)) if vals.size else 0.0,
    }
