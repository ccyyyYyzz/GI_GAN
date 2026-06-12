"""Audit correction scatter: v -> v - B_lambda(Av - y)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .checker import load_array_artifact
from .metrics import flatten_images, psnr


def b_lambda(A: np.ndarray, lam: float) -> np.ndarray:
    """Return B_lambda = A^T (A A^T + lambda I)^-1."""
    A64 = np.asarray(A, dtype=np.float64)
    gram = A64 @ A64.T + float(lam) * np.eye(A64.shape[0], dtype=np.float64)
    return A64.T @ np.linalg.inv(gram)


def audit_correct(v: np.ndarray, y: np.ndarray, A: np.ndarray, lam: float) -> np.ndarray:
    v_flat = flatten_images(v).astype(np.float64)
    y_flat = np.asarray(y, dtype=np.float64).reshape(v_flat.shape[0], -1)
    B = b_lambda(A, lam)
    residual = v_flat @ np.asarray(A, dtype=np.float64).T - y_flat
    return v_flat - residual @ B.T


def rel_measurement_error(v: np.ndarray, y: np.ndarray, A: np.ndarray) -> np.ndarray:
    v_flat = flatten_images(v).astype(np.float64)
    y_flat = np.asarray(y, dtype=np.float64).reshape(v_flat.shape[0], -1)
    residual = v_flat @ np.asarray(A, dtype=np.float64).T - y_flat
    denom = np.maximum(np.linalg.norm(y_flat, axis=1), 1e-300)
    return np.linalg.norm(residual, axis=1) / denom


def save_audit_arrow_plot(v: np.ndarray, y: np.ndarray, A: np.ndarray, lam: float, out_path: str | Path, x_true=None) -> np.ndarray:
    before = flatten_images(v).astype(np.float64)
    after = audit_correct(before, y, A, lam)
    before_rel = rel_measurement_error(before, y, A)
    after_rel = rel_measurement_error(after, y, A)
    if x_true is None:
        before_psnr = np.full(before_rel.shape, np.nan)
        after_psnr = np.full(after_rel.shape, np.nan)
    else:
        x_flat = flatten_images(x_true).astype(np.float64)
        before_psnr = psnr(before, x_flat)
        after_psnr = psnr(after, x_flat)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(before_rel, before_psnr, label="before", s=22)
    ax.scatter(after_rel, after_psnr, label="after", s=22)
    for x0, y0, x1, y1 in zip(before_rel, before_psnr, after_rel, after_psnr):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 0.8, "alpha": 0.7})
    ax.set_xscale("log")
    ax.set_xlabel("RelMeasErr")
    ax.set_ylabel("PSNR (dB)" if x_true is not None else "PSNR unavailable")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return after


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v", required=True, help=".npy/.npz artifact with candidates")
    parser.add_argument("--y", required=True, help=".npy/.npz artifact with measurements")
    parser.add_argument("--A", required=True)
    parser.add_argument("--lambda", dest="lam", type=float, required=True)
    parser.add_argument("--x-true", help="Optional .npy/.npz artifact with ground truth for PSNR")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    v = load_array_artifact(args.v)
    y = load_array_artifact(args.y)
    A = load_array_artifact(args.A)
    x_true = load_array_artifact(args.x_true) if args.x_true else None
    save_audit_arrow_plot(v, y, A, args.lam, args.out, x_true=x_true)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
