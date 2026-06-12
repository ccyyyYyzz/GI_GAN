"""Visualization helpers for ghost-imaging eval dumps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .checker import _pick, _rel_measurement_errors, _samples_to_flat, load_array_artifact, load_dump
from .metrics import flatten_images, infer_hw, to_nhw


def _load_core(dump_path: str | Path):
    path = Path(dump_path)
    data = load_dump(path)
    x = np.asarray(_pick(data, "x"), dtype=np.float64)
    samples = np.asarray(_pick(data, "samples"), dtype=np.float64)
    mean = np.asarray(_pick(data, "sample_mean"), dtype=np.float64)
    y = np.asarray(_pick(data, "y"), dtype=np.float64)
    samples_unclipped = np.asarray(_pick(data, "samples_unclipped", required=False, default=samples), dtype=np.float64)
    A_path = Path(str(_pick(data, "A_path")))
    if not A_path.is_absolute():
        A_path = path.parent / A_path
    return path, data, x, samples, mean, y, samples_unclipped, load_array_artifact(A_path)


def save_sample_grid(samples: np.ndarray, out_path: str | Path, image_index: int = 0, max_samples: int = 16) -> None:
    flat = _samples_to_flat(samples)
    k = min(max_samples, flat.shape[1])
    h, w = infer_hw(flat.shape[2])
    imgs = flat[image_index, :k].reshape(k, h, w)
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows), squeeze=False)
    for ax, img, idx in zip(axes.ravel(), imgs, range(k)):
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"k={idx}")
        ax.axis("off")
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_std_maps(samples: np.ndarray, out_path: str | Path, max_images: int = 8) -> None:
    flat = _samples_to_flat(samples)
    n = min(max_images, flat.shape[0])
    h, w = infer_hw(flat.shape[2])
    std = np.std(flat[:n], axis=1, ddof=0).reshape(n, h, w)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    vmax = float(np.max(std)) if std.size else 1.0
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.2 * rows), squeeze=False)
    last = None
    for i, ax in enumerate(axes.ravel()):
        if i < n:
            last = ax.imshow(std[i], cmap="magma", vmin=0.0, vmax=vmax)
            ax.set_title(f"image {i}")
        ax.axis("off")
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_std_error_scatter(x: np.ndarray, sample_mean: np.ndarray, samples: np.ndarray, out_path: str | Path) -> None:
    x_flat = flatten_images(x)
    mean_flat = flatten_images(sample_mean)
    std_flat = np.std(_samples_to_flat(samples), axis=1, ddof=0)
    err = np.abs(mean_flat - x_flat)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(std_flat.ravel(), err.ravel(), s=5, alpha=0.25)
    ax.set_xlabel("per-pixel sample std")
    ax.set_ylabel("|sample mean error|")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_certificate_table(samples_unclipped: np.ndarray, y: np.ndarray, A: np.ndarray, out_path: str | Path, lambdas=None) -> None:
    flat = _samples_to_flat(samples_unclipped)
    rel = _rel_measurement_errors(flat, y, A)
    with Path(out_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "k", "lambda", "rel_measurement_error"])
        writer.writeheader()
        for image_idx in range(rel.shape[0]):
            for k_idx in range(rel.shape[1]):
                lam = "" if lambdas is None else np.asarray(lambdas).reshape(rel.shape)[image_idx, k_idx]
                writer.writerow({"image": image_idx, "k": k_idx, "lambda": lam, "rel_measurement_error": rel[image_idx, k_idx]})


def save_all_visualizations(dump_path: str | Path, out_dir: str | Path, max_images: int = 8, max_samples: int = 16) -> list[Path]:
    path, data, x, samples, mean, y, samples_unclipped, A = _load_core(dump_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for image_index in range(min(max_images, _samples_to_flat(samples).shape[0])):
        file = out / f"{path.stem}_image{image_index}_samples.png"
        save_sample_grid(samples, file, image_index=image_index, max_samples=max_samples)
        written.append(file)
    file = out / f"{path.stem}_std_maps.png"
    save_std_maps(samples, file, max_images=max_images)
    written.append(file)
    file = out / f"{path.stem}_std_vs_abs_error.png"
    save_std_error_scatter(x, mean, samples, file)
    written.append(file)
    file = out / f"{path.stem}_certificate_table.csv"
    save_certificate_table(samples_unclipped, y, A, file, lambdas=data.get("lambda"))
    written.append(file)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=16)
    args = parser.parse_args(argv)
    for path in save_all_visualizations(args.dump, args.out_dir, args.max_images, args.max_samples):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
