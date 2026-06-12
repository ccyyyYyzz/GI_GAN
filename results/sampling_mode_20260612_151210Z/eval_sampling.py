from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


def fixed_z_seeds(base_seed: int, k: int) -> list[int]:
    return [int(base_seed + 1009 * i) for i in range(k)]


def save_stochastic_samples_npz(
    output_path: str | Path,
    reconstruct_fn: Callable[[int], np.ndarray],
    *,
    k: int = 32,
    base_seed: int = 6002,
    metadata: dict | None = None,
) -> Path:
    """Save K stochastic outputs for one test item.

    `reconstruct_fn(seed)` must return one H x W or C x H x W sample for the same y.
    The function is deliberately explicit about seeds so a future G2 run can prove that
    stochastic z was active and all samples were persisted before aggregate metrics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = fixed_z_seeds(base_seed, k)
    samples = [np.asarray(reconstruct_fn(seed), dtype=np.float32) for seed in seeds]
    stack = np.stack(samples, axis=0)
    np.savez_compressed(output_path, samples=stack, z_seeds=np.array(seeds, dtype=np.int64), metadata=json.dumps(metadata or {}))
    return output_path


def save_batch_stochastic_samples(
    output_dir: str | Path,
    reconstruct_batch_fn: Callable[[int], np.ndarray],
    *,
    image_ids: list[int],
    k: int = 32,
    base_seed: int = 6002,
    metadata: dict | None = None,
) -> list[Path]:
    """Save K stochastic outputs per image for one batch.

    `reconstruct_batch_fn(seed)` must return B x C x H x W for a fixed y batch. Files are
    z-indexed through the stored seed list and image-indexed through the filename.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = fixed_z_seeds(base_seed, k)
    by_seed = [np.asarray(reconstruct_batch_fn(seed), dtype=np.float32) for seed in seeds]
    stack = np.stack(by_seed, axis=0)  # K x B x C x H x W
    paths: list[Path] = []
    for local_idx, image_id in enumerate(image_ids):
        path = output_dir / f"image_{int(image_id):08d}_K{k}.npz"
        np.savez_compressed(
            path,
            samples=stack[:, local_idx],
            z_seeds=np.array(seeds, dtype=np.int64),
            image_id=np.array([int(image_id)], dtype=np.int64),
            metadata=json.dumps(metadata or {}),
        )
        paths.append(path)
    return paths
