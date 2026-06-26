from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .projections import exact_data_anchor, exact_null_project, exact_row_project, get_exact_projector


@dataclass
class SplitComponents:
    name: str
    x: torch.Tensor
    r: torch.Tensor
    n: torch.Tensor
    y: torch.Tensor
    labels: torch.Tensor
    source_indices: torch.Tensor
    projector_info: dict[str, Any]

    def subset(self, count: int | None) -> "SplitComponents":
        if count is None or int(count) >= int(self.x.shape[0]):
            return self
        idx = slice(0, int(count))
        return SplitComponents(
            self.name,
            self.x[idx],
            self.r[idx],
            self.n[idx],
            self.y[idx],
            self.labels[idx],
            self.source_indices[idx],
            self.projector_info,
        )

    @property
    def size(self) -> int:
        return int(self.x.shape[0])

    @property
    def img_size(self) -> int:
        return int(self.x.shape[-1])


def tensor_sha256(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_measurements(x: torch.Tensor, measurement, *, device: torch.device, batch_size: int = 64) -> torch.Tensor:
    ys = []
    for start in range(0, int(x.shape[0]), int(batch_size)):
        xb = x[start : start + batch_size].to(device)
        flat = measurement.flatten_img(xb.float())
        ys.append(measurement.A_forward(flat).detach().cpu())
    return torch.cat(ys, 0)


@torch.no_grad()
def decompose_split(
    split,
    measurement,
    *,
    device: torch.device,
    batch_size: int = 32,
    max_samples: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> SplitComponents:
    projector = get_exact_projector(measurement, dtype=dtype, device=device)
    x_cpu = split.x.float()
    labels = split.labels.long()
    indices = split.indices.long()
    if max_samples is not None:
        x_cpu = x_cpu[: int(max_samples)]
        labels = labels[: int(max_samples)]
        indices = indices[: int(max_samples)]
    rows, nulls, ys = [], [], []
    for start in range(0, int(x_cpu.shape[0]), int(batch_size)):
        xb = x_cpu[start : start + batch_size].to(device)
        flat = measurement.flatten_img(xb)
        yb = measurement.A_forward(flat)
        rb = exact_row_project(flat, measurement, dtype=dtype, device=device)
        nb = exact_null_project(flat, measurement, dtype=dtype, device=device)
        rows.append(rb.detach().cpu().float())
        nulls.append(nb.detach().cpu().float())
        ys.append(yb.detach().cpu().float())
    return SplitComponents(
        name=str(split.name),
        x=x_cpu.cpu().float(),
        r=torch.cat(rows, 0),
        n=torch.cat(nulls, 0),
        y=torch.cat(ys, 0),
        labels=labels.cpu(),
        source_indices=indices.cpu(),
        projector_info=projector.info_dict(),
    )


def load_rad5_96_components(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    device: torch.device,
) -> tuple[Any, dict[str, Any], dict[str, SplitComponents], dict[str, Any]]:
    """Load the canonical Phase78 Rad-5/96 split and exact decomposition."""
    from . import phase78_96px_rad5_one_seed_probe as p78

    out = Path(output_dir)
    p78.OUT = out / "_phase78_side_effects"
    p78.TRAIN_COUNT = int(config.get("train_count", p78.TRAIN_COUNT))
    p78.VAL_COUNT = int(config.get("val_count", p78.VAL_COUNT))
    p78.TEST_COUNT = int(config.get("test_count", p78.TEST_COUNT))
    p78.BATCH_SIZE = int(config.get("cache_batch_size", p78.BATCH_SIZE))
    rad_config = p78.make_config(device)
    rad_config["num_workers"] = int(config.get("num_workers", 0))
    measurement = p78.make_measurement(rad_config, device)
    train, val, test, split_manifest = p78.build_caches(measurement, device)
    batch_size = int(config.get("project_batch_size", 32))
    splits = {
        "train": decompose_split(train, measurement, device=device, batch_size=batch_size, max_samples=config.get("max_train_samples")),
        "val": decompose_split(val, measurement, device=device, batch_size=batch_size, max_samples=config.get("max_val_samples")),
        "test": decompose_split(test, measurement, device=device, batch_size=batch_size, max_samples=config.get("max_test_samples")),
    }
    return measurement, rad_config, splits, split_manifest


def make_derangement(n: int, seed: int) -> torch.Tensor:
    if n < 2:
        raise ValueError("Need at least two samples for mismatched negatives.")
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen)
    for _ in range(32):
        fixed = perm == torch.arange(n)
        if not bool(fixed.any()):
            return perm
        perm[fixed] = perm[fixed.roll(1)]
    return torch.roll(torch.arange(n), shifts=1)


def null_energy(nulls: torch.Tensor) -> torch.Tensor:
    flat = nulls.reshape(nulls.shape[0], -1)
    return torch.linalg.norm(flat, dim=1)


def make_semihard_donors(split: SplitComponents, *, seed: int, pool_size: int = 96) -> torch.Tensor:
    n = split.size
    if n < 2:
        raise ValueError("Need at least two samples for semi-hard donors.")
    gen = torch.Generator().manual_seed(int(seed))
    r_flat = split.r.reshape(n, -1)
    e = null_energy(split.n)
    donors = []
    for i in range(n):
        candidates = torch.randint(0, n - 1, (min(pool_size, n - 1),), generator=gen)
        candidates = candidates + (candidates >= i).long()
        rd = torch.linalg.norm(r_flat[candidates] - r_flat[i], dim=1)
        ed = (e[candidates] - e[i]).abs() / e[i].clamp_min(1e-8)
        score = rd / rd.median().clamp_min(1e-8) + ed
        donors.append(int(candidates[int(torch.argmin(score).item())].item()))
    return torch.tensor(donors, dtype=torch.long)


def energy_matched_donors(split: SplitComponents, *, seed: int, pool_size: int = 128) -> torch.Tensor:
    n = split.size
    gen = torch.Generator().manual_seed(int(seed))
    e = null_energy(split.n)
    donors = []
    for i in range(n):
        candidates = torch.randint(0, n - 1, (min(pool_size, n - 1),), generator=gen)
        candidates = candidates + (candidates >= i).long()
        score = (e[candidates] - e[i]).abs()
        donors.append(int(candidates[int(torch.argmin(score).item())].item()))
    return torch.tensor(donors, dtype=torch.long)


def verify_feasible_pairs(
    split: SplitComponents,
    measurement,
    donors: torch.Tensor,
    *,
    device: torch.device,
    max_pairs: int | None = None,
    batch_size: int = 64,
) -> dict[str, float | int | bool]:
    count = min(split.size, int(max_pairs) if max_pairs is not None else split.size)
    rel_u, rel_n = [], []
    for start in range(0, count, int(batch_size)):
        sl = slice(start, min(count, start + int(batch_size)))
        r = split.r[sl].to(device)
        n = split.n[donors[sl]].to(device)
        y = split.y[sl].to(device)
        u = r + n
        Au = measurement.A_forward(u)
        An = measurement.A_forward(n)
        rel_u.append((torch.linalg.norm(Au - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)).detach().cpu())
        rel_n.append((torch.linalg.norm(An, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)).detach().cpu())
    rel_u_t = torch.cat(rel_u)
    rel_n_t = torch.cat(rel_n)
    return {
        "pairs_checked": int(count),
        "u_rel_mean": float(rel_u_t.mean().item()),
        "u_rel_max": float(rel_u_t.max().item()),
        "donor_null_rel_mean": float(rel_n_t.mean().item()),
        "donor_null_rel_max": float(rel_n_t.max().item()),
        "pass_float32_proxy": bool(rel_u_t.max().item() < 1e-5 and rel_n_t.max().item() < 1e-5),
    }


def compute_train_normalization(train: SplitComponents) -> dict[str, float]:
    return {
        "r_mean": float(train.r.mean().item()),
        "r_std": float(train.r.std(unbiased=False).clamp_min(1e-6).item()),
        "n_mean": float(train.n.mean().item()),
        "n_std": float(train.n.std(unbiased=False).clamp_min(1e-6).item()),
    }


class RangeNullPairDataset(Dataset):
    def __init__(self, split: SplitComponents, *, normalization: dict[str, float] | None = None) -> None:
        self.split = split
        self.norm = normalization or compute_train_normalization(split)
        self.img_size = split.img_size

    def __len__(self) -> int:
        return self.split.size

    def _img(self, flat: torch.Tensor, key: str) -> torch.Tensor:
        img = flat.reshape(1, self.img_size, self.img_size).float()
        mean = float(self.norm[f"{key}_mean"])
        std = float(self.norm[f"{key}_std"])
        return (img - mean) / max(std, 1e-6)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "r": self._img(self.split.r[idx], "r"),
            "n": self._img(self.split.n[idx], "n"),
            "label": self.split.labels[idx],
            "source_index": self.split.source_indices[idx],
            "local_idx": torch.tensor(int(idx), dtype=torch.long),
        }


def normalize_images(flat: torch.Tensor, *, img_size: int, key: str, normalization: dict[str, float]) -> torch.Tensor:
    img = flat.reshape(flat.shape[0], 1, img_size, img_size).float()
    return (img - float(normalization[f"{key}_mean"])) / max(float(normalization[f"{key}_std"]), 1e-6)


def split_manifest(splits: dict[str, SplitComponents], measurement, split_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "a_sha256": tensor_sha256(measurement.A),
        "m": int(measurement.m),
        "n": int(measurement.n),
        "img_size": int(measurement.img_size),
        "pattern_type": str(measurement.pattern_type),
        "matrix_normalization": str(measurement.matrix_normalization),
        "split_info": split_info,
        "splits": {
            name: {
                "count": comp.size,
                "x_sha256": tensor_sha256(comp.x),
                "r_sha256": tensor_sha256(comp.r),
                "n_sha256": tensor_sha256(comp.n),
                "y_clean_sha256": tensor_sha256(comp.y),
                "indices_sha256": tensor_sha256(comp.source_indices),
            }
            for name, comp in splits.items()
        },
        "projector": splits["train"].projector_info,
    }


def save_split_cache(output_dir: str | Path, splits: dict[str, SplitComponents]) -> dict[str, str]:
    root = Path(output_dir) / "counterfactual_cache"
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, split in splits.items():
        path = root / f"{name}_components.pt"
        torch.save(asdict(split), path)
        paths[name] = str(path)
    return paths


def exact_data_anchor_from_y(y: torch.Tensor, measurement, *, device: torch.device, img: bool = True) -> torch.Tensor:
    return exact_data_anchor(y.to(device), measurement, device=device, as_image=img)
