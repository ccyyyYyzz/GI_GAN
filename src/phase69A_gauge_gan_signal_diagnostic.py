from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets

from .datasets import build_transform
from .eval import make_measurement
from .exact_measurement import tensor_sha256
from .models import build_generator
from .split_guard import assert_train_loader_disjoint_from_test, collect_sample_identities
from .utils import apply_experiment_defaults, load_config, set_seed


REQUESTED_PROJECT = Path(
    r"C:\Users\CYZ的computer\Documents\Codex\2026-06-04\files-mentioned-by-the-user-txt\ns_mc_gan_gi"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("E:/ns_mc_gan_gi")
OUT_DIR = DATA_ROOT / "outputs_phase69A_gauge_gan_signal_diagnostic"
CERT_ROOT = DATA_ROOT / "results" / "cert_package_20260612"
CERT_CACHE = CERT_ROOT / "cache"
CHECKPOINT = DATA_ROOT / "outputs_phase15" / "imported_noleak" / "scrambled_hadamard5_hq_noise001_colab" / "last.pt"
RESOLVED_CONFIG = DATA_ROOT / "outputs_phase15" / "imported_noleak" / "scrambled_hadamard5_hq_noise001_colab" / "resolved_config.yaml"
A_SCR5 = CERT_CACHE / "A_scr5.npy"
EVAL_CACHE = CERT_CACHE / "main_scr5.npz"
EVAL_MANIFEST = CERT_CACHE / "main_scr5_manifest.json"
PROVENANCE_JSON = CERT_ROOT / "PROVENANCE.json"
SPLIT_TRAIN = CERT_CACHE / "split_train_indices_stl10_train_unlabeled.npy"
SPLIT_EVAL = CERT_CACHE / "split_eval_indices_stl10_test.npy"


CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def append_log(out_dir: Path, message: str) -> None:
    ensure_dir(out_dir)
    with (out_dir / "RUNLOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {now()} {message}\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_np(a: np.ndarray, *, sort_int64: bool = False) -> str:
    arr = np.asarray(a)
    if sort_int64:
        arr = np.sort(arr.astype(np.int64))
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def row_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = "| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |"
    sep = "| " + " | ".join("-" * widths[c] for c in columns) + " |"
    body = ["| " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def unsafe_stop(out_dir: Path, failures: list[str], warnings: list[str] | None = None) -> None:
    warnings = warnings or []
    lines = [
        "# UNSAFE TO RUN",
        "",
        "Phase69A stopped before critic training.",
        "",
        "## Critical Failures",
        "",
    ]
    lines.extend(f"- {item}" for item in failures)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in warnings)
    lines.extend(["", "No generator, reconstruction network, or critic training was run."])
    write_text(out_dir / "UNSAFE_TO_RUN.md", "\n".join(lines) + "\n")
    append_log(out_dir, "unsafe_stop")


def load_provenance() -> dict[str, Any]:
    return read_json(PROVENANCE_JSON)


def make_config(device: str) -> dict[str, Any]:
    config = load_config(RESOLVED_CONFIG)
    config = apply_experiment_defaults(config)
    config["dataset_root"] = str(DATA_ROOT / "data")
    config["output_dir"] = str(OUT_DIR)
    config["device"] = device
    config["num_workers"] = 0
    config["batch_size"] = 32
    config["use_augmentation"] = False
    config["use_final_dc_project"] = True
    config = apply_experiment_defaults(config)
    return config


def load_checkpoint_and_generator(config: dict[str, Any], measurement, device: torch.device):
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    merged = dict(config)
    if isinstance(checkpoint, dict) and checkpoint.get("config"):
        merged.update(checkpoint["config"])
    merged["dataset_root"] = str(DATA_ROOT / "data")
    merged["output_dir"] = str(OUT_DIR)
    merged["device"] = str(device)
    merged["num_workers"] = 0
    merged["batch_size"] = 32
    merged["use_augmentation"] = False
    merged = apply_experiment_defaults(merged)
    generator = build_generator(merged, measurement=measurement).to(device)
    if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
        raise RuntimeError("Checkpoint has no generator state.")
    state = checkpoint.get("generator_ema") or checkpoint["generator"]
    generator.load_state_dict(state)
    generator.eval()
    for param in generator.parameters():
        param.requires_grad_(False)
    return checkpoint, generator, merged


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = [int(i) for i in indices.tolist()]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[int(idx)]
        x, y = self.base[base_idx]
        return x, int(y), base_idx


def stl10_dataset(split: str, *, train_transform: bool = False) -> Dataset:
    transform = build_transform(64, dataset_name="stl10", train=False, use_augmentation=False)
    del train_transform
    return datasets.STL10(root=str(DATA_ROOT / "data"), split=split, transform=transform, download=False)


def make_source_loaders(
    train_count: int,
    val_count: int,
    test_count: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    train_indices_full = np.load(SPLIT_TRAIN).astype(np.int64)
    eval_indices_full = np.load(SPLIT_EVAL).astype(np.int64)
    if train_count + val_count > len(train_indices_full):
        raise ValueError("Requested train+val count exceeds saved train split.")
    if test_count > len(eval_indices_full):
        raise ValueError("Requested test count exceeds saved eval split.")
    train_indices = train_indices_full[:train_count]
    val_indices = train_indices_full[train_count : train_count + val_count]
    test_indices = eval_indices_full[:test_count]

    base_train = stl10_dataset("train+unlabeled")
    train_ds = IndexedDataset(base_train, np.concatenate([train_indices, val_indices]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    guard_subset = Subset(base_train, train_indices.tolist())
    guard_loader = DataLoader(guard_subset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    guard = assert_train_loader_disjoint_from_test(guard_loader, context="Phase69A critic train source loader")

    split_info = {
        "train_indices_full_sorted_sha256": sha256_np(train_indices_full, sort_int64=True),
        "train_indices_full_loader_order_sha256": sha256_np(train_indices_full),
        "eval_indices_full_sorted_sha256": sha256_np(eval_indices_full, sort_int64=True),
        "eval_indices_full_loader_order_sha256": sha256_np(eval_indices_full),
        "critic_train_indices_loader_order_sha256": sha256_np(train_indices),
        "critic_val_indices_loader_order_sha256": sha256_np(val_indices),
        "critic_test_indices_loader_order_sha256": sha256_np(test_indices),
        "critic_train_count": int(train_count),
        "critic_val_count": int(val_count),
        "critic_test_count": int(test_count),
        "train_split_guard": guard,
        "train_source_split": "STL10 train+unlabeled",
        "heldout_test_source_split": "STL10 official test via frozen cert cache",
    }
    return train_loader, None, split_info


@torch.no_grad()
def exact_projectors(A: torch.Tensor, lambda_dc: float):
    A64 = A.detach().to(torch.float64)
    G = A64 @ A64.T
    eye = torch.eye(A64.shape[0], device=A64.device, dtype=torch.float64)
    K = G + float(lambda_dc) * eye
    return A64, G, K


def data_solution_safe(measurement, y: torch.Tensor, mode: str | None) -> torch.Tensor:
    mode_s = str(mode or getattr(measurement, "backprojection_mode", "ridge_pinv")).lower()
    if mode_s == "hadamard_zero_filled" and getattr(measurement, "hadamard_metadata", None) is None:
        # For the published Scr operator, rows are orthonormal; zero-filled
        # Hadamard inversion is exactly A^T y. set_A_override clears the
        # metadata, so use the algebraic equivalent rather than refusing.
        return measurement.AT_forward(y)
    return measurement.data_solution(y, mode=mode)


def blambda_y(y: torch.Tensor, A64: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    y64 = y.to(torch.float64)
    sol = torch.linalg.solve(K, y64.T).T
    return sol @ A64


def adag_y(y: torch.Tensor, A64: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    y64 = y.to(torch.float64)
    sol = torch.linalg.solve(G, y64.T).T
    return sol @ A64


def p0_exact(v: torch.Tensor, A64: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    v64 = v.to(torch.float64)
    av = v64 @ A64.T
    sol = torch.linalg.solve(G, av.T).T
    return v64 - sol @ A64


@torch.no_grad()
def mean_mode_pre_final(generator, measurement, y: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    y = y.float()
    x_data_flat = data_solution_safe(measurement, y, config.get("backprojection_mode", "ridge_pinv"))
    x_data = measurement.unflatten_img(x_data_flat)
    zero_noise = torch.zeros_like(x_data)
    residual = generator(x_data, zero_noise, y=y)
    residual_flat = measurement.flatten_img(residual.float())
    residual_ns_flat = (
        measurement.null_project(residual_flat)
        if bool(config.get("use_null_project", True))
        else residual_flat
    )
    x_tilde_flat = x_data_flat + residual_ns_flat
    x_stage1_flat = (
        measurement.dc_project(x_tilde_flat, y)
        if bool(config.get("use_dc_project", True))
        else x_tilde_flat
    )
    x_stage1 = measurement.unflatten_img(x_stage1_flat)
    if hasattr(generator, "refine"):
        refine_residual = generator.refine(x_data, x_stage1)
        refine_flat = measurement.flatten_img(refine_residual.float())
        pre_final_flat = x_stage1_flat + refine_flat
    else:
        pre_final_flat = x_stage1_flat
    return pre_final_flat, x_data_flat


@dataclass
class SplitArrays:
    split: str
    source_indices: np.ndarray
    labels: np.ndarray
    y: np.ndarray
    x: np.ndarray
    x_data: np.ndarray
    v_mean: np.ndarray
    real: np.ndarray
    fake: np.ndarray
    real_adag: np.ndarray
    fake_adag: np.ndarray
    residual_real_raw: np.ndarray
    residual_fake_raw: np.ndarray
    rel_real_raw: np.ndarray
    rel_fake_raw: np.ndarray


def canonical_stats(split: str, arr: SplitArrays, A: np.ndarray, lambda_dc: float) -> list[dict[str, Any]]:
    del lambda_dc
    rows: list[dict[str, Any]] = []
    A64 = A.astype(np.float64)
    y64 = arr.y.astype(np.float64)
    ABlambda = arr.real.astype(np.float64) @ A64.T
    # real and fake share the same B-lambda anchor after exact P0, so compare
    # each residual against the real canonical measurement anchor.
    anchor = ABlambda
    for kind, data in [("real", arr.real), ("fake", arr.fake), ("real_adagger", arr.real_adag), ("fake_adagger", arr.fake_adag)]:
        d64 = data.astype(np.float64)
        meas = d64 @ A64.T
        residual = meas - anchor
        rel = np.linalg.norm(residual, axis=1) / np.maximum(np.linalg.norm(anchor, axis=1), 1e-12)
        p0_energy = np.linalg.norm(d64 - arr.x_data.astype(np.float64), axis=1)
        rows.append(
            {
                "split": split,
                "kind": kind,
                "n": int(data.shape[0]),
                "min": float(np.min(d64)),
                "max": float(np.max(d64)),
                "mean": float(np.mean(d64)),
                "std": float(np.std(d64)),
                "measurement_residual_rel_median_vs_ABlambda": float(np.median(rel)),
                "measurement_residual_rel_max_vs_ABlambda": float(np.max(rel)),
                "p0_energy_median": float(np.median(p0_energy)),
                "p0_energy_mean": float(np.mean(p0_energy)),
            }
        )
    return rows


def build_train_val_arrays(
    loader: DataLoader,
    train_count: int,
    val_count: int,
    generator,
    measurement,
    config: dict[str, Any],
    A64: torch.Tensor,
    G: torch.Tensor,
    K: torch.Tensor,
    device: torch.device,
    out_dir: Path,
) -> tuple[SplitArrays, SplitArrays]:
    append_log(out_dir, "dataset_train_val_generation_start")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    xdatas: list[np.ndarray] = []
    vmeans: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    real: list[np.ndarray] = []
    fake: list[np.ndarray] = []
    real_adag: list[np.ndarray] = []
    fake_adag: list[np.ndarray] = []
    rraw: list[np.ndarray] = []
    fraw: list[np.ndarray] = []
    relr: list[np.ndarray] = []
    relf: list[np.ndarray] = []
    torch.manual_seed(6901)
    for x_cpu, y_label, idx in loader:
        x = x_cpu.to(device)
        y = measurement.measure(x)
        v_mean, x_data_flat = mean_mode_pre_final(generator, measurement, y, config)
        x_flat = measurement.flatten_img(x).to(torch.float64)
        v_flat = v_mean.to(torch.float64)
        b = blambda_y(y, A64, K)
        ad = adag_y(y, A64, G)
        p0x = p0_exact(x_flat, A64, G)
        p0v = p0_exact(v_flat, A64, G)
        real_b = p0x + b
        fake_b = p0v + b
        real_a = p0x + ad
        fake_a = p0v + ad
        raw_res_r = (x_flat @ A64.T) - y.to(torch.float64)
        raw_res_f = (v_flat @ A64.T) - y.to(torch.float64)
        denom = torch.linalg.norm(y.to(torch.float64), dim=1).clamp_min(1e-12)
        xs.append(x_flat.detach().cpu().numpy().astype(np.float32))
        ys.append(y.detach().cpu().numpy().astype(np.float32))
        xdatas.append(x_data_flat.detach().cpu().numpy().astype(np.float32))
        vmeans.append(v_flat.detach().cpu().numpy().astype(np.float32))
        labels.append(y_label.numpy().astype(np.int64))
        indices.append(idx.numpy().astype(np.int64))
        real.append(real_b.detach().cpu().numpy().astype(np.float32))
        fake.append(fake_b.detach().cpu().numpy().astype(np.float32))
        real_adag.append(real_a.detach().cpu().numpy().astype(np.float32))
        fake_adag.append(fake_a.detach().cpu().numpy().astype(np.float32))
        rraw.append(raw_res_r.detach().cpu().numpy().astype(np.float32))
        fraw.append(raw_res_f.detach().cpu().numpy().astype(np.float32))
        relr.append((torch.linalg.norm(raw_res_r, dim=1) / denom).detach().cpu().numpy().astype(np.float32))
        relf.append((torch.linalg.norm(raw_res_f, dim=1) / denom).detach().cpu().numpy().astype(np.float32))

    def cat(parts):
        return np.concatenate(parts, axis=0)

    all_arr = {
        "source_indices": cat(indices),
        "labels": cat(labels),
        "y": cat(ys),
        "x": cat(xs),
        "x_data": cat(xdatas),
        "v_mean": cat(vmeans),
        "real": cat(real),
        "fake": cat(fake),
        "real_adag": cat(real_adag),
        "fake_adag": cat(fake_adag),
        "residual_real_raw": cat(rraw),
        "residual_fake_raw": cat(fraw),
        "rel_real_raw": cat(relr),
        "rel_fake_raw": cat(relf),
    }

    def split_slice(name: str, sl: slice) -> SplitArrays:
        return SplitArrays(split=name, **{k: v[sl] for k, v in all_arr.items()})

    train = split_slice("train", slice(0, train_count))
    val = split_slice("val", slice(train_count, train_count + val_count))
    append_log(out_dir, "dataset_train_val_generation_complete")
    return train, val


def build_test_arrays(
    test_count: int,
    generator,
    measurement,
    config: dict[str, Any],
    A64: torch.Tensor,
    G: torch.Tensor,
    K: torch.Tensor,
    device: torch.device,
    out_dir: Path,
) -> SplitArrays:
    append_log(out_dir, "dataset_test_generation_start")
    z = np.load(EVAL_CACHE, allow_pickle=False)
    x = torch.from_numpy(z["x"][:test_count]).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(z["y"][:test_count]).to(device=device, dtype=torch.float32)
    labels = z["labels"][:test_count].astype(np.int64)
    eval_indices = np.load(SPLIT_EVAL).astype(np.int64)[:test_count]
    v_mean, x_data_flat = mean_mode_pre_final(generator, measurement, y, config)
    x_flat = x.to(torch.float64)
    v_flat = v_mean.to(torch.float64)
    b = blambda_y(y, A64, K)
    ad = adag_y(y, A64, G)
    p0x = p0_exact(x_flat, A64, G)
    p0v = p0_exact(v_flat, A64, G)
    raw_res_r = (x_flat @ A64.T) - y.to(torch.float64)
    raw_res_f = (v_flat @ A64.T) - y.to(torch.float64)
    denom = torch.linalg.norm(y.to(torch.float64), dim=1).clamp_min(1e-12)
    arr = SplitArrays(
        split="test",
        source_indices=eval_indices,
        labels=labels,
        y=y.detach().cpu().numpy().astype(np.float32),
        x=x.detach().cpu().numpy().astype(np.float32),
        x_data=x_data_flat.detach().cpu().numpy().astype(np.float32),
        v_mean=v_flat.detach().cpu().numpy().astype(np.float32),
        real=(p0x + b).detach().cpu().numpy().astype(np.float32),
        fake=(p0v + b).detach().cpu().numpy().astype(np.float32),
        real_adag=(p0x + ad).detach().cpu().numpy().astype(np.float32),
        fake_adag=(p0v + ad).detach().cpu().numpy().astype(np.float32),
        residual_real_raw=raw_res_r.detach().cpu().numpy().astype(np.float32),
        residual_fake_raw=raw_res_f.detach().cpu().numpy().astype(np.float32),
        rel_real_raw=(torch.linalg.norm(raw_res_r, dim=1) / denom).detach().cpu().numpy().astype(np.float32),
        rel_fake_raw=(torch.linalg.norm(raw_res_f, dim=1) / denom).detach().cpu().numpy().astype(np.float32),
    )
    append_log(out_dir, "dataset_test_generation_complete")
    return arr


def pair_arrays(arr: SplitArrays, *, gauge: str = "blambda", include_cond: bool = False, shuffle_cond: bool = False, seed: int = 0):
    if gauge == "blambda":
        real, fake = arr.real, arr.fake
    elif gauge == "adagger":
        real, fake = arr.real_adag, arr.fake_adag
    elif gauge == "raw":
        real, fake = arr.x, arr.v_mean
    else:
        raise ValueError(gauge)
    X = np.concatenate([real, fake], axis=0).astype(np.float32).reshape(-1, 1, 64, 64)
    y = np.concatenate([np.ones(real.shape[0]), np.zeros(fake.shape[0])], axis=0).astype(np.float32)
    src = np.concatenate([arr.source_indices, arr.source_indices], axis=0).astype(np.int64)
    labels = np.concatenate([arr.labels, arr.labels], axis=0).astype(np.int64)
    cond = None
    if include_cond:
        cond_flat = np.concatenate([arr.x_data, arr.x_data], axis=0).astype(np.float32)
        if shuffle_cond:
            rng = np.random.default_rng(seed)
            cond_flat = cond_flat[rng.permutation(cond_flat.shape[0])]
        cond = cond_flat.reshape(-1, 1, 64, 64)
    return X, y, cond, src, labels


def subset_pair_tuple(pair, n_source: int, n_keep: int):
    X, y, cond, src, labels = pair
    n_keep = min(int(n_keep), int(n_source))
    idx = np.concatenate([np.arange(n_keep), np.arange(n_source, n_source + n_keep)])
    cond_sub = None if cond is None else cond[idx]
    return X[idx], y[idx], cond_sub, src[idx], labels[idx]


class PatchCritic(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        def conv(cin, cout, stride):
            return nn.utils.spectral_norm(nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1))

        self.net = nn.Sequential(
            conv(in_channels, 16, 2),
            nn.LeakyReLU(0.2, inplace=True),
            conv(16, 32, 2),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=True),
            conv(32, 64, 2),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 1, kernel_size=3, padding=1)),
        )

    def forward(self, x):
        return self.net(x).mean(dim=(1, 2, 3))


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 12, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(48, 1)

    def forward(self, x):
        feat = self.net(x).flatten(1)
        return self.head(feat).squeeze(1)


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, *, n_boot: int = 300, seed: int = 123) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -60, 60)))
    pred = (probs >= 0.5).astype(int)
    acc = float(np.mean(pred == y_true))
    tpr = float(np.mean(pred[y_true == 1] == 1)) if np.any(y_true == 1) else float("nan")
    tnr = float(np.mean(pred[y_true == 0] == 0)) if np.any(y_true == 0) else float("nan")
    auc = auc_score(y_true, scores)
    brier = float(np.mean((probs - y_true) ** 2))
    rng = np.random.default_rng(seed)
    boots = []
    idx_all = np.arange(y_true.shape[0])
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=idx_all.shape[0], replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(auc_score(y_true[idx], scores[idx]))
    ci_lo, ci_hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"), float("nan"))
    return {
        "auc": auc,
        "auc_ci_low": ci_lo,
        "auc_ci_high": ci_hi,
        "accuracy": acc,
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "brier": brier,
    }


def make_tensor_loader(X: np.ndarray, y: np.ndarray, cond: np.ndarray | None, batch_size: int, shuffle: bool, seed: int):
    x_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    if cond is None:
        ds = TensorDataset(x_t, y_t)
    else:
        ds = TensorDataset(x_t, torch.from_numpy(cond), y_t)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, num_workers=0, drop_last=False)


@torch.no_grad()
def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device, conditional: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        if conditional:
            x, cond, y = batch
            inp = torch.cat([x.to(device), cond.to(device)], dim=1)
        else:
            x, y = batch
            inp = x.to(device)
        logit = model(inp).detach().cpu().numpy()
        scores.append(logit)
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def train_image_model(
    name: str,
    model: nn.Module,
    train_data,
    val_data,
    test_data,
    *,
    device: torch.device,
    out_dir: Path,
    conditional: bool = False,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 2e-4,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    append_log(out_dir, f"critic_train_start model={name}")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9))
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = make_tensor_loader(*train_data, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = make_tensor_loader(*val_data, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = make_tensor_loader(*test_data, batch_size=batch_size, shuffle=False, seed=seed)
    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_auc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            if conditional:
                x, cond, y = batch
                inp = torch.cat([x.to(device), cond.to(device)], dim=1)
            else:
                x, y = batch
                inp = x.to(device)
            target = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(inp)
            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        y_val, s_val = predict_scores(model, val_loader, device, conditional)
        val_metrics = metrics_from_scores(y_val, s_val, n_boot=80, seed=seed + epoch)
        row = {
            "model": name,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_brier": val_metrics["brier"],
        }
        history.append(row)
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
        append_log(out_dir, f"critic_epoch model={name} epoch={epoch} val_auc={val_metrics['auc']:.4f}")
    model.load_state_dict(best_state)
    y_test, s_test = predict_scores(model, test_loader, device, conditional)
    m = metrics_from_scores(y_test, s_test, n_boot=300, seed=seed + 999)
    result = {
        "model": name,
        "split": "test",
        "epochs": epochs,
        "best_val_auc": best_auc,
        **m,
    }
    append_log(out_dir, f"critic_train_complete model={name} test_auc={m['auc']:.4f}")
    return result, history, y_test, s_test


def train_residual_logistic(
    train: SplitArrays,
    val: SplitArrays,
    test: SplitArrays,
    *,
    device: torch.device,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    append_log(out_dir, "residual_control_train_start")

    def feats(arr: SplitArrays):
        xr = np.concatenate([arr.residual_real_raw, arr.rel_real_raw[:, None]], axis=1)
        xf = np.concatenate([arr.residual_fake_raw, arr.rel_fake_raw[:, None]], axis=1)
        X = np.concatenate([xr, xf], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(xr.shape[0]), np.zeros(xf.shape[0])], axis=0).astype(np.float32)
        return X, y

    Xtr, ytr = feats(train)
    Xv, yv = feats(val)
    Xte, yte = feats(test)
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xv = (Xv - mu) / sd
    Xte = (Xte - mu) / sd
    model = nn.Sequential(nn.Linear(Xtr.shape[1], 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    history: list[dict[str, Any]] = []
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    loader = DataLoader(ds, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(777))
    best_state = copy.deepcopy(model.state_dict())
    best_auc = -1.0
    for epoch in range(1, 81):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logit = model(xb.to(device)).squeeze(1)
            loss = loss_fn(logit, yb.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                sv = model(torch.from_numpy(Xv).to(device)).squeeze(1).detach().cpu().numpy()
            mv = metrics_from_scores(yv, sv, n_boot=50, seed=epoch)
            history.append({"model": "residual_features_logistic_raw", "epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": mv["auc"]})
            if mv["auc"] > best_auc:
                best_auc = mv["auc"]
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    with torch.no_grad():
        ste = model(torch.from_numpy(Xte).to(device)).squeeze(1).detach().cpu().numpy()
    m = metrics_from_scores(yte, ste, n_boot=300, seed=1234)
    append_log(out_dir, f"residual_control_train_complete test_auc={m['auc']:.4f}")
    return {"model": "residual_features_logistic_raw", "split": "test", "best_val_auc": best_auc, **m}, history, yte, ste


def per_class_auc(labels: np.ndarray, y_true: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for class_id in sorted(set(int(x) for x in labels.tolist() if int(x) >= 0)):
        mask = labels == class_id
        if mask.sum() < 4:
            continue
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id),
                "n_pairs": int(mask.sum()),
                "auc": auc_score(y_true[mask], scores[mask]),
            }
        )
    return rows


def save_example_grid(out_dir: Path, test: SplitArrays) -> None:
    ensure_dir(out_dir)
    n = min(8, test.real.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
    for i in range(n):
        imgs = [test.x[i], test.x_data[i], test.real[i], test.fake[i]]
        titles = ["x real", "x_data", "canon real", "canon fake"]
        for j, (img, title) in enumerate(zip(imgs, titles)):
            ax = axes[i, j] if n > 1 else axes[j]
            ax.imshow(img.reshape(64, 64), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "gauge_dataset_example_grid.png", dpi=160)
    fig.savefig(out_dir / "gauge_dataset_example_grid.pdf")
    plt.close(fig)


def save_score_histograms(out_dir: Path, score_sets: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    n = len(score_sets)
    fig, axes = plt.subplots(n, 1, figsize=(7, max(2.4, 2.2 * n)), squeeze=False)
    for ax, (name, (y, s)) in zip(axes[:, 0], score_sets.items()):
        ax.hist(s[y == 1], bins=30, alpha=0.6, label="real", density=True)
        ax.hist(s[y == 0], bins=30, alpha=0.6, label="fake", density=True)
        ax.set_title(name)
        ax.set_xlabel("critic logit")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "critic_score_histograms.png", dpi=160)
    fig.savefig(out_dir / "critic_score_histograms.pdf")
    plt.close(fig)


def save_training_curves(out_dir: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in history:
        by_model.setdefault(str(row["model"]), []).append(row)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, rows in by_model.items():
        rows = sorted(rows, key=lambda r: int(r["epoch"]))
        ax.plot([r["epoch"] for r in rows], [r.get("val_auc", np.nan) for r in rows], marker="o", label=name)
    ax.axhline(0.58, color="tab:orange", linestyle="--", linewidth=1, label="weak threshold 0.58")
    ax.axhline(0.65, color="tab:green", linestyle="--", linewidth=1, label="possible threshold 0.65")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation AUC")
    ax.set_ylim(0.35, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "critic_training_curves.png", dpi=160)
    fig.savefig(out_dir / "critic_training_curves.pdf")
    plt.close(fig)


def save_auc_bar(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    names = [str(r["model"]) for r in rows]
    aucs = [float(r["auc"]) for r in rows]
    lows = [float(r["auc_ci_low"]) for r in rows]
    highs = [float(r["auc_ci_high"]) for r in rows]
    yerr = np.array([[a - l for a, l in zip(aucs, lows)], [h - a for a, h in zip(aucs, highs)]])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(range(len(rows)), aucs, yerr=yerr, capsize=3)
    ax.axhline(0.5, color="black", linewidth=1)
    ax.axhline(0.58, color="tab:orange", linestyle="--", linewidth=1)
    ax.axhline(0.65, color="tab:green", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("held-out test AUC")
    ax.set_ylim(0.35, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "critic_auc_summary.png", dpi=160)
    fig.savefig(out_dir / "critic_auc_summary.pdf")
    plt.close(fig)


def save_dataset_cache(out_dir: Path, train: SplitArrays, val: SplitArrays, test: SplitArrays) -> None:
    payload = {}
    for split_name, arr in [("train", train), ("val", val), ("test", test)]:
        for key in [
            "source_indices",
            "labels",
            "y",
            "x_data",
            "real",
            "fake",
            "real_adag",
            "fake_adag",
        ]:
            value = getattr(arr, key)
            if value.dtype.kind == "f":
                value = value.astype(np.float16)
            payload[f"{split_name}_{key}"] = value
    np.savez_compressed(out_dir / "gauge_dataset_cache_float16.npz", **payload)


def dataset_entries(arrays: list[SplitArrays], checkpoint_sha: str, max_hash_rows: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for arr in arrays:
        n = arr.source_indices.shape[0]
        for i in range(n):
            include_hash = max_hash_rows is None or i < max_hash_rows
            for label_name, label_value in [("real", 1), ("fake", 0)]:
                rows.append(
                    {
                        "split": arr.split,
                        "source_index": int(arr.source_indices[i]),
                        "stl10_label": int(arr.labels[i]),
                        "pair_label_name": label_name,
                        "label": label_value,
                        "source_checkpoint": str(CHECKPOINT),
                        "source_checkpoint_sha256": checkpoint_sha,
                        "y_hash": row_hash(arr.y[i]) if include_hash else "",
                        "x_data_hash": row_hash(arr.x_data[i]) if include_hash else "",
                    }
                )
    return rows


def preflight(out_dir: Path, device: str, train_count: int, val_count: int, test_count: int) -> tuple[bool, dict[str, Any], list[str], list[str]]:
    append_log(out_dir, "preflight_start")
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    checks["requested_project_path"] = str(REQUESTED_PROJECT)
    checks["requested_project_path_exists"] = REQUESTED_PROJECT.exists()
    if not REQUESTED_PROJECT.exists():
        warnings.append(f"Requested C: project path is missing; using actual repo mirror {REPO_ROOT}.")
    for name, path in [
        ("repo_root", REPO_ROOT),
        ("checkpoint", CHECKPOINT),
        ("resolved_config", RESOLVED_CONFIG),
        ("A_scr5", A_SCR5),
        ("eval_cache", EVAL_CACHE),
        ("provenance_json", PROVENANCE_JSON),
        ("split_train", SPLIT_TRAIN),
        ("split_eval", SPLIT_EVAL),
    ]:
        exists = path.exists()
        checks[f"{name}_path"] = str(path)
        checks[f"{name}_exists"] = exists
        if not exists:
            failures.append(f"Missing required {name}: {path}")
    if failures:
        return False, checks, failures, warnings

    try:
        prov = load_provenance()
        train_idx = np.load(SPLIT_TRAIN).astype(np.int64)
        eval_idx = np.load(SPLIT_EVAL).astype(np.int64)
        prov_splits = prov["splits"]
        checks["train_split_hash_recomputed_sorted"] = sha256_np(train_idx, sort_int64=True)
        checks["train_split_hash_expected_sorted"] = prov_splits["train_indices_sha256_sorted_int64"]
        checks["eval_split_hash_recomputed_sorted"] = sha256_np(eval_idx, sort_int64=True)
        checks["eval_split_hash_expected_sorted"] = prov_splits["eval_indices_sha256_sorted_int64"]
        if checks["train_split_hash_recomputed_sorted"] != checks["train_split_hash_expected_sorted"]:
            failures.append("Train split sorted SHA256 does not match provenance.")
        if checks["eval_split_hash_recomputed_sorted"] != checks["eval_split_hash_expected_sorted"]:
            failures.append("Eval split sorted SHA256 does not match provenance.")
        if train_count + val_count > len(train_idx):
            failures.append("Requested train+val sample count exceeds saved train split.")
        if test_count > len(eval_idx):
            failures.append("Requested test sample count exceeds saved eval split.")
    except Exception as exc:
        failures.append(f"Split/provenance reconstruction failed: {exc}")

    try:
        ckpt_sha = sha256_file(CHECKPOINT)
        manifest = read_json(EVAL_MANIFEST)
        checks["checkpoint_sha256"] = ckpt_sha
        checks["checkpoint_sha256_expected"] = manifest["checkpoint_sha256"]
        if ckpt_sha != manifest["checkpoint_sha256"]:
            failures.append("Checkpoint SHA256 does not match main_scr5 manifest.")
        ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        checks["checkpoint_keys"] = sorted(list(ck.keys())) if isinstance(ck, dict) else []
        checks["checkpoint_generator_ema_available"] = bool(isinstance(ck, dict) and ck.get("generator_ema") is not None)
        if not checks["checkpoint_generator_ema_available"]:
            failures.append("Published checkpoint lacks generator_ema for mean-mode diagnostic.")
        del ck
    except Exception as exc:
        failures.append(f"Checkpoint load failed: {exc}")

    try:
        A = np.load(A_SCR5)
        checks["A_shape"] = list(A.shape)
        checks["A_sha256_float32_bytes"] = sha256_np(A.astype(np.float32))
        manifest = read_json(EVAL_MANIFEST)
        checks["A_sha256_expected"] = manifest.get("A_sha256_float32_bytes")
        if checks["A_sha256_float32_bytes"] != checks["A_sha256_expected"]:
            failures.append("A_scr5 SHA256 does not match main_scr5 manifest.")
        G = A.astype(np.float64) @ A.astype(np.float64).T
        checks["AAT_minus_I_max"] = float(np.max(np.abs(G - np.eye(G.shape[0]))))
        if checks["AAT_minus_I_max"] > 1e-5:
            failures.append("Scr-5 A is not row-orthonormal to tolerance.")
    except Exception as exc:
        failures.append(f"Exact A load/check failed: {exc}")

    try:
        ensure_dir(out_dir)
        probe = out_dir / "_write_probe.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks["output_saving_available"] = True
    except Exception as exc:
        failures.append(f"Output saving check failed: {exc}")

    try:
        config = make_config(device)
        torch_device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        measurement = make_measurement(config, torch_device)
        A_t = torch.from_numpy(np.load(A_SCR5).astype(np.float32)).to(torch_device)
        measurement.set_A_override(A_t, metadata={"source": str(A_SCR5), "phase": "phase69A_preflight"}, rebuild_cache=True)
        A64, G, K = exact_projectors(measurement.A, float(config["lambda_solver"]))
        z = np.load(EVAL_CACHE, allow_pickle=False)
        y = torch.from_numpy(z["y"][:2]).to(torch_device)
        x = torch.from_numpy(z["x"][:2]).to(torch_device).reshape(2, 1, 64, 64)
        x_flat = measurement.flatten_img(x)
        x_data = data_solution_safe(measurement, y, config.get("backprojection_mode"))
        b = blambda_y(y, A64, K)
        p0 = p0_exact(x_flat, A64, G)
        checks["preflight_x_data_shape"] = list(x_data.shape)
        checks["preflight_Blambda_shape"] = list(b.shape)
        checks["preflight_P0_shape"] = list(p0.shape)
        ck, gen, cfg = load_checkpoint_and_generator(config, measurement, torch_device)
        v, xd = mean_mode_pre_final(gen, measurement, y, cfg)
        checks["preflight_v_mean_shape"] = list(v.shape)
        checks["preflight_v_mean_finite"] = bool(torch.isfinite(v).all().item())
        if not checks["preflight_v_mean_finite"]:
            failures.append("v_mean contains non-finite values.")
        del ck, gen, measurement, A_t, A64, G, K, z, y, x, x_flat, x_data, b, p0, v, xd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        failures.append(f"P0/B_lambda/x_data/v_mean preflight failed: {exc}")

    checks["safe_to_run"] = len(failures) == 0
    append_log(out_dir, f"preflight_complete safe={checks['safe_to_run']}")
    return len(failures) == 0, checks, failures, warnings


def write_preflight_report(out_dir: Path, checks: dict[str, Any], warnings: list[str]) -> None:
    lines = [
        "# Phase69A Preflight Safety",
        "",
        f"- safe_to_run: `{checks.get('safe_to_run')}`",
        f"- requested_project_path_exists: `{checks.get('requested_project_path_exists')}`",
        f"- actual_repo_root: `{REPO_ROOT}`",
        f"- checkpoint_sha256: `{checks.get('checkpoint_sha256')}`",
        f"- train split sorted SHA256: `{checks.get('train_split_hash_recomputed_sorted')}`",
        f"- eval split sorted SHA256: `{checks.get('eval_split_hash_recomputed_sorted')}`",
        f"- A_scr5 SHA256: `{checks.get('A_sha256_float32_bytes')}`",
        f"- AAT_minus_I_max: `{checks.get('AAT_minus_I_max')}`",
        f"- output_saving_available: `{checks.get('output_saving_available')}`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {w}" for w in warnings] or ["- none"])
    lines.extend(["", "## Raw Checks", "", "```json", json.dumps(json_safe(checks), indent=2), "```", ""])
    write_text(out_dir / "PREFLIGHT_SAFETY.md", "\n".join(lines))


def decision_from_auc(rows: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any] | None]:
    main_rows = [r for r in rows if r["model"] in {"patchgan_unconditional_gauge", "patchgan_conditional_xdata_gauge", "simple_cnn_gauge"}]
    if not main_rows:
        return "NO_GO", "no main critic rows", None
    best = max(main_rows, key=lambda r: float(r["auc"]))
    auc = float(best["auc"])
    lo = float(best["auc_ci_low"])
    if auc < 0.58:
        return "NO_GO", f"best gauge-equalized AUC {auc:.3f} < 0.58: no useful adversarial signal", best
    if auc < 0.65:
        return "NO_GO_WEAK", f"best gauge-equalized AUC {auc:.3f} is weak (0.58-0.65); do not run cGAN unless required", best
    if lo > 0.58:
        return "GO_CONSIDER_PILOT", f"best gauge-equalized AUC {auc:.3f} with CI lower {lo:.3f} > 0.58: possible signal", best
    return "NO_GO_WEAK_CI", f"best gauge-equalized AUC {auc:.3f} but CI lower {lo:.3f} is not above 0.58", best


def write_reports(
    out_dir: Path,
    split_info: dict[str, Any],
    stats_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    decision: tuple[str, str, dict[str, Any] | None],
) -> None:
    go, reason, best = decision
    main_rows = [r for r in result_rows if r["model"] in {"patchgan_unconditional_gauge", "patchgan_conditional_xdata_gauge", "simple_cnn_gauge"}]
    cond = next((r for r in main_rows if r["model"] == "patchgan_conditional_xdata_gauge"), None)
    uncond = next((r for r in main_rows if r["model"] == "patchgan_unconditional_gauge"), None)
    residual = next((r for r in control_rows if r["model"] == "residual_features_logistic_raw"), None)
    shuffled = next((r for r in control_rows if r["model"] == "patchgan_conditional_shuffled_xdata_gauge"), None)
    cond_better = (
        bool(cond and uncond and float(cond["auc"]) > float(uncond["auc"]))
        if cond and uncond
        else None
    )
    residual_cheat = bool(residual and float(residual["auc"]) > max(0.8, float(best["auc"]) + 0.1 if best else 0.8))
    gauge_remove = bool(residual and best and float(residual["auc"]) > float(best["auc"]) + 0.1)
    stable = max(float(r["measurement_residual_rel_max_vs_ABlambda"]) for r in stats_rows if r["kind"] in {"real", "fake"}) < 1e-5
    recommended_input = "D(tilde_x, x_data) only if conditional AUC materially beats unconditional; otherwise D(tilde_x). Never feed residual/correction features."
    if cond_better is True and cond and uncond and float(cond["auc"]) - float(uncond["auc"]) > 0.02:
        recommended_input = "conditional gauge image plus x_data anchor: D(tilde_x, x_data), with shuffled-condition control retained."
    elif uncond:
        recommended_input = "unconditional gauge image only: D(tilde_x)."

    report_lines = [
        "# Phase69A Gauge-Equalized GAN Signal Diagnostic",
        "",
        f"Output directory: `{out_dir}`",
        "",
        "No generator training, reconstruction-network training, GAN fine-tune, checkpoint update, or main-result update was performed.",
        "",
        "## AUC Results",
        "",
        format_table(
            result_rows,
            ["model", "auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "brier", "best_val_auc"],
        ),
        "",
        "## Shortcut Controls",
        "",
        format_table(
            control_rows,
            ["model", "auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "brier", "best_val_auc"],
        ),
        "",
        "## Required Answers",
        "",
        f"1. Does gauge-equalized D have usable signal? {reason}.",
        f"2. Is conditional D better than unconditional? `{cond_better}`.",
        f"3. Does residual-fed D cheat? `{residual_cheat}`.",
        f"4. Does gauge remove shortcut? `{gauge_remove}`.",
        f"5. Are real/fake canonical images numerically stable? `{stable}`.",
        f"6. Should Phase69B controlled cGAN be run? `{go}`.",
        f"7. Recommended D input for Phase69B: {recommended_input}",
        "8. Risks: small-budget critic only; train-side y is deterministic local measurement noise while test-side y is frozen recorded cert y; conditional D may exploit anchor/style correlation; high AUC would require additional shortcut checks before any cGAN pilot.",
        "",
        "## Split / Provenance",
        "",
        f"- critic train count: `{split_info['critic_train_count']}`",
        f"- critic val count: `{split_info['critic_val_count']}`",
        f"- held-out test count: `{split_info['critic_test_count']}`",
        f"- train full sorted SHA256: `{split_info['train_indices_full_sorted_sha256']}`",
        f"- eval full sorted SHA256: `{split_info['eval_indices_full_sorted_sha256']}`",
        f"- train source split: `{split_info['train_source_split']}`",
        f"- held-out test source split: `{split_info['heldout_test_source_split']}`",
        "",
        "## Stability Stats",
        "",
        format_table(
            stats_rows,
            [
                "split",
                "kind",
                "n",
                "min",
                "max",
                "mean",
                "std",
                "measurement_residual_rel_median_vs_ABlambda",
                "measurement_residual_rel_max_vs_ABlambda",
                "p0_energy_median",
            ],
        ),
        "",
    ]
    write_text(out_dir / "PHASE69A_GAUGE_GAN_SIGNAL_REPORT.md", "\n".join(report_lines))
    go_lines = [
        "# Phase69A Go / No-Go",
        "",
        f"Decision: `{go}`",
        "",
        reason,
        "",
        f"Best main critic: `{best['model'] if best else 'none'}`",
        f"Best main AUC: `{best['auc'] if best else 'nan'}`",
        "",
        "Do not run Phase69B unless the user explicitly approves a controlled three-arm pilot.",
        "",
        "No generator training was performed.",
        "",
    ]
    write_text(out_dir / "PHASE69A_GO_NOGO.md", "\n".join(go_lines))
    critic_lines = [
        "# Gauge Critic Report",
        "",
        "Main critic inputs exclude `Au-y`, RelMeasErr, correction vectors, and `Pi_y(v)-v`.",
        "",
        format_table(
            result_rows,
            ["model", "auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "brier"],
        ),
        "",
    ]
    write_text(out_dir / "gauge_critic_report.md", "\n".join(critic_lines))
    control_lines = [
        "# Shortcut Control Report",
        "",
        "Residual features are used only in the explicit negative control.",
        "",
        format_table(
            control_rows,
            ["model", "auc", "auc_ci_low", "auc_ci_high", "accuracy", "balanced_accuracy", "brier"],
        ),
        "",
    ]
    write_text(out_dir / "shortcut_control_report.md", "\n".join(control_lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Phase69A gauge-equalized GAN signal diagnostic.")
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_count", type=int, default=1024)
    parser.add_argument("--val_count", type=int, default=384)
    parser.add_argument("--test_count", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    if args.quick:
        args.train_count = min(args.train_count, 256)
        args.val_count = min(args.val_count, 128)
        args.test_count = min(args.test_count, 256)
        args.epochs = min(args.epochs, 2)
    write_text(out_dir / "RUNLOG.md", f"# Phase69A Runlog\n- {now()} runner_start\n")
    safe, checks, failures, warnings = preflight(out_dir, args.device, args.train_count, args.val_count, args.test_count)
    checks["safe_to_run"] = safe
    save_json(out_dir / "preflight_checks.json", checks)
    if not safe:
        unsafe_stop(out_dir, failures, warnings)
        return 2
    write_preflight_report(out_dir, checks, warnings)

    append_log(out_dir, "main_start")
    random.seed(69)
    np.random.seed(69)
    set_seed(69)
    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    config = make_config(str(device))
    measurement = make_measurement(config, device)
    A_np = np.load(A_SCR5).astype(np.float32)
    A_t = torch.from_numpy(A_np).to(device)
    override_stats = measurement.set_A_override(
        A_t,
        metadata={"source": str(A_SCR5), "phase": "phase69A_main", "tensor_sha256": sha256_np(A_np)},
        rebuild_cache=True,
    )
    A64, G, K = exact_projectors(measurement.A, float(config["lambda_solver"]))
    checkpoint, generator, config = load_checkpoint_and_generator(config, measurement, device)
    checkpoint_sha = sha256_file(CHECKPOINT)

    train_loader, _, split_info = make_source_loaders(args.train_count, args.val_count, args.test_count, config["batch_size"])
    split_info["measurement_override_stats"] = override_stats
    split_info["checkpoint"] = str(CHECKPOINT)
    split_info["checkpoint_sha256"] = checkpoint_sha
    split_info["A_scr5"] = str(A_SCR5)
    split_info["A_scr5_sha256"] = sha256_np(A_np)
    save_json(out_dir / "split_manifest.json", split_info)

    train, val = build_train_val_arrays(
        train_loader,
        args.train_count,
        args.val_count,
        generator,
        measurement,
        config,
        A64,
        G,
        K,
        device,
        out_dir,
    )
    test = build_test_arrays(args.test_count, generator, measurement, config, A64, G, K, device, out_dir)
    del generator, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stats_rows: list[dict[str, Any]] = []
    for arr in [train, val, test]:
        stats_rows.extend(canonical_stats(arr.split, arr, A_np, float(config["lambda_solver"])))
    write_csv(out_dir / "gauge_dataset_stats.csv", stats_rows)
    save_example_grid(out_dir, test)
    save_dataset_cache(out_dir, train, val, test)

    entries = dataset_entries([train, val, test], checkpoint_sha, max_hash_rows=None)
    write_csv(out_dir / "gauge_dataset_entries.csv", entries)
    manifest = {
        "phase": "Phase69A",
        "created": now(),
        "source_checkpoint": str(CHECKPOINT),
        "source_checkpoint_sha256": checkpoint_sha,
        "A_path": str(A_SCR5),
        "A_sha256_float32_bytes": sha256_np(A_np),
        "split_info": split_info,
        "entries_csv": str(out_dir / "gauge_dataset_entries.csv"),
        "dataset_cache": str(out_dir / "gauge_dataset_cache_float16.npz"),
        "forbidden_main_critic_inputs": ["Au-y", "RelMeasErr", "correction vector", "Pi_y(v)-v"],
        "no_generator_training": True,
        "no_reconstruction_network_training": True,
    }
    save_json(out_dir / "gauge_dataset_manifest.json", manifest)

    append_log(out_dir, "critic_training_all_start")
    train_uncond = pair_arrays(train, include_cond=False)
    val_uncond = pair_arrays(val, include_cond=False)
    test_uncond = pair_arrays(test, include_cond=False)
    train_cond = pair_arrays(train, include_cond=True)
    val_cond = pair_arrays(val, include_cond=True)
    test_cond = pair_arrays(test, include_cond=True)
    train_cond_shuf = pair_arrays(train, include_cond=True, shuffle_cond=True, seed=691)
    val_cond_shuf = pair_arrays(val, include_cond=True, shuffle_cond=True, seed=692)
    test_cond_shuf = pair_arrays(test, include_cond=True, shuffle_cond=True, seed=693)
    train_adag = pair_arrays(train, gauge="adagger", include_cond=False)
    val_adag = pair_arrays(val, gauge="adagger", include_cond=False)
    test_adag = pair_arrays(test, gauge="adagger", include_cond=False)

    results: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    score_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    r, h, y_score, s_score = train_image_model(
        "patchgan_unconditional_gauge",
        PatchCritic(1),
        train_uncond[:3],
        val_uncond[:3],
        test_uncond[:3],
        device=device,
        out_dir=out_dir,
        conditional=False,
        epochs=args.epochs,
        seed=6910,
    )
    results.append(r)
    history.extend(h)
    score_sets[r["model"]] = (y_score, s_score)

    r, h, y_score, s_score = train_image_model(
        "patchgan_conditional_xdata_gauge",
        PatchCritic(2),
        train_cond[:3],
        val_cond[:3],
        test_cond[:3],
        device=device,
        out_dir=out_dir,
        conditional=True,
        epochs=args.epochs,
        seed=6911,
    )
    results.append(r)
    history.extend(h)
    score_sets[r["model"]] = (y_score, s_score)

    r, h, y_score, s_score = train_image_model(
        "simple_cnn_gauge",
        SimpleCNN(1),
        train_uncond[:3],
        val_uncond[:3],
        test_uncond[:3],
        device=device,
        out_dir=out_dir,
        conditional=False,
        epochs=args.epochs,
        seed=6912,
    )
    results.append(r)
    history.extend(h)
    score_sets[r["model"]] = (y_score, s_score)

    r, h, y_score, s_score = train_residual_logistic(train, val, test, device=device, out_dir=out_dir)
    controls.append(r)
    history.extend(h)
    score_sets[r["model"]] = (y_score, s_score)

    r, h, y_score, s_score = train_image_model(
        "patchgan_conditional_shuffled_xdata_gauge",
        PatchCritic(2),
        train_cond_shuf[:3],
        val_cond_shuf[:3],
        test_cond_shuf[:3],
        device=device,
        out_dir=out_dir,
        conditional=True,
        epochs=max(2, args.epochs - 1),
        seed=6913,
    )
    controls.append(r)
    history.extend(h)
    score_sets[r["model"]] = (y_score, s_score)

    # Gauge choice control: same images rebuilt with A^\dagger y rather than B_lambda y.
    small = min(args.train_count, 512)
    train_adag_small = subset_pair_tuple(train_adag, args.train_count, small)
    r, h, y_score, s_score = train_image_model(
        "simple_cnn_adagger_gauge_subset",
        SimpleCNN(1),
        train_adag_small[:3],
        val_adag[:3],
        test_adag[:3],
        device=device,
        out_dir=out_dir,
        conditional=False,
        epochs=max(2, args.epochs - 1),
        seed=6914,
    )
    controls.append(r)
    history.extend(h)
    score_sets[r["model"]] = (y_score, s_score)

    write_csv(out_dir / "critic_auc_results.csv", results)
    write_csv(out_dir / "shortcut_control_results.csv", controls)
    write_csv(out_dir / "critic_training_history.csv", history)

    labels_pair = np.concatenate([test.labels, test.labels], axis=0)
    per_class_rows = []
    for name, (yt, st) in score_sets.items():
        for row in per_class_auc(labels_pair, yt.astype(int), st):
            row["model"] = name
            per_class_rows.append(row)
    write_csv(out_dir / "critic_per_class_auc.csv", per_class_rows)

    save_score_histograms(out_dir, score_sets)
    save_training_curves(out_dir, history)
    save_auc_bar(out_dir, results + controls)

    # B_lambda versus A_dagger image stability.
    diff_rows = []
    for arr in [train, val, test]:
        for kind, b_arr, a_arr in [("real", arr.real, arr.real_adag), ("fake", arr.fake, arr.fake_adag)]:
            diff = a_arr.astype(np.float64) - b_arr.astype(np.float64)
            denom = np.linalg.norm(b_arr.astype(np.float64), axis=1).clip(1e-12)
            diff_rows.append(
                {
                    "split": arr.split,
                    "kind": kind,
                    "median_rel_l2_adagger_minus_blambda": float(np.median(np.linalg.norm(diff, axis=1) / denom)),
                    "max_abs_pixel_diff": float(np.max(np.abs(diff))),
                    "mean_abs_pixel_diff": float(np.mean(np.abs(diff))),
                }
            )
    write_csv(out_dir / "gauge_choice_stability.csv", diff_rows)

    decision = decision_from_auc(results)
    write_reports(out_dir, split_info, stats_rows, results, controls, decision)
    append_log(out_dir, f"runner_complete decision={decision[0]} no_generator_training=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
