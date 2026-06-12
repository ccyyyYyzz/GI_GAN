from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import _filter_dataset, _make_dataset, build_transform
from .eval import make_measurement
from .exact_measurement import apply_measurement_override_from_config
from .metrics import batch_metrics
from .phase48_49_common import file_sha256, load_bundle_task, write_csv, write_environment, write_markdown_table, write_sha256s
from .phase53C_exact_projector import build_rowspace_basis, project_null
from .phase53D_common import (
    DEFAULT_BUNDLE_ROOT,
    DEFAULT_DATASET_ROOT,
    pooled_image_features,
    save_bar,
    save_histogram,
    save_image_grid,
    save_scatter,
)
from .utils import apply_experiment_defaults, ensure_dir, save_config, save_json, set_seed


PHASE56_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase56_group_split_exact_null_critic")
PHASE55_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase55_cross_audit")
PHASE53C_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import")
PHASE53D_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase53D_local_preflight")
TASKS = ["rad5", "scr5", "rad10", "scr10"]
SPLIT_MODES = ["strict_both_group_split", "anchor_heldout_only", "null_heldout_only", "pair_split_reproduction"]
NEGATIVE_TYPES = ["random", "same_class", "nearest_anchor", "alpha_chimera"]


def resolve_device(requested: str = "cuda") -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def add_args(parser):
    parser.add_argument("--bundle_root", default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_dir", default=str(PHASE56_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", nargs="*", default=TASKS)
    parser.add_argument("--limit_samples", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303])
    parser.add_argument("--critic_epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=560)
    return parser


def write_rows(root: str | Path, stem: str, rows: list[dict[str, Any]], title: str) -> None:
    root = ensure_dir(root)
    write_csv(root / f"{stem}.csv", rows)
    write_markdown_table(root / f"{stem}.md", rows, title)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def fmt(value: Any, digits: int = 3) -> str:
    v = to_float(value)
    return "n/a" if math.isnan(v) else f"{v:.{digits}f}"


def mean(values: list[Any]) -> float:
    vals = [to_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def max_value(values: list[Any]) -> float:
    vals = [to_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return max(vals) if vals else float("nan")


def configure_task(args, task_key: str, task_out: Path, device: torch.device):
    info = load_bundle_task(args.bundle_root, task_key)
    config = apply_experiment_defaults(info["config"])
    config["dataset_root"] = args.dataset_root
    config["device"] = str(device)
    config["batch_size"] = int(args.batch_size)
    config["num_workers"] = int(args.num_workers)
    config["limit_val_samples"] = int(args.limit_samples)
    config["phase56_note"] = "Group-split exact-null critic repeat; no reconstruction generator training."
    if info["exact_A_path"] is not None:
        config["measurement_operator_exact_path"] = str(info["exact_A_path"])
        config["exact_A_required"] = bool(info["metadata"]["requires_exact_A"])
    ensure_dir(task_out)
    save_config(config, task_out / "config_used.yaml")
    save_json(
        {
            "task": task_key,
            "config_path": str(info["config_path"]),
            "checkpoint_path": str(info["checkpoint_path"]),
            "metrics_path": str(info["metrics_path"]) if info.get("metrics_path") else "",
            "exact_A_path": str(info["exact_A_path"]) if info.get("exact_A_path") else "",
            "exact_A_sha256": file_sha256(info["exact_A_path"]) if info.get("exact_A_path") else "",
        },
        task_out / "source_paths.json",
    )
    measurement = make_measurement(config, device)
    exact_info = apply_measurement_override_from_config(config, measurement, device)
    save_json(exact_info, task_out / "exact_A_info.json")
    return info, config, measurement, exact_info


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_ids: list[int]):
        self.dataset = dataset
        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, local_idx: int):
        img, label = self.dataset[local_idx]
        return img, int(label), int(self.image_ids[local_idx]), int(local_idx)


def dataset_with_ids(config: dict[str, Any], limit_samples: int, seed: int):
    transform = build_transform(int(config["img_size"]), dataset_name=config.get("dataset_name", "stl10"), train=False, use_augmentation=False)
    dataset = _make_dataset(config["dataset_root"], config.get("dataset_name", "stl10"), config.get("val_split", "test"), transform, download=True)
    dataset = _filter_dataset(dataset, config.get("dataset_name", "stl10"), config.get("class_filter"))
    base_ids = list(getattr(dataset, "indices", range(len(dataset))))
    limit = min(int(limit_samples), len(dataset))
    gen = torch.Generator().manual_seed(seed + 1)
    selected_local = torch.randperm(len(dataset), generator=gen)[:limit].tolist()
    from torch.utils.data import Subset

    selected_dataset = Subset(dataset, selected_local)
    selected_ids = [int(base_ids[i]) for i in selected_local]
    return IndexedDataset(selected_dataset, selected_ids)


def split_ids(image_ids: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    perm = image_ids[torch.randperm(image_ids.numel(), generator=gen)]
    n = perm.numel()
    n_train = max(4, int(0.6 * n))
    n_val = max(2, int(0.2 * n))
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def save_id_csv(path: Path, ids: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id"])
        writer.writeheader()
        for image_id in ids.tolist():
            writer.writerow({"image_id": int(image_id)})


def build_nearest_indices(anchor_flat: torch.Tensor, split_local: list[int]) -> dict[int, int]:
    if len(split_local) < 2:
        return {i: i for i in split_local}
    feats = pooled_image_features(anchor_flat[split_local], pool=8)
    feats = (feats - feats.mean(0, keepdim=True)) / feats.std(0, keepdim=True).clamp_min(1e-6)
    dist = torch.cdist(feats, feats)
    dist.fill_diagonal_(float("inf"))
    nearest = torch.argmin(dist, dim=1).tolist()
    return {int(split_local[i]): int(split_local[j]) for i, j in enumerate(nearest)}


def same_class_choice(labels: torch.Tensor, pool: list[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    by_cls: dict[int, list[int]] = {}
    for idx in pool:
        by_cls.setdefault(int(labels[idx]), []).append(int(idx))
    for members in by_cls.values():
        if len(members) == 1:
            out[members[0]] = members[0]
            continue
        for i, idx in enumerate(members):
            out[idx] = members[(i + 1) % len(members)]
    return out


def choose_negative(local_idx: int, pool: list[int], labels: torch.Tensor, nearest: dict[int, int], same_class: dict[int, int], negative_type: str) -> int:
    if len(pool) < 2:
        return local_idx
    if negative_type == "nearest_anchor":
        cand = nearest.get(local_idx, pool[0])
        return cand if cand != local_idx else pool[(pool.index(local_idx) + 1) % len(pool)] if local_idx in pool else cand
    if negative_type == "same_class":
        cand = same_class.get(local_idx, pool[0])
        return cand if cand != local_idx else pool[(pool.index(local_idx) + 1) % len(pool)] if local_idx in pool else cand
    pos = pool.index(local_idx) if local_idx in pool else 0
    return pool[(pos + 1) % len(pool)]


def binary_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu().float()
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    cmp = (pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()
    return float(cmp.mean())


def average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu().float()
    order = torch.argsort(scores, descending=True)
    y = labels[order]
    total_pos = y.sum().clamp_min(1)
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float32)
    return float((precision * y).sum() / total_pos)


def bootstrap_auc_ci(labels: torch.Tensor, scores: torch.Tensor, n_boot: int = 120) -> tuple[float, float]:
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu().float()
    n = labels.numel()
    gen = torch.Generator().manual_seed(56056)
    vals = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=gen)
        if labels[idx].unique().numel() < 2:
            continue
        vals.append(binary_auc(labels[idx], scores[idx]))
    if not vals:
        return float("nan"), float("nan")
    t = torch.tensor(vals)
    return float(torch.quantile(t, 0.025)), float(torch.quantile(t, 0.975))


def binary_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    labels = labels.detach().cpu().float()
    scores = scores.detach().cpu().float()
    probs = torch.sigmoid(scores)
    pred = (probs >= 0.5).float()
    tp = float(((pred == 1) & (labels == 1)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    auc = binary_auc(labels, scores)
    ci_low, ci_high = bootstrap_auc_ci(labels, scores)
    return {
        "auc": auc,
        "auc_abs": max(auc, 1.0 - auc) if auc == auc else float("nan"),
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "accuracy": float((pred == labels).float().mean()),
        "balanced_accuracy": 0.5 * (tp / max(1.0, tp + fn) + tn / max(1.0, tn + fp)),
        "average_precision": average_precision(labels, scores),
        "brier": float(torch.mean((probs - labels) ** 2)),
    }


def standardize_fit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp_min(1e-6)
    return mean, std


def standardize_apply(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def fit_pca(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.float().cpu()
    mean = x.mean(0, keepdim=True)
    xc = x - mean
    q = min(dim + 8, min(xc.shape) - 1)
    _u, _s, v = torch.pca_lowrank(xc, q=q, center=False, niter=2)
    return mean, v[:, :dim]


def project_pca(x: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (x.float().cpu() - mean) @ basis


def pair_tabular_features(p0: torch.Tensor, anchor: torch.Tensor, dim: int = 128) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    flat = torch.cat([p0.flatten(1), anchor.flatten(1)], dim=0)
    mean_, basis = fit_pca(flat, dim)
    p = project_pca(p0.flatten(1), mean_, basis)
    a = project_pca(anchor.flatten(1), mean_, basis)
    feats = torch.cat([p, a, torch.abs(p - a), p * a], dim=1)
    return feats, {"mean": mean_, "basis": basis}


def transform_pair_tabular(p0: torch.Tensor, anchor: torch.Tensor, pca: dict[str, torch.Tensor]) -> torch.Tensor:
    p = project_pca(p0.flatten(1), pca["mean"], pca["basis"])
    a = project_pca(anchor.flatten(1), pca["mean"], pca["basis"])
    return torch.cat([p, a, torch.abs(p - a), p * a], dim=1)


def fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float = 1.0) -> torch.Tensor:
    y2 = y.float() * 2.0 - 1.0
    X = torch.cat([x.float(), torch.ones(x.shape[0], 1)], dim=1)
    eye = torch.eye(X.shape[1], dtype=X.dtype)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(X.T @ X + ridge * eye, X.T @ y2)


def fit_gradient_linear(x: torch.Tensor, y: torch.Tensor, kind: str, steps: int = 120, lr: float = 0.04) -> torch.Tensor:
    X = torch.cat([x.float(), torch.ones(x.shape[0], 1)], dim=1)
    y = y.float()
    signed = y * 2.0 - 1.0
    w = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        scores = X @ w
        loss = F.binary_cross_entropy_with_logits(scores, y) if kind == "logistic" else torch.clamp(1.0 - signed * scores, min=0).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return w.detach()


def score_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    X = torch.cat([x.float(), torch.ones(x.shape[0], 1)], dim=1)
    return X @ w


class ExactNullCriticSmall(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, base, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(base * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).flatten(1)).squeeze(1)


class PairMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 192), nn.ReLU(inplace=True), nn.Linear(192, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_torch_model(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, *, device: torch.device, epochs: int, lr: float = 2e-4) -> tuple[nn.Module, dict[str, float]]:
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.float().to(device)
    x_val = x_val.to(device)
    y_val = y_val.float().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    batch = min(64, y_train.numel())
    for _ in range(epochs):
        perm = torch.randperm(y_train.numel(), device=device)
        for start in range(0, perm.numel(), batch):
            idx = perm[start : start + batch]
            logits = model(x_train[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        scores = model(x_val).detach().cpu()
    return model.cpu(), binary_metrics(y_val.detach().cpu(), scores)


def write_command_log(out: Path) -> None:
    ensure_dir(out)
    (out / "command_log.txt").write_text("$ " + " ".join(sys.argv) + "\n", encoding="utf-8")


def finalize(root: Path, payload: dict[str, Any]) -> None:
    save_json(payload, root / "PHASE56_MANIFEST.json")
    write_environment(root)
    write_sha256s(root)

