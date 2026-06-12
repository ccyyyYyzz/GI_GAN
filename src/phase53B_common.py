from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import get_val_dataloader
from .eval import make_measurement
from .split_guard import get_train_dataloader_guarded
from .exact_measurement import apply_measurement_override_from_config, torch_load
from .metrics import batch_metrics
from .models import build_generator
from .phase48_49_common import (
    TASKS,
    copy_required_bundle_leaf,
    load_bundle_task,
    save_run_config,
    write_csv,
    write_environment,
    write_markdown_table,
    write_session_manifest,
    write_sha256s,
)
from .utils import apply_experiment_defaults, ensure_dir, reconstruct_from_measurements, save_config, save_json, set_seed


TASK_ORDER = ["rad5", "scr5", "rad10", "scr10"]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--bundle_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--session_name", required=True)
    parser.add_argument("--dataset_root", default="/content/ns_mc_gan_gi_data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", nargs="*", default=["rad5", "scr5", "rad10", "scr10"])
    parser.add_argument("--limit_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--critic_epochs", type=int, default=8)
    parser.add_argument("--critic_lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def resolve_device(requested: str) -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def write_command_log(output_dir: Path, argv: list[str] | None = None) -> Path:
    path = output_dir / "command_log.txt"
    argv = argv or sys.argv
    path.write_text("$ " + " ".join(argv) + "\n", encoding="utf-8")
    return path


def finalize_session(
    output_dir: Path,
    session_name: str,
    ok: bool,
    payload: dict[str, Any] | None = None,
    report_lines: list[str] | None = None,
) -> None:
    payload = dict(payload or {})
    save_json({"ok": ok, "session": session_name, **payload}, output_dir / "SESSION_STATUS.json")
    write_session_manifest(output_dir, session_name, payload)
    if report_lines:
        (output_dir / f"{session_name.upper()}_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_environment(output_dir)
    write_sha256s(output_dir)


def write_rows(output_dir: Path, stem: str, rows: list[dict[str, Any]], title: str) -> None:
    write_csv(output_dir / f"{stem}.csv", rows)
    write_markdown_table(output_dir / f"{stem}.md", rows, title)


def configure_task(args: argparse.Namespace, task_key: str, task_out: Path, device: torch.device):
    info = load_bundle_task(args.bundle_root, task_key)
    config = apply_experiment_defaults(info["config"])
    config["dataset_root"] = args.dataset_root
    config["device"] = str(device)
    config["batch_size"] = int(args.batch_size)
    config["num_workers"] = int(args.num_workers)
    config["limit_val_samples"] = int(args.limit_samples)
    # Session-level cap for training-time loaders too: bundle resolved configs
    # carry limit_train_samples=50000, which would blow critic-data collection
    # up ~100x if inherited.
    config["limit_train_samples"] = int(args.limit_samples)
    config["phase53B_note"] = "Exploratory certified-blind null-space critic screening; no test-set checkpoint selection."
    if info["exact_A_path"] is not None:
        config["measurement_operator_exact_path"] = str(info["exact_A_path"])
        config["exact_A_required"] = bool(info["metadata"]["requires_exact_A"])
    task_out.mkdir(parents=True, exist_ok=True)
    save_run_config(config, task_out)
    save_config(config, task_out / "config_used.yaml")
    copy_required_bundle_leaf(args.bundle_root, task_out / "_source_bundle_leaf", task_key)
    measurement = make_measurement(config, device)
    exact_info = apply_measurement_override_from_config(config, measurement, device)
    save_json(exact_info, task_out / "exact_A_info.json")
    return info, config, measurement, exact_info


def make_loader(config: dict[str, Any], device: torch.device):
    """EVAL-ONLY loader (test split). Never use in a training-time loop:
    GAN/critic updates must use make_train_loader (split-guarded)."""
    return get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(config.get("batch_size", 16)),
        num_workers=int(config.get("num_workers", 2)),
        limit_val_samples=int(config.get("limit_val_samples", 512)),
        seed=int(config.get("seed", 123)),
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )


def make_train_loader(config: dict[str, Any], device: torch.device):
    """TRAIN-split loader for every training-time loop (GAN updates, critic
    data collection). Split-guarded at creation: raises SplitViolationError
    if it could reach test-split samples."""
    raw_limit = config.get("limit_train_samples")
    if raw_limit is None:
        # YAML "limit_train_samples: null" means full split for train.py, but
        # phase53 sessions are capped by the session-level sample limit.
        raw_limit = config.get("limit_val_samples", 512)
    return get_train_dataloader_guarded(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(config.get("batch_size", 16)),
        num_workers=int(config.get("num_workers", 2)),
        limit_train_samples=int(raw_limit) if raw_limit is not None else None,
        seed=int(config.get("seed", 123)),
        train_split=str(config.get("train_split", "train+unlabeled")),
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
        context="phase53 GAN/critic training loader",
    )


def load_generator(info: dict[str, Any], config: dict[str, Any], measurement, device: torch.device):
    checkpoint = torch_load(info["checkpoint_path"], map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        merged = dict(config)
        merged.update(checkpoint["config"])
        merged["dataset_root"] = config["dataset_root"]
        merged["device"] = str(device)
        merged["batch_size"] = config.get("batch_size", 16)
        merged["limit_val_samples"] = config.get("limit_val_samples", 512)
        if info.get("exact_A_path") is not None:
            merged["measurement_operator_exact_path"] = str(info["exact_A_path"])
            merged["exact_A_required"] = bool(info["metadata"]["requires_exact_A"])
        config = apply_experiment_defaults(merged)
    generator = build_generator(config, measurement=measurement).to(device)
    state = checkpoint.get("generator_ema") or checkpoint.get("generator") if isinstance(checkpoint, dict) else checkpoint
    if state is None:
        raise RuntimeError(f"No generator state in checkpoint: {info['checkpoint_path']}")
    generator.load_state_dict(state)
    generator.eval()
    return generator


def flatten_batch(measurement, x: torch.Tensor) -> torch.Tensor:
    return measurement.flatten_img(x.float())


def unflatten_batch(measurement, flat: torch.Tensor) -> torch.Tensor:
    return measurement.unflatten_img(flat.float())


def relmeas_tensor(measurement, flat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    err = measurement.A_forward(flat.float()) - y.float()
    return torch.linalg.norm(err, dim=1) / torch.linalg.norm(y.float(), dim=1).clamp_min(1e-12)


def null_component(measurement, image_or_flat: torch.Tensor) -> torch.Tensor:
    flat = image_or_flat if image_or_flat.ndim == 2 else flatten_batch(measurement, image_or_flat)
    return unflatten_batch(measurement, measurement.null_project(flat.float()))


def anchor_from_y(measurement, y: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=False):
        flat = measurement.data_solution(y.float(), mode=config.get("backprojection_mode", "ridge_pinv"))
    return unflatten_batch(measurement, flat)


def audited_cross_pair(measurement, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xj = torch.roll(x, shifts=1, dims=0)
    xj_flat = flatten_batch(measurement, xj)
    return unflatten_batch(measurement, measurement.dc_project(xj_flat, y.float()))


def residual_easy_wrong_pair(_measurement, x: torch.Tensor, _y: torch.Tensor) -> torch.Tensor:
    return torch.roll(x, shifts=1, dims=0)


def collect_pair_dataset(
    config: dict[str, Any],
    measurement,
    device: torch.device,
    negative_mode: str,
    limit_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    # Critic training data: TRAIN split only (split-guarded).
    loader = make_train_loader(config, device)
    u_images: list[torch.Tensor] = []
    anchors: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    y_refs: list[torch.Tensor] = []
    clean_targets: list[torch.Tensor] = []
    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        x = batch[0].to(device, non_blocking=True)
        y = measurement.measure(x)
        anchor = anchor_from_y(measurement, y, config)
        if x.shape[0] < 2:
            continue
        if negative_mode == "audited_cross_pair":
            neg = audited_cross_pair(measurement, x, y)
        elif negative_mode == "residual_easy_wrong_y":
            neg = residual_easy_wrong_pair(measurement, x, y)
        else:
            raise ValueError(f"Unknown negative_mode: {negative_mode}")
        u = torch.cat([x, neg], dim=0)
        anchor_rep = torch.cat([anchor, anchor], dim=0)
        label = torch.cat([torch.ones(x.shape[0], device=device), torch.zeros(x.shape[0], device=device)])
        y_ref = torch.cat([y, y], dim=0)
        target_rep = torch.cat([x, x], dim=0)
        u_images.append(u.detach().cpu())
        anchors.append(anchor_rep.detach().cpu())
        labels.append(label.detach().cpu())
        y_refs.append(y_ref.detach().cpu())
        clean_targets.append(target_rep.detach().cpu())
    return {
        "u": torch.cat(u_images, dim=0),
        "anchor": torch.cat(anchors, dim=0),
        "label": torch.cat(labels, dim=0),
        "y": torch.cat(y_refs, dim=0),
        "target": torch.cat(clean_targets, dim=0),
    }


class BlindCriticSmall(nn.Module):
    def __init__(self, in_channels: int = 2, base: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, stride=2, padding=1),
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


class ProjectionConditionedCritic(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        dim = base * 4
        self.null_head = nn.Linear(dim, dim)
        self.anchor_head = nn.Linear(dim, dim)
        self.bias = nn.Sequential(nn.Linear(dim * 2, dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        null_img, anchor = x[:, :1], x[:, 1:2]
        zn = F.normalize(self.null_head(self.encoder(null_img).flatten(1)), dim=1)
        za = F.normalize(self.anchor_head(self.encoder(anchor).flatten(1)), dim=1)
        return (zn * za).sum(dim=1) + self.bias(torch.cat([zn, za], dim=1)).squeeze(1)


class FullShortcutCritic(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(96, 48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_image_inputs(dataset: dict[str, torch.Tensor], measurement, kind: str, device: torch.device) -> torch.Tensor:
    u = dataset["u"].to(device)
    anchor = dataset["anchor"].to(device)
    if kind == "blind":
        null_img = null_component(measurement, u)
        return torch.cat([null_img, anchor], dim=1)
    if kind == "image_cond":
        return torch.cat([u, anchor], dim=1)
    raise ValueError(kind)


def make_full_shortcut_inputs(dataset: dict[str, torch.Tensor], measurement, device: torch.device) -> torch.Tensor:
    u = dataset["u"].to(device)
    anchor = dataset["anchor"].to(device)
    y = dataset["y"].to(device)
    u_flat = flatten_batch(measurement, u)
    anchor_flat = flatten_batch(measurement, anchor)
    residual = measurement.A_forward(u_flat) - y.float()
    delta = measurement.AT_forward(measurement.solve_K(residual.float()))
    rel = relmeas_tensor(measurement, u_flat, y).unsqueeze(1)
    feats = [
        u_flat.mean(dim=1, keepdim=True),
        u_flat.std(dim=1, keepdim=True),
        anchor_flat.mean(dim=1, keepdim=True),
        anchor_flat.std(dim=1, keepdim=True),
        residual.mean(dim=1, keepdim=True),
        residual.std(dim=1, keepdim=True),
        torch.linalg.norm(residual, dim=1, keepdim=True),
        rel,
        delta.mean(dim=1, keepdim=True),
        delta.std(dim=1, keepdim=True),
        torch.linalg.norm(delta, dim=1, keepdim=True),
    ]
    return torch.cat(feats, dim=1)


def binary_metrics(labels: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    labels = labels.detach().float().cpu()
    scores = scores.detach().float().cpu()
    probs = torch.sigmoid(scores)
    pred = (probs >= 0.5).float()
    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    tn = ((pred == 0) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    acc = (tp + tn) / max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    brier = torch.mean((probs - labels) ** 2).item()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        compare = (pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()
        auc = compare.mean().item()
    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "brier": float(brier),
    }


def train_binary_critic(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float], torch.Tensor, torch.Tensor]:
    set_seed(123)
    inputs = inputs.to(device)
    labels = labels.float().to(device)
    n = labels.numel()
    perm = torch.randperm(n, device=device)
    split = max(2, int(0.75 * n))
    train_idx, val_idx = perm[:split], perm[split:]
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    batch = min(64, split)
    for _epoch in range(int(epochs)):
        train_idx = train_idx[torch.randperm(train_idx.numel(), device=device)]
        for start in range(0, train_idx.numel(), batch):
            idx = train_idx[start : start + batch]
            logit = model(inputs[idx])
            loss = F.binary_cross_entropy_with_logits(logit, labels[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        val_scores = model(inputs[val_idx])
    metrics = binary_metrics(labels[val_idx], val_scores)
    return model, metrics, labels[val_idx].detach().cpu(), val_scores.detach().cpu()


def eval_binary_critic(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        scores = model(inputs.to(device)).detach().cpu()
    return binary_metrics(labels.float().cpu(), scores)


def save_score_histogram(path: Path, labels: torch.Tensor, scores: torch.Tensor, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = labels.float().cpu()
    scores = scores.float().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(scores[labels == 1].numpy(), bins=30, alpha=0.65, label="positive")
    plt.hist(scores[labels == 0].numpy(), bins=30, alpha=0.65, label="negative")
    plt.title(title)
    plt.xlabel("critic logit")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_bar_plot(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get(x_key, i)) for i, row in enumerate(rows)]
    values = []
    for row in rows:
        value = row.get(y_key, float("nan"))
        try:
            values.append(float(value))
        except Exception:
            values.append(float("nan"))
    plt.figure(figsize=(max(6, 0.6 * len(labels)), 4))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_image_grid(path: Path, rows: list[list[torch.Tensor]], titles: list[str], max_rows: int = 6) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = rows[:max_rows]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nrows, ncols = len(rows), len(titles)
    plt.figure(figsize=(2.2 * ncols, 2.1 * nrows))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = plt.subplot(nrows, ncols, r * ncols + c + 1)
            arr = img.detach().float().cpu().squeeze().numpy()
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def copy_checkpoint_for_manifest(info: dict[str, Any], output_dir: Path) -> None:
    shutil.copy2(info["checkpoint_path"], output_dir / "source_checkpoint.pt")
    if info.get("exact_A_path") is not None:
        shutil.copy2(info["exact_A_path"], output_dir / "measurement_operator_exact.pt")


def summarize_metric_rows(rows: list[dict[str, Any]], group_key: str, metric_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        key = str(row.get(group_key, ""))
        value = row.get(metric_key)
        try:
            val = float(value)
        except Exception:
            continue
        grouped.setdefault(key, []).append(val)
    return [
        {group_key: key, f"mean_{metric_key}": sum(vals) / max(1, len(vals)), "n": len(vals)}
        for key, vals in sorted(grouped.items())
    ]
