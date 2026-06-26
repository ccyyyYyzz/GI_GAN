from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
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

from . import phase69A_gauge_gan_signal_diagnostic as p69a
from .eval import make_measurement
from .models import build_generator
from .split_guard import (
    assert_train_loader_disjoint_from_test,
    collect_sample_identities,
)
from .utils import apply_experiment_defaults, load_config, set_seed


DATA_ROOT = Path("E:/ns_mc_gan_gi")
OUT_DIR = DATA_ROOT / "outputs_phase69B_controlled_gauge_cgan_pilot"
PHASE69A_DIR = DATA_ROOT / "outputs_phase69A_gauge_gan_signal_diagnostic"
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(p69a.json_safe(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    lines = [
        "| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |",
        "| " + " | ".join("-" * widths[c] for c in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns) + " |")
    return "\n".join(lines)


def unsafe_stop(out_dir: Path, failures: list[str], warnings: list[str] | None = None) -> int:
    lines = [
        "# UNSAFE TO RUN",
        "",
        "Phase69B stopped before supervised/cGAN fine-tuning.",
        "",
        "## Critical Failures",
        "",
    ]
    lines.extend(f"- {item}" for item in failures)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in warnings)
    lines.extend(["", "No generator/reconstruction network training was run."])
    write_text(out_dir / "UNSAFE_TO_RUN.md", "\n".join(lines) + "\n")
    append_log(out_dir, "unsafe_stop")
    return 2


class CachedDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor):
        self.x = x.float()
        self.y = y.float()
        self.labels = labels.long()
        self.indices = indices.long()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.labels[idx], self.indices[idx]


@dataclass
class SplitCache:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor

    def dataset(self) -> CachedDataset:
        return CachedDataset(self.x, self.y, self.labels, self.indices)


def load_phase69a_summary() -> dict[str, Any]:
    summary = {
        "manifest_exists": (PHASE69A_DIR / "gauge_dataset_manifest.json").exists(),
        "go_nogo_exists": (PHASE69A_DIR / "PHASE69A_GO_NOGO.md").exists(),
        "critic_auc_exists": (PHASE69A_DIR / "critic_auc_results.csv").exists(),
    }
    if summary["go_nogo_exists"]:
        summary["go_nogo_text"] = (PHASE69A_DIR / "PHASE69A_GO_NOGO.md").read_text(encoding="utf-8")
    return summary


def output_dir_is_clean(path: Path) -> bool:
    if not path.exists():
        return True
    return not any(path.iterdir())


def output_dir_has_only_runner_files(path: Path) -> bool:
    if not path.exists():
        return True
    allowed = {"RUNLOG.md", "phase69B_stdout.log", "phase69B_stderr.log"}
    return all(item.name in allowed for item in path.iterdir())


def import_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def make_config(device: str, batch_size: int) -> dict[str, Any]:
    config = load_config(p69a.RESOLVED_CONFIG)
    config = apply_experiment_defaults(config)
    if isinstance(config.get("device"), str):
        config["device"] = device
    config["dataset_root"] = str(DATA_ROOT / "data")
    config["output_dir"] = str(OUT_DIR)
    config["num_workers"] = 0
    config["batch_size"] = int(batch_size)
    config["use_augmentation"] = False
    config["use_final_dc_project"] = True
    config["output_range_mode"] = "clamp_eval_only"
    return apply_experiment_defaults(config)


def load_generator_from_checkpoint(config: dict[str, Any], measurement, device: torch.device):
    checkpoint = torch.load(p69a.CHECKPOINT, map_location=device, weights_only=False)
    merged = dict(config)
    if isinstance(checkpoint, dict) and checkpoint.get("config"):
        merged.update(checkpoint["config"])
    merged["dataset_root"] = str(DATA_ROOT / "data")
    merged["output_dir"] = str(OUT_DIR)
    merged["device"] = str(device)
    merged["num_workers"] = 0
    merged["batch_size"] = int(config["batch_size"])
    merged["use_augmentation"] = False
    merged = apply_experiment_defaults(merged)
    generator = build_generator(merged, measurement=measurement).to(device)
    state = checkpoint.get("generator_ema") or checkpoint.get("generator")
    if state is None:
        raise RuntimeError("Published checkpoint has no generator/generator_ema state.")
    generator.load_state_dict(state)
    generator.train()
    return generator, merged


def A_forward(v: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return v @ A.T


def AT_forward(y: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return y @ A


def p0_ortho(v: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return v - AT_forward(A_forward(v, A), A)


def blambda_ortho(y: torch.Tensor, A: torch.Tensor, lambda_dc: float) -> torch.Tensor:
    return AT_forward(y, A) / (1.0 + float(lambda_dc))


def pi_lambda_ortho(v: torch.Tensor, y: torch.Tensor, A: torch.Tensor, lambda_dc: float) -> torch.Tensor:
    return v - AT_forward(A_forward(v, A) - y, A) / (1.0 + float(lambda_dc))


def flatten_img(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def unflatten(v: torch.Tensor) -> torch.Tensor:
    return v.reshape(v.shape[0], 1, 64, 64)


def charbonnier(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((a - b) ** 2 + eps**2).mean()


def forward_candidate(generator, x: torch.Tensor, y: torch.Tensor, A: torch.Tensor, lambda_dc: float, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    x_flat = flatten_img(x)
    x_data_flat = AT_forward(y, A)
    x_data = unflatten(x_data_flat)
    zero_noise = torch.zeros_like(x_data)
    residual = generator(x_data, zero_noise, y=y)
    residual_flat = flatten_img(residual.float())
    residual_ns = p0_ortho(residual_flat, A) if bool(config.get("use_null_project", True)) else residual_flat
    v_stage0 = x_data_flat + residual_ns
    x_stage1 = pi_lambda_ortho(v_stage0, y, A, lambda_dc) if bool(config.get("use_dc_project", True)) else v_stage0
    if hasattr(generator, "refine"):
        refine = generator.refine(x_data, unflatten(x_stage1))
        v_pre = x_stage1 + flatten_img(refine.float())
    else:
        v_pre = x_stage1
    x_hat_flat = (
        pi_lambda_ortho(v_pre, y, A, lambda_dc)
        if bool(config.get("use_final_dc_project", True))
        else v_pre
    )
    b = blambda_ortho(y, A, lambda_dc)
    fake_gauge = p0_ortho(v_pre, A) + b
    real_gauge = p0_ortho(x_flat, A) + b
    return {
        "x_data_flat": x_data_flat,
        "v_pre": v_pre,
        "x_hat_flat": x_hat_flat,
        "x_hat": unflatten(x_hat_flat),
        "fake_gauge": unflatten(fake_gauge),
        "real_gauge": unflatten(real_gauge),
        "correction_flat": x_hat_flat - v_pre,
    }


def source_subset(dataset, indices: np.ndarray) -> Subset:
    return Subset(dataset, [int(i) for i in indices.tolist()])


@torch.no_grad()
def build_split_cache(
    name: str,
    base_dataset,
    indices: np.ndarray,
    measurement,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> SplitCache:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    subset = source_subset(base_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    seen_indices: list[torch.Tensor] = []
    offset = 0
    for batch in loader:
        x, label = batch
        x = x.to(device)
        y = measurement.measure(x)
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
        labels.append(torch.as_tensor(label).long())
        bsz = int(x.shape[0])
        seen_indices.append(torch.from_numpy(indices[offset : offset + bsz].astype(np.int64)))
        offset += bsz
    return SplitCache(
        name=name,
        x=torch.cat(xs, dim=0),
        y=torch.cat(ys, dim=0),
        labels=torch.cat(labels, dim=0),
        indices=torch.cat(seen_indices, dim=0),
    )


def build_caches(config: dict[str, Any], measurement, device: torch.device, train_count: int, val_count: int, test_count: int) -> tuple[SplitCache, SplitCache, SplitCache, dict[str, Any]]:
    train_indices_full = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    eval_indices_full = np.load(p69a.SPLIT_EVAL).astype(np.int64)
    train_indices = train_indices_full[:train_count]
    val_indices = train_indices_full[train_count : train_count + val_count]
    test_indices = eval_indices_full[:test_count]

    base_train = p69a.stl10_dataset("train+unlabeled")
    guard_loader = DataLoader(source_subset(base_train, train_indices), batch_size=config["batch_size"], shuffle=False, num_workers=0)
    train_guard = assert_train_loader_disjoint_from_test(guard_loader, context="Phase69B train cache source")
    train_ids = collect_sample_identities(source_subset(base_train, train_indices))
    val_ids = collect_sample_identities(source_subset(base_train, val_indices))
    if train_ids & val_ids:
        raise RuntimeError("Phase69B train and val partitions overlap.")

    train = build_split_cache("train", base_train, train_indices, measurement, device, config["batch_size"], seed=69021)
    val = build_split_cache("val", base_train, val_indices, measurement, device, config["batch_size"], seed=69022)

    z = np.load(p69a.EVAL_CACHE, allow_pickle=False)
    test = SplitCache(
        name="test",
        x=torch.from_numpy(z["x"][:test_count].reshape(test_count, 1, 64, 64)).float(),
        y=torch.from_numpy(z["y"][:test_count]).float(),
        labels=torch.from_numpy(z["labels"][:test_count]).long(),
        indices=torch.from_numpy(test_indices).long(),
    )
    split_info = {
        "train_count": int(train_count),
        "val_count": int(val_count),
        "test_count": int(test_count),
        "train_source": "STL10 train+unlabeled partition",
        "val_source": "held-out slice of STL10 train+unlabeled, not used for training",
        "test_source": "frozen cert cache main_scr5 / official STL10 test subset",
        "train_full_sorted_sha256": p69a.sha256_np(train_indices_full, sort_int64=True),
        "eval_full_sorted_sha256": p69a.sha256_np(eval_indices_full, sort_int64=True),
        "train_indices_sha256": p69a.sha256_np(train_indices),
        "val_indices_sha256": p69a.sha256_np(val_indices),
        "test_indices_sha256": p69a.sha256_np(test_indices),
        "train_val_overlap": 0,
        "train_guard": train_guard,
    }
    return train, val, test, split_info


def make_loader(cache: SplitCache, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        cache.dataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(int(seed)),
        num_workers=0,
        drop_last=False,
    )


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate_val_loss(generator, cache: SplitCache, A: torch.Tensor, lambda_dc: float, config: dict[str, Any], device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    generator.eval()
    loader = make_loader(cache, int(config["batch_size"]), shuffle=False, seed=1)
    losses = []
    rels = []
    for idx, (x, y, _, _) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        out = forward_candidate(generator, x, y, A, lambda_dc, config)
        losses.append(float(charbonnier(out["x_hat"], x).detach().cpu()))
        rel = relmeas_batch(out["x_hat_flat"], y, A)
        rels.extend(rel.tolist())
        if max_batches is not None and idx + 1 >= max_batches:
            break
    generator.train()
    return {"val_rec_loss": float(np.mean(losses)), "val_relmeas": float(np.mean(rels))}


def relmeas_batch(x_hat_flat: torch.Tensor, y: torch.Tensor, A: torch.Tensor) -> np.ndarray:
    A64 = A.detach().to(torch.float64)
    pred = x_hat_flat.detach().to(torch.float64) @ A64.T
    y64 = y.detach().to(torch.float64)
    rel = torch.linalg.norm(pred - y64, dim=1) / torch.linalg.norm(y64, dim=1).clamp_min(1e-12)
    return rel.detach().cpu().numpy()


def d_accuracy(real_score: torch.Tensor, fake_score: torch.Tensor) -> float:
    return float(0.5 * ((real_score.detach() > 0).float().mean() + (fake_score.detach() < 0).float().mean()).cpu())


def train_calibration_critic(
    generator,
    cache: SplitCache,
    A: torch.Tensor,
    lambda_dc: float,
    config: dict[str, Any],
    device: torch.device,
    steps: int = 20,
) -> nn.Module:
    critic = p69a.PatchCritic(1).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9))
    loader = cycle_loader(make_loader(cache, int(config["batch_size"]), shuffle=True, seed=69031))
    generator.eval()
    for _ in range(int(steps)):
        x, y, _, _ = next(loader)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = forward_candidate(generator, x, y, A, lambda_dc, config)
        real = out["real_gauge"]
        fake = out["fake_gauge"]
        opt.zero_grad(set_to_none=True)
        real_score = critic(real)
        fake_score = critic(fake)
        d_loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
        d_loss.backward()
        opt.step()
    return critic


def beta_calibration(generator, train: SplitCache, A: torch.Tensor, lambda_dc: float, config: dict[str, Any], device: torch.device, out_dir: Path) -> tuple[float, list[dict[str, Any]]]:
    append_log(out_dir, "beta_calibration_start")
    critic = train_calibration_critic(generator, train, A, lambda_dc, config, device, steps=20)
    loader = make_loader(train, int(config["batch_size"]), shuffle=False, seed=69032)
    x, y, _, _ = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        base = forward_candidate(generator, x, y, A, lambda_dc, config)
    v = base["v_pre"].detach().requires_grad_(True)
    x_hat = unflatten(pi_lambda_ortho(v, y, A, lambda_dc))
    fake_gauge = unflatten(p0_ortho(v, A) + blambda_ortho(y, A, lambda_dc))
    rec = charbonnier(x_hat, x)
    adv = -critic(fake_gauge).mean()
    grad_rec = torch.autograd.grad(rec, v, retain_graph=True)[0]
    grad_adv = torch.autograd.grad(adv, v)[0]
    rec_norm = float(torch.linalg.norm(grad_rec).detach().cpu())
    adv_norm = float(torch.linalg.norm(grad_adv).detach().cpu())
    ratio = adv_norm / max(rec_norm, 1e-12)
    target = 0.075
    beta0 = float(target / max(ratio, 1e-12))
    beta0 = float(np.clip(beta0, 1e-5, 1.0))
    rows = [
        {
            "grad_rec_norm": rec_norm,
            "grad_adv_norm": adv_norm,
            "adv_to_rec_ratio": ratio,
            "target_beta_times_ratio": target,
            "selected_beta0": beta0,
            "candidate_0p3_beta0": 0.3 * beta0,
            "candidate_beta0": beta0,
            "candidate_3_beta0": 3.0 * beta0,
            "candidate_sweep_run": False,
            "note": "Runtime kept to selected beta0 only after smoke; candidates recorded but not swept.",
        }
    ]
    append_log(out_dir, f"beta_calibration_complete beta0={beta0:.6g} ratio={ratio:.6g}")
    return beta0, rows


def save_checkpoint(path: Path, arm: str, step: int, generator, optimizer_g, config: dict[str, Any], metrics: dict[str, Any] | None = None, critic=None, optimizer_d=None, beta: float | None = None) -> None:
    ensure_dir(path.parent)
    payload = {
        "phase": "Phase69B",
        "arm": arm,
        "step": int(step),
        "generator": generator.state_dict(),
        "optimizer_g": optimizer_g.state_dict() if optimizer_g is not None else None,
        "config": config,
        "metrics": metrics or {},
        "source_checkpoint": str(p69a.CHECKPOINT),
        "source_checkpoint_sha256": p69a.sha256_file(p69a.CHECKPOINT),
        "beta": beta,
    }
    if critic is not None:
        payload["critic"] = critic.state_dict()
    if optimizer_d is not None:
        payload["optimizer_d"] = optimizer_d.state_dict()
    torch.save(payload, path)


def train_arm(
    arm: str,
    generator,
    train: SplitCache,
    val: SplitCache,
    A: torch.Tensor,
    lambda_dc: float,
    config: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    steps: int,
    beta: float = 0.0,
    adversarial: bool = False,
    eval_every: int = 100,
    save_every: int = 200,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    append_log(out_dir, f"train_arm_start arm={arm} steps={steps} adversarial={adversarial} beta={beta:.6g}")
    arm_dir = ensure_dir(out_dir / arm)
    ckpt_dir = ensure_dir(arm_dir / "checkpoints")
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-5, betas=(0.9, 0.999))
    critic = p69a.PatchCritic(1).to(device) if adversarial else None
    opt_d = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9)) if critic is not None else None
    loader = cycle_loader(make_loader(train, int(config["batch_size"]), shuffle=True, seed=69040 + (1 if adversarial else 0)))
    log_rows: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = ckpt_dir / "best_by_val.pt"
    final_metrics: dict[str, Any] = {}
    finite_ok = True
    d_acc_history: list[float] = []

    for step in range(1, int(steps) + 1):
        x, y, _, _ = next(loader)
        x = x.to(device)
        y = y.to(device)

        d_loss_value = float("nan")
        d_acc_value = float("nan")
        if adversarial and critic is not None and opt_d is not None:
            generator.eval()
            critic.train()
            with torch.no_grad():
                out_d = forward_candidate(generator, x, y, A, lambda_dc, config)
            real_g = out_d["real_gauge"]
            fake_g = out_d["fake_gauge"]
            opt_d.zero_grad(set_to_none=True)
            real_score = critic(real_g)
            fake_score = critic(fake_g)
            d_loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
            d_loss.backward()
            opt_d.step()
            d_loss_value = float(d_loss.detach().cpu())
            d_acc_value = d_accuracy(real_score, fake_score)
            d_acc_history.append(d_acc_value)
            generator.train()

        opt_g.zero_grad(set_to_none=True)
        out = forward_candidate(generator, x, y, A, lambda_dc, config)
        rec_loss = charbonnier(out["x_hat"], x)
        adv_loss = torch.zeros((), device=device)
        if adversarial and critic is not None:
            critic.eval()
            adv_loss = -critic(out["fake_gauge"]).mean()
        loss = rec_loss + float(beta) * adv_loss
        if not torch.isfinite(loss):
            finite_ok = False
            raise RuntimeError(f"Non-finite loss in {arm} at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()

        row = {
            "arm": arm,
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            "loss_rec": float(rec_loss.detach().cpu()),
            "loss_adv": float(adv_loss.detach().cpu()),
            "loss_d": d_loss_value,
            "d_accuracy": d_acc_value,
            "beta": float(beta),
        }
        if step % int(eval_every) == 0 or step == int(steps):
            val_metrics = evaluate_val_loss(generator, val, A, lambda_dc, config, device, max_batches=None)
            row.update(val_metrics)
            if val_metrics["val_rec_loss"] < best_val:
                best_val = val_metrics["val_rec_loss"]
                save_checkpoint(best_path, arm, step, generator, opt_g, config, metrics=val_metrics, critic=critic, optimizer_d=opt_d, beta=beta)
            final_metrics = val_metrics
            append_log(out_dir, f"train_arm_eval arm={arm} step={step} val_rec={val_metrics['val_rec_loss']:.6g}")
        if step % int(save_every) == 0 or step == int(steps):
            save_checkpoint(ckpt_dir / f"step_{step:05d}.pt", arm, step, generator, opt_g, config, metrics=row, critic=critic, optimizer_d=opt_d, beta=beta)
        log_rows.append(row)

    final_path = ckpt_dir / "final.pt"
    save_checkpoint(final_path, arm, int(steps), generator, opt_g, config, metrics=final_metrics, critic=critic, optimizer_d=opt_d, beta=beta)
    if not best_path.exists():
        save_checkpoint(best_path, arm, int(steps), generator, opt_g, config, metrics=final_metrics, critic=critic, optimizer_d=opt_d, beta=beta)
    summary = {
        "arm": arm,
        "steps": int(steps),
        "finite_losses": finite_ok,
        "best_val_rec_loss": best_val,
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "d_accuracy_last_mean": float(np.nanmean(d_acc_history[-50:])) if d_acc_history else float("nan"),
        "d_saturated_last_mean_gt_0p95": bool(np.nanmean(d_acc_history[-50:]) > 0.95) if d_acc_history else False,
    }
    write_csv(arm_dir / "training_log.csv", log_rows)
    save_json(arm_dir / "training_summary.json", summary)
    append_log(out_dir, f"train_arm_complete arm={arm} best_val={best_val:.6g}")
    return summary, log_rows, best_path


def load_generator_checkpoint_for_eval(path: Path, config: dict[str, Any], measurement, device: torch.device):
    generator = build_generator(config, measurement=measurement).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(payload["generator"])
    generator.eval()
    return generator


def psnr_one(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 0:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def ssim_one(pred: np.ndarray, target: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(target, pred, data_range=1.0))
    except Exception:
        return float("nan")


def grad_mag(img: np.ndarray) -> np.ndarray:
    gx = np.diff(img, axis=1, append=img[:, -1:])
    gy = np.diff(img, axis=0, append=img[-1:, :])
    return np.sqrt(gx * gx + gy * gy)


def hf_ratio(img: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(f) ** 2
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    mask = rr > min(h, w) * 0.25
    return float(power[mask].sum() / max(power.sum(), 1e-12))


def rapsd(img: np.ndarray, bins: int = 32) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(f) ** 2
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    max_r = rr.max()
    edges = np.linspace(0, max_r + 1e-6, bins + 1)
    prof = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        mask = (rr >= edges[i]) & (rr < edges[i + 1])
        prof[i] = float(power[mask].mean()) if np.any(mask) else 0.0
    total = prof.sum()
    return prof / max(total, 1e-12)


def metric_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.nanmean(arr)), "median": float(np.nanmedian(arr)), "std": float(np.nanstd(arr))}


@torch.no_grad()
def evaluate_arm(
    arm: str,
    generator,
    test: SplitCache,
    A: torch.Tensor,
    lambda_dc: float,
    config: dict[str, Any],
    device: torch.device,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    generator.eval()
    loader = make_loader(test, int(config["batch_size"]), shuffle=False, seed=7)
    per_rows: list[dict[str, Any]] = []
    outputs: list[np.ndarray] = []
    A64 = A.detach().to(torch.float64)
    p0_l2_vals: list[float] = []
    for batch_idx, (x, y, labels, indices) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        out = forward_candidate(generator, x, y, A, lambda_dc, config)
        x_hat_flat = out["x_hat_flat"]
        x_hat = out["x_hat"].detach().cpu().numpy()[:, 0]
        x_clip = np.clip(x_hat, 0.0, 1.0)
        x_true = x.detach().cpu().numpy()[:, 0]
        y_np = y.detach().cpu().numpy()
        rels = relmeas_batch(x_hat_flat, y, A)
        corr = torch.linalg.norm(out["correction_flat"].detach(), dim=1) / torch.linalg.norm(out["v_pre"].detach(), dim=1).clamp_min(1e-12)
        p0_pred = p0_ortho(x_hat_flat.detach().to(torch.float64), A64)
        p0_true = p0_ortho(flatten_img(x).detach().to(torch.float64), A64)
        p0_l2 = (torch.linalg.norm(p0_pred - p0_true, dim=1) / math.sqrt(4096)).detach().cpu().numpy()
        outputs.append(x_hat.astype(np.float32))
        for i in range(x.shape[0]):
            pred = x_clip[i]
            true = x_true[i]
            rdist = float(np.linalg.norm(rapsd(pred) - rapsd(true)))
            gerr = float(abs(grad_mag(pred).mean() - grad_mag(true).mean()))
            hferr = float(abs(hf_ratio(pred) - hf_ratio(true)))
            row = {
                "arm": arm,
                "sample_index": int(indices[i]),
                "label": int(labels[i]),
                "psnr": psnr_one(pred, true),
                "ssim": ssim_one(pred, true),
                "relmeaserr_unclipped_float64": float(rels[i]),
                "correction_norm_rel": float(corr[i].detach().cpu()),
                "rapsd_distance": rdist,
                "gradient_mean_abs_error": gerr,
                "highfreq_ratio_abs_error": hferr,
                "p0_l2": float(p0_l2[i]),
            }
            per_rows.append(row)
            p0_l2_vals.append(float(p0_l2[i]))
    out_arr = np.concatenate(outputs, axis=0)
    aggregate: dict[str, Any] = {"arm": arm, "n": int(out_arr.shape[0])}
    for metric in [
        "psnr",
        "ssim",
        "relmeaserr_unclipped_float64",
        "correction_norm_rel",
        "rapsd_distance",
        "gradient_mean_abs_error",
        "highfreq_ratio_abs_error",
        "p0_l2",
    ]:
        vals = [float(r[metric]) for r in per_rows]
        stats = metric_summary(vals)
        aggregate[f"{metric}_mean"] = stats["mean"]
        aggregate[f"{metric}_median"] = stats["median"]
        aggregate[f"{metric}_std"] = stats["std"]
    np.savez_compressed(out_dir / f"per_sample_outputs_{arm}.npz", x_hat_unclipped=out_arr.astype(np.float16))
    return aggregate, per_rows, out_arr


def save_visual_grid(out_dir: Path, test: SplitCache, outputs: dict[str, np.ndarray], n: int = 8) -> None:
    n = min(n, test.x.shape[0])
    arms = ["A", "B", "C"]
    fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
    for i in range(n):
        imgs = [test.x[i, 0].numpy(), *[np.clip(outputs[a][i], 0, 1) for a in arms]]
        titles = ["GT", "Arm A", "Arm B", "Arm C"]
        for j, (img, title) in enumerate(zip(imgs, titles)):
            ax = axes[i, j] if n > 1 else axes[j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "visual_grid_A_B_C.png", dpi=160)
    fig.savefig(out_dir / "visual_grid_A_B_C.pdf")
    plt.close(fig)


def save_rapsd_plot(out_dir: Path, test: SplitCache, outputs: dict[str, np.ndarray]) -> None:
    true_profiles = np.stack([rapsd(test.x[i, 0].numpy()) for i in range(test.x.shape[0])])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(true_profiles.mean(axis=0), label="GT", linewidth=2)
    for arm, arr in outputs.items():
        prof = np.stack([rapsd(np.clip(arr[i], 0, 1)) for i in range(arr.shape[0])])
        ax.plot(prof.mean(axis=0), label=f"Arm {arm}")
    ax.set_xlabel("radial frequency bin")
    ax.set_ylabel("normalized power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rapsd_comparison.png", dpi=160)
    fig.savefig(out_dir / "rapsd_comparison.pdf")
    plt.close(fig)


def bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 1000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    idx = np.arange(values.shape[0])
    for _ in range(n_boot):
        sample = rng.choice(idx, size=idx.shape[0], replace=True)
        means.append(float(np.nanmean(values[sample])))
    return float(np.nanmean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), {})[int(row["sample_index"])] = row
    common = sorted(set(by_arm.get("B", {})) & set(by_arm.get("C", {})))
    metrics = [
        ("psnr", "higher"),
        ("ssim", "higher"),
        ("relmeaserr_unclipped_float64", "lower"),
        ("correction_norm_rel", "lower"),
        ("rapsd_distance", "lower"),
        ("gradient_mean_abs_error", "lower"),
        ("highfreq_ratio_abs_error", "lower"),
        ("p0_l2", "lower"),
    ]
    out = []
    for metric, direction in metrics:
        b = np.asarray([float(by_arm["B"][i][metric]) for i in common])
        c = np.asarray([float(by_arm["C"][i][metric]) for i in common])
        delta = c - b
        improvement = delta if direction == "higher" else -delta
        mean, lo, hi = bootstrap_ci(improvement, seed=69050 + len(out))
        out.append(
            {
                "metric": metric,
                "direction": direction,
                "mean_B": float(np.nanmean(b)),
                "mean_C": float(np.nanmean(c)),
                "mean_C_minus_B": float(np.nanmean(delta)),
                "improvement_positive_means_C_better": mean,
                "ci_low": lo,
                "ci_high": hi,
                "ci_excludes_zero_in_favor_of_C": bool(lo > 0),
            }
        )
    return out


def availability_rows() -> list[dict[str, Any]]:
    checks = [
        ("LPIPS", "lpips"),
        ("DISTS", "DISTS_pytorch"),
        ("torchmetrics", "torchmetrics"),
        ("clean_fid", "cleanfid"),
        ("skimage", "skimage"),
    ]
    return [{"metric_package": name, "module": module, "available": import_available(module)} for name, module in checks]


def write_beta_report(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    row = rows[0]
    lines = [
        "# Beta Calibration",
        "",
        f"- `||grad_rec||`: `{row['grad_rec_norm']}`",
        f"- `||grad_adv||`: `{row['grad_adv_norm']}`",
        f"- ratio `||grad_adv||/||grad_rec||`: `{row['adv_to_rec_ratio']}`",
        f"- selected beta0: `{row['selected_beta0']}`",
        "",
        "Candidates were recorded as `0.3 beta0`, `beta0`, and `3 beta0`. Runtime was kept to the selected beta0 arm only; no beta sweep was run.",
        "",
    ]
    write_text(out_dir / "BETA_CALIBRATION_REPORT.md", "\n".join(lines))


def plot_smoke(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, arm_rows in by_arm.items():
        axes[0].plot([r["step"] for r in arm_rows], [r["loss_rec"] for r in arm_rows], label=arm)
        d_rows = [r for r in arm_rows if not math.isnan(float(r.get("d_accuracy", float("nan"))))]
        if d_rows:
            axes[1].plot([r["step"] for r in d_rows], [r["d_accuracy"] for r in d_rows], label=arm)
    axes[0].set_title("Smoke rec loss")
    axes[0].set_xlabel("step")
    axes[0].legend()
    axes[1].set_title("Smoke D accuracy")
    axes[1].set_xlabel("step")
    axes[1].axhline(0.95, color="tab:red", linestyle="--", linewidth=1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_training_curves.png", dpi=160)
    fig.savefig(out_dir / "smoke_training_curves.pdf")
    plt.close(fig)


def plot_pilot_curves(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for arm in sorted(set(str(r["arm"]) for r in rows)):
        arm_rows = [r for r in rows if str(r["arm"]) == arm]
        axes[0].plot([r["step"] for r in arm_rows], [r["loss_rec"] for r in arm_rows], label=arm)
        val_rows = [r for r in arm_rows if r.get("val_rec_loss", "") != "" and not np.isnan(float(r.get("val_rec_loss", np.nan)))]
        if val_rows:
            axes[1].plot([r["step"] for r in val_rows], [r["val_rec_loss"] for r in val_rows], marker="o", label=arm)
        d_rows = [r for r in arm_rows if not math.isnan(float(r.get("d_accuracy", float("nan"))))]
        if d_rows:
            axes[2].plot([r["step"] for r in d_rows], [r["d_accuracy"] for r in d_rows], label=arm)
    axes[0].set_title("train rec loss")
    axes[1].set_title("val rec loss")
    axes[2].set_title("D accuracy")
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
    axes[2].axhline(0.95, color="tab:red", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_dir / "pilot_training_curves.png", dpi=160)
    fig.savefig(out_dir / "pilot_training_curves.pdf")
    plt.close(fig)


def preflight(out_dir: Path, args) -> tuple[bool, dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    checks["requested_project_path"] = str(p69a.REQUESTED_PROJECT)
    checks["requested_project_path_exists"] = p69a.REQUESTED_PROJECT.exists()
    if not p69a.REQUESTED_PROJECT.exists():
        warnings.append(f"Requested C: project path is missing; using actual repo mirror {REPO_ROOT}.")
    checks["output_directory"] = str(out_dir)
    checks["output_directory_clean_before_run"] = output_dir_has_only_runner_files(out_dir)
    if not checks["output_directory_clean_before_run"]:
        failures.append(f"Output directory is not clean: {out_dir}")
    phase69a = load_phase69a_summary()
    checks["phase69a"] = phase69a
    if not all([phase69a["manifest_exists"], phase69a["go_nogo_exists"], phase69a["critic_auc_exists"]]):
        failures.append("Phase69A manifest/reports are missing.")
    for name, path in [
        ("checkpoint", p69a.CHECKPOINT),
        ("resolved_config", p69a.RESOLVED_CONFIG),
        ("A_scr5", p69a.A_SCR5),
        ("split_train", p69a.SPLIT_TRAIN),
        ("split_eval", p69a.SPLIT_EVAL),
        ("eval_cache", p69a.EVAL_CACHE),
        ("provenance", p69a.PROVENANCE_JSON),
    ]:
        checks[f"{name}_path"] = str(path)
        checks[f"{name}_exists"] = path.exists()
        if not path.exists():
            failures.append(f"Missing required {name}: {path}")
    if failures:
        return False, checks, failures, warnings

    prov = read_json(p69a.PROVENANCE_JSON)
    train_idx = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    eval_idx = np.load(p69a.SPLIT_EVAL).astype(np.int64)
    checks["train_split_sorted_sha256"] = p69a.sha256_np(train_idx, sort_int64=True)
    checks["train_split_expected_sha256"] = prov["splits"]["train_indices_sha256_sorted_int64"]
    checks["test_split_sorted_sha256"] = p69a.sha256_np(eval_idx, sort_int64=True)
    checks["test_split_expected_sha256"] = prov["splits"]["eval_indices_sha256_sorted_int64"]
    if checks["train_split_sorted_sha256"] != checks["train_split_expected_sha256"]:
        failures.append("Train split hash mismatch.")
    if checks["test_split_sorted_sha256"] != checks["test_split_expected_sha256"]:
        failures.append("Test/eval split hash mismatch.")
    if args.train_count + args.val_count > len(train_idx):
        failures.append("Requested train+val count exceeds train split length.")
    if args.test_count > len(eval_idx):
        failures.append("Requested test count exceeds eval split length.")

    try:
        ckpt_sha = p69a.sha256_file(p69a.CHECKPOINT)
        manifest = read_json(p69a.EVAL_MANIFEST)
        checks["checkpoint_sha256"] = ckpt_sha
        checks["checkpoint_expected_sha256"] = manifest["checkpoint_sha256"]
        if ckpt_sha != manifest["checkpoint_sha256"]:
            failures.append("Checkpoint SHA mismatch.")
        ck = torch.load(p69a.CHECKPOINT, map_location="cpu", weights_only=False)
        checks["checkpoint_loadable"] = isinstance(ck, dict) and "generator" in ck
        checks["checkpoint_generator_ema"] = bool(isinstance(ck, dict) and ck.get("generator_ema") is not None)
        del ck
    except Exception as exc:
        failures.append(f"Checkpoint load failed: {exc}")

    try:
        A_np = np.load(p69a.A_SCR5).astype(np.float32)
        checks["A_shape"] = list(A_np.shape)
        checks["A_sha256_float32"] = p69a.sha256_np(A_np)
        G = A_np.astype(np.float64) @ A_np.astype(np.float64).T
        checks["AAT_minus_I_max"] = float(np.max(np.abs(G - np.eye(G.shape[0]))))
        if checks["AAT_minus_I_max"] > 1e-5:
            failures.append("Scr-5 A is not orthonormal.")
    except Exception as exc:
        failures.append(f"Exact A load failed: {exc}")

    checks["availability"] = availability_rows()
    try:
        ensure_dir(out_dir)
        probe = out_dir / "_probe.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks["per_sample_saving_available"] = True
    except Exception as exc:
        failures.append(f"Output/per-sample saving failed: {exc}")

    try:
        device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        config = make_config(str(device), args.batch_size)
        measurement = make_measurement(config, device)
        A_t = torch.from_numpy(np.load(p69a.A_SCR5).astype(np.float32)).to(device)
        measurement.set_A_override(A_t, metadata={"phase": "phase69B_preflight"}, rebuild_cache=True)
        generator, cfg = load_generator_from_checkpoint(config, measurement, device)
        z = np.load(p69a.EVAL_CACHE, allow_pickle=False)
        x = torch.from_numpy(z["x"][:2].reshape(2, 1, 64, 64)).to(device)
        y = torch.from_numpy(z["y"][:2]).to(device)
        out = forward_candidate(generator, x, y, measurement.A, float(cfg["lambda_solver"]), cfg)
        checks["P0_shape"] = list(p0_ortho(flatten_img(x), measurement.A).shape)
        checks["B_lambda_shape"] = list(blambda_ortho(y, measurement.A, float(cfg["lambda_solver"])).shape)
        checks["Pi_shape"] = list(out["x_hat_flat"].shape)
        checks["mean_mode_finite"] = bool(torch.isfinite(out["v_pre"]).all().item())
        if not checks["mean_mode_finite"]:
            failures.append("Mean-mode output contains non-finite values.")
        del generator, measurement, A_t, x, y, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        failures.append(f"P0/B_lambda/Pi/mean-mode preflight failed: {exc}")

    return len(failures) == 0, checks, failures, warnings


def write_preflight_report(out_dir: Path, checks: dict[str, Any], warnings: list[str]) -> None:
    rows = checks.get("availability", [])
    lines = [
        "# Phase69B Preflight Safety",
        "",
        f"- safe_to_run: `{checks.get('safe_to_run')}`",
        f"- output directory clean before run: `{checks.get('output_directory_clean_before_run')}`",
        f"- requested C path exists: `{checks.get('requested_project_path_exists')}`",
        f"- actual repo root: `{REPO_ROOT}`",
        f"- checkpoint SHA256: `{checks.get('checkpoint_sha256')}`",
        f"- train split sorted SHA256: `{checks.get('train_split_sorted_sha256')}`",
        f"- test split sorted SHA256: `{checks.get('test_split_sorted_sha256')}`",
        f"- AAT_minus_I_max: `{checks.get('AAT_minus_I_max')}`",
        "",
        "## Availability",
        "",
        format_table(rows, ["metric_package", "module", "available"]),
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {w}" for w in warnings] or ["- none"])
    write_text(out_dir / "PREFLIGHT_SAFETY.md", "\n".join(lines) + "\n")


def gauge_input_preflight(out_dir: Path, generator, val: SplitCache, A: torch.Tensor, lambda_dc: float, config: dict[str, Any], device: torch.device) -> None:
    loader = make_loader(val, int(config["batch_size"]), shuffle=False, seed=8)
    rows = []
    x, y, _, _ = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        out = forward_candidate(generator, x, y, A, lambda_dc, config)
    for kind in ["real_gauge", "fake_gauge"]:
        arr = out[kind].detach().cpu().numpy()
        rows.append(
            {
                "kind": kind,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "finite": bool(np.isfinite(arr).all()),
                "A_gauge_minus_ABlambda_rel_max": float("not_applicable" == "never") if False else "",
            }
        )
    write_csv(out_dir / "gauge_input_preflight.csv", rows)
    n = min(6, x.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
    for i in range(n):
        imgs = [x[i, 0].detach().cpu().numpy(), out["real_gauge"][i, 0].detach().cpu().numpy(), out["fake_gauge"][i, 0].detach().cpu().numpy()]
        titles = ["x", "real gauge", "fake gauge"]
        for j, (img, title) in enumerate(zip(imgs, titles)):
            ax = axes[i, j] if n > 1 else axes[j]
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "gauge_input_examples.png", dpi=160)
    fig.savefig(out_dir / "gauge_input_examples.pdf")
    plt.close(fig)
    lines = [
        "# Gauge Input Preflight",
        "",
        "Phase69B uses the unconditional gauge image input `D(tilde_x)` only.",
        "",
        "`tilde_x_real = P0 x + B_lambda y` and `tilde_x_fake = P0 v_theta + B_lambda y`.",
        "",
        "This gauge is residual-shortcut-free and row-equalized, but it is not claimed to be exactly feasible. Deployment output remains `Pi_y^lambda(v_theta)`.",
        "",
        format_table(rows, ["kind", "min", "max", "mean", "std", "finite"]),
        "",
    ]
    write_text(out_dir / "GAUGE_INPUT_PREFLIGHT.md", "\n".join(lines))


def smoke_pass(summary_b: dict[str, Any], summary_c: dict[str, Any], smoke_rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons = []
    for summary in [summary_b, summary_c]:
        if not summary["finite_losses"]:
            reasons.append(f"{summary['arm']} has non-finite losses.")
        if not Path(summary["final_checkpoint"]).exists():
            reasons.append(f"{summary['arm']} final checkpoint missing.")
    if summary_c.get("d_saturated_last_mean_gt_0p95"):
        reasons.append("Arm C smoke D accuracy saturated above 0.95.")
    val_rel = [float(r.get("val_relmeas", np.nan)) for r in smoke_rows if r.get("val_relmeas", "") != ""]
    if val_rel and max(val_rel) > 0.02:
        reasons.append("Smoke RelMeasErr after audit exceeded 0.02.")
    return len(reasons) == 0, reasons


def write_smoke_report(out_dir: Path, passed: bool, reasons: list[str], summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Smoke Report",
        "",
        f"- smoke_passed: `{passed}`",
        "",
        "## Arm Summaries",
        "",
        format_table(summaries, ["arm", "steps", "finite_losses", "best_val_rec_loss", "d_accuracy_last_mean", "d_saturated_last_mean_gt_0p95", "final_checkpoint"]),
        "",
        "## Failure Reasons",
        "",
    ]
    lines.extend([f"- {r}" for r in reasons] or ["- none"])
    write_text(out_dir / "SMOKE_REPORT.md", "\n".join(lines) + "\n")


def write_outputs(
    out_dir: Path,
    preflight_checks: dict[str, Any],
    split_info: dict[str, Any],
    beta_rows: list[dict[str, Any]],
    smoke_summaries: list[dict[str, Any]],
    pilot_summaries: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    decision: str,
    decision_reason: str,
) -> None:
    c_vs_b = {r["metric"]: r for r in comparison_rows}
    auc_note = "Phase69B did not use D as a certificate; Pi_y^lambda remains the measurement certificate."
    lines = [
        "# Phase69B Controlled Gauge-Equalized Audited cGAN Pilot",
        "",
        f"Output directory: `{out_dir}`",
        "",
        "This is a controlled project/supplement branch, not a change to the main paper results.",
        "",
        "## Preflight",
        "",
        f"- safe_to_run: `{preflight_checks.get('safe_to_run')}`",
        f"- train split SHA256: `{preflight_checks.get('train_split_sorted_sha256')}`",
        f"- test split SHA256: `{preflight_checks.get('test_split_sorted_sha256')}`",
        f"- checkpoint SHA256: `{preflight_checks.get('checkpoint_sha256')}`",
        "",
        "## Beta Calibration",
        "",
        format_table(beta_rows, ["adv_to_rec_ratio", "selected_beta0", "candidate_0p3_beta0", "candidate_beta0", "candidate_3_beta0", "candidate_sweep_run"]),
        "",
        "## Smoke",
        "",
        format_table(smoke_summaries, ["arm", "steps", "finite_losses", "best_val_rec_loss", "d_accuracy_last_mean", "d_saturated_last_mean_gt_0p95"]),
        "",
        "## Pilot Summaries",
        "",
        format_table(pilot_summaries, ["arm", "steps", "best_val_rec_loss", "best_checkpoint", "d_accuracy_last_mean", "d_saturated_last_mean_gt_0p95"]),
        "",
        "## Evaluation",
        "",
        format_table(eval_rows, ["arm", "psnr_mean", "ssim_mean", "relmeaserr_unclipped_float64_mean", "rapsd_distance_mean", "gradient_mean_abs_error_mean", "highfreq_ratio_abs_error_mean", "p0_l2_mean"]),
        "",
        "## C vs B",
        "",
        format_table(comparison_rows, ["metric", "direction", "mean_B", "mean_C", "improvement_positive_means_C_better", "ci_low", "ci_high", "ci_excludes_zero_in_favor_of_C"]),
        "",
        "## Decision",
        "",
        f"- decision: `{decision}`",
        f"- reason: {decision_reason}",
        f"- certificate note: {auc_note}",
        "",
        "No published checkpoint or main result was modified.",
        "",
    ]
    write_text(out_dir / "PHASE69B_CONTROLLED_CGAN_REPORT.md", "\n".join(lines))

    write_text(
        out_dir / "GAN_BRANCH_DECISION.md",
        "\n".join(
            [
                "# GAN Branch Decision",
                "",
                f"Decision: `{decision}`",
                "",
                decision_reason,
                "",
                "Phase69B does not authorize changing the title, abstract, or main result tables.",
                "",
            ]
        ),
    )
    write_text(
        out_dir / "PROJECT_REPORT_WORDING.md",
        "\n".join(
            [
                "# Project Report Wording",
                "",
                "We ran a controlled Scr-5 gauge-equalized audited cGAN pilot initialized from the published mean-mode checkpoint. The adversarial branch was evaluated only as a project/supplement diagnostic against a budget-matched supervised fine-tune, with measurement consistency still enforced by the audit operator.",
                "",
                f"Outcome: {decision}. {decision_reason}",
                "",
            ]
        ),
    )
    write_text(
        out_dir / "PAPER_WORDING_IF_SUCCESS.md",
        "\n".join(
            [
                "# Paper Wording If Success",
                "",
                "A small controlled supplement may mention that a gauge-equalized adversarial regularizer showed a pilot-level texture/perceptual signal under strict no-test-training controls. Do not place GAN in the title or abstract and do not alter the main reconstruction table.",
                "",
            ]
        ),
    )
    write_text(
        out_dir / "PAPER_WORDING_IF_FAIL.md",
        "\n".join(
            [
                "# Paper Wording If Inert Or Fail",
                "",
                "The adversarial branch was integrated and stress-tested, but under a controlled budget-matched Scr-5 pilot it did not provide a reliable gain beyond supervised fine-tuning. The main paper should omit GAN claims and keep the adversarial branch as project-history/post-mortem evidence only.",
                "",
            ]
        ),
    )


def write_manifest(out_dir: Path) -> None:
    files = [
        "PREFLIGHT_SAFETY.md",
        "preflight_checks.json",
        "split_manifest.json",
        "gauge_input_preflight.csv",
        "gauge_input_examples.png",
        "GAUGE_INPUT_PREFLIGHT.md",
        "beta_calibration.csv",
        "BETA_CALIBRATION_REPORT.md",
        "smoke_training_curves.png",
        "SMOKE_REPORT.md",
        "pilot_training_curves.png",
        "evaluation_metrics.csv",
        "per_sample_metrics.csv",
        "paired_comparison_C_vs_B.csv",
        "visual_grid_A_B_C.png",
        "rapsd_comparison.png",
        "lpips_or_dists_results.csv",
        "relmeaserr_certificate_table.csv",
        "P0_texture_metrics.csv",
        "PHASE69B_CONTROLLED_CGAN_REPORT.md",
        "GAN_BRANCH_DECISION.md",
        "PROJECT_REPORT_WORDING.md",
        "PAPER_WORDING_IF_SUCCESS.md",
        "PAPER_WORDING_IF_FAIL.md",
        "MANIFEST.md",
    ]
    lines = [
        "# Phase69B Manifest",
        "",
        f"Output directory: `{out_dir}`",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in files if (out_dir / name).exists())
    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            "- `smoke/armB/checkpoints/`",
            "- `smoke/armC/checkpoints/`",
            "- `pilot/armB/checkpoints/`",
            "- `pilot/armC/checkpoints/`",
            "",
            "## No Main-Result Change",
            "",
            "No published checkpoint, main table, or main paper result was modified.",
            "",
        ]
    )
    write_text(out_dir / "MANIFEST.md", "\n".join(lines))


def decide(eval_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], pilot_c: dict[str, Any]) -> tuple[str, str]:
    by_arm = {r["arm"]: r for r in eval_rows}
    b = by_arm.get("B", {})
    c = by_arm.get("C", {})
    if not b or not c:
        return "FAIL", "Missing Arm B or Arm C evaluation rows."
    psnr_loss = float(b.get("psnr_mean", np.nan)) - float(c.get("psnr_mean", np.nan))
    rel_b = float(b.get("relmeaserr_unclipped_float64_mean", np.nan))
    rel_c = float(c.get("relmeaserr_unclipped_float64_mean", np.nan))
    texture_metrics = {"rapsd_distance", "gradient_mean_abs_error", "highfreq_ratio_abs_error", "p0_l2"}
    wins = [r for r in comparison_rows if r["metric"] in texture_metrics and bool(r["ci_excludes_zero_in_favor_of_C"])]
    d_saturated = bool(pilot_c.get("d_saturated_last_mean_gt_0p95"))
    rel_ok = rel_c <= rel_b + 0.001
    psnr_ok = psnr_loss <= 0.3
    if d_saturated:
        return "FAIL", "Arm C discriminator saturated in the pilot."
    if not rel_ok:
        return "FAIL", f"Arm C RelMeasErr is not comparable to Arm B ({rel_c:.4g} vs {rel_b:.4g})."
    if wins and psnr_ok:
        names = ", ".join(r["metric"] for r in wins)
        return "SUCCESS", f"Arm C improves predeclared texture/perceptual metric(s) with CI above zero: {names}; PSNR loss {psnr_loss:.3f} dB."
    if wins and not psnr_ok:
        return "FAIL", f"Arm C has texture gains but PSNR loss {psnr_loss:.3f} dB exceeds the 0.3 dB guardrail."
    return "INERT", "Arm C does not beat the budget-matched supervised Arm B on predeclared texture/perceptual metrics with CI excluding zero."


def parse_args():
    parser = argparse.ArgumentParser(description="Phase69B controlled gauge-equalized audited cGAN pilot.")
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_count", type=int, default=2048)
    parser.add_argument("--val_count", type=int, default=384)
    parser.add_argument("--test_count", type=int, default=512)
    parser.add_argument("--smoke_steps", type=int, default=200)
    parser.add_argument("--pilot_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.train_count = min(args.train_count, 256)
        args.val_count = min(args.val_count, 128)
        args.test_count = min(args.test_count, 128)
        args.smoke_steps = min(args.smoke_steps, 20)
        args.pilot_steps = min(args.pilot_steps, 40)
        args.eval_every = min(args.eval_every, 20)
        args.save_every = min(args.save_every, 20)
    out_dir = Path(args.out_dir)
    pre_clean = output_dir_is_clean(out_dir)
    ensure_dir(out_dir)
    if pre_clean:
        write_text(out_dir / "RUNLOG.md", f"# Phase69B Runlog\n- {now()} runner_start\n")
    else:
        write_text(out_dir / "RUNLOG.md", f"# Phase69B Runlog\n- {now()} runner_start output_dir_not_clean\n")
    safe, checks, failures, warnings = preflight(out_dir, args)
    checks["safe_to_run"] = safe
    save_json(out_dir / "preflight_checks.json", checks)
    if not safe:
        return unsafe_stop(out_dir, failures, warnings)
    write_preflight_report(out_dir, checks, warnings)

    set_seed(6900)
    random.seed(6900)
    np.random.seed(6900)
    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    config = make_config(str(device), args.batch_size)
    measurement = make_measurement(config, device)
    A_np = np.load(p69a.A_SCR5).astype(np.float32)
    A = torch.from_numpy(A_np).to(device)
    measurement.set_A_override(A, metadata={"phase": "phase69B_main", "tensor_sha256": p69a.sha256_np(A_np)}, rebuild_cache=True)
    lambda_dc = float(config["lambda_solver"])

    append_log(out_dir, "build_caches_start")
    train_cache, val_cache, test_cache, split_info = build_caches(config, measurement, device, args.train_count, args.val_count, args.test_count)
    save_json(out_dir / "split_manifest.json", split_info)
    append_log(out_dir, "build_caches_complete")

    gen_a, config = load_generator_from_checkpoint(config, measurement, device)
    gauge_input_preflight(out_dir, gen_a, val_cache, A, lambda_dc, config, device)

    beta0, beta_rows = beta_calibration(gen_a, train_cache, A, lambda_dc, config, device, out_dir)
    write_csv(out_dir / "beta_calibration.csv", beta_rows)
    write_beta_report(out_dir, beta_rows)
    del gen_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    append_log(out_dir, "smoke_start")
    smoke_dir = ensure_dir(out_dir / "smoke")
    gen_b_smoke, _ = load_generator_from_checkpoint(config, measurement, device)
    b_smoke, b_smoke_rows, _ = train_arm(
        "armB",
        gen_b_smoke,
        train_cache,
        val_cache,
        A,
        lambda_dc,
        config,
        device,
        smoke_dir,
        steps=args.smoke_steps,
        beta=0.0,
        adversarial=False,
        eval_every=max(50, min(args.eval_every, args.smoke_steps)),
        save_every=max(50, min(args.save_every, args.smoke_steps)),
    )
    del gen_b_smoke
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gen_c_smoke, _ = load_generator_from_checkpoint(config, measurement, device)
    c_smoke, c_smoke_rows, _ = train_arm(
        "armC",
        gen_c_smoke,
        train_cache,
        val_cache,
        A,
        lambda_dc,
        config,
        device,
        smoke_dir,
        steps=args.smoke_steps,
        beta=beta0,
        adversarial=True,
        eval_every=max(50, min(args.eval_every, args.smoke_steps)),
        save_every=max(50, min(args.save_every, args.smoke_steps)),
    )
    del gen_c_smoke
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    smoke_rows = b_smoke_rows + c_smoke_rows
    write_csv(out_dir / "smoke_training_log.csv", smoke_rows)
    plot_smoke(out_dir, smoke_rows)
    smoke_ok, smoke_reasons = smoke_pass(b_smoke, c_smoke, smoke_rows)
    write_smoke_report(out_dir, smoke_ok, smoke_reasons, [b_smoke, c_smoke])
    append_log(out_dir, f"smoke_complete pass={smoke_ok}")
    if not smoke_ok:
        write_manifest(out_dir)
        return 3

    append_log(out_dir, "pilot_start")
    pilot_dir = ensure_dir(out_dir / "pilot")
    gen_b, _ = load_generator_from_checkpoint(config, measurement, device)
    b_summary, b_rows, b_best = train_arm(
        "armB",
        gen_b,
        train_cache,
        val_cache,
        A,
        lambda_dc,
        config,
        device,
        pilot_dir,
        steps=args.pilot_steps,
        beta=0.0,
        adversarial=False,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    del gen_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gen_c, _ = load_generator_from_checkpoint(config, measurement, device)
    c_summary, c_rows, c_best = train_arm(
        "armC",
        gen_c,
        train_cache,
        val_cache,
        A,
        lambda_dc,
        config,
        device,
        pilot_dir,
        steps=args.pilot_steps,
        beta=beta0,
        adversarial=True,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    del gen_c
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pilot_rows = b_rows + c_rows
    write_csv(out_dir / "pilot_training_log.csv", pilot_rows)
    plot_pilot_curves(out_dir, pilot_rows)
    append_log(out_dir, "pilot_complete")

    append_log(out_dir, "evaluation_start")
    eval_dir = ensure_dir(out_dir / "evaluation")
    gen_a, _ = load_generator_from_checkpoint(config, measurement, device)
    gen_a.eval()
    gen_b_eval = load_generator_checkpoint_for_eval(b_best, config, measurement, device)
    gen_c_eval = load_generator_checkpoint_for_eval(c_best, config, measurement, device)
    eval_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    for arm, gen in [("A", gen_a), ("B", gen_b_eval), ("C", gen_c_eval)]:
        agg, per, out_arr = evaluate_arm(arm, gen, test_cache, A, lambda_dc, config, device, eval_dir)
        eval_rows.append(agg)
        per_rows.extend(per)
        outputs[arm] = out_arr
    write_csv(out_dir / "evaluation_metrics.csv", eval_rows)
    write_csv(out_dir / "per_sample_metrics.csv", per_rows)
    comparison_rows = paired_comparison(per_rows)
    write_csv(out_dir / "paired_comparison_C_vs_B.csv", comparison_rows)
    save_visual_grid(out_dir, test_cache, outputs)
    save_rapsd_plot(out_dir, test_cache, outputs)
    write_csv(out_dir / "lpips_or_dists_results.csv", availability_rows())
    rel_rows = [
        {
            "arm": row["arm"],
            "relmeaserr_unclipped_float64_mean": row["relmeaserr_unclipped_float64_mean"],
            "relmeaserr_unclipped_float64_median": row["relmeaserr_unclipped_float64_median"],
            "certificate_operator": "Pi_y^lambda audit, not D",
        }
        for row in eval_rows
    ]
    write_csv(out_dir / "relmeaserr_certificate_table.csv", rel_rows)
    p0_rows = [
        {
            "arm": row["arm"],
            "p0_l2_mean": row["p0_l2_mean"],
            "p0_l2_median": row["p0_l2_median"],
            "rapsd_distance_mean": row["rapsd_distance_mean"],
            "gradient_mean_abs_error_mean": row["gradient_mean_abs_error_mean"],
            "highfreq_ratio_abs_error_mean": row["highfreq_ratio_abs_error_mean"],
        }
        for row in eval_rows
    ]
    write_csv(out_dir / "P0_texture_metrics.csv", p0_rows)
    append_log(out_dir, "evaluation_complete")

    decision, reason = decide(eval_rows, comparison_rows, c_summary)
    write_outputs(out_dir, checks, split_info, beta_rows, [b_smoke, c_smoke], [b_summary, c_summary], eval_rows, comparison_rows, decision, reason)
    write_manifest(out_dir)
    append_log(out_dir, f"runner_complete decision={decision} main_results_unchanged=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
