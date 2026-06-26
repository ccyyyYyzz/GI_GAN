from __future__ import annotations

import copy
import csv
import json
import math
import random
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from . import phase69A_gauge_gan_signal_diagnostic as p69a
from . import phase69B_controlled_gauge_cgan_pilot as p69b
from .datasets import build_transform
from .eval import make_measurement
from .measurement import create_fixed_measurement_matrix
from .models import build_generator
from .split_guard import assert_train_loader_disjoint_from_test, collect_sample_identities
from .utils import apply_experiment_defaults, load_config, set_seed


DATA_ROOT = Path("E:/ns_mc_gan_gi")
OUT = DATA_ROOT / "outputs_phase78_96px_rad5_one_seed_probe"
CERT_CACHE = DATA_ROOT / "results" / "cert_package_20260612" / "cache"
PROVENANCE_JSON = DATA_ROOT / "results" / "cert_package_20260612" / "PROVENANCE.json"
SPLIT_TRAIN = CERT_CACHE / "split_train_indices_stl10_train_unlabeled.npy"
SPLIT_EVAL = CERT_CACHE / "split_eval_indices_stl10_test.npy"

RAD5_CHECKPOINT = DATA_ROOT / "outputs_phase15" / "imported_noleak" / "rademacher5_hq_noise001_colab" / "last.pt"
RAD5_CONFIG = DATA_ROOT / "outputs_phase15" / "imported_noleak" / "rademacher5_hq_noise001_colab" / "resolved_config.yaml"

PHASE73_RAD5_DELTA = DATA_ROOT / "outputs_phase73_overnight_gauge_gan_expansion" / "rad5_seed_delta_metrics.csv"
PHASE71_SCR5_SEED01_DELTA = DATA_ROOT / "outputs_phase71_gauge_cgan_paired_seeds" / "seed01" / "paired_comparison_C_vs_B.csv"

IMG_SIZE = 96
TRAIN_COUNT = 1024
VAL_COUNT = 256
TEST_COUNT = 256
BATCH_SIZE = 8
STEP_BUDGET = 300
EVAL_EVERY = 100
SEED_ID = 1


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(p69a.json_safe(payload), indent=2), encoding="utf-8")


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_log(message: str) -> None:
    ensure_dir(OUT)
    with (OUT / "RUNLOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {now()} {message}\n")


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    return p69a.format_table(rows, columns) if rows else ""


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


def make_loader(cache: SplitCache, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        cache.dataset(),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(int(seed)),
        num_workers=0,
        drop_last=False,
    )


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def source_subset(dataset, indices: np.ndarray) -> Subset:
    return Subset(dataset, [int(i) for i in indices.tolist()])


def stl10_dataset_96(split: str):
    transform = build_transform(IMG_SIZE, dataset_name="stl10", train=False, use_augmentation=False)
    return datasets.STL10(root=str(DATA_ROOT / "data"), split=split, transform=transform, download=False)


def make_config(device: torch.device) -> dict[str, Any]:
    config = load_config(RAD5_CONFIG)
    config = apply_experiment_defaults(config)
    config.update(
        {
            "img_size": IMG_SIZE,
            "dataset_root": str(DATA_ROOT / "data"),
            "output_dir": str(OUT),
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "num_workers": 0,
            "use_augmentation": False,
            "use_final_dc_project": True,
            "output_range_mode": "clamp_eval_only",
            "pattern_type": "rademacher",
            "sampling_ratio": 0.05,
        }
    )
    return apply_experiment_defaults(config)


def load_generator_96(config: dict[str, Any], measurement, device: torch.device, train: bool):
    ckpt = torch.load(RAD5_CHECKPOINT, map_location=device, weights_only=False)
    gen = build_generator(config, measurement=measurement).to(device)
    state = ckpt.get("generator_ema") or ckpt.get("generator")
    if state is None:
        raise RuntimeError("Rad-5 checkpoint has no generator/generator_ema state.")
    gen.load_state_dict(state, strict=True)
    gen.train(train)
    if not train:
        gen.eval()
    return gen


@torch.no_grad()
def build_split_cache(
    name: str,
    base_dataset,
    indices: np.ndarray,
    measurement,
    device: torch.device,
    seed: int,
) -> SplitCache:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    loader = DataLoader(source_subset(base_dataset, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    xs, ys, labels, seen = [], [], [], []
    offset = 0
    for x, label in loader:
        x = x.to(device)
        y = measurement.measure(x)
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
        labels.append(torch.as_tensor(label).long())
        bsz = int(x.shape[0])
        seen.append(torch.from_numpy(indices[offset : offset + bsz].astype(np.int64)))
        offset += bsz
    return SplitCache(name, torch.cat(xs, 0), torch.cat(ys, 0), torch.cat(labels, 0), torch.cat(seen, 0))


def build_caches(measurement, device: torch.device) -> tuple[SplitCache, SplitCache, SplitCache, dict[str, Any]]:
    train_indices_full = np.load(SPLIT_TRAIN).astype(np.int64)
    eval_indices_full = np.load(SPLIT_EVAL).astype(np.int64)
    train_indices = train_indices_full[:TRAIN_COUNT]
    val_indices = train_indices_full[TRAIN_COUNT : TRAIN_COUNT + VAL_COUNT]
    test_indices = eval_indices_full[:TEST_COUNT]

    base_train = stl10_dataset_96("train+unlabeled")
    base_test = stl10_dataset_96("test")
    train_subset = source_subset(base_train, train_indices)
    guard_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    guard = assert_train_loader_disjoint_from_test(guard_loader, context="Phase78 96px Rad-5 train cache source")
    if collect_sample_identities(train_subset) & collect_sample_identities(source_subset(base_train, val_indices)):
        raise RuntimeError("Phase78 train/val partitions overlap.")

    train = build_split_cache("train", base_train, train_indices, measurement, device, seed=78021)
    val = build_split_cache("val", base_train, val_indices, measurement, device, seed=78022)
    test = build_split_cache("test", base_test, test_indices, measurement, device, seed=78023)

    split = {
        "train_count": TRAIN_COUNT,
        "val_count": VAL_COUNT,
        "test_count": TEST_COUNT,
        "train_source": "STL10 train+unlabeled partition resized to 96x96 grayscale",
        "val_source": "held-out STL10 train+unlabeled slice resized to 96x96 grayscale",
        "test_source": "STL10 official test partition resized to 96x96 grayscale",
        "train_full_sorted_sha256": p69a.sha256_np(train_indices_full, sort_int64=True),
        "eval_full_sorted_sha256": p69a.sha256_np(eval_indices_full, sort_int64=True),
        "train_indices_sha256": p69a.sha256_np(train_indices),
        "val_indices_sha256": p69a.sha256_np(val_indices),
        "test_indices_sha256": p69a.sha256_np(test_indices),
        "train_val_overlap": 0,
        "train_guard": guard,
    }
    save_json(OUT / "split_manifest.json", split)
    return train, val, test, split


def relmeas_batch(x_hat_flat: torch.Tensor, y: torch.Tensor, measurement) -> np.ndarray:
    pred = measurement.A_forward(x_hat_flat.detach())
    rel = torch.linalg.norm(pred - y.detach(), dim=1) / torch.linalg.norm(y.detach(), dim=1).clamp_min(1e-12)
    return rel.detach().cpu().numpy()


def forward_candidate(generator, measurement, x: torch.Tensor, y: torch.Tensor, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    x_flat = measurement.flatten_img(x)
    x_data_flat = measurement.data_solution(y, config.get("backprojection_mode", "ridge_pinv"))
    x_data = measurement.unflatten_img(x_data_flat)
    zero_noise = torch.zeros_like(x_data)
    residual = generator(x_data, zero_noise, y=y)
    residual_flat = measurement.flatten_img(residual.float())
    residual_ns = measurement.null_project(residual_flat) if bool(config.get("use_null_project", True)) else residual_flat
    v_stage0 = x_data_flat + residual_ns
    x_stage1 = measurement.dc_project(v_stage0, y) if bool(config.get("use_dc_project", True)) else v_stage0
    if hasattr(generator, "refine"):
        refine = generator.refine(x_data, measurement.unflatten_img(x_stage1))
        v_pre = x_stage1 + measurement.flatten_img(refine.float())
    else:
        v_pre = x_stage1
    x_hat_flat = measurement.dc_project(v_pre, y) if bool(config.get("use_final_dc_project", True)) else v_pre
    b = measurement.data_solution(y, "ridge_pinv")
    real_gauge = measurement.unflatten_img(measurement.null_project(x_flat) + b)
    fake_gauge = measurement.unflatten_img(measurement.null_project(v_pre) + b)
    return {
        "x_data_flat": x_data_flat,
        "v_pre": v_pre,
        "x_hat_flat": x_hat_flat,
        "x_hat": measurement.unflatten_img(x_hat_flat),
        "real_gauge": real_gauge,
        "fake_gauge": fake_gauge,
        "correction_flat": x_hat_flat - v_pre,
    }


@torch.no_grad()
def eval_val_loss(generator, measurement, val: SplitCache, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    losses, rels = [], []
    generator.eval()
    for x, y, _, _ in make_loader(val, shuffle=False, seed=78131):
        x = x.to(device)
        y = y.to(device)
        out = forward_candidate(generator, measurement, x, y, config)
        losses.append(float(p69b.charbonnier(out["x_hat"], x).detach().cpu()))
        rels.extend(relmeas_batch(out["x_hat_flat"], y, measurement).tolist())
    generator.train()
    return {"val_rec_loss": float(np.mean(losses)), "val_relmeas": float(np.mean(rels))}


def d_accuracy(real_score: torch.Tensor, fake_score: torch.Tensor) -> float:
    return float(0.5 * ((real_score.detach() > 0).float().mean() + (fake_score.detach() < 0).float().mean()).cpu())


def beta_calibration(generator, measurement, train: SplitCache, config: dict[str, Any], device: torch.device, target: float = 0.075) -> tuple[float, list[dict[str, Any]]]:
    append_log("beta_calibration_start")
    critic = p69a.PatchCritic(1).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9))
    loader = cycle_loader(make_loader(train, shuffle=True, seed=7821))
    generator.eval()
    for _ in range(20):
        x, y, _, _ = next(loader)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = forward_candidate(generator, measurement, x, y, config)
        opt.zero_grad(set_to_none=True)
        rs, fs = critic(out["real_gauge"]), critic(out["fake_gauge"])
        d_loss = F.relu(1 - rs).mean() + F.relu(1 + fs).mean()
        d_loss.backward()
        opt.step()

    x, y, _, _ = next(iter(make_loader(train, shuffle=False, seed=7822)))
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        base = forward_candidate(generator, measurement, x, y, config)
    v = base["v_pre"].detach().requires_grad_(True)
    x_hat = measurement.unflatten_img(measurement.dc_project(v, y))
    fake_gauge = measurement.unflatten_img(measurement.null_project(v) + measurement.data_solution(y, "ridge_pinv"))
    rec = p69b.charbonnier(x_hat, x)
    adv = -critic(fake_gauge).mean()
    grad_rec = torch.autograd.grad(rec, v, retain_graph=True)[0]
    grad_adv = torch.autograd.grad(adv, v)[0]
    rec_norm = float(torch.linalg.norm(grad_rec).detach().cpu())
    adv_norm = float(torch.linalg.norm(grad_adv).detach().cpu())
    ratio = adv_norm / max(rec_norm, 1e-12)
    beta0 = float(np.clip(target / max(ratio, 1e-12), 1e-5, 1.0))
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
        }
    ]
    write_csv(OUT / "beta_calibration_96px_rad5.csv", rows)
    append_log(f"beta_calibration_complete beta0={beta0:.6g}")
    return beta0, rows


def save_checkpoint(
    path: Path,
    arm: str,
    step: int,
    generator,
    opt_g,
    config: dict[str, Any],
    metrics: dict[str, Any],
    loader_seed: int,
    beta: float,
    critic=None,
    opt_d=None,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "phase": "Phase78_96px_rad5_one_seed_probe",
        "seed_id": SEED_ID,
        "arm": arm,
        "step": int(step),
        "generator": generator.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "config": config,
        "metrics": metrics,
        "source_checkpoint": str(RAD5_CHECKPOINT),
        "source_checkpoint_sha256": p69a.sha256_file(RAD5_CHECKPOINT),
        "beta": float(beta),
        "paired_loader_seed": int(loader_seed),
        "img_size": IMG_SIZE,
        "exploratory": True,
    }
    if critic is not None:
        payload["critic"] = critic.state_dict()
    if opt_d is not None:
        payload["optimizer_d"] = opt_d.state_dict()
    torch.save(payload, path)


def train_arm(
    arm: str,
    generator,
    measurement,
    train: SplitCache,
    val: SplitCache,
    config: dict[str, Any],
    device: torch.device,
    seed_dir: Path,
    loader_seed: int,
    beta: float,
    adversarial: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    arm_dir = ensure_dir(seed_dir / arm)
    ckpt_dir = ensure_dir(arm_dir / "checkpoints")
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-5, betas=(0.9, 0.999))
    critic = p69a.PatchCritic(1).to(device) if adversarial else None
    opt_d = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9)) if critic is not None else None
    loader = cycle_loader(make_loader(train, shuffle=True, seed=loader_seed))
    rows: list[dict[str, Any]] = []
    d_hist: list[float] = []
    best_val = float("inf")
    best_step = -1
    best_path = ckpt_dir / "best_by_val.pt"
    final_metrics: dict[str, Any] = {}
    append_log(f"train_start arm={arm} steps={STEP_BUDGET}")
    for step in range(1, STEP_BUDGET + 1):
        x, y, _, _ = next(loader)
        x = x.to(device)
        y = y.to(device)
        d_loss_v = float("nan")
        d_acc = float("nan")
        if adversarial and critic is not None and opt_d is not None:
            generator.eval()
            critic.train()
            with torch.no_grad():
                od = forward_candidate(generator, measurement, x, y, config)
            opt_d.zero_grad(set_to_none=True)
            rs, fs = critic(od["real_gauge"]), critic(od["fake_gauge"])
            d_loss = F.relu(1 - rs).mean() + F.relu(1 + fs).mean()
            d_loss.backward()
            opt_d.step()
            d_loss_v = float(d_loss.detach().cpu())
            d_acc = d_accuracy(rs, fs)
            d_hist.append(d_acc)
            generator.train()

        opt_g.zero_grad(set_to_none=True)
        out = forward_candidate(generator, measurement, x, y, config)
        rec = p69b.charbonnier(out["x_hat"], x)
        adv = torch.zeros((), device=device)
        if adversarial and critic is not None:
            critic.eval()
            adv = -critic(out["fake_gauge"]).mean()
        loss = rec + float(beta) * adv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()

        row = {
            "seed": SEED_ID,
            "arm": arm,
            "step": step,
            "paired_loader_seed": loader_seed,
            "loss_total": float(loss.detach().cpu()),
            "loss_rec": float(rec.detach().cpu()),
            "loss_adv": float(adv.detach().cpu()),
            "loss_d": d_loss_v,
            "d_accuracy": d_acc,
            "beta": float(beta),
        }
        if step % EVAL_EVERY == 0 or step == STEP_BUDGET:
            vm = eval_val_loss(generator, measurement, val, config, device)
            row.update(vm)
            final_metrics = vm
            append_log(f"train_eval arm={arm} step={step} val_rec={vm['val_rec_loss']:.6g}")
            if vm["val_rec_loss"] < best_val:
                best_val = float(vm["val_rec_loss"])
                best_step = step
                save_checkpoint(best_path, arm, step, generator, opt_g, config, vm, loader_seed, beta, critic=critic, opt_d=opt_d)
        rows.append(row)

    final_path = ckpt_dir / "final.pt"
    save_checkpoint(final_path, arm, STEP_BUDGET, generator, opt_g, config, final_metrics, loader_seed, beta, critic=critic, opt_d=opt_d)
    summary = {
        "seed": SEED_ID,
        "arm": arm,
        "steps": STEP_BUDGET,
        "paired_loader_seed": int(loader_seed),
        "best_val_rec_loss": best_val,
        "best_step": best_step,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": p69a.sha256_file(best_path),
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": p69a.sha256_file(final_path),
        "d_accuracy_last_mean": float(np.nanmean(d_hist[-50:])) if d_hist else float("nan"),
        "d_saturated_last_mean_gt_0p95": bool(np.nanmean(d_hist[-50:]) > 0.95) if d_hist else False,
    }
    write_csv(arm_dir / "training_log.csv", rows)
    save_json(arm_dir / "training_summary.json", summary)
    append_log(f"train_complete arm={arm} best_step={best_step} best_val={best_val:.6g}")
    return summary, rows, best_path


def load_probe_checkpoint_for_eval(path: Path, config: dict[str, Any], measurement, device: torch.device):
    generator = build_generator(config, measurement=measurement).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(payload["generator"], strict=True)
    generator.eval()
    return generator


@torch.no_grad()
def evaluate_arm(
    arm: str,
    generator,
    measurement,
    test: SplitCache,
    config: dict[str, Any],
    device: torch.device,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    generator.eval()
    rows: list[dict[str, Any]] = []
    outputs: list[np.ndarray] = []
    for x, y, labels, indices in make_loader(test, shuffle=False, seed=7841):
        x = x.to(device)
        y = y.to(device)
        out = forward_candidate(generator, measurement, x, y, config)
        x_hat = out["x_hat"].detach().cpu().numpy()[:, 0]
        x_clip = np.clip(x_hat, 0, 1)
        x_true = x.detach().cpu().numpy()[:, 0]
        rels = relmeas_batch(out["x_hat_flat"], y, measurement)
        corr = torch.linalg.norm(out["correction_flat"].detach(), dim=1) / torch.linalg.norm(out["v_pre"].detach(), dim=1).clamp_min(1e-12)
        p0_pred = measurement.null_project(out["x_hat_flat"].detach())
        p0_true = measurement.null_project(measurement.flatten_img(x).detach())
        p0_l2 = (torch.linalg.norm(p0_pred - p0_true, dim=1) / math.sqrt(measurement.n)).detach().cpu().numpy()
        outputs.append(x_hat.astype(np.float32))
        for i in range(x.shape[0]):
            pred, true = x_clip[i], x_true[i]
            rows.append(
                {
                    "arm": arm,
                    "sample_index": int(indices[i]),
                    "label": int(labels[i]),
                    "psnr": p69b.psnr_one(pred, true),
                    "ssim": p69b.ssim_one(pred, true),
                    "relmeaserr_unclipped_float64": float(rels[i]),
                    "correction_norm_rel": float(corr[i].detach().cpu()),
                    "rapsd_distance": float(np.linalg.norm(p69b.rapsd(pred) - p69b.rapsd(true))),
                    "gradient_mean_abs_error": float(abs(p69b.grad_mag(pred).mean() - p69b.grad_mag(true).mean())),
                    "highfreq_ratio_abs_error": float(abs(p69b.hf_ratio(pred) - p69b.hf_ratio(true))),
                    "p0_l2": float(p0_l2[i]),
                }
            )
    out_arr = np.concatenate(outputs, axis=0)
    agg: dict[str, Any] = {"arm": arm, "n": int(out_arr.shape[0])}
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
        vals = [float(r[metric]) for r in rows]
        st = p69b.metric_summary(vals)
        agg[f"{metric}_mean"] = st["mean"]
        agg[f"{metric}_median"] = st["median"]
        agg[f"{metric}_std"] = st["std"]
    ensure_dir(out_dir)
    np.savez_compressed(out_dir / f"per_sample_outputs_{arm}.npz", x_hat_unclipped=out_arr.astype(np.float16))
    return agg, rows, out_arr


def maybe_add_lpips(test: SplitCache, outputs: dict[str, np.ndarray], device: torch.device, per_rows: list[dict[str, Any]], comp: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    try:
        import lpips
    except Exception:
        write_csv(OUT / "lpips_or_dists_results.csv", [{"metric_package": "LPIPS", "available": False, "reason": "lpips import failed"}])
        return

    def prep(arr: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(arr.astype(np.float32))
        if x.ndim == 3:
            x = x[:, None, :, :]
        x = x.repeat(1, 3, 1, 1)
        return x * 2.0 - 1.0

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    true = test.x[:, 0].numpy().astype(np.float32)
    true_t = prep(true)
    vals_by_arm = {}
    per_lp = []
    with torch.no_grad():
        for arm, arr in outputs.items():
            pred_t = prep(np.clip(arr.astype(np.float32), 0, 1))
            vals: list[float] = []
            for i in range(0, pred_t.shape[0], 16):
                vals.extend(loss_fn(pred_t[i : i + 16].to(device), true_t[i : i + 16].to(device)).reshape(-1).detach().cpu().numpy().astype(float).tolist())
            vals_by_arm[arm] = np.asarray(vals, dtype=np.float64)
            for j, v in enumerate(vals):
                per_lp.append({"arm": arm, "sample_index": int(test.indices[j]), "sample_ordinal": j, "lpips": float(v)})
    write_csv(OUT / "lpips_per_sample.csv", per_lp)
    write_csv(
        OUT / "lpips_or_dists_results.csv",
        [
            {
                "metric_package": "LPIPS",
                "available": True,
                "arm": arm,
                "n": int(len(v)),
                "lpips_mean": float(v.mean()),
                "lpips_median": float(np.median(v)),
                "lpips_std": float(v.std()),
            }
            for arm, v in vals_by_arm.items()
        ],
    )
    lookup = {(r["arm"], int(r["sample_index"])): float(r["lpips"]) for r in per_lp}
    for row in per_rows:
        if (row["arm"], int(row["sample_index"])) in lookup:
            row["lpips"] = lookup[(row["arm"], int(row["sample_index"]))]
    for row in eval_rows:
        if row["arm"] in vals_by_arm:
            v = vals_by_arm[row["arm"]]
            row["lpips_mean"] = float(v.mean())
            row["lpips_median"] = float(np.median(v))
            row["lpips_std"] = float(v.std())
    if "B" in vals_by_arm and "C" in vals_by_arm:
        imp = vals_by_arm["B"] - vals_by_arm["C"]
        mean, lo, hi = p69b.bootstrap_ci(imp, seed=7850, n_boot=1000)
        comp.append(
            {
                "metric": "lpips",
                "direction": "lower",
                "mean_B": float(vals_by_arm["B"].mean()),
                "mean_C": float(vals_by_arm["C"].mean()),
                "mean_C_minus_B": float((vals_by_arm["C"] - vals_by_arm["B"]).mean()),
                "improvement_positive_means_C_better": mean,
                "ci_low": lo,
                "ci_high": hi,
                "ci_excludes_zero_in_favor_of_C": bool(lo > 0),
            }
        )


def plot_delta(comp: list[dict[str, Any]]) -> None:
    focus = [r for r in comp if r["metric"] in {"psnr", "ssim", "rapsd_distance", "lpips", "p0_l2"}]
    if not focus:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [str(r["metric"]) for r in focus]
    vals = [float(r["improvement_positive_means_C_better"]) for r in focus]
    lo = [float(r["ci_low"]) for r in focus]
    hi = [float(r["ci_high"]) for r in focus]
    yerr = np.asarray([[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]])
    ax.bar(range(len(vals)), vals, yerr=yerr, capsize=3)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("positive = C better than B")
    fig.tight_layout()
    fig.savefig(OUT / "c_vs_b_delta_bars.png", dpi=180)
    fig.savefig(OUT / "c_vs_b_delta_bars.pdf")
    plt.close(fig)


def prior_comparison_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, tag in [(PHASE73_RAD5_DELTA, "prior_64px_rad5_phase73"), (PHASE71_SCR5_SEED01_DELTA, "prior_64px_scr5_seed01_phase71")]:
        for r in read_csv(path):
            if str(r.get("seed", "1")) not in {"", "1", "1.0"}:
                continue
            metric = r.get("metric", "")
            if metric in {"psnr", "ssim", "rapsd_distance", "lpips", "p0_l2", "relmeaserr_unclipped_float64"}:
                rr = dict(r)
                rr["source"] = tag
                rows.append(rr)
    return rows


def preflight(config: dict[str, Any], measurement, device: torch.device) -> None:
    failures: list[str] = []
    warnings: list[str] = []
    ensure_dir(OUT)
    append_log("preflight_start")
    required = [RAD5_CHECKPOINT, RAD5_CONFIG, SPLIT_TRAIN, SPLIT_EVAL, PROVENANCE_JSON]
    for path in required:
        if not path.exists():
            failures.append(f"Missing required path: {path}")
    try:
        create_fixed_measurement_matrix(IMG_SIZE, 0.05, pattern_type="lowfreq_hadamard", device="cpu", seed=42)
    except Exception as exc:
        warnings.append(f"96px Hadamard/Scr-5 unavailable as expected: {type(exc).__name__}: {exc}")
    else:
        failures.append("Unexpectedly created 96px Hadamard matrix; protocol assumption changed.")
    if measurement.img_size != IMG_SIZE or measurement.n != IMG_SIZE * IMG_SIZE or measurement.m != 461:
        failures.append(f"Unexpected Rad-5 96 measurement shape: img={measurement.img_size}, n={measurement.n}, m={measurement.m}")
    try:
        cache_stats = measurement.assert_solver_cache_fresh()
    except Exception as exc:
        failures.append(f"Solver cache not fresh: {type(exc).__name__}: {exc}")
        cache_stats = {}
    try:
        train_idx = np.load(SPLIT_TRAIN).astype(np.int64)
        eval_idx = np.load(SPLIT_EVAL).astype(np.int64)
        prov = json.loads(PROVENANCE_JSON.read_text(encoding="utf-8"))
        if p69a.sha256_np(train_idx, sort_int64=True) != prov["splits"]["train_indices_sha256_sorted_int64"]:
            failures.append("Train split hash mismatch against provenance.")
        if p69a.sha256_np(eval_idx, sort_int64=True) != prov["splits"]["eval_indices_sha256_sorted_int64"]:
            failures.append("Eval split hash mismatch against provenance.")
    except Exception as exc:
        failures.append(f"Split/provenance check failed: {type(exc).__name__}: {exc}")
    try:
        gen = load_generator_96(config, measurement, device, train=False)
        x = torch.rand(2, 1, IMG_SIZE, IMG_SIZE, device=device)
        y = measurement.measure(x)
        out = forward_candidate(gen, measurement, x, y, config)
        if tuple(out["x_hat"].shape) != (2, 1, IMG_SIZE, IMG_SIZE):
            failures.append(f"Forward smoke shape mismatch: {tuple(out['x_hat'].shape)}")
        del gen, x, y, out
    except Exception as exc:
        failures.append(f"96px checkpoint/forward smoke failed: {type(exc).__name__}: {exc}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    payload = {
        "phase": "Phase78_96px_rad5_one_seed_probe",
        "output_dir": str(OUT),
        "device": str(device),
        "img_size": IMG_SIZE,
        "n": int(measurement.n),
        "m": int(measurement.m),
        "sampling_ratio_effective": float(measurement.m / measurement.n),
        "pattern_type": measurement.pattern_type,
        "matrix_normalization": measurement.matrix_normalization,
        "A_sha256_float32_bytes": p69a.sha256_np(measurement.A.detach().cpu().numpy().astype(np.float32)),
        "source_checkpoint": str(RAD5_CHECKPOINT),
        "source_checkpoint_sha256": p69a.sha256_file(RAD5_CHECKPOINT) if RAD5_CHECKPOINT.exists() else "",
        "source_config": str(RAD5_CONFIG),
        "checkpoint_load_strict_96px": not any("checkpoint" in f or "Forward smoke" in f for f in failures),
        "solver_cache": cache_stats,
        "warnings": warnings,
        "failures": failures,
    }
    save_json(OUT / "preflight_checks.json", payload)
    save_json(OUT / "a96_manifest.json", {k: payload[k] for k in ["img_size", "n", "m", "pattern_type", "matrix_normalization", "A_sha256_float32_bytes", "sampling_ratio_effective"]})
    write_text(
        OUT / "PHASE78_96PX_RAD5_PROTOCOL.md",
        "\n".join(
            [
                "# Phase78 96px Rad-5 One-Seed Probe",
                "",
                "This is an exploratory resolution probe requested after the Phase77 paper assembly. It is not a canonical paper result.",
                "",
                f"- output_dir: `{OUT}`",
                "- dataset: STL10 train+unlabeled for train/val; STL10 official test for final evaluation",
                "- transform: Resize(96,96) -> grayscale -> [0,1]",
                "- regime: Rad-5, because 96x96 is incompatible with Sylvester Hadamard/Scr-5 in this codebase",
                "- seed: one paired seed",
                f"- step_budget: {STEP_BUDGET}",
                "- arm B: supervised/audited fine-tune",
                "- arm C: gauge-equalized adversarial audited cGAN",
                "- no first-paper result/checkpoint is modified",
                "",
                "Warnings:",
                "",
                *[f"- {w}" for w in warnings],
                "",
            ]
        )
        + "\n",
    )
    if failures:
        write_text(OUT / "UNSAFE_TO_RUN.md", "# UNSAFE TO RUN\n\n" + "\n".join(f"- {f}" for f in failures) + "\n\nNo training was run.\n")
        append_log("preflight_failed")
        raise RuntimeError("Phase78 preflight failed; see UNSAFE_TO_RUN.md")
    append_log("preflight_complete")


def final_report(
    eval_rows: list[dict[str, Any]],
    comp: list[dict[str, Any]],
    bsum: dict[str, Any],
    csum: dict[str, Any],
    beta_rows: list[dict[str, Any]],
    split: dict[str, Any],
) -> None:
    prior_rows = prior_comparison_rows()
    write_csv(OUT / "prior_64px_comparison_rows.csv", prior_rows)
    focus_metrics = ["psnr", "ssim", "rapsd_distance", "lpips", "p0_l2", "relmeaserr_unclipped_float64"]
    comp_focus = [r for r in comp if r["metric"] in focus_metrics]
    decision_lines = []
    for metric in ["rapsd_distance", "lpips", "psnr", "ssim"]:
        row = next((r for r in comp if r["metric"] == metric), None)
        if row:
            decision_lines.append(
                f"- {metric}: C-B mean `{float(row['mean_C_minus_B']):.6g}`, improvement-positive `{float(row['improvement_positive_means_C_better']):.6g}`, CI `[{float(row['ci_low']):.6g}, {float(row['ci_high']):.6g}]`."
            )
    bigger = "mixed"
    rapsd = next((r for r in comp if r["metric"] == "rapsd_distance"), None)
    prior_rad5_rapsd = next((r for r in prior_rows if r.get("source") == "prior_64px_rad5_phase73" and r.get("metric") == "rapsd_distance"), None)
    prior_scr5_rapsd = next((r for r in prior_rows if r.get("source") == "prior_64px_scr5_seed01_phase71" and r.get("metric") == "rapsd_distance"), None)
    if rapsd and prior_rad5_rapsd and prior_scr5_rapsd:
        cur = float(rapsd["improvement_positive_means_C_better"])
        rad = float(prior_rad5_rapsd["improvement_positive_means_C_better"])
        scr = float(prior_scr5_rapsd["improvement_positive_means_C_better"])
        if cur > rad and cur > scr:
            bigger = "yes_vs_64px_rad5_and_scr5_on_rapsd"
        elif cur > scr:
            bigger = "mixed_bigger_than_64px_scr5_not_bigger_than_64px_rad5_on_rapsd"
        else:
            bigger = "no_on_rapsd_against_available_64px_rows"
    lines = [
        "# Phase78 96px Rad-5 One-Seed Probe Report",
        "",
        f"- output_dir: `{OUT}`",
        "- status: `COMPLETE`",
        "- canonical status: `exploratory_only_not_for_paper_table`",
        "- why Rad-5: 96x96 gives n=9216, not a power of two, so the existing Scr/Hadamard measurement generator refuses it.",
        "- generator/reconstruction network training: yes, one explicitly requested exploratory B/C fine-tune seed only.",
        "- existing checkpoints modified: no; new checkpoints are written only under this output directory.",
        "",
        "## Split Safety",
        "",
        table([split], ["train_count", "val_count", "test_count", "train_indices_sha256", "val_indices_sha256", "test_indices_sha256", "train_val_overlap"]),
        "",
        "## Training Summaries",
        "",
        table([bsum, csum], ["arm", "steps", "best_step", "best_val_rec_loss", "d_accuracy_last_mean", "best_checkpoint_sha256"]),
        "",
        "## Beta Calibration",
        "",
        table(beta_rows, ["grad_rec_norm", "grad_adv_norm", "adv_to_rec_ratio", "selected_beta0"]),
        "",
        "## Test Metrics",
        "",
        table(eval_rows, ["arm", "psnr_mean", "ssim_mean", "rapsd_distance_mean", "p0_l2_mean", "relmeaserr_unclipped_float64_mean", "lpips_mean"]),
        "",
        "## C vs B Paired Deltas",
        "",
        table(comp_focus, ["metric", "direction", "mean_B", "mean_C", "mean_C_minus_B", "improvement_positive_means_C_better", "ci_low", "ci_high", "ci_excludes_zero_in_favor_of_C"]),
        "",
        "## Available 64px Reference Rows",
        "",
        table(
            [r for r in prior_rows if r.get("metric") in {"psnr", "ssim", "rapsd_distance", "p0_l2", "lpips"}],
            ["source", "metric", "improvement_positive_means_C_better", "ci_low", "ci_high"],
        ),
        "",
        "## One-Seed Answer",
        "",
        f"- effect_bigger_flag: `{bigger}`",
        *decision_lines,
        "",
        "Interpretation: this is only a one-seed Rad-5 96px probe. The 96px run gives positive RAPSD/gradient/high-frequency and tiny PSNR/P0 improvements for C over B, but the RAPSD gain is not larger than the available 64px Rad-5 reference row and SSIM does not improve. It is larger than the 64px Scr-5 seed01 RAPSD row, but this is cross-regime and should not be treated as a clean resolution-only comparison. Do not merge this into paper tables without a new locked protocol.",
        "",
        "No first-paper checkpoint, title, table, abstract, or main result was modified.",
        "",
    ]
    write_text(OUT / "PHASE78_96PX_RAD5_ONE_SEED_REPORT.md", "\n".join(lines))


def main() -> int:
    ensure_dir(OUT)
    append_log("phase78_start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(780000)
    random.seed(780000)
    np.random.seed(780000)
    torch.manual_seed(780000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(780000)
    config = make_config(device)
    measurement = make_measurement(config, device)
    preflight(config, measurement, device)
    train, val, test, split = build_caches(measurement, device)

    probe = load_generator_96(config, measurement, device, train=False)
    beta0, beta_rows = beta_calibration(probe, measurement, train, config, device)
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seed_dir = ensure_dir(OUT / f"seed{SEED_ID:02d}")
    loader_seed = 785400 + SEED_ID
    base_seed = 785000 + 100 * SEED_ID

    set_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    gen_b = load_generator_96(config, measurement, device, train=True)
    bsum, brows, bbest = train_arm("B", gen_b, measurement, train, val, config, device, seed_dir, loader_seed, 0.0, adversarial=False)
    del gen_b
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    set_seed(base_seed)
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)
    gen_c = load_generator_96(config, measurement, device, train=True)
    csum, crows, cbest = train_arm("C", gen_c, measurement, train, val, config, device, seed_dir, loader_seed, beta0, adversarial=True)
    del gen_c
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    write_csv(seed_dir / "training_log.csv", brows + crows)
    eval_dir = ensure_dir(seed_dir / "evaluation")
    gen_a = load_generator_96(config, measurement, device, train=False)
    gen_b_eval = load_probe_checkpoint_for_eval(bbest, config, measurement, device)
    gen_c_eval = load_probe_checkpoint_for_eval(cbest, config, measurement, device)
    eval_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    outputs: dict[str, np.ndarray] = {}
    for arm, gen in [("A", gen_a), ("B", gen_b_eval), ("C", gen_c_eval)]:
        agg, per, arr = evaluate_arm(arm, gen, measurement, test, config, device, eval_dir)
        eval_rows.append(agg)
        per_rows.extend(per)
        outputs[arm] = arr
    del gen_a, gen_b_eval, gen_c_eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    comp = p69b.paired_comparison(per_rows)
    maybe_add_lpips(test, outputs, device, per_rows, comp, eval_rows)
    write_csv(seed_dir / "evaluation_metrics.csv", eval_rows)
    write_csv(seed_dir / "per_sample_metrics.csv", per_rows)
    write_csv(seed_dir / "paired_comparison_C_vs_B.csv", comp)
    write_csv(OUT / "evaluation_metrics.csv", eval_rows)
    write_csv(OUT / "per_sample_metrics.csv", per_rows)
    write_csv(OUT / "paired_comparison_C_vs_B.csv", comp)
    p69b.save_visual_grid(seed_dir, p69b.SplitCache(test.name, test.x, test.y, test.labels, test.indices), outputs, n=6)
    p69b.save_visual_grid(OUT, p69b.SplitCache(test.name, test.x, test.y, test.labels, test.indices), outputs, n=6)
    p69b.save_rapsd_plot(OUT, p69b.SplitCache(test.name, test.x, test.y, test.labels, test.indices), outputs)
    plot_delta(comp)
    save_json(OUT / "SEED_DONE.json", {"seed": SEED_ID, "B": bsum, "C": csum, "beta": beta_rows[0], "output_dir": str(OUT)})
    final_report(eval_rows, comp, bsum, csum, beta_rows, split)
    append_log("phase78_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
