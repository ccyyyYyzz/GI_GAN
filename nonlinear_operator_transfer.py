from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import operator_conditioned_nullspace_canary as occ
from src import phase69A_gauge_gan_signal_diagnostic as p69a
from src import phase1_4v4b0_scoring as b0
from src.dc_balanced import (
    dc_row,
    dct2_basis_row,
    full_rmse,
    centered_rmse,
    dct_band_rmse,
    hadamard_lowsequency_non_dc_rows,
    random_zero_mean_rows,
    row_audit,
)
from src.operator_conditioned_nullspace import MatrixFreeNullProjector, SmallNullspaceUNet, reconstruct_with_projected_residual
from src.phase2_fresh_operator import resolve_device
from src.phase2_witness import repo_state, sha256_file, write_csv, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "nonlinear_operator_transfer_dev.yaml"


class NonlinearTransferError(RuntimeError):
    pass


@dataclass
class TransferArm:
    arm_id: str
    family: str
    budget: int
    pool: str
    operator_seed: int
    rows: np.ndarray
    projector: MatrixFreeNullProjector
    cond: torch.Tensor


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return json_safe(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val):
            return None
        if math.isinf(val):
            return "inf" if val > 0 else "-inf"
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise NonlinearTransferError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def sha256_np(arr: np.ndarray, *, sort_int64: bool = False) -> str:
    x = np.asarray(arr)
    if sort_int64:
        x = np.sort(x.astype(np.int64))
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_indices(config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_full = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    cfg = dict(config["splits"])
    out: dict[str, np.ndarray] = {}
    used: list[int] = []
    manifest: dict[str, Any] = {
        "source": "STL10 train+unlabeled only; no final-v4 or Phase2 locked indices",
        "train_full_sorted_sha256": sha256_np(train_full, sort_int64=True),
    }
    for name in ["train", "val", "dev"]:
        offset = int(cfg[name]["offset"])
        count = int(cfg[name]["count"])
        if offset < 0 or offset + count > train_full.shape[0]:
            raise NonlinearTransferError(f"SPLIT_OUT_OF_RANGE:{name}:{offset}:{count}:{train_full.shape[0]}")
        arr = train_full[offset : offset + count].copy()
        out[name] = arr
        used.extend(int(v) for v in arr.tolist())
        manifest[name] = {"offset": offset, "count": count, "indices_sha256": sha256_np(arr), "indices_sorted_sha256": sha256_np(arr, sort_int64=True)}
    if len(set(used)) != len(used):
        raise NonlinearTransferError("SPLIT_OVERLAP_DETECTED")
    manifest["combined_indices_sha256"] = sha256_np(np.asarray(used, dtype=np.int64))
    return out, manifest


def sample_hashes(indices: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    base = p69a.stl10_dataset("train+unlabeled")
    raw_hashes: list[str] = []
    transformed_hashes: list[str] = []
    rows: list[dict[str, Any]] = []
    data = getattr(base, "data", None)
    for idx in [int(v) for v in indices.tolist()]:
        x, label = base[idx]
        raw = np.asarray(data[idx]) if data is not None else np.asarray(x)
        raw_h = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
        trans = x.detach().cpu().contiguous().numpy().astype(np.float32)
        trans_h = hashlib.sha256(np.ascontiguousarray(trans).tobytes()).hexdigest()
        raw_hashes.append(raw_h)
        transformed_hashes.append(trans_h)
        rows.append({"source_index": idx, "label": int(label), "raw_source_sha256": raw_h, "transformed_64_sha256": trans_h})
    return rows, np.asarray(raw_hashes, dtype=object), np.asarray(transformed_hashes, dtype=object)


def duplicate_audit(splits: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_rows: list[dict[str, Any]] = []
    by_split: dict[str, dict[str, set[str]]] = {}
    for split, idx in splits.items():
        rows, raw, trans = sample_hashes(idx)
        for r in rows:
            r["split"] = split
        per_rows.extend(rows)
        by_split[split] = {"raw": set(raw.tolist()), "transformed": set(trans.tolist())}
    checks: dict[str, Any] = {"status": "PASS", "split_pairs": []}
    for a in sorted(by_split):
        checks[a] = {
            "raw_unique": len(by_split[a]["raw"]),
            "transformed_unique": len(by_split[a]["transformed"]),
            "raw_count": int(len(splits[a])),
            "transformed_count": int(len(splits[a])),
            "raw_exact_duplicates_within_split": int(len(splits[a]) - len(by_split[a]["raw"])),
            "transformed_exact_duplicates_within_split": int(len(splits[a]) - len(by_split[a]["transformed"])),
        }
        if checks[a]["raw_exact_duplicates_within_split"] or checks[a]["transformed_exact_duplicates_within_split"]:
            checks["status"] = "FAIL"
    for a in sorted(by_split):
        for b in sorted(by_split):
            if a >= b:
                continue
            raw_overlap = len(by_split[a]["raw"] & by_split[b]["raw"])
            trans_overlap = len(by_split[a]["transformed"] & by_split[b]["transformed"])
            checks["split_pairs"].append({"a": a, "b": b, "raw_overlap": raw_overlap, "transformed_overlap": trans_overlap})
            if raw_overlap or trans_overlap:
                checks["status"] = "FAIL"
    return checks, per_rows


def load_split_arrays(indices: np.ndarray, *, batch_size: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    loader = occ.make_loader(indices, batch_size=batch_size, shuffle=False, seed=12345)
    xs: list[torch.Tensor] = []
    labels: list[np.ndarray] = []
    seen: list[np.ndarray] = []
    for batch in loader:
        x, lab, idx = occ.batch_to_flat(batch, device)
        xs.append(x.detach().cpu().reshape(x.shape[0], -1))
        labels.append(lab.numpy().astype(np.int64))
        seen.append(idx.numpy().astype(np.int64))
    return torch.cat(xs, dim=0), np.concatenate(labels), np.concatenate(seen)


def dct_seeded_rows(num_rows: int, img_size: int, seed: int, *, pool_factor: int = 5) -> np.ndarray:
    size = int(img_size)
    coords = [(u, v) for u in range(size) for v in range(size) if not (u == 0 and v == 0)]
    coords.sort(key=lambda uv: (uv[0] * uv[0] + uv[1] * uv[1], uv[0] + uv[1], uv[0], uv[1]))
    pool_size = min(len(coords), max(int(num_rows), int(num_rows) * int(pool_factor)))
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(np.arange(pool_size), size=int(num_rows), replace=False))
    rows = [dct2_basis_row(size, *coords[int(i)]) for i in chosen]
    return np.stack(rows, axis=0).astype(np.float32)


def hadamard_seeded_rows(num_rows: int, dim: int, seed: int, *, pool_factor: int = 5) -> np.ndarray:
    pool = hadamard_lowsequency_non_dc_rows(max(int(num_rows) * int(pool_factor), int(num_rows)), int(dim))
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(np.arange(pool.shape[0]), size=int(num_rows), replace=False))
    return pool[chosen].astype(np.float32)


def build_rows(family: str, budget: int, *, img_size: int, seed: int) -> np.ndarray:
    dim = int(img_size) * int(img_size)
    non_dc = int(budget) - 1
    if non_dc < 0:
        raise NonlinearTransferError(f"BAD_BUDGET:{budget}")
    family = str(family).lower()
    if family == "random":
        extra = random_zero_mean_rows(non_dc, dim, seed)
    elif family == "dct":
        extra = dct_seeded_rows(non_dc, img_size, seed)
    elif family == "hadamard":
        extra = hadamard_seeded_rows(non_dc, dim, seed)
    elif family == "mixed":
        random_count = max(1, int(round(non_dc * 0.2)))
        dct_count = non_dc - random_count
        extra = np.concatenate(
            [
                random_zero_mean_rows(random_count, dim, seed + 17),
                dct_seeded_rows(dct_count, img_size, seed + 31),
            ],
            axis=0,
        )
    else:
        raise NonlinearTransferError(f"UNKNOWN_FAMILY:{family}")
    return np.concatenate([dc_row(dim)[None, :], extra], axis=0).astype(np.float32)


def dct_visibility(rows: np.ndarray, *, img_size: int) -> np.ndarray:
    from scipy.fft import dctn

    arr = np.asarray(rows, dtype=np.float32).reshape(rows.shape[0], int(img_size), int(img_size))
    coeff = dctn(arr, axes=(1, 2), norm="ortho")
    return np.sum(coeff * coeff, axis=0).reshape(-1)


def norm_channel(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64)
    return ((x - float(x.mean())) / max(float(x.std()), 1e-8)).astype(np.float32)


def condition_features(rows: np.ndarray, *, img_size: int, max_budget: int, device: torch.device) -> torch.Tensor:
    leverage = np.sum(np.asarray(rows, dtype=np.float64) ** 2, axis=0)
    visibility = dct_visibility(rows, img_size=img_size)
    budget = np.full_like(leverage, float(rows.shape[0]) / float(max_budget), dtype=np.float64)
    feat = np.stack(
        [
            norm_channel(leverage).reshape(img_size, img_size),
            budget.astype(np.float32).reshape(img_size, img_size),
            norm_channel(visibility).reshape(img_size, img_size),
        ],
        axis=0,
    )
    return torch.from_numpy(feat[None].astype(np.float32)).to(device)


def build_arm_pool(config: Mapping[str, Any], pool: str, *, device: torch.device) -> list[TransferArm]:
    op = dict(config["operator_protocol"])
    img_size = int(op.get("img_size", 64))
    max_budget = int(max(op["budgets"]))
    families = list(op["families"])
    if pool == "train":
        budgets = [int(v) for v in op["train_budgets"]]
        seeds = [int(v) for v in op["train_operator_seeds"]]
    elif pool == "val":
        budgets = [int(v) for v in op["train_budgets"]]
        seeds = [int(v) for v in op["val_operator_seeds"]]
    elif pool == "test":
        budgets = [int(v) for v in op["budgets"]]
        seeds = [int(v) for v in op["test_operator_seeds"]]
    else:
        raise NonlinearTransferError(f"UNKNOWN_POOL:{pool}")
    arms: list[TransferArm] = []
    for family in families:
        for budget in budgets:
            for seed in seeds:
                rows = build_rows(family, budget, img_size=img_size, seed=seed)
                projector = MatrixFreeNullProjector(torch.from_numpy(rows).to(device=device, dtype=torch.float32))
                cond = condition_features(rows, img_size=img_size, max_budget=max_budget, device=device)
                arm_id = f"{pool}_{family}_m{budget}_op{seed}"
                arms.append(TransferArm(arm_id, family, int(budget), pool, int(seed), rows, projector, cond))
    return arms


@dataclass
class EmpiricalStats:
    mu: np.ndarray
    z: np.ndarray
    components: np.ndarray


def fit_empirical_stats(x_train: np.ndarray, *, klt_rank: int) -> EmpiricalStats:
    mu = x_train.mean(axis=0).astype(np.float64)
    z = (x_train.astype(np.float64) - mu[None, :]).astype(np.float64)
    # Thin SVD; components are orthonormal rows in pixel space.
    _, _s, vt = np.linalg.svd(z, full_matrices=False)
    rank = min(int(klt_rank), vt.shape[0])
    return EmpiricalStats(mu=mu, z=z, components=vt[:rank].astype(np.float64))


def audit_numpy(xhat: np.ndarray, x: np.ndarray, arm: TransferArm, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    xb = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    y = arm.projector.measurement(xb)
    pred = torch.from_numpy(np.asarray(xhat, dtype=np.float32)).to(device)
    audited = arm.projector.data_anchor(y) + arm.projector.null_project(pred)
    rel = arm.projector.relmeaserr(audited, y)
    return audited.detach().cpu().numpy().astype(np.float32), rel.detach().cpu().numpy().astype(np.float64)


def joint_predict(x: np.ndarray, arm: TransferArm, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    return occ.predict_joint(arm.projector, x.astype(np.float32), device=device)


def null_mean_predict(x: np.ndarray, arm: TransferArm, stats: EmpiricalStats, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    xb = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    y = arm.projector.measurement(xb)
    r = arm.projector.data_anchor(y)
    mu = torch.from_numpy(stats.mu.astype(np.float32)).to(device).reshape(1, -1)
    pred = r + arm.projector.null_project(mu).expand(r.shape[0], -1)
    rel = arm.projector.relmeaserr(pred, y)
    return pred.detach().cpu().numpy().astype(np.float32), rel.detach().cpu().numpy().astype(np.float64)


def lmmse_predict(x: np.ndarray, arm: TransferArm, stats: EmpiricalStats, *, lambda_: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(arm.rows, dtype=np.float64)
    x64 = np.asarray(x, dtype=np.float64)
    y = x64 @ A.T
    y_mu = stats.mu @ A.T
    u = stats.z @ A.T
    scale = max(stats.z.shape[0] - 1, 1)
    s = (u.T @ u) / scale
    ca_t = (stats.z.T @ u) / scale
    coeff = np.linalg.solve(s + float(lambda_) * np.eye(s.shape[0]), (y - y_mu[None, :]).T).T
    pred = stats.mu[None, :] + coeff @ ca_t.T
    return audit_numpy(pred.astype(np.float32), x.astype(np.float32), arm, device=device)


def klt_predict(x: np.ndarray, arm: TransferArm, stats: EmpiricalStats, *, lambda_: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(arm.rows, dtype=np.float64)
    q = stats.components
    x64 = np.asarray(x, dtype=np.float64)
    y_res = x64 @ A.T - stats.mu @ A.T
    b = A @ q.T
    lhs = b.T @ b + float(lambda_) * np.eye(q.shape[0])
    beta = np.linalg.solve(lhs, (b.T @ y_res.T)).T
    pred = stats.mu[None, :] + beta @ q
    return audit_numpy(pred.astype(np.float32), x.astype(np.float32), arm, device=device)


def cond_for_arm(arm_index: int, arms: Sequence[TransferArm], mode: str, batch: int, device: torch.device) -> torch.Tensor:
    if mode == "full":
        return arms[arm_index].cond
    if mode == "no_condition":
        return torch.zeros_like(arms[arm_index].cond)
    if mode == "shuffled_condition":
        return arms[(arm_index + 1) % len(arms)].cond
    raise NonlinearTransferError(f"UNKNOWN_CONDITION_MODE:{mode}")


def train_variant(
    *,
    variant: str,
    train_seed: int,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    train_arms: Sequence[TransferArm],
    val_arms: Sequence[TransferArm],
    config: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    img_size = int(config["operator_protocol"].get("img_size", 64))
    model_cfg = dict(config.get("model", {}))
    torch.manual_seed(int(train_seed))
    np.random.seed(int(train_seed) % (2**32 - 1))
    model = SmallNullspaceUNet(in_channels=4, base_channels=int(model_cfg.get("base_channels", 24)), blocks=int(model_cfg.get("blocks", 1))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["training"].get("lr", 3e-4)), weight_decay=float(config["training"].get("weight_decay", 1e-4)))
    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    batch_size = int(config.get("batch_size", 32))
    epochs = int(config["training"].get("epochs", 6))
    clip_grad = float(config["training"].get("clip_grad", 1.0))
    train_x_dev = train_x.to(device=device, dtype=torch.float32)
    val_np = val_x.numpy().astype(np.float32)
    log: list[dict[str, Any]] = []
    best = {"epoch": -1, "val_centered_mean": float("inf"), "state": None}
    gen = torch.Generator(device="cpu").manual_seed(int(train_seed) + 991)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(train_x_dev.shape[0], generator=gen)
        losses: list[float] = []
        rels: list[float] = []
        for start in range(0, perm.numel(), batch_size):
            idx = perm[start : start + batch_size].to(device)
            x = train_x_dev[idx]
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = torch.zeros((), device=device)
                    for arm_i, arm in enumerate(train_arms):
                        y = arm.projector.measurement(x)
                        r = arm.projector.data_anchor(y)
                        target = x - r
                        out = reconstruct_with_projected_residual(model, arm.projector, r, y, img_size=img_size, cond_features=cond_for_arm(arm_i, train_arms, variant, x.shape[0], device))
                        loss = loss + F.mse_loss(out.null_hat, target)
                        rels.append(float(out.relmeaserr.detach().max().cpu()))
                    loss = loss / len(train_arms)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(opt)
                scaler.update()
            else:
                loss = torch.zeros((), device=device)
                for arm_i, arm in enumerate(train_arms):
                    y = arm.projector.measurement(x)
                    r = arm.projector.data_anchor(y)
                    target = x - r
                    out = reconstruct_with_projected_residual(model, arm.projector, r, y, img_size=img_size, cond_features=cond_for_arm(arm_i, train_arms, variant, x.shape[0], device))
                    loss = loss + F.mse_loss(out.null_hat, target)
                    rels.append(float(out.relmeaserr.detach().max().cpu()))
                loss = loss / len(train_arms)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                opt.step()
            losses.append(float(loss.detach().cpu()))
        val_centered: list[float] = []
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(val_np).to(device)
            for arm_i, arm in enumerate(val_arms):
                y = arm.projector.measurement(xb)
                r = arm.projector.data_anchor(y)
                out = reconstruct_with_projected_residual(model, arm.projector, r, y, img_size=img_size, cond_features=cond_for_arm(arm_i, val_arms, variant, xb.shape[0], device))
                val_centered.append(float(centered_rmse(out.x_hat.detach().cpu().numpy(), val_np).mean()))
        row = {
            "variant": variant,
            "train_seed": int(train_seed),
            "epoch": int(epoch),
            "loss_mean": float(np.mean(losses)),
            "train_relmeaserr_max": float(np.max(rels)) if rels else None,
            "val_centered_mean": float(np.mean(val_centered)),
        }
        log.append(row)
        if row["val_centered_mean"] < best["val_centered_mean"]:
            best = {"epoch": epoch, "val_centered_mean": row["val_centered_mean"], "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if best["state"] is None:
        raise NonlinearTransferError("NO_BEST_STATE")
    model.load_state_dict(best["state"])
    ckpt = output_dir / "checkpoints" / f"{variant}_seed{train_seed}_best.pt"
    ensure_dir(ckpt.parent)
    tmp = ckpt.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "variant": variant, "train_seed": int(train_seed), "best": best, "config": json_safe(config)}, tmp)
    os.replace(tmp, ckpt)
    return model, {"variant": variant, "train_seed": int(train_seed), "best_epoch": int(best["epoch"]), "best_val_centered_mean": float(best["val_centered_mean"]), "train_log": log}, ckpt


@torch.no_grad()
def nn_predict(
    model: torch.nn.Module,
    *,
    variant: str,
    arm: TransferArm,
    arm_i: int,
    arms: Sequence[TransferArm],
    x: np.ndarray,
    img_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    xb = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    y = arm.projector.measurement(xb)
    r = arm.projector.data_anchor(y)
    out = reconstruct_with_projected_residual(model, arm.projector, r, y, img_size=img_size, cond_features=cond_for_arm(arm_i, arms, variant, xb.shape[0], device))
    return out.x_hat.detach().cpu().numpy().astype(np.float32), out.relmeaserr.detach().cpu().numpy().astype(np.float64)


def scalar_metrics(xhat: np.ndarray, x: np.ndarray, rel: np.ndarray, *, img_size: int) -> dict[str, float]:
    f = full_rmse(xhat, x)
    c = centered_rmse(xhat, x)
    bands = dct_band_rmse(xhat, x, img_size=img_size, low_count=40)
    psnr = -20.0 * np.log10(np.maximum(f, 1e-12))
    return {
        "centered_rmse_mean": float(c.mean()),
        "full_rmse_mean": float(f.mean()),
        "psnr_mean": float(psnr.mean()),
        "dct_non_dc_low_rmse_mean": float(bands["dct_non_dc_low_rmse"].mean()),
        "dct_mid_rmse_mean": float(bands["dct_mid_rmse"].mean()),
        "dct_high_rmse_mean": float(bands["dct_high_rmse"].mean()),
        "relmeaserr_max": float(np.max(rel)),
        "relmeaserr_mean": float(np.mean(rel)),
    }


def add_per_rows(
    out: list[dict[str, Any]],
    *,
    method: str,
    variant_seed: int | str,
    arm: TransferArm,
    xhat: np.ndarray,
    x: np.ndarray,
    rel: np.ndarray,
    source_indices: np.ndarray,
    labels: np.ndarray,
    img_size: int,
) -> dict[str, Any]:
    f = full_rmse(xhat, x)
    c = centered_rmse(xhat, x)
    bands = dct_band_rmse(xhat, x, img_size=img_size, low_count=40)
    psnr = -20.0 * np.log10(np.maximum(f, 1e-12))
    for i in range(x.shape[0]):
        out.append(
            {
                "method": method,
                "variant_seed": variant_seed,
                "arm_id": arm.arm_id,
                "family": arm.family,
                "budget": arm.budget,
                "operator_seed": arm.operator_seed,
                "source_index": int(source_indices[i]),
                "label": int(labels[i]),
                "centered_rmse": float(c[i]),
                "full_rmse": float(f[i]),
                "psnr": float(psnr[i]),
                "dct_non_dc_low_rmse": float(bands["dct_non_dc_low_rmse"][i]),
                "dct_mid_rmse": float(bands["dct_mid_rmse"][i]),
                "dct_high_rmse": float(bands["dct_high_rmse"][i]),
                "relmeaserr": float(rel[i]),
            }
        )
    m = scalar_metrics(xhat, x, rel, img_size=img_size)
    m.update({"method": method, "variant_seed": variant_seed, "arm_id": arm.arm_id, "family": arm.family, "budget": arm.budget, "operator_seed": arm.operator_seed, "n": int(x.shape[0])})
    return m


def clustered_bootstrap_delta(
    per_rows: Sequence[Mapping[str, Any]],
    method: str,
    reference: str,
    metric: str,
    *,
    filter_family: str | None = None,
    filter_budget: int | None = None,
    variant_seed: int | str | None = None,
    reps: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    rows = []
    for r in per_rows:
        if filter_family is not None and str(r["family"]) != str(filter_family):
            continue
        if filter_budget is not None and int(r["budget"]) != int(filter_budget):
            continue
        if variant_seed is not None and str(r["variant_seed"]) not in {str(variant_seed), "baseline"}:
            continue
        if r["method"] in {method, reference}:
            rows.append(r)
    vals: dict[tuple[int, str], dict[str, float]] = {}
    for r in rows:
        key = (int(r["source_index"]), str(r["arm_id"]))
        vals.setdefault(key, {})[str(r["method"])] = float(r[metric])
    deltas_by_image: dict[int, list[float]] = {}
    for (src, _arm), v in vals.items():
        if method in v and reference in v:
            deltas_by_image.setdefault(src, []).append(v[method] - v[reference])
    image_ids = np.asarray(sorted(deltas_by_image), dtype=np.int64)
    image_means = np.asarray([np.mean(deltas_by_image[int(i)]) for i in image_ids], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    boots = []
    for _ in range(int(reps)):
        pick = rng.choice(np.arange(image_means.shape[0]), size=image_means.shape[0], replace=True)
        boots.append(float(image_means[pick].mean()))
    return {
        "method": method,
        "reference": reference,
        "metric": metric,
        "filter_family": filter_family or "ALL",
        "filter_budget": "ALL" if filter_budget is None else int(filter_budget),
        "variant_seed": "ALL" if variant_seed is None else variant_seed,
        "n_images": int(image_means.shape[0]),
        "n_image_operator_pairs": int(sum(len(v) for v in deltas_by_image.values())),
        "mean_delta": float(image_means.mean()),
        "ci_lower": float(np.percentile(boots, 2.5)),
        "ci_upper": float(np.percentile(boots, 97.5)),
        "image_wins": int(np.sum(image_means < 0)),
        "image_losses": int(np.sum(image_means > 0)),
    }


def compute_lpips_subset(per_rows: Sequence[Mapping[str, Any]], predictions: Mapping[tuple[str, str, str], np.ndarray], truths: Mapping[str, np.ndarray], *, device_name: str, max_pairs: int, seed: int, img_size: int) -> list[dict[str, Any]]:
    # predictions keyed by (method, variant_seed, arm_id), truths keyed by arm_id.
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for method in ["empirical_lmmse", "nn_full"]:
        vals = []
        keys = [k for k in predictions if k[0] == method]
        for key in keys:
            arm_id = key[2]
            pred = predictions[key].reshape(-1, img_size, img_size)
            truth = truths[arm_id].reshape(-1, img_size, img_size)
            if pred.shape[0] > max_pairs:
                idx = rng.choice(np.arange(pred.shape[0]), size=int(max_pairs), replace=False)
                pred = pred[idx]
                truth = truth[idx]
            try:
                lp = b0.compute_lpips_matrix(np.clip(pred[:, None], 0.0, 1.0), np.clip(truth, 0.0, 1.0), device_name=device_name).reshape(-1)
                vals.extend(float(v) for v in lp.tolist())
            except Exception:
                return [{"status": "DATA MISSING", "reason": "LPIPS_COMPUTE_FAILED"}]
        rows.append({"status": "PASS", "method": method, "n": int(len(vals)), "lpips_mean": float(np.mean(vals)) if vals else None})
    return rows


def run(config_path: Path) -> dict[str, Any]:
    started = time.time()
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/nonlinear_operator_transfer/dev"))
    reports = output_dir / "reports"
    ensure_dir(reports)
    (output_dir / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    device = resolve_device(str(config.get("device", "cuda")))
    base_seed = int(config.get("seed", 20260625))
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    img_size = int(config["operator_protocol"].get("img_size", 64))
    splits, split_manifest = split_indices(config)
    duplicate_report, duplicate_rows = duplicate_audit(splits)
    write_csv(reports / "sample_hash_audit.csv", duplicate_rows)
    write_json(reports / "duplicate_audit.json", duplicate_report)
    if duplicate_report["status"] != "PASS":
        raise NonlinearTransferError("DUPLICATE_AUDIT_FAILED")

    train_x_t, train_labels, train_seen = load_split_arrays(splits["train"], batch_size=int(config.get("batch_size", 32)), device=device)
    val_x_t, _val_labels, _val_seen = load_split_arrays(splits["val"], batch_size=int(config.get("batch_size", 32)), device=device)
    dev_x_t, dev_labels, dev_seen = load_split_arrays(splits["dev"], batch_size=int(config.get("batch_size", 32)), device=device)
    train_x = train_x_t.numpy().astype(np.float32)
    dev_x = dev_x_t.numpy().astype(np.float32)
    stats = fit_empirical_stats(train_x, klt_rank=int(config["baselines"].get("klt_rank", 128)))

    train_arms = build_arm_pool(config, "train", device=device)
    val_arms = build_arm_pool(config, "val", device=device)
    test_arms = build_arm_pool(config, "test", device=device)

    variants = list(config["training"].get("variants", ["full", "no_condition", "shuffled_condition"]))
    train_seeds = [int(v) for v in config["training"].get("train_seeds", [0, 1, 2])]
    models: dict[tuple[str, int], torch.nn.Module] = {}
    ckpt_rows: list[dict[str, Any]] = []
    train_log_rows: list[dict[str, Any]] = []
    for variant in variants:
        for ts in train_seeds:
            variant_seed_offset = int(hashlib.sha256(variant.encode("utf-8")).hexdigest()[:8], 16) % 997
            model, diag, ckpt = train_variant(
                variant=variant,
                train_seed=base_seed + ts * 1000 + variant_seed_offset,
                train_x=train_x_t,
                val_x=val_x_t,
                train_arms=train_arms,
                val_arms=val_arms,
                config=config,
                device=device,
                output_dir=output_dir,
            )
            models[(variant, ts)] = model
            ckpt_rows.append({"variant": variant, "train_seed_id": ts, "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt), "best_epoch": diag["best_epoch"], "best_val_centered_mean": diag["best_val_centered_mean"]})
            train_log_rows.extend(diag["train_log"])
    write_csv(reports / "checkpoint_manifest.csv", ckpt_rows)
    write_csv(reports / "train_log.csv", train_log_rows)

    metric_rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    predictions_for_lpips: dict[tuple[str, str, str], np.ndarray] = {}
    truths_for_lpips: dict[str, np.ndarray] = {}
    lmmse_lambda = float(config["baselines"].get("lmmse_lambda", 1e-4))
    klt_lambda = float(config["baselines"].get("klt_lambda", 1e-4))
    for arm_i, arm in enumerate(test_arms):
        truths_for_lpips[arm.arm_id] = dev_x
        base_methods = [
            ("joint_minimum_norm",) + joint_predict(dev_x, arm, device=device),
            ("null_mean",) + null_mean_predict(dev_x, arm, stats, device=device),
            ("empirical_lmmse",) + lmmse_predict(dev_x, arm, stats, lambda_=lmmse_lambda, device=device),
            ("klt_ridge",) + klt_predict(dev_x, arm, stats, lambda_=klt_lambda, device=device),
        ]
        for method, xhat, rel in base_methods:
            metric_rows.append(add_per_rows(per_rows, method=method, variant_seed="baseline", arm=arm, xhat=xhat, x=dev_x, rel=rel, source_indices=dev_seen, labels=dev_labels, img_size=img_size))
            if method == "empirical_lmmse":
                predictions_for_lpips[(method, "baseline", arm.arm_id)] = xhat
        for (variant, ts), model in models.items():
            xhat, rel = nn_predict(model, variant=variant, arm=arm, arm_i=arm_i, arms=test_arms, x=dev_x, img_size=img_size, device=device)
            method = f"nn_{variant}"
            metric_rows.append(add_per_rows(per_rows, method=method, variant_seed=ts, arm=arm, xhat=xhat, x=dev_x, rel=rel, source_indices=dev_seen, labels=dev_labels, img_size=img_size))
            if variant == "full":
                predictions_for_lpips[(method, str(ts), arm.arm_id)] = xhat

    write_csv(reports / "method_metrics.csv", metric_rows)
    write_csv(reports / "per_image_operator_metrics.csv", per_rows)

    reps = int(config["statistics"].get("bootstrap_replicates", 500))
    comparisons: list[dict[str, Any]] = []
    for ts in train_seeds:
        comparisons.append(clustered_bootstrap_delta(per_rows, "nn_full", "empirical_lmmse", "centered_rmse", variant_seed=ts, reps=reps, seed=base_seed + 10 + ts))
        comparisons.append(clustered_bootstrap_delta(per_rows, "nn_full", "joint_minimum_norm", "centered_rmse", variant_seed=ts, reps=reps, seed=base_seed + 20 + ts))
        comparisons.append(clustered_bootstrap_delta(per_rows, "nn_full", "nn_no_condition", "centered_rmse", variant_seed=ts, reps=reps, seed=base_seed + 30 + ts))
        comparisons.append(clustered_bootstrap_delta(per_rows, "nn_full", "nn_shuffled_condition", "centered_rmse", variant_seed=ts, reps=reps, seed=base_seed + 40 + ts))
        for fam in config["operator_protocol"]["families"]:
            comparisons.append(clustered_bootstrap_delta(per_rows, "nn_full", "empirical_lmmse", "centered_rmse", filter_family=str(fam), variant_seed=ts, reps=reps, seed=base_seed + 100 + ts))
        for budget in config["operator_protocol"]["budgets"]:
            comparisons.append(clustered_bootstrap_delta(per_rows, "nn_full", "empirical_lmmse", "centered_rmse", filter_budget=int(budget), variant_seed=ts, reps=reps, seed=base_seed + 200 + ts))
    # Risk ladder baselines independent of NN seed.
    for ref_method in ["joint_minimum_norm", "null_mean", "empirical_lmmse"]:
        next_method = {"joint_minimum_norm": "null_mean", "null_mean": "empirical_lmmse", "empirical_lmmse": "nn_full"}[ref_method]
        comparisons.append(clustered_bootstrap_delta(per_rows, next_method, ref_method, "centered_rmse", variant_seed=train_seeds[0] if next_method == "nn_full" else None, reps=reps, seed=base_seed + 300))
    write_json(reports / "clustered_paired_comparisons.json", comparisons)

    lpips_rows = compute_lpips_subset(
        per_rows,
        predictions_for_lpips,
        truths_for_lpips,
        device_name=str(config.get("quality", {}).get("lpips_device", "cuda")),
        max_pairs=int(config.get("quality", {}).get("lpips_max_pairs_per_operator", 16)),
        seed=base_seed + 777,
        img_size=img_size,
    )
    write_csv(reports / "lpips_subset.csv", lpips_rows)

    # Mechanical gate.
    full_vs_lmmse = [c for c in comparisons if c["method"] == "nn_full" and c["reference"] == "empirical_lmmse" and c["filter_family"] == "ALL" and c["filter_budget"] == "ALL"]
    family_cmp = {(c["variant_seed"], c["filter_family"]): c for c in comparisons if c["method"] == "nn_full" and c["reference"] == "empirical_lmmse" and c["filter_family"] != "ALL" and c["filter_budget"] == "ALL"}
    budget_cmp = {(c["variant_seed"], c["filter_budget"]): c for c in comparisons if c["method"] == "nn_full" and c["reference"] == "empirical_lmmse" and c["filter_budget"] != "ALL" and c["filter_family"] == "ALL"}
    cond_cmp = [c for c in comparisons if c["method"] == "nn_full" and c["reference"] in {"nn_no_condition", "nn_shuffled_condition"}]
    seed_success = [c["mean_delta"] < 0 and c["ci_upper"] < 0 for c in full_vs_lmmse]
    heldout_budget = int(config["operator_protocol"].get("heldout_budget", 41))
    heldout_success = [budget_cmp.get((ts, heldout_budget), {"mean_delta": 1, "ci_upper": 1})["mean_delta"] < 0 for ts in train_seeds]
    random_success = [family_cmp.get((ts, "random"), {"ci_upper": 1})["ci_upper"] < 0 for ts in train_seeds]
    dct_or_mixed_success = [
        (family_cmp.get((ts, "dct"), {"ci_upper": 1})["ci_upper"] < 0) or (family_cmp.get((ts, "mixed"), {"ci_upper": 1})["ci_upper"] < 0)
        for ts in train_seeds
    ]
    condition_success = [c["mean_delta"] < 0 and c["ci_upper"] < 0 for c in cond_cmp]
    rel_max = max(float(r["relmeaserr_max"]) for r in metric_rows if str(r["method"]).startswith("nn_") or r["method"] in {"empirical_lmmse", "null_mean", "klt_ridge"})
    lpips_status = "DATA MISSING"
    lpips_not_worse = False
    if all(r.get("status") == "PASS" for r in lpips_rows):
        means = {r["method"]: float(r["lpips_mean"]) for r in lpips_rows if r.get("lpips_mean") is not None}
        if "empirical_lmmse" in means and "nn_full" in means:
            lpips_status = "PASS"
            lpips_not_worse = bool(means["nn_full"] <= means["empirical_lmmse"] + float(config.get("gate", {}).get("lpips_nonworse_margin", 0.02)))
    gate_conditions = {
        "shared_unseen_ops_beats_lmmse_2_of_3": int(sum(seed_success)) >= 2,
        "random_family_ci_upper_lt_zero_2_of_3": int(sum(random_success)) >= 2,
        "dct_or_mixed_ci_upper_lt_zero_2_of_3": int(sum(dct_or_mixed_success)) >= 2,
        "heldout_budget_not_reversed_2_of_3": int(sum(heldout_success)) >= 2,
        "condition_ablation_full_better": int(sum(condition_success)) >= max(2, len(cond_cmp) // 2),
        "measurement_consistency_ok": bool(rel_max < float(config.get("gate", {}).get("relmeaserr_max", 1e-4))),
        "duplicate_audit_ok": duplicate_report["status"] == "PASS",
        "lpips_not_obviously_worse": bool(lpips_not_worse),
    }
    if all(gate_conditions.values()):
        classification = "NONLINEAR_OPERATOR_TRANSFER_CONFIRMED"
    elif not gate_conditions["shared_unseen_ops_beats_lmmse_2_of_3"]:
        classification = "LINEAR_COVARIANCE_EXPLAINS_GAIN"
    elif not gate_conditions["heldout_budget_not_reversed_2_of_3"]:
        classification = "FIXED_OPERATOR_ONLY"
    elif not gate_conditions["condition_ablation_full_better"]:
        classification = "CONDITION_UNUSED"
    else:
        classification = "DEVELOPMENT_INCONCLUSIVE"
    gate = {
        "status": "PASS",
        "classification": classification,
        "locked_test_authorized": classification == "NONLINEAR_OPERATOR_TRANSFER_CONFIRMED",
        "conditions": gate_conditions,
        "seed_success_vs_lmmse": seed_success,
        "random_success": random_success,
        "dct_or_mixed_success": dct_or_mixed_success,
        "heldout_budget_success": heldout_success,
        "condition_success_count": int(sum(condition_success)),
        "condition_comparison_count": int(len(condition_success)),
        "relmeaserr_max_all_audited": rel_max,
        "lpips_status": lpips_status,
        "scope": "fresh development only; no final-v4 and no existing locked test used",
    }
    write_json(reports / "gate_report.json", gate)

    write_json(
        reports / "lineage_and_operator_audit.json",
        {
            "status": "PASS",
            "split_manifest": split_manifest,
            "duplicate_audit": duplicate_report,
            "repo_state": repo_state(),
            "arms": [
                {
                    "arm_id": a.arm_id,
                    "family": a.family,
                    "budget": a.budget,
                    "pool": a.pool,
                    "operator_seed": a.operator_seed,
                    "rows_sha256": sha256_np(a.rows),
                    "row_audit": row_audit(a.rows, name=a.arm_id),
                    "projector": a.projector.diagnostics(),
                }
                for a in [*train_arms, *val_arms, *test_arms]
            ],
            "final_v4_or_existing_locked_used_for_selection": False,
        },
    )

    write_math_and_summary(reports, gate, comparisons, lpips_rows, config)
    runtime = {
        "status": "PASS",
        "elapsed_seconds": float(time.time() - started),
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "artifact_hashes": {
            "config_used.yaml": sha256_file(output_dir / "config_used.yaml"),
            "method_metrics.csv": sha256_file(reports / "method_metrics.csv"),
            "per_image_operator_metrics.csv": sha256_file(reports / "per_image_operator_metrics.csv"),
            "clustered_paired_comparisons.json": sha256_file(reports / "clustered_paired_comparisons.json"),
            "gate_report.json": sha256_file(reports / "gate_report.json"),
            "lineage_and_operator_audit.json": sha256_file(reports / "lineage_and_operator_audit.json"),
        },
    }
    write_json(reports / "runtime_and_hashes.json", runtime)
    summary = {
        "status": "NONLINEAR_OPERATOR_TRANSFER_COMPLETE",
        "output_dir": str(output_dir),
        "gate": gate,
        "runtime": runtime,
        "key_artifacts": {
            "decision": str(reports / "research_decision.md"),
            "gate_report": str(reports / "gate_report.json"),
            "method_metrics": str(reports / "method_metrics.csv"),
            "per_image_operator_metrics": str(reports / "per_image_operator_metrics.csv"),
            "clustered_comparisons": str(reports / "clustered_paired_comparisons.json"),
            "lineage": str(reports / "lineage_and_operator_audit.json"),
        },
    }
    write_json(reports / "summary.json", summary)
    atomic_json(output_dir / "NONLINEAR_OPERATOR_TRANSFER_COMPLETE.json", {"status": summary["status"], "classification": classification, "summary_sha256": sha256_file(reports / "summary.json")})
    return summary


def write_math_and_summary(reports: Path, gate: Mapping[str, Any], comparisons: Sequence[Mapping[str, Any]], lpips_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    math_text = r"""# LMMSE and Nonlinear Operator Transfer Math

For a fixed final operator \(A\), \(r=A^\dagger y\) and \(P_0^A=I-A^\dagger A\).
Minimum norm is \(r\). The affine null-mean completion is
\[
\hat x_\mu=r+P_0^A\mu.
\]
For empirical mean \(\mu\) and centered training matrix \(Z\), the LMMSE estimator is
\[
\hat x_L=\mu+C A^\top(A C A^\top+\lambda I)^{-1}(y-A\mu).
\]
The implementation never forms \(C\in\mathbb R^{n\times n}\). It computes
\[
U=ZA^\top,\quad A C A^\top=U^\top U/(N-1),\quad
C A^\top=Z^\top U/(N-1).
\]
All non-exact affine outputs are then exactly audited as \(A^\dagger y+P_0^A\hat x\).

The nonlinear model is
\[
\hat x_\theta=A^\dagger y+P_0^A f_\theta(r,E(A)),
\]
so \(A\hat x_\theta=y\) because \(A P_0^A=0\). The risk ladder is
\[
R_{\rm MN}\to R_\mu\to R_{\rm LMMSE}\to R_\theta.
\]
Only \(R_\theta<R_{\rm LMMSE}\) on unseen operators supports nonlinear prior gain beyond training-set covariance.
"""
    (reports / "math_derivation.md").write_text(math_text, encoding="utf-8")
    lines = [
        "# Nonlinear Operator Transfer Decision",
        "",
        f"- Classification: `{gate['classification']}`",
        f"- Locked test authorized: `{str(gate['locked_test_authorized']).lower()}`",
        f"- Scope: `{gate['scope']}`",
        "",
        "## Mechanical Gate",
        "",
    ]
    for k, v in gate["conditions"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.extend(["", "## Key Clustered Comparisons", ""])
    for c in comparisons:
        if c["method"] == "nn_full" and c["reference"] in {"empirical_lmmse", "nn_no_condition", "nn_shuffled_condition"}:
            lines.append(
                f"- seed `{c['variant_seed']}`, family `{c['filter_family']}`, budget `{c['filter_budget']}`, "
                f"`{c['method']}-{c['reference']}` {c['metric']} delta `{c['mean_delta']:.6f}` "
                f"CI `[{c['ci_lower']:.6f}, {c['ci_upper']:.6f}]`, images `{c['n_images']}`"
            )
    lines.extend(["", "## LPIPS Subset", ""])
    for r in lpips_rows:
        lines.append(f"- `{r}`")
    lines.extend(
        [
            "",
            "## Claim-Evidence Ledger",
            "",
            "| Claim | Evidence | Status |",
            "|---|---|---|",
            "| LMMSE baseline is matrix-free | `math_derivation.md` and code use \(Z A^T\), \(U^T U\), and \(Z^T U\), never dense \(C\). | supported |",
            "| Measurement consistency is exact-audited | `gate_report.json` records max RelMeasErr across audited methods. | supported |",
            "| Split/duplicate leakage is controlled | `duplicate_audit.json` and `sample_hash_audit.csv` include raw and transformed SHA256 checks. | supported if gate passes |",
            "| Nonlinear gain beyond covariance | Requires `shared_unseen_ops_beats_lmmse_2_of_3`. | see gate |",
            "| Operator condition is used | Requires full condition to beat no-condition and shuffled-condition. | see gate |",
            "",
            "## Unique Recommended Next Step",
            "",
        ]
    )
    if gate["classification"] == "NONLINEAR_OPERATOR_TRANSFER_CONFIRMED":
        lines.append("Prepare a frozen locked-test protocol with the exact same risk ladder, unseen operator pools, held-out budget, condition ablations, and duplicate audit. Do not tune further on this development result.")
    elif gate["classification"] == "LINEAR_COVARIANCE_EXPLAINS_GAIN":
        lines.append("Stop claiming nonlinear prior superiority. Reframe the result around empirical covariance/LMMSE sensor priors, then decide whether a nonlinear architecture needs stronger data/conditioning before any locked test.")
    elif gate["classification"] == "CONDITION_UNUSED":
        lines.append("Redesign operator encoding before any locked test; current transfer evidence cannot support operator-conditioned claims.")
    else:
        lines.append("Diagnose the failed gate item before locked testing; keep this as development evidence only.")
    (reports / "research_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (reports / "claim_evidence_ledger.md").write_text("\n".join(lines[lines.index("## Claim-Evidence Ledger") :]) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fresh-development protocol for nonlinear prior gain and unseen-operator transfer.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(Path(args.config))
    print(json.dumps(json_safe({"status": summary["status"], "classification": summary["gate"]["classification"], "output_dir": summary["output_dir"], "key_artifacts": summary["key_artifacts"]}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
