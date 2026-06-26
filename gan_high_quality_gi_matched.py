from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

import gan_high_quality_gi as hq


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "gan_high_quality_gi_matched_dev_64_5pct.yaml"
METHOD_NO_GAN = "matched_no_gan"
METHOD_GAN = "matched_gan"


def _hash_update_obj(hasher: Any, obj: Any) -> None:
    if torch.is_tensor(obj):
        t = obj.detach().cpu().contiguous()
        hasher.update(str(t.dtype).encode())
        hasher.update(str(tuple(t.shape)).encode())
        hasher.update(t.numpy().tobytes())
    elif isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        hasher.update(str(arr.dtype).encode())
        hasher.update(str(arr.shape).encode())
        hasher.update(arr.tobytes())
    elif isinstance(obj, Mapping):
        for key in sorted(obj):
            hasher.update(str(key).encode())
            _hash_update_obj(hasher, obj[key])
    elif isinstance(obj, (list, tuple)):
        hasher.update(str(type(obj)).encode())
        for item in obj:
            _hash_update_obj(hasher, item)
    else:
        hasher.update(repr(obj).encode())


def stable_hash(obj: Any) -> str:
    h = hashlib.sha256()
    _hash_update_obj(h, obj)
    return h.hexdigest()


def pickle_hash(obj: Any) -> str:
    return hashlib.sha256(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def rng_state_hash(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(pickle.dumps(state["python"], protocol=pickle.HIGHEST_PROTOCOL))
    np_state = state["numpy"]
    h.update(str(np_state[0]).encode())
    h.update(np.ascontiguousarray(np_state[1]).tobytes())
    h.update(str(np_state[2:]).encode())
    h.update(state["torch"].detach().cpu().contiguous().numpy().tobytes())
    for cuda_state in state.get("cuda", []):
        h.update(cuda_state.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    hq.ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def build_generator_state(config: Mapping[str, Any], measurement: hq.GhostMeasurementOperator, device: torch.device):
    model_cfg = dict(config["model"])
    model = hq.build_generator(
        {"model_type": model_cfg.get("model_type", "hq_unet"), "base_channels": int(model_cfg.get("base_channels", 32))},
        measurement=measurement,
    ).to(device)
    hq.zero_init_residual_head(model)
    ema = hq.ModelEMA(model, decay=float(model_cfg.get("ema_decay", 0.995)))
    train_cfg = dict(config["training"])
    opt_g = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr_g", 2e-4)), betas=tuple(train_cfg.get("betas", [0.5, 0.9])))
    scaler_g = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    return model, ema, opt_g, scaler_g


def make_discriminator(config: Mapping[str, Any], device: torch.device, seed: int) -> nn.Module:
    devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        return hq.DualPatchDiscriminator().to(device)


def save_generator_branch_state(
    path: Path,
    *,
    model: nn.Module,
    ema: hq.ModelEMA,
    opt_g: torch.optim.Optimizer,
    scaler_g: torch.cuda.amp.GradScaler,
    config: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> None:
    rng = capture_rng_state()
    payload = {
        "generator": model.state_dict(),
        "generator_ema": ema.module.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "amp_scaler_g": scaler_g.state_dict(),
        "scheduler": None,
        "rng_state": rng,
        "config": hq.json_safe(config),
        "meta": hq.json_safe(
            {
                **dict(meta),
                "generator_hash": stable_hash(model.state_dict()),
                "generator_ema_hash": stable_hash(ema.module.state_dict()),
                "optimizer_g_hash": stable_hash(opt_g.state_dict()),
                "amp_scaler_g_hash": stable_hash(scaler_g.state_dict()),
                "rng_state_hash": rng_state_hash(rng),
            }
        ),
        "saved_utc": hq.now_utc(),
    }
    atomic_torch_save(path, payload)


def load_generator_branch_state(
    path: Path,
    *,
    config: Mapping[str, Any],
    measurement: hq.GhostMeasurementOperator,
    device: torch.device,
):
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device)
    model, ema, opt_g, scaler_g = build_generator_state(config, measurement, device)
    model.load_state_dict(payload["generator"])
    ema.module.load_state_dict(payload["generator_ema"])
    opt_g.load_state_dict(payload["optimizer_g"])
    scaler_g.load_state_dict(payload.get("amp_scaler_g", {}))
    restore_rng_state(payload["rng_state"])
    return model, ema, opt_g, scaler_g, payload


def train_epochs(
    *,
    phase: str,
    model: nn.Module,
    ema: hq.ModelEMA,
    opt_g: torch.optim.Optimizer,
    scaler_g: torch.cuda.amp.GradScaler,
    disc: nn.Module | None,
    opt_d: torch.optim.Optimizer | None,
    scaler_d: torch.cuda.amp.GradScaler | None,
    lmmse: hq.EmpiricalLMMSE,
    measurement: hq.GhostMeasurementOperator,
    train_ds: torch.utils.data.Dataset,
    config: Mapping[str, Any],
    device: torch.device,
    epochs: int,
    shuffle_seed: int,
    gan_active: bool,
    lambda_adv: float,
    max_batches: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_cfg = dict(config["training"])
    logs: list[dict[str, Any]] = []
    batch_hasher = hashlib.sha256()
    g_updates = 0
    d_updates = 0
    first_batches: list[list[int]] = []
    adversarial_enabled = bool(gan_active and disc is not None and abs(float(lambda_adv)) > 0.0)
    for epoch in range(int(epochs)):
        loader = hq.build_loader(
            train_ds,
            batch_size=int(config["data"]["batch_size"]),
            workers=int(config["data"].get("num_workers", 0)),
            shuffle=True,
            seed=int(shuffle_seed) + epoch,
            device=device,
        )
        model.train()
        if disc is not None:
            disc.train()
        losses: list[dict[str, float]] = []
        for batch_i, (x, _label, source_idx) in enumerate(loader):
            if max_batches is not None and batch_i >= int(max_batches):
                break
            idx_np = source_idx.detach().cpu().numpy().astype(np.int64)
            batch_hasher.update(idx_np.tobytes())
            if len(first_batches) < 5:
                first_batches.append([int(v) for v in idx_np[: min(12, len(idx_np))]])
            x = x.to(device, non_blocking=True)
            flat = measurement.flatten_img(x)
            y = measurement.A_forward(flat)
            with torch.no_grad():
                x0_flat = lmmse.anchor(y, measurement, device=device)
                uncertainty = lmmse.uncertainty_map(
                    img_size=int(config["data"].get("img_size", 64)),
                    device=device,
                    batch_size=x.shape[0],
                    dtype=x.dtype,
                )
            if adversarial_enabled:
                assert opt_d is not None and scaler_d is not None
                for p in disc.parameters():
                    p.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler_d.is_enabled()):
                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        fake = hq.generator_forward(model, x0_flat, uncertainty, measurement).detach().clamp(0, 1)
                    model.train(was_training)
                    real_score = disc(x)
                    fake_score = disc(fake)
                    d_loss = hq.hinge_d_loss(real_score, fake_score)
                    if float(train_cfg.get("r1_gamma", 0.0)) > 0:
                        d_loss = d_loss + 0.5 * float(train_cfg.get("r1_gamma", 0.0)) * hq.r1_penalty(disc, x)
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
                d_updates += 1
            else:
                d_loss = torch.zeros((), device=device)

            if disc is not None:
                for p in disc.parameters():
                    p.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler_g.is_enabled()):
                xhat = hq.generator_forward(model, x0_flat, uncertainty, measurement)
                base_loss, parts = hq.image_losses(xhat.clamp(0, 1), x, train_cfg)
                adv = torch.zeros((), device=device)
                if adversarial_enabled:
                    adv = hq.hinge_g_loss(disc(xhat.clamp(0, 1)))
                g_loss = base_loss + float(lambda_adv) * adv
            scaler_g.scale(g_loss).backward()
            if float(train_cfg.get("grad_clip", 0.0)) > 0:
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            scaler_g.step(opt_g)
            scaler_g.update()
            ema.update(model)
            g_updates += 1
            parts.update({"g_loss": float(g_loss.detach().cpu()), "d_loss": float(d_loss.detach().cpu()), "adv": float(adv.detach().cpu())})
            losses.append(parts)
        row: dict[str, Any] = {
            "phase": phase,
            "epoch": epoch + 1,
            "gan_active": bool(gan_active),
            "g_updates_cumulative": int(g_updates),
            "d_updates_cumulative": int(d_updates),
            "shuffle_seed": int(shuffle_seed) + epoch,
        }
        for key in sorted({k for item in losses for k in item}):
            row[key] = float(np.mean([item.get(key, 0.0) for item in losses]))
        logs.append(row)
    manifest = {
        "phase": phase,
        "epochs": int(epochs),
        "g_updates": int(g_updates),
        "d_updates": int(d_updates),
        "batch_order_hash": batch_hasher.hexdigest(),
        "first_batches": first_batches,
        "shuffle_seed_base": int(shuffle_seed),
        "lambda_adv": float(lambda_adv),
        "gan_active": bool(gan_active),
    }
    return logs, manifest


def run_equivalence_test(
    *,
    config: Mapping[str, Any],
    lmmse: hq.EmpiricalLMMSE,
    measurement: hq.GhostMeasurementOperator,
    train_ds: torch.utils.data.Dataset,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    mcfg = dict(config["matched_training"])
    seed = int(mcfg.get("equivalence_seed", 987))
    max_batches = int(mcfg.get("equivalence_max_batches", 2))
    tol = float(mcfg.get("equivalence_tolerance", 1e-7))
    hq.set_seed(int(config["seed"]) + 900000 + seed)
    model, ema, opt_g, scaler_g = build_generator_state(config, measurement, device)
    state_path = output_dir / "equivalence" / "equivalence_start_state.pt"
    save_generator_branch_state(state_path, model=model, ema=ema, opt_g=opt_g, scaler_g=scaler_g, config=config, meta={"phase": "equivalence_start", "seed": seed})
    out: dict[str, Any] = {"seed": seed, "max_batches": max_batches, "tolerance": tol}
    final_hashes: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for branch in [METHOD_NO_GAN, METHOD_GAN]:
        model_b, ema_b, opt_b, scaler_b, payload = load_generator_branch_state(state_path, config=config, measurement=measurement, device=device)
        disc = None
        opt_d = None
        scaler_d = None
        if branch == METHOD_GAN:
            disc = make_discriminator(config, device, int(config["seed"]) + int(mcfg.get("discriminator_seed_offset", 410000)) + seed)
            restore_rng_state(payload["rng_state"])
            opt_d = torch.optim.Adam(disc.parameters(), lr=float(config["training"].get("lr_d", 2e-4)), betas=tuple(config["training"].get("betas", [0.5, 0.9])))
            scaler_d = torch.cuda.amp.GradScaler(enabled=bool(config["training"].get("amp", True)) and device.type == "cuda")
        logs, manifest = train_epochs(
            phase=f"equivalence_{branch}",
            model=model_b,
            ema=ema_b,
            opt_g=opt_b,
            scaler_g=scaler_b,
            disc=disc,
            opt_d=opt_d,
            scaler_d=scaler_d,
            lmmse=lmmse,
            measurement=measurement,
            train_ds=train_ds,
            config=config,
            device=device,
            epochs=1,
            shuffle_seed=int(config["seed"]) + int(mcfg.get("branch_shuffle_seed_offset", 310000)) + seed,
            gan_active=branch == METHOD_GAN,
            lambda_adv=0.0,
            max_batches=max_batches,
        )
        final_hashes[branch] = {
            "generator_hash": stable_hash(model_b.state_dict()),
            "generator_ema_hash": stable_hash(ema_b.module.state_dict()),
            "optimizer_g_hash": stable_hash(opt_b.state_dict()),
            "logs": logs,
        }
        manifests[branch] = manifest
    diffs = {}
    for label in ["generator", "generator_ema"]:
        a_path = "generator_hash" if label == "generator" else "generator_ema_hash"
        diffs[f"{label}_hash_equal"] = final_hashes[METHOD_NO_GAN][a_path] == final_hashes[METHOD_GAN][a_path]
    diffs["batch_order_hash_equal"] = manifests[METHOD_NO_GAN]["batch_order_hash"] == manifests[METHOD_GAN]["batch_order_hash"]
    diffs["g_update_count_equal"] = manifests[METHOD_NO_GAN]["g_updates"] == manifests[METHOD_GAN]["g_updates"]
    # Hash equality is the strictest check; max_abs is reported as zero only if hashes match.
    passed = all(bool(v) for v in diffs.values())
    out.update({"status": "PASS" if passed else "FAIL", "diffs": diffs, "final_hashes": final_hashes, "branch_manifests": manifests})
    hq.write_json(output_dir / "reports" / "lambda_adv0_equivalence_test.json", out)
    return out


def train_matched_seed(
    *,
    train_seed: int,
    config: Mapping[str, Any],
    lmmse: hq.EmpiricalLMMSE,
    measurement: hq.GhostMeasurementOperator,
    train_ds: torch.utils.data.Dataset,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    mcfg = dict(config["matched_training"])
    base_seed = int(config["seed"]) + 1000 * int(train_seed)
    hq.set_seed(base_seed)
    seed_dir = hq.ensure_dir(output_dir / "runs" / f"seed{train_seed}")
    model, ema, opt_g, scaler_g = build_generator_state(config, measurement, device)
    pre_logs, pre_manifest = train_epochs(
        phase="shared_pretrain",
        model=model,
        ema=ema,
        opt_g=opt_g,
        scaler_g=scaler_g,
        disc=None,
        opt_d=None,
        scaler_d=None,
        lmmse=lmmse,
        measurement=measurement,
        train_ds=train_ds,
        config=config,
        device=device,
        epochs=int(mcfg.get("pretrain_epochs", 3)),
        shuffle_seed=int(config["seed"]) + int(mcfg.get("pretrain_shuffle_seed_offset", 210000)) + int(train_seed),
        gan_active=False,
        lambda_adv=0.0,
    )
    pre_state = seed_dir / "branch_state" / f"seed{train_seed}_shared_pretrain_state.pt"
    save_generator_branch_state(
        pre_state,
        model=model,
        ema=ema,
        opt_g=opt_g,
        scaler_g=scaler_g,
        config=config,
        meta={"phase": "shared_pretrain_complete", "train_seed": int(train_seed), "pretrain_manifest": pre_manifest},
    )
    pre_payload = torch.load(pre_state, map_location=device)
    branch_manifests: list[dict[str, Any]] = []
    branch_epochs = int(mcfg.get("branch_epochs", 3))
    branch_seed = int(config["seed"]) + int(mcfg.get("branch_shuffle_seed_offset", 310000)) + int(train_seed)
    start_hashes: dict[str, dict[str, str]] = {}
    for branch in [METHOD_NO_GAN, METHOD_GAN]:
        model_b, ema_b, opt_b, scaler_b, payload = load_generator_branch_state(pre_state, config=config, measurement=measurement, device=device)
        start_hashes[branch] = {
            "generator_hash": stable_hash(model_b.state_dict()),
            "generator_ema_hash": stable_hash(ema_b.module.state_dict()),
            "optimizer_g_hash": stable_hash(opt_b.state_dict()),
            "amp_scaler_g_hash": stable_hash(scaler_b.state_dict()),
            "rng_state_hash": rng_state_hash(payload["rng_state"]),
        }
        disc = None
        opt_d = None
        scaler_d = None
        if branch == METHOD_GAN:
            disc_seed = int(config["seed"]) + int(mcfg.get("discriminator_seed_offset", 410000)) + int(train_seed)
            disc = make_discriminator(config, device, disc_seed)
            restore_rng_state(payload["rng_state"])
            opt_d = torch.optim.Adam(disc.parameters(), lr=float(config["training"].get("lr_d", 2e-4)), betas=tuple(config["training"].get("betas", [0.5, 0.9])))
            scaler_d = torch.cuda.amp.GradScaler(enabled=bool(config["training"].get("amp", True)) and device.type == "cuda")
        logs, manifest = train_epochs(
            phase=branch,
            model=model_b,
            ema=ema_b,
            opt_g=opt_b,
            scaler_g=scaler_b,
            disc=disc,
            opt_d=opt_d,
            scaler_d=scaler_d,
            lmmse=lmmse,
            measurement=measurement,
            train_ds=train_ds,
            config=config,
            device=device,
            epochs=branch_epochs,
            shuffle_seed=branch_seed,
            gan_active=branch == METHOD_GAN,
            lambda_adv=float(config["training"].get("lambda_adv", 0.01)) if branch == METHOD_GAN else 0.0,
        )
        run_dir = hq.ensure_dir(seed_dir / branch)
        hq.write_csv(run_dir / "train_log.csv", logs)
        ckpt = run_dir / "checkpoints" / f"{branch}_seed{train_seed}_final.pt"
        hq.save_checkpoint(ckpt, model_b, ema_b, disc, config, {"final": True, "branch_manifest": manifest, "train_seed": int(train_seed), "method": branch})
        branch_manifests.append(
            {
                "method": branch,
                "train_seed": int(train_seed),
                "checkpoint": str(ckpt),
                "checkpoint_sha256": hq.sha256_file(ckpt),
                "branch_manifest": manifest,
                "final_generator_hash": stable_hash(model_b.state_dict()),
                "final_generator_ema_hash": stable_hash(ema_b.module.state_dict()),
                "final_optimizer_g_hash": stable_hash(opt_b.state_dict()),
                "disc_rng_independent": branch == METHOD_GAN,
            }
        )
    fairness = {
        "train_seed": int(train_seed),
        "shared_pretrain_state": str(pre_state),
        "shared_pretrain_state_sha256": hq.sha256_file(pre_state),
        "pretrain_manifest": pre_manifest,
        "pretrain_log": pre_logs,
        "branch_start_hashes": start_hashes,
        "branch_start_generator_equal": start_hashes[METHOD_NO_GAN]["generator_hash"] == start_hashes[METHOD_GAN]["generator_hash"],
        "branch_start_optimizer_equal": start_hashes[METHOD_NO_GAN]["optimizer_g_hash"] == start_hashes[METHOD_GAN]["optimizer_g_hash"],
        "branch_start_rng_equal": start_hashes[METHOD_NO_GAN]["rng_state_hash"] == start_hashes[METHOD_GAN]["rng_state_hash"],
        "branch_manifests": branch_manifests,
        "branch_batch_order_equal": branch_manifests[0]["branch_manifest"]["batch_order_hash"] == branch_manifests[1]["branch_manifest"]["batch_order_hash"],
        "branch_g_updates_equal": branch_manifests[0]["branch_manifest"]["g_updates"] == branch_manifests[1]["branch_manifest"]["g_updates"],
        "branch_ema_updates_equal": branch_manifests[0]["branch_manifest"]["g_updates"] == branch_manifests[1]["branch_manifest"]["g_updates"],
        "scheduler": None,
        "augmentation": "deterministic_resize_grayscale_no_random_augmentation",
    }
    fairness["status"] = "PASS" if all(
        [
            fairness["branch_start_generator_equal"],
            fairness["branch_start_optimizer_equal"],
            fairness["branch_start_rng_equal"],
            fairness["branch_batch_order_equal"],
            fairness["branch_g_updates_equal"],
            fairness["branch_ema_updates_equal"],
        ]
    ) else "FAIL"
    hq.write_json(seed_dir / "fairness_manifest.json", fairness)
    return fairness


def load_eval_model(checkpoint: Path, *, config: Mapping[str, Any], measurement: hq.GhostMeasurementOperator, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location=device)
    model_cfg = dict(config["model"])
    model = hq.build_generator(
        {"model_type": model_cfg.get("model_type", "hq_unet"), "base_channels": int(model_cfg.get("base_channels", 32))},
        measurement=measurement,
    ).to(device)
    state = payload.get("generator_ema") or payload["generator"]
    model.load_state_dict(state)
    model.eval()
    return model


def paired_image_bootstrap(per_rows: Sequence[Mapping[str, Any]], method: str, reference: str, metric: str, *, higher_is_better: bool, reps: int, seed: int) -> dict[str, Any]:
    by: dict[int, dict[int, dict[str, float]]] = {}
    ref_vals: list[float] = []
    for row in per_rows:
        if row["method"] not in {method, reference}:
            continue
        try:
            value = float(row[metric])
        except (TypeError, ValueError):
            continue
        sample = int(row["sample_ordinal"])
        train_seed = int(row["train_seed"])
        by.setdefault(sample, {}).setdefault(train_seed, {})[str(row["method"])] = value
        if row["method"] == reference:
            ref_vals.append(value)
    image_deltas: list[float] = []
    seed_deltas: dict[int, list[float]] = {}
    for sample, seed_map in by.items():
        ds = []
        for train_seed, vals in seed_map.items():
            if method in vals and reference in vals:
                delta = vals[method] - vals[reference]
                ds.append(delta)
                seed_deltas.setdefault(train_seed, []).append(delta)
        if ds:
            image_deltas.append(float(np.mean(ds)))
    if not image_deltas:
        return {"method": method, "reference": reference, "metric": metric, "status": "NO_PAIRS"}
    arr = np.asarray(image_deltas, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    boots = np.asarray([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(int(reps))], dtype=np.float64)
    seed_summary = []
    for train_seed in sorted(seed_deltas):
        sarr = np.asarray(seed_deltas[train_seed], dtype=np.float64)
        seed_summary.append(
            {
                "train_seed": int(train_seed),
                "mean_delta": float(sarr.mean()),
                "direction_good": bool(sarr.mean() > 0 if higher_is_better else sarr.mean() < 0),
                "n_images": int(len(sarr)),
            }
        )
    mean_delta = float(arr.mean())
    ref_mean = float(np.mean(ref_vals)) if ref_vals else float("nan")
    good = arr > 0 if higher_is_better else arr < 0
    return {
        "method": method,
        "reference": reference,
        "metric": metric,
        "status": "PASS",
        "bootstrap": "image_level_mean_over_train_seeds",
        "n_images": int(len(arr)),
        "n_seed_image_pairs": int(sum(len(v) for v in seed_deltas.values())),
        "mean_delta": mean_delta,
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "reference_mean": ref_mean,
        "relative_gain": float((-mean_delta / ref_mean) if (not higher_is_better and ref_mean > 0) else (mean_delta / abs(ref_mean) if ref_mean else float("nan"))),
        "wins_method": int(good.sum()),
        "wins_reference": int((~good).sum()),
        "seed_summary": seed_summary,
    }


def summarize_matched_gate(all_per: Sequence[Mapping[str, Any]], all_method: Sequence[Mapping[str, Any]], fairness: Sequence[Mapping[str, Any]], equivalence: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    eval_cfg = dict(config["eval"])
    reps = int(eval_cfg.get("bootstrap_replicates", 1000))
    seed = int(eval_cfg.get("bootstrap_seed", 20260626))
    metrics = [
        ("lpips", False),
        ("rapsd", False),
        ("full_rmse", False),
        ("centered_rmse", False),
        ("psnr", True),
        ("ssim", True),
    ]
    comparisons = [
        paired_image_bootstrap(all_per, METHOD_GAN, METHOD_NO_GAN, metric, higher_is_better=higher, reps=reps, seed=seed + i)
        for i, (metric, higher) in enumerate(metrics)
    ]
    comp_by_metric = {c["metric"]: c for c in comparisons if c.get("status") == "PASS"}
    lp = comp_by_metric.get("lpips", {})
    lpips_gate = bool(lp and lp["relative_gain"] >= float(eval_cfg.get("lpips_relative_gain_gate", 0.05)) and lp["ci_high"] < 0)
    seed_dirs = [bool(s["direction_good"]) for s in lp.get("seed_summary", [])]
    two_seed_gate = sum(seed_dirs) >= 2
    rapsd = comp_by_metric.get("rapsd", {})
    rapsd_gate = bool(rapsd and rapsd["mean_delta"] < 0)
    psnr = comp_by_metric.get("psnr", {})
    psnr_gate = bool(psnr and psnr["mean_delta"] >= -float(eval_cfg.get("psnr_drop_tolerance_db", 0.5)))
    rels = []
    for row in all_per:
        if row["method"] in {METHOD_GAN, METHOD_NO_GAN}:
            rels.append(float(row["relmeaserr"]))
    rel_gate = bool(rels and max(rels) <= float(eval_cfg.get("relmeaserr_limit", 1e-5)))
    method_by_seed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in all_method:
        if row["method"] in {METHOD_GAN, METHOD_NO_GAN}:
            method_by_seed[(int(row["train_seed"]), str(row["method"]))] = row
    kid_dirs = []
    fid_dirs = []
    for train_seed in sorted({k[0] for k in method_by_seed}):
        a = method_by_seed.get((train_seed, METHOD_GAN))
        b = method_by_seed.get((train_seed, METHOD_NO_GAN))
        if not a or not b:
            continue
        try:
            kid_dirs.append(float(a["kid"]) < float(b["kid"]))
            fid_dirs.append(float(a["fid"]) < float(b["fid"]))
        except (TypeError, ValueError):
            pass
    kid_gate = bool(kid_dirs and sum(kid_dirs) >= 2)
    fair_gate = bool(fairness and all(f.get("status") == "PASS" for f in fairness))
    eq_gate = equivalence.get("status") == "PASS"
    conditions = {
        "lpips_relative_gain_ge_5pct_and_ci_upper_lt_0": lpips_gate,
        "at_least_2_of_3_seeds_lpips_same_direction": two_seed_gate,
        "rapsd_same_direction": rapsd_gate,
        "kid_same_direction": kid_gate,
        "psnr_not_below_tolerance": psnr_gate,
        "measurement_consistency": rel_gate,
        "fairness_hash_audit": fair_gate,
        "lambda_adv0_equivalence": eq_gate,
    }
    if not fair_gate or not eq_gate:
        classification = "INVALID_MATCHED_EXPERIMENT"
    elif lpips_gate and two_seed_gate and rapsd_gate and kid_gate and rel_gate and psnr_gate:
        classification = "MATCHED_ADVERSARIAL_GAIN_CONFIRMED"
    elif lpips_gate and two_seed_gate and rel_gate and not psnr_gate:
        classification = "GAN_GAIN_WITH_TRADEOFF"
    elif not lpips_gate:
        classification = "EXTRA_TRAINING_EXPLAINS_OLD_GAIN"
    else:
        classification = "GAN_GAIN_NOT_CONFIRMED"
    return {
        "classification": classification,
        "development_gate_passed": classification in {"MATCHED_ADVERSARIAL_GAIN_CONFIRMED", "GAN_GAIN_WITH_TRADEOFF"},
        "comparisons": comparisons,
        "conditions": conditions,
        "seed_lpips_better": seed_dirs,
        "kid_seed_better": kid_dirs,
        "fid_seed_better": fid_dirs,
        "max_relmeaserr": max(rels) if rels else None,
        "lpips_relative_gain": lp.get("relative_gain"),
        "statistics_note": "Primary CI bootstraps images after averaging paired train-seed deltas per image.",
    }


def evaluate_from_checkpoints(
    *,
    config: Mapping[str, Any],
    lmmse: hq.EmpiricalLMMSE,
    measurement: hq.GhostMeasurementOperator,
    dev_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    train_seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_method: list[dict[str, Any]] = []
    all_per: list[dict[str, Any]] = []
    eval_manifest: list[dict[str, Any]] = []
    for train_seed in train_seeds:
        for method in [METHOD_NO_GAN, METHOD_GAN]:
            run_dir = output_dir / "runs" / f"seed{train_seed}" / method
            ckpt = run_dir / "checkpoints" / f"{method}_seed{train_seed}_final.pt"
            model = load_eval_model(ckpt, config=config, measurement=measurement, device=device)
            method_rows, per_rows, diag = hq.evaluate_methods(
                methods={method: model},
                lmmse=lmmse,
                measurement=measurement,
                loader=dev_loader,
                device=device,
                config=config,
                output_dir=run_dir,
                epoch_tag=f"{method}_seed{train_seed}_matched_final",
            )
            for row in method_rows:
                row["train_seed"] = int(train_seed)
            for row in per_rows:
                row["train_seed"] = int(train_seed)
            all_method.extend(method_rows)
            all_per.extend(per_rows)
            eval_manifest.append({"method": method, "train_seed": int(train_seed), "checkpoint": str(ckpt), "checkpoint_sha256": hq.sha256_file(ckpt), "eval_diag": diag})
    return all_method, all_per, eval_manifest


def write_reports(
    *,
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    gate: Mapping[str, Any],
    all_method: Sequence[Mapping[str, Any]],
    all_per: Sequence[Mapping[str, Any]],
    fairness: Sequence[Mapping[str, Any]],
    equivalence: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    split_audit: Mapping[str, Any],
    op_meta: Mapping[str, Any],
    lmmse: hq.EmpiricalLMMSE,
    eval_manifest: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> None:
    reports = hq.ensure_dir(output_dir / "reports")
    hq.write_csv(reports / "method_metrics.csv", all_method)
    hq.write_csv(reports / "per_image_metrics.csv", all_per)
    hq.write_json(reports / "gate_report.json", gate)
    hq.write_json(reports / "fairness_manifest.json", list(fairness))
    hq.write_json(reports / "lambda_adv0_equivalence_test.json", equivalence)
    hq.write_json(reports / "operator_manifest.json", op_meta)
    hq.write_json(reports / "split_manifest.json", split_manifest)
    hq.write_json(reports / "duplicate_audit.json", split_audit)
    hq.write_json(reports / "lmmse_manifest.json", {"lambda": lmmse.lambda_, "rows_sha256": lmmse.rows_sha256})
    hq.write_json(reports / "eval_checkpoint_manifest.json", list(eval_manifest))
    hq.write_json(reports / "runtime_and_hashes.json", runtime)
    lines = [
        "# Matched Shared-Pretrain GAN Causal Development Report",
        "",
        f"Classification: `{gate['classification']}`",
        f"Development gate passed: `{gate['development_gate_passed']}`",
        "",
        "## Retired Old Evidence",
        "",
        "Previous unmatched result is retired as `INVALID_OLD_LOCKED_UNMATCHED_CONTROL`; see `matched_control/old_locked_retirement`.",
        "",
        "## Hierarchical Paired Comparisons",
        "",
        "| metric | delta matched_GAN - matched_no_GAN | 95% CI | relative gain | GAN image wins | no-GAN image wins | images | seed-image pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comp in gate.get("comparisons", []):
        if comp.get("status") != "PASS":
            lines.append(f"| {comp.get('metric')} | [DATA MISSING] | [DATA MISSING] | [DATA MISSING] | [DATA MISSING] | [DATA MISSING] | 0 | 0 |")
            continue
        rel = comp.get("relative_gain")
        lines.append(
            f"| {comp['metric']} | {float(comp['mean_delta']):.6g} | [{float(comp['ci_low']):.6g}, {float(comp['ci_high']):.6g}] | {float(rel):.4g} | {comp['wins_method']} | {comp['wins_reference']} | {comp['n_images']} | {comp['n_seed_image_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Gate Conditions",
            "",
            "```json",
            json.dumps(gate.get("conditions", {}), indent=2),
            "```",
            "",
            "## Seed-Level LPIPS Effects",
            "",
            "```json",
            json.dumps(next((c.get("seed_summary") for c in gate.get("comparisons", []) if c.get("metric") == "lpips"), []), indent=2),
            "```",
            "",
            "## Fairness Evidence",
            "",
            f"- `lambda_adv=0` equivalence: `{equivalence.get('status')}`",
            f"- per-seed fairness manifests: `{[f.get('status') for f in fairness]}`",
            "- Branches share generator initialization, optimizer state, AMP scaler, RNG state, LR path, batch-order hash and EMA update count.",
            "- Discriminator initialization uses forked RNG and its state is not allowed to perturb the generator branch stream.",
            "",
            "## Reproducibility",
            "",
            f"- config: `{config_path}`",
            f"- config sha256: `{runtime.get('config_sha256')}`",
            f"- git commit: `{runtime.get('git_commit')}`",
            "",
        ]
    )
    hq.write_text(reports / "MATCHED_DEVELOPMENT_REPORT.md", "\n".join(lines))
    hq.write_json(
        reports / "summary.json",
        {
            "status": "GAN_HQ_MATCHED_DEVELOPMENT_COMPLETE",
            "classification": gate["classification"],
            "development_gate_passed": gate["development_gate_passed"],
            "output_dir": str(output_dir),
            "key_artifacts": {
                "matched_report": str(reports / "MATCHED_DEVELOPMENT_REPORT.md"),
                "gate_report": str(reports / "gate_report.json"),
                "fairness_manifest": str(reports / "fairness_manifest.json"),
                "equivalence_test": str(reports / "lambda_adv0_equivalence_test.json"),
                "method_metrics": str(reports / "method_metrics.csv"),
                "per_image_metrics": str(reports / "per_image_metrics.csv"),
            },
            "runtime": dict(runtime),
        },
    )


def run(config_path: Path, *, mode: str) -> dict[str, Any]:
    started = time.time()
    config = hq.load_yaml(config_path)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = hq.ensure_dir(ROOT / str(config["output_dir"]))
    reports = hq.ensure_dir(output_dir / "reports")
    shutil.copyfile(config_path, output_dir / "config_used.yaml")
    train_ds, _val_ds, dev_ds, split_manifest = hq.build_split_datasets(config)
    split_audit = hq.save_split_hash_audit(reports / "sample_hash_audit.csv", {"train": train_ds, "dev": dev_ds})
    train_x, _labels, _indices = hq.tensor_dataset_to_matrix(train_ds)
    rows, op_meta = hq.build_structured_operator_rows(
        img_size=int(config["data"]["img_size"]),
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    measurement = hq.make_measurement_operator(rows, img_size=int(config["data"]["img_size"]), device=device, lambda_solver=float(config["operator"].get("lambda_solver", 1e-8)))
    lmmse = hq.EmpiricalLMMSE.fit(train_x, rows, lambda_=float(config["operator"].get("lmmse_lambda", 1e-4)))
    train_seeds = [int(v) for v in config["matched_training"].get("train_seeds", [0])]
    equivalence = run_equivalence_test(config=config, lmmse=lmmse, measurement=measurement, train_ds=train_ds, device=device, output_dir=output_dir)
    fairness: list[dict[str, Any]] = []
    if mode in {"train_eval", "train_only"}:
        for train_seed in train_seeds:
            fairness.append(train_matched_seed(train_seed=train_seed, config=config, lmmse=lmmse, measurement=measurement, train_ds=train_ds, device=device, output_dir=output_dir))
    else:
        for train_seed in train_seeds:
            p = output_dir / "runs" / f"seed{train_seed}" / "fairness_manifest.json"
            fairness.append(json.loads(p.read_text(encoding="utf-8")))
    all_method: list[dict[str, Any]] = []
    all_per: list[dict[str, Any]] = []
    eval_manifest: list[dict[str, Any]] = []
    gate: dict[str, Any] = {"classification": "NOT_EVALUATED", "development_gate_passed": False}
    if mode in {"train_eval", "score_only"}:
        dev_loader = hq.build_loader(dev_ds, batch_size=int(config["data"]["batch_size"]), workers=int(config["data"].get("num_workers", 0)), shuffle=False, seed=int(config["seed"]) + 1, device=device)
        all_method, all_per, eval_manifest = evaluate_from_checkpoints(config=config, lmmse=lmmse, measurement=measurement, dev_loader=dev_loader, device=device, output_dir=output_dir, train_seeds=train_seeds)
        gate = summarize_matched_gate(all_per, all_method, fairness, equivalence, config)
    runtime = {
        "status": "PASS",
        "mode": mode,
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
        "started_utc": hq.now_utc(),
        "config_sha256": hq.sha256_file(output_dir / "config_used.yaml"),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "script_sha256": hq.sha256_file(Path(__file__)),
    }
    write_reports(
        output_dir=output_dir,
        config_path=config_path,
        config=config,
        gate=gate,
        all_method=all_method,
        all_per=all_per,
        fairness=fairness,
        equivalence=equivalence,
        split_manifest=split_manifest,
        split_audit=split_audit,
        op_meta=op_meta,
        lmmse=lmmse,
        eval_manifest=eval_manifest,
        runtime=runtime,
    )
    summary = json.loads((reports / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="True shared-pretrain matched GAN causal validation for high-quality ghost imaging.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=["train_eval", "train_only", "score_only"], default="train_eval")
    args = parser.parse_args()
    run(Path(args.config), mode=str(args.mode))


if __name__ == "__main__":
    main()
