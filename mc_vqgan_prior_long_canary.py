from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils as tv_utils

import gan_high_quality_gi as hq
import gan_gauge_aligned_nsgan as ga
import measurement_conditioned_vqgan as mc


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "mc_vqgan_prior_long_canary_seed0.yaml"


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def state_hash(obj: Any) -> str:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    hq.ensure_dir(path.parent)
    exists = path.exists()
    keys = list(row.keys())
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if not exists:
            writer.writeheader()
        writer.writerow({k: hq.json_safe(row.get(k, "")) for k in keys})


def batch_stream(dataset: Dataset, *, batch_size: int, workers: int, seed: int, device: torch.device):
    epoch = 0
    while True:
        loader = hq.build_loader(dataset, batch_size=batch_size, workers=workers, shuffle=True, seed=int(seed) + epoch, device=device)
        for x, label, idx in loader:
            yield x, label, idx, epoch
        epoch += 1


def build_split_datasets(config: Mapping[str, Any]):
    data_cfg = dict(config["data"])
    if not bool(data_cfg.get("hash_clean", False)):
        return hq.build_split_datasets(config)
    root = str(data_cfg["dataset_root"])
    img_size = int(data_cfg.get("img_size", 64))
    source_split = str(data_cfg.get("source_split", "train+unlabeled"))
    base = datasets.STL10(root=root, split=source_split, download=True)
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    probe = hq.IndexedTensorDataset(base, [], transform)
    total_needed = int(data_cfg["train_count"]) + int(data_cfg["val_count"]) + int(data_cfg["dev_count"])
    seen: set[str] = set()
    unique_indices: list[int] = []
    duplicate_count = 0
    for source_index in range(len(base)):
        raw_h = probe.raw_hash(source_index)
        if raw_h in seen:
            duplicate_count += 1
            continue
        seen.add(raw_h)
        unique_indices.append(source_index)
        if len(unique_indices) >= total_needed:
            break
    if len(unique_indices) < total_needed:
        raise RuntimeError(f"HASH_CLEAN_SPLIT_TOO_SMALL:{len(unique_indices)}:{total_needed}")
    spans = {}
    out = []
    cursor = 0
    for name in ["train", "val", "dev"]:
        count = int(data_cfg[f"{name}_count"])
        idx = unique_indices[cursor : cursor + count]
        cursor += count
        spans[name] = {"count": count, "min": min(idx), "max": max(idx), "source_indices_sha256": hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()}
        out.append(hq.IndexedTensorDataset(base, idx, transform))
    manifest = {
        "source_split": source_split,
        "dataset_name": "STL10",
        "dataset_root": root,
        "img_size": img_size,
        "hash_clean": True,
        "unique_scan": {"scanned_until": int(unique_indices[-1]), "skipped_raw_duplicates_before_cut": int(duplicate_count), "unique_count": int(len(unique_indices))},
        "spans": spans,
        "note": "Hash-clean prior split uses the first occurrence of each raw STL10 image hash; no final-v4/test split is used.",
    }
    return out[0], out[1], out[2], manifest


def rng_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_random_state": repr(random.getstate()),
        "numpy_random_state": np.random.get_state()[1].tolist(),
        "torch_rng_state": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    return payload


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    hq.ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def make_model(config: Mapping[str, Any], device: torch.device) -> mc.VQAutoencoder:
    return mc.model_from_config(config, device)


def make_disc(config: Mapping[str, Any], device: torch.device) -> mc.FeaturePatchDiscriminator:
    return mc.disc_from_config(config, device)


def make_optim(model: nn.Module, lr: float, betas: Sequence[float]) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=float(lr), betas=tuple(float(x) for x in betas))


def codebook_stats(indices: torch.Tensor, codebook_size: int) -> dict[str, float]:
    flat = indices.detach().reshape(-1).cpu()
    hist = torch.bincount(flat, minlength=int(codebook_size)).float()
    probs = hist / hist.sum().clamp_min(1.0)
    perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
    used = int((hist > 0).sum().item())
    return {
        "codebook_perplexity": float(perplexity.item()),
        "used_codes": used,
        "dead_code_fraction": float(1.0 - used / max(1, int(codebook_size))),
    }


@torch.no_grad()
def evaluate_prior(
    *,
    model: mc.VQAutoencoder,
    loader: DataLoader,
    lpips_fn: Any,
    device: torch.device,
    config: Mapping[str, Any],
    phase: str,
    step: int,
    out_dir: Path,
) -> dict[str, Any]:
    model.eval()
    eval_cfg = dict(config["eval"])
    max_batches = int(eval_cfg.get("max_val_batches", 0))
    all_truth: list[torch.Tensor] = []
    all_recon: list[torch.Tensor] = []
    feat_truth: list[torch.Tensor] = []
    feat_recon: list[torch.Tensor] = []
    lpips_vals: list[float] = []
    psnr_vals: list[float] = []
    rmse_vals: list[float] = []
    ssim_vals: list[float] = []
    code_hist = torch.zeros(int(config["model"]["codebook_size"]), dtype=torch.float64)
    n_images = 0
    for bi, (x, _label, _idx) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        recon, indices, _vq_loss, _stats = model(x)
        clipped = recon.clamp(0, 1)
        lp = hq.lpips_batch(lpips_fn, clipped, x)
        rmse = hq.full_rmse_torch(clipped, x)
        if lp is not None:
            lpips_vals.extend([float(v) for v in lp])
        rmse_vals.extend([float(v) for v in rmse])
        psnr_vals.extend([float(-20.0 * math.log10(max(float(v), 1e-12))) for v in rmse])
        for i in range(x.shape[0]):
            ssim_vals.append(float(hq.ssim_metric(clipped[i : i + 1], x[i : i + 1])))
        hist = torch.bincount(indices.detach().reshape(-1).cpu(), minlength=int(config["model"]["codebook_size"])).double()
        code_hist += hist
        n_images += int(x.shape[0])
        if sum(t.shape[0] for t in feat_truth) < int(eval_cfg.get("kid_max_images", 128)):
            remaining = int(eval_cfg.get("kid_max_images", 128)) - sum(t.shape[0] for t in feat_truth)
            feat_truth.append(x.detach().cpu()[:remaining])
            feat_recon.append(clipped.detach().cpu()[:remaining])
        if sum(t.shape[0] for t in all_truth) < int(eval_cfg.get("qualitative_count", 12)):
            remaining = int(eval_cfg.get("qualitative_count", 12)) - sum(t.shape[0] for t in all_truth)
            all_truth.append(x.detach().cpu()[:remaining])
            all_recon.append(clipped.detach().cpu()[:remaining])
    probs = code_hist / code_hist.sum().clamp_min(1.0)
    perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
    used = int((code_hist > 0).sum().item())
    row: dict[str, Any] = {
        "phase": phase,
        "step": int(step),
        "n_images": int(n_images),
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else "[DATA MISSING]",
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else "[DATA MISSING]",
        "rmse": float(np.mean(rmse_vals)) if rmse_vals else "[DATA MISSING]",
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else "[DATA MISSING]",
        "codebook_perplexity": float(perplexity.item()),
        "used_codes": used,
        "dead_code_fraction": float(1.0 - used / max(1, int(config["model"]["codebook_size"]))),
    }
    if bool(eval_cfg.get("kid", True)):
        truth = torch.cat(feat_truth, dim=0) if feat_truth else torch.empty(0)
        recon = torch.cat(feat_recon, dim=0) if feat_recon else torch.empty(0)
        if truth.numel() and recon.numel():
            real_feat = hq.inception_features(truth, device=device, max_images=int(eval_cfg.get("kid_max_images", 128)))
            fake_feat = hq.inception_features(recon, device=device, max_images=int(eval_cfg.get("kid_max_images", 128)))
            row["kid"] = "[DATA MISSING]" if real_feat is None or fake_feat is None else hq.kid_from_features(real_feat, fake_feat)
        else:
            row["kid"] = "[DATA MISSING]"
    if all_truth and all_recon:
        grid = torch.cat([torch.cat(all_truth, dim=0)[: int(eval_cfg.get("qualitative_count", 12))], torch.cat(all_recon, dim=0)[: int(eval_cfg.get("qualitative_count", 12))]], dim=0)
        fig = out_dir / "figures" / f"qual_{phase}_step{int(step):06d}.png"
        hq.ensure_dir(fig.parent)
        tv_utils.save_image(grid, fig, nrow=int(eval_cfg.get("qualitative_count", 12)), padding=2)
    return row


def train_step(
    *,
    model: mc.VQAutoencoder,
    disc: mc.FeaturePatchDiscriminator | None,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer | None,
    x: torch.Tensor,
    config: Mapping[str, Any],
    lpips_fn: Any,
    gan_active: bool,
) -> dict[str, Any]:
    train_cfg = dict(config["training"])
    if disc is not None and gan_active:
        opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_det, _idx, _vq, _stats = model(x)
        real_score = disc(x)
        fake_score = disc(fake_det.detach())
        d_loss = mc.hinge_d_loss(real_score, fake_score)
        d_loss.backward()
        if float(train_cfg.get("grad_clip", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(disc.parameters(), float(train_cfg["grad_clip"]))
        opt_d.step()
    else:
        d_loss = torch.zeros((), device=x.device)
    opt_g.zero_grad(set_to_none=True)
    recon, indices, vq_loss, vq_stats = model(x)
    base_loss, parts = mc.image_base_loss(recon.clamp(0, 1), x, train_cfg, lpips_fn)
    adv = torch.zeros((), device=x.device)
    fm = torch.zeros((), device=x.device)
    if disc is not None and gan_active:
        fake_score, fake_feats = disc(recon.clamp(0, 1), return_features=True)
        with torch.no_grad():
            _real_score, real_feats = disc(x, return_features=True)
        adv = -fake_score.mean()
        fm = mc.feature_matching(real_feats, fake_feats)
    g_loss = base_loss + float(train_cfg.get("lambda_vq", 1.0)) * vq_loss + float(train_cfg.get("lambda_adv", 0.05)) * adv + float(train_cfg.get("lambda_fm", 0.5)) * fm
    g_loss.backward()
    if float(train_cfg.get("grad_clip", 0.0)) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
    opt_g.step()
    return {
        "g_loss": float(g_loss.detach().cpu()),
        "d_loss": float(d_loss.detach().cpu()),
        "adv": float(adv.detach().cpu()),
        "fm": float(fm.detach().cpu()),
        **parts,
        **vq_stats,
        **codebook_stats(indices, int(config["model"]["codebook_size"])),
    }


def run_validation_and_save(
    *,
    phase: str,
    step: int,
    model: mc.VQAutoencoder,
    disc: mc.FeaturePatchDiscriminator | None,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer | None,
    val_loader: DataLoader,
    lpips_fn: Any,
    device: torch.device,
    config: Mapping[str, Any],
    run_dir: Path,
    best: dict[str, float],
) -> dict[str, float]:
    row = evaluate_prior(model=model, loader=val_loader, lpips_fn=lpips_fn, device=device, config=config, phase=phase, step=step, out_dir=run_dir)
    append_csv(run_dir / "validation_metrics.csv", row)
    latest = run_dir / "checkpoints" / f"{phase}_latest.pt"
    payload = {
        "phase": phase,
        "step": int(step),
        "model": model.state_dict(),
        "disc": None if disc is None else disc.state_dict(),
        "ema": None,
        "opt_g": opt_g.state_dict(),
        "opt_d": None if opt_d is None else opt_d.state_dict(),
        "rng": rng_payload(),
        "config": hq.json_safe(config),
        "validation": hq.json_safe(row),
    }
    save_checkpoint(latest, payload)
    try:
        lp = float(row["lpips"])
    except (TypeError, ValueError):
        lp = float("inf")
    if lp < best.get("lpips", float("inf")):
        best["lpips"] = lp
        best["step"] = float(step)
        save_checkpoint(run_dir / "checkpoints" / f"{phase}_best_by_lpips.pt", payload)
    hq.write_json(run_dir / "best_by_lpips.json", best)
    return best


def train_phase(
    *,
    phase: str,
    model: mc.VQAutoencoder,
    disc: mc.FeaturePatchDiscriminator | None,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer | None,
    train_ds: Dataset,
    val_loader: DataLoader,
    lpips_fn: Any,
    device: torch.device,
    config: Mapping[str, Any],
    run_dir: Path,
    steps: int,
    stream_seed: int,
    gan_active: bool,
    start_step: int = 0,
) -> dict[str, Any]:
    batch_size = int(config["data"]["batch_size"])
    workers = int(config["data"].get("num_workers", 0))
    val_interval = int(config["training"].get("val_interval", 2000))
    train_log = run_dir / "train_log.csv"
    stream = batch_stream(train_ds, batch_size=batch_size, workers=workers, seed=stream_seed, device=device)
    hasher = hashlib.sha256()
    best: dict[str, float] = {"lpips": float("inf"), "step": -1.0}
    hq.ensure_dir(run_dir / "checkpoints")
    losses_window: list[dict[str, Any]] = []
    for local_step in range(1, int(steps) + 1):
        global_step = int(start_step) + local_step
        x, _label, idx, _epoch = next(stream)
        hasher.update(np.asarray([int(i) for i in idx], dtype=np.int64).tobytes())
        x = x.to(device, non_blocking=True)
        row = train_step(model=model, disc=disc, opt_g=opt_g, opt_d=opt_d, x=x, config=config, lpips_fn=lpips_fn, gan_active=gan_active)
        row.update({"phase": phase, "step": int(global_step), "gan_active": bool(gan_active)})
        losses_window.append(row)
        if global_step % max(1, int(config["training"].get("log_interval", 100))) == 0:
            avg: dict[str, Any] = {"phase": phase, "step": int(global_step), "gan_active": bool(gan_active)}
            for key in sorted({k for r in losses_window for k in r if k not in {"phase", "step", "gan_active"}}):
                avg[key] = float(np.mean([float(r.get(key, 0.0)) for r in losses_window]))
            append_csv(train_log, avg)
            losses_window.clear()
        if global_step % val_interval == 0 or local_step == int(steps):
            best = run_validation_and_save(phase=phase, step=global_step, model=model, disc=disc, opt_g=opt_g, opt_d=opt_d, val_loader=val_loader, lpips_fn=lpips_fn, device=device, config=config, run_dir=run_dir, best=best)
    return {
        "phase": phase,
        "steps": int(steps),
        "batch_order_hash": hasher.hexdigest(),
        "best_lpips": best.get("lpips"),
        "best_step": best.get("step"),
        "latest_checkpoint": str(run_dir / "checkpoints" / f"{phase}_latest.pt"),
        "best_checkpoint": str(run_dir / "checkpoints" / f"{phase}_best_by_lpips.pt"),
    }


def write_summary(out: Path, manifests: Sequence[Mapping[str, Any]], config: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    reports = hq.ensure_dir(out / "reports")
    hq.write_json(reports / "training_manifest.json", list(manifests))
    hq.write_json(reports / "runtime_and_hashes.json", runtime)
    lines = [
        "# Single-Seed Long VQGAN Prior Canary",
        "",
        "This run trains only the image prior. No measurement encoder is trained.",
        "",
        f"- Train count: `{config['data']['train_count']}`",
        f"- Batch size: `{config['data']['batch_size']}`",
        f"- Warmup VQAE steps: `{config['training']['warmup_steps']}`",
        f"- Matched continuation steps: `{config['training']['continuation_steps']}`",
        f"- Validation interval: `{config['training']['val_interval']}`",
        "",
        "| phase | best LPIPS | best step | batch order hash |",
        "|---|---:|---:|---|",
    ]
    for m in manifests:
        lines.append(f"| {m['phase']} | {m.get('best_lpips')} | {m.get('best_step')} | `{m.get('batch_order_hash')}` |")
    lines += [
        "",
        "Outputs include `validation_metrics.csv`, `train_log.csv`, `latest.pt`, and `best_by_lpips.pt` under each phase directory.",
    ]
    hq.write_text(reports / "LONG_CANARY_SUMMARY.md", "\n".join(lines) + "\n")


def run(config_path: Path) -> dict[str, Any]:
    t0 = time.time()
    config = load_yaml(config_path)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    out = hq.ensure_dir(ROOT / str(config["output_dir"]))
    hq.write_text(out / "config_used.yaml", config_path.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    train_ds, val_ds, dev_ds, split_manifest = build_split_datasets(config)
    duplicate_audit = hq.save_split_hash_audit(out / "reports" / "sample_hash_audit.csv", {"train": train_ds, "val": val_ds, "dev": dev_ds})
    del dev_ds
    val_loader = hq.build_loader(val_ds, batch_size=int(config["data"].get("eval_batch_size", config["data"]["batch_size"])), workers=int(config["data"].get("num_workers", 0)), shuffle=False, seed=int(config["seed"]) + 55, device=device)
    hq.write_json(out / "reports" / "split_manifest.json", split_manifest)
    hq.write_json(out / "reports" / "duplicate_audit.json", duplicate_audit)
    lpips_fn = ga.freeze_lpips(hq.load_lpips(device) if bool(config["training"].get("lpips_train", True)) else {"error": "disabled"})
    model = make_model(config, device)
    start_hash = state_hash(model.state_dict())
    opt_g = make_optim(model, float(config["training"]["prior_lr_g"]), config["training"].get("betas", [0.5, 0.9]))
    warmup_dir = hq.ensure_dir(out / "warmup_vqae")
    manifests = []
    warmup_manifest = train_phase(
        phase="warmup_vqae",
        model=model,
        disc=None,
        opt_g=opt_g,
        opt_d=None,
        train_ds=train_ds,
        val_loader=val_loader,
        lpips_fn=lpips_fn,
        device=device,
        config=config,
        run_dir=warmup_dir,
        steps=int(config["training"]["warmup_steps"]),
        stream_seed=int(config["seed"]) + 9000,
        gan_active=False,
        start_step=0,
    )
    warmup_manifest["start_model_hash"] = start_hash
    warmup_manifest["end_model_hash"] = state_hash(model.state_dict())
    manifests.append(warmup_manifest)

    warmup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for branch, use_gan in [("vqae_continuation", False), ("vqgan_continuation", True)]:
        branch_model = make_model(config, device)
        branch_model.load_state_dict(warmup_state)
        disc = make_disc(config, device) if use_gan else None
        opt_bg = make_optim(branch_model, float(config["training"]["prior_lr_g"]), config["training"].get("betas", [0.5, 0.9]))
        opt_bd = make_optim(disc, float(config["training"]["prior_lr_d"]), config["training"].get("betas", [0.5, 0.9])) if disc is not None else None
        run_dir = hq.ensure_dir(out / branch)
        manifest = train_phase(
            phase=branch,
            model=branch_model,
            disc=disc,
            opt_g=opt_bg,
            opt_d=opt_bd,
            train_ds=train_ds,
            val_loader=val_loader,
            lpips_fn=lpips_fn,
            device=device,
            config=config,
            run_dir=run_dir,
            steps=int(config["training"]["continuation_steps"]),
            stream_seed=int(config["seed"]) + 99000,
            gan_active=use_gan,
            start_step=int(config["training"]["warmup_steps"]),
        )
        manifest["branch_start_model_hash"] = state_hash(warmup_state)
        manifest["end_model_hash"] = state_hash(branch_model.state_dict())
        manifests.append(manifest)
    runtime = {
        "seconds": float(time.time() - t0),
        "device": str(device),
        "config": str(config_path),
        "config_sha256": hq.sha256_file(config_path),
        "script_sha256": hq.sha256_file(ROOT / "mc_vqgan_prior_long_canary.py"),
    }
    write_summary(out, manifests, config, runtime)
    summary = {"output_dir": str(out), "manifests": hq.json_safe(manifests), "runtime": runtime}
    hq.write_json(out / "reports" / "summary.json", summary)
    print(json.dumps(hq.json_safe(summary), indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-seed long canary for matched VQAE/VQGAN image-prior training.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
