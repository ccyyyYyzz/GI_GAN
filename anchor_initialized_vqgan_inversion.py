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
import zipfile
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
from src.losses import charbonnier_loss, frequency_loss, gradient_difference_loss
from src.metrics import ssim as ssim_metric
from src.projections import get_exact_projector, relative_measurement_error


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "anchor_vqgan_inversion_seed0.yaml"
VQAE = "vqae"
VQGAN = "vqgan"
BETAS_DEFAULT = [0.0, 0.25, 0.5, 0.75, 1.0]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def stable_hash(obj: Any) -> str:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    hq.write_csv(path, rows)


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
    splits = {}
    cursor = 0
    datasets_out = []
    for name in ["train", "val", "dev"]:
        count = int(data_cfg[f"{name}_count"])
        idx = unique_indices[cursor : cursor + count]
        cursor += count
        splits[name] = {"count": count, "min": min(idx), "max": max(idx), "source_indices_sha256": hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()}
        datasets_out.append(hq.IndexedTensorDataset(base, idx, transform))
    manifest = {
        "source_split": source_split,
        "dataset_name": "STL10",
        "dataset_root": root,
        "img_size": img_size,
        "hash_clean": True,
        "unique_scan": {"scanned_until": int(unique_indices[-1]), "skipped_raw_duplicates_before_cut": int(duplicate_count), "unique_count": int(len(unique_indices))},
        "spans": splits,
        "note": "Hash-clean development split uses the first occurrence of each raw STL10 image hash; no final-v4/test split is used for selection.",
    }
    return datasets_out[0], datasets_out[1], datasets_out[2], manifest


class AnchorLatentRefiner(nn.Module):
    def __init__(self, *, z_dim: int, codebook_size: int, base: int = 64, delta_scale: float = 0.25, logit_scale: float = 1.0) -> None:
        super().__init__()
        c = int(base)
        self.delta_scale = float(delta_scale)
        self.logit_scale = float(logit_scale)
        self.img_down = nn.Sequential(
            nn.Conv2d(2, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c + int(z_dim), c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
        )
        self.delta_z = nn.Conv2d(c, int(z_dim), 1)
        self.delta_logits = nn.Conv2d(c, int(codebook_size), 1)
        nn.init.zeros_(self.delta_z.weight)
        nn.init.zeros_(self.delta_z.bias)
        nn.init.zeros_(self.delta_logits.weight)
        nn.init.zeros_(self.delta_logits.bias)

    def forward(self, x0: torch.Tensor, uncertainty: torch.Tensor, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.img_down(torch.cat([x0, uncertainty], dim=1))
        h = self.fuse(torch.cat([h, z0], dim=1))
        return self.delta_scale * self.delta_z(h), self.logit_scale * self.delta_logits(h)


class PriorPack:
    def __init__(self, kind: str, model: mc.VQAutoencoder, disc: mc.FeaturePatchDiscriminator | None, ckpt: Path) -> None:
        self.kind = kind
        self.model = model
        self.disc = disc
        self.ckpt = ckpt


def load_prior(kind: str, ckpt: Path, config: Mapping[str, Any], device: torch.device) -> PriorPack:
    model = mc.model_from_config(config, device)
    payload = torch.load(ckpt, map_location=device)
    state = payload.get("model") or payload.get("generator") or payload
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    disc = None
    disc_state = payload.get("disc") or payload.get("discriminator")
    if kind == VQGAN and disc_state is not None:
        disc = mc.disc_from_config(config, device)
        disc.load_state_dict(disc_state)
        disc.eval()
        for p in disc.parameters():
            p.requires_grad_(False)
    return PriorPack(kind, model, disc, ckpt)


def audit_flat_or_img(x: torch.Tensor, y: torch.Tensor, measurement) -> torch.Tensor:
    return mc.audit_image(x, y, measurement)


def null_blend(x0: torch.Tensor, xg: torch.Tensor, beta: float, measurement) -> torch.Tensor:
    if float(beta) == 0.0:
        return x0
    with torch.cuda.amp.autocast(enabled=False):
        projector = get_exact_projector(measurement, dtype=torch.float64, device=x0.device)
        diff = measurement.flatten_img(xg - x0).double()
        p0 = projector.null_project_flat(diff)
        out = measurement.flatten_img(x0).double() + float(beta) * p0
    return measurement.unflatten_img(out.to(dtype=x0.dtype))


def logits_from_latent(z: torch.Tensor, prior: PriorPack, *, distance_temperature: float) -> torch.Tensor:
    emb = prior.model.quantizer.embedding.weight
    flat = z.permute(0, 2, 3, 1).reshape(-1, emb.shape[1])
    dist = flat.pow(2).sum(dim=1, keepdim=True) + emb.pow(2).sum(dim=1)[None, :] - 2.0 * flat @ emb.t()
    logits = -dist.reshape(z.shape[0], z.shape[2], z.shape[3], emb.shape[0]).permute(0, 3, 1, 2).contiguous()
    return logits / max(float(distance_temperature), 1e-6)


def quantize_from_logits(prior: PriorPack, logits: torch.Tensor, *, soft_temperature: float, straight_through: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits / max(float(soft_temperature), 1e-6), dim=1)
    z_soft = torch.einsum("bkhw,kc->bchw", probs, prior.model.quantizer.embedding.weight)
    idx = logits.argmax(dim=1)
    z_hard = prior.model.quantizer.lookup_indices(idx)
    zq = z_soft + (z_hard - z_soft).detach() if straight_through else z_soft
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=1).mean()
    return zq, idx, entropy


def image_loss(pred: torch.Tensor, truth: torch.Tensor, lpips_fn: Any, cfg: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = F.l1_loss(pred, truth)
    charb = charbonnier_loss(pred, truth)
    grad = gradient_difference_loss(pred, truth)
    spec = frequency_loss(pred, truth)
    lp = ga.differentiable_lpips(lpips_fn, pred, truth)
    total = (
        float(cfg.get("lambda_l1", 1.0)) * l1
        + float(cfg.get("lambda_charb", 1.0)) * charb
        + float(cfg.get("lambda_grad", 0.25)) * grad
        + float(cfg.get("lambda_spec", 0.1)) * spec
        + float(cfg.get("lambda_lpips", 0.25)) * lp
    )
    return total, {"l1": float(l1.detach().cpu()), "charb": float(charb.detach().cpu()), "grad": float(grad.detach().cpu()), "spec": float(spec.detach().cpu()), "lpips_train": float(lp.detach().cpu())}


def prediction_metrics(
    *,
    pred: torch.Tensor,
    truth: torch.Tensor,
    y: torch.Tensor,
    measurement,
    lpips_fn: Any,
    method: str,
    beta: float,
    source_idx: torch.Tensor,
    labels: torch.Tensor,
    train_seed: int,
    extra: Mapping[str, Any],
) -> list[dict[str, Any]]:
    clipped = pred.clamp(0, 1)
    lp_vals = hq.lpips_batch(lpips_fn, clipped, truth)
    rel = relative_measurement_error(pred, y, measurement).detach().cpu().numpy().astype(np.float64)
    rmse = hq.full_rmse_torch(clipped, truth)
    crmse = hq.centered_rmse_torch(clipped, truth)
    sharp = hq.edge_sharpness(clipped)
    pred_np = clipped.detach().cpu().numpy()[:, 0]
    truth_np = truth.detach().cpu().numpy()[:, 0]
    rapsd = np.asarray([np.linalg.norm(hq.rapsd_np(pred_np[i]) - hq.rapsd_np(truth_np[i])) for i in range(truth.shape[0])], dtype=np.float64)
    rows = []
    for i in range(truth.shape[0]):
        row = {
            "train_seed": int(train_seed),
            "method": method,
            "beta": float(beta),
            "source_index": int(source_idx[i]),
            "label": int(labels[i]),
            "full_rmse": float(rmse[i]),
            "centered_rmse": float(crmse[i]),
            "psnr": float(-20.0 * math.log10(max(float(rmse[i]), 1e-12))),
            "ssim": float(ssim_metric(clipped[i : i + 1], truth[i : i + 1])),
            "lpips": "[DATA MISSING]" if lp_vals is None else float(lp_vals[i]),
            "rapsd": float(rapsd[i]),
            "edge_sharpness": float(sharp[i]),
            "relmeaserr": float(rel[i]),
        }
        row.update(hq.json_safe(extra))
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_predictions(
    *,
    predictions: Mapping[str, list[torch.Tensor]],
    truth_chunks: list[torch.Tensor],
    lpips_fn: Any,
    device: torch.device,
    kid_max_images: int,
) -> dict[str, dict[str, Any]]:
    truth = torch.cat(truth_chunks, dim=0).clamp(0, 1)
    out: dict[str, dict[str, Any]] = {}
    real_feat = hq.inception_features(truth, device=device, max_images=int(kid_max_images))
    for name, chunks in predictions.items():
        pred = torch.cat(chunks, dim=0).clamp(0, 1)
        row: dict[str, Any] = {}
        if real_feat is not None:
            fake_feat = hq.inception_features(pred, device=device, max_images=int(kid_max_images))
            row["kid"] = "[DATA MISSING]" if fake_feat is None else hq.kid_from_features(real_feat, fake_feat)
        else:
            row["kid"] = "[DATA MISSING]"
        out[name] = row
    return out


@torch.no_grad()
def run_transfer_ceiling(
    *,
    config: Mapping[str, Any],
    priors: Mapping[str, PriorPack],
    loader: DataLoader,
    lmmse: hq.EmpiricalLMMSE,
    measurement,
    lpips_fn: Any,
    device: torch.device,
    out: Path,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    betas = [float(b) for b in config["eval"].get("beta_grid", BETAS_DEFAULT)]
    train_seed_id = int(config.get("experiment_seed", config.get("seed_id", 0)))
    rows: list[dict[str, Any]] = []
    truth_chunks: list[torch.Tensor] = []
    pred_chunks: dict[str, list[torch.Tensor]] = {}
    for x, label, idx in loader:
        x = x.to(device, non_blocking=True)
        flat = measurement.flatten_img(x)
        y = measurement.A_forward(flat)
        x0_flat = lmmse.anchor(y, measurement, device=device)
        x0 = measurement.unflatten_img(x0_flat)
        truth_chunks.append(x.detach().cpu())
        for beta in [0.0]:
            method = "lmmse_anchor"
            rows.extend(prediction_metrics(pred=x0, truth=x, y=y, measurement=measurement, lpips_fn=lpips_fn, method=method, beta=beta, source_idx=idx, labels=label, train_seed=train_seed_id, extra={"source": split_name, "projection_norm": 0.0, "pre_audit_rel": "[DATA MISSING]"}))
            pred_chunks.setdefault(f"{method}_b{beta:g}", []).append(x0.detach().cpu())
        for kind, prior in priors.items():
            teacher_idx = prior.model.encode_indices(x)
            x_teacher = prior.model.decode_indices(teacher_idx)
            z0 = prior.model.encode(x0)
            logits0 = logits_from_latent(z0, prior, distance_temperature=float(config["training"].get("distance_temperature", 1.0)))
            z_anchor, anchor_idx, entropy = quantize_from_logits(prior, logits0, soft_temperature=float(config["training"].get("soft_temperature", 1.0)), straight_through=False)
            x_anchor = prior.model.decode_embeddings(z_anchor)
            for source_name, xg in [(f"{kind}_teacher", x_teacher), (f"{kind}_anchor_latent", x_anchor)]:
                pre = mc.pre_audit_rel_mse(xg, y, measurement)
                audited = audit_flat_or_img(xg, y, measurement)
                proj_norm = (audited - xg).reshape(x.shape[0], -1).norm(dim=1).mean() / math.sqrt(x.shape[-1] * x.shape[-2])
                for method, pred, beta in [(f"{source_name}_raw", xg, -1.0), (f"{source_name}_audited", audited, 1.0)]:
                    rows.extend(prediction_metrics(pred=pred, truth=x, y=y, measurement=measurement, lpips_fn=lpips_fn, method=method, beta=beta, source_idx=idx, labels=label, train_seed=train_seed_id, extra={"source": split_name, "projection_norm": float(proj_norm.cpu()), "pre_audit_rel": float(pre.cpu()), "posterior_entropy": float(entropy.cpu()) if "anchor" in source_name else "[DATA MISSING]"}))
                    pred_chunks.setdefault(f"{method}_b{beta:g}", []).append(pred.detach().cpu())
                for beta in betas:
                    pred = null_blend(x0, xg, beta, measurement)
                    method = f"{source_name}_nullblend"
                    rows.extend(prediction_metrics(pred=pred, truth=x, y=y, measurement=measurement, lpips_fn=lpips_fn, method=method, beta=beta, source_idx=idx, labels=label, train_seed=train_seed_id, extra={"source": split_name, "projection_norm": float((pred - xg).reshape(x.shape[0], -1).norm(dim=1).mean().cpu() / math.sqrt(x.shape[-1] * x.shape[-2])), "pre_audit_rel": float(pre.cpu()), "posterior_entropy": float(entropy.cpu()) if "anchor" in source_name else "[DATA MISSING]"}))
                    pred_chunks.setdefault(f"{method}_b{beta:g}", []).append(pred.detach().cpu())
    kid = evaluate_predictions(predictions=pred_chunks, truth_chunks=truth_chunks, lpips_fn=lpips_fn, device=device, kid_max_images=int(config["eval"].get("kid_max_images", 128)))
    for row in rows:
        key = f"{row['method']}_b{float(row['beta']):g}"
        row["kid_group"] = kid.get(key, {}).get("kid", "[DATA MISSING]")
    method_rows = summarize_rows(rows)
    write_csv(out / "reports" / f"stage0_{split_name}_per_image.csv", rows)
    write_csv(out / "reports" / f"stage0_{split_name}_method_metrics.csv", method_rows)
    save_qualitative_from_chunks(out / "reports" / "figures" / f"stage0_{split_name}_qualitative.png", truth_chunks, pred_chunks, preferred=[
        "lmmse_anchor_b0",
        "vqae_teacher_nullblend_b1",
        "vqgan_teacher_nullblend_b0.25",
        "vqgan_teacher_nullblend_b0.5",
        "vqgan_teacher_nullblend_b1",
        "vqgan_anchor_latent_nullblend_b1",
    ])
    return method_rows, rows, kid


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault((str(r["method"]), float(r["beta"])), []).append(r)
    out = []
    for (method, beta), vals in sorted(groups.items()):
        row: dict[str, Any] = {"method": method, "beta": beta, "n": len(vals)}
        for metric in ["lpips", "full_rmse", "centered_rmse", "psnr", "ssim", "rapsd", "edge_sharpness", "relmeaserr", "projection_norm", "pre_audit_rel"]:
            arr = []
            for v in vals:
                try:
                    arr.append(float(v[metric]))
                except (TypeError, ValueError, KeyError):
                    pass
            row[f"{metric}_mean"] = float(np.mean(arr)) if arr else "[DATA MISSING]"
        kids = []
        for v in vals:
            try:
                kids.append(float(v["kid_group"]))
            except (TypeError, ValueError, KeyError):
                pass
        row["kid"] = float(np.mean(kids)) if kids else "[DATA MISSING]"
        out.append(row)
    return out


def save_qualitative_from_chunks(path: Path, truth_chunks: list[torch.Tensor], pred_chunks: Mapping[str, list[torch.Tensor]], preferred: Sequence[str], *, max_items: int = 10) -> None:
    hq.ensure_dir(path.parent)
    truth = torch.cat(truth_chunks, dim=0)[:max_items].clamp(0, 1)
    rows = [truth]
    names = ["truth"]
    for name in preferred:
        if name in pred_chunks:
            rows.append(torch.cat(pred_chunks[name], dim=0)[:max_items].clamp(0, 1))
            names.append(name)
    tv_utils.save_image(torch.cat(rows, dim=0), path, nrow=max_items, padding=2)
    hq.write_text(path.with_suffix(".txt"), "Rows: " + ", ".join(names) + "\n")


def make_pareto_figure(path: Path, method_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    points = []
    for row in method_rows:
        try:
            points.append((str(row["method"]), float(row["beta"]), float(row["lpips_mean"]), float(row["psnr_mean"])))
        except Exception:
            continue
    if not points:
        return
    hq.ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    marker = {
        "lmmse_anchor": "o",
        "vqae_refiner_nullblend": "s",
        "vqgan_refiner_nullblend": "^",
        "vqae_teacher_nullblend": "x",
        "vqgan_teacher_nullblend": "D",
        "vqae_anchor_latent_nullblend": "P",
        "vqgan_anchor_latent_nullblend": "*",
    }
    for method in sorted({p[0] for p in points}):
        xs = [p[3] for p in points if p[0] == method]
        ys = [p[2] for p in points if p[0] == method]
        betas = [p[1] for p in points if p[0] == method]
        ax.scatter(xs, ys, label=method, marker=marker.get(method, "o"), s=28)
        for x, y, beta in zip(xs, ys, betas):
            if beta >= 0:
                ax.annotate(f"{beta:g}", (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("LPIPS (lower is better)")
    ax.set_title("Anchor inversion perception-distortion frontier")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def choose_beta(method_rows: Sequence[Mapping[str, Any]], *, method_prefix: str, lmmse_psnr: float, tolerance: float) -> dict[str, Any]:
    candidates = []
    for r in method_rows:
        if str(r["method"]) == method_prefix and float(r["beta"]) >= 0:
            try:
                psnr = float(r["psnr_mean"])
                lpips = float(r["lpips_mean"])
            except (TypeError, ValueError):
                continue
            if psnr >= float(lmmse_psnr) - float(tolerance):
                candidates.append((lpips, r))
    if not candidates:
        all_rows = [r for r in method_rows if str(r["method"]) == method_prefix and float(r["beta"]) >= 0]
        if not all_rows:
            return {"status": "NO_CANDIDATE"}
        r = max(all_rows, key=lambda x: float(x["psnr_mean"]))
        return {"status": "FALLBACK_MAX_PSNR", "selected": r}
    return {"status": "PASS", "selected": min(candidates, key=lambda x: x[0])[1]}


def train_refiner(
    *,
    kind: str,
    prior: PriorPack,
    config: Mapping[str, Any],
    train_ds: Dataset,
    val_loader: DataLoader,
    lmmse: hq.EmpiricalLMMSE,
    measurement,
    lpips_fn: Any,
    device: torch.device,
    out: Path,
    seed: int,
    start_state: Mapping[str, torch.Tensor],
) -> tuple[AnchorLatentRefiner, dict[str, Any]]:
    train_cfg = dict(config["training"])
    refiner = AnchorLatentRefiner(z_dim=int(config["model"]["z_dim"]), codebook_size=int(config["model"]["codebook_size"]), base=int(config["model"].get("refiner_base_channels", 64)), delta_scale=float(train_cfg.get("delta_scale", 0.25)), logit_scale=float(train_cfg.get("logit_scale", 1.0))).to(device)
    refiner.load_state_dict(start_state)
    opt = torch.optim.Adam(refiner.parameters(), lr=float(train_cfg.get("refiner_lr", 2e-4)), betas=tuple(train_cfg.get("betas", [0.5, 0.9])))
    stream = batch_stream(train_ds, batch_size=int(config["data"]["batch_size"]), workers=int(config["data"].get("num_workers", 0)), seed=int(config["seed"]) + 80000 + int(seed), device=device)
    run_dir = hq.ensure_dir(out / "runs" / f"seed{seed}" / f"{kind}_refiner")
    batch_hasher = hashlib.sha256()
    best: dict[str, Any] = {"lpips": float("inf"), "step": -1}
    for step in range(1, int(train_cfg["refiner_steps"]) + 1):
        x, _label, idx, _epoch = next(stream)
        batch_hasher.update(np.asarray([int(i) for i in idx], dtype=np.int64).tobytes())
        x = x.to(device, non_blocking=True)
        flat = measurement.flatten_img(x)
        y = measurement.A_forward(flat)
        with torch.no_grad():
            x0_flat = lmmse.anchor(y, measurement, device=device)
            x0 = measurement.unflatten_img(x0_flat)
            uncertainty = lmmse.uncertainty_map(img_size=int(config["data"]["img_size"]), device=device, batch_size=x.shape[0], dtype=x.dtype)
            z0 = prior.model.encode(x0)
            z_teacher = prior.model.encode(x)
            teacher_idx = prior.model.encode_indices(x)
            logits0 = logits_from_latent(z0, prior, distance_temperature=float(train_cfg.get("distance_temperature", 1.0)))
        dz, dlogits = refiner(x0, uncertainty, z0)
        z_cont = z0 + dz
        logits = logits_from_latent(z_cont, prior, distance_temperature=float(train_cfg.get("distance_temperature", 1.0))) + dlogits
        zq, pred_idx, entropy = quantize_from_logits(prior, logits, soft_temperature=float(train_cfg.get("soft_temperature", 1.0)), straight_through=True)
        xg = prior.model.decode_embeddings(zq)
        xhat = null_blend(x0, xg, float(train_cfg.get("beta_train", 1.0)), measurement).clamp(0, 1)
        img_loss, parts = image_loss(xhat, x, lpips_fn, train_cfg)
        latent_loss = F.l1_loss(z_cont, z_teacher.detach())
        token_loss = F.cross_entropy(logits, teacher_idx)
        pre = mc.pre_audit_rel_mse(xg, y, measurement)
        guidance = torch.zeros((), device=device)
        if kind == VQGAN and prior.disc is not None and float(train_cfg.get("lambda_guidance", 0.0)) > 0:
            guidance = F.softplus(-prior.disc(xg.clamp(0, 1))).mean()
        loss = (
            float(train_cfg.get("lambda_image", 1.0)) * img_loss
            + float(train_cfg.get("lambda_latent", 0.2)) * latent_loss
            + float(train_cfg.get("lambda_token", 0.25)) * token_loss
            + float(train_cfg.get("lambda_pre_audit", 0.05)) * pre
            + float(train_cfg.get("lambda_guidance", 0.0)) * guidance
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(train_cfg.get("grad_clip", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), float(train_cfg["grad_clip"]))
        opt.step()
        if step % int(train_cfg.get("log_interval", 100)) == 0:
            top1 = (pred_idx == teacher_idx).float().mean()
            top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices.eq(teacher_idx[:, None]).any(dim=1).float().mean()
            append_csv(run_dir / "train_log.csv", {
                "kind": kind,
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "latent_loss": float(latent_loss.detach().cpu()),
                "token_loss": float(token_loss.detach().cpu()),
                "pre_audit": float(pre.detach().cpu()),
                "guidance": float(guidance.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "top1": float(top1.detach().cpu()),
                "top5": float(top5.detach().cpu()),
                **parts,
            })
        if step % int(train_cfg.get("val_interval", 2000)) == 0 or step == int(train_cfg["refiner_steps"]):
            method_rows, per_rows, _ = evaluate_refiner(kind=kind, prior=prior, refiner=refiner, config=config, loader=val_loader, lmmse=lmmse, measurement=measurement, lpips_fn=lpips_fn, device=device, out_dir=run_dir, split_name="val", train_seed=seed)
            write_csv(run_dir / f"val_step{step:06d}_method_metrics.csv", method_rows)
            write_csv(run_dir / f"val_step{step:06d}_per_image.csv", per_rows)
            selected = choose_beta(method_rows, method_prefix=f"{kind}_refiner_nullblend", lmmse_psnr=float(next(r for r in method_rows if r["method"] == "lmmse_anchor")["psnr_mean"]), tolerance=float(config["eval"].get("psnr_drop_tolerance_db", 2.5)))
            if selected.get("selected"):
                lp = float(selected["selected"]["lpips_mean"])
                if lp < best["lpips"]:
                    best = {"lpips": lp, "step": int(step), "selected_beta": float(selected["selected"]["beta"]), "selection": selected}
                    save_checkpoint(run_dir / "checkpoints" / f"{kind}_refiner_best_by_val_lpips.pt", refiner, opt, config, best)
            save_checkpoint(run_dir / "checkpoints" / f"{kind}_refiner_latest.pt", refiner, opt, config, {"step": int(step)})
            hq.write_json(run_dir / "best_by_val_lpips.json", best)
    manifest = {"kind": kind, "seed": int(seed), "steps": int(train_cfg["refiner_steps"]), "batch_order_hash": batch_hasher.hexdigest(), "best": best, "latest_checkpoint": str(run_dir / "checkpoints" / f"{kind}_refiner_latest.pt"), "best_checkpoint": str(run_dir / "checkpoints" / f"{kind}_refiner_best_by_val_lpips.pt")}
    return refiner.eval(), manifest


def save_checkpoint(path: Path, refiner: nn.Module, opt: torch.optim.Optimizer, config: Mapping[str, Any], meta: Mapping[str, Any]) -> None:
    hq.ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"refiner": refiner.state_dict(), "optimizer": opt.state_dict(), "config": hq.json_safe(config), "meta": hq.json_safe(meta)}, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate_refiner(
    *,
    kind: str,
    prior: PriorPack,
    refiner: AnchorLatentRefiner,
    config: Mapping[str, Any],
    loader: DataLoader,
    lmmse: hq.EmpiricalLMMSE,
    measurement,
    lpips_fn: Any,
    device: torch.device,
    out_dir: Path,
    split_name: str,
    train_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    betas = [float(b) for b in config["eval"].get("beta_grid", BETAS_DEFAULT)]
    refiner.eval()
    per: list[dict[str, Any]] = []
    truth_chunks: list[torch.Tensor] = []
    pred_chunks: dict[str, list[torch.Tensor]] = {}
    for x, label, idx in loader:
        x = x.to(device, non_blocking=True)
        flat = measurement.flatten_img(x)
        y = measurement.A_forward(flat)
        x0_flat = lmmse.anchor(y, measurement, device=device)
        x0 = measurement.unflatten_img(x0_flat)
        uncertainty = lmmse.uncertainty_map(img_size=int(config["data"]["img_size"]), device=device, batch_size=x.shape[0], dtype=x.dtype)
        z0 = prior.model.encode(x0)
        teacher_idx = prior.model.encode_indices(x)
        logits0 = logits_from_latent(z0, prior, distance_temperature=float(config["training"].get("distance_temperature", 1.0)))
        dz, dlogits = refiner(x0, uncertainty, z0)
        logits = logits_from_latent(z0 + dz, prior, distance_temperature=float(config["training"].get("distance_temperature", 1.0))) + dlogits
        zq, pred_idx, entropy = quantize_from_logits(prior, logits, soft_temperature=float(config["training"].get("soft_temperature", 1.0)), straight_through=False)
        xg = prior.model.decode_embeddings(zq)
        top1 = float((pred_idx == teacher_idx).float().mean().cpu())
        top5 = float(logits.topk(k=min(5, logits.shape[1]), dim=1).indices.eq(teacher_idx[:, None]).any(dim=1).float().mean().cpu())
        pre = float(mc.pre_audit_rel_mse(xg, y, measurement).cpu())
        truth_chunks.append(x.detach().cpu())
        lmmse_method = "lmmse_anchor"
        per.extend(prediction_metrics(pred=x0, truth=x, y=y, measurement=measurement, lpips_fn=lpips_fn, method=lmmse_method, beta=0.0, source_idx=idx, labels=label, train_seed=train_seed, extra={"source": split_name, "projection_norm": 0.0, "pre_audit_rel": "[DATA MISSING]", "top1": "[DATA MISSING]", "top5": "[DATA MISSING]", "entropy": "[DATA MISSING]", "latent_l1": "[DATA MISSING]"}))
        pred_chunks.setdefault("lmmse_anchor_b0", []).append(x0.detach().cpu())
        for beta in betas:
            pred = null_blend(x0, xg, beta, measurement)
            method = f"{kind}_refiner_nullblend"
            per.extend(prediction_metrics(pred=pred, truth=x, y=y, measurement=measurement, lpips_fn=lpips_fn, method=method, beta=beta, source_idx=idx, labels=label, train_seed=train_seed, extra={"source": split_name, "projection_norm": float((pred - xg).reshape(x.shape[0], -1).norm(dim=1).mean().cpu() / math.sqrt(x.shape[-1] * x.shape[-2])), "pre_audit_rel": pre, "top1": top1, "top5": top5, "entropy": float(entropy.cpu()), "latent_l1": float(dz.abs().mean().cpu())}))
            pred_chunks.setdefault(f"{method}_b{beta:g}", []).append(pred.detach().cpu())
    kid = evaluate_predictions(predictions=pred_chunks, truth_chunks=truth_chunks, lpips_fn=lpips_fn, device=device, kid_max_images=int(config["eval"].get("kid_max_images", 128)))
    for row in per:
        key = f"{row['method']}_b{float(row['beta']):g}"
        row["kid_group"] = kid.get(key, {}).get("kid", "[DATA MISSING]")
    method_rows = summarize_rows(per)
    save_qualitative_from_chunks(out_dir / "figures" / f"{split_name}_{kind}_refiner_qualitative.png", truth_chunks, pred_chunks, preferred=[
        "lmmse_anchor_b0",
        f"{kind}_refiner_nullblend_b0.25",
        f"{kind}_refiner_nullblend_b0.5",
        f"{kind}_refiner_nullblend_b0.75",
        f"{kind}_refiner_nullblend_b1",
    ])
    return method_rows, per, kid


def load_refiner_checkpoint(path: Path, config: Mapping[str, Any], device: torch.device) -> AnchorLatentRefiner:
    ref = AnchorLatentRefiner(z_dim=int(config["model"]["z_dim"]), codebook_size=int(config["model"]["codebook_size"]), base=int(config["model"].get("refiner_base_channels", 64)), delta_scale=float(config["training"].get("delta_scale", 0.25)), logit_scale=float(config["training"].get("logit_scale", 1.0))).to(device)
    payload = torch.load(path, map_location=device)
    ref.load_state_dict(payload["refiner"])
    ref.eval()
    return ref


def paired_bootstrap(rows: Sequence[Mapping[str, Any]], method: str, ref: str, metric: str, *, reps: int, seed: int, beta_method: float | None = None, beta_ref: float | None = None, lower: bool = True) -> dict[str, Any]:
    by: dict[int, dict[str, float]] = {}
    for r in rows:
        if r["method"] not in {method, ref}:
            continue
        if beta_method is not None and r["method"] == method and abs(float(r["beta"]) - float(beta_method)) > 1e-9:
            continue
        if beta_ref is not None and r["method"] == ref and abs(float(r["beta"]) - float(beta_ref)) > 1e-9:
            continue
        try:
            val = float(r[metric])
        except (TypeError, ValueError):
            continue
        by.setdefault(int(r["source_index"]), {})[str(r["method"])] = val
    pairs = [(v[method], v[ref]) for v in by.values() if method in v and ref in v]
    if not pairs:
        return {"status": "NO_PAIRS", "method": method, "reference": ref, "metric": metric}
    arr = np.asarray(pairs, dtype=np.float64)
    delta = arr[:, 0] - arr[:, 1]
    rng = np.random.default_rng(int(seed))
    boots = [float(delta[rng.integers(0, len(delta), size=len(delta))].mean()) for _ in range(int(reps))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ref_mean = float(arr[:, 1].mean())
    rel_gain = (-float(delta.mean()) / max(abs(ref_mean), 1e-12)) if lower else (float(delta.mean()) / max(abs(ref_mean), 1e-12))
    return {"status": "PASS", "method": method, "reference": ref, "metric": metric, "mean_delta": float(delta.mean()), "ci_low": float(lo), "ci_high": float(hi), "relative_gain": float(rel_gain), "n": int(len(delta)), "wins_method": int(np.sum(delta < 0 if lower else delta > 0)), "wins_reference": int(np.sum(delta > 0 if lower else delta < 0))}


def make_reports(out: Path, config: Mapping[str, Any], stage0: Mapping[str, Any], final_gate: Mapping[str, Any], manifests: Sequence[Mapping[str, Any]]) -> None:
    reports = hq.ensure_dir(out / "reports")
    hq.write_json(reports / "gate_report.json", final_gate)
    hq.write_json(reports / "refiner_manifests.json", list(manifests))
    report = [
        "# Anchor-Initialized VQGAN Inversion Report",
        "",
        f"Classification: `{final_gate['classification']}`",
        f"Development gate passed: `{final_gate['development_gate_passed']}`",
        "",
        "## Stage 0 Transfer Ceiling",
        "",
        f"Stage0 decision: `{stage0['decision']}`",
        "",
        "## Main Comparison",
        "",
    ]
    for c in final_gate.get("comparisons", []):
        report.append(f"- {c['metric']}: delta `{c.get('mean_delta')}`, CI `[{c.get('ci_low')}, {c.get('ci_high')}]`, relative gain `{c.get('relative_gain')}`")
    report += [
        "",
        "## Selected Betas",
        "",
        "```json",
        json.dumps(final_gate.get("selected_betas", {}), indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        final_gate.get("interpretation", ""),
    ]
    hq.write_text(reports / "ANCHOR_INITIALIZED_VQGAN_REPORT.md", "\n".join(report) + "\n")
    if len(manifests) >= 2:
        batch_evidence = f"`{manifests[0].get('batch_order_hash')}` / `{manifests[1].get('batch_order_hash')}`"
        batch_status = "supported" if manifests[0].get("batch_order_hash") == manifests[1].get("batch_order_hash") else "failed"
    else:
        batch_evidence = "[DATA MISSING: refiner training skipped]"
        batch_status = "not applicable"
    ledger = [
        "# Claim-Evidence Ledger",
        "",
        "| Claim | Evidence | Status |",
        "|---|---|---|",
        f"| Teacher/Pareto upper bound is useful | `{stage0['decision']}` | {'supported' if stage0['decision'] != 'STOP_NO_TRANSFER_HEADROOM' else 'not supported'} |",
        f"| Matched refiner training was fair | batch hashes {batch_evidence} | {batch_status} |",
        f"| VQGAN inversion beats VQAE inversion | `{final_gate['classification']}` | {'supported' if final_gate['classification'] == 'VQGAN_PRIOR_TRANSFER_CONFIRMED' else 'not supported'} |",
        f"| Measurement consistency holds | max RelMeasErr `{final_gate.get('max_relmeaserr')}` | {'supported' if final_gate['conditions'].get('relmeaserr_ok') else 'failed'} |",
    ]
    hq.write_text(reports / "CLAIM_EVIDENCE_LEDGER.md", "\n".join(ledger) + "\n")


def package_results(out: Path) -> dict[str, Any]:
    files = []
    reports = out / "reports"
    for rel in [
        "config_used.yaml",
        "reports/ANCHOR_INITIALIZED_VQGAN_REPORT.md",
        "reports/CLAIM_EVIDENCE_LEDGER.md",
        "reports/gate_report.json",
        "reports/stage0_val_method_metrics.csv",
        "reports/stage0_val_per_image.csv",
        "reports/stage0_dev_method_metrics.csv",
        "reports/stage0_dev_per_image.csv",
        "reports/final_dev_method_metrics.csv",
        "reports/final_dev_per_image.csv",
        "reports/refiner_manifests.json",
        "reports/runtime_and_hashes.json",
        "reports/split_manifest.json",
        "reports/duplicate_audit.json",
        "reports/sample_hash_audit.csv",
        "reports/operator_manifest.json",
        "reports/lmmse_manifest.json",
        "reports/prior_checkpoint_manifest.json",
        "reports/figures/stage0_val_qualitative.png",
        "reports/figures/final_dev_qualitative.png",
        "reports/figures/pareto_curves.png",
    ]:
        p = out / rel
        if p.exists():
            files.append(p)
    files += [
        ROOT / "anchor_initialized_vqgan_inversion.py",
        ROOT / "configs" / "compatibility" / "anchor_vqgan_inversion_seed0.yaml",
        ROOT / "tests" / "test_anchor_initialized_vqgan_inversion.py",
    ]
    manifest = []
    for p in files:
        if p.exists():
            data = p.read_bytes()
            manifest.append({"relative": str(p.relative_to(ROOT)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    hq.write_json(reports / "PACKAGE_MANIFEST.json", manifest)
    seed_tag = "seed0"
    parts = out.name.split("seed", 1)
    if len(parts) == 2 and parts[1]:
        seed_tag = "seed" + "".join(ch for ch in parts[1] if ch.isdigit())[:4]
    zip_path = out.parent / f"ANCHOR_INITIALIZED_VQGAN_INVERSION_{seed_tag.upper()}_PACKAGE.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for p in files + [reports / "PACKAGE_MANIFEST.json"]:
            if p.exists():
                z.write(p, p.relative_to(ROOT))
    return {"zip_path": str(zip_path), "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(), "zip_bytes": zip_path.stat().st_size}


def run(config_path: Path) -> dict[str, Any]:
    t0 = time.time()
    config = load_yaml(config_path)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    out = hq.ensure_dir(ROOT / str(config["output_dir"]))
    reports = hq.ensure_dir(out / "reports")
    hq.ensure_dir(reports / "figures")
    hq.write_text(out / "config_used.yaml", config_path.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    train_seed_id = int(config.get("experiment_seed", config.get("seed_id", 0)))
    rows_np, op_meta = hq.build_structured_operator_rows(img_size=int(config["data"]["img_size"]), total_m=int(config["operator"]["total_m"]), dct_rows=int(config["operator"]["dct_rows"]), hadamard_rows=int(config["operator"]["hadamard_rows"]), random_rows=int(config["operator"]["random_rows"]), seed=int(config["operator"]["seed"]))
    measurement = hq.make_measurement_operator(rows_np, img_size=int(config["data"]["img_size"]), device=device, lambda_solver=float(config["operator"].get("lambda_solver", 1e-6)))
    train_ds, val_ds, dev_ds, split_manifest = build_split_datasets(config)
    duplicate_audit = hq.save_split_hash_audit(reports / "sample_hash_audit.csv", {"train": train_ds, "val": val_ds, "dev": dev_ds})
    train_x, _labels, _idx = hq.tensor_dataset_to_matrix(train_ds, batch_size=int(config["data"].get("matrix_batch_size", 128)))
    lmmse = hq.EmpiricalLMMSE.fit(train_x, rows_np, lambda_=float(config["operator"].get("lmmse_lambda", 1e-3)))
    hq.write_json(reports / "split_manifest.json", split_manifest)
    hq.write_json(reports / "duplicate_audit.json", duplicate_audit)
    hq.write_json(reports / "operator_manifest.json", op_meta)
    hq.write_json(reports / "lmmse_manifest.json", {"train_count": int(train_x.shape[0]), "lambda": float(config["operator"].get("lmmse_lambda", 1e-3)), "rows_sha256": lmmse.rows_sha256})
    hq.write_json(reports / "prior_checkpoint_manifest.json", {
        VQAE: {"path": str(ROOT / str(config["priors"]["vqae_checkpoint"])), "sha256": hq.sha256_file(ROOT / str(config["priors"]["vqae_checkpoint"]))},
        VQGAN: {"path": str(ROOT / str(config["priors"]["vqgan_checkpoint"])), "sha256": hq.sha256_file(ROOT / str(config["priors"]["vqgan_checkpoint"]))},
    })
    priors = {
        VQAE: load_prior(VQAE, ROOT / str(config["priors"]["vqae_checkpoint"]), config, device),
        VQGAN: load_prior(VQGAN, ROOT / str(config["priors"]["vqgan_checkpoint"]), config, device),
    }
    lpips_fn = ga.freeze_lpips(hq.load_lpips(device) if bool(config["eval"].get("lpips", True)) else {"error": "disabled"})
    val_loader = hq.build_loader(val_ds, batch_size=int(config["data"].get("eval_batch_size", 16)), workers=int(config["data"].get("num_workers", 0)), shuffle=False, seed=int(config["seed"]) + 10, device=device)
    dev_loader = hq.build_loader(dev_ds, batch_size=int(config["data"].get("eval_batch_size", 16)), workers=int(config["data"].get("num_workers", 0)), shuffle=False, seed=int(config["seed"]) + 11, device=device)
    stage0_val_methods, stage0_val_per, _ = run_transfer_ceiling(config=config, priors=priors, loader=val_loader, lmmse=lmmse, measurement=measurement, lpips_fn=lpips_fn, device=device, out=out, split_name="val")
    lmmse_lp = float(next(r for r in stage0_val_methods if r["method"] == "lmmse_anchor")["lpips_mean"])
    vqgan_rows = [r for r in stage0_val_methods if str(r["method"]) == "vqgan_teacher_nullblend" and float(r["beta"]) >= 0]
    vqae_rows = [r for r in stage0_val_methods if str(r["method"]) == "vqae_teacher_nullblend" and float(r["beta"]) >= 0]
    vqgan_best = min(vqgan_rows, key=lambda r: float(r["lpips_mean"])) if vqgan_rows else None
    vqae_best = min(vqae_rows, key=lambda r: float(r["lpips_mean"])) if vqae_rows else None
    stage0_decision = "PASS_TRANSFER_HEADROOM" if vqgan_best and (float(vqgan_best["lpips_mean"]) < min(lmmse_lp, float(vqae_best["lpips_mean"]) if vqae_best else 1e9)) else "STOP_NO_TRANSFER_HEADROOM"
    stage0 = {"decision": stage0_decision, "lmmse_lpips": lmmse_lp, "vqgan_teacher_best": vqgan_best, "vqae_teacher_best": vqae_best}
    hq.write_json(reports / "stage0_decision.json", stage0)
    if stage0_decision == "STOP_NO_TRANSFER_HEADROOM":
        no_duplicates = not duplicate_audit.get("raw_duplicates") and not duplicate_audit.get("transformed_duplicates")
        final_gate = {"classification": "ANCHOR_LATENT_BASELINE_ALREADY_SUFFICIENT" if no_duplicates else "INVALID_EXPERIMENT", "development_gate_passed": False, "conditions": {"stage0_transfer_headroom": False, "relmeaserr_ok": True, "hash_clean_no_raw_or_transformed_duplicates": bool(no_duplicates)}, "comparisons": [], "selected_betas": {}, "max_relmeaserr": "[DATA MISSING]", "interpretation": "Teacher/Pareto upper bound did not beat anchor/VQAE on validation; refiner training was skipped by the preregistered gate."}
        manifests: list[dict[str, Any]] = []
    else:
        set_seed(int(config["seed"]) + 12345)
        start_ref = AnchorLatentRefiner(z_dim=int(config["model"]["z_dim"]), codebook_size=int(config["model"]["codebook_size"]), base=int(config["model"].get("refiner_base_channels", 64)), delta_scale=float(config["training"].get("delta_scale", 0.25)), logit_scale=float(config["training"].get("logit_scale", 1.0))).to(device)
        start_state = {k: v.detach().cpu().clone() for k, v in start_ref.state_dict().items()}
        del start_ref
        ref_vqae, man_vqae = train_refiner(kind=VQAE, prior=priors[VQAE], config=config, train_ds=train_ds, val_loader=val_loader, lmmse=lmmse, measurement=measurement, lpips_fn=lpips_fn, device=device, out=out, seed=train_seed_id, start_state=start_state)
        ref_vqgan, man_vqgan = train_refiner(kind=VQGAN, prior=priors[VQGAN], config=config, train_ds=train_ds, val_loader=val_loader, lmmse=lmmse, measurement=measurement, lpips_fn=lpips_fn, device=device, out=out, seed=train_seed_id, start_state=start_state)
        manifests = [man_vqae, man_vqgan]
        # Evaluate best checkpoints on dev using validation-selected beta.
        best_vqae = load_refiner_checkpoint(Path(man_vqae["best_checkpoint"]), config, device)
        best_vqgan = load_refiner_checkpoint(Path(man_vqgan["best_checkpoint"]), config, device)
        all_methods: list[dict[str, Any]] = []
        all_per: list[dict[str, Any]] = []
        for kind, prior, refiner in [(VQAE, priors[VQAE], best_vqae), (VQGAN, priors[VQGAN], best_vqgan)]:
            methods, per, _ = evaluate_refiner(kind=kind, prior=prior, refiner=refiner, config=config, loader=dev_loader, lmmse=lmmse, measurement=measurement, lpips_fn=lpips_fn, device=device, out_dir=out / "reports", split_name="dev", train_seed=train_seed_id)
            all_methods.extend([dict(r, prior_kind=kind) for r in methods])
            all_per.extend(per)
        write_csv(reports / "final_dev_method_metrics.csv", all_methods)
        write_csv(reports / "final_dev_per_image.csv", all_per)
        sel_vqae = dict(man_vqae.get("best", {}).get("selection", {"status": "NO_VALIDATION_SELECTION"}))
        sel_vqgan = dict(man_vqgan.get("best", {}).get("selection", {"status": "NO_VALIDATION_SELECTION"}))
        if "selected" not in sel_vqae or "selected" not in sel_vqgan:
            raise RuntimeError(f"VALIDATION_BETA_SELECTION_MISSING:vqae={sel_vqae}:vqgan={sel_vqgan}")
        beta_ae = float(sel_vqae["selected"]["beta"])
        beta_gan = float(sel_vqgan["selected"]["beta"])
        make_pareto_figure(reports / "figures" / "pareto_curves.png", all_methods)
        for candidate in [
            reports / "figures" / f"dev_{VQGAN}_refiner_qualitative.png",
            reports / "figures" / f"dev_{VQAE}_refiner_qualitative.png",
        ]:
            if candidate.exists():
                target = reports / "figures" / "final_dev_qualitative.png"
                if candidate != target:
                    target.write_bytes(candidate.read_bytes())
                break
        comparisons = []
        for metric in ["lpips", "rapsd", "psnr", "ssim", "full_rmse", "centered_rmse"]:
            comparisons.append(paired_bootstrap(all_per, "vqgan_refiner_nullblend", "vqae_refiner_nullblend", metric, reps=int(config["eval"].get("bootstrap_reps", 500)), seed=int(config["seed"]) + len(comparisons), beta_method=beta_gan, beta_ref=beta_ae, lower=metric not in {"psnr", "ssim"}))
        rels = [float(r["relmeaserr"]) for r in all_per if "refiner" in r["method"]]
        lp = next(c for c in comparisons if c["metric"] == "lpips")
        psnr = next(c for c in comparisons if c["metric"] == "psnr")
        conditions = {
            "stage0_transfer_headroom": stage0_decision == "PASS_TRANSFER_HEADROOM",
            "matched_batch_order": man_vqae["batch_order_hash"] == man_vqgan["batch_order_hash"],
            "hash_clean_no_raw_or_transformed_duplicates": bool(not duplicate_audit.get("raw_duplicates") and not duplicate_audit.get("transformed_duplicates")),
            "lpips_gain_ge_5pct_ci_upper_lt0": bool(lp["relative_gain"] >= float(config["eval"].get("lpips_relative_gain_gate", 0.05)) and lp["ci_high"] < 0),
            "psnr_within_tolerance": bool(psnr["mean_delta"] >= -float(config["eval"].get("psnr_drop_tolerance_db", 2.5))),
            "relmeaserr_ok": bool(rels and max(rels) < float(config["eval"].get("relmeaserr_limit", 1e-5))),
        }
        if not conditions["matched_batch_order"] or not conditions["relmeaserr_ok"] or not conditions["hash_clean_no_raw_or_transformed_duplicates"]:
            classification = "INVALID_EXPERIMENT"
        elif conditions["lpips_gain_ge_5pct_ci_upper_lt0"] and conditions["psnr_within_tolerance"]:
            classification = "VQGAN_PRIOR_TRANSFER_CONFIRMED"
        elif conditions["lpips_gain_ge_5pct_ci_upper_lt0"] and not conditions["psnr_within_tolerance"]:
            classification = "VQGAN_DISTORTION_TRADEOFF_TOO_LARGE"
        else:
            classification = "VQGAN_PRIOR_HAS_HEADROOM_BUT_ENCODER_BOTTLENECK"
        final_gate = {"classification": classification, "development_gate_passed": classification == "VQGAN_PRIOR_TRANSFER_CONFIRMED", "conditions": conditions, "comparisons": comparisons, "selected_betas": {"vqae": sel_vqae, "vqgan": sel_vqgan}, "max_relmeaserr": max(rels) if rels else "[DATA MISSING]", "interpretation": "Seed0 anchor-initialized refiner canary; locked test is not authorized unless classification confirms transfer."}
    stage0_dev_methods, stage0_dev_per, _ = run_transfer_ceiling(config=config, priors=priors, loader=dev_loader, lmmse=lmmse, measurement=measurement, lpips_fn=lpips_fn, device=device, out=out, split_name="dev")
    make_reports(out, config, stage0, final_gate, manifests)
    runtime = {"seconds": float(time.time() - t0), "device": str(device), "config": str(config_path), "config_sha256": hq.sha256_file(config_path), "script_sha256": hq.sha256_file(ROOT / "anchor_initialized_vqgan_inversion.py")}
    hq.write_json(reports / "runtime_and_hashes.json", runtime)
    package = package_results(out)
    summary = {"classification": final_gate["classification"], "development_gate_passed": final_gate["development_gate_passed"], "output_dir": str(out), "package": package, "runtime": runtime}
    hq.write_json(reports / "summary.json", summary)
    print(json.dumps(hq.json_safe(summary), indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Anchor-initialized VQGAN latent inversion for 5% ghost imaging.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
