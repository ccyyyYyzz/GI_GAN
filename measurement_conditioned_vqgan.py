from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import utils as tv_utils

import gan_high_quality_gi as hq
import gan_gauge_aligned_nsgan as ga
from src.losses import charbonnier_loss, frequency_loss, gradient_difference_loss
from src.metrics import ssim as ssim_metric
from src.projections import get_exact_projector, relative_measurement_error


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "mc_vqgan_smoke.yaml"
VQAE = "vqae"
VQGAN = "vqgan"
VQGAN_GUIDED = "vqgan_guided"


class MCVQGANError(RuntimeError):
    pass


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def stable_hash(obj: Any) -> str:
    buf = torch.save(obj, _ := Path(os.environ.get("TEMP", "/tmp")) / f"mc_vq_hash_{os.getpid()}.pt")
    del buf
    data = _.read_bytes()
    try:
        _.unlink()
    except OSError:
        pass
    return hashlib.sha256(data).hexdigest()


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def batch_stream(dataset: Dataset, *, batch_size: int, workers: int, seed: int, device: torch.device):
    epoch = 0
    while True:
        loader = hq.build_loader(dataset, batch_size=batch_size, workers=workers, shuffle=True, seed=int(seed) + epoch, device=device)
        for x, label, idx in loader:
            yield x, label, idx, epoch
        epoch += 1


def prep_input(x0: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    return torch.cat([x0, uncertainty], dim=1)


def differentiable_lpips(loss_fn: Any, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if isinstance(loss_fn, dict):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return loss_fn(hq.prep_lpips(pred), hq.prep_lpips(target)).mean()


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embed_dim: int, beta: float = 0.25) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.codebook_size, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z.reshape(-1, self.embed_dim)
        emb = self.embedding.weight
        dist = flat.pow(2).sum(dim=1, keepdim=True) + emb.pow(2).sum(dim=1)[None, :] - 2.0 * flat @ emb.t()
        indices = torch.argmin(dist, dim=1)
        z_q = emb[indices].view_as(z).permute(0, 3, 1, 2).contiguous()
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())
        loss = codebook_loss + self.beta * commit_loss
        z_st = z_e + (z_q - z_e).detach()
        encodings = F.one_hot(indices, self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        used = int((avg_probs > 0).sum().detach().cpu())
        stats = {
            "vq_loss": float(loss.detach().cpu()),
            "codebook_loss": float(codebook_loss.detach().cpu()),
            "commit_loss": float(commit_loss.detach().cpu()),
            "perplexity": float(perplexity.detach().cpu()),
            "used_codes": used,
            "dead_code_fraction": float(1.0 - used / max(1, self.codebook_size)),
        }
        return z_st, indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3]), loss, stats

    def lookup_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z = self.embedding(indices.long())
        return z.permute(0, 3, 1, 2).contiguous()

    def soft_embed(self, logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
        probs = F.softmax(logits / max(float(temperature), 1e-6), dim=1)
        return torch.einsum("bkhw,kc->bchw", probs, self.embedding.weight)


class VQAutoencoder(nn.Module):
    def __init__(self, *, codebook_size: int = 128, z_dim: int = 64, base: int = 48, beta: float = 0.25) -> None:
        super().__init__()
        c = int(base)
        z = int(z_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 2), c * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 4), c * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 4, z, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, z), z),
            nn.SiLU(inplace=True),
            nn.Conv2d(z, z, 3, padding=1),
        )
        self.quantizer = VectorQuantizer(codebook_size, z, beta=beta)
        self.decoder = nn.Sequential(
            nn.Conv2d(z, c * 4, 3, padding=1),
            nn.GroupNorm(min(8, c * 4), c * 4),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 4, c * 2, 3, padding=1),
            nn.GroupNorm(min(8, c * 2), c * 2),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 2, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(x)
        _, indices, _, _ = self.quantizer(z_e)
        return indices

    def decode_embeddings(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.decode_embeddings(self.quantizer.lookup_indices(indices))

    def decode_logits(self, logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
        return self.decode_embeddings(self.quantizer.soft_embed(logits, temperature=temperature))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        z_e = self.encode(x)
        z_q, indices, vq_loss, stats = self.quantizer(z_e)
        return self.decode_embeddings(z_q), indices, vq_loss, stats


class FeaturePatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 48) -> None:
        super().__init__()
        c = int(base)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(in_channels, c, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)),
                nn.Sequential(nn.Conv2d(c, c * 2, 4, stride=2, padding=1), nn.GroupNorm(min(8, c * 2), c * 2), nn.LeakyReLU(0.2, inplace=True)),
                nn.Sequential(nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1), nn.GroupNorm(min(8, c * 4), c * 4), nn.LeakyReLU(0.2, inplace=True)),
                nn.Sequential(nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.GroupNorm(min(8, c * 4), c * 4), nn.LeakyReLU(0.2, inplace=True)),
            ]
        )
        self.head = nn.Conv2d(c * 4, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, *, return_features: bool = False):
        feats = []
        h = x
        for block in self.blocks:
            h = block(h)
            feats.append(h)
        score = self.head(h).mean(dim=(1, 2, 3))
        return (score, feats) if return_features else score


class MeasurementEncoder(nn.Module):
    def __init__(self, *, in_channels: int = 2, codebook_size: int = 128, base: int = 48) -> None:
        super().__init__()
        c = int(base)
        k = int(codebook_size)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 2), c * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 4), c * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, c * 4), c * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, 3, padding=1),
            nn.GroupNorm(min(8, c * 4), c * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 4, k, 1),
        )

    def forward(self, x0: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        return self.net(prep_input(x0, uncertainty))


@dataclass
class PriorBundle:
    kind: str
    model: VQAutoencoder
    discriminator: FeaturePatchDiscriminator | None
    checkpoint: Path


def audit_image(x: torch.Tensor, y: torch.Tensor, measurement) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=False):
        projector = get_exact_projector(measurement, dtype=torch.float64, device=x.device)
        flat = measurement.flatten_img(x).double()
        out = projector.audit_flat(flat, y.double())
    return measurement.unflatten_img(out.to(dtype=x.dtype))


def pre_audit_rel_mse(x: torch.Tensor, y: torch.Tensor, measurement) -> torch.Tensor:
    pred_y = measurement.A_forward(measurement.flatten_img(x))
    return ((pred_y - y).pow(2).mean(dim=1) / (y.pow(2).mean(dim=1) + 1e-8)).mean()


def image_base_loss(pred: torch.Tensor, truth: torch.Tensor, cfg: Mapping[str, Any], lpips_fn: Any | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    l_l1 = F.l1_loss(pred, truth)
    l_charb = charbonnier_loss(pred, truth)
    l_grad = gradient_difference_loss(pred, truth)
    l_spec = frequency_loss(pred, truth)
    l_lpips = differentiable_lpips(lpips_fn, pred, truth) if lpips_fn is not None else torch.zeros((), device=pred.device)
    total = (
        float(cfg.get("lambda_l1", 1.0)) * l_l1
        + float(cfg.get("lambda_charb", 1.0)) * l_charb
        + float(cfg.get("lambda_grad", 0.5)) * l_grad
        + float(cfg.get("lambda_spec", 0.1)) * l_spec
        + float(cfg.get("lambda_lpips_train", 0.0)) * l_lpips
    )
    return total, {
        "l1": float(l_l1.detach().cpu()),
        "charb": float(l_charb.detach().cpu()),
        "grad": float(l_grad.detach().cpu()),
        "spec": float(l_spec.detach().cpu()),
        "lpips_train": float(l_lpips.detach().cpu()),
    }


def hinge_d_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()


def feature_matching(real_feats: Sequence[torch.Tensor], fake_feats: Sequence[torch.Tensor]) -> torch.Tensor:
    loss = torch.zeros((), device=fake_feats[0].device)
    for real, fake in zip(real_feats, fake_feats):
        loss = loss + F.l1_loss(fake, real.detach())
    return loss / max(1, len(fake_feats))


def save_ckpt_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    hq.ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def model_from_config(config: Mapping[str, Any], device: torch.device) -> VQAutoencoder:
    m = dict(config["model"])
    return VQAutoencoder(
        codebook_size=int(m.get("codebook_size", 128)),
        z_dim=int(m.get("z_dim", 64)),
        base=int(m.get("base_channels", 48)),
        beta=float(m.get("commit_beta", 0.25)),
    ).to(device)


def disc_from_config(config: Mapping[str, Any], device: torch.device) -> FeaturePatchDiscriminator:
    return FeaturePatchDiscriminator(in_channels=1, base=int(config["model"].get("disc_base_channels", 40))).to(device)


def train_prior_pair(*, seed: int, config: Mapping[str, Any], train_ds: Dataset, dev_ds: Dataset, device: torch.device, out: Path, lpips_fn: Any) -> tuple[PriorBundle, PriorBundle, dict[str, Any]]:
    set_seed(int(config["seed"]) + 1000 * int(seed))
    base_model = model_from_config(config, device)
    start_state = copy.deepcopy(base_model.state_dict())
    start_hash = stable_hash(start_state)
    train_cfg = dict(config["training"])
    batch_size = int(config["data"].get("batch_size", 32))
    workers = int(config["data"].get("num_workers", 0))
    steps = int(train_cfg.get("prior_steps", 50))
    gan_start = int(train_cfg.get("gan_start_step", max(1, steps // 3)))
    branch_manifests = []
    bundles: dict[str, PriorBundle] = {}
    for kind in [VQAE, VQGAN]:
        set_seed(int(config["seed"]) + 1000 * int(seed) + (0 if kind == VQAE else 17))
        model = model_from_config(config, device)
        model.load_state_dict(start_state)
        disc = disc_from_config(config, device) if kind == VQGAN else None
        opt_g = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("prior_lr_g", 2e-4)), betas=tuple(train_cfg.get("betas", [0.5, 0.9])))
        opt_d = torch.optim.Adam(disc.parameters(), lr=float(train_cfg.get("prior_lr_d", 2e-4)), betas=tuple(train_cfg.get("betas", [0.5, 0.9]))) if disc is not None else None
        stream = batch_stream(train_ds, batch_size=batch_size, workers=workers, seed=int(config["seed"]) + 50000 + int(seed), device=device)
        log_rows = []
        batch_order: list[int] = []
        run_dir = hq.ensure_dir(out / "runs" / f"seed{seed}" / f"{kind}_prior")
        for step in range(1, steps + 1):
            x, _label, idx, epoch = next(stream)
            x = x.to(device, non_blocking=True)
            batch_order.extend([int(i) for i in idx])
            gan_active = kind == VQGAN and step > gan_start
            if disc is not None and gan_active:
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_det, _, _, _ = model(x)
                real_score = disc(x)
                fake_score = disc(fake_det.detach())
                d_loss = hinge_d_loss(real_score, fake_score)
                d_loss.backward()
                if float(train_cfg.get("grad_clip", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), float(train_cfg["grad_clip"]))
                opt_d.step()
            else:
                d_loss = torch.zeros((), device=device)
            opt_g.zero_grad(set_to_none=True)
            recon, _indices, vq_loss, vq_stats = model(x)
            base_loss, base_parts = image_base_loss(recon, x, train_cfg, lpips_fn)
            adv = torch.zeros((), device=device)
            fm = torch.zeros((), device=device)
            if disc is not None and gan_active:
                fake_score, fake_feats = disc(recon, return_features=True)
                with torch.no_grad():
                    _real_score, real_feats = disc(x, return_features=True)
                adv = -fake_score.mean()
                fm = feature_matching(real_feats, fake_feats)
            g_loss = base_loss + float(train_cfg.get("lambda_vq", 1.0)) * vq_loss + float(train_cfg.get("lambda_adv", 0.05)) * adv + float(train_cfg.get("lambda_fm", 0.5)) * fm
            g_loss.backward()
            if float(train_cfg.get("grad_clip", 0.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            opt_g.step()
            row = {
                "seed": int(seed),
                "kind": kind,
                "step": int(step),
                "epoch": int(epoch),
                "gan_active": bool(gan_active),
                "g_loss": float(g_loss.detach().cpu()),
                "d_loss": float(d_loss.detach().cpu()),
                "adv": float(adv.detach().cpu()),
                "fm": float(fm.detach().cpu()),
                **base_parts,
                **vq_stats,
            }
            log_rows.append(row)
        ckpt = run_dir / "checkpoints" / f"{kind}_prior_seed{seed}_final.pt"
        save_ckpt_atomic(ckpt, {"model": model.state_dict(), "discriminator": None if disc is None else disc.state_dict(), "config": hq.json_safe(config), "seed": int(seed), "kind": kind})
        hq.write_csv(run_dir / "train_log.csv", log_rows)
        branch_manifests.append({"kind": kind, "start_hash": start_hash, "batch_order_hash": hashlib.sha256(np.asarray(batch_order, dtype=np.int64).tobytes()).hexdigest(), "g_updates": steps, "checkpoint": str(ckpt)})
        model.eval()
        if disc is not None:
            disc.eval()
        bundles[kind] = PriorBundle(kind=kind, model=model, discriminator=disc, checkpoint=ckpt)
    fairness = {
        "seed": int(seed),
        "status": "PASS" if branch_manifests[0]["start_hash"] == branch_manifests[1]["start_hash"] and branch_manifests[0]["batch_order_hash"] == branch_manifests[1]["batch_order_hash"] and branch_manifests[0]["g_updates"] == branch_manifests[1]["g_updates"] else "FAIL",
        "manifests": branch_manifests,
        "only_prior_difference": "VQGAN receives adversarial and feature-matching terms after gan_start_step; architecture/data/G-step/batch-order are matched.",
    }
    return bundles[VQAE], bundles[VQGAN], fairness


def train_measurement_encoder(*, method: str, prior: PriorBundle, seed: int, config: Mapping[str, Any], train_ds: Dataset, lmmse: hq.EmpiricalLMMSE, measurement, device: torch.device, out: Path, lpips_fn: Any) -> tuple[MeasurementEncoder, dict[str, Any]]:
    train_cfg = dict(config["training"])
    batch_size = int(config["data"].get("batch_size", 32))
    workers = int(config["data"].get("num_workers", 0))
    steps = int(train_cfg.get("inverse_steps", 50))
    set_seed(int(config["seed"]) + 200000 + int(seed))
    base = MeasurementEncoder(codebook_size=int(config["model"].get("codebook_size", 128)), base=int(config["model"].get("meas_base_channels", 48))).to(device)
    start_state = copy.deepcopy(base.state_dict())
    enc = MeasurementEncoder(codebook_size=int(config["model"].get("codebook_size", 128)), base=int(config["model"].get("meas_base_channels", 48))).to(device)
    enc.load_state_dict(start_state)
    opt = torch.optim.Adam(enc.parameters(), lr=float(train_cfg.get("inverse_lr", 2e-4)), betas=tuple(train_cfg.get("betas", [0.5, 0.9])))
    stream = batch_stream(train_ds, batch_size=batch_size, workers=workers, seed=int(config["seed"]) + 70000 + int(seed), device=device)
    run_dir = hq.ensure_dir(out / "runs" / f"seed{seed}" / method)
    log_rows = []
    batch_order: list[int] = []
    prior.model.eval()
    if prior.discriminator is not None:
        prior.discriminator.eval()
    for p in prior.model.parameters():
        p.requires_grad_(False)
    if prior.discriminator is not None:
        for p in prior.discriminator.parameters():
            p.requires_grad_(False)
    for step in range(1, steps + 1):
        x, _label, idx, epoch = next(stream)
        x = x.to(device, non_blocking=True)
        batch_order.extend([int(i) for i in idx])
        flat = measurement.flatten_img(x)
        y = measurement.A_forward(flat)
        with torch.no_grad():
            x0_flat = lmmse.anchor(y, measurement, device=device)
            x0 = measurement.unflatten_img(x0_flat)
            uncertainty = lmmse.uncertainty_map(img_size=int(config["data"].get("img_size", 64)), device=device, batch_size=x.shape[0], dtype=x.dtype)
            teacher = prior.model.encode_indices(x)
        opt.zero_grad(set_to_none=True)
        logits = enc(x0, uncertainty)
        token_loss = F.cross_entropy(logits, teacher)
        xg = prior.model.decode_logits(logits, temperature=float(train_cfg.get("soft_decode_temperature", 1.0)))
        xhat = audit_image(xg, y, measurement)
        image_loss, image_parts = image_base_loss(xhat.clamp(0, 1), x, train_cfg, lpips_fn)
        meas_loss = pre_audit_rel_mse(xg, y, measurement)
        guidance = torch.zeros((), device=device)
        if method == VQGAN_GUIDED and prior.discriminator is not None:
            guidance = F.softplus(-prior.discriminator(xg.clamp(0, 1))).mean()
        loss = (
            float(train_cfg.get("lambda_token", 1.0)) * token_loss
            + float(train_cfg.get("lambda_inverse_image", 1.0)) * image_loss
            + float(train_cfg.get("lambda_pre_audit", 0.1)) * meas_loss
            + float(train_cfg.get("lambda_guidance", 0.0)) * guidance
        )
        loss.backward()
        if float(train_cfg.get("grad_clip", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(enc.parameters(), float(train_cfg["grad_clip"]))
        opt.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            top1 = (pred == teacher).float().mean()
            top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices.eq(teacher[:, None]).any(dim=1).float().mean()
            correction = (xhat - xg).reshape(x.shape[0], -1).norm(dim=1).mean() / math.sqrt(x.shape[-1] * x.shape[-2])
        log_rows.append({
            "seed": int(seed),
            "method": method,
            "step": int(step),
            "epoch": int(epoch),
            "loss": float(loss.detach().cpu()),
            "token_loss": float(token_loss.detach().cpu()),
            "meas_loss": float(meas_loss.detach().cpu()),
            "guidance": float(guidance.detach().cpu()),
            "token_top1": float(top1.detach().cpu()),
            "token_top5": float(top5.detach().cpu()),
            "correction_norm": float(correction.detach().cpu()),
            **image_parts,
        })
    ckpt = run_dir / "checkpoints" / f"{method}_encoder_seed{seed}_final.pt"
    save_ckpt_atomic(ckpt, {"encoder": enc.state_dict(), "config": hq.json_safe(config), "seed": int(seed), "method": method, "prior_checkpoint": str(prior.checkpoint)})
    hq.write_csv(run_dir / "train_log.csv", log_rows)
    manifest = {
        "method": method,
        "seed": int(seed),
        "start_hash": stable_hash(start_state),
        "batch_order_hash": hashlib.sha256(np.asarray(batch_order, dtype=np.int64).tobytes()).hexdigest(),
        "updates": int(steps),
        "checkpoint": str(ckpt),
    }
    return enc.eval(), manifest


@torch.no_grad()
def evaluate_all(*, config: Mapping[str, Any], dev_ds: Dataset, lmmse: hq.EmpiricalLMMSE, measurement, device: torch.device, out: Path, seeds: Sequence[int], prior_by_seed: Mapping[int, Mapping[str, PriorBundle]], enc_by_seed: Mapping[int, Mapping[str, MeasurementEncoder]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eval_cfg = dict(config["eval"])
    loader = hq.build_loader(dev_ds, batch_size=int(config["data"].get("eval_batch_size", config["data"].get("batch_size", 32))), workers=int(config["data"].get("num_workers", 0)), shuffle=False, seed=int(config["seed"]) + 123, device=device)
    lpips_fn = hq.load_lpips(device) if bool(eval_cfg.get("lpips", True)) else {"error": "disabled"}
    all_per: list[dict[str, Any]] = []
    all_method: list[dict[str, Any]] = []
    qualitative_payload: dict[str, torch.Tensor] = {}
    truth_chunks: list[torch.Tensor] = []
    by_method_chunks: dict[str, list[torch.Tensor]] = {}
    for seed in seeds:
        priors = prior_by_seed[int(seed)]
        encs = enc_by_seed[int(seed)]
        source_counter = 0
        for x, label, source_idx in loader:
            x = x.to(device, non_blocking=True)
            flat = measurement.flatten_img(x)
            y = measurement.A_forward(flat)
            x0_flat = lmmse.anchor(y, measurement, device=device)
            x0 = measurement.unflatten_img(x0_flat)
            uncertainty = lmmse.uncertainty_map(img_size=int(config["data"].get("img_size", 64)), device=device, batch_size=x.shape[0], dtype=x.dtype)
            preds: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {"lmmse_anchor": (x0, {"token_top1": "[DATA MISSING]", "token_top5": "[DATA MISSING]", "pre_audit_rel": "[DATA MISSING]", "correction_norm": 0.0})}
            for kind, prior in priors.items():
                teacher_idx = prior.model.encode_indices(x)
                teacher_xg = prior.model.decode_indices(teacher_idx)
                teacher_xhat = audit_image(teacher_xg, y, measurement)
                preds[f"{kind}_teacher_oracle"] = (teacher_xhat, {"token_top1": 1.0, "token_top5": 1.0, "pre_audit_rel": float(pre_audit_rel_mse(teacher_xg, y, measurement).cpu()), "correction_norm": float(((teacher_xhat - teacher_xg).reshape(x.shape[0], -1).norm(dim=1).mean() / math.sqrt(x.shape[-1] * x.shape[-2])).cpu())})
            for method, enc in encs.items():
                prior = priors[VQGAN if method.startswith(VQGAN) else VQAE]
                logits = enc(x0, uncertainty)
                pred_idx = logits.argmax(dim=1)
                xg = prior.model.decode_indices(pred_idx)
                xhat = audit_image(xg, y, measurement)
                teacher = prior.model.encode_indices(x)
                top1 = float((pred_idx == teacher).float().mean().cpu())
                top5 = float(logits.topk(k=min(5, logits.shape[1]), dim=1).indices.eq(teacher[:, None]).any(dim=1).float().mean().cpu())
                preds[f"{method}_inversion"] = (xhat, {"token_top1": top1, "token_top5": top5, "pre_audit_rel": float(pre_audit_rel_mse(xg, y, measurement).cpu()), "correction_norm": float(((xhat - xg).reshape(x.shape[0], -1).norm(dim=1).mean() / math.sqrt(x.shape[-1] * x.shape[-2])).cpu())})
            if not truth_chunks:
                truth_chunks.append(x.detach().cpu())
            for method, (pred, extra) in preds.items():
                by_method_chunks.setdefault(method, []).append(pred.detach().cpu())
                clipped = pred.clamp(0, 1)
                lp_vals = hq.lpips_batch(lpips_fn, clipped, x)
                rel = relative_measurement_error(pred, y, measurement).detach().cpu().numpy().astype(np.float64)
                rmse = hq.full_rmse_torch(clipped, x)
                crmse = hq.centered_rmse_torch(clipped, x)
                sharp = hq.edge_sharpness(clipped)
                pred_np = clipped.detach().cpu().numpy()[:, 0]
                truth_np = x.detach().cpu().numpy()[:, 0]
                rapsd = np.asarray([np.linalg.norm(hq.rapsd_np(pred_np[i]) - hq.rapsd_np(truth_np[i])) for i in range(x.shape[0])], dtype=np.float64)
                for i in range(x.shape[0]):
                    all_per.append({
                        "train_seed": int(seed),
                        "method": method,
                        "sample_ordinal": int(source_counter + i),
                        "source_index": int(source_idx[i]),
                        "label": int(label[i]),
                        "full_rmse": float(rmse[i]),
                        "centered_rmse": float(crmse[i]),
                        "psnr": float(-20.0 * math.log10(max(float(rmse[i]), 1e-12))),
                        "ssim": float(ssim_metric(clipped[i:i+1], x[i:i+1])),
                        "lpips": "[DATA MISSING]" if lp_vals is None else float(lp_vals[i]),
                        "rapsd": float(rapsd[i]),
                        "edge_sharpness": float(sharp[i]),
                        "relmeaserr": float(rel[i]),
                        "token_top1": extra["token_top1"],
                        "token_top5": extra["token_top5"],
                        "pre_audit_rel": extra["pre_audit_rel"],
                        "correction_norm": extra["correction_norm"],
                    })
            source_counter += x.shape[0]
    for method in sorted(by_method_chunks):
        vals = [r for r in all_per if r["method"] == method]
        row: dict[str, Any] = {"method": method, "n": len(vals)}
        for metric in ["full_rmse", "centered_rmse", "psnr", "ssim", "rapsd", "edge_sharpness", "relmeaserr"]:
            arr = np.asarray([float(v[metric]) for v in vals], dtype=np.float64)
            row[f"{metric}_mean"] = float(arr.mean())
        for metric in ["lpips", "token_top1", "token_top5", "pre_audit_rel", "correction_norm"]:
            arr = []
            for v in vals:
                try:
                    arr.append(float(v[metric]))
                except (TypeError, ValueError):
                    pass
            row[f"{metric}_mean"] = float(np.mean(arr)) if arr else "[DATA MISSING]"
        all_method.append(row)
    truth_cat = truth_chunks[0] if truth_chunks else torch.empty(0)
    pred_cat = {k: torch.cat(v, dim=0)[: truth_cat.shape[0]] for k, v in by_method_chunks.items()}
    hq.save_qualitative_grid(out / "reports" / "figures" / "mc_vqgan_qualitative.png", truth_cat, pred_cat, max_items=int(eval_cfg.get("qualitative_count", 10)))
    return all_method, all_per, {"lpips_status": "PASS" if not isinstance(lpips_fn, dict) else lpips_fn}


def paired_delta(per_rows: Sequence[Mapping[str, Any]], method: str, reference: str, metric: str, *, reps: int, seed: int, lower_is_better: bool = True) -> dict[str, Any]:
    by: dict[tuple[int, int], dict[str, float]] = {}
    for r in per_rows:
        if r["method"] not in {method, reference}:
            continue
        try:
            val = float(r[metric])
        except (TypeError, ValueError):
            continue
        by.setdefault((int(r.get("train_seed", -1)), int(r["source_index"])), {})[str(r["method"])] = val
    pairs = [(v[method], v[reference]) for v in by.values() if method in v and reference in v]
    if not pairs:
        return {"comparison": f"{method}_vs_{reference}", "method": method, "reference": reference, "metric": metric, "status": "NO_PAIRS"}
    arr = np.asarray(pairs, dtype=np.float64)
    delta = arr[:, 0] - arr[:, 1]
    rng = np.random.default_rng(int(seed))
    boots = []
    for _ in range(int(reps)):
        idx = rng.integers(0, len(delta), size=len(delta))
        boots.append(float(delta[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ref_mean = float(arr[:, 1].mean())
    rel_gain = (-float(delta.mean()) / max(abs(ref_mean), 1e-12)) if lower_is_better else (float(delta.mean()) / max(abs(ref_mean), 1e-12))
    return {
        "comparison": f"{method}_vs_{reference}",
        "method": method,
        "reference": reference,
        "metric": metric,
        "status": "PASS",
        "n_pairs": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "reference_mean": ref_mean,
        "relative_gain": float(rel_gain),
        "wins_method": int(np.sum(delta < 0 if lower_is_better else delta > 0)),
        "wins_reference": int(np.sum(delta > 0 if lower_is_better else delta < 0)),
    }


def summarize_gate(all_per: Sequence[Mapping[str, Any]], all_method: Sequence[Mapping[str, Any]], fairness: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    eval_cfg = dict(config["eval"])
    reps = int(eval_cfg.get("bootstrap_replicates", 500))
    seed = int(eval_cfg.get("bootstrap_seed", 20260625))
    comparisons = []
    for metric in ["lpips", "rapsd", "full_rmse", "centered_rmse", "psnr", "ssim"]:
        comparisons.append(paired_delta(all_per, "vqgan_inversion", "vqae_inversion", metric, reps=reps, seed=seed + len(comparisons), lower_is_better=metric not in {"psnr", "ssim"}))
    teacher_lpips = paired_delta(all_per, "vqgan_teacher_oracle", "vqae_teacher_oracle", "lpips", reps=reps, seed=seed + 101)
    guided_lpips = paired_delta(all_per, "vqgan_guided_inversion", "vqae_inversion", "lpips", reps=reps, seed=seed + 202) if any(r["method"] == "vqgan_guided_inversion" for r in all_per) else {"status": "NOT_RUN"}
    method_rows = {r["method"]: r for r in all_method}
    seed_dirs = []
    for train_seed in sorted({int(r["train_seed"]) for r in all_per}):
        vals = {}
        for r in all_per:
            if int(r["train_seed"]) == train_seed and r["method"] in {"vqgan_inversion", "vqae_inversion"}:
                try:
                    vals.setdefault(r["method"], []).append(float(r["lpips"]))
                except (TypeError, ValueError):
                    pass
        if vals.get("vqgan_inversion") and vals.get("vqae_inversion"):
            seed_dirs.append(float(np.mean(vals["vqgan_inversion"])) < float(np.mean(vals["vqae_inversion"])))
    lp = next(c for c in comparisons if c["metric"] == "lpips")
    rap = next(c for c in comparisons if c["metric"] == "rapsd")
    psnr = next(c for c in comparisons if c["metric"] == "psnr")
    rels = [float(r["relmeaserr"]) for r in all_per if r["method"] in {"vqae_inversion", "vqgan_inversion", "vqgan_guided_inversion"}]
    conditions = {
        "fairness_pass": bool(fairness and all(f.get("status") == "PASS" for f in fairness)),
        "stage_a_vqgan_teacher_lpips_better": bool(teacher_lpips.get("status") == "PASS" and teacher_lpips["mean_delta"] < 0 and teacher_lpips["ci_high"] < 0),
        "vqgan_inversion_lpips_gain_ge_5pct_ci_upper_lt0": bool(lp.get("status") == "PASS" and lp["mean_delta"] < 0 and lp["ci_high"] < 0 and lp["relative_gain"] >= float(eval_cfg.get("lpips_relative_gain_gate", 0.05))),
        "two_of_three_seeds_same_direction": bool(sum(seed_dirs) >= 2 and len(seed_dirs) >= 3),
        "rapsd_same_direction": bool(rap.get("status") == "PASS" and rap["mean_delta"] < 0),
        "psnr_not_below_tolerance": bool(psnr.get("status") == "PASS" and psnr["mean_delta"] >= -float(eval_cfg.get("psnr_drop_tolerance_db", 0.5))),
        "relmeaserr_ok": bool(rels and max(rels) <= float(eval_cfg.get("relmeaserr_limit", 1e-5))),
    }
    if not conditions["fairness_pass"] or not conditions["relmeaserr_ok"]:
        classification = "INVALID_EXPERIMENT"
    elif conditions["vqgan_inversion_lpips_gain_ge_5pct_ci_upper_lt0"] and conditions["two_of_three_seeds_same_direction"] and conditions["rapsd_same_direction"] and conditions["psnr_not_below_tolerance"]:
        classification = "VQGAN_PRIOR_GAIN_CONFIRMED"
    elif guided_lpips.get("status") == "PASS" and guided_lpips["mean_delta"] < 0 and guided_lpips["ci_high"] < 0 and guided_lpips["relative_gain"] >= float(eval_cfg.get("lpips_relative_gain_gate", 0.05)):
        classification = "LATENT_REFINEMENT_RESCUES_GAN_PRIOR"
    elif conditions["stage_a_vqgan_teacher_lpips_better"]:
        classification = "GAN_PRIOR_GOOD_BUT_INVERSION_BOTTLENECK"
    else:
        classification = "VQGAN_PRIOR_NOT_BETTER_THAN_VQAE"
    return {
        "classification": classification,
        "development_gate_passed": classification in {"VQGAN_PRIOR_GAIN_CONFIRMED", "LATENT_REFINEMENT_RESCUES_GAN_PRIOR"},
        "locked_test_authorized": classification in {"VQGAN_PRIOR_GAIN_CONFIRMED", "LATENT_REFINEMENT_RESCUES_GAN_PRIOR"},
        "conditions": conditions,
        "comparisons": comparisons,
        "teacher_lpips": teacher_lpips,
        "guided_lpips": guided_lpips,
        "seed_dirs": seed_dirs,
        "max_relmeaserr": max(rels) if rels else "[DATA MISSING]",
        "method_means": all_method,
    }


def write_reports(out: Path, config: Mapping[str, Any], gate: Mapping[str, Any], fairness: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any]) -> None:
    reports = hq.ensure_dir(out / "reports")
    hq.write_json(reports / "gate_report.json", gate)
    hq.write_json(reports / "fairness_manifest.json", list(fairness))
    math = r"""# Measurement-Conditioned VQGAN Inversion

The prior stage learns a discrete image manifold
\[
z_q=Q(E_x(x)),\qquad x_g=G(z_q).
\]
The inverse stage freezes \(Q,G\) and predicts code logits from the audited LMMSE anchor and posterior uncertainty:
\[
\ell_\phi=F_\phi(x_0,U_A),\qquad x_g=G(Q(\ell_\phi)).
\]
The final estimator is the exact audited image
\[
\hat x=\Pi_y(x_g)=x_g-A^\top(AA^\top)^{-1}(Ax_g-y)=A^\dagger y+P_0^A x_g.
\]
Since \(A\Pi_y(x_g)=y\), the generated image can provide only null-space visual content after the audit has fixed the measured row-space content.

The matched causal question is whether the adversarially pretrained VQGAN prior transfers through the same measurement encoder better than the same-architecture VQAE prior. The measurement encoder, data, loss, update count, and seeds are matched; the only intended difference is the frozen prior.
"""
    hq.write_text(reports / "math_and_system_description.md", math)
    key = {c["metric"]: c for c in gate["comparisons"] if c.get("comparison") == "vqgan_inversion_vs_vqae_inversion"}
    lp = key.get("lpips", {})
    summary = [
        "# Measurement-Conditioned VQGAN Development Summary",
        "",
        f"Classification: `{gate['classification']}`",
        f"Development gate passed: `{gate['development_gate_passed']}`",
        "",
        "## Main Result",
        "",
        f"- VQGAN inversion vs VQAE inversion LPIPS delta: `{lp.get('mean_delta', '[DATA MISSING]')}`",
        f"- 95% CI: `[{lp.get('ci_low', '[DATA MISSING]')}, {lp.get('ci_high', '[DATA MISSING]')}]`",
        f"- Relative LPIPS gain: `{lp.get('relative_gain', '[DATA MISSING]')}`",
        f"- Max RelMeasErr: `{gate.get('max_relmeaserr')}`",
        "",
        "## Gate Conditions",
        "",
        "```json",
        json.dumps(gate["conditions"], indent=2),
        "```",
        "",
        "## Locked Decision",
        "",
        "A locked split is opened only if the development gate passes. This run did not score any locked split unless `development_gate_passed` is true.",
        "",
        "## Unique Next Step",
        "",
        "If the VQGAN teacher is better but inversion is not, the bottleneck is measurement-to-token inference; move to a stronger token posterior or masked-token transformer. If the VQGAN teacher is not better, the VQ latent prior itself is the bottleneck and a continuous pretrained prior is the next route.",
    ]
    hq.write_text(reports / "FINAL_MC_VQGAN_DEVELOPMENT_SUMMARY.md", "\n".join(summary) + "\n")
    ledger = [
        "# Claim-Evidence Ledger",
        "",
        "| Claim | Evidence | Status |",
        "|---|---|---|",
        f"| Matched VQAE/VQGAN prior fork preserved | fairness status `{all(f.get('status') == 'PASS' for f in fairness)}` | {'supported' if all(f.get('status') == 'PASS' for f in fairness) else 'failed'} |",
        f"| VQGAN inversion beats VQAE inversion by preregistered LPIPS gate | classification `{gate['classification']}` | {'supported' if gate['classification'] == 'VQGAN_PRIOR_GAIN_CONFIRMED' else 'not supported'} |",
        f"| Measurement consistency maintained | max RelMeasErr `{gate.get('max_relmeaserr')}` | {'supported' if gate['conditions'].get('relmeaserr_ok') else 'failed'} |",
        f"| Locked test authorized | development gate `{gate['development_gate_passed']}` | {'yes' if gate['development_gate_passed'] else 'no'} |",
    ]
    hq.write_text(reports / "FINAL_MC_VQGAN_CLAIM_EVIDENCE_LEDGER.md", "\n".join(ledger) + "\n")
    hq.write_json(reports / "runtime_and_hashes.json", runtime)


def package_results(out: Path) -> dict[str, Any]:
    package = out.parent / "MEASUREMENT_CONDITIONED_VQGAN_PACKAGE.zip"
    reports = out / "reports"
    files = [
        out / "config_used.yaml",
        reports / "FINAL_MC_VQGAN_DEVELOPMENT_SUMMARY.md",
        reports / "MC_VQGAN_DEVELOPMENT_REPORT.md",
        reports / "FINAL_MC_VQGAN_CLAIM_EVIDENCE_LEDGER.md",
        reports / "gate_report.json",
        reports / "method_metrics.csv",
        reports / "per_image_metrics.csv",
        reports / "fairness_manifest.json",
        reports / "math_and_system_description.md",
        reports / "runtime_and_hashes.json",
        reports / "split_manifest.json",
        reports / "operator_manifest.json",
        reports / "lmmse_manifest.json",
        reports / "figures" / "mc_vqgan_qualitative.png",
        reports / "figures" / "mc_vqgan_labeled_qualitative.png",
        reports / "figures" / "mc_vqgan_paired_deltas.png",
        reports / "figures" / "mc_vqgan_method_means.png",
        ROOT / "measurement_conditioned_vqgan.py",
        ROOT / "configs" / "compatibility" / "mc_vqgan_smoke.yaml",
        ROOT / "configs" / "compatibility" / "mc_vqgan_dev_64_5pct.yaml",
        ROOT / "tests" / "test_measurement_conditioned_vqgan.py",
    ]
    manifest = []
    for p in files:
        if p.exists():
            data = p.read_bytes()
            manifest.append({"relative": str(p.relative_to(ROOT)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    hq.write_json(reports / "FINAL_PACKAGE_MANIFEST.json", manifest)
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as zf:
        for p in files + [reports / "FINAL_PACKAGE_MANIFEST.json"]:
            if p.exists():
                zf.write(p, p.relative_to(ROOT))
    return {"zip_path": str(package), "zip_bytes": package.stat().st_size, "zip_sha256": hashlib.sha256(package.read_bytes()).hexdigest()}


def run(config_path: Path, *, mode: str = "train_eval") -> dict[str, Any]:
    t0 = time.time()
    config = hq.load_yaml(config_path)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    out = hq.ensure_dir(ROOT / str(config["output_dir"]))
    reports = hq.ensure_dir(out / "reports")
    hq.write_text(out / "config_used.yaml", config_path.read_text(encoding="utf-8"))
    hq.ensure_dir(reports / "figures")
    rows, op_meta = hq.build_structured_operator_rows(
        img_size=int(config["data"]["img_size"]),
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    measurement = hq.make_measurement_operator(rows, img_size=int(config["data"]["img_size"]), device=device, lambda_solver=float(config["operator"].get("lambda_solver", 1e-8)))
    train_ds, val_ds, dev_ds, split_manifest = hq.build_split_datasets(config)
    del val_ds
    train_x, _labels, _idx = hq.tensor_dataset_to_matrix(train_ds, batch_size=int(config["data"].get("matrix_batch_size", 128)))
    lmmse = hq.EmpiricalLMMSE.fit(train_x, rows, lambda_=float(config["operator"].get("lmmse_lambda", 1e-3)))
    hq.write_json(reports / "split_manifest.json", split_manifest)
    hq.write_json(reports / "operator_manifest.json", op_meta)
    hq.write_json(reports / "lmmse_manifest.json", {"train_count": int(train_x.shape[0]), "lambda": float(config["operator"].get("lmmse_lambda", 1e-3)), "rows_sha256": lmmse.rows_sha256})
    lpips_fn = ga.freeze_lpips(hq.load_lpips(device) if bool(config["training"].get("lpips_train", True)) else {"error": "disabled"})
    seeds = [int(s) for s in config["training"].get("seeds", [0])]
    prior_by_seed: dict[int, dict[str, PriorBundle]] = {}
    enc_by_seed: dict[int, dict[str, MeasurementEncoder]] = {}
    fairness_all = []
    for seed in seeds:
        vqae, vqgan, fair = train_prior_pair(seed=seed, config=config, train_ds=train_ds, dev_ds=dev_ds, device=device, out=out, lpips_fn=lpips_fn)
        fairness_all.append(fair)
        prior_by_seed[int(seed)] = {VQAE: vqae, VQGAN: vqgan}
        encs: dict[str, MeasurementEncoder] = {}
        enc_ae, man_ae = train_measurement_encoder(method=VQAE, prior=vqae, seed=seed, config=config, train_ds=train_ds, lmmse=lmmse, measurement=measurement, device=device, out=out, lpips_fn=lpips_fn)
        enc_gan, man_gan = train_measurement_encoder(method=VQGAN, prior=vqgan, seed=seed, config=config, train_ds=train_ds, lmmse=lmmse, measurement=measurement, device=device, out=out, lpips_fn=lpips_fn)
        encs[VQAE] = enc_ae
        encs[VQGAN] = enc_gan
        if bool(config["training"].get("run_guided", False)):
            enc_guided, man_guided = train_measurement_encoder(method=VQGAN_GUIDED, prior=vqgan, seed=seed, config=config, train_ds=train_ds, lmmse=lmmse, measurement=measurement, device=device, out=out, lpips_fn=lpips_fn)
            encs[VQGAN_GUIDED] = enc_guided
        enc_by_seed[int(seed)] = encs
    all_method, all_per, eval_diag = evaluate_all(config=config, dev_ds=dev_ds, lmmse=lmmse, measurement=measurement, device=device, out=out, seeds=seeds, prior_by_seed=prior_by_seed, enc_by_seed=enc_by_seed)
    hq.write_csv(reports / "method_metrics.csv", all_method)
    hq.write_csv(reports / "per_image_metrics.csv", all_per)
    gate = summarize_gate(all_per, all_method, fairness_all, config)
    runtime = {
        "config": str(config_path),
        "config_sha256": hq.sha256_file(config_path),
        "script_sha256": hq.sha256_file(ROOT / "measurement_conditioned_vqgan.py"),
        "device": str(device),
        "seconds": float(time.time() - t0),
        "eval_diag": eval_diag,
    }
    write_reports(out, config, gate, fairness_all, runtime)
    package = package_results(out)
    summary = {
        "classification": gate["classification"],
        "development_gate_passed": gate["development_gate_passed"],
        "locked_test_authorized": gate["locked_test_authorized"],
        "output_dir": str(out),
        "package": package,
        "runtime_seconds": runtime["seconds"],
    }
    hq.write_json(reports / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measurement-conditioned VQGAN inversion for low-rate ghost imaging.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config.")
    parser.add_argument("--mode", choices=["train_eval"], default="train_eval", help="Run mode.")
    args = parser.parse_args(argv)
    run(args.config, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
