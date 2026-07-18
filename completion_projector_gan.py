from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils as tv_utils

import gan_high_quality_gi as hq
from src.gauge_geometry import (
    GaugeEmpiricalAnchor,
    GaugeGeometry,
    GaugeDykstraResult,
    project_box_fiber_exact_dual,
    project_box_fiber_q,
)
from src.losses import charbonnier_loss, differentiable_ssim_loss, gradient_difference_loss
from src.metrics import ssim as ssim_metric
from src.projector_gated_fiber_gan import (
    FiberConditionalDiscriminator,
    ProjectorGatedFiberGenerator,
    parameter_count,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "completion_gan_round18" / "smoke.yaml"


class PQBFGANError(RuntimeError):
    pass


@dataclass(frozen=True)
class FiberCacheDiagnostics:
    split: str
    count: int
    max_anchor_relative_record_error: float
    max_anchor_box_violation: float
    max_anchor_iterations: int
    source_indices_sha256: str
    max_anchor_intrinsic_infinity_residual: float = 0.0
    max_anchor_proximal_residual: float = 0.0
    max_anchor_complementarity_residual: float = 0.0


class FiberCacheDataset(Dataset):
    """Cached truth, bounded anchor, intrinsic record, and original record."""

    def __init__(
        self,
        *,
        truth: torch.Tensor,
        anchor: torch.Tensor,
        intrinsic: torch.Tensor,
        measurement: torch.Tensor,
        label: torch.Tensor,
        source_index: torch.Tensor,
    ) -> None:
        lengths = {
            int(value.shape[0])
            for value in [truth, anchor, intrinsic, measurement, label, source_index]
        }
        if len(lengths) != 1:
            raise PQBFGANError(f"CACHE_LENGTH_MISMATCH:{sorted(lengths)}")
        self.truth = truth.contiguous()
        self.anchor = anchor.contiguous()
        self.intrinsic = intrinsic.contiguous()
        self.measurement = measurement.contiguous()
        self.label = label.contiguous()
        self.source_index = source_index.contiguous()

    def __len__(self) -> int:
        return int(self.truth.shape[0])

    def __getitem__(self, index: int):
        i = int(index)
        return (
            self.truth[i],
            self.anchor[i],
            self.intrinsic[i],
            self.measurement[i],
            self.label[i],
            self.source_index[i],
        )


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PQBFGANError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return payload


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def build_hash_clean_splits(
    config: Mapping[str, Any],
) -> tuple[hq.IndexedTensorDataset, hq.IndexedTensorDataset, hq.IndexedTensorDataset, dict[str, Any]]:
    data_cfg = dict(config["data"])
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
    names = ["train", "val", "test"]
    counts = {
        "train": int(data_cfg["train_count"]),
        "val": int(data_cfg["val_count"]),
        "test": int(data_cfg.get("test_count", data_cfg.get("dev_count", 0))),
    }
    total_needed = sum(counts.values())
    seen: set[str] = set()
    unique_indices: list[int] = []
    skipped = 0
    for source_index in range(len(base)):
        raw_hash = probe.raw_hash(source_index)
        if raw_hash in seen:
            skipped += 1
            continue
        seen.add(raw_hash)
        unique_indices.append(int(source_index))
        if len(unique_indices) == total_needed:
            break
    if len(unique_indices) != total_needed:
        raise PQBFGANError(
            f"HASH_CLEAN_SPLIT_TOO_SMALL:{len(unique_indices)}:{total_needed}"
        )

    cursor = 0
    split_datasets: list[hq.IndexedTensorDataset] = []
    split_meta: dict[str, Any] = {}
    for name in names:
        indices = unique_indices[cursor : cursor + counts[name]]
        cursor += counts[name]
        split_datasets.append(hq.IndexedTensorDataset(base, indices, transform))
        packed = np.asarray(indices, dtype=np.int64)
        split_meta[name] = {
            "count": int(len(indices)),
            "min_source_index": int(min(indices)),
            "max_source_index": int(max(indices)),
            "source_indices_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        }
    manifest = {
        "dataset": "STL10",
        "source_split": source_split,
        "dataset_root": root,
        "transform": f"resize_{img_size}x{img_size}_grayscale_tensor_0_1",
        "allocation": "first raw-hash-unique images in source order",
        "scanned_through_source_index": int(unique_indices[-1]),
        "raw_duplicates_skipped_before_cut": int(skipped),
        "splits": split_meta,
    }
    return split_datasets[0], split_datasets[1], split_datasets[2], manifest


def load_or_run_split_audit(
    path: Path,
    datasets_by_name: Mapping[str, hq.IndexedTensorDataset],
) -> dict[str, Any]:
    expected = {
        (str(split), int(source_index))
        for split, dataset in datasets_by_name.items()
        for source_index in dataset.indices
    }
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        observed = {(str(row["split"]), int(row["source_index"])) for row in rows}
        if observed == expected and len(rows) == len(expected):
            raw = [str(row["raw_sha256"]) for row in rows]
            transformed = [str(row["transformed_sha256"]) for row in rows]
            return {
                "raw_duplicates": [] if len(raw) == len(set(raw)) else ["duplicate"],
                "transformed_duplicates": (
                    [] if len(transformed) == len(set(transformed)) else ["duplicate"]
                ),
                "rows": len(rows),
                "reused": True,
            }
    result = hq.save_split_hash_audit(path, datasets_by_name)
    result["reused"] = False
    return result


def _cache_identity(
    *, split: str, dataset: hq.IndexedTensorDataset, geometry: GaugeGeometry
) -> dict[str, Any]:
    indices = np.asarray(dataset.indices, dtype=np.int64)
    return {
        "format": "pqbf_fiber_cache_v2_kkt",
        "split": str(split),
        "count": int(len(dataset)),
        "source_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "rows_sha256": geometry.info.rows_sha256,
        "rank": geometry.rank,
    }


@torch.no_grad()
def build_fiber_cache(
    *,
    split: str,
    dataset: hq.IndexedTensorDataset,
    geometry: GaugeGeometry,
    anchor_model: GaugeEmpiricalAnchor,
    rows64: torch.Tensor,
    device: torch.device,
    batch_size: int,
    cache_path: Path,
    reuse: bool,
) -> tuple[FiberCacheDataset, FiberCacheDiagnostics]:
    identity = _cache_identity(split=split, dataset=dataset, geometry=geometry)
    if reuse and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("identity") == identity:
            cached = FiberCacheDataset(**payload["tensors"])
            diagnostics = FiberCacheDiagnostics(**payload["diagnostics"])
            return cached, diagnostics

    partial_path = cache_path.with_suffix(cache_path.suffix + ".partial")
    truth_chunks: list[torch.Tensor] = []
    anchor_chunks: list[torch.Tensor] = []
    intrinsic_chunks: list[torch.Tensor] = []
    measurement_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    max_record = 0.0
    max_box = 0.0
    max_iterations = 0
    max_infinity = 0.0
    max_proximal = 0.0
    max_complementarity = 0.0
    img_size = int(round(math.sqrt(geometry.n)))
    processed = 0
    if reuse and partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu")
        if partial.get("identity") == identity:
            tensors = partial["tensors"]
            prefix = tensors["source_index"].to(torch.int64).tolist()
            if prefix == dataset.indices[: len(prefix)]:
                truth_chunks.append(tensors["truth"])
                anchor_chunks.append(tensors["anchor"])
                intrinsic_chunks.append(tensors["intrinsic"])
                measurement_chunks.append(tensors["measurement"])
                label_chunks.append(tensors["label"])
                index_chunks.append(tensors["source_index"])
                processed = len(prefix)
                progress = dict(partial.get("progress", {}))
                max_record = float(progress.get("max_record", 0.0))
                max_box = float(progress.get("max_box", 0.0))
                max_iterations = int(progress.get("max_iterations", 0))
                max_infinity = float(progress.get("max_infinity", 0.0))
                max_proximal = float(progress.get("max_proximal", 0.0))
                max_complementarity = float(progress.get("max_complementarity", 0.0))

    remaining_dataset = hq.IndexedTensorDataset(
        dataset.base, dataset.indices[processed:], dataset.transform
    )
    loader = DataLoader(
        remaining_dataset, batch_size=int(batch_size), shuffle=False, num_workers=0
    )

    def consolidate_chunks() -> dict[str, torch.Tensor]:
        return {
            "truth": torch.cat(truth_chunks),
            "anchor": torch.cat(anchor_chunks),
            "intrinsic": torch.cat(intrinsic_chunks),
            "measurement": torch.cat(measurement_chunks),
            "label": torch.cat(label_chunks),
            "source_index": torch.cat(index_chunks),
        }

    def save_partial() -> None:
        tensors = consolidate_chunks()
        payload = {
            "identity": identity,
            "tensors": tensors,
            "progress": {
                "processed": int(processed),
                "max_record": float(max_record),
                "max_box": float(max_box),
                "max_iterations": int(max_iterations),
                "max_infinity": float(max_infinity),
                "max_proximal": float(max_proximal),
                "max_complementarity": float(max_complementarity),
            },
        }
        hq.ensure_dir(partial_path.parent)
        temporary = partial_path.with_suffix(partial_path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, partial_path)
        truth_chunks[:] = [tensors["truth"]]
        anchor_chunks[:] = [tensors["anchor"]]
        intrinsic_chunks[:] = [tensors["intrinsic"]]
        measurement_chunks[:] = [tensors["measurement"]]
        label_chunks[:] = [tensors["label"]]
        index_chunks[:] = [tensors["source_index"]]

    for truth, label, source_index in loader:
        truth = truth.to(device=device, dtype=torch.float32, non_blocking=True)
        flat64 = truth.reshape(truth.shape[0], -1).to(torch.float64)
        y64 = flat64 @ rows64.T
        affine_anchor, intrinsic = anchor_model(y64, geometry)
        projection = project_box_fiber_exact_dual(
            affine_anchor,
            intrinsic,
            geometry,
            record_tolerance=1e-10,
            step_tolerance=1e-8,
        )
        if not projection.converged:
            raise PQBFGANError(
                "BOUNDED_ANCHOR_DID_NOT_CONVERGE:"
                f"{split}:offset={processed}:{projection.max_relative_record_error}:"
                f"{projection.max_step_change}:{projection.iterations}"
            )
        bounded = projection.image_flat.reshape(-1, 1, img_size, img_size)
        truth_chunks.append(truth.detach().cpu())
        anchor_chunks.append(bounded.to(torch.float32).detach().cpu())
        intrinsic_chunks.append(intrinsic.detach().cpu().to(torch.float64))
        measurement_chunks.append(y64.detach().cpu().to(torch.float64))
        label_chunks.append(label.detach().cpu().to(torch.int64))
        index_chunks.append(source_index.detach().cpu().to(torch.int64))
        max_record = max(max_record, projection.max_relative_record_error)
        max_box = max(max_box, projection.max_box_violation)
        max_iterations = max(max_iterations, projection.iterations)
        max_infinity = max(max_infinity, projection.max_intrinsic_infinity_residual)
        max_proximal = max(max_proximal, projection.max_proximal_residual)
        max_complementarity = max(
            max_complementarity, projection.max_complementarity_residual
        )
        processed += int(truth.shape[0])
        if processed % max(1, int(batch_size) * 8) == 0:
            save_partial()

    final_tensors = consolidate_chunks()
    cached = FiberCacheDataset(**final_tensors)
    diagnostics = FiberCacheDiagnostics(
        split=str(split),
        count=len(cached),
        max_anchor_relative_record_error=float(max_record),
        max_anchor_box_violation=float(max_box),
        max_anchor_iterations=int(max_iterations),
        source_indices_sha256=identity["source_indices_sha256"],
        max_anchor_intrinsic_infinity_residual=float(max_infinity),
        max_anchor_proximal_residual=float(max_proximal),
        max_anchor_complementarity_residual=float(max_complementarity),
    )
    hq.ensure_dir(cache_path.parent)
    torch.save(
        {
            "identity": identity,
            "tensors": {
                "truth": cached.truth,
                "anchor": cached.anchor,
                "intrinsic": cached.intrinsic,
                "measurement": cached.measurement,
                "label": cached.label,
                "source_index": cached.source_index,
            },
            "diagnostics": diagnostics.__dict__,
        },
        cache_path,
    )
    if partial_path.exists():
        partial_path.unlink()
    return cached, diagnostics


def batch_stream(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, ...]]:
    epoch = 0
    while True:
        loader = hq.build_loader(
            dataset,
            batch_size=int(batch_size),
            workers=int(workers),
            shuffle=True,
            seed=int(seed) + epoch,
            device=device,
        )
        for batch in loader:
            yield (*batch, torch.tensor(epoch, dtype=torch.int64))
        epoch += 1


def prep_lpips_no_clamp(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    return 2.0 * image - 1.0


def freeze_lpips(loss_fn: Any) -> Any:
    if isinstance(loss_fn, dict):
        return loss_fn
    loss_fn.eval()
    for parameter in loss_fn.parameters():
        parameter.requires_grad_(False)
    return loss_fn


def lpips_loss(loss_fn: Any, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    if isinstance(loss_fn, dict):
        raise PQBFGANError(f"LPIPS_UNAVAILABLE:{loss_fn}")
    return loss_fn(prep_lpips_no_clamp(pred), prep_lpips_no_clamp(truth)).mean()


def supervised_loss(
    *,
    prediction: torch.Tensor,
    raw_prediction: torch.Tensor,
    anchor: torch.Tensor,
    truth: torch.Tensor,
    geometry: GaugeGeometry,
    lpips_fn: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l2 = F.mse_loss(prediction, truth)
    charb = charbonnier_loss(prediction, truth, eps=1e-3)
    ssim = differentiable_ssim_loss(prediction, truth)
    lpips = lpips_loss(lpips_fn, prediction, truth)
    grad = gradient_difference_loss(prediction, truth)
    raw_null = geometry.project_feature_maps(raw_prediction - anchor, null=True)
    target_null = geometry.project_feature_maps(truth - anchor, null=True).detach()
    null = F.l1_loss(raw_null, target_null)
    move = F.l1_loss(prediction, raw_prediction)
    total = (
        5.0 * l2
        + 2.0 * charb
        + ssim
        + 0.5 * lpips
        + 0.25 * grad
        + 0.25 * null
        + 0.10 * move
    )
    return total, {
        "l2": l2,
        "charb": charb,
        "ssim_loss": ssim,
        "lpips": lpips,
        "gradient": grad,
        "null": null,
        "move": move,
    }


def projected_generator_output(
    generator: ProjectorGatedFiberGenerator,
    *,
    anchor: torch.Tensor,
    uncertainty: torch.Tensor,
    intrinsic: torch.Tensor,
    geometry: GaugeGeometry,
    train_iterations: int,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    with torch.cuda.amp.autocast(enabled=bool(amp)):
        generated = generator(anchor, uncertainty)
    with torch.cuda.amp.autocast(enabled=False):
        projection = project_box_fiber_q(
            generated.raw_image.reshape(anchor.shape[0], -1).float(),
            intrinsic.float(),
            geometry,
            iterations=int(train_iterations),
            exact=False,
        )
    prediction = projection.image_flat.reshape_as(anchor)
    return prediction, generated.raw_image, generated.diagnostics


def _scheduler_factor(step: int, *, total: int, warmup: int, min_ratio: float) -> float:
    current = int(step)
    if warmup > 0 and current < warmup:
        return max(1.0 / float(warmup), (current + 1) / float(warmup))
    span = max(1, int(total) - int(warmup))
    progress = min(1.0, max(0.0, (current - int(warmup)) / float(span)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio) + (1.0 - float(min_ratio)) * cosine


def save_generator_checkpoint(
    path: Path,
    *,
    generator: ProjectorGatedFiberGenerator,
    ema: hq.ModelEMA,
    optimizer: torch.optim.Optimizer,
    stage: str,
    step: int,
    config: Mapping[str, Any],
) -> None:
    hq.ensure_dir(path.parent)
    torch.save(
        {
            "stage": str(stage),
            "step": int(step),
            "generator": generator.state_dict(),
            "ema": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": hq.json_safe(config),
        },
        path,
    )


def train_stage_a(
    *,
    generator: ProjectorGatedFiberGenerator,
    geometry: GaugeGeometry,
    uncertainty: torch.Tensor,
    train_data: FiberCacheDataset,
    lpips_fn: Any,
    config: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[hq.ModelEMA, list[dict[str, Any]], dict[str, Any]]:
    cfg = dict(config["training"])
    steps = int(cfg["stage_a_steps"])
    batch_size = int(config["data"]["batch_size"])
    workers = int(config["data"].get("num_workers", 0))
    amp = bool(cfg.get("amp", True) and device.type == "cuda")
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=float(cfg.get("stage_a_lr", 2e-4)),
        betas=tuple(float(v) for v in cfg.get("betas", [0.5, 0.9])),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ema = hq.ModelEMA(generator, decay=float(cfg.get("ema_decay", 0.999)))
    stream = batch_stream(
        train_data,
        batch_size=batch_size,
        workers=workers,
        seed=int(config["seeds"]["loader"]),
        device=device,
    )
    order_hash = hashlib.sha256()
    log: list[dict[str, Any]] = []
    generator.train()
    started = time.time()
    for step in range(steps):
        truth, anchor, intrinsic, _y, _label, source_index, epoch = next(stream)
        truth = truth.to(device, non_blocking=True)
        anchor = anchor.to(device, non_blocking=True)
        intrinsic = intrinsic.to(device, non_blocking=True)
        sigma = uncertainty.expand(truth.shape[0], -1, -1, -1)
        order_hash.update(source_index.numpy().astype(np.int64).tobytes())
        optimizer.zero_grad(set_to_none=True)
        prediction, raw_prediction, diagnostics = projected_generator_output(
            generator,
            anchor=anchor,
            uncertainty=sigma,
            intrinsic=intrinsic,
            geometry=geometry,
            train_iterations=int(cfg.get("training_dykstra_iterations", 12)),
            amp=amp,
        )
        loss, parts = supervised_loss(
            prediction=prediction,
            raw_prediction=raw_prediction,
            anchor=anchor,
            truth=truth,
            geometry=geometry,
            lpips_fn=lpips_fn,
        )
        if not torch.isfinite(loss):
            raise PQBFGANError(f"NONFINITE_STAGE_A_LOSS:{step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            generator.parameters(), float(cfg.get("grad_clip", 1.0))
        )
        scaler.step(optimizer)
        scaler.update()
        ema.update(generator)
        factor = _scheduler_factor(
            step + 1,
            total=steps,
            warmup=int(cfg.get("stage_a_warmup_steps", 200)),
            min_ratio=float(cfg.get("stage_a_min_lr", 5e-5))
            / float(cfg.get("stage_a_lr", 2e-4)),
        )
        for group in optimizer.param_groups:
            group["lr"] = float(cfg.get("stage_a_lr", 2e-4)) * factor
        row = {
            "stage": "A",
            "step": step + 1,
            "epoch": int(epoch),
            "loss_sup": float(loss.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{key: float(value.detach().cpu()) for key, value in parts.items()},
            **{key: float(value.detach().cpu()) for key, value in diagnostics.items()},
        }
        log.append(row)
        if (step + 1) == steps or (step + 1) % max(1, int(cfg.get("log_every", 20))) == 0:
            hq.write_csv(output_dir / "stage_a_train_log.csv", log)
    checkpoint = output_dir / "checkpoints" / f"stage_a_step{steps:06d}.pt"
    save_generator_checkpoint(
        checkpoint,
        generator=generator,
        ema=ema,
        optimizer=optimizer,
        stage="A",
        step=steps,
        config=config,
    )
    return ema, log, {
        "steps": steps,
        "batch_order_sha256": order_hash.hexdigest(),
        "checkpoint": str(checkpoint),
        "runtime_seconds": time.time() - started,
    }


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(bool(enabled))


def _tail_parameters(generator: ProjectorGatedFiberGenerator) -> list[nn.Parameter]:
    layer = generator.shared_step.correction_head[-1]
    return [parameter for parameter in layer.parameters() if parameter.requires_grad]


def _gradient_norm(
    loss: torch.Tensor, parameters: Sequence[nn.Parameter], *, retain_graph: bool
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        list(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    values = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    if not values:
        return torch.zeros((), device=loss.device)
    return torch.stack(values).sum().sqrt()


def train_stage_b(
    *,
    generator: ProjectorGatedFiberGenerator,
    geometry: GaugeGeometry,
    uncertainty: torch.Tensor,
    train_data: FiberCacheDataset,
    lpips_fn: Any,
    config: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[hq.ModelEMA, FiberConditionalDiscriminator, list[dict[str, Any]], dict[str, Any]]:
    cfg = dict(config["training"])
    steps = int(cfg["stage_b_steps"])
    batch_size = int(config["data"]["batch_size"])
    workers = int(config["data"].get("num_workers", 0))
    amp = bool(cfg.get("amp", True) and device.type == "cuda")
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(cfg.get("stage_b_lr_g", 1e-4)),
        betas=tuple(float(v) for v in cfg.get("betas", [0.5, 0.9])),
    )
    set_seed(int(config["seeds"]["discriminator"]))
    discriminator = FiberConditionalDiscriminator().to(device)
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(cfg.get("stage_b_lr_d", 2e-4)),
        betas=tuple(float(v) for v in cfg.get("betas", [0.5, 0.9])),
    )
    scaler_g = torch.cuda.amp.GradScaler(enabled=amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=amp)
    ema = hq.ModelEMA(generator, decay=float(cfg.get("ema_decay", 0.999)))
    stream = batch_stream(
        train_data,
        batch_size=batch_size,
        workers=workers,
        seed=int(config["seeds"]["loader"]),
        device=device,
    )
    order_hash = hashlib.sha256()
    target_ratio = float(cfg.get("adaptive_target_ratio", 0.10))
    adaptive_lambda = float(cfg.get("adaptive_lambda_initial", 0.01))
    last_sup_norm = float("nan")
    last_adv_norm = float("nan")
    last_achieved_ratio = 0.0
    update_every = int(cfg.get("adaptive_update_every", 20))
    ramp_steps = int(cfg.get("adversarial_ramp_steps", 300))
    r1_interval = int(cfg.get("r1_interval", 16))
    r1_gamma = float(cfg.get("r1_gamma", 1.0))
    checkpoint_steps = {
        int(value) for value in cfg.get("stage_b_checkpoint_steps", [])
    }
    checkpoint_steps.add(steps)
    if any(value <= 0 or value > steps for value in checkpoint_steps):
        raise PQBFGANError(
            f"INVALID_STAGE_B_CHECKPOINT_STEPS:{sorted(checkpoint_steps)}:{steps}"
        )
    checkpoint_paths: list[Path] = []
    log: list[dict[str, Any]] = []
    generator.train()
    discriminator.train()
    started = time.time()
    for step in range(steps):
        truth, anchor, intrinsic, _y, _label, source_index, epoch = next(stream)
        truth = truth.to(device, non_blocking=True)
        anchor = anchor.to(device, non_blocking=True)
        intrinsic = intrinsic.to(device, non_blocking=True)
        sigma = uncertainty.expand(truth.shape[0], -1, -1, -1)
        order_hash.update(source_index.numpy().astype(np.int64).tobytes())

        _set_requires_grad(discriminator, True)
        optimizer_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake_for_d, _raw, _diagnostics = projected_generator_output(
                generator,
                anchor=anchor,
                uncertainty=sigma,
                intrinsic=intrinsic,
                geometry=geometry,
                train_iterations=int(cfg.get("training_dykstra_iterations", 12)),
                amp=amp,
            )
        do_r1 = r1_interval > 0 and (step + 1) % r1_interval == 0
        real_for_d = truth.detach().requires_grad_(do_r1)
        with torch.cuda.amp.autocast(enabled=amp):
            real_logits = discriminator(anchor.detach(), real_for_d)
            fake_logits = discriminator(anchor.detach(), fake_for_d.detach())
            d_hinge = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
        r1 = torch.zeros((), device=device)
        if do_r1:
            real_scores_per_sample = real_logits.flatten(1).mean(dim=1)
            real_gradient = torch.autograd.grad(
                real_scores_per_sample.sum(),
                real_for_d,
                create_graph=True,
                retain_graph=True,
            )[0]
            r1 = real_gradient.float().square().reshape(truth.shape[0], -1).sum(dim=1).mean()
        d_loss = d_hinge + (0.5 * r1_gamma * r1 if do_r1 else 0.0)
        if not torch.isfinite(d_loss):
            raise PQBFGANError(f"NONFINITE_STAGE_B_D_LOSS:{step + 1}")
        scaler_d.scale(d_loss).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()

        _set_requires_grad(discriminator, False)
        optimizer_g.zero_grad(set_to_none=True)
        prediction, raw_prediction, diagnostics = projected_generator_output(
            generator,
            anchor=anchor,
            uncertainty=sigma,
            intrinsic=intrinsic,
            geometry=geometry,
            train_iterations=int(cfg.get("training_dykstra_iterations", 12)),
            amp=amp,
        )
        sup, sup_parts = supervised_loss(
            prediction=prediction,
            raw_prediction=raw_prediction,
            anchor=anchor,
            truth=truth,
            geometry=geometry,
            lpips_fn=lpips_fn,
        )
        with torch.no_grad():
            _real_logits_g, real_features = discriminator(
                anchor, truth, return_features=True
            )
        fake_logits_g, fake_features = discriminator(
            anchor, prediction, return_features=True
        )
        g_adv = -fake_logits_g.mean()
        feature_matching = torch.stack(
            [
                F.l1_loss(fake_feature, real_feature.detach())
                for fake_feature, real_feature in zip(fake_features, real_features)
            ]
        ).mean()
        adversarial_bundle = g_adv + 2.0 * feature_matching
        if step == 0 or (step + 1) % update_every == 0:
            tail = _tail_parameters(generator)
            sup_norm = _gradient_norm(sup, tail, retain_graph=True)
            adv_norm = _gradient_norm(adversarial_bundle, tail, retain_graph=True)
            raw_lambda = torch.clamp(
                target_ratio * sup_norm / (adv_norm + 1e-8),
                min=float(cfg.get("adaptive_lambda_min", 5e-4)),
                max=float(cfg.get("adaptive_lambda_max", 5e-2)),
            )
            adaptive_lambda = 0.9 * adaptive_lambda + 0.1 * float(raw_lambda.detach().cpu())
            last_sup_norm = float(sup_norm.detach().cpu())
            last_adv_norm = float(adv_norm.detach().cpu())
        ramp = min(1.0, (step + 1) / float(max(1, ramp_steps)))
        effective_lambda = adaptive_lambda * ramp
        if math.isfinite(last_sup_norm) and last_sup_norm > 0:
            last_achieved_ratio = effective_lambda * last_adv_norm / last_sup_norm
        g_loss = sup + effective_lambda * adversarial_bundle
        if not torch.isfinite(g_loss):
            raise PQBFGANError(f"NONFINITE_STAGE_B_G_LOSS:{step + 1}")
        scaler_g.scale(g_loss).backward()
        scaler_g.unscale_(optimizer_g)
        g_grad_norm = torch.nn.utils.clip_grad_norm_(
            generator.parameters(), float(cfg.get("grad_clip", 1.0))
        )
        scaler_g.step(optimizer_g)
        scaler_g.update()
        ema.update(generator)
        _set_requires_grad(discriminator, True)

        row = {
            "stage": "B",
            "step": step + 1,
            "epoch": int(epoch),
            "d_loss": float(d_loss.detach().cpu()),
            "d_hinge": float(d_hinge.detach().cpu()),
            "r1": float(r1.detach().cpu()),
            "real_margin": float(real_logits.detach().mean().cpu()),
            "fake_margin": float(fake_logits.detach().mean().cpu()),
            "discriminator_gap": float((real_logits.detach().mean() - fake_logits.detach().mean()).cpu()),
            "g_loss": float(g_loss.detach().cpu()),
            "g_adv": float(g_adv.detach().cpu()),
            "feature_matching": float(feature_matching.detach().cpu()),
            "loss_sup": float(sup.detach().cpu()),
            "adaptive_lambda": float(adaptive_lambda),
            "effective_lambda": float(effective_lambda),
            "tail_grad_sup": float(last_sup_norm),
            "tail_grad_adv": float(last_adv_norm),
            "achieved_adv_to_sup_ratio": float(last_achieved_ratio),
            "g_grad_norm": float(torch.as_tensor(g_grad_norm).detach().cpu()),
            **{key: float(value.detach().cpu()) for key, value in sup_parts.items()},
            **{key: float(value.detach().cpu()) for key, value in diagnostics.items()},
        }
        log.append(row)
        if (step + 1) == steps or (step + 1) % max(1, int(cfg.get("log_every", 20))) == 0:
            hq.write_csv(output_dir / "stage_b_train_log.csv", log)
        if (step + 1) in checkpoint_steps:
            checkpoint = output_dir / "checkpoints" / f"stage_b_step{step + 1:06d}.pt"
            hq.ensure_dir(checkpoint.parent)
            torch.save(
                {
                    "stage": "B",
                    "step": step + 1,
                    "generator": generator.state_dict(),
                    "ema": ema.module.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "config": hq.json_safe(config),
                },
                checkpoint,
            )
            checkpoint_paths.append(checkpoint)

    checkpoint = checkpoint_paths[-1]
    return ema, discriminator, log, {
        "generator_updates": steps,
        "discriminator_updates": steps,
        "batch_order_sha256": order_hash.hexdigest(),
        "checkpoint": str(checkpoint),
        "checkpoints": [
            {
                "step": int(path.stem.rsplit("step", 1)[-1]),
                "path": str(path),
                "sha256": hq.sha256_file(path),
            }
            for path in checkpoint_paths
        ],
        "runtime_seconds": time.time() - started,
    }


@torch.no_grad()
def exact_model_prediction(
    model: ProjectorGatedFiberGenerator,
    *,
    anchor: torch.Tensor,
    intrinsic: torch.Tensor,
    uncertainty: torch.Tensor,
    geometry: GaugeGeometry,
) -> tuple[torch.Tensor, GaugeDykstraResult]:
    model.eval()
    generated = model(anchor, uncertainty)
    projection = project_box_fiber_exact_dual(
        generated.raw_image.reshape(anchor.shape[0], -1).to(torch.float64),
        intrinsic.to(torch.float64),
        geometry,
        record_tolerance=1e-10,
        step_tolerance=1e-8,
    )
    if not projection.converged:
        raise PQBFGANError(
            "EVAL_PROJECTION_DID_NOT_CONVERGE:"
            f"{projection.max_relative_record_error}:"
            f"{projection.max_step_change}:{projection.iterations}"
        )
    return projection.image_flat.reshape_as(anchor).to(torch.float32), projection


def paired_bootstrap(
    values: np.ndarray, *, reps: int, seed: int
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    sampled = np.empty(int(reps), dtype=np.float64)
    for index in range(int(reps)):
        draw = rng.integers(0, len(array), size=len(array))
        sampled[index] = float(array[draw].mean())
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci_low_95": float(np.quantile(sampled, 0.025)),
        "ci_high_95": float(np.quantile(sampled, 0.975)),
        "one_sided_lower_95": float(np.quantile(sampled, 0.05)),
    }


@torch.no_grad()
def evaluate_methods(
    *,
    methods: Mapping[str, ProjectorGatedFiberGenerator | None],
    data: FiberCacheDataset,
    geometry: GaugeGeometry,
    rows64: torch.Tensor,
    uncertainty: torch.Tensor,
    lpips_fn: Any,
    config: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    batch_size = int(config["data"].get("eval_batch_size", config["data"]["batch_size"]))
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    per_image: list[dict[str, Any]] = []
    galleries: dict[str, list[torch.Tensor]] = {name: [] for name in methods}
    truth_gallery: list[torch.Tensor] = []
    projection_diagnostics: dict[str, dict[str, Any]] = {}
    for name, model in methods.items():
        ordinal = 0
        max_iterations = 0
        max_intrinsic_residual = 0.0
        max_intrinsic_infinity = 0.0
        max_proximal = 0.0
        max_complementarity = 0.0
        for truth, anchor, intrinsic, measurement, label, source_index in loader:
            truth = truth.to(device, non_blocking=True)
            anchor = anchor.to(device, non_blocking=True)
            intrinsic = intrinsic.to(device, non_blocking=True)
            measurement = measurement.to(device, non_blocking=True)
            if model is None:
                prediction = anchor
                projection_iterations = 0
                intrinsic_residual = float(
                    geometry.relative_record_error(
                        prediction.reshape(prediction.shape[0], -1).to(torch.float64),
                        intrinsic,
                    )
                    .max()
                    .cpu()
                )
                intrinsic_infinity = 0.0
                proximal_residual = 0.0
                complementarity_residual = 0.0
            else:
                sigma = uncertainty.expand(truth.shape[0], -1, -1, -1)
                prediction, projection = exact_model_prediction(
                    model,
                    anchor=anchor,
                    intrinsic=intrinsic,
                    uncertainty=sigma,
                    geometry=geometry,
                )
                projection_iterations = projection.iterations
                intrinsic_residual = projection.max_relative_record_error
                intrinsic_infinity = projection.max_intrinsic_infinity_residual
                proximal_residual = projection.max_proximal_residual
                complementarity_residual = projection.max_complementarity_residual
            if float(prediction.min()) < -1e-7 or float(prediction.max()) > 1.0 + 1e-7:
                raise PQBFGANError(
                    f"EVAL_OUTPUT_OUTSIDE_BOX:{name}:{float(prediction.min())}:{float(prediction.max())}"
                )
            flat64 = prediction.reshape(prediction.shape[0], -1).to(torch.float64)
            rel_intrinsic_score = geometry.relative_record_error(flat64, intrinsic)
            predicted_y = flat64 @ rows64.T
            rel_original = torch.linalg.norm(predicted_y - measurement, dim=1) / torch.linalg.norm(
                measurement, dim=1
            ).clamp_min(1e-12)
            if float(rel_intrinsic_score.max()) > 1e-6 or float(rel_original.max()) > 1e-6:
                raise PQBFGANError(
                    f"SCORED_TENSOR_RECORD_AUDIT_FAILED:{name}:"
                    f"{float(rel_intrinsic_score.max())}:{float(rel_original.max())}"
                )
            rmse = (prediction - truth).float().square().mean(dim=(1, 2, 3)).sqrt()
            psnr = -20.0 * torch.log10(rmse.clamp_min(1e-12))
            if isinstance(lpips_fn, dict):
                raise PQBFGANError(f"LPIPS_UNAVAILABLE_DURING_EVAL:{lpips_fn}")
            lpips_values = lpips_fn(
                prep_lpips_no_clamp(prediction), prep_lpips_no_clamp(truth)
            ).reshape(-1)
            for item in range(truth.shape[0]):
                per_image.append(
                    {
                        "method": name,
                        "sample_ordinal": ordinal + item,
                        "source_index": int(source_index[item]),
                        "label": int(label[item]),
                        "psnr": float(psnr[item].cpu()),
                        "ssim": float(ssim_metric(prediction[item : item + 1], truth[item : item + 1])),
                        "lpips": float(lpips_values[item].cpu()),
                        "relative_original_measurement_residual": float(rel_original[item].cpu()),
                        "relative_intrinsic_record_residual": float(rel_intrinsic_score[item].cpu()),
                        "pixel_min": float(prediction[item].min().cpu()),
                        "pixel_max": float(prediction[item].max().cpu()),
                    }
                )
            if len(galleries[name]) < 2:
                remaining = max(0, int(config["eval"].get("qualitative_count", 6)) - sum(v.shape[0] for v in galleries[name]))
                if remaining:
                    galleries[name].append(prediction[:remaining].detach().cpu())
                    if name == next(iter(methods)):
                        truth_gallery.append(truth[:remaining].detach().cpu())
            ordinal += truth.shape[0]
            max_iterations = max(max_iterations, int(projection_iterations))
            max_intrinsic_residual = max(max_intrinsic_residual, float(intrinsic_residual))
            max_intrinsic_infinity = max(max_intrinsic_infinity, float(intrinsic_infinity))
            max_proximal = max(max_proximal, float(proximal_residual))
            max_complementarity = max(
                max_complementarity, float(complementarity_residual)
            )
        projection_diagnostics[name] = {
            "max_exact_projection_iterations": int(max_iterations),
            "max_intrinsic_record_error": float(max_intrinsic_residual),
            "max_intrinsic_infinity_residual": float(max_intrinsic_infinity),
            "max_proximal_residual": float(max_proximal),
            "max_complementarity_residual": float(max_complementarity),
        }

    method_rows: list[dict[str, Any]] = []
    for name in methods:
        rows = [row for row in per_image if row["method"] == name]
        method_rows.append(
            {
                "method": name,
                "n": len(rows),
                "psnr_mean": float(np.mean([row["psnr"] for row in rows])),
                "ssim_mean": float(np.mean([row["ssim"] for row in rows])),
                "lpips_mean": float(np.mean([row["lpips"] for row in rows])),
                "max_relative_original_measurement_residual": float(
                    max(row["relative_original_measurement_residual"] for row in rows)
                ),
                "max_relative_intrinsic_record_residual": float(
                    max(row["relative_intrinsic_record_residual"] for row in rows)
                ),
                "pixel_min": float(min(row["pixel_min"] for row in rows)),
                "pixel_max": float(max(row["pixel_max"] for row in rows)),
            }
        )

    by_method = {
        name: sorted(
            [row for row in per_image if row["method"] == name],
            key=lambda row: int(row["sample_ordinal"]),
        )
        for name in methods
    }
    comparisons: list[dict[str, Any]] = []
    bootstrap_reps = int(config["eval"].get("bootstrap_reps", 500))
    bootstrap_seed = int(config["seeds"]["bootstrap"])
    comparison_pairs = [
        ("pqbf_content", "box_fiber_lmmse"),
        ("pqbf_gan", "box_fiber_lmmse"),
        ("pqbf_gan", "pqbf_content"),
    ]
    for method in methods:
        if method.startswith("pqbf_gan_step"):
            comparison_pairs.extend(
                [
                    (method, "box_fiber_lmmse"),
                    (method, "pqbf_content"),
                ]
            )
    for method, reference in comparison_pairs:
        if method not in by_method or reference not in by_method:
            continue
        for metric in ["psnr", "ssim", "lpips"]:
            method_values = np.asarray([row[metric] for row in by_method[method]], dtype=np.float64)
            reference_values = np.asarray([row[metric] for row in by_method[reference]], dtype=np.float64)
            delta = method_values - reference_values
            statistics = paired_bootstrap(
                delta,
                reps=bootstrap_reps,
                seed=bootstrap_seed + len(comparisons),
            )
            comparisons.append(
                {
                    "comparison": f"{method}_minus_{reference}",
                    "metric": metric,
                    **statistics,
                    "relative_reduction": (
                        float((reference_values.mean() - method_values.mean()) / reference_values.mean())
                        if metric == "lpips"
                        else "not_applicable"
                    ),
                }
            )

    count = int(config["eval"].get("qualitative_count", 6))
    if truth_gallery:
        image_rows = [torch.cat(truth_gallery, dim=0)[:count]]
        labels = ["truth"]
        for name in methods:
            if galleries[name]:
                image_rows.append(torch.cat(galleries[name], dim=0)[:count])
                labels.append(name)
        grid = torch.cat(image_rows, dim=0)
        figure_path = output_dir / "qualitative_grid.png"
        hq.ensure_dir(figure_path.parent)
        tv_utils.save_image(grid, figure_path, nrow=count, padding=2)
        hq.write_text(figure_path.with_suffix(".txt"), "Rows: " + ", ".join(labels) + "\n")
    hq.write_csv(output_dir / "per_image_metrics.csv", per_image)
    hq.write_csv(output_dir / "method_metrics.csv", method_rows)
    hq.write_json(output_dir / "paired_comparisons.json", comparisons)
    hq.write_json(output_dir / "projection_diagnostics.json", projection_diagnostics)
    return method_rows, per_image, {
        "comparisons": comparisons,
        "projection": projection_diagnostics,
    }


def summarize_game(log: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not log:
        return {"status": "NO_STAGE_B_ROWS"}
    tail = list(log[-min(1000, len(log)) :])
    medians = {}
    for key in [
        "d_loss",
        "g_adv",
        "real_margin",
        "fake_margin",
        "discriminator_gap",
        "adaptive_lambda",
        "achieved_adv_to_sup_ratio",
    ]:
        medians[f"tail_median_{key}"] = float(np.median([float(row[key]) for row in tail]))
    medians["generator_updates"] = len(log)
    medians["discriminator_updates"] = len(log)
    medians["healthy_reference_ranges"] = {
        "d_loss": [0.3, 1.7],
        "discriminator_gap": [0.2, 2.5],
        "achieved_adv_to_sup_ratio": [0.03, 0.20],
    }
    return medians


def run(config_path: Path) -> dict[str, Any]:
    started = time.time()
    config = load_config(config_path)
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise PQBFGANError("CUDA_REQUIRED_BUT_UNAVAILABLE")
    device = torch.device(requested_device)
    output_dir = Path(str(config["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    hq.ensure_dir(output_dir)
    hq.ensure_dir(output_dir / "reports")
    hq.write_text(output_dir / "config_used.yaml", config_path.read_text(encoding="utf-8"))
    set_seed(int(config["seeds"]["global_generator"]))

    train_ds, val_ds, test_ds, split_manifest = build_hash_clean_splits(config)
    duplicate_audit = load_or_run_split_audit(
        output_dir / "reports" / "sample_hash_audit.csv",
        {"train": train_ds, "val": val_ds, "test": test_ds},
    )
    if duplicate_audit["raw_duplicates"] or duplicate_audit["transformed_duplicates"]:
        raise PQBFGANError(f"SPLIT_DUPLICATE_AUDIT_FAILED:{duplicate_audit}")
    rows_np, operator_manifest = hq.build_structured_operator_rows(
        img_size=int(config["data"]["img_size"]),
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    rows64 = torch.from_numpy(rows_np).to(device=device, dtype=torch.float64)
    geometry = GaugeGeometry(
        torch.from_numpy(rows_np).to(torch.float64),
        relative_cutoff=float(config["operator"].get("svd_relative_cutoff", 1e-12)),
    ).to(device)
    expected_rank = int(config["operator"].get("expected_rank", 200))
    if geometry.rank != expected_rank:
        raise PQBFGANError(f"OPERATOR_RANK_MISMATCH:{geometry.rank}:{expected_rank}")
    if operator_manifest["rows_sha256"] != geometry.info.rows_sha256:
        raise PQBFGANError(
            f"OPERATOR_HASH_MISMATCH:{operator_manifest['rows_sha256']}:{geometry.info.rows_sha256}"
        )

    train_matrix, _train_labels, _train_indices = hq.tensor_dataset_to_matrix(
        train_ds, batch_size=int(config["data"].get("matrix_batch_size", 128))
    )
    anchor_model = GaugeEmpiricalAnchor.fit(
        train_matrix,
        geometry,
        lambda_=float(config["operator"].get("lmmse_lambda", 1e-3)),
    ).to(device)
    uncertainty = anchor_model.normalized_posterior_map(
        img_size=int(config["data"]["img_size"]),
        device=device,
    )
    cache_dir = Path(str(config["data"].get("cache_dir", output_dir / "cache")))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    caches: dict[str, FiberCacheDataset] = {}
    cache_diagnostics: dict[str, Any] = {}
    for split, dataset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        cache, diagnostics = build_fiber_cache(
            split=split,
            dataset=dataset,
            geometry=geometry,
            anchor_model=anchor_model,
            rows64=rows64,
            device=device,
            batch_size=int(config["data"].get("cache_batch_size", 16)),
            cache_path=cache_dir / f"{split}.pt",
            reuse=bool(config["data"].get("reuse_cache", True)),
        )
        caches[split] = cache
        cache_diagnostics[split] = diagnostics.__dict__

    generator = ProjectorGatedFiberGenerator(
        geometry,
        steps=int(config["model"].get("steps", 3)),
        step_scale=float(config["model"].get("step_scale", 0.25)),
    ).to(device)
    if parameter_count(generator) != 787_107:
        raise PQBFGANError(f"GENERATOR_PARAMETER_COUNT:{parameter_count(generator)}")
    lpips_fn = freeze_lpips(hq.load_lpips(device))
    if isinstance(lpips_fn, dict):
        raise PQBFGANError(f"LPIPS_REQUIRED:{lpips_fn}")
    probe = torch.full((1, 1, 64, 64), 0.5, device=device, requires_grad=True)
    probe_loss = lpips_loss(lpips_fn, probe, torch.zeros_like(probe))
    probe_gradient = torch.autograd.grad(probe_loss, probe)[0]
    lpips_input_gradient_norm = float(probe_gradient.norm().detach().cpu())
    if not math.isfinite(lpips_input_gradient_norm) or lpips_input_gradient_norm <= 0:
        raise PQBFGANError(f"LPIPS_INPUT_GRADIENT_ZERO:{lpips_input_gradient_norm}")
    resume_stage_a = str(config["training"].get("resume_stage_a_checkpoint", "")).strip()
    if resume_stage_a:
        checkpoint_path = Path(resume_stage_a)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        payload = torch.load(checkpoint_path, map_location=device)
        if str(payload.get("stage")) != "A":
            raise PQBFGANError(f"RESUME_CHECKPOINT_NOT_STAGE_A:{checkpoint_path}")
        generator.load_state_dict(payload["ema"])
        stage_a_log = []
        stage_a_manifest = {
            "steps": int(payload.get("step", -1)),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hq.sha256_file(checkpoint_path),
            "reused_without_retraining": True,
        }
    else:
        stage_a_ema, stage_a_log, stage_a_manifest = train_stage_a(
            generator=generator,
            geometry=geometry,
            uncertainty=uncertainty,
            train_data=caches["train"],
            lpips_fn=lpips_fn,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        generator.load_state_dict(stage_a_ema.module.state_dict())
    content_model = copy.deepcopy(generator).to(device).eval()
    stage_b_ema, discriminator, stage_b_log, stage_b_manifest = train_stage_b(
        generator=generator,
        geometry=geometry,
        uncertainty=uncertainty,
        train_data=caches["train"],
        lpips_fn=lpips_fn,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    gan_model = stage_b_ema.module.to(device).eval()
    methods: dict[str, ProjectorGatedFiberGenerator | None] = {
        "box_fiber_lmmse": None,
        "pqbf_content": content_model,
        "pqbf_gan": gan_model,
    }
    for requested_step in config["eval"].get("stage_b_checkpoint_sweep", []):
        checkpoint_step = int(requested_step)
        checkpoint_path = output_dir / "checkpoints" / f"stage_b_step{checkpoint_step:06d}.pt"
        if not checkpoint_path.is_file():
            raise PQBFGANError(f"MISSING_STAGE_B_SWEEP_CHECKPOINT:{checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=device)
        if str(payload.get("stage")) != "B" or int(payload.get("step", -1)) != checkpoint_step:
            raise PQBFGANError(f"INVALID_STAGE_B_SWEEP_CHECKPOINT:{checkpoint_path}")
        checkpoint_model = ProjectorGatedFiberGenerator(
            geometry,
            steps=int(config["model"].get("steps", 3)),
            step_scale=float(config["model"].get("step_scale", 0.25)),
        ).to(device)
        checkpoint_model.load_state_dict(payload["ema"])
        methods[f"pqbf_gan_step{checkpoint_step:04d}"] = checkpoint_model.eval()
        del payload
    method_rows, _per_image, evaluation = evaluate_methods(
        methods=methods,
        data=caches["val"],
        geometry=geometry,
        rows64=rows64,
        uncertainty=uncertainty,
        lpips_fn=lpips_fn,
        config=config,
        device=device,
        output_dir=output_dir / "reports" / "validation",
    )
    metric_by_method = {str(row["method"]): row for row in method_rows}
    content_metrics = metric_by_method["pqbf_content"]
    eligible_checkpoints = []
    for name, metrics in metric_by_method.items():
        if not name.startswith("pqbf_gan_step"):
            continue
        psnr_delta = float(metrics["psnr_mean"]) - float(content_metrics["psnr_mean"])
        ssim_delta = float(metrics["ssim_mean"]) - float(content_metrics["ssim_mean"])
        if psnr_delta >= -0.10 and ssim_delta >= -0.002:
            eligible_checkpoints.append(
                {
                    "method": name,
                    "psnr_delta_vs_content": psnr_delta,
                    "ssim_delta_vs_content": ssim_delta,
                    "lpips": float(metrics["lpips_mean"]),
                }
            )
    selected_checkpoint = (
        min(eligible_checkpoints, key=lambda row: float(row["lpips"]))
        if eligible_checkpoints
        else None
    )
    game = summarize_game(stage_b_log)
    mode = str(config.get("mode", "smoke"))
    classification = "SMOKE_COMPLETED_DIRECTIONAL_ONLY" if mode != "formal" else "FORMAL_PENDING_TEST_GATE"
    summary = {
        "classification": classification,
        "method": "PQBF-GAN",
        "mode": mode,
        "config": str(config_path.resolve()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "not_applicable",
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "generator_parameters": parameter_count(generator),
        "discriminator_parameters": parameter_count(discriminator),
        "lpips_input_gradient_norm": lpips_input_gradient_norm,
        "operator": {**operator_manifest, **geometry.info_dict()},
        "split": split_manifest,
        "duplicate_audit": duplicate_audit,
        "cache": cache_diagnostics,
        "stage_a": stage_a_manifest,
        "stage_b": stage_b_manifest,
        "gan_game": game,
        "validation_metrics": method_rows,
        "validation_comparisons": evaluation["comparisons"],
        "stage_b_checkpoint_selection": {
            "constraints": {
                "psnr_delta_vs_content_min": -0.10,
                "ssim_delta_vs_content_min": -0.002,
            },
            "eligible": eligible_checkpoints,
            "selected": selected_checkpoint,
        },
        "test_opened": False,
        "runtime_seconds": time.time() - started,
        "scientific_scope": (
            "The GAN learns a deterministic prior-supported null-space correction. "
            "The unmeasured component is not claimed to be uniquely identified by the record."
        ),
    }
    hq.write_json(output_dir / "reports" / "run_summary.json", summary)
    hq.write_json(output_dir / "reports" / "gan_game_summary.json", game)
    hq.write_json(output_dir / "reports" / "operator_manifest.json", summary["operator"])
    hq.write_json(output_dir / "reports" / "split_manifest.json", split_manifest)
    hq.write_json(output_dir / "reports" / "cache_manifest.json", cache_diagnostics)
    print(yaml.safe_dump(hq.json_safe(summary), sort_keys=False), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the PQBF-GAN completion-project experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
