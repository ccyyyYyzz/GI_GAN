from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils as tv_utils

from src.dc_balanced import dc_row, dct_lowfreq_non_dc_rows, hadamard_lowsequency_non_dc_rows, random_zero_mean_rows
from src.losses import charbonnier_loss, frequency_loss, gradient_difference_loss
from src.measurement import GhostMeasurementOperator
from src.metrics import psnr as psnr_metric
from src.metrics import ssim as ssim_metric
from src.models import PatchDiscriminator, build_generator
from src.projections import get_exact_projector, relative_measurement_error
from src.phase2_witness import sha256_file


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "gan_high_quality_gi_smoke.yaml"


class GANHighQualityGIError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json_safe(row.get(k, "")) for k in keys})
    os.replace(tmp, path)


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise GANHighQualityGIError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256_numpy(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def build_structured_operator_rows(
    *,
    img_size: int,
    total_m: int,
    dct_rows: int,
    hadamard_rows: int,
    random_rows: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    dim = int(img_size) * int(img_size)
    counts = [int(dct_rows), int(hadamard_rows), int(random_rows)]
    if int(total_m) != 1 + sum(counts):
        raise GANHighQualityGIError(
            f"BAD_OPERATOR_BUDGET: total_m={total_m}, expected={1 + sum(counts)}"
        )
    rows = [dc_row(dim)[None, :]]
    if dct_rows:
        rows.append(dct_lowfreq_non_dc_rows(int(dct_rows), int(img_size)))
    if hadamard_rows:
        rows.append(hadamard_lowsequency_non_dc_rows(int(hadamard_rows), dim))
    if random_rows:
        rows.append(random_zero_mean_rows(int(random_rows), dim, int(seed)))
    A = np.concatenate(rows, axis=0).astype(np.float32)
    norms = np.linalg.norm(A, axis=1)
    means = A.mean(axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 1e-5:
        raise GANHighQualityGIError("OPERATOR_ROWS_NOT_UNIT_NORM")
    if float(np.max(np.abs(means[1:]))) > 1e-6:
        raise GANHighQualityGIError("NON_DC_ROWS_NOT_ZERO_MEAN")
    meta = {
        "img_size": int(img_size),
        "n": dim,
        "total_m": int(total_m),
        "dct_rows": int(dct_rows),
        "hadamard_rows": int(hadamard_rows),
        "random_rows": int(random_rows),
        "seed": int(seed),
        "rows_sha256": sha256_numpy(A),
        "dc_first_row": True,
        "row_norm_max_abs_error": float(np.max(np.abs(norms - 1.0))),
        "non_dc_mean_max_abs": float(np.max(np.abs(means[1:]))) if A.shape[0] > 1 else 0.0,
        "signed_exposure_note": "Rows after the DC row are zero-mean unit-norm; DC is counted in the fixed 5% budget.",
    }
    return A, meta


def make_measurement_operator(
    rows: np.ndarray,
    *,
    img_size: int,
    device: torch.device,
    lambda_solver: float,
) -> GhostMeasurementOperator:
    measurement = GhostMeasurementOperator(
        img_size=int(img_size),
        sampling_ratio=float(rows.shape[0] / rows.shape[1]),
        pattern_type="rademacher",
        noise_std=0.0,
        lambda_dc=float(lambda_solver),
        backprojection_mode="ridge_pinv",
        matrix_normalization="orthonormal_rows",
        device=device,
        seed=0,
    )
    measurement.set_A_override(
        torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(device),
        metadata={
            "operator_family": "dc_lowfreq_dct_lowsequency_hadamard_random_mix",
            "rows_sha256": sha256_numpy(rows),
        },
        rebuild_cache=True,
    )
    return measurement


class IndexedTensorDataset(Dataset):
    def __init__(self, base, indices: Sequence[int], transform) -> None:
        self.base = base
        self.indices = [int(i) for i in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def _raw_item(self, source_index: int):
        data = getattr(self.base, "data", None)
        if data is not None:
            return data[int(source_index)]
        return self.base[int(source_index)][0]

    def raw_hash(self, source_index: int) -> str:
        item = self._raw_item(int(source_index))
        if hasattr(item, "tobytes"):
            payload = item.tobytes()
        else:
            payload = repr(item).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __getitem__(self, idx: int):
        source_index = self.indices[int(idx)]
        img, label = self.base[source_index]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label), int(source_index)


def build_split_datasets(config: Mapping[str, Any]) -> tuple[IndexedTensorDataset, IndexedTensorDataset, IndexedTensorDataset, dict[str, Any]]:
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
    n = len(base)
    spans = {}
    used: set[int] = set()
    out = []
    for name in ["train", "val", "dev"]:
        off = int(data_cfg[f"{name}_offset"])
        count = int(data_cfg[f"{name}_count"])
        idx = list(range(off, min(off + count, n)))
        if len(idx) != count:
            raise GANHighQualityGIError(f"SPLIT_OUT_OF_RANGE:{name}:{off}:{count}:{n}")
        overlap = used.intersection(idx)
        if overlap:
            raise GANHighQualityGIError(f"SPLIT_OVERLAP:{name}:{len(overlap)}")
        used.update(idx)
        spans[name] = {"offset": off, "count": count, "min": min(idx), "max": max(idx)}
        out.append(IndexedTensorDataset(base, idx, transform))
    manifest = {
        "source_split": source_split,
        "dataset_name": "STL10",
        "dataset_root": root,
        "img_size": img_size,
        "spans": spans,
        "note": "Fresh development uses STL10 train+unlabeled only; no final-v4/test split is used for selection.",
    }
    return out[0], out[1], out[2], manifest


def build_loader(dataset: Dataset, *, batch_size: int, workers: int, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        drop_last=bool(shuffle),
        generator=generator,
    )


def tensor_dataset_to_matrix(dataset: Dataset, *, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    xs, labels, indices = [], [], []
    for x, label, idx in loader:
        xs.append(x.reshape(x.shape[0], -1).numpy().astype(np.float32))
        labels.append(label.numpy().astype(np.int64))
        indices.append(idx.numpy().astype(np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(labels, axis=0), np.concatenate(indices, axis=0)


@dataclass
class EmpiricalLMMSE:
    mu: np.ndarray
    z_scaled: np.ndarray
    u: np.ndarray
    inv_s: np.ndarray
    posterior_std: np.ndarray
    rows_sha256: str
    lambda_: float

    @classmethod
    def fit(cls, x_train: np.ndarray, rows: np.ndarray, *, lambda_: float) -> "EmpiricalLMMSE":
        x64 = np.asarray(x_train, dtype=np.float64)
        rows64 = np.asarray(rows, dtype=np.float64)
        mu = x64.mean(axis=0)
        denom = math.sqrt(max(1, x64.shape[0] - 1))
        z_scaled = (x64 - mu[None, :]) / denom
        u = z_scaled @ rows64.T
        s = u.T @ u + float(lambda_) * np.eye(rows64.shape[0], dtype=np.float64)
        inv_s = np.linalg.inv(s)
        diag_c = np.sum(z_scaled * z_scaled, axis=0)
        # diag(Z^T U (U^T U + lambda I)^-1 U^T Z) without forming the
        # N x N middle matrix.  This keeps empirical LMMSE usable for the
        # 20k+ image regimes needed by the GAN experiments.
        q = u.T @ z_scaled
        post_diag = diag_c - np.einsum("ij,ik,kj->j", q, inv_s, q, optimize=True)
        post_diag = np.maximum(post_diag, 0.0)
        posterior_std = np.sqrt(post_diag).astype(np.float32)
        return cls(
            mu=mu.astype(np.float32),
            z_scaled=z_scaled.astype(np.float32),
            u=u.astype(np.float32),
            inv_s=inv_s.astype(np.float32),
            posterior_std=posterior_std,
            rows_sha256=sha256_numpy(rows),
            lambda_=float(lambda_),
        )

    def predict_flat(self, y: torch.Tensor, rows: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        mu = torch.from_numpy(self.mu).to(device=device, dtype=y.dtype).reshape(1, -1)
        z = torch.from_numpy(self.z_scaled).to(device=device, dtype=y.dtype)
        u = torch.from_numpy(self.u).to(device=device, dtype=y.dtype)
        inv_s = torch.from_numpy(self.inv_s).to(device=device, dtype=y.dtype)
        y_mu = mu @ rows.T
        beta = (y - y_mu) @ inv_s.T
        coeff = beta @ u.T
        return mu + coeff @ z

    def anchor(self, y: torch.Tensor, measurement: GhostMeasurementOperator, *, device: torch.device) -> torch.Tensor:
        rows = measurement.A.to(device=device, dtype=y.dtype)
        pred = self.predict_flat(y, rows, device=device)
        projector = get_exact_projector(measurement, dtype=torch.float64, device=device)
        audited = projector.audit_flat(pred.to(torch.float64), y.to(torch.float64))
        return audited.to(dtype=y.dtype)

    def uncertainty_map(self, *, img_size: int, device: torch.device, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        arr = self.posterior_std.astype(np.float32)
        lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
        t = torch.from_numpy(arr).to(device=device, dtype=dtype).reshape(1, 1, img_size, img_size)
        return t.repeat(int(batch_size), 1, 1, 1)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        import copy

        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for key, value in self.module.state_dict().items():
            src = msd[key].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                value.copy_(src)


class DualPatchDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.global_d = PatchDiscriminator(in_channels=1)
        self.high_d = PatchDiscriminator(in_channels=1)

    @staticmethod
    def highpass(x: torch.Tensor) -> torch.Tensor:
        return x - F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_d(x).mean(dim=(1, 2, 3)) + self.high_d(self.highpass(x)).mean(dim=(1, 2, 3))


def zero_init_residual_head(model: nn.Module) -> None:
    """Make the residual model start exactly at the audited LMMSE anchor."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        return
    last = convs[-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


def nullspace_reconstruct(x0_flat: torch.Tensor, delta_flat: torch.Tensor, measurement: GhostMeasurementOperator) -> torch.Tensor:
    # Keep the linear algebra out of AMP.  CUDA autocast may downcast the
    # projector Gram solve to float16, which is both unsupported for pinv and
    # too fragile for the measurement-certificate path.
    with torch.cuda.amp.autocast(enabled=False):
        x0_f = x0_flat.double()
        delta_f = delta_flat.double()
        projector = get_exact_projector(measurement, dtype=torch.float64, device=x0_f.device)
        out = x0_f + projector.null_project_flat(delta_f)
    return out.to(dtype=x0_flat.dtype)


def generator_forward(
    generator: nn.Module,
    x0_flat: torch.Tensor,
    uncertainty: torch.Tensor,
    measurement: GhostMeasurementOperator,
) -> torch.Tensor:
    x0_img = measurement.unflatten_img(x0_flat)
    delta_img = generator(x0_img, uncertainty)
    delta_flat = measurement.flatten_img(delta_img)
    xhat_flat = nullspace_reconstruct(x0_flat, delta_flat, measurement)
    return measurement.unflatten_img(xhat_flat)


def r1_penalty(discriminator: nn.Module, real: torch.Tensor) -> torch.Tensor:
    real = real.detach().requires_grad_(True)
    score = discriminator(real).sum()
    grad = torch.autograd.grad(score, real, create_graph=True)[0]
    return grad.pow(2).reshape(real.shape[0], -1).sum(dim=1).mean()


def hinge_d_loss(real_score: torch.Tensor, fake_score: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()


def hinge_g_loss(fake_score: torch.Tensor) -> torch.Tensor:
    return -fake_score.mean()


def image_losses(xhat: torch.Tensor, x: torch.Tensor, cfg: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    l_charb = charbonnier_loss(xhat, x)
    l_l1 = F.l1_loss(xhat, x)
    l_grad = gradient_difference_loss(xhat, x)
    l_spec = frequency_loss(xhat, x)
    total = (
        float(cfg.get("lambda_charb", 20.0)) * l_charb
        + float(cfg.get("lambda_l1", 20.0)) * l_l1
        + float(cfg.get("lambda_grad", 1.0)) * l_grad
        + float(cfg.get("lambda_spec", 0.2)) * l_spec
    )
    return total, {
        "charb": float(l_charb.detach().cpu()),
        "l1": float(l_l1.detach().cpu()),
        "grad": float(l_grad.detach().cpu()),
        "spec": float(l_spec.detach().cpu()),
    }


def prep_lpips(x: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
    """Map grayscale images to LPIPS input space without silently cutting gradients.

    Evaluation callers already run under ``torch.no_grad()``.  Training callers
    need gradients through the prediction, so detaching is an explicit opt-in
    instead of an unconditional preprocessing side effect.
    """
    if detach:
        x = x.detach()
    x = x.clamp(0, 1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x * 2.0 - 1.0


def load_lpips(device: torch.device):
    try:
        import lpips  # type: ignore

        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        return {"error": repr(exc)}


@torch.no_grad()
def lpips_batch(loss_fn, pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray | None:
    if isinstance(loss_fn, dict):
        return None
    vals = loss_fn(prep_lpips(pred), prep_lpips(truth))
    return vals.reshape(vals.shape[0]).detach().cpu().numpy().astype(np.float64)


def rapsd_np(img: np.ndarray, bins: int = 32) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    f = np.fft.fftshift(np.fft.fft2(arr))
    power = np.abs(f) ** 2
    h, w = arr.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    edges = np.linspace(0.0, rr.max() + 1e-9, bins + 1)
    out = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        mask = (rr >= edges[i]) & (rr < edges[i + 1])
        out[i] = float(power[mask].mean()) if np.any(mask) else 0.0
    s = out.sum()
    return out / max(s, 1e-12)


def edge_sharpness(x: torch.Tensor) -> np.ndarray:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    val = dx.abs().mean(dim=(1, 2, 3)) + dy.abs().mean(dim=(1, 2, 3))
    return val.detach().cpu().numpy().astype(np.float64)


def centered_rmse_torch(xhat: torch.Tensor, x: torch.Tensor) -> np.ndarray:
    err = (xhat - x).detach().reshape(x.shape[0], -1).float()
    err = err - err.mean(dim=1, keepdim=True)
    return torch.sqrt(torch.mean(err * err, dim=1)).cpu().numpy().astype(np.float64)


def full_rmse_torch(xhat: torch.Tensor, x: torch.Tensor) -> np.ndarray:
    err = (xhat - x).detach().reshape(x.shape[0], -1).float()
    return torch.sqrt(torch.mean(err * err, dim=1)).cpu().numpy().astype(np.float64)


def fid_from_features(real: np.ndarray, fake: np.ndarray) -> float:
    from scipy.linalg import sqrtm

    mu1, mu2 = real.mean(axis=0), fake.mean(axis=0)
    c1 = np.cov(real, rowvar=False)
    c2 = np.cov(fake, rowvar=False)
    covmean = sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(c1 + c2 - 2.0 * covmean))


def kid_from_features(real: np.ndarray, fake: np.ndarray) -> float:
    gamma = 1.0 / real.shape[1]
    k_xx = (gamma * real @ real.T + 1.0) ** 3
    k_yy = (gamma * fake @ fake.T + 1.0) ** 3
    k_xy = (gamma * real @ fake.T + 1.0) ** 3
    n = real.shape[0]
    if n < 2:
        return float("nan")
    return float((k_xx.sum() - np.trace(k_xx)) / (n * (n - 1)) + (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1)) - 2.0 * k_xy.mean())


@torch.no_grad()
def inception_features(images: torch.Tensor, *, device: torch.device, max_images: int) -> np.ndarray | None:
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except Exception:
        return None
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True).to(device).eval()
    model.fc = nn.Identity()
    xs = images[: int(max_images)].detach().clamp(0, 1)
    if xs.shape[1] == 1:
        xs = xs.repeat(1, 3, 1, 1)
    feats = []
    for start in range(0, xs.shape[0], 16):
        xb = F.interpolate(xs[start : start + 16].to(device), size=(299, 299), mode="bilinear", align_corners=False)
        out = model(xb)
        feats.append(out.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def evaluate_methods(
    *,
    methods: Mapping[str, nn.Module | None],
    lmmse: EmpiricalLMMSE,
    measurement: GhostMeasurementOperator,
    loader: DataLoader,
    device: torch.device,
    config: Mapping[str, Any],
    output_dir: Path,
    epoch_tag: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    eval_cfg = dict(config["eval"])
    img_size = int(config["data"].get("img_size", 64))
    rows: list[dict[str, Any]] = []
    per: list[dict[str, Any]] = []
    all_truth: list[torch.Tensor] = []
    all_pred_by_method: dict[str, list[torch.Tensor]] = {k: [] for k in methods}
    all_pred_by_method["lmmse_anchor"] = []
    lpips_fn = load_lpips(device) if bool(eval_cfg.get("lpips", False)) else {"error": "disabled"}
    source_counter = 0
    for x, label, source_idx in loader:
        x = x.to(device, non_blocking=True)
        flat = measurement.flatten_img(x)
        y = measurement.A_forward(flat)
        x0_flat = lmmse.anchor(y, measurement, device=device)
        x0 = measurement.unflatten_img(x0_flat)
        preds: dict[str, torch.Tensor] = {"lmmse_anchor": x0}
        for name, model in methods.items():
            if model is None:
                continue
            model.eval()
            uncertainty = lmmse.uncertainty_map(
                img_size=img_size, device=device, batch_size=x.shape[0], dtype=x.dtype
            )
            preds[name] = generator_forward(model, x0_flat, uncertainty, measurement)
        all_truth.append(x.detach().cpu())
        for name, pred in preds.items():
            all_pred_by_method.setdefault(name, []).append(pred.detach().cpu())
            clipped = pred.clamp(0, 1)
            lp_vals = lpips_batch(lpips_fn, clipped, x)
            rel = relative_measurement_error(pred, y, measurement).detach().cpu().numpy().astype(np.float64)
            rmse = full_rmse_torch(clipped, x)
            crmse = centered_rmse_torch(clipped, x)
            sharp = edge_sharpness(clipped)
            pred_np = clipped.detach().cpu().numpy()[:, 0]
            truth_np = x.detach().cpu().numpy()[:, 0]
            rapsd = np.asarray([np.linalg.norm(rapsd_np(pred_np[i]) - rapsd_np(truth_np[i])) for i in range(x.shape[0])], dtype=np.float64)
            for i in range(x.shape[0]):
                per.append(
                    {
                        "epoch_tag": epoch_tag,
                        "method": name,
                        "sample_ordinal": int(source_counter + i),
                        "source_index": int(source_idx[i]),
                        "label": int(label[i]),
                        "full_rmse": float(rmse[i]),
                        "centered_rmse": float(crmse[i]),
                        "psnr": float(-20.0 * math.log10(max(float(rmse[i]), 1e-12))),
                        "ssim": float(ssim_metric(clipped[i : i + 1], x[i : i + 1])),
                        "lpips": "[DATA MISSING]" if lp_vals is None else float(lp_vals[i]),
                        "rapsd": float(rapsd[i]),
                        "edge_sharpness": float(sharp[i]),
                        "relmeaserr": float(rel[i]),
                    }
                )
        source_counter += x.shape[0]
    truth_cat = torch.cat(all_truth, dim=0)
    fid_kid: dict[str, dict[str, Any]] = {}
    if bool(eval_cfg.get("kid_fid", False)):
        real_feat = inception_features(truth_cat, device=device, max_images=int(eval_cfg.get("kid_fid_max_images", 96)))
    else:
        real_feat = None
    for name, chunks in all_pred_by_method.items():
        if not chunks:
            continue
        vals = [r for r in per if r["method"] == name]
        row = {"epoch_tag": epoch_tag, "method": name, "n": len(vals)}
        for metric in ["full_rmse", "centered_rmse", "psnr", "ssim", "rapsd", "edge_sharpness", "relmeaserr"]:
            arr = np.asarray([float(v[metric]) for v in vals], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(arr))
        lp = []
        for v in vals:
            try:
                lp.append(float(v["lpips"]))
            except (TypeError, ValueError):
                pass
        row["lpips_mean"] = float(np.mean(lp)) if lp else "[DATA MISSING]"
        pred_cat = torch.cat(chunks, dim=0).clamp(0, 1)
        if real_feat is not None:
            fake_feat = inception_features(pred_cat, device=device, max_images=int(eval_cfg.get("kid_fid_max_images", 96)))
            if fake_feat is not None:
                fid_kid[name] = {"fid": fid_from_features(real_feat, fake_feat), "kid": kid_from_features(real_feat, fake_feat)}
                row["fid"] = fid_kid[name]["fid"]
                row["kid"] = fid_kid[name]["kid"]
            else:
                row["fid"] = "[DATA MISSING]"
                row["kid"] = "[DATA MISSING]"
        else:
            row["fid"] = "[DATA MISSING]"
            row["kid"] = "[DATA MISSING]"
        rows.append(row)
    save_qualitative_grid(
        output_dir / "figures" / f"qualitative_{epoch_tag}.png",
        truth_cat,
        {name: torch.cat(chunks, dim=0) for name, chunks in all_pred_by_method.items() if chunks},
        max_items=int(eval_cfg.get("qualitative_count", 12)),
    )
    return rows, per, {"lpips_status": "PASS" if not isinstance(lpips_fn, dict) else lpips_fn, "fid_kid": fid_kid}


def save_qualitative_grid(path: Path, truth: torch.Tensor, preds: Mapping[str, torch.Tensor], *, max_items: int) -> None:
    ensure_dir(path.parent)
    names = ["truth"] + list(preds.keys())
    rows = [truth[:max_items].clamp(0, 1)]
    for name in preds:
        rows.append(preds[name][:max_items].clamp(0, 1))
    grid = torch.cat(rows, dim=0)
    tv_utils.save_image(grid, path, nrow=max_items, padding=2)
    write_text(path.with_suffix(".txt"), "Rows: " + ", ".join(names) + "\n")


def train_one_variant(
    *,
    variant: str,
    train_seed: int,
    config: Mapping[str, Any],
    lmmse: EmpiricalLMMSE,
    measurement: GhostMeasurementOperator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(int(config["seed"]) + 1000 * int(train_seed) + (17 if variant == "gan" else 0))
    model_cfg = dict(config["model"])
    model = build_generator({"model_type": model_cfg.get("model_type", "hq_unet"), "base_channels": int(model_cfg.get("base_channels", 32))}, measurement=measurement).to(device)
    zero_init_residual_head(model)
    disc = DualPatchDiscriminator().to(device) if variant == "gan" else None
    ema = ModelEMA(model, decay=float(model_cfg.get("ema_decay", 0.995)))
    train_cfg = dict(config["training"])
    opt_g = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr_g", 2e-4)), betas=tuple(train_cfg.get("betas", [0.5, 0.9])))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(train_cfg.get("lr_d", 2e-4)), betas=tuple(train_cfg.get("betas", [0.5, 0.9]))) if disc is not None else None
    scaler_g = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    scaler_d = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    total_epochs = int(train_cfg.get("epochs_pretrain", 1)) + (int(train_cfg.get("epochs_gan", 0)) if variant == "gan" else 0)
    log_rows: list[dict[str, Any]] = []
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    for epoch in range(total_epochs):
        gan_active = variant == "gan" and epoch >= int(train_cfg.get("epochs_pretrain", 1))
        model.train()
        if disc is not None:
            disc.train()
        losses: list[dict[str, float]] = []
        for x, _label, _idx in train_loader:
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
            if disc is not None and gan_active:
                for p in disc.parameters():
                    p.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler_d.is_enabled()):
                    fake = generator_forward(model, x0_flat, uncertainty, measurement).detach().clamp(0, 1)
                    real_score = disc(x)
                    fake_score = disc(fake)
                    d_loss = hinge_d_loss(real_score, fake_score)
                    if float(train_cfg.get("r1_gamma", 0.0)) > 0:
                        d_loss = d_loss + 0.5 * float(train_cfg.get("r1_gamma", 0.0)) * r1_penalty(disc, x)
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                d_loss = torch.zeros((), device=device)

            if disc is not None:
                for p in disc.parameters():
                    p.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler_g.is_enabled()):
                xhat = generator_forward(model, x0_flat, uncertainty, measurement)
                base_loss, parts = image_losses(xhat.clamp(0, 1), x, train_cfg)
                adv = torch.zeros((), device=device)
                if disc is not None and gan_active:
                    adv = hinge_g_loss(disc(xhat.clamp(0, 1)))
                g_loss = base_loss + float(train_cfg.get("lambda_adv", 0.01)) * adv
            scaler_g.scale(g_loss).backward()
            if float(train_cfg.get("grad_clip", 0.0)) > 0:
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            scaler_g.step(opt_g)
            scaler_g.update()
            ema.update(model)
            parts.update({"g_loss": float(g_loss.detach().cpu()), "d_loss": float(d_loss.detach().cpu()), "adv": float(adv.detach().cpu())})
            losses.append(parts)
        row = {"variant": variant, "train_seed": int(train_seed), "epoch": epoch + 1, "gan_active": bool(gan_active)}
        for key in sorted({k for item in losses for k in item}):
            row[key] = float(np.mean([item.get(key, 0.0) for item in losses]))
        log_rows.append(row)
        if (epoch + 1) % int(train_cfg.get("save_every_epochs", 1)) == 0:
            save_checkpoint(ckpt_dir / f"{variant}_seed{train_seed}_epoch{epoch+1:03d}.pt", model, ema, disc, config, row)
    write_csv(output_dir / "train_log.csv", log_rows)
    save_checkpoint(ckpt_dir / f"{variant}_seed{train_seed}_final.pt", model, ema, disc, config, {"final": True})
    return ema.module, {"train_log": log_rows, "checkpoint": str(ckpt_dir / f"{variant}_seed{train_seed}_final.pt")}


def save_checkpoint(path: Path, model: nn.Module, ema: ModelEMA, disc: nn.Module | None, config: Mapping[str, Any], meta: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    payload = {
        "generator": model.state_dict(),
        "generator_ema": ema.module.state_dict(),
        "discriminator": None if disc is None else disc.state_dict(),
        "config": json_safe(config),
        "meta": json_safe(meta),
        "saved_utc": now_utc(),
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def paired_bootstrap_delta(per_rows: Sequence[Mapping[str, Any]], method: str, reference: str, metric: str, *, reps: int, seed: int) -> dict[str, Any]:
    by: dict[tuple[int, int], dict[str, float]] = {}
    for r in per_rows:
        if r["method"] not in {method, reference}:
            continue
        try:
            val = float(r[metric])
        except (TypeError, ValueError):
            continue
        train_seed = int(r.get("train_seed", -1))
        by.setdefault((train_seed, int(r["sample_ordinal"])), {})[str(r["method"])] = val
    pairs = [(v[method], v[reference]) for v in by.values() if method in v and reference in v]
    if not pairs:
        return {"method": method, "reference": reference, "metric": metric, "status": "NO_PAIRS"}
    arr = np.asarray(pairs, dtype=np.float64)
    delta = arr[:, 0] - arr[:, 1]
    rng = np.random.default_rng(int(seed))
    boots = []
    for _ in range(int(reps)):
        idx = rng.integers(0, len(delta), size=len(delta))
        boots.append(float(delta[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "method": method,
        "reference": reference,
        "metric": metric,
        "status": "PASS",
        "n": len(delta),
        "mean_delta": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "wins_method_lower": int(np.sum(delta < 0)),
        "wins_reference_lower": int(np.sum(delta > 0)),
    }


def summarize_gate(all_per: Sequence[Mapping[str, Any]], all_method: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    eval_cfg = dict(config["eval"])
    reps = int(eval_cfg.get("bootstrap_replicates", 500))
    seed = int(eval_cfg.get("bootstrap_seed", 20260626))
    comparisons = []
    for metric in ["lpips", "rapsd", "full_rmse", "centered_rmse", "psnr"]:
        comparisons.append(paired_bootstrap_delta(all_per, "gan", "no_gan", metric, reps=reps, seed=seed + len(comparisons)))
    by_seed: dict[int, dict[str, float]] = {}
    for r in all_method:
        if r["method"] in {"gan", "no_gan"}:
            try:
                by_seed.setdefault(int(r["train_seed"]), {})[str(r["method"])] = float(r["lpips_mean"])
            except (TypeError, ValueError):
                pass
    seed_lpips_better = [v["gan"] < v["no_gan"] for v in by_seed.values() if "gan" in v and "no_gan" in v]
    cmp_lpips = next(c for c in comparisons if c["metric"] == "lpips")
    cmp_rapsd = next(c for c in comparisons if c["metric"] == "rapsd")
    cmp_psnr = next(c for c in comparisons if c["metric"] == "psnr")
    rel_vals = [float(r["relmeaserr"]) for r in all_per if r["method"] in {"gan", "no_gan"}]
    lpips_gain = None
    if cmp_lpips["status"] == "PASS":
        ref = np.mean([float(r["lpips"]) for r in all_per if r["method"] == "no_gan" and not isinstance(r["lpips"], str)])
        lpips_gain = -float(cmp_lpips["mean_delta"]) / max(float(ref), 1e-12)
    conditions = {
        "lpips_gain_ge_5pct_and_ci": bool(
            cmp_lpips["status"] == "PASS"
            and cmp_lpips["mean_delta"] < 0
            and cmp_lpips["ci_high"] < 0
            and lpips_gain is not None
            and lpips_gain >= float(eval_cfg.get("lpips_relative_gain_gate", 0.05))
        ),
        "rapsd_improves_ci_or_direction": bool(cmp_rapsd["status"] == "PASS" and cmp_rapsd["mean_delta"] < 0),
        "psnr_drop_within_tolerance": bool(
            cmp_psnr["status"] == "PASS"
            and cmp_psnr["mean_delta"] >= -float(eval_cfg.get("psnr_drop_tolerance_db", 0.5))
        ),
        "relmeaserr_ok": bool(rel_vals and max(rel_vals) <= float(eval_cfg.get("relmeaserr_limit", 1e-5))),
        "two_of_three_seeds_lpips_same_direction": bool(sum(seed_lpips_better) >= 2 and len(seed_lpips_better) >= 3),
    }
    if all(conditions.values()):
        classification = "GAN_PERCEPTUAL_GAIN_CONFIRMED"
    elif conditions["lpips_gain_ge_5pct_and_ci"] and not conditions["psnr_drop_within_tolerance"]:
        classification = "GAN_GAIN_WITH_DISTORTION_TRADEOFF"
    elif not conditions["relmeaserr_ok"]:
        classification = "INVALID_EXPERIMENT"
    else:
        classification = "GAN_NOT_YET_EFFECTIVE"
    return {
        "classification": classification,
        "conditions": conditions,
        "comparisons": comparisons,
        "lpips_relative_gain": lpips_gain,
        "seed_lpips_better": seed_lpips_better,
        "locked_test_authorized": classification in {"GAN_PERCEPTUAL_GAIN_CONFIRMED", "GAN_GAIN_WITH_DISTORTION_TRADEOFF"},
    }


def write_math_and_ledger(reports: Path, gate: Mapping[str, Any]) -> None:
    math_text = r"""# LMMSE-Anchored Null-Space GAN for Ghost Imaging

The measurement model is \(y=Ax\).  For \(P_0^A=I-A^\dagger A\), the estimator is
\[
\hat x=x_0+P_0^A G_\theta(x_0,U_A,z).
\]
The anchor \(x_L=\mu+C A^\top(A C A^\top+\lambda I)^{-1}(y-A\mu)\) is implemented with the centered training matrix \(Z\), never a dense \(C\):
\[
C A^\top=Z^\top(ZA^\top)/(N-1),\quad
A C A^\top=(ZA^\top)^\top(ZA^\top)/(N-1).
\]
The audited anchor is \(x_0=x_L-A^\dagger(Ax_L-y)\), so \(Ax_0=y\).  Because \(AP_0^A=0\),
\[
A\hat x=A x_0 + A P_0^A G_\theta(\cdot)=y.
\]
Plain language: the LMMSE anchor carries the measurement-supported structure, and the GAN can only add content in directions invisible to the bucket rows.
"""
    write_text(reports / "math_and_system_description.md", math_text)
    ledger = [
        "# Claim-Evidence Ledger",
        "",
        "| Claim | Evidence | Status |",
        "|---|---|---|",
        "| Fixed 5% structured operator is used for every method | `operator_manifest.json` | required control |",
        "| LMMSE is matrix-free | `math_and_system_description.md`, `lmmse_manifest.json` | supported |",
        "| Final outputs preserve measurement consistency | per-image `relmeaserr`, `gate_report.json` | see gate |",
        "| GAN is the performance source | paired `gan` vs `no_gan` comparisons with same architecture | see gate |",
        "| Locked test may be opened | gate classification | `" + str(gate.get("classification")) + "` |",
    ]
    write_text(reports / "claim_evidence_ledger.md", "\n".join(ledger) + "\n")


def save_split_hash_audit(path: Path, datasets_by_name: Mapping[str, IndexedTensorDataset]) -> dict[str, Any]:
    rows = []
    seen_raw: dict[str, str] = {}
    seen_trans: dict[str, str] = {}
    duplicate_raw = []
    duplicate_trans = []
    for split, ds in datasets_by_name.items():
        for local_i, source_index in enumerate(ds.indices):
            x, _label, _idx = ds[local_i]
            raw_h = ds.raw_hash(source_index)
            trans_h = hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
            uid = f"{split}:{source_index}"
            if raw_h in seen_raw:
                duplicate_raw.append([uid, seen_raw[raw_h]])
            if trans_h in seen_trans:
                duplicate_trans.append([uid, seen_trans[trans_h]])
            seen_raw[raw_h] = uid
            seen_trans[trans_h] = uid
            rows.append({"split": split, "source_index": source_index, "raw_sha256": raw_h, "transformed_sha256": trans_h})
    write_csv(path, rows)
    return {"raw_duplicates": duplicate_raw, "transformed_duplicates": duplicate_trans, "rows": len(rows)}


def run(config_path: Path, *, variants_override: list[str] | None = None, seeds_override: list[int] | None = None) -> dict[str, Any]:
    started = time.time()
    config = load_yaml(config_path)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    set_seed(int(config.get("seed", 20260625)))
    out = ensure_dir(ROOT / str(config["output_dir"]))
    reports = ensure_dir(out / "reports")
    shutil.copyfile(config_path, out / "config_used.yaml")

    train_ds, val_ds, dev_ds, split_manifest = build_split_datasets(config)
    split_audit = save_split_hash_audit(reports / "sample_hash_audit.csv", {"train": train_ds, "val": val_ds, "dev": dev_ds})
    train_x, _train_labels, _train_indices = tensor_dataset_to_matrix(train_ds)
    rows, op_meta = build_structured_operator_rows(
        img_size=int(config["data"]["img_size"]),
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    measurement = make_measurement_operator(
        rows,
        img_size=int(config["data"]["img_size"]),
        device=device,
        lambda_solver=float(config["operator"].get("lambda_solver", 1e-8)),
    )
    lmmse = EmpiricalLMMSE.fit(train_x, rows, lambda_=float(config["operator"].get("lmmse_lambda", 1e-4)))
    train_loader = build_loader(
        train_ds,
        batch_size=int(config["data"]["batch_size"]),
        workers=int(config["data"].get("num_workers", 0)),
        shuffle=True,
        seed=int(config["seed"]),
        device=device,
    )
    dev_loader = build_loader(
        dev_ds,
        batch_size=int(config["data"]["batch_size"]),
        workers=int(config["data"].get("num_workers", 0)),
        shuffle=False,
        seed=int(config["seed"]) + 1,
        device=device,
    )
    all_method: list[dict[str, Any]] = []
    all_per: list[dict[str, Any]] = []
    train_manifests: list[dict[str, Any]] = []
    variants = variants_override or list(config["training"].get("variants", ["no_gan", "gan"]))
    train_seeds = seeds_override or [int(v) for v in config["training"].get("train_seeds", [0])]
    for ts in train_seeds:
        for variant in variants:
            run_dir = ensure_dir(out / "runs" / f"{variant}_seed{ts}")
            model, manifest = train_one_variant(
                variant=str(variant),
                train_seed=int(ts),
                config=config,
                lmmse=lmmse,
                measurement=measurement,
                train_loader=train_loader,
                val_loader=dev_loader,
                device=device,
                output_dir=run_dir,
            )
            method_rows, per_rows, eval_diag = evaluate_methods(
                methods={str(variant): model},
                lmmse=lmmse,
                measurement=measurement,
                loader=dev_loader,
                device=device,
                config=config,
                output_dir=run_dir,
                epoch_tag=f"{variant}_seed{ts}_final",
            )
            for r in method_rows:
                r["train_seed"] = int(ts)
            for r in per_rows:
                r["train_seed"] = int(ts)
            all_method.extend(method_rows)
            all_per.extend(per_rows)
            manifest.update({"variant": str(variant), "train_seed": int(ts), "eval_diag": eval_diag})
            train_manifests.append(manifest)
    write_csv(reports / "method_metrics.csv", all_method)
    write_csv(reports / "per_image_metrics.csv", all_per)
    gate = summarize_gate(all_per, all_method, config)
    write_json(reports / "gate_report.json", gate)
    write_json(reports / "operator_manifest.json", op_meta)
    write_json(reports / "split_manifest.json", split_manifest)
    write_json(reports / "duplicate_audit.json", split_audit)
    write_json(reports / "lmmse_manifest.json", {"lambda": lmmse.lambda_, "rows_sha256": lmmse.rows_sha256, "train_count": int(train_x.shape[0])})
    write_json(reports / "training_manifest.json", train_manifests)
    write_math_and_ledger(reports, gate)
    runtime = {
        "status": "PASS",
        "elapsed_seconds": float(time.time() - started),
        "device": str(device),
        "started_utc": now_utc(),
        "config_sha256": sha256_file(out / "config_used.yaml"),
        "method_metrics_sha256": sha256_file(reports / "method_metrics.csv"),
        "per_image_metrics_sha256": sha256_file(reports / "per_image_metrics.csv"),
        "gate_report_sha256": sha256_file(reports / "gate_report.json"),
    }
    write_json(reports / "runtime_and_hashes.json", runtime)
    summary = {
        "status": "GAN_HIGH_QUALITY_GI_CANARY_COMPLETE",
        "output_dir": str(out),
        "classification": gate["classification"],
        "locked_test_authorized": gate["locked_test_authorized"],
        "key_artifacts": {
            "method_metrics": str(reports / "method_metrics.csv"),
            "per_image_metrics": str(reports / "per_image_metrics.csv"),
            "gate_report": str(reports / "gate_report.json"),
            "claim_evidence_ledger": str(reports / "claim_evidence_ledger.md"),
            "math_and_system_description": str(reports / "math_and_system_description.md"),
            "operator_manifest": str(reports / "operator_manifest.json"),
            "duplicate_audit": str(reports / "duplicate_audit.json"),
        },
        "runtime": runtime,
    }
    write_json(reports / "summary.json", summary)
    write_json(out / "GAN_HIGH_QUALITY_GI_CANARY_COMPLETE.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate LMMSE-anchored null-space GAN for ghost imaging.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--variants", default="", help="Comma-separated subset, e.g. no_gan or gan.")
    parser.add_argument("--train-seeds", default="", help="Comma-separated seed subset, e.g. 0,1.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()] or None
    seeds = [int(v.strip()) for v in args.train_seeds.split(",") if v.strip()] or None
    summary = run(Path(args.config), variants_override=variants, seeds_override=seeds)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
