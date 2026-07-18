from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

import gan_high_quality_gi as hq
from src.metrics import ssim as ssim_metric


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = Path(
    "E:/GAN_FCC_WORK/experiments/completion_gan_round18/"
    "pqbf_pilot_seed0/cache"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round23/"
    "unpaired_optical_calibration_micro"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def gaussian_kernel(sigma: torch.Tensor, size: int = 11) -> torch.Tensor:
    coord = torch.arange(size, device=sigma.device, dtype=sigma.dtype)
    coord = coord - float(size // 2)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma.square()))
    kernel = kernel / kernel.sum()
    return kernel.reshape(1, 1, size, size)


def blur_images(images: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    kernel = gaussian_kernel(sigma)
    return F.conv2d(images, kernel, padding=kernel.shape[-1] // 2)


def measure(
    images: torch.Tensor,
    rows: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    # Blurring the projected patterns is adjoint-equivalent to applying the
    # same symmetric optical transfer to the object before nominal buckets.
    effective = blur_images(images, sigma)
    return effective.flatten(1) @ rows.T


def normalized_record(record: torch.Tensor) -> torch.Tensor:
    # Remove unknown global flux while retaining the pattern-dependent optical
    # signature.  The first row is the positive DC exposure.
    dc = record[:, :1].abs().clamp_min(1.0e-4)
    return record[:, 1:] / dc


class BucketCritic(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(dim, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(128, 1)),
        )

    def forward(self, record: torch.Tensor) -> torch.Tensor:
        return self.net(record).flatten()


class DefocusGenerator(nn.Module):
    def __init__(self, initial_sigma: float) -> None:
        super().__init__()
        floor = 0.05
        self.floor = floor
        self.raw_sigma = nn.Parameter(
            torch.tensor(inv_softplus(initial_sigma - floor), dtype=torch.float32)
        )

    def sigma(self) -> torch.Tensor:
        return self.floor + F.softplus(self.raw_sigma)


def sample_batch(pool: torch.Tensor, batch_size: int) -> torch.Tensor:
    index = torch.randint(0, pool.shape[0], (batch_size,), device=pool.device)
    return pool[index]


def fit_adversarial_sigma(
    real_images: torch.Tensor,
    reference_images: torch.Tensor,
    rows: torch.Tensor,
    true_sigma: float,
    *,
    steps: int,
    batch_size: int,
) -> tuple[float, list[dict[str, float]]]:
    with torch.no_grad():
        real_records = normalized_record(
            measure(
                real_images,
                rows,
                torch.tensor(true_sigma, device=rows.device),
            )
        )
        center = real_records.mean(0)
        scale = real_records.std(0).clamp_min(1.0e-3)

    generator = DefocusGenerator(initial_sigma=0.30).to(rows.device)
    critic = BucketCritic(real_records.shape[1]).to(rows.device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=2.0e-3, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=2.0e-4, betas=(0.5, 0.9))
    log: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        for _ in range(3):
            real = sample_batch(real_records, batch_size)
            clean = sample_batch(reference_images, batch_size)
            with torch.no_grad():
                fake = normalized_record(measure(clean, rows, generator.sigma()))
            real_z = (real - center) / scale
            fake_z = (fake - center) / scale
            loss_d = F.softplus(-critic(real_z)).mean() + F.softplus(
                critic(fake_z)
            ).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

        clean = sample_batch(reference_images, batch_size)
        fake = normalized_record(measure(clean, rows, generator.sigma()))
        fake_z = (fake - center) / scale
        loss_g = F.softplus(-critic(fake_z)).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if step == 1 or step % 25 == 0 or step == steps:
            with torch.no_grad():
                real_probe = (sample_batch(real_records, batch_size) - center) / scale
                fake_probe = normalized_record(
                    measure(sample_batch(reference_images, batch_size), rows, generator.sigma())
                )
                fake_probe = (fake_probe - center) / scale
                gap = critic(real_probe).mean() - critic(fake_probe).mean()
            row = {
                "step": float(step),
                "sigma": float(generator.sigma().detach().cpu()),
                "d_loss": float(loss_d.detach().cpu()),
                "g_loss": float(loss_g.detach().cpu()),
                "critic_gap": float(gap.detach().cpu()),
            }
            log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return float(generator.sigma().detach().cpu()), log


def fit_moment_sigma(
    real_images: torch.Tensor,
    reference_images: torch.Tensor,
    rows: torch.Tensor,
    true_sigma: float,
    *,
    steps: int,
    batch_size: int,
) -> tuple[float, list[dict[str, float]]]:
    with torch.no_grad():
        real_records = normalized_record(
            measure(
                real_images,
                rows,
                torch.tensor(true_sigma, device=rows.device),
            )
        )
        target_mean = real_records.mean(0)
        target_std = real_records.std(0).clamp_min(1.0e-4)

    generator = DefocusGenerator(initial_sigma=0.30).to(rows.device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=2.0e-3)
    log: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        fake = normalized_record(
            measure(sample_batch(reference_images, batch_size), rows, generator.sigma())
        )
        mean_loss = ((fake.mean(0) - target_mean) / target_std).square().mean()
        std_loss = ((fake.std(0) - target_std) / target_std).square().mean()
        loss = mean_loss + std_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            row = {
                "step": float(step),
                "sigma": float(generator.sigma().detach().cpu()),
                "moment_loss": float(loss.detach().cpu()),
            }
            log.append(row)
    return float(generator.sigma().detach().cpu()), log


@torch.no_grad()
def ridge_reconstruct(
    train_images: torch.Tensor,
    validation_images: torch.Tensor,
    rows: torch.Tensor,
    assumed_sigma: float,
    true_sigma: float,
    ridge: float = 1.0e-3,
) -> torch.Tensor:
    sigma_assumed = torch.tensor(assumed_sigma, device=rows.device)
    sigma_true = torch.tensor(true_sigma, device=rows.device)
    y_train = measure(train_images, rows, sigma_assumed)
    y_val = measure(validation_images, rows, sigma_true)
    x_train = train_images.flatten(1)
    mean_x = x_train.mean(0, keepdim=True)
    mean_y = y_train.mean(0, keepdim=True)
    xc = x_train - mean_x
    yc = y_train - mean_y
    gram = yc.T @ yc / float(yc.shape[0])
    cross = yc.T @ xc / float(yc.shape[0])
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    weight = torch.linalg.solve(gram + ridge * eye, cross)
    prediction = mean_x + (y_val - mean_y) @ weight
    return prediction.reshape_as(validation_images).clamp(0.0, 1.0)


@torch.no_grad()
def evaluate(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    lpips_model: nn.Module,
) -> dict[str, float]:
    mse = (prediction - truth).square().flatten(1).mean(1)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1.0e-12))
    ssim = ssim_metric(prediction, truth)
    pred_rgb = prediction.repeat(1, 3, 1, 1) * 2.0 - 1.0
    truth_rgb = truth.repeat(1, 3, 1, 1) * 2.0 - 1.0
    lpips = lpips_model(pred_rgb, truth_rgb).flatten()
    return {
        "psnr": float(psnr.mean().cpu()),
        "ssim": float(ssim.mean().cpu()) if torch.is_tensor(ssim) else float(ssim),
        "lpips": float(lpips.mean().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--true-sigma", type=float, default=1.30)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device("cuda")
    set_seed(args.seed)
    started = time.time()

    train_payload = torch.load(args.cache_dir / "train.pt", map_location="cpu")
    val_payload = torch.load(args.cache_dir / "val.pt", map_location="cpu")
    train = train_payload["tensors"]["truth"].float().to(device)
    val = val_payload["tensors"]["truth"].float().to(device)
    rows_np, manifest = hq.build_structured_operator_rows(
        img_size=64,
        total_m=205,
        dct_rows=128,
        hadamard_rows=56,
        random_rows=20,
        seed=772001,
    )
    rows = torch.from_numpy(rows_np).float().to(device)
    real_images = train[: train.shape[0] // 2]
    reference_images = train[train.shape[0] // 2 :]

    adversarial_sigma, adversarial_log = fit_adversarial_sigma(
        real_images,
        reference_images,
        rows,
        args.true_sigma,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    moment_sigma, moment_log = fit_moment_sigma(
        real_images,
        reference_images,
        rows,
        args.true_sigma,
        steps=args.steps,
        batch_size=args.batch_size,
    )

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    methods = {
        "nominal_uncalibrated": 0.05,
        "moment_calibrated": moment_sigma,
        "adversarial_calibrated": adversarial_sigma,
        "oracle_calibrated": args.true_sigma,
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, sigma in methods.items():
        prediction = ridge_reconstruct(
            train,
            val,
            rows,
            assumed_sigma=sigma,
            true_sigma=args.true_sigma,
        )
        metrics[name] = evaluate(prediction, val, lpips_model)
        metrics[name]["assumed_sigma"] = float(sigma)
        print(name, json.dumps(metrics[name], sort_keys=True), flush=True)

    payload = {
        "status": "MICRO_PILOT",
        "validation_only": True,
        "test_split_opened": False,
        "true_sigma": float(args.true_sigma),
        "adversarial_sigma": float(adversarial_sigma),
        "moment_sigma": float(moment_sigma),
        "operator_sha256": manifest["rows_sha256"],
        "train_real_count": int(real_images.shape[0]),
        "train_unpaired_reference_count": int(reference_images.shape[0]),
        "validation_count": int(val.shape[0]),
        "metrics": metrics,
        "adversarial_log": adversarial_log,
        "moment_log": moment_log,
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
