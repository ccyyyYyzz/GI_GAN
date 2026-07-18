from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import gan_high_quality_gi as hq
from diagnose_unpaired_optical_calibration import (
    BucketCritic,
    DEFAULT_CACHE,
    evaluate,
    normalized_record,
    sample_batch,
    set_seed,
)


DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round23/"
    "unpaired_optical_calibration_multi"
)


def inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class PhysicalTransfer(nn.Module):
    """Positive anisotropic PSF and positive low-order illumination field."""

    def __init__(self, *, initial_sigma: float = 0.30) -> None:
        super().__init__()
        self.floor = 0.05
        raw = inv_softplus(initial_sigma - self.floor)
        self.raw_sigma_x = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self.raw_sigma_y = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self.raw_vignette_x = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.raw_vignette_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def parameters_physical(self) -> dict[str, torch.Tensor]:
        return {
            "sigma_x": self.floor + F.softplus(self.raw_sigma_x),
            "sigma_y": self.floor + F.softplus(self.raw_sigma_y),
            "vignette_x": 0.60 * torch.tanh(self.raw_vignette_x),
            "vignette_y": 0.60 * torch.tanh(self.raw_vignette_y),
        }


def fixed_parameters(
    *,
    sigma_x: float,
    sigma_y: float,
    vignette_x: float,
    vignette_y: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "sigma_x": torch.tensor(sigma_x, device=device),
        "sigma_y": torch.tensor(sigma_y, device=device),
        "vignette_x": torch.tensor(vignette_x, device=device),
        "vignette_y": torch.tensor(vignette_y, device=device),
    }


def transfer_images(
    images: torch.Tensor,
    physical: dict[str, torch.Tensor],
    *,
    kernel_size: int = 11,
) -> torch.Tensor:
    dtype = images.dtype
    device = images.device
    coord = torch.linspace(-1.0, 1.0, images.shape[-1], device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    illumination = torch.exp(
        physical["vignette_x"] * xx + physical["vignette_y"] * yy
    )
    illumination = illumination / illumination.mean()

    kcoord = torch.arange(kernel_size, device=device, dtype=dtype)
    kcoord = kcoord - float(kernel_size // 2)
    ky = torch.exp(-kcoord.square() / (2.0 * physical["sigma_y"].square()))
    kx = torch.exp(-kcoord.square() / (2.0 * physical["sigma_x"].square()))
    kernel = ky[:, None] * kx[None, :]
    kernel = (kernel / kernel.sum()).reshape(1, 1, kernel_size, kernel_size)

    # Actual pattern = illumination * blur(nominal pattern).  The adjoint
    # effective object is blur(illumination * object) for the symmetric PSF.
    weighted = images * illumination.reshape(1, 1, *illumination.shape)
    return F.conv2d(weighted, kernel, padding=kernel_size // 2)


def measure_multi(
    images: torch.Tensor,
    rows: torch.Tensor,
    physical: dict[str, torch.Tensor],
) -> torch.Tensor:
    return transfer_images(images, physical).flatten(1) @ rows.T


def serialize_physical(physical: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in physical.items()}


def fit_adversarial(
    real_images: torch.Tensor,
    reference_images: torch.Tensor,
    rows: torch.Tensor,
    truth: dict[str, torch.Tensor],
    *,
    steps: int,
    batch_size: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    with torch.no_grad():
        real_records = normalized_record(measure_multi(real_images, rows, truth))
        center = real_records.mean(0)
        scale = real_records.std(0).clamp_min(1.0e-3)

    generator = PhysicalTransfer().to(rows.device)
    critic = BucketCritic(real_records.shape[1]).to(rows.device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1.5e-3, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=2.0e-4, betas=(0.5, 0.9))
    log: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        for _ in range(3):
            real = sample_batch(real_records, batch_size)
            clean = sample_batch(reference_images, batch_size)
            with torch.no_grad():
                fake = normalized_record(
                    measure_multi(clean, rows, generator.parameters_physical())
                )
            loss_d = F.softplus(-critic((real - center) / scale)).mean()
            loss_d = loss_d + F.softplus(critic((fake - center) / scale)).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

        clean = sample_batch(reference_images, batch_size)
        fake = normalized_record(
            measure_multi(clean, rows, generator.parameters_physical())
        )
        loss_g = F.softplus(-critic((fake - center) / scale)).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if step == 1 or step % 50 == 0 or step == steps:
            with torch.no_grad():
                real_probe = (sample_batch(real_records, batch_size) - center) / scale
                fake_probe = normalized_record(
                    measure_multi(
                        sample_batch(reference_images, batch_size),
                        rows,
                        generator.parameters_physical(),
                    )
                )
                fake_probe = (fake_probe - center) / scale
                gap = critic(real_probe).mean() - critic(fake_probe).mean()
            row = {
                "step": float(step),
                **serialize_physical(generator.parameters_physical()),
                "d_loss": float(loss_d.detach().cpu()),
                "g_loss": float(loss_g.detach().cpu()),
                "critic_gap": float(gap.detach().cpu()),
            }
            log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return serialize_physical(generator.parameters_physical()), log


def fit_moments(
    real_images: torch.Tensor,
    reference_images: torch.Tensor,
    rows: torch.Tensor,
    truth: dict[str, torch.Tensor],
    *,
    steps: int,
    batch_size: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    with torch.no_grad():
        real_records = normalized_record(measure_multi(real_images, rows, truth))
        target_mean = real_records.mean(0)
        target_std = real_records.std(0).clamp_min(1.0e-4)
    generator = PhysicalTransfer().to(rows.device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=1.5e-3)
    log: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        fake = normalized_record(
            measure_multi(
                sample_batch(reference_images, batch_size),
                rows,
                generator.parameters_physical(),
            )
        )
        mean_loss = ((fake.mean(0) - target_mean) / target_std).square().mean()
        std_loss = ((fake.std(0) - target_std) / target_std).square().mean()
        loss = mean_loss + std_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            row = {
                "step": float(step),
                **serialize_physical(generator.parameters_physical()),
                "moment_loss": float(loss.detach().cpu()),
            }
            log.append(row)
    return serialize_physical(generator.parameters_physical()), log


def float_physical(
    values: dict[str, float], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(value, device=device) for key, value in values.items()}


@torch.no_grad()
def ridge_reconstruct(
    train_images: torch.Tensor,
    validation_images: torch.Tensor,
    rows: torch.Tensor,
    assumed: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
    ridge: float = 1.0e-3,
) -> torch.Tensor:
    y_train = measure_multi(train_images, rows, assumed)
    y_val = measure_multi(validation_images, rows, truth)
    x_train = train_images.flatten(1)
    mean_x = x_train.mean(0, keepdim=True)
    mean_y = y_train.mean(0, keepdim=True)
    xc = x_train - mean_x
    yc = y_train - mean_y
    gram = yc.T @ yc / float(yc.shape[0])
    cross = yc.T @ xc / float(yc.shape[0])
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    weight = torch.linalg.solve(gram + ridge * eye, cross)
    return (mean_x + (y_val - mean_y) @ weight).reshape_as(validation_images).clamp(
        0.0, 1.0
    )


def parameter_error(
    estimate: dict[str, float], truth: dict[str, torch.Tensor]
) -> dict[str, float]:
    return {
        key: float(estimate[key] - float(value.detach().cpu()))
        for key, value in truth.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
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
    truth = fixed_parameters(
        sigma_x=1.40,
        sigma_y=0.70,
        vignette_x=0.35,
        vignette_y=-0.25,
        device=device,
    )
    real_images = train[: train.shape[0] // 2]
    reference_images = train[train.shape[0] // 2 :]

    adversarial, adversarial_log = fit_adversarial(
        real_images,
        reference_images,
        rows,
        truth,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    moments, moment_log = fit_moments(
        real_images,
        reference_images,
        rows,
        truth,
        steps=args.steps,
        batch_size=args.batch_size,
    )

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    nominal = fixed_parameters(
        sigma_x=0.05,
        sigma_y=0.05,
        vignette_x=0.0,
        vignette_y=0.0,
        device=device,
    )
    methods = {
        "nominal_uncalibrated": nominal,
        "moment_calibrated": float_physical(moments, device),
        "adversarial_calibrated": float_physical(adversarial, device),
        "oracle_calibrated": truth,
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, physical in methods.items():
        prediction = ridge_reconstruct(train, val, rows, physical, truth)
        metrics[name] = evaluate(prediction, val, lpips_model)
        metrics[name].update(serialize_physical(physical))
        print(name, json.dumps(metrics[name], sort_keys=True), flush=True)

    payload = {
        "status": "MULTI_PARAMETER_MICRO_PILOT",
        "validation_only": True,
        "test_split_opened": False,
        "truth": serialize_physical(truth),
        "adversarial": adversarial,
        "adversarial_error": parameter_error(adversarial, truth),
        "moments": moments,
        "moment_error": parameter_error(moments, truth),
        "operator_sha256": manifest["rows_sha256"],
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
