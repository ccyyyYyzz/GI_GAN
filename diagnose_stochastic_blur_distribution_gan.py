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
    blur_images,
    evaluate,
    normalized_record,
    sample_batch,
    set_seed,
)


DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round24/"
    "stochastic_blur_distribution"
)


def inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class TwoStateBlurGenerator(nn.Module):
    def __init__(self, low: float = 0.25, high: float = 0.90) -> None:
        super().__init__()
        self.floor = 0.05
        self.min_gap = 0.10
        self.raw_low = nn.Parameter(
            torch.tensor(inv_softplus(low - self.floor), dtype=torch.float32)
        )
        self.raw_gap = nn.Parameter(
            torch.tensor(inv_softplus(high - low - self.min_gap), dtype=torch.float32)
        )

    def sigmas(self) -> torch.Tensor:
        low = self.floor + F.softplus(self.raw_low)
        high = low + self.min_gap + F.softplus(self.raw_gap)
        return torch.stack([low, high])


def measure_states(
    images: torch.Tensor,
    rows: torch.Tensor,
    sigmas: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        images.shape[0], rows.shape[0], device=images.device, dtype=images.dtype
    )
    for state in range(2):
        index = labels == state
        if bool(index.any()):
            effective = blur_images(images[index], sigmas[state])
            output[index] = effective.flatten(1) @ rows.T
    return output


def fixed_labels(count: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 2, (count,), generator=generator).to(device)


def fit_adversarial(
    real_images: torch.Tensor,
    reference_images: torch.Tensor,
    rows: torch.Tensor,
    true_sigmas: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    state_seed: int,
) -> tuple[list[float], list[dict[str, float]]]:
    real_labels = fixed_labels(real_images.shape[0], state_seed, rows.device)
    with torch.no_grad():
        real_records = normalized_record(
            measure_states(real_images, rows, true_sigmas, real_labels)
        )
        center = real_records.mean(0)
        scale = real_records.std(0).clamp_min(1.0e-3)

    generator = TwoStateBlurGenerator().to(rows.device)
    critic = BucketCritic(real_records.shape[1]).to(rows.device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1.5e-3, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=2.0e-4, betas=(0.5, 0.9))
    log: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        for _ in range(3):
            real = sample_batch(real_records, batch_size)
            clean = sample_batch(reference_images, batch_size)
            labels = torch.randint(0, 2, (batch_size,), device=rows.device)
            with torch.no_grad():
                fake = normalized_record(
                    measure_states(clean, rows, generator.sigmas(), labels)
                )
            loss_d = F.softplus(-critic((real - center) / scale)).mean()
            loss_d = loss_d + F.softplus(critic((fake - center) / scale)).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

        clean = sample_batch(reference_images, batch_size)
        labels = torch.randint(0, 2, (batch_size,), device=rows.device)
        fake = normalized_record(measure_states(clean, rows, generator.sigmas(), labels))
        loss_g = F.softplus(-critic((fake - center) / scale)).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        if step == 1 or step % 50 == 0 or step == steps:
            sigmas = generator.sigmas().detach().cpu().tolist()
            row = {
                "step": float(step),
                "sigma_low": float(sigmas[0]),
                "sigma_high": float(sigmas[1]),
                "d_loss": float(loss_d.detach().cpu()),
                "g_loss": float(loss_g.detach().cpu()),
            }
            log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    return [float(v) for v in generator.sigmas().detach().cpu().tolist()], log


def fit_moments(
    real_images: torch.Tensor,
    reference_images: torch.Tensor,
    rows: torch.Tensor,
    true_sigmas: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    state_seed: int,
) -> tuple[list[float], list[dict[str, float]]]:
    real_labels = fixed_labels(real_images.shape[0], state_seed, rows.device)
    with torch.no_grad():
        real_records = normalized_record(
            measure_states(real_images, rows, true_sigmas, real_labels)
        )
        target_mean = real_records.mean(0)
        target_std = real_records.std(0).clamp_min(1.0e-4)
    generator = TwoStateBlurGenerator().to(rows.device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=1.5e-3)
    log: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        clean = sample_batch(reference_images, batch_size)
        labels = torch.randint(0, 2, (batch_size,), device=rows.device)
        fake = normalized_record(measure_states(clean, rows, generator.sigmas(), labels))
        mean_loss = ((fake.mean(0) - target_mean) / target_std).square().mean()
        std_loss = ((fake.std(0) - target_std) / target_std).square().mean()
        loss = mean_loss + std_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            sigmas = generator.sigmas().detach().cpu().tolist()
            log.append(
                {
                    "step": float(step),
                    "sigma_low": float(sigmas[0]),
                    "sigma_high": float(sigmas[1]),
                    "moment_loss": float(loss.detach().cpu()),
                }
            )
    return [float(v) for v in generator.sigmas().detach().cpu().tolist()], log


@torch.no_grad()
def fit_ridge_model(
    train_images: torch.Tensor,
    rows: torch.Tensor,
    sigma: torch.Tensor,
    ridge: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.zeros(train_images.shape[0], dtype=torch.long, device=rows.device)
    y_train = measure_states(train_images, rows, torch.stack([sigma, sigma]), labels)
    x_train = train_images.flatten(1)
    mean_x = x_train.mean(0, keepdim=True)
    mean_y = y_train.mean(0, keepdim=True)
    xc = x_train - mean_x
    yc = y_train - mean_y
    gram = yc.T @ yc / float(yc.shape[0])
    cross = yc.T @ xc / float(yc.shape[0])
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    weight = torch.linalg.solve(gram + ridge * eye, cross)
    return mean_x, mean_y, weight


@torch.no_grad()
def predict_ridge(
    record: torch.Tensor,
    model: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    shape: tuple[int, ...],
) -> torch.Tensor:
    mean_x, mean_y, weight = model
    return (mean_x + (record - mean_y) @ weight).reshape(shape).clamp(0.0, 1.0)


@torch.no_grad()
def state_gaussians(
    images: torch.Tensor,
    rows: torch.Tensor,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    means = []
    variances = []
    for state in range(2):
        labels = torch.full(
            (images.shape[0],), state, dtype=torch.long, device=images.device
        )
        record = normalized_record(measure_states(images, rows, sigmas, labels))
        means.append(record.mean(0))
        variances.append(record.var(0).clamp_min(1.0e-5))
    return torch.stack(means), torch.stack(variances)


@torch.no_grad()
def classify_states(
    record: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
) -> torch.Tensor:
    feature = normalized_record(record)
    scores = []
    for state in range(2):
        score = -0.5 * (
            (feature - means[state]).square() / variances[state]
            + variances[state].log()
        ).mean(1)
        scores.append(score)
    return torch.stack(scores, dim=1).argmax(1)


@torch.no_grad()
def evaluate_mixture(
    train_images: torch.Tensor,
    reference_images: torch.Tensor,
    validation_images: torch.Tensor,
    rows: torch.Tensor,
    assumed_sigmas: torch.Tensor,
    true_sigmas: torch.Tensor,
    validation_labels: torch.Tensor,
    lpips_model: nn.Module,
    *,
    use_true_labels: bool = False,
) -> dict[str, float]:
    true_record = measure_states(
        validation_images, rows, true_sigmas, validation_labels
    )
    if use_true_labels:
        selected = validation_labels
    else:
        means, variances = state_gaussians(reference_images, rows, assumed_sigmas)
        selected = classify_states(true_record, means, variances)
    predictions = torch.empty_like(validation_images)
    for state in range(2):
        index = selected == state
        if bool(index.any()):
            model = fit_ridge_model(train_images, rows, assumed_sigmas[state])
            predictions[index] = predict_ridge(
                true_record[index], model, tuple(validation_images[index].shape)
            )
    metrics = evaluate(predictions, validation_images, lpips_model)
    metrics["state_accuracy"] = float(
        (selected == validation_labels).float().mean().cpu()
    )
    metrics["sigma_low"] = float(assumed_sigmas[0].cpu())
    metrics["sigma_high"] = float(assumed_sigmas[1].cpu())
    return metrics


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
    true_sigmas = torch.tensor([0.45, 1.55], device=device)
    real_images = train[: train.shape[0] // 2]
    reference_images = train[train.shape[0] // 2 :]
    state_seed = args.seed + 101

    adversarial, adversarial_log = fit_adversarial(
        real_images,
        reference_images,
        rows,
        true_sigmas,
        steps=args.steps,
        batch_size=args.batch_size,
        state_seed=state_seed,
    )
    moments, moment_log = fit_moments(
        real_images,
        reference_images,
        rows,
        true_sigmas,
        steps=args.steps,
        batch_size=args.batch_size,
        state_seed=state_seed,
    )

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    validation_labels = fixed_labels(val.shape[0], state_seed + 1, device)
    methods = {
        "nominal_single_state": torch.tensor([0.05, 0.05], device=device),
        "moment_mixture": torch.tensor(moments, device=device),
        "adversarial_mixture": torch.tensor(adversarial, device=device),
        "oracle_mixture_blind_state": true_sigmas,
    }
    metrics = {}
    for name, sigmas in methods.items():
        metrics[name] = evaluate_mixture(
            train,
            reference_images,
            val,
            rows,
            sigmas,
            true_sigmas,
            validation_labels,
            lpips_model,
        )
        print(name, json.dumps(metrics[name], sort_keys=True), flush=True)
    metrics["oracle_mixture_true_state"] = evaluate_mixture(
        train,
        reference_images,
        val,
        rows,
        true_sigmas,
        true_sigmas,
        validation_labels,
        lpips_model,
        use_true_labels=True,
    )
    print(
        "oracle_mixture_true_state",
        json.dumps(metrics["oracle_mixture_true_state"], sort_keys=True),
        flush=True,
    )

    payload = {
        "status": "STOCHASTIC_BLUR_DISTRIBUTION_MICRO",
        "validation_only": True,
        "test_split_opened": False,
        "true_sigmas": true_sigmas.cpu().tolist(),
        "adversarial_sigmas": adversarial,
        "moment_sigmas": moments,
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
