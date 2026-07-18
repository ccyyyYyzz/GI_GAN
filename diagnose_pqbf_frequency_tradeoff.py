from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import gan_high_quality_gi as hq
from completion_projector_gan import exact_model_prediction
from src.gauge_geometry import GaugeEmpiricalAnchor, GaugeGeometry
from src.projector_gated_fiber_gan import ProjectorGatedFiberGenerator


def _load_checkpoint_model(
    path: Path,
    *,
    geometry: GaugeGeometry,
    steps: int,
    step_scale: float,
    device: torch.device,
) -> ProjectorGatedFiberGenerator:
    payload = torch.load(path, map_location=device)
    model = ProjectorGatedFiberGenerator(
        geometry,
        steps=int(steps),
        step_scale=float(step_scale),
    ).to(device)
    model.load_state_dict(payload["ema"])
    return model.eval()


def _radial_masks(height: int, width: int, device: torch.device) -> dict[str, torch.Tensor]:
    fy = torch.fft.fftfreq(height, device=device).reshape(height, 1)
    fx = torch.fft.fftfreq(width, device=device).reshape(1, width)
    radius = torch.sqrt(fx.square() + fy.square()) / (2.0**0.5 / 2.0)
    return {
        "low": radius <= 0.20,
        "mid": (radius > 0.20) & (radius <= 0.50),
        "high": radius > 0.50,
    }


def _band_error(error: torch.Tensor, masks: dict[str, torch.Tensor]) -> dict[str, float]:
    spectrum = torch.fft.fft2(error.squeeze(1), norm="ortho")
    power = spectrum.abs().square()
    values: dict[str, float] = {}
    for name, mask in masks.items():
        values[name] = float(power[:, mask].mean().detach().cpu())
    values["all"] = float(power.mean().detach().cpu())
    return values


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    numerator = (a_flat * b_flat).sum(dim=1)
    denominator = a_flat.norm(dim=1) * b_flat.norm(dim=1)
    return float((numerator / denominator.clamp_min(1e-12)).mean().detach().cpu())


def _predict(
    model: ProjectorGatedFiberGenerator,
    *,
    anchor: torch.Tensor,
    intrinsic: torch.Tensor,
    uncertainty: torch.Tensor,
    geometry: GaugeGeometry,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, anchor.shape[0], int(batch_size)):
        stop = min(anchor.shape[0], start + int(batch_size))
        prediction, _projection = exact_model_prediction(
            model,
            anchor=anchor[start:stop],
            intrinsic=intrinsic[start:stop],
            uncertainty=uncertainty.expand(stop - start, -1, -1, -1),
            geometry=geometry,
        )
        outputs.append(prediction.cpu())
    return torch.cat(outputs, dim=0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    rows_np, operator_manifest = hq.build_structured_operator_rows(
        img_size=int(config["data"]["img_size"]),
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    geometry = GaugeGeometry(
        torch.from_numpy(rows_np).to(torch.float64),
        relative_cutoff=float(config["operator"].get("svd_relative_cutoff", 1e-12)),
    ).to(device)
    if geometry.info.rows_sha256 != operator_manifest["rows_sha256"]:
        raise RuntimeError("operator identity mismatch")

    train_payload = torch.load(args.train_cache, map_location="cpu")
    val_payload = torch.load(args.val_cache, map_location="cpu")
    train_truth = train_payload["tensors"]["truth"].reshape(
        train_payload["tensors"]["truth"].shape[0], -1
    ).numpy()
    anchor_model = GaugeEmpiricalAnchor.fit(
        train_truth,
        geometry,
        lambda_=float(config["operator"].get("lmmse_lambda", 1e-3)),
    )
    uncertainty = anchor_model.normalized_posterior_map(
        img_size=int(config["data"]["img_size"]),
        device=device,
    )

    truth = val_payload["tensors"]["truth"].to(device)
    anchor = val_payload["tensors"]["anchor"].to(device)
    intrinsic = val_payload["tensors"]["intrinsic"].to(device)
    common = {
        "geometry": geometry,
        "steps": int(config["model"]["steps"]),
        "step_scale": float(config["model"]["step_scale"]),
        "device": device,
    }
    models = {
        "content": _load_checkpoint_model(args.stage_a, **common),
        "gan": _load_checkpoint_model(args.stage_b, **common),
        "matched": _load_checkpoint_model(args.matched, **common),
    }
    predictions = {
        name: _predict(
            model,
            anchor=anchor,
            intrinsic=intrinsic,
            uncertainty=uncertainty,
            geometry=geometry,
            batch_size=int(args.batch_size),
        ).to(device)
        for name, model in models.items()
    }

    masks = _radial_masks(truth.shape[-2], truth.shape[-1], device)
    rows: list[dict[str, Any]] = []
    errors: dict[str, dict[str, float]] = {}
    for name, prediction in {"anchor": anchor, **predictions}.items():
        band_values = _band_error(prediction - truth, masks)
        errors[name] = band_values
        for band, value in band_values.items():
            rows.append({"method": name, "band": band, "fourier_mse": value})

    gan_minus_matched = predictions["gan"] - predictions["matched"]
    remaining_error = truth - predictions["matched"]
    helpful_fraction = float(
        (
            (predictions["gan"] - truth).square().reshape(truth.shape[0], -1).mean(dim=1)
            < (predictions["matched"] - truth)
            .square()
            .reshape(truth.shape[0], -1)
            .mean(dim=1)
        )
        .float()
        .mean()
        .cpu()
    )
    result = {
        "status": "PQBF_FREQUENCY_TRADEOFF_DIAG_COMPLETE",
        "split": "validation_only",
        "n": int(truth.shape[0]),
        "operator_rows_sha256": geometry.info.rows_sha256,
        "frequency_band_definition": {
            "low": "normalized radial frequency <= 0.20",
            "mid": "0.20 < normalized radial frequency <= 0.50",
            "high": "normalized radial frequency > 0.50",
        },
        "fourier_mse": errors,
        "gan_minus_matched_relative_change": {
            band: (errors["gan"][band] - errors["matched"][band])
            / max(errors["matched"][band], 1e-12)
            for band in errors["gan"]
        },
        "gan_update_alignment_with_matched_remaining_error": _cosine(
            gan_minus_matched, remaining_error
        ),
        "fraction_of_images_with_lower_pixel_mse_gan_vs_matched": helpful_fraction,
        "inputs": {
            "config": str(args.config),
            "stage_a": str(args.stage_a),
            "stage_b": str(args.stage_b),
            "matched": str(args.matched),
            "train_cache": str(args.train_cache),
            "val_cache": str(args.val_cache),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frequency_tradeoff.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "frequency_tradeoff.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "band", "fourier_mse"])
        writer.writeheader()
        writer.writerows(rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--stage-b", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
