from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import gan_high_quality_gi as hq
from diagnose_afrb_proposal_headroom import project_predictions
from diagnose_vqgan_causal_disagreement_controls import metric_vectors, paired_summary
from src.gauge_geometry import GaugeGeometry
from src.losses import differentiable_ssim_loss
from src.vqae_centered_residual_adapter import VQAECenteredResidualAdapter


def prepare_split(
    primary: dict,
    source: dict,
    geometry: GaugeGeometry,
    *,
    arm: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    for key in ("truth", "x0", "y", "source_index"):
        if not torch.equal(primary[key], source[key]):
            raise RuntimeError(f"CACHE_IDENTITY_MISMATCH:{key}")
    truth = primary["truth"].float().to(device)
    anchor = primary["x0"].float().to(device).clamp(0.0, 1.0)
    intrinsic = anchor.flatten(1).to(torch.float64) @ geometry.Q.T
    base_residual = geometry.null_project_flat(
        primary["x_A"].float().to(device).flatten(1) - anchor.flatten(1)
    )
    base, _ = project_predictions(
        (anchor.flatten(1) + base_residual).reshape_as(anchor),
        intrinsic,
        geometry,
        batch_size=int(batch_size),
    )
    source_key = "x_G" if arm == "gan" else "x_A"
    source_residual = geometry.null_project_flat(
        source[source_key].float().to(device).flatten(1) - anchor.flatten(1)
    )
    alternative, _ = project_predictions(
        (anchor.flatten(1) + source_residual).reshape_as(anchor),
        intrinsic,
        geometry,
        batch_size=int(batch_size),
    )
    direction = geometry.null_project_flat(
        alternative.flatten(1) - base.flatten(1)
    ).reshape_as(base)
    return {
        "truth": truth.cpu(),
        "anchor": anchor.cpu(),
        "base": base.cpu(),
        "direction": direction.cpu(),
        "intrinsic": intrinsic.cpu(),
        "source_index": primary["source_index"].cpu(),
    }


def apply_model(
    model: VQAECenteredResidualAdapter,
    base: torch.Tensor,
    direction: torch.Tensor,
    anchor: torch.Tensor,
    geometry: GaugeGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_correction, weight = model(base, direction, anchor)
    correction = geometry.null_project_flat(raw_correction.flatten(1)).reshape_as(base)
    return (base + correction).clamp(0.0, 1.0), weight


@torch.no_grad()
def predict_all(
    model: VQAECenteredResidualAdapter,
    split: dict[str, torch.Tensor],
    geometry: GaugeGeometry,
    *,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    predictions = []
    weights = []
    model.eval()
    for start in range(0, len(indices), int(batch_size)):
        index = indices[start : start + int(batch_size)]
        pred, weight = apply_model(
            model,
            split["base"][index].to(device),
            split["direction"][index].to(device),
            split["anchor"][index].to(device),
            geometry,
        )
        predictions.append(pred.cpu())
        weights.append(weight.cpu())
    weight = torch.cat(weights)
    return torch.cat(predictions), {
        "weight_mean": float(weight.mean()),
        "weight_std": float(weight.std(unbiased=False)),
        "weight_min": float(weight.min()),
        "weight_max": float(weight.max()),
    }


def metric_means(vectors: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(value.mean()) for name, value in vectors.items()}


def joint_score(candidate: dict[str, float], reference: dict[str, float]) -> float:
    deltas = {
        "psnr": candidate["psnr"] - reference["psnr"],
        "ssim": candidate["ssim"] - reference["ssim"],
        "lpips": reference["lpips"] - candidate["lpips"],
    }
    normalized = deltas["psnr"] / 0.05 + deltas["ssim"] / 0.001 + deltas["lpips"] / 0.01
    penalty = sum(min(value, 0.0) for value in deltas.values()) * 1000.0
    return float(normalized + penalty)


def train_variant(
    *,
    split: dict[str, torch.Tensor],
    geometry: GaugeGeometry,
    lpips_model: torch.nn.Module,
    architecture: str,
    lambda_lpips: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[VQAECenteredResidualAdapter, dict]:
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(split["truth"]), generator=generator)
    train_index = permutation[:384]
    calibration_index = permutation[384:]
    model = VQAECenteredResidualAdapter(
        architecture=architecture,
        maximum_weight=0.35,
        initial_weight=0.10,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    base_calibration = split["base"][calibration_index].to(device)
    truth_calibration = split["truth"][calibration_index].to(device)
    reference_vectors = metric_vectors(base_calibration, truth_calibration, lpips_model)
    reference_means = metric_means(reference_vectors)
    best = {"score": float("-inf"), "step": -1, "means": None, "weight": None}
    history = []
    for step in range(1, int(steps) + 1):
        selected = train_index[
            torch.randint(0, len(train_index), (int(batch_size),), generator=generator)
        ]
        base = split["base"][selected].to(device)
        direction = split["direction"][selected].to(device)
        anchor = split["anchor"][selected].to(device)
        truth = split["truth"][selected].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            prediction, _ = apply_model(model, base, direction, anchor, geometry)
            mse = F.mse_loss(prediction, truth)
            l1 = F.l1_loss(prediction, truth)
            ssim_loss = differentiable_ssim_loss(prediction, truth)
            pred_rgb = prediction.repeat(1, 3, 1, 1) * 2.0 - 1.0
            truth_rgb = truth.repeat(1, 3, 1, 1) * 2.0 - 1.0
            perceptual = lpips_model(pred_rgb, truth_rgb).mean()
            loss = mse + 0.05 * l1 + 0.01 * ssim_loss + float(lambda_lpips) * perceptual
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 100 == 0 or step == int(steps):
            prediction, weight_audit = predict_all(
                model,
                split,
                geometry,
                indices=calibration_index,
                batch_size=int(batch_size),
                device=device,
            )
            vectors = metric_vectors(prediction.to(device), truth_calibration, lpips_model)
            means = metric_means(vectors)
            score = joint_score(means, reference_means)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "score": score,
                "means": means,
                "weight": weight_audit,
            }
            history.append(row)
            print(json.dumps({"variant": f"{architecture}_lp{lambda_lpips:g}", **row}), flush=True)
            if score > float(best["score"]):
                best = {**row}
                torch.save(
                    {
                        "model": model.state_dict(),
                        "architecture": architecture,
                        "lambda_lpips": float(lambda_lpips),
                        "step": step,
                        "seed": int(seed),
                    },
                    output_dir / f"checkpoint_{architecture}_lp{lambda_lpips:g}.pt",
                )
    checkpoint = torch.load(
        output_dir / f"checkpoint_{architecture}_lp{lambda_lpips:g}.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, {
        "architecture": architecture,
        "lambda_lpips": float(lambda_lpips),
        "reference_calibration": reference_means,
        "best": best,
        "history": history,
        "train_indices_sha256": hq.sha256_numpy(train_index.numpy()),
        "calibration_indices_sha256": hq.sha256_numpy(calibration_index.numpy()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dev", type=Path, required=True)
    parser.add_argument("--primary-val", type=Path, required=True)
    parser.add_argument("--control-dev", type=Path)
    parser.add_argument("--control-val", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=["gan", "vqae_control"], required=True)
    parser.add_argument("--architectures", default="spatial")
    parser.add_argument("--lpips-weights", default="0,0.005,0.015")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed))
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    started = time.time()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows_np, manifest = hq.build_structured_operator_rows(
        img_size=int(config["data"]["img_size"]),
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    geometry = GaugeGeometry.from_rows_qr(
        torch.from_numpy(rows_np).to(torch.float64)
    ).to(device)
    if geometry.info.rows_sha256 != manifest["rows_sha256"]:
        raise RuntimeError("OPERATOR_IDENTITY_MISMATCH")

    primary_dev = torch.load(args.primary_dev, map_location="cpu", weights_only=False)
    primary_val = torch.load(args.primary_val, map_location="cpu", weights_only=False)
    if args.arm == "gan":
        source_dev, source_val = primary_dev, primary_val
    else:
        if args.control_dev is None or args.control_val is None:
            raise ValueError("VQAE_CONTROL_REQUIRES_CONTROL_CACHES")
        source_dev = torch.load(args.control_dev, map_location="cpu", weights_only=False)
        source_val = torch.load(args.control_val, map_location="cpu", weights_only=False)
    print("PREPARE_DEV", flush=True)
    dev = prepare_split(
        primary_dev,
        source_dev,
        geometry,
        arm=args.arm,
        batch_size=int(args.batch_size),
        device=device,
    )
    print("PREPARE_VAL", flush=True)
    val = prepare_split(
        primary_val,
        source_val,
        geometry,
        arm=args.arm,
        batch_size=int(args.batch_size),
        device=device,
    )
    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)
    val_index = torch.arange(len(val["truth"]))
    base_vectors = metric_vectors(val["base"].to(device), val["truth"].to(device), lpips_model)
    base_means = metric_means(base_vectors)

    results = []
    for architecture_index, architecture in enumerate(args.architectures.split(",")):
        for weight_index, lambda_lpips in enumerate(
            float(value) for value in args.lpips_weights.split(",")
        ):
            variant_seed = int(args.seed) + 1000 * architecture_index + 17 * weight_index
            model, training = train_variant(
                split=dev,
                geometry=geometry,
                lpips_model=lpips_model,
                architecture=architecture,
                lambda_lpips=float(lambda_lpips),
                steps=int(args.steps),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                seed=variant_seed,
                device=device,
                output_dir=args.output_dir,
            )
            prediction, weight_audit = predict_all(
                model,
                val,
                geometry,
                indices=val_index,
                batch_size=int(args.batch_size),
                device=device,
            )
            projected, projection_audit = project_predictions(
                prediction.to(device),
                val["intrinsic"].to(device),
                geometry,
                batch_size=int(args.batch_size),
            )
            vectors = metric_vectors(projected, val["truth"].to(device), lpips_model)
            means = metric_means(vectors)
            paired = paired_summary(
                vectors,
                base_vectors,
                bootstrap_reps=int(args.bootstrap_reps),
                seed=variant_seed + 90001,
            )
            result = {
                "architecture": architecture,
                "lambda_lpips": float(lambda_lpips),
                "training": training,
                "validation_means": means,
                "paired_vs_vqae": paired,
                "weight_audit": weight_audit,
                "projection_audit": projection_audit,
            }
            results.append(result)
            (args.output_dir / "partial_results.json").write_text(
                json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
            )

    payload = {
        "status": "VQAE_CENTERED_RESIDUAL_ADAPTER_PILOT",
        "arm": args.arm,
        "validation_only": True,
        "test_split_opened": False,
        "operator_sha256": geometry.info.rows_sha256,
        "dev_images": len(dev["truth"]),
        "validation_images": len(val["truth"]),
        "base_validation_means": base_means,
        "results": results,
        "runtime_seconds": time.time() - started,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
