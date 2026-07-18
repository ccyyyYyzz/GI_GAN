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
from src.fiber_residual_phase_gan import (
    ConditionalHighPassDiscriminator,
    FiberResidualPhaseGenerator,
    hinge_discriminator_loss,
)
from src.gauge_geometry import GaugeGeometry
from src.losses import differentiable_ssim_loss
from train_vqae_centered_residual_adapter import (
    joint_score,
    metric_means,
    prepare_split,
)


def apply_generator(
    model: FiberResidualPhaseGenerator,
    base: torch.Tensor,
    direction: torch.Tensor,
    anchor: torch.Tensor,
    geometry: GaugeGeometry,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    raw_correction, audit = model(base, direction, anchor)
    correction = geometry.null_project_flat(raw_correction.flatten(1)).reshape_as(base)
    prediction = (base + correction).clamp(0.0, 1.0)
    return prediction, correction, audit


@torch.no_grad()
def predict_all(
    model: FiberResidualPhaseGenerator,
    split: dict[str, torch.Tensor],
    geometry: GaugeGeometry,
    *,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    predictions = []
    corrections = []
    weights = []
    rotations = []
    model.eval()
    for start in range(0, len(indices), int(batch_size)):
        index = indices[start : start + int(batch_size)]
        prediction, correction, audit = apply_generator(
            model,
            split["base"][index].to(device),
            split["direction"][index].to(device),
            split["anchor"][index].to(device),
            geometry,
        )
        predictions.append(prediction.cpu())
        corrections.append(correction.cpu())
        weights.append(audit["weight"].cpu())
        rotations.append(audit["rotation"].cpu())
    weight = torch.cat(weights)
    rotation = torch.cat(rotations)
    return torch.cat(predictions), torch.cat(corrections), {
        "weight_mean": float(weight.mean()),
        "weight_std": float(weight.std(unbiased=False)),
        "weight_min": float(weight.min()),
        "weight_max": float(weight.max()),
        "rotation_abs_mean": float(rotation.abs().mean()),
        "rotation_rms": float(rotation.square().mean().sqrt()),
    }


def phase_audit(
    correction: torch.Tensor,
    split: dict[str, torch.Tensor],
    geometry: GaugeGeometry,
    *,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    flat = geometry.null_project_flat(correction.to(device).flatten(1))
    target = geometry.null_project_flat(
        (split["truth"].to(device) - split["base"].to(device)).flatten(1)
    )
    norm = flat.square().sum(dim=1).sqrt().clamp_min(1.0e-12)
    target_norm = target.square().sum(dim=1).sqrt().clamp_min(1.0e-12)
    crest = math.sqrt(flat.shape[1]) * flat.abs().amax(dim=1) / norm
    cosine = (flat * target).sum(dim=1) / (norm * target_norm)
    beta = ((flat * target).sum(dim=1) / norm.square()).clamp(0.0, 4.0)
    audit = {
        "crest_mean": float(crest.mean()),
        "crest_median": float(crest.median()),
        "crest_max": float(crest.max()),
        "crest_le_4_fraction": float((crest <= 4.0).float().mean()),
        "cosine_mean": float(cosine.mean()),
        "cosine_positive_fraction": float((cosine > 0.0).float().mean()),
        "oracle_beta_mean": float(beta.mean()),
        "oracle_beta_median": float(beta.median()),
        "oracle_beta_positive_fraction": float((beta > 0.0).float().mean()),
    }
    return audit, beta


def train_variant(
    *,
    split: dict[str, torch.Tensor],
    geometry: GaugeGeometry,
    lpips_model: torch.nn.Module,
    source_arm: str,
    rotation_scale: float,
    adversarial_weight: float,
    lpips_weight: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
    discriminator_learning_rate: float,
    gan_start_step: int,
    channels: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[FiberResidualPhaseGenerator, dict]:
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(split["truth"]), generator=generator)
    train_index = permutation[:384]
    calibration_index = permutation[384:]
    model = FiberResidualPhaseGenerator(
        channels=int(channels),
        maximum_weight=0.35,
        initial_weight=0.10,
        rotation_scale=float(rotation_scale),
    ).to(device)
    discriminator = (
        ConditionalHighPassDiscriminator(channels=int(channels)).to(device)
        if float(adversarial_weight) > 0.0
        else None
    )
    optimizer_g = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4
    )
    optimizer_d = (
        torch.optim.Adam(
            discriminator.parameters(),
            lr=float(discriminator_learning_rate),
            betas=(0.0, 0.99),
        )
        if discriminator is not None
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    truth_calibration = split["truth"][calibration_index].to(device)
    reference_vectors = metric_vectors(
        split["base"][calibration_index].to(device), truth_calibration, lpips_model
    )
    reference_means = metric_means(reference_vectors)
    best = {"score": float("-inf"), "step": -1, "means": None, "audit": None}
    history = []
    variant_name = (
        f"{source_arm}_rot{float(rotation_scale):g}_adv{float(adversarial_weight):g}"
    )
    checkpoint_path = output_dir / f"checkpoint_{variant_name}.pt"

    for step in range(1, int(steps) + 1):
        selected = train_index[
            torch.randint(0, len(train_index), (int(batch_size),), generator=generator)
        ]
        base = split["base"][selected].to(device)
        direction = split["direction"][selected].to(device)
        anchor = split["anchor"][selected].to(device)
        truth = split["truth"][selected].to(device)
        gan_active = discriminator is not None and step >= int(gan_start_step)

        d_loss = torch.zeros((), device=device)
        if gan_active:
            assert optimizer_d is not None and discriminator is not None
            discriminator.train()
            optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake, _, _ = apply_generator(model, base, direction, anchor, geometry)
            real_score = discriminator(base, truth)
            fake_score = discriminator(base, fake.detach())
            d_loss = hinge_discriminator_loss(real_score, fake_score)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_d.step()

        model.train()
        optimizer_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            prediction, correction, _ = apply_generator(
                model, base, direction, anchor, geometry
            )
            unclipped = base + correction
            mse = F.mse_loss(prediction, truth)
            l1 = F.l1_loss(prediction, truth)
            ssim_loss = differentiable_ssim_loss(prediction, truth)
            perceptual = lpips_model(
                prediction.repeat(1, 3, 1, 1) * 2.0 - 1.0,
                truth.repeat(1, 3, 1, 1) * 2.0 - 1.0,
            ).mean()
            box_penalty = (
                F.relu(-unclipped).square().mean()
                + F.relu(unclipped - 1.0).square().mean()
            )
            adversarial = torch.zeros((), device=device)
            if gan_active:
                assert discriminator is not None
                adversarial = -discriminator(base, prediction).mean()
            loss = (
                mse
                + 0.05 * l1
                + 0.01 * ssim_loss
                + float(lpips_weight) * perceptual
                + 0.10 * box_penalty
                + float(adversarial_weight) * adversarial
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer_g)
        scaler.update()

        if step == 1 or step % 100 == 0 or step == int(steps):
            prediction, _, model_audit = predict_all(
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
                "d_loss": float(d_loss.detach().cpu()),
                "adversarial": float(adversarial.detach().cpu()),
                "score": score,
                "means": means,
                "audit": model_audit,
            }
            history.append(row)
            print(json.dumps({"variant": variant_name, **row}), flush=True)
            if score > float(best["score"]):
                best = {**row}
                torch.save(
                    {
                        "generator": model.state_dict(),
                        "discriminator": (
                            None if discriminator is None else discriminator.state_dict()
                        ),
                        "source_arm": source_arm,
                        "rotation_scale": float(rotation_scale),
                        "adversarial_weight": float(adversarial_weight),
                        "lpips_weight": float(lpips_weight),
                        "step": step,
                        "seed": int(seed),
                        "channels": int(channels),
                    },
                    checkpoint_path,
                )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model, {
        "source_arm": source_arm,
        "rotation_scale": float(rotation_scale),
        "adversarial_weight": float(adversarial_weight),
        "lpips_weight": float(lpips_weight),
        "reference_calibration": reference_means,
        "best": best,
        "history": history,
        "train_indices_sha256": hq.sha256_numpy(train_index.numpy()),
        "calibration_indices_sha256": hq.sha256_numpy(calibration_index.numpy()),
        "checkpoint": checkpoint_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dev", type=Path, required=True)
    parser.add_argument("--primary-val", type=Path, required=True)
    parser.add_argument("--control-dev", type=Path)
    parser.add_argument("--control-val", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-arm", choices=["gan", "vqae_control"], required=True)
    parser.add_argument("--rotation-scales", default="0.25,0.5")
    parser.add_argument("--adv-weights", default="0,0.0005,0.0015")
    parser.add_argument("--lpips-weight", type=float, default=0.003)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--discriminator-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gan-start-step", type=int, default=100)
    parser.add_argument("--channels", type=int, default=32)
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
    geometry = GaugeGeometry.from_rows_qr(torch.from_numpy(rows_np).to(torch.float64)).to(device)
    if geometry.info.rows_sha256 != manifest["rows_sha256"]:
        raise RuntimeError("OPERATOR_IDENTITY_MISMATCH")

    primary_dev = torch.load(args.primary_dev, map_location="cpu", weights_only=False)
    primary_val = torch.load(args.primary_val, map_location="cpu", weights_only=False)
    if args.source_arm == "gan":
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
        arm=args.source_arm,
        batch_size=int(args.batch_size),
        device=device,
    )
    print("PREPARE_VAL", flush=True)
    val = prepare_split(
        primary_val,
        source_val,
        geometry,
        arm=args.source_arm,
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

    rotation_scales = [float(value) for value in args.rotation_scales.split(",")]
    adversarial_weights = [float(value) for value in args.adv_weights.split(",")]
    results = []
    for rotation_index, rotation_scale in enumerate(rotation_scales):
        for weight_index, adversarial_weight in enumerate(adversarial_weights):
            variant_seed = int(args.seed) + 1000 * rotation_index + 17 * weight_index
            model, training = train_variant(
                split=dev,
                geometry=geometry,
                lpips_model=lpips_model,
                source_arm=args.source_arm,
                rotation_scale=float(rotation_scale),
                adversarial_weight=float(adversarial_weight),
                lpips_weight=float(args.lpips_weight),
                steps=int(args.steps),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                discriminator_learning_rate=float(args.discriminator_learning_rate),
                gan_start_step=int(args.gan_start_step),
                channels=int(args.channels),
                seed=variant_seed,
                device=device,
                output_dir=args.output_dir,
            )
            prediction, correction, model_audit = predict_all(
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
            paired = paired_summary(
                vectors,
                base_vectors,
                bootstrap_reps=int(args.bootstrap_reps),
                seed=variant_seed + 90001,
            )
            phase, oracle_beta = phase_audit(correction, val, geometry, device=device)
            oracle_raw = val["base"].to(device) + (
                oracle_beta[:, None, None, None] * correction.to(device)
            )
            oracle, oracle_projection_audit = project_predictions(
                oracle_raw,
                val["intrinsic"].to(device),
                geometry,
                batch_size=int(args.batch_size),
            )
            oracle_vectors = metric_vectors(oracle, val["truth"].to(device), lpips_model)
            result = {
                "source_arm": args.source_arm,
                "rotation_scale": float(rotation_scale),
                "adversarial_weight": float(adversarial_weight),
                "training": training,
                "validation_means": metric_means(vectors),
                "paired_vs_vqae": paired,
                "model_audit": model_audit,
                "phase_audit": phase,
                "projection_audit": projection_audit,
                "oracle_phase_means": metric_means(oracle_vectors),
                "oracle_phase_paired_vs_vqae": paired_summary(
                    oracle_vectors,
                    base_vectors,
                    bootstrap_reps=int(args.bootstrap_reps),
                    seed=variant_seed + 130001,
                ),
                "oracle_projection_audit": oracle_projection_audit,
            }
            results.append(result)
            (args.output_dir / "partial_results.json").write_text(
                json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
            )

    payload = {
        "status": "FIBER_RESIDUAL_PHASE_GAN_PILOT",
        "source_arm": args.source_arm,
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
