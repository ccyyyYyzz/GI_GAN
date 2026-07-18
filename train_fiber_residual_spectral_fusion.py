from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import gan_high_quality_gi as hq
from diagnose_afrb_proposal_headroom import project_predictions
from diagnose_fiber_residual_frequency_fusion import load_generator
from diagnose_vqgan_causal_disagreement_controls import metric_vectors, paired_summary
from src.fiber_residual_spectral_fusion import FiberResidualSpectralFusionGate
from src.gauge_geometry import GaugeGeometry
from src.losses import differentiable_ssim_loss
from train_fiber_residual_phase_gan import predict_all as predict_phase_model
from train_vqae_centered_residual_adapter import (
    joint_score,
    metric_means,
    prepare_split,
)


def apply_gate(
    model: FiberResidualSpectralFusionGate,
    base: torch.Tensor,
    reference: torch.Tensor,
    proposal: torch.Tensor,
    anchor: torch.Tensor,
    geometry: GaugeGeometry,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_correction, weights = model(base, reference, proposal, anchor)
    correction = geometry.null_project_flat(raw_correction.flatten(1)).reshape_as(base)
    return (base + correction).clamp(0.0, 1.0), correction, weights


@torch.no_grad()
def predict_gate(
    model: FiberResidualSpectralFusionGate,
    split: dict[str, torch.Tensor],
    geometry: GaugeGeometry,
    *,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    predictions = []
    corrections = []
    weights = []
    model.eval()
    for start in range(0, len(indices), int(batch_size)):
        index = indices[start : start + int(batch_size)]
        prediction, correction, weight = apply_gate(
            model,
            split["base"][index].to(device),
            split["reference_correction"][index].to(device),
            split["proposal_correction"][index].to(device),
            split["anchor"][index].to(device),
            geometry,
        )
        predictions.append(prediction.cpu())
        corrections.append(correction.cpu())
        weights.append(weight.cpu())
    stacked = torch.cat(weights)
    return torch.cat(predictions), torch.cat(corrections), {
        "band_weight_mean": [float(value) for value in stacked.mean(dim=0)],
        "band_weight_std": [float(value) for value in stacked.std(dim=0, unbiased=False)],
        "weight_mean": float(stacked.mean()),
        "weight_min": float(stacked.min()),
        "weight_max": float(stacked.max()),
    }


def train_variant(
    *,
    split: dict[str, torch.Tensor],
    geometry: GaugeGeometry,
    lpips_model: torch.nn.Module,
    proposal_arm: str,
    lpips_weight: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
    channels: int,
    bands: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[FiberResidualSpectralFusionGate, dict]:
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(split["truth"]), generator=generator)
    train_index = permutation[:384]
    calibration_index = permutation[384:]
    model = FiberResidualSpectralFusionGate(
        channels=int(channels),
        bands=int(bands),
        maximum_mix=1.0,
        initial_mix=0.02,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    truth_calibration = split["truth"][calibration_index].to(device)
    reference_prediction = (
        split["base"][calibration_index]
        + split["reference_correction"][calibration_index]
    ).clamp(0.0, 1.0)
    reference_vectors = metric_vectors(
        reference_prediction.to(device), truth_calibration, lpips_model
    )
    reference_means = metric_means(reference_vectors)
    best = {"score": float("-inf"), "step": -1, "means": None, "audit": None}
    history = []
    variant_name = f"{proposal_arm}_lp{float(lpips_weight):g}"
    checkpoint_path = output_dir / f"checkpoint_{variant_name}.pt"

    for step in range(1, int(steps) + 1):
        selected = train_index[
            torch.randint(0, len(train_index), (int(batch_size),), generator=generator)
        ]
        base = split["base"][selected].to(device)
        reference = split["reference_correction"][selected].to(device)
        proposal = split["proposal_correction"][selected].to(device)
        anchor = split["anchor"][selected].to(device)
        truth = split["truth"][selected].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            prediction, _, weights = apply_gate(
                model, base, reference, proposal, anchor, geometry
            )
            mse = F.mse_loss(prediction, truth)
            l1 = F.l1_loss(prediction, truth)
            ssim_loss = differentiable_ssim_loss(prediction, truth)
            perceptual = lpips_model(
                prediction.repeat(1, 3, 1, 1) * 2.0 - 1.0,
                truth.repeat(1, 3, 1, 1) * 2.0 - 1.0,
            ).mean()
            loss = (
                mse
                + 0.05 * l1
                + 0.01 * ssim_loss
                + float(lpips_weight) * perceptual
                + 1.0e-5 * weights.mean()
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 100 == 0 or step == int(steps):
            prediction, _, audit = predict_gate(
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
                "audit": audit,
            }
            history.append(row)
            print(json.dumps({"variant": variant_name, **row}), flush=True)
            if score > float(best["score"]):
                best = {**row}
                torch.save(
                    {
                        "model": model.state_dict(),
                        "proposal_arm": proposal_arm,
                        "lpips_weight": float(lpips_weight),
                        "step": step,
                        "seed": int(seed),
                        "channels": int(channels),
                        "bands": int(bands),
                    },
                    checkpoint_path,
                )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, {
        "proposal_arm": proposal_arm,
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
    parser.add_argument("--control-dev", type=Path, required=True)
    parser.add_argument("--control-val", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--proposal-arm", choices=["gan", "vqae_control"], required=True)
    parser.add_argument("--lpips-weights", default="0,0.001,0.003,0.006")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--bands", type=int, default=6)
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
    control_dev = torch.load(args.control_dev, map_location="cpu", weights_only=False)
    control_val = torch.load(args.control_val, map_location="cpu", weights_only=False)
    reference_dev = prepare_split(
        primary_dev,
        control_dev,
        geometry,
        arm="vqae_control",
        batch_size=int(args.batch_size),
        device=device,
    )
    reference_val = prepare_split(
        primary_val,
        control_val,
        geometry,
        arm="vqae_control",
        batch_size=int(args.batch_size),
        device=device,
    )
    if args.proposal_arm == "gan":
        proposal_dev_source, proposal_val_source = primary_dev, primary_val
    else:
        proposal_dev_source, proposal_val_source = control_dev, control_val
    proposal_dev = prepare_split(
        primary_dev,
        proposal_dev_source,
        geometry,
        arm=args.proposal_arm,
        batch_size=int(args.batch_size),
        device=device,
    )
    proposal_val = prepare_split(
        primary_val,
        proposal_val_source,
        geometry,
        arm=args.proposal_arm,
        batch_size=int(args.batch_size),
        device=device,
    )
    for key in ("truth", "anchor", "base", "intrinsic", "source_index"):
        if not torch.equal(reference_dev[key], proposal_dev[key]):
            raise RuntimeError(f"DEV_SPLIT_MISMATCH:{key}")
        if not torch.equal(reference_val[key], proposal_val[key]):
            raise RuntimeError(f"VAL_SPLIT_MISMATCH:{key}")

    reference_model, reference_manifest = load_generator(args.reference_checkpoint, device)
    proposal_model, proposal_manifest = load_generator(args.proposal_checkpoint, device)
    dev_indices = torch.arange(len(reference_dev["truth"]))
    val_indices = torch.arange(len(reference_val["truth"]))
    _, reference_dev_correction, reference_dev_audit = predict_phase_model(
        reference_model,
        reference_dev,
        geometry,
        indices=dev_indices,
        batch_size=int(args.batch_size),
        device=device,
    )
    _, reference_val_correction, reference_val_audit = predict_phase_model(
        reference_model,
        reference_val,
        geometry,
        indices=val_indices,
        batch_size=int(args.batch_size),
        device=device,
    )
    _, proposal_dev_correction, proposal_dev_audit = predict_phase_model(
        proposal_model,
        proposal_dev,
        geometry,
        indices=dev_indices,
        batch_size=int(args.batch_size),
        device=device,
    )
    _, proposal_val_correction, proposal_val_audit = predict_phase_model(
        proposal_model,
        proposal_val,
        geometry,
        indices=val_indices,
        batch_size=int(args.batch_size),
        device=device,
    )
    dev = {
        **reference_dev,
        "reference_correction": reference_dev_correction,
        "proposal_correction": proposal_dev_correction,
    }
    val = {
        **reference_val,
        "reference_correction": reference_val_correction,
        "proposal_correction": proposal_val_correction,
    }

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)
    base_vectors = metric_vectors(val["base"].to(device), val["truth"].to(device), lpips_model)
    reference_raw = val["base"].to(device) + val["reference_correction"].to(device)
    reference_exact, reference_projection_audit = project_predictions(
        reference_raw,
        val["intrinsic"].to(device),
        geometry,
        batch_size=int(args.batch_size),
    )
    reference_vectors = metric_vectors(reference_exact, val["truth"].to(device), lpips_model)

    results = []
    for weight_index, lpips_weight in enumerate(
        float(value) for value in args.lpips_weights.split(",")
    ):
        variant_seed = int(args.seed) + 17 * weight_index
        model, training = train_variant(
            split=dev,
            geometry=geometry,
            lpips_model=lpips_model,
            proposal_arm=args.proposal_arm,
            lpips_weight=float(lpips_weight),
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            channels=int(args.channels),
            bands=int(args.bands),
            seed=variant_seed,
            device=device,
            output_dir=args.output_dir,
        )
        prediction, _, gate_audit = predict_gate(
            model,
            val,
            geometry,
            indices=val_indices,
            batch_size=int(args.batch_size),
            device=device,
        )
        exact, projection_audit = project_predictions(
            prediction.to(device),
            val["intrinsic"].to(device),
            geometry,
            batch_size=int(args.batch_size),
        )
        vectors = metric_vectors(exact, val["truth"].to(device), lpips_model)
        results.append(
            {
                "proposal_arm": args.proposal_arm,
                "lpips_weight": float(lpips_weight),
                "training": training,
                "validation_means": metric_means(vectors),
                "paired_vs_reference": paired_summary(
                    vectors,
                    reference_vectors,
                    bootstrap_reps=int(args.bootstrap_reps),
                    seed=variant_seed + 90001,
                ),
                "paired_vs_vqae": paired_summary(
                    vectors,
                    base_vectors,
                    bootstrap_reps=int(args.bootstrap_reps),
                    seed=variant_seed + 130001,
                ),
                "gate_audit": gate_audit,
                "projection_audit": projection_audit,
            }
        )
        (args.output_dir / "partial_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
        )

    payload = {
        "status": "FIBER_RESIDUAL_SPECTRAL_FUSION_GATE_PILOT",
        "proposal_arm": args.proposal_arm,
        "validation_only": True,
        "test_split_opened": False,
        "operator_sha256": geometry.info.rows_sha256,
        "dev_images": len(dev["truth"]),
        "validation_images": len(val["truth"]),
        "base_validation_means": metric_means(base_vectors),
        "reference_validation_means": metric_means(reference_vectors),
        "reference_manifest": reference_manifest,
        "proposal_manifest": proposal_manifest,
        "reference_dev_audit": reference_dev_audit,
        "reference_val_audit": reference_val_audit,
        "proposal_dev_audit": proposal_dev_audit,
        "proposal_val_audit": proposal_val_audit,
        "reference_projection_audit": reference_projection_audit,
        "results": results,
        "runtime_seconds": time.time() - started,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
