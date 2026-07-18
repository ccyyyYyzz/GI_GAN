from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import gan_high_quality_gi as hq
from diagnose_afrb_proposal_headroom import project_predictions
from diagnose_fiber_residual_frequency_fusion import (
    load_generator,
    smooth_radial_high_pass,
)
from diagnose_vqgan_causal_disagreement_controls import metric_vectors, paired_summary
from src.fiber_orthogonal_innovation import fiber_orthogonal_innovation
from src.gauge_geometry import GaugeGeometry
from train_fiber_residual_phase_gan import predict_all
from train_vqae_centered_residual_adapter import metric_means, prepare_split


def distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().double().flatten()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "q05": float(torch.quantile(values, 0.05)),
        "median": float(values.median()),
        "q95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def scalar_paired_summary(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, float]:
    delta = (candidate - reference).detach().cpu().double().numpy()
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(delta), size=(int(bootstrap_reps), len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "sample_improved_fraction": float((delta < 0.0).mean()),
    }


def triple_ci_favorable(paired: dict[str, Any]) -> bool:
    return bool(
        paired["psnr"]["ci95_low"] > 0.0
        and paired["ssim"]["ci95_low"] > 0.0
        and paired["lpips"]["ci95_high"] < 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-val", type=Path, required=True)
    parser.add_argument("--control-val", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--proposal-checkpoint", "--gan-checkpoint", dest="proposal_checkpoint", type=Path, required=True
    )
    parser.add_argument("--filter-mode", choices=("highpass", "lowpass"), default="highpass")
    parser.add_argument("--cutoff", type=float, default=0.12)
    parser.add_argument("--transition", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exact-iterations", type=int, default=1024)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda")
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows_np, operator_manifest = hq.build_structured_operator_rows(
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
    if geometry.info.rows_sha256 != operator_manifest["rows_sha256"]:
        raise RuntimeError("OPERATOR_IDENTITY_MISMATCH")

    primary = torch.load(args.primary_val, map_location="cpu", weights_only=False)
    control_source = torch.load(args.control_val, map_location="cpu", weights_only=False)
    gan_split = prepare_split(
        primary,
        primary,
        geometry,
        arm="gan",
        batch_size=int(args.batch_size),
        device=device,
    )
    control_split = prepare_split(
        primary,
        control_source,
        geometry,
        arm="vqae_control",
        batch_size=int(args.batch_size),
        device=device,
    )
    for key in ("truth", "anchor", "base", "intrinsic", "source_index"):
        if not torch.equal(gan_split[key], control_split[key]):
            raise RuntimeError(f"PREPARED_SPLIT_MISMATCH:{key}")
    indices = torch.arange(len(gan_split["truth"]))
    control_model, control_manifest = load_generator(args.control_checkpoint, device)
    proposal_model, proposal_manifest = load_generator(args.proposal_checkpoint, device)
    control_prediction, control_correction, control_model_audit = predict_all(
        control_model,
        control_split,
        geometry,
        indices=indices,
        batch_size=int(args.batch_size),
        device=device,
    )
    proposal_split = (
        gan_split if proposal_manifest["source_arm"] == "gan" else control_split
    )
    _, proposal_correction, proposal_model_audit = predict_all(
        proposal_model,
        proposal_split,
        geometry,
        indices=indices,
        batch_size=int(args.batch_size),
        device=device,
    )
    structural, structural_projection_audit = project_predictions(
        control_prediction.to(device),
        control_split["intrinsic"].to(device),
        geometry,
        batch_size=int(args.batch_size),
        exact_iterations=int(args.exact_iterations),
    )

    base = gan_split["base"].to(device)
    truth = gan_split["truth"].to(device)
    structural_direction = geometry.null_project_flat(
        control_correction.to(device).flatten(1)
    )
    difference = geometry.null_project_flat(
        (proposal_correction.to(device) - control_correction.to(device)).flatten(1)
    ).reshape_as(base)
    high_difference = smooth_radial_high_pass(
        difference,
        cutoff=float(args.cutoff),
        transition=float(args.transition),
    )
    filtered_difference = (
        high_difference if args.filter_mode == "highpass" else difference - high_difference
    )
    innovation = geometry.null_project_flat(filtered_difference.flatten(1))
    orthogonal, beta, orthogonal_audit = fiber_orthogonal_innovation(
        structural_direction, innovation
    )
    parallel_fraction = orthogonal_audit["parallel_energy_fraction"]
    orthogonality = orthogonal_audit["relative_orthogonality_residual"]

    fixed_proposal = base.flatten(1) + structural_direction + float(args.alpha) * innovation
    fohi_proposal = base.flatten(1) + structural_direction + float(args.alpha) * orthogonal
    fixed, fixed_projection_audit = project_predictions(
        fixed_proposal.reshape_as(base),
        gan_split["intrinsic"].to(device),
        geometry,
        batch_size=int(args.batch_size),
        exact_iterations=int(args.exact_iterations),
    )
    fohi, fohi_projection_audit = project_predictions(
        fohi_proposal.reshape_as(base),
        gan_split["intrinsic"].to(device),
        geometry,
        batch_size=int(args.batch_size),
        exact_iterations=int(args.exact_iterations),
    )

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    structural_vectors = metric_vectors(structural, truth, lpips_model)
    fixed_vectors = metric_vectors(fixed, truth, lpips_model)
    fohi_vectors = metric_vectors(fohi, truth, lpips_model)
    fixed_vs_structural = paired_summary(
        fixed_vectors,
        structural_vectors,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 1,
    )
    fohi_vs_structural = paired_summary(
        fohi_vectors,
        structural_vectors,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 2,
    )
    fohi_vs_fixed = paired_summary(
        fohi_vectors,
        fixed_vectors,
        bootstrap_reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 3,
    )
    np.savez_compressed(
        args.output_dir / "metric_vectors.npz",
        **{
            f"{arm}_{metric}": values
            for arm, vectors in {
                "structural": structural_vectors,
                "fixed": fixed_vectors,
                "fohi": fohi_vectors,
            }.items()
            for metric, values in vectors.items()
        },
    )
    structural_mse = (structural - truth).square().flatten(1).mean(dim=1)
    fixed_mse = (fixed - truth).square().flatten(1).mean(dim=1)
    fohi_mse = (fohi - truth).square().flatten(1).mean(dim=1)

    missing = truth.flatten(1) - base.flatten(1) - structural_direction
    alignment = (missing * orthogonal).sum(dim=1)
    energy = orthogonal.square().sum(dim=1)
    positive_interval = 2.0 * alignment / energy.clamp_min(1.0e-12)
    alpha_inside = (alignment > 0.0) & (float(args.alpha) < positive_interval)
    fixed_box_violation = torch.maximum(
        torch.relu(-fixed_proposal).amax(dim=1),
        torch.relu(fixed_proposal - 1.0).amax(dim=1),
    )
    fohi_box_violation = torch.maximum(
        torch.relu(-fohi_proposal).amax(dim=1),
        torch.relu(fohi_proposal - 1.0).amax(dim=1),
    )

    payload = {
        "status": "FIBER_ORTHOGONAL_HIGHPASS_INNOVATION_DIAGNOSTIC",
        "validation_only": True,
        "test_split_opened": False,
        "operator_sha256": geometry.info.rows_sha256,
        "validation_images": len(truth),
        "cutoff": float(args.cutoff),
        "transition": float(args.transition),
        "alpha": float(args.alpha),
        "filter_mode": str(args.filter_mode),
        "exact_iterations": int(args.exact_iterations),
        "control_manifest": control_manifest,
        "proposal_manifest": proposal_manifest,
        "gan_manifest": proposal_manifest,
        "control_model_audit": control_model_audit,
        "proposal_model_audit": proposal_model_audit,
        "gan_model_audit": proposal_model_audit,
        "structural_means": metric_means(structural_vectors),
        "fixed_means": metric_means(fixed_vectors),
        "fohi_means": metric_means(fohi_vectors),
        "fixed_vs_structural": fixed_vs_structural,
        "fohi_vs_structural": fohi_vs_structural,
        "fohi_vs_fixed": fohi_vs_fixed,
        "fixed_mse_vs_structural": scalar_paired_summary(
            fixed_mse,
            structural_mse,
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed) + 4,
        ),
        "fohi_mse_vs_structural": scalar_paired_summary(
            fohi_mse,
            structural_mse,
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed) + 5,
        ),
        "fohi_mse_vs_fixed": scalar_paired_summary(
            fohi_mse,
            fixed_mse,
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed) + 6,
        ),
        "fohi_triple_ci_favorable_vs_structural": triple_ci_favorable(
            fohi_vs_structural
        ),
        "fixed_triple_ci_favorable_vs_structural": triple_ci_favorable(
            fixed_vs_structural
        ),
        "fohi_triple_ci_favorable_vs_fixed": triple_ci_favorable(fohi_vs_fixed),
        "parallel_energy_fraction": distribution(parallel_fraction),
        "parallel_energy_mean_at_least_one_percent": bool(
            parallel_fraction.mean() >= 0.01
        ),
        "beta_structural": distribution(beta),
        "innovation_norm": distribution(innovation.norm(dim=1)),
        "orthogonal_innovation_norm": distribution(orthogonal.norm(dim=1)),
        "relative_orthogonality_residual": distribution(orthogonality),
        "oracle_alignment": distribution(alignment),
        "frozen_alpha_inside_mse_improvement_interval_fraction": float(
            alpha_inside.float().mean()
        ),
        "fixed_preprojection_box_violation": distribution(fixed_box_violation),
        "fohi_preprojection_box_violation": distribution(fohi_box_violation),
        "structural_projection_audit": structural_projection_audit,
        "fixed_projection_audit": fixed_projection_audit,
        "fohi_projection_audit": fohi_projection_audit,
        "runtime_seconds": time.time() - started,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
