from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import gan_high_quality_gi as hq
from diagnose_factorial_moment_dithered_residual import (
    exact_project_predictions,
    simulate_counts,
)
from diagnose_fiber_residual_frequency_fusion import (
    load_generator,
    smooth_radial_high_pass,
)
from diagnose_vqgan_causal_disagreement_controls import metric_vectors, paired_summary
from src.complementary_poisson import compile_equal_reference_photon_schedule
from src.factorial_moment_dithered_residual import (
    clip_to_line_box,
    compile_dithered_phase_bank,
    estimate_factorial_moments,
    positive_part_risk_shrink,
)
from src.gauge_geometry import GaugeGeometry
from train_fiber_residual_phase_gan import predict_all
from train_vqae_centered_residual_adapter import metric_means, prepare_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-val", type=Path, required=True)
    parser.add_argument("--control-val", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--gan-checkpoint", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--rho", type=float, default=0.75)
    parser.add_argument("--photon-levels", default="1e4,1e5,1e6")
    parser.add_argument("--poisson-replicates", type=int, default=8)
    parser.add_argument("--background-fraction", type=float, default=0.01)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260718)
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

    primary = torch.load(args.primary_val, map_location="cpu", weights_only=False)
    control_source = torch.load(args.control_val, map_location="cpu", weights_only=False)
    limit = min(int(args.limit), len(primary["truth"]))
    primary = {key: value[:limit] for key, value in primary.items()}
    control_source = {key: value[:limit] for key, value in control_source.items()}
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
    indices = torch.arange(limit)
    control_model, control_manifest = load_generator(args.control_checkpoint, device)
    gan_model, gan_manifest = load_generator(args.gan_checkpoint, device)
    _, control_correction, control_model_audit = predict_all(
        control_model,
        control_split,
        geometry,
        indices=indices,
        batch_size=int(args.batch_size),
        device=device,
    )
    _, gan_correction, gan_model_audit = predict_all(
        gan_model,
        gan_split,
        geometry,
        indices=indices,
        batch_size=int(args.batch_size),
        device=device,
    )
    intrinsic = gan_split["intrinsic"].to(device)
    base_proposal = gan_split["base"].to(device) + control_correction.to(device)
    structural, structural_projection_audit = exact_project_predictions(
        base_proposal,
        intrinsic,
        geometry,
        batch_size=16,
    )
    difference = gan_correction.to(device) - control_correction.to(device)
    high_difference = smooth_radial_high_pass(
        difference, cutoff=float(args.cutoff)
    )
    fused_raw = gan_split["base"].to(device) + control_correction.to(device) + (
        float(args.alpha) * high_difference
    )
    fused, fused_projection_audit = exact_project_predictions(
        fused_raw,
        intrinsic,
        geometry,
        batch_size=16,
    )
    truth = gan_split["truth"].to(device)
    structural_flat = structural.flatten(1)
    fused_direction = geometry.null_project_flat(
        fused.flatten(1).to(torch.float64) - structural_flat.to(torch.float64)
    ).float()
    direction_norm = torch.linalg.vector_norm(fused_direction, dim=1)
    if bool((direction_norm <= 1.0e-8).any()):
        raise RuntimeError("FUSED_DIRECTION_DEGENERATE")
    phase = fused_direction / direction_norm[:, None]
    phase_bank = compile_dithered_phase_bank(
        phase,
        pairs=int(args.pairs),
        rho=float(args.rho),
        seed=int(args.seed) + 3901,
    )
    error = truth.flatten(1) - structural_flat
    true_beta = (phase * error).sum(dim=1)
    # Both endpoints are already exact members of the convex box-fiber set.
    # Restrict the physical coefficient to their convex segment.  Recomputing
    # the interval pixel-by-pixel in float32 can spuriously shorten an endpoint
    # that lies on a box face by a few ulps.
    lower = torch.zeros_like(direction_norm)
    upper = direction_norm
    fixed_beta = direction_norm
    oracle_beta = clip_to_line_box(true_beta, lower, upper)

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)
    structural_vectors_single = metric_vectors(structural, truth, lpips_model)
    fused_vectors_single = metric_vectors(fused, truth, lpips_model)
    oracle = (structural_flat + oracle_beta[:, None] * phase).reshape_as(truth)
    oracle_vectors_single = metric_vectors(oracle, truth, lpips_model)

    repeat = int(args.poisson_replicates)
    truth_repeated = truth[:, None].expand(-1, repeat, -1, -1, -1).reshape(
        limit * repeat, *truth.shape[1:]
    )
    structural_repeated = structural[:, None].expand(-1, repeat, -1, -1, -1).reshape_as(
        truth_repeated
    )
    structural_vectors = metric_vectors(structural_repeated, truth_repeated, lpips_model)
    old_rows = torch.from_numpy(rows_np).to(device=device, dtype=torch.float64)
    dummy = torch.where(
        torch.arange(geometry.n, device=device) % 2 == 0,
        torch.ones(geometry.n, device=device, dtype=torch.float64),
        -torch.ones(geometry.n, device=device, dtype=torch.float64),
    ) / math.sqrt(geometry.n)
    reference = torch.full((geometry.n,), 0.5, device=device, dtype=torch.float64)
    results = {}
    for level_index, photon_level in enumerate(
        float(value) for value in args.photon_levels.split(",")
    ):
        schedule = compile_equal_reference_photon_schedule(
            torch.cat([old_rows, dummy[None].repeat(int(args.pairs), 1)], dim=0),
            reference,
            total_signal_photons=float(rows_np.shape[0]) * float(photon_level),
            background_fraction=float(args.background_fraction),
        )
        exposure = schedule.exposure[-int(args.pairs) :]
        if not torch.allclose(exposure, exposure[:1].expand_as(exposure)):
            raise RuntimeError("DITHER_EXPOSURES_NOT_EQUAL")
        calibration = float(schedule.gain * exposure[0])
        background_each = (
            0.5
            * float(schedule.background_fraction)
            * float(schedule.signal_photons_per_pair)
        )
        bucket, shot_variance, positive_counts, negative_counts = simulate_counts(
            phase_bank.rows,
            truth,
            calibration=calibration,
            background_each=background_each,
            replicates=repeat,
            seed=int(args.seed) + 10000 * level_index + 3929,
        )
        anchor_response = torch.einsum("bkn,bn->bk", phase_bank.rows, structural_flat)
        estimate = estimate_factorial_moments(
            bucket - anchor_response[:, None],
            shot_variance,
            phase_bank.coherent_scale,
        )
        mean_only = clip_to_line_box(estimate.beta, lower, upper)
        shrunk = clip_to_line_box(
            positive_part_risk_shrink(estimate.beta, estimate.beta_variance),
            lower,
            upper,
        )
        standard_error = estimate.beta_variance.clamp_min(0.0).sqrt()
        accept = (estimate.beta.abs() > 2.131 * standard_error) & (
            estimate.beta.square() > estimate.beta_variance
        )
        gated = clip_to_line_box(
            torch.where(accept, shrunk, torch.zeros_like(shrunk)), lower, upper
        )
        coefficients = {
            "fixed_fusion": fixed_beta[:, None].expand(-1, repeat),
            "oracle_line": oracle_beta[:, None].expand(-1, repeat),
            "fm_mean_only": mean_only,
            "fm_risk_shrink": shrunk,
            "fm_significance_gate": gated,
        }
        arms = {}
        for arm_index, (name, coefficient) in enumerate(coefficients.items()):
            prediction = (
                structural_flat[:, None] + coefficient[:, :, None] * phase[:, None]
            ).reshape_as(truth_repeated)
            vectors = metric_vectors(prediction, truth_repeated, lpips_model)
            arms[name] = {
                "means": metric_means(vectors),
                "paired_vs_structural": paired_summary(
                    vectors,
                    structural_vectors,
                    bootstrap_reps=int(args.bootstrap_reps),
                    seed=int(args.seed) + 50000 * level_index + 101 * arm_index,
                ),
            }
        beta_error = estimate.beta - true_beta[:, None]
        results[f"B_{photon_level:g}"] = {
            "fixed_total_signal_photons": schedule.total_signal_photons,
            "signal_photons_per_pair": schedule.signal_photons_per_pair,
            "new_pair_calibration": calibration,
            "background_each_half": background_each,
            "arms": arms,
            "identifiability": {
                "beta_bias": float(beta_error.mean()),
                "beta_rmse": float(beta_error.square().mean().sqrt()),
                "beta_correlation": float(
                    np.corrcoef(
                        estimate.beta.detach().cpu().numpy().reshape(-1),
                        true_beta[:, None].expand(-1, repeat).cpu().numpy().reshape(-1),
                    )[0, 1]
                ),
                "significant_fraction": float(accept.float().mean()),
                "mean_positive_counts": float(positive_counts.mean()),
                "mean_negative_counts": float(negative_counts.mean()),
            },
        }

    payload = {
        "status": "FROZEN_FUSED_RESIDUAL_PHYSICAL_READOUT_SCREEN",
        "validation_only": True,
        "test_split_opened": False,
        "old_anchor_noise_simulated": False,
        "physical_readout_is_optional_to_fixed_fusion": True,
        "cutoff": float(args.cutoff),
        "alpha": float(args.alpha),
        "limit": limit,
        "pairs": int(args.pairs),
        "rho": float(args.rho),
        "poisson_replicates": repeat,
        "operator_sha256": geometry.info.rows_sha256,
        "structural_means": metric_means(structural_vectors_single),
        "fixed_fused_means": metric_means(fused_vectors_single),
        "fixed_fused_paired_vs_structural": paired_summary(
            fused_vectors_single,
            structural_vectors_single,
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed) + 70001,
        ),
        "oracle_line_means": metric_means(oracle_vectors_single),
        "oracle_line_paired_vs_structural": paired_summary(
            oracle_vectors_single,
            structural_vectors_single,
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed) + 70003,
        ),
        "phase_audit": {
            "direction_norm_mean": float(direction_norm.mean()),
            "crest_mean": float(phase_bank.crest_factor.mean()),
            "crest_max": float(phase_bank.crest_factor.max()),
            "crest_le_4_fraction": float(
                (phase_bank.crest_factor <= 4.0).float().mean()
            ),
            "old_row_component_norm_max": float(
                geometry.row_project_flat(phase).norm(dim=1).max()
            ),
            "true_beta_mean": float(true_beta.mean()),
            "true_beta_positive_fraction": float((true_beta > 0.0).float().mean()),
        },
        "control_manifest": control_manifest,
        "gan_manifest": gan_manifest,
        "control_model_audit": control_model_audit,
        "gan_model_audit": gan_model_audit,
        "structural_projection_audit": structural_projection_audit,
        "fused_projection_audit": fused_projection_audit,
        "results": results,
        "runtime_seconds": time.time() - started,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
