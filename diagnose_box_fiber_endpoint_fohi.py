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
from diagnose_fiber_orthogonal_highpass_innovation import (
    distribution,
    triple_ci_favorable,
)
from diagnose_fiber_residual_frequency_fusion import (
    load_generator,
    smooth_radial_high_pass,
)
from diagnose_vqgan_causal_disagreement_controls import metric_vectors, paired_summary
from src.fiber_orthogonal_innovation import fiber_orthogonal_innovation
from src.gauge_geometry import GaugeGeometry
from train_fiber_residual_phase_gan import predict_all
from train_vqae_centered_residual_adapter import metric_means, prepare_split


def favorable_mean_signs(paired: dict[str, Any]) -> bool:
    return bool(
        paired["psnr"]["mean_delta"] > 0.0
        and paired["ssim"]["mean_delta"] > 0.0
        and paired["lpips"]["mean_delta"] < 0.0
    )


def projection_certified(audit: dict[str, Any]) -> bool:
    return bool(
        audit["all_converged"]
        and audit["max_box_violation"] == 0.0
        and audit["max_relative_record_error"] < 1.0e-7
    )


def box_violation(flat: torch.Tensor) -> torch.Tensor:
    return torch.maximum(
        torch.relu(-flat).amax(dim=1),
        torch.relu(flat - 1.0).amax(dim=1),
    )


def endpoint_faces(image: torch.Tensor, tolerance: float = 1.0e-7) -> dict[str, Any]:
    flat = image.flatten(1)
    return {
        "tolerance": float(tolerance),
        "lower_count": distribution((flat <= tolerance).sum(dim=1).float()),
        "upper_count": distribution((flat >= 1.0 - tolerance).sum(dim=1).float()),
    }


def direction_comparison(raw: torch.Tensor, endpoint: torch.Tensor) -> dict[str, Any]:
    raw_norm = raw.norm(dim=1)
    endpoint_norm = endpoint.norm(dim=1)
    cosine = (raw * endpoint).sum(dim=1) / (
        raw_norm * endpoint_norm
    ).clamp_min(1.0e-12)
    energy_ratio = endpoint.square().sum(dim=1) / raw.square().sum(dim=1).clamp_min(1.0e-12)
    return {
        "cosine": distribution(cosine),
        "endpoint_to_raw_energy_ratio": distribution(energy_ratio),
        "raw_norm": distribution(raw_norm),
        "endpoint_norm": distribution(endpoint_norm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-val", type=Path, required=True)
    parser.add_argument("--control-val", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--proposal-checkpoint", "--gan-checkpoint", dest="proposal_checkpoint", type=Path, required=True
    )
    parser.add_argument("--cutoff", type=float, default=0.12)
    parser.add_argument("--transition", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
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
    geometry = GaugeGeometry.from_rows_qr(torch.from_numpy(rows_np).to(torch.float64)).to(device)
    if geometry.info.rows_sha256 != operator_manifest["rows_sha256"]:
        raise RuntimeError("OPERATOR_IDENTITY_MISMATCH")

    primary = torch.load(args.primary_val, map_location="cpu", weights_only=False)
    control_source = torch.load(args.control_val, map_location="cpu", weights_only=False)
    gan_split = prepare_split(
        primary, primary, geometry, arm="gan", batch_size=int(args.batch_size), device=device
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
    control_prediction, control_raw, control_model_audit = predict_all(
        control_model,
        control_split,
        geometry,
        indices=indices,
        batch_size=int(args.batch_size),
        device=device,
    )
    proposal_split = gan_split if proposal_manifest["source_arm"] == "gan" else control_split
    proposal_prediction, proposal_raw, proposal_model_audit = predict_all(
        proposal_model,
        proposal_split,
        geometry,
        indices=indices,
        batch_size=int(args.batch_size),
        device=device,
    )

    intrinsic = gan_split["intrinsic"].to(device)
    base = gan_split["base"].to(device)
    truth = gan_split["truth"].to(device)
    structural, structural_audit = project_predictions(
        control_prediction.to(device),
        intrinsic,
        geometry,
        batch_size=int(args.batch_size),
        exact_iterations=1024,
    )
    proposal_endpoint, proposal_endpoint_audit = project_predictions(
        proposal_prediction.to(device),
        intrinsic,
        geometry,
        batch_size=int(args.batch_size),
        exact_iterations=1024,
    )

    # Frozen incumbent FOHI, reproduced from raw network corrections.
    raw_structural = geometry.null_project_flat(control_raw.to(device).flatten(1))
    raw_difference = geometry.null_project_flat(
        (proposal_raw.to(device) - control_raw.to(device)).flatten(1)
    ).reshape_as(base)
    raw_high = smooth_radial_high_pass(
        raw_difference, cutoff=float(args.cutoff), transition=float(args.transition)
    )
    raw_innovation = geometry.null_project_flat(raw_high.flatten(1))
    raw_orthogonal, raw_beta, raw_orthogonal_audit = fiber_orthogonal_innovation(
        raw_structural, raw_innovation
    )
    raw0_proposal = base.flatten(1) + raw_structural
    raw_fohi_proposal = raw0_proposal + float(args.alpha) * raw_orthogonal
    raw0, raw0_audit = project_predictions(
        raw0_proposal.reshape_as(base),
        intrinsic,
        geometry,
        batch_size=int(args.batch_size),
        exact_iterations=1024,
    )
    raw_fohi, raw_fohi_audit = project_predictions(
        raw_fohi_proposal.reshape_as(base),
        intrinsic,
        geometry,
        batch_size=int(args.batch_size),
        exact_iterations=1024,
    )

    # Endpoint-quotiented FOHI: only exact box-fiber endpoint chords remain.
    endpoint_structural = geometry.null_project_flat((structural - base).flatten(1))
    endpoint_difference = geometry.null_project_flat(
        (proposal_endpoint - structural).flatten(1)
    ).reshape_as(base)
    endpoint_high = smooth_radial_high_pass(
        endpoint_difference, cutoff=float(args.cutoff), transition=float(args.transition)
    )
    endpoint_innovation = geometry.null_project_flat(endpoint_high.flatten(1))
    endpoint_orthogonal, endpoint_beta, endpoint_orthogonal_audit = (
        fiber_orthogonal_innovation(endpoint_structural, endpoint_innovation)
    )
    eq_proposal = structural.flatten(1) + float(args.alpha) * endpoint_orthogonal
    eq_fohi, eq_fohi_audit = project_predictions(
        eq_proposal.reshape_as(base),
        intrinsic,
        geometry,
        batch_size=int(args.batch_size),
        exact_iterations=1024,
    )

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    vectors = {
        "structural": metric_vectors(structural, truth, lpips_model),
        "raw0": metric_vectors(raw0, truth, lpips_model),
        "raw_fohi": metric_vectors(raw_fohi, truth, lpips_model),
        "eq_fohi": metric_vectors(eq_fohi, truth, lpips_model),
    }
    raw0_vs_structural = paired_summary(
        vectors["raw0"], vectors["structural"],
        bootstrap_reps=int(args.bootstrap_reps), seed=int(args.seed) + 1,
    )
    raw_fohi_vs_structural = paired_summary(
        vectors["raw_fohi"], vectors["structural"],
        bootstrap_reps=int(args.bootstrap_reps), seed=int(args.seed) + 2,
    )
    eq_vs_structural = paired_summary(
        vectors["eq_fohi"], vectors["structural"],
        bootstrap_reps=int(args.bootstrap_reps), seed=int(args.seed) + 3,
    )
    eq_vs_raw_fohi = paired_summary(
        vectors["eq_fohi"], vectors["raw_fohi"],
        bootstrap_reps=int(args.bootstrap_reps), seed=int(args.seed) + 4,
    )

    np.savez_compressed(
        args.output_dir / "metric_vectors.npz",
        **{
            f"{arm}_{metric}": value
            for arm, arm_vectors in vectors.items()
            for metric, value in arm_vectors.items()
        },
    )
    raw_endpoint_gap = (raw0 - structural).flatten(1).norm(dim=1)
    eq_displacement = (eq_fohi.flatten(1) - eq_proposal).norm(dim=1)
    raw_displacement = (raw_fohi.flatten(1) - raw_fohi_proposal).norm(dim=1)
    projection_audits = {
        "structural": structural_audit,
        "proposal_endpoint": proposal_endpoint_audit,
        "raw_zero_anchor": raw0_audit,
        "raw_fohi": raw_fohi_audit,
        "eq_fohi": eq_fohi_audit,
    }
    payload = {
        "status": "BOX_FIBER_ENDPOINT_QUOTIENTED_FOHI_DIAGNOSTIC",
        "validation_only": True,
        "test_split_opened": False,
        "operator_sha256": geometry.info.rows_sha256,
        "validation_images": len(truth),
        "cutoff": float(args.cutoff),
        "transition": float(args.transition),
        "alpha": float(args.alpha),
        "control_manifest": control_manifest,
        "proposal_manifest": proposal_manifest,
        "control_model_audit": control_model_audit,
        "proposal_model_audit": proposal_model_audit,
        "means": {arm: metric_means(value) for arm, value in vectors.items()},
        "raw0_vs_structural": raw0_vs_structural,
        "raw_fohi_vs_structural": raw_fohi_vs_structural,
        "eq_fohi_vs_structural": eq_vs_structural,
        "eq_fohi_vs_raw_fohi": eq_vs_raw_fohi,
        "raw_fohi_triple_ci_favorable_vs_structural": triple_ci_favorable(
            raw_fohi_vs_structural
        ),
        "eq_fohi_triple_ci_favorable_vs_structural": triple_ci_favorable(
            eq_vs_structural
        ),
        "eq_fohi_triple_ci_favorable_vs_raw_fohi": triple_ci_favorable(
            eq_vs_raw_fohi
        ),
        "eq_fohi_mean_signs_favorable_vs_structural": favorable_mean_signs(
            eq_vs_structural
        ),
        "eq_fohi_mean_signs_favorable_vs_raw_fohi": favorable_mean_signs(
            eq_vs_raw_fohi
        ),
        "all_projection_certificates_pass": all(
            projection_certified(audit) for audit in projection_audits.values()
        ),
        "projection_audits": projection_audits,
        "raw_zero_anchor_to_structural_l2": distribution(raw_endpoint_gap),
        "raw_vs_endpoint_highpass_innovation": direction_comparison(
            raw_innovation, endpoint_innovation
        ),
        "structural_endpoint_faces": endpoint_faces(structural),
        "proposal_endpoint_faces": endpoint_faces(proposal_endpoint),
        "raw_removed_parallel_energy_fraction": distribution(
            raw_orthogonal_audit["parallel_energy_fraction"]
        ),
        "endpoint_removed_parallel_energy_fraction": distribution(
            endpoint_orthogonal_audit["parallel_energy_fraction"]
        ),
        "raw_relative_orthogonality_residual": distribution(
            raw_orthogonal_audit["relative_orthogonality_residual"]
        ),
        "endpoint_relative_orthogonality_residual": distribution(
            endpoint_orthogonal_audit["relative_orthogonality_residual"]
        ),
        "raw_beta_structural": distribution(raw_beta),
        "endpoint_beta_structural": distribution(endpoint_beta),
        "raw_preprojection_box_violation": distribution(box_violation(raw_fohi_proposal)),
        "eq_preprojection_box_violation": distribution(box_violation(eq_proposal)),
        "raw_final_projection_displacement": distribution(raw_displacement),
        "eq_final_projection_displacement": distribution(eq_displacement),
        "per_seed_replacement_gate_before_crossed_bootstrap": bool(
            all(projection_certified(audit) for audit in projection_audits.values())
            and favorable_mean_signs(eq_vs_structural)
            and favorable_mean_signs(eq_vs_raw_fohi)
            and triple_ci_favorable(eq_vs_structural)
        ),
        "runtime_seconds": time.time() - started,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
