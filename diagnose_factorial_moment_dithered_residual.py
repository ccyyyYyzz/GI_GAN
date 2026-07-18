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
from diagnose_vqgan_causal_disagreement_controls import metric_vectors, paired_summary
from src.complementary_poisson import compile_equal_reference_photon_schedule
from src.factorial_moment_dithered_residual import (
    clip_to_line_box,
    compile_dithered_phase_bank,
    estimate_factorial_moments,
    line_box_interval,
    positive_part_risk_shrink,
)
from src.gauge_geometry import GaugeGeometry, project_box_fiber_exact_dual


DEFAULT_ORIGINAL = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
DEFAULT_CACHE = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round31/"
    "vq_disagreement_bucket/cache/seed0_val.pt"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round39/"
    "factorial_moment_dithered_residual"
)


def _means(vectors: dict[str, np.ndarray]) -> dict[str, float]:
    return {metric: float(value.mean()) for metric, value in vectors.items()}


@torch.no_grad()
def exact_project_predictions(
    proposal: torch.Tensor,
    intrinsic: torch.Tensor,
    geometry: GaugeGeometry,
    *,
    batch_size: int = 16,
) -> tuple[torch.Tensor, dict[str, float | bool | int]]:
    images = []
    audits = []
    for start in range(0, proposal.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), proposal.shape[0])
        result = project_box_fiber_exact_dual(
            proposal[start:stop].flatten(1), intrinsic[start:stop], geometry
        )
        images.append(result.image_flat.reshape_as(proposal[start:stop]).float())
        audits.append(result)
    return torch.cat(images), {
        "all_converged": bool(all(item.converged for item in audits)),
        "max_iterations": int(max(item.iterations for item in audits)),
        "max_relative_record_error": float(
            max(item.max_relative_record_error for item in audits)
        ),
        "max_box_violation": float(max(item.max_box_violation for item in audits)),
        "max_complementarity_residual": float(
            max(item.max_complementarity_residual for item in audits)
        ),
    }


@torch.no_grad()
def simulate_counts(
    rows: torch.Tensor,
    truth: torch.Tensor,
    *,
    calibration: float,
    background_each: float,
    replicates: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = rows.clamp_min(0.0)
    negative = (-rows).clamp_min(0.0)
    truth_flat = truth.flatten(1)
    positive_rate = float(calibration) * torch.einsum(
        "bkn,bn->bk", positive, truth_flat
    ) + float(background_each)
    negative_rate = float(calibration) * torch.einsum(
        "bkn,bn->bk", negative, truth_flat
    ) + float(background_each)
    generator = torch.Generator(device=truth.device).manual_seed(int(seed))
    shape = (truth.shape[0], int(replicates), rows.shape[1])
    positive_counts = torch.poisson(
        positive_rate[:, None].expand(shape), generator=generator
    )
    negative_counts = torch.poisson(
        negative_rate[:, None].expand(shape), generator=generator
    )
    bucket = (positive_counts - negative_counts) / float(calibration)
    shot_variance = (positive_counts + negative_counts) / float(calibration) ** 2
    return bucket, shot_variance, positive_counts, negative_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--rho", type=float, default=0.75)
    parser.add_argument("--photon-levels", default="1e4,1e5")
    parser.add_argument("--poisson-replicates", type=int, default=8)
    parser.add_argument("--background-fraction", type=float, default=0.01)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda")
    started = time.time()

    config_path = (
        args.original_root
        / "outputs/compatibility/measurement_conditioned_vqgan/"
        "anchor_multiseed_hashclean_seed0/config_used.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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
    pack = torch.load(args.cache, map_location="cpu", weights_only=False)
    limit = min(int(args.limit), len(pack["truth"]))
    truth = pack["truth"][:limit].float().to(device)
    anchor = pack["x0"][:limit].float().to(device).clamp(0.0, 1.0)
    intrinsic = anchor.flatten(1).to(torch.float64) @ geometry.Q.T
    vqae_residual = geometry.null_project_flat(
        pack["x_A"][:limit].float().to(device).flatten(1) - anchor.flatten(1)
    )
    base, base_projection_audit = exact_project_predictions(
        (anchor.flatten(1) + vqae_residual).reshape_as(anchor),
        intrinsic,
        geometry,
        batch_size=16,
    )
    vqgan_residual = geometry.null_project_flat(
        pack["x_G"][:limit].float().to(device).flatten(1) - anchor.flatten(1)
    )
    vqgan, vqgan_projection_audit = exact_project_predictions(
        (anchor.flatten(1) + vqgan_residual).reshape_as(anchor),
        intrinsic,
        geometry,
        batch_size=16,
    )
    direction = vqgan.flatten(1) - base.flatten(1)
    direction_norm = torch.linalg.vector_norm(direction, dim=1)
    phase = direction / direction_norm.clamp_min(1.0e-12)[:, None]
    phase_bank = compile_dithered_phase_bank(
        phase,
        pairs=int(args.pairs),
        rho=float(args.rho),
        seed=int(args.seed) + 3901,
    )
    base_flat = base.flatten(1)
    truth_flat = truth.flatten(1)
    error = truth_flat - base_flat
    true_beta = (phase * error).sum(dim=1)
    maximum = phase.abs().amax(dim=1)
    true_nuisance = (
        (1.0 - float(args.rho) ** 2 * phase.square() / maximum[:, None].square())
        * error.square()
    ).sum(dim=1) / geometry.n
    line_lower, line_upper = line_box_interval(base_flat, phase)
    oracle_beta = clip_to_line_box(true_beta, line_lower, line_upper)
    fixed_beta = clip_to_line_box(0.10 * direction_norm, line_lower, line_upper)

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    base_vectors_single = metric_vectors(base, truth, lpips_model)
    results: dict[str, dict] = {}
    old_rows = torch.from_numpy(rows_np).to(device=device, dtype=torch.float64)
    dummy = torch.where(
        torch.arange(geometry.n, device=device) % 2 == 0,
        torch.ones(geometry.n, device=device, dtype=torch.float64),
        -torch.ones(geometry.n, device=device, dtype=torch.float64),
    ) / math.sqrt(geometry.n)
    reference = torch.full(
        (geometry.n,), 0.5, device=device, dtype=torch.float64
    )
    repeat = int(args.poisson_replicates)
    truth_repeated = truth[:, None].expand(-1, repeat, -1, -1, -1).reshape(
        limit * repeat, *truth.shape[1:]
    )
    base_repeated = base[:, None].expand(-1, repeat, -1, -1, -1).reshape_as(
        truth_repeated
    )
    base_vectors = metric_vectors(base_repeated, truth_repeated, lpips_model)

    for level_index, photon_level in enumerate(
        float(value) for value in args.photon_levels.split(",")
    ):
        schedule = compile_equal_reference_photon_schedule(
            torch.cat([old_rows, dummy[None].repeat(int(args.pairs), 1)], dim=0),
            reference,
            total_signal_photons=float(rows_np.shape[0]) * float(photon_level),
            background_fraction=float(args.background_fraction),
        )
        new_exposure = schedule.exposure[-int(args.pairs) :]
        if not torch.allclose(new_exposure, new_exposure[:1].expand_as(new_exposure)):
            raise RuntimeError("DITHER_EXPOSURES_NOT_EQUAL")
        calibration = float(schedule.gain * new_exposure[0])
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
        anchor_response = torch.einsum("bkn,bn->bk", phase_bank.rows, base_flat)
        centered = bucket - anchor_response[:, None]
        estimate = estimate_factorial_moments(
            centered, shot_variance, phase_bank.coherent_scale
        )
        mean_only = clip_to_line_box(estimate.beta, line_lower, line_upper)
        shrunk = positive_part_risk_shrink(estimate.beta, estimate.beta_variance)
        shrunk = clip_to_line_box(shrunk, line_lower, line_upper)
        standard_error = estimate.beta_variance.clamp_min(0.0).sqrt()
        significant = estimate.beta.abs() > 2.131 * standard_error
        energy_positive = estimate.beta.square() > estimate.beta_variance
        crest_ok = phase_bank.crest_factor[:, None] <= 4.0
        variance_upper = estimate.nuisance_variance.clamp_min(0.0) + (
            2.326
            * math.sqrt(2.0 / max(int(args.pairs) - 1, 1))
            * estimate.sample_variance
        )
        energy_cap = torch.sqrt(
            geometry.n
            * variance_upper.clamp_min(0.0)
            / (1.0 - float(args.rho) ** 2)
        )
        gated = torch.maximum(torch.minimum(shrunk, energy_cap), -energy_cap)
        accept = significant & energy_positive & crest_ok
        gated = torch.where(accept, gated, torch.zeros_like(gated))
        gated = clip_to_line_box(gated, line_lower, line_upper)

        coefficients = {
            "oracle_phase": oracle_beta[:, None].expand(-1, repeat),
            "fixed_beta0p1": fixed_beta[:, None].expand(-1, repeat),
            "fm_mean_only": mean_only,
            "fm_risk_shrink": shrunk,
            "fm_approx_gate": gated,
        }
        arm_results = {}
        for arm_index, (name, coefficient) in enumerate(coefficients.items()):
            prediction = (
                base_flat[:, None] + coefficient[:, :, None] * phase[:, None]
            ).reshape_as(truth_repeated)
            vectors = metric_vectors(prediction, truth_repeated, lpips_model)
            arm_results[name] = {
                "means": _means(vectors),
                "paired_vs_vqae": paired_summary(
                    vectors,
                    base_vectors,
                    bootstrap_reps=int(args.bootstrap_reps),
                    seed=int(args.seed) + 50000 * level_index + 100 * arm_index,
                ),
            }
        beta_error = estimate.beta - true_beta[:, None]
        nuisance_denominator = true_nuisance[:, None].abs().clamp_min(1.0e-8)
        results[f"B_{photon_level:g}"] = {
            "fixed_total_signal_photons": schedule.total_signal_photons,
            "signal_photons_per_pair": schedule.signal_photons_per_pair,
            "new_pair_calibration": calibration,
            "background_each_half": background_each,
            "arms": arm_results,
            "identifiability": {
                "beta_bias": float(beta_error.mean().cpu()),
                "beta_rmse": float(beta_error.square().mean().sqrt().cpu()),
                "beta_correlation": float(
                    np.corrcoef(
                        estimate.beta.detach().cpu().numpy().reshape(-1),
                        true_beta[:, None].expand(-1, repeat).cpu().numpy().reshape(-1),
                    )[0, 1]
                ),
                "nuisance_relative_bias_mean": float(
                    ((estimate.nuisance_variance - true_nuisance[:, None]) / nuisance_denominator)
                    .mean()
                    .cpu()
                ),
                "significant_fraction": float(significant.float().mean().cpu()),
                "energy_positive_fraction": float(energy_positive.float().mean().cpu()),
                "accept_fraction": float(accept.float().mean().cpu()),
                "abstention_fraction": float((~accept).float().mean().cpu()),
                "mean_positive_counts": float(positive_counts.mean().cpu()),
                "mean_negative_counts": float(negative_counts.mean().cpu()),
            },
        }

    payload = {
        "status": "FM_DRP_FROZEN_PHASE_PHYSICAL_SCREEN",
        "validation_only": True,
        "test_split_opened": False,
        "phase_source": "normalized old-fiber deterministic VQGAN minus VQAE direction",
        "no_new_phase_network_trained": True,
        "old_anchor_noise_simulated": False,
        "likelihood_and_bootstrap_gates_implemented": False,
        "screen_interpretation": (
            "favorable-to-method physical identifiability screen; failure kills, success only permits full pilot"
        ),
        "limit": limit,
        "pairs": int(args.pairs),
        "half_exposures": 2 * (rows_np.shape[0] + int(args.pairs)),
        "rho": float(args.rho),
        "poisson_replicates": repeat,
        "operator_sha256": geometry.info.rows_sha256,
        "base_means": _means(base_vectors_single),
        "phase_audit": {
            "direction_norm_mean": float(direction_norm.mean().cpu()),
            "direction_norm_min": float(direction_norm.min().cpu()),
            "crest_mean": float(phase_bank.crest_factor.mean().cpu()),
            "crest_max": float(phase_bank.crest_factor.max().cpu()),
            "crest_pass_fraction": float(
                (phase_bank.crest_factor <= 4.0).float().mean().cpu()
            ),
            "dc_sum_abs_max": float(phase.sum(dim=1).abs().max().cpu()),
            "old_row_component_norm_max": float(
                geometry.row_project_flat(phase).norm(dim=1).max().cpu()
            ),
            "true_beta_mean": float(true_beta.mean().cpu()),
            "true_beta_positive_fraction": float((true_beta > 0).float().mean().cpu()),
            "true_nuisance_mean": float(true_nuisance.mean().cpu()),
        },
        "base_projection_audit": base_projection_audit,
        "vqgan_projection_audit": vqgan_projection_audit,
        "results": results,
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"summary_val_{limit}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
