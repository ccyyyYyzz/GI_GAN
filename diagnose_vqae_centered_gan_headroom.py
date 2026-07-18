from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import anchor_initialized_vqgan_inversion as ai
import gan_high_quality_gi as hq
from diagnose_afrb_proposal_headroom import (
    normalized_uncertainty_map,
    project_predictions,
    stochastic_vqgan_residuals,
)
from diagnose_vqgan_causal_disagreement_controls import metric_vectors, paired_summary
from src.fiber_ratio_barycenter import oracle_simplex_weights
from src.gauge_geometry import GaugeGeometry


DEFAULT_ORIGINAL = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
DEFAULT_CACHE = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round31/"
    "vq_disagreement_bucket/cache/seed0_val.pt"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round37/"
    "vqae_centered_gan_headroom"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--particles", type=int, default=8)
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

    base_root = args.original_root / "outputs/compatibility/measurement_conditioned_vqgan"
    config = yaml.safe_load(
        (base_root / "anchor_multiseed_hashclean_seed0/config_used.yaml").read_text(
            encoding="utf-8"
        )
    )
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
    limit = min(int(args.limit), int(pack["truth"].shape[0]))
    truth = pack["truth"][:limit].float().to(device)
    lmmse = pack["x0"][:limit].float().to(device).clamp(0.0, 1.0)
    intrinsic = lmmse.flatten(1).to(torch.float64) @ geometry.Q.T

    vqae_residual = geometry.null_project_flat(
        pack["x_A"][:limit].float().to(device).flatten(1) - lmmse.flatten(1)
    )
    vqae, vqae_audit = project_predictions(
        (lmmse.flatten(1) + vqae_residual).reshape_as(lmmse),
        intrinsic,
        geometry,
        batch_size=int(args.batch_size),
    )
    vqgan_residual = geometry.null_project_flat(
        pack["x_G"][:limit].float().to(device).flatten(1) - lmmse.flatten(1)
    )
    deterministic_vqgan, vqgan_audit = project_predictions(
        (lmmse.flatten(1) + vqgan_residual).reshape_as(lmmse),
        intrinsic,
        geometry,
        batch_size=int(args.batch_size),
    )

    prior = ai.load_prior(
        ai.VQGAN,
        args.original_root / config["priors"]["vqgan_checkpoint"],
        config,
        device,
    )
    refiner = ai.load_refiner_checkpoint(
        base_root
        / "anchor_multiseed_hashclean_seed0/runs/seed0/vqgan_refiner/"
        "checkpoints/vqgan_refiner_best_by_val_lpips.pt",
        config,
        device,
    )
    uncertainty = normalized_uncertainty_map(
        pack,
        prior,
        refiner,
        device,
        Path(
            "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round36/"
            "afrb_proposal_headroom/uncertainty_map_seed0_proxy.pt"
        ),
    )

    particle_chunks = []
    proposal_audits = []
    for start in range(0, limit, int(args.batch_size)):
        stop = min(start + int(args.batch_size), limit)
        residual, audit = stochastic_vqgan_residuals(
            vqae[start:stop],
            uncertainty.expand(stop - start, -1, -1, -1),
            prior,
            refiner,
            geometry,
            particles=int(args.particles),
            token_temperature=0.85,
            smoothing_sigma=1.0 / 255.0,
            radius=72.0,
            seed=int(args.seed) + 2017 * start,
        )
        particle_chunks.append(residual.cpu())
        proposal_audits.append(audit)
    particles = torch.cat(particle_chunks).to(device)
    target_null = geometry.null_project_flat((truth - vqae).flatten(1))
    oracle_weights, oracle_audit = oracle_simplex_weights(
        particles, target_null, iterations=2048, tolerance=1.0e-10
    )
    uniform_weights = torch.full(
        (limit, int(args.particles)),
        1.0 / float(args.particles),
        device=device,
    )
    nearest = F.one_hot(
        (particles - target_null[:, None]).square().mean(dim=2).argmin(dim=1),
        num_classes=int(args.particles),
    ).float()

    direct = geometry.null_project_flat(
        deterministic_vqgan.flatten(1) - vqae.flatten(1)
    )
    direct_denominator = direct.square().sum(dim=1).clamp_min(1.0e-12)
    direct_beta = (
        (direct * target_null).sum(dim=1) / direct_denominator
    ).clamp(0.0, 1.0)
    residuals = {
        "vqae_centered_uniform_particles": torch.einsum(
            "bk,bkn->bn", uniform_weights, particles
        ),
        "vqae_centered_oracle_nearest": torch.einsum(
            "bk,bkn->bn", nearest, particles
        ),
        "vqae_centered_oracle_simplex": torch.einsum(
            "bk,bkn->bn", oracle_weights.float(), particles
        ),
        "vqae_to_vqgan_oracle_segment": direct_beta[:, None] * direct,
    }
    predictions = {
        "box_lmmse_anchor": lmmse,
        "deterministic_vqae": vqae,
        "deterministic_vqgan": deterministic_vqgan,
    }
    projection_audits = {
        "deterministic_vqae": vqae_audit,
        "deterministic_vqgan": vqgan_audit,
    }
    for name, residual in residuals.items():
        prediction, audit = project_predictions(
            (vqae.flatten(1) + residual).reshape_as(vqae),
            intrinsic,
            geometry,
            batch_size=int(args.batch_size),
        )
        predictions[name] = prediction
        projection_audits[name] = audit

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    vectors = {
        name: metric_vectors(prediction, truth, lpips_model)
        for name, prediction in predictions.items()
    }
    means = {
        name: {metric: float(value.mean()) for metric, value in metrics.items()}
        for name, metrics in vectors.items()
    }
    paired_vs_vqae = {
        name: paired_summary(
            vectors[name],
            vectors["deterministic_vqae"],
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed) + 3571 * index,
        )
        for index, name in enumerate(residuals)
    }
    payload = {
        "status": "VQAE_CENTERED_GAN_PROPOSAL_HEADROOM",
        "validation_only": True,
        "test_split_opened": False,
        "truth_used_only_for_oracle_headroom_and_metrics": True,
        "proposal_refiner_was_not_yet_finetuned_on_vqae_inputs": True,
        "limit": limit,
        "particles": int(args.particles),
        "operator_sha256": geometry.info.rows_sha256,
        "means": means,
        "paired_vs_deterministic_vqae": paired_vs_vqae,
        "direct_segment_beta": {
            "mean": float(direct_beta.mean().cpu()),
            "median": float(direct_beta.median().cpu()),
            "positive_fraction": float((direct_beta > 0.0).float().mean().cpu()),
            "interior_fraction": float(
                ((direct_beta > 0.0) & (direct_beta < 1.0)).float().mean().cpu()
            ),
        },
        "oracle_weight_audit": oracle_audit,
        "proposal_audit": {
            key: float(np.mean([item[key] for item in proposal_audits]))
            for key in proposal_audits[0]
        },
        "projection_audits": projection_audits,
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"summary_val_{limit}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
