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
from diagnose_vqgan_causal_disagreement_controls import metric_vectors, paired_summary
from src.fiber_ratio_barycenter import oracle_simplex_weights, radial_bound
from src.gauge_geometry import GaugeGeometry, project_box_fiber_q


DEFAULT_ORIGINAL = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
DEFAULT_CACHE = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round31/"
    "vq_disagreement_bucket/cache/seed0_val.pt"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round36/afrb_proposal_headroom"
)


def normalized_uncertainty_map(
    pack: dict,
    prior: ai.PriorPack,
    refiner: ai.AnchorLatentRefiner,
    device: torch.device,
    cache_path: Path,
) -> torch.Tensor:
    if cache_path.exists():
        value = torch.load(cache_path, map_location="cpu", weights_only=True)
        if value.shape != (1, 1, 64, 64):
            raise RuntimeError("UNCERTAINTY_CACHE_SHAPE_MISMATCH")
        return value.to(device=device, dtype=torch.float32)
    for model_parameter in refiner.parameters():
        model_parameter.requires_grad_(False)
    x0 = pack["x0"][:32].float().to(device)
    target = pack["x_G"][:32].float().to(device)
    parameter = torch.full((1, 1, 64, 64), 0.85, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=0.08)
    generator = torch.Generator().manual_seed(20260718)
    for _ in range(201):
        index = torch.randperm(32, generator=generator)[:8].to(device)
        anchor = x0[index]
        with torch.no_grad():
            z0 = prior.model.encode(anchor)
        uncertainty = parameter.sigmoid().expand(8, -1, -1, -1)
        dz, delta_logits = refiner(anchor, uncertainty, z0)
        logits = ai.logits_from_latent(
            z0 + dz, prior, distance_temperature=1.0
        ) + delta_logits
        zq, _, _ = ai.quantize_from_logits(
            prior, logits, soft_temperature=1.0, straight_through=False
        )
        prediction = prior.model.decode_embeddings(zq)
        smoothness = (
            (uncertainty[:, :, 1:] - uncertainty[:, :, :-1]).abs().mean()
            + (uncertainty[:, :, :, 1:] - uncertainty[:, :, :, :-1]).abs().mean()
        )
        loss = (prediction - target[index]).abs().mean() + 0.001 * smoothness
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    value = parameter.sigmoid().detach().cpu()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, cache_path)
    return value.to(device=device)


@torch.no_grad()
def stochastic_vqgan_residuals(
    anchor: torch.Tensor,
    uncertainty: torch.Tensor,
    prior: ai.PriorPack,
    refiner: ai.AnchorLatentRefiner,
    geometry: GaugeGeometry,
    *,
    particles: int,
    token_temperature: float,
    smoothing_sigma: float,
    radius: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    generator = torch.Generator(device=anchor.device).manual_seed(int(seed))
    z0 = prior.model.encode(anchor)
    dz, delta_logits = refiner(anchor, uncertainty, z0)
    logits = ai.logits_from_latent(z0 + dz, prior, distance_temperature=1.0) + delta_logits
    probabilities = F.softmax(logits / float(token_temperature), dim=1)
    batch, codebook, height, width = probabilities.shape
    categorical = probabilities.permute(0, 2, 3, 1).reshape(-1, codebook)
    rng = np.random.default_rng(int(seed) + 1)
    residuals = []
    decode_min = float("inf")
    decode_max = float("-inf")
    for _ in range(int(particles)):
        index = torch.multinomial(categorical, 1, replacement=True, generator=generator)
        index = index.reshape(batch, height, width)
        embedding = prior.model.quantizer.lookup_indices(index)
        decoded = prior.model.decode_embeddings(embedding)
        decode_min = min(decode_min, float(decoded.min().cpu()))
        decode_max = max(decode_max, float(decoded.max().cpu()))
        alpha = torch.from_numpy(rng.beta(1.0, 7.0, size=batch).astype(np.float32)).to(
            anchor.device
        )
        centered = alpha[:, None] * (decoded - anchor).flatten(1)
        proposal = geometry.null_project_flat(centered)
        noise = torch.randn(
            proposal.shape,
            generator=generator,
            device=proposal.device,
            dtype=proposal.dtype,
        )
        proposal = proposal + float(smoothing_sigma) * geometry.null_project_flat(noise)
        residuals.append(radial_bound(proposal, float(radius)))
    result = torch.stack(residuals, dim=1)
    norms = torch.linalg.vector_norm(result, dim=2)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=1)
    return result, {
        "token_entropy_mean": float(entropy.mean().cpu()),
        "token_entropy_min": float(entropy.min().cpu()),
        "residual_norm_mean": float(norms.mean().cpu()),
        "residual_norm_max": float(norms.max().cpu()),
        "decoded_min": decode_min,
        "decoded_max": decode_max,
    }


@torch.no_grad()
def project_predictions(
    proposal: torch.Tensor,
    intrinsic: torch.Tensor,
    geometry: GaugeGeometry,
    *,
    batch_size: int,
    exact_iterations: int = 256,
) -> tuple[torch.Tensor, dict[str, float | bool | int]]:
    chunks = []
    audits = []
    for start in range(0, proposal.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), proposal.shape[0])
        result = project_box_fiber_q(
            proposal[start:stop].flatten(1),
            intrinsic[start:stop],
            geometry,
            exact=True,
            exact_iterations=int(exact_iterations),
            record_tolerance=1.0e-7,
            step_tolerance=1.0e-8,
        )
        chunks.append(result.image_flat.reshape_as(proposal[start:stop]).float())
        audits.append(result)
    return torch.cat(chunks), {
        "all_converged": bool(all(item.converged for item in audits)),
        "max_iterations": int(max(item.iterations for item in audits)),
        "max_relative_record_error": float(
            max(item.max_relative_record_error for item in audits)
        ),
        "max_box_violation": float(max(item.max_box_violation for item in audits)),
        "max_step_change": float(max(item.max_step_change for item in audits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--token-temperature", type=float, default=0.85)
    parser.add_argument("--smoothing-sigma", type=float, default=1.0 / 255.0)
    parser.add_argument("--radius", type=float, default=72.0)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    if int(args.particles) < 2:
        raise ValueError("AT_LEAST_TWO_PARTICLES_REQUIRED")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed))
    started = time.time()
    device = torch.device("cuda")

    base = args.original_root / "outputs/compatibility/measurement_conditioned_vqgan"
    config_path = base / "anchor_multiseed_hashclean_seed0/config_used.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["dataset_root"] = config["data"].get("dataset_root", "E:/datasets")
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
    anchor = pack["x0"][:limit].float().to(device).clamp(0.0, 1.0)
    anchor_flat = anchor.flatten(1)
    intrinsic = anchor_flat.to(torch.float64) @ geometry.Q.T
    target_null = geometry.null_project_flat((truth - anchor).flatten(1))

    prior = ai.load_prior(
        ai.VQGAN,
        args.original_root / config["priors"]["vqgan_checkpoint"],
        config,
        device,
    )
    refiner_path = (
        base
        / "anchor_multiseed_hashclean_seed0/runs/seed0/vqgan_refiner/"
        "checkpoints/vqgan_refiner_best_by_val_lpips.pt"
    )
    refiner = ai.load_refiner_checkpoint(refiner_path, config, device)
    uncertainty_one = normalized_uncertainty_map(
        pack,
        prior,
        refiner,
        device,
        args.output_dir / "uncertainty_map_seed0_proxy.pt",
    )

    particle_chunks = []
    proposal_audits = []
    for start in range(0, limit, int(args.batch_size)):
        stop = min(start + int(args.batch_size), limit)
        residual, audit = stochastic_vqgan_residuals(
            anchor[start:stop],
            uncertainty_one.expand(stop - start, -1, -1, -1),
            prior,
            refiner,
            geometry,
            particles=int(args.particles),
            token_temperature=float(args.token_temperature),
            smoothing_sigma=float(args.smoothing_sigma),
            radius=float(args.radius),
            seed=int(args.seed) + 1009 * start,
        )
        particle_chunks.append(residual.cpu())
        proposal_audits.append(audit)
    particles = torch.cat(particle_chunks).to(device)

    uniform_weights = torch.full(
        (limit, int(args.particles)),
        1.0 / float(args.particles),
        device=device,
    )
    oracle_weights, oracle_audit = oracle_simplex_weights(
        particles, target_null, iterations=2048, tolerance=1.0e-10
    )
    particle_mse = (particles - target_null[:, None, :]).square().mean(dim=2)
    nearest_index = particle_mse.argmin(dim=1)
    nearest_weights = F.one_hot(nearest_index, num_classes=int(args.particles)).float()

    barycenters = {
        "vqgan_uniform_barycenter": torch.einsum("bk,bkn->bn", uniform_weights, particles),
        "vqgan_oracle_nearest_particle": torch.einsum(
            "bk,bkn->bn", nearest_weights, particles
        ),
        "vqgan_oracle_simplex_barycenter": torch.einsum(
            "bk,bkn->bn", oracle_weights.float(), particles
        ),
    }
    predictions = {"box_lmmse_anchor": anchor}
    projection_audits: dict[str, dict[str, float | bool | int]] = {}
    for name, residual in barycenters.items():
        prediction, audit = project_predictions(
            (anchor_flat + residual).reshape_as(anchor),
            intrinsic,
            geometry,
            batch_size=int(args.batch_size),
        )
        predictions[name] = prediction
        projection_audits[name] = audit

    for name, key in (("deterministic_vqae", "x_A"), ("deterministic_vqgan", "x_G")):
        residual = geometry.null_project_flat(
            pack[key][:limit].float().to(device).flatten(1) - anchor_flat
        )
        prediction, audit = project_predictions(
            (anchor_flat + residual).reshape_as(anchor),
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
        name: {metric: float(value.mean()) for metric, value in per_metric.items()}
        for name, per_metric in vectors.items()
    }
    references = ("box_lmmse_anchor", "deterministic_vqae", "deterministic_vqgan")
    paired = {
        candidate: {
            reference: paired_summary(
                vectors[candidate],
                vectors[reference],
                bootstrap_reps=int(args.bootstrap_reps),
                seed=int(args.seed) + 7919 * cidx + 101 * ridx,
            )
            for ridx, reference in enumerate(references)
        }
        for cidx, candidate in enumerate(barycenters)
    }
    payload = {
        "status": "AFRB_PROPOSAL_SUPPORT_ORACLE_SCREEN",
        "validation_only": True,
        "test_split_opened": False,
        "truth_used_for_deployed_weights": False,
        "truth_used_for_oracle_headroom_diagnostics": True,
        "limit": limit,
        "particles": int(args.particles),
        "token_temperature": float(args.token_temperature),
        "alpha_law": "Beta(1,7)",
        "smoothing_sigma": float(args.smoothing_sigma),
        "radial_bound": float(args.radius),
        "uncertainty_map_status": (
            "cache-reproduction proxy; no truth or image metric used; exact LMMSE map required after GO"
        ),
        "operator_sha256": geometry.info.rows_sha256,
        "means": means,
        "paired": paired,
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
