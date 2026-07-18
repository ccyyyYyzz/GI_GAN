from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import anchor_initialized_vqgan_inversion as ai
import gan_high_quality_gi as hq
from diagnose_active_binary_query_headroom import (
    augmented_box_projection,
    normalized_balanced_binary,
)
from diagnose_vqgan_causal_disagreement_controls import (
    metric_vectors,
    paired_summary,
)
from src.dc_balanced import hadamard_lowsequency_non_dc_rows
from src.gauge_geometry import GaugeGeometry


DEFAULT_ORIGINAL = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
DEFAULT_BASE_CACHE = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round31/"
    "vq_disagreement_bucket/cache/seed0_val.pt"
)
DEFAULT_CONTROL_CACHE = Path(
    "E:/ns_mc_gan_gi_code_fcc_phase1/outputs/compatibility/"
    "measurement_conditioned_vqgan/detail_fusion/cache/seed2_val.pt"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round33/"
    "vq_disagreement_poisson_bucket"
)


def repeat_images(value: torch.Tensor, replicates: int) -> torch.Tensor:
    return value[:, None].expand(-1, int(replicates), *value.shape[1:]).reshape(
        value.shape[0] * int(replicates), *value.shape[1:]
    )


def repeat_flat(value: torch.Tensor, replicates: int) -> torch.Tensor:
    return value[:, None].expand(-1, int(replicates), value.shape[1]).reshape(
        value.shape[0] * int(replicates), value.shape[1]
    )


def cluster_vectors(
    vectors: dict[str, np.ndarray],
    *,
    images: int,
    replicates: int,
) -> dict[str, np.ndarray]:
    return {
        metric: values.reshape(int(images), int(replicates)).mean(axis=1)
        for metric, values in vectors.items()
    }


@torch.no_grad()
def complementary_poisson_record(
    query: torch.Tensor,
    truth: torch.Tensor,
    flux_estimate: torch.Tensor,
    *,
    signal_photons: float,
    background_fraction: float,
    replicates: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch, pixels = query.shape
    positive = query > 0.0
    positive_count = positive.sum(dim=1)
    if not bool((positive_count == pixels // 2).all()):
        raise RuntimeError("QUERY_NOT_HALF_ON_HALF_OFF")
    flat = truth.flatten(1)
    exposure = float(signal_photons) / flux_estimate.clamp_min(1.0e-8)
    rate_positive = exposure * (positive.to(flat.dtype) * flat).sum(dim=1)
    rate_negative = exposure * ((~positive).to(flat.dtype) * flat).sum(dim=1)
    background_each = 0.5 * float(background_fraction) * float(signal_photons)
    rate_positive = rate_positive + background_each
    rate_negative = rate_negative + background_each
    generator = torch.Generator(device=truth.device)
    generator.manual_seed(int(seed))
    count_positive = torch.poisson(
        rate_positive[:, None].expand(-1, int(replicates)), generator=generator
    )
    count_negative = torch.poisson(
        rate_negative[:, None].expand(-1, int(replicates)), generator=generator
    )
    record = (count_positive - count_negative) / (
        exposure[:, None] * math.sqrt(pixels)
    )
    true_record = (query * flat).sum(dim=1)
    noise = record - true_record[:, None]
    return record.reshape(batch * int(replicates)), {
        "positive_pixels": float(positive_count.float().mean().cpu()),
        "requested_signal_photons": float(signal_photons),
        "expected_signal_photons_mean": float(
            ((rate_positive + rate_negative) - 2.0 * background_each).mean().cpu()
        ),
        "expected_background_photons_per_pair": float(2.0 * background_each),
        "observed_total_counts_mean": float(
            (count_positive + count_negative).float().mean().cpu()
        ),
        "record_noise_rmse": float(noise.square().mean().sqrt().cpu()),
        "record_noise_mean": float(noise.mean().cpu()),
    }


@torch.no_grad()
def run_pair(
    base: torch.Tensor,
    alternative: torch.Tensor,
    truth: torch.Tensor,
    intrinsic: torch.Tensor,
    query: torch.Tensor,
    noisy_record: torch.Tensor,
    geometry: GaugeGeometry,
    *,
    replicates: int,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    base_r = repeat_images(base, replicates)
    alternative_r = repeat_images(alternative, replicates)
    query_r = repeat_flat(query, replicates)
    intrinsic_r = repeat_flat(intrinsic, replicates)
    center = 0.5 * (base_r + alternative_r)
    half = 0.5 * (alternative_r - base_r)
    denominator = (query_r * half.flatten(1)).sum(dim=1)
    safe_denominator = torch.where(
        denominator.abs() >= 1.0e-8,
        denominator,
        torch.where(
            denominator >= 0.0,
            torch.full_like(denominator, 1.0e-8),
            torch.full_like(denominator, -1.0e-8),
        ),
    )
    raw_coordinate = (
        noisy_record - (query_r * center.flatten(1)).sum(dim=1)
    ) / safe_denominator
    coordinate = raw_coordinate.clamp(-1.0, 1.0)
    proposal = center + coordinate[:, None, None, None] * half
    prediction, audit = augmented_box_projection(
        proposal,
        intrinsic_r,
        query_r,
        noisy_record,
        geometry,
        iterations=int(iterations),
    )
    absolute_bucket_residual = (
        (prediction.flatten(1) * query_r).sum(dim=1) - noisy_record
    ).abs()
    audit.update(
        {
            "coordinate_clipped_fraction": float(
                (raw_coordinate.abs() > 1.0).float().mean().cpu()
            ),
            "coordinate_mean": float(coordinate.mean().cpu()),
            "absolute_bucket_residual_max": float(
                absolute_bucket_residual.max().cpu()
            ),
        }
    )
    return prediction, audit


@torch.no_grad()
def run_fixed(
    base: torch.Tensor,
    intrinsic: torch.Tensor,
    query: torch.Tensor,
    noisy_record: torch.Tensor,
    geometry: GaugeGeometry,
    *,
    replicates: int,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    base_r = repeat_images(base, replicates)
    query_r = repeat_flat(query, replicates)
    intrinsic_r = repeat_flat(intrinsic, replicates)
    prediction, audit = augmented_box_projection(
        base_r,
        intrinsic_r,
        query_r,
        noisy_record,
        geometry,
        iterations=int(iterations),
    )
    absolute_bucket_residual = (
        (prediction.flatten(1) * query_r).sum(dim=1) - noisy_record
    ).abs()
    audit["absolute_bucket_residual_max"] = float(
        absolute_bucket_residual.max().cpu()
    )
    return prediction, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--control-cache", type=Path, default=DEFAULT_CONTROL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2048)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--signal-photons", type=float, nargs="+", default=[1.0e4, 1.0e5])
    parser.add_argument("--background-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    started = time.time()
    device = torch.device("cuda")

    base_root = (
        args.original_root
        / "outputs/compatibility/measurement_conditioned_vqgan"
    )
    config = yaml.safe_load(
        (base_root / "anchor_multiseed_hashclean_seed0/config_used.yaml").read_text(
            encoding="utf-8"
        )
    )
    operator = config["operator"]
    size = int(config["data"]["img_size"])
    rows_np, manifest = hq.build_structured_operator_rows(
        img_size=size,
        total_m=int(operator["total_m"]),
        dct_rows=int(operator["dct_rows"]),
        hadamard_rows=int(operator["hadamard_rows"]),
        random_rows=int(operator["random_rows"]),
        seed=int(operator["seed"]),
    )
    geometry = GaugeGeometry(torch.from_numpy(rows_np).to(torch.float64)).to(device)
    if geometry.info.rows_sha256 != manifest["rows_sha256"]:
        raise RuntimeError("OPERATOR_IDENTITY_MISMATCH")
    if not np.allclose(rows_np[0], 1.0 / math.sqrt(size * size), atol=1.0e-7):
        raise RuntimeError("FIRST_ROW_NOT_DC_FLUX_CHANNEL")
    measurement = hq.make_measurement_operator(
        rows_np,
        img_size=size,
        device=device,
        lambda_solver=float(operator["lambda_solver"]),
    )

    pack0 = torch.load(args.base_cache, map_location="cpu")
    pack2 = torch.load(args.control_cache, map_location="cpu")
    for field in ("source_index", "truth", "y", "x0"):
        if not torch.equal(pack0[field], pack2[field]):
            raise RuntimeError(f"CONTROL_CACHE_IDENTITY_MISMATCH:{field}")
    limit = min(int(args.limit), int(pack0["truth"].shape[0]))
    truth = pack0["truth"][:limit].float().to(device)
    x0 = pack0["x0"][:limit].float().to(device)
    y = pack0["y"][:limit].float().to(device)
    base = ai.null_blend(
        x0, pack0["x_A"][:limit].float().to(device), 1.0, measurement
    ).clamp(0.0, 1.0)
    vqgan = ai.null_blend(
        x0, pack0["x_G"][:limit].float().to(device), 1.0, measurement
    ).clamp(0.0, 1.0)
    vqae_control = ai.null_blend(
        x0, pack2["x_A"][:limit].float().to(device), 1.0, measurement
    ).clamp(0.0, 1.0)
    intrinsic = geometry.intrinsic_record(y.to(torch.float64))
    flux_estimate = y[:, 0] * math.sqrt(size * size)
    flux_truth = truth.flatten(1).sum(dim=1)
    flux_relative_error = (
        (flux_estimate - flux_truth).abs() / flux_truth.clamp_min(1.0e-8)
    )

    query_gan = normalized_balanced_binary(
        geometry.null_project_flat((vqgan - base).flatten(1))
    )
    query_vqae = normalized_balanced_binary(
        geometry.null_project_flat((vqae_control - base).flatten(1))
    )
    hadamard_pool = torch.from_numpy(
        hadamard_lowsequency_non_dc_rows(512, geometry.n)
    ).to(device=device, dtype=torch.float32)
    hadamard_null_norm = geometry.null_project_flat(hadamard_pool).norm(dim=1)
    hadamard_index = int(torch.argmax(hadamard_null_norm).item())
    query_hadamard = hadamard_pool[hadamard_index].expand(limit, -1)

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    truth_r = repeat_images(truth, int(args.replicates))
    base_r = repeat_images(base, int(args.replicates))
    base_vectors = metric_vectors(base_r, truth_r, lpips_model)
    base_cluster = cluster_vectors(
        base_vectors, images=limit, replicates=int(args.replicates)
    )

    levels: dict[str, dict] = {}
    for level_index, photons in enumerate(args.signal_photons):
        method_queries = {
            "vqgan_disagreement": query_gan,
            "vqae_disagreement": query_vqae,
            "fixed_hadamard": query_hadamard,
        }
        noisy_records: dict[str, torch.Tensor] = {}
        photon_audits: dict[str, dict[str, float]] = {}
        for method_index, (name, query) in enumerate(method_queries.items()):
            record, audit = complementary_poisson_record(
                query,
                truth,
                flux_estimate,
                signal_photons=float(photons),
                background_fraction=float(args.background_fraction),
                replicates=int(args.replicates),
                seed=int(args.seed) + 1000 * level_index + 100 * method_index,
            )
            noisy_records[name] = record
            photon_audits[name] = audit

        predictions: dict[str, torch.Tensor] = {"base": base_r}
        projection_audits: dict[str, dict[str, float]] = {}
        predictions["vqgan_disagreement"], projection_audits["vqgan_disagreement"] = run_pair(
            base,
            vqgan,
            truth,
            intrinsic,
            query_gan,
            noisy_records["vqgan_disagreement"],
            geometry,
            replicates=int(args.replicates),
            iterations=int(args.iterations),
        )
        predictions["vqae_disagreement"], projection_audits["vqae_disagreement"] = run_pair(
            base,
            vqae_control,
            truth,
            intrinsic,
            query_vqae,
            noisy_records["vqae_disagreement"],
            geometry,
            replicates=int(args.replicates),
            iterations=int(args.iterations),
        )
        predictions["fixed_hadamard"], projection_audits["fixed_hadamard"] = run_fixed(
            base,
            intrinsic,
            query_hadamard,
            noisy_records["fixed_hadamard"],
            geometry,
            replicates=int(args.replicates),
            iterations=int(args.iterations),
        )

        vectors = {
            name: metric_vectors(prediction, truth_r, lpips_model)
            for name, prediction in predictions.items()
        }
        clusters = {
            name: cluster_vectors(
                values, images=limit, replicates=int(args.replicates)
            )
            for name, values in vectors.items()
        }
        means = {
            name: {metric: float(values.mean()) for metric, values in per_metric.items()}
            for name, per_metric in vectors.items()
        }
        paired_vs_base = {
            name: paired_summary(
                values,
                base_cluster,
                bootstrap_reps=int(args.bootstrap_reps),
                seed=int(args.seed) + 10000 * level_index + index,
            )
            for index, (name, values) in enumerate(clusters.items())
            if name != "base"
        }
        paired_vs_hadamard = {
            name: paired_summary(
                clusters[name],
                clusters["fixed_hadamard"],
                bootstrap_reps=int(args.bootstrap_reps),
                seed=int(args.seed) + 20000 * level_index + index,
            )
            for index, name in enumerate(
                ["vqgan_disagreement", "vqae_disagreement"]
            )
        }
        levels[str(float(photons))] = {
            "means": means,
            "paired_vs_base": paired_vs_base,
            "paired_vs_fixed_hadamard": paired_vs_hadamard,
            "photon_audits": photon_audits,
            "projection_audits": projection_audits,
        }

    payload = {
        "status": "INCREMENTAL_BUCKET_POISSON_SCREEN",
        "scope": "new complementary pair noisy; original 205-row record remains noiseless",
        "limit": limit,
        "noise_replicates_per_image": int(args.replicates),
        "bootstrap_reps": int(args.bootstrap_reps),
        "background_fraction_of_signal_photons": float(args.background_fraction),
        "test_split_opened": False,
        "truth_used_for_query_selection": False,
        "flux_exposure_set_from_existing_dc_measurement": True,
        "flux_dc_relative_error_max": float(flux_relative_error.max().cpu()),
        "operator_sha256": geometry.info.rows_sha256,
        "fixed_hadamard_index_in_512_lowsequency_pool": hadamard_index,
        "fixed_hadamard_null_norm": float(hadamard_null_norm[hadamard_index].cpu()),
        "levels": levels,
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"summary_val_{limit}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
