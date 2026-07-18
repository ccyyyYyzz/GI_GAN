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
from diagnose_unpaired_optical_calibration import evaluate
from src.dc_balanced import dct_lowfreq_non_dc_rows
from src.gauge_geometry import GaugeGeometry


DEFAULT_ORIGINAL = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
DEFAULT_CACHE = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round31/"
    "vq_disagreement_bucket/cache"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round31/"
    "vq_disagreement_bucket/pilot"
)


@torch.no_grad()
def pair_diagnostics(
    x_a: torch.Tensor,
    x_g: torch.Tensor,
    truth: torch.Tensor,
    geometry: GaugeGeometry,
) -> tuple[torch.Tensor, dict[str, float]]:
    pair = geometry.null_project_flat(x_g.flatten(1) - x_a.flatten(1))
    error = geometry.null_project_flat(truth.flatten(1) - x_a.flatten(1))
    query = normalized_balanced_binary(pair)
    effective = geometry.null_project_flat(query)
    query_cosine = torch.abs((effective * error).sum(dim=1)) / (
        effective.norm(dim=1) * error.norm(dim=1)
    ).clamp_min(1.0e-12)
    pair_cosine = torch.abs((pair * error).sum(dim=1)) / (
        pair.norm(dim=1) * error.norm(dim=1)
    ).clamp_min(1.0e-12)
    bucket_true = (query * truth.flatten(1)).sum(dim=1)
    bucket_a = (query * x_a.flatten(1)).sum(dim=1)
    bucket_g = (query * x_g.flatten(1)).sum(dim=1)
    low = torch.minimum(bucket_a, bucket_g)
    high = torch.maximum(bucket_a, bucket_g)
    covered = ((bucket_true >= low) & (bucket_true <= high)).float()
    return query, {
        "query_error_abs_cosine_mean": float(query_cosine.mean().cpu()),
        "query_error_abs_cosine_median": float(query_cosine.median().cpu()),
        "pair_error_abs_cosine_mean": float(pair_cosine.mean().cpu()),
        "pair_error_abs_cosine_median": float(pair_cosine.median().cpu()),
        "truth_bucket_between_pair_fraction": float(covered.mean().cpu()),
        "mean_pair_bucket_separation": float((bucket_g - bucket_a).abs().mean().cpu()),
    }


@torch.no_grad()
def assimilate(
    proposal: torch.Tensor,
    truth: torch.Tensor,
    intrinsic: torch.Tensor,
    query: torch.Tensor,
    geometry: GaugeGeometry,
    *,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    record = (query * truth.flatten(1)).sum(dim=1)
    return augmented_box_projection(
        proposal,
        intrinsic,
        query,
        record,
        geometry,
        iterations=int(iterations),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["val", "dev"], default="val")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    started = time.time()
    device = torch.device("cuda")

    config_path = (
        args.original_root
        / "outputs/compatibility/measurement_conditioned_vqgan/"
        "anchor_multiseed_hashclean_seed0/config_used.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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

    pack = torch.load(args.cache_dir / f"seed0_{args.split}.pt", map_location="cpu")
    limit = min(int(args.limit), int(pack["truth"].shape[0]))
    truth = pack["truth"][:limit].float().to(device)
    x0 = pack["x0"][:limit].float().to(device)
    raw_a = pack["x_A"][:limit].float().to(device)
    raw_g = pack["x_G"][:limit].float().to(device)
    y = pack["y"][:limit].float().to(device)

    measurement = hq.make_measurement_operator(
        rows_np,
        img_size=size,
        device=device,
        lambda_solver=float(operator["lambda_solver"]),
    )
    x_a = ai.null_blend(x0, raw_a, 1.0, measurement).clamp(0.0, 1.0)
    x_g = ai.null_blend(x0, raw_g, 1.0, measurement).clamp(0.0, 1.0)
    balanced_fusion = ai.null_blend(
        x0, raw_a + 0.55 * (raw_g - raw_a), 1.0, measurement
    ).clamp(0.0, 1.0)
    intrinsic = geometry.intrinsic_record(y.to(torch.float64))

    query, diagnostics = pair_diagnostics(x_a, x_g, truth, geometry)
    minimal, minimal_audit = assimilate(
        x_a,
        truth,
        intrinsic,
        query,
        geometry,
        iterations=int(args.iterations),
    )
    center = 0.5 * (x_a + x_g)
    half = 0.5 * (x_g - x_a)
    record = (query * truth.flatten(1)).sum(dim=1)
    numerator = record - (query * center.flatten(1)).sum(dim=1)
    denominator = (query * half.flatten(1)).sum(dim=1)
    safe_denominator = torch.where(
        denominator.abs() >= 1.0e-8,
        denominator,
        torch.where(denominator >= 0.0, torch.full_like(denominator, 1.0e-8), torch.full_like(denominator, -1.0e-8)),
    )
    coordinate = (numerator / safe_denominator).clamp(-1.0, 1.0)
    pair_proposal = center + coordinate[:, None, None, None] * half
    pair_update, pair_audit = assimilate(
        pair_proposal,
        truth,
        intrinsic,
        query,
        geometry,
        iterations=int(args.iterations),
    )

    dct_pool = dct_lowfreq_non_dc_rows(int(operator["dct_rows"]) + 1, size)
    dct_query = torch.from_numpy(dct_pool[-1]).to(device=device, dtype=torch.float32)
    dct_query = dct_query.expand(limit, -1)
    fixed_dct, dct_audit = assimilate(
        x_a,
        truth,
        intrinsic,
        dct_query,
        geometry,
        iterations=int(args.iterations),
    )
    rng = np.random.default_rng(int(args.seed) + 55)
    template = np.concatenate(
        [-np.ones(geometry.n // 2, dtype=np.float32), np.ones(geometry.n // 2, dtype=np.float32)]
    ) / math.sqrt(geometry.n)
    random_query_np = np.stack([rng.permutation(template) for _ in range(limit)])
    random_query = torch.from_numpy(random_query_np).to(device)
    fixed_random, random_audit = assimilate(
        x_a,
        truth,
        intrinsic,
        random_query,
        geometry,
        iterations=int(args.iterations),
    )

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    predictions = {
        "vqae": x_a,
        "vqgan": x_g,
        "fusion_0p55": balanced_fusion,
        "fixed_next_dct_one_bucket": fixed_dct,
        "fixed_random_one_bucket": fixed_random,
        "vq_disagreement_minimal_update": minimal,
        "vq_disagreement_pair_segment": pair_update,
    }
    metrics = {name: evaluate(value, truth, lpips_model) for name, value in predictions.items()}
    baseline = metrics["vqae"]
    for name, values in metrics.items():
        values.update(
            {
                "delta_psnr_vs_vqae": values["psnr"] - baseline["psnr"],
                "delta_ssim_vs_vqae": values["ssim"] - baseline["ssim"],
                "delta_lpips_vs_vqae": values["lpips"] - baseline["lpips"],
            }
        )
    payload = {
        "status": "VQGAN_VQAE_DISAGREEMENT_BUCKET_PILOT",
        "split": args.split,
        "limit": limit,
        "test_split_opened": False,
        "truth_used_for_query_selection": False,
        "truth_used_only_as_simulated_new_bucket_and_metrics": True,
        "original_operator_sha256": geometry.info.rows_sha256,
        "metrics": metrics,
        "pair_diagnostics": diagnostics,
        "projection_audits": {
            "minimal": minimal_audit,
            "pair_segment": pair_audit,
            "fixed_dct": dct_audit,
            "fixed_random": random_audit,
        },
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"summary_{args.split}_{limit}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
