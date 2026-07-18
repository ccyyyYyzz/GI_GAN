from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity

import anchor_initialized_vqgan_inversion as ai
import gan_high_quality_gi as hq
from diagnose_vqgan_vqae_disagreement_bucket import assimilate, pair_diagnostics
from src.dc_balanced import dct_lowfreq_non_dc_rows
from src.gauge_geometry import GaugeGeometry


DEFAULT_ORIGINAL = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
DEFAULT_BASE_CACHE = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round31/"
    "vq_disagreement_bucket/cache/seed0_val.pt"
)
DEFAULT_CONTROL_CACHE = Path(
    "E:/ns_mc_gan_gi_code_fcc_phase1/outputs/compatibility/"
    "measurement_conditioned_vqgan/detail_fusion/cache"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round32/"
    "vq_disagreement_causal_controls"
)


@torch.no_grad()
def metric_vectors(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    lpips_model: torch.nn.Module,
) -> dict[str, np.ndarray]:
    prediction = prediction.clamp(0.0, 1.0)
    truth = truth.clamp(0.0, 1.0)
    mse = (prediction - truth).square().flatten(1).mean(1)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1.0e-12))
    pred_rgb = prediction.repeat(1, 3, 1, 1) * 2.0 - 1.0
    truth_rgb = truth.repeat(1, 3, 1, 1) * 2.0 - 1.0
    lpips = lpips_model(pred_rgb, truth_rgb).flatten()
    pred_np = prediction[:, 0].detach().cpu().float().numpy()
    truth_np = truth[:, 0].detach().cpu().float().numpy()
    ssim = np.asarray(
        [
            structural_similarity(target, pred, data_range=1.0)
            for pred, target in zip(pred_np, truth_np)
        ],
        dtype=np.float64,
    )
    return {
        "psnr": psnr.detach().cpu().double().numpy(),
        "ssim": ssim,
        "lpips": lpips.detach().cpu().double().numpy(),
    }


def paired_summary(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    n = len(candidate["psnr"])
    indices = rng.integers(0, n, size=(int(bootstrap_reps), n))
    result: dict[str, dict[str, float]] = {}
    for metric in ("psnr", "ssim", "lpips"):
        delta = candidate[metric] - reference[metric]
        bootstrap = delta[indices].mean(axis=1)
        if metric == "lpips":
            improved = delta < 0.0
        else:
            improved = delta > 0.0
        result[metric] = {
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "sample_improved_fraction": float(improved.mean()),
        }
    return result


@torch.no_grad()
def pair_segment_update(
    base: torch.Tensor,
    alternative: torch.Tensor,
    truth: torch.Tensor,
    intrinsic: torch.Tensor,
    geometry: GaugeGeometry,
    *,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    query, diagnostics = pair_diagnostics(base, alternative, truth, geometry)
    center = 0.5 * (base + alternative)
    half = 0.5 * (alternative - base)
    record = (query * truth.flatten(1)).sum(dim=1)
    numerator = record - (query * center.flatten(1)).sum(dim=1)
    denominator = (query * half.flatten(1)).sum(dim=1)
    safe_denominator = torch.where(
        denominator.abs() >= 1.0e-8,
        denominator,
        torch.where(
            denominator >= 0.0,
            torch.full_like(denominator, 1.0e-8),
            torch.full_like(denominator, -1.0e-8),
        ),
    )
    raw_coordinate = numerator / safe_denominator
    coordinate = raw_coordinate.clamp(-1.0, 1.0)
    proposal = center + coordinate[:, None, None, None] * half
    updated, audit = assimilate(
        proposal,
        truth,
        intrinsic,
        query,
        geometry,
        iterations=int(iterations),
    )
    diagnostics.update(
        {
            "coordinate_clipped_fraction": float(
                (raw_coordinate.abs() > 1.0).float().mean().cpu()
            ),
            "coordinate_mean": float(coordinate.mean().cpu()),
            "coordinate_std": float(coordinate.std(unbiased=False).cpu()),
            "coordinate_denominator_abs_min": float(denominator.abs().min().cpu()),
        }
    )
    return updated, diagnostics, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--control-cache-dir", type=Path, default=DEFAULT_CONTROL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=8192)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
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
    measurement = hq.make_measurement_operator(
        rows_np,
        img_size=size,
        device=device,
        lambda_solver=float(operator["lambda_solver"]),
    )

    pack0 = torch.load(args.base_cache, map_location="cpu")
    limit = min(int(args.limit), int(pack0["truth"].shape[0]))
    truth = pack0["truth"][:limit].float().to(device)
    x0 = pack0["x0"][:limit].float().to(device)
    y = pack0["y"][:limit].float().to(device)
    base = ai.null_blend(
        x0, pack0["x_A"][:limit].float().to(device), 1.0, measurement
    ).clamp(0.0, 1.0)
    alternatives = {
        "vqgan_seed0": ai.null_blend(
            x0, pack0["x_G"][:limit].float().to(device), 1.0, measurement
        ).clamp(0.0, 1.0)
    }
    identity_audit: dict[str, bool] = {}
    for seed in (1, 2):
        pack = torch.load(
            args.control_cache_dir / f"seed{seed}_val.pt", map_location="cpu"
        )
        for field in ("source_index", "truth", "y", "x0"):
            same = torch.equal(pack0[field], pack[field])
            identity_audit[f"seed{seed}_{field}_equal"] = bool(same)
            if not same:
                raise RuntimeError(f"CONTROL_CACHE_IDENTITY_MISMATCH:seed{seed}:{field}")
        alternatives[f"vqae_seed{seed}"] = ai.null_blend(
            x0, pack["x_A"][:limit].float().to(device), 1.0, measurement
        ).clamp(0.0, 1.0)
        alternatives[f"vqgan_seed{seed}"] = ai.null_blend(
            x0, pack["x_G"][:limit].float().to(device), 1.0, measurement
        ).clamp(0.0, 1.0)

    intrinsic = geometry.intrinsic_record(y.to(torch.float64))
    predictions: dict[str, torch.Tensor] = {"vqae_seed0": base}
    diagnostics: dict[str, dict[str, float]] = {}
    projection_audits: dict[str, dict[str, float]] = {}
    for name, alternative in alternatives.items():
        prediction, diagnostic, audit = pair_segment_update(
            base,
            alternative,
            truth,
            intrinsic,
            geometry,
            iterations=int(args.iterations),
        )
        predictions[name] = prediction
        diagnostics[name] = diagnostic
        projection_audits[name] = audit

    dct_pool = dct_lowfreq_non_dc_rows(int(operator["dct_rows"]) + 1, size)
    dct_query = torch.from_numpy(dct_pool[-1]).to(device=device, dtype=torch.float32)
    dct_query = dct_query.expand(limit, -1)
    predictions["fixed_next_dct"], projection_audits["fixed_next_dct"] = assimilate(
        base,
        truth,
        intrinsic,
        dct_query,
        geometry,
        iterations=int(args.iterations),
    )
    rng = np.random.default_rng(int(args.seed) + 55)
    template = np.concatenate(
        [
            -np.ones(geometry.n // 2, dtype=np.float32),
            np.ones(geometry.n // 2, dtype=np.float32),
        ]
    ) / math.sqrt(geometry.n)
    random_query = torch.from_numpy(
        np.stack([rng.permutation(template) for _ in range(limit)])
    ).to(device)
    predictions["fixed_random"], projection_audits["fixed_random"] = assimilate(
        base,
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
    vectors = {
        name: metric_vectors(prediction, truth, lpips_model)
        for name, prediction in predictions.items()
    }
    means = {
        name: {metric: float(values.mean()) for metric, values in per_metric.items()}
        for name, per_metric in vectors.items()
    }
    paired_vs_vqae = {
        name: paired_summary(
            values,
            vectors["vqae_seed0"],
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed) + index,
        )
        for index, (name, values) in enumerate(vectors.items())
        if name != "vqae_seed0"
    }
    paired_vs_dct = {
        name: paired_summary(
            vectors[name],
            vectors["fixed_next_dct"],
            bootstrap_reps=int(args.bootstrap_reps),
            seed=int(args.seed) + 100 + index,
        )
        for index, name in enumerate(
            ["vqgan_seed0", "vqgan_seed1", "vqgan_seed2", "vqae_seed1", "vqae_seed2"]
        )
    }

    payload = {
        "status": "VQGAN_CAUSAL_DISAGREEMENT_CONTROL",
        "limit": limit,
        "bootstrap_reps": int(args.bootstrap_reps),
        "test_split_opened": False,
        "truth_used_for_query_selection": False,
        "truth_used_only_as_simulated_new_bucket_and_metrics": True,
        "operator_sha256": geometry.info.rows_sha256,
        "cache_identity_audit": identity_audit,
        "means": means,
        "paired_vs_vqae_seed0": paired_vs_vqae,
        "paired_vs_fixed_next_dct": paired_vs_dct,
        "diagnostics": diagnostics,
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
