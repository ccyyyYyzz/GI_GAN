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
from completion_projector_gan import exact_model_prediction
from diagnose_active_binary_query_headroom import (
    augmented_box_projection,
    load_generator,
    normalized_balanced_binary,
)
from diagnose_fiber_critic_score_refinement import load_discriminator
from diagnose_unpaired_optical_calibration import evaluate, set_seed
from src.gauge_geometry import GaugeEmpiricalAnchor, GaugeGeometry, project_box_fiber_q


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/completion_gan_round18/pilot_checkpoint_sweep.yaml"
DEFAULT_CACHE = Path(
    "E:/GAN_FCC_WORK/experiments/completion_gan_round18/pqbf_pilot_seed0/cache"
)
DEFAULT_MATCHED = Path(
    "E:/GAN_FCC_WORK/experiments/completion_gan_round18/"
    "pqbf_pilot_selected_test_once_seed0/checkpoints/"
    "matched_supervised_step000750.pt"
)
DEFAULT_GAN = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round26/"
    "pqbf_adv_ratio_0p025/checkpoints/stage_b_step000750.pt"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round28/"
    "adversarial_extreme_pair_micro"
)


def per_image_critic_score(discriminator, anchor: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    logits = discriminator(anchor, image)
    return logits.flatten(1).mean(dim=1)


def box_penalty(image: torch.Tensor) -> torch.Tensor:
    return F.relu(-image).mean() + F.relu(image - 1.0).mean()


def optimize_extreme_pair(
    *,
    base: torch.Tensor,
    anchor: torch.Tensor,
    intrinsic: torch.Tensor,
    geometry: GaugeGeometry,
    discriminator,
    critic_weight: float,
    steps: int,
    lr: float,
    initial_scale: float,
    plausibility_margin: float,
    box_weight: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    generator = torch.Generator(device=base.device)
    generator.manual_seed(int(seed))
    parameter = torch.nn.Parameter(
        float(initial_scale) * torch.randn(base.shape, device=base.device, generator=generator)
    )
    optimizer = torch.optim.Adam([parameter], lr=float(lr), betas=(0.5, 0.9))
    with torch.no_grad():
        reference_score = per_image_critic_score(discriminator, anchor, base)
    last: dict[str, float] = {}
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        null_delta = geometry.project_feature_maps(parameter, null=True)
        plus = base + null_delta
        minus = base - null_delta
        separation = (plus - minus).abs().flatten(1).mean(dim=1)
        plus_score = per_image_critic_score(discriminator, anchor, plus)
        minus_score = per_image_critic_score(discriminator, anchor, minus)
        threshold = reference_score - float(plausibility_margin)
        plausibility = F.relu(threshold - plus_score).mean() + F.relu(
            threshold - minus_score
        ).mean()
        bounds = box_penalty(plus) + box_penalty(minus)
        loss = (
            -separation.mean()
            + float(critic_weight) * plausibility
            + float(box_weight) * bounds
        )
        loss.backward()
        optimizer.step()
        last = {
            "loss": float(loss.detach().cpu()),
            "mean_pair_l1_raw": float(separation.mean().detach().cpu()),
            "plausibility_hinge_raw": float(plausibility.detach().cpu()),
            "box_penalty_raw": float(bounds.detach().cpu()),
            "critic_plus_raw": float(plus_score.mean().detach().cpu()),
            "critic_minus_raw": float(minus_score.mean().detach().cpu()),
            "critic_reference": float(reference_score.mean().detach().cpu()),
        }

    with torch.no_grad():
        null_delta = geometry.project_feature_maps(parameter, null=True)
        raw_plus = base + null_delta
        raw_minus = base - null_delta
        plus_projection = project_box_fiber_q(
            raw_plus.flatten(1), intrinsic, geometry, iterations=256, exact=False
        )
        minus_projection = project_box_fiber_q(
            raw_minus.flatten(1), intrinsic, geometry, iterations=256, exact=False
        )
        plus = plus_projection.image_flat.reshape_as(base)
        minus = minus_projection.image_flat.reshape_as(base)
        last.update(
            {
                "mean_pair_l1_projected": float(
                    (plus - minus).abs().flatten(1).mean(dim=1).mean().cpu()
                ),
                "plus_record_residual": float(plus_projection.max_relative_record_error),
                "minus_record_residual": float(minus_projection.max_relative_record_error),
                "plus_box_violation": float(plus_projection.max_box_violation),
                "minus_box_violation": float(minus_projection.max_box_violation),
                "critic_plus_projected": float(
                    per_image_critic_score(discriminator, anchor, plus).mean().cpu()
                ),
                "critic_minus_projected": float(
                    per_image_critic_score(discriminator, anchor, minus).mean().cpu()
                ),
            }
        )
    return plus, minus, last


@torch.no_grad()
def query_diagnostics(
    plus: torch.Tensor,
    minus: torch.Tensor,
    base: torch.Tensor,
    truth: torch.Tensor,
    geometry: GaugeGeometry,
) -> tuple[torch.Tensor, dict[str, float]]:
    delta = geometry.null_project_flat(plus.flatten(1) - minus.flatten(1))
    query = normalized_balanced_binary(delta)
    effective = geometry.null_project_flat(query)
    error = geometry.null_project_flat(truth.flatten(1) - base.flatten(1))
    cosine = torch.abs((effective * error).sum(dim=1)) / (
        effective.norm(dim=1) * error.norm(dim=1)
    ).clamp_min(1.0e-12)
    pair_cosine = torch.abs((delta * error).sum(dim=1)) / (
        delta.norm(dim=1) * error.norm(dim=1)
    ).clamp_min(1.0e-12)
    true_bucket = (query * truth.flatten(1)).sum(dim=1)
    plus_bucket = (query * plus.flatten(1)).sum(dim=1)
    minus_bucket = (query * minus.flatten(1)).sum(dim=1)
    low = torch.minimum(plus_bucket, minus_bucket)
    high = torch.maximum(plus_bucket, minus_bucket)
    covered = ((true_bucket >= low) & (true_bucket <= high)).float()
    return query, {
        "query_error_abs_cosine_mean": float(cosine.mean().cpu()),
        "pair_error_abs_cosine_mean": float(pair_cosine.mean().cpu()),
        "truth_bucket_between_pair_fraction": float(covered.mean().cpu()),
        "mean_pair_bucket_separation": float((plus_bucket - minus_bucket).abs().mean().cpu()),
        "mean_truth_distance_to_pair_interval": float(
            (F.relu(low - true_bucket) + F.relu(true_bucket - high)).mean().cpu()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--matched-checkpoint", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--gan-checkpoint", type=Path, default=DEFAULT_GAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--initial-scale", type=float, default=0.005)
    parser.add_argument("--plausibility-margin", type=float, default=0.05)
    parser.add_argument("--critic-weight", type=float, default=10.0)
    parser.add_argument("--box-weight", type=float, default=50.0)
    parser.add_argument("--assimilation-iterations", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    set_seed(args.seed)
    started = time.time()
    device = torch.device("cuda")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    rows_np, operator_manifest = hq.build_structured_operator_rows(
        img_size=int(config["data"]["img_size"]),
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    geometry = GaugeGeometry(
        torch.from_numpy(rows_np).to(torch.float64),
        relative_cutoff=float(config["operator"].get("svd_relative_cutoff", 1.0e-12)),
    ).to(device)
    if geometry.info.rows_sha256 != operator_manifest["rows_sha256"]:
        raise RuntimeError("OPERATOR_IDENTITY_MISMATCH")

    train_payload = torch.load(args.cache_dir / "train.pt", map_location="cpu")
    val_payload = torch.load(args.cache_dir / "val.pt", map_location="cpu")
    anchor_model = GaugeEmpiricalAnchor.fit(
        train_payload["tensors"]["truth"].flatten(1).numpy(),
        geometry,
        lambda_=float(config["operator"].get("lmmse_lambda", 1.0e-3)),
    )
    uncertainty = anchor_model.normalized_posterior_map(
        img_size=int(config["data"]["img_size"]), device=device
    )
    limit = min(int(args.limit), int(val_payload["tensors"]["truth"].shape[0]))
    truth = val_payload["tensors"]["truth"][:limit].float().to(device)
    anchor = val_payload["tensors"]["anchor"][:limit].float().to(device)
    intrinsic = val_payload["tensors"]["intrinsic"][:limit].to(
        device=device, dtype=torch.float64
    )
    matched = load_generator(args.matched_checkpoint, geometry, config, device)
    discriminator = load_discriminator(args.gan_checkpoint, device)
    base_parts: list[torch.Tensor] = []
    for start in range(0, limit, int(args.batch_size)):
        stop = min(limit, start + int(args.batch_size))
        base, _ = exact_model_prediction(
            matched,
            anchor=anchor[start:stop],
            intrinsic=intrinsic[start:stop],
            uncertainty=uncertainty.expand(stop - start, -1, -1, -1),
            geometry=geometry,
        )
        base_parts.append(base)
    base = torch.cat(base_parts)

    variants = {
        "no_critic": 0.0,
        "adversarial_plausibility": float(args.critic_weight),
    }
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    pair_diagnostics: dict[str, list[dict[str, float]]] = {name: [] for name in variants}
    optimization_diagnostics: dict[str, list[dict[str, float]]] = {
        name: [] for name in variants
    }
    assimilation_audits: dict[str, list[dict[str, float]]] = {name: [] for name in variants}
    for start in range(0, limit, int(args.batch_size)):
        stop = min(limit, start + int(args.batch_size))
        for variant_index, (name, critic_weight) in enumerate(variants.items()):
            plus, minus, optimization = optimize_extreme_pair(
                base=base[start:stop],
                anchor=anchor[start:stop],
                intrinsic=intrinsic[start:stop],
                geometry=geometry,
                discriminator=discriminator,
                critic_weight=critic_weight,
                steps=int(args.steps),
                lr=float(args.lr),
                initial_scale=float(args.initial_scale),
                plausibility_margin=float(args.plausibility_margin),
                box_weight=float(args.box_weight),
                seed=int(args.seed) + 1000 * variant_index + start,
            )
            query, query_info = query_diagnostics(
                plus,
                minus,
                base[start:stop],
                truth[start:stop],
                geometry,
            )
            record = (query * truth[start:stop].flatten(1)).sum(dim=1)
            updated, audit = augmented_box_projection(
                base[start:stop],
                intrinsic[start:stop],
                query,
                record,
                geometry,
                iterations=int(args.assimilation_iterations),
            )
            predictions[name].append(updated)
            optimization_diagnostics[name].append(optimization)
            pair_diagnostics[name].append(query_info)
            assimilation_audits[name].append(audit)

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    baseline = evaluate(base, truth, lpips_model)
    metrics = {"base": baseline}
    for name, chunks in predictions.items():
        values = evaluate(torch.cat(chunks), truth, lpips_model)
        values.update(
            {
                "delta_psnr_vs_base": values["psnr"] - baseline["psnr"],
                "delta_ssim_vs_base": values["ssim"] - baseline["ssim"],
                "delta_lpips_vs_base": values["lpips"] - baseline["lpips"],
            }
        )
        metrics[name] = values

    def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
        keys = rows[0].keys()
        return {key: float(np.mean([row[key] for row in rows])) for key in keys}

    payload = {
        "status": "ADVERSARIAL_EXTREME_PAIR_QUERY_MICRO",
        "validation_only": True,
        "test_split_opened": False,
        "limit": limit,
        "operator_sha256": geometry.info.rows_sha256,
        "metrics": metrics,
        "pair_diagnostics": {
            name: aggregate(rows) for name, rows in pair_diagnostics.items()
        },
        "optimization_diagnostics": {
            name: aggregate(rows) for name, rows in optimization_diagnostics.items()
        },
        "assimilation_audits": {
            name: aggregate(rows) for name, rows in assimilation_audits.items()
        },
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
