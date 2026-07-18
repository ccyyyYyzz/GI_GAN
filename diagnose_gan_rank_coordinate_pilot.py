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
from diagnose_vqgan_causal_disagreement_controls import (
    metric_vectors,
    paired_summary,
)
from src.dc_balanced import dct_lowfreq_non_dc_rows
from src.gauge_geometry import GaugeGeometry


DEFAULT_ORIGINAL = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
DEFAULT_BASE_CACHE = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round31/"
    "vq_disagreement_bucket/cache/seed0_val.pt"
)
DEFAULT_CONTROL_CACHE = Path(
    "E:/ns_mc_gan_gi_code_fcc_phase1/outputs/compatibility/"
    "measurement_conditioned_vqgan/detail_fusion/cache/seed1_val.pt"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round35/"
    "gan_rank_coordinate"
)


def _bit_reverse(value: int, bits: int) -> int:
    result = 0
    for _ in range(bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def sequency_walsh_rows(n: int, count: int) -> torch.Tensor:
    """Return the first non-DC sequency-ordered balanced Walsh rows."""

    bits = int(round(math.log2(n)))
    if 2**bits != n:
        raise ValueError("WALSH_REQUIRES_POWER_OF_TWO_PIXELS")
    if not 1 <= count < n:
        raise ValueError("INVALID_WALSH_ROW_COUNT")
    positions = torch.arange(n, dtype=torch.int64)
    rows = []
    for sequency in range(1, count + 1):
        sylvester_index = _bit_reverse(sequency ^ (sequency >> 1), bits)
        parity = torch.zeros(n, dtype=torch.int64)
        masked = positions & sylvester_index
        for bit in range(bits):
            parity ^= (masked >> bit) & 1
        rows.append(torch.where(parity == 0, 1.0, -1.0))
    result = torch.stack(rows) / math.sqrt(n)
    if not bool((result.sum(dim=1).abs() < 1.0e-6).all()):
        raise RuntimeError("WALSH_ROWS_NOT_BALANCED")
    gram = result @ result.T
    if not torch.allclose(gram, torch.eye(count), atol=1.0e-6, rtol=0.0):
        raise RuntimeError("WALSH_ROWS_NOT_ORTHONORMAL")
    return result


def rank_walsh_queries(guide: torch.Tensor, count: int) -> torch.Tensor:
    """Pull low-sequency Walsh rows back through a sample-specific rank map."""

    if guide.ndim != 2:
        raise ValueError("GUIDE_MUST_BE_BATCH_BY_PIXELS")
    batch, n = guide.shape
    rank_rows = sequency_walsh_rows(n, count).to(guide.device, guide.dtype)
    order = torch.argsort(guide, dim=1, stable=True)
    queries = torch.empty(batch, count, n, device=guide.device, dtype=guide.dtype)
    queries.scatter_(
        2,
        order[:, None, :].expand(-1, count, -1),
        rank_rows[None, :, :].expand(batch, -1, -1),
    )
    return queries


@torch.no_grad()
def augmented_box_projection_many(
    proposal: torch.Tensor,
    intrinsic: torch.Tensor,
    queries: torch.Tensor,
    records: torch.Tensor,
    geometry: GaugeGeometry,
    *,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project onto the old fiber, several new signed buckets, and the image box."""

    flat = proposal.flatten(1).float()
    q = queries.to(device=flat.device, dtype=flat.dtype)
    target = records.to(device=flat.device, dtype=flat.dtype)
    batch, count, n = q.shape
    if flat.shape != (batch, n) or target.shape != (batch, count):
        raise ValueError("AUGMENTED_PROJECTION_SHAPE_MISMATCH")
    q_null = geometry.null_project_flat(q.reshape(batch * count, n)).reshape(
        batch, count, n
    )
    gram = q_null @ q_null.transpose(1, 2)
    singular_values = torch.linalg.svdvals(gram)
    if bool((singular_values[:, -1] <= 1.0e-7).any()):
        raise RuntimeError("AUGMENTED_QUERIES_RANK_DEFICIENT")

    def affine_project(value: torch.Tensor) -> torch.Tensor:
        old_fiber = geometry.affine_project_flat(value, intrinsic)
        residual = target - torch.einsum("bkn,bn->bk", q, old_fiber)
        coefficient = torch.linalg.solve(gram, residual.unsqueeze(-1)).squeeze(-1)
        return old_fiber + torch.einsum("bk,bkn->bn", coefficient, q_null)

    current = flat
    box_dual = torch.zeros_like(current)
    affine_dual = torch.zeros_like(current)
    max_step = float("inf")
    for _ in range(int(iterations)):
        box_input = current + box_dual
        on_box = box_input.clamp(0.0, 1.0)
        box_dual = box_input - on_box
        affine_input = on_box + affine_dual
        updated = affine_project(affine_input)
        affine_dual = affine_input - updated
        max_step = float((updated - current).abs().max().detach().cpu())
        current = updated

    old_residual = float(
        geometry.relative_record_error(current, intrinsic).max().detach().cpu()
    )
    predicted = torch.einsum("bkn,bn->bk", q, current)
    scale = target.abs().clamp_min(1.0e-8)
    new_residual = float(((predicted - target).abs() / scale).max().detach().cpu())
    box_violation = float(
        torch.maximum((-current).clamp_min(0.0), (current - 1.0).clamp_min(0.0))
        .max()
        .detach()
        .cpu()
    )
    return current.reshape_as(proposal), {
        "old_fiber_relative_residual": old_residual,
        "new_bucket_relative_residual": new_residual,
        "box_violation": box_violation,
        "last_step_infinity": max_step,
        "effective_gram_min_singular": float(singular_values[:, -1].min().cpu()),
        "effective_gram_max_condition": float(
            (singular_values[:, 0] / singular_values[:, -1]).max().cpu()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--control-cache", type=Path, default=DEFAULT_CONTROL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--budgets", default="1,2,4,8")
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    budgets = sorted({int(value) for value in args.budgets.split(",")})
    if not budgets or min(budgets) < 1:
        raise ValueError("BUDGETS_MUST_BE_POSITIVE")
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    started = time.time()
    device = torch.device("cuda")

    base_root = args.original_root / "outputs/compatibility/measurement_conditioned_vqgan"
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
    pack1 = torch.load(args.control_cache, map_location="cpu")
    for field in ("truth", "x0", "y", "source_index"):
        if not torch.equal(pack0[field], pack1[field]):
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
        x0, pack1["x_A"][:limit].float().to(device), 1.0, measurement
    ).clamp(0.0, 1.0)
    intrinsic = geometry.intrinsic_record(y.to(torch.float64))
    base_flat = base.flatten(1)
    gan_guide = geometry.null_project_flat(vqgan.flatten(1) - base_flat)
    vqae_guide = geometry.null_project_flat(vqae_control.flatten(1) - base_flat)
    guides = {
        "gan_residual_rank": gan_guide,
        "vqae_residual_rank": vqae_guide,
        "blend_rank_w0p25": 0.75 * vqae_guide + 0.25 * gan_guide,
        "blend_rank_w0p50": 0.50 * vqae_guide + 0.50 * gan_guide,
        "blend_rank_w0p75": 0.25 * vqae_guide + 0.75 * gan_guide,
    }
    rng = np.random.default_rng(int(args.seed) + 3501)
    guides["random_rank"] = torch.from_numpy(
        rng.standard_normal((limit, geometry.n), dtype=np.float32)
    ).to(device)

    maximum = max(budgets)
    query_banks = {
        name: rank_walsh_queries(guide, maximum) for name, guide in guides.items()
    }
    dct_np = dct_lowfreq_non_dc_rows(int(operator["dct_rows"]) + maximum, size)[
        int(operator["dct_rows"]) :
    ]
    dct = torch.from_numpy(dct_np).to(device=device, dtype=torch.float32)
    query_banks["fixed_next_dct"] = dct[None, :, :].expand(limit, -1, -1)

    predictions: dict[str, torch.Tensor] = {"vqae_seed0": base}
    audits: dict[str, dict[str, float]] = {}
    truth_flat = truth.flatten(1)
    for name, bank in query_banks.items():
        for budget in budgets:
            queries = bank[:, :budget]
            records = torch.einsum("bkn,bn->bk", queries, truth_flat)
            key = f"{name}_b{budget}"
            try:
                predictions[key], audits[key] = augmented_box_projection_many(
                    base,
                    intrinsic,
                    queries,
                    records,
                    geometry,
                    iterations=int(args.iterations),
                )
            except RuntimeError as error:
                raise RuntimeError(f"{key}:{error}") from error

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
    paired: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for budget in budgets:
        reference_name = f"fixed_next_dct_b{budget}"
        paired[f"b{budget}"] = {}
        for index, name in enumerate(guides):
            candidate_name = f"{name}_b{budget}"
            paired[f"b{budget}"][name] = paired_summary(
                vectors[candidate_name],
                vectors[reference_name],
                bootstrap_reps=int(args.bootstrap_reps),
                seed=int(args.seed) + 1000 * budget + index,
            )

    payload = {
        "status": "GAN_RANK_COORDINATE_VALIDATION_PILOT",
        "hypothesis": (
            "adversarial residual values induce a sample-specific rank coordinate in "
            "which the unknown old-fiber error is concentrated in low-sequency Walsh modes"
        ),
        "limit": limit,
        "budgets": budgets,
        "validation_only": True,
        "test_split_opened": False,
        "truth_used_for_query_selection": False,
        "truth_used_only_as_simulated_new_buckets_and_metrics": True,
        "one_signed_query_is_one_complementary_dmd_pair": True,
        "operator_sha256": geometry.info.rows_sha256,
        "means": means,
        "paired_vs_fixed_next_dct": paired,
        "projection_audits": audits,
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"summary_val_{limit}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
