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
from completion_projector_gan import exact_model_prediction
from diagnose_unpaired_optical_calibration import evaluate, set_seed
from src.dc_balanced import dct_lowfreq_non_dc_rows
from src.gauge_geometry import GaugeEmpiricalAnchor, GaugeGeometry
from src.projector_gated_fiber_gan import ProjectorGatedFiberGenerator


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
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round27/"
    "active_binary_query_headroom"
)


def load_generator(
    path: Path,
    geometry: GaugeGeometry,
    config: dict,
    device: torch.device,
) -> ProjectorGatedFiberGenerator:
    payload = torch.load(path, map_location=device)
    model = ProjectorGatedFiberGenerator(
        geometry,
        steps=int(config["model"]["steps"]),
        step_scale=float(config["model"]["step_scale"]),
    ).to(device)
    model.load_state_dict(payload["ema"])
    return model.eval()


@torch.no_grad()
def augmented_box_projection(
    proposal: torch.Tensor,
    intrinsic: torch.Tensor,
    query: torch.Tensor,
    query_record: torch.Tensor,
    geometry: GaugeGeometry,
    *,
    iterations: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Project onto [0,1]^n intersected with the old fiber and one new bucket.

    The effective part of the new row is P_null q.  It is orthogonal to the old
    row space, so projection onto the augmented affine set needs only a scalar
    correction after the existing intrinsic affine projection.  Dykstra then
    handles the box without forming a different 206 x 4096 SVD per image.
    """

    flat = proposal.flatten(1).float()
    q = query.to(device=flat.device, dtype=flat.dtype)
    z = intrinsic.to(device=flat.device, dtype=flat.dtype)
    target = query_record.to(device=flat.device, dtype=flat.dtype).reshape(-1)
    q_null = geometry.null_project_flat(q)
    norm2 = q_null.square().sum(dim=1).clamp_min(1.0e-10)
    if bool((norm2 <= 1.0e-9).any()):
        raise RuntimeError("QUERY_HAS_NO_NEW_NULLSPACE_COMPONENT")

    def affine_project(value: torch.Tensor) -> torch.Tensor:
        old_fiber = geometry.affine_project_flat(value, z)
        residual = target - (old_fiber * q).sum(dim=1)
        return old_fiber + (residual / norm2)[:, None] * q_null

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

    old_residual = float(geometry.relative_record_error(current, z).max().detach().cpu())
    new_scale = target.abs().clamp_min(1.0e-8)
    new_residual = float(
        (((current * q).sum(dim=1) - target).abs() / new_scale).max().detach().cpu()
    )
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
        "query_effective_norm_min": float(norm2.sqrt().min().detach().cpu()),
    }


def normalized_binary_sign(value: torch.Tensor) -> torch.Tensor:
    sign = torch.where(value >= 0.0, torch.ones_like(value), -torch.ones_like(value))
    return sign / math.sqrt(value.shape[1])


def normalized_balanced_binary(value: torch.Tensor) -> torch.Tensor:
    """Maximizer of q^T value over half-on/half-off complementary rows."""

    if value.shape[1] % 2:
        raise ValueError("BALANCED_BINARY_REQUIRES_EVEN_PIXEL_COUNT")
    order = torch.argsort(value, dim=1)
    query = -torch.ones_like(value)
    query.scatter_(1, order[:, value.shape[1] // 2 :], 1.0)
    return query / math.sqrt(value.shape[1])


@torch.no_grad()
def run_strategy(
    *,
    name: str,
    base: torch.Tensor,
    truth: torch.Tensor,
    intrinsic: torch.Tensor,
    geometry: GaugeGeometry,
    budgets: list[int],
    fixed_rows: torch.Tensor | None,
    oracle_kind: str | None,
    dykstra_iterations: int,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, float]]]:
    maximum = max(budgets)
    current = base.clone()
    outputs: dict[int, torch.Tensor] = {0: current.clone()}
    audits: dict[int, dict[str, float]] = {}
    truth_flat = truth.flatten(1)
    for step in range(1, maximum + 1):
        error = geometry.null_project_flat(truth_flat - current.flatten(1))
        if oracle_kind == "binary":
            query = normalized_binary_sign(error)
        elif oracle_kind == "balanced_binary":
            query = normalized_balanced_binary(error)
        elif oracle_kind == "continuous":
            query = error / error.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        else:
            if fixed_rows is None:
                raise RuntimeError(f"MISSING_FIXED_ROWS:{name}")
            query = fixed_rows[step - 1].to(current.device).expand(current.shape[0], -1)
        query_record = (query * truth_flat).sum(dim=1)
        current, audit = augmented_box_projection(
            current,
            intrinsic,
            query,
            query_record,
            geometry,
            iterations=dykstra_iterations,
        )
        if step in budgets:
            outputs[step] = current.clone()
            audits[step] = audit
    return outputs, audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--matched-checkpoint", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--gan-checkpoint", type=Path, default=DEFAULT_GAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--budgets", default="1,2,4,8")
    parser.add_argument("--dykstra-iterations", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    set_seed(args.seed)
    started = time.time()
    device = torch.device("cuda")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    budgets = sorted({int(value) for value in args.budgets.split(",")})
    if not budgets or min(budgets) < 1:
        raise ValueError("BUDGETS_MUST_BE_POSITIVE")

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
    truth = val_payload["tensors"]["truth"].float().to(device)
    anchor = val_payload["tensors"]["anchor"].float().to(device)
    intrinsic = val_payload["tensors"]["intrinsic"].to(device=device, dtype=torch.float64)

    models = {
        "matched": load_generator(args.matched_checkpoint, geometry, config, device),
        "gan_0p025": load_generator(args.gan_checkpoint, geometry, config, device),
    }
    bases: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    for start in range(0, truth.shape[0], int(args.batch_size)):
        stop = min(truth.shape[0], start + int(args.batch_size))
        sigma = uncertainty.expand(stop - start, -1, -1, -1)
        for name, model in models.items():
            prediction, _ = exact_model_prediction(
                model,
                anchor=anchor[start:stop],
                intrinsic=intrinsic[start:stop],
                uncertainty=sigma,
                geometry=geometry,
            )
            bases[name].append(prediction)
    base_images = {name: torch.cat(parts) for name, parts in bases.items()}

    size = int(config["data"]["img_size"])
    acquired_dct = int(config["operator"]["dct_rows"])
    dct_pool_np = dct_lowfreq_non_dc_rows(acquired_dct + max(budgets), size)[
        acquired_dct:
    ]
    dct_pool = torch.from_numpy(dct_pool_np).to(device=device, dtype=torch.float32)
    rng = np.random.default_rng(int(args.seed) + 8103)
    random_pool_np = np.empty((max(budgets), geometry.n), dtype=np.float32)
    balanced_template = np.concatenate(
        [
            -np.ones(geometry.n // 2, dtype=np.float32),
            np.ones(geometry.n // 2, dtype=np.float32),
        ]
    )
    for row_index in range(max(budgets)):
        random_pool_np[row_index] = rng.permutation(balanced_template)
    random_pool_np /= math.sqrt(geometry.n)
    random_pool = torch.from_numpy(random_pool_np).to(device)

    strategies = {
        "oracle_continuous": {"fixed_rows": None, "oracle_kind": "continuous"},
        "oracle_binary_dmd": {"fixed_rows": None, "oracle_kind": "binary"},
        "oracle_balanced_binary_dmd": {
            "fixed_rows": None,
            "oracle_kind": "balanced_binary",
        },
        "fixed_next_dct": {"fixed_rows": dct_pool, "oracle_kind": None},
        "fixed_random_binary": {"fixed_rows": random_pool, "oracle_kind": None},
    }
    outputs: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    audits: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for base_name, base in base_images.items():
        outputs[base_name] = {}
        audits[base_name] = {}
        for strategy_name, specification in strategies.items():
            values, diagnostics = run_strategy(
                name=strategy_name,
                base=base,
                truth=truth,
                intrinsic=intrinsic,
                geometry=geometry,
                budgets=budgets,
                fixed_rows=specification["fixed_rows"],
                oracle_kind=specification["oracle_kind"],
                dykstra_iterations=int(args.dykstra_iterations),
            )
            outputs[base_name][strategy_name] = values
            audits[base_name][strategy_name] = diagnostics

    # First implementable bridge: the adversarial and equal-step supervised
    # reconstructions are two truth-free, same-record hypotheses.  Their sign
    # difference is the binary DMD row that maximally separates that pair.
    # This is not yet a posterior GAN, but it tests whether the existing GAN
    # disagreement contains physically useful acquisition information.
    pair_delta = geometry.null_project_flat(
        base_images["gan_0p025"].flatten(1) - base_images["matched"].flatten(1)
    )
    pair_query = normalized_balanced_binary(pair_delta)
    pair_record = (pair_query * truth.flatten(1)).sum(dim=1)
    pair_outputs: dict[str, torch.Tensor] = {}
    pair_audits: dict[str, dict[str, float]] = {}
    for base_name, base in base_images.items():
        projected, audit = augmented_box_projection(
            base,
            intrinsic,
            pair_query,
            pair_record,
            geometry,
            iterations=int(args.dykstra_iterations),
        )
        pair_outputs[f"{base_name}/gan_supervised_pair_query"] = projected
        pair_audits[f"{base_name}/gan_supervised_pair_query"] = audit

    pair_center = 0.5 * (base_images["matched"] + base_images["gan_0p025"])
    pair_half = 0.5 * (base_images["gan_0p025"] - base_images["matched"])
    numerator = pair_record - (pair_query * pair_center.flatten(1)).sum(dim=1)
    denominator = (pair_query * pair_half.flatten(1)).sum(dim=1).clamp_min(1.0e-8)
    pair_coordinate = (numerator / denominator).clamp(-1.0, 1.0)
    pair_proposal = pair_center + pair_coordinate[:, None, None, None] * pair_half
    projected_pair, audit_pair = augmented_box_projection(
        pair_proposal,
        intrinsic,
        pair_query,
        pair_record,
        geometry,
        iterations=int(args.dykstra_iterations),
    )
    pair_outputs["pair_segment/gan_supervised_pair_query"] = projected_pair
    pair_audits["pair_segment/gan_supervised_pair_query"] = audit_pair

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    metrics: dict[str, dict] = {}
    for base_name, base in base_images.items():
        baseline = evaluate(base, truth, lpips_model)
        metrics[f"{base_name}/base"] = baseline
        for strategy_name, by_budget in outputs[base_name].items():
            for budget in budgets:
                name = f"{base_name}/{strategy_name}/b{budget}"
                values = evaluate(by_budget[budget], truth, lpips_model)
                values["delta_psnr_vs_base"] = values["psnr"] - baseline["psnr"]
                values["delta_ssim_vs_base"] = values["ssim"] - baseline["ssim"]
                values["delta_lpips_vs_base"] = values["lpips"] - baseline["lpips"]
                metrics[name] = values
    for name, prediction in pair_outputs.items():
        values = evaluate(prediction, truth, lpips_model)
        reference_name = name.split("/", 1)[0]
        if reference_name in base_images:
            baseline = metrics[f"{reference_name}/base"]
        else:
            baseline = metrics["matched/base"]
        values["delta_psnr_vs_base"] = values["psnr"] - baseline["psnr"]
        values["delta_ssim_vs_base"] = values["ssim"] - baseline["ssim"]
        values["delta_lpips_vs_base"] = values["lpips"] - baseline["lpips"]
        metrics[name] = values

    payload = {
        "status": "ACTIVE_BINARY_QUERY_ORACLE_HEADROOM",
        "validation_only": True,
        "test_split_opened": False,
        "interpretation": (
            "oracle_* strategies use truth to establish physical headroom only; "
            "they are not implementable reconstruction methods"
        ),
        "signed_query_exposure_note": (
            "one signed binary row is physically implemented by a complementary "
            "DMD exposure pair; all strategies use the same signed-row budget"
        ),
        "operator_sha256": geometry.info.rows_sha256,
        "budgets": budgets,
        "metrics": metrics,
        "projection_audits": audits,
        "truth_free_pair_query_audits": pair_audits,
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
