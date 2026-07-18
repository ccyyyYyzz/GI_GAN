from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

import gan_high_quality_gi as hq
from completion_projector_gan import exact_model_prediction
from diagnose_unpaired_optical_calibration import evaluate, set_seed
from src.gauge_geometry import (
    GaugeEmpiricalAnchor,
    GaugeGeometry,
    project_box_fiber_exact_dual,
)
from src.projector_gated_fiber_gan import (
    FiberConditionalDiscriminator,
    ProjectorGatedFiberGenerator,
)


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
    "E:/GAN_FCC_WORK/experiments/completion_gan_round18/"
    "pqbf_pilot_checkpoint_sweep_seed0/checkpoints/stage_b_step000750.pt"
)
DEFAULT_OUTPUT = Path(
    "E:/GAN_FCC_WORK/experiments/gan_gi_journal_round25/"
    "fiber_critic_score_refinement"
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


def load_discriminator(path: Path, device: torch.device) -> FiberConditionalDiscriminator:
    payload = torch.load(path, map_location=device)
    model = FiberConditionalDiscriminator().to(device)
    model.load_state_dict(payload["discriminator"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def critic_refine_raw(
    start: torch.Tensor,
    anchor: torch.Tensor,
    discriminator: FiberConditionalDiscriminator,
    geometry: GaugeGeometry,
    *,
    step_size: float,
    steps: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    current = start.detach()
    score_before = float(discriminator(anchor, current).mean().detach().cpu())
    null_rms_values = []
    for _ in range(int(steps)):
        current = current.detach().requires_grad_(True)
        score = discriminator(anchor, current).mean()
        gradient = torch.autograd.grad(score, current, create_graph=False)[0]
        null_gradient = geometry.project_feature_maps(gradient, null=True)
        null_rms = null_gradient.flatten(1).square().mean(1).sqrt().clamp_min(1.0e-12)
        normalized = null_gradient / null_rms[:, None, None, None]
        current = current.detach() + float(step_size) * normalized.detach()
        null_rms_values.append(float(null_rms.mean().detach().cpu()))
    score_after_raw = float(discriminator(anchor, current).mean().detach().cpu())
    return current.detach(), {
        "critic_score_before": score_before,
        "critic_score_after_raw": score_after_raw,
        "raw_null_gradient_rms_mean": sum(null_rms_values) / len(null_rms_values),
    }


@torch.no_grad()
def exact_project(
    proposal: torch.Tensor,
    intrinsic: torch.Tensor,
    geometry: GaugeGeometry,
) -> tuple[torch.Tensor, dict[str, float]]:
    result = project_box_fiber_exact_dual(
        proposal.flatten(1).to(torch.float64),
        intrinsic.to(torch.float64),
        geometry,
        record_tolerance=1.0e-10,
        step_tolerance=1.0e-8,
    )
    if not result.converged:
        raise RuntimeError(
            "EXACT_PROJECTION_FAILED:"
            f"{result.max_relative_record_error}:{result.iterations}"
        )
    return result.image_flat.reshape_as(proposal).float(), {
        "projection_iterations": float(result.iterations),
        "relative_record_error": float(result.max_relative_record_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--matched-checkpoint", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--gan-checkpoint", type=Path, default=DEFAULT_GAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--step-sizes", default="0.00025,0.0005,0.001,0.002,0.004,0.008")
    parser.add_argument("--score-steps", default="1,3")
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    set_seed(args.seed)
    started = time.time()
    device = torch.device("cuda")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    step_sizes = [float(value) for value in args.step_sizes.split(",")]
    score_steps = [int(value) for value in args.score_steps.split(",")]

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
    train_truth = train_payload["tensors"]["truth"].flatten(1).numpy()
    anchor_model = GaugeEmpiricalAnchor.fit(
        train_truth,
        geometry,
        lambda_=float(config["operator"].get("lmmse_lambda", 1.0e-3)),
    )
    uncertainty = anchor_model.normalized_posterior_map(
        img_size=int(config["data"]["img_size"]),
        device=device,
    )
    truth = val_payload["tensors"]["truth"].float()
    anchor = val_payload["tensors"]["anchor"].float()
    intrinsic = val_payload["tensors"]["intrinsic"].to(torch.float64)

    matched_model = load_generator(args.matched_checkpoint, geometry, config, device)
    gan_payload = torch.load(args.gan_checkpoint, map_location=device)
    gan_model = load_generator(args.gan_checkpoint, geometry, config, device)
    discriminator = load_discriminator(args.gan_checkpoint, device)
    if int(gan_payload.get("step", -1)) != 750:
        raise RuntimeError(f"UNEXPECTED_GAN_STEP:{gan_payload.get('step')}")

    methods: dict[str, list[torch.Tensor]] = {"matched": [], "gan": []}
    diagnostics: dict[str, list[dict[str, float]]] = {}
    for count in score_steps:
        for size in step_sizes:
            name = f"critic_s{count}_eta{size:g}"
            methods[name] = []
            diagnostics[name] = []

    for start in range(0, truth.shape[0], int(args.batch_size)):
        stop = min(truth.shape[0], start + int(args.batch_size))
        batch_anchor = anchor[start:stop].to(device)
        batch_intrinsic = intrinsic[start:stop].to(device)
        sigma = uncertainty.expand(stop - start, -1, -1, -1)
        matched, _ = exact_model_prediction(
            matched_model,
            anchor=batch_anchor,
            intrinsic=batch_intrinsic,
            uncertainty=sigma,
            geometry=geometry,
        )
        gan, _ = exact_model_prediction(
            gan_model,
            anchor=batch_anchor,
            intrinsic=batch_intrinsic,
            uncertainty=sigma,
            geometry=geometry,
        )
        methods["matched"].append(matched.cpu())
        methods["gan"].append(gan.cpu())
        for count in score_steps:
            for size in step_sizes:
                name = f"critic_s{count}_eta{size:g}"
                proposal, row = critic_refine_raw(
                    matched,
                    batch_anchor,
                    discriminator,
                    geometry,
                    step_size=size,
                    steps=count,
                )
                projected, audit = exact_project(proposal, batch_intrinsic, geometry)
                row.update(audit)
                row["critic_score_after_projected"] = float(
                    discriminator(batch_anchor, projected).mean().detach().cpu()
                )
                diagnostics[name].append(row)
                methods[name].append(projected.cpu())

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    truth_device = truth.to(device)
    metrics = {
        name: evaluate(torch.cat(parts).to(device), truth_device, lpips_model)
        for name, parts in methods.items()
    }
    baseline = metrics["matched"]
    deltas = {
        name: {
            metric: values[metric] - baseline[metric]
            for metric in ["psnr", "ssim", "lpips"]
        }
        for name, values in metrics.items()
        if name not in {"matched", "gan"}
    }
    eligible = [
        name
        for name, values in deltas.items()
        if values["psnr"] >= 0.0
        and values["ssim"] >= 0.0
        and values["lpips"] < 0.0
    ]
    best = min(eligible, key=lambda name: metrics[name]["lpips"]) if eligible else None
    payload = {
        "status": "FIBER_CRITIC_SCORE_REFINEMENT_PILOT",
        "validation_only": True,
        "test_split_opened": False,
        "operator_sha256": geometry.info.rows_sha256,
        "gan_checkpoint_step": int(gan_payload["step"]),
        "metrics": metrics,
        "deltas_vs_matched": deltas,
        "eligible_all_metric_improvement": eligible,
        "selected": best,
        "diagnostics": diagnostics,
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()

