from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import gan_high_quality_gi as hq
from diagnose_afrb_proposal_headroom import project_predictions
from diagnose_vqgan_causal_disagreement_controls import metric_vectors, paired_summary
from src.fiber_residual_phase_gan import FiberResidualPhaseGenerator
from src.gauge_geometry import GaugeGeometry
from train_fiber_residual_phase_gan import predict_all
from train_vqae_centered_residual_adapter import metric_means, prepare_split


def smooth_radial_high_pass(
    image: torch.Tensor,
    *,
    cutoff: float,
    transition: float = 0.03,
) -> torch.Tensor:
    height, width = image.shape[-2:]
    fy = torch.fft.fftfreq(height, device=image.device, dtype=torch.float32)
    fx = torch.fft.rfftfreq(width, device=image.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    radius = radius / radius.max().clamp_min(1.0e-12)
    mask = torch.sigmoid((radius - float(cutoff)) / max(float(transition), 1.0e-6))
    spectrum = torch.fft.rfft2(image.float(), norm="ortho")
    return torch.fft.irfft2(
        spectrum * mask[None, None],
        s=(height, width),
        norm="ortho",
    ).to(image.dtype)


def load_generator(path: Path, device: torch.device) -> tuple[FiberResidualPhaseGenerator, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FiberResidualPhaseGenerator(
        channels=int(checkpoint["channels"]),
        maximum_weight=0.35,
        initial_weight=0.10,
        rotation_scale=float(checkpoint["rotation_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model, {
        key: checkpoint[key]
        for key in (
            "source_arm",
            "rotation_scale",
            "adversarial_weight",
            "lpips_weight",
            "step",
            "seed",
            "channels",
        )
    }


def normalized_joint_score(candidate: dict[str, float], reference: dict[str, float]) -> float:
    normalized = np.asarray(
        [
            (candidate["psnr"] - reference["psnr"]) / 0.02,
            (candidate["ssim"] - reference["ssim"]) / 0.0005,
            (reference["lpips"] - candidate["lpips"]) / 0.005,
        ],
        dtype=np.float64,
    )
    return float(normalized.sum() + 100.0 * np.minimum(normalized, 0.0).sum())


def dominates(candidate: dict[str, float], reference: dict[str, float]) -> bool:
    return bool(
        candidate["psnr"] > reference["psnr"]
        and candidate["ssim"] > reference["ssim"]
        and candidate["lpips"] < reference["lpips"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-val", type=Path, required=True)
    parser.add_argument("--control-val", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--proposal-checkpoints", type=str)
    parser.add_argument("--gan-checkpoints", type=str)
    parser.add_argument("--cutoffs", default="0.08,0.12,0.18,0.25,0.35")
    parser.add_argument("--alphas", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--top-exact", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda")
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows_np, manifest = hq.build_structured_operator_rows(
        img_size=int(config["data"]["img_size"]),
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    geometry = GaugeGeometry.from_rows_qr(torch.from_numpy(rows_np).to(torch.float64)).to(device)
    if geometry.info.rows_sha256 != manifest["rows_sha256"]:
        raise RuntimeError("OPERATOR_IDENTITY_MISMATCH")

    primary = torch.load(args.primary_val, map_location="cpu", weights_only=False)
    control_source = torch.load(args.control_val, map_location="cpu", weights_only=False)
    gan_split = prepare_split(
        primary,
        primary,
        geometry,
        arm="gan",
        batch_size=int(args.batch_size),
        device=device,
    )
    control_split = prepare_split(
        primary,
        control_source,
        geometry,
        arm="vqae_control",
        batch_size=int(args.batch_size),
        device=device,
    )
    for key in ("truth", "anchor", "base", "intrinsic", "source_index"):
        if not torch.equal(gan_split[key], control_split[key]):
            raise RuntimeError(f"PREPARED_SPLIT_MISMATCH:{key}")
    indices = torch.arange(len(gan_split["truth"]))

    control_model, control_manifest = load_generator(args.control_checkpoint, device)
    control_prediction, control_correction, control_audit = predict_all(
        control_model,
        control_split,
        geometry,
        indices=indices,
        batch_size=int(args.batch_size),
        device=device,
    )
    control_exact, control_projection_audit = project_predictions(
        control_prediction.to(device),
        control_split["intrinsic"].to(device),
        geometry,
        batch_size=int(args.batch_size),
    )

    lpips_model = hq.load_lpips(device)
    if isinstance(lpips_model, dict):
        raise RuntimeError(f"LPIPS_UNAVAILABLE:{lpips_model}")
    lpips_model.eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)
    truth = gan_split["truth"].to(device)
    base_vectors = metric_vectors(gan_split["base"].to(device), truth, lpips_model)
    control_vectors = metric_vectors(control_exact, truth, lpips_model)
    base_means = metric_means(base_vectors)
    control_means = metric_means(control_vectors)

    cutoffs = [float(value) for value in args.cutoffs.split(",")]
    alphas = [float(value) for value in args.alphas.split(",")]
    checkpoint_text = args.proposal_checkpoints or args.gan_checkpoints
    if not checkpoint_text:
        raise ValueError("PROPOSAL_CHECKPOINTS_REQUIRED")
    proposal_paths = [Path(value) for value in checkpoint_text.split(",")]
    coarse = []
    cached_proposals: list[tuple[Path, dict, torch.Tensor, dict[str, float]]] = []
    for proposal_index, path in enumerate(proposal_paths):
        proposal_model, proposal_manifest = load_generator(path, device)
        active_split = (
            gan_split if proposal_manifest["source_arm"] == "gan" else control_split
        )
        _, proposal_correction, proposal_audit = predict_all(
            proposal_model,
            active_split,
            geometry,
            indices=indices,
            batch_size=int(args.batch_size),
            device=device,
        )
        cached_proposals.append(
            (path, proposal_manifest, proposal_correction, proposal_audit)
        )
        difference = proposal_correction.to(device) - control_correction.to(device)
        for cutoff in cutoffs:
            high_difference = smooth_radial_high_pass(difference, cutoff=float(cutoff))
            for alpha in alphas:
                raw = control_correction.to(device) + float(alpha) * high_difference
                correction = geometry.null_project_flat(raw.flatten(1)).reshape_as(raw)
                prediction = (gan_split["base"].to(device) + correction).clamp(0.0, 1.0)
                vectors = metric_vectors(prediction, truth, lpips_model)
                means = metric_means(vectors)
                coarse.append(
                    {
                        "proposal_index": proposal_index,
                        "proposal_checkpoint": path.name,
                        "proposal_arm": proposal_manifest["source_arm"],
                        "cutoff": float(cutoff),
                        "alpha": float(alpha),
                        "means": means,
                        "score_vs_control": normalized_joint_score(means, control_means),
                        "dominates_control_coarse": dominates(means, control_means),
                    }
                )
    coarse.sort(key=lambda row: float(row["score_vs_control"]), reverse=True)
    exact_results = []
    for rank, row in enumerate(coarse[: int(args.top_exact)]):
        _, proposal_manifest, proposal_correction, proposal_audit = cached_proposals[
            int(row["proposal_index"])
        ]
        difference = proposal_correction.to(device) - control_correction.to(device)
        high_difference = smooth_radial_high_pass(difference, cutoff=float(row["cutoff"]))
        raw = control_correction.to(device) + float(row["alpha"]) * high_difference
        correction = geometry.null_project_flat(raw.flatten(1)).reshape_as(raw)
        proposal = gan_split["base"].to(device) + correction
        exact, projection_audit = project_predictions(
            proposal,
            gan_split["intrinsic"].to(device),
            geometry,
            batch_size=int(args.batch_size),
        )
        vectors = metric_vectors(exact, truth, lpips_model)
        means = metric_means(vectors)
        exact_results.append(
            {
                **row,
                "coarse_rank": rank,
                "means": means,
                "score_vs_control": normalized_joint_score(means, control_means),
                "dominates_control": dominates(means, control_means),
                "paired_vs_control": paired_summary(
                    vectors,
                    control_vectors,
                    bootstrap_reps=int(args.bootstrap_reps),
                    seed=int(args.seed) + 1009 * rank,
                ),
                "paired_vs_vqae": paired_summary(
                    vectors,
                    base_vectors,
                    bootstrap_reps=int(args.bootstrap_reps),
                    seed=int(args.seed) + 2003 * rank,
                ),
                "projection_audit": projection_audit,
                "proposal_manifest": proposal_manifest,
                "proposal_model_audit": proposal_audit,
            }
        )
    exact_results.sort(key=lambda row: float(row["score_vs_control"]), reverse=True)

    payload = {
        "status": "FIBER_RESIDUAL_FREQUENCY_FUSION_SCREEN",
        "validation_only": True,
        "test_split_opened": False,
        "operator_sha256": geometry.info.rows_sha256,
        "validation_images": len(gan_split["truth"]),
        "base_means": base_means,
        "control_means": control_means,
        "control_manifest": control_manifest,
        "control_model_audit": control_audit,
        "control_projection_audit": control_projection_audit,
        "coarse_candidates": coarse,
        "exact_candidates": exact_results,
        "any_exact_dominates_control": any(
            bool(row["dominates_control"]) for row in exact_results
        ),
        "runtime_seconds": time.time() - started,
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()
