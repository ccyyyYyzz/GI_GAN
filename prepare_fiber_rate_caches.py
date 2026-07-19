from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import anchor_initialized_vqgan_inversion as ai
import gan_high_quality_gi as hq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def bundle_paths(bundle_root: Path, rate: str) -> dict[str, Path]:
    return {
        "config": require_file(bundle_root / f"config_rate{rate}.yaml"),
        "vqae_prior": require_file(bundle_root / "priors/vqae.pt"),
        "vqgan_prior": require_file(bundle_root / "priors/vqgan.pt"),
        "vqae_refiner": require_file(bundle_root / f"rate{rate}/vqae_refiner.pt"),
        "vqgan_refiner": require_file(bundle_root / f"rate{rate}/vqgan_refiner.pt"),
    }


def load_config(path: Path, dataset_root: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["data"]["dataset_root"] = str(dataset_root)
    if str(config["data"].get("source_split", "")) != "train+unlabeled":
        raise RuntimeError("DEVELOPMENT_SOURCE_MUST_BE_TRAIN_PLUS_UNLABELED")
    if int(config["data"]["val_count"]) != 512 or int(config["data"]["dev_count"]) != 512:
        raise RuntimeError("EXPECTED_512_IMAGE_DEV_AND_VALIDATION_SPLITS")
    return config


def compatible_data_config(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "source_split",
        "hash_clean",
        "img_size",
        "train_count",
        "val_count",
        "dev_count",
    )
    return all(left["data"].get(key) == right["data"].get(key) for key in keys)


@torch.no_grad()
def generate_split(
    *,
    dataset,
    split: str,
    config: dict[str, Any],
    measurement,
    lmmse: hq.EmpiricalLMMSE,
    priors: dict[str, ai.PriorPack],
    refiners: dict[str, ai.AnchorLatentRefiner],
    device: torch.device,
    bucket_snr_db: float | None = None,
    noise_seed: int | None = None,
) -> dict[str, torch.Tensor]:
    loader = hq.build_loader(
        dataset,
        batch_size=int(config["data"].get("eval_batch_size", 16)),
        workers=0,
        shuffle=False,
        seed=int(config["seed"]) + (10 if split == "val" else 11),
        device=device,
    )
    distance_temperature = float(config["training"].get("distance_temperature", 1.0))
    soft_temperature = float(config["training"].get("soft_temperature", 1.0))

    def refine(kind: str, x0: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        prior = priors[kind]
        z0 = prior.model.encode(x0)
        delta_z, delta_logits = refiners[kind](x0, uncertainty, z0)
        logits = (
            ai.logits_from_latent(
                z0 + delta_z,
                prior,
                distance_temperature=distance_temperature,
            )
            + delta_logits
        )
        zq, _, _ = ai.quantize_from_logits(
            prior,
            logits,
            soft_temperature=soft_temperature,
            straight_through=False,
        )
        return prior.model.decode_embeddings(zq)

    fields: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "x0",
            "x_A",
            "x_G",
            "y",
            "y_clean",
            "truth",
            "source_index",
            "label",
        )
    }
    noise_generator = None
    if bucket_snr_db is not None:
        if noise_seed is None:
            raise RuntimeError("NOISE_SEED_REQUIRED_WHEN_BUCKET_SNR_IS_SET")
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(int(noise_seed))
    for truth, label, source_index in loader:
        truth = truth.to(device, non_blocking=True)
        y_clean = measurement.A_forward(measurement.flatten_img(truth))
        y = y_clean
        if bucket_snr_db is not None:
            signal_rms = torch.linalg.vector_norm(y_clean, dim=1, keepdim=True) / np.sqrt(
                float(y_clean.shape[1])
            )
            noise_std = signal_rms / (10.0 ** (float(bucket_snr_db) / 20.0))
            noise = torch.randn(
                y_clean.shape,
                generator=noise_generator,
                device=y_clean.device,
                dtype=y_clean.dtype,
            )
            y = y_clean + noise_std * noise
        x0 = measurement.unflatten_img(lmmse.anchor(y, measurement, device=device))
        uncertainty = lmmse.uncertainty_map(
            img_size=int(config["data"]["img_size"]),
            device=device,
            batch_size=truth.shape[0],
            dtype=truth.dtype,
        )
        fields["x0"].append(x0.cpu())
        fields["x_A"].append(refine(ai.VQAE, x0, uncertainty).cpu())
        fields["x_G"].append(refine(ai.VQGAN, x0, uncertainty).cpu())
        fields["y"].append(y.cpu())
        fields["y_clean"].append(y_clean.cpu())
        fields["truth"].append(truth.cpu())
        fields["source_index"].append(source_index.cpu())
        fields["label"].append(label.cpu())
    return {key: torch.cat(chunks) for key, chunks in fields.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rates", default="02,10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bucket-snr-db", type=float)
    parser.add_argument("--noise-seed-base", type=int, default=20264000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device("cuda")
    started = time.time()
    rates = [part.strip() for part in str(args.rates).split(",") if part.strip()]
    if not rates:
        raise RuntimeError("AT_LEAST_ONE_RATE_REQUIRED")
    paths = {rate: bundle_paths(args.bundle_root, rate) for rate in rates}
    configs = {
        rate: load_config(paths[rate]["config"], args.dataset_root) for rate in rates
    }
    reference_config = configs[rates[0]]
    if any(
        not compatible_data_config(reference_config, configs[rate]) for rate in rates[1:]
    ):
        raise RuntimeError("RATE_DATA_SPLIT_CONFIG_MISMATCH")

    train_dataset, val_dataset, dev_dataset, split_manifest = ai.build_split_datasets(
        reference_config
    )
    train_x, _, _ = hq.tensor_dataset_to_matrix(
        train_dataset,
        batch_size=int(reference_config["data"].get("matrix_batch_size", 128)),
    )
    priors = {
        ai.VQAE: ai.load_prior(ai.VQAE, paths[rates[0]]["vqae_prior"], reference_config, device),
        ai.VQGAN: ai.load_prior(ai.VQGAN, paths[rates[0]]["vqgan_prior"], reference_config, device),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rate_manifests = {}
    for rate in rates:
        rate_started = time.time()
        config = configs[rate]
        rows_np, operator_manifest = hq.build_structured_operator_rows(
            img_size=int(config["data"]["img_size"]),
            total_m=int(config["operator"]["total_m"]),
            dct_rows=int(config["operator"]["dct_rows"]),
            hadamard_rows=int(config["operator"]["hadamard_rows"]),
            random_rows=int(config["operator"]["random_rows"]),
            seed=int(config["operator"]["seed"]),
        )
        measurement = hq.make_measurement_operator(
            rows_np,
            img_size=int(config["data"]["img_size"]),
            device=device,
            lambda_solver=float(config["operator"]["lambda_solver"]),
        )
        lmmse = hq.EmpiricalLMMSE.fit(
            train_x,
            rows_np,
            lambda_=float(config["operator"]["lmmse_lambda"]),
        )
        refiners = {
            ai.VQAE: ai.load_refiner_checkpoint(paths[rate]["vqae_refiner"], config, device),
            ai.VQGAN: ai.load_refiner_checkpoint(paths[rate]["vqgan_refiner"], config, device),
        }
        rate_dir = args.output_dir / f"rate{rate}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        split_shapes = {}
        for split, dataset in (("dev", dev_dataset), ("val", val_dataset)):
            split_noise_seed = (
                int(args.noise_seed_base)
                + 1000 * int(args.seed)
                + (0 if split == "dev" else 1)
            )
            pack = generate_split(
                dataset=dataset,
                split=split,
                config=config,
                measurement=measurement,
                lmmse=lmmse,
                priors=priors,
                refiners=refiners,
                device=device,
                bucket_snr_db=args.bucket_snr_db,
                noise_seed=split_noise_seed,
            )
            output = rate_dir / f"seed{int(args.seed)}_{split}.pt"
            torch.save(pack, output)
            split_shapes[split] = {
                key: list(value.shape) for key, value in pack.items()
            }
            print(f"WROTE {output}", flush=True)
        rate_manifests[rate] = {
            "operator_manifest": operator_manifest,
            "config_sha256": sha256(paths[rate]["config"]),
            "checkpoint_sha256": {
                key: sha256(path)
                for key, path in paths[rate].items()
                if key != "config"
            },
            "split_shapes": split_shapes,
            "runtime_seconds": time.time() - rate_started,
        }
        del refiners, measurement, lmmse
        torch.cuda.empty_cache()
    payload = {
        "status": "FIBER_RATE_CACHE_PREPARATION_COMPLETE",
        "seed": int(args.seed),
        "rates": rates,
        "noise_protocol": {
            "kind": "per_image_white_gaussian_bucket_noise",
            "bucket_snr_db": args.bucket_snr_db,
            "noise_seed_base": int(args.noise_seed_base),
            "interpretation": (
                "algorithmic exact-fit stress test; clean truth is not generally in the noisy equality fiber"
                if args.bucket_snr_db is not None
                else "noiseless exact-fiber reconstruction"
            ),
        },
        "validation_only": True,
        "test_split_opened": False,
        "split_manifest": split_manifest,
        "rate_manifests": rate_manifests,
        "runtime_seconds": time.time() - started,
    }
    manifest_path = args.output_dir / "cache_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

