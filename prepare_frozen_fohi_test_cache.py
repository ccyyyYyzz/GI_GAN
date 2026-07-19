from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision import datasets, transforms

import anchor_initialized_vqgan_inversion as ai
import gan_high_quality_gi as hq
import prepare_fiber_rate_caches as prep


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rate", choices=("05", "10"), required=True)
    parser.add_argument("--lane-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    access_marker = args.output_dir / "TEST_ACCESS_STARTED.json"
    cache_path = args.output_dir / "test_cache.pt"
    manifest_path = args.output_dir / "test_cache_manifest.json"
    if access_marker.exists() or cache_path.exists() or manifest_path.exists():
        raise RuntimeError("ONE_SHOT_TEST_CACHE_ALREADY_STARTED")
    started = time.time()
    write_json(
        access_marker,
        {
            "status": "TEST_ACCESS_STARTED",
            "rate": str(args.rate),
            "lane_index": int(args.lane_index),
            "test_split_opened": True,
            "unix_time": started,
        },
    )

    paths = prep.bundle_paths(args.bundle_root, str(args.rate))
    config = prep.load_config(paths["config"], args.dataset_root)
    if str(config["data"]["source_split"]) != "train+unlabeled":
        raise RuntimeError("FROZEN_DEVELOPMENT_SOURCE_MISMATCH")
    device = torch.device("cuda")
    seed = int(args.lane_index)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset, val_dataset, dev_dataset, development_manifest = ai.build_split_datasets(
        config
    )
    development_hashes = set()
    for dataset in (train_dataset, val_dataset, dev_dataset):
        for source_index in dataset.indices:
            development_hashes.add(dataset.raw_hash(int(source_index)))

    img_size = int(config["data"]["img_size"])
    test_base = datasets.STL10(root=str(args.dataset_root), split="test", download=True)
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    test_indices = list(range(len(test_base)))
    test_dataset = hq.IndexedTensorDataset(test_base, test_indices, transform)
    hash_rows = []
    test_hashes = []
    for source_index in test_indices:
        raw_hash = test_dataset.raw_hash(source_index)
        test_hashes.append(raw_hash)
        hash_rows.append(
            {
                "source_index": int(source_index),
                "raw_sha256": raw_hash,
                "label": int(test_base.labels[source_index]),
            }
        )
    overlap = sorted(set(test_hashes) & development_hashes)
    if overlap:
        raise RuntimeError(f"TEST_DEVELOPMENT_RAW_HASH_OVERLAP:{len(overlap)}")

    train_x, _, _ = hq.tensor_dataset_to_matrix(
        train_dataset,
        batch_size=int(config["data"].get("matrix_batch_size", 128)),
    )
    rows_np, operator_manifest = hq.build_structured_operator_rows(
        img_size=img_size,
        total_m=int(config["operator"]["total_m"]),
        dct_rows=int(config["operator"]["dct_rows"]),
        hadamard_rows=int(config["operator"]["hadamard_rows"]),
        random_rows=int(config["operator"]["random_rows"]),
        seed=int(config["operator"]["seed"]),
    )
    measurement = hq.make_measurement_operator(
        rows_np,
        img_size=img_size,
        device=device,
        lambda_solver=float(config["operator"]["lambda_solver"]),
    )
    lmmse = hq.EmpiricalLMMSE.fit(
        train_x,
        rows_np,
        lambda_=float(config["operator"]["lmmse_lambda"]),
    )
    priors = {
        ai.VQAE: ai.load_prior(ai.VQAE, paths["vqae_prior"], config, device),
        ai.VQGAN: ai.load_prior(ai.VQGAN, paths["vqgan_prior"], config, device),
    }
    refiners = {
        ai.VQAE: ai.load_refiner_checkpoint(paths["vqae_refiner"], config, device),
        ai.VQGAN: ai.load_refiner_checkpoint(paths["vqgan_refiner"], config, device),
    }
    pack = prep.generate_split(
        dataset=test_dataset,
        split="heldout",
        config=config,
        measurement=measurement,
        lmmse=lmmse,
        priors=priors,
        refiners=refiners,
        device=device,
    )
    temporary = cache_path.with_suffix(".pt.partial")
    torch.save(pack, temporary)
    temporary.replace(cache_path)
    manifest = {
        "status": "FROZEN_FOHI_TEST_CACHE_COMPLETE",
        "evaluation_scope": "heldout",
        "validation_only": False,
        "test_split_opened": True,
        "source_split": "test",
        "test_images": int(len(test_indices)),
        "rate": str(args.rate),
        "lane_index": int(args.lane_index),
        "operator_manifest": operator_manifest,
        "config_sha256": sha256(paths["config"]),
        "checkpoint_sha256": {
            key: sha256(path)
            for key, path in paths.items()
            if key != "config"
        },
        "development_manifest": development_manifest,
        "development_raw_hash_count": int(len(development_hashes)),
        "test_development_raw_hash_overlap": 0,
        "test_duplicate_raw_hashes": int(len(test_hashes) - len(set(test_hashes))),
        "test_source_indices_sha256": hashlib.sha256(
            np.asarray(test_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "test_raw_hash_sequence_sha256": hashlib.sha256(
            "\n".join(test_hashes).encode("ascii")
        ).hexdigest(),
        "test_samples": hash_rows,
        "cache_sha256": sha256(cache_path),
        "cache_shapes": {key: list(value.shape) for key, value in pack.items()},
        "runtime_seconds": time.time() - started,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {cache_path}", flush=True)
    print(f"WROTE {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
