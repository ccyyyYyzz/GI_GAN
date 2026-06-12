from __future__ import annotations

import hashlib
import json

import torch
import torchvision

from .datasets import build_transform, get_val_dataloader
from .phase15r_common import RADEMACHER_METHODS, REPRO_DEBUG, base_config_for, method_dir, write_rows_all_formats


FIELDS = [
    "method_id",
    "split",
    "dataset_name",
    "dataset_root",
    "limit_val_samples",
    "batch_size",
    "torch_version",
    "torchvision_version",
    "transform",
    "sample_count",
    "first_batch_shape",
    "first_10_image_hashes",
    "notes",
]


def hash_tensor(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def inspect_split(method_id: str, split: str) -> dict:
    config = base_config_for(method_id, method_dir(method_id) / "last.pt")
    transform = build_transform(
        int(config["img_size"]),
        dataset_name=config.get("dataset_name", "stl10"),
        train=False,
        use_augmentation=False,
    )
    row = {
        "method_id": method_id,
        "split": split,
        "dataset_name": config.get("dataset_name", ""),
        "dataset_root": config.get("dataset_root", ""),
        "limit_val_samples": config.get("limit_val_samples", ""),
        "batch_size": config.get("batch_size", ""),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "transform": str(transform),
        "sample_count": "",
        "first_batch_shape": "",
        "first_10_image_hashes": "",
        "notes": "",
    }
    try:
        loader = get_val_dataloader(
            dataset_root=config["dataset_root"],
            img_size=int(config["img_size"]),
            batch_size=int(config["batch_size"]),
            num_workers=0,
            limit_val_samples=config.get("limit_val_samples"),
            seed=int(config["seed"]),
            val_split=split,
            pin_memory=False,
            dataset_name=config.get("dataset_name", "stl10"),
            class_filter=config.get("class_filter"),
        )
        row["sample_count"] = len(loader.dataset)
        batch = next(iter(loader))
        x = batch[0]
        row["first_batch_shape"] = tuple(x.shape)
        n = min(10, x.shape[0])
        row["first_10_image_hashes"] = ";".join(hash_tensor(x[i]) for i in range(n))
        row["notes"] = "sample identifiers unavailable; hashes are computed after local transform"
    except Exception as exc:
        row["notes"] = f"failed: {type(exc).__name__}: {exc}"
    return row


def main() -> None:
    rows = []
    for method in RADEMACHER_METHODS:
        for split in ["test", "train"]:
            rows.append(inspect_split(method["method_id"], split))
    write_rows_all_formats(REPRO_DEBUG / "dataset_split_audit", rows, FIELDS)
    print(json.dumps({"rows": len(rows), "output": str(REPRO_DEBUG / "dataset_split_audit.csv")}, indent=2))


if __name__ == "__main__":
    main()
