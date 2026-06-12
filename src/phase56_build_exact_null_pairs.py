from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .metrics import batch_metrics
from .phase56_common import (
    NEGATIVE_TYPES,
    SPLIT_MODES,
    add_args,
    build_nearest_indices,
    choose_negative,
    configure_task,
    dataset_with_ids,
    file_sha256,
    project_null,
    resolve_device,
    same_class_choice,
    write_command_log,
    write_rows,
)
from .phase53C_exact_projector import build_rowspace_basis
from .utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase56 exact P0/anchor feature cache and pair metadata.")
    add_args(parser)
    return parser.parse_args()


def load_ids(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {int(r["image_id"]) for r in csv.DictReader(f)}


def write_pair_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "pair_id",
        "split",
        "split_mode",
        "negative_type",
        "alpha",
        "anchor_image_id",
        "null_image_id",
        "anchor_local_idx",
        "null_local_idx",
        "label",
        "family",
        "anchor_class_label",
        "null_class_label",
        "source_anchor_index",
        "source_null_index",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_rows_for_split(task: str, family: str, split: str, split_mode: str, negative_type: str, split_locals: dict[str, list[int]], ids: torch.Tensor, labels: torch.Tensor, anchor: torch.Tensor) -> list[dict]:
    if split_mode == "pair_split_reproduction":
        pool = split_locals["all"]
        active = pool
    elif split_mode == "anchor_heldout_only":
        active = split_locals[split]
        pool = split_locals["train"] if split == "test" else split_locals[split]
    elif split_mode == "null_heldout_only":
        active = split_locals["train"] if split == "test" else split_locals[split]
        pool = split_locals[split]
    else:
        active = split_locals[split]
        pool = split_locals[split]
    nearest = build_nearest_indices(anchor, pool)
    same_cls = same_class_choice(labels, pool)
    rows: list[dict] = []
    pair_i = 0
    for local_idx in active:
        null_idx = local_idx if split_mode != "null_heldout_only" or split != "test" else pool[pair_i % len(pool)]
        rows.append(
            {
                "pair_id": f"{task}_{split_mode}_{negative_type}_{split}_{pair_i:06d}_pos",
                "split": split,
                "split_mode": split_mode,
                "negative_type": negative_type,
                "alpha": "",
                "anchor_image_id": int(ids[local_idx]),
                "null_image_id": int(ids[null_idx]),
                "anchor_local_idx": int(local_idx),
                "null_local_idx": int(null_idx),
                "label": 1,
                "family": family,
                "anchor_class_label": int(labels[local_idx]),
                "null_class_label": int(labels[null_idx]),
                "source_anchor_index": int(ids[local_idx]),
                "source_null_index": int(ids[null_idx]),
            }
        )
        if negative_type == "alpha_chimera":
            neg_base = choose_negative(local_idx, pool, labels, nearest, same_cls, "random")
            for alpha in [0.0, 0.25, 0.5, 0.75]:
                pair_i += 1
                rows.append(
                    {
                        "pair_id": f"{task}_{split_mode}_{negative_type}_{split}_{pair_i:06d}_alpha{alpha}",
                        "split": split,
                        "split_mode": split_mode,
                        "negative_type": negative_type,
                        "alpha": alpha,
                        "anchor_image_id": int(ids[local_idx]),
                        "null_image_id": int(ids[neg_base]),
                        "anchor_local_idx": int(local_idx),
                        "null_local_idx": int(neg_base),
                        "label": 0,
                        "family": family,
                        "anchor_class_label": int(labels[local_idx]),
                        "null_class_label": int(labels[neg_base]),
                        "source_anchor_index": int(ids[local_idx]),
                        "source_null_index": int(ids[neg_base]),
                    }
                )
        else:
            neg_idx = choose_negative(local_idx, pool, labels, nearest, same_cls, negative_type)
            pair_i += 1
            rows.append(
                {
                    "pair_id": f"{task}_{split_mode}_{negative_type}_{split}_{pair_i:06d}_neg",
                    "split": split,
                    "split_mode": split_mode,
                    "negative_type": negative_type,
                    "alpha": "",
                    "anchor_image_id": int(ids[local_idx]),
                    "null_image_id": int(ids[neg_idx]),
                    "anchor_local_idx": int(local_idx),
                    "null_local_idx": int(neg_idx),
                    "label": 0,
                    "family": family,
                    "anchor_class_label": int(labels[local_idx]),
                    "null_class_label": int(labels[neg_idx]),
                    "source_anchor_index": int(ids[local_idx]),
                    "source_null_index": int(ids[neg_idx]),
                }
            )
        pair_i += 1
    return rows


def assign_pair_level_split(rows: list[dict], seed: int) -> list[dict]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(rows), generator=gen).tolist()
    n = len(rows)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    split_for = {}
    for pos, idx in enumerate(perm):
        if pos < n_train:
            split_for[idx] = "train"
        elif pos < n_train + n_val:
            split_for[idx] = "val"
        else:
            split_for[idx] = "test"
    out = []
    for idx, row in enumerate(rows):
        new = dict(row)
        new["split"] = split_for[idx]
        new["pair_id"] = str(new["pair_id"]).replace("_all_", f"_{new['split']}_")
        out.append(new)
    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_dir)
    write_command_log(root)
    device = resolve_device(args.device)
    set_seed(args.seed)
    feature_root = ensure_dir(root / "features")
    pairs_root = ensure_dir(root / "pairs")
    consistency_rows = []
    manifest = {"pairs": []}
    for task in args.tasks:
        task_out = ensure_dir(root / "features" / task)
        info, config, measurement, exact_info = configure_task(args, task, task_out, device)
        dataset = dataset_with_ids(config, args.limit_samples, args.seed)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        A = measurement.get_current_A().detach().float().to(device)
        Q = build_rowspace_basis(A)
        xs, anchors, p0s, ids, labels, ys = [], [], [], [], [], []
        rel_anchor, p0_anchor_energy, bp_psnr_vals = [], [], []
        for x, label, image_id, local_idx in loader:
            x = x.to(device, non_blocking=True)
            y = measurement.measure(x)
            flat = measurement.flatten_img(x.float())
            anchor_flat = measurement.data_solution(y.float(), mode=config.get("backprojection_mode", "ridge_pinv"))
            p0_flat = project_null(flat, Q)
            anchor_img = measurement.unflatten_img(anchor_flat)
            p0_img = measurement.unflatten_img(p0_flat)
            xs.append(x.detach().cpu())
            anchors.append(anchor_img.detach().cpu())
            p0s.append(p0_img.detach().cpu())
            ids.append(image_id.long())
            labels.append(label.long())
            ys.append(y.detach().cpu())
            rel = torch.linalg.norm(measurement.A_forward(anchor_flat) - y.float(), dim=1) / torch.linalg.norm(y.float(), dim=1).clamp_min(1e-12)
            rel_anchor.extend(rel.detach().cpu().tolist())
            p0_anchor_energy.extend((torch.linalg.norm(project_null(anchor_flat, Q), dim=1) / torch.linalg.norm(anchor_flat, dim=1).clamp_min(1e-12)).detach().cpu().tolist())
            bp_psnr_vals.append(batch_metrics(anchor_img.clamp(0, 1), x.clamp(0, 1), measurement, y)["psnr"])
        data = {
            "x": torch.cat(xs),
            "anchor": torch.cat(anchors),
            "p0": torch.cat(p0s),
            "image_ids": torch.cat(ids),
            "labels": torch.cat(labels),
            "y": torch.cat(ys),
            "task": task,
            "family": info["metadata"]["display"],
        }
        feature_path = feature_root / f"{task}_features.pt"
        torch.save(data, feature_path)
        qtq = torch.linalg.norm(Q.T @ Q - torch.eye(Q.shape[1], device=device)).item()
        probe = torch.randn(min(64, A.shape[1]), A.shape[1], device=device)
        ap0 = (torch.linalg.norm(A @ project_null(probe, Q).T) / torch.linalg.norm(probe).clamp_min(1e-12)).item()
        row_gram = torch.linalg.norm(A @ A.T - torch.eye(A.shape[0], device=device)).item()
        consistency_rows.append(
            {
                "task": task,
                "family": info["metadata"]["display"],
                "A_P0_probe_relative": ap0,
                "QTQ_minus_I": qtq,
                "Axdata_minus_y_rel_mean": sum(rel_anchor) / max(1, len(rel_anchor)),
                "P0_xdata_energy_ratio_mean": sum(p0_anchor_energy) / max(1, len(p0_anchor_energy)),
                "BP_anchor_PSNR_mean": sum(bp_psnr_vals) / max(1, len(bp_psnr_vals)),
                "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                "exact_A_sha256": file_sha256(info["exact_A_path"]) if info.get("exact_A_path") else "",
                "row_gram_minus_I": row_gram,
                "scr_anchor_mismatch_status": "ok" if (info["metadata"]["requires_exact_A"] or row_gram < 1e-2) else "check_scr_normalization",
            }
        )
        train_ids = load_ids(root / "splits" / f"{task}_train_ids.csv")
        val_ids = load_ids(root / "splits" / f"{task}_val_ids.csv")
        test_ids = load_ids(root / "splits" / f"{task}_test_ids.csv")
        image_ids = data["image_ids"]
        split_locals = {
            "train": [i for i, v in enumerate(image_ids.tolist()) if int(v) in train_ids],
            "val": [i for i, v in enumerate(image_ids.tolist()) if int(v) in val_ids],
            "test": [i for i, v in enumerate(image_ids.tolist()) if int(v) in test_ids],
            "all": list(range(image_ids.numel())),
        }
        for split_mode in SPLIT_MODES:
            for neg in NEGATIVE_TYPES:
                if split_mode == "pair_split_reproduction":
                    rows = make_rows_for_split(task, info["metadata"]["display"], "all", split_mode, neg, split_locals, image_ids, data["labels"], data["anchor"])
                    rows = assign_pair_level_split(rows, args.seed + sum(ord(c) for c in task + neg))
                else:
                    rows = []
                    for split in ["train", "val", "test"]:
                        rows.extend(make_rows_for_split(task, info["metadata"]["display"], split, split_mode, neg, split_locals, image_ids, data["labels"], data["anchor"]))
                path = pairs_root / f"{task}_{split_mode}_{neg}_pairs.csv"
                write_pair_csv(path, rows)
                manifest["pairs"].append({"task": task, "split_mode": split_mode, "negative_type": neg, "path": str(path), "rows": len(rows)})
    write_rows(root, "projector_anchor_consistency", consistency_rows, "Phase56 Projector Anchor Consistency")
    save_json(manifest, root / "pairs_manifest.json")
    print(root / "projector_anchor_consistency.csv")


if __name__ == "__main__":
    main()
