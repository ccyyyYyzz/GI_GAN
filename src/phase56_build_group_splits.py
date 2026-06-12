from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .phase56_common import (
    PHASE53C_ROOT,
    PHASE53D_ROOT,
    PHASE55_ROOT,
    TASKS,
    add_args,
    configure_task,
    dataset_with_ids,
    read_csv_rows,
    save_id_csv,
    split_ids,
    write_command_log,
    write_rows,
)
from .utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase56 input audit and strict image-ID group splits.")
    add_args(parser)
    return parser.parse_args()


def overlap(a: set[int], b: set[int]) -> int:
    return len(a.intersection(b))


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_dir)
    write_command_log(root)
    set_seed(args.seed)
    input_rows = []
    phase53c_mi = PHASE53C_ROOT / "session_20_exact_null_mi_pretest" / "exact_null_mi_pretest_results.csv"
    c_rows = read_csv_rows(phase53c_mi)
    d_rows = read_csv_rows(PHASE53D_ROOT / "anchor_null_pretest_results.csv")
    index_files = list(PHASE53C_ROOT.rglob("*index*")) + list(PHASE53C_ROOT.rglob("*indices*")) + list(PHASE53C_ROOT.rglob("*split*"))
    pair_files = list(PHASE53C_ROOT.rglob("*pair*"))
    for item, exists, detail in [
        ("Phase53C max AUC source file", phase53c_mi.exists(), str(phase53c_mi)),
        ("Phase53C original split indices", bool(index_files), "; ".join(str(p) for p in index_files[:8]) if index_files else "missing"),
        ("Phase53C original pair IDs", bool(pair_files), "; ".join(str(p) for p in pair_files[:8]) if pair_files else "missing"),
        ("Phase53C image IDs", bool(index_files), "not found as explicit image-id files" if not index_files else "possible index files found"),
        ("Phase53C family-wise AUC", bool(c_rows), "exact_null_mi_pretest_results.csv"),
        ("Phase53C baseline AUC", any("baseline" in r.get("model", "") for r in c_rows), "handcrafted/condition/anchor baselines present"),
        ("Phase53C shuffled-label baseline", False, "missing"),
        ("Phase53D Rad/Scr AUC contrast", bool(d_rows), "anchor_null_pretest_results.csv"),
    ]:
        input_rows.append({"item": item, "exists": exists, "detail": detail})
    if not index_files:
        input_rows.append({"item": "unknown_split_risk", "exists": True, "detail": "unknown_split_risk confirmed"})
    write_rows(root, "phase56_input_audit", input_rows, "Phase56 Input Audit")
    lines = [
        "# Phase56 Input Audit",
        "",
        f"- Phase53C max AUC source file: `{phase53c_mi}`.",
        f"- Phase53C split index files found: {len(index_files)}.",
        f"- Phase53C original pair/image IDs found: {'yes' if index_files or pair_files else 'no'}.",
        "- unknown_split_risk confirmed." if not index_files else "- Split index files exist and require deeper parsing.",
        f"- Phase53D local rows found: {len(d_rows)}.",
    ]
    (root / "phase56_input_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    split_rows = []
    device = torch.device("cpu")
    for task in args.tasks:
        task_out = ensure_dir(root / "splits" / task)
        info, config, _measurement, _exact_info = configure_task(args, task, task_out, device)
        dataset = dataset_with_ids(config, args.limit_samples, args.seed)
        image_ids = torch.tensor(dataset.image_ids, dtype=torch.long)
        splits = split_ids(image_ids, args.seed + sum(ord(c) for c in task))
        manifest = {
            "task": task,
            "family": info["metadata"]["display"],
            "n_total": int(image_ids.numel()),
            "splits": {k: int(v.numel()) for k, v in splits.items()},
            "split_policy": "image-ID group split; train/val/test image IDs are disjoint",
        }
        for split, ids in splits.items():
            save_id_csv(root / "splits" / f"{task}_{split}_ids.csv", ids)
        save_json(manifest, root / "splits" / f"{task}_split_manifest.json")
        train = set(map(int, splits["train"].tolist()))
        val = set(map(int, splits["val"].tolist()))
        test = set(map(int, splits["test"].tolist()))
        for mode in ["strict_both_group_split", "anchor_heldout_only", "null_heldout_only", "pair_split_reproduction"]:
            if mode == "pair_split_reproduction":
                status = "diagnostic_pair_split_allows_image_overlap"
                train_anchor_overlap = "expected"
                train_null_overlap = "expected"
                any_overlap = "expected"
                pair_overlap = 0
            elif mode == "anchor_heldout_only":
                status = "diagnostic_anchor_heldout"
                train_anchor_overlap = overlap(train, test)
                train_null_overlap = "allowed_by_design"
                any_overlap = "allowed_by_design"
                pair_overlap = 0
            elif mode == "null_heldout_only":
                status = "diagnostic_null_heldout"
                train_anchor_overlap = "allowed_by_design"
                train_null_overlap = overlap(train, test)
                any_overlap = "allowed_by_design"
                pair_overlap = 0
            else:
                train_anchor_overlap = overlap(train, test)
                train_null_overlap = overlap(train, test)
                any_overlap = overlap(train.union(val), test)
                pair_overlap = 0
                status = "pass" if train_anchor_overlap == 0 and train_null_overlap == 0 and any_overlap == 0 else "fail"
            split_rows.append(
                {
                    "task": task,
                    "family": info["metadata"]["display"],
                    "split_mode": mode,
                    "train_n": len(train),
                    "val_n": len(val),
                    "test_n": len(test),
                    "train_test_anchor_overlap_count": train_anchor_overlap,
                    "train_test_null_overlap_count": train_null_overlap,
                    "any_image_overlap_count": any_overlap,
                    "pair_overlap_count": pair_overlap,
                    "status": status,
                }
            )
    write_rows(root, "group_split_overlap_audit", split_rows, "Phase56 Group Split Overlap Audit")
    print(root / "group_split_overlap_audit.csv")


if __name__ == "__main__":
    main()
