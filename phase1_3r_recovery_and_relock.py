from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import phase1_2_rad5_64_pipeline as p12
import phase1_3_freeze_and_audit as p13
from src.datasets import build_transform
from src.phase1_1_controls import pair_features, sum_image_features
from src.compatibility_model import CompatibilityCritic


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
DATA_ROOT = Path("E:/ns_mc_gan_gi")
PHASE12 = ROOT / "outputs" / "compatibility" / "phase1_2_rad5_64_candidate_transfer"
PHASE13 = ROOT / "outputs" / "compatibility" / "phase1_3_final_locked_eval"
OUT = ROOT / "outputs" / "compatibility" / "phase1_3r_recovery_and_relock"
TRAIN_CACHE = PHASE12 / "candidate_cache" / "train_64_selector_k16.pt"
VAL_CACHE = PHASE12 / "candidate_cache" / "val_64_selector_k16.pt"
FINAL_V1_INDICES = ROOT / "outputs" / "compatibility" / "phase1_1_corrected_rad5" / "reports" / "final_locked_test_indices.npy"
DEV_MANIFEST = PHASE12 / "manifests" / "candidate_pool_dev_64.json"
REPAIR_SALT = "FCC_PHASE1_3R_FINAL_REPAIR_V1"
RANKER_NAMES = [
    "scratch_seed1",
    "scratch_seed2",
    "scratch_seed3",
    "raw_fcc_seed1",
    "raw_fcc_seed2",
    "raw_fcc_seed3",
    "dm_fcc_seed1",
    "dm_fcc_seed2",
    "dm_fcc_seed3",
    "structural_dm_fcc_seed1",
    "structural_dm_fcc_seed2",
    "structural_dm_fcc_seed3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.3R qualified split audit and selector artifact recovery.")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1202)
    parser.add_argument("--roundtrip-torch-artifact")
    parser.add_argument("--roundtrip-sklearn-artifact")
    parser.add_argument("--cache", default=str(VAL_CACHE))
    parser.add_argument("--finalize-existing", action="store_true", help="Re-run audits/readiness from existing recovered artifacts without retraining.")
    return parser.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def save_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_np(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise RuntimeError(f"RECOVERY_CACHE_INVALID: {path} did not load as dict.")
    required = {"r", "cand_n", "p0_error", "indices", "candidate_seed_rows", "k", "img_size"}
    missing = sorted(required - set(obj))
    if missing:
        raise RuntimeError(f"RECOVERY_CACHE_INVALID: {path} missing fields {missing}.")
    if int(obj["k"]) != 16:
        raise RuntimeError(f"RECOVERY_CACHE_INVALID: {path} has k={obj['k']}, expected 16.")
    return obj


def tensor_hash(cache: dict[str, Any], key: str) -> str:
    return hashlib.sha256(cache[key].detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class STL10Lineage:
    def __init__(self, root: Path = DATA_ROOT / "data", img_size: int = 64) -> None:
        self.root = str(root)
        self.transform = build_transform(img_size, dataset_name="stl10", train=False, use_augmentation=False)
        self.train = datasets.STL10(root=self.root, split="train", transform=self.transform, download=False)
        self.unlabeled = datasets.STL10(root=self.root, split="unlabeled", transform=self.transform, download=False)
        self.train_plus = datasets.STL10(root=self.root, split="train+unlabeled", transform=self.transform, download=False)
        self.test = datasets.STL10(root=self.root, split="test", transform=self.transform, download=False)
        self.labeled_count = len(self.train)

    def physical(self, source_namespace: str, integer_index: int) -> tuple[str, int, Any]:
        idx = int(integer_index)
        if source_namespace == "train+unlabeled":
            if idx < self.labeled_count:
                return "stl10/train", idx, self.train
            return "stl10/unlabeled", idx - self.labeled_count, self.unlabeled
        if source_namespace == "test":
            return "stl10/test", idx, self.test
        raise ValueError(f"Unknown source namespace {source_namespace!r}.")

    def sample(self, source_namespace: str, integer_index: int, collection: str, ordinal: int) -> dict[str, Any]:
        official_split, official_index, ds = self.physical(source_namespace, int(integer_index))
        raw = np.ascontiguousarray(ds.data[int(official_index)])
        transformed, _label = ds[int(official_index)]
        raw_hash = sha256_bytes(raw.tobytes())
        transformed_hash = hashlib.sha256(transformed.detach().cpu().contiguous().numpy().astype(np.float32).tobytes()).hexdigest()
        uid = qualified_uid("stl10", official_split, official_index, raw_hash)
        return {
            "collection": collection,
            "dataset_name": "stl10",
            "source_namespace": source_namespace,
            "integer_index": int(integer_index),
            "official_split": official_split,
            "source_index": int(official_index),
            "raw_source_sha256": raw_hash,
            "transformed_64_sha256": transformed_hash,
            "sample_uid": uid,
            "sample_ordinal": int(ordinal),
            "hash_source": "torchvision.datasets.STL10 .data and repository build_transform(64)",
        }

    def all_test_candidates(self) -> list[dict[str, Any]]:
        return [self.sample("test", i, "stl10_official_test_candidate_pool", i) for i in range(len(self.test))]


def qualified_uid(dataset_name: str, official_split: str, source_index: int, raw_hash: str) -> str:
    text = f"{dataset_name}|{official_split}|{int(source_index)}|{raw_hash}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def average_hash_hex(sample: dict[str, Any], lineage: STL10Lineage) -> str:
    _official_split, official_index, ds = lineage.physical(sample["source_namespace"], int(sample["integer_index"]))
    x, _ = ds[int(official_index)]
    img = x[0].numpy().reshape(8, 8, 8, 8).mean(axis=(1, 3))
    bits = (img > img.mean()).astype(np.uint8).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_hex(a: str, b: str) -> int:
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))


def indices_from_cache(cache: dict[str, Any]) -> list[int]:
    return [int(x) for x in cache["indices"].detach().cpu().numpy().astype(np.int64).tolist()]


def dev_indices() -> list[int]:
    obj = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    return [int(row["source_index"]) for row in obj.get("images", [])]


def build_manifest(lineage: STL10Lineage, indices: list[int], namespace: str, collection: str) -> list[dict[str, Any]]:
    return [lineage.sample(namespace, idx, collection, ord_) for ord_, idx in enumerate(indices)]


def count_overlap(a: list[dict[str, Any]], b: list[dict[str, Any]], key: str) -> int:
    return len({row[key] for row in a} & {row[key] for row in b})


def qualified_lineage_audit(out: Path, train_cache: dict[str, Any], val_cache: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    lineage = STL10Lineage()
    train = build_manifest(lineage, indices_from_cache(train_cache), "train+unlabeled", "phase1_2_selector_train")
    val = build_manifest(lineage, indices_from_cache(val_cache), "train+unlabeled", "phase1_2_selector_validation")
    dev = build_manifest(lineage, dev_indices(), "test", "phase1_2_legacy_dev_coverage")
    final_idx = [int(i) for i in np.load(FINAL_V1_INDICES).astype(np.int64).tolist()]
    final_v1 = build_manifest(lineage, final_idx, "test", "phase1_1_2_final_locked_v1")

    save_json(out / "manifests" / "train_qualified_samples.json", train)
    save_json(out / "manifests" / "val_qualified_samples.json", val)
    save_json(out / "manifests" / "dev_qualified_samples.json", dev)
    save_json(out / "manifests" / "final_v1_qualified_samples.json", final_v1)

    naked = sorted(set(row["integer_index"] for row in train) & set(row["integer_index"] for row in final_v1))
    table: list[dict[str, Any]] = []
    train_by_int = {row["integer_index"]: row for row in train}
    final_by_int = {row["integer_index"]: row for row in final_v1}
    for idx in naked:
        tr = train_by_int[idx]
        fn = final_by_int[idx]
        same_uid = tr["sample_uid"] == fn["sample_uid"]
        same_raw = tr["raw_source_sha256"] == fn["raw_source_sha256"]
        same_transformed = tr["transformed_64_sha256"] == fn["transformed_64_sha256"]
        classification = (
            "TRUE_SAMPLE_OVERLAP"
            if same_uid
            else "TRUE_EXACT_IMAGE_DUPLICATE_WITH_DIFFERENT_INDEX"
            if same_raw or same_transformed
            else "FALSE_POSITIVE_INDEX_NAMESPACE_COLLISION"
        )
        table.append(
            {
                "integer_index": idx,
                "train_official_split": tr["official_split"],
                "final_official_split": fn["official_split"],
                "train_source_index": tr["source_index"],
                "final_source_index": fn["source_index"],
                "train_raw_hash": tr["raw_source_sha256"],
                "final_raw_hash": fn["raw_source_sha256"],
                "train_transformed_hash": tr["transformed_64_sha256"],
                "final_transformed_hash": fn["transformed_64_sha256"],
                "same_real_image": bool(same_uid or same_raw or same_transformed),
                "classification": classification,
            }
        )
    write_csv(out / "reports" / "qualified_overlap_table.csv", table)

    collections = {"train": train, "val": val, "dev": dev, "final_v1": final_v1}
    pair_rows: list[dict[str, Any]] = []
    for a_name, a_rows in collections.items():
        for b_name, b_rows in collections.items():
            if a_name >= b_name:
                continue
            pair_rows.append(
                {
                    "a": a_name,
                    "b": b_name,
                    "naked_integer_overlap": len({r["integer_index"] for r in a_rows} & {r["integer_index"] for r in b_rows}),
                    "sample_uid_overlap": count_overlap(a_rows, b_rows, "sample_uid"),
                    "raw_hash_overlap": count_overlap(a_rows, b_rows, "raw_source_sha256"),
                    "transformed_hash_overlap": count_overlap(a_rows, b_rows, "transformed_64_sha256"),
                }
            )

    exact_duplicate_rows: list[dict[str, Any]] = []
    for group_name, rows in [("train", train), ("val", val), ("dev", dev)]:
        for frow in final_v1:
            for grow in rows:
                raw_match = frow["raw_source_sha256"] == grow["raw_source_sha256"]
                transformed_match = frow["transformed_64_sha256"] == grow["transformed_64_sha256"]
                uid_match = frow["sample_uid"] == grow["sample_uid"]
                if raw_match or transformed_match or uid_match:
                    exact_duplicate_rows.append(
                        {
                            "group": group_name,
                            "classification": "TRUE_SAMPLE_OVERLAP" if uid_match else "TRUE_EXACT_IMAGE_DUPLICATE_WITH_DIFFERENT_INDEX",
                            "final_integer_index": frow["integer_index"],
                            "final_official_split": frow["official_split"],
                            "final_source_index": frow["source_index"],
                            "other_integer_index": grow["integer_index"],
                            "other_official_split": grow["official_split"],
                            "other_source_index": grow["source_index"],
                            "raw_hash_match": bool(raw_match),
                            "transformed_hash_match": bool(transformed_match),
                            "sample_uid_match": bool(uid_match),
                            "raw_hash": frow["raw_source_sha256"] if raw_match else "",
                            "transformed_hash": frow["transformed_64_sha256"] if transformed_match else "",
                        }
                    )
    uid_overlap_any = any(row["sample_uid_match"] for row in exact_duplicate_rows)
    exact_duplicate_any = any(row["raw_hash_match"] or row["transformed_hash_match"] for row in exact_duplicate_rows)

    # Auxiliary near-duplicate screen: 64-bit average hash, hamming <= 2.
    near_rows: list[dict[str, Any]] = []
    for group_name, rows in [("train", train), ("val", val), ("dev", dev)]:
        group_hash = [(row, average_hash_hex(row, lineage)) for row in rows]
        final_avg = [(row, average_hash_hex(row, lineage)) for row in final_v1]
        best = 65
        n_near = 0
        for frow, fh in final_avg:
            for grow, gh in group_hash:
                d = hamming_hex(fh, gh)
                best = min(best, d)
                if d <= 2:
                    n_near += 1
                    if len(near_rows) < 25:
                        near_rows.append({"group": group_name, "final_integer_index": frow["integer_index"], "other_integer_index": grow["integer_index"], "hamming": d})
        pair_rows.append({"a": group_name, "b": "final_v1", "near_duplicate_average_hash_hamming_le_2": n_near, "nearest_average_hash_hamming": best})

    if uid_overlap_any:
        audit_status = "TRUE_SAMPLE_OVERLAP"
    elif exact_duplicate_any:
        audit_status = "TRUE_EXACT_IMAGE_DUPLICATE_WITH_DIFFERENT_INDEX"
    else:
        audit_status = "FALSE_POSITIVE_INDEX_NAMESPACE_COLLISION"
    audit = {
        "phase": "Phase1.3R",
        "status": audit_status,
        "naked_integer_overlaps_train_final": naked,
        "naked_integer_overlap_count_train_final": len(naked),
        "qualified_uid_overlap_train_final": count_overlap(train, final_v1, "sample_uid"),
        "raw_hash_overlap_train_final": count_overlap(train, final_v1, "raw_source_sha256"),
        "transformed_hash_overlap_train_final": count_overlap(train, final_v1, "transformed_64_sha256"),
        "pairwise_overlap_summary": pair_rows,
        "three_reported_overlap_details": table,
        "exact_duplicate_details": exact_duplicate_rows,
        "near_duplicate_screen": {"method": "64-bit average hash on transformed 64px image; hamming<=2 auxiliary only", "examples": near_rows},
        "phase1_3_interpretation": "BLOCKED_PREFLIGHT_NO_FINAL_CONSUMPTION",
        "final_candidate_generated": False,
        "final_metrics_computed": False,
    }
    save_json(out / "reports" / "qualified_data_lineage_audit.json", audit)
    return audit, final_v1, {"train": train, "val": val, "dev": dev}


def final_v2_relock(out: Path, audit: dict[str, Any], final_v1: list[dict[str, Any]], dev_sets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if audit["status"] == "INCONCLUSIVE_PROVENANCE_FAILURE":
        result = {"status": "BLOCKED_INCONCLUSIVE_PROVENANCE_FAILURE", "ready": False}
        save_json(out / "reports" / "final_v2_replacement_audit.json", result)
        return result
    old_indices = np.asarray([row["integer_index"] for row in final_v1], dtype=np.int64)
    if audit["status"] == "FALSE_POSITIVE_INDEX_NAMESPACE_COLLISION":
        v2 = {
            "status": "CLEAN_UNSEEN_FINAL_V2",
            "repair_type": "namespace_audit_only",
            "samples_identical_to_v1": True,
            "source_indices_count": int(old_indices.size),
            "source_indices_sha256": sha256_np(old_indices),
            "qualified_sample_uid_sha256": sha256_json([row["sample_uid"] for row in final_v1]),
            "final_test_evaluated": False,
            "final_candidates_generated": False,
            "final_metrics_computed": False,
            "samples": final_v1,
        }
        np.save(out / "manifests" / "final_locked_test_64_v2_indices.npy", old_indices)
        save_json(out / "manifests" / "final_locked_test_64_v2_manifest.json", v2)
        save_json(out / "reports" / "final_v1_retirement_record.json", {"status": "NOT_RETIRED", "reason": "qualified audit found only index namespace collision"})
        repl = {"status": "namespace_audit_only", "replacement_count": 0, "salt": REPAIR_SALT}
        save_json(out / "reports" / "final_v2_replacement_audit.json", repl)
        return v2

    lineage = STL10Lineage()
    used_hashes = set()
    used_uids = set()
    for rows in dev_sets.values():
        used_hashes |= {row["raw_source_sha256"] for row in rows}
        used_hashes |= {row["transformed_64_sha256"] for row in rows}
        used_uids |= {row["sample_uid"] for row in rows}
    clean = [row for row in final_v1 if row["sample_uid"] not in used_uids and row["raw_source_sha256"] not in used_hashes and row["transformed_64_sha256"] not in used_hashes]
    removed = [row for row in final_v1 if row not in clean]
    clean_hashes = {row["raw_source_sha256"] for row in clean} | {row["transformed_64_sha256"] for row in clean}
    clean_uids = {row["sample_uid"] for row in clean}
    candidates = []
    for row in lineage.all_test_candidates():
        if row["sample_uid"] in clean_uids or row["raw_source_sha256"] in clean_hashes or row["transformed_64_sha256"] in clean_hashes:
            continue
        if row["sample_uid"] in used_uids or row["raw_source_sha256"] in used_hashes or row["transformed_64_sha256"] in used_hashes:
            continue
        key = hashlib.sha256(f"{REPAIR_SALT}|stl10|test|{row['integer_index']}".encode("utf-8")).hexdigest()
        candidates.append((key, row))
    need = 512 - len(clean)
    if need < 0 or len(candidates) < need:
        result = {"status": "BLOCKED_FINAL_V2_ELIGIBLE_POOL_INSUFFICIENT", "clean_count": len(clean), "need": need, "eligible": len(candidates)}
        save_json(out / "reports" / "final_v2_replacement_audit.json", result)
        return result
    replacements = [row for _key, row in sorted(candidates, key=lambda t: t[0])[:need]]
    samples = clean + replacements
    idx = np.asarray([row["integer_index"] for row in samples], dtype=np.int64)
    np.save(out / "manifests" / "final_locked_test_64_v2_indices.npy", idx)
    v2 = {
        "status": "CLEAN_UNSEEN_FINAL_V2",
        "repair_type": "deterministic_replacement_after_true_overlap",
        "replacement_salt": REPAIR_SALT,
        "source_indices_count": int(idx.size),
        "source_indices_sha256": sha256_np(idx),
        "final_test_evaluated": False,
        "final_candidates_generated": False,
        "final_metrics_computed": False,
        "removed_count": len(removed),
        "replacement_count": len(replacements),
        "samples": samples,
    }
    save_json(out / "manifests" / "final_locked_test_64_v2_manifest.json", v2)
    save_json(out / "reports" / "final_v1_retirement_record.json", {"status": "RETIRED_BEFORE_EVALUATION_DUE_TO_SPLIT_OVERLAP", "removed": removed})
    save_json(out / "reports" / "final_v2_replacement_audit.json", {"status": "deterministic_repair", "removed_count": len(removed), "replacement_count": len(replacements), "salt": REPAIR_SALT})
    return v2


def feature_matrix_for_cache(cache: dict[str, Any], mode: str) -> tuple[np.ndarray, list[str]]:
    n_img, k, n_pix = cache["cand_n"].shape
    r = cache["r"][:, None, :].repeat(1, k, 1).reshape(n_img * k, n_pix)
    cn = cache["cand_n"].reshape(n_img * k, n_pix)
    if mode == "pair":
        return pair_features(r, cn, int(cache["img_size"]))
    if mode == "sum":
        return sum_image_features(r, cn, int(cache["img_size"]))
    raise ValueError(mode)


def train_scalar_selector_recoverable(train_cache: dict[str, Any], val_cache: dict[str, Any], mode: str) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    xtr, names = feature_matrix_for_cache(train_cache, mode)
    ytr = -train_cache["p0_error"].reshape(-1).numpy()
    xv, _ = feature_matrix_for_cache(val_cache, mode)
    candidates: dict[str, Any] = {}
    for name, model in [
        ("ridge", Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])),
        ("hist_gradient_boosting", HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, random_state=44)),
    ]:
        model.fit(xtr, ytr)
        scores = model.predict(xv).astype(np.float32)
        metrics = p12.evaluate_scores(val_cache, scores, f"{mode}_{name}")
        candidates[name] = {"model": model, "scores": scores, "metrics": metrics}
    selected_name = min(candidates, key=lambda key: candidates[key]["metrics"]["selected_p0_rmse_mean"])
    selected = candidates[selected_name]
    metrics = dict(selected["metrics"])
    metrics["selected_model"] = selected_name
    metrics["feature_count"] = int(len(names))
    return {
        "mode": mode,
        "selected_name": selected_name,
        "selected_model": selected["model"],
        "selected_scores": selected["scores"],
        "selected_metrics": metrics,
        "candidate_models": {k: v["model"] for k, v in candidates.items()},
        "candidate_metrics": {k: v["metrics"] for k, v in candidates.items()},
        "candidate_scores_hashes": {k: hashlib.sha256(v["scores"].tobytes()).hexdigest() for k, v in candidates.items()},
        "feature_names": names,
    }


def score_ranker(model: CompatibilityCritic, cache: dict[str, Any], device: torch.device, mode: str) -> np.ndarray:
    ds = p12.CandidateRankDataset(cache, mode=mode)
    scores_all = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=8, shuffle=False, num_workers=0):
            r = batch["r"].to(device)
            n = batch["n"].to(device)
            b, k = batch["err"].shape
            r_rep = r[:, None].repeat(1, k, 1, 1, 1).reshape(b * k, 1, cache["img_size"], cache["img_size"])
            n_flat = n.reshape(b * k, 1, cache["img_size"], cache["img_size"]).to(device)
            scores_all.append(model.score_pairs(r_rep, n_flat).reshape(b, k).detach().cpu())
    return torch.cat(scores_all, 0).numpy().astype(np.float32)


def train_ranker_recoverable(train_cache: dict[str, Any], val_cache: dict[str, Any], *, device: torch.device, seed: int, pretrain: str | None, structural: bool) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mode = "zscore" if structural else "global"
    model = CompatibilityCritic(embed_dim=128, base_channels=24, temperature=0.07).to(device)
    pre_report = {"kind": "none"}
    if pretrain:
        pre_report = p12.pretrain_counterfactual(model, train_cache, pretrain, device, seed=seed + 99, epochs=1, mode=mode)
    ds = p12.CandidateRankDataset(train_cache, mode=mode)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed + 1))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for _epoch in range(2):
        model.train()
        for batch in loader:
            r = batch["r"].to(device)
            n = batch["n"].to(device)
            err = batch["err"].to(device)
            b, k = err.shape
            r_rep = r[:, None].repeat(1, k, 1, 1, 1).reshape(b * k, 1, train_cache["img_size"], train_cache["img_size"])
            n_flat = n.reshape(b * k, 1, train_cache["img_size"], train_cache["img_size"])
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                scores = model.score_pairs(r_rep, n_flat).reshape(b, k)
                q = torch.softmax(-err / err.std(dim=1, keepdim=True).clamp_min(1e-4), dim=1)
                loss = -(q * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
    scores = score_ranker(model, val_cache, device, mode)
    method = ("structural_" if structural else "") + (pretrain or "scratch") + "_dual_ranker"
    metrics = p12.evaluate_scores(val_cache, scores, method)
    model.eval()
    return {"model": model, "scores": scores, "metrics": metrics, "pre_report": pre_report, "mode": mode}


def atomic_torch_artifact(path: Path, payload: dict[str, Any]) -> str:
    ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    loaded = torch.load(tmp, map_location="cpu", weights_only=False)
    if "state_dict" not in loaded:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Artifact verification failed for {path}.")
    os.replace(tmp, path)
    return sha256_file(path)


def atomic_joblib_artifact(path: Path, payload: dict[str, Any]) -> str:
    ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, tmp)
    loaded = joblib.load(tmp)
    if "selected_model" not in loaded:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Artifact verification failed for {path}.")
    os.replace(tmp, path)
    return sha256_file(path)


def cache_integrity(out: Path, train_cache: dict[str, Any], val_cache: dict[str, Any]) -> dict[str, Any]:
    report = {}
    for name, path, cache in [("train", TRAIN_CACHE, train_cache), ("val", VAL_CACHE, val_cache)]:
        report[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "name": cache["name"],
            "tensor_shapes": {k: list(v.shape) for k, v in cache.items() if torch.is_tensor(v)},
            "k": int(cache["k"]),
            "indices_hash_int64": sha256_np(cache["indices"].numpy().astype(np.int64)),
            "candidate_seed_manifest_hash": sha256_json(cache["candidate_seed_rows"]),
            "p0_error_hash": tensor_hash(cache, "p0_error"),
            "r_hash": tensor_hash(cache, "r"),
            "cand_n_hash": tensor_hash(cache, "cand_n"),
        }
    save_json(out / "reports" / "candidate_cache_integrity.json", report)
    return report


def recover_artifacts(out: Path, train_cache: dict[str, Any], val_cache: dict[str, Any], *, seed: int, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_dir = ensure(out / "recovered_selector_artifacts")
    cache_report = cache_integrity(out, train_cache, val_cache)
    source_hash = sha256_json({p.name: sha256_file(p) for p in [Path(__file__), Path("phase1_2_rad5_64_pipeline.py"), ROOT / "src" / "compatibility_model.py"] if p.exists()})
    phase13_manifest = json.loads((PHASE13 / "freeze_bundle" / "phase1_3_freeze_manifest.json").read_text(encoding="utf-8"))
    common_prov = {
        "phase": "Phase1.3R",
        "recovery_note": "fixed-recipe reproduction from Phase1.2 candidate caches; not the lost Phase1.2 checkpoint bytes",
        "train_cache_sha256": cache_report["train"]["sha256"],
        "validation_cache_sha256": cache_report["val"]["sha256"],
        "A_array_content_hash": phase13_manifest.get("A_file_sha256"),
        "A_file_hash": phase13_manifest.get("A_file_sha256"),
        "generator_checkpoint_hash": phase13_manifest.get("generator_checkpoint_sha256"),
        "source_code_hash": source_hash,
        "torch_version": torch.__version__,
        "timestamp": now(),
    }

    recovered: dict[str, Any] = {"rankers": {}, "scalar": {}}
    for run_seed in [1, 2, 3]:
        for name, pre, structural in [
            ("scratch", None, False),
            ("raw_fcc", "raw", False),
            ("dm_fcc", "dm", False),
            ("structural_dm_fcc", "dm", True),
        ]:
            key = f"{name}_seed{run_seed}"
            effective_seed = seed + run_seed * 100 + len(name)
            result = train_ranker_recoverable(train_cache, val_cache, device=device, seed=effective_seed, pretrain=pre, structural=structural)
            path = artifact_dir / f"{key}.pt"
            payload = {
                "state_dict": {k: v.detach().cpu() for k, v in result["model"].state_dict().items()},
                "model_config": {"class": "CompatibilityCritic", "embed_dim": 128, "base_channels": 24, "temperature": 0.07, "learn_temperature": False, "use_joint_mlp": False},
                "training_recipe": {
                    "preprocessing_mode": result["mode"],
                    "structural": bool(structural),
                    "pretraining_type": pre or "none",
                    "exact_effective_seed": effective_seed,
                    "base_phase_seed": seed,
                    "epoch_count": 2,
                    "pretraining_epoch_count": 1 if pre else 0,
                    "optimizer_config": {"optimizer": "AdamW", "lr": 2e-4, "weight_decay": 1e-4, "batch_size": 8, "grad_clip": 1.0, "amp": device.type == "cuda"},
                    "loss": "soft target cross entropy over K candidate P0 errors",
                },
                "validation_metrics": result["metrics"],
                "pretraining_report": result["pre_report"],
                "validation_scores": result["scores"],
                "provenance": {**common_prov, "validation_score_matrix_hash": hashlib.sha256(result["scores"].tobytes()).hexdigest(), "candidate_ordering_hash": sha256_json(val_cache["candidate_seed_rows"])},
            }
            artifact_hash = atomic_torch_artifact(path, payload)
            recovered["rankers"][key] = {"path": str(path), "sha256": artifact_hash, "metrics": result["metrics"], "pre_report": result["pre_report"], "score_hash": payload["provenance"]["validation_score_matrix_hash"]}

    for mode, filename in [("pair", "scalar_pair_selector.joblib"), ("sum", "sum_image_selector.joblib")]:
        result = train_scalar_selector_recoverable(train_cache, val_cache, mode)
        path = artifact_dir / filename
        payload = {
            "mode": mode,
            "selected_model_name": result["selected_name"],
            "selected_model": result["selected_model"],
            "candidate_models": result["candidate_models"],
            "feature_names": result["feature_names"],
            "feature_order": list(result["feature_names"]),
            "validation_metrics": result["selected_metrics"],
            "candidate_metrics": result["candidate_metrics"],
            "validation_scores": result["selected_scores"],
            "candidate_scores_hashes": result["candidate_scores_hashes"],
            "provenance": {**common_prov, "sklearn_version": __import__("sklearn").__version__, "validation_score_matrix_hash": hashlib.sha256(result["selected_scores"].tobytes()).hexdigest()},
        }
        artifact_hash = atomic_joblib_artifact(path, payload)
        recovered["scalar"][mode] = {"path": str(path), "sha256": artifact_hash, "metrics": result["selected_metrics"], "score_hash": payload["provenance"]["validation_score_matrix_hash"]}
    return recovered, common_prov


def roundtrip_torch_artifact(artifact_path: Path, cache_path: Path) -> dict[str, Any]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    cache = load_cache(cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = artifact["model_config"]
    model = CompatibilityCritic(
        embed_dim=int(cfg["embed_dim"]),
        base_channels=int(cfg["base_channels"]),
        temperature=float(cfg["temperature"]),
        learn_temperature=bool(cfg.get("learn_temperature", False)),
        use_joint_mlp=bool(cfg.get("use_joint_mlp", False)),
    )
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    model.eval()
    scores = score_ranker(model, cache, device, artifact["training_recipe"]["preprocessing_mode"])
    saved = np.asarray(artifact["validation_scores"], dtype=np.float32)
    selected = np.argmax(scores, axis=1)
    saved_selected = np.argmax(saved, axis=1)
    return {
        "artifact": str(artifact_path),
        "type": "torch",
        "max_abs_score_difference": float(np.max(np.abs(scores - saved))),
        "selected_indices_identical": bool(np.array_equal(selected, saved_selected)),
        "metric": p12.evaluate_scores(cache, scores, artifact["validation_metrics"]["method"]),
        "model_eval": model.training is False,
        "device": str(device),
        "passed": bool(np.max(np.abs(scores - saved)) <= 1e-5 and np.array_equal(selected, saved_selected) and model.training is False),
    }


def roundtrip_sklearn_artifact(artifact_path: Path, cache_path: Path) -> dict[str, Any]:
    artifact = joblib.load(artifact_path)
    cache = load_cache(cache_path)
    x, names = feature_matrix_for_cache(cache, artifact["mode"])
    scores = artifact["selected_model"].predict(x).astype(np.float32)
    saved = np.asarray(artifact["validation_scores"], dtype=np.float32)
    selected = np.argmax(scores.reshape(cache["p0_error"].shape), axis=1)
    saved_selected = np.argmax(saved.reshape(cache["p0_error"].shape), axis=1)
    return {
        "artifact": str(artifact_path),
        "type": "sklearn",
        "max_abs_prediction_difference": float(np.max(np.abs(scores - saved))),
        "selected_indices_identical": bool(np.array_equal(selected, saved_selected)),
        "feature_order_identical": bool(list(names) == list(artifact["feature_order"])),
        "candidate_index_or_seed_features": [name for name in names if "seed" in name.lower() or "index" in name.lower()],
        "passed": bool(np.max(np.abs(scores - saved)) <= 1e-8 and np.array_equal(selected, saved_selected) and list(names) == list(artifact["feature_order"])),
    }


def run_roundtrip_subprocess(out: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    rows = []
    py = sys.executable
    for key, info in artifacts["rankers"].items():
        cmd = [py, str(Path(__file__).resolve()), "--roundtrip-torch-artifact", info["path"], "--cache", str(VAL_CACHE)]
        res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=True)
        row = json.loads(res.stdout)
        row["name"] = key
        rows.append(row)
    for key, info in artifacts["scalar"].items():
        cmd = [py, str(Path(__file__).resolve()), "--roundtrip-sklearn-artifact", info["path"], "--cache", str(VAL_CACHE)]
        res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=True)
        row = json.loads(res.stdout)
        row["name"] = "scalar_pair_selector" if key == "pair" else "sum_image_selector"
        rows.append(row)
    audit = {
        "status": "PASS" if all(row.get("passed") for row in rows) else "FAIL",
        "max_torch_score_difference": max([row.get("max_abs_score_difference", 0.0) for row in rows] or [0.0]),
        "max_sklearn_prediction_difference": max([row.get("max_abs_prediction_difference", 0.0) for row in rows] or [0.0]),
        "all_selected_indices_identical": all(row.get("selected_indices_identical") for row in rows),
        "candidate_index_or_seed_used_as_feature": any(row.get("candidate_index_or_seed_features") for row in rows),
        "rows": rows,
    }
    save_json(out / "reports" / "artifact_roundtrip_audit.json", audit)
    return audit


def reproduction_report(out: Path, artifacts: dict[str, Any], train_cache: dict[str, Any], val_cache: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    old = json.loads((PHASE12 / "reports" / "gate_report_e2b_64_selector.json").read_text(encoding="utf-8"))
    baselines = p12.evaluate_baselines(val_cache)
    rankers = {k: v["metrics"] for k, v in artifacts["rankers"].items()}
    scalar_pair = artifacts["scalar"]["pair"]["metrics"]
    sum_image = artifacts["scalar"]["sum"]["metrics"]
    directions = [rankers[f"dm_fcc_seed{s}"]["selected_p0_rmse_mean"] < rankers[f"scratch_seed{s}"]["selected_p0_rmse_mean"] for s in [1, 2, 3]]
    best_dual_key = min(rankers, key=lambda k: rankers[k]["selected_p0_rmse_mean"])
    best_dual = rankers[best_dual_key]
    best_natural = min(scalar_pair["selected_p0_rmse_mean"], sum_image["selected_p0_rmse_mean"])
    gate = {
        "best_dual_key": best_dual_key,
        "best_dual_beats_random": best_dual["selected_p0_rmse_mean"] < baselines["random_expectation"]["p0_rmse_mean"],
        "best_dual_oracle_gain_fraction_ge_0_2": best_dual["oracle_gain_fraction_mean"] >= 0.2,
        "dm_fcc_beats_scratch_2_of_3": int(sum(directions)) >= 2,
        "dual_beats_best_naturalness": best_dual["selected_p0_rmse_mean"] < best_natural,
        "k": 16,
    }
    classification = "DM_FCC_ADDS_VALUE" if all([gate["best_dual_beats_random"], gate["dual_beats_best_naturalness"], gate["dm_fcc_beats_scratch_2_of_3"]]) else "REPRODUCTION_NOT_CONFIRMED"
    reproduced = {
        "classification": classification,
        "baselines": baselines,
        "scalar_pair_selector": scalar_pair,
        "sum_image_selector": sum_image,
        "rankers": rankers,
        "selector_gate": gate,
        "pretraining_reports": {k: v["pre_report"] for k, v in artifacts["rankers"].items()},
        "final_locked_test_run": False,
    }
    rows = []

    def add(name: str, old_metrics: dict[str, Any], new_metrics: dict[str, Any]) -> None:
        for metric in ["selected_p0_rmse_mean", "random_expected_p0_rmse_mean", "oracle_p0_rmse_mean", "deterministic_p0_rmse_mean", "posterior_mean_p0_rmse_mean", "oracle_gain_fraction_mean", "top_oracle_hit_rate"]:
            if metric in old_metrics or metric in new_metrics:
                ov = old_metrics.get(metric)
                nv = new_metrics.get(metric)
                rows.append(
                    {
                        "model": name,
                        "metric": metric,
                        "old_value": ov,
                        "reproduced_value": nv,
                        "absolute_difference": None if ov is None or nv is None else abs(float(nv) - float(ov)),
                        "relative_difference": None if ov in (None, 0) or nv is None else abs(float(nv) - float(ov)) / abs(float(ov)),
                        "old_selected_candidate_agreement": "DATA MISSING: Phase1.2 did not persist per-image scores/indices",
                        "score_correlation": "DATA MISSING: Phase1.2 did not persist score matrix",
                    }
                )

    add("scalar_pair_selector", old["scalar_pair_selector"], scalar_pair)
    add("sum_image_selector", old["sum_image_selector"], sum_image)
    for name, metrics in rankers.items():
        add(name, old["rankers"][name], metrics)
    write_csv(out / "reports" / "old_vs_reproduced_validation.csv", rows)
    save_json(out / "reports" / "phase1_2_validation_reproduction.json", reproduced)

    primary_diff = abs(rankers["dm_fcc_seed3"]["selected_p0_rmse_mean"] - old["rankers"]["dm_fcc_seed3"]["selected_p0_rmse_mean"])
    pair_diff = abs(scalar_pair["selected_p0_rmse_mean"] - old["scalar_pair_selector"]["selected_p0_rmse_mean"])
    sum_diff = abs(sum_image["selected_p0_rmse_mean"] - old["sum_image_selector"]["selected_p0_rmse_mean"])
    gate_report = {
        "status": "PASS" if classification == "DM_FCC_ADDS_VALUE" and primary_diff <= 5e-4 and pair_diff <= 5e-4 and sum_diff <= 5e-4 else "REPRODUCTION_NOT_CONFIRMED",
        "classification": classification,
        "primary_model": "reproduced_dm_fcc_seed3_v2",
        "primary_p0_rmse_abs_diff": primary_diff,
        "scalar_pair_abs_diff": pair_diff,
        "sum_image_abs_diff": sum_diff,
        "dm_fcc_beats_scratch_flags": directions,
        "old_best_dual_key": old["selector_gate"]["best_dual_key"],
        "reproduced_best_dual_key": best_dual_key,
        "cache_derived_baselines_exact": all(abs(baselines[k]["p0_rmse_mean"] - old["baselines"][k]["p0_rmse_mean"]) < 1e-8 for k in ["deterministic", "random_expectation", "posterior_mean", "oracle_best_of_k"]),
    }
    save_json(out / "reports" / "reproduction_gate.json", gate_report)
    return reproduced, gate_report


def scientific_status(out: Path) -> None:
    text = """# Phase 1.3R Scientific Status

Phase 1.3 is interpreted as `BLOCKED_PREFLIGHT_NO_FINAL_CONSUMPTION`.

- Stage A was not executed.
- Stage B was not executed.
- No final candidate pool was generated.
- No final P0 RMSE, PSNR, LPIPS, oracle, or selector metric was computed.
- The final test remains unconsumed.

The historical Phase 1.3 `FINAL_EVALUATION_INVALID` files are left unchanged.
Phase 1.3R exists only to audit split identity, recover selector artifacts from
the fixed Phase 1.2 train/validation candidate caches, and prepare a relocked
final-v2 manifest if all gates pass.
"""
    ensure(out / "reports")
    (out / "reports" / "phase1_3r_scientific_status.md").write_text(text, encoding="utf-8")


def artifact_registry(out: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for key, info in artifacts["rankers"].items():
        rows[key] = {"type": "torch_ranker", **info}
    rows["scalar_pair_selector"] = {"type": "sklearn_selector", **artifacts["scalar"]["pair"]}
    rows["sum_image_selector"] = {"type": "sklearn_selector", **artifacts["scalar"]["sum"]}
    save_json(out / "reports" / "artifact_registry_v2.json", rows)
    save_json(out / "reports" / "selector_registry_v2.json", rows)
    return rows


def load_existing_artifacts(out: Path) -> dict[str, Any]:
    artifact_dir = out / "recovered_selector_artifacts"
    rankers: dict[str, Any] = {}
    for key in RANKER_NAMES:
        path = artifact_dir / f"{key}.pt"
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        scores = np.asarray(artifact["validation_scores"], dtype=np.float32)
        rankers[key] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "metrics": artifact["validation_metrics"],
            "pre_report": artifact.get("pretraining_report", {}),
            "score_hash": hashlib.sha256(scores.tobytes()).hexdigest(),
        }
    scalar: dict[str, Any] = {}
    for mode, filename in [("pair", "scalar_pair_selector.joblib"), ("sum", "sum_image_selector.joblib")]:
        path = artifact_dir / filename
        artifact = joblib.load(path)
        scores = np.asarray(artifact["validation_scores"], dtype=np.float32)
        scalar[mode] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "metrics": artifact["validation_metrics"],
            "score_hash": hashlib.sha256(scores.tobytes()).hexdigest(),
        }
    return {"rankers": rankers, "scalar": scalar}


def copy_freeze_bundle(out: Path, ready: dict[str, Any], registry: dict[str, Any]) -> None:
    bundle = ensure(out / "freeze_bundle_v2")
    for rel in [
        "reports/qualified_data_lineage_audit.json",
        "manifests/final_locked_test_64_v2_manifest.json",
        "manifests/final_locked_test_64_v2_indices.npy",
        "reports/artifact_registry_v2.json",
        "reports/selector_registry_v2.json",
        "reports/phase1_2_validation_reproduction.json",
        "reports/artifact_roundtrip_audit.json",
        "reports/candidate_cache_integrity.json",
    ]:
        src = out / rel
        if src.exists():
            shutil.copy2(src, bundle / Path(rel).name)
    prereg = """# Phase 1.3R Preregistration Draft

This bundle is ready for a later Phase 1.4 final evaluation. It is not itself
`FINAL_EVAL_FROZEN.json`, and no final inference or truth-based final metric
has been run in Phase 1.3R.
"""
    (bundle / "phase1_3r_preregistration_draft.md").write_text(prereg, encoding="utf-8")
    save_json(bundle / "READY_FOR_PHASE1_4_FINAL.json", ready)
    source_zip = bundle / "source_snapshot_v2.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [Path(__file__), Path("phase1_2_rad5_64_pipeline.py"), ROOT / "src" / "compatibility_model.py", ROOT / "src" / "phase1_1_controls.py"]:
            if path.exists():
                zf.write(path, arcname=path.name)
    hashes = {
        "checkpoint_hashes_v2": {k: v["sha256"] for k, v in registry.items()},
        "config_hashes_v2": {"phase1_3r_script": sha256_file(Path(__file__))},
        "cache_hashes_v2": {"train_cache": sha256_file(TRAIN_CACHE), "val_cache": sha256_file(VAL_CACHE)},
        "source_file_hashes_v2": {p.name: sha256_file(p) for p in [Path(__file__), Path("phase1_2_rad5_64_pipeline.py")] if p.exists()},
        "dependency_versions_v2": {"torch": torch.__version__, "sklearn": __import__("sklearn").__version__, "numpy": np.__version__},
        "metric_definitions_v2": {"primary_endpoint": "exact canonicalized P0 RMSE", "K": 16},
        "candidate_seed_policy_v2": {"source": "Phase1.2 cache candidate_seed_rows; final candidates not generated"},
        "data_lineage_hashes": {"qualified_data_lineage_audit": sha256_file(out / "reports" / "qualified_data_lineage_audit.json")},
    }
    for name, payload in hashes.items():
        save_json(bundle / f"{name}.json", payload)
    git_status = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--short"], cwd=str(ROOT), text=True, capture_output=True)
    (bundle / "git_status.txt").write_text(git_status.stdout, encoding="utf-8")
    shutil.copy2(out / "reports" / "command_log.txt", bundle / "command_log.txt")


def bundle_hash(bundle: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(bundle)).encode("utf-8"))
        h.update(sha256_file(path).encode("utf-8"))
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    if args.roundtrip_torch_artifact:
        print(json.dumps(roundtrip_torch_artifact(Path(args.roundtrip_torch_artifact), Path(args.cache)), sort_keys=True))
        return 0
    if args.roundtrip_sklearn_artifact:
        print(json.dumps(roundtrip_sklearn_artifact(Path(args.roundtrip_sklearn_artifact), Path(args.cache)), sort_keys=True))
        return 0

    start = time.time()
    out = ensure(Path(args.output_dir))
    ensure(out / "reports")
    ensure(out / "manifests")
    ensure(out / "recovered_selector_artifacts")
    (out / "reports" / "command_log.txt").write_text("$ " + " ".join(sys.argv) + "\n", encoding="utf-8")
    scientific_status(out)
    device = p12.resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    blockers: list[str] = []
    train_cache = load_cache(TRAIN_CACHE)
    val_cache = load_cache(VAL_CACHE)
    if args.finalize_existing:
        lineage_audit = json.loads((out / "reports" / "qualified_data_lineage_audit.json").read_text(encoding="utf-8"))
        final_v2 = json.loads((out / "manifests" / "final_locked_test_64_v2_manifest.json").read_text(encoding="utf-8"))
        artifacts = load_existing_artifacts(out)
        phase13_manifest = json.loads((PHASE13 / "freeze_bundle" / "phase1_3_freeze_manifest.json").read_text(encoding="utf-8"))
        provenance = {
            "A_array_content_hash": phase13_manifest.get("A_file_sha256"),
            "A_file_hash": phase13_manifest.get("A_file_sha256"),
            "generator_checkpoint_hash": phase13_manifest.get("generator_checkpoint_sha256"),
        }
    else:
        lineage_audit, final_v1, dev_sets = qualified_lineage_audit(out, train_cache, val_cache)
        final_v2 = final_v2_relock(out, lineage_audit, final_v1, dev_sets)
        artifacts, provenance = recover_artifacts(out, train_cache, val_cache, seed=args.seed, device=device)
    registry = artifact_registry(out, artifacts)
    roundtrip = run_roundtrip_subprocess(out, artifacts)
    reproduced, gate = reproduction_report(out, artifacts, train_cache, val_cache)

    final_v2_clean = final_v2.get("status") == "CLEAN_UNSEEN_FINAL_V2" and len(final_v2.get("samples", [])) == 512
    all_artifacts = len(registry) == 14 and all(Path(info["path"]).exists() for info in registry.values())
    ready_ok = all(
        [
            lineage_audit["status"] in {"FALSE_POSITIVE_INDEX_NAMESPACE_COLLISION", "TRUE_SAMPLE_OVERLAP", "TRUE_EXACT_IMAGE_DUPLICATE_WITH_DIFFERENT_INDEX"},
            final_v2_clean,
            all_artifacts,
            roundtrip["status"] == "PASS",
            gate["status"] == "PASS",
            final_v2.get("final_test_evaluated") is False,
            final_v2.get("final_candidates_generated") is False,
            final_v2.get("final_metrics_computed") is False,
        ]
    )
    if not final_v2_clean:
        blockers.append("final-v2 manifest is not CLEAN_UNSEEN_FINAL_V2 with 512 samples.")
    if not all_artifacts:
        blockers.append("not all 14 selector artifacts exist.")
    if roundtrip["status"] != "PASS":
        blockers.append("artifact round-trip verification failed.")
    if gate["status"] != "PASS":
        blockers.append("validation reproduction gate failed.")

    ready_payload = {
        "status": "READY_FOR_PHASE1_4_FINAL" if ready_ok else "BLOCKED_PHASE1_3R",
        "primary_model": "reproduced_dm_fcc_seed3_v2",
        "primary_checkpoint_hash": registry.get("dm_fcc_seed3", {}).get("sha256"),
        "all_baseline_hashes": {k: v["sha256"] for k, v in registry.items() if k != "dm_fcc_seed3"},
        "final_v2_manifest_hash": sha256_file(out / "manifests" / "final_locked_test_64_v2_manifest.json") if (out / "manifests" / "final_locked_test_64_v2_manifest.json").exists() else None,
        "A_content_hash": provenance.get("A_array_content_hash"),
        "A_file_hash": provenance.get("A_file_hash"),
        "generator_hash": provenance.get("generator_checkpoint_hash"),
        "K": 16,
        "primary_endpoint": "exact canonicalized P0 RMSE",
        "validation_reproduction_status": gate["status"],
        "final_test_evaluated": False,
        "final_candidates_generated": False,
        "final_metrics_computed": False,
        "blockers": blockers,
    }
    if ready_ok:
        copy_freeze_bundle(out, ready_payload, registry)
        ready_payload["freeze_bundle_hash"] = bundle_hash(out / "freeze_bundle_v2")
        save_json(out / "READY_FOR_PHASE1_4_FINAL.json", ready_payload)
    else:
        save_json(out / "BLOCKED_PHASE1_3R.json", ready_payload)

    if blockers:
        (out / "reports" / "BLOCKERS_PHASE1_3R.md").write_text("# BLOCKERS_PHASE1_3R\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n", encoding="utf-8")
    else:
        (out / "reports" / "BLOCKERS_PHASE1_3R.md").write_text("# BLOCKERS_PHASE1_3R\n\nNo blockers. READY was produced for the next phase; no final evaluation was run.\n", encoding="utf-8")

    runtime = {
        "runtime_seconds": time.time() - start,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "device": str(device),
    }
    save_json(out / "reports" / "runtime_and_memory.json", runtime)
    status = {
        "phase": "Phase1.3R",
        "status": ready_payload["status"],
        "final_v2_status": final_v2.get("status"),
        "lineage_status": lineage_audit["status"],
        "artifact_count": len(registry),
        "roundtrip_status": roundtrip["status"],
        "reproduction_gate_status": gate["status"],
        "final_test_consumed": False,
        "final_candidates_generated": False,
        "final_metrics_computed": False,
        "runtime_seconds": runtime["runtime_seconds"],
        "peak_gpu_memory_bytes": runtime["peak_gpu_memory_bytes"],
    }
    save_json(out / "reports" / "implementation_status_phase1_3r.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
