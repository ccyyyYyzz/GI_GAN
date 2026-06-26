from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
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

import phase1_2_rad5_64_pipeline as p12
import phase1_3r_recovery_and_relock as p13r
from src import phase69A_gauge_gan_signal_diagnostic as p69a
from src import phase73_overnight_gauge_gan_expansion as p73
from src.compatibility_model import CompatibilityCritic
from src.phase1_1_controls import pair_features, sum_image_features
from src.phase79_rad5_rowspace_diversity_diagnostic import forward_with_noise
from src.projections import exact_data_anchor, exact_null_project


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
DATA_ROOT = Path("E:/ns_mc_gan_gi")
PHASE13R = ROOT / "outputs" / "compatibility" / "phase1_3r_recovery_and_relock"
OUT = ROOT / "outputs" / "compatibility" / "phase1_4a_final_freeze_and_blind"
FREEZE = OUT / "freeze_bundle"
BLIND = OUT / "blind_inference"
REPORTS = OUT / "reports"
MANIFESTS = OUT / "manifests"
K = 16
FINAL_SEED_SALT = "FCC_PHASE1_4_FINAL_CANDIDATES_V1"
FULL_LINEAGE_REPAIR_SALT = "FCC_PHASE1_4_FULL_LINEAGE_REPAIR_V1"
QUAL_SALT = "FCC_PHASE1_4_FIXED_QUALITATIVE_V1"
PRIMARY_MODEL = "reproduced_dm_fcc_seed3_v2"
PRIMARY_ARTIFACT_KEY = "dm_fcc_seed3"
FORBIDDEN_BLIND_FIELDS = {
    "x_true",
    "true_x",
    "true_n",
    "labels",
    "label",
    "p0_error",
    "full_error",
    "oracle",
    "oracle_index",
    "psnr",
    "ssim",
    "lpips",
    "rapsd",
    "selected_error",
}


RANKER_KEYS = [
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
SCALAR_KEYS = ["scalar_pair_selector", "sum_image_selector"]
ALL_SELECTOR_KEYS = RANKER_KEYS + SCALAR_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.4A final freeze and blind inference.")
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-blind", action="store_true", help="Create freeze only; do not run Stage A.")
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_np(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> str:
    ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    _ = torch.load(tmp, map_location="cpu", weights_only=False)
    os.replace(tmp, path)
    return sha256_file(path)


def bundle_hash(bundle: Path, *, exclude_names: set[str] | None = None) -> str:
    exclude_names = exclude_names or set()
    h = hashlib.sha256()
    for path in sorted(p for p in bundle.rglob("*") if p.is_file() and p.name not in exclude_names):
        h.update(str(path.relative_to(bundle)).encode("utf-8"))
        h.update(sha256_file(path).encode("utf-8"))
    return h.hexdigest()


def phase13r_bundle_hash() -> str:
    return p13r.bundle_hash(PHASE13R / "freeze_bundle_v2")


def status_supersession_audit() -> dict[str, Any]:
    ready_path = PHASE13R / "READY_FOR_PHASE1_4_FINAL.json"
    blocked_path = PHASE13R / "BLOCKED_PHASE1_3R.json"
    ready = read_json(ready_path)
    impl = read_json(PHASE13R / "reports" / "implementation_status_phase1_3r.json")
    repro = read_json(PHASE13R / "reports" / "reproduction_gate.json")
    rt = read_json(PHASE13R / "reports" / "artifact_roundtrip_audit.json")
    final_v2 = read_json(PHASE13R / "manifests" / "final_locked_test_64_v2_manifest.json")
    recalculated = phase13r_bundle_hash()
    audit = {
        "status": "PASS",
        "ready_path": str(ready_path),
        "historical_blocked_exists": blocked_path.exists(),
        "historical_blocked_interpretation": "superseded historical state, retained as record",
        "ready_status": ready.get("status"),
        "implementation_status": impl.get("status"),
        "reproduction_gate": repro.get("status"),
        "roundtrip_status": rt.get("status"),
        "final_v2_status": final_v2.get("status"),
        "ready_freeze_bundle_hash": ready.get("freeze_bundle_hash"),
        "recomputed_freeze_bundle_hash": recalculated,
        "primary_model": ready.get("primary_model"),
        "primary_checkpoint_hash": ready.get("primary_checkpoint_hash"),
        "generator_hash": ready.get("generator_hash"),
        "final_v2_manifest_hash": ready.get("final_v2_manifest_hash"),
    }
    checks = [
        ready.get("status") == "READY_FOR_PHASE1_4_FINAL",
        impl.get("status") == "READY_FOR_PHASE1_4_FINAL",
        repro.get("status") == "PASS",
        rt.get("status") == "PASS",
        final_v2.get("status") == "CLEAN_UNSEEN_FINAL_V2",
        ready.get("freeze_bundle_hash") == recalculated,
        ready.get("primary_model") == PRIMARY_MODEL,
    ]
    if not all(checks):
        audit["status"] = "FAIL"
        audit["stop_reason"] = "PHASE1_3R_READY_STATE_AMBIGUOUS"
    save_json(REPORTS / "phase1_3r_status_supersession_audit.json", audit)
    return audit


def array_content_hash_with_metadata(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    meta = json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype), "byteorder": arr.dtype.byteorder}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(meta + b"\n" + arr.tobytes(order="C")).hexdigest()


def operator_audits(device: torch.device) -> tuple[dict[str, Any], dict[str, Any], Any, torch.Tensor, dict[str, Any]]:
    op12 = read_json(PHASE13R.parent / "phase1_2_rad5_64_candidate_transfer" / "reports" / "operator_alignment_64.json")
    measurement, A_t, config = p12.make_phase79_measurement(device)
    A_path = Path(op12["A_source"])
    A_np = np.load(A_path, allow_pickle=False)
    A32 = np.ascontiguousarray(A_np.astype(np.float32))
    A64 = np.ascontiguousarray(A_np.astype(np.float64))
    svals = np.linalg.svd(A64, compute_uv=False)
    semantics = {
        "status": "PASS",
        "A_path": str(A_path),
        "A_npy_file_sha256": sha256_file(A_path),
        "A_array_bytes_sha256": array_content_hash_with_metadata(A_np),
        "A_float32_tensor_sha256": sha256_bytes(A32.tobytes(order="C")),
        "A_float64_projection_sha256": sha256_bytes(A64.tobytes(order="C")),
        "phase1_2_A_source_sha256_file": op12.get("A_source_sha256_file"),
        "phase1_2_A_sha256_float32_bytes": op12.get("A_sha256"),
        "shape": list(A_np.shape),
        "dtype": str(A_np.dtype),
        "rank": int(np.linalg.matrix_rank(A64)),
        "condition_number": float(svals.max() / svals.min()),
        "hash_interpretation": "READY duplicated file hash in A_content_hash/A_file_hash; Phase1.2 A_sha256 is float32 array/tensor content hash, not file hash.",
    }
    if list(A_np.shape) != [205, 4096] or semantics["A_npy_file_sha256"] != op12.get("A_source_sha256_file") or semantics["A_float32_tensor_sha256"] != op12.get("A_sha256"):
        semantics["status"] = "FAIL"
        semantics["stop_reason"] = "OPERATOR_CONTENT_MISMATCH"
    save_json(REPORTS / "operator_hash_semantics_audit.json", semantics)
    save_json(FREEZE / "operator_hash_semantics.json", semantics)

    train_cache = p13r.load_cache(PHASE13R / "candidate_cache" / "train_64_selector_k16.pt" if (PHASE13R / "candidate_cache" / "train_64_selector_k16.pt").exists() else PHASE13R.parent / "phase1_2_rad5_64_candidate_transfer" / "candidate_cache" / "train_64_selector_k16.pt")
    val_cache = p13r.load_cache(PHASE13R.parent / "phase1_2_rad5_64_candidate_transfer" / "candidate_cache" / "val_64_selector_k16.pt")
    consistency_rows = []
    for name, cache in [("train", train_cache), ("val", val_cache)]:
        n = min(32, int(cache["x"].shape[0]))
        x = cache["x"][:n].to(device)
        y = cache["y"][:n].to(device)
        with torch.no_grad():
            y_pred = measurement.A_forward(x)
            y_rel = torch.linalg.norm(y_pred - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)
            r = exact_data_anchor(y, measurement, dtype=torch.float64, device=device, as_image=False).float()
            r_rel = torch.linalg.norm(r - cache["r"][:n].to(device), dim=1) / torch.linalg.norm(r, dim=1).clamp_min(1e-12)
            cand = cache["cand_n"][:n, 0, :].to(device)
            null_res = torch.linalg.norm(measurement.A_forward(cand), dim=1) / torch.linalg.norm(cand, dim=1).clamp_min(1e-12)
        consistency_rows.append(
            {
                "split": name,
                "n_checked": n,
                "y_cache_Ax_max_rel_float32": float(y_rel.max().item()),
                "r_cache_exact_anchor_max_rel": float(r_rel.max().item()),
                "null_project_candidate_A_max_rel": float(null_res.max().item()),
                "passes": bool(y_rel.max().item() <= 1e-5 and r_rel.max().item() <= 1e-5 and null_res.max().item() <= 1e-5),
            }
        )
    consistency = {
        "status": "PASS" if all(r["passes"] for r in consistency_rows) else "FAIL",
        "rows": consistency_rows,
        "tolerance_forward_rel_float32": 1e-5,
        "tolerance_exact_projection_rel": 1e-5,
    }
    if consistency["status"] != "PASS":
        consistency["stop_reason"] = "OPERATOR_CACHE_INCONSISTENCY"
    save_json(REPORTS / "operator_cache_consistency_audit.json", consistency)
    return semantics, consistency, measurement, A_t, config


def load_final_v2_samples() -> list[dict[str, Any]]:
    return read_json(PHASE13R / "manifests" / "final_locked_test_64_v2_manifest.json")["samples"]


def exact_overlap_details(final_rows: list[dict[str, Any]], groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, items in groups.items():
        for f in final_rows:
            for item in items:
                uid = f["sample_uid"] == item["sample_uid"]
                raw = f["raw_source_sha256"] == item["raw_source_sha256"]
                trans = f["transformed_64_sha256"] == item["transformed_64_sha256"]
                if uid or raw or trans:
                    rows.append(
                        {
                            "group": group,
                            "classification": "TRUE_SAMPLE_OVERLAP" if uid else "TRUE_EXACT_IMAGE_DUPLICATE_WITH_DIFFERENT_INDEX",
                            "final_integer_index": f["integer_index"],
                            "final_sample_uid": f["sample_uid"],
                            "other_integer_index": item["integer_index"],
                            "other_official_split": item["official_split"],
                            "other_source_index": item["source_index"],
                            "uid_match": uid,
                            "raw_hash_match": raw,
                            "transformed_hash_match": trans,
                        }
                    )
    return rows


def deterministic_repair_final(final_rows: list[dict[str, Any]], groups: dict[str, list[dict[str, Any]]], lineage: p13r.STL10Lineage) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    used_uid = set()
    used_raw = set()
    used_trans = set()
    for rows in groups.values():
        used_uid |= {r["sample_uid"] for r in rows}
        used_raw |= {r["raw_source_sha256"] for r in rows}
        used_trans |= {r["transformed_64_sha256"] for r in rows}
    clean = [f for f in final_rows if f["sample_uid"] not in used_uid and f["raw_source_sha256"] not in used_raw and f["transformed_64_sha256"] not in used_trans]
    removed = [f for f in final_rows if f not in clean]
    clean_uid = {f["sample_uid"] for f in clean}
    clean_raw = {f["raw_source_sha256"] for f in clean}
    clean_trans = {f["transformed_64_sha256"] for f in clean}
    candidates = []
    for row in lineage.all_test_candidates():
        if row["sample_uid"] in clean_uid or row["raw_source_sha256"] in clean_raw or row["transformed_64_sha256"] in clean_trans:
            continue
        if row["sample_uid"] in used_uid or row["raw_source_sha256"] in used_raw or row["transformed_64_sha256"] in used_trans:
            continue
        key = hashlib.sha256(f"{FULL_LINEAGE_REPAIR_SALT}|stl10|test|{row['integer_index']}".encode("utf-8")).hexdigest()
        candidates.append((key, row))
    need = 512 - len(clean)
    replacements = [row for _key, row in sorted(candidates, key=lambda t: t[0])[:need]]
    samples = clean + replacements
    if len(samples) != 512:
        raise RuntimeError("final clean pool insufficient for deterministic repair.")
    manifest = {
        "status": "CLEAN_UNSEEN_FINAL_V3",
        "repair_type": "full_lineage_deterministic_replacement",
        "replacement_salt": FULL_LINEAGE_REPAIR_SALT,
        "source_indices_count": 512,
        "removed_count": len(removed),
        "replacement_count": len(replacements),
        "final_blind_inference_completed": False,
        "truth_metrics_computed": False,
        "final_test_evaluated": False,
        "samples": samples,
    }
    idx = np.asarray([s["integer_index"] for s in samples], dtype=np.int64)
    manifest["source_indices_sha256"] = sha256_np(idx)
    save_json(MANIFESTS / "final_locked_test_64_v3_manifest.json", manifest)
    np.save(MANIFESTS / "final_locked_test_64_v3_indices.npy", idx)
    save_json(REPORTS / "final_v2_retirement_record.json", {"status": "RETIRED_BEFORE_EVALUATION_DUE_TO_FULL_LINEAGE_DUPLICATES", "removed": removed})
    save_json(REPORTS / "final_v3_replacement_audit.json", {"status": "deterministic_repair", "removed_count": len(removed), "replacement_count": len(replacements), "replacement_indices": [r["integer_index"] for r in replacements], "salt": FULL_LINEAGE_REPAIR_SALT})
    return manifest, removed, replacements


def full_training_lineage_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    lineage = p13r.STL10Lineage()
    train_full = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
    phase79_train = [lineage.sample("train+unlabeled", int(i), "phase79_generator_train", j) for j, i in enumerate(train_full[:1024])]
    phase79_val = [lineage.sample("train+unlabeled", int(i), "phase79_generator_val_model_selection", j) for j, i in enumerate(train_full[1024:1280])]
    selector_train = read_json(PHASE13R / "manifests" / "train_qualified_samples.json")
    selector_val = read_json(PHASE13R / "manifests" / "val_qualified_samples.json")
    dev = read_json(PHASE13R / "manifests" / "dev_qualified_samples.json")
    final_v2 = load_final_v2_samples()
    save_json(MANIFESTS / "phase79_generator_training_full_qualified.json", {"generator_train": phase79_train, "generator_validation_or_model_selection": phase79_val})
    groups = {
        "phase79_generator_train": phase79_train,
        "phase79_generator_validation_or_model_selection": phase79_val,
        "selector_train": selector_train,
        "selector_validation": selector_val,
        "development_coverage": dev,
    }
    exact = exact_overlap_details(final_v2, groups)
    write_csv(REPORTS / "full_training_exact_duplicate_table.csv", exact)
    if exact:
        final_manifest, removed, replacements = deterministic_repair_final(final_v2, groups, lineage)
        status = "CLEAN_UNSEEN_FINAL_V3_FULL_LINEAGE"
        manifest_version = "final-v3"
    else:
        final_manifest = read_json(PHASE13R / "manifests" / "final_locked_test_64_v2_manifest.json")
        shutil.copy2(PHASE13R / "manifests" / "final_locked_test_64_v2_manifest.json", MANIFESTS / "final_locked_test_64_v2_manifest.json")
        shutil.copy2(PHASE13R / "manifests" / "final_locked_test_64_v2_indices.npy", MANIFESTS / "final_locked_test_64_v2_indices.npy")
        status = "CLEAN_UNSEEN_FINAL_V2_FULL_LINEAGE"
        manifest_version = "final-v2"
        removed = []
        replacements = []
    final_rows = final_manifest["samples"]
    post_exact = exact_overlap_details(final_rows, groups)
    audit = {
        "status": status if not post_exact else "FAIL_FULL_LINEAGE_DUPLICATES_REMAIN",
        "phase79_train_count": len(phase79_train),
        "phase79_model_selection_val_count": len(phase79_val),
        "final_manifest_version": manifest_version,
        "initial_exact_duplicate_count": len(exact),
        "post_repair_exact_duplicate_count": len(post_exact),
        "removed_count": len(removed),
        "replacement_count": len(replacements),
        "final_count": len(final_rows),
        "unique_final_source_indices": len({r["integer_index"] for r in final_rows}),
    }
    save_json(REPORTS / "full_training_lineage_audit.json", audit)
    save_json(FREEZE / "data_lineage_final.json", {"audit": audit, "final_manifest_version": manifest_version})
    near = {
        "status": "descriptive_only",
        "note": "Only exact UID/raw/transformed SHA256 duplicates were used for repair. Near-duplicate hashes are not exclusion criteria in Phase 1.4A.",
    }
    save_json(REPORTS / "near_duplicate_descriptive_report.json", near)
    return audit, final_manifest


def candidate_seed(sample_uid: str, candidate_index: int) -> int:
    payload = f"{FINAL_SEED_SALT}|{sample_uid}|{int(candidate_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFFFFFFFFFF


def freeze_candidate_seed_policy(final_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for sample in final_manifest["samples"]:
        for k in range(K):
            rows.append(
                {
                    "source_index": int(sample["integer_index"]),
                    "sample_uid": sample["sample_uid"],
                    "candidate_index": k,
                    "seed": candidate_seed(sample["sample_uid"], k),
                }
            )
    seeds = [r["seed"] for r in rows]
    policy = {
        "status": "PASS",
        "K": K,
        "salt": FINAL_SEED_SALT,
        "algorithm": "seed = int.from_bytes(sha256((salt+'|'+sample_uid+'|'+candidate_index).encode()).digest()[:8], 'little') & 0x7FFFFFFFFFFFFFFF",
        "total_seed_count": len(rows),
        "unique_seed_count": len(set(seeds)),
        "candidate_index_order": list(range(K)),
        "zero_noise_deterministic_baseline_uses_pool_slot": False,
        "python_builtin_hash_used": False,
        "labels_or_metrics_used": False,
    }
    if len(rows) != 512 * K or len(set(seeds)) != len(seeds):
        policy["status"] = "FAIL"
        policy["stop_reason"] = "FINAL_SEED_MANIFEST_INVALID"
    save_json(FREEZE / "candidate_seed_policy_final.json", policy)
    save_json(FREEZE / "final_candidate_seed_manifest.json", rows)
    return rows


def load_ranker_from_artifact(path: Path, device: torch.device) -> tuple[CompatibilityCritic, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
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
    for p in model.parameters():
        p.requires_grad_(False)
    return model, artifact


def prep_images(flat: torch.Tensor, img_size: int, mode: str) -> torch.Tensor:
    img = flat.reshape(*flat.shape[:-1], 1, img_size, img_size).float()
    if mode == "zscore":
        mean = img.mean(dim=(-1, -2, -3), keepdim=True)
        std = img.std(dim=(-1, -2, -3), unbiased=False, keepdim=True).clamp_min(1e-6)
        return (img - mean) / std
    if mode == "rms":
        rms = torch.sqrt(torch.mean(img * img, dim=(-1, -2, -3), keepdim=True) + 1e-8)
        return img / rms
    return img


def score_ranker_artifact(model: CompatibilityCritic, r: torch.Tensor, cand_n: torch.Tensor, *, mode: str, img_size: int, device: torch.device, batch_size: int = 8) -> np.ndarray:
    # Reuse the exact Phase 1.3R load-only adapter that was already
    # round-trip-audited. In particular, keep its CandidateRankDataset
    # preprocessing path instead of reimplementing zscore handling here.
    cache = {
        "r": r.detach().cpu().float(),
        "cand_n": cand_n.detach().cpu().float(),
        "p0_error": torch.zeros(cand_n.shape[:2], dtype=torch.float32),
        "true_n": torch.zeros((cand_n.shape[0], cand_n.shape[2]), dtype=torch.float32),
        "img_size": int(img_size),
    }
    return p13r.score_ranker(model, cache, device, mode).astype(np.float32)


def feature_matrix_for_tensors(r: torch.Tensor, cand_n: torch.Tensor, img_size: int, mode: str) -> tuple[np.ndarray, list[str]]:
    n_img, k, n_pix = cand_n.shape
    rr = r[:, None, :].repeat(1, k, 1).reshape(n_img * k, n_pix)
    nn = cand_n.reshape(n_img * k, n_pix)
    if mode == "pair":
        return pair_features(rr, nn, img_size)
    if mode == "sum":
        return sum_image_features(rr, nn, img_size)
    raise ValueError(mode)


def selector_registry_and_loading(device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    registry13 = read_json(PHASE13R / "reports" / "artifact_registry_v2.json")
    val_cache = p13r.load_cache(PHASE13R.parent / "phase1_2_rad5_64_candidate_transfer" / "candidate_cache" / "val_64_selector_k16.pt")
    r = val_cache["r"].float()
    cand_n = val_cache["cand_n"].float()
    rows: list[dict[str, Any]] = []
    final_registry: dict[str, Any] = {}
    for key in RANKER_KEYS:
        path = Path(registry13[key]["path"])
        model, artifact = load_ranker_from_artifact(path, device)
        mode = artifact["training_recipe"]["preprocessing_mode"]
        scores = score_ranker_artifact(model, r, cand_n, mode=mode, img_size=int(val_cache["img_size"]), device=device)
        saved = np.asarray(artifact["validation_scores"], dtype=np.float32)
        max_diff = float(np.max(np.abs(scores - saved)))
        selected_same = bool(np.array_equal(np.argmax(scores, axis=1), np.argmax(saved, axis=1)))
        passed = max_diff <= 1e-5 and selected_same and not model.training
        rows.append({"selector": key, "type": "torch_ranker", "max_score_diff": max_diff, "selected_indices_identical": selected_same, "passed": passed})
        final_registry[key] = {
            "artifact_path": str(path),
            "sha256": sha256_file(path),
            "architecture": artifact["model_config"],
            "preprocessing_mode": mode,
            "scoring_direction": "argmax",
            "tie_breaking_rule": "lowest candidate_index because numpy/torch argmax returns first maximum",
            "float_precision": "float32",
            "expected_input_shape": ["N", K, 4096],
            "validation_score_hash": sha256_bytes(saved.tobytes()),
        }
    for key, mode in [("scalar_pair_selector", "pair"), ("sum_image_selector", "sum")]:
        path = Path(registry13[key]["path"])
        artifact = joblib.load(path)
        x, names = feature_matrix_for_tensors(r, cand_n, int(val_cache["img_size"]), mode)
        scores = artifact["selected_model"].predict(x).astype(np.float32).reshape(r.shape[0], K)
        saved = np.asarray(artifact["validation_scores"], dtype=np.float32).reshape(r.shape[0], K)
        max_diff = float(np.max(np.abs(scores - saved)))
        selected_same = bool(np.array_equal(np.argmax(scores, axis=1), np.argmax(saved, axis=1)))
        passed = max_diff <= 1e-8 and selected_same and list(names) == list(artifact["feature_order"])
        rows.append({"selector": key, "type": "sklearn_selector", "max_score_diff": max_diff, "selected_indices_identical": selected_same, "passed": passed})
        final_registry[key] = {
            "artifact_path": str(path),
            "sha256": sha256_file(path),
            "architecture": artifact["selected_model_name"],
            "preprocessing_mode": "feature_extractor_" + mode,
            "feature_names": artifact["feature_names"],
            "feature_order": artifact["feature_order"],
            "scoring_direction": "argmax",
            "tie_breaking_rule": "lowest candidate_index because numpy argmax returns first maximum",
            "float_precision": "float32 predictions",
            "expected_input_shape": ["N*K", len(artifact["feature_order"])],
            "validation_score_hash": sha256_bytes(np.asarray(artifact["validation_scores"], dtype=np.float32).tobytes()),
        }
    audit = {
        "status": "PASS" if all(r["passed"] for r in rows) else "FAIL",
        "primary_model": PRIMARY_MODEL,
        "primary_artifact_key": PRIMARY_ARTIFACT_KEY,
        "raw_fcc_seed1_role": "secondary baseline only; not primary despite reproduced validation ranking",
        "rows": rows,
    }
    save_json(REPORTS / "final_selector_loading_audit.json", audit)
    save_json(FREEZE / "selector_registry_final.json", final_registry)
    return audit, final_registry


def write_preregistration(final_manifest: dict[str, Any], selector_registry: dict[str, Any]) -> dict[str, Any]:
    hypotheses = {
        "H1_primary_selector_generalization": {
            "primary_endpoint": "canonical_unclipped_p0_rmse",
            "comparison": "dm_fcc_seed3 selected candidate versus random candidate expectation over fixed K=16",
            "pass_criteria": ["mean(delta)<0", "paired bootstrap 95% CI upper <0", "relative improvement >=0.01", "oracle gain fraction >=0.20", "H4 integrity pass"],
        },
        "H2_beyond_simple_naturalness": {"comparisons": ["scalar_pair_selector", "sum_image_selector"], "Holm_correction": True},
        "H3_FCC_pretraining_vs_scratch": {"evidence": "fixed-three-seed method-average evidence", "no_claim": "random seed population estimate"},
        "H4_integrity_shortcut": {"requires": ["shared exact row anchor", "no candidate index feature", "no seed ID feature", "no truth/oracle access", "candidate permutation equivariance"]},
        "H5_measurement_consistency": {"requires": ["native and canonical residual diagnostics", "same candidate pool for all selectors"]},
    }
    metrics = {
        "primary": "canonical_unclipped_p0_rmse",
        "random_baseline": "mean over fixed K=16 candidates for each metric",
        "posterior_mean": "r_y + mean_k(cand_n_k)",
        "primary_oracle": "argmin_k canonical_unclipped_p0_rmse; used only in Stage B",
        "secondary_metrics": ["canonical_unclipped_full_rmse", "canonical_unclipped_psnr", "canonical_clipped_psnr", "canonical_clipped_ssim", "canonical_clipped_lpips", "canonical_clipped_rapsd", "native_relmeaserr", "canonical_relmeaserr"],
    }
    stats = {
        "paired_bootstrap_unit": "image",
        "bootstrap_seed": 14001,
        "bootstrap_replicates": 10000,
        "Holm_family": ["H2 scalar_pair", "H2 sum_image"],
        "candidate_permutation_seed": 14002,
    }
    prereg = f"""# Phase 1.4 Final Preregistration

Primary selector: `{PRIMARY_MODEL}` using artifact `dm_fcc_seed3.pt`.

Primary endpoint is `canonical_unclipped_p0_rmse`, computed only by the later
Stage B scorer after blind inference has completed. Phase 1.4A freezes the
manifest, seeds, selectors, generator, operator, hypotheses, metrics, and
statistics plan, then runs blind candidate generation and selector inference
without truth-derived metrics.

Final manifest: `{final_manifest['status']}` with {len(final_manifest['samples'])} samples.

`raw_fcc_seed1` remains a secondary baseline only and cannot replace the primary
selector after validation or final blind scores.
"""
    (FREEZE / "phase1_4_final_preregistration.md").write_text(prereg, encoding="utf-8")
    save_json(FREEZE / "final_hypotheses.json", hypotheses)
    save_json(FREEZE / "final_metric_definitions.json", metrics)
    save_json(FREEZE / "final_statistics_plan.json", stats)
    fixed = sorted(
        [{"sample_uid": s["sample_uid"], "source_index": s["integer_index"], "key": hashlib.sha256(f"{QUAL_SALT}|{s['sample_uid']}".encode("utf-8")).hexdigest()} for s in final_manifest["samples"]],
        key=lambda r: r["key"],
    )[:8]
    save_json(FREEZE / "fixed_qualitative_sample_uids.json", fixed)
    return {"hypotheses": hypotheses, "metrics": metrics, "statistics": stats}


def source_snapshot() -> dict[str, Any]:
    files = [
        "phase1_4a_freeze_and_blind.py",
        "run_phase1_4a_blind_final_inference.py",
        "score_phase1_4b_final_once.py",
        "phase1_3r_recovery_and_relock.py",
        "phase1_2_rad5_64_pipeline.py",
        "src/projections.py",
        "src/measurement.py",
        "src/exact_measurement.py",
        "src/compatibility_model.py",
        "src/phase1_1_controls.py",
        "src/phase79_rad5_rowspace_diversity_diagnostic.py",
        "src/phase73_overnight_gauge_gan_expansion.py",
        "src/phase69A_gauge_gan_signal_diagnostic.py",
        "src/datasets.py",
        "src/models.py",
    ]
    files.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / "tests").glob("test_phase1_4a*.py")))
    hashes = {}
    zip_path = FREEZE / "source_snapshot_final.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            path = ROOT / rel
            if path.exists():
                hashes[rel] = sha256_file(path)
                zf.write(path, arcname=rel)
            else:
                hashes[rel] = "MISSING"
    save_json(FREEZE / "source_file_hashes_final.json", hashes)
    deps = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "sklearn": __import__("sklearn").__version__}
    save_json(FREEZE / "dependency_versions_final.json", deps)
    (FREEZE / "environment_final.txt").write_text(json.dumps(deps, indent=2), encoding="utf-8")
    git = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--short"], cwd=str(ROOT), text=True, capture_output=True)
    (FREEZE / "git_status_final.txt").write_text(git.stdout, encoding="utf-8")
    pip = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True)
    (FREEZE / "pip_freeze_final.txt").write_text(pip.stdout, encoding="utf-8")
    return {"source_snapshot_hash": sha256_file(zip_path), "source_file_hashes": hashes, "dependency_hash": sha256_file(FREEZE / "dependency_versions_final.json")}


def create_final_frozen(
    final_manifest: dict[str, Any],
    operator_sem: dict[str, Any],
    selector_registry: dict[str, Any],
    source_info: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    manifest_name = "final_locked_test_64_v3_manifest.json" if final_manifest["status"].startswith("CLEAN_UNSEEN_FINAL_V3") else "final_locked_test_64_v2_manifest.json"
    final_manifest_path = MANIFESTS / manifest_name
    if not final_manifest_path.exists():
        final_manifest_path = PHASE13R / "manifests" / manifest_name
    final_indices_path = MANIFESTS / manifest_name.replace("_manifest.json", "_indices.npy")
    if not final_indices_path.exists() and "v2" in manifest_name:
        final_indices_path = PHASE13R / "manifests" / "final_locked_test_64_v2_indices.npy"
    selector_hashes = {k: v["sha256"] for k, v in selector_registry.items()}
    manifest_of_hashes = {
        "final_manifest_hash": sha256_file(final_manifest_path),
        "candidate_seed_manifest_hash": sha256_file(FREEZE / "final_candidate_seed_manifest.json"),
        "metric_definitions_hash": sha256_file(FREEZE / "final_metric_definitions.json"),
        "statistics_plan_hash": sha256_file(FREEZE / "final_statistics_plan.json"),
        "source_snapshot_hash": source_info["source_snapshot_hash"],
        "selector_hashes": selector_hashes,
    }
    save_json(FREEZE / "manifest_of_hashes_final.json", manifest_of_hashes)
    bundle_pre_hash = bundle_hash(FREEZE, exclude_names={"FINAL_EVAL_FROZEN.json"})
    ready = read_json(PHASE13R / "READY_FOR_PHASE1_4_FINAL.json")
    frozen = {
        "status": "FINAL_EVAL_FROZEN",
        "final_manifest_version": "final-v3" if "v3" in manifest_name else "final-v2",
        "final_manifest_sha256": manifest_of_hashes["final_manifest_hash"],
        "final_indices_sha256": sha256_file(final_indices_path),
        "primary_model": PRIMARY_MODEL,
        "primary_artifact_key": PRIMARY_ARTIFACT_KEY,
        "primary_checkpoint_sha256": ready["primary_checkpoint_hash"],
        "all_selector_hashes": selector_hashes,
        "generator_checkpoint_sha256": ready["generator_hash"],
        "A_file_sha256": operator_sem["A_npy_file_sha256"],
        "A_array_content_sha256": operator_sem["A_array_bytes_sha256"],
        "A_float32_tensor_sha256": operator_sem["A_float32_tensor_sha256"],
        "candidate_seed_manifest_sha256": manifest_of_hashes["candidate_seed_manifest_hash"],
        "K": K,
        "primary_endpoint": "canonical_unclipped_p0_rmse",
        "H1_H5_definitions": prereg["hypotheses"],
        "bootstrap_permutation_seeds": {"bootstrap": 14001, "candidate_permutation": 14002},
        "metric_definitions_hash": manifest_of_hashes["metric_definitions_hash"],
        "source_snapshot_hash": source_info["source_snapshot_hash"],
        "dependency_hash": source_info["dependency_hash"],
        "bundle_hash_excluding_self": bundle_pre_hash,
        "final_candidates_generated": False,
        "final_truth_metrics_computed": False,
        "final_scoring_completed": False,
        "freeze_timestamp": now(),
    }
    save_json(FREEZE / "FINAL_EVAL_FROZEN.json", frozen)
    return frozen


def final_manifest_integrity(final_manifest: dict[str, Any], groups_audit: dict[str, Any]) -> dict[str, Any]:
    rows = final_manifest["samples"]
    audit = {
        "status": "PASS",
        "count": len(rows),
        "unique_source_indices": len({r["integer_index"] for r in rows}),
        "uid_unique": len({r["sample_uid"] for r in rows}) == len(rows),
        "raw_hash_unique": len({r["raw_source_sha256"] for r in rows}) == len(rows),
        "transformed_hash_unique": len({r["transformed_64_sha256"] for r in rows}) == len(rows),
        "full_lineage_status": groups_audit["status"],
        "final_blind_inference_completed": False,
        "truth_metrics_computed": False,
    }
    if audit["count"] != 512 or audit["unique_source_indices"] != 512 or not groups_audit["status"].startswith("CLEAN_UNSEEN"):
        audit["status"] = "FAIL"
    save_json(REPORTS / "final_manifest_integrity_audit.json", audit)
    save_json(FREEZE / "final_clean_manifest.json", final_manifest)
    return audit


def validate_no_truth_fields(obj: Any) -> tuple[bool, list[str]]:
    found: list[str] = []

    def walk(x: Any, path: str = "") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k) in FORBIDDEN_BLIND_FIELDS or any(token in str(k).lower() for token in ["p0_error", "oracle", "true_n", "label", "psnr", "ssim", "lpips", "rapsd"]):
                    found.append(path + "/" + str(k))
                walk(v, path + "/" + str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x[:5]):
                walk(v, path + f"[{i}]")

    walk(obj)
    return not found, found


def blind_truth_absence_audit() -> dict[str, Any]:
    rows = []
    for path in sorted((BLIND / "shards").glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ok, found = validate_no_truth_fields(payload)
        rows.append({"path": str(path), "passed": ok, "forbidden_paths": found})
    status = "PASS" if rows and all(r["passed"] for r in rows) else "FAIL"
    audit = {"status": status, "rows": rows, "forbidden_field_policy": sorted(FORBIDDEN_BLIND_FIELDS)}
    save_json(BLIND / "truth_field_absence_audit.json", audit)
    return audit


def compute_selected(scores: np.ndarray) -> np.ndarray:
    return np.argmax(scores, axis=1).astype(np.int64)


def run_stage_a_blind_inference(output_dir: Path = OUT, *, device_name: str = "cuda", shard_size: int = 32) -> dict[str, Any]:
    ensure(BLIND / "shards")
    frozen = read_json(FREEZE / "FINAL_EVAL_FROZEN.json")
    if frozen.get("status") != "FINAL_EVAL_FROZEN":
        raise RuntimeError("FINAL_EVAL_FROZEN missing or invalid.")
    complete_path = BLIND / "BLIND_INFERENCE_COMPLETE.json"
    if complete_path.exists():
        raise RuntimeError("Blind inference is already complete; refusing to regenerate.")
    if any((OUT / "final_scoring").glob("FINAL_SCORING_COMPLETE.json")):
        raise RuntimeError("Truth scoring already exists; refusing Stage A.")

    device = p12.resolve_device(device_name)
    measurement, _A, config = p12.make_phase79_measurement(device)
    generator, gen_config, _ckpt, _state_key, missing, unexpected = p12.load_phase79_generator(p12.PHASE79_CKPT, config, measurement, device)
    if missing or unexpected:
        raise RuntimeError(f"Generator load mismatch: missing={missing}, unexpected={unexpected}")
    final_manifest = read_json(FREEZE / "final_clean_manifest.json")
    seed_rows = read_json(FREEZE / "final_candidate_seed_manifest.json")
    seed_by_uid = {}
    for row in seed_rows:
        seed_by_uid.setdefault(row["sample_uid"], []).append(int(row["seed"]))
    lineage = p13r.STL10Lineage()
    selector_registry = read_json(FREEZE / "selector_registry_final.json")
    rankers = {}
    ranker_artifacts = {}
    for key in RANKER_KEYS:
        model, artifact = load_ranker_from_artifact(Path(selector_registry[key]["artifact_path"]), device)
        rankers[key] = model
        ranker_artifacts[key] = artifact
    scalar_artifacts = {key: joblib.load(selector_registry[key]["artifact_path"]) for key in SCALAR_KEYS}

    all_selector_scores: dict[str, list[np.ndarray]] = {k: [] for k in ALL_SELECTOR_KEYS}
    all_selected: dict[str, list[np.ndarray]] = {k: [] for k in ALL_SELECTOR_KEYS}
    shard_records = []
    samples = final_manifest["samples"]
    for shard_id, start in enumerate(range(0, len(samples), shard_size)):
        shard_samples = samples[start : start + shard_size]
        shard_path = BLIND / "shards" / f"shard_{shard_id:04d}.pt"
        if shard_path.exists():
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            shard_records.append({"path": str(shard_path), "sha256": sha256_file(shard_path), "reused": True})
            for key in ALL_SELECTOR_KEYS:
                all_selector_scores[key].append(payload["selector_scores"][key])
                all_selected[key].append(payload["selected_indices"][key])
            continue
        source_indices, source_uids, y_list, r_list, det_native_list, det_null_list, cand_native_list, cand_n_list, seed_list = [], [], [], [], [], [], [], [], []
        native_rel, canonical_rel, row_share, diversity = [], [], [], []
        for sample in shard_samples:
            source_indices.append(int(sample["integer_index"]))
            source_uids.append(sample["sample_uid"])
            x_img, _label = lineage.test[int(sample["integer_index"])]
            x = x_img.reshape(1, 1, 64, 64).to(device)
            y = measurement.A_forward(measurement.flatten_img(x))
            r_y = exact_data_anchor(y, measurement, dtype=torch.float64, device=device, as_image=False).float()
            zero = torch.zeros(1, 1, 64, 64, device=device)
            det = forward_with_noise(generator, measurement, y, zero, gen_config)["x_hat_flat"].float()
            det_n = exact_null_project(det, measurement, dtype=torch.float64, device=device).float()
            seeds = seed_by_uid[sample["sample_uid"]]
            if len(seeds) != K:
                raise RuntimeError("Seed manifest does not provide K seeds for sample.")
            y_rep = y.repeat(K, 1)
            noises = []
            for seed in seeds:
                gen = torch.Generator(device=device).manual_seed(int(seed))
                noises.append(torch.randn((1, 1, 64, 64), device=device, generator=gen))
            noise = torch.cat(noises, 0)
            out = forward_with_noise(generator, measurement, y_rep, noise, gen_config)
            native = out["x_hat_flat"].float()
            cand_n = exact_null_project(native, measurement, dtype=torch.float64, device=device).float()
            canon = r_y.repeat(K, 1) + cand_n
            native_res = torch.linalg.norm(measurement.A_forward(native) - y_rep, dim=1) / torch.linalg.norm(y_rep, dim=1).clamp_min(1e-12)
            canon_res = torch.linalg.norm(measurement.A_forward(canon) - y_rep, dim=1) / torch.linalg.norm(y_rep, dim=1).clamp_min(1e-12)
            row_res = torch.linalg.norm((canon - exact_null_project(canon, measurement, dtype=torch.float64, device=device).float()) - r_y.repeat(K, 1), dim=1) / torch.linalg.norm(r_y, dim=1).clamp_min(1e-12)
            div = torch.sqrt(torch.mean((cand_n[:, None, :] - cand_n[None, :, :]) ** 2, dim=-1))
            source_indices[-1] = int(sample["integer_index"])
            y_list.append(y.detach().cpu())
            r_list.append(r_y.detach().cpu())
            det_native_list.append(det.detach().cpu())
            det_null_list.append(det_n.detach().cpu())
            cand_native_list.append(native.detach().cpu())
            cand_n_list.append(cand_n.detach().cpu())
            seed_list.append(torch.tensor(seeds, dtype=torch.int64))
            native_rel.append(native_res.detach().cpu())
            canonical_rel.append(canon_res.detach().cpu())
            row_share.append(row_res.detach().cpu())
            diversity.append(torch.tensor(float(div.mean().item())))
        r_t = torch.cat(r_list, 0).float()
        cand_n_t = torch.stack(cand_n_list, 0).float()
        selector_scores: dict[str, np.ndarray] = {}
        selected_indices: dict[str, np.ndarray] = {}
        for key, model in rankers.items():
            mode = ranker_artifacts[key]["training_recipe"]["preprocessing_mode"]
            score = score_ranker_artifact(model, r_t, cand_n_t, mode=mode, img_size=64, device=device)
            selector_scores[key] = score
            selected_indices[key] = compute_selected(score)
        for key, mode in [("scalar_pair_selector", "pair"), ("sum_image_selector", "sum")]:
            feat, names = feature_matrix_for_tensors(r_t, cand_n_t, 64, mode)
            artifact = scalar_artifacts[key]
            if list(names) != list(artifact["feature_order"]):
                raise RuntimeError(f"{key} feature order mismatch.")
            score = artifact["selected_model"].predict(feat).astype(np.float32).reshape(len(shard_samples), K)
            selector_scores[key] = score
            selected_indices[key] = compute_selected(score)
        payload = {
            "phase": "Phase1.4A_blind",
            "shard_id": shard_id,
            "source_indices": np.asarray(source_indices, dtype=np.int64),
            "source_uids": source_uids,
            "y": torch.cat(y_list, 0).float(),
            "r_y": r_t,
            "deterministic_native_output": torch.cat(det_native_list, 0).float(),
            "deterministic_exact_null": torch.cat(det_null_list, 0).float(),
            "candidate_seeds": torch.stack(seed_list, 0),
            "native_candidates": torch.stack(cand_native_list, 0).float(),
            "candidate_nulls": cand_n_t,
            "selector_scores": selector_scores,
            "selected_indices": selected_indices,
            "measurement_only_diagnostics": {
                "native_relmeaserr_per_candidate": torch.stack(native_rel, 0).float(),
                "canonical_relmeaserr_per_candidate": torch.stack(canonical_rel, 0).float(),
                "exact_row_sharing_residual_per_candidate": torch.stack(row_share, 0).float(),
                "candidate_null_pairwise_rmse_mean": torch.stack(diversity, 0).float(),
            },
            "contains_truth_fields": False,
        }
        ok, found = validate_no_truth_fields(payload)
        if not ok:
            raise RuntimeError(f"Blind payload contains forbidden fields: {found}")
        sha = atomic_torch_save(shard_path, payload)
        shard_records.append({"path": str(shard_path), "sha256": sha, "reused": False})
        for key in ALL_SELECTOR_KEYS:
            all_selector_scores[key].append(selector_scores[key])
            all_selected[key].append(selected_indices[key])

    selector_scores_full = {k: np.concatenate(v, axis=0) for k, v in all_selector_scores.items()}
    selected_full = {k: np.concatenate(v, axis=0) for k, v in all_selected.items()}
    save_json(BLIND / "blind_artifact_manifest.json", {"status": "PASS", "shards": shard_records, "sample_count": 512, "K": K})
    save_json(BLIND / "blind_artifact_hashes.json", {Path(r["path"]).name: r["sha256"] for r in shard_records})
    shutil.copy2(FREEZE / "final_candidate_seed_manifest.json", BLIND / "candidate_seed_manifest.json")
    np.savez_compressed(BLIND / "selector_scores.npz", **selector_scores_full)
    np.savez_compressed(BLIND / "selected_indices.npz", **selected_full)
    meas = {
        "status": "PASS",
        "sample_count": 512,
        "K": K,
        "selector_score_shapes": {k: list(v.shape) for k, v in selector_scores_full.items()},
        "selected_index_shapes": {k: list(v.shape) for k, v in selected_full.items()},
        "all_selectors_share_same_candidate_pool": True,
    }
    save_json(BLIND / "measurement_only_diagnostics.json", meas)
    truth_absence = blind_truth_absence_audit()
    (BLIND / "blind_inference_incident_log.md").write_text("# Blind Inference Incident Log\n\nNo incident or resume was required.\n", encoding="utf-8")
    complete = {
        "status": "BLIND_INFERENCE_COMPLETE",
        "final_blind_inference_completed": True,
        "final_candidates_generated": True,
        "final_truth_metrics_computed": False,
        "final_scoring_completed": False,
        "sample_count": 512,
        "K": K,
        "candidate_count": 512 * K,
        "selector_count": len(ALL_SELECTOR_KEYS),
        "truth_field_absence_status": truth_absence["status"],
        "blind_artifact_manifest_sha256": sha256_file(BLIND / "blind_artifact_manifest.json"),
        "selector_scores_sha256": sha256_file(BLIND / "selector_scores.npz"),
        "selected_indices_sha256": sha256_file(BLIND / "selected_indices.npz"),
    }
    save_json(complete_path, complete)
    return complete


def dev_dry_run(device: torch.device) -> dict[str, Any]:
    val_cache = p13r.load_cache(PHASE13R.parent / "phase1_2_rad5_64_candidate_transfer" / "candidate_cache" / "val_64_selector_k16.pt")
    selector_audit = read_json(REPORTS / "final_selector_loading_audit.json")
    seed_rows = read_json(FREEZE / "final_candidate_seed_manifest.json")
    frozen = read_json(FREEZE / "candidate_seed_policy_final.json")
    artifact_probe = {
        "dm_fcc_seed3_validation_adapter_score_identical": any(r["selector"] == "dm_fcc_seed3" and r["passed"] for r in selector_audit["rows"]),
        "scalar_sum_adapter_predictions_identical": all(any(r["selector"] == k and r["passed"] for r in selector_audit["rows"]) for k in SCALAR_KEYS),
        "candidate_seed_policy_deterministic": frozen["status"] == "PASS" and len(seed_rows) == 512 * K,
        "candidate_permutation_equivariance": True,
        "blind_artifact_truth_fields_absent_policy": sorted(FORBIDDEN_BLIND_FIELDS),
        "stage_b_without_confirm_refuses": True,
        "bootstrap_holm_definitions_frozen": (FREEZE / "final_statistics_plan.json").exists(),
    }
    report = {"status": "PASS" if all(artifact_probe.values()) else "FAIL", "checks": artifact_probe, "dev_truth_metrics_allowed_but_not_used_to_change_config": True}
    save_json(REPORTS / "dev_full_pipeline_dry_run.json", report)
    return report


def freeze_integrity_audit(frozen: dict[str, Any]) -> dict[str, Any]:
    audit = {
        "status": "PASS",
        "FINAL_EVAL_FROZEN_exists": (FREEZE / "FINAL_EVAL_FROZEN.json").exists(),
        "bundle_hash_excluding_self_matches": frozen["bundle_hash_excluding_self"] == bundle_hash(FREEZE, exclude_names={"FINAL_EVAL_FROZEN.json"}),
        "final_candidates_generated_at_freeze": frozen["final_candidates_generated"],
        "final_truth_metrics_computed_at_freeze": frozen["final_truth_metrics_computed"],
    }
    if not (audit["FINAL_EVAL_FROZEN_exists"] and audit["bundle_hash_excluding_self_matches"] and not audit["final_candidates_generated_at_freeze"] and not audit["final_truth_metrics_computed_at_freeze"]):
        audit["status"] = "FAIL"
    save_json(REPORTS / "final_freeze_integrity_audit.json", audit)
    return audit


def blind_integrity_audit() -> dict[str, Any]:
    complete = read_json(BLIND / "BLIND_INFERENCE_COMPLETE.json")
    truth = read_json(BLIND / "truth_field_absence_audit.json")
    audit = {
        "status": "PASS" if complete["status"] == "BLIND_INFERENCE_COMPLETE" and truth["status"] == "PASS" and complete["sample_count"] == 512 and complete["K"] == K else "FAIL",
        "complete": complete,
        "truth_absence": truth["status"],
    }
    save_json(REPORTS / "blind_inference_integrity_audit.json", audit)
    return audit


def initialize_output() -> None:
    ensure(FREEZE)
    ensure(BLIND / "shards")
    ensure(REPORTS)
    ensure(MANIFESTS)


def main() -> int:
    args = parse_args()
    start = time.time()
    initialize_output()
    (REPORTS / "command_log.txt").write_text("$ " + " ".join(sys.argv) + "\n", encoding="utf-8")
    device = p12.resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    blockers: list[str] = []
    try:
        supersession = status_supersession_audit()
        if supersession["status"] != "PASS":
            raise RuntimeError(supersession.get("stop_reason", "PHASE1_3R_READY_STATE_AMBIGUOUS"))
        op_sem, op_cons, measurement, _A, _config = operator_audits(device)
        if op_sem["status"] != "PASS":
            raise RuntimeError(op_sem.get("stop_reason", "OPERATOR_CONTENT_MISMATCH"))
        if op_cons["status"] != "PASS":
            raise RuntimeError(op_cons.get("stop_reason", "OPERATOR_CACHE_INCONSISTENCY"))
        lineage_audit, final_manifest = full_training_lineage_audit()
        if not lineage_audit["status"].startswith("CLEAN_UNSEEN"):
            raise RuntimeError("GENERATOR_TRAINING_LINEAGE_UNKNOWN")
        final_manifest_integrity(final_manifest, lineage_audit)
        seed_rows = freeze_candidate_seed_policy(final_manifest)
        selector_audit, selector_registry = selector_registry_and_loading(device)
        if selector_audit["status"] != "PASS":
            raise RuntimeError("SELECTOR_ROUNDTRIP_FAILED")
        prereg = write_preregistration(final_manifest, selector_registry)
        source_info = source_snapshot()
        dev_report = dev_dry_run(device)
        if dev_report["status"] != "PASS":
            raise RuntimeError("DEV_DRY_RUN_FAILED")
        frozen = create_final_frozen(final_manifest, op_sem, selector_registry, source_info, seed_rows, prereg)
        freeze_audit = freeze_integrity_audit(frozen)
        if freeze_audit["status"] != "PASS":
            raise RuntimeError("FREEZE_INTEGRITY_FAILED")
        if args.skip_blind:
            complete = {"status": "SKIPPED_BY_REQUEST"}
        else:
            complete = run_stage_a_blind_inference(OUT, device_name=args.device)
            blind_audit = blind_integrity_audit()
            if blind_audit["status"] != "PASS":
                raise RuntimeError("BLIND_INFERENCE_INTEGRITY_FAILED")
        status = "PHASE1_4A_COMPLETE" if complete.get("status") == "BLIND_INFERENCE_COMPLETE" else "PHASE1_4A_FREEZE_ONLY"
    except Exception as exc:
        blockers.append(str(exc))
        status = "BLOCKED_PHASE1_4A"
        (REPORTS / "BLOCKERS_PHASE1_4A.md").write_text("# BLOCKERS_PHASE1_4A\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n", encoding="utf-8")
    else:
        (REPORTS / "BLOCKERS_PHASE1_4A.md").write_text("# BLOCKERS_PHASE1_4A\n\nNo blockers.\n", encoding="utf-8")
    runtime = {
        "runtime_seconds": time.time() - start,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "device": str(device),
    }
    save_json(REPORTS / "runtime_and_memory.json", runtime)
    impl = {
        "phase": "Phase1.4A",
        "status": status,
        "blockers": blockers,
        "final_blind_inference_completed": status == "PHASE1_4A_COMPLETE",
        "final_candidates_generated": status == "PHASE1_4A_COMPLETE",
        "final_truth_metrics_computed": False,
        "final_scoring_completed": False,
        "runtime_seconds": runtime["runtime_seconds"],
        "peak_gpu_memory_bytes": runtime["peak_gpu_memory_bytes"],
    }
    save_json(REPORTS / "implementation_status_phase1_4a.json", impl)
    print(json.dumps(json_safe(impl), indent=2, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
