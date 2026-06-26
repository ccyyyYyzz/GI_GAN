from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import re
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
from PIL import Image

import phase1_3r_recovery_and_relock as p13r
from src.phase1_4ir_uid_safe_scoring import (
    ALL_SELECTOR_KEYS,
    K,
    BlindRecord,
    TruthRecord,
    build_selected_by_uid_from_scores,
    score_uid_maps,
    sha256_json,
    stable_candidate_seed,
    truth_rows_hash_from_verified_rows,
)


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
PHASE14A = ROOT / "outputs" / "compatibility" / "phase1_4a_final_freeze_and_blind"
PHASE14B0 = ROOT / "outputs" / "compatibility" / "phase1_4b0_scoring_protocol"
PHASE13R = ROOT / "outputs" / "compatibility" / "phase1_3r_recovery_and_relock"
PHASE12 = ROOT / "outputs" / "compatibility" / "phase1_2_rad5_64_candidate_transfer"
OUT = ROOT / "outputs" / "compatibility" / "phase1_4ir_incident_recovery"
REPORTS = OUT / "reports"
MANIFESTS = OUT / "manifests"
EVIDENCE = OUT / "incident_evidence"
EVIDENCE_RUN = EVIDENCE / "original_final_v3_run"
FREEZE_V4 = OUT / "freeze_bundle_v4"

FINAL_V4_SELECTION_SALT = "FCC_PHASE1_4IR_FRESH_FINAL_V4_V1"
FINAL_V4_CANDIDATE_SALT = "FCC_PHASE1_4IR_FINAL_V4_CANDIDATES_V1"
PRIMARY_SELECTOR = "dm_fcc_seed3"
PRIMARY_MODEL = "reproduced_dm_fcc_seed3_v2"


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
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in keys})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def append_command(text: str) -> None:
    ensure(REPORTS)
    with (REPORTS / "command_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def copy_evidence_file(src: Path, evidence_root: Path) -> dict[str, Any]:
    if not src.exists():
        return {"source": str(src), "exists": False, "copied": False, "sha256": "MISSING"}
    rel = src.relative_to(ROOT) if src.is_relative_to(ROOT) else Path(src.name)
    dst = evidence_root / rel
    ensure(dst.parent)
    shutil.copy2(src, dst)
    src_hash = sha256_file(src)
    dst_hash = sha256_file(dst)
    return {
        "source": str(src),
        "evidence_path": str(dst),
        "exists": True,
        "copied": True,
        "sha256": src_hash,
        "copy_sha256": dst_hash,
        "byte_exact_copy": src_hash == dst_hash,
        "bytes": src.stat().st_size,
    }


def preserve_original_final_v3_run() -> dict[str, Any]:
    ensure(EVIDENCE_RUN)
    files = [
        PHASE14A / "final_scoring_v2" / "FINAL_SCORING_STARTED.json",
        PHASE14A / "final_scoring_v2" / "FINAL_SCORING_COMPLETE.json",
        PHASE14A / "final_scoring_v2" / "results" / "final_summary.json",
        PHASE14A / "final_scoring_v2" / "results" / "per_image_final_metrics.csv",
        PHASE14A / "final_scoring_v2" / "results" / "selector_descriptive_summary.json",
        PHASE14A / "final_scoring_v2" / "final_run_status.json",
        PHASE14A / "final_scoring_v2" / "final_scientific_conclusion.md",
        PHASE14A / "final_scoring_v2" / "preflight_audit.json",
        PHASE14A / "final_scoring_v2" / "postrun_integrity_check.json",
        ROOT / "src" / "phase1_4b_scoring.py",
        ROOT / "score_phase1_4b_final_once_v2.py",
        PHASE14B0 / "freeze_bundle_scoring" / "FINAL_SCORING_PROTOCOL_FROZEN.json",
        PHASE14B0 / "READY_FOR_PHASE1_4B_ONE_SHOT.json",
        PHASE14A / "manifests" / "final_locked_test_64_v3_manifest.json",
        PHASE14A / "manifests" / "final_locked_test_64_v3_indices.npy",
        PHASE14A / "blind_inference" / "blind_artifact_hashes.json",
        PHASE14A / "blind_inference" / "blind_artifact_manifest.json",
        PHASE14A / "blind_inference" / "selector_scores.npz",
        PHASE14A / "blind_inference" / "selected_indices.npz",
    ]
    rows = [copy_evidence_file(path, EVIDENCE_RUN) for path in files]
    for shard in sorted((PHASE14A / "blind_inference" / "shards").glob("shard_*.pt")):
        rows.append(
            {
                "source": str(shard),
                "exists": shard.exists(),
                "copied": False,
                "referenced_only": True,
                "sha256": sha256_file(shard),
                "bytes": shard.stat().st_size,
            }
        )
    manifest = {
        "status": "PASS" if all(row.get("exists") for row in rows) else "FAIL",
        "created_at": now(),
        "policy": "Small evidence files copied byte-exact; large blind shards referenced and hashed.",
        "original_files_left_in_place": True,
        "entries": rows,
    }
    save_json(EVIDENCE / "evidence_manifest.json", manifest)
    save_json(REPORTS / "original_final_v3_evidence_manifest.json", manifest)
    return manifest


def source_incident_audit() -> dict[str, Any]:
    source_path = ROOT / "src" / "phase1_4b_scoring.py"
    runner_path = ROOT / "score_phase1_4b_final_once_v2.py"
    text = source_path.read_text(encoding="utf-8")
    runner = runner_path.read_text(encoding="utf-8")
    start_pattern = re.search(r"start\s*=\s*len\s*\(\s*all_p0\s*\)", text) is not None
    x_slice_pattern = re.search(r"x_part\s*=\s*x_true\s*\[\s*start\s*:\s*start\s*\+\s*count\s*\]", text) is not None
    hardcoded_hash_verified = re.search(r"hash_verified[\"']?\s*:\s*True", text) is not None
    actual_hash_compare_tokens = [
        "actual_raw",
        "reconstructed_sample_uid",
        "actual_transformed",
        "raw_hash !=",
        "transformed_hash !=",
    ]
    actual_verification_evidence = [token for token in actual_hash_compare_tokens if token in text]
    audit = {
        "status": "PASS",
        "classification": "SUSPICIOUS_POSITION_BASED_TRUTH_SLICING_AND_HARDCODED_HASH_VERIFIED",
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "contains_start_equals_len_all_p0": start_pattern,
        "contains_x_true_position_slice": x_slice_pattern,
        "contains_hardcoded_hash_verified_true": hardcoded_hash_verified,
        "actual_hash_verification_tokens_found": actual_verification_evidence,
        "runner_delegates_to_score_final_once": "score_final_once" in runner,
        "old_scorer_source_excerpt_lines": _grep_excerpt(text, ["start = len(all_p0)", "hash_verified"]),
    }
    save_json(REPORTS / "incident_source_audit.json", audit)
    return audit


def _grep_excerpt(text: str, needles: list[str]) -> list[dict[str, Any]]:
    rows = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        if any(needle in line for needle in needles):
            lo = max(1, i - 2)
            hi = min(len(lines), i + 2)
            rows.append({"line": i, "context": "\n".join(f"{j}: {lines[j-1]}" for j in range(lo, hi + 1))})
    return rows


def audit_truth_shard_alignment() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(PHASE14A / "manifests" / "final_locked_test_64_v3_manifest.json")
    samples = manifest["samples"]
    uid_to_pos = {row["sample_uid"]: i for i, row in enumerate(samples)}
    seed_rows = read_json(PHASE14A / "blind_inference" / "candidate_seed_manifest.json")
    first_seed_by_global: dict[int, dict[str, Any]] = {}
    for idx in range(0, len(seed_rows), K):
        block = seed_rows[idx : idx + K]
        if len(block) == K and sorted(int(row["candidate_index"]) for row in block) == list(range(K)):
            first_seed_by_global[idx // K] = block[0]
    rows: list[dict[str, Any]] = []
    shard_summaries = []
    correct_count = 0
    cumulative = 0
    affected_shards: set[int] = set()
    for shard_ordinal, shard_path in enumerate(sorted((PHASE14A / "blind_inference" / "shards").glob("shard_*.pt"))):
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        count = int(payload["candidate_nulls"].shape[0])
        old_start = shard_ordinal
        correct_start = cumulative
        shard_correct = 0
        source_indices = as_numpy(payload["source_indices"]).astype(np.int64).tolist()
        source_uids = [str(uid) for uid in payload["source_uids"]]
        for local_row in range(count):
            global_row = cumulative + local_row
            blind_uid = source_uids[local_row]
            blind_source_index = int(source_indices[local_row])
            seed_manifest = first_seed_by_global.get(global_row, {})
            old_pos = old_start + local_row
            old_sample = samples[old_pos] if old_pos < len(samples) else {}
            correct_pos = uid_to_pos.get(blind_uid, -1)
            correct_sample = samples[correct_pos] if correct_pos >= 0 else {}
            uid_match = old_sample.get("sample_uid") == blind_uid
            source_match = int(old_sample.get("integer_index", -1)) == blind_source_index
            old_alignment_correct = bool(uid_match and source_match)
            if old_alignment_correct:
                correct_count += 1
                shard_correct += 1
            else:
                affected_shards.add(int(payload["shard_id"]))
            rows.append(
                {
                    "shard_id": int(payload["shard_id"]),
                    "shard_file": shard_path.name,
                    "local_row": local_row,
                    "global_candidate_row": global_row,
                    "blind_sample_uid": blind_uid,
                    "blind_source_index": blind_source_index,
                    "candidate_seed_manifest_uid": seed_manifest.get("sample_uid", ""),
                    "candidate_seed_manifest_source_index": seed_manifest.get("source_index", ""),
                    "old_truth_position": old_pos,
                    "old_truth_sample_uid": old_sample.get("sample_uid", ""),
                    "old_truth_source_index": old_sample.get("integer_index", ""),
                    "correct_truth_position": correct_pos,
                    "correct_truth_sample_uid": correct_sample.get("sample_uid", ""),
                    "correct_truth_source_index": correct_sample.get("integer_index", ""),
                    "old_alignment_correct": old_alignment_correct,
                    "exact_uid_match": bool(uid_match),
                    "source_index_match": bool(source_match),
                    "candidate_seed_manifest_match": seed_manifest.get("sample_uid") == blind_uid
                    and int(seed_manifest.get("source_index", -1)) == blind_source_index,
                }
            )
        shard_summaries.append(
            {
                "shard_id": int(payload["shard_id"]),
                "shard_file": shard_path.name,
                "size": count,
                "old_start_len_all_p0": old_start,
                "old_end_exclusive": old_start + count,
                "correct_cumulative_start": correct_start,
                "correct_cumulative_end_exclusive": correct_start + count,
                "old_correct_rows": shard_correct,
                "old_wrong_rows": count - shard_correct,
            }
        )
        cumulative += count
    total = len(rows)
    wrong = total - correct_count
    classification = "CONFIRMED_TRUTH_SHARD_MISALIGNMENT" if wrong else "NO_MISALIGNMENT_DESPITE_SUSPICIOUS_CODE"
    summary = {
        "status": "PASS",
        "classification": classification,
        "total_images": total,
        "alignment_correct_count": correct_count,
        "alignment_error_count": wrong,
        "affected_shard_count": len(affected_shards),
        "affected_shards": sorted(affected_shards),
        "first_shard_correct": all(row["old_alignment_correct"] for row in rows if row["shard_id"] == 0),
        "shard_sizes": [row["size"] for row in shard_summaries],
        "shards": shard_summaries,
        "candidate_seed_manifest_order_provenance": "candidate_seed_manifest has 8192 rows, grouped in K=16 blocks; each first row matches shard UID/source_index order.",
        "blind_artifact_manifest_uid_limitation": "blind_artifact_manifest records shard paths and hashes, not per-sample UIDs; per-sample order is proven by shard source_uids/source_indices plus candidate_seed_manifest.",
    }
    write_csv(REPORTS / "final_v3_truth_shard_alignment.csv", rows)
    save_json(REPORTS / "final_v3_alignment_summary.json", summary)
    return summary, rows


def sample_from_stl10_data_no_label(lineage: p13r.STL10Lineage, source_namespace: str, integer_index: int, collection: str, ordinal: int) -> dict[str, Any]:
    official_split, official_index, dataset = lineage.physical(source_namespace, int(integer_index))
    raw = np.ascontiguousarray(dataset.data[int(official_index)])
    pil = Image.fromarray(np.transpose(raw, (1, 2, 0)))
    transformed = lineage.transform(pil)
    raw_hash = sha256_bytes(raw.tobytes())
    transformed_arr = as_numpy(transformed).astype(np.float32)
    transformed_hash = sha256_bytes(transformed_arr.tobytes())
    uid = p13r.qualified_uid("stl10", official_split, official_index, raw_hash)
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
        "hash_source": "torchvision.datasets.STL10 .data and repository build_transform(64); labels not read for selection",
    }


def audit_final_v3_truth_hashes(lineage: p13r.STL10Lineage) -> dict[str, Any]:
    manifest = read_json(PHASE14A / "manifests" / "final_locked_test_64_v3_manifest.json")
    rows: list[dict[str, Any]] = []
    for i, expected in enumerate(manifest["samples"]):
        actual = sample_from_stl10_data_no_label(lineage, "test", int(expected["integer_index"]), "final_v3_actual_truth_hash_audit", i)
        row = {
            "row": i,
            "source_index": int(expected["integer_index"]),
            "expected_raw_source_sha256": expected["raw_source_sha256"],
            "actual_raw_source_sha256": actual["raw_source_sha256"],
            "expected_transformed_64_sha256": expected["transformed_64_sha256"],
            "actual_transformed_64_sha256": actual["transformed_64_sha256"],
            "expected_sample_uid": expected["sample_uid"],
            "reconstructed_sample_uid": actual["sample_uid"],
            "raw_match": expected["raw_source_sha256"] == actual["raw_source_sha256"],
            "transformed_match": expected["transformed_64_sha256"] == actual["transformed_64_sha256"],
            "uid_match": expected["sample_uid"] == actual["sample_uid"],
        }
        row["all_match"] = bool(row["raw_match"] and row["transformed_match"] and row["uid_match"])
        rows.append(row)
    write_csv(REPORTS / "final_v3_actual_truth_hash_rows.csv", rows)
    source_audit = read_json(REPORTS / "incident_source_audit.json")
    final_summary = read_json(PHASE14A / "final_scoring_v2" / "results" / "final_summary.json")
    old_truth_rows = []
    for sample in manifest["samples"]:
        old_truth_rows.append(
            {
                "dataset_name": sample.get("dataset_name"),
                "official_split": sample.get("official_split"),
                "source_index": sample.get("source_index"),
                "sample_uid": sample.get("sample_uid"),
                "raw_source_sha256": sample.get("raw_source_sha256"),
                "transformed_64_sha256": sample.get("transformed_64_sha256"),
                "shape": [1, 64, 64],
                "dtype": "torch.float32",
                "hash_verified": True,
            }
        )
    audit = {
        "status": "PASS",
        "classification": "OLD_SCORER_USED_UNVERIFIED_TRUTH_BUT_CURRENT_HASHES_MATCH"
        if all(row["all_match"] for row in rows) and source_audit["contains_hardcoded_hash_verified_true"]
        else ("ALL_512_ACTUALLY_VERIFIED" if all(row["all_match"] for row in rows) else "TRUTH_MANIFEST_HASH_MISMATCH"),
        "rows_checked": len(rows),
        "all_512_current_hashes_match": all(row["all_match"] for row in rows),
        "raw_matches": sum(1 for row in rows if row["raw_match"]),
        "transformed_matches": sum(1 for row in rows if row["transformed_match"]),
        "uid_matches": sum(1 for row in rows if row["uid_match"]),
        "old_scorer_actual_hash_verification_executed": False,
        "old_reported_hash_verified_true_has_real_computation_support": False,
        "old_truth_rows_hash_from_metadata_rows": sha256_json(old_truth_rows),
        "old_final_summary_truth_rows_hash": final_summary.get("truth_rows_hash"),
        "old_truth_rows_hash_matches_metadata_reconstruction": sha256_json(old_truth_rows) == final_summary.get("truth_rows_hash"),
        "actual_verified_truth_rows_hash": truth_rows_hash_from_verified_rows(rows),
        "labels_used": False,
        "notes": [
            "The old scorer loaded STL10 transformed tensors and copied expected hashes from the manifest, then wrote hash_verified=True.",
            "This audit recomputed raw bytes hash, transformed tensor hash, and sample_uid from STL10 .data without using labels.",
        ],
    }
    save_json(REPORTS / "final_v3_actual_truth_hash_audit.json", audit)
    return audit


def retire_final_v3_if_needed(alignment: dict[str, Any], truth_audit: dict[str, Any]) -> dict[str, Any]:
    final_summary = read_json(PHASE14A / "final_scoring_v2" / "results" / "final_summary.json")
    misaligned = alignment["classification"] != "NO_MISALIGNMENT_DESPITE_SUSPICIOUS_CODE"
    unverified = not truth_audit["old_scorer_actual_hash_verification_executed"]
    invalid = bool(misaligned or unverified or truth_audit["classification"] in {"TRUTH_MANIFEST_HASH_MISMATCH", "INCONCLUSIVE_TRUTH_PROVENANCE"})
    rel = (0.3262930388 - 0.3244862575) / 0.3262930388
    record = {
        "status": "FINAL_EVALUATION_INVALID" if invalid else "FINAL_EVALUATION_REMAINS_VALID",
        "final_v3_scientific_status": "FINAL_EVALUATION_INVALID" if invalid else "FINAL_EVALUATION_REMAINS_VALID",
        "final_v3_seen": True,
        "final_v3_reusable_for_model_selection": False,
        "final_v3_reusable_as_confirmatory_test": False if invalid else True,
        "classification": "scorer_implementation_failure" if invalid else "no_incident_confirmed",
        "model_failure": False,
        "scorer_implementation_failure": invalid,
        "truth_candidate_misalignment_confirmed": misaligned,
        "old_truth_hash_verification_unverified": unverified,
        "invalid_due_to_truth_candidate_misalignment": misaligned,
        "old_effect_evidence_only": {
            "selected": final_summary.get("primary_mean_selected"),
            "random": final_summary.get("primary_mean_random"),
            "oracle": final_summary.get("primary_mean_oracle"),
            "relative_improvement_from_old_numbers": rel,
            "below_preregistered_one_percent_threshold": rel < 0.01,
            "scientific_use": "incident evidence only",
        },
    }
    save_json(REPORTS / "final_v3_retirement_record.json", record)
    save_json(OUT / "FINAL_V3_EVALUATION_INVALID.json", record)
    md = [
        "# Final-v3 Incident Report",
        "",
        "## Classification",
        "",
        f"- Scientific status: `{record['final_v3_scientific_status']}`",
        "- Incident type: scorer implementation failure, not model generalization failure.",
        f"- Truth/candidate UID misalignment confirmed: `{misaligned}`",
        f"- Old truth hash verification actually executed: `{truth_audit['old_scorer_actual_hash_verification_executed']}`",
        "",
        "## Consequence",
        "",
        "The previous final-v3 numbers are retained only as incident evidence and must not be used in paper tables, claims, model selection, threshold selection, or confirmatory conclusions.",
        "",
        "## Old numeric fact",
        "",
        "The old output had selected=0.3244862575, random=0.3262930388, oracle=0.3222364023. Its relative improvement was about 0.5537%, below the 1% preregistered threshold even before considering the scorer bug.",
        "",
        "## Label",
        "",
        "`INVALID_DUE_TO_TRUTH_CANDIDATE_MISALIGNMENT`",
    ]
    (REPORTS / "final_v3_incident_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return record


def rows_from_manifest(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, list):
        return data
    if "samples" in data:
        return data["samples"]
    rows: list[dict[str, Any]] = []
    for value in data.values():
        if isinstance(value, list):
            rows.extend(value)
    return rows


def select_final_v4(lineage: p13r.STL10Lineage) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exclusion_specs = {
        "selector_train": PHASE13R / "manifests" / "train_qualified_samples.json",
        "selector_validation": PHASE13R / "manifests" / "val_qualified_samples.json",
        "development_coverage": PHASE13R / "manifests" / "dev_qualified_samples.json",
        "phase79_generator_full_training": PHASE14A / "manifests" / "phase79_generator_training_full_qualified.json",
        "final_v1": PHASE13R / "manifests" / "final_v1_qualified_samples.json",
        "final_v2": PHASE13R / "manifests" / "final_locked_test_64_v2_manifest.json",
        "final_v3_seen_final_test": PHASE14A / "manifests" / "final_locked_test_64_v3_manifest.json",
    }
    used_uid: set[str] = set()
    used_raw: set[str] = set()
    used_trans: set[str] = set()
    exclusion_counts: dict[str, int] = {}
    for name, path in exclusion_specs.items():
        rows = rows_from_manifest(path)
        exclusion_counts[name] = len(rows)
        used_uid |= {row["sample_uid"] for row in rows}
        used_raw |= {row["raw_source_sha256"] for row in rows}
        used_trans |= {row["transformed_64_sha256"] for row in rows}
    candidates = [sample_from_stl10_data_no_label(lineage, "test", i, "final_v4_candidate_pool", i) for i in range(len(lineage.test))]
    eligible = [
        row
        for row in candidates
        if row["sample_uid"] not in used_uid
        and row["raw_source_sha256"] not in used_raw
        and row["transformed_64_sha256"] not in used_trans
    ]
    for row in eligible:
        row["selection_salt"] = FINAL_V4_SELECTION_SALT
        row["selection_key"] = hashlib.sha256(f"{FINAL_V4_SELECTION_SALT}|stl10|test|{row['integer_index']}".encode("utf-8")).hexdigest()
    eligible_sorted = sorted(eligible, key=lambda row: row["selection_key"])
    if len(eligible_sorted) < 512:
        raise RuntimeError(f"FINAL_V4_ELIGIBLE_POOL_TOO_SMALL:{len(eligible_sorted)}")
    selected = []
    for ordinal, row in enumerate(eligible_sorted[:512]):
        selected_row = dict(row)
        selected_row["collection"] = "final_locked_test_64_v4"
        selected_row["sample_ordinal"] = ordinal
        selected.append(selected_row)
    manifest = {
        "status": "CLEAN_UNSCORED_FINAL_V4",
        "selection_salt": FINAL_V4_SELECTION_SALT,
        "samples": selected,
        "source_indices_count": len(selected),
        "source_indices_sha256": sha256_json([row["integer_index"] for row in selected]),
        "final_v4_candidates_generated": False,
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "final_v4_never_used_for_model_selection": True,
    }
    save_json(MANIFESTS / "final_locked_test_64_v4_manifest.json", manifest)
    np.save(MANIFESTS / "final_locked_test_64_v4_indices.npy", np.asarray([row["integer_index"] for row in selected], dtype=np.int64))
    dev_rows = []
    for path in exclusion_specs.values():
        dev_rows.extend(rows_from_manifest(path))
    audit = overlap_audit(selected, dev_rows)
    selection_audit = {
        "status": "PASS",
        "eligible_pool_size": len(eligible_sorted),
        "official_test_pool_size": len(candidates),
        "selected_count": len(selected),
        "selection_salt": FINAL_V4_SELECTION_SALT,
        "selection_rule": "sort sha256(salt+'|stl10|test|'+source_index) lexicographically and take first 512",
        "forbidden_features_used": [],
        "labels_used_for_selection": False,
        "image_statistics_used_for_selection": False,
        "model_outputs_used_for_selection": False,
        "exclusion_counts": exclusion_counts,
        "overlap_audit": audit,
    }
    save_json(REPORTS / "final_v4_selection_audit.json", selection_audit)
    full_lineage = {
        "status": "PASS" if audit["uid_overlap"] == 0 and audit["raw_hash_overlap"] == 0 and audit["transformed_hash_overlap"] == 0 else "FAIL",
        "final_v4_count": len(selected),
        "unique_uid_count": len({row["sample_uid"] for row in selected}),
        "unique_source_index_count": len({row["integer_index"] for row in selected}),
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "final_v4_never_used_for_model_selection": True,
        "overlap_audit": audit,
        "near_duplicate_report": "Descriptive near-duplicate screening not used for sample selection; no perceptual threshold was applied.",
    }
    save_json(REPORTS / "final_v4_full_lineage_audit.json", full_lineage)
    return full_lineage, selected


def overlap_audit(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]) -> dict[str, Any]:
    a_uid = {row["sample_uid"] for row in a_rows}
    a_raw = {row["raw_source_sha256"] for row in a_rows}
    a_trans = {row["transformed_64_sha256"] for row in a_rows}
    b_uid = {row["sample_uid"] for row in b_rows}
    b_raw = {row["raw_source_sha256"] for row in b_rows}
    b_trans = {row["transformed_64_sha256"] for row in b_rows}
    final_v3 = rows_from_manifest(PHASE14A / "manifests" / "final_locked_test_64_v3_manifest.json")
    v3_uid = {row["sample_uid"] for row in final_v3}
    return {
        "uid_overlap": len(a_uid & b_uid),
        "raw_hash_overlap": len(a_raw & b_raw),
        "transformed_hash_overlap": len(a_trans & b_trans),
        "final_v3_uid_overlap": len(a_uid & v3_uid),
        "selected_uid_unique": len(a_uid) == len(a_rows),
        "selected_raw_unique": len(a_raw) == len(a_rows),
        "selected_transformed_unique": len(a_trans) == len(a_rows),
    }


def load_validation_scores() -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    artifact_dir = PHASE13R / "recovered_selector_artifacts"
    for key in ALL_SELECTOR_KEYS:
        pt_path = artifact_dir / f"{key}.pt"
        joblib_path = artifact_dir / f"{key}.joblib"
        if pt_path.exists():
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            arr = np.asarray(payload["validation_scores"], dtype=np.float64)
        elif joblib_path.exists():
            payload = joblib.load(joblib_path)
            arr = np.asarray(payload["validation_scores"], dtype=np.float64).reshape(-1, K)
        else:
            raise RuntimeError(f"MISSING_SELECTOR_ARTIFACT:{key}")
        if arr.ndim == 1:
            arr = arr.reshape(-1, K)
        scores[key] = arr
    return scores


def uid_safe_dev_reproduction() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = torch.load(PHASE12 / "candidate_cache" / "val_64_selector_k16.pt", map_location="cpu", weights_only=False)
    val_manifest = rows_from_manifest(PHASE13R / "manifests" / "val_qualified_samples.json")
    if len(val_manifest) != int(cache["cand_n"].shape[0]):
        raise RuntimeError("VAL_MANIFEST_CACHE_LENGTH_MISMATCH")
    uids = [row["sample_uid"] for row in val_manifest]
    truth_by_uid = {}
    blind_by_uid = {}
    for i, row in enumerate(val_manifest):
        truth_by_uid[row["sample_uid"]] = TruthRecord(
            sample_uid=row["sample_uid"],
            source_index=int(row["integer_index"]),
            true_null=as_numpy(cache["true_n"][i]),
            transformed_64_sha256=row["transformed_64_sha256"],
        )
        blind_by_uid[row["sample_uid"]] = BlindRecord(
            sample_uid=row["sample_uid"],
            source_index=int(row["integer_index"]),
            r_y=as_numpy(cache["r"][i]),
            candidate_nulls=as_numpy(cache["cand_n"][i]),
            transformed_64_sha256=row["transformed_64_sha256"],
        )
    scores = load_validation_scores()
    selected_by_uid = build_selected_by_uid_from_scores(uids, scores)
    result = score_uid_maps(truth_by_uid, blind_by_uid, selected_by_uid)
    p0 = result["p0_error"]
    cache_row_by_uid = {uid: i for i, uid in enumerate(uids)}
    cache_p0 = np.stack([as_numpy(cache["p0_error"])[cache_row_by_uid[uid]] for uid in result["ordered_uids"]], axis=0)
    p0_max_abs_diff = float(np.max(np.abs(p0 - cache_p0)))
    registry = read_json(PHASE13R / "reports" / "artifact_registry_v2.json")
    rows = []
    max_metric_diff = 0.0
    for key in ALL_SELECTOR_KEYS:
        got = result["per_selector"][key]
        expected = registry[key]["metrics"]
        comparisons = {
            "selected_p0_rmse_mean": got["mean_selected"],
            "random_expected_p0_rmse_mean": got["mean_random"],
            "oracle_p0_rmse_mean": got["mean_oracle"],
            "selection_regret_mean": got["selection_regret_mean"],
            "oracle_gain_fraction_mean": got["oracle_gain_fraction_mean"],
            "top_oracle_hit_rate": got["top_oracle_hit_rate"],
            "selected_beats_random_fraction": got["selected_beats_random_fraction"],
            "deterministic_p0_rmse_mean": float(cache["deterministic_p0_error"].mean()),
            "posterior_mean_p0_rmse_mean": float(cache["posterior_mean_p0_error"].mean()),
        }
        for metric, actual in comparisons.items():
            expected_value = float(expected[metric])
            diff = abs(float(actual) - expected_value)
            max_metric_diff = max(max_metric_diff, diff)
            rows.append(
                {
                    "selector": key,
                    "metric": metric,
                    "uid_safe_value": float(actual),
                    "phase1_3r_expected_value": expected_value,
                    "absolute_difference": diff,
                }
            )
    write_csv(REPORTS / "old_vs_uid_safe_dev_metrics.csv", rows)
    report = {
        "status": "PASS" if p0_max_abs_diff < 1e-7 and max_metric_diff < 1e-6 else "FAIL",
        "validation_images": len(uids),
        "selector_count": len(ALL_SELECTOR_KEYS),
        "primary_selector": PRIMARY_SELECTOR,
        "primary_selected_p0_rmse_mean": result["per_selector"][PRIMARY_SELECTOR]["mean_selected"],
        "primary_random_expected_p0_rmse_mean": result["per_selector"][PRIMARY_SELECTOR]["mean_random"],
        "p0_error_cache_max_abs_diff": p0_max_abs_diff,
        "max_metric_abs_diff_vs_phase1_3r": max_metric_diff,
        "deterministic_p0_rmse_mean": float(cache["deterministic_p0_error"].mean()),
        "posterior_mean_p0_rmse_mean": float(cache["posterior_mean_p0_error"].mean()),
        "oracle_p0_rmse_mean": float(result["oracle"].mean()),
        "H1_bootstrap": paired_percentile_bootstrap(
            result["per_selector"][PRIMARY_SELECTOR]["selected_errors"] - result["random_expected"]
        ),
    }
    save_json(REPORTS / "uid_safe_dev_reproduction.json", report)
    return report, rows


def synthetic_alignment_demo() -> dict[str, Any]:
    selector_keys = ["dm_fcc_seed3"]
    uids = [f"uid{i}" for i in range(6)]
    truth_by_uid = {}
    blind_by_uid = {}
    selected_by_uid = {}
    for i, uid in enumerate(uids):
        truth = np.asarray([float(i), 0.0])
        candidates = np.stack([truth + (j * 1.0) for j in range(K)], axis=0)
        truth_by_uid[uid] = TruthRecord(uid, i, truth, f"h{i}")
        blind_by_uid[uid] = BlindRecord(uid, i, np.zeros(2), candidates, f"h{i}")
        selected_by_uid[uid] = {"dm_fcc_seed3": 0}
    correct = score_uid_maps(truth_by_uid, blind_by_uid, selected_by_uid, selector_keys=selector_keys)
    shards = [uids[0:3], uids[3:6]]
    wrong_errors = []
    for shard_ordinal, shard_uids in enumerate(shards):
        old_start = shard_ordinal
        for local, uid in enumerate(shard_uids):
            old_truth_uid = uids[old_start + local]
            cand = blind_by_uid[uid].candidate_nulls[0]
            old_truth = truth_by_uid[old_truth_uid].true_null
            wrong_errors.append(float(np.sqrt(np.mean((cand - old_truth) ** 2))))
    permuted_blind = {uid: blind_by_uid[uid] for uid in reversed(uids)}
    permuted_truth = {uid: truth_by_uid[uid] for uid in reversed(uids)}
    permuted_selected = {uid: selected_by_uid[uid] for uid in reversed(uids)}
    permuted = score_uid_maps(permuted_truth, permuted_blind, permuted_selected, selector_keys=selector_keys)
    demo = {
        "status": "PASS",
        "old_position_based_mean_error": float(np.mean(wrong_errors)),
        "uid_safe_mean_error": float(correct["per_selector"]["dm_fcc_seed3"]["mean_selected"]),
        "old_position_bug_exposed": float(np.mean(wrong_errors)) > 0.0,
        "uid_safe_permutation_invariant": np.allclose(
            np.sort(correct["per_selector"]["dm_fcc_seed3"]["selected_errors"]),
            np.sort(permuted["per_selector"]["dm_fcc_seed3"]["selected_errors"]),
        ),
        "note": "Synthetic two-shard fixture: second shard old_start=1 instead of cumulative 3.",
    }
    save_json(REPORTS / "old_position_bug_synthetic_demo.json", demo)
    return demo


def paired_percentile_bootstrap(delta: np.ndarray, seed: int = 14001, replicates: int = 10000) -> dict[str, Any]:
    arr = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    means = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        means[b] = arr[rng.integers(0, n, size=n)].mean()
    return {
        "B": replicates,
        "seed": seed,
        "observed_mean": float(arr.mean()),
        "ci_lower": float(np.quantile(means, 0.025)),
        "ci_upper": float(np.quantile(means, 0.975)),
        "fraction_negative": float(np.mean(means < 0)),
        "unit": "image",
    }


def freeze_final_v4_protocol(selected: list[dict[str, Any]], recovery_gate_pretests: dict[str, Any]) -> dict[str, Any]:
    ensure(FREEZE_V4)
    seed_rows = []
    for row in selected:
        for k in range(K):
            seed_rows.append(
                {
                    "sample_uid": row["sample_uid"],
                    "source_index": int(row["integer_index"]),
                    "candidate_index": k,
                    "seed": stable_candidate_seed(row["sample_uid"], k, FINAL_V4_CANDIDATE_SALT),
                }
            )
    save_json(FREEZE_V4 / "final_v4_candidate_seed_manifest.json", seed_rows)
    seed_policy = {
        "status": "FROZEN",
        "salt": FINAL_V4_CANDIDATE_SALT,
        "algorithm": "int.from_bytes(sha256((salt+'|'+sample_uid+'|'+candidate_index).encode()).digest()[:8], 'little') & 0x7FFFFFFFFFFFFFFF",
        "K": K,
        "seed_count": len(seed_rows),
    }
    save_json(FREEZE_V4 / "final_v4_candidate_seed_policy.json", seed_policy)
    prereg = [
        "# Final-v4 Blind Protocol Preregistration",
        "",
        f"- Generator: Phase79 frozen generator.",
        f"- Operator: Rad-5 A from the existing Phase 1.4A protocol.",
        f"- Primary selector: {PRIMARY_MODEL} / {PRIMARY_SELECTOR}.",
        f"- Selectors: original {len(ALL_SELECTOR_KEYS)} selectors.",
        f"- K: {K}.",
        "- Canonicalization: exact shared row anchor.",
        "- H1-H5: unchanged from Phase 1.4B0; H3 remains incomplete-decision-rule limited; S1 remains an amendment.",
        "- No final-v4 truth metrics are computed in this phase.",
    ]
    (FREEZE_V4 / "final_v4_preregistration.md").write_text("\n".join(prereg) + "\n", encoding="utf-8")
    save_json(FREEZE_V4 / "final_v4_manifest.json", read_json(MANIFESTS / "final_locked_test_64_v4_manifest.json"))
    shutil.copy2(PHASE14A / "freeze_bundle" / "selector_registry_final.json", FREEZE_V4 / "selector_registry.json")
    shutil.copy2(PHASE14A / "freeze_bundle" / "final_metric_definitions.json", FREEZE_V4 / "metric_contract.json")
    shutil.copy2(PHASE14A / "freeze_bundle" / "final_statistics_plan.json", FREEZE_V4 / "statistics_contract.json")
    source_files = [
        ROOT / "src" / "phase1_4ir_uid_safe_scoring.py",
        ROOT / "phase1_4ir_incident_recovery.py",
        ROOT / "tests" / "test_phase1_4ir_uid_safe.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) if path.exists() else "MISSING" for path in source_files}
    save_json(FREEZE_V4 / "corrected_scorer_source_hashes.json", source_hashes)
    data_lineage = {
        "status": "PASS",
        "final_v4_manifest_sha256": sha256_file(MANIFESTS / "final_locked_test_64_v4_manifest.json"),
        "final_v4_indices_sha256": sha256_file(MANIFESTS / "final_locked_test_64_v4_indices.npy"),
        "selection_audit_sha256": sha256_file(REPORTS / "final_v4_selection_audit.json"),
        "full_lineage_audit_sha256": sha256_file(REPORTS / "final_v4_full_lineage_audit.json"),
    }
    save_json(FREEZE_V4 / "data_lineage.json", data_lineage)
    incident_reference = {
        "final_v3_retirement_record_sha256": sha256_file(REPORTS / "final_v3_retirement_record.json"),
        "final_v3_incident_report_sha256": sha256_file(REPORTS / "final_v3_incident_report.md"),
        "final_v3_seen_final_test": True,
    }
    save_json(FREEZE_V4 / "incident_reference.json", incident_reference)
    with zipfile.ZipFile(FREEZE_V4 / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_files:
            if path.exists():
                zf.write(path, path.relative_to(ROOT))
    frozen = {
        "status": "FINAL_V4_BLIND_PROTOCOL_FROZEN",
        "created_at": now(),
        "primary_model": PRIMARY_MODEL,
        "primary_artifact_key": PRIMARY_SELECTOR,
        "K": K,
        "selector_count": len(ALL_SELECTOR_KEYS),
        "selection_salt": FINAL_V4_SELECTION_SALT,
        "candidate_salt": FINAL_V4_CANDIDATE_SALT,
        "final_v4_count": len(selected),
        "final_v4_candidate_seed_manifest_sha256": sha256_file(FREEZE_V4 / "final_v4_candidate_seed_manifest.json"),
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "corrected_final_v3_diagnostic_run": False,
        "source_snapshot_sha256": sha256_file(FREEZE_V4 / "source_snapshot.zip"),
        "gate_pretests": recovery_gate_pretests,
    }
    save_json(FREEZE_V4 / "FINAL_V4_BLIND_PROTOCOL_FROZEN.json", frozen)
    return frozen


def run_pytest() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "tests", "-q"]
    append_command("$ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    text = res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else "")
    (REPORTS / "pytest_summary.txt").write_text(text, encoding="utf-8")
    return {"status": "PASS" if res.returncode == 0 else "FAIL", "returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr}


def create_ready_optional_v3_diagnostic() -> None:
    save_json(
        OUT / "READY_FOR_OPTIONAL_V3_DIAGNOSTIC.json",
        {
            "status": "READY_FOR_OPTIONAL_V3_DIAGNOSTIC",
            "purpose": "post-incident technical diagnostic only",
            "corrected_final_v3_diagnostic_run": False,
            "forbidden_uses": ["model_selection", "hypothesis_modification", "primary_selector_change", "independent_generalization_claim"],
        },
    )


def final_v4_truth_absence_audit() -> dict[str, Any]:
    suspicious = []
    allowed_names = {"final_v4_truth_metric_absence_audit.json"}
    for path in OUT.rglob("*"):
        if path.name in allowed_names:
            continue
        if path.is_file() and re.search(r"(final_v4.*(p0_rmse|psnr|lpips|oracle|truth_metric|scoring_complete)|FINAL_V4_SCORING_COMPLETE)", path.name, re.I):
            suspicious.append(str(path))
    audit = {
        "status": "PASS" if not suspicious else "FAIL",
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "suspicious_files": suspicious,
    }
    save_json(REPORTS / "final_v4_truth_metric_absence_audit.json", audit)
    return audit


def package_outputs() -> tuple[dict[str, Any], dict[str, Any]]:
    brief_files = [
        REPORTS / "incident_source_audit.json",
        REPORTS / "final_v3_alignment_summary.json",
        REPORTS / "final_v3_truth_shard_alignment.csv",
        REPORTS / "final_v3_actual_truth_hash_audit.json",
        REPORTS / "final_v3_incident_report.md",
        REPORTS / "final_v3_retirement_record.json",
        REPORTS / "uid_safe_dev_reproduction.json",
        REPORTS / "old_position_bug_synthetic_demo.json",
        REPORTS / "final_v4_selection_audit.json",
        REPORTS / "final_v4_full_lineage_audit.json",
        REPORTS / "implementation_status_phase1_4ir.json",
        REPORTS / "pytest_summary.txt",
        MANIFESTS / "final_locked_test_64_v4_manifest.json",
        MANIFESTS / "final_locked_test_64_v4_indices.npy",
        FREEZE_V4 / "FINAL_V4_BLIND_PROTOCOL_FROZEN.json",
        FREEZE_V4 / "final_v4_candidate_seed_manifest.json",
        FREEZE_V4 / "corrected_scorer_source_hashes.json",
        OUT / "FINAL_V3_EVALUATION_INVALID.json",
        ROOT / "src" / "phase1_4ir_uid_safe_scoring.py",
        ROOT / "phase1_4ir_incident_recovery.py",
        ROOT / "tests" / "test_phase1_4ir_uid_safe.py",
    ]
    brief_zip = OUT / "phase1_4ir_incident_recovery_gpt_brief.zip"
    full_zip = OUT / "phase1_4ir_incident_recovery_full_archive.zip"
    for zip_path in [brief_zip, full_zip]:
        if zip_path.exists():
            zip_path.unlink()
    with zipfile.ZipFile(brief_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in brief_files:
            if path.exists():
                zf.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(full_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base in [OUT]:
            for path in base.rglob("*"):
                if path.is_file() and path != full_zip:
                    zf.write(path, path.relative_to(ROOT))
        for path in [ROOT / "src" / "phase1_4ir_uid_safe_scoring.py", ROOT / "phase1_4ir_incident_recovery.py", ROOT / "tests" / "test_phase1_4ir_uid_safe.py"]:
            if path.exists():
                zf.write(path, path.relative_to(ROOT))
    brief = {"path": str(brief_zip), "sha256": sha256_file(brief_zip), "bytes": brief_zip.stat().st_size}
    full = {"path": str(full_zip), "sha256": sha256_file(full_zip), "bytes": full_zip.stat().st_size}
    save_json(OUT / "package_hashes.json", {"gpt_brief": brief, "full_archive": full})
    return brief, full


def run_phase1_4ir(run_blind: bool = False) -> dict[str, Any]:
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ensure(REPORTS)
    ensure(MANIFESTS)
    ensure(EVIDENCE)
    append_command("$ python phase1_4ir_incident_recovery.py --run")
    blockers: list[str] = []
    final_status = "BLOCKED_INCIDENT_NOT_CONFIRMED"
    try:
        evidence = preserve_original_final_v3_run()
        source = source_incident_audit()
        alignment, _alignment_rows = audit_truth_shard_alignment()
        lineage = p13r.STL10Lineage()
        truth_audit = audit_final_v3_truth_hashes(lineage)
        retirement = retire_final_v3_if_needed(alignment, truth_audit)
        if retirement["final_v3_scientific_status"] != "FINAL_EVALUATION_INVALID":
            raise RuntimeError("INCIDENT_NOT_CONFIRMED")
        final_v4_lineage, selected_v4 = select_final_v4(lineage)
        if final_v4_lineage["status"] != "PASS":
            raise RuntimeError("FINAL_V4_LINEAGE_FAILED")
        synthetic = synthetic_alignment_demo()
        dev_repro, _dev_rows = uid_safe_dev_reproduction()
        pytest_report = run_pytest()
        gates_pre = {
            "incident_recorded": True,
            "final_v3_retired": True,
            "final_v4_locked": True,
            "final_v4_lineage_pass": final_v4_lineage["status"] == "PASS",
            "uid_safe_dev_reproduction_pass": dev_repro["status"] == "PASS",
            "synthetic_alignment_pass": synthetic["status"] == "PASS",
            "pytest_pass": pytest_report["status"] == "PASS",
            "selector_artifact_hash_pass": selector_artifact_hash_audit()["status"] == "PASS",
            "generator_operator_hash_pass": generator_operator_hash_audit()["status"] == "PASS",
            "seed_count": len(selected_v4) * K,
        }
        frozen = freeze_final_v4_protocol(selected_v4, gates_pre)
        create_ready_optional_v3_diagnostic()
        truth_absence = final_v4_truth_absence_audit()
        if not all(v is True for k, v in gates_pre.items() if k.endswith("_pass") or k in {"incident_recorded", "final_v3_retired", "final_v4_locked"}):
            final_status = "BLOCKED_UID_SCORER_REPRODUCTION"
        elif truth_absence["status"] != "PASS":
            final_status = "BLOCKED_ARTIFACT_INTEGRITY"
        elif run_blind:
            final_status = "READY_FOR_FINAL_V4_BLIND"
            blockers.append("V4 blind inference intentionally not implemented in this incident-recovery run.")
        else:
            final_status = "READY_FOR_FINAL_V4_BLIND"
        blockers_path_text = "# BLOCKERS_PHASE1_4IR\n\nNo blockers. Final-v4 blind inference was not run in this phase.\n"
    except Exception as exc:
        blockers.append(repr(exc))
        blockers_path_text = "# BLOCKERS_PHASE1_4IR\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n"
        frozen = {}
        pytest_report = {"status": "NOT_RUN"}
    runtime = {
        "runtime_seconds": time.time() - start,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    save_json(REPORTS / "runtime_and_memory.json", runtime)
    (REPORTS / "BLOCKERS_PHASE1_4IR.md").write_text(blockers_path_text, encoding="utf-8")
    implementation = {
        "phase": "Phase 1.4IR",
        "status": final_status,
        "blockers": blockers,
        "final_v3_seen_final_test": True,
        "corrected_final_v3_diagnostic_run": False,
        "final_v4_locked_before_any_corrected_v3_metric": True,
        "final_v4_candidates_generated": False,
        "final_v4_blind_inference_completed": False,
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "pytest_status": pytest_report.get("status"),
        **runtime,
    }
    save_json(REPORTS / "implementation_status_phase1_4ir.json", implementation)
    brief, full = package_outputs()
    implementation["gpt_brief_package"] = brief
    implementation["full_archive_package"] = full
    save_json(REPORTS / "implementation_status_phase1_4ir.json", implementation)
    print(json.dumps(json_safe(implementation), indent=2, sort_keys=True))
    return implementation


def selector_artifact_hash_audit() -> dict[str, Any]:
    registry = read_json(PHASE13R / "reports" / "artifact_registry_v2.json")
    rows = []
    for key, info in registry.items():
        path = Path(info["path"])
        rows.append({"selector": key, "path": str(path), "expected": info["sha256"], "actual": sha256_file(path), "pass": sha256_file(path) == info["sha256"]})
    audit = {"status": "PASS" if all(row["pass"] for row in rows) else "FAIL", "rows": rows}
    save_json(REPORTS / "selector_artifact_hash_audit.json", audit)
    return audit


def generator_operator_hash_audit() -> dict[str, Any]:
    frozen = read_json(PHASE14A / "freeze_bundle" / "FINAL_EVAL_FROZEN.json")
    ready = read_json(PHASE14B0 / "READY_FOR_PHASE1_4B_ONE_SHOT.json")
    generator = Path("E:/ns_mc_gan_gi/outputs_phase79_posterior_anti_collapse/rad5_rowspace_diversity_diagnostic/checkpoints/final.pt")
    operator = Path(ready["original_hashes"]["operator_hash_semantics"]["A_path"])
    rows = [
        {"name": "generator", "path": str(generator), "expected": frozen["generator_checkpoint_sha256"], "actual": sha256_file(generator)},
        {"name": "operator_A_file", "path": str(operator), "expected": frozen["A_file_sha256"], "actual": sha256_file(operator)},
    ]
    for row in rows:
        row["pass"] = row["expected"] == row["actual"]
    audit = {"status": "PASS" if all(row["pass"] for row in rows) else "FAIL", "rows": rows}
    save_json(REPORTS / "generator_operator_hash_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.4IR incident forensics and final-v4 relock.")
    parser.add_argument("--run", action="store_true", help="Run incident recovery through final-v4 blind protocol freeze.")
    parser.add_argument("--run-v4-blind", action="store_true", help="Reserved; this implementation stops at READY_FOR_FINAL_V4_BLIND.")
    parser.add_argument("--corrected-final-v3-diagnostic", action="store_true", help="Refuses by default; final-v3 diagnostics are not run in this phase.")
    parser.add_argument("--score-final-v4", action="store_true", help="Always refuses in Phase 1.4IR.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.corrected_final_v3_diagnostic:
        print("REFUSING: corrected final-v3 diagnostic is disabled by default in Phase 1.4IR.")
        return 2
    if args.score_final_v4:
        print("REFUSING: final-v4 truth scoring is forbidden in Phase 1.4IR.")
        return 2
    if args.run:
        status = run_phase1_4ir(run_blind=args.run_v4_blind)
        return 0 if not status["blockers"] else 2
    print("No action requested. Use --run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
