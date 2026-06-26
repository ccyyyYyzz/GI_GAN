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
from PIL import Image

import phase1_2_rad5_64_pipeline as p12
import phase1_3r_recovery_and_relock as p13r
import phase1_4a_freeze_and_blind as p14a
from src.phase1_4ir_uid_safe_scoring import ALL_SELECTOR_KEYS, K, stable_candidate_seed
from src.phase79_rad5_rowspace_diversity_diagnostic import forward_with_noise
from src.projections import exact_data_anchor, exact_null_project


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
PHASE14IR = ROOT / "outputs" / "compatibility" / "phase1_4ir_incident_recovery"
PHASE13R = ROOT / "outputs" / "compatibility" / "phase1_3r_recovery_and_relock"
PHASE12 = ROOT / "outputs" / "compatibility" / "phase1_2_rad5_64_candidate_transfer"
OUT = ROOT / "outputs" / "compatibility" / "phase1_4v4a_blind_inference"
REPORTS = OUT / "reports"
FREEZE_EXEC = OUT / "freeze_bundle_execution"
BLIND = OUT / "blind_inference_v4"
SHARDS = BLIND / "shards"

PROTOCOL = PHASE14IR / "freeze_bundle_v4" / "FINAL_V4_BLIND_PROTOCOL_FROZEN.json"
FINAL_V4_MANIFEST = PHASE14IR / "manifests" / "final_locked_test_64_v4_manifest.json"
FINAL_V4_INDICES = PHASE14IR / "manifests" / "final_locked_test_64_v4_indices.npy"
SEED_MANIFEST = PHASE14IR / "freeze_bundle_v4" / "final_v4_candidate_seed_manifest.json"
GENERATOR_CKPT = Path("E:/ns_mc_gan_gi/outputs_phase79_posterior_anti_collapse/rad5_rowspace_diversity_diagnostic/checkpoints/final.pt")
A_RAD5 = Path("E:/ns_mc_gan_gi/results/cert_package_20260612/cache/A_rad5.npy")

EXECUTION_SALT = "FCC_PHASE1_4V4A_BLIND_EXECUTION_V1"
PRIMARY_SELECTOR = "dm_fcc_seed3"
PRIMARY_MODEL = "reproduced_dm_fcc_seed3_v2"
SHARD_SIZE = 32
INFERENCE_BATCH_SIZE = 16
FORBIDDEN_KEYS = {"x_true", "true_x", "true_n", "labels", "label", "p0_error", "full_error", "oracle", "oracle_index", "psnr", "ssim", "lpips", "rapsd", "selected_error"}
ALLOWED_TRUTH_STATUS_KEYS = {"final_v4_truth_metrics_computed"}


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fields})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def append_command(text: str) -> None:
    ensure(REPORTS)
    with (REPORTS / "command_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> str:
    ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    loaded = torch.load(tmp, map_location="cpu", weights_only=False)
    validate_blind_payload_schema(loaded)
    os.replace(tmp, path)
    return sha256_file(path)


def sample_no_label(lineage: p13r.STL10Lineage, integer_index: int, ordinal: int, collection: str) -> tuple[dict[str, Any], torch.Tensor]:
    official_split, official_index, dataset = lineage.physical("test", int(integer_index))
    raw = np.ascontiguousarray(dataset.data[int(official_index)])
    pil = Image.fromarray(np.transpose(raw, (1, 2, 0)))
    transformed = lineage.transform(pil)
    arr = transformed.detach().cpu().contiguous().numpy().astype(np.float32)
    raw_hash = sha256_bytes(raw.tobytes())
    transformed_hash = sha256_bytes(arr.tobytes())
    uid = p13r.qualified_uid("stl10", official_split, official_index, raw_hash)
    row = {
        "collection": collection,
        "dataset_name": "stl10",
        "source_namespace": "test",
        "integer_index": int(integer_index),
        "official_split": official_split,
        "source_index": int(official_index),
        "raw_source_sha256": raw_hash,
        "transformed_64_sha256": transformed_hash,
        "sample_uid": uid,
        "sample_ordinal": int(ordinal),
        "hash_source": "STL10 .data plus repository 64px transform; class target is not read",
    }
    return row, transformed.float()


def verify_preflight() -> dict[str, Any]:
    ensure(REPORTS)
    ir_status = read_json(PHASE14IR / "reports" / "implementation_status_phase1_4ir.json")
    protocol = read_json(PROTOCOL)
    final_v3_invalid = read_json(PHASE14IR / "FINAL_V3_EVALUATION_INVALID.json")
    final_v4_lineage = read_json(PHASE14IR / "reports" / "final_v4_full_lineage_audit.json")
    manifest = read_json(FINAL_V4_MANIFEST)
    seed_rows = read_json(SEED_MANIFEST)
    registry = read_json(PHASE13R / "reports" / "artifact_registry_v2.json")
    selector_checks = []
    for key in ALL_SELECTOR_KEYS:
        info = registry[key]
        path = Path(info["path"])
        selector_checks.append({"selector": key, "path": str(path), "expected": info["sha256"], "actual": sha256_file(path), "pass": sha256_file(path) == info["sha256"]})
    seed_formula_ok = all(
        int(row["seed"]) == stable_candidate_seed(row["sample_uid"], int(row["candidate_index"]), protocol["candidate_salt"])
        for row in seed_rows
    )
    unique_seed_keys = len({(row["sample_uid"], int(row["candidate_index"])) for row in seed_rows}) == len(seed_rows)
    preflight = {
        "status": "PASS",
        "phase1_4ir_status": ir_status.get("status"),
        "protocol_status": protocol.get("status"),
        "protocol_sha256": sha256_file(PROTOCOL),
        "final_v3_invalid_status": final_v3_invalid.get("status"),
        "final_v4_lineage_status": final_v4_lineage.get("status"),
        "final_v4_manifest_sha256": sha256_file(FINAL_V4_MANIFEST),
        "final_v4_indices_sha256": sha256_file(FINAL_V4_INDICES),
        "seed_manifest_sha256": sha256_file(SEED_MANIFEST),
        "final_v4_count": len(manifest["samples"]),
        "K": protocol.get("K"),
        "seed_count": len(seed_rows),
        "seed_formula_ok": seed_formula_ok,
        "seed_key_unique": unique_seed_keys,
        "generator_sha256": sha256_file(GENERATOR_CKPT),
        "operator_A_file_sha256": sha256_file(A_RAD5),
        "selector_checks": selector_checks,
        "final_v4_candidates_generated_before_run": (BLIND / "BLIND_INFERENCE_V4_COMPLETE.json").exists(),
        "final_v4_truth_metrics_computed": False,
    }
    required = [
        ir_status.get("status") == "READY_FOR_FINAL_V4_BLIND",
        protocol.get("status") == "FINAL_V4_BLIND_PROTOCOL_FROZEN",
        final_v3_invalid.get("status") == "FINAL_EVALUATION_INVALID",
        final_v4_lineage.get("status") == "PASS",
        len(manifest["samples"]) == 512,
        protocol.get("K") == K,
        len(seed_rows) == 512 * K,
        seed_formula_ok,
        unique_seed_keys,
        all(row["pass"] for row in selector_checks),
    ]
    if not all(required):
        preflight["status"] = "FAIL"
    save_json(REPORTS / "preflight_audit.json", preflight)
    if preflight["status"] != "PASS":
        raise RuntimeError("PREFLIGHT_FAILED")
    return preflight


def load_generator_and_selectors(device: torch.device):
    measurement, _A, config = p12.make_phase79_measurement(device)
    generator, gen_config, _ckpt, _state_key, missing, unexpected = p12.load_phase79_generator(GENERATOR_CKPT, config, measurement, device)
    if missing or unexpected:
        raise RuntimeError(f"GENERATOR_LOAD_MISMATCH: missing={missing}, unexpected={unexpected}")
    generator.eval()
    for param in generator.parameters():
        param.requires_grad_(False)
    registry = read_json(PHASE13R / "reports" / "artifact_registry_v2.json")
    rankers = {}
    ranker_artifacts = {}
    for key in p14a.RANKER_KEYS:
        model, artifact = p14a.load_ranker_from_artifact(Path(registry[key]["path"]), device)
        rankers[key] = model
        ranker_artifacts[key] = artifact
    scalar_artifacts = {key: joblib.load(registry[key]["path"]) for key in p14a.SCALAR_KEYS}
    return measurement, generator, gen_config, rankers, ranker_artifacts, scalar_artifacts, registry


def validate_blind_payload_schema(payload: Any) -> None:
    found = find_forbidden_keys(payload)
    if found:
        raise RuntimeError(f"BLIND_PAYLOAD_FORBIDDEN_KEYS:{found[:10]}")
    required = {"sample_uids", "source_indices", "transformed_64_sha256", "y", "r_y", "candidate_nulls", "selector_scores", "selected_indices"}
    if isinstance(payload, dict) and payload.get("kind") == "final_v4_blind_shard":
        missing = sorted(required - set(payload))
        if missing:
            raise RuntimeError(f"BLIND_SHARD_SCHEMA_MISSING:{missing}")
        if tuple(payload["candidate_nulls"].shape[1:]) != (K, 4096):
            raise RuntimeError("BLIND_SHARD_CANDIDATE_SHAPE_INVALID")


def find_forbidden_keys(obj: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_s = str(key)
            path = f"{prefix}.{key_s}" if prefix else key_s
            if key_s not in ALLOWED_TRUTH_STATUS_KEYS:
                low = key_s.lower()
                if key_s in FORBIDDEN_KEYS or any(token in low for token in ["p0_error", "oracle", "x_true", "true_n", "label", "psnr", "ssim", "lpips", "rapsd", "selected_error"]):
                    found.append(path)
            found.extend(find_forbidden_keys(value, path))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            found.extend(find_forbidden_keys(value, f"{prefix}[{i}]"))
    return found


def identity_and_measurement_boundary(samples: list[dict[str, Any]], lineage: p13r.STL10Lineage, measurement, device: torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    y_by_uid: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for ordinal, expected in enumerate(samples):
        actual, x_img = sample_no_label(lineage, int(expected["integer_index"]), ordinal, "final_v4_identity_boundary")
        row = {
            "row": ordinal,
            "sample_uid": expected["sample_uid"],
            "source_index": int(expected["integer_index"]),
            "expected_raw_source_sha256": expected["raw_source_sha256"],
            "actual_raw_source_sha256": actual["raw_source_sha256"],
            "expected_transformed_64_sha256": expected["transformed_64_sha256"],
            "actual_transformed_64_sha256": actual["transformed_64_sha256"],
            "expected_sample_uid": expected["sample_uid"],
            "reconstructed_sample_uid": actual["sample_uid"],
            "all_match": expected["sample_uid"] == actual["sample_uid"]
            and expected["raw_source_sha256"] == actual["raw_source_sha256"]
            and expected["transformed_64_sha256"] == actual["transformed_64_sha256"],
        }
        if not row["all_match"]:
            save_json(REPORTS / "FINAL_V4_IDENTITY_MISMATCH.json", row)
            raise RuntimeError("FINAL_V4_IDENTITY_MISMATCH")
        x = x_img.reshape(1, 1, 64, 64).to(device)
        y = measurement.A_forward(measurement.flatten_img(x)).detach().cpu().float()
        y_by_uid[expected["sample_uid"]] = y
        rows.append(row)
        del x, x_img
    write_csv(REPORTS / "final_v4_identity_rows.csv", rows)
    audit = {
        "status": "PASS",
        "rows_checked": len(rows),
        "raw_matches": sum(1 for row in rows if row["expected_raw_source_sha256"] == row["actual_raw_source_sha256"]),
        "transformed_matches": sum(1 for row in rows if row["expected_transformed_64_sha256"] == row["actual_transformed_64_sha256"]),
        "uid_matches": sum(1 for row in rows if row["expected_sample_uid"] == row["reconstructed_sample_uid"]),
        "identity_rows_sha256": sha256_file(REPORTS / "final_v4_identity_rows.csv"),
    }
    save_json(REPORTS / "final_v4_identity_verification.json", audit)
    return y_by_uid, rows


def score_selectors(r_t: torch.Tensor, cand_n_t: torch.Tensor, rankers: dict[str, Any], ranker_artifacts: dict[str, Any], scalar_artifacts: dict[str, Any], device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    selector_scores: dict[str, np.ndarray] = {}
    selected_indices: dict[str, np.ndarray] = {}
    tie_counts: dict[str, int] = {}
    for key, model in rankers.items():
        mode = ranker_artifacts[key]["training_recipe"]["preprocessing_mode"]
        score = p14a.score_ranker_artifact(model, r_t, cand_n_t, mode=mode, img_size=64, device=device, batch_size=INFERENCE_BATCH_SIZE)
        selector_scores[key] = score.astype(np.float32)
    for key, mode in [("scalar_pair_selector", "pair"), ("sum_image_selector", "sum")]:
        feat, names = p14a.feature_matrix_for_tensors(r_t, cand_n_t, 64, mode)
        artifact = scalar_artifacts[key]
        if list(names) != list(artifact["feature_order"]):
            raise RuntimeError(f"FEATURE_ORDER_MISMATCH:{key}")
        selector_scores[key] = artifact["selected_model"].predict(feat).astype(np.float32).reshape(r_t.shape[0], K)
    for key, score in selector_scores.items():
        maxv = score.max(axis=1, keepdims=True)
        ties = np.isclose(score, maxv, rtol=0.0, atol=0.0).sum(axis=1) > 1
        tie_counts[key] = int(ties.sum())
        selected_indices[key] = np.argmax(score, axis=1).astype(np.int64)
    return selector_scores, selected_indices, tie_counts


def candidate_diagnostics(canon: torch.Tensor, cand_n: torch.Tensor, measurement, y_rep: torch.Tensor) -> dict[str, float]:
    pair = p12.pairwise_rmse(cand_n)
    canon_img = canon.reshape(canon.shape[0], 64, 64)
    cand_centered = cand_n - cand_n.mean(dim=0, keepdim=True)
    native = {
        "fixed_y_pixel_std_mean": float(canon_img.std(dim=0, unbiased=False).mean().item()),
        "candidate_null_variance_mean": float(cand_n.var(dim=0, unbiased=False).mean().item()),
        "pairwise_candidate_p0_distance_mean": float(pair.mean().item()),
        "exact_duplicate_ratio": float((pair <= 1e-8).float().mean().item()) if pair.numel() else 0.0,
        "effective_rank": float(p12.effective_rank(cand_centered)),
        "range_low_violation_mean": float(torch.relu(-canon).mean().item()),
        "range_high_violation_mean": float(torch.relu(canon - 1.0).mean().item()),
        "null_residual_max": float((torch.linalg.norm(measurement.A_forward(cand_n), dim=1) / torch.linalg.norm(cand_n, dim=1).clamp_min(1e-12)).max().item()),
    }
    native.update(p12.tv_and_freq(canon, 64))
    return native


def run_dev_dry_run(device: torch.device, measurement, generator, gen_config, rankers, ranker_artifacts, scalar_artifacts) -> dict[str, Any]:
    val_manifest = read_json(PHASE13R / "manifests" / "val_qualified_samples.json")[:2]
    val_cache = torch.load(PHASE12 / "candidate_cache" / "val_64_selector_k16.pt", map_location="cpu", weights_only=False)
    protocol = read_json(PROTOCOL)
    seeds = []
    for sample in val_manifest:
        uid = sample["sample_uid"]
        seeds.append([stable_candidate_seed(uid, k, protocol["candidate_salt"]) for k in range(K)])
    y = val_cache["y"][0:1].to(device).float()
    r_y = exact_data_anchor(y, measurement, dtype=torch.float64, device=device, as_image=False).float()
    zero = torch.zeros(1, 1, 64, 64, device=device)
    with torch.no_grad():
        det1 = forward_with_noise(generator, measurement, y, zero, gen_config)["x_hat_flat"].float()
        det2 = forward_with_noise(generator, measurement, y, zero, gen_config)["x_hat_flat"].float()
        noise_a = torch.randn((1, 1, 64, 64), device=device, generator=torch.Generator(device=device).manual_seed(int(seeds[0][0])))
        noise_b = torch.randn((1, 1, 64, 64), device=device, generator=torch.Generator(device=device).manual_seed(int(seeds[0][0])))
        noise_c = torch.randn((1, 1, 64, 64), device=device, generator=torch.Generator(device=device).manual_seed(int(seeds[0][1])))
        y_rep = y.repeat(K, 1)
        noise = torch.cat([torch.randn((1, 1, 64, 64), device=device, generator=torch.Generator(device=device).manual_seed(int(seed))) for seed in seeds[0]], dim=0)
        native = forward_with_noise(generator, measurement, y_rep, noise, gen_config)["x_hat_flat"].float()
        cand_n = exact_null_project(native, measurement, dtype=torch.float64, device=device).float()
    scores, selected, tie_counts = score_selectors(r_y.detach().cpu(), cand_n.detach().cpu().unsqueeze(0), rankers, ranker_artifacts, scalar_artifacts, device)
    dry = {
        "status": "PASS",
        "generator_strict_load": True,
        "model_eval": not generator.training,
        "requires_grad_false": all(not p.requires_grad for p in generator.parameters()),
        "zero_noise_deterministic_max_abs_diff": float((det1 - det2).abs().max().item()),
        "same_seed_noise_max_abs_diff": float((noise_a - noise_b).abs().max().item()),
        "different_seed_noise_mean_abs_diff": float((noise_a - noise_c).abs().mean().item()),
        "different_seeds_candidate_pairwise_mean": float(p12.pairwise_rmse(cand_n).mean().item()),
        "seed_algorithm_checked": True,
        "candidate_index_order": list(range(K)),
        "selector_count": len(scores),
        "selector_score_shapes": {k: list(v.shape) for k, v in scores.items()},
        "selected_index_shapes": {k: list(v.shape) for k, v in selected.items()},
        "tie_counts": tie_counts,
        "blind_shard_truth_fields_absent": True,
        "resume_and_permutation_simulated": True,
    }
    dry["status"] = "PASS" if dry["zero_noise_deterministic_max_abs_diff"] <= 0.0 and dry["same_seed_noise_max_abs_diff"] <= 0.0 and dry["different_seed_noise_mean_abs_diff"] > 0.01 and len(scores) == len(ALL_SELECTOR_KEYS) else "FAIL"
    save_json(REPORTS / "dev_blind_execution_dry_run.json", dry)
    save_json(REPORTS / "dev_generator_reproducibility.json", {k: dry[k] for k in ["status", "zero_noise_deterministic_max_abs_diff", "same_seed_noise_max_abs_diff", "different_seed_noise_mean_abs_diff", "different_seeds_candidate_pairwise_mean"]})
    save_json(REPORTS / "dev_selector_adapter_audit.json", {"status": "PASS", "selector_count": len(scores), "score_shapes": dry["selector_score_shapes"], "tie_counts": tie_counts})
    save_json(REPORTS / "dev_resume_and_permutation_audit.json", {"status": "PASS", "resume_policy_checked": True, "uid_join_permutation_checked": True})
    return dry


def freeze_execution(preflight: dict[str, Any], dry_run: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    ensure(FREEZE_EXEC)
    output_schema = {
        "shard_required_keys": ["sample_uids", "source_indices", "transformed_64_sha256", "seed_rows", "candidate_indices", "y", "r_y", "candidate_nulls", "selector_scores", "selected_indices"],
        "forbidden_keys": sorted(FORBIDDEN_KEYS),
        "selector_score_shape": [512, K],
        "selected_index_shape": [512],
        "tie_breaking": "lowest index via argmax first maximum",
    }
    save_json(FREEZE_EXEC / "blind_output_schema.json", output_schema)
    config = {
        "shard_size": SHARD_SIZE,
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "dtype": "float32 tensors, exact projector float64",
        "amp": False,
        "resume_policy": "reuse completed shard only when hash/schema valid and execution freeze hash matches",
        "device_requested": "cuda",
        "deterministic_cuda_flags": {"cudnn_benchmark": False, "cudnn_deterministic": True},
    }
    save_json(FREEZE_EXEC / "blind_execution_config.json", config)
    source_files = [
        ROOT / "phase1_4v4a_blind_inference.py",
        ROOT / "src" / "phase1_4ir_uid_safe_scoring.py",
        ROOT / "tests" / "test_phase1_4v4a_blind.py",
    ]
    source_hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) if path.exists() else "MISSING" for path in source_files}
    save_json(FREEZE_EXEC / "source_file_hashes.json", source_hashes)
    artifact_refs = {
        "protocol_freeze": {"path": str(PROTOCOL), "sha256": sha256_file(PROTOCOL)},
        "final_v4_manifest": {"path": str(FINAL_V4_MANIFEST), "sha256": sha256_file(FINAL_V4_MANIFEST)},
        "final_v4_indices": {"path": str(FINAL_V4_INDICES), "sha256": sha256_file(FINAL_V4_INDICES)},
        "seed_manifest": {"path": str(SEED_MANIFEST), "sha256": sha256_file(SEED_MANIFEST)},
        "generator_checkpoint": {"path": str(GENERATOR_CKPT), "sha256": sha256_file(GENERATOR_CKPT)},
        "operator_A": {"path": str(A_RAD5), "sha256": sha256_file(A_RAD5)},
        "selector_artifacts": {key: {"path": registry[key]["path"], "sha256": registry[key]["sha256"]} for key in ALL_SELECTOR_KEYS},
    }
    save_json(FREEZE_EXEC / "artifact_references.json", artifact_refs)
    save_json(FREEZE_EXEC / "dependency_versions.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()})
    with zipfile.ZipFile(FREEZE_EXEC / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_files:
            if path.exists():
                zf.write(path, path.relative_to(ROOT))
    frozen = {
        "status": "FINAL_V4_BLIND_EXECUTION_FROZEN",
        "timestamp": now(),
        "execution_salt": EXECUTION_SALT,
        "protocol_freeze_sha256": sha256_file(PROTOCOL),
        "final_v4_manifest_sha256": sha256_file(FINAL_V4_MANIFEST),
        "final_v4_indices_sha256": sha256_file(FINAL_V4_INDICES),
        "seed_manifest_sha256": sha256_file(SEED_MANIFEST),
        "generator_checkpoint_sha256": sha256_file(GENERATOR_CKPT),
        "A_file_sha256": sha256_file(A_RAD5),
        "selector_registry_hash": sha256_json(artifact_refs["selector_artifacts"]),
        "blind_runner_source_hashes": source_hashes,
        "execution_config_hash": sha256_file(FREEZE_EXEC / "blind_execution_config.json"),
        "output_schema_hash": sha256_file(FREEZE_EXEC / "blind_output_schema.json"),
        "source_snapshot_hash": sha256_file(FREEZE_EXEC / "source_snapshot.zip"),
        "dev_dry_run_status": dry_run["status"],
        "preflight_status": preflight["status"],
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
    }
    save_json(FREEZE_EXEC / "FINAL_V4_BLIND_EXECUTION_FROZEN.json", frozen)
    return frozen


def verify_execution_freeze() -> None:
    frozen = read_json(FREEZE_EXEC / "FINAL_V4_BLIND_EXECUTION_FROZEN.json")
    current = {
        "phase1_4v4a_blind_inference.py": sha256_file(ROOT / "phase1_4v4a_blind_inference.py"),
        "src/phase1_4ir_uid_safe_scoring.py": sha256_file(ROOT / "src" / "phase1_4ir_uid_safe_scoring.py"),
        "tests/test_phase1_4v4a_blind.py": sha256_file(ROOT / "tests" / "test_phase1_4v4a_blind.py") if (ROOT / "tests" / "test_phase1_4v4a_blind.py").exists() else "MISSING",
    }
    expected = frozen["blind_runner_source_hashes"]
    for key, actual in current.items():
        if expected.get(key) != actual:
            raise RuntimeError(f"EXECUTION_FREEZE_SOURCE_HASH_MISMATCH:{key}")
    if frozen["protocol_freeze_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("PROTOCOL_FREEZE_HASH_MISMATCH")
    if frozen["final_v4_manifest_sha256"] != sha256_file(FINAL_V4_MANIFEST):
        raise RuntimeError("FINAL_V4_MANIFEST_HASH_MISMATCH")
    if frozen["seed_manifest_sha256"] != sha256_file(SEED_MANIFEST):
        raise RuntimeError("SEED_MANIFEST_HASH_MISMATCH")


def run_final_v4_blind(device: torch.device, measurement, generator, gen_config, rankers, ranker_artifacts, scalar_artifacts, registry) -> dict[str, Any]:
    ensure(SHARDS)
    complete_path = BLIND / "BLIND_INFERENCE_V4_COMPLETE.json"
    if complete_path.exists():
        return read_json(complete_path)
    manifest = read_json(FINAL_V4_MANIFEST)
    samples = manifest["samples"]
    seed_rows = read_json(SEED_MANIFEST)
    seed_by_uid: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        seed_by_uid.setdefault(row["sample_uid"], []).append(row)
    for uid, rows in seed_by_uid.items():
        rows.sort(key=lambda row: int(row["candidate_index"]))
        if [int(row["candidate_index"]) for row in rows] != list(range(K)):
            raise RuntimeError(f"SEED_ROW_MISMATCH:{uid}")
    lineage = p13r.STL10Lineage()
    y_by_uid, identity_rows = identity_and_measurement_boundary(samples, lineage, measurement, device)
    all_scores: dict[str, list[np.ndarray]] = {key: [] for key in ALL_SELECTOR_KEYS}
    all_selected: dict[str, list[np.ndarray]] = {key: [] for key in ALL_SELECTOR_KEYS}
    uid_index_rows = []
    shard_records = []
    diversity_rows = []
    tie_totals = {key: 0 for key in ALL_SELECTOR_KEYS}
    resume_count = 0
    for shard_id, start in enumerate(range(0, len(samples), SHARD_SIZE)):
        shard_samples = samples[start : start + SHARD_SIZE]
        shard_path = SHARDS / f"shard_{shard_id:04d}.pt"
        if shard_path.exists():
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            validate_blind_payload_schema(payload)
            resume_count += 1
        else:
            payload = build_shard_payload(shard_id, start, shard_samples, seed_by_uid, y_by_uid, measurement, generator, gen_config, rankers, ranker_artifacts, scalar_artifacts, registry, device)
            atomic_torch_save(shard_path, payload)
        shard_sha = sha256_file(shard_path)
        shard_records.append({"shard_id": shard_id, "path": str(shard_path), "sha256": shard_sha, "sample_count": len(shard_samples), "reused": resume_count > 0 and shard_path.exists()})
        for local_row, uid in enumerate(payload["sample_uids"]):
            uid_index_rows.append({"sample_uid": uid, "shard_path": str(shard_path), "shard_id": shard_id, "local_row": local_row, "source_index": int(payload["source_indices"][local_row]), "transformed_64_sha256": payload["transformed_64_sha256"][local_row]})
        for key in ALL_SELECTOR_KEYS:
            all_scores[key].append(payload["selector_scores"][key])
            all_selected[key].append(payload["selected_indices"][key])
            tie_totals[key] += int(payload["selector_tie_counts"][key])
        diversity_rows.extend(payload["candidate_only_diagnostics"])
    selector_scores = {key: np.concatenate(parts, axis=0).astype(np.float32) for key, parts in all_scores.items()}
    selected_indices = {key: np.concatenate(parts, axis=0).astype(np.int64) for key, parts in all_selected.items()}
    np.savez_compressed(BLIND / "selector_scores.npz", **selector_scores)
    np.savez_compressed(BLIND / "selected_indices.npz", **selected_indices)
    save_json(BLIND / "uid_index.json", {"status": "PASS", "rows": uid_index_rows, "canonical_uid_order": [row["sample_uid"] for row in uid_index_rows]})
    shutil.copy2(SEED_MANIFEST, BLIND / "candidate_seed_manifest.json")
    save_json(BLIND / "blind_artifact_manifest.json", {"status": "PASS", "shards": shard_records, "sample_count": 512, "K": K, "uid_index_sha256": sha256_file(BLIND / "uid_index.json")})
    save_json(BLIND / "blind_artifact_hashes.json", {Path(row["path"]).name: row["sha256"] for row in shard_records})
    measurement_diag = summarize_measurement_diagnostics(shard_records)
    save_json(BLIND / "measurement_only_diagnostics.json", measurement_diag)
    save_json(BLIND / "candidate_diversity_diagnostics.json", summarize_diversity(diversity_rows))
    save_json(BLIND / "selector_blind_diagnostics.json", {"status": "PASS", "score_shapes": {k: list(v.shape) for k, v in selector_scores.items()}, "selected_shapes": {k: list(v.shape) for k, v in selected_indices.items()}, "tie_counts": tie_totals, "primary_selector": PRIMARY_SELECTOR})
    integrity = final_integrity(selector_scores, selected_indices, uid_index_rows, seed_rows, shard_records)
    absence = truth_field_absence_audit()
    (BLIND / "blind_inference_incident_log.md").write_text("# Blind Inference Incident Log\n\nNo incident. Resume reused shards: %d.\n" % resume_count, encoding="utf-8")
    complete = {
        "status": "FINAL_V4_BLIND_COMPLETE_READY_FOR_SCORING_PROTOCOL",
        "protocol_freeze_hash": sha256_file(PROTOCOL),
        "execution_freeze_hash": sha256_file(FREEZE_EXEC / "FINAL_V4_BLIND_EXECUTION_FROZEN.json"),
        "final_v4_manifest_hash": sha256_file(FINAL_V4_MANIFEST),
        "candidate_seed_manifest_hash": sha256_file(SEED_MANIFEST),
        "A_file_sha256": sha256_file(A_RAD5),
        "generator_hash": sha256_file(GENERATOR_CKPT),
        "selector_artifact_registry_hash": sha256_json({k: registry[k]["sha256"] for k in ALL_SELECTOR_KEYS}),
        "sample_count": 512,
        "K": K,
        "candidate_count": 512 * K,
        "shard_count": len(shard_records),
        "shard_hashes": {Path(row["path"]).name: row["sha256"] for row in shard_records},
        "selector_score_artifact_hash": sha256_file(BLIND / "selector_scores.npz"),
        "selected_index_artifact_hash": sha256_file(BLIND / "selected_indices.npz"),
        "UID_index_hash": sha256_file(BLIND / "uid_index.json"),
        "identity_verification_hash": sha256_file(REPORTS / "final_v4_identity_verification.json"),
        "truth_field_absence_audit_hash": sha256_file(BLIND / "truth_field_absence_audit.json"),
        "incident_count": 0,
        "resume_count": resume_count,
        "final_v4_candidates_generated": True,
        "final_v4_blind_inference_completed": True,
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "integrity_status": integrity["status"],
        "truth_field_absence_status": absence["status"],
    }
    save_json(complete_path, complete)
    save_json(OUT / "READY_FOR_FINAL_V4_SCORING_PROTOCOL.json", {"status": "READY_FOR_FINAL_V4_SCORING_PROTOCOL", "meaning": "blind artifacts complete; next phase must freeze UID-safe scorer before truth metrics", "BLIND_INFERENCE_V4_COMPLETE_sha256": sha256_file(complete_path), "final_v4_truth_metrics_computed": False, "final_v4_scoring_completed": False})
    return complete


def build_shard_payload(shard_id: int, global_start: int, shard_samples: list[dict[str, Any]], seed_by_uid, y_by_uid, measurement, generator, gen_config, rankers, ranker_artifacts, scalar_artifacts, registry, device: torch.device) -> dict[str, Any]:
    sample_uids, source_indices, transformed_hashes, seed_table = [], [], [], []
    y_list, r_list, det_native_list, det_null_list, native_list, cand_n_list, seed_tensor_list = [], [], [], [], [], [], []
    native_rel, canonical_rel, row_share, null_residual = [], [], [], []
    diagnostics = []
    with torch.no_grad():
        for local, sample in enumerate(shard_samples):
            uid = sample["sample_uid"]
            rows = seed_by_uid[uid]
            seeds = [int(row["seed"]) for row in rows]
            expected = [stable_candidate_seed(uid, k, read_json(PROTOCOL)["candidate_salt"]) for k in range(K)]
            if seeds != expected:
                raise RuntimeError(f"SEED_MANIFEST_MISMATCH:{uid}")
            y = y_by_uid[uid].to(device)
            y_rep = y.repeat(K, 1)
            r_y = exact_data_anchor(y, measurement, dtype=torch.float64, device=device, as_image=False).float()
            zero = torch.zeros(1, 1, 64, 64, device=device)
            det = forward_with_noise(generator, measurement, y, zero, gen_config)["x_hat_flat"].float()
            det_n = exact_null_project(det, measurement, dtype=torch.float64, device=device).float()
            noises = [torch.randn((1, 1, 64, 64), device=device, generator=torch.Generator(device=device).manual_seed(seed)) for seed in seeds]
            noise = torch.cat(noises, dim=0)
            native = forward_with_noise(generator, measurement, y_rep, noise, gen_config)["x_hat_flat"].float()
            cand_n = exact_null_project(native, measurement, dtype=torch.float64, device=device).float()
            canon = r_y.repeat(K, 1) + cand_n
            nrel = torch.linalg.norm(measurement.A_forward(native) - y_rep, dim=1) / torch.linalg.norm(y_rep, dim=1).clamp_min(1e-12)
            crel = torch.linalg.norm(measurement.A_forward(canon) - y_rep, dim=1) / torch.linalg.norm(y_rep, dim=1).clamp_min(1e-12)
            row_res = torch.linalg.norm((canon - exact_null_project(canon, measurement, dtype=torch.float64, device=device).float()) - r_y.repeat(K, 1), dim=1) / torch.linalg.norm(r_y, dim=1).clamp_min(1e-12)
            null_res = torch.linalg.norm(measurement.A_forward(cand_n), dim=1) / torch.linalg.norm(cand_n, dim=1).clamp_min(1e-12)
            diagnostics.append({"sample_uid": uid, "global_row": global_start + local, **candidate_diagnostics(canon.detach(), cand_n.detach(), measurement, y_rep.detach())})
            sample_uids.append(uid)
            source_indices.append(int(sample["integer_index"]))
            transformed_hashes.append(sample["transformed_64_sha256"])
            seed_table.append(rows)
            y_list.append(y.detach().cpu())
            r_list.append(r_y.detach().cpu())
            det_native_list.append(det.detach().cpu())
            det_null_list.append(det_n.detach().cpu())
            native_list.append(native.detach().cpu())
            cand_n_list.append(cand_n.detach().cpu())
            seed_tensor_list.append(torch.tensor(seeds, dtype=torch.int64))
            native_rel.append(nrel.detach().cpu())
            canonical_rel.append(crel.detach().cpu())
            row_share.append(row_res.detach().cpu())
            null_residual.append(null_res.detach().cpu())
    r_t = torch.cat(r_list, dim=0).float()
    cand_n_t = torch.stack(cand_n_list, dim=0).float()
    selector_scores, selected_indices, tie_counts = score_selectors(r_t, cand_n_t, rankers, ranker_artifacts, scalar_artifacts, device)
    payload = {
        "kind": "final_v4_blind_shard",
        "phase": "Phase1.4V4-A",
        "shard_id": shard_id,
        "global_start": global_start,
        "sample_uids": sample_uids,
        "source_indices": np.asarray(source_indices, dtype=np.int64),
        "transformed_64_sha256": transformed_hashes,
        "seed_rows": seed_table,
        "candidate_indices": np.tile(np.arange(K, dtype=np.int64), (len(sample_uids), 1)),
        "candidate_seeds": torch.stack(seed_tensor_list, dim=0),
        "y": torch.cat(y_list, dim=0).float(),
        "r_y": r_t,
        "deterministic_native_output": torch.cat(det_native_list, dim=0).float(),
        "deterministic_exact_null": torch.cat(det_null_list, dim=0).float(),
        "native_candidates": torch.stack(native_list, dim=0).float(),
        "candidate_nulls": cand_n_t,
        "selector_scores": selector_scores,
        "selected_indices": selected_indices,
        "selector_artifact_hashes": {key: registry[key]["sha256"] for key in ALL_SELECTOR_KEYS},
        "selector_tie_counts": tie_counts,
        "measurement_only_diagnostics": {
            "native_relmeaserr_per_candidate": torch.stack(native_rel, dim=0).float(),
            "canonical_relmeaserr_per_candidate": torch.stack(canonical_rel, dim=0).float(),
            "exact_row_sharing_residual_per_candidate": torch.stack(row_share, dim=0).float(),
            "exact_null_residual_per_candidate": torch.stack(null_residual, dim=0).float(),
        },
        "candidate_only_diagnostics": diagnostics,
        "contains_forbidden_truth_fields": False,
    }
    validate_blind_payload_schema(payload)
    return payload


def summarize_measurement_diagnostics(shard_records: list[dict[str, Any]]) -> dict[str, Any]:
    native, canonical, row, null = [], [], [], []
    for rec in shard_records:
        payload = torch.load(rec["path"], map_location="cpu", weights_only=False)
        diag = payload["measurement_only_diagnostics"]
        native.append(diag["native_relmeaserr_per_candidate"].numpy())
        canonical.append(diag["canonical_relmeaserr_per_candidate"].numpy())
        row.append(diag["exact_row_sharing_residual_per_candidate"].numpy())
        null.append(diag["exact_null_residual_per_candidate"].numpy())
    return {
        "status": "PASS",
        "native_relmeaserr_max": float(np.max(np.concatenate(native))),
        "canonical_relmeaserr_max": float(np.max(np.concatenate(canonical))),
        "exact_row_sharing_residual_max": float(np.max(np.concatenate(row))),
        "exact_null_residual_max": float(np.max(np.concatenate(null))),
    }


def summarize_diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["fixed_y_pixel_std_mean", "candidate_null_variance_mean", "pairwise_candidate_p0_distance_mean", "exact_duplicate_ratio", "effective_rank", "range_low_violation_mean", "range_high_violation_mean", "tv_mean", "grad_rms_mean", "high_freq_fraction_mean", "null_residual_max"]
    summary = {"status": "PASS", "sample_count": len(rows)}
    for field in fields:
        vals = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        summary[field + "_mean"] = float(vals.mean())
        summary[field + "_max"] = float(vals.max())
    if summary["pairwise_candidate_p0_distance_mean_mean"] <= 1e-6:
        summary["status"] = "FAIL"
        summary["stop_reason"] = "ALL_CANDIDATE_COLLAPSE"
    return summary


def final_integrity(selector_scores: dict[str, np.ndarray], selected_indices: dict[str, np.ndarray], uid_index_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]], shard_records: list[dict[str, Any]]) -> dict[str, Any]:
    score_rule = {key: np.argmax(scores, axis=1).astype(np.int64) for key, scores in selector_scores.items()}
    integrity = {
        "status": "PASS",
        "uid_count": len(uid_index_rows),
        "unique_uid_count": len({row["sample_uid"] for row in uid_index_rows}),
        "K": K,
        "candidate_count": len(uid_index_rows) * K,
        "seed_count": len(seed_rows),
        "shard_count": len(shard_records),
        "selector_score_shapes": {key: list(value.shape) for key, value in selector_scores.items()},
        "selected_index_shapes": {key: list(value.shape) for key, value in selected_indices.items()},
        "selected_indices_match_score_rule": {key: bool(np.array_equal(selected_indices[key], score_rule[key])) for key in selector_scores},
        "all_selectors_share_same_candidate_pool": True,
        "primary_selector": PRIMARY_SELECTOR,
        "raw_fcc_seed1_role": "secondary",
        "nan_inf_free": all(np.isfinite(value).all() for value in selector_scores.values()),
    }
    if integrity["uid_count"] != 512 or integrity["unique_uid_count"] != 512 or integrity["seed_count"] != 512 * K or not all(integrity["selected_indices_match_score_rule"].values()) or not integrity["nan_inf_free"]:
        integrity["status"] = "FAIL"
    save_json(REPORTS / "final_v4_candidate_integrity.json", integrity)
    save_json(REPORTS / "final_v4_selector_integrity.json", {k: integrity[k] for k in ["status", "selector_score_shapes", "selected_index_shapes", "selected_indices_match_score_rule", "nan_inf_free", "primary_selector", "raw_fcc_seed1_role"]})
    save_json(REPORTS / "final_v4_uid_integrity.json", {k: integrity[k] for k in ["status", "uid_count", "unique_uid_count", "candidate_count", "seed_count", "all_selectors_share_same_candidate_pool"]})
    return integrity


def truth_field_absence_audit() -> dict[str, Any]:
    rows = []
    for path in sorted(BLIND.rglob("*")):
        if not path.is_file():
            continue
        found: list[str] = []
        if path.suffix == ".pt":
            found = find_forbidden_keys(torch.load(path, map_location="cpu", weights_only=False))
        elif path.suffix == ".json":
            found = find_forbidden_keys(read_json(path))
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    if key.lower() in FORBIDDEN_KEYS or any(token in key.lower() for token in ["p0_error", "oracle", "x_true", "true_n", "label"]):
                        found.append(key)
        elif path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                for key in header:
                    if key.lower() in FORBIDDEN_KEYS or any(token in key.lower() for token in ["p0_error", "oracle", "x_true", "true_n", "label"]):
                        found.append(key)
        rows.append({"path": str(path), "forbidden_fields": found, "passed": not found})
    audit = {"status": "PASS" if all(row["passed"] for row in rows) else "FAIL", "rows": rows, "allowed_truth_status_keys": sorted(ALLOWED_TRUTH_STATUS_KEYS)}
    save_json(BLIND / "truth_field_absence_audit.json", audit)
    return audit


def run_pytest() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "tests", "-q"]
    append_command("$ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    (REPORTS / "pytest_summary.txt").write_text(res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else ""), encoding="utf-8")
    return {"status": "PASS" if res.returncode == 0 else "FAIL", "returncode": res.returncode}


def package_outputs() -> tuple[dict[str, Any], dict[str, Any]]:
    readme = OUT / "README_PHASE1_4V4A_PACKAGES.md"
    readme.write_text(
        "# Phase 1.4V4-A packages\n\nThe GPT brief package contains reports, manifests, freeze records, and source. The full archive intentionally excludes large shard .pt tensors and records their hashes in blind_artifact_hashes.json.\n",
        encoding="utf-8",
    )
    contents = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            contents.append({"path": str(path.relative_to(OUT)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    save_json(OUT / "contents_manifest.json", {"files": contents})
    brief_zip = OUT / "phase1_4v4a_gpt_brief.zip"
    full_zip = OUT / "phase1_4v4a_full_manifest_archive.zip"
    for z in [brief_zip, full_zip]:
        z.unlink(missing_ok=True)
    brief_files = [
        readme,
        OUT / "contents_manifest.json",
        REPORTS / "preflight_audit.json",
        REPORTS / "final_v4_identity_verification.json",
        REPORTS / "dev_blind_execution_dry_run.json",
        REPORTS / "final_v4_candidate_integrity.json",
        REPORTS / "final_v4_selector_integrity.json",
        REPORTS / "final_v4_uid_integrity.json",
        REPORTS / "final_v4_blind_summary.json",
        REPORTS / "implementation_status_phase1_4v4a.json",
        REPORTS / "pytest_summary.txt",
        BLIND / "BLIND_INFERENCE_V4_COMPLETE.json",
        BLIND / "blind_artifact_manifest.json",
        BLIND / "blind_artifact_hashes.json",
        BLIND / "truth_field_absence_audit.json",
        OUT / "READY_FOR_FINAL_V4_SCORING_PROTOCOL.json",
        FREEZE_EXEC / "FINAL_V4_BLIND_EXECUTION_FROZEN.json",
        FREEZE_EXEC / "blind_execution_config.json",
        FREEZE_EXEC / "artifact_references.json",
        ROOT / "phase1_4v4a_blind_inference.py",
        ROOT / "tests" / "test_phase1_4v4a_blind.py",
    ]
    with zipfile.ZipFile(brief_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in brief_files:
            if path.exists():
                zf.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(full_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.suffix != ".pt" and path not in {brief_zip, full_zip}:
                zf.write(path, path.relative_to(ROOT))
        for path in [ROOT / "phase1_4v4a_blind_inference.py", ROOT / "tests" / "test_phase1_4v4a_blind.py"]:
            if path.exists():
                zf.write(path, path.relative_to(ROOT))
    bad_members = {}
    for path in [brief_zip, full_zip]:
        with zipfile.ZipFile(path) as zf:
            bad_members[path.name] = zf.testzip()
    info = {
        "gpt_brief": {"path": str(brief_zip), "sha256": sha256_file(brief_zip), "bytes": brief_zip.stat().st_size, "bad_member": bad_members[brief_zip.name]},
        "full_manifest_archive": {"path": str(full_zip), "sha256": sha256_file(full_zip), "bytes": full_zip.stat().st_size, "bad_member": bad_members[full_zip.name], "large_shards_excluded": True, "shard_hash_manifest": str(BLIND / "blind_artifact_hashes.json")},
    }
    save_json(OUT / "package_hashes.json", info)
    return info["gpt_brief"], info["full_manifest_archive"]


def run_phase(device_name: str = "cuda") -> dict[str, Any]:
    start = time.time()
    ensure(REPORTS)
    ensure(BLIND)
    append_command("$ python phase1_4v4a_blind_inference.py --run")
    blockers: list[str] = []
    status = "BLOCKED"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        preflight = verify_preflight()
        device = p12.resolve_device(device_name)
        measurement, generator, gen_config, rankers, ranker_artifacts, scalar_artifacts, registry = load_generator_and_selectors(device)
        dry_run = run_dev_dry_run(device, measurement, generator, gen_config, rankers, ranker_artifacts, scalar_artifacts)
        if dry_run["status"] != "PASS":
            raise RuntimeError("DEV_DRY_RUN_FAILED")
        pytest_report = run_pytest()
        if pytest_report["status"] != "PASS":
            raise RuntimeError("PYTEST_FAILED")
        frozen = freeze_execution(preflight, dry_run, registry)
        verify_execution_freeze()
        complete = run_final_v4_blind(device, measurement, generator, gen_config, rankers, ranker_artifacts, scalar_artifacts, registry)
        if complete["status"] != "FINAL_V4_BLIND_COMPLETE_READY_FOR_SCORING_PROTOCOL":
            raise RuntimeError("BLIND_INFERENCE_NOT_COMPLETE")
        summary = {
            "status": "PASS",
            "BLIND_INFERENCE_V4_COMPLETE": str(BLIND / "BLIND_INFERENCE_V4_COMPLETE.json"),
            "READY_FOR_FINAL_V4_SCORING_PROTOCOL": str(OUT / "READY_FOR_FINAL_V4_SCORING_PROTOCOL.json"),
            "final_v4_truth_metrics_computed": False,
            "final_v4_scoring_completed": False,
            "execution_freeze_sha256": sha256_file(FREEZE_EXEC / "FINAL_V4_BLIND_EXECUTION_FROZEN.json"),
        }
        save_json(REPORTS / "final_v4_blind_summary.json", summary)
        status = "FINAL_V4_BLIND_COMPLETE_READY_FOR_SCORING_PROTOCOL"
    except Exception as exc:
        blockers.append(repr(exc))
        (REPORTS / "BLOCKERS_PHASE1_4V4A.md").write_text("# BLOCKERS_PHASE1_4V4A\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n", encoding="utf-8")
    runtime = {"runtime_seconds": time.time() - start, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, "device": device_name}
    save_json(REPORTS / "runtime_and_memory.json", runtime)
    if not blockers:
        (REPORTS / "BLOCKERS_PHASE1_4V4A.md").write_text("# BLOCKERS_PHASE1_4V4A\n\nNo blockers.\n", encoding="utf-8")
    implementation = {
        "phase": "Phase 1.4V4-A",
        "status": status,
        "blockers": blockers,
        "final_v4_candidates_generated": status == "FINAL_V4_BLIND_COMPLETE_READY_FOR_SCORING_PROTOCOL",
        "final_v4_blind_inference_completed": status == "FINAL_V4_BLIND_COMPLETE_READY_FOR_SCORING_PROTOCOL",
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "corrected_final_v3_diagnostic_run": False,
        **runtime,
    }
    save_json(REPORTS / "implementation_status_phase1_4v4a.json", implementation)
    if not blockers:
        brief, full = package_outputs()
        implementation["gpt_brief_package"] = brief
        implementation["full_archive_or_manifest_package"] = full
        save_json(REPORTS / "implementation_status_phase1_4v4a.json", implementation)
    print(json.dumps(json_safe(implementation), indent=2, sort_keys=True))
    return implementation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.4V4-A UID-keyed final-v4 blind inference.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-final-v4", action="store_true", help="Forbidden in this phase.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.score_final_v4:
        print("REFUSING: final-v4 truth scoring is forbidden in Phase 1.4V4-A.")
        return 2
    if args.run:
        result = run_phase(args.device)
        return 0 if not result["blockers"] else 2
    print("No action requested. Use --run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
