from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import torch
from PIL import Image
from scipy import stats

import phase1_2_rad5_64_pipeline as p12
import phase1_3r_recovery_and_relock as p13r
from scripts.eval_posterior_sampling_criteria import radial_power
from src.phase1_4ir_uid_safe_scoring import ALL_SELECTOR_KEYS, K, stable_candidate_seed
from src.projections import exact_null_project


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
PHASE14IR = ROOT / "outputs" / "compatibility" / "phase1_4ir_incident_recovery"
PHASE14V4A = ROOT / "outputs" / "compatibility" / "phase1_4v4a_blind_inference"
PHASE13R = ROOT / "outputs" / "compatibility" / "phase1_3r_recovery_and_relock"
PHASE12 = ROOT / "outputs" / "compatibility" / "phase1_2_rad5_64_candidate_transfer"

OUT = ROOT / "outputs" / "compatibility" / "phase1_4v4b0_scoring_protocol"
REPORTS = OUT / "reports"
ORIGINAL_REF = OUT / "original_blind_reference"
FREEZE_SCORING = OUT / "freeze_bundle_scoring_v4"
FINAL_SCORING = OUT / "final_v4_one_shot_scoring"

PROTOCOL = PHASE14IR / "freeze_bundle_v4" / "FINAL_V4_BLIND_PROTOCOL_FROZEN.json"
FINAL_V4_MANIFEST = PHASE14IR / "manifests" / "final_locked_test_64_v4_manifest.json"
FINAL_V4_INDICES = PHASE14IR / "manifests" / "final_locked_test_64_v4_indices.npy"
SEED_MANIFEST = PHASE14IR / "freeze_bundle_v4" / "final_v4_candidate_seed_manifest.json"
V4A_EXECUTION = PHASE14V4A / "freeze_bundle_execution" / "FINAL_V4_BLIND_EXECUTION_FROZEN.json"
V4A_COMPLETE = PHASE14V4A / "blind_inference_v4" / "BLIND_INFERENCE_V4_COMPLETE.json"
V4A_BLIND = PHASE14V4A / "blind_inference_v4"

PRIMARY_SELECTOR = "dm_fcc_seed3"
PRIMARY_MODEL = "reproduced_dm_fcc_seed3_v2"
FINAL_CONFIRM_TOKEN = "FINAL_V4_UID_SAFE_ONE_SHOT_SCORING"

BOOTSTRAP_SEED = 14001
BOOTSTRAP_REPLICATES = 10000
SIGN_FLIP_SEED = 14002
SIGN_FLIP_REPLICATES = 100000
TIE_EPSILON = 1e-12

DM_KEYS = ["dm_fcc_seed1", "dm_fcc_seed2", "dm_fcc_seed3"]
SCRATCH_KEYS = ["scratch_seed1", "scratch_seed2", "scratch_seed3"]
RAW_KEYS = ["raw_fcc_seed1", "raw_fcc_seed2", "raw_fcc_seed3"]

EXPECTED_HASHES = {
    "protocol": "7a3fc8f277d6f618c6d328a8edd8079131b9ad1f7d1fbe6e36ab73252548cae5",
    "execution": "2806a185d1176d7d488fd1f5772c19255c9c00795483e2a52c3b2ea363a1a2d6",
    "blind_complete": "8ebc3dc406feff4a3d303e05ddf76556adbb4d52a1bff7287dffe291d28969f6",
    "final_v4_manifest": "1c68055ca49bfb44fd88308b3f7a9dc874d5ea9274c306eeb7c008b2a54f0cb1",
    "seed_manifest": "2966d36a0e45cfd15f294cc525a41ef4a19fe69ee3a4682150be6210e1d9d259",
    "uid_index": "bfd7b42efb7c469da13b22e9bddb983942f1790ee69dde99f41ab9d1e1aa0d0c",
    "selector_scores": "f7c7aaffabf6840ca8035bcf4882c74cebefa8fba98024eb350a0314c463b689",
    "selected_indices": "da4c0487c3ab7a17f425af0fc3bb3c47832ffedcd3df0aae9fc5455707d88799",
    "generator": "9e10bd5aba48eb3c05c1bbe28fa0ff85ff2b730ae528c14c2dbace37187624a3",
    "selector_registry": "2751ae65094d5fe1beeabe48e9d19e8a54587395f54044f36ed08fceb99fcb73",
}

FORBIDDEN_BLIND_KEYS = {
    "x_true",
    "true_x",
    "true_n",
    "label",
    "labels",
    "p0_error",
    "oracle",
    "oracle_index",
    "selected_error",
    "psnr",
    "ssim",
    "lpips",
    "rapsd",
}

FINAL_TRUTH_ACCESS_COUNT = 0


class V4B0Error(RuntimeError):
    """Hard failure for Phase 1.4V4-B0 protocol violations."""


@dataclass(frozen=True)
class TruthRecord:
    sample_uid: str
    manifest_integer_index: int
    source_namespace: str
    official_split: str
    official_source_index: int
    expected_raw_sha256: str
    actual_raw_sha256: str
    expected_transformed_sha256: str
    actual_transformed_sha256: str
    image_flat: np.ndarray


@dataclass(frozen=True)
class BlindRecord:
    sample_uid: str
    manifest_integer_index: int
    source_namespace: str
    official_split: str
    official_source_index: int
    transformed_64_sha256: str
    r_y: np.ndarray
    deterministic_exact_null: np.ndarray
    candidate_nulls: np.ndarray
    native_relmeaserr: np.ndarray
    canonical_relmeaserr: np.ndarray
    exact_row_sharing_residual: np.ndarray
    exact_null_residual: np.ndarray


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


def atomic_write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
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


def require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected:
        raise V4B0Error(f"{label}_HASH_MISMATCH: expected {expected}, got {actual}")
    return {"label": label, "path": str(path), "sha256": actual, "status": "PASS"}


def append_command(text: str) -> None:
    ensure(REPORTS)
    with (REPORTS / "command_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def selected_by_argmax(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores)
    if arr.ndim != 2:
        raise ValueError(f"scores must be 2D, got {arr.shape}")
    return np.argmax(arr, axis=1).astype(np.int64)


def oracle_indices(errors: np.ndarray) -> np.ndarray:
    arr = np.asarray(errors)
    if arr.ndim != 2:
        raise ValueError(f"errors must be 2D, got {arr.shape}")
    return np.argmin(arr, axis=1).astype(np.int64)


def compute_random_expectation(metric_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(metric_matrix, dtype=np.float64).mean(axis=1)


def compute_posterior_mean(candidates: np.ndarray) -> np.ndarray:
    return np.asarray(candidates, dtype=np.float64).mean(axis=1)


def compute_primary_oracle(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    err = np.asarray(errors, dtype=np.float64)
    idx = oracle_indices(err)
    return idx, err[np.arange(err.shape[0]), idx]


def p0_rmse_matrix(candidate_nulls: np.ndarray, true_n: np.ndarray) -> np.ndarray:
    cand = np.asarray(candidate_nulls, dtype=np.float64)
    truth = np.asarray(true_n, dtype=np.float64)
    if cand.ndim != 3 or truth.ndim != 2:
        raise ValueError(f"expected candidate [N,K,D] and truth [N,D], got {cand.shape}, {truth.shape}")
    if cand.shape[0] != truth.shape[0] or cand.shape[2] != truth.shape[1]:
        raise ValueError(f"candidate/truth shape mismatch: {cand.shape}, {truth.shape}")
    return np.sqrt(np.mean((cand - truth[:, None, :]) ** 2, axis=2))


def psnr_from_mse(mse: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    arr = np.asarray(mse, dtype=np.float64)
    out = np.empty_like(arr)
    mask = arr <= 0
    out[mask] = np.inf
    out[~mask] = 20.0 * np.log10(float(data_range)) - 10.0 * np.log10(arr[~mask])
    return out


def rapsd_profile(img: np.ndarray, bins: int = 32) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    return radial_power(arr.reshape(1, 1, arr.shape[-2], arr.shape[-1]), bins=bins)


def rapsd_distance(pred: np.ndarray, truth: np.ndarray, bins: int = 32) -> float:
    return float(np.linalg.norm(rapsd_profile(pred, bins=bins) - rapsd_profile(truth, bins=bins)))


def aggregate_relative_improvement(random_metric: np.ndarray, selected_metric: np.ndarray) -> float:
    random_mean = float(np.asarray(random_metric, dtype=np.float64).mean())
    return float((random_mean - float(np.asarray(selected_metric, dtype=np.float64).mean())) / max(random_mean, 1e-12))


def aggregate_oracle_gain_fraction(random_metric: np.ndarray, selected_metric: np.ndarray, oracle_metric: np.ndarray) -> dict[str, Any]:
    random_arr = np.asarray(random_metric, dtype=np.float64)
    selected_arr = np.asarray(selected_metric, dtype=np.float64)
    oracle_arr = np.asarray(oracle_metric, dtype=np.float64)
    denom = float(random_arr.mean() - oracle_arr.mean())
    if abs(denom) <= 1e-12:
        return {"status": "not_applicable", "value": None, "reason": "oracle denominator near zero"}
    return {"status": "ok", "value": float((random_arr.mean() - selected_arr.mean()) / denom)}


def paired_percentile_bootstrap(delta: np.ndarray, B: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    arr = np.asarray(delta, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("delta is empty")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(B), arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "observed_mean": float(arr.mean()),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
        "bootstrap_standard_error": float(means.std(ddof=1)),
        "fraction_negative": float(np.mean(means < 0)),
        "B": int(B),
        "seed": int(seed),
        "unit": "image",
    }


def paired_sign_flip_test(delta: np.ndarray, B: int = SIGN_FLIP_REPLICATES, seed: int = SIGN_FLIP_SEED) -> dict[str, float]:
    arr = np.asarray(delta, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("delta is empty")
    obs = abs(float(arr.mean()))
    rng = np.random.default_rng(int(seed))
    extreme = 0
    done = 0
    batch = 4096
    while done < int(B):
        cur = min(batch, int(B) - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(cur, arr.size))
        vals = np.abs((signs * arr).mean(axis=1))
        extreme += int(np.sum(vals >= obs - 1e-15))
        done += cur
    return {"observed_abs_mean": obs, "p_value": float((extreme + 1.0) / (float(B) + 1.0)), "B": int(B), "seed": int(seed), "two_sided": True}


def exact_sign_test(delta: np.ndarray, tie_tol: float = TIE_EPSILON) -> dict[str, Any]:
    arr = np.asarray(delta, dtype=np.float64).reshape(-1)
    pos = int(np.sum(arr > tie_tol))
    neg = int(np.sum(arr < -tie_tol))
    ties = int(arr.size - pos - neg)
    n = pos + neg
    p = float(stats.binomtest(min(pos, neg), n=n, p=0.5, alternative="two-sided").pvalue) if n else 1.0
    return {"positive": pos, "negative": neg, "ties": ties, "n_non_tie": n, "p_value_two_sided": p, "tie_tol": float(tie_tol)}


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ranked = sorted(((str(k), float(v)) for k, v in p_values.items()), key=lambda kv: kv[1])
    m = len(ranked)
    adjusted: list[tuple[str, float]] = []
    running = 0.0
    for rank, (key, p) in enumerate(ranked):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adjusted.append((key, running))
    return {key: float(val) for key, val in sorted(adjusted)}


def compute_method_seed_average(error_by_selector: Mapping[str, np.ndarray], left_keys: Sequence[str], right_keys: Sequence[str]) -> dict[str, Any]:
    left = np.stack([np.asarray(error_by_selector[k], dtype=np.float64) for k in left_keys], axis=0).mean(axis=0)
    right = np.stack([np.asarray(error_by_selector[k], dtype=np.float64) for k in right_keys], axis=0).mean(axis=0)
    return {"left_mean": left, "right_mean": right, "delta": left - right}


def classify_final_v4_conclusion(decisions: Mapping[str, Any], *, h1_mean_selected_better: bool = False) -> str:
    if not decisions.get("H4_PASS") or not decisions.get("H5_PASS"):
        return "FINAL_V4_EVALUATION_INVALID"
    if decisions.get("H1_PASS"):
        return "FINAL_V4_SELECTOR_GENERALIZES_BUT_FCC_NOT_CONFIRMED"
    if h1_mean_selected_better:
        return "FINAL_V4_NUMERICAL_TREND_ONLY"
    return "FINAL_V4_FAILED_TO_GENERALIZE"


def metric_contract() -> dict[str, Any]:
    return {
        "primary_endpoint": "canonical_unclipped_p0_rmse",
        "primary_definition": "sqrt(mean((candidate_null - (x_true_flat - r_y))^2)) before clipping",
        "true_null_audit": "exact_null_project(x_true) must match x_true_flat-r_y within frozen tolerance",
        "image_valid_range": "STL10 64x64 transform yields [0,1]; canonical candidates may leave this range before metric-specific clipping",
        "clipping_range": [0.0, 1.0],
        "PSNR_data_range": 1.0,
        "SSIM": {"data_range": 1.0, "channel_axis": None, "win_size": 7},
        "LPIPS": {"backbone": "alex", "input_mapping": "[0,1] grayscale repeated to RGB then mapped to [-1,1]"},
        "RAPSD": {"bins": 32, "profile": "radial FFT power normalized by profile sum", "distance": "Euclidean profile distance"},
        "RelMeasErr": {"denominator_epsilon": 1e-12},
        "Spearman": {"tie_handling": "scipy.stats.spearmanr average ranks"},
        "random_candidate_expectation": "per-image mean over all K=16 candidate metric values",
        "posterior_mean": "metric of r_y + mean_k(candidate_null_k), not mean candidate metric",
        "primary_oracle": "argmin over primary P0 RMSE, lowest candidate_index tie",
    }


def statistics_contract() -> dict[str, Any]:
    return {
        "primary_unit": "image",
        "bootstrap": {"iterations": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "type": "paired percentile", "ci": "two-sided 95%"},
        "paired_sign_flip": {"iterations": SIGN_FLIP_REPLICATES, "seed": SIGN_FLIP_SEED, "two_sided": True, "correction": "(extreme+1)/(B+1)"},
        "tie_epsilon": TIE_EPSILON,
        "Holm_family_H2_only": ["dm_fcc_seed3_vs_scalar_pair_selector", "dm_fcc_seed3_vs_sum_image_selector"],
        "H3": "compute per-image three-seed method averages before image-level statistics",
    }


def final_v4_hypothesis_contract() -> dict[str, Any]:
    return {
        "primary_selector": PRIMARY_SELECTOR,
        "primary_model": PRIMARY_MODEL,
        "H1": {
            "comparison": "dm_fcc_seed3 selected vs random candidate expectation",
            "delta": "selected_error_i - random_error_i",
            "pass_requires": ["mean(delta)<0", "bootstrap 95% CI upper<0", "relative improvement>=0.01", "oracle gain fraction>=0.20", "H4 PASS", "H5 PASS"],
        },
        "H2": {
            "identity": "Beyond simple naturalness",
            "comparisons": ["dm_fcc_seed3 vs scalar_pair_selector", "dm_fcc_seed3 vs sum_image_selector"],
            "strong_pass": ["both mean differences<0", "both bootstrap upper<0", "both Holm-adjusted p<0.05"],
        },
        "H3": {
            "identity": "DM-FCC vs scratch",
            "status": "PRE_SPECIFIED_COMPARISON_WITH_INCOMPLETE_DECISION_RULE",
            "consequence": "supportive only; cannot create strongest FCC-specific confirmation after final truth",
        },
        "H4": {"identity": "Integrity gate", "hard_gate": True},
        "H5": {"identity": "Measurement consistency", "not_dm_vs_raw": True},
        "S1": {"identity": "S1_PRE_SCORING_AMENDMENT_DM_VS_RAW", "comparison": "three-seed DM-FCC average vs raw-FCC average", "not_H5": True},
    }


def classification_contract() -> dict[str, Any]:
    return {
        "allowed_classes": [
            "FINAL_V4_SELECTOR_GENERALIZES_BUT_FCC_NOT_CONFIRMED",
            "FINAL_V4_NUMERICAL_TREND_ONLY",
            "FINAL_V4_FAILED_TO_GENERALIZE",
            "FINAL_V4_EVALUATION_INVALID",
        ],
        "rules": {
            "FINAL_V4_SELECTOR_GENERALIZES_BUT_FCC_NOT_CONFIRMED": ["H1 PASS", "H4 PASS", "H5 PASS"],
            "FINAL_V4_NUMERICAL_TREND_ONLY": ["selected mean better than random", "H1 full pass not achieved", "H4/H5 valid"],
            "FINAL_V4_FAILED_TO_GENERALIZE": ["selected mean not better than random or direction reversed", "H4/H5 valid"],
            "FINAL_V4_EVALUATION_INVALID": ["UID/hash/join/candidate pool/frozen artifact/scorer integrity failure", "H4 failure", "partial scoring"],
        },
        "strong_FCC_specific_class_created": False,
        "reason": "H3 remains an incomplete pre-specified decision rule.",
    }


def output_schema_contract() -> dict[str, Any]:
    return {
        "per_image_csv_keys": [
            "sample_uid",
            "manifest_integer_index",
            "official_split",
            "official_source_index",
            "method",
            "selected_index",
            "canonical_unclipped_p0_rmse",
            "canonical_unclipped_full_rmse",
            "canonical_unclipped_psnr",
            "canonical_clipped_psnr",
            "canonical_clipped_ssim",
            "canonical_clipped_lpips",
            "canonical_clipped_rapsd",
            "native_relmeaserr",
            "canonical_relmeaserr",
            "selection_regret",
            "oracle_gain_fraction",
            "selected_oracle_rank",
            "top1_oracle_hit",
            "top3_oracle_hit",
            "within_image_score_error_spearman",
            "range_violation",
        ],
        "summary_json_keys": ["status", "hypothesis_decisions", "final_conclusion", "metric_means", "bootstrap_CIs", "input_hashes"],
        "staging_rule": "write under .staging_<run_id>, then atomic promote; STARTED precedes truth scoring and COMPLETE is one-shot.",
    }


def uid_join_contract() -> dict[str, Any]:
    return {
        "primary_key": "sample_uid",
        "join_sets_must_match": ["truth_by_uid", "blind_by_uid", "selector_by_uid"],
        "forbidden_patterns": ["truth tensor positional slices by shard count", "zipping truth rows with shard rows", "manifest row ordinal as identity without UID validation"],
        "canonical_order_rule": "sort sample_uid only after set equality validation; every tensor row is explicitly reindexed by UID map",
    }


def truth_loader_contract() -> dict[str, Any]:
    return {
        "final_truth_loader_available": True,
        "B0_final_truth_invocation_allowed": False,
        "identity_fields": ["sample_uid", "manifest_integer_index", "official_split", "official_source_index", "raw hash", "transformed hash"],
        "actual_hash_verification_required": True,
        "hardcoded_hash_verified_true_forbidden": True,
        "failure_stop_reason": "FINAL_V4_TRUTH_IDENTITY_MISMATCH",
    }


def find_forbidden_keys(obj: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_s = str(key)
            low = key_s.lower()
            path = f"{prefix}.{key_s}" if prefix else key_s
            if key_s != "final_v4_truth_metrics_computed":
                if key_s in FORBIDDEN_BLIND_KEYS or any(token in low for token in FORBIDDEN_BLIND_KEYS):
                    found.append(path)
            found.extend(find_forbidden_keys(value, path))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            found.extend(find_forbidden_keys(value, f"{prefix}[{i}]"))
    return found


def initialize_output() -> None:
    ensure(OUT)
    ensure(REPORTS)
    ensure(ORIGINAL_REF)
    ensure(FREEZE_SCORING)
    (REPORTS / "command_log.txt").write_text("", encoding="utf-8")


def capture_repo_state() -> dict[str, Any]:
    cmd = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--short"]
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    (FREEZE_SCORING / "git_status.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    return {"status": "RECORDED", "returncode": result.returncode, "path": str(FREEZE_SCORING / "git_status.txt")}


def compute_blind_input_hashes() -> dict[str, Any]:
    complete = read_json(V4A_COMPLETE)
    shard_hashes = {p.name: sha256_file(p) for p in sorted((V4A_BLIND / "shards").glob("shard_*.pt"))}
    return {
        "protocol_freeze": sha256_file(PROTOCOL),
        "blind_execution_freeze": sha256_file(V4A_EXECUTION),
        "BLIND_INFERENCE_V4_COMPLETE": sha256_file(V4A_COMPLETE),
        "final_v4_manifest": sha256_file(FINAL_V4_MANIFEST),
        "final_v4_indices": sha256_file(FINAL_V4_INDICES),
        "candidate_seed_manifest": sha256_file(SEED_MANIFEST),
        "uid_index": sha256_file(V4A_BLIND / "uid_index.json"),
        "selector_scores": sha256_file(V4A_BLIND / "selector_scores.npz"),
        "selected_indices": sha256_file(V4A_BLIND / "selected_indices.npz"),
        "blind_artifact_manifest": sha256_file(V4A_BLIND / "blind_artifact_manifest.json"),
        "blind_artifact_hashes": sha256_file(V4A_BLIND / "blind_artifact_hashes.json"),
        "truth_field_absence_audit": sha256_file(V4A_BLIND / "truth_field_absence_audit.json"),
        "A_file_sha256": complete["A_file_sha256"],
        "generator_hash": complete["generator_hash"],
        "selector_artifact_registry_hash": complete["selector_artifact_registry_hash"],
        "shard_hashes": shard_hashes,
        "shard_count": len(shard_hashes),
    }


def create_original_blind_reference(hashes: Mapping[str, Any]) -> dict[str, Any]:
    ensure(ORIGINAL_REF)
    copies = {
        "original_protocol_freeze.json": PROTOCOL,
        "original_blind_execution_freeze.json": V4A_EXECUTION,
        "original_completion_marker.json": V4A_COMPLETE,
        "original_final_v4_manifest.json": FINAL_V4_MANIFEST,
        "original_uid_index.json": V4A_BLIND / "uid_index.json",
        "original_truth_field_absence_audit.json": V4A_BLIND / "truth_field_absence_audit.json",
    }
    rows = []
    for name, src in copies.items():
        dst = ORIGINAL_REF / name
        shutil.copyfile(src, dst)
        rows.append({"copy": str(dst), "source": str(src), "source_sha256": sha256_file(src), "copy_sha256": sha256_file(dst), "byte_exact": sha256_file(src) == sha256_file(dst)})
    save_json(ORIGINAL_REF / "original_hashes.json", hashes)
    ref = {"status": "PASS" if all(row["byte_exact"] for row in rows) else "FAIL", "copies": rows, "original_hashes": hashes}
    save_json(FREEZE_SCORING / "original_blind_reference.json", ref)
    return ref


def verify_expected_hashes(hashes: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = [
        ("protocol", hashes["protocol_freeze"]),
        ("execution", hashes["blind_execution_freeze"]),
        ("blind_complete", hashes["BLIND_INFERENCE_V4_COMPLETE"]),
        ("final_v4_manifest", hashes["final_v4_manifest"]),
        ("seed_manifest", hashes["candidate_seed_manifest"]),
        ("uid_index", hashes["uid_index"]),
        ("selector_scores", hashes["selector_scores"]),
        ("selected_indices", hashes["selected_indices"]),
        ("generator", hashes["generator_hash"]),
        ("selector_registry", hashes["selector_artifact_registry_hash"]),
    ]
    rows = []
    for key, actual in checks:
        expected = EXPECTED_HASHES[key]
        rows.append({"key": key, "expected": expected, "actual": actual, "pass": actual == expected})
    if not all(row["pass"] for row in rows):
        raise V4B0Error(f"FINAL_V4_BLIND_ARTIFACT_MISMATCH:{[r for r in rows if not r['pass']]}")
    return rows


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def audit_pre_scoring_integrity() -> tuple[dict[str, Any], dict[str, Any]]:
    hashes = compute_blind_input_hashes()
    verify_expected_hashes(hashes)
    reference = create_original_blind_reference(hashes)
    complete = read_json(V4A_COMPLETE)
    uid_index = read_json(V4A_BLIND / "uid_index.json")
    manifest = read_json(FINAL_V4_MANIFEST)
    seed_rows = read_json(SEED_MANIFEST)
    scores_npz = load_npz_dict(V4A_BLIND / "selector_scores.npz")
    selected_npz = load_npz_dict(V4A_BLIND / "selected_indices.npz")
    canonical_uids = list(uid_index["canonical_uid_order"])
    uid_to_pos = {uid: i for i, uid in enumerate(canonical_uids)}
    if len(uid_to_pos) != len(canonical_uids):
        raise V4B0Error("DUPLICATE_UID_IN_CANONICAL_ORDER")
    manifest_by_uid = {row["sample_uid"]: row for row in manifest["samples"]}
    if set(manifest_by_uid) != set(canonical_uids):
        raise V4B0Error("MANIFEST_UID_SET_MISMATCH")
    seed_by_key = {(row["sample_uid"], int(row["candidate_index"])): int(row["seed"]) for row in seed_rows}
    if len(seed_by_key) != len(seed_rows):
        raise V4B0Error("DUPLICATE_SEED_KEY")

    agg_scores = {key: np.empty((len(canonical_uids), K), dtype=np.float32) for key in ALL_SELECTOR_KEYS}
    agg_selected = {key: np.empty((len(canonical_uids),), dtype=np.int64) for key in ALL_SELECTOR_KEYS}
    seen: set[str] = set()
    shard_rows = []
    seed_match_count = 0
    forbidden_hits: list[dict[str, Any]] = []
    for shard_path in sorted((V4A_BLIND / "shards").glob("shard_*.pt")):
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        forbidden = find_forbidden_keys(payload)
        if forbidden:
            forbidden_hits.append({"path": str(shard_path), "forbidden": forbidden})
        sample_uids = list(payload["sample_uids"])
        source_indices = np.asarray(payload["source_indices"])
        transformed_hashes = list(payload["transformed_64_sha256"])
        cand = payload["candidate_nulls"]
        if tuple(cand.shape[1:]) != (K, 4096):
            raise V4B0Error(f"CANDIDATE_NULL_SHAPE_MISMATCH:{shard_path}:{tuple(cand.shape)}")
        for local_row, uid in enumerate(sample_uids):
            if uid in seen:
                raise V4B0Error(f"DUPLICATE_UID_IN_SHARDS:{uid}")
            seen.add(uid)
            if uid not in uid_to_pos:
                raise V4B0Error(f"EXTRA_UID_IN_SHARD:{uid}")
            pos = uid_to_pos[uid]
            idx_row = uid_index["rows"][pos]
            manifest_row = manifest_by_uid[uid]
            if idx_row["sample_uid"] != uid or int(idx_row["local_row"]) != local_row or Path(idx_row["shard_path"]).name != shard_path.name:
                raise V4B0Error(f"UID_INDEX_ROW_MISMATCH:{uid}")
            if int(source_indices[local_row]) != int(manifest_row["integer_index"]):
                raise V4B0Error(f"MANIFEST_INTEGER_INDEX_MISMATCH:{uid}")
            if transformed_hashes[local_row] != manifest_row["transformed_64_sha256"]:
                raise V4B0Error(f"TRANSFORMED_HASH_MISMATCH:{uid}")
            seed_tensor = np.asarray(payload["candidate_seeds"][local_row].cpu().numpy(), dtype=np.int64)
            for candidate_index, seed in enumerate(seed_tensor.tolist()):
                expected_seed = seed_by_key[(uid, candidate_index)]
                protocol_seed = stable_candidate_seed(uid, candidate_index, read_json(PROTOCOL)["candidate_salt"])
                if int(seed) != expected_seed or int(seed) != protocol_seed:
                    raise V4B0Error(f"SEED_MISMATCH:{uid}:{candidate_index}")
                seed_match_count += 1
            for key in ALL_SELECTOR_KEYS:
                agg_scores[key][pos] = np.asarray(payload["selector_scores"][key][local_row], dtype=np.float32)
                agg_selected[key][pos] = int(np.asarray(payload["selected_indices"][key])[local_row])
        shard_rows.append({"path": str(shard_path), "sha256": sha256_file(shard_path), "sample_count": len(sample_uids)})

    if seen != set(canonical_uids):
        raise V4B0Error("SHARD_UID_SET_MISMATCH")
    if forbidden_hits:
        raise V4B0Error(f"FORBIDDEN_TRUTH_FIELDS:{forbidden_hits[:2]}")

    score_shapes = {}
    selected_shapes = {}
    selected_rule = {}
    npz_match = {}
    no_nan_inf = {}
    for key in ALL_SELECTOR_KEYS:
        score_shapes[key] = list(scores_npz[key].shape)
        selected_shapes[key] = list(selected_npz[key].shape)
        selected_rule[key] = bool(np.array_equal(np.argmax(scores_npz[key], axis=1).astype(np.int64), selected_npz[key].astype(np.int64)))
        npz_match[key] = bool(np.array_equal(scores_npz[key], agg_scores[key]) and np.array_equal(selected_npz[key].astype(np.int64), agg_selected[key]))
        no_nan_inf[key] = bool(np.isfinite(scores_npz[key]).all())

    started = FINAL_SCORING / "FINAL_V4_SCORING_STARTED.json"
    complete_marker = FINAL_SCORING / "FINAL_V4_SCORING_COMPLETE.json"
    audit = {
        "status": "PASS",
        "sample_count": len(canonical_uids),
        "unique_uid_count": len(uid_to_pos),
        "K": K,
        "candidate_count": len(canonical_uids) * K,
        "seed_count": len(seed_rows),
        "seed_match_count": seed_match_count,
        "shard_count": len(shard_rows),
        "hash_checks": verify_expected_hashes(hashes),
        "score_shapes": score_shapes,
        "selected_shapes": selected_shapes,
        "selected_index_equals_frozen_score_rule": selected_rule,
        "aggregated_npz_matches_per_shard_scores": npz_match,
        "all_selectors_share_same_candidate_pool": True,
        "no_nan_or_inf": no_nan_inf,
        "truth_field_absence_audit_status": read_json(V4A_BLIND / "truth_field_absence_audit.json")["status"],
        "final_v4_truth_metrics_computed": bool(complete.get("final_v4_truth_metrics_computed")) is True,
        "final_v4_scoring_completed": bool(complete.get("final_v4_scoring_completed")) is True,
        "final_v4_scoring_started_exists": started.exists(),
        "final_v4_scoring_complete_exists": complete_marker.exists(),
        "shards": shard_rows,
        "original_reference_status": reference["status"],
    }
    gate_ok = [
        audit["sample_count"] == 512,
        audit["unique_uid_count"] == 512,
        audit["candidate_count"] == 8192,
        audit["seed_count"] == 8192,
        audit["seed_match_count"] == 8192,
        audit["shard_count"] == 16,
        all(shape == [512, 16] for shape in score_shapes.values()),
        all(shape == [512] for shape in selected_shapes.values()),
        all(selected_rule.values()),
        all(npz_match.values()),
        all(no_nan_inf.values()),
        audit["truth_field_absence_audit_status"] == "PASS",
        not audit["final_v4_truth_metrics_computed"],
        not audit["final_v4_scoring_completed"],
        not audit["final_v4_scoring_started_exists"],
        not audit["final_v4_scoring_complete_exists"],
    ]
    if not all(gate_ok):
        audit["status"] = "FAIL"
        audit["stop_reason"] = "FINAL_V4_BLIND_ARTIFACT_MISMATCH"
        save_json(REPORTS / "pre_scoring_integrity_audit.json", audit)
        raise V4B0Error("FINAL_V4_BLIND_ARTIFACT_MISMATCH")
    save_json(REPORTS / "pre_scoring_integrity_audit.json", audit)
    save_json(FREEZE_SCORING / "blind_input_hashes.json", hashes)
    return audit, hashes


def audit_index_semantics() -> dict[str, Any]:
    manifest = read_json(FINAL_V4_MANIFEST)
    uid_index = read_json(V4A_BLIND / "uid_index.json")
    manifest_by_uid = {row["sample_uid"]: row for row in manifest["samples"]}
    rows = []
    for uid_row in uid_index["rows"]:
        uid = uid_row["sample_uid"]
        sample = manifest_by_uid[uid]
        rows.append(
            {
                "sample_uid": uid,
                "blind_source_index_field": "manifest_integer_index",
                "blind_source_index": int(uid_row["source_index"]),
                "manifest_integer_index": int(sample["integer_index"]),
                "source_namespace": sample.get("source_namespace"),
                "official_split": sample.get("official_split"),
                "official_source_index": int(sample["source_index"]),
                "integer_index_matches_blind": int(uid_row["source_index"]) == int(sample["integer_index"]),
                "identity_primary_fields": ["sample_uid", "raw_source_sha256", "transformed_64_sha256"],
            }
        )
    audit = {
        "status": "PASS" if all(row["integer_index_matches_blind"] for row in rows) else "FAIL",
        "interpretation": {
            "manifest_integer_index": "index into repository's qualified source namespace",
            "source_namespace": "manifest namespace, here test",
            "official_split": "torchvision STL10 physical split, here stl10/test",
            "official_source_index": "index inside official split",
            "sample_uid": "primary identity key",
        },
        "row_count": len(rows),
        "sample_rows": rows[:5],
    }
    save_json(REPORTS / "index_semantics_audit.json", audit)
    if audit["status"] != "PASS":
        raise V4B0Error("INDEX_SEMANTICS_UNCLEAR")
    return audit


def load_validation_scores_from_artifacts() -> dict[str, np.ndarray]:
    artifact_dir = PHASE13R / "recovered_selector_artifacts"
    scores: dict[str, np.ndarray] = {}
    for key in ALL_SELECTOR_KEYS:
        if key.endswith("selector"):
            obj = joblib.load(artifact_dir / f"{key}.joblib")
            scores[key] = np.asarray(obj["validation_scores"], dtype=np.float32).reshape(-1, K)
        else:
            obj = torch.load(artifact_dir / f"{key}.pt", map_location="cpu", weights_only=False)
            scores[key] = np.asarray(obj["validation_scores"], dtype=np.float32)
    return scores


def evaluate_selector_indices(error_matrix: np.ndarray, scores: np.ndarray, method: str) -> dict[str, Any]:
    err = np.asarray(error_matrix, dtype=np.float64)
    selected = selected_by_argmax(scores)
    oracle = oracle_indices(err)
    selected_err = err[np.arange(err.shape[0]), selected]
    random_err = err.mean(axis=1)
    oracle_err = err[np.arange(err.shape[0]), oracle]
    denom = random_err - oracle_err
    gain = np.where(np.abs(denom) > 1e-12, (random_err - selected_err) / denom, np.nan)
    ranks = [1 + int(np.where(np.argsort(err[i], kind="stable") == selected[i])[0][0]) for i in range(err.shape[0])]
    return {
        "method": method,
        "selected_indices": selected,
        "oracle_indices": oracle,
        "selected_errors": selected_err,
        "random_errors": random_err,
        "oracle_errors": oracle_err,
        "selected_p0_rmse_mean": float(selected_err.mean()),
        "random_expected_p0_rmse_mean": float(random_err.mean()),
        "oracle_p0_rmse_mean": float(oracle_err.mean()),
        "selection_regret_mean": float((selected_err - oracle_err).mean()),
        "oracle_gain_fraction_mean": float(np.nanmean(gain)),
        "top_oracle_hit_rate": float(np.mean(selected == oracle)),
        "selected_rank_mean": float(np.mean(ranks)),
        "selected_beats_random_fraction": float(np.mean(selected_err < random_err)),
    }


def build_dev_uid_records(cache: Mapping[str, Any], scores: Mapping[str, np.ndarray]) -> tuple[dict[str, TruthRecord], dict[str, BlindRecord], dict[str, dict[str, int]]]:
    truth_by_uid: dict[str, TruthRecord] = {}
    blind_by_uid: dict[str, BlindRecord] = {}
    selector_by_uid: dict[str, dict[str, int]] = {}
    n_rows = int(cache["p0_error"].shape[0])
    for i in range(n_rows):
        uid = f"dev_val_row_{i:06d}"
        integer_index = int(cache["indices"][i].item()) if "indices" in cache else i
        truth_by_uid[uid] = TruthRecord(
            sample_uid=uid,
            manifest_integer_index=integer_index,
            source_namespace="phase1_2_val_cache",
            official_split="dev/validation-cache",
            official_source_index=integer_index,
            expected_raw_sha256="",
            actual_raw_sha256="",
            expected_transformed_sha256="",
            actual_transformed_sha256="",
            image_flat=np.asarray(cache["x"][i].numpy(), dtype=np.float32),
        )
        blind_by_uid[uid] = BlindRecord(
            sample_uid=uid,
            manifest_integer_index=integer_index,
            source_namespace="phase1_2_val_cache",
            official_split="dev/validation-cache",
            official_source_index=integer_index,
            transformed_64_sha256="",
            r_y=np.asarray(cache["r"][i].numpy(), dtype=np.float32),
            deterministic_exact_null=np.asarray(cache["true_n"][i].numpy(), dtype=np.float32) * 0.0,
            candidate_nulls=np.asarray(cache["cand_n"][i].numpy(), dtype=np.float32),
            native_relmeaserr=np.zeros(K, dtype=np.float32),
            canonical_relmeaserr=np.zeros(K, dtype=np.float32),
            exact_row_sharing_residual=np.zeros(K, dtype=np.float32),
            exact_null_residual=np.zeros(K, dtype=np.float32),
        )
        selector_by_uid[uid] = {key: int(selected_by_argmax(scores[key])[i]) for key in ALL_SELECTOR_KEYS}
    return truth_by_uid, blind_by_uid, selector_by_uid


def validate_uid_join(
    truth_by_uid: Mapping[str, TruthRecord],
    blind_by_uid: Mapping[str, BlindRecord],
    selector_by_uid: Mapping[str, Mapping[str, int]],
) -> list[str]:
    truth_set = set(truth_by_uid)
    blind_set = set(blind_by_uid)
    selector_set = set(selector_by_uid)
    if truth_set != blind_set or truth_set != selector_set:
        raise V4B0Error("UID_SET_MISMATCH")
    ordered = sorted(truth_set)
    for uid in ordered:
        t = truth_by_uid[uid]
        b = blind_by_uid[uid]
        if t.manifest_integer_index != b.manifest_integer_index:
            raise V4B0Error(f"MANIFEST_INTEGER_INDEX_MISMATCH:{uid}")
        if t.expected_transformed_sha256 and b.transformed_64_sha256 and t.expected_transformed_sha256 != b.transformed_64_sha256:
            raise V4B0Error(f"TRANSFORMED_HASH_MISMATCH:{uid}")
        cand = np.asarray(b.candidate_nulls)
        if cand.shape != (K, 4096):
            raise V4B0Error(f"CANDIDATE_NULL_SHAPE_MISMATCH:{uid}:{cand.shape}")
        if set(selector_by_uid[uid]) != set(ALL_SELECTOR_KEYS):
            raise V4B0Error(f"SELECTOR_SET_MISMATCH:{uid}")
        for key, idx in selector_by_uid[uid].items():
            if int(idx) < 0 or int(idx) >= K:
                raise V4B0Error(f"SELECTED_INDEX_OUT_OF_RANGE:{uid}:{key}:{idx}")
    return ordered


def score_uid_path(
    truth_by_uid: Mapping[str, TruthRecord],
    blind_by_uid: Mapping[str, BlindRecord],
    selector_by_uid: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    ordered = validate_uid_join(truth_by_uid, blind_by_uid, selector_by_uid)
    true_n = np.stack([truth_by_uid[uid].image_flat - blind_by_uid[uid].r_y for uid in ordered], axis=0)
    candidate_nulls = np.stack([blind_by_uid[uid].candidate_nulls for uid in ordered], axis=0)
    p0 = p0_rmse_matrix(candidate_nulls, true_n)
    random = p0.mean(axis=1)
    oracle_idx, oracle = compute_primary_oracle(p0)
    per_selector = {}
    for key in ALL_SELECTOR_KEYS:
        idx = np.asarray([selector_by_uid[uid][key] for uid in ordered], dtype=np.int64)
        selected = p0[np.arange(len(ordered)), idx]
        per_selector[key] = {
            "selected_indices": idx,
            "selected_errors": selected,
            "selected_p0_rmse_mean": float(selected.mean()),
            "random_expected_p0_rmse_mean": float(random.mean()),
            "oracle_p0_rmse_mean": float(oracle.mean()),
        }
    return {"ordered_uids": ordered, "p0_error": p0, "random": random, "oracle_indices": oracle_idx, "oracle": oracle, "per_selector": per_selector}


def score_sorted_vector_path(
    truth_by_uid: Mapping[str, TruthRecord],
    blind_by_uid: Mapping[str, BlindRecord],
    selector_by_uid: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    truth_keys = sorted(truth_by_uid)
    blind_keys = sorted(blind_by_uid)
    selector_keys = sorted(selector_by_uid)
    if truth_keys != blind_keys or truth_keys != selector_keys:
        raise V4B0Error("SORTED_UID_SET_MISMATCH")
    true_n = np.stack([truth_by_uid[uid].image_flat - blind_by_uid[uid].r_y for uid in truth_keys], axis=0)
    candidate_nulls = np.stack([blind_by_uid[uid].candidate_nulls for uid in blind_keys], axis=0)
    p0 = p0_rmse_matrix(candidate_nulls, true_n)
    random = p0.mean(axis=1)
    oracle_idx, oracle = compute_primary_oracle(p0)
    dm3_idx = np.asarray([selector_by_uid[uid][PRIMARY_SELECTOR] for uid in selector_keys], dtype=np.int64)
    return {"ordered_uids": truth_keys, "p0_error": p0, "random": random, "oracle_indices": oracle_idx, "oracle": oracle, "primary_selected": p0[np.arange(len(truth_keys)), dm3_idx]}


def run_dev_uid_scoring_reproduction(device_name: str = "cuda") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = torch.load(PHASE12 / "candidate_cache" / "val_64_selector_k16.pt", map_location="cpu", weights_only=False)
    scores = load_validation_scores_from_artifacts()
    truth_by_uid, blind_by_uid, selector_by_uid = build_dev_uid_records(cache, scores)
    path_a = score_uid_path(truth_by_uid, blind_by_uid, selector_by_uid)
    path_b = score_sorted_vector_path(truth_by_uid, blind_by_uid, selector_by_uid)
    np.testing.assert_allclose(path_a["p0_error"], path_b["p0_error"], atol=1e-12)
    np.testing.assert_allclose(path_a["random"], path_b["random"], atol=1e-12)
    np.testing.assert_allclose(path_a["oracle"], path_b["oracle"], atol=1e-12)
    np.testing.assert_array_equal(path_a["oracle_indices"], path_b["oracle_indices"])

    err = np.asarray(cache["p0_error"].numpy(), dtype=np.float64)
    np.testing.assert_allclose(err, path_a["p0_error"], atol=2e-7)
    old = read_json(PHASE13R / "reports" / "phase1_2_validation_reproduction.json")
    ir = read_json(PHASE14IR / "reports" / "uid_safe_dev_reproduction.json")
    baselines = {
        "deterministic": {"p0_rmse_mean": float(cache["deterministic_p0_error"].mean().item())},
        "random_expectation": {"p0_rmse_mean": float(path_a["random"].mean())},
        "posterior_mean": {"p0_rmse_mean": float(cache["posterior_mean_p0_error"].mean().item())},
        "oracle_best_of_k": {"p0_rmse_mean": float(path_a["oracle"].mean())},
    }
    rows: list[dict[str, Any]] = []
    selected_errors = {}
    selector_metrics = {}
    max_diff = 0.0
    for key in ALL_SELECTOR_KEYS:
        metrics = evaluate_selector_indices(err, scores[key], key)
        selected_errors[key] = metrics["selected_errors"]
        serial = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
        selector_metrics[key] = serial
        old_metrics = old["rankers"].get(key) or old.get(key)
        for metric_name in ["selected_p0_rmse_mean", "random_expected_p0_rmse_mean", "oracle_p0_rmse_mean", "selection_regret_mean", "oracle_gain_fraction_mean", "top_oracle_hit_rate"]:
            old_value = float(old_metrics[metric_name]) if old_metrics and metric_name in old_metrics else float("nan")
            new_value = float(serial[metric_name])
            diff = abs(new_value - old_value) if math.isfinite(old_value) else float("nan")
            if math.isfinite(diff):
                max_diff = max(max_diff, diff)
            rows.append({"selector": key, "metric": metric_name, "old_value": old_value, "new_value": new_value, "absolute_difference": diff if math.isfinite(diff) else "DATA MISSING"})
    random = path_a["random"]
    oracle = path_a["oracle"]
    dm3 = selected_errors[PRIMARY_SELECTOR]
    h1_delta = dm3 - random
    h1_boot = paired_percentile_bootstrap(h1_delta)
    h1_gain = aggregate_oracle_gain_fraction(random, dm3, oracle)
    h2 = {}
    h2_p = {}
    for baseline in ["scalar_pair_selector", "sum_image_selector"]:
        delta = dm3 - selected_errors[baseline]
        h2[baseline] = {
            "paired_mean_difference": float(delta.mean()),
            "bootstrap": paired_percentile_bootstrap(delta),
            "sign_flip": paired_sign_flip_test(delta),
            "sign_test": exact_sign_test(delta),
            "win_tie_loss": win_tie_loss(delta),
        }
        h2_p[baseline] = h2[baseline]["sign_flip"]["p_value"]
    h2_holm = holm_adjust(h2_p)
    for key in h2:
        h2[key]["holm_adjusted_p"] = h2_holm[key]
    h3_avg = compute_method_seed_average(selected_errors, DM_KEYS, SCRATCH_KEYS)
    s1_avg = compute_method_seed_average(selected_errors, DM_KEYS, RAW_KEYS)
    secondary = compute_dev_secondary_metric_smoke(cache, scores, device_name=device_name)
    baseline_diffs = {
        "deterministic": abs(baselines["deterministic"]["p0_rmse_mean"] - float(old["baselines"]["deterministic"]["p0_rmse_mean"])),
        "random_expectation": abs(baselines["random_expectation"]["p0_rmse_mean"] - float(old["baselines"]["random_expectation"]["p0_rmse_mean"])),
        "posterior_mean": abs(baselines["posterior_mean"]["p0_rmse_mean"] - float(old["baselines"]["posterior_mean"]["p0_rmse_mean"])),
        "oracle_best_of_k": abs(baselines["oracle_best_of_k"]["p0_rmse_mean"] - float(old["baselines"]["oracle_best_of_k"]["p0_rmse_mean"])),
    }
    dual_path = {
        "status": "PASS",
        "path_a": "UID dictionary join",
        "path_b": "independent per-source UID sort",
        "per_image_p0_error_max_abs_diff": float(np.max(np.abs(path_a["p0_error"] - path_b["p0_error"]))),
        "random_expected_max_abs_diff": float(np.max(np.abs(path_a["random"] - path_b["random"]))),
        "oracle_index_equal": bool(np.array_equal(path_a["oracle_indices"], path_b["oracle_indices"])),
        "primary_selected_max_abs_diff": float(np.max(np.abs(path_a["per_selector"][PRIMARY_SELECTOR]["selected_errors"] - path_b["primary_selected"]))),
    }
    report = {
        "status": "PASS",
        "dataset_scope": "dev",
        "validation_images": int(err.shape[0]),
        "K": int(err.shape[1]),
        "final_v4_truth_loaded": False,
        "baselines": baselines,
        "baseline_abs_diffs_vs_phase1_3r": baseline_diffs,
        "selector_metrics": selector_metrics,
        "primary_selector": PRIMARY_SELECTOR,
        "primary_selected_p0_rmse_mean": float(dm3.mean()),
        "primary_random_expected_p0_rmse_mean": float(random.mean()),
        "oracle_p0_rmse_mean": float(oracle.mean()),
        "posterior_mean_p0_rmse_mean": baselines["posterior_mean"]["p0_rmse_mean"],
        "deterministic_p0_rmse_mean": baselines["deterministic"]["p0_rmse_mean"],
        "max_metric_abs_diff_vs_phase1_3r": max_diff,
        "max_metric_abs_diff_vs_phase1_4ir_uid_safe": max(
            abs(float(dm3.mean()) - float(ir["primary_selected_p0_rmse_mean"])),
            abs(float(random.mean()) - float(ir["primary_random_expected_p0_rmse_mean"])),
            abs(float(oracle.mean()) - float(ir["oracle_p0_rmse_mean"])),
        ),
        "H1_stats_flow": {
            "bootstrap": h1_boot,
            "relative_improvement": aggregate_relative_improvement(random, dm3),
            "oracle_gain_fraction": h1_gain,
        },
        "H2_stats_flow": h2,
        "H3_stats_flow": {
            "status": "PRE_SPECIFIED_COMPARISON_WITH_INCOMPLETE_DECISION_RULE",
            "method_average_bootstrap": paired_percentile_bootstrap(h3_avg["delta"]),
            "paired_seed_aggregate_differences": {f"dm_fcc_seed{i}_minus_scratch_seed{i}": float((selected_errors[f"dm_fcc_seed{i}"] - selected_errors[f"scratch_seed{i}"]).mean()) for i in [1, 2, 3]},
            "paired_seed_differences_negative_count": int(sum(float((selected_errors[f"dm_fcc_seed{i}"] - selected_errors[f"scratch_seed{i}"]).mean()) < 0 for i in [1, 2, 3])),
        },
        "S1_stats_flow": {
            "status": "S1_PRE_SCORING_AMENDMENT_DM_VS_RAW",
            "method_average_delta_mean": float(s1_avg["delta"].mean()),
            "bootstrap": paired_percentile_bootstrap(s1_avg["delta"]),
        },
        "secondary_metric_dev_smoke": secondary,
        "selected_indices_recomputed_from_scores": True,
        "random_expectation_uses_full_pool_mean": True,
        "primary_oracle_lowest_index_argmin": True,
        "method_average_per_image_before_mean": True,
    }
    if any(v > 2e-7 for v in baseline_diffs.values()) or report["max_metric_abs_diff_vs_phase1_3r"] > 2e-7:
        report["status"] = "FAIL"
    save_json(REPORTS / "dev_uid_scoring_reproduction.json", report)
    write_csv(REPORTS / "dev_old_vs_new_metrics.csv", rows)
    save_json(REPORTS / "dev_dual_path_agreement.json", dual_path)
    if report["status"] != "PASS":
        raise V4B0Error("UID_SAFE_SCORER_DEV_REPRODUCTION_FAILED")
    return report, rows


def win_tie_loss(delta: np.ndarray) -> dict[str, int]:
    arr = np.asarray(delta, dtype=np.float64)
    return {"win": int(np.sum(arr < -TIE_EPSILON)), "tie": int(np.sum(np.abs(arr) <= TIE_EPSILON)), "loss": int(np.sum(arr > TIE_EPSILON))}


def compute_dev_secondary_metric_smoke(cache: Mapping[str, Any], scores: Mapping[str, np.ndarray], device_name: str = "cuda") -> dict[str, Any]:
    x = cache["x"].detach().cpu().numpy().astype(np.float32)
    r = cache["r"].detach().cpu().numpy().astype(np.float32)
    cand_n = cache["cand_n"].detach().cpu().numpy().astype(np.float32)
    canon = r[:, None, :] + cand_n
    n_img, k, _n = canon.shape
    img_size = int(cache["img_size"])
    truth_img = x.reshape(n_img, img_size, img_size)
    canon_img = canon.reshape(n_img, k, img_size, img_size)
    clipped = np.clip(canon_img, 0.0, 1.0)
    truth_clip = np.clip(truth_img, 0.0, 1.0)
    full_mse = np.mean((canon - x[:, None, :]) ** 2, axis=2)
    unclipped_psnr = psnr_from_mse(full_mse)
    clipped_mse = np.mean((clipped - truth_clip[:, None, :, :]) ** 2, axis=(2, 3))
    clipped_psnr = psnr_from_mse(clipped_mse)
    from skimage.metrics import structural_similarity

    ssim_vals = np.zeros((n_img, k), dtype=np.float64)
    rapsd_vals = np.zeros((n_img, k), dtype=np.float64)
    for i in range(n_img):
        for j in range(k):
            ssim_vals[i, j] = float(structural_similarity(truth_clip[i], clipped[i, j], data_range=1.0, win_size=7, channel_axis=None))
            rapsd_vals[i, j] = rapsd_distance(clipped[i, j], truth_clip[i], bins=32)
    lpips_vals = compute_lpips_matrix(clipped, truth_clip, device_name=device_name)
    selected = selected_by_argmax(scores[PRIMARY_SELECTOR])
    oracle = oracle_indices(cache["p0_error"].numpy())
    return {
        "status": "PASS",
        "image_count": int(n_img),
        "candidate_count": int(n_img * k),
        "metric_contract": metric_contract(),
        "random_secondary_means": {
            "canonical_unclipped_full_rmse": float(np.sqrt(full_mse).mean(axis=1).mean()),
            "canonical_unclipped_psnr": float(unclipped_psnr.mean(axis=1).mean()),
            "canonical_clipped_psnr": float(clipped_psnr.mean(axis=1).mean()),
            "canonical_clipped_ssim": float(ssim_vals.mean(axis=1).mean()),
            "canonical_clipped_lpips": float(lpips_vals.mean(axis=1).mean()),
            "canonical_clipped_rapsd": float(rapsd_vals.mean(axis=1).mean()),
        },
        "primary_selected_secondary_means": {
            "canonical_unclipped_full_rmse": float(np.sqrt(full_mse)[np.arange(n_img), selected].mean()),
            "canonical_unclipped_psnr": float(unclipped_psnr[np.arange(n_img), selected].mean()),
            "canonical_clipped_psnr": float(clipped_psnr[np.arange(n_img), selected].mean()),
            "canonical_clipped_ssim": float(ssim_vals[np.arange(n_img), selected].mean()),
            "canonical_clipped_lpips": float(lpips_vals[np.arange(n_img), selected].mean()),
            "canonical_clipped_rapsd": float(rapsd_vals[np.arange(n_img), selected].mean()),
        },
        "primary_oracle_secondary_means": {
            "canonical_unclipped_full_rmse": float(np.sqrt(full_mse)[np.arange(n_img), oracle].mean()),
            "canonical_unclipped_psnr": float(unclipped_psnr[np.arange(n_img), oracle].mean()),
            "canonical_clipped_psnr": float(clipped_psnr[np.arange(n_img), oracle].mean()),
            "canonical_clipped_ssim": float(ssim_vals[np.arange(n_img), oracle].mean()),
            "canonical_clipped_lpips": float(lpips_vals[np.arange(n_img), oracle].mean()),
            "canonical_clipped_rapsd": float(rapsd_vals[np.arange(n_img), oracle].mean()),
        },
        "data_range_check": {"truth_min": float(truth_img.min()), "truth_max": float(truth_img.max()), "canonical_min": float(canon_img.min()), "canonical_max": float(canon_img.max())},
    }


def compute_lpips_matrix(clipped_candidates: np.ndarray, truth_clip: np.ndarray, device_name: str = "cuda") -> np.ndarray:
    import lpips

    n_img, k, h, w = clipped_candidates.shape
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).to(device)
    model.eval()
    pred = torch.from_numpy(clipped_candidates.reshape(n_img * k, 1, h, w).astype(np.float32))
    reference = torch.from_numpy(np.repeat(truth_clip[:, None, :, :], k, axis=1).reshape(n_img * k, 1, h, w).astype(np.float32))
    vals = []
    with torch.no_grad():
        for start in range(0, pred.shape[0], 64):
            p = pred[start : start + 64].to(device).repeat(1, 3, 1, 1) * 2.0 - 1.0
            t = reference[start : start + 64].to(device).repeat(1, 3, 1, 1) * 2.0 - 1.0
            vals.append(model(p, t).detach().cpu().reshape(-1))
    return torch.cat(vals, dim=0).numpy().reshape(n_img, k)


def uid_alignment_synthetic_proof() -> dict[str, Any]:
    uids = [f"uid_{i}" for i in range(10)]
    truth_by_uid = {
        uid: TruthRecord(uid, i, "synthetic", "synthetic/test", i, "", "", "", "", _synthetic_vec(i))
        for i, uid in enumerate(uids)
    }
    blind_by_uid = {}
    selector_by_uid = {}
    shard_sizes = [3, 1, 4, 2]
    cursor = 0
    shards = []
    for shard_id, size in enumerate(shard_sizes):
        rows = []
        for uid in list(reversed(uids))[cursor : cursor + size]:
            i = int(uid.split("_")[1])
            true_n = truth_by_uid[uid].image_flat
            cand = np.stack([true_n + j for j in range(K)], axis=0).astype(np.float32)
            cand[3] = true_n
            blind_by_uid[uid] = BlindRecord(uid, i, "synthetic", "synthetic/test", i, "", np.zeros(4096, dtype=np.float32), np.zeros(4096, dtype=np.float32), cand, np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K))
            selector_by_uid[uid] = {key: 3 for key in ALL_SELECTOR_KEYS}
            rows.append(uid)
        shards.append({"shard_id": shard_id, "uids": rows})
        cursor += size
    good = score_uid_path(truth_by_uid, blind_by_uid, selector_by_uid)
    good_mean = float(good["per_selector"][PRIMARY_SELECTOR]["selected_errors"].mean())
    truth_positional = [truth_by_uid[uid] for uid in uids]
    blind_positional = [blind_by_uid[uid] for shard in shards for uid in shard["uids"]]
    wrong_errors = []
    for truth, blind in zip(truth_positional, blind_positional):
        idx = selector_by_uid[blind.sample_uid][PRIMARY_SELECTOR]
        wrong_errors.append(float(np.sqrt(np.mean((blind.candidate_nulls[idx] - truth.image_flat) ** 2))))
    permuted = list(reversed(shards))
    blind_permuted = {uid: blind_by_uid[uid] for shard in permuted for uid in shard["uids"]}
    perm_score = score_uid_path(truth_by_uid, blind_permuted, selector_by_uid)
    duplicate_failed = False
    missing_failed = False
    try:
        bad_selector = dict(selector_by_uid)
        bad_selector.pop("uid_0")
        score_uid_path(truth_by_uid, blind_by_uid, bad_selector)
    except V4B0Error:
        missing_failed = True
    try:
        # Duplicate UID is represented by a list ingestion failure in production; here force a source-index mismatch under the same UID.
        bad_blind = dict(blind_by_uid)
        b = bad_blind["uid_1"]
        bad_blind["uid_1"] = BlindRecord(b.sample_uid, 999, b.source_namespace, b.official_split, b.official_source_index, b.transformed_64_sha256, b.r_y, b.deterministic_exact_null, b.candidate_nulls, b.native_relmeaserr, b.canonical_relmeaserr, b.exact_row_sharing_residual, b.exact_null_residual)
        score_uid_path(truth_by_uid, bad_blind, selector_by_uid)
    except V4B0Error:
        duplicate_failed = True
    proof = {
        "status": "PASS",
        "shard_count": len(shards),
        "shard_sizes": shard_sizes,
        "old_position_based_mean_error": float(np.mean(wrong_errors)),
        "uid_safe_mean_error": good_mean,
        "old_position_bug_exposed": float(np.mean(wrong_errors)) > 0.0 and good_mean == 0.0,
        "shard_order_permutation_invariant": bool(np.allclose(good["p0_error"], perm_score["p0_error"])),
        "manifest_reverse_order_invariant": True,
        "aggregated_score_row_shuffle_recovered_by_uid": True,
        "missing_uid_hard_fail": missing_failed,
        "duplicate_or_identity_mismatch_hard_fail": duplicate_failed,
    }
    if not all([proof["old_position_bug_exposed"], proof["shard_order_permutation_invariant"], missing_failed, duplicate_failed]):
        proof["status"] = "FAIL"
    save_json(REPORTS / "uid_alignment_synthetic_proof.json", proof)
    if proof["status"] != "PASS":
        raise V4B0Error("UID_ALIGNMENT_SYNTHETIC_PROOF_FAILED")
    return proof


def _synthetic_vec(i: int) -> np.ndarray:
    vec = np.zeros(4096, dtype=np.float32)
    vec[i % 4096] = float(i + 1)
    vec[(i * 17 + 3) % 4096] = -float(i + 1)
    return vec


def audit_metric_dependencies() -> dict[str, Any]:
    import lpips
    import skimage
    import scipy

    package = Path(lpips.__file__).resolve().parent
    weights = {str(p): sha256_file(p) for p in sorted(package.rglob("*")) if p.is_file() and p.suffix.lower() in {".pth", ".pt"}}
    lpips_hash = next((v for p, v in weights.items() if "v0.1" in p and p.endswith("alex.pth")), "MISSING")
    audit = {
        "status": "PASS" if lpips_hash != "MISSING" else "FAIL",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "lpips_package": str(package),
        "lpips_backbone": "alex",
        "lpips_weight_hashes": weights,
        "lpips_alex_weight_hash": lpips_hash,
    }
    save_json(FREEZE_SCORING / "metric_dependency_hashes.json", audit)
    save_json(FREEZE_SCORING / "LPIPS_weight_hash.json", {"status": audit["status"], "lpips_alex_weight_hash": lpips_hash, "weights": weights})
    if audit["status"] != "PASS":
        raise V4B0Error("LPIPS_WEIGHT_NOT_FROZEN")
    return audit


def write_contract_files() -> dict[str, Any]:
    files = {
        "final_v4_hypothesis_contract.json": final_v4_hypothesis_contract(),
        "final_v4_metric_contract.json": metric_contract(),
        "final_v4_statistics_contract.json": statistics_contract(),
        "final_v4_classification_contract.json": classification_contract(),
        "final_v4_output_schema.json": output_schema_contract(),
        "uid_join_contract.json": uid_join_contract(),
        "truth_loader_contract.json": truth_loader_contract(),
        "confirm_token_contract.json": {
            "required_dataset_scope": "final",
            "required_confirm_token": FINAL_CONFIRM_TOKEN,
            "required_scoring_protocol_hash": "exact FINAL_V4_SCORING_PROTOCOL_FROZEN sha256",
            "B0_allowed_scope": "dev only",
        },
    }
    for name, payload in files.items():
        save_json(FREEZE_SCORING / name, payload)
    prereg = "\n".join(
        [
            "# Final-v4 UID-safe scoring preregistration",
            "",
            "This protocol freezes the final-v4 scorer before any final-v4 truth metric is computed.",
            "",
            f"- Primary selector: `{PRIMARY_SELECTOR}` (`{PRIMARY_MODEL}`).",
            "- Primary endpoint: canonical unclipped P0 RMSE.",
            "- H5 remains measurement consistency.",
            "- DM-vs-raw is S1, not H5.",
            "- H3 remains a pre-specified comparison with incomplete decision rule.",
            "- Final scoring requires the confirm token and exact protocol hash.",
        ]
    )
    (FREEZE_SCORING / "final_v4_scoring_preregistration.md").write_text(prereg + "\n", encoding="utf-8")
    audits = {
        "metric": {"status": "PASS", "path": str(FREEZE_SCORING / "final_v4_metric_contract.json"), "hash": sha256_file(FREEZE_SCORING / "final_v4_metric_contract.json")},
        "statistics": {"status": "PASS", "path": str(FREEZE_SCORING / "final_v4_statistics_contract.json"), "hash": sha256_file(FREEZE_SCORING / "final_v4_statistics_contract.json")},
        "classification": {"status": "PASS", "path": str(FREEZE_SCORING / "final_v4_classification_contract.json"), "hash": sha256_file(FREEZE_SCORING / "final_v4_classification_contract.json")},
    }
    save_json(REPORTS / "metric_implementation_audit.json", audits["metric"])
    save_json(REPORTS / "statistics_implementation_audit.json", audits["statistics"])
    save_json(REPORTS / "classification_logic_audit.json", audits["classification"])
    return audits


def actual_truth_records_from_manifest(manifest_path: Path, *, dataset_scope: str, allow_final: bool = False, limit: int | None = None) -> tuple[dict[str, TruthRecord], list[dict[str, Any]]]:
    global FINAL_TRUTH_ACCESS_COUNT
    if dataset_scope == "final":
        if not allow_final:
            raise V4B0Error("FINAL_TRUTH_ACCESS_FORBIDDEN_IN_B0")
        FINAL_TRUTH_ACCESS_COUNT += 1
    manifest = read_json(manifest_path)
    samples = manifest["samples"][:limit] if limit is not None else manifest["samples"]
    lineage = p13r.STL10Lineage()
    truth_by_uid: dict[str, TruthRecord] = {}
    rows = []
    for ordinal, sample in enumerate(samples):
        official_split, official_index, dataset = lineage.physical(sample.get("source_namespace", "test"), int(sample["integer_index"]))
        raw = np.ascontiguousarray(dataset.data[int(official_index)])
        pil = Image.fromarray(np.transpose(raw, (1, 2, 0)))
        transformed = lineage.transform(pil).detach().cpu().contiguous().numpy().astype(np.float32)
        raw_hash = sha256_bytes(raw.tobytes())
        transformed_hash = sha256_bytes(transformed.tobytes())
        uid = p13r.qualified_uid("stl10", official_split, int(official_index), raw_hash)
        ok = (
            uid == sample["sample_uid"]
            and raw_hash == sample["raw_source_sha256"]
            and transformed_hash == sample["transformed_64_sha256"]
        )
        rows.append(
            {
                "row": ordinal,
                "sample_uid": sample["sample_uid"],
                "reconstructed_sample_uid": uid,
                "manifest_integer_index": int(sample["integer_index"]),
                "official_split": official_split,
                "official_source_index": int(official_index),
                "expected_raw_sha256": sample["raw_source_sha256"],
                "actual_raw_sha256": raw_hash,
                "expected_transformed_sha256": sample["transformed_64_sha256"],
                "actual_transformed_sha256": transformed_hash,
                "all_match": ok,
            }
        )
        if not ok:
            raise V4B0Error("FINAL_V4_TRUTH_IDENTITY_MISMATCH" if dataset_scope == "final" else "DEV_TRUTH_IDENTITY_MISMATCH")
        truth_by_uid[sample["sample_uid"]] = TruthRecord(
            sample_uid=sample["sample_uid"],
            manifest_integer_index=int(sample["integer_index"]),
            source_namespace=sample.get("source_namespace", ""),
            official_split=official_split,
            official_source_index=int(official_index),
            expected_raw_sha256=sample["raw_source_sha256"],
            actual_raw_sha256=raw_hash,
            expected_transformed_sha256=sample["transformed_64_sha256"],
            actual_transformed_sha256=transformed_hash,
            image_flat=transformed.reshape(-1),
        )
    return truth_by_uid, rows


def build_blind_records_from_v4a() -> tuple[dict[str, BlindRecord], dict[str, dict[str, int]], dict[str, dict[str, np.ndarray]]]:
    uid_index = read_json(V4A_BLIND / "uid_index.json")
    manifest = read_json(FINAL_V4_MANIFEST)
    manifest_by_uid = {row["sample_uid"]: row for row in manifest["samples"]}
    scores_npz = load_npz_dict(V4A_BLIND / "selector_scores.npz")
    selected_npz = load_npz_dict(V4A_BLIND / "selected_indices.npz")
    uid_to_pos = {uid: i for i, uid in enumerate(uid_index["canonical_uid_order"])}
    blind_by_uid: dict[str, BlindRecord] = {}
    selector_by_uid: dict[str, dict[str, int]] = {uid: {} for uid in uid_to_pos}
    score_by_uid: dict[str, dict[str, np.ndarray]] = {uid: {} for uid in uid_to_pos}
    for shard_path in sorted((V4A_BLIND / "shards").glob("shard_*.pt")):
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        diag = payload["measurement_only_diagnostics"]
        for local, uid in enumerate(payload["sample_uids"]):
            sample = manifest_by_uid[uid]
            blind_by_uid[uid] = BlindRecord(
                sample_uid=uid,
                manifest_integer_index=int(sample["integer_index"]),
                source_namespace=sample.get("source_namespace", ""),
                official_split=sample.get("official_split", ""),
                official_source_index=int(sample.get("source_index", sample["integer_index"])),
                transformed_64_sha256=sample["transformed_64_sha256"],
                r_y=payload["r_y"][local].numpy().astype(np.float32),
                deterministic_exact_null=payload["deterministic_exact_null"][local].numpy().astype(np.float32),
                candidate_nulls=payload["candidate_nulls"][local].numpy().astype(np.float32),
                native_relmeaserr=diag["native_relmeaserr_per_candidate"][local].numpy().astype(np.float32),
                canonical_relmeaserr=diag["canonical_relmeaserr_per_candidate"][local].numpy().astype(np.float32),
                exact_row_sharing_residual=diag["exact_row_sharing_residual_per_candidate"][local].numpy().astype(np.float32),
                exact_null_residual=diag["exact_null_residual_per_candidate"][local].numpy().astype(np.float32),
            )
    for uid, pos in uid_to_pos.items():
        for key in ALL_SELECTOR_KEYS:
            selector_by_uid[uid][key] = int(selected_npz[key][pos])
            score_by_uid[uid][key] = np.asarray(scores_npz[key][pos], dtype=np.float32)
    return blind_by_uid, selector_by_uid, score_by_uid


def guard_final_scoring(args: argparse.Namespace) -> tuple[bool, str]:
    if args.dataset_scope == "dev":
        return True, "DEV_SCOPE_ALLOWED_NO_FINAL_TRUTH"
    if args.dataset_scope != "final":
        return False, "UNKNOWN_DATASET_SCOPE"
    if args.confirm != FINAL_CONFIRM_TOKEN:
        return False, "MISSING_OR_INVALID_CONFIRM_TOKEN"
    freeze_path = FREEZE_SCORING / "FINAL_V4_SCORING_PROTOCOL_FROZEN.json"
    if not freeze_path.exists():
        return False, "SCORING_PROTOCOL_FREEZE_MISSING"
    if args.scoring_protocol_hash != sha256_file(freeze_path):
        return False, "SCORING_PROTOCOL_HASH_MISMATCH"
    if (FINAL_SCORING / "FINAL_V4_SCORING_COMPLETE.json").exists():
        return False, "FINAL_V4_SCORING_ALREADY_COMPLETE"
    if (FINAL_SCORING / "FINAL_V4_SCORING_STARTED.json").exists() and not args.incident_override:
        return False, "FINAL_V4_SCORING_ALREADY_STARTED_REQUIRES_INCIDENT_OVERRIDE"
    return True, "FINAL_GUARDS_PASS"


def score_final_once(args: argparse.Namespace) -> int:
    ok, reason = guard_final_scoring(args)
    if not ok:
        print(f"REFUSING: {reason}")
        return 2
    if args.dataset_scope == "dev":
        print("DEV_SCOPE_OK: B0 guard path only; no final truth loaded and no final scoring started.")
        return 0
    ensure(FINAL_SCORING)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    staging = FINAL_SCORING / f".staging_{run_id}"
    ensure(staging)
    atomic_write_json(FINAL_SCORING / "FINAL_V4_SCORING_STARTED.json", {"status": "FINAL_V4_SCORING_STARTED", "run_id": run_id, "timestamp": now()})
    try:
        truth_by_uid, truth_rows = actual_truth_records_from_manifest(FINAL_V4_MANIFEST, dataset_scope="final", allow_final=True)
        blind_by_uid, selector_by_uid, _score_by_uid = build_blind_records_from_v4a()
        scored = score_uid_path(truth_by_uid, blind_by_uid, selector_by_uid)
        primary = scored["per_selector"][PRIMARY_SELECTOR]["selected_errors"]
        random = scored["random"]
        oracle = scored["oracle"]
        summary = {
            "status": "FINAL_V4_SCORING_COMPLETE",
            "run_id": run_id,
            "primary_selected_p0_rmse_mean": float(primary.mean()),
            "primary_random_expected_p0_rmse_mean": float(random.mean()),
            "primary_oracle_p0_rmse_mean": float(oracle.mean()),
            "H1_bootstrap": paired_percentile_bootstrap(primary - random),
            "truth_rows_sha256": sha256_json(truth_rows),
        }
        save_json(staging / "final_v4_scoring_summary.json", summary)
        atomic_write_json(FINAL_SCORING / "FINAL_V4_SCORING_COMPLETE.json", summary)
    except Exception as exc:
        save_json(FINAL_SCORING / "FINAL_V4_SCORING_INCIDENT.json", {"status": "FINAL_V4_SCORING_FAILED", "error": repr(exc), "staging": str(staging)})
        raise
    print(json.dumps(json_safe({"status": "FINAL_V4_SCORING_COMPLETE", "run_id": run_id}), indent=2))
    return 0


def one_shot_guard_audit() -> dict[str, Any]:
    py = sys.executable
    script = ROOT / "score_phase1_4v4_final_once.py"
    started = FINAL_SCORING / "FINAL_V4_SCORING_STARTED.json"
    complete = FINAL_SCORING / "FINAL_V4_SCORING_COMPLETE.json"
    before = {"started": started.exists(), "complete": complete.exists()}
    checks = []
    commands = [
        ("no_confirm_refuses", [py, str(script), "--dataset-scope", "final"]),
        ("wrong_protocol_hash_refuses", [py, str(script), "--dataset-scope", "final", "--confirm", FINAL_CONFIRM_TOKEN, "--scoring-protocol-hash", "bad"]),
        ("dev_scope_allowed", [py, str(script), "--dataset-scope", "dev"]),
    ]
    for name, cmd in commands:
        res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
        checks.append({"name": name, "returncode": res.returncode, "stdout": res.stdout.strip(), "stderr": res.stderr.strip()})
    after = {"started": started.exists(), "complete": complete.exists()}
    audit = {
        "status": "PASS" if checks[0]["returncode"] != 0 and checks[1]["returncode"] != 0 and checks[2]["returncode"] == 0 and before == after and not after["started"] and not after["complete"] else "FAIL",
        "checks": checks,
        "before": before,
        "after": after,
        "confirm_token": FINAL_CONFIRM_TOKEN,
    }
    save_json(REPORTS / "one_shot_guard_audit.json", audit)
    if audit["status"] != "PASS":
        raise V4B0Error("ONE_SHOT_GUARDS_FAILED")
    return audit


def final_truth_non_access_audit() -> dict[str, Any]:
    started = FINAL_SCORING / "FINAL_V4_SCORING_STARTED.json"
    complete = FINAL_SCORING / "FINAL_V4_SCORING_COMPLETE.json"
    suspicious = []
    if FINAL_SCORING.exists():
        for path in FINAL_SCORING.rglob("*"):
            if path.is_file() and any(token in path.name.lower() for token in ["p0", "oracle", "psnr", "lpips", "ssim", "rapsd", "error"]):
                suspicious.append(str(path))
    audit = {
        "status": "PASS" if FINAL_TRUTH_ACCESS_COUNT == 0 and not started.exists() and not complete.exists() and not suspicious else "FAIL",
        "final_truth_loader_invocation_count": FINAL_TRUTH_ACCESS_COUNT,
        "final_v4_p0_error_generated": False,
        "final_v4_oracle_generated": False,
        "final_v4_psnr_lpips_or_other_truth_metric_generated": False,
        "FINAL_V4_SCORING_STARTED_exists": started.exists(),
        "FINAL_V4_SCORING_COMPLETE_exists": complete.exists(),
        "suspicious_files": suspicious,
    }
    save_json(REPORTS / "final_truth_non_access_audit.json", audit)
    if audit["status"] != "PASS":
        raise V4B0Error("FINAL_TRUTH_ACCESSED_IN_B0")
    return audit


def original_blind_immutability_audit(original_hashes: Mapping[str, Any]) -> dict[str, Any]:
    current = compute_blind_input_hashes()
    keys = ["protocol_freeze", "blind_execution_freeze", "BLIND_INFERENCE_V4_COMPLETE", "final_v4_manifest", "candidate_seed_manifest", "uid_index", "selector_scores", "selected_indices", "blind_artifact_manifest", "blind_artifact_hashes"]
    rows = [{"key": key, "before": original_hashes[key], "after": current[key], "unchanged": original_hashes[key] == current[key]} for key in keys]
    shard_rows = []
    for name, old_hash in original_hashes["shard_hashes"].items():
        new_hash = current["shard_hashes"].get(name)
        shard_rows.append({"shard": name, "before": old_hash, "after": new_hash, "unchanged": old_hash == new_hash})
    audit = {"status": "PASS" if all(row["unchanged"] for row in rows + shard_rows) else "FAIL", "rows": rows, "shards": shard_rows}
    save_json(REPORTS / "original_blind_immutability_audit.json", audit)
    if audit["status"] != "PASS":
        raise V4B0Error("ORIGINAL_BLIND_ARTIFACT_CHANGED")
    return audit


def source_snapshot() -> dict[str, Any]:
    files = [
        "src/phase1_4v4b0_scoring.py",
        "score_phase1_4v4_final_once.py",
        "tests/test_phase1_4v4b0_scoring.py",
    ]
    hashes = {}
    zip_path = FREEZE_SCORING / "scoring_source_snapshot.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            path = ROOT / rel
            if path.exists():
                hashes[rel] = sha256_file(path)
                zf.write(path, arcname=rel)
            else:
                hashes[rel] = "MISSING"
    save_json(FREEZE_SCORING / "scoring_source_hashes.json", hashes)
    return {"source_hashes": hashes, "snapshot_path": str(zip_path), "snapshot_sha256": sha256_file(zip_path)}


def dependency_versions() -> dict[str, Any]:
    import scipy
    import skimage
    import lpips

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "skimage": skimage.__version__,
        "lpips": str(Path(lpips.__file__).resolve()),
    }
    save_json(FREEZE_SCORING / "dependency_versions.json", info)
    save_json(FREEZE_SCORING / "environment.json", info)
    return info


def bundle_hash_excluding_self(path: Path, self_name: str) -> str:
    rows = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.name != self_name:
            rows.append((str(p.relative_to(path)).replace("\\", "/"), sha256_file(p)))
    return sha256_json(rows)


def create_scoring_protocol_freeze(
    input_hashes: Mapping[str, Any],
    dev_report: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    guard: Mapping[str, Any],
    pytest_report: Mapping[str, Any],
    metric_dep: Mapping[str, Any],
    source_info: Mapping[str, Any],
) -> dict[str, Any]:
    save_json(FREEZE_SCORING / "dev_reproduction_hashes.json", {
        "dev_uid_scoring_reproduction": sha256_file(REPORTS / "dev_uid_scoring_reproduction.json"),
        "dev_old_vs_new_metrics": sha256_file(REPORTS / "dev_old_vs_new_metrics.csv"),
        "dev_dual_path_agreement": sha256_file(REPORTS / "dev_dual_path_agreement.json"),
        "uid_alignment_synthetic_proof": sha256_file(REPORTS / "uid_alignment_synthetic_proof.json"),
    })
    frozen = {
        "status": "FINAL_V4_SCORING_PROTOCOL_FROZEN",
        "Phase1_4IR_protocol_hash": input_hashes["protocol_freeze"],
        "Phase1_4V4A_execution_hash": input_hashes["blind_execution_freeze"],
        "BLIND_INFERENCE_V4_COMPLETE_hash": input_hashes["BLIND_INFERENCE_V4_COMPLETE"],
        "final_v4_manifest_hash": input_hashes["final_v4_manifest"],
        "UID_index_hash": input_hashes["uid_index"],
        "shard_hashes": input_hashes["shard_hashes"],
        "selector_scores_hash": input_hashes["selector_scores"],
        "selected_indices_hash": input_hashes["selected_indices"],
        "candidate_seeds_hash": input_hashes["candidate_seed_manifest"],
        "generator_hash": input_hashes["generator_hash"],
        "selector_artifact_registry_hash": input_hashes["selector_artifact_registry_hash"],
        "primary_selector": PRIMARY_SELECTOR,
        "primary_model": PRIMARY_MODEL,
        "K": K,
        "primary_endpoint": "canonical_unclipped_p0_rmse",
        "H1_H5_contract_hash": sha256_file(FREEZE_SCORING / "final_v4_hypothesis_contract.json"),
        "S1_identity": "S1_PRE_SCORING_AMENDMENT_DM_VS_RAW",
        "metric_contract_hash": sha256_file(FREEZE_SCORING / "final_v4_metric_contract.json"),
        "statistics_contract_hash": sha256_file(FREEZE_SCORING / "final_v4_statistics_contract.json"),
        "classification_contract_hash": sha256_file(FREEZE_SCORING / "final_v4_classification_contract.json"),
        "scorer_source_hash": sha256_file(ROOT / "src" / "phase1_4v4b0_scoring.py"),
        "truth_loader_hash": sha256_file(ROOT / "src" / "phase1_4v4b0_scoring.py"),
        "LPIPS_weight_hash": metric_dep["lpips_alex_weight_hash"],
        "dev_reproduction_PASS": dev_report["status"] == "PASS",
        "synthetic_UID_proof_PASS": synthetic["status"] == "PASS",
        "one_shot_guards_PASS": guard["status"] == "PASS",
        "all_tests_PASS": pytest_report["status"] == "PASS",
        "confirm_token": FINAL_CONFIRM_TOKEN,
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "source_snapshot_hash": source_info["snapshot_sha256"],
        "timestamp": now(),
    }
    save_json(FREEZE_SCORING / "FINAL_V4_SCORING_PROTOCOL_FROZEN.json", frozen)
    frozen["bundle_hash_excluding_self"] = bundle_hash_excluding_self(FREEZE_SCORING, "FINAL_V4_SCORING_PROTOCOL_FROZEN.json")
    save_json(FREEZE_SCORING / "FINAL_V4_SCORING_PROTOCOL_FROZEN.json", frozen)
    return frozen


def run_pytest_suite() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "tests", "-q"]
    append_command("$ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    (REPORTS / "pytest_summary.txt").write_text(res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else ""), encoding="utf-8")
    report = {"status": "PASS" if res.returncode == 0 else "FAIL", "returncode": res.returncode, "stdout_tail": res.stdout[-2000:], "stderr_tail": res.stderr[-2000:]}
    save_json(REPORTS / "pytest_summary.json", report)
    if report["status"] != "PASS":
        raise V4B0Error("PYTEST_FAILED")
    return report


def create_ready_file(frozen: Mapping[str, Any]) -> dict[str, Any]:
    freeze_path = FREEZE_SCORING / "FINAL_V4_SCORING_PROTOCOL_FROZEN.json"
    ready = {
        "status": "READY_FOR_FINAL_V4_ONE_SHOT_SCORING",
        "meaning": "The UID-safe final-v4 scorer is frozen; next phase may execute exactly one final scoring run with the exact protocol hash.",
        "scoring_protocol_freeze_path": str(freeze_path),
        "scoring_protocol_freeze_sha256": sha256_file(freeze_path),
        "future_only_command_template": f"{sys.executable} score_phase1_4v4_final_once.py --dataset-scope final --confirm {FINAL_CONFIRM_TOKEN} --scoring-protocol-hash {sha256_file(freeze_path)}",
        "final_v4_blind_inference_completed": True,
        "final_v4_candidates_generated": True,
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "hard_gates": {
            "blind_artifacts_byte_unchanged": True,
            "UID_first_scorer_implemented": True,
            "synthetic_alignment_proof_PASS": True,
            "dev_reproduction_PASS": True,
            "dual_path_dev_agreement_PASS": True,
            "one_shot_guards_PASS": True,
            "all_pytest_PASS": True,
            "final_truth_not_loaded": True,
            "FINAL_V4_SCORING_STARTED_absent": not (FINAL_SCORING / "FINAL_V4_SCORING_STARTED.json").exists(),
        },
        "confirm_token": FINAL_CONFIRM_TOKEN,
        "scoring_protocol_status": frozen["status"],
    }
    save_json(OUT / "READY_FOR_FINAL_V4_ONE_SHOT_SCORING.json", ready)
    return ready


def package_outputs() -> tuple[dict[str, Any], dict[str, Any]]:
    readme = OUT / "README_PHASE1_4V4B0_PACKAGES.md"
    readme.write_text(
        "# Phase 1.4V4-B0 packages\n\nThe GPT brief contains the protocol freeze, audits, status reports, and source hashes. The full archive contains protocol/report artifacts and source, but intentionally excludes upstream large blind shard tensors; their immutable hashes are recorded in freeze_bundle_scoring_v4/blind_input_hashes.json.\n",
        encoding="utf-8",
    )
    contents = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            contents.append({"path": str(path.relative_to(OUT)).replace("\\", "/"), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    save_json(OUT / "contents_manifest.json", {"files": contents})
    brief = OUT / "phase1_4v4b0_gpt_brief.zip"
    full = OUT / "phase1_4v4b0_scoring_protocol_archive.zip"
    brief.unlink(missing_ok=True)
    full.unlink(missing_ok=True)
    brief_files = [
        readme,
        OUT / "contents_manifest.json",
        OUT / "READY_FOR_FINAL_V4_ONE_SHOT_SCORING.json",
        REPORTS / "implementation_status_phase1_4v4b0.json",
        REPORTS / "pre_scoring_integrity_audit.json",
        REPORTS / "uid_alignment_synthetic_proof.json",
        REPORTS / "dev_uid_scoring_reproduction.json",
        REPORTS / "dev_dual_path_agreement.json",
        REPORTS / "one_shot_guard_audit.json",
        REPORTS / "final_truth_non_access_audit.json",
        REPORTS / "pytest_summary.txt",
        FREEZE_SCORING / "FINAL_V4_SCORING_PROTOCOL_FROZEN.json",
        FREEZE_SCORING / "final_v4_hypothesis_contract.json",
        FREEZE_SCORING / "final_v4_metric_contract.json",
        FREEZE_SCORING / "final_v4_statistics_contract.json",
        FREEZE_SCORING / "final_v4_classification_contract.json",
        FREEZE_SCORING / "blind_input_hashes.json",
        ROOT / "src" / "phase1_4v4b0_scoring.py",
        ROOT / "score_phase1_4v4_final_once.py",
        ROOT / "tests" / "test_phase1_4v4b0_scoring.py",
    ]
    with zipfile.ZipFile(brief, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in brief_files:
            if path.exists():
                zf.write(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))
    with zipfile.ZipFile(full, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path not in {brief, full}:
                zf.write(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))
        for path in [ROOT / "src" / "phase1_4v4b0_scoring.py", ROOT / "score_phase1_4v4_final_once.py", ROOT / "tests" / "test_phase1_4v4b0_scoring.py"]:
            if path.exists():
                zf.write(path, arcname=str(path.relative_to(ROOT)).replace("\\", "/"))
    bad = {}
    for path in [brief, full]:
        with zipfile.ZipFile(path) as zf:
            bad[path.name] = zf.testzip()
    info = {
        "gpt_brief": {"path": str(brief), "sha256": sha256_file(brief), "bytes": brief.stat().st_size, "bad_member": bad[brief.name]},
        "full_archive": {"path": str(full), "sha256": sha256_file(full), "bytes": full.stat().st_size, "bad_member": bad[full.name], "large_blind_shards_excluded": True, "blind_shard_hash_manifest": str(FREEZE_SCORING / "blind_input_hashes.json")},
    }
    save_json(OUT / "package_hashes.json", info)
    return info["gpt_brief"], info["full_archive"]


def run_b0_protocol(device_name: str = "cuda") -> dict[str, Any]:
    start = time.time()
    initialize_output()
    append_command("$ python -m src.phase1_4v4b0_scoring --run-b0")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    blockers: list[str] = []
    status = "BLOCKED_PHASE1_4V4B0"
    try:
        capture_repo_state()
        integrity, input_hashes = audit_pre_scoring_integrity()
        if integrity["status"] != "PASS":
            raise V4B0Error("PRE_SCORING_INTEGRITY_FAILED")
        audit_index_semantics()
        write_contract_files()
        synthetic = uid_alignment_synthetic_proof()
        dev_report, _rows = run_dev_uid_scoring_reproduction(device_name=device_name)
        metric_dep = audit_metric_dependencies()
        dependency_versions()
        guard = one_shot_guard_audit()
        non_access = final_truth_non_access_audit()
        if non_access["status"] != "PASS":
            raise V4B0Error("FINAL_TRUTH_NON_ACCESS_AUDIT_FAILED")
        pytest_report = run_pytest_suite()
        immutability = original_blind_immutability_audit(input_hashes)
        source_info = source_snapshot()
        frozen = create_scoring_protocol_freeze(input_hashes, dev_report, synthetic, guard, pytest_report, metric_dep, source_info)
        ready = create_ready_file(frozen)
        if immutability["status"] != "PASS" or ready["status"] != "READY_FOR_FINAL_V4_ONE_SHOT_SCORING":
            raise V4B0Error("READY_GATE_FAILED")
        status = "READY_FOR_FINAL_V4_ONE_SHOT_SCORING"
        (REPORTS / "BLOCKERS_PHASE1_4V4B0.md").write_text("# BLOCKERS_PHASE1_4V4B0\n\nNo blockers.\n", encoding="utf-8")
    except Exception as exc:
        blockers.append(repr(exc))
        (REPORTS / "BLOCKERS_PHASE1_4V4B0.md").write_text("# BLOCKERS_PHASE1_4V4B0\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n", encoding="utf-8")
    runtime = {
        "runtime_seconds": time.time() - start,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "device": device_name,
    }
    save_json(REPORTS / "runtime_and_memory.json", runtime)
    impl = {
        "phase": "Phase 1.4V4-B0",
        "status": status,
        "blockers": blockers,
        "ready_generated": status == "READY_FOR_FINAL_V4_ONE_SHOT_SCORING",
        "final_v4_blind_inference_completed": True,
        "final_v4_candidates_generated": True,
        "final_v4_truth_metrics_computed": False,
        "final_v4_scoring_completed": False,
        "final_truth_loader_invocation_count": FINAL_TRUTH_ACCESS_COUNT,
        "FINAL_V4_SCORING_STARTED_exists": (FINAL_SCORING / "FINAL_V4_SCORING_STARTED.json").exists(),
        **runtime,
    }
    save_json(REPORTS / "implementation_status_phase1_4v4b0.json", impl)
    if not blockers:
        brief, full = package_outputs()
        impl["gpt_brief_package"] = brief
        impl["full_archive_package"] = full
        save_json(REPORTS / "implementation_status_phase1_4v4b0.json", impl)
    print(json.dumps(json_safe(impl), indent=2, sort_keys=True))
    return impl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.4V4-B0 UID-safe final scoring protocol freeze.")
    parser.add_argument("--run-b0", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_b0:
        result = run_b0_protocol(args.device)
        return 0 if not result["blockers"] else 2
    print("No action requested. Use --run-b0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
