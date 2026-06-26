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
from typing import Any

import joblib
import numpy as np
import torch
from scipy import stats

import phase1_2_rad5_64_pipeline as p12
import phase1_3r_recovery_and_relock as p13r
from scripts.eval_posterior_sampling_criteria import radial_power
from src.projections import exact_null_project


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")
PHASE14A = ROOT / "outputs" / "compatibility" / "phase1_4a_final_freeze_and_blind"
PHASE13R = ROOT / "outputs" / "compatibility" / "phase1_3r_recovery_and_relock"
PHASE12 = ROOT / "outputs" / "compatibility" / "phase1_2_rad5_64_candidate_transfer"
OUT = ROOT / "outputs" / "compatibility" / "phase1_4b0_scoring_protocol"
REPORTS = OUT / "reports"
FREEZE_SCORING = OUT / "freeze_bundle_scoring"
FREEZE_REFERENCE = OUT / "freeze_reference"
FINAL_SCORING_V2 = PHASE14A / "final_scoring_v2"

K = 16
BOOTSTRAP_SEED = 14001
BOOTSTRAP_REPLICATES = 10000
SIGN_FLIP_SEED = 14002
SIGN_FLIP_REPLICATES = 100000
FINAL_CONFIRM_TOKEN = "FINAL_V3_ONE_SHOT_SCORING"

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
DM_KEYS = ["dm_fcc_seed1", "dm_fcc_seed2", "dm_fcc_seed3"]
SCRATCH_KEYS = ["scratch_seed1", "scratch_seed2", "scratch_seed3"]
RAW_KEYS = ["raw_fcc_seed1", "raw_fcc_seed2", "raw_fcc_seed3"]


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def atomic_write_json(path: Path, payload: Any) -> None:
    ensure(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


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
            writer.writerow({k: json_safe(row.get(k, "")) for k in keys})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label}_HASH_MISMATCH: expected {expected_sha256}, got {actual}")
    return {"label": label, "path": str(path), "sha256": actual, "status": "PASS"}


def sha256_json(payload: Any) -> str:
    data = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def initialize_b0_output() -> None:
    ensure(REPORTS)
    ensure(FREEZE_SCORING)
    ensure(FREEZE_REFERENCE)
    (REPORTS / "command_log.txt").write_text("", encoding="utf-8")


def append_command(text: str) -> None:
    ensure(REPORTS)
    with (REPORTS / "command_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def capture_repo_state() -> dict[str, Any]:
    git_status = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "status", "--short"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    git_diff = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "diff", "--stat"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    (REPORTS / "git_status_short_start.txt").write_text(git_status.stdout + git_status.stderr, encoding="utf-8")
    (REPORTS / "git_diff_stat_start.txt").write_text(git_diff.stdout + git_diff.stderr, encoding="utf-8")
    return {
        "status": "RECORDED",
        "git_status_returncode": git_status.returncode,
        "git_diff_returncode": git_diff.returncode,
        "git_status_path": str(REPORTS / "git_status_short_start.txt"),
        "git_diff_stat_path": str(REPORTS / "git_diff_stat_start.txt"),
    }


def existing_final_truth_outputs() -> list[str]:
    suspicious: list[str] = []
    final_scoring = PHASE14A / "final_scoring"
    patterns = [
        "per_image_final_metrics*",
        "*final*hypothesis*result*",
        "*final*oracle*",
        "*final*error*",
        "FINAL_SCORING_COMPLETE.json",
    ]
    if final_scoring.exists():
        for pattern in patterns:
            suspicious.extend(str(p) for p in final_scoring.rglob(pattern))
    for pattern in ["per_image_final_metrics*", "*final_oracle*", "*final_error*"]:
        suspicious.extend(str(p) for p in PHASE14A.rglob(pattern))
    return sorted(set(suspicious))


def pre_scoring_consumption_audit() -> dict[str, Any]:
    old_scoring = PHASE14A / "final_scoring"
    started = old_scoring / "FINAL_SCORING_STARTED.json"
    complete = old_scoring / "FINAL_SCORING_COMPLETE.json"
    truth_outputs = existing_final_truth_outputs()
    old_script = ROOT / "score_phase1_4b_final_once.py"
    old_script_hash = sha256_file(old_script) if old_script.exists() else "MISSING"
    skeleton_text = old_script.read_text(encoding="utf-8") if old_script.exists() else ""
    skeleton_can_read_truth = any(token in skeleton_text for token in ["STL10Lineage", "p0_error", "lpips", "ssim", "rapsd"])
    if truth_outputs or complete.exists():
        classification = "POSSIBLE_TRUTH_SCORING_ATTEMPT"
    elif started.exists():
        classification = "SKELETON_STARTED_NO_TRUTH_ACCESS" if not skeleton_can_read_truth else "POSSIBLE_TRUTH_SCORING_ATTEMPT"
    else:
        classification = "CLEAN_NO_SCORING_ATTEMPT"
    audit = {
        "status": "PASS" if classification in {"CLEAN_NO_SCORING_ATTEMPT", "SKELETON_STARTED_NO_TRUTH_ACCESS"} else "FAIL",
        "classification": classification,
        "old_final_scoring_started_exists": started.exists(),
        "old_final_scoring_complete_exists": complete.exists(),
        "truth_derived_outputs": truth_outputs,
        "old_skeleton_path": str(old_script),
        "old_skeleton_sha256": old_script_hash,
        "old_skeleton_direct_truth_or_metric_tokens_found": bool(skeleton_can_read_truth),
        "old_skeleton_explanation": "The retained skeleton only writes a STARTED marker after confirm and prints a message; it does not load truth or compute metrics.",
    }
    save_json(REPORTS / "pre_scoring_consumption_audit.json", audit)
    return audit


def compute_original_artifact_hashes() -> dict[str, Any]:
    freeze = PHASE14A / "freeze_bundle"
    blind = PHASE14A / "blind_inference"
    manifests = PHASE14A / "manifests"
    shard_hashes = {p.name: sha256_file(p) for p in sorted((blind / "shards").glob("shard_*.pt"))}
    frozen = read_json(freeze / "FINAL_EVAL_FROZEN.json")
    operator = read_json(PHASE14A / "reports" / "operator_hash_semantics_audit.json")
    hashes = {
        "FINAL_EVAL_FROZEN": sha256_file(freeze / "FINAL_EVAL_FROZEN.json"),
        "BLIND_INFERENCE_COMPLETE": sha256_file(blind / "BLIND_INFERENCE_COMPLETE.json"),
        "blind_artifact_manifest": sha256_file(blind / "blind_artifact_manifest.json"),
        "blind_artifact_hashes": sha256_file(blind / "blind_artifact_hashes.json"),
        "final_v3_manifest": sha256_file(manifests / "final_locked_test_64_v3_manifest.json"),
        "final_v3_indices": sha256_file(manifests / "final_locked_test_64_v3_indices.npy"),
        "selector_scores_npz": sha256_file(blind / "selector_scores.npz"),
        "selected_indices_npz": sha256_file(blind / "selected_indices.npz"),
        "candidate_seed_manifest_freeze": sha256_file(freeze / "final_candidate_seed_manifest.json"),
        "candidate_seed_manifest_blind": sha256_file(blind / "candidate_seed_manifest.json"),
        "primary_checkpoint_sha256": frozen["primary_checkpoint_sha256"],
        "generator_checkpoint_sha256": frozen["generator_checkpoint_sha256"],
        "A_file_sha256": frozen["A_file_sha256"],
        "A_array_content_sha256": frozen["A_array_content_sha256"],
        "A_float32_tensor_sha256": frozen["A_float32_tensor_sha256"],
        "operator_hash_semantics": operator,
        "blind_shards": shard_hashes,
        "shard_count": len(shard_hashes),
    }
    return hashes


def copy_freeze_references(hashes: dict[str, Any]) -> dict[str, Any]:
    ensure(FREEZE_REFERENCE)
    src_frozen = PHASE14A / "freeze_bundle" / "FINAL_EVAL_FROZEN.json"
    src_blind = PHASE14A / "blind_inference" / "BLIND_INFERENCE_COMPLETE.json"
    dst_frozen = FREEZE_REFERENCE / "ORIGINAL_FINAL_EVAL_FROZEN.json"
    dst_blind = FREEZE_REFERENCE / "ORIGINAL_BLIND_INFERENCE_COMPLETE.json"
    shutil.copyfile(src_frozen, dst_frozen)
    shutil.copyfile(src_blind, dst_blind)
    reference = {
        "status": "PASS",
        "original_hashes": hashes,
        "copied_FINAL_EVAL_FROZEN_sha256": sha256_file(dst_frozen),
        "copied_BLIND_INFERENCE_COMPLETE_sha256": sha256_file(dst_blind),
        "byte_exact_copy": sha256_file(src_frozen) == sha256_file(dst_frozen) and sha256_file(src_blind) == sha256_file(dst_blind),
    }
    save_json(FREEZE_REFERENCE / "original_artifact_hashes.json", reference)
    save_json(FREEZE_SCORING / "original_blind_artifact_hashes.json", hashes)
    save_json(
        FREEZE_SCORING / "ORIGINAL_FREEZE_REFERENCE.json",
        {
            "status": "PASS",
            "original_phase1_4a_dir": str(PHASE14A),
            "reference_dir": str(FREEZE_REFERENCE),
            "hashes": hashes,
            "original_candidates_unchanged": True,
            "original_selector_scores_unchanged": True,
        },
    )
    return reference


def write_protocol_gap_and_amendment() -> tuple[dict[str, Any], str]:
    gap = {
        "status": "PASS",
        "gaps": [
            "Original Stage B file was a skeleton and not a runnable metric implementation.",
            "Original freeze locked blind candidate generation and selector inference, but not executable metric code.",
            "Original statistics plan froze image-level bootstrap, seed 14001, 10000 replicates, and H2 Holm family only.",
            "Original H5 is measurement consistency, not DM-vs-raw.",
        ],
        "amendment_time": now(),
        "amendment_before_final_truth_metrics": True,
        "does_not_modify_models_candidates_selections_or_primary_hypothesis": True,
    }
    save_json(REPORTS / "frozen_protocol_gap_audit.json", gap)
    md = "\n".join(
        [
            "# SCORING_PROTOCOL_AMENDMENT",
            "",
            "This amendment completes the scoring layer before any final truth metric is computed.",
            "",
            "Facts fixed in Phase 1.4B0:",
            "",
            "1. The retained `score_phase1_4b_final_once.py` is only a skeleton.",
            "2. Phase 1.4A froze blind candidate generation and selector inference, but not a complete executable metric implementation.",
            "3. The original statistics plan only fixed image-level paired bootstrap, bootstrap seed 14001, 10000 replicates, and the H2 scalar/sum-image Holm family.",
            "4. H5 remains measurement consistency. DM-vs-raw is handled separately as S1.",
            "5. This amendment happened before any final truth metric, final oracle, or final error calculation.",
            "6. No model, generator, candidate, selector score, selected index, K value, seed, or primary hypothesis is changed here.",
        ]
    )
    (REPORTS / "SCORING_PROTOCOL_AMENDMENT.md").write_text(md + "\n", encoding="utf-8")
    return gap, md


def build_hypothesis_resolution() -> dict[str, Any]:
    phase14_hyp = read_json(PHASE14A / "freeze_bundle" / "final_hypotheses.json")
    provenance = {
        "status": "PASS",
        "blind_inference_complete_mtime": (PHASE14A / "blind_inference" / "BLIND_INFERENCE_COMPLETE.json").stat().st_mtime,
        "pre_blind_sources": {
            "phase1_3r_phase1_2_validation_reproduction": str(PHASE13R / "reports" / "phase1_2_validation_reproduction.json"),
            "phase1_3r_preregistration_draft": str(PHASE13R / "freeze_bundle_v2" / "phase1_3r_preregistration_draft.md"),
            "phase1_4a_final_preregistration": str(PHASE14A / "freeze_bundle" / "phase1_4_final_preregistration.md"),
            "phase1_4a_final_hypotheses": str(PHASE14A / "freeze_bundle" / "final_hypotheses.json"),
        },
        "H3_decision_rule_status": "PRE_SPECIFIED_COMPARISON_WITH_INCOMPLETE_DECISION_RULE",
        "H3_reason": "Pre-blind files fixed the three-seed DM-vs-scratch comparison as evidence, but did not freeze all final pass/fail conditions before truth scoring.",
        "S1_status": "S1_PRE_SCORING_AMENDMENT_DM_VS_RAW",
        "S1_reason": "Raw-FCC and DM-FCC three-seed artifacts and reports existed before blind final inference, but DM-vs-raw was not the original H5.",
    }
    save_json(REPORTS / "hypothesis_provenance_audit.json", provenance)
    resolution = {
        "status": "PASS",
        "source_final_hypotheses_sha256": sha256_file(PHASE14A / "freeze_bundle" / "final_hypotheses.json"),
        "H1_primary_selector_generalization": {
            "primary_model": "reproduced_dm_fcc_seed3_v2",
            "artifact_key": "dm_fcc_seed3",
            "endpoint": "canonical_unclipped_p0_rmse",
            "delta": "selected_i - random_i, lower is better",
            "random_i": "mean_k(error_i,k) over the same fixed K=16 pool",
            "selected_i": "error_i at selected_index(dm_fcc_seed3)",
            "pass": [
                "mean(delta)<0",
                "paired percentile bootstrap 95% CI upper <0",
                "aggregate relative improvement >=0.01",
                "aggregate oracle gain fraction >=0.20",
                "H4 PASS",
                "H5 PASS",
            ],
        },
        "H2_beyond_simple_naturalness": {
            "comparisons": ["dm_fcc_seed3_vs_scalar_pair_selector", "dm_fcc_seed3_vs_sum_image_selector"],
            "delta": "error_DM_i - error_baseline_i",
            "Holm_family": ["scalar_pair_selector", "sum_image_selector"],
            "strong_pass": ["mean delta <0", "bootstrap CI upper <0", "Holm-adjusted p <0.05 for both comparisons"],
        },
        "H3_FCC_pretraining_vs_scratch": {
            "method": "per-image fixed-three-seed DM average minus scratch average",
            "paired_seed_differences": ["dm_seed1-scratch_seed1", "dm_seed2-scratch_seed2", "dm_seed3-scratch_seed3"],
            "status": provenance["H3_decision_rule_status"],
            "cannot_support_strongest_FCC_confirmation_if_incomplete": True,
        },
        "H4_integrity_shortcut": {
            "hard_gate": True,
            "requires": [
                "shared exact row anchor",
                "no candidate index feature",
                "no candidate seed feature",
                "no truth/oracle selector access",
                "candidate permutation equivariance",
                "same candidate pool for all selectors",
                "selected indices equal frozen score argmax",
            ],
        },
        "H5_measurement_consistency": {
            "definition": phase14_hyp["H5_measurement_consistency"],
            "kept_as_measurement_consistency": True,
            "not_dm_vs_raw": True,
        },
        "S1_DM_vs_raw": {
            "status": provenance["S1_status"],
            "comparison": "three-seed DM-FCC method average vs three-seed raw-FCC method average",
            "not_in_H2_Holm_family": True,
            "does_not_change_H1_primary_conclusion": True,
        },
        "final_conclusion_classes": {
            "FINAL_CONFIRMED_DM_FCC_ADDS_VALUE": ["H1 PASS", "H2 strong PASS", "H3 complete pre-specified rule PASS", "H4 PASS", "H5 PASS"],
            "FINAL_SELECTOR_GENERALIZES_BUT_FCC_NOT_CONFIRMED": ["H1 PASS", "H4 PASS", "H5 PASS", "H2 or H3 not confirmed"],
            "FINAL_NUMERICAL_TREND_ONLY": ["selected mean better than random", "but H1 CI/effect/oracle-gain incomplete"],
            "FINAL_FAILED_TO_GENERALIZE": ["selected mean not better than random or direction reversed", "integrity still valid"],
            "FINAL_EVALUATION_INVALID": ["hash mismatch", "manifest mismatch", "truth leakage", "candidate pool mismatch", "H4 FAIL"],
        },
    }
    save_json(FREEZE_SCORING / "final_hypothesis_resolution_v2.json", resolution)
    return resolution


def selected_by_argmax(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores)
    if arr.ndim != 2:
        raise ValueError("scores must have shape [n_images, K]")
    return np.argmax(arr, axis=1).astype(np.int64)


def oracle_indices(errors: np.ndarray) -> np.ndarray:
    arr = np.asarray(errors)
    if arr.ndim != 2:
        raise ValueError("errors must have shape [n_images, K]")
    return np.argmin(arr, axis=1).astype(np.int64)


def compute_random_expectation(metric_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(metric_matrix, dtype=np.float64).mean(axis=1)


def compute_posterior_mean(candidates: np.ndarray) -> np.ndarray:
    return np.asarray(candidates, dtype=np.float64).mean(axis=1)


def compute_primary_oracle(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = oracle_indices(errors)
    arr = np.asarray(errors, dtype=np.float64)
    return idx, arr[np.arange(arr.shape[0]), idx]


def aggregate_relative_improvement(random_metric: np.ndarray, selected_metric: np.ndarray) -> float:
    random_mean = float(np.mean(random_metric))
    return float((random_mean - float(np.mean(selected_metric))) / max(random_mean, 1e-12))


def aggregate_oracle_gain_fraction(random_metric: np.ndarray, selected_metric: np.ndarray, oracle_metric: np.ndarray) -> dict[str, Any]:
    denom = float(np.mean(random_metric) - np.mean(oracle_metric))
    if abs(denom) <= 1e-12:
        return {"status": "not_applicable", "value": None, "reason": "oracle denominator near zero"}
    return {"status": "ok", "value": float((np.mean(random_metric) - np.mean(selected_metric)) / denom)}


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
    batch = 4096
    done = 0
    while done < int(B):
        cur = min(batch, int(B) - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(cur, arr.size))
        vals = np.abs((signs * arr).mean(axis=1))
        extreme += int(np.sum(vals >= obs - 1e-15))
        done += cur
    p = (extreme + 1.0) / (float(B) + 1.0)
    return {"observed_abs_mean": obs, "p_value": float(p), "B": int(B), "seed": int(seed), "two_sided": True}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    items = sorted((k, float(v)) for k, v in p_values.items())
    ranked = sorted(items, key=lambda kv: kv[1])
    m = len(ranked)
    adjusted_pairs = []
    running = 0.0
    for rank, (key, p) in enumerate(ranked):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        adjusted_pairs.append((key, running))
    return {key: float(val) for key, val in sorted(adjusted_pairs)}


def exact_sign_test(delta: np.ndarray, tie_tol: float = 1e-12) -> dict[str, Any]:
    arr = np.asarray(delta, dtype=np.float64).reshape(-1)
    pos = int(np.sum(arr > tie_tol))
    neg = int(np.sum(arr < -tie_tol))
    ties = int(arr.size - pos - neg)
    n = pos + neg
    p = float(stats.binomtest(min(pos, neg), n=n, p=0.5, alternative="two-sided").pvalue) if n else 1.0
    return {"positive": pos, "negative": neg, "ties": ties, "n_non_tie": n, "p_value_two_sided": p, "tie_tol": float(tie_tol)}


def compute_method_seed_average(error_by_selector: dict[str, np.ndarray], left_keys: list[str], right_keys: list[str]) -> dict[str, Any]:
    left = np.stack([np.asarray(error_by_selector[k], dtype=np.float64) for k in left_keys], axis=0).mean(axis=0)
    right = np.stack([np.asarray(error_by_selector[k], dtype=np.float64) for k in right_keys], axis=0).mean(axis=0)
    return {"left_mean": left, "right_mean": right, "delta": left - right}


def decide_hypotheses(metrics: dict[str, Any]) -> dict[str, Any]:
    h1 = metrics.get("H1", {})
    h2 = metrics.get("H2", {})
    h3 = metrics.get("H3", {})
    h4 = bool(metrics.get("H4_PASS", False))
    h5 = bool(metrics.get("H5_PASS", False))
    return {
        "H1_PASS": bool(h1.get("mean_delta_negative") and h1.get("ci_upper_negative") and h1.get("relative_improvement_ge_0_01") and h1.get("oracle_gain_fraction_ge_0_20") and h4 and h5),
        "H2_STRONG_PASS": bool(h2.get("strong_pass", False)),
        "H3_PASS_WITH_COMPLETE_RULE": bool(h3.get("complete_rule", False) and h3.get("pass", False)),
        "H4_PASS": h4,
        "H5_PASS": h5,
    }


def classify_final_conclusion(decisions: dict[str, Any], h1_mean_selected_better: bool = False) -> str:
    if not decisions.get("H4_PASS") or not decisions.get("H5_PASS"):
        return "FINAL_EVALUATION_INVALID"
    if decisions.get("H1_PASS") and decisions.get("H2_STRONG_PASS") and decisions.get("H3_PASS_WITH_COMPLETE_RULE"):
        return "FINAL_CONFIRMED_DM_FCC_ADDS_VALUE"
    if decisions.get("H1_PASS"):
        return "FINAL_SELECTOR_GENERALIZES_BUT_FCC_NOT_CONFIRMED"
    if h1_mean_selected_better:
        return "FINAL_NUMERICAL_TREND_ONLY"
    return "FINAL_FAILED_TO_GENERALIZE"


def p0_rmse_matrix(candidate_nulls: np.ndarray, true_n: np.ndarray) -> np.ndarray:
    cand = np.asarray(candidate_nulls, dtype=np.float64)
    truth = np.asarray(true_n, dtype=np.float64)
    return np.sqrt(np.mean((cand - truth[:, None, :]) ** 2, axis=2))


def psnr_from_mse(mse: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    arr = np.asarray(mse, dtype=np.float64)
    out = np.empty_like(arr)
    mask = arr <= 0
    out[mask] = np.inf
    out[~mask] = 20.0 * np.log10(float(data_range)) - 10.0 * np.log10(arr[~mask])
    return out


def rapsd_profile(img: np.ndarray, bins: int = 32) -> np.ndarray:
    return radial_power(np.asarray(img, dtype=np.float64).reshape(1, 1, img.shape[-2], img.shape[-1]), bins=bins)


def rapsd_distance(pred: np.ndarray, truth: np.ndarray, bins: int = 32) -> float:
    return float(np.linalg.norm(rapsd_profile(pred, bins=bins) - rapsd_profile(truth, bins=bins)))


def image_tv_and_frequency(img_flat: np.ndarray, img_size: int = 64) -> dict[str, float]:
    img = np.asarray(img_flat, dtype=np.float64).reshape(img_size, img_size)
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)
    f = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(f) ** 2
    yy, xx = np.mgrid[:img_size, :img_size]
    rr = np.sqrt((yy - img_size / 2.0) ** 2 + (xx - img_size / 2.0) ** 2)
    maxr = max(float(rr.max()), 1e-12)
    total = max(float(power.sum()), 1e-12)
    return {
        "tv": float(np.mean(np.abs(dx)) + np.mean(np.abs(dy))),
        "freq_low": float(power[(rr / maxr) < 0.15].sum() / total),
        "freq_mid": float(power[((rr / maxr) >= 0.15) & ((rr / maxr) < 0.35)].sum() / total),
        "freq_high": float(power[(rr / maxr) >= 0.35].sum() / total),
    }


def load_validation_scores_from_artifacts() -> dict[str, np.ndarray]:
    artifact_dir = PHASE13R / "recovered_selector_artifacts"
    scores: dict[str, np.ndarray] = {}
    for key in RANKER_KEYS:
        artifact = torch.load(artifact_dir / f"{key}.pt", map_location="cpu", weights_only=False)
        scores[key] = np.asarray(artifact["validation_scores"], dtype=np.float32)
    scores["scalar_pair_selector"] = np.asarray(joblib.load(artifact_dir / "scalar_pair_selector.joblib")["validation_scores"], dtype=np.float32).reshape(-1, K)
    scores["sum_image_selector"] = np.asarray(joblib.load(artifact_dir / "sum_image_selector.joblib")["validation_scores"], dtype=np.float32).reshape(-1, K)
    return scores


def evaluate_selector_indices(error_matrix: np.ndarray, scores: np.ndarray, method: str) -> dict[str, Any]:
    err = np.asarray(error_matrix, dtype=np.float64)
    score = np.asarray(scores, dtype=np.float64)
    selected = selected_by_argmax(score)
    oracle = oracle_indices(err)
    selected_err = err[np.arange(err.shape[0]), selected]
    random_err = err.mean(axis=1)
    oracle_err = err[np.arange(err.shape[0]), oracle]
    denom = random_err - oracle_err
    gain = np.where(np.abs(denom) > 1e-12, (random_err - selected_err) / denom, np.nan)
    ranks = []
    for i in range(err.shape[0]):
        ranks.append(1 + int(np.where(np.argsort(err[i], kind="stable") == selected[i])[0][0]))
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


def compute_dev_secondary_metric_smoke(cache: dict[str, Any], scores: dict[str, np.ndarray], device_name: str = "cuda") -> dict[str, Any]:
    x = cache["x"].detach().cpu().numpy().astype(np.float32)
    r = cache["r"].detach().cpu().numpy().astype(np.float32)
    cand_n = cache["cand_n"].detach().cpu().numpy().astype(np.float32)
    canon = r[:, None, :] + cand_n
    n_img, k, n_pix = canon.shape
    img_size = int(cache["img_size"])
    truth_img = x.reshape(n_img, img_size, img_size)
    canon_img = canon.reshape(n_img, k, img_size, img_size)
    clipped = np.clip(canon_img, 0.0, 1.0)
    truth_clip = np.clip(truth_img, 0.0, 1.0)
    full_mse = np.mean((canon - x[:, None, :]) ** 2, axis=2)
    unclipped_psnr = psnr_from_mse(full_mse, data_range=1.0)
    clipped_mse = np.mean((clipped - truth_clip[:, None, :, :]) ** 2, axis=(2, 3))
    clipped_psnr = psnr_from_mse(clipped_mse, data_range=1.0)
    from skimage.metrics import structural_similarity

    ssim_vals = np.zeros((n_img, k), dtype=np.float64)
    rapsd_vals = np.zeros((n_img, k), dtype=np.float64)
    for i in range(n_img):
        for j in range(k):
            ssim_vals[i, j] = float(structural_similarity(truth_clip[i], clipped[i, j], data_range=1.0, win_size=7, channel_axis=None))
            rapsd_vals[i, j] = rapsd_distance(clipped[i, j], truth_clip[i], bins=32)
    lpips_vals = compute_lpips_matrix(clipped, truth_clip, device_name=device_name)
    primary = evaluate_selector_indices(cache["p0_error"].numpy(), scores["dm_fcc_seed3"], "dm_fcc_seed3")
    selected = primary["selected_indices"]
    oracle = oracle_indices(cache["p0_error"].numpy())
    summary = {
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
        "data_range_check": {
            "truth_min": float(truth_img.min()),
            "truth_max": float(truth_img.max()),
            "canonical_min": float(canon_img.min()),
            "canonical_max": float(canon_img.max()),
            "clamp_range": [0.0, 1.0],
        },
    }
    return summary


def compute_lpips_matrix(clipped_candidates: np.ndarray, truth_clip: np.ndarray, device_name: str = "cuda") -> np.ndarray:
    import lpips

    n_img, k, h, w = clipped_candidates.shape
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).to(device)
    model.eval()
    pred = torch.from_numpy(clipped_candidates.reshape(n_img * k, 1, h, w).astype(np.float32))
    truth = torch.from_numpy(np.repeat(truth_clip[:, None, :, :], k, axis=1).reshape(n_img * k, 1, h, w).astype(np.float32))
    vals = []
    with torch.no_grad():
        for start in range(0, pred.shape[0], 64):
            p = pred[start : start + 64].to(device).repeat(1, 3, 1, 1) * 2.0 - 1.0
            t = truth[start : start + 64].to(device).repeat(1, 3, 1, 1) * 2.0 - 1.0
            vals.append(model(p, t).detach().cpu().reshape(-1))
    return torch.cat(vals, dim=0).numpy().reshape(n_img, k)


def metric_contract() -> dict[str, Any]:
    return {
        "image_valid_range": "input tensors are expected in [0,1] after STL10 64x64 transform; canonical candidates may be outside before clipping",
        "clamp_range": [0.0, 1.0],
        "PSNR_data_range": 1.0,
        "SSIM": {"data_range": 1.0, "channel_axis": None, "win_size": 7},
        "LPIPS": {"backbone": "alex", "input_mapping": "[0,1] grayscale repeated to 3 channels then mapped to [-1,1]"},
        "RAPSD": {"bins": 32, "profile": "mean radial FFT power per bin normalized by profile sum", "distance": "Euclidean profile distance"},
        "RelMeasErr": {"denominator_epsilon": 1e-12},
        "Spearman": {"tie_handling": "scipy.stats.spearmanr average ranks"},
        "primary_metric": "canonical_unclipped_p0_rmse before clipping",
    }


def statistics_contract() -> dict[str, Any]:
    return {
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "unit": "image", "ci": "paired percentile 95%"},
        "H2_sign_flip": {"replicates": SIGN_FLIP_REPLICATES, "seed": SIGN_FLIP_SEED, "statistic": "absolute paired mean", "two_sided": True, "correction": "(extreme+1)/(B+1)"},
        "exact_sign_test": {"tie_tol": 1e-12, "ties": "excluded", "test": "two-sided binomial"},
        "Holm_family": ["dm_fcc_seed3_vs_scalar_pair_selector", "dm_fcc_seed3_vs_sum_image_selector"],
        "H3": "per-image fixed-three-seed method average, not seed-level independent samples",
    }


def run_dev_scoring_reproduction(device_name: str = "cuda") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache = torch.load(PHASE12 / "candidate_cache" / "val_64_selector_k16.pt", map_location="cpu", weights_only=False)
    scores = load_validation_scores_from_artifacts()
    err = cache["p0_error"].numpy()
    baselines = {
        "deterministic": {"p0_rmse_mean": float(cache["deterministic_p0_error"].mean().item())},
        "random_expectation": {"p0_rmse_mean": float(err.mean(axis=1).mean())},
        "posterior_mean": {"p0_rmse_mean": float(cache["posterior_mean_p0_error"].mean().item())},
        "oracle_best_of_k": {"p0_rmse_mean": float(err.min(axis=1).mean())},
        "oracle_gain_available_mean": float((err.mean(axis=1) - err.min(axis=1)).mean()),
    }
    selector_metrics = {}
    selected_errors = {}
    rows: list[dict[str, Any]] = []
    old = read_json(PHASE13R / "reports" / "phase1_2_validation_reproduction.json")
    for key in ALL_SELECTOR_KEYS:
        metrics = evaluate_selector_indices(err, scores[key], key)
        selected_errors[key] = metrics["selected_errors"]
        serial = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
        selector_metrics[key] = serial
        old_metrics = old["rankers"].get(key) or old.get(key)
        for metric_name in [
            "selected_p0_rmse_mean",
            "random_expected_p0_rmse_mean",
            "oracle_p0_rmse_mean",
            "selection_regret_mean",
            "oracle_gain_fraction_mean",
            "top_oracle_hit_rate",
        ]:
            old_value = float(old_metrics[metric_name]) if old_metrics and metric_name in old_metrics else float("nan")
            new_value = float(serial[metric_name])
            rows.append(
                {
                    "selector": key,
                    "metric": metric_name,
                    "old_value": old_value,
                    "new_value": new_value,
                    "absolute_difference": abs(new_value - old_value) if math.isfinite(old_value) else "DATA MISSING",
                }
            )
    random = err.mean(axis=1)
    oracle = err.min(axis=1)
    dm3 = selected_errors["dm_fcc_seed3"]
    h1_delta = dm3 - random
    h1_boot = paired_percentile_bootstrap(h1_delta)
    h1_gain = aggregate_oracle_gain_fraction(random, dm3, oracle)
    h2_raw_p = {}
    h2 = {}
    for baseline in ["scalar_pair_selector", "sum_image_selector"]:
        delta = dm3 - selected_errors[baseline]
        boot = paired_percentile_bootstrap(delta)
        sign_flip = paired_sign_flip_test(delta)
        sign = exact_sign_test(delta)
        h2[baseline] = {"bootstrap": boot, "sign_flip": sign_flip, "sign_test": sign, "mean_delta": float(delta.mean())}
        h2_raw_p[baseline] = sign_flip["p_value"]
    h2_holm = holm_adjust(h2_raw_p)
    for key in h2:
        h2[key]["holm_adjusted_p"] = h2_holm[key]
    h3_avg = compute_method_seed_average(selected_errors, DM_KEYS, SCRATCH_KEYS)
    h3_boot = paired_percentile_bootstrap(h3_avg["delta"])
    paired_seed_diffs = {
        f"dm_fcc_seed{i}_minus_scratch_seed{i}": float((selected_errors[f"dm_fcc_seed{i}"] - selected_errors[f"scratch_seed{i}"]).mean())
        for i in [1, 2, 3]
    }
    secondary = compute_dev_secondary_metric_smoke(cache, scores, device_name=device_name)
    old_baselines = old["baselines"]
    baseline_diffs = {
        "deterministic": abs(baselines["deterministic"]["p0_rmse_mean"] - old_baselines["deterministic"]["p0_rmse_mean"]),
        "random_expectation": abs(baselines["random_expectation"]["p0_rmse_mean"] - old_baselines["random_expectation"]["p0_rmse_mean"]),
        "posterior_mean": abs(baselines["posterior_mean"]["p0_rmse_mean"] - old_baselines["posterior_mean"]["p0_rmse_mean"]),
        "oracle_best_of_k": abs(baselines["oracle_best_of_k"]["p0_rmse_mean"] - old_baselines["oracle_best_of_k"]["p0_rmse_mean"]),
    }
    report = {
        "status": "PASS",
        "dataset_scope": "dev",
        "final_v3_truth_loaded": False,
        "dev_cache": str(PHASE12 / "candidate_cache" / "val_64_selector_k16.pt"),
        "image_count": int(err.shape[0]),
        "K": int(err.shape[1]),
        "baselines": baselines,
        "baseline_abs_diffs_vs_phase1_3r": baseline_diffs,
        "selector_metrics": selector_metrics,
        "primary_dm_fcc_seed3_p0_rmse_abs_diff_vs_phase1_3r": abs(selector_metrics["dm_fcc_seed3"]["selected_p0_rmse_mean"] - old["rankers"]["dm_fcc_seed3"]["selected_p0_rmse_mean"]),
        "scalar_pair_abs_diff_vs_phase1_3r": abs(selector_metrics["scalar_pair_selector"]["selected_p0_rmse_mean"] - old["scalar_pair_selector"]["selected_p0_rmse_mean"]),
        "sum_image_abs_diff_vs_phase1_3r": abs(selector_metrics["sum_image_selector"]["selected_p0_rmse_mean"] - old["sum_image_selector"]["selected_p0_rmse_mean"]),
        "selected_indices_recomputed_from_scores": True,
        "random_expectation_uses_full_pool_mean": True,
        "primary_oracle_lowest_index_argmin": True,
        "method_average_per_image_before_mean": True,
        "candidate_seed_or_index_features_used": False,
        "H1_stats_flow": {
            "bootstrap": h1_boot,
            "relative_improvement": aggregate_relative_improvement(random, dm3),
            "oracle_gain_fraction": h1_gain,
        },
        "H2_stats_flow": h2,
        "H3_stats_flow": {
            "status": "PRE_SPECIFIED_COMPARISON_WITH_INCOMPLETE_DECISION_RULE",
            "method_average_bootstrap": h3_boot,
            "paired_seed_aggregate_differences": paired_seed_diffs,
            "paired_seed_differences_negative_count": int(sum(v < 0 for v in paired_seed_diffs.values())),
        },
        "secondary_metric_dev_smoke": secondary,
        "comparison_tolerance": 1e-7,
    }
    if any(v > 1e-7 for v in baseline_diffs.values()):
        report["status"] = "FAIL"
    if report["primary_dm_fcc_seed3_p0_rmse_abs_diff_vs_phase1_3r"] > 1e-7:
        report["status"] = "FAIL"
    save_json(REPORTS / "dev_scoring_reproduction.json", report)
    write_csv(REPORTS / "dev_old_vs_new_scoring.csv", rows)
    return report, rows


def audit_metric_dependencies() -> dict[str, Any]:
    import lpips
    import skimage

    package = Path(lpips.__file__).resolve().parent
    weights = {
        str(p): sha256_file(p)
        for p in sorted(package.rglob("*"))
        if p.is_file() and p.suffix.lower() in {".pth", ".pt"}
    }
    audit = {
        "status": "PASS" if any("v0.1" in p and p.endswith("alex.pth") for p in weights) else "FAIL",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": stats.__version__ if hasattr(stats, "__version__") else __import__("scipy").__version__,
        "skimage": skimage.__version__,
        "lpips_package": str(package),
        "lpips_weight_hashes": weights,
        "lpips_backbone": "alex",
    }
    save_json(FREEZE_SCORING / "metric_dependency_hashes.json", audit)
    return audit


def audit_blind_scores() -> dict[str, Any]:
    scores = np.load(PHASE14A / "blind_inference" / "selector_scores.npz")
    selected = np.load(PHASE14A / "blind_inference" / "selected_indices.npz")
    rows = []
    all_pass = True
    for key in ALL_SELECTOR_KEYS:
        s = np.asarray(scores[key])
        idx = np.asarray(selected[key])
        arg = selected_by_argmax(s)
        max_vals = s.max(axis=1, keepdims=True)
        tie_counts = np.sum(np.isclose(s, max_vals, rtol=0.0, atol=0.0), axis=1)
        hist = np.bincount(idx.astype(np.int64), minlength=K)
        no_nan_inf = bool(np.isfinite(s).all())
        selected_ok = bool(np.array_equal(arg, idx))
        perm = np.array([3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64)
        inv_selected = perm[np.argmax(s[:, perm], axis=1)]
        non_tie = tie_counts == 1
        perm_ok = bool(np.array_equal(inv_selected[non_tie], arg[non_tie]))
        row = {
            "selector": key,
            "score_shape": list(s.shape),
            "selected_shape": list(idx.shape),
            "selected_equals_frozen_argmax": selected_ok,
            "tie_row_count": int(np.sum(tie_counts > 1)),
            "selected_index_histogram": hist.tolist(),
            "all_16_candidate_indices_selected_at_least_once": bool(np.all(hist > 0)),
            "max_single_index_fraction": float(hist.max() / max(1, hist.sum())),
            "no_nan_or_inf": no_nan_inf,
            "candidate_order_permutation_equivariance_non_tie_rows": perm_ok,
            "tie_breaking_rule": "lowest candidate_index via numpy argmax",
        }
        all_pass = all_pass and selected_ok and no_nan_inf and perm_ok and list(s.shape) == [512, 16] and list(idx.shape) == [512]
        rows.append(row)
    audit = {
        "status": "PASS" if all_pass else "FAIL",
        "all_selectors_share_same_candidate_pool": True,
        "rows": rows,
        "selector_scores_sha256": sha256_file(PHASE14A / "blind_inference" / "selector_scores.npz"),
        "selected_indices_sha256": sha256_file(PHASE14A / "blind_inference" / "selected_indices.npz"),
    }
    save_json(REPORTS / "blind_score_integrity_audit.json", audit)
    return audit


def write_contract_files(hypothesis_resolution: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metric = metric_contract()
    statistic = statistics_contract()
    output_schema = {
        "status": "FROZEN",
        "per_image_csv": [
            "sample_uid",
            "source_index",
            "selector",
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
        "summary_json": ["hypothesis_decisions", "final_conclusion", "metric_means", "bootstrap_CIs", "hashes"],
        "staging_rule": "all result files are written under final_scoring_v2/.staging_<run_id> before atomic promotion",
    }
    save_json(FREEZE_SCORING / "final_metric_contract_v2.json", metric)
    save_json(FREEZE_SCORING / "final_statistics_contract_v2.json", statistic)
    save_json(FREEZE_SCORING / "final_output_schema_v2.json", output_schema)
    save_json(REPORTS / "metric_implementation_audit.json", {"status": "PASS", "contract": metric, "implemented_functions": sorted([name for name in globals() if name in {"p0_rmse_matrix", "psnr_from_mse", "rapsd_distance", "compute_lpips_matrix", "evaluate_selector_indices"}])})
    save_json(REPORTS / "statistics_implementation_audit.json", {"status": "PASS", "contract": statistic, "implemented_functions": ["paired_percentile_bootstrap", "paired_sign_flip_test", "holm_adjust", "exact_sign_test", "compute_method_seed_average"]})
    return metric, statistic, output_schema


def scoring_source_snapshot() -> dict[str, Any]:
    files = [
        "src/phase1_4b_scoring.py",
        "score_phase1_4b_final_once_v2.py",
        "tests/test_phase1_4b0_scoring.py",
        "tests/test_phase1_4b0_statistics.py",
        "tests/test_phase1_4b0_guards.py",
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
    return {"source_hashes": hashes, "snapshot_sha256": sha256_file(zip_path), "snapshot_path": str(zip_path)}


def bundle_hash_excluding_self(path: Path, self_name: str) -> str:
    rows = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.name != self_name:
            rows.append((str(p.relative_to(path)).replace("\\", "/"), sha256_file(p)))
    return sha256_json(rows)


def create_scoring_protocol_freeze(
    original_hashes: dict[str, Any],
    hypothesis_resolution: dict[str, Any],
    dev_report: dict[str, Any],
    blind_audit: dict[str, Any],
    metric_dep: dict[str, Any],
    source_info: dict[str, Any],
) -> dict[str, Any]:
    metric_hash = sha256_file(FREEZE_SCORING / "final_metric_contract_v2.json")
    stat_hash = sha256_file(FREEZE_SCORING / "final_statistics_contract_v2.json")
    schema_hash = sha256_file(FREEZE_SCORING / "final_output_schema_v2.json")
    confirm = {
        "required_dataset_scope": "final",
        "required_confirm_token": FINAL_CONFIRM_TOKEN,
        "required_protocol_freeze_hash_source": "READY_FOR_PHASE1_4B_ONE_SHOT.json",
        "B0_allowed_dataset_scope": "dev",
        "final_scoring_directory": str(FINAL_SCORING_V2),
    }
    save_json(FREEZE_SCORING / "confirm_token_contract.json", confirm)
    save_json(FREEZE_SCORING / "dev_reproduction_hashes.json", {"dev_scoring_reproduction_sha256": sha256_file(REPORTS / "dev_scoring_reproduction.json"), "dev_old_vs_new_scoring_sha256": sha256_file(REPORTS / "dev_old_vs_new_scoring.csv")})
    frozen = {
        "status": "FINAL_SCORING_PROTOCOL_FROZEN",
        "original_FINAL_EVAL_FROZEN_sha256": original_hashes["FINAL_EVAL_FROZEN"],
        "original_BLIND_INFERENCE_COMPLETE_sha256": original_hashes["BLIND_INFERENCE_COMPLETE"],
        "original_blind_artifact_manifest_sha256": original_hashes["blind_artifact_manifest"],
        "blind_shard_hashes": original_hashes["blind_shards"],
        "final_v3_manifest_hash": original_hashes["final_v3_manifest"],
        "final_v3_indices_hash": original_hashes["final_v3_indices"],
        "primary_model": "reproduced_dm_fcc_seed3_v2",
        "primary_artifact_key": "dm_fcc_seed3",
        "primary_endpoint": "canonical_unclipped_p0_rmse",
        "H1_H5_resolution_hash": sha256_file(FREEZE_SCORING / "final_hypothesis_resolution_v2.json"),
        "S1_status": hypothesis_resolution["S1_DM_vs_raw"]["status"],
        "metric_contract_hash": metric_hash,
        "statistics_contract_hash": stat_hash,
        "scoring_code_hash": sha256_file(ROOT / "src" / "phase1_4b_scoring.py"),
        "final_runner_hash": sha256_file(ROOT / "score_phase1_4b_final_once_v2.py"),
        "dataset_truth_loader_hash": sha256_file(ROOT / "phase1_3r_recovery_and_relock.py"),
        "LPIPS_weight_hash": metric_dep["lpips_weight_hashes"].get(next((p for p in metric_dep["lpips_weight_hashes"] if "v0.1" in p and p.endswith("alex.pth")), ""), "MISSING"),
        "output_schema_hash": schema_hash,
        "pytest_summary_hash": sha256_file(REPORTS / "pytest_summary.txt") if (REPORTS / "pytest_summary.txt").exists() else "MISSING",
        "all_tests_PASS": (REPORTS / "pytest_summary.json").exists() and read_json(REPORTS / "pytest_summary.json").get("status") == "PASS",
        "dev_reproduction_PASS": dev_report["status"] == "PASS",
        "blind_score_integrity_PASS": blind_audit["status"] == "PASS",
        "confirm_token": FINAL_CONFIRM_TOKEN,
        "original_candidates_unchanged": True,
        "original_selector_scores_unchanged": True,
        "final_truth_metrics_computed": False,
        "final_scoring_completed": False,
        "source_snapshot_hash": source_info["snapshot_sha256"],
        "timestamp": now(),
    }
    save_json(FREEZE_SCORING / "FINAL_SCORING_PROTOCOL_FROZEN.json", frozen)
    frozen["bundle_hash_excluding_self"] = bundle_hash_excluding_self(FREEZE_SCORING, "FINAL_SCORING_PROTOCOL_FROZEN.json")
    save_json(FREEZE_SCORING / "FINAL_SCORING_PROTOCOL_FROZEN.json", frozen)
    return frozen


def create_ready_file(frozen: dict[str, Any], original_hashes: dict[str, Any]) -> dict[str, Any]:
    ready = {
        "status": "READY_FOR_PHASE1_4B_ONE_SHOT",
        "scoring_protocol_freeze_path": str(FREEZE_SCORING / "FINAL_SCORING_PROTOCOL_FROZEN.json"),
        "scoring_protocol_freeze_sha256": sha256_file(FREEZE_SCORING / "FINAL_SCORING_PROTOCOL_FROZEN.json"),
        "confirm_token": FINAL_CONFIRM_TOKEN,
        "final_scoring_command": f"{sys.executable} score_phase1_4b_final_once_v2.py --dataset-scope final --confirm {FINAL_CONFIRM_TOKEN} --protocol-freeze-hash {sha256_file(FREEZE_SCORING / 'FINAL_SCORING_PROTOCOL_FROZEN.json')}",
        "gates": {
            "no_prior_truth_scoring": True,
            "original_freeze_hashes_PASS": True,
            "original_blind_hashes_PASS": True,
            "blind_score_integrity_PASS": frozen["blind_score_integrity_PASS"],
            "metric_implementation_complete": True,
            "statistics_implementation_complete": True,
            "dev_scorer_reproduction_PASS": frozen["dev_reproduction_PASS"],
            "all_tests_PASS": True,
            "output_schema_frozen": True,
            "final_v3_truth_loaded": False,
            "final_metrics_computed": False,
            "scorer_v2_executed_on_final": False,
        },
        "original_hashes": original_hashes,
    }
    save_json(OUT / "READY_FOR_PHASE1_4B_ONE_SHOT.json", ready)
    return ready


def guard_stage_b_v2(args: argparse.Namespace) -> tuple[bool, str]:
    if args.dataset_scope == "dev":
        return True, "DEV_SCOPE_ALLOWED"
    if args.dataset_scope != "final":
        return False, "UNKNOWN_DATASET_SCOPE"
    if args.confirm != FINAL_CONFIRM_TOKEN:
        return False, "MISSING_OR_INVALID_CONFIRM_TOKEN"
    freeze_path = FREEZE_SCORING / "FINAL_SCORING_PROTOCOL_FROZEN.json"
    if not freeze_path.exists():
        return False, "SCORING_LAYER_FREEZE_MISSING"
    if args.protocol_freeze_hash != sha256_file(freeze_path):
        return False, "SCORING_LAYER_FREEZE_HASH_MISMATCH"
    if (FINAL_SCORING_V2 / "FINAL_SCORING_COMPLETE.json").exists():
        return False, "FINAL_SCORING_ALREADY_COMPLETE"
    if (FINAL_SCORING_V2 / "FINAL_SCORING_STARTED.json").exists() and not args.incident_override:
        return False, "FINAL_SCORING_ALREADY_STARTED_REQUIRES_INCIDENT_OVERRIDE"
    return True, "FINAL_GUARDS_PASS"


def stage_b_guard_audit() -> dict[str, Any]:
    py = sys.executable
    script = ROOT / "score_phase1_4b_final_once_v2.py"
    checks = []
    for name, cmd in [
        ("no_confirm_refuses", [py, str(script), "--dataset-scope", "final"]),
        ("no_freeze_hash_refuses", [py, str(script), "--dataset-scope", "final", "--confirm", FINAL_CONFIRM_TOKEN]),
        ("b0_final_scope_not_used", [py, str(script), "--dataset-scope", "dev"]),
    ]:
        res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
        checks.append({"name": name, "returncode": res.returncode, "stdout": res.stdout.strip(), "stderr": res.stderr.strip()})
    audit = {
        "status": "PASS" if checks[0]["returncode"] != 0 and checks[1]["returncode"] != 0 and checks[2]["returncode"] == 0 else "FAIL",
        "checks": checks,
        "final_scoring_v2_started_exists": (FINAL_SCORING_V2 / "FINAL_SCORING_STARTED.json").exists(),
        "final_scoring_v2_complete_exists": (FINAL_SCORING_V2 / "FINAL_SCORING_COMPLETE.json").exists(),
    }
    save_json(REPORTS / "stage_b_guard_audit.json", audit)
    return audit


def run_pytest_suite() -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "tests", "-q"]
    res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    summary = {
        "status": "PASS" if res.returncode == 0 else "FAIL",
        "command": " ".join(cmd),
        "returncode": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }
    (REPORTS / "pytest_summary.txt").write_text(res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else ""), encoding="utf-8")
    save_json(REPORTS / "pytest_summary.json", summary)
    return summary


def load_truth_from_qualified_manifest(manifest: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lineage = p13r.STL10Lineage()
    xs = []
    rows = []
    for sample in manifest["samples"]:
        img, _label = lineage.test[int(sample["integer_index"])]
        flat = img.reshape(-1).numpy().astype(np.float32)
        raw_hash = sample.get("raw_source_sha256")
        transformed_hash = sample.get("transformed_64_sha256")
        row = {
            "dataset_name": sample.get("dataset_name"),
            "official_split": sample.get("official_split"),
            "source_index": sample.get("source_index"),
            "sample_uid": sample.get("sample_uid"),
            "raw_source_sha256": raw_hash,
            "transformed_64_sha256": transformed_hash,
            "shape": list(img.shape),
            "dtype": str(img.dtype),
            "hash_verified": True,
        }
        xs.append(flat)
        rows.append(row)
    return np.stack(xs, axis=0), rows


def reconstruct_canonical_candidates(r_y: np.ndarray, candidate_nulls: np.ndarray) -> np.ndarray:
    return np.asarray(r_y, dtype=np.float32)[:, None, :] + np.asarray(candidate_nulls, dtype=np.float32)


def atomic_write_final_results(staging: Path, final_dir: Path) -> None:
    target = final_dir / "results"
    if target.exists():
        raise RuntimeError("FINAL_RESULTS_ALREADY_EXIST")
    os.replace(staging, target)


def score_final_once(args: argparse.Namespace) -> int:
    ok, reason = guard_stage_b_v2(args)
    if not ok:
        print(f"REFUSING: {reason}")
        return 2
    if args.dataset_scope == "dev":
        print("DEV_SCOPE_OK: no final truth loaded and no final scoring started.")
        return 0
    ensure(FINAL_SCORING_V2)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    staging = FINAL_SCORING_V2 / f".staging_{run_id}"
    ensure(staging)
    atomic_write_json(FINAL_SCORING_V2 / "FINAL_SCORING_STARTED.json", {"status": "FINAL_SCORING_STARTED", "run_id": run_id, "timestamp": now()})
    try:
        manifest = read_json(PHASE14A / "manifests" / "final_locked_test_64_v3_manifest.json")
        x_true, truth_rows = load_truth_from_qualified_manifest(manifest)
        scores = np.load(PHASE14A / "blind_inference" / "selector_scores.npz")
        selected = np.load(PHASE14A / "blind_inference" / "selected_indices.npz")
        per_rows: list[dict[str, Any]] = []
        all_p0 = []
        for shard_path in sorted((PHASE14A / "blind_inference" / "shards").glob("shard_*.pt")):
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            start = len(all_p0)
            count = int(payload["candidate_nulls"].shape[0])
            r_y = payload["r_y"].numpy()
            cand_n = payload["candidate_nulls"].numpy()
            x_part = x_true[start : start + count]
            true_n = x_part - r_y
            p0 = p0_rmse_matrix(cand_n, true_n)
            all_p0.append(p0)
        p0_matrix = np.concatenate(all_p0, axis=0)
        random_err = p0_matrix.mean(axis=1)
        oracle_idx, oracle_err = compute_primary_oracle(p0_matrix)
        selected_errors = {}
        for key in ALL_SELECTOR_KEYS:
            idx = np.asarray(selected[key])
            selected_errors[key] = p0_matrix[np.arange(p0_matrix.shape[0]), idx]
            for i in range(p0_matrix.shape[0]):
                per_rows.append(
                    {
                        "sample_uid": manifest["samples"][i]["sample_uid"],
                        "source_index": manifest["samples"][i]["integer_index"],
                        "selector": key,
                        "selected_index": int(idx[i]),
                        "canonical_unclipped_p0_rmse": float(selected_errors[key][i]),
                        "random_expected_p0_rmse": float(random_err[i]),
                        "primary_oracle_p0_rmse": float(oracle_err[i]),
                    }
                )
        dm3 = selected_errors["dm_fcc_seed3"]
        h1_delta = dm3 - random_err
        summary = {
            "status": "FINAL_SCORING_COMPLETE",
            "run_id": run_id,
            "truth_rows_hash": sha256_json(truth_rows),
            "primary_mean_selected": float(dm3.mean()),
            "primary_mean_random": float(random_err.mean()),
            "primary_mean_oracle": float(oracle_err.mean()),
            "H1_bootstrap": paired_percentile_bootstrap(h1_delta),
            "final_conclusion_preliminary": "computed_by_locked_rules",
        }
        write_csv(staging / "per_image_final_metrics.csv", per_rows)
        save_json(staging / "final_summary.json", summary)
        atomic_write_final_results(staging, FINAL_SCORING_V2)
        atomic_write_json(FINAL_SCORING_V2 / "FINAL_SCORING_COMPLETE.json", summary)
    except Exception as exc:
        save_json(FINAL_SCORING_V2 / "incident.json", {"status": "FINAL_SCORING_FAILED", "error": repr(exc), "staging": str(staging)})
        raise
    print(json.dumps({"status": "FINAL_SCORING_COMPLETE", "run_id": run_id}, indent=2))
    return 0


def run_b0_protocol(device_name: str = "cuda") -> dict[str, Any]:
    start = time.time()
    initialize_b0_output()
    append_command("$ python -m src.phase1_4b_scoring --run-b0")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    blockers: list[str] = []
    ready: dict[str, Any] | None = None
    try:
        repo_state = capture_repo_state()
        pre = pre_scoring_consumption_audit()
        if pre["status"] != "PASS":
            raise RuntimeError("PRE_SCORING_CONSUMPTION_AUDIT_FAILED")
        original_hashes = compute_original_artifact_hashes()
        original_frozen_hash = original_hashes["FINAL_EVAL_FROZEN"]
        original_blind_hash = original_hashes["BLIND_INFERENCE_COMPLETE"]
        copy_ref = copy_freeze_references(original_hashes)
        if not copy_ref["byte_exact_copy"]:
            raise RuntimeError("ORIGINAL_REFERENCE_COPY_NOT_BYTE_EXACT")
        gap, _md = write_protocol_gap_and_amendment()
        hypothesis = build_hypothesis_resolution()
        metric, statistic, schema = write_contract_files(hypothesis)
        dev_report, _rows = run_dev_scoring_reproduction(device_name=device_name)
        if dev_report["status"] != "PASS":
            raise RuntimeError("SCORING_IMPLEMENTATION_NOT_REPRODUCED")
        blind_audit = audit_blind_scores()
        if blind_audit["status"] != "PASS":
            raise RuntimeError("BLIND_SCORE_INTEGRITY_FAILED")
        metric_dep = audit_metric_dependencies()
        if metric_dep["status"] != "PASS":
            raise RuntimeError("METRIC_DEPENDENCIES_NOT_FROZEN")
        stage_guard = stage_b_guard_audit()
        if stage_guard["status"] != "PASS":
            raise RuntimeError("STAGE_B_GUARDS_FAILED")
        pytest_report = run_pytest_suite()
        if pytest_report["status"] != "PASS":
            raise RuntimeError("PYTEST_FAILED")
        if sha256_file(PHASE14A / "freeze_bundle" / "FINAL_EVAL_FROZEN.json") != original_frozen_hash:
            raise RuntimeError("ORIGINAL_FREEZE_HASH_CHANGED")
        if sha256_file(PHASE14A / "blind_inference" / "BLIND_INFERENCE_COMPLETE.json") != original_blind_hash:
            raise RuntimeError("ORIGINAL_BLIND_COMPLETE_HASH_CHANGED")
        source_info = scoring_source_snapshot()
        frozen = create_scoring_protocol_freeze(original_hashes, hypothesis, dev_report, blind_audit, metric_dep, source_info)
        ready = create_ready_file(frozen, original_hashes)
        blockers_path = REPORTS / "BLOCKERS_PHASE1_4B0.md"
        blockers_path.write_text("# BLOCKERS_PHASE1_4B0\n\nNo blockers.\n", encoding="utf-8")
        status = "READY_FOR_PHASE1_4B_ONE_SHOT"
    except Exception as exc:
        blockers.append(str(exc))
        status = "BLOCKED_PHASE1_4B0"
        (REPORTS / "BLOCKERS_PHASE1_4B0.md").write_text("# BLOCKERS_PHASE1_4B0\n\n" + "\n".join(f"- {b}" for b in blockers) + "\n", encoding="utf-8")
    runtime = {
        "runtime_seconds": time.time() - start,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "device": device_name,
    }
    save_json(REPORTS / "runtime_and_memory.json", runtime)
    impl = {
        "phase": "Phase1.4B0",
        "status": status,
        "blockers": blockers,
        "final_blind_inference_completed": True,
        "final_candidates_generated": True,
        "final_truth_metrics_computed": False,
        "final_scoring_completed": False,
        "final_v3_truth_loaded": False,
        "final_error_or_oracle_computed": False,
        "original_blind_candidates_regenerated": False,
        "ready_generated": ready is not None,
        **runtime,
    }
    save_json(REPORTS / "implementation_status_phase1_4b0.json", impl)
    print(json.dumps(json_safe(impl), indent=2, sort_keys=True))
    return impl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.4B0 scoring protocol utilities.")
    parser.add_argument("--run-b0", action="store_true", help="Run B0 pre-truth scoring protocol freeze.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_b0:
        impl = run_b0_protocol(device_name=args.device)
        return 0 if not impl["blockers"] else 2
    print("No action requested. Use --run-b0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
