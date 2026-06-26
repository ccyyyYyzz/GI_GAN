from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src import phase69A_gauge_gan_signal_diagnostic as p69a
from src.measurement import create_fixed_measurement_matrix
from src.phase2_witness import (
    atomic_write_json,
    ensure_dir,
    json_safe,
    make_witness_rows,
    repo_state,
    sha256_file,
    sha256_json,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "compatibility" / "phase2_addon_locked_test.yaml"
CONFIRM_TOKEN = "PHASE2_ADDON_WITNESS_LOCKED_ONE_SHOT_SCORING"
FINAL_V4_INDICES = (
    ROOT
    / "outputs"
    / "compatibility"
    / "phase1_4ir_incident_recovery"
    / "manifests"
    / "final_locked_test_64_v4_indices.npy"
)


class Phase2LockedProtocolError(RuntimeError):
    """Hard-fail exception for Phase 2 locked-test preflight/protocol errors."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise Phase2LockedProtocolError(f"CONFIG_MUST_BE_MAPPING:{path}")
    return obj


def sha256_int64(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha256_float32(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def locked_sample_uid(cache_name: str, split_name: str, source_index: int, row: int) -> str:
    return (
        f"phase2_locked:{cache_name}:split:{split_name}:"
        f"source_index:{int(source_index)}:row:{int(row)}"
    )


def stable_index_order(indices: np.ndarray, salt: str) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    keyed = [
        (
            hashlib.sha256(f"{salt}:{int(idx)}".encode("utf-8")).hexdigest(),
            int(idx),
        )
        for idx in arr.tolist()
    ]
    keyed.sort()
    return np.asarray([idx for _key, idx in keyed], dtype=np.int64)


def interval_overlap(a_start: int, a_count: int, b_start: int, b_count: int) -> bool:
    a0, a1 = int(a_start), int(a_start) + int(a_count)
    b0, b1 = int(b_start), int(b_start) + int(b_count)
    return max(a0, b0) < min(a1, b1)


def validate_development_exclusion(
    *,
    locked_offset: int,
    locked_count: int,
    exclusions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    overlaps = []
    for item in exclusions:
        name = str(item.get("name", "unnamed"))
        offset = int(item["train_unlabeled_offset"])
        count = int(item["sample_count"])
        overlap = interval_overlap(locked_offset, locked_count, offset, count)
        row = {
            "name": name,
            "train_unlabeled_offset": offset,
            "sample_count": count,
            "overlap_with_locked_slice": bool(overlap),
        }
        rows.append(row)
        if overlap:
            overlaps.append(row)
    return {
        "status": "PASS" if not overlaps else "FAIL",
        "locked_offset": int(locked_offset),
        "locked_count": int(locked_count),
        "checked_exclusions": rows,
    }


def build_locked_split_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    split = dict(config["split"])
    source = str(split.get("source"))
    count = int(split["sample_count"])
    if count <= 0:
        raise Phase2LockedProtocolError(f"LOCKED_SAMPLE_COUNT_MUST_BE_POSITIVE:{count}")
    excluded_final_v4: np.ndarray | None = None
    if source == "STL10 train+unlabeled":
        offset = int(split["train_unlabeled_offset"])
        pool = np.load(p69a.SPLIT_TRAIN).astype(np.int64)
        if offset < 0 or offset + count > int(pool.shape[0]):
            raise Phase2LockedProtocolError(f"LOCKED_SPLIT_EXCEEDS_POOL:{offset}:{count}:{pool.shape[0]}")
        indices = np.asarray(pool[offset : offset + count], dtype=np.int64)
        pool_path = str(p69a.SPLIT_TRAIN)
        selection = {
            "mode": "contiguous_train_unlabeled_slice",
            "train_unlabeled_offset": offset,
        }
        exclusion = validate_development_exclusion(
            locked_offset=offset,
            locked_count=count,
            exclusions=config.get("development_exclusions", []),
        )
        if exclusion["status"] != "PASS":
            raise Phase2LockedProtocolError(f"LOCKED_SPLIT_OVERLAPS_DEVELOPMENT:{exclusion}")
        final_boundary = {
            "final_v4_consumed": True,
            "final_v4_physical_split": "STL10 test",
            "this_locked_split": "STL10 train+unlabeled holdout slice",
            "final_v4_used_for_method_selection": False,
        }
    elif source == "STL10 test":
        total = int(split.get("test_pool_count", 8000))
        offset = int(split.get("test_selection_offset", 0))
        salt = str(split.get("selection_salt", "PHASE2_LOCKED_TEST_SELECTION_V1"))
        pool = np.arange(total, dtype=np.int64)
        exclude_path = Path(str(split.get("exclude_indices_npy", FINAL_V4_INDICES)))
        if not exclude_path.exists():
            raise Phase2LockedProtocolError(f"FINAL_V4_EXCLUSION_INDICES_MISSING:{exclude_path}")
        excluded_final_v4 = np.load(exclude_path).astype(np.int64)
        candidates = np.setdiff1d(pool, excluded_final_v4, assume_unique=False)
        ordered = stable_index_order(candidates, salt)
        if offset < 0 or offset + count > int(ordered.shape[0]):
            raise Phase2LockedProtocolError(f"LOCKED_TEST_SPLIT_EXCEEDS_POOL:{offset}:{count}:{ordered.shape[0]}")
        indices = np.asarray(ordered[offset : offset + count], dtype=np.int64)
        pool_path = f"synthetic_range_0_{total - 1}_excluding_{exclude_path}"
        selection = {
            "mode": "stable_hash_ordered_stl10_test_minus_final_v4",
            "test_selection_offset": offset,
            "selection_salt": salt,
            "excluded_final_v4_indices_path": str(exclude_path),
            "excluded_final_v4_count": int(excluded_final_v4.shape[0]),
            "candidate_pool_count_after_exclusion": int(candidates.shape[0]),
        }
        exclusion = {
            "status": "PASS",
            "note": "development_exclusions are train+unlabeled slices; locked source is STL10 test minus final-v4.",
            "checked_exclusions": [],
        }
        final_boundary = {
            "final_v4_consumed": True,
            "final_v4_physical_split": "STL10 test",
            "this_locked_split": "STL10 test excluding final-v4 source indices",
            "final_v4_excluded_count": int(excluded_final_v4.shape[0]),
            "numeric_overlap_with_final_v4_source_indices": int(np.intersect1d(indices, excluded_final_v4).shape[0]),
            "final_v4_used_for_method_selection": False,
        }
        if final_boundary["numeric_overlap_with_final_v4_source_indices"] != 0:
            raise Phase2LockedProtocolError(f"LOCKED_TEST_OVERLAPS_FINAL_V4:{final_boundary}")
    else:
        raise Phase2LockedProtocolError(f"UNSUPPORTED_LOCKED_SPLIT_SOURCE:{source}")
    cache_name = str(split.get("name", "phase2_addon_locked_test"))
    uids = [locked_sample_uid(cache_name, "locked_test", int(idx), i) for i, idx in enumerate(indices)]
    if len(set(uids)) != len(uids):
        raise Phase2LockedProtocolError("LOCKED_SAMPLE_UIDS_NOT_UNIQUE")
    if len(set(indices.tolist())) != int(indices.shape[0]):
        raise Phase2LockedProtocolError("LOCKED_SOURCE_INDICES_NOT_UNIQUE")
    rows = [
        {
            "sample_uid": uids[i],
            "source": split["source"],
            "source_index": int(indices[i]),
            "row": int(i),
        }
        for i in range(int(indices.shape[0]))
    ]
    return {
        "status": "PASS",
        "split_name": cache_name,
        "source": source,
        "purpose": "Phase 2 locked add-on witness test identity manifest; no image truth loaded.",
        "selection": selection,
        "sample_count": count,
        "pool_path": pool_path,
        "pool_length": int(pool.shape[0]),
        "pool_sha256_unsorted": p69a.sha256_np(pool),
        "pool_sha256_sorted": p69a.sha256_np(pool, sort_int64=True),
        "indices_sha256": sha256_int64(indices),
        "sample_uid_sha256": hashlib.sha256("\n".join(uids).encode("utf-8")).hexdigest(),
        "first_three_uids": uids[:3],
        "development_exclusion_audit": exclusion,
        "final_v4_boundary": final_boundary,
        "rows": rows,
    }


def build_operator_witness_freeze(config: Mapping[str, Any]) -> dict[str, Any]:
    op = dict(config["context_operator"])
    img_size = int(op.get("img_size", 64))
    n = img_size * img_size
    A, metadata = create_fixed_measurement_matrix(
        img_size=img_size,
        sampling_ratio=float(op.get("sampling_ratio", 0.05)),
        pattern_type=str(op.get("pattern_type", "rademacher")),
        device="cpu",
        seed=int(op["seed"]),
        matrix_normalization=str(op.get("matrix_normalization", "legacy_sqrt_m")),
        return_metadata=True,
    )
    A_np = A.detach().cpu().numpy().astype(np.float32)
    witness = dict(config["witness"])
    budgets = [int(v) for v in witness.get("budgets", [24, 32, 64])]
    if not budgets or max(budgets) <= 0:
        raise Phase2LockedProtocolError(f"INVALID_WITNESS_BUDGETS:{budgets}")
    random_max = make_witness_rows(
        str(witness.get("random_witness", "rademacher")),
        max(budgets),
        n,
        seed=int(witness.get("random_seed", 0)),
    )
    fixed_max = make_witness_rows(
        str(witness.get("fixed_witness", "dct2_low_frequency")),
        max(budgets),
        n,
        seed=int(witness.get("fixed_seed", 0)),
    )
    library = make_witness_rows(
        str(witness.get("adaptive_library", "dct2_low_frequency")),
        int(witness.get("adaptive_library_size", 256)),
        n,
        seed=int(witness.get("adaptive_library_seed", 0)),
    )
    return {
        "status": "PASS",
        "context_operator": {
            "role": "candidate generation context only",
            "pattern_type": str(op.get("pattern_type", "rademacher")),
            "sampling_ratio": float(op.get("sampling_ratio", 0.05)),
            "seed": int(op["seed"]),
            "img_size": img_size,
            "n": n,
            "m": int(A_np.shape[0]),
            "matrix_normalization": str(op.get("matrix_normalization", "legacy_sqrt_m")),
            "noise_std": float(op.get("noise_std", 0.0)),
            "A_sha256_float32": sha256_float32(A_np),
            "metadata": metadata,
        },
        "witness": {
            "role": "selection only after candidate generation",
            "seen_by_generator": False,
            "used_for_training_or_early_stopping": False,
            "budgets": budgets,
            "primary_budget": int(config["locked_test"]["primary_budget"]),
            "random_witness_kind": str(witness.get("random_witness", "rademacher")),
            "random_seed": int(witness.get("random_seed", 0)),
            "random_rows_max_budget_sha256": sha256_float32(random_max),
            "fixed_witness_kind": str(witness.get("fixed_witness", "dct2_low_frequency")),
            "fixed_rows_max_budget_sha256": sha256_float32(fixed_max),
            "adaptive_library_kind": str(witness.get("adaptive_library", "dct2_low_frequency")),
            "adaptive_library_seed": int(witness.get("adaptive_library_seed", 0)),
            "adaptive_library_size": int(witness.get("adaptive_library_size", 256)),
            "adaptive_library_sha256": sha256_float32(library),
        },
    }


def preregistration_text(config: Mapping[str, Any], split_manifest: Mapping[str, Any], freeze: Mapping[str, Any]) -> str:
    locked = config["locked_test"]
    primary_budget = int(locked["primary_budget"])
    rows = [
        "# Phase 2 Add-On Witnessed Candidate Selection Locked Test Protocol",
        "",
        "Status: `READY_FOR_LOCKED_TEST_SCORING_NOT_STARTED`.",
        "",
        "This protocol freezes the add-on witnessed candidate-selection test. It does not load image truth, generate candidates, or compute locked metrics.",
        "",
        "## Primary Question",
        "",
        "Can generator-unseen add-on witness measurements select a single context-feasible generated candidate with lower current-sample P0-RMSE than posterior mean and random candidate expectation?",
        "",
        "## Frozen Identity",
        "",
        f"- Split: `{split_manifest['split_name']}`",
        f"- Source: `{split_manifest['source']}`",
        f"- Sample count: `{split_manifest['sample_count']}`",
        f"- Source indices SHA256: `{split_manifest['indices_sha256']}`",
        f"- Sample UID SHA256: `{split_manifest['sample_uid_sha256']}`",
        "",
        "## Frozen Operator And Witness",
        "",
        f"- Context operator hash: `{freeze['context_operator']['A_sha256_float32']}`",
        f"- Context rows: `{freeze['context_operator']['m']}`",
        f"- Primary witness method: `adaptive_witness_b{primary_budget}` using `{freeze['witness']['adaptive_library_kind']}` library",
        f"- Witness budgets: `{freeze['witness']['budgets']}`",
        f"- Adaptive library hash: `{freeze['witness']['adaptive_library_sha256']}`",
        "",
        "## Primary Endpoint",
        "",
        "- Canonical unclipped P0-RMSE of the selected single candidate.",
        "- The main table must include random expectation, posterior mean, scalar, sum-image, scratch dual, frozen compatibility selector, random witness, fixed witness, adaptive witness, compatibility-prefiltered witness, and oracle.",
        "",
        "## Primary Success Conditions",
        "",
        "All must pass:",
        "",
        "1. adaptive witness b64 minus posterior mean has mean < 0.",
        "2. paired image-level bootstrap 95% CI upper bound for adaptive b64 minus posterior mean is < 0.",
        "3. adaptive witness b64 minus random expectation has mean < 0 and 95% CI upper bound < 0.",
        "4. adaptive witness b64 captures at least 50% aggregate oracle headroom.",
        "5. canonical context RelMeasErr max remains below 1e-5.",
        "6. UID/hash/schema/leakage audits pass.",
        "",
        "## Secondary Questions",
        "",
        "- Is adaptive witness more efficient than random witness and fixed low-frequency witness?",
        "- Does compatibility prefilter plus witness beat adaptive witness alone?",
        "- Do LPIPS/RAPSD/SSIM move consistently with P0-RMSE?",
        "- Fixed-total benefit is not tested by this locked add-on protocol.",
        "",
        "## Boundary",
        "",
        "Witness rows add observed directions. They do not certify the complete remaining null space, and perceptual improvement must not be interpreted as proof of true unmeasured detail.",
        "",
    ]
    return "\n".join(rows)


def build_protocol_payload(config_path: Path, config: Mapping[str, Any], split_manifest: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "status": "PHASE2_ADDON_WITNESS_LOCKED_PROTOCOL_FROZEN",
        "created_utc": now_utc(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "run_name": config.get("run_name"),
        "candidate_k": int(config.get("candidate_k", 16)),
        "candidate_seed": int(config.get("candidate_seed")),
        "checkpoint": str(config.get("checkpoint")),
        "checkpoint_hash": sha256_file(Path(config["checkpoint"])),
        "split": {
            "name": split_manifest["split_name"],
            "source": split_manifest["source"],
            "sample_count": split_manifest["sample_count"],
            "indices_sha256": split_manifest["indices_sha256"],
            "sample_uid_sha256": split_manifest["sample_uid_sha256"],
        },
        "operator": freeze["context_operator"],
        "witness": freeze["witness"],
        "statistics": config["statistics"],
        "locked_test": config["locked_test"],
        "classification_rules": config["classification_rules"],
        "truth_loaded": False,
        "candidates_generated": False,
        "locked_metrics_computed": False,
        "final_v4_used_for_method_selection": False,
    }
    payload["protocol_hash"] = sha256_json({k: v for k, v in payload.items() if k != "protocol_hash"})
    return payload


def classify_locked_addon_result(
    summaries: Mapping[str, Mapping[str, Any]],
    feasibility: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    locked = dict(config["locked_test"])
    method = str(locked.get("primary_method", f"adaptive_witness_b{int(locked.get('primary_budget', 64))}"))
    rules = dict(config.get("classification_rules", {}))
    if method not in summaries:
        return {
            "classification": rules.get("invalid", "WITNESS_EVALUATION_INVALID"),
            "status": "FAIL",
            "reason": f"PRIMARY_METHOD_MISSING:{method}",
        }
    primary = summaries[method]
    threshold = float(locked.get("relmeaserr_max_threshold", 1e-5))
    min_gain = float(locked.get("min_oracle_gain_fraction", 0.50))
    cond = {
        "adaptive_vs_posterior_mean_negative": bool(primary["delta_vs_posterior_mean"] < 0.0),
        "adaptive_vs_posterior_ci_upper_negative": bool(primary["bootstrap_vs_posterior"]["ci_upper"] < 0.0),
        "adaptive_vs_random_mean_negative": bool(primary["delta_vs_random_mean"] < 0.0),
        "adaptive_vs_random_ci_upper_negative": bool(primary["bootstrap_vs_random"]["ci_upper"] < 0.0),
        "oracle_gain_fraction_at_least_threshold": bool(
            primary.get("oracle_gain_fraction_aggregate") is not None
            and float(primary["oracle_gain_fraction_aggregate"]) >= min_gain
        ),
        "context_relmeaserr_below_threshold": bool(
            str(feasibility.get("status")) == "PASS"
            and float(feasibility.get("canonical_relmeaserr_max", float("inf"))) < threshold
        ),
    }
    if not cond["context_relmeaserr_below_threshold"]:
        classification = rules.get("invalid", "WITNESS_EVALUATION_INVALID")
    elif all(cond.values()):
        classification = rules.get("success", "WITNESS_ADDON_CONFIRMED")
    elif cond["adaptive_vs_posterior_mean_negative"] or cond["adaptive_vs_random_mean_negative"]:
        classification = rules.get("trend", "WITNESS_ADDON_TREND_ONLY")
    else:
        classification = rules.get("no_benefit", "WITNESS_ADDON_NO_BENEFIT")
    return {
        "classification": classification,
        "status": "PASS",
        "primary_method": method,
        "conditions": cond,
        "primary_mean_p0_rmse": primary["mean_p0_rmse"],
        "delta_vs_posterior_mean": primary["delta_vs_posterior_mean"],
        "delta_vs_random_mean": primary["delta_vs_random_mean"],
        "oracle_gain_fraction_aggregate": primary.get("oracle_gain_fraction_aggregate"),
        "canonical_relmeaserr_max": feasibility.get("canonical_relmeaserr_max"),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_locked_ready(config_path: Path, protocol_hash: str) -> tuple[dict[str, Any], Path]:
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/phase2_addon_locked_test/lock_v1"))
    ready_path = output_dir / "READY_FOR_PHASE2_ADDON_WITNESS_LOCKED_SCORING.json"
    if not ready_path.exists():
        raise Phase2LockedProtocolError(f"LOCKED_READY_MISSING:{ready_path}")
    ready = read_json(ready_path)
    if ready.get("status") != "READY_FOR_PHASE2_ADDON_WITNESS_LOCKED_SCORING":
        raise Phase2LockedProtocolError(f"LOCKED_READY_BAD_STATUS:{ready.get('status')}")
    if str(ready.get("protocol_hash")) != str(protocol_hash):
        raise Phase2LockedProtocolError(f"LOCKED_PROTOCOL_HASH_MISMATCH:{ready.get('protocol_hash')}:{protocol_hash}")
    if ready.get("truth_loaded") or ready.get("candidates_generated") or ready.get("locked_metrics_computed"):
        raise Phase2LockedProtocolError("LOCKED_READY_ALREADY_CONSUMED")
    return config, output_dir


def score_locked_once(
    config_path: Path,
    *,
    confirm: str,
    protocol_hash: str,
) -> dict[str, Any]:
    if confirm != CONFIRM_TOKEN:
        raise Phase2LockedProtocolError(f"CONFIRM_TOKEN_REQUIRED:{CONFIRM_TOKEN}")
    config, output_dir = verify_locked_ready(config_path, protocol_hash)
    started = output_dir / "PHASE2_ADDON_WITNESS_LOCKED_SCORING_STARTED.json"
    complete = output_dir / "PHASE2_ADDON_WITNESS_LOCKED_SCORING_COMPLETE.json"
    if complete.exists():
        raise Phase2LockedProtocolError(f"LOCKED_SCORING_ALREADY_COMPLETE:{complete}")
    if started.exists():
        raise Phase2LockedProtocolError(f"LOCKED_SCORING_ALREADY_STARTED_NO_AUTORERUN:{started}")
    atomic_write_json(
        started,
        {
            "status": "PHASE2_ADDON_WITNESS_LOCKED_SCORING_STARTED",
            "created_utc": now_utc(),
            "protocol_hash": protocol_hash,
            "config_sha256": sha256_file(config_path),
        },
    )
    try:
        from src.phase2_fresh_operator import run_fresh_operator_smoke

        summary = run_fresh_operator_smoke(config_path)
        reports = output_dir / "reports"
        summaries = read_json(reports / "method_summaries.json")
        feasibility = read_json(reports / "candidate_feasibility_audit.json")
        classification = classify_locked_addon_result(summaries, feasibility, config)
        result = {
            "status": "PHASE2_ADDON_WITNESS_LOCKED_SCORING_COMPLETE",
            "created_utc": now_utc(),
            "protocol_hash": protocol_hash,
            "summary_status": summary["status"],
            "classification": classification["classification"],
            "classification_report": classification,
            "pilot_summary_sha256": sha256_file(reports / "pilot_summary.json"),
            "method_summaries_sha256": sha256_file(reports / "method_summaries.json"),
            "candidate_feasibility_sha256": sha256_file(reports / "candidate_feasibility_audit.json"),
            "truth_loaded": True,
            "candidates_generated": True,
            "locked_metrics_computed": True,
        }
        atomic_write_json(complete, result)
        return result
    except Exception as exc:
        incident = output_dir / f"locked_scoring_incident_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic_write_json(
            incident,
            {
                "status": "PHASE2_ADDON_WITNESS_LOCKED_SCORING_FAILED",
                "created_utc": now_utc(),
                "protocol_hash": protocol_hash,
                "error": repr(exc),
                "started_marker": str(started),
                "no_autorerun_policy": True,
            },
        )
        raise


def run_locked_preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    output_dir = ROOT / str(config.get("output_dir", "outputs/compatibility/phase2_addon_locked_test/lock_v1"))
    reports = output_dir / "reports"
    freeze_dir = output_dir / "freeze_bundle"
    ensure_dir(reports)
    ensure_dir(freeze_dir)
    shutil.copyfile(config_path, output_dir / "config_used.yaml")
    split_manifest = build_locked_split_manifest(config)
    freeze = build_operator_witness_freeze(config)
    leak = {
        "status": "PASS",
        "truth_loaded": False,
        "candidates_generated": False,
        "locked_metrics_computed": False,
        "final_v4_consumed": True,
        "final_v4_inputs_loaded": False,
        "final_v4_used_for_method_selection": False,
        "witness_seen_by_generator": False,
        "witness_used_for_training_or_early_stopping": False,
        "position_only_join_forbidden": True,
        "primary_key": "qualified sample_uid",
        "repo_state": repo_state(),
    }
    protocol = build_protocol_payload(config_path, config, split_manifest, freeze)
    prereg = preregistration_text(config, split_manifest, freeze)
    write_json(reports / "locked_split_manifest.json", split_manifest)
    write_json(reports / "operator_witness_freeze.json", freeze)
    write_json(reports / "leakage_independence_audit.json", leak)
    write_json(freeze_dir / "PHASE2_ADDON_WITNESS_LOCKED_PROTOCOL_FROZEN.json", protocol)
    (reports / "locked_preregistration.md").write_text(prereg, encoding="utf-8")
    try:
        config_cli = config_path.relative_to(ROOT).as_posix()
    except ValueError:
        config_cli = str(config_path)
    command = (
        "D:\\Anacondar\\anaconda3\\python.exe phase2_locked_test_score_once.py "
        f"--config {config_cli} "
        f"--confirm {CONFIRM_TOKEN} "
        f"--protocol-hash {protocol['protocol_hash']}"
    )
    (reports / "future_one_shot_command_template.txt").write_text(command + "\n", encoding="utf-8")
    hashes = {
        "config_used.yaml": sha256_file(output_dir / "config_used.yaml"),
        "locked_split_manifest.json": sha256_file(reports / "locked_split_manifest.json"),
        "operator_witness_freeze.json": sha256_file(reports / "operator_witness_freeze.json"),
        "leakage_independence_audit.json": sha256_file(reports / "leakage_independence_audit.json"),
        "locked_preregistration.md": sha256_file(reports / "locked_preregistration.md"),
        "protocol_freeze": sha256_file(freeze_dir / "PHASE2_ADDON_WITNESS_LOCKED_PROTOCOL_FROZEN.json"),
    }
    write_json(reports / "preflight_hashes.json", hashes)
    ready = {
        "status": "READY_FOR_PHASE2_ADDON_WITNESS_LOCKED_SCORING",
        "created_utc": now_utc(),
        "protocol_hash": protocol["protocol_hash"],
        "protocol_freeze_sha256": hashes["protocol_freeze"],
        "locked_split_manifest_sha256": hashes["locked_split_manifest.json"],
        "operator_witness_freeze_sha256": hashes["operator_witness_freeze.json"],
        "truth_loaded": False,
        "candidates_generated": False,
        "locked_metrics_computed": False,
        "output_dir": str(output_dir),
        "future_one_shot_command_template": command,
    }
    atomic_write_json(output_dir / "READY_FOR_PHASE2_ADDON_WITNESS_LOCKED_SCORING.json", ready)
    return {
        "status": ready["status"],
        "output_dir": str(output_dir),
        "protocol_hash": protocol["protocol_hash"],
        "ready_sha256": sha256_file(output_dir / "READY_FOR_PHASE2_ADDON_WITNESS_LOCKED_SCORING.json"),
        "hashes": hashes,
    }
