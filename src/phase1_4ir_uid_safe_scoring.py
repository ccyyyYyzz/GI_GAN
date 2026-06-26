from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


K = 16


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


class UIDScoringError(RuntimeError):
    """Hard-fail exception for UID-safe scoring contract violations."""


@dataclass(frozen=True)
class TruthRecord:
    sample_uid: str
    source_index: int
    true_null: np.ndarray
    transformed_64_sha256: str = ""


@dataclass(frozen=True)
class BlindRecord:
    sample_uid: str
    source_index: int
    r_y: np.ndarray
    candidate_nulls: np.ndarray
    transformed_64_sha256: str = ""


def sha256_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_candidate_seed(sample_uid: str, candidate_index: int, salt: str) -> int:
    if not 0 <= int(candidate_index) < K:
        raise UIDScoringError(f"CANDIDATE_INDEX_OUT_OF_RANGE_FOR_SEED: {candidate_index}")
    digest = hashlib.sha256(f"{salt}|{sample_uid}|{int(candidate_index)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF


def p0_rmse_matrix(candidate_nulls: np.ndarray, true_null: np.ndarray) -> np.ndarray:
    cand = np.asarray(candidate_nulls, dtype=np.float64)
    truth = np.asarray(true_null, dtype=np.float64)
    if cand.ndim != 3:
        raise UIDScoringError(f"CANDIDATE_NULLS_MUST_BE_3D: got shape {cand.shape}")
    if truth.ndim != 2:
        raise UIDScoringError(f"TRUE_NULL_MUST_BE_2D: got shape {truth.shape}")
    if cand.shape[0] != truth.shape[0] or cand.shape[2] != truth.shape[1]:
        raise UIDScoringError(f"CANDIDATE_TRUTH_SHAPE_MISMATCH: candidate {cand.shape}, truth {truth.shape}")
    return np.sqrt(np.mean((cand - truth[:, None, :]) ** 2, axis=2))


def uid_map_from_records(records: Sequence[Any], *, uid_attr: str = "sample_uid") -> dict[str, Any]:
    out: dict[str, Any] = {}
    duplicates: list[str] = []
    for record in records:
        uid = _get(record, uid_attr)
        if uid in out:
            duplicates.append(str(uid))
        out[str(uid)] = record
    if duplicates:
        raise UIDScoringError(f"DUPLICATE_UID: {sorted(set(duplicates))[:5]}")
    return out


def validate_uid_maps(
    truth_by_uid: Mapping[str, TruthRecord],
    blind_by_uid: Mapping[str, BlindRecord],
    selected_by_uid: Mapping[str, Mapping[str, int]],
    *,
    selector_keys: Sequence[str] = ALL_SELECTOR_KEYS,
    k: int = K,
) -> None:
    if k != K:
        raise UIDScoringError(f"K_MUST_BE_16: got {k}")
    truth_uids = set(truth_by_uid)
    blind_uids = set(blind_by_uid)
    selected_uids = set(selected_by_uid)
    if len(truth_uids) != len(truth_by_uid):
        raise UIDScoringError("TRUTH_UIDS_NOT_UNIQUE")
    if len(blind_uids) != len(blind_by_uid):
        raise UIDScoringError("BLIND_UIDS_NOT_UNIQUE")
    if len(selected_uids) != len(selected_by_uid):
        raise UIDScoringError("SELECTED_UIDS_NOT_UNIQUE")
    if truth_uids != blind_uids or truth_uids != selected_uids:
        raise UIDScoringError(
            "UID_SET_MISMATCH: "
            f"missing_blind={sorted(truth_uids - blind_uids)[:3]} "
            f"extra_blind={sorted(blind_uids - truth_uids)[:3]} "
            f"missing_selected={sorted(truth_uids - selected_uids)[:3]} "
            f"extra_selected={sorted(selected_uids - truth_uids)[:3]}"
        )
    for uid in sorted(truth_uids):
        truth = truth_by_uid[uid]
        blind = blind_by_uid[uid]
        if int(truth.source_index) != int(blind.source_index):
            raise UIDScoringError(f"SOURCE_INDEX_MISMATCH:{uid}:{truth.source_index}!={blind.source_index}")
        if truth.transformed_64_sha256 and blind.transformed_64_sha256:
            if truth.transformed_64_sha256 != blind.transformed_64_sha256:
                raise UIDScoringError(f"TRANSFORMED_HASH_MISMATCH:{uid}")
        cand = np.asarray(blind.candidate_nulls)
        if cand.ndim != 2 or cand.shape[0] != k:
            raise UIDScoringError(f"CANDIDATE_COUNT_MISMATCH:{uid}:shape={cand.shape}")
        selected = selected_by_uid[uid]
        missing = [key for key in selector_keys if key not in selected]
        if missing:
            raise UIDScoringError(f"SELECTOR_MISSING:{uid}:{missing[:3]}")
        for key in selector_keys:
            idx = int(selected[key])
            if idx < 0 or idx >= k:
                raise UIDScoringError(f"SELECTED_INDEX_OUT_OF_RANGE:{uid}:{key}:{idx}")


def score_uid_maps(
    truth_by_uid: Mapping[str, TruthRecord],
    blind_by_uid: Mapping[str, BlindRecord],
    selected_by_uid: Mapping[str, Mapping[str, int]],
    *,
    selector_keys: Sequence[str] = ALL_SELECTOR_KEYS,
    k: int = K,
) -> dict[str, Any]:
    validate_uid_maps(truth_by_uid, blind_by_uid, selected_by_uid, selector_keys=selector_keys, k=k)
    ordered_uids = sorted(truth_by_uid)
    true_null = np.stack([np.asarray(truth_by_uid[uid].true_null, dtype=np.float64) for uid in ordered_uids], axis=0)
    candidate_nulls = np.stack([np.asarray(blind_by_uid[uid].candidate_nulls, dtype=np.float64) for uid in ordered_uids], axis=0)
    p0 = p0_rmse_matrix(candidate_nulls, true_null)
    random_expected = p0.mean(axis=1)
    oracle = p0.min(axis=1)
    oracle_indices = p0.argmin(axis=1)
    per_selector: dict[str, dict[str, Any]] = {}
    for key in selector_keys:
        idx = np.asarray([int(selected_by_uid[uid][key]) for uid in ordered_uids], dtype=np.int64)
        selected = p0[np.arange(len(ordered_uids)), idx]
        per_selector[key] = {
            "selected_indices": idx,
            "selected_errors": selected,
            "mean_selected": float(selected.mean()),
            "mean_random": float(random_expected.mean()),
            "mean_oracle": float(oracle.mean()),
            "mean_selected_minus_random": float((selected - random_expected).mean()),
            "selected_beats_random_fraction": float(np.mean(selected < random_expected)),
            "selection_regret_mean": float(np.mean(selected - oracle)),
            "oracle_gain_fraction_mean": _oracle_gain_fraction(random_expected, selected, oracle),
            "top_oracle_hit_rate": float(np.mean(idx == oracle_indices)),
        }
    return {
        "ordered_uids": ordered_uids,
        "p0_error": p0,
        "random_expected": random_expected,
        "oracle": oracle,
        "oracle_indices": oracle_indices,
        "per_selector": per_selector,
    }


def build_selected_by_uid_from_scores(
    uids: Sequence[str],
    score_by_selector: Mapping[str, np.ndarray],
    *,
    selector_keys: Sequence[str] = ALL_SELECTOR_KEYS,
    k: int = K,
) -> dict[str, dict[str, int]]:
    selected_by_uid = {str(uid): {} for uid in uids}
    if len(selected_by_uid) != len(uids):
        raise UIDScoringError("DUPLICATE_UID_IN_SCORE_ROWS")
    for key in selector_keys:
        if key not in score_by_selector:
            raise UIDScoringError(f"SCORE_SELECTOR_MISSING:{key}")
        scores = np.asarray(score_by_selector[key])
        if scores.shape != (len(uids), k):
            raise UIDScoringError(f"SCORE_SHAPE_MISMATCH:{key}:{scores.shape}")
        indices = np.argmax(scores, axis=1).astype(np.int64)
        for uid, idx in zip(uids, indices):
            selected_by_uid[str(uid)][key] = int(idx)
    return selected_by_uid


def selected_by_uid_from_index_arrays(
    uids: Sequence[str],
    index_by_selector: Mapping[str, np.ndarray],
    *,
    selector_keys: Sequence[str] = ALL_SELECTOR_KEYS,
    k: int = K,
) -> dict[str, dict[str, int]]:
    selected_by_uid = {str(uid): {} for uid in uids}
    if len(selected_by_uid) != len(uids):
        raise UIDScoringError("DUPLICATE_UID_IN_SELECTED_ROWS")
    for key in selector_keys:
        if key not in index_by_selector:
            raise UIDScoringError(f"SELECTED_SELECTOR_MISSING:{key}")
        indices = np.asarray(index_by_selector[key], dtype=np.int64)
        if indices.shape != (len(uids),):
            raise UIDScoringError(f"SELECTED_SHAPE_MISMATCH:{key}:{indices.shape}")
        for uid, idx in zip(uids, indices):
            if int(idx) < 0 or int(idx) >= k:
                raise UIDScoringError(f"SELECTED_INDEX_OUT_OF_RANGE:{uid}:{key}:{int(idx)}")
            selected_by_uid[str(uid)][key] = int(idx)
    return selected_by_uid


def truth_rows_hash_from_verified_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = []
    for row in rows:
        canonical.append(
            {
                "sample_uid": row["expected_sample_uid"],
                "reconstructed_sample_uid": row["reconstructed_sample_uid"],
                "source_index": int(row["source_index"]),
                "expected_raw_source_sha256": row["expected_raw_source_sha256"],
                "actual_raw_source_sha256": row["actual_raw_source_sha256"],
                "expected_transformed_64_sha256": row["expected_transformed_64_sha256"],
                "actual_transformed_64_sha256": row["actual_transformed_64_sha256"],
                "all_match": bool(row["all_match"]),
            }
        )
    return sha256_json(canonical)


def _oracle_gain_fraction(random_expected: np.ndarray, selected: np.ndarray, oracle: np.ndarray) -> float | None:
    denom = np.asarray(random_expected, dtype=np.float64) - np.asarray(oracle, dtype=np.float64)
    numer = np.asarray(random_expected, dtype=np.float64) - np.asarray(selected, dtype=np.float64)
    valid = np.abs(denom) > 1e-12
    if not np.any(valid):
        return None
    return float(np.mean(numer[valid] / denom[valid]))


def _get(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
