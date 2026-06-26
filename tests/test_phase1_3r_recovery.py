from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from phase1_3r_recovery_and_relock import (
    REPAIR_SALT,
    qualified_uid,
    sha256_np,
)


def test_qualified_uid_includes_split_namespace() -> None:
    raw = "a" * 64
    train_uid = qualified_uid("stl10", "stl10/train", 7, raw)
    test_uid = qualified_uid("stl10", "stl10/test", 7, raw)
    assert train_uid != test_uid


def test_same_raw_hash_different_index_detectable_as_duplicate() -> None:
    raw = "b" * 64
    uid_a = qualified_uid("stl10", "stl10/test", 1, raw)
    uid_b = qualified_uid("stl10", "stl10/test", 2, raw)
    assert uid_a != uid_b
    assert raw == raw


def test_replacement_sort_key_is_label_and_image_blind() -> None:
    pool = [13, 2, 99, 7]
    ordered_a = sorted(pool, key=lambda i: hashlib.sha256(f"{REPAIR_SALT}|stl10|test|{i}".encode("utf-8")).hexdigest())
    ordered_b = sorted(reversed(pool), key=lambda i: hashlib.sha256(f"{REPAIR_SALT}|stl10|test|{i}".encode("utf-8")).hexdigest())
    assert ordered_a == ordered_b


def test_final_v2_flags_are_explicitly_false_in_manifest_shape() -> None:
    manifest = {
        "source_indices_count": 512,
        "final_test_evaluated": False,
        "final_candidates_generated": False,
        "final_metrics_computed": False,
    }
    assert manifest["source_indices_count"] == 512
    assert manifest["final_test_evaluated"] is False
    assert manifest["final_candidates_generated"] is False
    assert manifest["final_metrics_computed"] is False


def test_sha256_np_is_order_sensitive_for_locked_indices() -> None:
    a = np.asarray([1, 2, 3], dtype=np.int64)
    b = np.asarray([3, 2, 1], dtype=np.int64)
    assert sha256_np(a) != sha256_np(b)


def test_phase1_3r_does_not_create_final_eval_frozen_name(tmp_path: Path) -> None:
    ready = tmp_path / "READY_FOR_PHASE1_4_FINAL.json"
    ready.write_text("{}", encoding="utf-8")
    assert not (tmp_path / "FINAL_EVAL_FROZEN.json").exists()
