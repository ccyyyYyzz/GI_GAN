from __future__ import annotations

from pathlib import Path

import numpy as np

from phase1_3_freeze_and_audit import (
    PHASE12,
    PARENT_FINAL_INDICES,
    expected_selector_artifacts,
    final_integrity_audit,
    hash_paths,
    sha256_file,
)


def test_phase1_3_primary_selector_artifact_is_not_silently_available() -> None:
    artifacts = expected_selector_artifacts(PHASE12)
    primary = artifacts["primary_dm_fcc_seed3_checkpoint"]
    assert not primary.exists()


def test_phase1_3_missing_selector_artifacts_are_explicit() -> None:
    artifacts = expected_selector_artifacts(PHASE12)
    missing = [name for name, path in artifacts.items() if not path.exists()]
    assert "primary_dm_fcc_seed3_checkpoint" in missing
    assert "scalar_pair_selector_model" in missing


def test_phase1_3_final_indices_exist_and_not_empty() -> None:
    assert PARENT_FINAL_INDICES.exists()
    idx = np.load(PARENT_FINAL_INDICES)
    assert idx.shape[0] == 512


def test_phase1_3_hash_paths_detect_existing_and_missing(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    missing = tmp_path / "missing.txt"
    hashes = hash_paths([p, missing])
    assert hashes[str(p)]["exists"] is True
    assert hashes[str(p)]["sha256"] == sha256_file(p)
    assert hashes[str(missing)]["exists"] is False


def test_phase1_3_integrity_audit_does_not_mark_evaluated_in_temp_dir(tmp_path: Path) -> None:
    audit = final_integrity_audit(tmp_path, PHASE12)
    assert audit["final_source_indices_count"] == 512
    assert audit["final_test_evaluated_flag"] is False
    assert audit["status"] in {"CLEAN_UNSEEN_FINAL_TEST", "INVALID_SPLIT_OVERLAP", "POSSIBLY_SEEN_OR_CONTAMINATED"}
    assert not (tmp_path / "freeze_bundle" / "FINAL_EVAL_FROZEN.json").exists()
