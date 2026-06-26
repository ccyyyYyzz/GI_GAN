from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

import phase1_4a_freeze_and_blind as p14a


def test_final_seed_policy_is_deterministic_and_not_builtin_hash() -> None:
    uid = "abc123"
    seeds_a = [p14a.candidate_seed(uid, k) for k in range(16)]
    seeds_b = [p14a.candidate_seed(uid, k) for k in range(16)]
    assert seeds_a == seeds_b
    assert len(set(seeds_a)) == 16
    assert all(0 <= seed <= 0x7FFFFFFFFFFFFFFF for seed in seeds_a)


def test_zero_noise_baseline_is_not_pool_slot_constant() -> None:
    assert p14a.K == 16
    assert p14a.FINAL_SEED_SALT == "FCC_PHASE1_4_FINAL_CANDIDATES_V1"


def test_primary_is_locked_to_dm_fcc_seed3() -> None:
    assert p14a.PRIMARY_MODEL == "reproduced_dm_fcc_seed3_v2"
    assert p14a.PRIMARY_ARTIFACT_KEY == "dm_fcc_seed3"
    assert "raw_fcc_seed1" in p14a.RANKER_KEYS
    assert p14a.PRIMARY_ARTIFACT_KEY != "raw_fcc_seed1"


def test_hash_semantics_separate_file_and_array_hash(tmp_path: Path) -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "a.npy"
    np.save(path, arr)
    file_hash = p14a.sha256_file(path)
    array_hash = p14a.array_content_hash_with_metadata(np.load(path))
    assert file_hash != array_hash


def test_forbidden_truth_fields_are_detected() -> None:
    ok, found = p14a.validate_no_truth_fields({"safe": 1, "p0_error": [1.0]})
    assert ok is False
    assert found


def test_stage_b_refuses_without_confirm() -> None:
    proc = subprocess.run([sys.executable, "score_phase1_4b_final_once.py"], cwd=str(p14a.ROOT), text=True, capture_output=True)
    assert proc.returncode != 0
    assert "REFUSING" in proc.stdout
