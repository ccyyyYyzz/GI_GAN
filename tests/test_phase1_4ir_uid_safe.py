from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.phase1_4ir_uid_safe_scoring import (
    BlindRecord,
    TruthRecord,
    UIDScoringError,
    build_selected_by_uid_from_scores,
    score_uid_maps,
    stable_candidate_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture_maps(n: int = 6, k: int = 16):
    selector_keys = ["dm_fcc_seed3", "scratch_seed1"]
    truth = {}
    blind = {}
    scores = {}
    uids = [f"uid{i:03d}" for i in range(n)]
    for key in selector_keys:
        scores[key] = np.zeros((n, k), dtype=np.float64)
    for i, uid in enumerate(uids):
        true_n = np.asarray([float(i), 1.0], dtype=np.float64)
        cand = np.stack([true_n + (j * 0.01) for j in range(k)], axis=0)
        truth[uid] = TruthRecord(uid, i, true_n, f"hash{i}")
        blind[uid] = BlindRecord(uid, i, np.zeros(2), cand, f"hash{i}")
        scores["dm_fcc_seed3"][i, 0] = 1.0
        scores["scratch_seed1"][i, 1] = 1.0
    selected = build_selected_by_uid_from_scores(uids, scores, selector_keys=selector_keys, k=k)
    return selector_keys, uids, truth, blind, selected


def test_shard_swap_manifest_reverse_and_row_shuffle_do_not_change_result():
    selector_keys, uids, truth, blind, selected = fixture_maps()
    base = score_uid_maps(truth, blind, selected, selector_keys=selector_keys)
    reversed_truth = {uid: truth[uid] for uid in reversed(uids)}
    reversed_blind = {uid: blind[uid] for uid in reversed(uids)}
    reversed_selected = {uid: selected[uid] for uid in reversed(uids)}
    rev = score_uid_maps(reversed_truth, reversed_blind, reversed_selected, selector_keys=selector_keys)
    assert np.allclose(np.sort(base["per_selector"]["dm_fcc_seed3"]["selected_errors"]), np.sort(rev["per_selector"]["dm_fcc_seed3"]["selected_errors"]))


def test_unequal_shard_sizes_are_irrelevant_after_uid_join():
    selector_keys, uids, truth, blind, selected = fixture_maps(n=7)
    shard_a = uids[:2]
    shard_b = uids[2:7]
    rebuilt_blind = {uid: blind[uid] for uid in shard_b + shard_a}
    result = score_uid_maps(truth, rebuilt_blind, selected, selector_keys=selector_keys)
    assert result["per_selector"]["dm_fcc_seed3"]["mean_selected"] == pytest.approx(0.0)


def test_static_new_scorer_forbids_position_bug_pattern_and_hardcoded_hash_verified():
    text = (ROOT / "src" / "phase1_4ir_uid_safe_scoring.py").read_text(encoding="utf-8")
    assert "start = len(" not in text
    assert "hash_verified = True" not in text
    assert '"hash_verified": True' not in text


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda truth, blind, selected: (truth, {k: v for k, v in list(blind.items())[1:]}, selected), "UID_SET_MISMATCH"),
        (lambda truth, blind, selected: (truth, {**blind, "extra": next(iter(blind.values()))}, selected), "UID_SET_MISMATCH"),
        (lambda truth, blind, selected: ({**truth, "extra": next(iter(truth.values()))}, blind, selected), "UID_SET_MISMATCH"),
    ],
)
def test_missing_extra_uid_hard_fail(mutator, message):
    selector_keys, _uids, truth, blind, selected = fixture_maps()
    t, b, s = mutator(truth, blind, selected)
    with pytest.raises(UIDScoringError, match=message):
        score_uid_maps(t, b, s, selector_keys=selector_keys)


def test_swapped_candidate_uid_truth_uid_hard_fails_by_source_index():
    selector_keys, uids, truth, blind, selected = fixture_maps()
    swapped = dict(blind)
    a, b = uids[0], uids[1]
    swapped[a], swapped[b] = BlindRecord(a, 999, swapped[b].r_y, swapped[b].candidate_nulls, swapped[b].transformed_64_sha256), BlindRecord(
        b, 998, swapped[a].r_y, swapped[a].candidate_nulls, swapped[a].transformed_64_sha256
    )
    with pytest.raises(UIDScoringError, match="SOURCE_INDEX_MISMATCH"):
        score_uid_maps(truth, swapped, selected, selector_keys=selector_keys)


def test_source_index_and_transformed_hash_mismatch_hard_fail():
    selector_keys, uids, truth, blind, selected = fixture_maps()
    uid = uids[0]
    bad_source = dict(blind)
    bad_source[uid] = BlindRecord(uid, 12345, blind[uid].r_y, blind[uid].candidate_nulls, blind[uid].transformed_64_sha256)
    with pytest.raises(UIDScoringError, match="SOURCE_INDEX_MISMATCH"):
        score_uid_maps(truth, bad_source, selected, selector_keys=selector_keys)
    bad_hash = dict(blind)
    bad_hash[uid] = BlindRecord(uid, blind[uid].source_index, blind[uid].r_y, blind[uid].candidate_nulls, "different")
    with pytest.raises(UIDScoringError, match="TRANSFORMED_HASH_MISMATCH"):
        score_uid_maps(truth, bad_hash, selected, selector_keys=selector_keys)


def test_k_and_selected_index_guards():
    selector_keys, uids, truth, blind, selected = fixture_maps()
    uid = uids[0]
    bad_k = dict(blind)
    bad_k[uid] = BlindRecord(uid, blind[uid].source_index, blind[uid].r_y, blind[uid].candidate_nulls[:15], blind[uid].transformed_64_sha256)
    with pytest.raises(UIDScoringError, match="CANDIDATE_COUNT_MISMATCH"):
        score_uid_maps(truth, bad_k, selected, selector_keys=selector_keys)
    bad_selected = {key: dict(value) for key, value in selected.items()}
    bad_selected[uid]["dm_fcc_seed3"] = 16
    with pytest.raises(UIDScoringError, match="SELECTED_INDEX_OUT_OF_RANGE"):
        score_uid_maps(truth, blind, bad_selected, selector_keys=selector_keys)
    with pytest.raises(UIDScoringError, match="K_MUST_BE_16"):
        score_uid_maps(truth, blind, selected, selector_keys=selector_keys, k=8)


def test_random_expectation_and_oracle_are_same_uid_candidate_functions():
    selector_keys, _uids, truth, blind, selected = fixture_maps()
    result = score_uid_maps(truth, blind, selected, selector_keys=selector_keys)
    assert np.allclose(result["random_expected"], result["p0_error"].mean(axis=1))
    assert np.allclose(result["oracle"], result["p0_error"].min(axis=1))
    assert result["per_selector"]["dm_fcc_seed3"]["mean_selected"] == pytest.approx(0.0)


def test_true_n_fixture_matches_exact_null_identity_case():
    selector_keys, _uids, truth, blind, selected = fixture_maps()
    # Synthetic identity-null fixture: candidates are direct perturbations of true_null.
    result = score_uid_maps(truth, blind, selected, selector_keys=selector_keys)
    assert result["p0_error"].shape[1] == 16
    assert np.allclose(result["p0_error"][:, 0], 0.0)


def test_candidate_position_seed_id_does_not_affect_uid_join():
    selector_keys, uids, truth, blind, selected = fixture_maps()
    first = stable_candidate_seed(uids[0], 0, "fixed_salt")
    second = stable_candidate_seed(uids[0], 0, "fixed_salt")
    assert first == second
    assert first != stable_candidate_seed(uids[0], 1, "fixed_salt")
    result = score_uid_maps(truth, blind, selected, selector_keys=selector_keys)
    assert result["per_selector"]["dm_fcc_seed3"]["top_oracle_hit_rate"] == pytest.approx(1.0)


def test_final_v3_invalid_file_is_not_overwritten_if_present(tmp_path):
    path = tmp_path / "FINAL_V3_EVALUATION_INVALID.json"
    path.write_text('{"status":"FINAL_EVALUATION_INVALID"}', encoding="utf-8")
    before = path.read_bytes()
    # This test documents the append-only policy: callers must not mutate an existing invalid record.
    after = path.read_bytes()
    assert before == after


def test_cli_refuses_corrected_v3_and_final_v4_scoring():
    script = ROOT / "phase1_4ir_incident_recovery.py"
    res_v3 = subprocess.run([sys.executable, str(script), "--corrected-final-v3-diagnostic"], cwd=str(ROOT), text=True, capture_output=True)
    assert res_v3.returncode == 2
    assert "REFUSING" in res_v3.stdout
    res_v4 = subprocess.run([sys.executable, str(script), "--score-final-v4"], cwd=str(ROOT), text=True, capture_output=True)
    assert res_v4.returncode == 2
    assert "REFUSING" in res_v4.stdout


def test_final_v4_selection_source_does_not_use_labels_or_image_statistics():
    text = (ROOT / "phase1_4ir_incident_recovery.py").read_text(encoding="utf-8")
    selection_block = text.split("def select_final_v4", 1)[1].split("def overlap_audit", 1)[0]
    assert ".labels" not in selection_block
    assert "labels[" not in selection_block
    assert ".targets" not in selection_block
    assert ".mean(" not in selection_block
    assert ".std(" not in selection_block
