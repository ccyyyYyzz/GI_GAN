from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import phase1_4v4b0_scoring as s


ROOT = Path("E:/ns_mc_gan_gi_code_fcc_phase1")


def test_v4_random_posterior_oracle_definitions() -> None:
    metric = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]])
    np.testing.assert_allclose(s.compute_random_expectation(metric), [3.0, 14.0 / 3.0])
    candidates = np.array([[[0.0, 0.0], [2.0, 2.0]]])
    np.testing.assert_allclose(s.compute_posterior_mean(candidates), [[1.0, 1.0]])
    errors = np.array([[1.0, 1.0, 2.0], [3.0, 0.5, 0.5]])
    np.testing.assert_array_equal(s.oracle_indices(errors), [0, 1])


def test_v4_primary_p0_rmse_is_unclipped_null_metric() -> None:
    cand = np.array([[[2.0, -1.0], [0.0, 1.0]]])
    truth = np.array([[1.0, 0.0]])
    rmse = s.p0_rmse_matrix(cand, truth)
    np.testing.assert_allclose(rmse[0, 0], 1.0)
    assert rmse[0, 0] != 0.0


def test_v4_statistics_are_image_level_and_seeded() -> None:
    delta = np.array([-1.0, 0.0, 1.0, 2.0])
    a = s.paired_percentile_bootstrap(delta, B=200, seed=17)
    b = s.paired_percentile_bootstrap(delta, B=200, seed=17)
    assert a == b
    assert a["unit"] == "image"
    sign = s.exact_sign_test(np.array([-1.0, 0.0, 2.0]))
    assert sign["ties"] == 1


def test_v4_holm_family_has_only_two_h2_comparisons() -> None:
    adjusted = s.holm_adjust({"dm_vs_scalar": 0.03, "dm_vs_sum": 0.02})
    assert adjusted == {"dm_vs_scalar": 0.04, "dm_vs_sum": 0.04}
    contract = s.statistics_contract()
    assert contract["Holm_family_H2_only"] == ["dm_fcc_seed3_vs_scalar_pair_selector", "dm_fcc_seed3_vs_sum_image_selector"]


def test_v4_classification_no_strong_fcc_class_due_h3_limit() -> None:
    assert s.classify_final_v4_conclusion({"H1_PASS": True, "H4_PASS": True, "H5_PASS": True}) == "FINAL_V4_SELECTOR_GENERALIZES_BUT_FCC_NOT_CONFIRMED"
    assert s.classify_final_v4_conclusion({"H1_PASS": False, "H4_PASS": True, "H5_PASS": True}, h1_mean_selected_better=True) == "FINAL_V4_NUMERICAL_TREND_ONLY"
    assert s.classify_final_v4_conclusion({"H1_PASS": False, "H4_PASS": True, "H5_PASS": True}) == "FINAL_V4_FAILED_TO_GENERALIZE"
    assert s.classify_final_v4_conclusion({"H1_PASS": True, "H4_PASS": False, "H5_PASS": True}) == "FINAL_V4_EVALUATION_INVALID"
    assert "FINAL_V4_CONFIRMED" not in "".join(s.classification_contract()["allowed_classes"])


def test_v4_h5_and_s1_identity_are_frozen() -> None:
    h = s.final_v4_hypothesis_contract()
    assert h["H5"]["identity"] == "Measurement consistency"
    assert h["H5"]["not_dm_vs_raw"] is True
    assert h["S1"]["identity"] == "S1_PRE_SCORING_AMENDMENT_DM_VS_RAW"
    assert h["H3"]["status"] == "PRE_SPECIFIED_COMPARISON_WITH_INCOMPLETE_DECISION_RULE"
    assert h["primary_selector"] == "dm_fcc_seed3"


def test_v4_uid_join_ignores_input_order() -> None:
    uid_a, uid_b = "a", "b"
    vec_a = np.zeros(4096, dtype=np.float32)
    vec_b = np.zeros(4096, dtype=np.float32)
    vec_a[0] = 1.0
    vec_b[1] = 1.0
    truth = {
        uid_a: s.TruthRecord(uid_a, 1, "syn", "syn/test", 1, "", "", "", "", vec_a),
        uid_b: s.TruthRecord(uid_b, 2, "syn", "syn/test", 2, "", "", "", "", vec_b),
    }
    cand_a = np.zeros((s.K, 4096), dtype=np.float32)
    cand_b = np.zeros((s.K, 4096), dtype=np.float32)
    cand_a[2] = truth[uid_a].image_flat
    cand_b[2] = truth[uid_b].image_flat
    blind = {
        uid_b: s.BlindRecord(uid_b, 2, "syn", "syn/test", 2, "", np.zeros(4096), np.zeros(4096), cand_b, np.zeros(s.K), np.zeros(s.K), np.zeros(s.K), np.zeros(s.K)),
        uid_a: s.BlindRecord(uid_a, 1, "syn", "syn/test", 1, "", np.zeros(4096), np.zeros(4096), cand_a, np.zeros(s.K), np.zeros(s.K), np.zeros(s.K), np.zeros(s.K)),
    }
    selector = {uid_a: {key: 2 for key in s.ALL_SELECTOR_KEYS}, uid_b: {key: 2 for key in s.ALL_SELECTOR_KEYS}}
    out = s.score_uid_path(truth, blind, selector)
    assert out["per_selector"]["dm_fcc_seed3"]["selected_p0_rmse_mean"] == pytest.approx(0.0)


def test_v4_uid_join_hard_fails_missing_uid() -> None:
    truth = {"a": s.TruthRecord("a", 1, "syn", "syn/test", 1, "", "", "", "", np.zeros(4096, dtype=np.float32))}
    blind = {}
    selector = {"a": {key: 0 for key in s.ALL_SELECTOR_KEYS}}
    with pytest.raises(s.V4B0Error, match="UID_SET_MISMATCH"):
        s.score_uid_path(truth, blind, selector)


def test_v4_guard_refuses_final_without_confirm_and_does_not_start() -> None:
    started = s.FINAL_SCORING / "FINAL_V4_SCORING_STARTED.json"
    before = started.exists()
    args = argparse.Namespace(dataset_scope="final", confirm="", scoring_protocol_hash="", incident_override="")
    ok, reason = s.guard_final_scoring(args)
    assert not ok
    assert reason == "MISSING_OR_INVALID_CONFIRM_TOKEN"
    assert started.exists() == before


def test_v4_runner_dev_scope_is_guard_only() -> None:
    res = subprocess.run(
        [sys.executable, str(ROOT / "score_phase1_4v4_final_once.py"), "--dataset-scope", "dev"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    assert res.returncode == 0
    assert "DEV_SCOPE_OK" in res.stdout
    assert not (s.FINAL_SCORING / "FINAL_V4_SCORING_STARTED.json").exists()


def test_v4_scorer_source_has_no_positional_truth_slice_pattern() -> None:
    tree = ast.parse((ROOT / "src" / "phase1_4v4b0_scoring.py").read_text(encoding="utf-8"))
    bad_names = {"start", "count"}
    for node in ast.walk(tree):
        value_name = node.value.id if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) else ""
        assert not (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Slice)
            and isinstance(node.slice.lower, ast.Name)
            and node.slice.lower.id in bad_names
            and value_name in {"truth", "x_true", "truth_rows", "truth_tensor"}
        )
    text = (ROOT / "src" / "phase1_4v4b0_scoring.py").read_text(encoding="utf-8")
    assert "hash_verified = True" not in text
    assert "x_true[start" not in text
