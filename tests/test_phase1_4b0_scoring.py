from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import phase1_4b_scoring as s


def test_random_expectation_is_candidate_metric_mean() -> None:
    metric = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]])
    np.testing.assert_allclose(s.compute_random_expectation(metric), [3.0, 14.0 / 3.0])


def test_posterior_mean_is_not_random_expectation() -> None:
    candidates = np.array([[[0.0, 0.0], [2.0, 2.0]]])
    posterior = s.compute_posterior_mean(candidates)
    metric_random = s.compute_random_expectation(np.array([[0.5, 1.5]]))
    np.testing.assert_allclose(posterior, [[1.0, 1.0]])
    assert posterior.shape != metric_random.shape


def test_lowest_index_tie_handling_for_selector_and_oracle() -> None:
    scores = np.array([[2.0, 2.0, 1.0], [0.0, 4.0, 4.0]])
    errors = np.array([[1.0, 1.0, 2.0], [3.0, 0.5, 0.5]])
    np.testing.assert_array_equal(s.selected_by_argmax(scores), [0, 1])
    np.testing.assert_array_equal(s.oracle_indices(errors), [0, 1])


def test_relative_improvement_and_oracle_gain_formulas() -> None:
    random = np.array([10.0, 10.0])
    selected = np.array([8.0, 9.0])
    oracle = np.array([5.0, 5.0])
    assert s.aggregate_relative_improvement(random, selected) == pytest.approx(0.15)
    gain = s.aggregate_oracle_gain_fraction(random, selected, oracle)
    assert gain["status"] == "ok"
    assert gain["value"] == pytest.approx(0.3)


def test_oracle_gain_denominator_near_zero_not_applicable() -> None:
    out = s.aggregate_oracle_gain_fraction(np.array([1.0]), np.array([1.0]), np.array([1.0]))
    assert out["status"] == "not_applicable"


def test_p0_rmse_is_before_clipping() -> None:
    cand = np.array([[[2.0, -1.0], [0.0, 1.0]]])
    truth = np.array([[1.0, 0.0]])
    rmse = s.p0_rmse_matrix(cand, truth)
    np.testing.assert_allclose(rmse[0, 0], 1.0)
    assert rmse[0, 0] != 0.0


def test_true_null_two_definitions_consistent_in_fixture() -> None:
    x_true = np.array([[3.0, 4.0]])
    r_y = np.array([[1.0, 1.0]])
    true_n_b = x_true - r_y
    true_n_a = np.array([[2.0, 3.0]])
    np.testing.assert_allclose(true_n_a, true_n_b)


def test_random_secondary_metric_after_per_candidate_metric() -> None:
    candidate_psnr = np.array([[10.0, 20.0]])
    assert s.compute_random_expectation(candidate_psnr)[0] == 15.0


def test_primary_oracle_reused_for_secondary_rows() -> None:
    p0 = np.array([[0.2, 0.1, 0.3]])
    secondary = np.array([[5.0, 7.0, 1.0]])
    oracle_idx, _ = s.compute_primary_oracle(p0)
    assert int(oracle_idx[0]) == 1
    assert secondary[0, oracle_idx[0]] == 7.0


def test_metric_contract_fixed_ranges_and_channels() -> None:
    contract = s.metric_contract()
    assert contract["PSNR_data_range"] == 1.0
    assert contract["SSIM"]["channel_axis"] is None
    assert contract["LPIPS"]["input_mapping"].startswith("[0,1]")
    assert contract["RAPSD"]["bins"] == 32


def test_rapsd_profile_is_normalized() -> None:
    img = np.zeros((64, 64), dtype=np.float64)
    img[32, 32] = 1.0
    prof = s.rapsd_profile(img, bins=32)
    assert prof.shape == (32,)
    assert float(prof.sum()) == pytest.approx(1.0)


def test_hash_mismatch_rejected(tmp_path: Path) -> None:
    p = tmp_path / "artifact.txt"
    p.write_text("abc", encoding="utf-8")
    with pytest.raises(RuntimeError, match="FINAL_MANIFEST_HASH_MISMATCH"):
        s.require_hash(p, "bad", "FINAL_MANIFEST")
