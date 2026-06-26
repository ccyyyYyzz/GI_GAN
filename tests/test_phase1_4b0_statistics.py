from __future__ import annotations

import numpy as np

from src import phase1_4b_scoring as s


def test_bootstrap_is_image_level_and_seed_reproducible() -> None:
    delta = np.array([-1.0, 0.0, 1.0, 2.0])
    a = s.paired_percentile_bootstrap(delta, B=200, seed=17)
    b = s.paired_percentile_bootstrap(delta, B=200, seed=17)
    assert a == b
    assert a["unit"] == "image"
    assert a["observed_mean"] == np.mean(delta)


def test_negative_delta_means_selector_better_and_zero_delta() -> None:
    selected = np.array([1.0, 2.0, 3.0])
    random = np.array([2.0, 2.0, 4.0])
    delta = selected - random
    assert delta.mean() < 0
    assert np.all((selected - selected) == 0)


def test_sign_flip_two_sided_and_exact_sign_ties_excluded() -> None:
    delta = np.array([-1.0, -2.0, 0.0, 3.0, 0.0])
    flip = s.paired_sign_flip_test(delta, B=500, seed=19)
    sign = s.exact_sign_test(delta, tie_tol=1e-12)
    assert flip["two_sided"] is True
    assert 0.0 <= flip["p_value"] <= 1.0
    assert sign["ties"] == 2
    assert sign["n_non_tie"] == 3


def test_holm_two_comparison_family() -> None:
    adjusted = s.holm_adjust({"scalar": 0.03, "sum": 0.02})
    assert set(adjusted) == {"scalar", "sum"}
    assert adjusted["sum"] == 0.04
    assert adjusted["scalar"] == 0.04


def test_three_seed_method_average_is_per_image() -> None:
    errors = {
        "dm1": np.array([1.0, 5.0]),
        "dm2": np.array([2.0, 5.0]),
        "dm3": np.array([3.0, 5.0]),
        "sc1": np.array([4.0, 1.0]),
        "sc2": np.array([5.0, 1.0]),
        "sc3": np.array([6.0, 1.0]),
    }
    out = s.compute_method_seed_average(errors, ["dm1", "dm2", "dm3"], ["sc1", "sc2", "sc3"])
    np.testing.assert_allclose(out["left_mean"], [2.0, 5.0])
    np.testing.assert_allclose(out["right_mean"], [5.0, 1.0])
    np.testing.assert_allclose(out["delta"], [-3.0, 4.0])


def test_final_classification_rules() -> None:
    assert s.classify_final_conclusion({"H1_PASS": True, "H2_STRONG_PASS": True, "H3_PASS_WITH_COMPLETE_RULE": True, "H4_PASS": True, "H5_PASS": True}) == "FINAL_CONFIRMED_DM_FCC_ADDS_VALUE"
    assert s.classify_final_conclusion({"H1_PASS": True, "H2_STRONG_PASS": False, "H3_PASS_WITH_COMPLETE_RULE": False, "H4_PASS": True, "H5_PASS": True}) == "FINAL_SELECTOR_GENERALIZES_BUT_FCC_NOT_CONFIRMED"
    assert s.classify_final_conclusion({"H1_PASS": False, "H4_PASS": True, "H5_PASS": True}, h1_mean_selected_better=True) == "FINAL_NUMERICAL_TREND_ONLY"
    assert s.classify_final_conclusion({"H1_PASS": False, "H4_PASS": False, "H5_PASS": True}) == "FINAL_EVALUATION_INVALID"
