from __future__ import annotations

import numpy as np

from src.phase2_witness import (
    adaptive_witness_order,
    compute_gate,
    make_dct2_lowfreq_rows,
    make_rademacher_rows,
    make_witness_rows,
    qualified_sample_uid,
    select_adaptive_witness,
    select_random_witness,
    witness_residual_scores,
)
from src.phase2_locked_protocol import (
    classify_locked_addon_result,
    interval_overlap,
    locked_sample_uid,
    stable_index_order,
    validate_development_exclusion,
)


def _summary(mean, delta_random=-0.1, gain=0.5, delta_post=0.1):
    return {
        "mean_p0_rmse": mean,
        "delta_vs_random_mean": delta_random,
        "delta_vs_posterior_mean": delta_post,
        "oracle_gain_fraction_aggregate": gain,
        "bootstrap_vs_random": {"ci_upper": -0.01},
    }


def test_witness_residual_selects_exact_candidate() -> None:
    truth = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    candidates = np.stack(
        [
            truth + np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            truth.copy(),
            truth + np.array([0.0, 0.0, -2.0, 0.0], dtype=np.float32),
        ],
        axis=0,
    )
    rows = np.eye(4, dtype=np.float32)
    scores = witness_residual_scores(candidates, truth, rows)
    assert int(np.argmin(scores)) == 1
    assert scores[1] == 0.0


def test_random_witness_reproducible_and_bounded_indices() -> None:
    rng = np.random.default_rng(123)
    truth = rng.normal(size=(5, 8)).astype(np.float32)
    candidates = truth[:, None, :] + rng.normal(scale=0.1, size=(5, 16, 8)).astype(np.float32)
    rows_a = make_rademacher_rows(4, 8, 99)
    rows_b = make_rademacher_rows(4, 8, 99)
    assert np.array_equal(rows_a, rows_b)
    selected = select_random_witness(candidates, truth, rows_a, 3)
    assert selected.shape == (5,)
    assert np.all((selected >= 0) & (selected < 16))


def test_dct_lowfreq_rows_are_normalized() -> None:
    rows = make_dct2_lowfreq_rows(5, 8)
    assert rows.shape == (5, 64)
    norms = np.linalg.norm(rows, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert np.isclose(rows[0].std(), 0.0, atol=1e-6)


def test_configurable_witness_rows_support_dct_and_hybrid() -> None:
    dct = make_witness_rows("dct2_low_frequency", 6, 64, seed=123)
    hybrid = make_witness_rows("hybrid_dct_rademacher", 7, 64, seed=123)
    assert dct.shape == (6, 64)
    assert hybrid.shape == (7, 64)
    assert np.allclose(np.linalg.norm(dct, axis=1), 1.0, atol=1e-6)
    assert np.allclose(hybrid[:4], make_dct2_lowfreq_rows(4, 8), atol=1e-6)


def test_adaptive_witness_uses_candidate_variance() -> None:
    candidates = np.zeros((16, 4), dtype=np.float32)
    candidates[:, 2] = np.linspace(-2, 2, 16)
    library = np.eye(4, dtype=np.float32)
    order = adaptive_witness_order(candidates, library)
    assert int(order[0]) == 2
    truth = np.zeros((1, 4), dtype=np.float32)
    idx, rows = select_adaptive_witness(candidates[None, :, :], truth, library, 1)
    assert idx.shape == (1,)
    assert rows[0] == [2]


def test_qualified_uid_contains_namespace() -> None:
    uid = qualified_sample_uid("val", 42, 3, "cache_name")
    assert "phase2_dev" in uid
    assert "source_index:42" in uid
    assert "row:3" in uid


def test_gate_keeps_posterior_challenge_separate() -> None:
    summaries = {
        "random_expectation": _summary(1.0, 0.0, None),
        "posterior_mean": _summary(0.40, -0.6, 0.9, 0.0),
        "dm_fcc_seed3": _summary(0.60, -0.4, 0.5, 0.2),
        "random_witness_b64": _summary(0.58, -0.42, 0.55, 0.18),
        "fixed_lowfreq_witness_b64": _summary(0.55, -0.45, 0.65, 0.15),
        "adaptive_witness_b64": _summary(0.50, -0.50, 0.70, 0.10),
        "adaptive_witness_b16": _summary(0.56, -0.44, 0.60, 0.16),
        "compat_top4_adaptive_witness_b16": _summary(0.57, -0.43, 0.58, 0.17),
        "compat_top4_adaptive_witness_b64": _summary(0.49, -0.51, 0.72, 0.09),
        "oracle_best_of_16": _summary(0.35, -0.65, 1.0, -0.05),
    }
    config = {
        "witness": {"budgets": [16, 64], "primary_selector": "dm_fcc_seed3", "compatibility_prefilter_top_m": 4},
        "pilot_gate": {"primary_budget": 64, "low_budget": 16, "min_oracle_gain_fraction": 0.45},
    }
    gate = compute_gate(summaries, [], config)
    assert gate["conditions"]["adaptive_primary_budget_beats_random_expectation_with_ci"]
    assert not gate["conditions"]["adaptive_primary_budget_beats_posterior_mean_by_mean"]
    assert gate["decision"] == "CONTINUE_WITNESS_DEVELOPMENT_DO_NOT_LOCK_TEST_YET"


def test_locked_protocol_uid_namespace_and_overlap_guard() -> None:
    uid = locked_sample_uid("phase2_lock", "locked_test", 123, 4)
    assert uid.startswith("phase2_locked:")
    assert "source_index:123" in uid
    assert interval_overlap(10, 5, 14, 2)
    assert not interval_overlap(10, 5, 15, 2)
    audit = validate_development_exclusion(
        locked_offset=100,
        locked_count=10,
        exclusions=[
            {"name": "dev_a", "train_unlabeled_offset": 10, "sample_count": 5},
            {"name": "dev_b", "train_unlabeled_offset": 110, "sample_count": 5},
        ],
    )
    assert audit["status"] == "PASS"
    bad = validate_development_exclusion(
        locked_offset=100,
        locked_count=10,
        exclusions=[{"name": "dev_overlap", "train_unlabeled_offset": 109, "sample_count": 2}],
    )
    assert bad["status"] == "FAIL"


def test_locked_addon_classification_uses_preregistered_primary_conditions() -> None:
    summaries = {
        "adaptive_witness_b64": {
            "mean_p0_rmse": 0.30,
            "delta_vs_posterior_mean": -0.01,
            "delta_vs_random_mean": -0.02,
            "bootstrap_vs_posterior": {"ci_upper": -0.001},
            "bootstrap_vs_random": {"ci_upper": -0.002},
            "oracle_gain_fraction_aggregate": 0.6,
        }
    }
    config = {
        "locked_test": {
            "primary_method": "adaptive_witness_b64",
            "relmeaserr_max_threshold": 1e-5,
            "min_oracle_gain_fraction": 0.5,
        },
        "classification_rules": {
            "success": "WITNESS_ADDON_CONFIRMED",
            "trend": "WITNESS_ADDON_TREND_ONLY",
            "no_benefit": "WITNESS_ADDON_NO_BENEFIT",
            "invalid": "WITNESS_EVALUATION_INVALID",
        },
    }
    out = classify_locked_addon_result(
        summaries,
        {"status": "PASS", "canonical_relmeaserr_max": 1e-7},
        config,
    )
    assert out["classification"] == "WITNESS_ADDON_CONFIRMED"
    bad = classify_locked_addon_result(
        summaries,
        {"status": "PASS", "canonical_relmeaserr_max": 2e-5},
        config,
    )
    assert bad["classification"] == "WITNESS_EVALUATION_INVALID"


def test_stable_index_order_is_salt_dependent_and_permutation_safe() -> None:
    arr = np.arange(10, dtype=np.int64)
    a = stable_index_order(arr, "salt_a")
    b = stable_index_order(arr[::-1], "salt_a")
    c = stable_index_order(arr, "salt_b")
    assert a.tolist() == b.tolist()
    assert sorted(a.tolist()) == arr.tolist()
    assert a.tolist() != c.tolist()
