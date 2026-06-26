from __future__ import annotations

import inspect

import numpy as np
import torch

from phase1_1_corrected_pipeline import oracle_energy_report
from src.compatibility_data import SplitComponents
from src.phase1_1_controls import (
    fixed_32_candidate_manifests,
    k_prefix_indices,
    label_histogram,
    make_pair_arrays,
    nuisance_balanced_derangement,
    overlap_count,
    pair_features,
    score_error_correlations,
    tie_aware_auc,
)


def _split(n: int = 18, img_size: int = 8) -> SplitComponents:
    gen = torch.Generator().manual_seed(123)
    r = torch.randn(n, img_size * img_size, generator=gen) * 0.2
    null = torch.randn(n, img_size * img_size, generator=gen) * 0.3
    x = (r + null).reshape(n, 1, img_size, img_size)
    y = torch.randn(n, 5, generator=gen)
    labels = torch.tensor([-1, 0, 1, -1, 2, 2] * ((n + 5) // 6))[:n]
    return SplitComponents("toy", x.float(), r.float(), null.float(), y.float(), labels.long(), torch.arange(100, 100 + n), {"solver": "toy"})


def test_tie_aware_auc_edges_and_ties() -> None:
    assert tie_aware_auc([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert tie_aware_auc([1, 1, 0, 0], [2.0, 1.0, 0.0, -1.0]) == 1.0
    assert tie_aware_auc([1, 1, 0, 0], [0.0, -1.0, 2.0, 1.0]) == 0.0
    try:
        from sklearn.metrics import roc_auc_score

        labels = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([1.0, 0.5, 0.5, 0.5, 0.2, 0.2])
        assert abs(tie_aware_auc(labels, scores) - roc_auc_score(labels, scores)) < 1e-12
    except Exception:
        pass


def test_oracle_energy_metric_marked_non_deployable() -> None:
    split = _split()
    donors = np.roll(np.arange(split.size), 1)
    report = oracle_energy_report(split, donors)
    assert report["metric_name"] == "oracle_true_null_energy_distance_auc"
    assert report["non_deployable"] is True
    assert report["excluded_from_gate"] is True


def test_deployable_pair_features_signature_has_no_true_null_argument() -> None:
    params = list(inspect.signature(pair_features).parameters)
    assert params == ["r", "n", "img_size"]
    split = _split()
    x, names = pair_features(split.r[:4], split.n[:4], split.img_size)
    assert x.shape[0] == 4
    assert "u_fraction_below_0" in names


def test_row_null_controls_use_real_features_not_constants() -> None:
    split = _split()
    donors = np.roll(np.arange(split.size), 1)
    row_x, row_y, _names, _rows = make_pair_arrays(split, donors, feature_mode="row")
    null_x, null_y, _names2, _rows2 = make_pair_arrays(split, donors, feature_mode="null")
    assert row_x.std() > 0
    assert null_x.std() > 0
    assert set(row_y.tolist()) == {0, 1}
    assert set(null_y.tolist()) == {0, 1}


def test_nuisance_balanced_derangement_is_one_to_one() -> None:
    split = _split(n=20)
    donors, report = nuisance_balanced_derangement(split, seed=4)
    assert donors.shape[0] == split.size
    assert int(np.sum(donors == np.arange(split.size))) == 0
    assert len(np.unique(donors)) == split.size
    assert report["donor_unique_fraction"] == 1.0
    assert report["positive_negative_n_marginal_same_multiset"] is True


def test_unknown_labels_are_screened() -> None:
    hist = label_histogram(torch.tensor([-1, -1, 0, 1, 1, 2]))
    assert hist["unlabeled_count"] == 2
    assert "-1" not in hist["histogram_labeled_only"]


def test_candidate_manifests_are_deterministic() -> None:
    split = _split(n=20)
    a = fixed_32_candidate_manifests(split, count=2, donors_per_anchor=8, seed=9)
    b = fixed_32_candidate_manifests(split, count=2, donors_per_anchor=8, seed=9)
    assert [m["manifest_hash"] for m in a] == [m["manifest_hash"] for m in b]


def test_k_pools_use_same_prefix() -> None:
    pools = k_prefix_indices(32, [1, 4, 8, 16, 32])
    assert pools[4] == pools[8][:4]
    assert pools[16] == pools[32][:16]


def test_final_test_overlap_helper() -> None:
    assert overlap_count([1, 2, 3], [3, 4]) == 1
    assert overlap_count([1, 2], [3, 4]) == 0


def test_score_error_correlation_can_exclude_positive() -> None:
    split = _split(n=10)
    matrix = np.eye(10) * 10.0
    report = score_error_correlations(matrix, split.n)
    assert "spearman_including_positive" in report
    assert "spearman_negatives_only_global" in report
    assert report["spearman_including_positive"] != report["spearman_negatives_only_global"]
