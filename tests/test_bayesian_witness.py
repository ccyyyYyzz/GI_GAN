from __future__ import annotations

import numpy as np
import torch

from src.bayesian_witness import (
    barycenter_null,
    conditional_nullspace_audit,
    gaussian_risk_utilities,
    posterior_weights_for_rows,
    sequential_risk_order,
    stable_softmax,
)
from src.projections import get_exact_projector


class ToyOperator:
    def __init__(self) -> None:
        self.A = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        self.m = 2
        self.n = 4
        self.img_size = 2

    def A_forward(self, v: torch.Tensor) -> torch.Tensor:
        return v.to(dtype=self.A.dtype, device=self.A.device) @ self.A.T


def test_stable_softmax_rows_sum_to_one() -> None:
    weights = stable_softmax(np.array([[1000.0, 999.0], [-1000.0, -1001.0]]), axis=1)
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(2), atol=1e-12)


def test_posterior_barycenter_is_mse_stationary_point() -> None:
    cand = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]], dtype=np.float64)
    q = np.array([[0.2, 0.5, 0.3]], dtype=np.float64)
    bary = barycenter_null(cand, q)[0]
    gradient = 2.0 * np.sum(q[0, :, None] * (bary[None, :] - cand[0]), axis=0)
    np.testing.assert_allclose(gradient, np.zeros(2), atol=1e-12)


def test_conditional_audit_preserves_context_and_shrinks_witness() -> None:
    operator = ToyOperator()
    projector = get_exact_projector(operator, dtype=torch.float64, use_cache=False)
    true_null = np.array([[0.0, 0.0, 1.0, -0.5]], dtype=np.float64)
    estimate = np.array([[0.0, 0.0, -0.25, 0.25]], dtype=np.float64)
    rows = [np.array([[0.0, 0.0, 1.0, 0.0], [0.2, 0.0, 0.0, 1.0]], dtype=np.float64)]
    audited, diag = conditional_nullspace_audit(estimate, true_null, rows, projector, lambda_=1e-9)
    update = audited - estimate
    context_update = operator.A_forward(torch.from_numpy(update).to(dtype=torch.float64)).numpy()
    np.testing.assert_allclose(context_update, np.zeros((1, 2)), atol=1e-10)
    before = np.linalg.norm((true_null[0] - estimate[0]) @ rows[0].T)
    after = np.linalg.norm((true_null[0] - audited[0]) @ rows[0].T)
    assert after < before * 1e-6
    assert diag["implementation"].startswith("matrix_free_P0")


def test_posterior_weights_favor_witness_consistent_candidate() -> None:
    cand = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    truth = np.array([1.0, 0.0], dtype=np.float64)
    rows = np.eye(2, dtype=np.float64)
    q = posterior_weights_for_rows(cand, truth, rows, None, alpha=0.0, tau=0.05)
    assert int(np.argmax(q)) == 1
    assert q[1] > 0.99


def test_gaussian_risk_utility_selects_discriminative_row() -> None:
    cand = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    q = np.array([1.0 / 3.0] * 3, dtype=np.float64)
    rows = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    util = gaussian_risk_utilities(cand, q, rows, sigma2=1e-6)
    assert int(np.argmax(util)) == 0
    order, trace = sequential_risk_order(
        cand,
        true_null_i=np.array([1.0, 0.0], dtype=np.float64),
        row_pool=rows,
        prior_scores_i=None,
        alpha=0.0,
        tau=0.1,
        max_budget=1,
        sigma2=1e-6,
    )
    assert order == [0]
    assert trace[0]["risk_after_observed_update"] < trace[0]["risk_before"]
