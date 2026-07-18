from __future__ import annotations

import torch

from src.fiber_ratio_barycenter import (
    effective_sample_size,
    oracle_simplex_weights,
    project_simplex,
    radial_bound,
)


def test_project_simplex_is_feasible_and_known_case() -> None:
    value = torch.tensor([[0.2, -0.1, 1.4], [1.0, 1.0, 1.0]], dtype=torch.float64)
    result = project_simplex(value)
    assert torch.all(result >= 0.0)
    assert torch.allclose(result.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(result[0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    assert torch.allclose(result[1], torch.full((3,), 1.0 / 3.0, dtype=torch.float64))


def test_oracle_simplex_recovers_exact_convex_target() -> None:
    particles = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]], dtype=torch.float32
    )
    target = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    weights, diagnostics = oracle_simplex_weights(particles, target, iterations=4096)
    fitted = torch.einsum("bk,bkn->bn", weights.float(), particles)
    assert torch.allclose(fitted, target, atol=1.0e-6, rtol=0.0)
    assert diagnostics["mean_oracle_null_mse"] < 1.0e-12


def test_radial_bound_and_effective_sample_size() -> None:
    residual = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    bounded = radial_bound(residual, 2.0)
    assert torch.allclose(torch.linalg.vector_norm(bounded, dim=1), torch.tensor([2.0, 0.0]))
    weights = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    assert torch.allclose(effective_sample_size(weights), torch.tensor([2.0, 1.0]))
