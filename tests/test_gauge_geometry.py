from __future__ import annotations

import numpy as np
import pytest
import torch

from src.gauge_geometry import (
    GaugeEmpiricalAnchor,
    GaugeGeometry,
    _piqp_box_fiber_projection,
    project_box_fiber_exact_dual,
    project_box_fiber_q,
)


def test_gauge_anchor_is_invariant_to_invertible_row_reparameterization() -> None:
    rng = np.random.default_rng(17)
    rows = torch.from_numpy(rng.normal(size=(5, 16))).double()
    transform = torch.from_numpy(rng.normal(size=(5, 5))).double()
    transform = transform + 2.0 * torch.eye(5, dtype=torch.float64)
    changed_rows = transform @ rows
    train = rng.uniform(size=(80, 16)).astype(np.float32)
    truth = torch.from_numpy(rng.uniform(size=(4, 16))).double()
    y = truth @ rows.T
    changed_y = truth @ changed_rows.T
    geometry = GaugeGeometry(rows)
    changed_geometry = GaugeGeometry(changed_rows)
    anchor = GaugeEmpiricalAnchor.fit(train, geometry, lambda_=1e-3)
    changed_anchor = GaugeEmpiricalAnchor.fit(train, changed_geometry, lambda_=1e-3)

    native, native_z = anchor(y, geometry)
    changed, changed_z = changed_anchor(changed_y, changed_geometry)

    assert torch.max(torch.abs(native - changed)).item() < 1e-7
    assert geometry.relative_record_error(native, native_z).max().item() < 1e-10
    assert changed_geometry.relative_record_error(changed, changed_z).max().item() < 1e-10


def test_exact_q_dykstra_returns_bounded_fiber_point() -> None:
    torch.manual_seed(19)
    rows = torch.randn(6, 25, dtype=torch.float64)
    geometry = GaugeGeometry(rows)
    truth = torch.rand(3, 25, dtype=torch.float64)
    y = truth @ rows.T
    z = geometry.intrinsic_record(y)
    proposal = truth + 0.4 * torch.randn_like(truth)

    result = project_box_fiber_q(proposal, z, geometry, exact=True)

    assert result.converged
    assert result.max_relative_record_error <= 1e-7
    assert result.max_box_violation == 0.0


def test_exact_dual_projection_handles_boundary_active_solution() -> None:
    torch.manual_seed(23)
    rows = torch.randn(7, 36, dtype=torch.float64)
    geometry = GaugeGeometry(rows)
    feasible = torch.rand(4, 36, dtype=torch.float64)
    feasible[:, :5] = 0.0
    feasible[:, 5:9] = 1.0
    z = feasible @ geometry.Q.T
    proposal = geometry.affine_project_flat(
        feasible + 0.8 * torch.randn_like(feasible), z
    )

    result = project_box_fiber_exact_dual(proposal, z, geometry)

    assert result.converged
    assert result.max_relative_record_error <= 1e-10
    assert result.max_box_violation == 0.0
    assert result.iterations <= 64


def test_piqp_reference_projection_passes_primal_kkt_on_small_problem() -> None:
    pytest.importorskip("piqp")
    torch.manual_seed(29)
    geometry = GaugeGeometry(torch.randn(5, 25, dtype=torch.float64))
    feasible = torch.rand(2, 25, dtype=torch.float64)
    z = feasible @ geometry.Q.T
    proposal = geometry.affine_project_flat(
        feasible + 0.9 * torch.randn_like(feasible), z
    )

    image, dual = _piqp_box_fiber_projection(proposal, z, geometry)
    fixed_point = (proposal - dual @ geometry.Q).clamp(0.0, 1.0)

    torch.testing.assert_close(image, fixed_point, atol=1e-10, rtol=1e-10)
    assert geometry.relative_record_error(image, z).max().item() <= 1e-10
    assert image.min().item() >= 0.0
    assert image.max().item() <= 1.0
