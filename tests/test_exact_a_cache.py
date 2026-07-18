"""Exact-A / solver-cache discipline tests with deliberate violations
(incident (c): an exact-A override once left a stale Cholesky/solver cache)."""

import pytest
import torch

from src.exact_measurement import apply_measurement_override_from_config
from src.measurement import GhostMeasurementOperator, StaleSolverCacheError


@pytest.fixture()
def op():
    return GhostMeasurementOperator(
        img_size=8,
        sampling_ratio=0.25,
        pattern_type="rademacher",
        noise_std=0.0,
        lambda_dc=1e-3,
        device="cpu",
        seed=7,
    )


def test_init_asserts_cache_fresh(op):
    checks = op.assert_solver_cache_fresh()
    assert checks["rel_solve_fresh_vs_cached"] <= 1e-10
    assert checks["rel_K64_vs_fresh"] <= 1e-10
    assert checks["rel_cached_backward_residual"] <= 1e-10
    assert checks["rel_K32_vs_fresh"] <= 1e-4


def test_violation_stale_cache_after_raw_A_swap_detected(op):
    # Deliberate violation reproducing incident (c): replace A without going
    # through set_A_override -> the cached K/Cholesky is now stale.
    gen = torch.Generator().manual_seed(99)
    op.A = torch.randn(op.m, op.n, generator=gen)
    with pytest.raises(StaleSolverCacheError):
        op.assert_solver_cache_fresh()


def test_fresh_ill_conditioned_cache_is_judged_by_backward_error(op):
    # A fresh ill-conditioned system can make LU and Cholesky solutions differ
    # in forward error even though both solve the current K.  Freshness must be
    # based on K identity and backward residual, not cross-solver forward error.
    gen = torch.Generator().manual_seed(123)
    A = torch.randn(op.m, op.n, generator=gen)
    A[1] = A[0] + 1e-5 * A[1]
    op.set_A_override(A)
    checks = op.assert_solver_cache_fresh()
    assert checks["rel_K64_vs_fresh"] <= 1e-10
    assert checks["rel_cached_backward_residual"] <= 1e-10


def test_violation_rebuild_cache_false_is_forbidden(op):
    gen = torch.Generator().manual_seed(11)
    A2 = torch.randn(op.m, op.n, generator=gen)
    with pytest.raises(StaleSolverCacheError):
        op.set_A_override(A2, rebuild_cache=False)


def test_override_forces_rebuild_and_asserts(op):
    gen = torch.Generator().manual_seed(11)
    A2 = torch.randn(op.m, op.n, generator=gen)
    stats = op.set_A_override(A2)
    assert stats["cache_rebuilt"] is True
    assert stats["solver_cache_rel_solve"] <= 1e-10
    op.assert_solver_cache_fresh()
    # The runtime solve must now match a fresh float64 solve of the NEW K.
    b = torch.randn(3, op.m, generator=torch.Generator().manual_seed(5))
    z = op.solve_K(b)
    A64 = A2.to(torch.float64)
    K64 = A64 @ A64.T + op.lambda_dc * torch.eye(op.m, dtype=torch.float64)
    z_ref = torch.linalg.solve(K64, b.to(torch.float64).T).T
    rel = (z.to(torch.float64) - z_ref).norm() / z_ref.norm()
    assert rel < 1e-4  # float32 runtime precision against float64 reference


def test_override_with_different_m_rebuilds_consistently(op):
    gen = torch.Generator().manual_seed(13)
    A3 = torch.randn(op.m // 2, op.n, generator=gen)
    op.set_A_override(A3)
    assert op.m == A3.shape[0]
    assert op.K.shape == (op.m, op.m)
    op.assert_solver_cache_fresh()


def test_exact_file_override_path_end_to_end(op, tmp_path):
    gen = torch.Generator().manual_seed(21)
    A2 = torch.randn(op.m, op.n, generator=gen)
    exact_path = tmp_path / "measurement_operator_exact.pt"
    torch.save({"A": A2}, exact_path)
    info = apply_measurement_override_from_config(
        {"measurement_operator_exact_path": str(exact_path), "exact_A_required": True},
        op,
        "cpu",
    )
    assert info["exact_A_loaded"] is True
    assert info["cache_rebuilt"] is True
    op.assert_solver_cache_fresh()
    assert torch.allclose(op.A, A2)


def test_dc_project_audit_uses_fresh_cache_after_override(op):
    # Pi_y(v) = v - B_lambda(Av - y) must be computed against the NEW operator.
    gen = torch.Generator().manual_seed(31)
    A2 = torch.randn(op.m, op.n, generator=gen)
    op.set_A_override(A2)
    v = torch.randn(2, op.n, generator=torch.Generator().manual_seed(32))
    y = torch.randn(2, op.m, generator=torch.Generator().manual_seed(33))
    out = op.dc_project(v, y)
    A64 = A2.to(torch.float64)
    K64 = A64 @ A64.T + op.lambda_dc * torch.eye(op.m, dtype=torch.float64)
    resid = v.to(torch.float64) @ A64.T - y.to(torch.float64)
    ref = v.to(torch.float64) - torch.linalg.solve(K64, resid.T).T @ A64
    rel = (out.to(torch.float64) - ref).norm() / ref.norm()
    assert rel < 1e-4
