"""P0 projector builder tests with a deliberate violation."""

import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("build_p0", REPO_ROOT / "scripts" / "g2r" / "build_p0.py")
build_p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_p0)


def _synthetic_A(m=12, n=64, seed=3):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=gen)


def test_p0_verifies_on_synthetic_A():
    A = _synthetic_A()
    P0, rank = build_p0.build_p0_float64(A)
    assert rank == 12
    checks = build_p0.verify_p0_float64(A, P0)
    assert checks["pass"] is True
    assert checks["range_leak_max"] <= 1e-12
    assert checks["idempotence_fro"] <= 1e-10


def test_p0_orthonormal_rows_case():
    # Scrambled-Hadamard-like case: A with orthonormal rows -> P0 = I - A^T A.
    A = _synthetic_A(16, 64, seed=4)
    Q, _ = torch.linalg.qr(A.to(torch.float64).T)
    A_orth = Q.T[:16].to(torch.float32)
    P0, rank = build_p0.build_p0_float64(A_orth)
    assert rank == 16
    checks = build_p0.verify_p0_float64(A_orth, P0)
    assert checks["pass"] is True


def test_violation_corrupted_p0_fails_verification():
    # Deliberate violation: perturb P0 -> verification must FAIL.
    A = _synthetic_A()
    P0, _ = build_p0.build_p0_float64(A)
    gen = torch.Generator().manual_seed(8)
    P0_bad = P0 + 1e-6 * torch.randn(P0.shape, dtype=torch.float64, generator=gen)
    checks = build_p0.verify_p0_float64(A, P0_bad)
    assert checks["pass"] is False


def test_violation_identity_is_not_a_null_projector():
    A = _synthetic_A()
    I = torch.eye(A.shape[1], dtype=torch.float64)
    checks = build_p0.verify_p0_float64(A, I)
    assert checks["range_leak_pass"] is False
    assert checks["pass"] is False
