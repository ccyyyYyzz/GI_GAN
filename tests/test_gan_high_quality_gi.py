from __future__ import annotations

import numpy as np
import torch

from gan_high_quality_gi import (
    EmpiricalLMMSE,
    build_structured_operator_rows,
    make_measurement_operator,
    nullspace_reconstruct,
)


def test_structured_operator_has_fixed_budget_and_dc_row() -> None:
    rows, meta = build_structured_operator_rows(
        img_size=64,
        total_m=41,
        dct_rows=20,
        hadamard_rows=10,
        random_rows=10,
        seed=123,
    )
    assert rows.shape == (41, 4096)
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(rows[0].mean(), 1.0 / np.sqrt(4096), atol=1e-7)
    np.testing.assert_allclose(rows[1:].mean(axis=1), 0.0, atol=1e-7)
    assert meta["total_m"] == 41


def test_lmmse_anchor_exact_audit_and_nullspace_reconstruction() -> None:
    device = torch.device("cpu")
    rows, _ = build_structured_operator_rows(
        img_size=8,
        total_m=8,
        dct_rows=3,
        hadamard_rows=2,
        random_rows=2,
        seed=7,
    )
    measurement = make_measurement_operator(rows, img_size=8, device=device, lambda_solver=1e-10)
    rng = np.random.default_rng(5)
    train = rng.random((20, 64), dtype=np.float32)
    lmmse = EmpiricalLMMSE.fit(train, rows, lambda_=1e-4)
    x = torch.from_numpy(rng.random((4, 64), dtype=np.float32)).to(device)
    y = x @ torch.from_numpy(rows).to(device).T
    x0 = lmmse.anchor(y, measurement, device=device)
    rel = torch.linalg.norm(x0 @ measurement.A.T - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)
    assert float(rel.max()) < 1e-7

    delta = torch.randn_like(x0)
    xhat = nullspace_reconstruct(x0, delta, measurement)
    rel_hat = torch.linalg.norm(xhat @ measurement.A.T - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)
    assert float(rel_hat.max()) < 1e-6

