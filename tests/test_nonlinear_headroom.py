from __future__ import annotations

import numpy as np
import torch

import nonlinear_headroom as nh


def test_local_residual_with_zero_residual_reduces_to_lmmse() -> None:
    rng = np.random.default_rng(5)
    x_fit = rng.normal(size=(20, 16)).astype(np.float32)
    x = rng.normal(size=(4, 16)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(16, 8)))
    rows = q.T[:4].astype(np.float32)
    arm = nh.Arm("unit", 4, 1, rows, nh.MatrixFreeNullProjector(torch.from_numpy(rows)))
    fit = nh.fit_lmmse(x_fit)
    lm, _ = nh.lmmse_predict(x, arm, fit, 1e-4, device=torch.device("cpu"))
    calib_y = x_fit[:6].astype(np.float64) @ rows.astype(np.float64).T
    zero_res = np.zeros((6, 16), dtype=np.float32)
    sinv = np.eye(4)
    loc, rel = nh.local_residual_predict(x, arm, fit, 1e-4, calib_y, zero_res, sinv, k=3, device=torch.device("cpu"))
    np.testing.assert_allclose(loc, lm, atol=1e-5)
    assert float(rel.max()) < 1e-5


def test_mfa_single_component_is_measurement_consistent() -> None:
    rng = np.random.default_rng(9)
    x_fit = rng.normal(size=(24, 16)).astype(np.float32)
    x = rng.normal(size=(5, 16)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(16, 8)))
    rows = q.T[:5].astype(np.float32)
    arm = nh.Arm("unit", 5, 1, rows, nh.MatrixFreeNullProjector(torch.from_numpy(rows)))
    comps = nh.fit_mfa_components(x_fit, arm, k=1, lambda_=1e-4, seed=1)
    pred, rel = nh.mfa_predict(x, arm, comps, device=torch.device("cpu"))
    assert pred.shape == x.shape
    assert float(rel.max()) < 1e-5
