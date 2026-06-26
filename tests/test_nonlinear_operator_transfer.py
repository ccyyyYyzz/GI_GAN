from __future__ import annotations

import numpy as np
import torch

import nonlinear_operator_transfer as nlt


def test_lmmse_uses_low_rank_products_and_audit_preserves_measurements() -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(12, 16)).astype(np.float32)
    x_test = rng.normal(size=(5, 16)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(16, 8)))
    rows = q.T[:4].astype(np.float32)
    projector = nlt.MatrixFreeNullProjector(torch.from_numpy(rows))
    arm = nlt.TransferArm("unit_random_m4_op0", "random", 4, "test", 0, rows, projector, torch.zeros(1, 3, 4, 4))
    stats = nlt.fit_empirical_stats(x_train, klt_rank=4)
    pred, rel = nlt.lmmse_predict(x_test, arm, stats, lambda_=1e-4, device=torch.device("cpu"))
    assert pred.shape == x_test.shape
    assert float(rel.max()) < 1e-5


def test_clustered_bootstrap_clusters_by_image() -> None:
    rows = []
    for src in [1, 2, 3]:
        for arm in ["a", "b"]:
            rows.append({"source_index": src, "arm_id": arm, "family": "random", "budget": 21, "variant_seed": 0, "method": "m", "centered_rmse": 1.0})
            rows.append({"source_index": src, "arm_id": arm, "family": "random", "budget": 21, "variant_seed": "baseline", "method": "r", "centered_rmse": 2.0})
    out = nlt.clustered_bootstrap_delta(rows, "m", "r", "centered_rmse", variant_seed=0, reps=20, seed=3)
    assert out["n_images"] == 3
    assert out["n_image_operator_pairs"] == 6
    assert out["mean_delta"] == -1.0
