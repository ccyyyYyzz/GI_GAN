from __future__ import annotations

import numpy as np
import torch

from src.bayesian_witness import conditional_nullspace_audit
from src.dc_balanced import (
    build_dc_balanced_rows,
    dc_row,
    dct_lowfreq_non_dc_rows,
    hadamard_lowsequency_non_dc_rows,
    random_zero_mean_rows,
    row_audit,
)
from src.projections import exact_data_anchor, get_exact_projector


class TinyOperator:
    def __init__(self, A: np.ndarray, img_size: int = 4) -> None:
        self.A = torch.as_tensor(A, dtype=torch.float64)
        self.m = int(self.A.shape[0])
        self.n = int(self.A.shape[1])
        self.img_size = int(img_size)

    def flatten_img(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    def unflatten_img(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], 1, self.img_size, self.img_size)


def test_dc_balanced_rows_are_unit_norm_and_non_dc_zero_mean() -> None:
    dim = 64 * 64
    dc = dc_row(dim)
    np.testing.assert_allclose(np.linalg.norm(dc), 1.0, atol=1e-7)
    np.testing.assert_allclose(dc.mean(), 1.0 / np.sqrt(dim), atol=1e-7)

    families = {
        "random": random_zero_mean_rows(8, dim, seed=123),
        "dct": dct_lowfreq_non_dc_rows(8, img_size=64),
        "hadamard": hadamard_lowsequency_non_dc_rows(8, dim),
    }
    for name, rows in families.items():
        assert rows.shape == (8, dim), name
        np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(rows.mean(axis=1), 0.0, atol=1e-7)
        np.testing.assert_allclose(rows @ dc, 0.0, atol=1e-6)


def test_row_audit_passes_constructed_designs() -> None:
    dim = 64 * 64
    for kind in ["random", "dct", "hadamard"]:
        rows = build_dc_balanced_rows(kind, 10, dim=dim, img_size=64, seed=777)
        audit = row_audit(rows, name=kind)
        assert audit["dc_balanced_pass"], audit
        assert audit["shape"] == [11, dim]
        assert audit["non_dc_row_mean_max_abs"] < 1e-6
        assert audit["non_dc_dc_dot_max_abs"] < 1e-5


def test_conditional_audit_preserves_context_and_matches_joint_minimum_norm() -> None:
    n = 16
    A_c = np.eye(n, dtype=np.float64)[:3]
    W = np.eye(n, dtype=np.float64)[3:7]
    x = torch.linspace(-0.5, 1.0, n, dtype=torch.float64).reshape(1, n)

    context = TinyOperator(A_c)
    combined = TinyOperator(np.concatenate([A_c, W], axis=0))
    projector = get_exact_projector(context, dtype=torch.float64, device="cpu", use_cache=False)

    y_c = x @ torch.as_tensor(A_c, dtype=torch.float64).T
    y_b = x @ torch.as_tensor(combined.A, dtype=torch.float64).T
    anchor = exact_data_anchor(y_c, context, dtype=torch.float64, device="cpu", as_image=False).numpy()
    true_null = projector.null_project_flat(x).numpy()

    audited_null, diag = conditional_nullspace_audit(
        np.zeros_like(true_null),
        true_null,
        [W.astype(np.float32)],
        projector,
        lambda_=0.0,
    )
    x_plus = torch.as_tensor(anchor + audited_null, dtype=torch.float64)
    joint = exact_data_anchor(y_b, combined, dtype=torch.float64, device="cpu", as_image=False)

    np.testing.assert_allclose((x_plus @ torch.as_tensor(A_c).T).numpy(), y_c.numpy(), atol=1e-10)
    np.testing.assert_allclose(x_plus.numpy(), joint.numpy(), atol=1e-10)
    assert diag["max_context_A_update_norm"] < 1e-10
    assert diag["max_witness_residual_after"] < 1e-10
