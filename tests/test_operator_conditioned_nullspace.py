from __future__ import annotations

import torch

from src.operator_conditioned_nullspace import MatrixFreeNullProjector, SmallNullspaceUNet, reconstruct_with_projected_residual


def test_matrix_free_null_projector_annihilates_rows_without_dense_p0() -> None:
    torch.manual_seed(7)
    q, _ = torch.linalg.qr(torch.randn(32, 16), mode="reduced")
    rows = q.T[:5].contiguous()
    projector = MatrixFreeNullProjector(rows)
    v = torch.randn(4, 32)
    n = projector.null_project(v)
    assert torch.linalg.norm(projector.measurement(n)).item() < 1e-5
    r = projector.row_project(v)
    torch.testing.assert_close(r + n, v, atol=1e-5, rtol=1e-5)


def test_projected_predictor_keeps_measurement_consistency() -> None:
    torch.manual_seed(11)
    q, _ = torch.linalg.qr(torch.randn(16, 12), mode="reduced")
    rows = q.T[:6].contiguous()
    projector = MatrixFreeNullProjector(rows)
    x = torch.randn(3, 16)
    y = projector.measurement(x)
    r = projector.data_anchor(y)
    model = SmallNullspaceUNet(in_channels=2, base_channels=8, blocks=1)
    out = reconstruct_with_projected_residual(model, projector, r, y, img_size=4, cond_scalar=6 / 32)
    assert float(out.relmeaserr.max()) < 1e-5
    torch.testing.assert_close(projector.measurement(out.null_hat), torch.zeros_like(y), atol=1e-5, rtol=1e-5)
