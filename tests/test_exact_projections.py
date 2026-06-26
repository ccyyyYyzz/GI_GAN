from __future__ import annotations

import pytest
import torch

from src.measurement import GhostMeasurementOperator
from src.projections import exact_data_anchor, exact_null_project, exact_row_project, get_exact_projector


@pytest.mark.parametrize(
    "pattern,matrix_normalization",
    [
        ("rademacher", "legacy_sqrt_m"),
        ("scrambled_hadamard", "orthonormal_rows"),
    ],
)
def test_exact_range_null_projectors(pattern: str, matrix_normalization: str) -> None:
    op = GhostMeasurementOperator(
        img_size=16,
        sampling_ratio=0.25,
        pattern_type=pattern,
        noise_std=0.0,
        lambda_dc=1e-3,
        matrix_normalization=matrix_normalization,
        device="cpu",
        seed=123,
    )
    gen = torch.Generator().manual_seed(5)
    v64 = torch.randn(5, op.n, dtype=torch.float64, generator=gen)
    projector = get_exact_projector(op, dtype=torch.float64, device="cpu", rcond=1e-12, use_cache=False)
    pr64 = projector.row_project_flat(v64)
    p064 = projector.null_project_flat(v64)
    ap0 = torch.linalg.norm(projector.A_forward(p064), dim=1) / torch.linalg.norm(projector.A_forward(v64), dim=1).clamp_min(1e-12)
    recon = torch.linalg.norm(pr64 + p064 - v64, dim=1) / torch.linalg.norm(v64, dim=1)
    dot = torch.sum(pr64 * p064, dim=1).abs() / (torch.linalg.norm(pr64, dim=1) * torch.linalg.norm(p064, dim=1)).clamp_min(1e-12)
    idem = torch.linalg.norm(projector.null_project_flat(p064) - p064, dim=1) / torch.linalg.norm(v64, dim=1)
    assert float(ap0.max()) < 1e-9
    assert float(recon.max()) < 1e-12
    assert float(dot.max()) < 1e-8
    assert float(idem.max()) < 1e-9

    y = projector.A_forward(v64)
    anchor = projector.data_anchor_flat(y)
    anchor_rel = torch.linalg.norm(projector.A_forward(anchor) - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)
    assert float(anchor_rel.max()) < 1e-9

    v32 = v64.float()
    pr32 = exact_row_project(v32, op, dtype=torch.float32, device="cpu")
    p032 = exact_null_project(v32, op, dtype=torch.float32, device="cpu")
    y32 = op.A_forward(v32)
    ap0_32 = torch.linalg.norm(op.A_forward(p032), dim=1) / torch.linalg.norm(y32, dim=1).clamp_min(1e-12)
    recon32 = torch.linalg.norm(pr32 + p032 - v32, dim=1) / torch.linalg.norm(v32, dim=1)
    assert float(ap0_32.max()) < 1e-5
    assert float(recon32.max()) < 1e-6


def test_exact_data_anchor_public_api_image_shape() -> None:
    op = GhostMeasurementOperator(img_size=16, sampling_ratio=0.25, pattern_type="rademacher", noise_std=0.0, device="cpu", seed=9)
    x = torch.randn(3, 1, 16, 16)
    y = op.A_forward(op.flatten_img(x))
    anchor_img = exact_data_anchor(y, op, as_image=True)
    assert tuple(anchor_img.shape) == (3, 1, 16, 16)
    rel = torch.linalg.norm(op.A_forward(op.flatten_img(anchor_img)) - y, dim=1) / torch.linalg.norm(y, dim=1).clamp_min(1e-12)
    assert float(rel.max()) < 1e-5
