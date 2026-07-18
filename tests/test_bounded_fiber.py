from __future__ import annotations

import torch

from src.bounded_fiber import project_box_and_fiber
from src.projections import ExactRangeNullProjector


class _TinyOperator:
    def __init__(self, rows: torch.Tensor) -> None:
        self.A = rows
        self.img_size = 4


def test_dykstra_delivers_the_scored_image_inside_box_and_on_fiber() -> None:
    torch.manual_seed(7)
    rows = torch.randn(5, 16, dtype=torch.float64)
    truth = torch.rand(3, 16, dtype=torch.float64)
    y = truth @ rows.T
    proposal = 2.5 * torch.randn_like(truth)
    projector = ExactRangeNullProjector(_TinyOperator(rows), dtype=torch.float64)

    result = project_box_and_fiber(
        proposal,
        y,
        projector,
        max_iterations=2000,
        measurement_tolerance=1e-8,
    )

    assert result.converged
    assert result.max_measurement_residual <= 1e-8
    assert result.max_box_violation == 0.0
    assert torch.all(result.image_flat >= 0.0)
    assert torch.all(result.image_flat <= 1.0)


def test_fixed_unrolling_remains_differentiable() -> None:
    torch.manual_seed(11)
    rows = torch.randn(3, 16, dtype=torch.float64)
    truth = torch.rand(1, 16, dtype=torch.float64)
    y = truth @ rows.T
    proposal = torch.full((1, 16), 0.4, dtype=torch.float64, requires_grad=True)
    projector = ExactRangeNullProjector(_TinyOperator(rows), dtype=torch.float64)

    result = project_box_and_fiber(
        proposal,
        y,
        projector,
        max_iterations=4,
        stop_early=False,
    )
    result.image_flat.square().mean().backward()

    assert proposal.grad is not None
    assert torch.isfinite(proposal.grad).all()
