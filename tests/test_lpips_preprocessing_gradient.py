from __future__ import annotations

import torch

import gan_high_quality_gi as hq


def test_prep_lpips_preserves_prediction_gradient() -> None:
    pred = torch.full((2, 1, 8, 8), 0.25, requires_grad=True)

    hq.prep_lpips(pred).square().mean().backward()

    assert pred.grad is not None
    assert torch.count_nonzero(pred.grad).item() > 0


def test_prep_lpips_can_detach_for_explicit_metric_use() -> None:
    pred = torch.full((1, 1, 8, 8), 0.5, requires_grad=True)

    prepared = hq.prep_lpips(pred, detach=True)

    assert prepared.requires_grad is False
