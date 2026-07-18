from __future__ import annotations

import pytest
import torch

from src.vqae_centered_residual_adapter import (
    VQAECenteredResidualAdapter,
    radial_frequency_masks,
)


@pytest.mark.parametrize("architecture", ["spatial", "spectral", "global"])
def test_adapter_is_bounded_and_shape_preserving(architecture: str) -> None:
    torch.manual_seed(3)
    model = VQAECenteredResidualAdapter(
        architecture=architecture, maximum_weight=0.35, initial_weight=0.10
    )
    base = torch.rand(4, 1, 64, 64)
    direction = torch.randn_like(base)
    anchor = torch.rand_like(base)
    correction, weight = model(base, direction, anchor)
    assert correction.shape == direction.shape
    assert float(weight.min()) >= 0.0
    assert float(weight.max()) <= 0.35


def test_radial_masks_partition_rfft_plane() -> None:
    masks = radial_frequency_masks(
        64, 64, 6, device=torch.device("cpu"), dtype=torch.float32
    )
    assert masks.shape == (6, 64, 33)
    torch.testing.assert_close(masks.sum(dim=0), torch.ones(64, 33))
