from __future__ import annotations

import torch

from src.compatibility_model import CompatibilityCritic, symmetric_infonce_loss


def test_compatibility_critic_shapes_and_loss() -> None:
    model = CompatibilityCritic(embed_dim=32, base_channels=8, temperature=0.1)
    r = torch.randn(6, 1, 32, 32)
    n = torch.randn(6, 1, 32, 32)
    zr, zn = model.forward_embeddings(r, n)
    assert tuple(zr.shape) == (6, 32)
    assert tuple(zn.shape) == (6, 32)
    assert torch.allclose(torch.linalg.norm(zr, dim=1), torch.ones(6), atol=1e-5)
    scores = model.score_matrix(r, n)
    assert tuple(scores.shape) == (6, 6)
    loss = symmetric_infonce_loss(scores)
    assert torch.isfinite(loss)
