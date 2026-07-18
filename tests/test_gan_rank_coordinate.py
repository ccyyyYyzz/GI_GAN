from __future__ import annotations

import math

import torch

from diagnose_gan_rank_coordinate_pilot import (
    augmented_box_projection_many,
    rank_walsh_queries,
    sequency_walsh_rows,
)
from src.gauge_geometry import GaugeGeometry


def test_sequency_walsh_rows_are_balanced_and_orthonormal() -> None:
    rows = sequency_walsh_rows(64, 12)
    assert torch.allclose(rows.sum(1), torch.zeros(12), atol=1.0e-6)
    assert torch.allclose(rows @ rows.T, torch.eye(12), atol=1.0e-6)
    assert int((rows[0, 1:] != rows[0, :-1]).sum()) == 1


def test_rank_pullback_preserves_binary_balance_and_gram() -> None:
    guide = torch.Generator().manual_seed(7)
    values = torch.rand(3, 64, generator=guide)
    queries = rank_walsh_queries(values, 8)
    assert queries.shape == (3, 8, 64)
    assert torch.allclose(queries.sum(2), torch.zeros(3, 8), atol=1.0e-6)
    gram = queries @ queries.transpose(1, 2)
    assert torch.allclose(gram, torch.eye(8).expand(3, -1, -1), atol=1.0e-6)
    assert set(torch.unique(queries).tolist()) == {-1.0 / 8.0, 1.0 / 8.0}


def test_many_bucket_projection_enforces_old_and_new_records() -> None:
    n = 16
    rows = torch.ones(1, n, dtype=torch.float64) / math.sqrt(n)
    geometry = GaugeGeometry(rows)
    truth = torch.linspace(0.2, 0.8, n).reshape(1, 1, 4, 4)
    y = truth.flatten(1).double() @ rows.T
    intrinsic = geometry.intrinsic_record(y)
    guide = torch.arange(n, dtype=torch.float32).reshape(1, n)
    queries = rank_walsh_queries(guide, 2)
    records = torch.einsum("bkn,bn->bk", queries, truth.flatten(1))
    proposal = torch.full_like(truth, 0.5)

    result, audit = augmented_box_projection_many(
        proposal,
        intrinsic,
        queries,
        records,
        geometry,
        iterations=64,
    )

    flat = result.flatten(1)
    assert torch.allclose(flat.double() @ rows.T, y, atol=2.0e-6, rtol=0.0)
    assert torch.allclose(
        torch.einsum("bkn,bn->bk", queries, flat), records, atol=2.0e-6, rtol=0.0
    )
    assert audit["box_violation"] <= 1.0e-6
