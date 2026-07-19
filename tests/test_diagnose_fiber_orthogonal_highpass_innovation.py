from __future__ import annotations

import torch

from diagnose_fiber_orthogonal_highpass_innovation import final_projection_target
from src.gauge_geometry import GaugeGeometry


def test_raw_y_final_target_is_explicit_and_differs_from_legacy_anchor_target() -> None:
    geometry = GaugeGeometry.from_rows_qr(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    )
    raw_y = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    split = {
        "raw_y": raw_y,
        "intrinsic": torch.tensor([[0.0, 0.0]], dtype=torch.float64),
    }

    raw_target, raw_receipt = final_projection_target(
        split, geometry, mode="raw_y"
    )
    legacy_target, legacy_receipt = final_projection_target(
        split, geometry, mode="legacy_clipped_anchor"
    )

    torch.testing.assert_close(raw_target, geometry.intrinsic_record(raw_y))
    assert not torch.equal(raw_target, legacy_target)
    assert raw_receipt["target_is_cached_raw_y"] is True
    assert legacy_receipt["target_is_cached_raw_y"] is False
