import numpy as np

from aggregate_frozen_fohi_heldout import (
    favorable_interval,
    hierarchical_test,
    holm_adjust,
)


def test_hierarchical_test_constant_favorable_deltas() -> None:
    deltas = {
        "psnr": [np.full(8, 0.2), np.full(8, 0.1), np.full(8, 0.3)],
        "ssim": [np.full(8, 0.02), np.full(8, 0.01), np.full(8, 0.03)],
        "lpips": [np.full(8, -0.2), np.full(8, -0.1), np.full(8, -0.3)],
    }
    result = hierarchical_test(deltas, reps=200, seed=7)
    assert favorable_interval("psnr", result["psnr"])
    assert favorable_interval("ssim", result["ssim"])
    assert favorable_interval("lpips", result["lpips"])
    assert result["psnr"]["one_sided_p"] == 1 / 201
    assert result["lpips"]["one_sided_p"] == 1 / 201


def test_holm_adjust_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust({"a": 0.001, "b": 0.01, "c": 0.04})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert adjusted == {"a": 0.003, "b": 0.02, "c": 0.04}
