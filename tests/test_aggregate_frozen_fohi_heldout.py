import json

import numpy as np

from aggregate_frozen_fohi_heldout import (
    canonical_json_sha256,
    favorable_interval,
    hierarchical_test,
    holm_adjust,
    resolve_lane_input,
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


def test_canonical_json_hash_ignores_newlines_and_key_order() -> None:
    left = json.loads('{\r\n  "b": 2,\r\n  "a": 1\r\n}')
    right = json.loads('{"a":1,"b":2}')
    assert canonical_json_sha256(left) == canonical_json_sha256(right)


def test_resolve_lane_input_accepts_original_and_release_layouts(tmp_path) -> None:
    original = tmp_path / "lane1"
    original.mkdir()
    assert resolve_lane_input(original) == (1, original, original)

    release = tmp_path / "lane2_frozen_fohi_release"
    (release / "results").mkdir(parents=True)
    (release / "receipts").mkdir()
    (release / "RELEASE_MANIFEST.json").write_text(
        json.dumps({"status": "FROZEN_FOHI_LANE_RELEASE_COMPLETE", "lane_index": 2}),
        encoding="utf-8",
    )
    assert resolve_lane_input(release) == (2, release / "results", release / "receipts")
    assert resolve_lane_input(release / "results") == (
        2,
        release / "results",
        release / "receipts",
    )
