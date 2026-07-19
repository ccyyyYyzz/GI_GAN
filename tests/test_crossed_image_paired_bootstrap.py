import json

import numpy as np
import pytest

from crossed_image_paired_bootstrap import (
    _load_rate_deltas,
    crossed_image_paired_bootstrap,
    resolve_lane_result_root,
    sha256,
)


def _constant_metrics(value: float) -> dict[str, np.ndarray]:
    return {
        "psnr": np.full((3, 4), value),
        "ssim": np.full((3, 4), value / 10),
        "lpips": np.full((3, 4), -value),
    }


def test_crossed_bootstrap_averages_fixed_lanes_before_common_image_resampling() -> None:
    # The lane effects vary but their per-image average is exactly one.  A
    # nested lane/image bootstrap would vary; crossed fixed-lane resampling must
    # remain exactly one for all percentile endpoints.
    lanes = np.array([[0.0, 2.0], [2.0, 0.0]])
    result = crossed_image_paired_bootstrap(
        {"psnr": lanes, "ssim": lanes, "lpips": -lanes}, reps=101, seed=3
    )
    assert result["bootstrap"]["fixed_lane_count"] == 2
    assert result["bootstrap"]["common_image_count"] == 2
    assert result["metrics"]["psnr"] == {
        "direction": "higher",
        "mean_delta": 1.0,
        "ci95_percentile_low": 1.0,
        "ci95_percentile_high": 1.0,
    }
    assert result["metrics"]["lpips"]["mean_delta"] == -1.0
    assert result["metrics"]["lpips"]["ci95_percentile_low"] == -1.0
    assert result["metrics"]["lpips"]["ci95_percentile_high"] == -1.0
    assert "p_values" in result["design"]["not_reported"]


def test_crossed_bootstrap_is_seed_reproducible_and_exposes_optional_bounds() -> None:
    lanes = _constant_metrics(0.2)
    first = crossed_image_paired_bootstrap(
        lanes, reps=37, seed=11, bonferroni_family_size=6
    )
    second = crossed_image_paired_bootstrap(
        lanes, reps=37, seed=11, bonferroni_family_size=6
    )
    assert first == second
    bounds = first["bonferroni_simultaneous_directional_bounds"]
    assert bounds["family_size"] == 6
    assert bounds["metrics"]["psnr"]["favorable_bound"] == "lower"
    assert bounds["metrics"]["psnr"]["favorable_if"] == "value > 0"
    assert bounds["metrics"]["psnr"]["value"] == pytest.approx(0.2)
    assert bounds["metrics"]["lpips"]["favorable_bound"] == "upper"
    assert bounds["metrics"]["lpips"]["favorable_if"] == "value < 0"
    assert bounds["metrics"]["lpips"]["value"] == pytest.approx(-0.2)


def test_crossed_bootstrap_rejects_noncommon_shapes() -> None:
    with pytest.raises(ValueError, match="CROSSED_SHAPE_MISMATCH"):
        crossed_image_paired_bootstrap(
            {
                "psnr": np.zeros((2, 3)),
                "ssim": np.zeros((2, 4)),
                "lpips": np.zeros((2, 3)),
            },
            reps=3,
            seed=1,
        )


def test_resolve_lane_result_root_accepts_release_and_raw_fiber_layouts(tmp_path) -> None:
    raw = tmp_path / "future_raw_fiber_lane7"
    (raw / "rate05").mkdir(parents=True)
    assert resolve_lane_result_root(raw) == (7, raw)

    release = tmp_path / "lane2_frozen_release"
    (release / "results" / "rate05").mkdir(parents=True)
    (release / "RELEASE_MANIFEST.json").write_text(
        json.dumps({"lane_index": 2}), encoding="utf-8"
    )
    assert resolve_lane_result_root(release) == (2, release / "results")
    assert resolve_lane_result_root(release / "results") == (2, release / "results")


def _write_round59_lane(tmp_path, lane: int, *, sequence_hash: str = "shared"):
    source_manifest = tmp_path / "round56" / f"lane{lane}" / "rate05" / "cache" / "test_cache_manifest.json"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(
        json.dumps({"test_raw_hash_sequence_sha256": sequence_hash}), encoding="utf-8"
    )
    root = tmp_path / "gan_r59_raw_fiber" / f"lane{lane}"
    vectors_path = root / "rate05" / "fohi" / "metric_vectors.npz"
    vectors_path.parent.mkdir(parents=True)
    np.savez(
        vectors_path,
        structural_psnr=np.array([1.0, 2.0]),
        fohi_psnr=np.array([2.0, 3.0]),
        structural_ssim=np.array([0.1, 0.2]),
        fohi_ssim=np.array([0.2, 0.3]),
        structural_lpips=np.array([0.7, 0.6]),
        fohi_lpips=np.array([0.6, 0.5]),
    )
    receipt = {
        "status": "ROUND59_RAW_FIBER_LANE_COMPLETE",
        "lane_index": lane,
        "final_target": "raw_y",
        "reused_caches": {
            "05": {
                "cache_manifest": str(source_manifest),
                "cache_manifest_sha256": sha256(source_manifest),
            }
        },
    }
    (root / "ROUND59_COMPLETE.json").write_text(json.dumps(receipt), encoding="utf-8")
    return root, receipt, source_manifest


def test_load_rate_deltas_accepts_round59_reused_cache_manifest(tmp_path) -> None:
    roots = [_write_round59_lane(tmp_path, lane)[0] for lane in range(3)]
    deltas, provenance = _load_rate_deltas(
        [(lane, root) for lane, root in enumerate(roots)],
        rate="05",
        baseline_prefix="structural",
        candidate_prefix="fohi",
    )
    assert deltas["psnr"].shape == (3, 2)
    assert np.all(deltas["psnr"] == 1.0)
    assert provenance["test_raw_hash_sequence_sha256"] == "shared"
    assert all(item["cache_provenance"] == "round59_reused_manifest" for item in provenance["inputs"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "NOT_COMPLETE", "ROUND59_COMPLETE_RECEIPT_STATUS_MISMATCH"),
        ("final_target", "projected_y", "ROUND59_FINAL_TARGET_MISMATCH"),
    ],
)
def test_round59_receipt_rejects_invalid_identity(tmp_path, field, value, message) -> None:
    root, receipt, _ = _write_round59_lane(tmp_path, 0)
    receipt[field] = value
    (root / "ROUND59_COMPLETE.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load_rate_deltas(
            [(0, root)], rate="05", baseline_prefix="structural", candidate_prefix="fohi"
        )


def test_round59_receipt_rejects_referenced_manifest_hash_drift(tmp_path) -> None:
    root, _, manifest = _write_round59_lane(tmp_path, 0)
    manifest.write_text(json.dumps({"test_raw_hash_sequence_sha256": "changed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="ROUND59_REFERENCED_CACHE_MANIFEST_HASH_MISMATCH"):
        _load_rate_deltas(
            [(0, root)], rate="05", baseline_prefix="structural", candidate_prefix="fohi"
        )
