from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from aggregate_round59_raw_y import (
    EXPECTED_IMAGES,
    aggregate_round59_raw_y,
    sha256,
    write_decision,
)


def _hash(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _summary(*, lane: int, rate: str, raw_certificate: bool = True) -> dict:
    audit = {"all_converged": True, "max_box_violation": 0.0, "max_relative_record_error": 0.0}
    return {
        "status": "FIBER_ORTHOGONAL_HIGHPASS_INNOVATION_DIAGNOSTIC",
        "final_target": "raw_y",
        "evaluation_scope": "heldout",
        "validation_only": False,
        "test_split_opened": True,
        "evaluation_images": EXPECTED_IMAGES,
        "heldout_images": EXPECTED_IMAGES,
        "structural_projection_audit": audit,
        "fixed_projection_audit": audit,
        "fohi_projection_audit": audit,
        "raw_measurement_residual_certificate": {
            arm: {"passed": raw_certificate, "target": "cached_raw_y"}
            for arm in ("structural", "fixed", "fohi")
        },
        "lane": lane,
        "rate": rate,
    }


def _write_lane(tmp_path: Path, lane: int, *, sequence_hash: str = "a" * 64, raw_certificate: bool = True) -> Path:
    root = tmp_path / f"lane{lane}"
    root.mkdir()
    code = {"diagnose.py": _hash(f"code-{lane}")}
    weights = {"proposal.pt": _hash(f"weights-{lane}")}
    reused_caches = {}
    for rate in ("05", "10"):
        cache_manifest_path = (
            tmp_path / "round56_cache_manifests" / f"lane{lane}" / f"rate{rate}" / "test_cache_manifest.json"
        )
        cache_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        cache_manifest_path.write_text(
            json.dumps(
                {
                    "test_images": EXPECTED_IMAGES,
                    "included_development_raw_hash_overlap": 0,
                    "test_raw_hash_sequence_sha256": sequence_hash,
                }
            ),
            encoding="utf-8",
        )
        reused_caches[rate] = {
            "cache": f"/content/round56/lane{lane}/rate{rate}/cache/test_cache.pt",
            "cache_sha256": _hash(f"cache-{lane}-{rate}"),
            "cache_manifest": str(cache_manifest_path),
            "cache_manifest_sha256": sha256(cache_manifest_path),
            "test_images": EXPECTED_IMAGES,
            "included_development_raw_hash_overlap": 0,
        }
    preflight = {
        "status": "ROUND59_RAW_FIBER_PREFLIGHT_VERIFIED",
        "lane_index": lane,
        "current_code_sha256": code,
        "frozen_weight_and_config_sha256": weights,
        "reused_caches": reused_caches,
    }
    (root / "preflight_receipt.json").write_text(json.dumps(preflight), encoding="utf-8")

    rates = {}
    for rate_index, rate in enumerate(("05", "10")):
        result = root / f"rate{rate}" / "fohi"
        result.mkdir(parents=True)
        summary = _summary(lane=lane, rate=rate, raw_certificate=raw_certificate)
        summary_path = result / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        # All arms are deliberately different.  FOHI improves every metric.
        n = EXPECTED_IMAGES
        structural_psnr = np.full(n, 20.0 + lane)
        structural_ssim = np.full(n, 0.50 + 0.01 * lane)
        structural_lpips = np.full(n, 0.40 - 0.01 * lane)
        vectors_path = result / "metric_vectors.npz"
        np.savez_compressed(
            vectors_path,
            structural_psnr=structural_psnr,
            fohi_psnr=structural_psnr + 0.10 + 0.01 * rate_index,
            structural_ssim=structural_ssim,
            fohi_ssim=structural_ssim + 0.01,
            structural_lpips=structural_lpips,
            fohi_lpips=structural_lpips - 0.01,
        )
        rates[rate] = {
            "summary": str(summary_path),
            "summary_sha256": sha256(summary_path),
            "metric_vectors": str(vectors_path),
            "metric_vectors_sha256": sha256(vectors_path),
            "reused_cache": reused_caches[rate],
        }
    complete = {
        "status": "ROUND59_RAW_FIBER_LANE_COMPLETE",
        "lane_index": lane,
        "final_target": "raw_y",
        "evaluation_scope": "heldout",
        "method_parameters": {"final_target": "raw_y"},
        "code_sha256": code,
        "frozen_weight_and_config_sha256": weights,
        "reused_caches": reused_caches,
        "rates": rates,
    }
    (root / "ROUND59_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    return root


def _complete_fixture(tmp_path: Path) -> list[Path]:
    return [_write_lane(tmp_path, lane) for lane in range(3)]


def _move_receipt_manifests_to_portable_fallback(roots: list[Path]) -> None:
    """Simulate a downloaded archive where the original /content path is gone."""
    for root in roots:
        for filename in ("preflight_receipt.json", "ROUND59_COMPLETE.json"):
            path = root / filename
            payload = json.loads(path.read_text(encoding="utf-8"))
            for rate in ("05", "10"):
                original = Path(payload["reused_caches"][rate]["cache_manifest"])
                fallback = root / "reused_cache_manifests" / f"rate{rate}" / "test_cache_manifest.json"
                fallback.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, fallback)
                payload["reused_caches"][rate]["cache_manifest"] = f"/content/unavailable/lane{root.name[-1]}/rate{rate}/test_cache_manifest.json"
                if filename == "ROUND59_COMPLETE.json":
                    payload["rates"][rate]["reused_cache"]["cache_manifest"] = payload["reused_caches"][rate]["cache_manifest"]
            path.write_text(json.dumps(payload), encoding="utf-8")


def test_round59_aggregator_checks_complete_fixture_and_writes_concise_artifacts(tmp_path: Path) -> None:
    roots = _complete_fixture(tmp_path)
    payload = aggregate_round59_raw_y(roots, bootstrap_reps=31, bootstrap_seed=7)
    assert payload["headline_all_gates_passed"] is True
    assert payload["all_18_lane_metric_means_favorable"] is True
    assert payload["test_images"] == EXPECTED_IMAGES
    assert payload["test_raw_hash_sequence_sha256"] == "a" * 64
    assert payload["statistical_design"]["not_reported"] == ["p_values", "Holm adjustments"]
    for rate in ("05", "10"):
        statistics = payload["rates"][rate]["crossed_fixed_lane_statistics"]
        assert statistics["bootstrap"]["fixed_lane_count"] == 3
        assert statistics["bootstrap"]["common_image_count"] == EXPECTED_IMAGES
        assert payload["rates"][rate]["all_three_95_percent_intervals_favorable"] is True
        assert payload["rates"][rate]["all_three_bonferroni_directional_bounds_favorable"] is True
    json_path, markdown_path = write_decision(tmp_path / "decision", payload)
    assert json.loads(json_path.read_text(encoding="utf-8"))["headline_all_gates_passed"] is True
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "not represented as preregistered" in markdown
    assert "p-value" not in markdown.lower()
    assert "holm" not in markdown.lower()


def test_round59_aggregator_rejects_summary_receipt_hash_drift(tmp_path: Path) -> None:
    roots = _complete_fixture(tmp_path)
    (roots[0] / "rate05" / "fohi" / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ROUND59_SUMMARY_RECEIPT_HASH_MISMATCH"):
        aggregate_round59_raw_y(roots, bootstrap_reps=3)


def test_round59_aggregator_accepts_only_the_hash_checked_portable_manifest_fallback(tmp_path: Path) -> None:
    roots = _complete_fixture(tmp_path)
    _move_receipt_manifests_to_portable_fallback(roots)
    payload = aggregate_round59_raw_y(roots, bootstrap_reps=3)
    assert payload["headline_all_gates_passed"] is True
    assert {
        item["cache_manifest_source"] for item in payload["input_hashes"].values()
    } == {"portable_fallback"}


def test_round59_aggregator_rejects_missing_receipt_manifest_and_missing_portable_fallback(tmp_path: Path) -> None:
    roots = _complete_fixture(tmp_path)
    for filename in ("preflight_receipt.json", "ROUND59_COMPLETE.json"):
        path = roots[0] / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rate in ("05", "10"):
            payload["reused_caches"][rate]["cache_manifest"] = "/content/absent/test_cache_manifest.json"
            if filename == "ROUND59_COMPLETE.json":
                payload["rates"][rate]["reused_cache"]["cache_manifest"] = "/content/absent/test_cache_manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="ROUND59_REFERENCED_CACHE_MANIFEST_MISSING"):
        aggregate_round59_raw_y(roots, bootstrap_reps=3)


def test_round59_aggregator_rejects_raw_hash_sequence_drift(tmp_path: Path) -> None:
    roots = _complete_fixture(tmp_path)
    changed = roots[2].parents[0] / "round56_cache_manifests" / "lane2" / "rate05" / "test_cache_manifest.json"
    changed.write_text(
        json.dumps(
            {
                "test_images": EXPECTED_IMAGES,
                "included_development_raw_hash_overlap": 0,
                "test_raw_hash_sequence_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    for filename in ("preflight_receipt.json", "ROUND59_COMPLETE.json"):
        path = roots[2] / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rate in ("05", "10"):
            manifest_path = Path(payload["reused_caches"][rate]["cache_manifest"])
            if rate == "10":
                manifest_path.write_text(
                    json.dumps(
                        {
                            "test_images": EXPECTED_IMAGES,
                            "included_development_raw_hash_overlap": 0,
                            "test_raw_hash_sequence_sha256": "b" * 64,
                        }
                    ),
                    encoding="utf-8",
                )
            payload["reused_caches"][rate]["cache_manifest_sha256"] = sha256(manifest_path)
            if filename == "ROUND59_COMPLETE.json":
                payload["rates"][rate]["reused_cache"]["cache_manifest_sha256"] = sha256(manifest_path)
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ROUND59_TEST_RAW_HASH_SEQUENCE_DRIFT"):
        aggregate_round59_raw_y(roots, bootstrap_reps=3)


def test_round59_aggregator_rejects_failed_raw_measurement_certificate(tmp_path: Path) -> None:
    roots = [_write_lane(tmp_path, lane, raw_certificate=(lane != 1)) for lane in range(3)]
    # Refresh the changed lane's receipt hash so the test reaches certificate validation.
    root = roots[1]
    complete_path = root / "ROUND59_COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    summary_path = root / "rate05" / "fohi" / "summary.json"
    complete["rates"]["05"]["summary_sha256"] = sha256(summary_path)
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ROUND59_RAW_MEASUREMENT_NOT_CERTIFIED"):
        aggregate_round59_raw_y(roots, bootstrap_reps=3)


def test_round59_aggregator_rejects_missing_or_wrong_lanes(tmp_path: Path) -> None:
    roots = _complete_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="ROUND59_EXACTLY_THREE_LANES_REQUIRED"):
        aggregate_round59_raw_y(roots[:2], bootstrap_reps=3)
    with pytest.raises(RuntimeError, match="ROUND59_LANES_MUST_BE_0_1_2"):
        aggregate_round59_raw_y([roots[0], roots[1], roots[1]], bootstrap_reps=3)
