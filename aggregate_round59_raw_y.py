"""Fail-closed final aggregation for the three Round59 raw-bucket lanes.

Round59 is a fixed re-evaluation after correcting the final physical
constraint from the legacy clipped anchor to the cached raw bucket record. It
is not described here as preregistered before the held-out test was opened.
The program reads only extracted ``lane0``, ``lane1``, and ``lane2`` Round59
directories and writes a new decision directory; it never changes a lane.

The three lanes are fixed reconstruction conditions on one common held-out
image sequence.  Consequently the statistical resampling unit is the image,
not a random lane factor.  For each rate, the implementation reuses
``crossed_image_paired_bootstrap``: it averages lane-wise paired effects for
each image, then applies one shared image-index draw to every lane and metric.
It reports 95% percentile intervals and six one-sided Bonferroni directional
bounds, with no p-values or Holm adjustments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from crossed_image_paired_bootstrap import (
    FAVORABLE_DIRECTION,
    METRICS,
    crossed_image_paired_bootstrap,
)


RATES = ("05", "10")
EXPECTED_LANES = (0, 1, 2)
EXPECTED_IMAGES = 6740
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260719
FAMILY_SIZE = len(RATES) * len(METRICS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ROUND59_{label}_MISSING:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"ROUND59_{label}_OBJECT_REQUIRED:{path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    _require(actual == expected, f"{message}:{actual!r}!={expected!r}")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def resolve_round59_lane(path: Path) -> tuple[int, Path]:
    """Resolve one extracted Round59 lane directory, without guessing layout."""
    root = path.resolve()
    receipt = root / "ROUND59_COMPLETE.json"
    if not receipt.is_file():
        raise FileNotFoundError(f"ROUND59_COMPLETE_RECEIPT_MISSING:{root}")
    complete = read_object(receipt, label="COMPLETE_RECEIPT")
    lane = complete.get("lane_index")
    if not isinstance(lane, int):
        raise ValueError(f"ROUND59_COMPLETE_LANE_NOT_INTEGER:{root}")
    return lane, root


def _projection_audit_converged(summary: Mapping[str, Any], arm: str) -> bool:
    audit = summary.get(f"{arm}_projection_audit")
    if not isinstance(audit, dict):
        return False
    return bool(audit.get("all_converged") is True)


def _raw_measurement_certificate_passed(summary: Mapping[str, Any], arm: str) -> bool:
    certificates = summary.get("raw_measurement_residual_certificate")
    if not isinstance(certificates, dict):
        return False
    certificate = certificates.get(arm)
    return isinstance(certificate, dict) and certificate.get("passed") is True


def _cache_receipt_is_valid(cache: Any, *, lane: int, rate: str) -> bool:
    if not isinstance(cache, dict):
        return False
    required_hashes = ("cache_sha256", "cache_manifest_sha256")
    return bool(
        cache.get("test_images") == EXPECTED_IMAGES
        and cache.get("included_development_raw_hash_overlap") == 0
        and isinstance(cache.get("cache_manifest"), str)
        and all(_valid_sha256(cache.get(key)) for key in required_hashes)
    )


def _resolve_reused_cache_manifest(
    *, lane_root: Path, cache: Mapping[str, Any], lane: int, rate: str
) -> tuple[Path, str]:
    """Resolve only the receipt path or the one declared portable fallback.

    A Round59 archive normally records an absolute Colab path to the reused
    Round56 manifest.  The archive packer may preserve that external manifest
    at ``reused_cache_manifests/rateXX/test_cache_manifest.json``.  No search
    across a machine or arbitrary sibling directory is permitted.
    """
    receipt_path = Path(str(cache["cache_manifest"]))
    if receipt_path.is_file():
        return receipt_path, "receipt_path"
    fallback = lane_root / "reused_cache_manifests" / f"rate{rate}" / "test_cache_manifest.json"
    if fallback.is_file():
        return fallback, "portable_fallback"
    raise FileNotFoundError(
        f"ROUND59_REFERENCED_CACHE_MANIFEST_MISSING:lane{lane}:rate{rate}:"
        f"receipt={receipt_path}:fallback={fallback}"
    )


def _verify_lane_rate(
    *, lane: int, root: Path, rate: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Verify one raw-y lane/rate and return structural-to-FOHI deltas."""
    complete_path = root / "ROUND59_COMPLETE.json"
    preflight_path = root / "preflight_receipt.json"
    complete = read_object(complete_path, label="COMPLETE_RECEIPT")
    preflight = read_object(preflight_path, label="PREFLIGHT_RECEIPT")

    _require_equal(complete.get("status"), "ROUND59_RAW_FIBER_LANE_COMPLETE", "ROUND59_COMPLETE_STATUS")
    _require_equal(complete.get("lane_index"), lane, "ROUND59_COMPLETE_LANE")
    _require_equal(complete.get("final_target"), "raw_y", "ROUND59_COMPLETE_FINAL_TARGET")
    _require_equal(complete.get("evaluation_scope"), "heldout", "ROUND59_COMPLETE_SCOPE")
    _require_equal(preflight.get("status"), "ROUND59_RAW_FIBER_PREFLIGHT_VERIFIED", "ROUND59_PREFLIGHT_STATUS")
    _require_equal(preflight.get("lane_index"), lane, "ROUND59_PREFLIGHT_LANE")

    # This is deliberately receipt-to-receipt: downloaded Round59 archives do
    # not duplicate the multi-GB cache and frozen model files.  Their hashes
    # are nevertheless carried in both authenticated launch receipts.
    _require_equal(complete.get("code_sha256"), preflight.get("current_code_sha256"), "ROUND59_CODE_RECEIPT_INCONSISTENT")
    _require_equal(
        complete.get("frozen_weight_and_config_sha256"),
        preflight.get("frozen_weight_and_config_sha256"),
        "ROUND59_WEIGHT_RECEIPT_INCONSISTENT",
    )
    _require_equal(complete.get("reused_caches"), preflight.get("reused_caches"), "ROUND59_CACHE_RECEIPT_INCONSISTENT")
    _require(
        isinstance(complete.get("code_sha256"), dict)
        and bool(complete["code_sha256"])
        and all(_valid_sha256(value) for value in complete["code_sha256"].values()),
        "ROUND59_CODE_RECEIPT_INVALID",
    )
    _require(
        isinstance(complete.get("frozen_weight_and_config_sha256"), dict)
        and bool(complete["frozen_weight_and_config_sha256"])
        and all(
            _valid_sha256(value)
            for value in complete["frozen_weight_and_config_sha256"].values()
        ),
        "ROUND59_WEIGHT_RECEIPT_INVALID",
    )

    cache = complete.get("reused_caches", {}).get(rate) if isinstance(complete.get("reused_caches"), dict) else None
    _require(_cache_receipt_is_valid(cache, lane=lane, rate=rate), f"ROUND59_CACHE_RECEIPT_INVALID:lane{lane}:rate{rate}")
    assert isinstance(cache, dict)  # narrows type for the provenance record
    cache_manifest_path, cache_manifest_source = _resolve_reused_cache_manifest(
        lane_root=root, cache=cache, lane=lane, rate=rate
    )
    _require_equal(
        sha256(cache_manifest_path),
        cache["cache_manifest_sha256"],
        f"ROUND59_REFERENCED_CACHE_MANIFEST_HASH_MISMATCH:lane{lane}:rate{rate}",
    )
    cache_manifest = read_object(cache_manifest_path, label="REFERENCED_CACHE_MANIFEST")
    _require_equal(
        cache_manifest.get("test_images"), EXPECTED_IMAGES,
        f"ROUND59_REFERENCED_CACHE_IMAGE_COUNT_INVALID:lane{lane}:rate{rate}",
    )
    _require_equal(
        cache_manifest.get("included_development_raw_hash_overlap"), 0,
        f"ROUND59_REFERENCED_CACHE_OVERLAP_INVALID:lane{lane}:rate{rate}",
    )
    raw_hash_sequence = cache_manifest.get("test_raw_hash_sequence_sha256")
    _require(
        _valid_sha256(raw_hash_sequence),
        f"ROUND59_REFERENCED_CACHE_RAW_HASH_SEQUENCE_MISSING:lane{lane}:rate{rate}",
    )

    rate_receipt = complete.get("rates", {}).get(rate) if isinstance(complete.get("rates"), dict) else None
    _require(isinstance(rate_receipt, dict), f"ROUND59_RATE_RECEIPT_MISSING:lane{lane}:rate{rate}")
    _require_equal(rate_receipt.get("reused_cache"), cache, f"ROUND59_RATE_CACHE_RECEIPT_INCONSISTENT:lane{lane}:rate{rate}")

    result_dir = root / f"rate{rate}" / "fohi"
    summary_path = result_dir / "summary.json"
    vectors_path = result_dir / "metric_vectors.npz"
    summary = read_object(summary_path, label="SUMMARY")
    _require_equal(rate_receipt.get("summary_sha256"), sha256(summary_path), f"ROUND59_SUMMARY_RECEIPT_HASH_MISMATCH:lane{lane}:rate{rate}")
    _require_equal(rate_receipt.get("metric_vectors_sha256"), sha256(vectors_path), f"ROUND59_VECTORS_RECEIPT_HASH_MISMATCH:lane{lane}:rate{rate}")

    _require_equal(summary.get("status"), "FIBER_ORTHOGONAL_HIGHPASS_INNOVATION_DIAGNOSTIC", f"ROUND59_SUMMARY_STATUS:lane{lane}:rate{rate}")
    _require_equal(summary.get("final_target"), "raw_y", f"ROUND59_SUMMARY_FINAL_TARGET:lane{lane}:rate{rate}")
    _require_equal(summary.get("evaluation_scope"), "heldout", f"ROUND59_SUMMARY_SCOPE:lane{lane}:rate{rate}")
    _require(summary.get("validation_only") is False, f"ROUND59_SUMMARY_VALIDATION_ONLY:lane{lane}:rate{rate}")
    _require(summary.get("test_split_opened") is True, f"ROUND59_SUMMARY_TEST_NOT_OPENED:lane{lane}:rate{rate}")
    _require_equal(summary.get("evaluation_images"), EXPECTED_IMAGES, f"ROUND59_SUMMARY_EVALUATION_COUNT:lane{lane}:rate{rate}")
    _require_equal(summary.get("heldout_images"), EXPECTED_IMAGES, f"ROUND59_SUMMARY_HELDOUT_COUNT:lane{lane}:rate{rate}")

    for arm in ("structural", "fixed", "fohi"):
        _require(_projection_audit_converged(summary, arm), f"ROUND59_PROJECTION_NOT_CONVERGED:lane{lane}:rate{rate}:{arm}")
        _require(_raw_measurement_certificate_passed(summary, arm), f"ROUND59_RAW_MEASUREMENT_NOT_CERTIFIED:lane{lane}:rate{rate}:{arm}")

    deltas: dict[str, np.ndarray] = {}
    with np.load(vectors_path) as archive:
        for metric in METRICS:
            structural_key = f"structural_{metric}"
            fohi_key = f"fohi_{metric}"
            _require(structural_key in archive and fohi_key in archive, f"ROUND59_METRIC_VECTOR_KEYS_MISSING:lane{lane}:rate{rate}:{metric}")
            structural = np.asarray(archive[structural_key], dtype=np.float64)
            fohi = np.asarray(archive[fohi_key], dtype=np.float64)
            _require(structural.ndim == 1 and fohi.ndim == 1 and structural.shape == fohi.shape, f"ROUND59_METRIC_VECTOR_SHAPE_INVALID:lane{lane}:rate{rate}:{metric}")
            _require(len(structural) == EXPECTED_IMAGES, f"ROUND59_METRIC_VECTOR_COUNT_INVALID:lane{lane}:rate{rate}:{metric}")
            _require(np.all(np.isfinite(structural)) and np.all(np.isfinite(fohi)), f"ROUND59_METRIC_VECTOR_NONFINITE:lane{lane}:rate{rate}:{metric}")
            deltas[metric] = fohi - structural

    return deltas, {
        "lane_index": lane,
        "complete_sha256": sha256(complete_path),
        "preflight_sha256": sha256(preflight_path),
        "summary_sha256": sha256(summary_path),
        "metric_vectors_sha256": sha256(vectors_path),
        "test_raw_hash_sequence_sha256": raw_hash_sequence,
        "cache_sha256": cache["cache_sha256"],
        "cache_manifest_sha256": cache["cache_manifest_sha256"],
        "cache_manifest_source": cache_manifest_source,
        "projection_and_raw_measurement_certificates_passed": True,
    }


def _favorable_ci(metric: str, item: Mapping[str, Any]) -> bool:
    return bool(item["ci95_percentile_high"] < 0.0) if FAVORABLE_DIRECTION[metric] == "lower" else bool(item["ci95_percentile_low"] > 0.0)


def _favorable_bound(metric: str, item: Mapping[str, Any]) -> bool:
    return bool(item["value"] < 0.0) if FAVORABLE_DIRECTION[metric] == "lower" else bool(item["value"] > 0.0)


def _favorable_mean(metric: str, value: float) -> bool:
    return bool(value < 0.0) if FAVORABLE_DIRECTION[metric] == "lower" else bool(value > 0.0)


def aggregate_round59_raw_y(
    input_dirs: Sequence[Path], *, bootstrap_reps: int = BOOTSTRAP_REPS, bootstrap_seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Aggregate three complete raw-y lanes into a serializable decision payload."""
    _require(len(input_dirs) == len(EXPECTED_LANES), "ROUND59_EXACTLY_THREE_LANES_REQUIRED")
    lanes = [resolve_round59_lane(path) for path in input_dirs]
    lane_ids = [lane for lane, _ in lanes]
    _require(sorted(lane_ids) == list(EXPECTED_LANES), f"ROUND59_LANES_MUST_BE_0_1_2:{lane_ids}")
    _require(len(set(lane_ids)) == len(lane_ids), f"ROUND59_DUPLICATE_LANES:{lane_ids}")
    _require(int(bootstrap_reps) > 0, "ROUND59_BOOTSTRAP_REPS_MUST_BE_POSITIVE")

    ordered_lanes = sorted(lanes)
    per_rate: dict[str, Any] = {}
    input_hashes: dict[str, Any] = {}
    sequence_hashes: set[str] = set()
    all_lane_means_favorable = True

    for rate in RATES:
        metric_lanes: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}
        rate_lanes: list[dict[str, Any]] = []
        for lane, root in ordered_lanes:
            deltas, provenance = _verify_lane_rate(lane=lane, root=root, rate=rate)
            sequence_hashes.add(provenance["test_raw_hash_sequence_sha256"])
            means = {metric: float(deltas[metric].mean()) for metric in METRICS}
            favorable = all(_favorable_mean(metric, means[metric]) for metric in METRICS)
            all_lane_means_favorable = all_lane_means_favorable and favorable
            for metric in METRICS:
                metric_lanes[metric].append(deltas[metric])
            rate_lanes.append({
                "lane_index": lane,
                "means": means,
                "all_three_means_favorable": favorable,
                "certificates_passed": True,
            })
            input_hashes[f"lane{lane}/rate{rate}"] = provenance

        # Same seed deliberately produces the same common image-index draws at
        # both rates; lanes remain fixed rather than being resampled.
        statistics = crossed_image_paired_bootstrap(
            metric_lanes,
            reps=int(bootstrap_reps),
            seed=int(bootstrap_seed),
            bonferroni_family_size=FAMILY_SIZE,
            family_confidence=0.95,
        )
        metrics = statistics["metrics"]
        bounds = statistics["bonferroni_simultaneous_directional_bounds"]["metrics"]
        ci_pass = all(_favorable_ci(metric, metrics[metric]) for metric in METRICS)
        bound_pass = all(_favorable_bound(metric, bounds[metric]) for metric in METRICS)
        per_rate[rate] = {
            "crossed_fixed_lane_statistics": statistics,
            "all_three_95_percent_intervals_favorable": ci_pass,
            "all_three_bonferroni_directional_bounds_favorable": bound_pass,
            "all_nine_rate_lane_metric_means_favorable": all(item["all_three_means_favorable"] for item in rate_lanes),
            "per_lane": rate_lanes,
        }

    _require(len(sequence_hashes) == 1, "ROUND59_TEST_RAW_HASH_SEQUENCE_DRIFT_ACROSS_LANES_OR_RATES")
    six_ci_favorable = all(per_rate[rate]["all_three_95_percent_intervals_favorable"] for rate in RATES)
    six_bound_favorable = all(per_rate[rate]["all_three_bonferroni_directional_bounds_favorable"] for rate in RATES)
    certificates_and_hashes_passed = True  # fail-closed checks above must all return first
    headline_pass = bool(six_ci_favorable and six_bound_favorable and all_lane_means_favorable and certificates_and_hashes_passed)

    return {
        "status": "ROUND59_RAW_Y_THREE_LANE_FINAL_REEVALUATION",
        "evaluation_scope": "heldout",
        "final_target": "raw_y",
        "test_images": EXPECTED_IMAGES,
        "test_raw_hash_sequence_sha256": next(iter(sequence_hashes)),
        "description": (
            "Physical-constraint-corrected fixed re-evaluation: the final projection targets cached raw bucket measurements. "
            "This is not represented as preregistered before the held-out test was opened."
        ),
        "statistical_design": {
            "lanes": "three fixed reconstruction conditions",
            "resampling_unit": "held-out image",
            "common_image_sampling": "one shared image-index draw is applied across all fixed lanes and all metrics within each rate",
            "bootstrap_reps": int(bootstrap_reps),
            "bootstrap_seed": int(bootstrap_seed),
            "confidence_interval": "95% percentile interval",
            "multiplicity_control": "six one-sided Bonferroni directional percentile bounds; familywise confidence 0.95",
            "not_reported": ["p_values", "Holm adjustments"],
        },
        "decision_rules": {
            "six_ci_gate": "all six rate-by-metric 95% intervals are favorable",
            "six_bonferroni_gate": "all six one-sided Bonferroni directional bounds are favorable",
            "eighteen_lane_mean_gate": "all 18 rate-by-lane-by-metric means are favorable",
            "integrity_gate": "all receipt hashes, raw-hash sequence identities, projection convergence audits, and raw-measurement certificates pass",
        },
        "headline_all_gates_passed": headline_pass,
        "six_ci_gate_passed": six_ci_favorable,
        "six_bonferroni_directional_bound_gate_passed": six_bound_favorable,
        "all_18_lane_metric_means_favorable": all_lane_means_favorable,
        "all_integrity_certificates_and_hashes_passed": certificates_and_hashes_passed,
        "rates": per_rate,
        "input_hashes": input_hashes,
        "post_test_tuning_permitted": False,
    }


def write_decision(output_dir: Path, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "round59_raw_y_decision.json"
    markdown_path = output_dir / "ROUND59_RAW_Y_DECISION.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Round59 raw-y three-lane final re-evaluation",
        "",
        f"All fixed decision gates: **{'PASS' if payload['headline_all_gates_passed'] else 'FAIL'}**.",
        "",
        "Round59 is a physical-constraint-corrected fixed re-evaluation using cached raw bucket measurements; it is not represented as preregistered before the held-out test was opened.",
        "",
        "| Rate | ΔPSNR, 95% CI | ΔSSIM, 95% CI | ΔLPIPS, 95% CI | six-bound gate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rate in RATES:
        statistics = payload["rates"][rate]["crossed_fixed_lane_statistics"]
        metrics = statistics["metrics"]
        line_metrics = []
        for metric in METRICS:
            item = metrics[metric]
            line_metrics.append(f"{item['mean_delta']:+.6f} [{item['ci95_percentile_low']:+.6f}, {item['ci95_percentile_high']:+.6f}]")
        passed = payload["rates"][rate]["all_three_bonferroni_directional_bounds_favorable"]
        lines.append(f"| {int(rate)}% | {line_metrics[0]} | {line_metrics[1]} | {line_metrics[2]} | {'PASS' if passed else 'FAIL'} |")
    lines.extend([
        "",
        "The decision requires favorable 95% intervals and Bonferroni directional bounds for all six rate-by-metric effects, favorable means for all 18 lane-by-rate-by-metric effects, and passing receipt/hash and raw-measurement certification checks.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True, help="Extracted Round59 lane0, lane1, and lane2 directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    payload = aggregate_round59_raw_y(args.input_dirs, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.bootstrap_seed)
    json_path, markdown_path = write_decision(args.output_dir, payload)
    print(json.dumps({"decision_json": str(json_path), "decision_markdown": str(markdown_path), "headline_all_gates_passed": payload["headline_all_gates_passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
