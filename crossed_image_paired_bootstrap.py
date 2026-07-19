"""Crossed-image paired bootstrap for fixed reconstruction lanes.

The three FOHI lanes reconstruct the *same* images under fixed, frozen
conditions.  They are not independent image cohorts and are not resampled as a
random factor here.  This module therefore averages the lane-wise paired
effects for each image first, then resamples one shared image-index vector per
bootstrap replicate.  It reports percentile confidence intervals only; it does
not compute bootstrap sign tails, p-values, or Holm adjustments.

The CLI accepts both the original ``laneN`` result trees and the ``results``
trees in a frozen release archive.  It deliberately writes to an explicit new
output path and never modifies a lane result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METRICS = ("psnr", "ssim", "lpips")
FAVORABLE_DIRECTION = {"psnr": "higher", "ssim": "higher", "lpips": "lower"}


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lane_index_from_path(path: Path) -> int:
    """Find a ``laneN`` identifier in a portable lane/release path."""
    for candidate in (path, *path.parents):
        match = re.search(r"(?:^|[_-])lane[_-]?(\d+)(?:$|[_-])", candidate.name)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"LANE_INDEX_NOT_ENCODED_IN_PATH:{path}")


def resolve_lane_result_root(path: Path) -> tuple[int, Path]:
    """Resolve original, release, or direct raw-fiber result layouts.

    Supported layouts are ``lane0/rate05/...``,
    ``lane0_frozen_fohi_release/results/rate05/...``, a direct ``results``
    child of such a release, and future raw-fiber result directories named
    ``laneN`` or nested under a ``laneN`` parent.
    """
    path = path.resolve()
    release_root = path if (path / "RELEASE_MANIFEST.json").is_file() else None
    if release_root is None and path.name == "results" and (
        path.parent / "RELEASE_MANIFEST.json"
    ).is_file():
        release_root = path.parent
    if release_root is not None:
        manifest = json.loads(
            (release_root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8")
        )
        if "lane_index" not in manifest:
            raise ValueError(f"RELEASE_MANIFEST_MISSING_LANE_INDEX:{release_root}")
        result_root = release_root / "results"
        if not result_root.is_dir():
            raise FileNotFoundError(f"RELEASE_RESULTS_DIRECTORY_MISSING:{release_root}")
        return int(manifest["lane_index"]), result_root

    if (path / "rate05").is_dir() or (path / "rate10").is_dir():
        return _lane_index_from_path(path), path
    if (path / "results" / "rate05").is_dir() or (path / "results" / "rate10").is_dir():
        return _lane_index_from_path(path), path / "results"
    raise FileNotFoundError(f"UNRECOGNIZED_LANE_RESULTS_LAYOUT:{path}")


def _coerce_lane_matrix(
    lane_deltas: Mapping[str, Sequence[np.ndarray] | np.ndarray]
) -> dict[str, np.ndarray]:
    """Validate and coerce metric effects to ``[fixed_lane, common_image]``."""
    matrices: dict[str, np.ndarray] = {}
    lane_count: int | None = None
    image_count: int | None = None
    for metric in METRICS:
        if metric not in lane_deltas:
            raise KeyError(f"MISSING_METRIC:{metric}")
        matrix = np.asarray(lane_deltas[metric], dtype=np.float64)
        if matrix.ndim != 2 or min(matrix.shape) == 0:
            raise ValueError(f"EXPECTED_NONEMPTY_2D_LANE_BY_IMAGE_MATRIX:{metric}:{matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"NONFINITE_DELTA:{metric}")
        if lane_count is None:
            lane_count, image_count = matrix.shape
        elif matrix.shape != (lane_count, image_count):
            raise ValueError(
                f"CROSSED_SHAPE_MISMATCH:{metric}:{matrix.shape}!={(lane_count, image_count)}"
            )
        matrices[metric] = matrix
    return matrices


def crossed_image_paired_bootstrap(
    lane_deltas: Mapping[str, Sequence[np.ndarray] | np.ndarray],
    *,
    reps: int,
    seed: int,
    bonferroni_family_size: int | None = None,
    family_confidence: float = 0.95,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Bootstrap fixed lanes crossed with a common set of images.

    Each replicate draws exactly one length-``n_images`` vector of image
    indices.  The same vector indexes every fixed lane, after which lane means
    are averaged.  Thus an image's repeated lane measurements remain paired.

    Optional Bonferroni bounds are directional one-sided percentile bounds for
    a declared family, not p-values or a multiplicity-adjusted hypothesis test.
    """
    if reps <= 0:
        raise ValueError("BOOTSTRAP_REPS_MUST_BE_POSITIVE")
    if batch_size <= 0:
        raise ValueError("BOOTSTRAP_BATCH_SIZE_MUST_BE_POSITIVE")
    if not 0.0 < family_confidence < 1.0:
        raise ValueError("FAMILY_CONFIDENCE_MUST_BE_IN_0_1")
    if bonferroni_family_size is not None and bonferroni_family_size <= 0:
        raise ValueError("BONFERRONI_FAMILY_SIZE_MUST_BE_POSITIVE")

    matrices = _coerce_lane_matrix(lane_deltas)
    lane_count, image_count = next(iter(matrices.values())).shape
    per_image = np.stack([matrices[metric].mean(axis=0) for metric in METRICS])
    draws = np.empty((len(METRICS), reps), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for start in range(0, reps, batch_size):
        stop = min(start + batch_size, reps)
        common_image_indices = rng.integers(0, image_count, size=(stop - start, image_count))
        draws[:, start:stop] = per_image[:, common_image_indices].mean(axis=2)

    result: dict[str, Any] = {
        "design": {
            "lanes": "fixed reconstruction conditions",
            "resampling_unit": "image",
            "crossed_pairing": "each bootstrap replicate uses one common image-index draw across all lanes",
            "lane_aggregation": "lane-wise paired effects are averaged for each image before resampling",
            "not_reported": ["bootstrap sign-tail fractions", "p_values", "Holm adjustments"],
        },
        "bootstrap": {
            "reps": int(reps),
            "seed": int(seed),
            "percentile_ci_level": 0.95,
            "fixed_lane_count": int(lane_count),
            "common_image_count": int(image_count),
        },
        "metrics": {},
    }
    for metric_index, metric in enumerate(METRICS):
        metric_draws = draws[metric_index]
        result["metrics"][metric] = {
            "direction": FAVORABLE_DIRECTION[metric],
            "mean_delta": float(per_image[metric_index].mean()),
            "ci95_percentile_low": float(np.quantile(metric_draws, 0.025)),
            "ci95_percentile_high": float(np.quantile(metric_draws, 0.975)),
        }

    if bonferroni_family_size is not None:
        alpha_per_bound = (1.0 - family_confidence) / bonferroni_family_size
        result["bonferroni_simultaneous_directional_bounds"] = {
            "family_confidence": float(family_confidence),
            "family_size": int(bonferroni_family_size),
            "per_bound_one_sided_alpha": float(alpha_per_bound),
            "interpretation": (
                "Bonferroni directional percentile bounds over the declared family; "
                "these bounds are not p-values and are not a hypothesis-test adjustment."
            ),
            "metrics": {},
        }
        for metric_index, metric in enumerate(METRICS):
            metric_draws = draws[metric_index]
            if FAVORABLE_DIRECTION[metric] == "higher":
                bound = float(np.quantile(metric_draws, alpha_per_bound))
                result["bonferroni_simultaneous_directional_bounds"]["metrics"][metric] = {
                    "favorable_bound": "lower",
                    "value": bound,
                    "favorable_if": "value > 0",
                }
            else:
                bound = float(np.quantile(metric_draws, 1.0 - alpha_per_bound))
                result["bonferroni_simultaneous_directional_bounds"]["metrics"][metric] = {
                    "favorable_bound": "upper",
                    "value": bound,
                    "favorable_if": "value < 0",
                }
    return result


def _load_rate_deltas(
    lane_roots: Sequence[tuple[int, Path]],
    *,
    rate: str,
    baseline_prefix: str,
    candidate_prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ordered_roots = sorted(lane_roots, key=lambda item: item[0])
    lane_ids = [lane for lane, _ in ordered_roots]
    if len(set(lane_ids)) != len(lane_ids):
        raise ValueError(f"DUPLICATE_LANE_INDEX:{lane_ids}")
    lane_deltas: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}
    sequence_hashes: list[str] = []
    inputs: list[dict[str, Any]] = []
    for lane_index, root in ordered_roots:
        vectors_path = root / f"rate{rate}" / "fohi" / "metric_vectors.npz"
        cache_manifest_path, cache_provenance = _resolve_cache_manifest(
            root=root, lane_index=lane_index, rate=rate
        )
        if not vectors_path.is_file():
            raise FileNotFoundError(f"MISSING_METRIC_VECTORS:lane{lane_index}:rate{rate}")
        manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        sequence_hash = manifest.get("test_raw_hash_sequence_sha256")
        if not isinstance(sequence_hash, str):
            raise ValueError(f"COMMON_IMAGE_SEQUENCE_HASH_MISSING:lane{lane_index}:rate{rate}")
        sequence_hashes.append(sequence_hash)
        with np.load(vectors_path) as archive:
            for metric in METRICS:
                baseline_key = f"{baseline_prefix}_{metric}"
                candidate_key = f"{candidate_prefix}_{metric}"
                if baseline_key not in archive or candidate_key not in archive:
                    raise KeyError(
                        f"MISSING_PAIRED_METRIC_KEYS:lane{lane_index}:rate{rate}:{baseline_key}:{candidate_key}"
                    )
                baseline = np.asarray(archive[baseline_key], dtype=np.float64)
                candidate = np.asarray(archive[candidate_key], dtype=np.float64)
                if baseline.shape != candidate.shape or baseline.ndim != 1:
                    raise ValueError(f"INVALID_PAIRED_VECTOR_SHAPES:lane{lane_index}:rate{rate}:{metric}")
                lane_deltas[metric].append(candidate - baseline)
        inputs.append(
            {
                "lane_index": lane_index,
                "metric_vectors_sha256": sha256(vectors_path),
                "test_cache_manifest_sha256": sha256(cache_manifest_path),
                **cache_provenance,
            }
        )
    if len(set(sequence_hashes)) != 1:
        raise ValueError(f"CROSSED_IMAGE_SEQUENCE_DRIFT:rate{rate}")
    matrices = {metric: np.stack(values) for metric, values in lane_deltas.items()}
    _coerce_lane_matrix(matrices)
    return matrices, {
        "lane_indices": lane_ids,
        "test_raw_hash_sequence_sha256": sequence_hashes[0],
        "inputs": inputs,
    }


def _resolve_cache_manifest(
    *, root: Path, lane_index: int, rate: str
) -> tuple[Path, dict[str, Any]]:
    """Resolve and verify the cache-manifest provenance for one lane and rate.

    Legacy and release trees carry ``rateXX/cache/test_cache_manifest.json``.
    Round59 deliberately does not copy that cache: its completion receipt points
    at the Round56 manifest that supplied the fixed held-out image sequence.
    The receipt is consequently an integrity boundary, not merely metadata.
    """
    round59_receipt_path = root / "ROUND59_COMPLETE.json"
    if not round59_receipt_path.is_file():
        manifest_path = root / f"rate{rate}" / "cache" / "test_cache_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"MISSING_RATE_INPUTS:lane{lane_index}:rate{rate}")
        return manifest_path, {"cache_provenance": "local_manifest"}

    receipt = json.loads(round59_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "ROUND59_RAW_FIBER_LANE_COMPLETE":
        raise ValueError(f"ROUND59_COMPLETE_RECEIPT_STATUS_MISMATCH:lane{lane_index}")
    if receipt.get("final_target") != "raw_y":
        raise ValueError(f"ROUND59_FINAL_TARGET_MISMATCH:lane{lane_index}")
    if int(receipt.get("lane_index", -1)) != lane_index:
        raise ValueError(f"ROUND59_COMPLETE_RECEIPT_LANE_MISMATCH:lane{lane_index}")

    reused_caches = receipt.get("reused_caches")
    if not isinstance(reused_caches, dict) or not isinstance(reused_caches.get(rate), dict):
        raise ValueError(f"ROUND59_REUSED_CACHE_MISSING:lane{lane_index}:rate{rate}")
    cache_record = reused_caches[rate]
    manifest_ref = cache_record.get("cache_manifest")
    expected_sha256 = cache_record.get("cache_manifest_sha256")
    if not isinstance(manifest_ref, str) or not isinstance(expected_sha256, str):
        raise ValueError(f"ROUND59_CACHE_MANIFEST_RECEIPT_MISSING:lane{lane_index}:rate{rate}")
    manifest_path = Path(manifest_ref)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ROUND59_REFERENCED_CACHE_MANIFEST_MISSING:lane{lane_index}:rate{rate}")
    actual_sha256 = sha256(manifest_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"ROUND59_REFERENCED_CACHE_MANIFEST_HASH_MISMATCH:lane{lane_index}:rate{rate}")
    return manifest_path, {
        "cache_provenance": "round59_reused_manifest",
        "round59_complete_receipt_sha256": sha256(round59_receipt_path),
        "round59_reused_cache_manifest_path": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rates", nargs="+", choices=("05", "10"), default=("05", "10"))
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--baseline-prefix", default="structural")
    parser.add_argument("--candidate-prefix", default="fohi")
    parser.add_argument(
        "--bonferroni-simultaneous",
        action="store_true",
        help="Add optional one-sided directional percentile bounds for all rate-by-metric effects.",
    )
    args = parser.parse_args()
    lane_roots = [resolve_lane_result_root(path) for path in args.input_dirs]
    rates = tuple(args.rates)
    family_size = len(rates) * len(METRICS) if args.bonferroni_simultaneous else None
    payload: dict[str, Any] = {
        "status": "CROSSED_IMAGE_PAIRED_BOOTSTRAP_SENSITIVITY",
        "scope": "read_only_statistical_sensitivity; does_not_modify_scientific_result_files",
        "baseline_prefix": args.baseline_prefix,
        "candidate_prefix": args.candidate_prefix,
        "rates": {},
    }
    for rate_index, rate in enumerate(rates):
        deltas, provenance = _load_rate_deltas(
            lane_roots,
            rate=rate,
            baseline_prefix=args.baseline_prefix,
            candidate_prefix=args.candidate_prefix,
        )
        statistics = crossed_image_paired_bootstrap(
            deltas,
            reps=args.bootstrap_reps,
            seed=args.bootstrap_seed + rate_index,
            bonferroni_family_size=family_size,
        )
        payload["rates"][rate] = {"provenance": provenance, "statistics": statistics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
