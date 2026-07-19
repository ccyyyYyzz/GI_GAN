from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aggregate_endpoint_fohi import METRICS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    """Hash parsed JSON independently of checkout newline conventions."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_lane_input(path: Path) -> tuple[int, Path, Path]:
    """Resolve either an original lane tree or an extracted release tree.

    Returns ``(lane_index, result_root, receipt_root)``.  Release archives keep
    immutable receipts separate from result arrays; original Colab trees keep
    both at the lane root.
    """
    release_root = None
    if (path / "RELEASE_MANIFEST.json").is_file():
        release_root = path
    elif path.name == "results" and (path.parent / "RELEASE_MANIFEST.json").is_file():
        release_root = path.parent
    if release_root is not None:
        release = json.loads(
            (release_root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8")
        )
        if release.get("status") != "FROZEN_FOHI_LANE_RELEASE_COMPLETE":
            raise RuntimeError(f"INVALID_FOHI_RELEASE_STATUS:{release_root}")
        lane_index = int(release["lane_index"])
        result_root = release_root / "results"
        receipt_root = release_root / "receipts"
        if not result_root.is_dir() or not receipt_root.is_dir():
            raise FileNotFoundError(f"INCOMPLETE_FOHI_RELEASE_LAYOUT:{release_root}")
        return lane_index, result_root, receipt_root
    name = path.name
    if not (len(name) == 5 and name.startswith("lane") and name[-1] in "012"):
        raise ValueError(f"UNRECOGNIZED_HELDOUT_LANE_LAYOUT:{path}")
    return int(name[-1]), path, path


def projection_certified(summary: dict[str, Any]) -> bool:
    return all(
        audit["all_converged"]
        and audit["max_box_violation"] == 0.0
        and audit["max_relative_record_error"] < 1.0e-7
        for audit in (
            summary["structural_projection_audit"],
            summary["fohi_projection_audit"],
        )
    )


def hierarchical_test(
    deltas: dict[str, list[np.ndarray]], *, reps: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    lane_count = len(next(iter(deltas.values())))
    draws = {metric: np.empty(int(reps), dtype=np.float64) for metric in METRICS}
    for replicate in range(int(reps)):
        selected_lanes = rng.integers(0, lane_count, size=lane_count)
        for metric in METRICS:
            lane_means = []
            for selected in selected_lanes:
                values = deltas[metric][int(selected)]
                indices = rng.integers(0, len(values), size=len(values))
                lane_means.append(float(values[indices].mean()))
            draws[metric][replicate] = float(np.mean(lane_means))
    result = {}
    for metric in METRICS:
        values = np.concatenate(deltas[metric])
        unfavorable = draws[metric] <= 0.0 if metric != "lpips" else draws[metric] >= 0.0
        result[metric] = {
            "mean_delta": float(values.mean()),
            "ci95_low": float(np.quantile(draws[metric], 0.025)),
            "ci95_high": float(np.quantile(draws[metric], 0.975)),
            "one_sided_p": float((1 + unfavorable.sum()) / (int(reps) + 1)),
        }
    return result


def favorable_interval(metric: str, item: dict[str, float]) -> bool:
    return bool(item["ci95_high"] < 0.0) if metric == "lpips" else bool(item["ci95_low"] > 0.0)


def favorable_mean(metric: str, value: float) -> bool:
    return bool(value < 0.0) if metric == "lpips" else bool(value > 0.0)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        raw = min(1.0, (total - rank) * p_values[name])
        running = max(running, raw)
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input_dirs) != 3:
        raise ValueError("EXACTLY_THREE_HELDOUT_LANES_REQUIRED")
    lane_inputs = [resolve_lane_input(path) for path in args.input_dirs]
    lane_indices = [item[0] for item in lane_inputs]
    if sorted(lane_indices) != [0, 1, 2]:
        raise ValueError(f"HELDOUT_LANES_MUST_BE_EXACTLY_0_1_2:{lane_indices}")
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("status") != "VQGAN_GUIDED_FOHI_HELDOUT_FROZEN":
        raise RuntimeError("HELDOUT_FREEZE_MANIFEST_INVALID")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_images = int(freeze["expected_test_images"])
    per_rate: dict[str, Any] = {}
    input_hashes = {}
    split_hashes = set()
    all_certified = True
    all_scope_correct = True
    all_receipt_hashes_match = True
    for rate_index, rate in enumerate(("05", "10")):
        deltas = {metric: [] for metric in METRICS}
        per_lane = []
        for lane_index, result_root, receipt_root in lane_inputs:
            complete_path = receipt_root / "HELDOUT_ONCE_COMPLETE.json"
            summary_path = result_root / f"rate{rate}" / "fohi" / "summary.json"
            vectors_path = result_root / f"rate{rate}" / "fohi" / "metric_vectors.npz"
            cache_manifest_path = result_root / f"rate{rate}" / "cache" / "test_cache_manifest.json"
            for path in (complete_path, summary_path, vectors_path, cache_manifest_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            if not (
                complete.get("status") == "VQGAN_GUIDED_FOHI_HELDOUT_LANE_COMPLETE"
                and complete.get("lane_index") == lane_index
                and complete.get("evaluation_scope") == "heldout"
                and complete.get("validation_only") is False
            ):
                raise RuntimeError(f"HELDOUT_COMPLETE_RECEIPT_INVALID:{lane_index}")
            receipt_rate = complete.get("rates", {}).get(rate, {})
            receipt_hashes_match = bool(
                receipt_rate.get("summary_sha256") == sha256(summary_path)
                and receipt_rate.get("metric_vectors_sha256") == sha256(vectors_path)
                and receipt_rate.get("test_cache_manifest_sha256")
                == sha256(cache_manifest_path)
            )
            if not receipt_hashes_match:
                raise RuntimeError(f"HELDOUT_RESULT_RECEIPT_HASH_MISMATCH:{lane_index}:{rate}")
            all_receipt_hashes_match = all_receipt_hashes_match and receipt_hashes_match
            scope_correct = bool(
                complete.get("test_split_opened") is True
                and complete.get("validation_only") is False
                and summary.get("evaluation_scope") == "heldout"
                and summary.get("test_split_opened") is True
                and summary.get("heldout_images") == expected_images
                and cache_manifest.get("source_split") == "test"
                and cache_manifest.get("test_images") == expected_images
                and cache_manifest.get("included_development_raw_hash_overlap") == 0
            )
            all_scope_correct = all_scope_correct and scope_correct
            expected_operator = freeze["lanes"][str(lane_index)]["rates"][rate]["operator_sha256"]
            if summary.get("operator_sha256") != expected_operator:
                raise RuntimeError(f"HELDOUT_OPERATOR_DRIFT:{lane_index}:{rate}")
            split_hashes.add(cache_manifest["test_raw_hash_sequence_sha256"])
            archive = np.load(vectors_path)
            structural = {
                metric: np.asarray(archive[f"structural_{metric}"], dtype=np.float64)
                for metric in METRICS
            }
            fohi = {
                metric: np.asarray(archive[f"fohi_{metric}"], dtype=np.float64)
                for metric in METRICS
            }
            means = {}
            for metric in METRICS:
                if not (
                    np.all(np.isfinite(structural[metric]))
                    and np.all(np.isfinite(fohi[metric]))
                ):
                    raise RuntimeError(
                        f"HELDOUT_NONFINITE_METRIC:{lane_index}:{rate}:{metric}"
                    )
                delta = fohi[metric] - structural[metric]
                if len(delta) != expected_images:
                    raise RuntimeError(f"HELDOUT_VECTOR_LENGTH_MISMATCH:{lane_index}:{rate}:{metric}")
                deltas[metric].append(delta)
                means[metric] = float(delta.mean())
            certified = projection_certified(summary)
            all_certified = all_certified and certified
            per_lane.append(
                {
                    "lane_index": lane_index,
                    "means": means,
                    "all_means_favorable": all(
                        favorable_mean(metric, means[metric]) for metric in METRICS
                    ),
                    "projection_certified": certified,
                    "scope_correct": scope_correct,
                }
            )
            input_hashes[f"lane{lane_index}/rate{rate}"] = {
                "complete_sha256": sha256(complete_path),
                "summary_sha256": sha256(summary_path),
                "metric_vectors_sha256": sha256(vectors_path),
                "test_cache_manifest_sha256": sha256(cache_manifest_path),
            }
        combined = hierarchical_test(
            deltas,
            reps=int(args.bootstrap_reps),
            seed=int(args.bootstrap_seed) + rate_index,
        )
        per_rate[rate] = {
            "hierarchical": combined,
            "all_intervals_favorable": all(
                favorable_interval(metric, combined[metric]) for metric in METRICS
            ),
            "all_lane_means_favorable": all(item["all_means_favorable"] for item in per_lane),
            "per_lane": sorted(per_lane, key=lambda item: item["lane_index"]),
        }
    if len(split_hashes) != 1:
        raise RuntimeError("HELDOUT_SPLIT_DRIFT_ACROSS_LANES_OR_RATES")
    p_values = {
        f"rate{rate}_{metric}": per_rate[rate]["hierarchical"][metric]["one_sided_p"]
        for rate in ("05", "10")
        for metric in METRICS
    }
    holm = holm_adjust(p_values)
    six_gate = all(
        per_rate[rate]["all_intervals_favorable"]
        and per_rate[rate]["all_lane_means_favorable"]
        for rate in ("05", "10")
    )
    headline_pass = bool(
        six_gate and all_certified and all_scope_correct and all_receipt_hashes_match
    )
    payload = {
        "status": "VQGAN_GUIDED_FOHI_HELDOUT_FINAL_DECISION",
        "evaluation_scope": "heldout",
        "validation_only": False,
        "test_split_opened": True,
        "test_images": expected_images,
        "test_raw_hash_sequence_sha256": next(iter(split_hashes)),
        "headline_six_component_gate_passed": headline_pass,
        "all_projection_certificates_passed": all_certified,
        "all_scope_and_hash_gates_passed": bool(
            all_scope_correct and all_receipt_hashes_match
        ),
        "all_result_receipt_hashes_matched": all_receipt_hashes_match,
        "rates": per_rate,
        "one_sided_p_values": p_values,
        "holm_adjusted_p_values": holm,
        "holm_all_six_below_0_05": all(value < 0.05 for value in holm.values()),
        "freeze_manifest_sha256": sha256(args.freeze_manifest),
        "freeze_manifest_canonical_json_sha256": canonical_json_sha256(freeze),
        "input_hashes": input_hashes,
        "post_test_tuning_permitted": False,
    }
    (args.output_dir / "heldout_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# VQGAN-guided FOHI one-shot held-out decision",
        "",
        f"Headline six-component gate: **{'PASS' if headline_pass else 'FAIL'}**.",
        "",
        "| Rate | Delta PSNR | Delta SSIM | Delta LPIPS | Joint interval gate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rate in ("05", "10"):
        item = per_rate[rate]
        lines.append(
            f"| {int(rate)}% | {item['hierarchical']['psnr']['mean_delta']:+.6f} | "
            f"{item['hierarchical']['ssim']['mean_delta']:+.6f} | "
            f"{item['hierarchical']['lpips']['mean_delta']:+.6f} | "
            f"{'PASS' if item['all_intervals_favorable'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"All {expected_images} raw-hash-disjoint STL-10 test images are used once. "
            "The 1260 byte-identical development overlaps are excluded before any quality metric. "
            "No post-test method change is permitted.",
        ]
    )
    (args.output_dir / "HELDOUT_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
