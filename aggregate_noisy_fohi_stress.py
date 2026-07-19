from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aggregate_endpoint_fohi import (
    METRICS,
    crossed_bootstrap,
    direct_ci_favorable,
    paired_image_bootstrap,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_certified(summary: dict[str, Any]) -> bool:
    return all(
        audit["all_converged"]
        and audit["max_box_violation"] == 0.0
        and audit["max_relative_record_error"] < 1.0e-7
        for audit in (
            summary["structural_projection_audit"],
            summary["fixed_projection_audit"],
            summary["fohi_projection_audit"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input_dirs) != 3:
        raise ValueError("EXACTLY_THREE_OPERATOR_LANES_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions: dict[str, Any] = {}
    hashes: dict[str, dict[str, str]] = {}
    for condition_index, condition in enumerate(("snr30", "snr20")):
        deltas = {metric: [] for metric in METRICS}
        per_operator = []
        for operator_index, directory in enumerate(args.input_dirs):
            campaign_path = directory / "campaign_summary.json"
            summary_path = directory / condition / "fohi" / "summary.json"
            vectors_path = directory / condition / "fohi" / "metric_vectors.npz"
            for path in (campaign_path, summary_path, vectors_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                campaign.get("validation_only") is not True
                or campaign.get("test_split_opened") is not False
                or campaign.get("frozen_clean_checkpoints") is not True
                or summary.get("validation_only") is not True
                or summary.get("test_split_opened") is not False
            ):
                raise RuntimeError(f"PROTOCOL_VIOLATION:{directory}:{condition}")
            archive = np.load(vectors_path)
            structural = {
                metric: np.asarray(archive[f"structural_{metric}"], dtype=np.float64)
                for metric in METRICS
            }
            fohi = {
                metric: np.asarray(archive[f"fohi_{metric}"], dtype=np.float64)
                for metric in METRICS
            }
            paired = paired_image_bootstrap(
                fohi,
                structural,
                reps=10000,
                seed=int(args.seed) + 100 * condition_index + operator_index,
            )
            for metric in METRICS:
                deltas[metric].append(fohi[metric] - structural[metric])
            requested = float(condition.removeprefix("snr"))
            condition_meta = next(
                item
                for item in campaign["conditions"]
                if float(item["requested_bucket_snr_db"]) == requested
            )
            per_operator.append(
                {
                    "operator_seed": int(campaign["operator_seed"]),
                    "achieved_validation_bucket_snr": condition_meta[
                        "achieved_validation_bucket_snr"
                    ],
                    "paired_vs_structural": paired,
                    "triple_ci_favorable": direct_ci_favorable(paired),
                    "projection_certified": projection_certified(summary),
                }
            )
            hashes[f"{directory.name}/{condition}"] = {
                "summary_sha256": sha256(summary_path),
                "metric_vectors_sha256": sha256(vectors_path),
            }
        combined = crossed_bootstrap(
            deltas,
            reps=int(args.bootstrap_reps),
            seed=int(args.seed) + 1000 + condition_index,
        )
        positive_count = sum(
            item["triple_ci_favorable"] and item["projection_certified"]
            for item in per_operator
        )
        conditions[condition] = {
            "operator_positive_count": int(positive_count),
            "combined_hierarchical_bootstrap": combined,
            "stress_robust": bool(positive_count == 3 and direct_ci_favorable(combined)),
            "per_operator": sorted(per_operator, key=lambda item: item["operator_seed"]),
        }

    payload = {
        "status": "FROZEN_FOHI_NOISY_EQUALITY_STRESS_DECISION",
        "interpretation": (
            "algorithmic exact-fit stress only; this is not a statistically calibrated noisy-fiber claim"
        ),
        "selected_method": "discriminator-off VQGAN-source FOHI",
        "validation_only": True,
        "test_split_opened": False,
        "frozen_clean_checkpoints": True,
        "conditions": conditions,
        "input_hashes": hashes,
    }
    (args.output_dir / "noise_stress_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Frozen FOHI noisy-bucket equality stress",
        "",
        "This is an algorithmic exact-fit stress test, not a calibrated noisy-fiber physical claim.",
        "",
        "| Requested bucket SNR | Operators passing the joint CI gate |",
        "|---:|---:|",
    ]
    for condition in ("snr30", "snr20"):
        lines.append(
            f"| {condition.removeprefix('snr')} dB | "
            f"{conditions[condition]['operator_positive_count']}/3 |"
        )
    lines.extend(["", "The held-out test remains unopened."])
    (args.output_dir / "NOISE_STRESS_FREEZE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
