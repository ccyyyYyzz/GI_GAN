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
        raise ValueError("EXACTLY_THREE_OPERATOR_SEEDS_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_operator = []
    deltas = {metric: [] for metric in METRICS}
    hashes: dict[str, dict[str, str]] = {}
    operator_hashes = set()
    operator_seeds = set()
    for index, directory in enumerate(args.input_dirs):
        campaign_path = directory / "campaign_summary.json"
        summary_path = directory / "fohi" / "summary.json"
        vectors_path = directory / "fohi" / "metric_vectors.npz"
        for path in (campaign_path, summary_path, vectors_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            campaign.get("validation_only") is not True
            or campaign.get("test_split_opened") is not False
            or summary.get("validation_only") is not True
            or summary.get("test_split_opened") is not False
        ):
            raise RuntimeError(f"SPLIT_PROTOCOL_VIOLATION:{directory}")
        operator_seeds.add(int(campaign["operator_seed"]))
        operator_hashes.add(str(campaign["operator_sha256"]))
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
            seed=int(args.seed) + index,
        )
        for metric in METRICS:
            deltas[metric].append(fohi[metric] - structural[metric])
        certified = projection_certified(summary)
        per_operator.append(
            {
                "operator_seed": int(campaign["operator_seed"]),
                "operator_sha256": str(campaign["operator_sha256"]),
                "paired_vs_structural": paired,
                "triple_ci_favorable": direct_ci_favorable(paired),
                "projection_certified": certified,
            }
        )
        hashes[directory.name] = {
            "campaign_sha256": sha256(campaign_path),
            "summary_sha256": sha256(summary_path),
            "metric_vectors_sha256": sha256(vectors_path),
        }

    if len(operator_seeds) != 3 or len(operator_hashes) != 3:
        raise RuntimeError("OPERATOR_SEEDS_OR_MATRICES_NOT_INDEPENDENT")
    combined = crossed_bootstrap(
        deltas,
        reps=int(args.bootstrap_reps),
        seed=int(args.seed) + 1000,
    )
    positive_count = sum(
        item["triple_ci_favorable"] and item["projection_certified"]
        for item in per_operator
    )
    payload = {
        "status": "FOHI_INDEPENDENT_OPERATOR_SEED_DECISION",
        "validation_only": True,
        "test_split_opened": False,
        "selected_method": "discriminator-off VQGAN-source FOHI",
        "operator_positive_count": int(positive_count),
        "operator_robust": bool(positive_count == 3 and direct_ci_favorable(combined)),
        "combined_hierarchical_bootstrap": combined,
        "per_operator": sorted(per_operator, key=lambda item: item["operator_seed"]),
        "input_hashes": hashes,
    }
    (args.output_dir / "operator_seed_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# FOHI independent-operator validation",
        "",
        f"Independent operator seeds passing the joint CI gate: **{positive_count}/3**.",
        "",
        (
            "The selected discriminator-off VQGAN-source FOHI is robust to the three "
            "predeclared independent measurement operators."
            if payload["operator_robust"]
            else "Operator robustness is not established; the held-out test must remain closed."
        ),
        "",
        "The held-out test remains unopened.",
    ]
    (args.output_dir / "OPERATOR_FREEZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
