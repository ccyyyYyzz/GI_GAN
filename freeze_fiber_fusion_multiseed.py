from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCALE = {"psnr": 0.02, "ssim": 0.0005, "lpips": 0.005}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_split_opened") is not False:
        raise RuntimeError(f"TEST_SPLIT_NOT_CLOSED:{path}")
    if payload.get("validation_only") is not True:
        raise RuntimeError(f"NOT_VALIDATION_ONLY:{path}")
    if not payload.get("exact_candidates"):
        raise RuntimeError(f"NO_EXACT_CANDIDATES:{path}")
    return payload


def candidate_key(row: dict[str, Any]) -> tuple[float, float]:
    return float(row["cutoff"]), float(row["alpha"])


def favorable_mean(row: dict[str, Any]) -> bool:
    paired = row["paired_vs_control"]
    return bool(
        paired["psnr"]["mean_delta"] > 0.0
        and paired["ssim"]["mean_delta"] > 0.0
        and paired["lpips"]["mean_delta"] < 0.0
    )


def favorable_ci(row: dict[str, Any]) -> bool:
    paired = row["paired_vs_control"]
    return bool(
        paired["psnr"]["ci95_low"] > 0.0
        and paired["ssim"]["ci95_low"] > 0.0
        and paired["lpips"]["ci95_high"] < 0.0
    )


def normalized_benefit(row: dict[str, Any]) -> float:
    paired = row["paired_vs_control"]
    return float(
        paired["psnr"]["mean_delta"] / SCALE["psnr"]
        + paired["ssim"]["mean_delta"] / SCALE["ssim"]
        - paired["lpips"]["mean_delta"] / SCALE["lpips"]
    )


def freeze(summary_paths: list[Path]) -> dict[str, Any]:
    if len(summary_paths) != 3:
        raise RuntimeError(f"THREE_VALIDATION_SEEDS_REQUIRED:{len(summary_paths)}")
    records = []
    operator_hashes = set()
    grids = []
    for path in summary_paths:
        payload = load_summary(path)
        label = path.parent.name
        by_key = {candidate_key(row): row for row in payload["exact_candidates"]}
        records.append((label, path, payload, by_key))
        operator_hashes.add(str(payload["operator_sha256"]))
        grids.append(set(by_key))
    if len(operator_hashes) != 1:
        raise RuntimeError(f"OPERATOR_MISMATCH:{sorted(operator_hashes)}")
    if any(grid != grids[0] for grid in grids[1:]):
        raise RuntimeError("CANDIDATE_GRID_MISMATCH")

    candidates = []
    for key in sorted(grids[0]):
        per_seed = {}
        for label, _, _, by_key in records:
            row = by_key[key]
            per_seed[label] = {
                "means": row["means"],
                "paired_vs_control": row["paired_vs_control"],
                "normalized_benefit": normalized_benefit(row),
                "projection_audit": row["projection_audit"],
            }
        benefits = [entry["normalized_benefit"] for entry in per_seed.values()]
        candidates.append(
            {
                "cutoff": key[0],
                "alpha": key[1],
                "all_seed_means_favorable": all(
                    favorable_mean(by_key[key]) for _, _, _, by_key in records
                ),
                "all_seed_ci95_favorable": all(
                    favorable_ci(by_key[key]) for _, _, _, by_key in records
                ),
                "worst_seed_normalized_benefit": min(benefits),
                "mean_normalized_benefit": sum(benefits) / len(benefits),
                "per_seed": per_seed,
            }
        )
    eligible = [
        row
        for row in candidates
        if row["all_seed_means_favorable"] and row["all_seed_ci95_favorable"]
    ]
    if not eligible:
        raise RuntimeError("NO_THREE_SEED_CI_ROBUST_CANDIDATE")
    eligible.sort(
        key=lambda row: (
            -row["worst_seed_normalized_benefit"],
            -row["mean_normalized_benefit"],
            row["cutoff"],
            row["alpha"],
        )
    )
    winner = eligible[0]
    winner_key = (winner["cutoff"], winner["alpha"])
    winner_rows = {
        label: by_key[winner_key] for label, _, _, by_key in records
    }
    hyperparameter_keys = (
        "source_arm",
        "rotation_scale",
        "adversarial_weight",
        "lpips_weight",
        "step",
        "channels",
    )
    proposal_hyperparameters = {
        key: next(iter(winner_rows.values()))["proposal_manifest"][key]
        for key in hyperparameter_keys
    }
    if any(
        any(
            row["proposal_manifest"][key] != proposal_hyperparameters[key]
            for key in hyperparameter_keys
        )
        for row in winner_rows.values()
    ):
        raise RuntimeError("PROPOSAL_HYPERPARAMETER_MISMATCH")
    payload = {
        "status": "FROZEN_BEFORE_HELD_OUT_TEST",
        "selection_rule": {
            "gate_1": "PSNR and SSIM mean deltas are positive and LPIPS mean delta is negative on every validation seed",
            "gate_2": "the corresponding seedwise paired 95% confidence intervals exclude zero in the favorable direction",
            "ranking": "maximize the minimum normalized benefit across seeds",
            "normalized_benefit": "delta_PSNR/0.02 + delta_SSIM/0.0005 - delta_LPIPS/0.005",
        },
        "frozen_method": {
            "name": "fiber-orthogonal adversarial detail fusion",
            "cutoff": winner["cutoff"],
            "alpha": winner["alpha"],
            "transition": 0.03,
            "proposal_arm": next(iter(winner_rows.values()))["proposal_arm"],
            "proposal_hyperparameters": proposal_hyperparameters,
            "proposal_training_seeds": {
                label: row["proposal_manifest"]["seed"]
                for label, row in winner_rows.items()
            },
        },
        "winner": winner,
        "eligible_candidates": [
            {
                key: row[key]
                for key in (
                    "cutoff",
                    "alpha",
                    "worst_seed_normalized_benefit",
                    "mean_normalized_benefit",
                )
            }
            for row in eligible
        ],
        "all_candidates": candidates,
        "operator_sha256": next(iter(operator_hashes)),
        "source_summaries": [
            {
                "label": label,
                "filename": path.name,
                "sha256": sha256(path),
                "validation_images": payload["validation_images"],
                "test_split_opened": payload["test_split_opened"],
            }
            for label, path, payload, _ in records
        ],
        "test_split_opened": False,
    }
    return payload


def markdown(payload: dict[str, Any]) -> str:
    method = payload["frozen_method"]
    winner = payload["winner"]
    lines = [
        "# Round 45 three-seed validation freeze",
        "",
        "The held-out test remains unopened. The method and its two scalar settings are frozen from three 512-image validation seeds before any test evaluation.",
        "",
        f"Frozen setting: cutoff = {method['cutoff']:.2f}, alpha = {method['alpha']:.2f}, transition = {method['transition']:.2f}.",
        "",
        "Selection rule: require favorable seedwise means and favorable seedwise paired 95% confidence intervals for PSNR, SSIM, and LPIPS; among survivors, maximize the worst-seed normalized benefit.",
        "",
        "| validation pairing | delta PSNR (95% CI) | delta SSIM (95% CI) | delta LPIPS (95% CI) | normalized benefit |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, entry in sorted(winner["per_seed"].items()):
        paired = entry["paired_vs_control"]
        cells = []
        for metric in ("psnr", "ssim", "lpips"):
            value = paired[metric]
            cells.append(
                f"{value['mean_delta']:+.6f} [{value['ci95_low']:+.6f}, {value['ci95_high']:+.6f}]"
            )
        lines.append(
            f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {entry['normalized_benefit']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Operator SHA-256: `{payload['operator_sha256']}`.",
            "",
            "The GAN proposal is compared against a matched VQAE-only residual generator. The same spectral rule, projection, and validation images are used in both arms; the only causal difference is the adversarial proposal source.",
            "",
            "Source summary hashes:",
            "",
        ]
    )
    for source in payload["source_summaries"]:
        lines.append(f"- `{source['label']}`: `{source['sha256']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "freeze.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "FREEZE.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["frozen_method"], sort_keys=True))
    print(f"WROTE {args.output_dir / 'freeze.json'}")
    print(f"WROTE {args.output_dir / 'FREEZE.md'}")


if __name__ == "__main__":
    main()
