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


CELLS = (
    "A_gan_adv_highpass",
    "B_gan_adv0_highpass",
    "C_vqae2_highpass",
    "D_gan_adv_lowpass",
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


def no_significant_control_advantage(a_minus_control: dict[str, Any]) -> bool:
    return bool(
        a_minus_control["psnr"]["ci95_high"] >= 0.0
        and a_minus_control["ssim"]["ci95_high"] >= 0.0
        and a_minus_control["lpips"]["ci95_low"] <= 0.0
    )


def any_significant_primary_advantage(a_minus_control: dict[str, Any]) -> bool:
    return bool(
        a_minus_control["psnr"]["ci95_low"] > 0.0
        or a_minus_control["ssim"]["ci95_low"] > 0.0
        or a_minus_control["lpips"]["ci95_high"] < 0.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input_dirs) != 3:
        raise ValueError("EXACTLY_THREE_FROZEN_PAIRINGS_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data: list[dict[str, Any]] = []
    hashes = {}
    for seed_index, directory in enumerate(args.input_dirs):
        cells = {}
        structural_reference = None
        for cell_index, cell in enumerate(CELLS):
            summary_path = directory / cell / "summary.json"
            vectors_path = directory / cell / "metric_vectors.npz"
            if not summary_path.is_file() or not vectors_path.is_file():
                raise FileNotFoundError(f"MISSING_CAUSAL_CELL:{directory}:{cell}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("validation_only") is not True or summary.get("test_split_opened") is not False:
                raise RuntimeError(f"SPLIT_PROTOCOL_VIOLATION:{directory}:{cell}")
            archive = np.load(vectors_path)
            structural = {
                metric: np.asarray(archive[f"structural_{metric}"], dtype=np.float64)
                for metric in METRICS
            }
            fohi = {
                metric: np.asarray(archive[f"fohi_{metric}"], dtype=np.float64)
                for metric in METRICS
            }
            if structural_reference is None:
                structural_reference = structural
            elif not all(
                np.array_equal(structural_reference[metric], structural[metric])
                for metric in METRICS
            ):
                raise RuntimeError(f"STRUCTURAL_VECTOR_DRIFT:{directory}:{cell}")
            paired = paired_image_bootstrap(
                fohi,
                structural,
                reps=10000,
                seed=int(args.seed) + 100 * seed_index + cell_index,
            )
            cells[cell] = {
                "summary": summary,
                "structural": structural,
                "fohi": fohi,
                "paired_vs_structural": paired,
                "triple_ci_favorable": direct_ci_favorable(paired),
                "projection_certified": projection_certified(summary),
            }
            hashes[f"{directory.name}/{cell}"] = {
                "summary_sha256": sha256(summary_path),
                "metric_vectors_sha256": sha256(vectors_path),
            }
        data.append({"pairing": directory.name, "cells": cells})

    crossed_direct = {}
    for control in CELLS[1:]:
        deltas = {metric: [] for metric in METRICS}
        for seed in data:
            for metric in METRICS:
                deltas[metric].append(
                    seed["cells"]["A_gan_adv_highpass"]["fohi"][metric]
                    - seed["cells"][control]["fohi"][metric]
                )
        crossed_direct[f"A_minus_{control}"] = crossed_bootstrap(
            deltas,
            reps=int(args.bootstrap_reps),
            seed=int(args.seed) + len(crossed_direct),
        )

    cell_positive_counts = {
        cell: sum(
            seed["cells"][cell]["triple_ci_favorable"]
            and seed["cells"][cell]["projection_certified"]
            for seed in data
        )
        for cell in CELLS
    }
    a_vs_b = crossed_direct["A_minus_B_gan_adv0_highpass"]
    b_robust = cell_positive_counts["B_gan_adv0_highpass"] == 3
    b_has_no_significant_disadvantage = not any_significant_primary_advantage(a_vs_b)
    b_has_significant_advantage = not no_significant_control_advantage(a_vs_b)
    select_b = bool(
        b_robust and b_has_no_significant_disadvantage and b_has_significant_advantage
    )
    c_robust = cell_positive_counts["C_vqae2_highpass"] == 3
    d_robust = cell_positive_counts["D_gan_adv_lowpass"] == 3
    selected = "B_gan_adv0_highpass" if select_b else "A_gan_adv_highpass"
    payload = {
        "status": "FOHI_FOUR_CELL_CAUSAL_DECISION",
        "validation_only": True,
        "test_split_opened": False,
        "pairings": [item["pairing"] for item in data],
        "input_hashes": hashes,
        "cell_positive_counts": cell_positive_counts,
        "crossed_direct": crossed_direct,
        "selected_primary_cell": selected,
        "adapter_discriminator_essential": bool(not b_robust),
        "gan_source_distinct_from_second_vqae": bool(not c_robust),
        "highpass_exclusive": bool(not d_robust),
        "interpretation": (
            "VQGAN proposal alignment; the added adapter discriminator is unnecessary"
            if b_robust
            else "adapter adversarial training remains causally implicated"
        ),
        "per_seed": [
            {
                "pairing": seed["pairing"],
                "cells": {
                    cell: {
                        "paired_vs_structural": seed["cells"][cell]["paired_vs_structural"],
                        "triple_ci_favorable": seed["cells"][cell]["triple_ci_favorable"],
                        "projection_certified": seed["cells"][cell]["projection_certified"],
                    }
                    for cell in CELLS
                },
            }
            for seed in data
        ],
    }
    (args.output_dir / "causal_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# FOHI four-cell causal decision",
        "",
        f"Selected primary cell: **{selected}**.",
        "",
        f"Interpretation: {payload['interpretation']}.",
        "",
        "| Cell | Seeds passing joint CI gate |",
        "|---|---:|",
    ]
    for cell in CELLS:
        lines.append(f"| {cell} | {cell_positive_counts[cell]}/3 |")
    lines.extend(["", "The held-out test remains unopened."])
    (args.output_dir / "CAUSAL_FREEZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
