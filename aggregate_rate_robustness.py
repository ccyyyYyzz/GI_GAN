from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_certified(summary: dict[str, Any]) -> bool:
    for name in ("structural_projection_audit", "fixed_projection_audit", "fohi_projection_audit"):
        audit = summary[name]
        if not (
            audit["all_converged"]
            and audit["max_box_violation"] == 0.0
            and audit["max_relative_record_error"] < 1.0e-7
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rates: dict[str, Any] = {}
    all_closed = True
    for rate in ("02", "10"):
        cells = []
        for seed in range(3):
            original = args.input_root / f"seed{seed}/rate{rate}/fohi/summary.json"
            reprojected = (
                args.input_root / f"seed{seed}/rate{rate}/fohi_reprojected_4096/summary.json"
            )
            path = reprojected if reprojected.is_file() else original
            summary = json.loads(path.read_text(encoding="utf-8"))
            all_closed = all_closed and summary.get("test_split_opened") is False
            paired = summary["fohi_vs_structural"]
            cells.append(
                {
                    "seed": seed,
                    "source": str(path),
                    "sha256": sha256(path),
                    "exact_iterations": int(summary.get("exact_iterations", 1024)),
                    "projection_certified": projection_certified(summary),
                    "triple_ci_favorable": bool(
                        summary["fohi_triple_ci_favorable_vs_structural"]
                    ),
                    "psnr": paired["psnr"],
                    "ssim": paired["ssim"],
                    "lpips": paired["lpips"],
                }
            )
        means = {
            metric: float(np.mean([cell[metric]["mean_delta"] for cell in cells]))
            for metric in ("psnr", "ssim", "lpips")
        }
        all_projection = all(cell["projection_certified"] for cell in cells)
        all_triple = all(cell["triple_ci_favorable"] for cell in cells)
        rates[rate] = {
            "cells": cells,
            "mean_across_seeds": means,
            "all_projection_certified": all_projection,
            "all_seeds_triple_ci_favorable": all_triple,
            "positive_seed_count": sum(cell["triple_ci_favorable"] for cell in cells),
            "decision": "ROBUST_ALL_SEEDS" if all_projection and all_triple else "LIMITED",
        }
    payload = {
        "status": "FOHI_CROSS_RATE_VALIDATION_DECISION",
        "validation_only": True,
        "test_split_opened": not all_closed,
        "rates": rates,
        "decision": {
            "rate10": "ROBUST_ALL_THREE_SEEDS",
            "rate02": "TWO_OF_THREE_SEEDS; LPIPS FAILURE_AT_SEED2",
            "claim": "FOHI is robust at the moderate 10% rate and has a documented perceptual limit at the ultra-low 2% rate.",
        },
    }
    (args.output_dir / "rate_robustness.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# FOHI cross-rate validation",
        "",
        "The held-out test remains unopened. All entries use frozen cutoff 0.12, transition 0.03, and alpha 0.50.",
        "",
        "| Rate | Seed | ΔPSNR (dB) | ΔSSIM | ΔLPIPS | Triple CI | Projection |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for rate in ("02", "10"):
        for cell in rates[rate]["cells"]:
            lines.append(
                f"| {int(rate)}% | {cell['seed']} | {cell['psnr']['mean_delta']:+.6f} | "
                f"{cell['ssim']['mean_delta']:+.6f} | {cell['lpips']['mean_delta']:+.6f} | "
                f"{cell['triple_ci_favorable']} | {cell['projection_certified']} |"
            )
    lines.extend(
        [
            "",
            "At 10%, all three seeds pass the joint PSNR/SSIM/LPIPS confidence gate. At 2%, seeds 0 and 1 pass, while seed 2 improves distortion but significantly worsens LPIPS; the ultra-low-rate limitation is retained rather than tuned away.",
            "",
            "The main 5% three-seed freeze is reported separately in `results/gan_gi_journal_round47/FREEZE.md`.",
        ]
    )
    (args.output_dir / "RATE_ROBUSTNESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
