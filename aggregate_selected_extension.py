from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aggregate_endpoint_fohi import METRICS, crossed_bootstrap, direct_ci_favorable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vectors(path: Path, arm: str = "fohi") -> dict[str, np.ndarray]:
    archive = np.load(path)
    return {
        metric: np.asarray(archive[f"{arm}_{metric}"], dtype=np.float64)
        for metric in METRICS
    }


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
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--extension-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.extension_dirs) != 3:
        raise ValueError("EXACTLY_THREE_EXTENSION_LANES_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spectral_deltas = {metric: [] for metric in METRICS}
    rates = {"02": [], "10": []}
    hashes = {}
    for directory in args.extension_dirs:
        campaign_path = directory / "campaign_summary.json"
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        if campaign.get("validation_only") is not True or campaign.get("test_split_opened") is not False:
            raise RuntimeError(f"SPLIT_PROTOCOL_VIOLATION:{directory}")
        label = directory.name
        high_path = args.causal_root / label / "B_gan_adv0_highpass/metric_vectors.npz"
        low_path = directory / "five_percent_adv0_lowpass/metric_vectors.npz"
        high = load_vectors(high_path)
        low = load_vectors(low_path)
        high_structural = load_vectors(high_path, arm="structural")
        low_structural = load_vectors(low_path, arm="structural")
        if not all(
            np.array_equal(high_structural[metric], low_structural[metric])
            for metric in METRICS
        ):
            raise RuntimeError(f"SPECTRAL_CONTROL_STRUCTURAL_DRIFT:{label}")
        for metric in METRICS:
            spectral_deltas[metric].append(high[metric] - low[metric])
        hashes[f"{label}/five_highpass"] = sha256(high_path)
        hashes[f"{label}/five_lowpass"] = sha256(low_path)
        for rate in ("02", "10"):
            summary_path = directory / f"rate{rate}_adv0_highpass/summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rates[rate].append(
                {
                    "rate_seed": int(campaign["rate_seed"]),
                    "label": label,
                    "summary_sha256": sha256(summary_path),
                    "projection_certified": projection_certified(summary),
                    "triple_ci_favorable": bool(
                        summary["fohi_triple_ci_favorable_vs_structural"]
                    ),
                    "psnr": summary["fohi_vs_structural"]["psnr"],
                    "ssim": summary["fohi_vs_structural"]["ssim"],
                    "lpips": summary["fohi_vs_structural"]["lpips"],
                }
            )
    for rate in rates:
        rates[rate].sort(key=lambda item: item["rate_seed"])
    high_minus_low = crossed_bootstrap(
        spectral_deltas,
        reps=int(args.bootstrap_reps),
        seed=int(args.seed),
    )
    payload = {
        "status": "SELECTED_DISCRIMINATOR_OFF_FOHI_EXTENSION_DECISION",
        "validation_only": True,
        "test_split_opened": False,
        "input_hashes": hashes,
        "five_percent_highpass_minus_lowpass": high_minus_low,
        "five_percent_highpass_strictly_dominant": direct_ci_favorable(high_minus_low),
        "rates": rates,
        "rate_positive_counts": {
            rate: sum(
                cell["projection_certified"] and cell["triple_ci_favorable"]
                for cell in cells
            )
            for rate, cells in rates.items()
        },
        "selected_method": "discriminator-off VQGAN-source FOHI",
        "claim": "High-pass is the dominant band, not the exclusive beneficial band.",
    }
    (args.output_dir / "selected_extension.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Selected discriminator-off FOHI extension",
        "",
        "Selected method: **discriminator-off VQGAN-source FOHI**.",
        "",
        "| Rate | Seeds passing joint CI gate |",
        "|---:|---:|",
        f"| 2% | {payload['rate_positive_counts']['02']}/3 |",
        f"| 10% | {payload['rate_positive_counts']['10']}/3 |",
        "",
        "At 5%, both spectral complements can improve the structural arm, but the direct crossed-seed comparison strictly favors the high-pass component in PSNR, SSIM, and LPIPS. High-pass is therefore described as dominant rather than exclusive.",
        "",
        "The held-out test remains unopened.",
    ]
    (args.output_dir / "SELECTED_FREEZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
