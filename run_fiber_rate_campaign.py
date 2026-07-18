from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rates", default="02,10")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-seed", type=int, default=20260718)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rates = [part.strip() for part in str(args.rates).split(",") if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cache_root = args.output_dir / "cache"
    run(
        [
            sys.executable,
            "-u",
            str(root / "prepare_fiber_rate_caches.py"),
            "--bundle-root",
            str(args.bundle_root),
            "--dataset-root",
            str(args.dataset_root),
            "--rates",
            ",".join(rates),
            "--output-dir",
            str(cache_root),
            "--seed",
            str(int(args.seed)),
        ],
        cwd=root,
        log_path=args.output_dir / "cache_driver.log",
    )
    rate_results: dict[str, dict[str, Any]] = {}
    for rate in rates:
        rate_output = args.output_dir / f"rate{rate}"
        cache_dir = cache_root / f"rate{rate}"
        dev = cache_dir / f"seed{int(args.seed)}_dev.pt"
        val = cache_dir / f"seed{int(args.seed)}_val.pt"
        config = args.bundle_root / f"config_rate{rate}.yaml"
        run(
            [
                sys.executable,
                "-u",
                str(root / "run_fiber_fusion_multiseed_pipeline.py"),
                "--primary-dev",
                str(dev),
                "--primary-val",
                str(val),
                "--control-dev",
                str(dev),
                "--control-val",
                str(val),
                "--config",
                str(config),
                "--output-dir",
                str(rate_output),
                "--label",
                f"rate{rate}_seed{int(args.seed)}_matched_control",
                "--seed",
                str(int(args.training_seed) + 100 * int(args.seed) + int(rate)),
                "--steps",
                str(int(args.steps)),
                "--cutoffs",
                "0.18",
                "--alphas",
                "0.58",
                "--top-exact",
                "1",
            ],
            cwd=root,
            log_path=args.output_dir / f"rate{rate}_pipeline_driver.log",
        )
        fusion_path = rate_output / "fusion/summary.json"
        fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
        if fusion.get("test_split_opened") is not False:
            raise RuntimeError(f"TEST_SPLIT_OPENED_AT_RATE:{rate}")
        candidate = fusion["exact_candidates"][0]
        if float(candidate["cutoff"]) != 0.18 or float(candidate["alpha"]) != 0.58:
            raise RuntimeError(f"FROZEN_PARAMETER_DRIFT:{rate}")
        rate_results[rate] = {
            "operator_sha256": fusion["operator_sha256"],
            "base_means": fusion["base_means"],
            "control_means": fusion["control_means"],
            "fused_means": candidate["means"],
            "paired_vs_control": candidate["paired_vs_control"],
            "paired_vs_vqae": candidate["paired_vs_vqae"],
            "dominates_control": candidate["dominates_control"],
            "projection_audit": candidate["projection_audit"],
            "fusion_summary": str(fusion_path),
        }
    payload = {
        "status": "FIBER_RATE_ROBUSTNESS_CAMPAIGN_COMPLETE",
        "seed": int(args.seed),
        "training_seed": int(args.training_seed),
        "steps": int(args.steps),
        "frozen_cutoff": 0.18,
        "frozen_alpha": 0.58,
        "matched_control_uses_same_seed_cache": True,
        "validation_only": True,
        "test_split_opened": False,
        "rates": rate_results,
        "runtime_seconds": time.time() - started,
    }
    output = args.output_dir / "campaign_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {output}", flush=True)


if __name__ == "__main__":
    main()

