from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def run(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--operator-seed", type=int, required=True)
    parser.add_argument("--lane-index", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    config = yaml.safe_load((args.asset_root / "config_base.yaml").read_text(encoding="utf-8"))
    config["operator"]["seed"] = int(args.operator_seed)
    config["data"]["dataset_root"] = str(args.dataset_root)
    config_path = args.asset_root / "config_rate05.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cache_root = args.output_dir / "cache"
    run(
        [
            sys.executable, "-u", str(root / "prepare_fiber_rate_caches.py"),
            "--bundle-root", str(args.asset_root),
            "--dataset-root", str(args.dataset_root),
            "--rates", "05",
            "--output-dir", str(cache_root),
            "--seed", str(int(args.lane_index)),
        ],
        cwd=root,
        log_path=args.output_dir / "cache.log",
    )
    cache_dir = cache_root / "rate05"
    dev = cache_dir / f"seed{int(args.lane_index)}_dev.pt"
    val = cache_dir / f"seed{int(args.lane_index)}_val.pt"
    training_seed = 20262000 + int(args.lane_index)
    gan_dir = args.output_dir / "gan_adv0"
    control_dir = args.output_dir / "control"
    common = [
        "--primary-dev", str(dev),
        "--primary-val", str(val),
        "--config", str(config_path),
        "--rotation-scales", "0.5",
        "--lpips-weight", "0.003",
        "--steps", str(int(args.steps)),
        "--batch-size", "32",
        "--bootstrap-reps", "5000",
        "--seed", str(training_seed),
    ]
    run(
        [
            sys.executable, "-u", str(root / "train_fiber_residual_phase_gan.py"),
            *common,
            "--source-arm", "gan",
            "--adv-weights", "0",
            "--output-dir", str(gan_dir),
        ],
        cwd=root,
        log_path=args.output_dir / "gan_adv0.log",
    )
    run(
        [
            sys.executable, "-u", str(root / "train_fiber_residual_phase_gan.py"),
            *common,
            "--source-arm", "vqae_control",
            "--adv-weights", "0",
            "--control-dev", str(dev),
            "--control-val", str(val),
            "--output-dir", str(control_dir),
        ],
        cwd=root,
        log_path=args.output_dir / "control.log",
    )
    fohi_dir = args.output_dir / "fohi"
    run(
        [
            sys.executable, "-u", str(root / "diagnose_fiber_orthogonal_highpass_innovation.py"),
            "--primary-val", str(val),
            "--control-val", str(val),
            "--config", str(config_path),
            "--control-checkpoint", str(control_dir / "checkpoint_vqae_control_rot0.5_adv0.pt"),
            "--proposal-checkpoint", str(gan_dir / "checkpoint_gan_rot0.5_adv0.pt"),
            "--filter-mode", "highpass",
            "--cutoff", "0.12",
            "--transition", "0.03",
            "--alpha", "0.5",
            "--batch-size", "32",
            "--exact-iterations", "4096",
            "--bootstrap-reps", "10000",
            "--seed", str(training_seed + 1000),
            "--output-dir", str(fohi_dir),
        ],
        cwd=root,
        log_path=args.output_dir / "fohi.log",
    )
    fohi = json.loads((fohi_dir / "summary.json").read_text(encoding="utf-8"))
    payload = {
        "status": "FOHI_OPERATOR_SEED_CAMPAIGN_COMPLETE",
        "operator_seed": int(args.operator_seed),
        "lane_index": int(args.lane_index),
        "steps": int(args.steps),
        "validation_only": True,
        "test_split_opened": False,
        "operator_sha256": fohi["operator_sha256"],
        "fohi_summary": str(fohi_dir / "summary.json"),
        "runtime_seconds": time.time() - started,
    }
    (args.output_dir / "campaign_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
