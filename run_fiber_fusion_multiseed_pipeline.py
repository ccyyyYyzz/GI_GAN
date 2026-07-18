from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


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
    parser.add_argument("--primary-dev", type=Path, required=True)
    parser.add_argument("--primary-val", type=Path, required=True)
    parser.add_argument("--control-dev", type=Path, required=True)
    parser.add_argument("--control-val", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    common = [
        "--primary-dev",
        str(args.primary_dev),
        "--primary-val",
        str(args.primary_val),
        "--config",
        str(args.config),
        "--rotation-scales",
        "0.5",
        "--lpips-weight",
        "0.003",
        "--steps",
        "1500",
        "--batch-size",
        "32",
        "--bootstrap-reps",
        "5000",
        "--seed",
        str(int(args.seed)),
    ]
    gan_dir = args.output_dir / "gan"
    control_dir = args.output_dir / "control"
    fusion_dir = args.output_dir / "fusion"
    run(
        [
            sys.executable,
            "-u",
            str(root / "train_fiber_residual_phase_gan.py"),
            *common,
            "--source-arm",
            "gan",
            "--adv-weights",
            "0.0015",
            "--output-dir",
            str(gan_dir),
        ],
        cwd=root,
        log_path=args.output_dir / "gan_driver.log",
    )
    run(
        [
            sys.executable,
            "-u",
            str(root / "train_fiber_residual_phase_gan.py"),
            *common,
            "--source-arm",
            "vqae_control",
            "--adv-weights",
            "0",
            "--control-dev",
            str(args.control_dev),
            "--control-val",
            str(args.control_val),
            "--output-dir",
            str(control_dir),
        ],
        cwd=root,
        log_path=args.output_dir / "control_driver.log",
    )
    run(
        [
            sys.executable,
            "-u",
            str(root / "diagnose_fiber_residual_frequency_fusion.py"),
            "--primary-val",
            str(args.primary_val),
            "--control-val",
            str(args.control_val),
            "--config",
            str(args.config),
            "--control-checkpoint",
            str(control_dir / "checkpoint_vqae_control_rot0.5_adv0.pt"),
            "--proposal-checkpoints",
            str(gan_dir / "checkpoint_gan_rot0.5_adv0.0015.pt"),
            "--cutoffs",
            "0.12,0.18",
            "--alphas",
            "0.5,0.75",
            "--top-exact",
            "4",
            "--batch-size",
            "32",
            "--bootstrap-reps",
            "5000",
            "--seed",
            str(int(args.seed)),
            "--output-dir",
            str(fusion_dir),
        ],
        cwd=root,
        log_path=args.output_dir / "fusion_driver.log",
    )
    payload = {
        "status": "FIBER_FUSION_MULTISEED_PIPELINE_COMPLETE",
        "label": args.label,
        "seed": int(args.seed),
        "primary_dev": str(args.primary_dev),
        "primary_val": str(args.primary_val),
        "control_dev": str(args.control_dev),
        "control_val": str(args.control_val),
        "gan_summary": str(gan_dir / "summary.json"),
        "control_summary": str(control_dir / "summary.json"),
        "fusion_summary": str(fusion_dir / "summary.json"),
        "runtime_seconds": time.time() - started,
    }
    output = args.output_dir / "pipeline_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
