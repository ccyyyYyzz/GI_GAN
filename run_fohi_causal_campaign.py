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
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dev", type=Path, required=True)
    parser.add_argument("--primary-val", type=Path, required=True)
    parser.add_argument("--control-dev", type=Path, required=True)
    parser.add_argument("--control-val", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--structural-checkpoint", type=Path, required=True)
    parser.add_argument("--main-gan-checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    adv0_dir = args.output_dir / "train_gan_adv0"
    vqae2_dir = args.output_dir / "train_vqae2"
    train_common = [
        "--primary-dev", str(args.primary_dev),
        "--primary-val", str(args.primary_val),
        "--config", str(args.config),
        "--lpips-weight", "0.003",
        "--steps", str(int(args.steps)),
        "--batch-size", "32",
        "--bootstrap-reps", "5000",
        "--seed", str(int(args.seed)),
    ]
    run(
        [
            sys.executable, "-u", str(root / "train_fiber_residual_phase_gan.py"),
            *train_common,
            "--rotation-scales", "0.5",
            "--source-arm", "gan",
            "--adv-weights", "0",
            "--output-dir", str(adv0_dir),
        ],
        cwd=root,
        log_path=args.output_dir / "train_gan_adv0.log",
    )
    run(
        [
            sys.executable, "-u", str(root / "train_fiber_residual_phase_gan.py"),
            *train_common,
            "--rotation-scales", "0.25",
            "--source-arm", "vqae_control",
            "--adv-weights", "0",
            "--control-dev", str(args.control_dev),
            "--control-val", str(args.control_val),
            "--output-dir", str(vqae2_dir),
        ],
        cwd=root,
        log_path=args.output_dir / "train_vqae2.log",
    )

    cells = {
        "A_gan_adv_highpass": (args.main_gan_checkpoint, "highpass"),
        "B_gan_adv0_highpass": (adv0_dir / "checkpoint_gan_rot0.5_adv0.pt", "highpass"),
        "C_vqae2_highpass": (vqae2_dir / "checkpoint_vqae_control_rot0.25_adv0.pt", "highpass"),
        "D_gan_adv_lowpass": (args.main_gan_checkpoint, "lowpass"),
    }
    for index, (name, (proposal, filter_mode)) in enumerate(cells.items()):
        cell_dir = args.output_dir / name
        run(
            [
                sys.executable, "-u",
                str(root / "diagnose_fiber_orthogonal_highpass_innovation.py"),
                "--primary-val", str(args.primary_val),
                "--control-val", str(args.control_val),
                "--config", str(args.config),
                "--control-checkpoint", str(args.structural_checkpoint),
                "--proposal-checkpoint", str(proposal),
                "--filter-mode", filter_mode,
                "--cutoff", "0.12",
                "--transition", "0.03",
                "--alpha", "0.5",
                "--batch-size", "32",
                "--exact-iterations", "4096",
                "--bootstrap-reps", "10000",
                "--seed", str(int(args.seed) + 100 + index),
                "--output-dir", str(cell_dir),
            ],
            cwd=root,
            log_path=args.output_dir / f"{name}.log",
        )

    payload = {
        "status": "FOHI_FOUR_CELL_CAUSAL_CAMPAIGN_COMPLETE",
        "label": args.label,
        "steps": int(args.steps),
        "seed": int(args.seed),
        "validation_only": True,
        "test_split_opened": False,
        "cells": {name: str(args.output_dir / name / "summary.json") for name in cells},
        "runtime_seconds": time.time() - started,
    }
    (args.output_dir / "campaign_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
