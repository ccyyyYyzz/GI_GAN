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


def diagnostic_command(
    *,
    root: Path,
    primary_val: Path,
    control_val: Path,
    config: Path,
    structural_checkpoint: Path,
    proposal_checkpoint: Path,
    filter_mode: str,
    seed: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable, "-u", str(root / "diagnose_fiber_orthogonal_highpass_innovation.py"),
        "--primary-val", str(primary_val),
        "--control-val", str(control_val),
        "--config", str(config),
        "--control-checkpoint", str(structural_checkpoint),
        "--proposal-checkpoint", str(proposal_checkpoint),
        "--filter-mode", filter_mode,
        "--cutoff", "0.12",
        "--transition", "0.03",
        "--alpha", "0.5",
        "--batch-size", "32",
        "--exact-iterations", "4096",
        "--bootstrap-reps", "10000",
        "--seed", str(int(seed)),
        "--final-target", "legacy_clipped_anchor",
        "--output-dir", str(output_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--five-primary-val", type=Path, required=True)
    parser.add_argument("--five-control-val", type=Path, required=True)
    parser.add_argument("--five-config", type=Path, required=True)
    parser.add_argument("--five-structural-checkpoint", type=Path, required=True)
    parser.add_argument("--five-adv0-checkpoint", type=Path, required=True)
    parser.add_argument("--rate-seed", type=int, required=True)
    parser.add_argument("--rate-cache-root", type=Path, required=True)
    parser.add_argument("--rate-bundle-root", type=Path, required=True)
    parser.add_argument("--rate-result-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    five_lowpass = args.output_dir / "five_percent_adv0_lowpass"
    run(
        diagnostic_command(
            root=root,
            primary_val=args.five_primary_val,
            control_val=args.five_control_val,
            config=args.five_config,
            structural_checkpoint=args.five_structural_checkpoint,
            proposal_checkpoint=args.five_adv0_checkpoint,
            filter_mode="lowpass",
            seed=20261000 + int(args.rate_seed),
            output_dir=five_lowpass,
        ),
        cwd=root,
        log_path=args.output_dir / "five_percent_adv0_lowpass.log",
    )

    rate_outputs = {}
    for rate in ("02", "10"):
        cache_dir = args.rate_cache_root / f"rate{rate}"
        rate_root = args.rate_result_root / f"rate{rate}"
        config = args.rate_bundle_root / f"config_rate{rate}.yaml"
        dev = cache_dir / f"seed{int(args.rate_seed)}_dev.pt"
        val = cache_dir / f"seed{int(args.rate_seed)}_val.pt"
        train_dir = args.output_dir / f"rate{rate}_train_gan_adv0"
        training_seed = 20260718 + 100 * int(args.rate_seed) + int(rate)
        run(
            [
                sys.executable, "-u", str(root / "train_fiber_residual_phase_gan.py"),
                "--primary-dev", str(dev),
                "--primary-val", str(val),
                "--config", str(config),
                "--rotation-scales", "0.5",
                "--lpips-weight", "0.003",
                "--steps", str(int(args.steps)),
                "--batch-size", "32",
                "--bootstrap-reps", "5000",
                "--seed", str(training_seed),
                "--source-arm", "gan",
                "--adv-weights", "0",
                "--output-dir", str(train_dir),
            ],
            cwd=root,
            log_path=args.output_dir / f"rate{rate}_train_gan_adv0.log",
        )
        evaluation = args.output_dir / f"rate{rate}_adv0_highpass"
        run(
            diagnostic_command(
                root=root,
                primary_val=val,
                control_val=val,
                config=config,
                structural_checkpoint=rate_root / "control/checkpoint_vqae_control_rot0.5_adv0.pt",
                proposal_checkpoint=train_dir / "checkpoint_gan_rot0.5_adv0.pt",
                filter_mode="highpass",
                seed=training_seed + 1000,
                output_dir=evaluation,
            ),
            cwd=root,
            log_path=args.output_dir / f"rate{rate}_adv0_highpass.log",
        )
        rate_outputs[rate] = str(evaluation / "summary.json")

    payload = {
        "status": "SELECTED_DISCRIMINATOR_OFF_FOHI_EXTENSION_COMPLETE",
        "rate_seed": int(args.rate_seed),
        "steps": int(args.steps),
        "validation_only": True,
        "test_split_opened": False,
        "five_percent_lowpass": str(five_lowpass / "summary.json"),
        "rate_outputs": rate_outputs,
        "runtime_seconds": time.time() - started,
    }
    (args.output_dir / "campaign_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
