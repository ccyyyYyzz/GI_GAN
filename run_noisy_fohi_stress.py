from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml


def run(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def achieved_bucket_snr(cache_path: Path) -> dict[str, float]:
    pack = torch.load(cache_path, map_location="cpu", weights_only=True)
    clean = pack["y_clean"].double()
    noisy = pack["y"].double()
    signal = torch.linalg.vector_norm(clean, dim=1)
    error = torch.linalg.vector_norm(noisy - clean, dim=1).clamp_min(1.0e-15)
    values = 20.0 * torch.log10(signal / error)
    return {
        "mean_db": float(values.mean()),
        "min_db": float(values.min()),
        "max_db": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clean-checkpoint-root", type=Path, required=True)
    parser.add_argument("--operator-seed", type=int, required=True)
    parser.add_argument("--lane-index", type=int, required=True)
    parser.add_argument("--snr-db", type=float, nargs="+", default=(30.0, 20.0))
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

    control_checkpoint = (
        args.clean_checkpoint_root / "control" / "checkpoint_vqae_control_rot0.5_adv0.pt"
    )
    proposal_checkpoint = (
        args.clean_checkpoint_root / "gan_adv0" / "checkpoint_gan_rot0.5_adv0.pt"
    )
    if not control_checkpoint.is_file() or not proposal_checkpoint.is_file():
        raise FileNotFoundError("CLEAN_FROZEN_CHECKPOINTS_REQUIRED")

    conditions = []
    for snr_db in args.snr_db:
        tag = f"snr{int(round(float(snr_db))):02d}"
        condition_dir = args.output_dir / tag
        cache_root = condition_dir / "cache"
        run(
            [
                sys.executable,
                "-u",
                str(root / "prepare_fiber_rate_caches.py"),
                "--bundle-root",
                str(args.asset_root),
                "--dataset-root",
                str(args.dataset_root),
                "--rates",
                "05",
                "--output-dir",
                str(cache_root),
                "--seed",
                str(int(args.lane_index)),
                "--bucket-snr-db",
                str(float(snr_db)),
                "--noise-seed-base",
                "20264000",
            ],
            cwd=root,
            log_path=condition_dir / "cache.log",
        )
        val = cache_root / "rate05" / f"seed{int(args.lane_index)}_val.pt"
        fohi_dir = condition_dir / "fohi"
        run(
            [
                sys.executable,
                "-u",
                str(root / "diagnose_fiber_orthogonal_highpass_innovation.py"),
                "--primary-val",
                str(val),
                "--control-val",
                str(val),
                "--config",
                str(config_path),
                "--control-checkpoint",
                str(control_checkpoint),
                "--proposal-checkpoint",
                str(proposal_checkpoint),
                "--filter-mode",
                "highpass",
                "--cutoff",
                "0.12",
                "--transition",
                "0.03",
                "--alpha",
                "0.5",
                "--batch-size",
                "32",
                "--exact-iterations",
                "4096",
                "--bootstrap-reps",
                "10000",
                "--seed",
                str(20265000 + int(args.lane_index) + int(round(float(snr_db))) * 10),
                "--output-dir",
                str(fohi_dir),
            ],
            cwd=root,
            log_path=condition_dir / "fohi.log",
        )
        summary = json.loads((fohi_dir / "summary.json").read_text(encoding="utf-8"))
        conditions.append(
            {
                "requested_bucket_snr_db": float(snr_db),
                "achieved_validation_bucket_snr": achieved_bucket_snr(val),
                "summary": str(fohi_dir / "summary.json"),
                "projection_converged": bool(summary["fohi_projection_audit"]["all_converged"]),
            }
        )

    payload = {
        "status": "FROZEN_FOHI_NOISY_EQUALITY_STRESS_COMPLETE",
        "interpretation": (
            "algorithmic exact-fit stress only; the clean truth is not generally in the noisy equality fiber"
        ),
        "operator_seed": int(args.operator_seed),
        "lane_index": int(args.lane_index),
        "validation_only": True,
        "test_split_opened": False,
        "frozen_clean_checkpoints": True,
        "conditions": conditions,
        "runtime_seconds": time.time() - started,
    }
    (args.output_dir / "campaign_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
