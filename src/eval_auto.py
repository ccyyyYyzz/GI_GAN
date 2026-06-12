from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .checkpoint_utils import find_best_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best available checkpoint in an output directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit_val_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint = find_best_checkpoint(output_dir)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found in {output_dir}. Expected one of best_hq.pt, "
            "best_score.pt, best_ssim.pt, best_psnr.pt, best_mse.pt, last.pt, or epoch_*.pt."
        )
    cmd = [
        sys.executable,
        "-m",
        "src.eval",
        "--config",
        args.config,
        "--checkpoint",
        str(checkpoint),
        "--output_dir",
        str(output_dir),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.limit_val_samples is not None:
        cmd.extend(["--limit_val_samples", str(args.limit_val_samples)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
