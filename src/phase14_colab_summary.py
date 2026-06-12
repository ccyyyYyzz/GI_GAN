from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Any

from .phase14_common import PHASE14_IMPORTS, ensure_dir, row_from_output, write_csv, write_json, write_md_table


def gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 14 Colab outputs.")
    parser.add_argument("--base_dir", default="/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase14_colab")
    parser.add_argument("--output_dir", default="/content/drive/MyDrive/ns_mc_gan_gi/outputs_phase14_colab/_summary")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    rows: list[dict[str, Any]] = []
    for exp in PHASE14_IMPORTS:
        colab_name = exp["config"].stem
        rows.append(row_from_output({**exp, "path": base_dir / colab_name}))

    fields = [
        "method_id",
        "display_name",
        "status",
        "sampling_ratio",
        "pattern_type",
        "psnr",
        "ssim",
        "backproj_psnr",
        "backproj_ssim",
        "threshold_reached",
        "best_checkpoint_path",
        "checkpoint_sha256",
    ]
    write_csv(output_dir / "phase14_colab_summary.csv", rows, fields)
    write_md_table(output_dir / "phase14_colab_summary.md", rows, fields)
    write_json(
        output_dir / "phase14_colab_runtime.json",
        {"gpu": gpu_name(), "python": platform.python_version(), "platform": platform.platform()},
    )
    print(f"Wrote {output_dir / 'phase14_colab_summary.md'}")


if __name__ == "__main__":
    main()
