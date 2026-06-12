from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .sample_export import save_example_grid
from .utils import ensure_dir


DEFAULT_OUTPUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase8/quality_audit/examples")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", nargs="*")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["psnr", "ssim", "mse", "rel_meas_err"]:
            try:
                row[key] = float(row[key])
            except Exception:
                row[key] = float("nan")
    return rows


def _median_eight(rows: list[dict], key: str) -> list[dict]:
    if not rows:
        return []
    mid = len(rows) // 2
    start = max(0, mid - 4)
    return rows[start : start + 8]


def export_for_csv(path: Path, output_dir: Path) -> list[Path]:
    method = path.parent.parent.name if path.parent.name == "eval_samples_individual" else path.stem
    rows = _read_rows(path)
    written = []
    for key in ["psnr", "ssim"]:
        sorted_rows = sorted(rows, key=lambda r: r[key])
        groups = {
            "best": list(reversed(sorted_rows[-8:])),
            "median": _median_eight(sorted_rows, key),
            "worst": sorted_rows[:8],
        }
        for group, selected in groups.items():
            out = output_dir / f"{method}_{group}_{key}.png"
            title = f"{method} {group} {key}"
            save_example_grid(selected, out, title=title, max_items=8)
            if out.exists():
                written.append(out)
    return written


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    csv_paths = [Path(p) for p in args.csv_paths]
    if not csv_paths:
        roots = [
            Path("E:/ns_mc_gan_gi/outputs_phase8/fixed_wide_5pct"),
            Path("E:/ns_mc_gan_gi/outputs_phase8/fixed_wide_refiner_5pct"),
            Path("E:/ns_mc_gan_gi/outputs_phase8/continuous_physical_wide_5pct"),
            Path("E:/ns_mc_gan_gi/outputs_phase8/continuous_g_only_wide_5pct"),
            Path("E:/ns_mc_gan_gi/outputs_phase8/direct_y_fixed_5pct"),
            Path("E:/ns_mc_gan_gi/outputs_phase8/mnist_fixed_5pct"),
            Path("E:/ns_mc_gan_gi/outputs_phase8/mnist_continuous_5pct"),
        ]
        csv_paths = [root / "eval_samples_individual" / "per_sample_metrics.csv" for root in roots]
    written = []
    for path in csv_paths:
        if path.exists():
            written.extend(export_for_csv(path, output_dir))
    print(f"Wrote {len(written)} example grids to {output_dir}")


if __name__ == "__main__":
    main()
