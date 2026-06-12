from __future__ import annotations

import csv
from pathlib import Path

from .sample_export import save_example_grid
from .utils import ensure_dir


OUTPUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase8_hq")
METHODS = [
    "hadamard_hq_10pct",
    "hadamard_hq_5pct",
    "scrambled_hadamard_hq_10pct",
    "rademacher_hq_10pct",
    "rademacher_hq_5pct",
    "continuous_physical_hq_10pct",
    "continuous_physical_hq_5pct",
    "mnist_hq_5pct",
    "fashion_mnist_hq_5pct",
    "cifar10_gray_hq_10pct",
]


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


def _median(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    mid = len(rows) // 2
    start = max(0, mid - 4)
    return rows[start : start + 8]


def export_method(method: str) -> int:
    csv_path = OUTPUT_DIR / method / "eval_samples_individual" / "per_sample_metrics.csv"
    if not csv_path.exists():
        return 0
    rows = _read_rows(csv_path)
    out_dir = ensure_dir(OUTPUT_DIR / "paper_examples" / method)
    count = 0
    for key in ["psnr", "ssim"]:
        sorted_rows = sorted(rows, key=lambda r: r[key])
        groups = {
            "best": list(reversed(sorted_rows[-8:])),
            "median": _median(sorted_rows),
            "worst": sorted_rows[:8],
        }
        for group, selected in groups.items():
            path = out_dir / f"{group}_{key}_grid.png"
            title = f"{method} {group} {key}"
            save_example_grid(selected, path, title=title, max_items=8)
            if path.exists():
                count += 1
    return count


def main() -> None:
    ensure_dir(OUTPUT_DIR / "paper_examples")
    total = sum(export_method(method) for method in METHODS)
    print(f"Wrote {total} HQ example grids to {OUTPUT_DIR / 'paper_examples'}")


if __name__ == "__main__":
    main()
