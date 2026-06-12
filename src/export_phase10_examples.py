from __future__ import annotations

import csv
import shutil
from pathlib import Path

from .sample_export import save_example_grid
from .utils import ensure_dir


ROOT = Path("E:/ns_mc_gan_gi/outputs_phase10")
RESULTS = ROOT / "phase10_results.csv"
EXPORT_ROOT = ROOT / "paper_examples"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value):
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def experiment_dir(row: dict) -> Path:
    checkpoint = row.get("checkpoint")
    if checkpoint:
        return Path(checkpoint).parent
    sample = row.get("sample_image")
    if sample:
        return Path(sample).parents[1]
    return ROOT / row.get("method", "missing")


def select_triplet(rows: list[dict], key: str) -> list[tuple[str, list[dict]]]:
    valid = [row for row in rows if as_float(row.get(key)) is not None]
    if not valid:
        return []
    valid.sort(key=lambda row: float(row[key]))
    return [
        ("worst", [valid[0]]),
        ("median", [valid[len(valid) // 2]]),
        ("best", [valid[-1]]),
    ]


def export_from_individuals(exp_dir: Path, out_dir: Path, row: dict) -> bool:
    per_sample = exp_dir / "eval_samples_individual" / "per_sample_metrics.csv"
    samples = read_rows(per_sample)
    if not samples:
        return False
    for metric in ["psnr", "ssim"]:
        for label, selected in select_triplet(samples, metric):
            first = selected[0]
            title = (
                f"{row.get('method')} | {row.get('dataset_name')} | "
                f"{row.get('sampling_ratio')} | {row.get('pattern_type')} | "
                f"PSNR={first.get('psnr')} | SSIM={first.get('ssim')} | "
                f"RelMeasErr={first.get('rel_meas_err')} | {metric.upper()} {label}"
            )
            save_example_grid(selected, out_dir / f"{label}_{metric}_grid.png", title=title, max_items=len(selected))
    return True


def export_from_grid(exp_dir: Path, out_dir: Path, row: dict) -> bool:
    grid = exp_dir / "eval_samples" / "recon_grid.png"
    if not grid.exists() and row.get("sample_image"):
        grid = Path(row["sample_image"])
    if not grid.exists():
        return False
    for name in [
        "best_psnr_grid.png",
        "median_psnr_grid.png",
        "worst_psnr_grid.png",
        "best_ssim_grid.png",
        "median_ssim_grid.png",
        "worst_ssim_grid.png",
    ]:
        shutil.copyfile(grid, out_dir / name)
    (out_dir / "README.txt").write_text(
        "Fallback export: per-sample metrics were unavailable, so the eval grid was copied "
        "to the standard best/median/worst filenames.\n",
        encoding="utf-8",
    )
    return True


def main() -> None:
    ensure_dir(EXPORT_ROOT)
    rows = [row for row in read_rows(RESULTS) if row.get("phase") == "phase10" and row.get("status") == "completed"]
    manifest = []
    for row in rows:
        method = row.get("method", "unknown")
        out_dir = ensure_dir(EXPORT_ROOT / method)
        exp_dir = experiment_dir(row)
        mode = "individual" if export_from_individuals(exp_dir, out_dir, row) else ""
        if not mode:
            mode = "grid_fallback" if export_from_grid(exp_dir, out_dir, row) else "missing_samples"
        manifest.append({"method": method, "experiment_dir": str(exp_dir), "export_dir": str(out_dir), "mode": mode})
    with (EXPORT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["method", "experiment_dir", "export_dir", "mode"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Phase 10 examples exported to: {EXPORT_ROOT}")


if __name__ == "__main__":
    main()
