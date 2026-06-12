from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .phase14_ablation_pack_common import (
    bootstrap_ci,
    copy_sample_grid,
    f,
    load_main_rows,
    out_dir,
    per_sample_rows,
    write_rows,
)


def values(rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for row in rows:
        v = f(row.get(key))
        if v is not None and math.isfinite(v):
            out.append(v)
    return out


def metric_summary(vals: list[float]) -> dict[str, float | str]:
    if not vals:
        return {"mean": "", "std": "", "median": "", "ci95_low": "", "ci95_high": ""}
    arr = np.array(vals, dtype=float)
    lo, hi = bootstrap_ci(vals)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "ci95_low": lo,
        "ci95_high": hi,
    }


def plot_hist(all_values: dict[str, list[float]], path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(8.5, 5.0))
    for label, vals in all_values.items():
        if vals:
            plt.hist(vals, bins=18, alpha=0.45, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    rows = []
    psnr_hists: dict[str, list[float]] = {}
    ssim_hists: dict[str, list[float]] = {}
    for method in load_main_rows():
        label = method.get("display_name") or method.get("method_id")
        samples = per_sample_rows(method)
        psnr_values = values(samples, "psnr")
        ssim_values = values(samples, "ssim")
        if not psnr_values:
            v = f(method.get("psnr"))
            psnr_values = [v] if v is not None else []
        if not ssim_values:
            v = f(method.get("ssim"))
            ssim_values = [v] if v is not None else []
        ps = metric_summary(psnr_values)
        ss = metric_summary(ssim_values)
        short = str(method.get("method_id", label)).replace("stl10_", "").replace("_colab_full", "")
        grid = copy_sample_grid(method, short, "representative_grid")
        rows.append(
            {
                "method": label,
                "dataset": method.get("dataset"),
                "sampling_ratio": method.get("sampling_ratio"),
                "num_samples": len(samples) if samples else len(psnr_values),
                "psnr_mean": ps["mean"],
                "psnr_std": ps["std"],
                "psnr_median": ps["median"],
                "psnr_ci95_low": ps["ci95_low"],
                "psnr_ci95_high": ps["ci95_high"],
                "ssim_mean": ss["mean"],
                "ssim_std": ss["std"],
                "ssim_median": ss["median"],
                "ssim_ci95_low": ss["ci95_low"],
                "ssim_ci95_high": ss["ci95_high"],
                "sample_grid": grid,
                "status": "per_sample_available" if samples else "aggregate_only",
            }
        )
        psnr_hists[label] = psnr_values
        ssim_hists[label] = ssim_values
    fields = [
        "method",
        "dataset",
        "sampling_ratio",
        "num_samples",
        "psnr_mean",
        "psnr_std",
        "psnr_median",
        "psnr_ci95_low",
        "psnr_ci95_high",
        "ssim_mean",
        "ssim_std",
        "ssim_median",
        "ssim_ci95_low",
        "ssim_ci95_high",
        "sample_grid",
        "status",
    ]
    write_rows("statistics_summary", rows, fields)
    plot_hist(psnr_hists, out_dir() / "psnr_histograms.png", "Per-sample PSNR distributions", "PSNR")
    plot_hist(ssim_hists, out_dir() / "ssim_histograms.png", "Per-sample SSIM distributions", "SSIM")
    print(f"Wrote statistics summary with {len(rows)} rows")


if __name__ == "__main__":
    main()
