from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

from .utils import ensure_dir, load_config


DEFAULT_OUTPUT_ROOT = Path("E:/ns_mc_gan_gi/outputs")
DEFAULT_OUTPUT_PREFIX = "phase2"
FIELDS = [
    "sampling_ratio",
    "m",
    "n",
    "backproj_mse",
    "backproj_psnr",
    "backproj_ssim",
    "backproj_rel_meas_err",
    "model_mse",
    "model_psnr",
    "model_ssim",
    "model_rel_meas_err",
    "delta_mse",
    "delta_psnr",
    "delta_ssim",
    "checkpoint",
    "sample_image",
    "run_report",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate quick sampling results.")
    parser.add_argument("--base_dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def make_runs(base_dir: Path) -> list[tuple[float, str, Path]]:
    return [
        (0.01, "quick_1pct", base_dir / "quick_1pct"),
        (0.02, "quick_2pct", base_dir / "quick_2pct"),
        (0.05, "quick_5pct", base_dir / "quick_5pct"),
        (0.10, "quick_10pct", base_dir / "quick_10pct"),
    ]


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_config(run_dir: Path):
    path = run_dir / "resolved_config.yaml"
    if path.exists():
        return load_config(path)
    return {}


def build_row(ratio: float, label: str, run_dir: Path) -> dict:
    metrics = read_json(run_dir / "eval_metrics.json") or read_json(run_dir / "best_metrics.json")
    config = read_config(run_dir)
    checkpoint = run_dir / "best_ssim.pt"
    sample = run_dir / "eval_samples" / "recon_grid.png"
    if not sample.exists():
        sample = run_dir / "samples" / "epoch_010.png"
    report = run_dir / "RUN_REPORT.md"
    n = int(config.get("img_size", 64)) ** 2
    m = int(round(ratio * n))

    row = {field: "" for field in FIELDS}
    row.update(
        {
            "sampling_ratio": ratio,
            "m": m,
            "n": n,
            "checkpoint": str(checkpoint) if checkpoint.exists() else "",
            "sample_image": str(sample) if sample.exists() else "",
            "run_report": str(report) if report.exists() else "",
            "status": "ok" if metrics else "missing",
        }
    )
    if not metrics:
        return row

    back = metrics.get("backprojection", {})
    model = metrics.get("model", {})
    improve = metrics.get("improvement", {})
    row.update(
        {
            "backproj_mse": back.get("mse", ""),
            "backproj_psnr": back.get("psnr", ""),
            "backproj_ssim": back.get("ssim", ""),
            "backproj_rel_meas_err": back.get("rel_meas_error", ""),
            "model_mse": model.get("mse", ""),
            "model_psnr": model.get("psnr", ""),
            "model_ssim": model.get("ssim", ""),
            "model_rel_meas_err": model.get("rel_meas_error", ""),
            "delta_mse": improve.get("delta_mse", ""),
            "delta_psnr": improve.get("delta_psnr", ""),
            "delta_ssim": improve.get("delta_ssim", ""),
        }
    )
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(rows: list[dict], path: Path) -> None:
    cols = [
        "sampling_ratio",
        "m",
        "backproj_psnr",
        "model_psnr",
        "delta_psnr",
        "backproj_ssim",
        "model_ssim",
        "delta_ssim",
        "status",
    ]
    lines = ["# Phase 2 Sampling Results", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric(rows: list[dict], metric: str, ylabel: str, path: Path) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        _fallback_plot([], [], [], ylabel, path)
        return
    xs = [float(row["sampling_ratio"]) for row in ok_rows]
    back = [float(row[f"backproj_{metric}"]) for row in ok_rows]
    model = [float(row[f"model_{metric}"]) for row in ok_rows]
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, back, marker="o", label="Backprojection")
        ax.plot(xs, model, marker="o", label="NS-MC-GAN")
        ax.set_xlabel("sampling_ratio")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        _fallback_plot(xs, back, model, ylabel, path)


def _fallback_plot(xs: list[float], back: list[float], model: list[float], ylabel: str, path: Path) -> None:
    try:
        from PIL import Image, ImageDraw

        width, height = 720, 460
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 10), f"{ylabel} vs sampling_ratio", fill="black")
        left, top, right, bottom = 70, 50, 680, 410
        draw.rectangle((left, top, right, bottom), outline="black")
        if xs and back and model:
            y_values = back + model
            y_min, y_max = min(y_values), max(y_values)
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0
            x_min, x_max = min(xs), max(xs)
            if x_min == x_max:
                x_min -= 0.01
                x_max += 0.01

            def pt(x, y):
                px = left + (x - x_min) / (x_max - x_min) * (right - left)
                py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
                return int(px), int(py)

            for series, color in [(back, "gray"), (model, "blue")]:
                points = [pt(x, y) for x, y in zip(xs, series)]
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
                for p in points:
                    draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=color)
            draw.text((left, bottom + 8), "Backprojection=gray, NS-MC-GAN=blue", fill="black")
        else:
            draw.text((left + 20, top + 30), "missing results", fill="black")
        img.save(path)
    except Exception:
        path.with_suffix(".txt").write_text("Plot unavailable.\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.base_dir)
    prefix = args.output_prefix
    rows = [build_row(ratio, label, run_dir) for ratio, label, run_dir in make_runs(output_root)]
    write_csv(rows, output_root / f"{prefix}_results.csv")
    write_markdown(rows, output_root / f"{prefix}_results.md")
    plot_metric(rows, "psnr", "PSNR", output_root / f"{prefix}_psnr_vs_sampling.png")
    plot_metric(rows, "ssim", "SSIM", output_root / f"{prefix}_ssim_vs_sampling.png")
    plot_metric(rows, "mse", "MSE", output_root / f"{prefix}_mse_vs_sampling.png")
    plot_metric(
        rows,
        "rel_meas_err",
        "relative measurement error",
        output_root / f"{prefix}_relmeaserr_vs_sampling.png",
    )
    print(f"Wrote sampling aggregation to: {output_root}")


if __name__ == "__main__":
    main()
