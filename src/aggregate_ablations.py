from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

from .utils import ensure_dir, load_config


DEFAULT_OUTPUT_ROOT = Path("E:/ns_mc_gan_gi/outputs")
DEFAULT_OUTPUT_PREFIX = "phase2_ablation"
FIELDS = [
    "method",
    "use_null_project",
    "use_dc_project",
    "use_adversarial",
    "model_mse",
    "model_psnr",
    "model_ssim",
    "model_rel_meas_err",
    "backproj_mse",
    "backproj_psnr",
    "backproj_ssim",
    "backproj_rel_meas_err",
    "delta_psnr",
    "delta_ssim",
    "checkpoint",
    "sample_image",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate 5% ablation results.")
    parser.add_argument("--base_dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def make_runs(base_dir: Path) -> list[tuple[str, Path]]:
    return [
        ("Full NS-MC-GAN", base_dir / "quick_5pct"),
        ("No Null Projection", base_dir / "ablation_5pct_no_null"),
        ("No DC Projection", base_dir / "ablation_5pct_no_dc"),
        ("No Adversarial", base_dir / "ablation_5pct_no_adv"),
    ]


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_row(method: str, run_dir: Path) -> dict:
    config_path = run_dir / "resolved_config.yaml"
    config = load_config(config_path) if config_path.exists() else {}
    metrics = read_json(run_dir / "eval_metrics.json") or read_json(run_dir / "best_metrics.json")
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "method": method,
            "use_null_project": config.get("use_null_project", ""),
            "use_dc_project": config.get("use_dc_project", ""),
            "use_adversarial": config.get("use_adversarial", ""),
            "checkpoint": str(run_dir / "best_ssim.pt") if (run_dir / "best_ssim.pt").exists() else "",
            "sample_image": str(run_dir / "eval_samples" / "recon_grid.png")
            if (run_dir / "eval_samples" / "recon_grid.png").exists()
            else "",
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
            "model_mse": model.get("mse", ""),
            "model_psnr": model.get("psnr", ""),
            "model_ssim": model.get("ssim", ""),
            "model_rel_meas_err": model.get("rel_meas_error", ""),
            "backproj_mse": back.get("mse", ""),
            "backproj_psnr": back.get("psnr", ""),
            "backproj_ssim": back.get("ssim", ""),
            "backproj_rel_meas_err": back.get("rel_meas_error", ""),
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
    cols = ["method", "model_psnr", "model_ssim", "model_rel_meas_err", "delta_psnr", "delta_ssim", "status"]
    lines = ["# Phase 2 Ablation Results", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_bar(rows: list[dict], field: str, ylabel: str, path: Path) -> None:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    labels = [row["method"] for row in ok_rows]
    values = [float(row[field]) for row in ok_rows if row.get(field) != ""]
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        _fallback_bar(labels, values, ylabel, path)


def _fallback_bar(labels: list[str], values: list[float], ylabel: str, path: Path) -> None:
    try:
        from PIL import Image, ImageDraw

        width, height = 820, 460
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 10), ylabel, fill="black")
        if not labels or not values:
            draw.text((80, 80), "missing results", fill="black")
            img.save(path)
            return
        max_value = max(values) if max(values) > 0 else 1.0
        bar_w = 120
        gap = 40
        base = 390
        for idx, (label, value) in enumerate(zip(labels, values)):
            x = 60 + idx * (bar_w + gap)
            h = int((value / max_value) * 300)
            draw.rectangle((x, base - h, x + bar_w, base), fill="steelblue")
            draw.text((x, base + 8), label[:16], fill="black")
            draw.text((x, base - h - 18), f"{value:.3f}", fill="black")
        img.save(path)
    except Exception:
        path.with_suffix(".txt").write_text("Plot unavailable.\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.base_dir)
    prefix = args.output_prefix
    rows = [build_row(method, run_dir) for method, run_dir in make_runs(output_root)]
    write_csv(rows, output_root / f"{prefix}_results.csv")
    write_markdown(rows, output_root / f"{prefix}_results.md")
    plot_bar(rows, "model_psnr", "Model PSNR", output_root / f"{prefix}_bar_psnr.png")
    plot_bar(rows, "model_ssim", "Model SSIM", output_root / f"{prefix}_bar_ssim.png")
    plot_bar(
        rows,
        "model_rel_meas_err",
        "Model relative measurement error",
        output_root / f"{prefix}_bar_relmeaserr.png",
    )
    print(f"Wrote ablation aggregation to: {output_root}")


if __name__ == "__main__":
    main()
