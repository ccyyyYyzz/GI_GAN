from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze training convergence.")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_history(output_dir: Path) -> tuple[list[dict], str]:
    per_epoch = read_csv_rows(output_dir / "per_epoch_metrics.csv")
    if per_epoch:
        return normalize_rows(per_epoch, source="per_epoch_metrics"), "per_epoch_metrics.csv"
    eval_history = read_csv_rows(output_dir / "eval_history.csv")
    if eval_history:
        return normalize_rows(eval_history, source="eval_history"), "eval_history.csv"
    fallback_rows = []
    for name in ["val_metrics_latest.json", "best_hq_metrics.json", "best_score_metrics.json", "best_ssim_metrics.json"]:
        p = output_dir / name
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        model = data.get("model", {})
        back = data.get("backprojection", {})
        fallback_rows.append(
            {
                "epoch": name,
                "model_psnr": model.get("psnr", ""),
                "model_ssim": model.get("ssim", ""),
                "model_mse": model.get("mse", ""),
                "model_rel_meas_err": model.get("rel_meas_error", ""),
                "rel_meas_err_unclamped": model.get("rel_meas_err_unclamped", ""),
                "rel_meas_err_clamped": model.get("rel_meas_err_clamped", ""),
                "backproj_psnr": back.get("psnr", ""),
                "backproj_ssim": back.get("ssim", ""),
                "train_total_loss": "",
                "hq_score": "",
            }
        )
    return normalize_rows(fallback_rows, source="fallback_json"), "metric_snapshots"


def normalize_rows(rows: list[dict], source: str) -> list[dict]:
    normalized = []
    for row in rows:
        psnr = row.get("val_model_psnr", row.get("model_psnr", ""))
        ssim = row.get("val_model_ssim", row.get("model_ssim", ""))
        rel = row.get("val_model_rel_meas_err", row.get("model_rel_meas_error", row.get("model_rel_meas_err", "")))
        back_psnr = row.get("val_backproj_psnr", row.get("backproj_psnr", ""))
        back_ssim = row.get("val_backproj_ssim", row.get("backproj_ssim", ""))
        hq = row.get("hq_score", "")
        if hq in {"", None}:
            psnr_f = as_float(psnr)
            ssim_f = as_float(ssim)
            rel_f = as_float(rel) or 0.0
            hq = "" if psnr_f is None or ssim_f is None else psnr_f + 20.0 * ssim_f - 20.0 * rel_f
        normalized.append(
            {
                "epoch": row.get("epoch", ""),
                "model_psnr": psnr,
                "model_ssim": ssim,
                "model_mse": row.get("val_model_mse", row.get("model_mse", "")),
                "model_rel_meas_err": rel,
                "rel_meas_err_unclamped": row.get("rel_meas_err_unclamped", ""),
                "rel_meas_err_clamped": row.get("rel_meas_err_clamped", ""),
                "backproj_psnr": back_psnr,
                "backproj_ssim": back_ssim,
                "train_total_loss": row.get("train_total_loss", row.get("train_g_loss", "")),
                "hq_score": hq,
                "source": source,
            }
        )
    return normalized


def as_float(value):
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def epoch_label(row: dict, idx: int):
    value = as_float(row.get("epoch"))
    return idx if value is None else value


def plot(rows: list[dict], key: str, path: Path, ylabel: str) -> None:
    points = [(epoch_label(row, idx), as_float(row.get(key))) for idx, row in enumerate(rows)]
    points = [(x, y) for x, y in points if y is not None]
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".txt").write_text(f"Could not render plot: {exc}\n", encoding="utf-8")


def slope_last(rows: list[dict], key: str, window: int = 5) -> float:
    vals = [as_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    recent = vals[-window:]
    return float(recent[-1] - recent[0]) / max(1, len(recent) - 1)


def best_row(rows: list[dict]) -> tuple[dict | None, float]:
    best = None
    best_value = float("-inf")
    for row in rows:
        value = as_float(row.get("hq_score"))
        if value is None:
            psnr = as_float(row.get("model_psnr"))
            ssim = as_float(row.get("model_ssim"))
            value = None if psnr is None or ssim is None else psnr + 20.0 * ssim
        if value is not None and value > best_value:
            best = row
            best_value = value
    return best, best_value


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    rows, source = read_history(output_dir)
    plot(rows, "model_psnr", output_dir / "curve_psnr.png", "model PSNR")
    plot(rows, "model_ssim", output_dir / "curve_ssim.png", "model SSIM")
    plot(rows, "train_total_loss", output_dir / "curve_loss.png", "train total loss")
    plot(rows, "rel_meas_err_unclamped", output_dir / "curve_relmeaserr.png", "relative measurement error")
    plot(rows, "hq_score", output_dir / "curve_hq_score.png", "HQ score")

    psnr_slope = slope_last(rows, "model_psnr")
    ssim_slope = slope_last(rows, "model_ssim")
    loss_slope = slope_last(rows, "train_total_loss")
    hq_slope = slope_last(rows, "hq_score")
    best, best_hq = best_row(rows)
    continue_training = (psnr_slope > 0.02) or (ssim_slope > 0.001) or (hq_slope > 0.05)
    summary = {
        "rows_analyzed": len(rows),
        "source": source,
        "last_5_epoch_psnr_slope": psnr_slope,
        "last_5_epoch_ssim_slope": ssim_slope,
        "last_5_epoch_loss_slope": loss_slope,
        "last_5_epoch_hq_score_slope": hq_slope,
        "best_epoch": None if best is None else best.get("epoch"),
        "best_hq_score": None if best is None else best_hq,
        "continue_training_recommended": bool(continue_training),
    }
    (output_dir / "convergence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Convergence Summary",
        "",
        f"- rows_analyzed: {len(rows)}",
        f"- source: {source}",
        f"- last_5_epoch_psnr_slope: {psnr_slope:.6g}",
        f"- last_5_epoch_ssim_slope: {ssim_slope:.6g}",
        f"- last_5_epoch_loss_slope: {loss_slope:.6g}",
        f"- last_5_epoch_hq_score_slope: {hq_slope:.6g}",
        f"- best_epoch: {summary['best_epoch']}",
        f"- best_hq_score: {summary['best_hq_score']}",
        f"- continue_training_recommended: {continue_training}",
        "",
        "## Notes",
        "",
        "- Positive late PSNR, SSIM, or HQ-score slope indicates the run has not saturated.",
        "- Missing per-epoch history falls back to eval history or metric snapshots.",
    ]
    (output_dir / "convergence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Convergence summary written to: {output_dir / 'convergence_summary.md'}")


if __name__ == "__main__":
    main()
