from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .checkpoint_utils import find_best_checkpoint
from .models import build_generator
from .phase26_common import (
    drive_root,
    ensure_dir,
    fmt,
    markdown_table,
    output_root,
    read_csv,
    read_json,
    safe_float,
    write_csv,
    write_json,
    write_text,
)
from .utils import apply_experiment_defaults, load_config


FIELDS = [
    "config_name",
    "family",
    "model_type",
    "sampling_ratio",
    "epochs_actual",
    "train_samples",
    "val_samples",
    "psnr",
    "ssim",
    "mse",
    "rel_meas_err",
    "params",
    "runtime_sec_per_image",
    "best_checkpoint",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 26 architecture pilot results.")
    parser.add_argument("--drive_root", default=None)
    return parser.parse_args()


def count_params(config: dict[str, Any]) -> int:
    model = build_generator(config, measurement=None)
    return int(sum(param.numel() for param in model.parameters()))


def metric(metrics: dict[str, Any], key: str) -> Any:
    return (metrics.get("model") or {}).get(key, "")


def infer_family(config_name: str) -> str:
    if "_rad5_" in config_name or config_name.endswith("_rad5_pilot"):
        return "rademacher"
    if "_scr5_" in config_name or config_name.endswith("_scr5_pilot"):
        return "scrambled_hadamard"
    return ""


def rows_from_manifest(root: Path) -> list[dict[str, Any]]:
    out = output_root(root)
    manifest_rows = read_csv(out / "arch_pilot_config_manifest.csv")
    rows = []
    for item in manifest_rows:
        config_name = item.get("config_name", "")
        config_path = Path(item.get("path", ""))
        output_dir = Path(item.get("output_dir", ""))
        notes = []
        status = "pending"
        metrics = {}
        if (output_dir / "eval_metrics.json").exists():
            metrics = read_json(output_dir / "eval_metrics.json")
            status = "complete"
        elif output_dir.exists():
            status = "incomplete"
            notes.append("output_dir exists but eval_metrics.json is missing")
        else:
            notes.append("not started")
        config = apply_experiment_defaults(load_config(config_path)) if config_path.exists() else {}
        per_epoch = read_csv(output_dir / "per_epoch_metrics.csv")
        epochs_actual = len(per_epoch)
        checkpoint = find_best_checkpoint(output_dir)
        try:
            params = count_params(config) if config else ""
        except Exception as exc:
            params = ""
            notes.append(f"param_count_failed={exc!r}")
        rows.append(
            {
                "config_name": config_name,
                "family": item.get("family") or infer_family(config_name),
                "model_type": item.get("model_type") or config.get("model_type", ""),
                "sampling_ratio": config.get("sampling_ratio", item.get("sampling_ratio", "")),
                "epochs_actual": epochs_actual,
                "train_samples": config.get("limit_train_samples", item.get("limit_train_samples", "")),
                "val_samples": config.get("limit_val_samples", item.get("limit_val_samples", "")),
                "psnr": metric(metrics, "psnr"),
                "ssim": metric(metrics, "ssim"),
                "mse": metric(metrics, "mse"),
                "rel_meas_err": metric(metrics, "rel_meas_error"),
                "params": params,
                "runtime_sec_per_image": "",
                "best_checkpoint": str(checkpoint) if checkpoint else "",
                "status": status,
                "notes": "; ".join(notes),
            }
        )
    return rows


def plot_bar(rows: list[dict[str, Any]], metric_name: str, path: Path, ylabel: str) -> None:
    ensure_dir(path.parent)
    completed = [row for row in rows if row.get("status") == "complete" and safe_float(row.get(metric_name)) == safe_float(row.get(metric_name))]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=160)
    labels = [row["config_name"].replace("_pilot", "") for row in completed]
    values = [safe_float(row.get(metric_name)) for row in completed]
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_runtime_params(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    completed = [row for row in rows if row.get("status") == "complete"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=160)
    for row in completed:
        ax.scatter(safe_float(row.get("params")) / 1e6, safe_float(row.get("psnr")), label=row["config_name"].replace("_pilot", ""))
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Pilot PSNR (dB)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = drive_root(args.drive_root)
    out = output_root(root)
    rows = rows_from_manifest(root)
    write_csv(out / "arch_pilot_results.csv", rows, FIELDS)
    pretty = []
    for row in rows:
        item = dict(row)
        for key in ["sampling_ratio", "psnr", "ssim", "mse", "rel_meas_err", "runtime_sec_per_image"]:
            item[key] = fmt(item.get(key), 4)
        pretty.append(item)
    write_text(
        out / "arch_pilot_results.md",
        "# Phase 26 Architecture Medium Pilot Results\n\n"
        + "These are medium pilot planning results, not final paper results.\n\n"
        + markdown_table(pretty, FIELDS),
    )
    plot_bar(rows, "psnr", out / "arch_pilot_psnr.png", "Pilot PSNR (dB)")
    plot_bar(rows, "ssim", out / "arch_pilot_ssim.png", "Pilot SSIM")
    plot_runtime_params(rows, out / "arch_pilot_runtime_params.png")
    write_json(out / "arch_pilot_results.json", {"rows": rows})
    print({"rows": len(rows), "output": str(out / "arch_pilot_results.csv")})


if __name__ == "__main__":
    main()
