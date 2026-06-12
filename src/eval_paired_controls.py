from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .datasets import get_val_dataloader
from .eval import make_measurement
from .metrics import batch_metrics
from .models import ResidualUNetGenerator
from .utils import (
    apply_experiment_defaults,
    ensure_dir,
    load_config,
    reconstruct_from_measurements,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Paired control evaluation with bootstrap CIs.")
    parser.add_argument("--experiments_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def read_experiments(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_experiment(exp: dict, device: torch.device) -> tuple[dict, dict | None, ResidualUNetGenerator | None, object | None, str]:
    checkpoint_path = Path(exp["checkpoint"])
    config_path = Path(exp["config"])
    if not config_path.exists():
        return {}, None, None, None, "missing_config"
    config = apply_experiment_defaults(load_config(config_path))
    if not checkpoint_path.exists():
        return config, None, None, None, "missing_checkpoint"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        merged = dict(config)
        merged.update(checkpoint["config"])
        config = apply_experiment_defaults(merged)
    measurement = make_measurement(config, device)
    pattern_bank = getattr(measurement, "pattern_bank", None)
    if bool(config.get("use_learned_patterns", False)):
        if not isinstance(checkpoint, dict) or "pattern_bank" not in checkpoint:
            return config, checkpoint, None, None, "missing_pattern_bank"
        pattern_bank.load_state_dict(checkpoint["pattern_bank"])
        pattern_bank.set_tau(float(config.get("pattern_tau_final", config.get("pattern_tau", 1.0))))
        pattern_bank.eval()
    generator = ResidualUNetGenerator().to(device)
    state = checkpoint["generator"] if isinstance(checkpoint, dict) else checkpoint
    generator.load_state_dict(state)
    generator.eval()
    return config, checkpoint, generator, measurement, "ok"


def evaluate_experiment(exp: dict, loader, device: torch.device, seed: int) -> list[dict]:
    config, _checkpoint, generator, measurement, status = load_experiment(exp, device)
    if status != "ok":
        return [{"name": exp["name"], "sample_index": "", "status": status}]
    rows = []
    set_seed(seed)
    with torch.no_grad():
        for sample_index, batch in enumerate(tqdm(loader, desc=exp["name"])):
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            x_hat, _x_data = reconstruct_from_measurements(
                generator,
                measurement,
                y,
                use_null_project=bool(config["use_null_project"]),
                use_dc_project=bool(config["use_dc_project"]),
            )
            metrics = batch_metrics(x_hat, x, measurement, y)
            rows.append(
                {
                    "name": exp["name"],
                    "sample_index": sample_index,
                    "mse": metrics["mse"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "rel_meas_err": metrics["rel_meas_error"],
                    "score": metrics["psnr"] + float(config.get("score_ssim_weight", 10.0)) * metrics["ssim"],
                    "status": "ok",
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_delta(values_a: np.ndarray, values_b: np.ndarray, n: int, rng: np.random.Generator) -> dict:
    diff = values_b - values_a
    if diff.size == 0:
        return {"mean_delta": "missing", "ci_low": "missing", "ci_high": "missing", "p_gt_0": "missing"}
    boot = np.empty(n, dtype=np.float64)
    for idx in range(n):
        sample = rng.choice(diff, size=diff.size, replace=True)
        boot[idx] = float(sample.mean())
    return {
        "mean_delta": float(diff.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_gt_0": float((boot > 0.0).mean()),
    }


def paired_summary(rows: list[dict], experiments: list[dict], n_boot: int, seed: int) -> list[dict]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_name: dict[str, list[dict]] = {}
    for row in ok_rows:
        by_name.setdefault(row["name"], []).append(row)
    rng = np.random.default_rng(seed)
    summary = []
    for i, exp_a in enumerate(experiments):
        for exp_b in experiments[i + 1 :]:
            name_a = exp_a["name"]
            name_b = exp_b["name"]
            rows_a = {int(row["sample_index"]): row for row in by_name.get(name_a, [])}
            rows_b = {int(row["sample_index"]): row for row in by_name.get(name_b, [])}
            common = sorted(set(rows_a).intersection(rows_b))
            if not common:
                summary.append({"method_a": name_a, "method_b": name_b, "status": "missing"})
                continue
            row = {"method_a": name_a, "method_b": name_b, "n": len(common), "status": "ok"}
            for metric in ["psnr", "ssim", "score"]:
                a = np.array([float(rows_a[idx][metric]) for idx in common], dtype=np.float64)
                b = np.array([float(rows_b[idx][metric]) for idx in common], dtype=np.float64)
                stats = bootstrap_delta(a, b, n_boot, rng)
                for key, value in stats.items():
                    row[f"{metric}_{key}"] = value
            summary.append(row)
    return summary


def write_markdown(rows: list[dict], path: Path) -> None:
    cols = [
        "method_a",
        "method_b",
        "n",
        "score_mean_delta",
        "score_ci_low",
        "score_ci_high",
        "score_p_gt_0",
        "status",
    ]
    lines = ["# Paired Bootstrap Summary", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        def fmt(value):
            if value in ("", None):
                return "missing"
            try:
                return f"{float(value):.6f}"
            except Exception:
                return str(value)

        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_delta(rows: list[dict], metric: str, path: Path) -> None:
    ok = [row for row in rows if row.get("status") == "ok" and f"{metric}_mean_delta" in row]
    if not ok:
        return
    labels = [f"{row['method_b']}\nvs {row['method_a']}" for row in ok]
    means = np.array([float(row[f"{metric}_mean_delta"]) for row in ok])
    lows = np.array([float(row[f"{metric}_ci_low"]) for row in ok])
    highs = np.array([float(row[f"{metric}_ci_high"]) for row in ok])
    fig, ax = plt.subplots(figsize=(max(7, len(ok) * 1.1), 4.5))
    x = np.arange(len(ok))
    yerr = np.vstack([means - lows, highs - means])
    ax.bar(x, means, yerr=yerr, capsize=3)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel(f"Delta {metric.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    experiments = read_experiments(args.experiments_json)
    if not experiments:
        raise RuntimeError("experiments_json is empty.")
    first_config = apply_experiment_defaults(load_config(experiments[0]["config"]))
    device = resolve_device(args.device or first_config["device"])
    loader = get_val_dataloader(
        dataset_root=first_config["dataset_root"],
        img_size=int(first_config["img_size"]),
        batch_size=1,
        num_workers=int(first_config.get("num_workers", 0)),
        limit_val_samples=int(first_config.get("limit_val_samples", 500)),
        seed=int(args.seed),
        pin_memory=device.type == "cuda",
    )
    all_rows = []
    for exp in experiments:
        all_rows.extend(evaluate_experiment(exp, loader, device, int(args.seed)))
    write_csv(all_rows, output_dir / "per_sample_metrics.csv")
    summary = paired_summary(all_rows, experiments, int(args.num_bootstrap), int(args.seed))
    write_csv(summary, output_dir / "paired_summary.csv")
    write_markdown(summary, output_dir / "paired_summary.md")
    plot_delta(summary, "psnr", output_dir / "paired_delta_psnr.png")
    plot_delta(summary, "ssim", output_dir / "paired_delta_ssim.png")
    plot_delta(summary, "score", output_dir / "paired_delta_score.png")
    save_json({"experiments": experiments, "num_bootstrap": args.num_bootstrap, "seed": args.seed}, output_dir / "paired_eval_meta.json")
    print(f"Wrote paired evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
