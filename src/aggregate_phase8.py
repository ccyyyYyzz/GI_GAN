from __future__ import annotations

import csv
import json
from pathlib import Path

from .utils import ensure_dir, load_config


OUTPUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase8")

EXPERIMENTS = [
    ("Phase2 Fixed 5%", Path("E:/ns_mc_gan_gi/outputs_clean_phase2/quick_5pct")),
    ("Phase4 Best 5%", Path("E:/ns_mc_gan_gi/outputs_phase4/matched_binary_no_freeze_5pct")),
    ("Phase7 Continuous G-only 5%", Path("E:/ns_mc_gan_gi/outputs_phase7/continuous_g_only_5pct")),
    ("Phase7 Continuous Physical 5%", Path("E:/ns_mc_gan_gi/outputs_phase7/continuous_physical_5pct")),
    ("Phase8 Fixed Wide 5%", Path("E:/ns_mc_gan_gi/outputs_phase8/fixed_wide_5pct")),
    ("Phase8 Fixed Wide Refiner 5%", Path("E:/ns_mc_gan_gi/outputs_phase8/fixed_wide_refiner_5pct")),
    ("Phase8 Continuous Physical Wide 5%", Path("E:/ns_mc_gan_gi/outputs_phase8/continuous_physical_wide_5pct")),
    ("Phase8 Continuous G-only Wide 5%", Path("E:/ns_mc_gan_gi/outputs_phase8/continuous_g_only_wide_5pct")),
    ("Phase8 Direct Y Fixed 5%", Path("E:/ns_mc_gan_gi/outputs_phase8/direct_y_fixed_5pct")),
    ("Phase8 MNIST Fixed 5%", Path("E:/ns_mc_gan_gi/outputs_phase8/mnist_fixed_5pct")),
    ("Phase8 MNIST Continuous 5%", Path("E:/ns_mc_gan_gi/outputs_phase8/mnist_continuous_5pct")),
]

FIELDS = [
    "method",
    "dataset_name",
    "sampling_ratio",
    "model_type",
    "pattern_type",
    "use_learned_patterns",
    "effective_A_mode",
    "model_psnr",
    "model_ssim",
    "model_mse",
    "model_rel_meas_err",
    "backproj_psnr",
    "backproj_ssim",
    "backproj_mse",
    "backproj_rel_meas_err",
    "score",
    "delta_vs_phase7_continuous_physical_psnr",
    "delta_vs_phase7_continuous_physical_ssim",
    "delta_vs_fixed_phase2_psnr",
    "delta_vs_fixed_phase2_ssim",
    "checkpoint",
    "sample_image",
    "per_sample_metrics",
    "status",
]


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _best_checkpoint(root: Path) -> str:
    for name in ["best_score.pt", "best_hq.pt", "best_ssim.pt", "best_psnr.pt", "last.pt"]:
        path = root / name
        if path.exists():
            return str(path)
    return "missing"


def _sample(root: Path) -> str:
    for path in [root / "eval_samples" / "recon_grid.png", root / "samples" / "epoch_000.png"]:
        if path.exists():
            return str(path)
    return "missing"


def _row(method: str, root: Path) -> dict:
    metrics = _read_json(root / "eval_metrics.json")
    try:
        cfg = load_config(root / "resolved_config.yaml") if (root / "resolved_config.yaml").exists() else {}
    except Exception:
        cfg = {}
    if metrics is None:
        # resolved_config is YAML in this project, so keep inferred defaults simple.
        return {field: "missing" for field in FIELDS} | {"method": method, "status": "missing"}
    model = metrics.get("model", {})
    back = metrics.get("backprojection", {})
    row = {
        "method": method,
        "dataset_name": cfg.get("dataset_name", "stl10"),
        "sampling_ratio": cfg.get("sampling_ratio", "missing"),
        "model_type": cfg.get("model_type", "missing"),
        "pattern_type": cfg.get("pattern_type", "missing"),
        "use_learned_patterns": cfg.get("use_learned_patterns", "missing"),
        "effective_A_mode": cfg.get("effective_A_mode", "missing"),
        "model_psnr": model.get("psnr", "missing"),
        "model_ssim": model.get("ssim", "missing"),
        "model_mse": model.get("mse", "missing"),
        "model_rel_meas_err": model.get("rel_meas_error", "missing"),
        "backproj_psnr": back.get("psnr", "missing"),
        "backproj_ssim": back.get("ssim", "missing"),
        "backproj_mse": back.get("mse", "missing"),
        "backproj_rel_meas_err": back.get("rel_meas_error", "missing"),
        "score": "missing",
        "checkpoint": _best_checkpoint(root),
        "sample_image": _sample(root),
        "per_sample_metrics": metrics.get("per_sample_metrics", str(root / "eval_samples_individual" / "per_sample_metrics.csv") if (root / "eval_samples_individual" / "per_sample_metrics.csv").exists() else "missing"),
        "status": "ok",
    }
    try:
        row["score"] = float(row["model_psnr"]) + 10.0 * float(row["model_ssim"])
    except Exception:
        pass
    return row


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def collect() -> list[dict]:
    rows = [_row(method, root) for method, root in EXPERIMENTS]
    phase7 = next((r for r in rows if r["method"] == "Phase7 Continuous Physical 5%" and r["status"] == "ok"), None)
    phase2 = next((r for r in rows if r["method"] == "Phase2 Fixed 5%" and r["status"] == "ok"), None)
    for row in rows:
        for key in [
            "delta_vs_phase7_continuous_physical_psnr",
            "delta_vs_phase7_continuous_physical_ssim",
            "delta_vs_fixed_phase2_psnr",
            "delta_vs_fixed_phase2_ssim",
        ]:
            row[key] = "missing"
        if row["status"] != "ok":
            continue
        if phase7:
            psnr, base = _to_float(row["model_psnr"]), _to_float(phase7["model_psnr"])
            ssim, base_s = _to_float(row["model_ssim"]), _to_float(phase7["model_ssim"])
            row["delta_vs_phase7_continuous_physical_psnr"] = psnr - base if psnr is not None and base is not None else "missing"
            row["delta_vs_phase7_continuous_physical_ssim"] = ssim - base_s if ssim is not None and base_s is not None else "missing"
        if phase2:
            psnr, base = _to_float(row["model_psnr"]), _to_float(phase2["model_psnr"])
            ssim, base_s = _to_float(row["model_ssim"]), _to_float(phase2["model_ssim"])
            row["delta_vs_fixed_phase2_psnr"] = psnr - base if psnr is not None and base is not None else "missing"
            row["delta_vs_fixed_phase2_ssim"] = ssim - base_s if ssim is not None and base_s is not None else "missing"
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict], path: Path) -> None:
    headers = ["method", "model_psnr", "model_ssim", "score", "model_rel_meas_err", "status"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(h, "missing")) for h in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_bar(rows: list[dict], field: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    ok = [(r["method"], _to_float(r[field])) for r in rows if _to_float(r[field]) is not None]
    if not ok:
        return
    labels, values = zip(*ok)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(field)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    rows = collect()
    write_csv(rows, OUTPUT_DIR / "phase8_results.csv")
    write_md(rows, OUTPUT_DIR / "phase8_results.md")
    plot_bar(rows, "model_psnr", OUTPUT_DIR / "phase8_quality_bar_psnr.png")
    plot_bar(rows, "model_ssim", OUTPUT_DIR / "phase8_quality_bar_ssim.png")
    plot_bar(rows, "score", OUTPUT_DIR / "phase8_quality_bar_score.png")
    plot_bar(rows, "model_rel_meas_err", OUTPUT_DIR / "phase8_quality_bar_relmeaserr.png")
    print(f"Wrote Phase 8 aggregate to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
