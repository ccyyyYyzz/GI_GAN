from __future__ import annotations

import csv
import json
from pathlib import Path

from .utils import ensure_dir, load_config


OUTPUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase8_hq")

EXPERIMENTS = [
    ("hadamard_hq_10pct", OUTPUT_DIR / "hadamard_hq_10pct"),
    ("hadamard_hq_5pct", OUTPUT_DIR / "hadamard_hq_5pct"),
    ("scrambled_hadamard_hq_10pct", OUTPUT_DIR / "scrambled_hadamard_hq_10pct"),
    ("rademacher_hq_10pct", OUTPUT_DIR / "rademacher_hq_10pct"),
    ("rademacher_hq_5pct", OUTPUT_DIR / "rademacher_hq_5pct"),
    ("continuous_physical_hq_10pct", OUTPUT_DIR / "continuous_physical_hq_10pct"),
    ("continuous_physical_hq_5pct", OUTPUT_DIR / "continuous_physical_hq_5pct"),
    ("mnist_hq_5pct", OUTPUT_DIR / "mnist_hq_5pct"),
    ("fashion_mnist_hq_5pct", OUTPUT_DIR / "fashion_mnist_hq_5pct"),
    ("cifar10_gray_hq_10pct", OUTPUT_DIR / "cifar10_gray_hq_10pct"),
]

FIELDS = [
    "method",
    "dataset_name",
    "sampling_ratio",
    "pattern_type",
    "backprojection_mode",
    "model_type",
    "use_learned_patterns",
    "model_psnr",
    "model_ssim",
    "model_mse",
    "model_rel_meas_err",
    "backproj_psnr",
    "backproj_ssim",
    "score",
    "hq_score",
    "reaches_stl10_10pct_hq_threshold",
    "reaches_stl10_5pct_hq_threshold",
    "reaches_simple_domain_hq_threshold",
    "checkpoint",
    "sample_image",
    "status",
]


def _read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_cfg(root: Path) -> dict:
    path = root / "resolved_config.yaml"
    if not path.exists():
        return {}
    try:
        return load_config(path)
    except Exception:
        return {}


def _float(value):
    try:
        return float(value)
    except Exception:
        return None


def _best_checkpoint(root: Path) -> str:
    for name in ["best_hq.pt", "best_score.pt", "best_ssim.pt", "best_psnr.pt", "last.pt"]:
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
    cfg = _load_cfg(root)
    if metrics is None:
        return {field: "missing" for field in FIELDS} | {"method": method, "status": "missing"}
    model = metrics.get("model", {})
    back = metrics.get("backprojection", {})
    psnr = _float(model.get("psnr"))
    ssim = _float(model.get("ssim"))
    rel = _float(model.get("rel_meas_error")) or 0.0
    score = psnr + 10.0 * ssim if psnr is not None and ssim is not None else "missing"
    hq_score = psnr + 20.0 * ssim - 20.0 * rel if psnr is not None and ssim is not None else "missing"
    dataset = cfg.get("dataset_name", "stl10")
    ratio = _float(cfg.get("sampling_ratio"))
    row = {
        "method": method,
        "dataset_name": dataset,
        "sampling_ratio": cfg.get("sampling_ratio", "missing"),
        "pattern_type": cfg.get("pattern_type", "missing"),
        "backprojection_mode": cfg.get("backprojection_mode", "missing"),
        "model_type": cfg.get("model_type", "missing"),
        "use_learned_patterns": cfg.get("use_learned_patterns", "missing"),
        "model_psnr": model.get("psnr", "missing"),
        "model_ssim": model.get("ssim", "missing"),
        "model_mse": model.get("mse", "missing"),
        "model_rel_meas_err": model.get("rel_meas_error", "missing"),
        "backproj_psnr": back.get("psnr", "missing"),
        "backproj_ssim": back.get("ssim", "missing"),
        "score": score,
        "hq_score": hq_score,
        "reaches_stl10_10pct_hq_threshold": bool(dataset == "stl10" and ratio == 0.10 and psnr is not None and ssim is not None and psnr >= 22.0 and ssim >= 0.65),
        "reaches_stl10_5pct_hq_threshold": bool(dataset == "stl10" and ratio == 0.05 and psnr is not None and ssim is not None and psnr >= 20.0 and ssim >= 0.60),
        "reaches_simple_domain_hq_threshold": bool(dataset in {"mnist", "fashion_mnist"} and psnr is not None and ssim is not None and psnr >= 25.0 and ssim >= 0.80),
        "checkpoint": _best_checkpoint(root),
        "sample_image": _sample(root),
        "status": "ok",
    }
    return row


def collect() -> list[dict]:
    return [_row(method, root) for method, root in EXPERIMENTS]


def write_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict], path: Path) -> None:
    headers = ["method", "dataset_name", "sampling_ratio", "pattern_type", "model_psnr", "model_ssim", "hq_score", "status"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(h, "missing")) for h in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_bar(rows: list[dict], field: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    pairs = [(r["method"], _float(r[field])) for r in rows if _float(r[field]) is not None]
    if not pairs:
        return
    labels, values = zip(*pairs)
    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(labels)), 4))
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
    write_csv(rows, OUTPUT_DIR / "phase8_hq_results.csv")
    write_md(rows, OUTPUT_DIR / "phase8_hq_results.md")
    plot_bar(rows, "model_psnr", OUTPUT_DIR / "phase8_hq_psnr.png")
    plot_bar(rows, "model_ssim", OUTPUT_DIR / "phase8_hq_ssim.png")
    plot_bar(rows, "score", OUTPUT_DIR / "phase8_hq_score.png")
    plot_bar(rows, "model_rel_meas_err", OUTPUT_DIR / "phase8_hq_relmeaserr.png")
    print(f"Wrote Phase 8-HQ aggregate to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
