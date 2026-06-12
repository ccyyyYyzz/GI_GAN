from __future__ import annotations

import json
from pathlib import Path

from .utils import ensure_dir, save_json


OLD_METRICS = Path("E:/ns_mc_gan_gi/outputs/quick_5pct/eval_metrics.json")
CLEAN_METRICS = Path("E:/ns_mc_gan_gi/outputs_clean_phase2/quick_5pct/eval_metrics.json")
OUTPUT_ROOT = Path("E:/ns_mc_gan_gi/outputs_clean_phase2")


FIELDS = [
    "old_env_model_psnr",
    "clean_env_model_psnr",
    "old_env_model_ssim",
    "clean_env_model_ssim",
    "old_env_model_mse",
    "clean_env_model_mse",
    "old_env_backproj_psnr",
    "clean_env_backproj_psnr",
    "clean_minus_old_psnr",
    "clean_minus_old_ssim",
    "note",
]


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(metrics: dict | None, section: str, key: str):
    if not metrics:
        return "missing"
    return metrics.get(section, {}).get(key, "missing")


def delta(clean_value, old_value):
    if clean_value == "missing" or old_value == "missing":
        return "missing"
    return float(clean_value) - float(old_value)


def fmt(value) -> str:
    if value == "missing" or value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(row: dict, path: Path) -> None:
    lines = [
        "# Clean 5% vs Old ABI 5%",
        "",
        f"- old metrics: {OLD_METRICS if OLD_METRICS.exists() else 'missing'}",
        f"- clean metrics: {CLEAN_METRICS if CLEAN_METRICS.exists() else 'missing'}",
        "",
        "|" + "|".join(FIELDS) + "|",
        "|" + "|".join(["---"] * len(FIELDS)) + "|",
        "|" + "|".join(fmt(row.get(field)) for field in FIELDS) + "|",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    old = read_json(OLD_METRICS)
    clean = read_json(CLEAN_METRICS)
    old_psnr = get_metric(old, "model", "psnr")
    clean_psnr = get_metric(clean, "model", "psnr")
    old_ssim = get_metric(old, "model", "ssim")
    clean_ssim = get_metric(clean, "model", "ssim")
    note = []
    if old is None:
        note.append("old_metrics_missing")
    if clean is None:
        note.append("clean_metrics_missing")
    if not note:
        note.append("ok")

    row = {
        "old_env_model_psnr": old_psnr,
        "clean_env_model_psnr": clean_psnr,
        "old_env_model_ssim": old_ssim,
        "clean_env_model_ssim": clean_ssim,
        "old_env_model_mse": get_metric(old, "model", "mse"),
        "clean_env_model_mse": get_metric(clean, "model", "mse"),
        "old_env_backproj_psnr": get_metric(old, "backprojection", "psnr"),
        "clean_env_backproj_psnr": get_metric(clean, "backprojection", "psnr"),
        "clean_minus_old_psnr": delta(clean_psnr, old_psnr),
        "clean_minus_old_ssim": delta(clean_ssim, old_ssim),
        "note": ",".join(note),
    }
    json_path = save_json(row, OUTPUT_ROOT / "compare_old_vs_clean_5pct.json")
    md_path = OUTPUT_ROOT / "compare_old_vs_clean_5pct.md"
    write_markdown(row, md_path)
    print(f"Wrote comparison JSON to: {json_path}")
    print(f"Wrote comparison Markdown to: {md_path}")


if __name__ == "__main__":
    main()
