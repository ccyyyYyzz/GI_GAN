from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path("E:/ns_mc_gan_gi")
DEFAULT_DRIVE_ROOT = Path(os.environ.get("NS_MC_GAN_GI_ROOT", "/content/drive/MyDrive/ns_mc_gan_gi"))
OUT_NAME = "outputs_phase26"

CURRENT_MAIN_RESULTS = {
    "rademacher5_hq_noise001_colab": {
        "family": "rademacher",
        "sampling_ratio": 0.05,
        "current_model_psnr": 22.315998762433527,
        "current_model_ssim": 0.6345529191150999,
    },
    "scrambled_hadamard5_hq_noise001_colab": {
        "family": "scrambled_hadamard",
        "sampling_ratio": 0.05,
        "current_model_psnr": 22.270757736570168,
        "current_model_ssim": 0.6316637063606377,
    },
    "rademacher10_full_noise001_colab": {
        "family": "rademacher",
        "sampling_ratio": 0.10,
        "current_model_psnr": 24.78116836352519,
        "current_model_ssim": 0.7472215678423729,
    },
    "scrambled_hadamard10_full_noise001_colab": {
        "family": "scrambled_hadamard",
        "sampling_ratio": 0.10,
        "current_model_psnr": 24.730141547548826,
        "current_model_ssim": 0.7456643261872284,
    },
}

PILOT_CONFIGS = [
    "current_hq_rad5_pilot",
    "nafnet_small_rad5_pilot",
    "unrolled_ista_rad5_pilot",
    "current_hq_scr5_pilot",
    "nafnet_small_scr5_pilot",
    "unrolled_ista_scr5_pilot",
]


def drive_root(value: str | None = None) -> Path:
    return Path(value) if value else DEFAULT_DRIVE_ROOT


def output_root(root: str | Path | None = None) -> Path:
    base = Path(root) if root else DEFAULT_DRIVE_ROOT
    return base / OUT_NAME


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    value = safe_float(value)
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    return ""


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(field, "")).replace("|", "/") for field in fields) + "|")
    return "\n".join(lines)


def main_results_from_drive(root: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(root / "outputs_phase16" / "supplementary_experiments" / "attribution" / "attribution_final.csv")
    results = {key: dict(value) for key, value in CURRENT_MAIN_RESULTS.items()}
    for row in rows:
        method_id = row.get("method_id", "")
        if method_id not in results:
            continue
        results[method_id].update(
            {
                "family": row.get("measurement_family", results[method_id]["family"]),
                "sampling_ratio": safe_float(row.get("sampling_ratio")),
                "current_model_psnr": safe_float(row.get("model_psnr")),
                "current_model_ssim": safe_float(row.get("model_ssim")),
            }
        )
    return results


def best_by_metric(rows: list[dict[str, Any]], metric: str, group_key: str | None = None) -> list[dict[str, Any]]:
    if group_key is None:
        valid = [row for row in rows if math.isfinite(safe_float(row.get(metric)))]
        return sorted(valid, key=lambda row: safe_float(row.get(metric)), reverse=True)[:1]
    winners = []
    groups = sorted({row.get(group_key, "") for row in rows})
    for group in groups:
        valid = [
            row
            for row in rows
            if row.get(group_key, "") == group and math.isfinite(safe_float(row.get(metric)))
        ]
        if valid:
            winners.append(sorted(valid, key=lambda row: safe_float(row.get(metric)), reverse=True)[0])
    return winners
