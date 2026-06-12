from __future__ import annotations

import json
from pathlib import Path

from .phase11_common import CONFIG10, CONFIG11, ROOT10, ROOT11, ensure_dir, read_csv_rows, safe_copy, write_json


ASSET_ROOT = ROOT11 / "paper_assets"


TABLES = [
    ROOT10 / "phase10_results.csv",
    ROOT11 / "phase11_summary.csv",
    ROOT11 / "multiseed_summary.csv",
    ROOT11 / "attribution_results.csv",
    ROOT11 / "noise_sweep_summary.csv",
]

FIGURES = [
    ROOT10 / "phase10_psnr.png",
    ROOT10 / "phase10_ssim.png",
    ROOT10 / "phase10_backproj_vs_model.png",
    ROOT11 / "phase11_psnr.png",
    ROOT11 / "phase11_ssim.png",
    ROOT11 / "phase11_hq_score.png",
    ROOT11 / "attribution_delta_psnr.png",
    ROOT11 / "attribution_delta_ssim.png",
]

CONFIGS = [
    CONFIG10 / "hadamard10_full_noise001.yaml",
    CONFIG10 / "hadamard5_medium_noise001.yaml",
    CONFIG10 / "rademacher10_full_noise001.yaml",
    CONFIG10 / "scrambled_hadamard10_full_noise001.yaml",
    CONFIG11 / "hadamard5_push_hq.yaml",
    CONFIG11 / "hadamard10_seed43.yaml",
    CONFIG11 / "hadamard10_seed44.yaml",
]


def best_row(rows: list[dict]) -> dict | None:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("hq_score") not in {"", None}]
    return max(completed, key=lambda row: float(row["hq_score"]), default=None)


def claims(rows: list[dict]) -> list[str]:
    best = best_row(rows)
    stl10_10 = any(row.get("status") == "completed" and str(row.get("reaches_stl10_10pct_hq")).lower() == "true" for row in rows)
    stl10_5 = any(row.get("status") == "completed" and str(row.get("reaches_stl10_5pct_hq")).lower() == "true" for row in rows)
    simple = any(row.get("status") == "completed" and str(row.get("reaches_simple_domain_hq")).lower() == "true" for row in rows)
    supported = [
        "DC row is essential for low-frequency Hadamard reconstruction, based on Phase 9 calibration.",
        "Low-frequency Hadamard backprojection is far stronger than random Rademacher at the same 10% ratio in Phase 9 references.",
    ]
    if stl10_10:
        supported.append("STL-10 10% lowfreq Hadamard reaches the internal HQ threshold under completed long training.")
    if simple:
        supported.append("Structured simple targets reach the internal 5% HQ threshold.")
    partial = []
    if best:
        partial.append(f"Best current method is {best.get('method')} with PSNR={best.get('model_psnr')} and SSIM={best.get('model_ssim')}.")
    partial.append("HQ network refinement should be described in proportion to attribution deltas.")
    unsupported = []
    if not stl10_5:
        unsupported.append("STL-10 5% high-quality reconstruction is not supported until a completed 5% row reaches threshold.")
    unsupported.append("Learned binary illumination improves HQ reconstruction is unsupported by the fixed Hadamard Phase 10/11 evidence.")
    unsupported.append("Network alone is the main source of high quality is unsupported when attribution is backprojection dominated.")
    lines = ["# Paper Claims Draft", "", "## Supported Claims", *[f"- {x}" for x in supported], "", "## Partially Supported Claims", *[f"- {x}" for x in partial], "", "## Unsupported Claims", *[f"- {x}" for x in unsupported], ""]
    return lines


def main() -> None:
    tables_dir = ensure_dir(ASSET_ROOT / "tables")
    figures_dir = ensure_dir(ASSET_ROOT / "figures")
    configs_dir = ensure_dir(ASSET_ROOT / "configs")
    rows = read_csv_rows(ROOT11 / "phase11_summary.csv")
    copied = []
    for path in TABLES:
        copied.append({"path": str(path), "copied": safe_copy(path, tables_dir / path.name)})
    for path in FIGURES:
        copied.append({"path": str(path), "copied": safe_copy(path, figures_dir / path.name)})
    best = best_row(rows)
    if best and best.get("sample_image"):
        copied.append({"path": best["sample_image"], "copied": safe_copy(best["sample_image"], figures_dir / "best_method_recon_grid.png")})
    for path in CONFIGS:
        if path.exists():
            copied.append({"path": str(path), "copied": safe_copy(path, configs_dir / path.name)})
    manifest = []
    for row in rows:
        if row.get("checkpoint") or row.get("status") == "completed":
            manifest.append(
                {
                    "method": row.get("method"),
                    "checkpoint_path": row.get("checkpoint"),
                    "config_path": str((CONFIG11 if row.get("phase") == "phase11" else CONFIG10) / f"{row.get('method')}.yaml"),
                    "metrics_path": str(Path(row.get("checkpoint", "")).parent / "eval_metrics.json") if row.get("checkpoint") else "",
                    "sample_image_path": row.get("sample_image"),
                    "status": row.get("status"),
                }
            )
    write_json({"checkpoints": manifest, "copied": copied}, ASSET_ROOT / "checkpoints_manifest.json")
    (ASSET_ROOT / "PAPER_CLAIMS_DRAFT.md").write_text("\n".join(claims(rows)), encoding="utf-8")
    print(f"Phase 11 paper assets exported to: {ASSET_ROOT}")


if __name__ == "__main__":
    main()
