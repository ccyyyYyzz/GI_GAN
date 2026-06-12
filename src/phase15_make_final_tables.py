from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .phase15_common import (
    PHASE14,
    PHASE15,
    ensure_dir,
    load_registry,
    numeric,
    read_csv,
    threshold_for,
    write_csv,
    write_json,
    write_md_table,
    write_tex_table,
)


OUT_DIR = PHASE15 / "paper_tables_final"
RESCUE_AUDIT = PHASE14 / "noleak_rescue_audit" / "NOLEAK_RESCUE_AUDIT.csv"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt(value: Any) -> Any:
    x = numeric(value)
    if x != x:
        return value
    return round(x, 4)


def get_field(row: dict[str, Any], name: str) -> Any:
    if name in row:
        return row[name]
    for key, value in row.items():
        normalized = key.replace("\ufeff", "").replace('"', "").strip()
        if normalized == name:
            return value
    return ""


def save_table(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_csv(OUT_DIR / f"{name}.csv", rows, fields)
    write_md_table(OUT_DIR / f"{name}.md", rows, fields)
    write_tex_table(OUT_DIR / f"{name}.tex", rows, fields)


def exact_status_by_id() -> dict[str, str]:
    rows = read_csv(PHASE15 / "exactA_reeval" / "exactA_reeval_results.csv")
    return {row.get("method_id", ""): row.get("status", "") for row in rows}


def rescue_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(RESCUE_AUDIT):
        method_id = get_field(row, "method_id")
        if not method_id:
            continue
        rows.append(
            {
                "method_id": method_id,
                "dataset": "Fashion-MNIST" if "fashion" in method_id else "STL-10",
                "sampling_ratio": 0.10 if "10" in method_id else 0.05,
                "measurement_family": "lowfreq_hadamard",
                "psnr": fmt(row.get("last_eval_psnr")),
                "ssim": fmt(row.get("last_eval_ssim")),
                "source": row.get("source", ""),
                "paper_status": "auxiliary only",
                "notes": row.get("leak_status", ""),
            }
        )
    return rows


def registry_to_final_row(row: dict[str, Any], exact_status: dict[str, str]) -> dict[str, Any]:
    ratio = numeric(row.get("sampling_ratio"))
    threshold_type, threshold_psnr, threshold_ssim = threshold_for(row.get("dataset", ""), ratio)
    psnr = numeric(row.get("psnr"))
    ssim = numeric(row.get("ssim"))
    reached = psnr >= threshold_psnr and ssim >= threshold_ssim
    reeval_status = exact_status.get(row.get("method_id", ""), "not_required")
    if reeval_status == "mismatch":
        paper_status = "conditional: local exact-A mismatch"
    elif reached:
        paper_status = "main strict no-leak"
    else:
        paper_status = "strict no-leak but below threshold"
    return {
        "method_id": row.get("method_id", ""),
        "dataset": row.get("dataset", ""),
        "sampling_ratio": ratio,
        "measurement_family": row.get("measurement_family", ""),
        "pattern_type": row.get("pattern_type", ""),
        "noise_std": fmt(row.get("noise_std")),
        "model_psnr": fmt(row.get("psnr")),
        "model_ssim": fmt(row.get("ssim")),
        "model_mse": fmt(row.get("mse")),
        "backproj_psnr": fmt(row.get("backproj_psnr")),
        "backproj_ssim": fmt(row.get("backproj_ssim")),
        "delta_psnr": fmt(row.get("delta_psnr")),
        "delta_ssim": fmt(row.get("delta_ssim")),
        "threshold_type": threshold_type,
        "threshold_psnr": threshold_psnr,
        "threshold_ssim": threshold_ssim,
        "threshold_reached": reached,
        "strict_noleak": truthy(row.get("strict_noleak")),
        "exact_A_available": truthy(row.get("exact_A_available")),
        "exactA_reeval_status": reeval_status,
        "paper_status": paper_status,
    }


def main() -> None:
    ensure_dir(OUT_DIR)
    registry = load_registry()
    exact_status = exact_status_by_id()
    final_rows = [registry_to_final_row(row, exact_status) for row in registry]
    rescue = rescue_rows()

    main_fields = [
        "method_id",
        "dataset",
        "sampling_ratio",
        "measurement_family",
        "model_psnr",
        "model_ssim",
        "backproj_psnr",
        "backproj_ssim",
        "delta_psnr",
        "delta_ssim",
        "threshold_reached",
        "strict_noleak",
        "exact_A_available",
        "exactA_reeval_status",
        "paper_status",
    ]
    save_table("table_main_strict_noleak_results", final_rows, main_fields)

    stl5 = [row for row in final_rows if row["dataset"] == "STL-10" and abs(row["sampling_ratio"] - 0.05) < 1e-9]
    stl5 += [row for row in rescue if row["dataset"] == "STL-10" and abs(row["sampling_ratio"] - 0.05) < 1e-9]
    save_table("table_stl10_5pct_final", stl5, main_fields[:10] + ["threshold_reached", "paper_status", "notes"])

    stl10 = [row for row in final_rows if row["dataset"] == "STL-10" and abs(row["sampling_ratio"] - 0.10) < 1e-9]
    stl10 += [row for row in rescue if row["dataset"] == "STL-10" and abs(row["sampling_ratio"] - 0.10) < 1e-9]
    save_table("table_stl10_10pct_final", stl10, main_fields[:10] + ["threshold_reached", "paper_status", "notes"])

    simple = [row for row in final_rows if row["dataset"] in {"MNIST", "Fashion-MNIST"}]
    save_table("table_simple_domains_final", simple, main_fields)

    attribution_fields = [
        "method_id",
        "dataset",
        "sampling_ratio",
        "measurement_family",
        "pattern_type",
        "model_psnr",
        "model_ssim",
        "delta_psnr",
        "strict_noleak",
        "exact_A_available",
        "paper_status",
    ]
    save_table("table_measurement_attribution_final", final_rows, attribution_fields)

    integrity = []
    for row in registry:
        integrity.append(
            {
                "method_id": row.get("method_id", ""),
                "checkpoint_sha256": row.get("checkpoint_sha256", ""),
                "sha_verified": row.get("sha_verified", ""),
                "exact_A_available": row.get("exact_A_available", ""),
                "exact_A_path": row.get("exact_A_path", ""),
                "eval_metrics_path": row.get("eval_metrics_path", ""),
                "paper_status": "main strict no-leak" if truthy(row.get("preferred_for_paper")) else "excluded",
            }
        )
    integrity_fields = [
        "method_id",
        "checkpoint_sha256",
        "sha_verified",
        "exact_A_available",
        "exact_A_path",
        "eval_metrics_path",
        "paper_status",
    ]
    save_table("table_checkpoint_integrity_final", integrity, integrity_fields)

    audit = read_csv(PHASE15 / "noleak_audit" / "noleak_audit.csv")
    if audit:
        save_table("table_noleak_audit_final", audit, list(audit[0].keys()))

    excluded = []
    for row in rescue:
        excluded.append(
            {
                "method_id": row["method_id"],
                "source": row["source"],
                "psnr": row["psnr"],
                "ssim": row["ssim"],
                "paper_status": row["paper_status"],
                "reason": row["notes"],
            }
        )
    excluded_fields = ["method_id", "source", "psnr", "ssim", "paper_status", "reason"]
    save_table("table_auxiliary_and_excluded_results", excluded, excluded_fields)

    manifest = {
        "output_dir": str(OUT_DIR),
        "strict_rows": len(final_rows),
        "auxiliary_rows": len(rescue),
        "tables": sorted(p.name for p in OUT_DIR.glob("*.csv")),
    }
    write_json(OUT_DIR / "paper_tables_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
