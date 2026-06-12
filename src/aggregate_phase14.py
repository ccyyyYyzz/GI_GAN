from __future__ import annotations

from pathlib import Path

from .phase14_common import (
    PHASE14,
    PHASE14_IMPORTS,
    ensure_dir,
    load_phase12_rows,
    row_from_output,
    write_csv,
    write_json,
    write_md_table,
)


FIELDS = [
    "method_id",
    "display_name",
    "source",
    "dataset",
    "sampling_ratio",
    "measurement_family",
    "noise_std",
    "epochs",
    "gpu",
    "psnr",
    "ssim",
    "backproj_psnr",
    "backproj_ssim",
    "delta_psnr",
    "delta_ssim",
    "threshold_type",
    "threshold_reached",
    "best_checkpoint_path",
    "checkpoint_exists",
    "checkpoint_sha256",
    "eval_metrics_path",
    "sample_image_path",
    "status",
    "notes",
]


def main() -> None:
    out = ensure_dir(PHASE14)
    rows = []
    rows.extend(load_phase12_rows())
    rows.extend(row_from_output(exp) for exp in PHASE14_IMPORTS)

    phase14_rows = rows[-len(PHASE14_IMPORTS) :]
    write_csv(out / "phase14_final_results.csv", rows, FIELDS)
    write_md_table(out / "phase14_final_results.md", rows, FIELDS)
    write_md_table(
        out / "phase14_5pct_colab_status.md",
        phase14_rows,
        ["method_id", "status", "psnr", "ssim", "threshold_reached", "best_checkpoint_path", "checkpoint_sha256"],
    )
    write_json(out / "phase14_final_results.json", rows)

    reached = [r for r in phase14_rows if str(r.get("threshold_reached")).lower() == "true"]
    missing = [r for r in phase14_rows if r.get("status") != "completed"]
    lines = [
        "# Phase 14 Threshold Status",
        "",
        f"- Phase 14 Colab imports completed: {len(phase14_rows) - len(missing)}/{len(phase14_rows)}",
        f"- STL-10 5% HQ threshold reached: {len(reached)}/{len(phase14_rows)}",
        "- Threshold used here: PSNR >= 20.0 and SSIM >= 0.60 for STL-10 5%.",
        "",
        "## Rows",
    ]
    for row in phase14_rows:
        lines.append(
            f"- {row['method_id']}: status={row['status']}, PSNR={row.get('psnr', '')}, "
            f"SSIM={row.get('ssim', '')}, threshold={row.get('threshold_reached', '')}"
        )
    (out / "phase14_threshold_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'phase14_final_results.md'}")


if __name__ == "__main__":
    main()
