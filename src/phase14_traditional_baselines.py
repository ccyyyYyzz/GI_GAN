from __future__ import annotations

from .phase14_common import PHASE14, PHASE14_IMPORTS, ensure_dir, row_from_output, write_md_table


def main() -> None:
    out = ensure_dir(PHASE14 / "traditional_baselines")
    rows = []
    for exp in PHASE14_IMPORTS:
        row = row_from_output(exp)
        rows.append(
            {
                "method_id": row["method_id"],
                "baseline": "configured_backprojection",
                "backproj_psnr": row.get("backproj_psnr", ""),
                "backproj_ssim": row.get("backproj_ssim", ""),
                "model_psnr": row.get("psnr", ""),
                "model_ssim": row.get("ssim", ""),
                "delta_psnr": row.get("delta_psnr", ""),
                "delta_ssim": row.get("delta_ssim", ""),
                "status": row.get("status", ""),
            }
        )
    write_md_table(out / "phase14_traditional_baselines.md", rows, list(rows[0].keys()) if rows else [])
    print(f"Wrote {out / 'phase14_traditional_baselines.md'}")


if __name__ == "__main__":
    main()
