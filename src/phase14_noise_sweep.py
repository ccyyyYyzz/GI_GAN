from __future__ import annotations

from .phase14_common import PHASE14, PHASE14_IMPORTS, ensure_dir, row_from_output, write_md_table


def main() -> None:
    out = ensure_dir(PHASE14 / "noise_sweep")
    rows = []
    for exp in PHASE14_IMPORTS:
        row = row_from_output(exp)
        rows.append(
            {
                "method_id": row["method_id"],
                "noise_std": row.get("noise_std", ""),
                "status": row.get("status", ""),
                "psnr": row.get("psnr", ""),
                "ssim": row.get("ssim", ""),
                "note": "Recorded completed noise=0.01 run; wider sweep requires explicit Colab jobs.",
            }
        )
    write_md_table(out / "phase14_noise001_status.md", rows, list(rows[0].keys()) if rows else [])
    print(f"Wrote {out / 'phase14_noise001_status.md'}")


if __name__ == "__main__":
    main()
