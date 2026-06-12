from __future__ import annotations

from .phase14_common import PHASE14, PHASE14_IMPORTS, ensure_dir, row_from_output, write_md_table


def main() -> None:
    out = ensure_dir(PHASE14 / "checkpoint_ablation")
    rows = []
    for exp in PHASE14_IMPORTS:
        row = row_from_output(exp)
        rows.append(
            {
                "method_id": row["method_id"],
                "checkpoint_mode": "best_hq_or_best_score",
                "status": row["status"],
                "psnr": row.get("psnr", ""),
                "ssim": row.get("ssim", ""),
                "checkpoint": row.get("best_checkpoint_path", ""),
                "note": "No local retraining or checkpoint surgery was performed.",
            }
        )
    write_md_table(out / "phase14_checkpoint_ablation.md", rows, list(rows[0].keys()) if rows else [])
    print(f"Wrote {out / 'phase14_checkpoint_ablation.md'}")


if __name__ == "__main__":
    main()
