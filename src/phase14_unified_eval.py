from __future__ import annotations

import argparse

from .phase14_common import PHASE14, PHASE14_IMPORTS, ensure_dir, row_from_output, write_csv, write_md_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Record unified Phase 14 eval status without starting training.")
    parser.add_argument("--output_dir", default=str(PHASE14 / "unified_eval"))
    args = parser.parse_args()
    out = ensure_dir(PHASE14 / "unified_eval")
    rows = []
    for exp in PHASE14_IMPORTS:
        row = row_from_output(exp)
        row["unified_eval_mode"] = "read_existing_colab_eval_metrics"
        rows.append(row)
    fields = ["method_id", "status", "unified_eval_mode", "psnr", "ssim", "eval_metrics_path", "best_checkpoint_path"]
    write_csv(out / "phase14_unified_eval_status.csv", rows, fields)
    write_md_table(out / "phase14_unified_eval_status.md", rows, fields)
    print(f"Wrote {out / 'phase14_unified_eval_status.md'}")


if __name__ == "__main__":
    main()
