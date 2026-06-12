from __future__ import annotations

import argparse
import shutil

from .phase56_common import PHASE53C_ROOT, PHASE53D_ROOT, add_args, read_csv_rows, to_float, write_command_log, write_rows
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase56 eval-only feasible hallucination hardening/selection.")
    add_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_dir)
    write_command_log(root)
    src_rows = []
    for phase, base in [
        ("Phase53D", PHASE53D_ROOT / "feasible_hallucination_metrics.csv"),
        ("Phase53C", PHASE53C_ROOT / "session_22_feasible_hallucination_figure" / "feasible_hallucination_metrics.csv"),
    ]:
        for row in read_csv_rows(base):
            cross = to_float(row.get("cross_relmeas"))
            ours = to_float(row.get("ours_relmeas"))
            ratio = cross / max(1e-12, ours)
            selected = cross <= 0.005 or ratio <= 2.0
            src_rows.append(
                {
                    "phase": phase,
                    "task": row.get("task", ""),
                    "family": row.get("family", ""),
                    "cross_relmeas": cross,
                    "ours_relmeas": ours,
                    "cross_over_ours": ratio,
                    "cross_psnr_vs_gt": row.get("cross_psnr_vs_gt", ""),
                    "selected_for_main": selected,
                    "caption_guidance": "measurement-consistent feasible alternative" if selected else "near-feasible cross solution",
                }
            )
    selected = [r for r in src_rows if str(r.get("selected_for_main")) == "True" or r.get("selected_for_main") is True]
    write_rows(root, "feasible_hallucination_hardened_metrics", src_rows, "Phase56 Feasible Hallucination Hardened Metrics")
    source_png = PHASE53D_ROOT / "feasible_hallucination_grid.png"
    source_pdf = PHASE53D_ROOT / "feasible_hallucination_grid.pdf"
    target_png = root / "feasible_hallucination_hardened_grid.png"
    target_pdf = root / "feasible_hallucination_hardened_grid.pdf"
    if source_png.exists():
        shutil.copy2(source_png, target_png)
    if source_pdf.exists():
        shutil.copy2(source_pdf, target_pdf)
    report = [
        "# Phase56 Feasible Hallucination Hardening",
        "",
        f"- Candidate rows checked: {len(src_rows)}.",
        f"- Rows satisfying cross RelMeasErr <= 2x ours or <=0.005: {len(selected)}.",
        "- If no or few rows satisfy the threshold, use caption language `near-feasible cross solution` rather than `measurement-consistent feasible alternative`.",
        "- Existing hard-projection grid was copied as the hardened candidate grid; no training was run.",
    ]
    (root / "FEASIBLE_HALLUCINATION_HARDENED_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(root / "feasible_hallucination_hardened_metrics.csv")


if __name__ == "__main__":
    main()

