from __future__ import annotations

from .phase14_common import PHASE14, ensure_dir, read_csv


def main() -> None:
    out = ensure_dir(PHASE14)
    rows = read_csv(out / "phase14_final_results.csv")
    phase14_rows = [r for r in rows if r.get("method_id") in {"stl10_rademacher5_colab_full", "stl10_scrambled5_colab_full"}]
    complete = [r for r in phase14_rows if r.get("status") == "completed"]
    reached = [r for r in phase14_rows if r.get("threshold_reached") == "True"]
    lines = [
        "# Phase 14 Final Missing Experiments Report",
        "",
        "## Local execution policy",
        "",
        "- Local large training was not started.",
        "- New STL-10 5% Rademacher and scrambled Hadamard runs are Colab-first.",
        "- Local machine is used for packaging, import, aggregation, and reporting.",
        "",
        "## 5% Colab import status",
        "",
        f"- Imported/completed rows: {len(complete)}/{len(phase14_rows)}",
        f"- Rows meeting STL-10 5% HQ threshold: {len(reached)}/{len(phase14_rows)}",
        "",
    ]
    if not phase14_rows:
        lines.append("- Phase 14 aggregation has not been run yet.")
    for row in phase14_rows:
        lines.append(
            f"- {row['method_id']}: status={row.get('status')}, PSNR={row.get('psnr')}, "
            f"SSIM={row.get('ssim')}, checkpoint={row.get('best_checkpoint_path')}"
        )
    lines.extend(
        [
            "",
            "## Next manuscript gate",
            "",
            "- If both 5% Colab rows are completed and reach threshold, the manuscript can treat STL-10 5% as supported.",
            "- If either row is missing or below threshold, keep the 5% STL-10 claim as exploratory and use 10% STL-10 plus MNIST/Fashion 5% as the stronger core evidence.",
        ]
    )
    report = out / "PHASE14_FINAL_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
