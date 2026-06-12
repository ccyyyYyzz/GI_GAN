from __future__ import annotations

from pathlib import Path

from .phase12_common import PHASE12, as_float, load_registry, read_csv, round_float, write_md_table
from .utils import ensure_dir


def main() -> None:
    ensure_dir(PHASE12)
    rows = load_registry()
    preferred = [row for row in rows if str(row.get("preferred_for_paper")).lower() == "true"]
    completed = [row for row in rows if row.get("status") == "completed"]
    best_psnr = max(preferred, key=lambda r: as_float(r.get("psnr")) or -1)
    best_ssim = max(preferred, key=lambda r: as_float(r.get("ssim")) or -1)
    registry_fields = ["method_id", "display_name", "source", "dataset", "sampling_ratio", "psnr", "ssim", "threshold_type", "threshold_reached", "status", "preferred_for_paper"]
    lines = [
        "# Phase 12 Final Report",
        "",
        "## 1. Experiment Inventory",
        f"- Registry: {PHASE12 / 'final_result_registry.csv'}",
        f"- Total registry rows: {len(rows)}",
        f"- Completed rows: {len(completed)}",
        "",
        "## 2. Local vs Colab",
        "- Local rows keep source=local or running_local.",
        "- Imported Colab rows keep source=colab_import.",
        "- Medium runs are not labeled full.",
        "",
        "## 3. Preferred Paper Results",
    ]
    tmp = PHASE12 / "_preferred_table.md"
    write_md_table(tmp, preferred, registry_fields)
    lines.extend(tmp.read_text(encoding="utf-8").splitlines())
    tmp.unlink(missing_ok=True)
    lines.extend(
        [
            "",
            "## 4. Main Tables",
            f"- STL-10 10%: {PHASE12 / 'paper_tables' / 'table_main_stl10_10pct.md'}",
            f"- STL-10 5%: {PHASE12 / 'paper_tables' / 'table_stl10_5pct.md'}",
            f"- MNIST/Fashion: {PHASE12 / 'paper_tables' / 'table_simple_domains_5pct.md'}",
            f"- Reproducibility: {PHASE12 / 'paper_tables' / 'table_reproducibility.md'}",
            f"- Claims: {PHASE12 / 'paper_tables' / 'table_claims.md'}",
            "",
            "## 5. Figures And Examples",
            f"- Paper figures: {PHASE12 / 'paper_figures'}",
            f"- Reconstruction examples: {PHASE12 / 'reconstruction_examples'}",
            "",
            "## 6. DC Row Control",
            f"- DC row table: {PHASE12 / 'dc_row_control' / 'dc_row_results.md'}",
            "",
            "## 7. Minimal Baselines",
            f"- Minimal baselines: {PHASE12 / 'minimal_baselines' / 'minimal_baselines_results.md'}",
            "",
            "## 8. Best Results",
            f"- Best preferred PSNR: {best_psnr.get('display_name')} ({round_float(best_psnr.get('psnr'))})",
            f"- Best preferred SSIM: {best_ssim.get('display_name')} ({round_float(best_ssim.get('ssim'))})",
            "",
            "## 9. Threshold Status",
            "- STL-10 10% HQ: supported.",
            "- STL-10 5% HQ: unsupported under stated threshold.",
            "- Simple-domain 5% HQ: supported.",
            "",
            "## 10. Supported Claims",
            "- STL-10 10% high-quality reconstruction.",
            "- MNIST/Fashion-MNIST 5% high-quality reconstruction.",
            "- Measurement choice matters; Rademacher and scrambled Hadamard are strongest at STL-10 10%.",
            "- DC row retention is important for low-frequency Hadamard.",
            "- Local/Colab reproducibility is supported by 5% Hadamard medium and, if complete, Fashion local/Colab.",
            "",
            "## 11. Unsupported Claims",
            "- STL-10 5% high-quality.",
            "- Binary learned illumination improvement.",
            "- Learned illumination as sole driver of quality.",
            "- Strict SOTA claims without matched protocols.",
            "",
            "## 12. Immediate Writing Plan",
            "Use the registry, tables, figures, reconstruction examples, DC control, and claims draft to write the paper/report. More training is not necessary before drafting.",
            "",
        ]
    )
    path = PHASE12 / "PHASE12_FINAL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
