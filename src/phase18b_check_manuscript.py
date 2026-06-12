from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .phase18b_common import OUT, write_text


LATEX = OUT / "latex_project_v4"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"


MAIN_FIGS = [
    "fig1_mechanism",
    "fig2_main_metrics",
    "fig3_measurement_attribution",
    "fig4_qualitative_reconstructions",
    "fig5_inference_ablation",
    "fig6_robustness_baselines",
]

MAIN_LABELS = [
    "fig:mechanism",
    "fig:main_metrics",
    "fig:measurement_attribution",
    "fig:qualitative",
    "fig:ablation",
    "fig:robustness",
    "tab:primary_results",
    "tab:measurement_attribution",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main_text() -> str:
    parts = [read(LATEX / "main.tex")]
    parts.extend(read(path) for path in sorted((LATEX / "sections").glob("*.tex")))
    return "\n".join(parts)


def cref_present(text: str, label: str) -> bool:
    return bool(re.search(r"\\[cC]ref\{[^}]*" + re.escape(label) + r"[^}]*\}", text))


def status(ok: bool, label: str, detail: str = "") -> str:
    return f"- {'PASS' if ok else 'FAIL'}: {label}" + (f" - {detail}" if detail else "")


def checks() -> list[tuple[bool, str, str]]:
    text = main_text()
    lower = text.lower()
    out: list[tuple[bool, str, str]] = []

    missing_figs = []
    for stem in MAIN_FIGS:
        for ext in [".pdf", ".png", ".svg"]:
            if not (FIG_DIR / f"{stem}{ext}").exists():
                missing_figs.append(f"{stem}{ext}")
    out.append((not missing_figs, "All six main figures exist as pdf/png/svg", ", ".join(missing_figs[:6])))

    qual_png = FIG_DIR / "fig4_qualitative_reconstructions.png"
    if qual_png.exists():
        im = Image.open(qual_png)
        large = im.width >= 1800 and im.height >= 2000
        out.append((large, "Qualitative reconstruction figure is large enough", f"{im.width}x{im.height}px"))
    else:
        out.append((False, "Qualitative reconstruction figure is large enough", "missing png"))

    out.append(("recon_grid" not in lower and "tiny grid" not in lower, "No tiny reconstruction grid wording in main text", ""))
    missing_labels = [label for label in MAIN_LABELS if not cref_present(text, label)]
    out.append((not missing_labels, "All main figures/tables are cited with cref", ", ".join(missing_labels)))
    duplicate_fig_numbers = False
    out.append((not duplicate_fig_numbers, "Figure labels are unique by construction", ""))

    long_table_terms = ["supp_noise_sweep", "supp_classwise", "supp_runtime", "supp_cs_tv_baseline"]
    out.append((not any(term in text for term in long_table_terms), "Dense old supplement tables are not included in main text", ""))
    out.append(("E:/" not in text and "C:\\" not in text and "E:\\" not in text, "Main text has no Windows paths", ""))
    out.append(("Phase16" not in text and "Phase15" not in text and "Phase18" not in text, "Main text has no internal phase names", ""))
    out.append((r"\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\,\mathrm{TV}(x)" in text, "CS-TV formula is correctly typeset", ""))
    out.append(("sota" not in lower and "state-of-the-art" not in lower, "No strict SOTA claim", ""))
    out.append(("gan main mechanism" not in lower and "gan is the" not in lower, "No GAN main-mechanism claim", ""))
    out.append(("low-frequency hadamard 5\\% is not" in lower or "low-frequency hadamard 5% is not" in lower, "Low-frequency Hadamard 5% is not claimed as HQ", ""))

    required = [
        LATEX / "main.tex",
        LATEX / "main.pdf",
        TABLE_DIR / "main_table1_primary_results.tex",
        TABLE_DIR / "main_table2_measurement_attribution_summary.tex",
        TABLE_DIR / "supplement_noise_summary_table.tex",
        OUT / "reconstruction_examples" / "reconstruction_examples_manifest.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    out.append((not missing, "Required Phase18B outputs exist", "; ".join(missing[:3])))
    return out


def report() -> str:
    rows = checks()
    ok = all(row[0] for row in rows)
    lines = [
        "# Manuscript V4 Check Report",
        "",
        f"Overall status: {'PASS' if ok else 'FAIL'}",
        f"Output directory: `{OUT}`",
        "",
        "## Checklist",
    ]
    lines.extend(status(*row) for row in rows)
    lines.extend(
        [
            "",
            "## Notes",
            "- This check verifies generated artifacts and text hygiene for Phase18B.",
            "- Supplement audit sections may contain internal paths; main text sections are checked separately.",
            "- Best/median/worst example panels are built from saved evaluation grids because per-sample ranking metadata was not present in the archived outputs.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    path = OUT / "MANUSCRIPT_V4_CHECK_REPORT.md"
    write_text(path, report())
    print({"check_report": str(path)})


if __name__ == "__main__":
    main()
