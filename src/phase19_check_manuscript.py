from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .phase19_common import OUT, write_text


LATEX = OUT / "latex_project_v5"
FIG_DIR = OUT / "figures"

FIGURES = [
    "fig0_graphical_abstract",
    "fig1_mechanism",
    "fig2_main_metrics",
    "fig3_qualitative_grid_v2",
    "fig4_measurement_regime_map",
    "fig5_inference_ablation",
    "fig6_validation_summary",
]

LABELS = [
    "fig:mechanism",
    "fig:main_metrics",
    "fig:qualitative",
    "fig:regime_map",
    "fig:ablation",
    "fig:validation_summary",
    "tab:primary_results",
    "tab:attribution",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main_text() -> str:
    parts = [read(LATEX / "main.tex")]
    parts.extend(read(path) for path in sorted((LATEX / "sections").glob("*.tex")))
    return "\n".join(parts)


def cref(text: str, label: str) -> bool:
    return bool(re.search(r"\\[cC]ref\{[^}]*" + re.escape(label) + r"[^}]*\}", text))


def section_order_ok() -> tuple[bool, str]:
    main = read(LATEX / "main.tex")
    order = [
        r"\input{sections/forward_model.tex}",
        r"\input{sections/method.tex}",
        r"\input{sections/results.tex}",
        r"\input{sections/validation.tex}",
        r"\input{sections/discussion.tex}",
    ]
    positions = [main.find(term) for term in order]
    return all(p >= 0 for p in positions) and positions == sorted(positions), str(positions)


def status(ok: bool, label: str, detail: str = "") -> str:
    return f"- {'PASS' if ok else 'FAIL'}: {label}" + (f" - {detail}" if detail else "")


def checks() -> list[tuple[bool, str, str]]:
    text = main_text()
    lower = text.lower()
    rows: list[tuple[bool, str, str]] = []

    missing_figs = []
    for stem in FIGURES:
        for ext in [".pdf", ".png", ".svg"]:
            if not (FIG_DIR / f"{stem}{ext}").exists():
                missing_figs.append(f"{stem}{ext}")
    rows.append((not missing_figs, "All Phase19 figures exist as pdf/png/svg", ", ".join(missing_figs[:6])))

    q = FIG_DIR / "fig3_qualitative_grid_v2.png"
    if q.exists():
        im = Image.open(q)
        rows.append((im.width >= 1800 and im.height >= 2600, "Figure 3 qualitative is large enough", f"{im.width}x{im.height}px"))
    else:
        rows.append((False, "Figure 3 qualitative is large enough", "missing"))

    missing_labels = [label for label in LABELS if not cref(text, label)]
    rows.append((not missing_labels, "Main figures/tables are cited with cref", ", ".join(missing_labels)))

    ordered, positions = section_order_ok()
    rows.append((ordered, "Narrative follows problem -> method -> results -> validation", positions))

    rows.append(("contributions are" in lower and "null-space formulation" in lower and "discussion" in lower, "Innovation points are clear in Introduction and Discussion", ""))
    rows.append(("Reference Placeholders" not in text, "No Reference Placeholders section", ""))
    rows.append(("E:/" not in text and "E:\\" not in text and "C:\\" not in text, "No Windows paths in main text", ""))
    rows.append(("Phase15" not in text and "Phase16" not in text and "Phase18" not in text and "Phase19" not in text, "No internal phase names in main text", ""))
    rows.append((r"\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\,\mathrm{TV}(x)" in text, "CS-TV formula is correct", ""))
    rows.append(("tv-regularized compressed-sensing baseline solved by projected gradient descent" in lower, "CS-TV explained as compressed sensing PGD", ""))
    rows.append(("sota" not in lower and "state-of-the-art" not in lower, "No strict SOTA claim", ""))
    rows.append(("gan is" not in lower and "adversarial generation is the final mechanism" not in lower, "No GAN main-mechanism claim", ""))
    rows.append(("low-frequency hadamard 5\\% is not used as a primary" in lower, "No lowfreq Hadamard 5% HQ claim", ""))
    rows.append(((LATEX / "main.pdf").exists(), "Compiled latex_project_v5/main.pdf exists", str(LATEX / "main.pdf")))
    rows.append(((OUT / "main_v5.pdf").exists(), "Copied main_v5.pdf exists", str(OUT / "main_v5.pdf")))
    rows.append(((OUT / "citations_to_verify.md").exists(), "citations_to_verify.md exists", ""))
    return rows


def report() -> str:
    rows = checks()
    ok = all(r[0] for r in rows)
    lines = [
        "# Phase19 Logic Rewrite Report",
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
            "## Figure Placement Recommendation",
            "- Main text: Figure 1 mechanism, Figure 2 metrics, Figure 3 qualitative, Figure 4 regime map, Figure 5 ablation, Figure 6 validation summary.",
            "- Supplement: Figure 0 graphical abstract, exact-A summary, confidence intervals, and detailed CSV pointers.",
            "",
            "## Manual Follow-up",
            "- Replace TODO_VERIFY BibTeX entries with verified references.",
            "- Consider hand-polishing graphical abstract and mechanism colors for the target journal style.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    path = OUT / "PHASE19_LOGIC_REWRITE_REPORT.md"
    write_text(path, report())
    print({"report": str(path)})


if __name__ == "__main__":
    main()
