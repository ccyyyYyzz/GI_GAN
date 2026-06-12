from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .phase20_common import OUT, write_text


LATEX = OUT / "latex_project"
FIG_DIR = OUT / "figures"

FIGURES = [
    "fig1_mechanism",
    "fig2_main_results",
    "fig3_qualitative_reconstruction",
    "fig4_measurement_attribution",
    "fig5_inference_ablation",
    "fig6_robustness_baselines",
]

SUPP_FIGURES = ["figS_ablation_relmeaserr"]

LABELS = [
    "fig:mechanism",
    "fig:main_results",
    "fig:qualitative_reconstruction",
    "fig:measurement_attribution",
    "fig:inference_ablation",
    "fig:robustness_baselines",
    "tab:primary_results",
    "tab:measurement_attribution",
    "tab:ablation_summary",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def submission_text() -> str:
    parts = [read(LATEX / "main.tex")]
    parts.extend(read(path) for path in sorted((LATEX / "sections").glob("*.tex")))
    parts.extend(read(path) for path in sorted((LATEX / "supplement").glob("*.tex")))
    parts.extend(read(path) for path in sorted((LATEX / "tables").glob("*.tex")))
    return "\n".join(parts)


def main_only_text() -> str:
    parts = [read(LATEX / "main.tex")]
    parts.extend(read(path) for path in sorted((LATEX / "sections").glob("*.tex")))
    return "\n".join(parts)


def cref(text: str, label: str) -> bool:
    return bool(re.search(r"\\[cC]ref\{[^}]*" + re.escape(label) + r"[^}]*\}", text))


def section_order_ok() -> tuple[bool, str]:
    main = read(LATEX / "main.tex")
    order = [
        r"\input{sections/introduction.tex}",
        r"\input{sections/related_work.tex}",
        r"\input{sections/problem_formulation.tex}",
        r"\input{sections/method.tex}",
        r"\input{sections/measurement_families.tex}",
        r"\input{sections/experimental_protocol.tex}",
        r"\input{sections/results.tex}",
        r"\input{sections/validation_ablation.tex}",
        r"\input{sections/discussion.tex}",
        r"\input{sections/limitations.tex}",
        r"\input{sections/conclusion.tex}",
    ]
    positions = [main.find(term) for term in order]
    return all(p >= 0 for p in positions) and positions == sorted(positions), str(positions)


def status(ok: bool, label: str, detail: str = "") -> str:
    return f"- {'PASS' if ok else 'FAIL'}: {label}" + (f" - {detail}" if detail else "")


def checks() -> list[tuple[bool, str, str]]:
    text = submission_text()
    main_text = main_only_text()
    lower = text.lower()
    main_lower = main_text.lower()
    rows: list[tuple[bool, str, str]] = []

    missing_figs = []
    for stem in FIGURES:
        for ext in [".pdf", ".png", ".svg"]:
            if not (FIG_DIR / f"{stem}{ext}").exists():
                missing_figs.append(f"{stem}{ext}")
    for stem in SUPP_FIGURES:
        for ext in [".pdf", ".png"]:
            if not (FIG_DIR / f"{stem}{ext}").exists():
                missing_figs.append(f"{stem}{ext}")
    rows.append((not missing_figs, "All required Phase20 figures exist", ", ".join(missing_figs[:8])))

    q = FIG_DIR / "fig3_qualitative_reconstruction.png"
    if q.exists():
        im = Image.open(q)
        rows.append((im.width >= 1800 and im.height >= 2600, "Qualitative reconstruction figure is large enough", f"{im.width}x{im.height}px"))
    else:
        rows.append((False, "Qualitative reconstruction figure is large enough", "missing"))

    missing_labels = [label for label in LABELS if not cref(main_text, label)]
    rows.append((not missing_labels, "All main figures/tables are cited with cref", ", ".join(missing_labels)))

    ordered, positions = section_order_ok()
    rows.append((ordered, "Section order follows v6 manuscript structure", positions))
    rows.append(("Reference Placeholders" not in text, "Reference Placeholders section removed", ""))
    rows.append(("E:/" not in text and "E:\\" not in text and "C:\\" not in text, "No Windows paths in main/supplement text", ""))
    rows.append((not re.search(r"Phase1[5-9]|Phase20", text), "No internal phase names in main/supplement text", ""))
    rows.append(("rac12" not in lower and "鈥" not in text and "蔚" not in text, "No copied mojibake formulas in LaTeX text", ""))
    rows.append((r"\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\,\mathrm{TV}(x)" in text, "CS-TV formula is correct", ""))
    rows.append(("tv-regularized compressed-sensing baseline solved by projected gradient descent" in lower or "tv-regularized compressed-sensing baseline solved by pgd" in lower, "CS-TV explained as compressed-sensing PGD baseline", ""))
    rows.append(("sota" not in lower and "state-of-the-art" not in lower, "No SOTA claim", ""))
    rows.append(("gan main" not in lower and "gan is" not in lower and "adversarial generation is the final" not in lower, "No GAN main-mechanism claim", ""))
    rows.append(("binary learned illumination is not claimed as successful" in lower, "No binary learned illumination success claim", ""))
    rows.append(("low-frequency hadamard at 5\\% is not the primary" in main_lower and "low-frequency hadamard at 5\\% is not a high-quality stl-10 setting" in main_lower, "No lowfreq Hadamard 5% HQ claim", ""))
    rows.append(((LATEX / "main.pdf").exists(), "Compiled latex_project/main.pdf exists", str(LATEX / "main.pdf")))
    rows.append(((OUT / "main_v6.pdf").exists(), "Copied main_v6.pdf exists", str(OUT / "main_v6.pdf")))
    rows.append(((OUT / "citations_to_verify.md").exists(), "citations_to_verify.md exists", ""))
    return rows


def report() -> str:
    rows = checks()
    ok = all(row[0] for row in rows)
    lines = [
        "# Manuscript V6 Check Report",
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
            "## Figure Placement",
            "- Main text: Figure 1 mechanism, Figure 2 main results, Figure 3 qualitative reconstruction, Figure 4 measurement attribution, Figure 5 inference ablation, Figure 6 robustness/baselines.",
            "- Supplement: figS ablation RelMeasErr and curated summary tables.",
            "",
            "## Manual Follow-up",
            "- Replace TODO BibTeX entries with verified references.",
            "- Consider journal-specific formatting and final figure color polish.",
            "- Hardware validation and broader external baselines remain future work, not claims in this PDF.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    path = OUT / "MANUSCRIPT_V6_CHECK_REPORT.md"
    write_text(path, report())
    print({"report": str(path)})


if __name__ == "__main__":
    main()
