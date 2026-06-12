from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase38_professional_figure"
PROJECT = OUT / "latex_project_v38"
FIG_DIR = OUT / "figures"
EDIT_PACK = OUT / "figure1_edit_pack"
REPORT = OUT / "PHASE38_CHECK_REPORT.md"
INFO = OUT / "INKSCAPE_INFO.json"

FIG_SVG = FIG_DIR / "fig1_professional_mechanism_v38.svg"
FIG_SOURCE_SVG = FIG_DIR / "fig1_professional_mechanism_v38_source.svg"
FIG_PDF = FIG_DIR / "fig1_professional_mechanism_v38.pdf"
FIG_PNG = FIG_DIR / "fig1_professional_mechanism_v38_600dpi.png"
FIG_TIKZ = FIG_DIR / "fig1_professional_mechanism_v38.tikz.tex"
FIG4_SVG = FIG_DIR / "fig4_measurement_attribution_v36.svg"

RESULT_STRINGS = ["22.316", "22.271", "24.781", "24.730", "27.692", "25.019"]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def pdf_text(pdf: Path) -> str:
    txt = pdf.with_suffix(".txt")
    if pdf.exists():
        subprocess.run(["pdftotext", str(pdf), str(txt)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return read(txt)


def tex_files() -> list[Path]:
    paths = [PROJECT / "main.tex", PROJECT / "supplement.tex"]
    for folder in ["sections", "supplement", "tables"]:
        base = PROJECT / folder
        if base.exists():
            paths.extend(sorted(base.glob("*.tex")))
    return [path for path in paths if path.exists()]


def all_source() -> str:
    parts = [read(path) for path in tex_files()]
    parts.append(read(PROJECT / "references.bib"))
    return "\n".join(parts)


def main_source() -> str:
    parts = [read(PROJECT / "main.tex")]
    section_dir = PROJECT / "sections"
    if section_dir.exists():
        parts.extend(read(path) for path in sorted(section_dir.glob("*.tex")))
    return "\n".join(parts)


def strip_svg_images(text: str) -> str:
    return re.sub(r'xlink:href="data:image/[^"]+"', 'xlink:href="[embedded-image]"', text, flags=re.DOTALL)


def absent(text: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
    return re.search(pattern, text, flags=flags) is None


def get_inkscape_info() -> dict[str, object]:
    if not INFO.exists():
        return {"found": False, "command_path": None, "version": None}
    try:
        return json.loads(INFO.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"found": False, "command_path": None, "version": None}


def heading_order(text: str, headings: list[str]) -> bool:
    positions = []
    for heading in headings:
        pos = text.find(heading)
        if pos < 0:
            return False
        positions.append(pos)
    return positions == sorted(positions)


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    inkscape = get_inkscape_info()

    source = all_source()
    main_tex = main_source()
    compiled_main = pdf_text(OUT / "main_v38.pdf")
    compiled_supp = pdf_text(OUT / "supplement_v38.pdf")
    public_all = source + "\n" + compiled_main + "\n" + compiled_supp
    public_main = main_tex + "\n" + compiled_main
    fig_svg = strip_svg_images(read(FIG_SVG))
    fig4_svg = strip_svg_images(read(FIG4_SVG))
    method = read(PROJECT / "sections" / "method.tex")
    results = read(PROJECT / "sections" / "results.tex")
    validation = read(PROJECT / "sections" / "validation_ablation.tex")

    expected = {
        OUT / "main_v38.pdf": "main PDF",
        OUT / "supplement_v38.pdf": "supplement PDF",
        FIG_SVG: "Figure 1 SVG",
        FIG_SOURCE_SVG: "Figure 1 source SVG",
        FIG_PDF: "Figure 1 PDF",
        FIG_PNG: "Figure 1 600dpi PNG",
        FIG_TIKZ: "Figure 1 TikZ source",
        FIG4_SVG: "Figure 4 SVG",
        EDIT_PACK / "fig1_professional_mechanism_v38.svg": "edit-pack SVG",
        EDIT_PACK / "fig1_professional_mechanism_v38.pdf": "edit-pack PDF",
        EDIT_PACK / "fig1_professional_mechanism_v38_600dpi.png": "edit-pack PNG",
        EDIT_PACK / "FIGURE1_EDITING_GUIDE.md": "editing guide",
        EDIT_PACK / "FIGURE1_STYLE_GUIDE.md": "style guide",
    }
    for path, label in expected.items():
        if not path.exists():
            failures.append(f"Missing {label}: {path}")

    checks = [
        ("Inkscape is available", bool(inkscape.get("found"))),
        (
            "Figure 1 is a new professional vector figure",
            "<svg" in fig_svg
            and f'width="{1800}"' in fig_svg
            and f'height="{950}"' in fig_svg
            and "Known-pattern" in fig_svg
            and "Conventional GI correlation" not in fig_svg,
        ),
        (
            r"Figure 1 explains conventional GI \(A^Ty\)",
            "Conventional GI / raw BP" in fig_svg and "A^T y" in fig_svg and "sum_i y_i a_i" in fig_svg,
        ),
        (
            r"Figure 1 explains \(x_data=A^Tq\) and \(q=(AA^T+\lambda I)^{-1}y\)",
            "q = (AA^T + lambda I)^-1 y" in fig_svg and "x_data = A^T q" in fig_svg,
        ),
        (
            r"Figure 1 clearly separates \(G_\theta\), \(P_N\), and \(\Pi_y\)",
            "Candidate residual" in fig_svg
            and (
                "r_theta = G_theta(x_data)" in fig_svg
                or ("r_theta = G_theta" in fig_svg and "(x_data)" in fig_svg)
            )
            and "Residual filter" in fig_svg
            and "P_N filter" in fig_svg
            and (("Pi_y audit" in fig_svg) or ("Bucket" in fig_svg and "audit" in fig_svg and "Pi_y" in fig_svg)),
        ),
        (
            "Figure 1 has no hardware-experiment implication",
            absent(fig_svg, r"laser|DMD|lens|CCD|camera|optical path|hardware"),
        ),
        ("SVG/PDF/600dpi PNG generated", FIG_SVG.exists() and FIG_PDF.exists() and FIG_PNG.exists()),
        (
            "figure1_edit_pack generated",
            EDIT_PACK.exists()
            and (EDIT_PACK / "FIGURE1_EDITING_GUIDE.md").exists()
            and (EDIT_PACK / "FIGURE1_STYLE_GUIDE.md").exists(),
        ),
        (
            r"CS-TV formula uses \operatorname{TV}(x)",
            r"\lambda\operatorname{TV}(x)" in source
            and r"\lambda TV" not in source
            and "lambda TV" not in source,
        ),
        (
            "Results order is attribution before simple-domain sanity",
            heading_order(
                results,
                [
                    r"\subsection{STL-10 reconstruction at 5\% and 10\%}",
                    r"\subsection{Qualitative reconstruction}",
                    r"\subsection{Similar final quality arises from different reconstruction regimes}",
                    r"\subsection{Simple-domain sanity checks}",
                ],
            )
            and "Similar final PSNR does not imply similar sensing behavior." in results,
        ),
        (
            "Validation headings are question-style",
            heading_order(
                validation,
                [
                    r"\subsection{Is random sensing reproducible?}",
                    r"\subsection{Is the final measurement audit necessary?}",
                    r"\subsection{Does the network depend on the bucket vector?}",
                    r"\subsection{Is the method stronger than a classical CSGI-style prior?}",
                    r"\subsection{Stability diagnostics}",
                ],
            ),
        ),
        (
            "Figure 4 y-axis is fixed",
            "Delta PSNR (dB)" in fig4_svg
            and "Neural gain PSN R" not in fig4_svg
            and "Neural gain PSNR" not in fig4_svg
            and fig4_svg.count("Lowfreq-10") <= 1
            and "MNIST" not in fig4_svg
            and "Fashion" not in fig4_svg,
        ),
        ("Main result numbers unchanged", all(value in public_all for value in RESULT_STRINGS)),
        (
            "No PCA/oracle/architecture exploration",
            absent(public_all, r"PCA oracle|architecture pilot|sampling scaling|network replacement"),
        ),
        (
            "No Windows path / Colab / Phase words in main text",
            absent(public_main, r"[A-Za-z]:[\\/]|Colab|\bPhase\s+\d+\b|internal phase"),
        ),
        (
            "No strict SOTA / first GI / first pseudoinverse / GAN main mechanism claim",
            absent(
                public_all,
                r"strict SOTA|state-of-the-art ranking|first\s+(GI|ghost|pseudoinverse|pseudo-inverse|deep)|GAN main mechanism|GAN-based method",
            ),
        ),
    ]

    for label, ok in checks:
        if not ok:
            failures.append(label)

    method_checks = [
        r"\subsection{From bucket-pattern correlation to regularized data solution}",
        r"\hat{x}_{\rm GI}=A^Ty=\sum_i y_i a_i",
        r"q=(AA^T+\lambda I)^{-1}y",
        r"x_{\rm data}=A^Tq",
        "not a new standalone reconstructor",
        "candidate residual proposal",
        "Figure 1 anchors the proposed pipeline to conventional GI.",
        "fig1_professional_mechanism_v38.pdf",
    ]
    for item in method_checks:
        if item not in method:
            failures.append(f"Method missing required item: {item}")

    if "Fig. 1 visualizes the same logic" in public_all:
        failures.append("Forbidden Figure 1 sentence remains.")
    if "hardware experiment" in public_main.lower() or "hardware optical experiment" in public_main.lower():
        failures.append("Hardware experiment claim remains in main text.")
    if "low-frequency Hadamard 5\\% high-quality" in public_all:
        failures.append("Forbidden low-frequency Hadamard 5% high-quality claim remains.")

    log_text = read(PROJECT / "main.log") + "\n" + read(PROJECT / "supplement.log")
    if re.search(r"undefined references|Citation `|Rerun to get cross-references right|LaTeX Warning: Reference", log_text):
        failures.append("LaTeX log contains unresolved references/citations or rerun warnings.")
    if "Overfull" in log_text:
        warnings.append("LaTeX log contains overfull box warnings; visual inspection is recommended.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Phase 38 Check Report",
        "",
        f"Status: {status}",
        "",
        "## Inkscape",
        f"- Found: {'yes' if inkscape.get('found') else 'no'}",
        f"- Command path: {inkscape.get('command_path') or 'not found'}",
        f"- Version: {inkscape.get('version') or 'not available'}",
        "",
        "## Required Questions",
        "",
    ]
    for i, (label, ok) in enumerate(checks, 1):
        lines.append(f"{i}. {label}: {'yes' if ok else 'no'}")
    lines += ["", "## Failures"]
    lines.extend([f"- {item}" for item in failures] or ["- None."])
    lines += ["", "## Warnings"]
    lines.extend([f"- {item}" for item in warnings] or ["- None."])
    lines += [
        "",
        "## Output Paths",
        f"- Main PDF: {OUT / 'main_v38.pdf'}",
        f"- Supplement PDF: {OUT / 'supplement_v38.pdf'}",
        f"- Figure 1 SVG: {FIG_SVG}",
        f"- Figure 1 PDF: {FIG_PDF}",
        f"- Figure 1 600dpi PNG: {FIG_PNG}",
        f"- Edit pack: {EDIT_PACK}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
