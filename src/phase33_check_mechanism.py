from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase33_mechanism_overhaul"
PROJECT = OUT / "latex_project_mechanism_v33"
FIG_DIR = OUT / "figures"
REPORT = OUT / "MECHANISM_FIGURE_CHECK_REPORT.md"

RESULT_STRINGS = ["22.316", "22.271", "24.781", "24.730", "27.692", "25.019"]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def pdf_text(pdf: Path) -> str:
    txt = pdf.with_suffix(".txt")
    if pdf.exists():
        subprocess.run(["pdftotext", str(pdf), str(txt)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return read(txt)


def public_source() -> str:
    parts = [read(PROJECT / "main.tex"), read(PROJECT / "supplement.tex")]
    for folder in ["sections", "supplement", "tables"]:
        for path in sorted((PROJECT / folder).glob("*.tex")):
            parts.append(read(path))
    return "\n".join(parts)


def main_source_only() -> str:
    parts = [read(PROJECT / "main.tex")]
    for path in sorted((PROJECT / "sections").glob("*.tex")):
        parts.append(read(path))
    for path in sorted((PROJECT / "tables").glob("table1*.tex")) + sorted((PROJECT / "tables").glob("table2*.tex")) + sorted((PROJECT / "tables").glob("table3*.tex")):
        parts.append(read(path))
    return "\n".join(parts)


def exists(path: Path, failures: list[str], label: str) -> bool:
    if not path.exists():
        failures.append(f"Missing {label}: {path}")
        return False
    return True


def no_match(text: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
    return re.search(pattern, text, flags=flags) is None


def strip_embedded_svg_images(text: str) -> str:
    return re.sub(r'xlink:href="data:image/[^"]+"', 'xlink:href="[embedded-image]"', text, flags=re.DOTALL)


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []

    main_pdf = OUT / "main_mechanism_v33.pdf"
    supp_pdf = OUT / "supplement_mechanism_v33.pdf"
    exists(main_pdf, failures, "main PDF")
    exists(supp_pdf, failures, "supplement PDF")

    expected_sets = {
        "fig1_variant_A_problem_solution": "Figure 1 variant A",
        "fig1_variant_B_geometry_pipeline": "Figure 1 variant B",
        "fig1_variant_C_equation_decomposition": "Figure 1 variant C",
        "fig1_mechanism_final_v33": "final Figure 1",
        "figS_mechanism_equations": "supplement mechanism equations",
        "fig4_measurement_attribution_v33": "Figure 4",
    }
    for stem, label in expected_sets.items():
        for ext in ("pdf", "png", "svg"):
            exists(FIG_DIR / f"{stem}.{ext}", failures, f"{label}.{ext}")
    exists(FIG_DIR / "fig1_variants_comparison.pdf", failures, "Figure 1 variants comparison PDF")
    exists(OUT / "FIGURE1_STORYBOARD.md", failures, "Figure 1 storyboard")
    exists(OUT / "FIGURE1_VARIANT_DECISION.md", failures, "Figure 1 decision note")

    source = public_source()
    main_source = main_source_only()
    compiled = pdf_text(main_pdf) + "\n" + pdf_text(supp_pdf)
    public_text = source + "\n" + compiled
    fig1_svg = strip_embedded_svg_images(read(FIG_DIR / "fig1_mechanism_final_v33.svg"))
    fig4_svg = strip_embedded_svg_images(read(FIG_DIR / "fig4_measurement_attribution_v33.svg"))
    storyboard = read(OUT / "FIGURE1_STORYBOARD.md")
    decision = read(OUT / "FIGURE1_VARIANT_DECISION.md")

    q1 = (
        "Few bucket" in fig1_svg
        and "measurements" in fig1_svg
        and "Unconstrained" in fig1_svg
        and "neural" in fig1_svg
        and "reconstruction can drift" in fig1_svg
        and "Measured part + missing" in fig1_svg
        and "part + audit" in fig1_svg
    )
    q2 = no_match(fig1_svg, r"experimental setup|laser|lens|DMD|CCD|optical path|hardware setup")
    q3 = "residual" in fig1_svg and "filter" in fig1_svg and "final audit" in fig1_svg and r"$P_N$" in fig1_svg and r"$\Pi_y$" in fig1_svg
    q4 = "Many images can" in fig1_svg and "not measurement-audited" in fig1_svg and "measurement-consistent reconstruction" in fig1_svg
    q5 = all((FIG_DIR / f"{stem}.pdf").exists() for stem in ["fig1_variant_A_problem_solution", "fig1_variant_B_geometry_pipeline", "fig1_variant_C_equation_decomposition"])
    q6 = "Variant A is selected" in decision
    q7 = (FIG_DIR / "figS_mechanism_equations.pdf").exists() and "Algebraic form" in read(FIG_DIR / "figS_mechanism_equations.svg")
    q8 = "Neural gain" in fig4_svg and "PSN R" not in fig4_svg and "PSNR" in fig4_svg
    q9 = r"\lambda\operatorname{TV}(x)" in source and r"\lambda TV(x)" not in source
    q10 = all(value in source for value in RESULT_STRINGS)
    q11 = no_match(public_text, r"PCA oracle|architecture pilot|sampling scaling|network replacement")
    q12 = no_match(public_text, r"[A-Za-z]:[\\/]|Colab|\bPhase\s+\d+|Phase internal")
    q13 = no_match(public_text, r"strict SOTA|state-of-the-art ranking|GAN main mechanism|GAN-based method|binary learned illumination.*successful|low-frequency Hadamard.*5\\%.*high-quality|high-quality.*low-frequency Hadamard.*5\\%")

    checks = [
        ("Figure 1 explains problem -> risk -> solution", q1),
        ("Figure 1 does not imply hardware experiment", q2),
        ("Figure 1 distinguishes P_N residual filter and Pi_y final audit", q3),
        ("Figure 1 is understandable to non-specialist readers", q4),
        ("Three Figure 1 variants were generated", q5),
        ("Best variant was selected", q6),
        ("Supplement formula figure was generated", q7),
        ("Figure 4 y-axis was fixed", q8),
        ("CS-TV formula uses operatorname TV", q9),
        ("Main result numbers were unchanged", q10),
        ("No PCA / architecture / sampling scaling content", q11),
        ("No Windows path / Colab / internal phase words", q12),
        ("No strict SOTA / GAN main mechanism / lowfreq 5pct HQ claim", q13),
    ]
    for label, ok in checks:
        if not ok:
            failures.append(label)

    if "fig7_gi_csgi_ours_visual_comparison" in main_source:
        failures.append("GI/CSGI/Ours visual comparison is still included in the main text.")
    if "Visual comparison with GI/BP and CS-TV(PGD) is provided for reference" not in source:
        failures.append("Supplement GI/CSGI visual comparison caption was not updated.")
    if "Figure 1 summarizes the mechanism" not in main_source:
        failures.append("Method text does not explain Figure 1 mechanism.")
    if "Conceptual mechanism of the proposed computational reconstruction" not in main_source:
        failures.append("Final Figure 1 caption was not installed.")
    if "This baseline represents a classical compressed-sensing prior" not in main_source:
        failures.append("CS-TV main text paragraph was not updated.")

    log_text = read(PROJECT / "main.log") + "\n" + read(PROJECT / "supplement.log")
    if re.search(r"undefined references|Citation `|Rerun to get cross-references right|LaTeX Warning: Reference", log_text):
        failures.append("LaTeX log contains unresolved references/citations or rerun warnings.")
    if "Overfull" in log_text:
        warnings.append("LaTeX log contains overfull box warnings; visual inspection is recommended.")

    if "The bucket measurements" not in storyboard or "桶测量" not in storyboard:
        failures.append("Storyboard does not include both English and Chinese explanations.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Mechanism Figure Check Report",
        "",
        f"Status: {status}",
        "",
        "## Required Checks",
        "",
    ]
    for i, (label, ok) in enumerate(checks, 1):
        lines.append(f"{i}. {label}: {'yes' if ok else 'no'}")
    lines += [
        "",
        "## Failures",
    ]
    lines.extend([f"- {item}" for item in failures] or ["- None."])
    lines += ["", "## Warnings"]
    lines.extend([f"- {item}" for item in warnings] or ["- None."])
    lines += [
        "",
        "## Output Paths",
        f"- Storyboard: {OUT / 'FIGURE1_STORYBOARD.md'}",
        f"- Variant A: {FIG_DIR / 'fig1_variant_A_problem_solution.pdf'}",
        f"- Variant B: {FIG_DIR / 'fig1_variant_B_geometry_pipeline.pdf'}",
        f"- Variant C: {FIG_DIR / 'fig1_variant_C_equation_decomposition.pdf'}",
        f"- Final Figure 1: {FIG_DIR / 'fig1_mechanism_final_v33.pdf'}",
        f"- Supplement mechanism equation figure: {FIG_DIR / 'figS_mechanism_equations.pdf'}",
        f"- Figure 4: {FIG_DIR / 'fig4_measurement_attribution_v33.pdf'}",
        f"- Main PDF: {main_pdf}",
        f"- Supplement PDF: {supp_pdf}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
