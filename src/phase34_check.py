from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase34_mechanism_teaser"
PROJECT = OUT / "latex_project_mechanism_v34"
FIG_DIR = OUT / "figures"
REPORT = OUT / "MECHANISM_TEASER_CHECK_REPORT.md"

RESULT_STRINGS = ["22.316", "22.271", "24.781", "24.730", "27.692", "25.019"]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def pdf_text(pdf: Path) -> str:
    txt = pdf.with_suffix(".txt")
    if pdf.exists():
        subprocess.run(["pdftotext", str(pdf), str(txt)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return read(txt)


def tex_source() -> str:
    parts = [read(PROJECT / "main.tex"), read(PROJECT / "supplement.tex")]
    for folder in ("sections", "supplement", "tables"):
        base = PROJECT / folder
        if base.exists():
            for path in sorted(base.glob("*.tex")):
                parts.append(read(path))
    return "\n".join(parts)


def main_tex_source() -> str:
    parts = [read(PROJECT / "main.tex")]
    base = PROJECT / "sections"
    if base.exists():
        for path in sorted(base.glob("*.tex")):
            parts.append(read(path))
    return "\n".join(parts)


def strip_svg_images(text: str) -> str:
    return re.sub(r'xlink:href="data:image/[^"]+"', 'xlink:href="[embedded-image]"', text, flags=re.DOTALL)


def exists(path: Path, failures: list[str], label: str) -> bool:
    if path.exists():
        return True
    failures.append(f"Missing {label}: {path}")
    return False


def absent(text: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
    return re.search(pattern, text, flags=flags) is None


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []

    expected = {
        OUT / "FIGURE1_REDESIGN_RATIONALE.md": "Figure 1 redesign rationale",
        FIG_DIR / "fig1_mechanism_teaser_v34.pdf": "Figure 1 PDF",
        FIG_DIR / "fig1_mechanism_teaser_v34.png": "Figure 1 PNG",
        FIG_DIR / "fig1_mechanism_teaser_v34.svg": "Figure 1 SVG",
        FIG_DIR / "figS1_equation_decomposition_v34.pdf": "Figure S1 PDF",
        FIG_DIR / "figS1_equation_decomposition_v34.png": "Figure S1 PNG",
        FIG_DIR / "figS1_equation_decomposition_v34.svg": "Figure S1 SVG",
        FIG_DIR / "fig4_measurement_attribution_v34.pdf": "Figure 4 PDF",
        FIG_DIR / "fig4_measurement_attribution_v34.png": "Figure 4 PNG",
        FIG_DIR / "fig4_measurement_attribution_v34.svg": "Figure 4 SVG",
        OUT / "main_v34.pdf": "main PDF",
        OUT / "supplement_v34.pdf": "supplement PDF",
    }
    for path, label in expected.items():
        exists(path, failures, label)

    source = tex_source()
    main_source = main_tex_source()
    compiled_text = pdf_text(OUT / "main_v34.pdf") + "\n" + pdf_text(OUT / "supplement_v34.pdf")
    public_text = source + "\n" + compiled_text
    fig1_svg = strip_svg_images(read(FIG_DIR / "fig1_mechanism_teaser_v34.svg"))
    figs1_svg = strip_svg_images(read(FIG_DIR / "figS1_equation_decomposition_v34.svg"))
    fig4_svg = strip_svg_images(read(FIG_DIR / "fig4_measurement_attribution_v34.svg"))
    rationale = read(OUT / "FIGURE1_REDESIGN_RATIONALE.md")

    checks = [
        (
            "Figure 1 rationale explains why old figure was insufficient",
            all(
                item in rationale
                for item in [
                    "only listed",
                    "visual failure mode",
                    "real-image or image-like anchor",
                    "roles of",
                    "too many formulas",
                ]
            ),
        ),
        (
            "Figure 1 is a 2x3 problem-failure-solution teaser",
            all(
                item in fig1_svg
                for item in [
                    "Low-sampling",
                    "bucket measurement",
                    "Physical inverse:",
                    "faithful but incomplete",
                    "Unconstrained neural",
                    "inverse can drift",
                    "Neural residual is filtered",
                    "Completed image is audited",
                    "Output",
                ]
            ),
        ),
        (
            "Figure 1 states computational forward model and reconstruction mechanism",
            "computational forward model and reconstruction mechanism" in fig1_svg,
        ),
        (
            "Figure 1 shows low sampling ambiguity and physical inverse incompleteness",
            "many possible images" in fig1_svg and "measurement-tied" in fig1_svg and "incomplete" in fig1_svg,
        ),
        (
            "Figure 1 shows unconstrained drift warning",
            "weak audit" in fig1_svg and "not" in fig1_svg and "free" in fig1_svg,
        ),
        (
            "Figure 1 distinguishes P_N residual filter and Pi_y final audit",
            "P_N" in fig1_svg and "residual filter" in fig1_svg and "Pi_y final audit" in fig1_svg,
        ),
        (
            "Figure 1 includes conceptual output comparison",
            "GT" in fig1_svg and "BP" in fig1_svg and "Ours" in fig1_svg,
        ),
        (
            "Figure 1 does not imply hardware experiment",
            absent(fig1_svg, r"hardware|laser|lens|DMD|CCD|camera|optical setup|optical path"),
        ),
        (
            "Figure S1 contains equation decomposition",
            all(item in figs1_svg for item in [r"x_{\rm data}", r"G_\theta", "P_N", r"\Pi_y"]),
        ),
        (
            "Figure 4 y-axis label is fixed",
            "\u0394PSNR (dB)" in fig4_svg and "PSN R" not in fig4_svg and "Neural gain PSN" not in fig4_svg,
        ),
        (
            "Figure 4 excludes MNIST/Fashion and keeps lowfreq diagnostic",
            "MNIST" not in fig4_svg and "Fashion" not in fig4_svg and "Lowfreq-5" in fig4_svg and "Lowfreq-10" in fig4_svg,
        ),
        (
            "CS-TV formula uses operatorname TV",
            r"\lambda\operatorname{TV}(x)" in source and r"\lambda TV" not in source,
        ),
        (
            "Main result numbers are unchanged",
            all(value in source for value in RESULT_STRINGS),
        ),
        (
            "No PCA / architecture / sampling scaling content",
            absent(public_text, r"PCA oracle|architecture pilot|sampling scaling|network replacement"),
        ),
        (
            "No Windows path / Colab / internal phase words in paper text",
            absent(public_text, r"[A-Za-z]:[\\/]|Colab|\bPhase\s+\d+\b|internal phase"),
        ),
        (
            "No strict SOTA / GAN main mechanism / binary learned illumination / lowfreq 5pct HQ claim",
            absent(
                public_text,
                r"strict SOTA|state-of-the-art ranking|GAN main mechanism|binary learned illumination|low-frequency Hadamard.*5\\%.*high-quality|high-quality.*low-frequency Hadamard.*5\\%",
            ),
        ),
        (
            "Main text uses new mechanism paragraph and caption",
            "problem--failure--solution sequence" in main_source
            and "Conceptual mechanism." in main_source
            and "measurement-constrained completion module" in main_source,
        ),
        (
            "Main text references new Figure 1 and Figure 4 files",
            "fig1_mechanism_teaser_v34.pdf" in main_source and "fig4_measurement_attribution_v34.pdf" in main_source,
        ),
        (
            "Supplement references Figure S1 equation file",
            "figS1_equation_decomposition_v34.pdf" in source,
        ),
    ]

    for label, ok in checks:
        if not ok:
            failures.append(label)

    if "fig7_gi_csgi_ours_visual_comparison" in main_source:
        failures.append("GI/CSGI/Ours visual comparison is still in the main text.")
    if "TODO" in public_text or "Reference Placeholder" in public_text:
        failures.append("TODO or reference placeholder remains in paper text.")

    log_text = read(PROJECT / "main.log") + "\n" + read(PROJECT / "supplement.log")
    if re.search(r"undefined references|Citation `|Rerun to get cross-references right|LaTeX Warning: Reference", log_text):
        failures.append("LaTeX log contains unresolved references/citations or rerun warnings.")
    if "Overfull" in log_text:
        warnings.append("LaTeX log contains overfull box warnings; visual inspection is recommended.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Mechanism Teaser Check Report",
        "",
        f"Status: {status}",
        "",
        "## Required Checks",
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
        f"- Rationale: {OUT / 'FIGURE1_REDESIGN_RATIONALE.md'}",
        f"- Figure 1: {FIG_DIR / 'fig1_mechanism_teaser_v34.pdf'}",
        f"- Figure S1: {FIG_DIR / 'figS1_equation_decomposition_v34.pdf'}",
        f"- Figure 4: {FIG_DIR / 'fig4_measurement_attribution_v34.pdf'}",
        f"- Main PDF: {OUT / 'main_v34.pdf'}",
        f"- Supplement PDF: {OUT / 'supplement_v34.pdf'}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
