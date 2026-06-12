from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase37_author_guided_rewrite"
PROJECT = OUT / "latex_project_v37"
FIG_DIR = OUT / "figures"
REPORT = OUT / "PHASE37_CHECK_REPORT.md"

RESULT_STRINGS = ["22.316", "22.271", "24.781", "24.730", "27.692", "25.019"]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def pdf_text(pdf: Path) -> str:
    txt = pdf.with_suffix(".txt")
    if pdf.exists():
        subprocess.run(["pdftotext", str(pdf), str(txt)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return read(txt)


def source_text(include_refs: bool = True) -> str:
    parts = [read(PROJECT / "main.tex"), read(PROJECT / "supplement.tex")]
    if include_refs:
        parts.append(read(PROJECT / "references.bib"))
    for folder in ("sections", "supplement", "tables"):
        base = PROJECT / folder
        if base.exists():
            for path in sorted(base.glob("*.tex")):
                parts.append(read(path))
    return "\n".join(parts)


def main_source() -> str:
    parts = [read(PROJECT / "main.tex")]
    for path in sorted((PROJECT / "sections").glob("*.tex")):
        parts.append(read(path))
    return "\n".join(parts)


def strip_svg_images(text: str) -> str:
    return re.sub(r'xlink:href="data:image/[^"]+"', 'xlink:href="[embedded-image]"', text, flags=re.DOTALL)


def paragraph_count_intro(text: str) -> int:
    body = text.replace(r"\section{Introduction}", "").strip()
    before_items = body.split(r"\begin{itemize}")[0]
    paras = [p.strip() for p in re.split(r"\n\s*\n", before_items) if p.strip()]
    return len(paras)


def absent(text: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
    return re.search(pattern, text, flags=flags) is None


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []

    expected = {
        FIG_DIR / "fig1_minimal_mechanism_v37.pdf": "Figure 1 PDF",
        FIG_DIR / "fig1_minimal_mechanism_v37.png": "Figure 1 PNG",
        FIG_DIR / "fig1_minimal_mechanism_v37.svg": "Figure 1 SVG",
        FIG_DIR / "fig4_measurement_attribution_v36.pdf": "Figure 4 PDF",
        OUT / "main_v37.pdf": "main PDF",
        OUT / "supplement_v37.pdf": "supplement PDF",
    }
    for path, label in expected.items():
        if not path.exists():
            failures.append(f"Missing {label}: {path}")

    abstract = read(PROJECT / "sections" / "abstract.tex")
    introduction = read(PROJECT / "sections" / "introduction.tex")
    method = read(PROJECT / "sections" / "method.tex")
    results = read(PROJECT / "sections" / "results.tex")
    validation = read(PROJECT / "sections" / "validation_ablation.tex")
    all_source = source_text()
    main_tex_source = main_source()
    compiled_main = pdf_text(OUT / "main_v37.pdf")
    compiled_supp = pdf_text(OUT / "supplement_v37.pdf")
    public_main = main_tex_source + "\n" + compiled_main
    public_all = all_source + "\n" + compiled_main + "\n" + compiled_supp
    fig1_svg = strip_svg_images(read(FIG_DIR / "fig1_minimal_mechanism_v37.svg"))
    fig4_svg = strip_svg_images(read(FIG_DIR / "fig4_measurement_attribution_v36.svg"))

    checks = [
        (
            "Abstract replaced",
            "No hardware optical experiment" not in abstract
            and "regularized GI/SPI data solution" in abstract
            and "Exact-operator audit" in abstract
            and all(value in abstract for value in RESULT_STRINGS),
        ),
        (
            "Introduction is 6-paragraph logic",
            paragraph_count_intro(introduction) == 6
            and "Conventional GI can be viewed as bucket-pattern correlation" in introduction
            and "The main contributions are" in introduction,
        ),
        (
            r"Method starts from \(A^Ty\), then \(x_data\), then \(G_\theta/P_N/\Pi_y\)",
            r"\hat{x}_{\rm GI}=A^Ty=\sum_i y_i a_i" in method
            and r"x_{\rm data}=A^Tq" in method
            and r"r_\theta=G_\theta(x_{\rm data},z)" in method
            and r"P_N(v)" in method
            and r"\Pi_y(v)" in method,
        ),
        (
            "Figure 1 is minimal, readable, and conventional-GI anchored",
            "Conventional GI correlation" in fig1_svg
            and "Regularized data solution" in fig1_svg
            and "Measurement-audited" in fig1_svg
            and "neural completion" in fig1_svg
            and "raw bucket weights" in fig1_svg
            and "decorrelated bucket weights" in fig1_svg
            and "candidate residual" in fig1_svg
            and absent(fig1_svg, r"DMD|lens|laser|camera|optical setup"),
        ),
        (
            r"\(x_data\) not claimed as new standalone reconstructor",
            "not a new standalone reconstructor" in method
            and absent(public_all, r"x_?\{?\\?rm data\}? is (a )?new invention|new standalone basic reconstructor"),
        ),
        (
            r"CS-TV formula uses \operatorname{TV}",
            r"\lambda\operatorname{TV}(x)" in public_all and r"\lambda TV" not in public_all,
        ),
        (
            "Results and validation headings are question/logic driven",
            r"\subsection{Similar final quality arises from different reconstruction regimes}" in results
            and r"\subsection{Simple-domain sanity checks}" in results
            and r"\subsection{Is random sensing reproducible?}" in validation
            and r"\subsection{Is the final measurement audit necessary?}" in validation
            and r"\subsection{Does the network depend on the bucket vector?}" in validation
            and r"\subsection{Is the method stronger than a classical CSGI-style prior?}" in validation
            and r"\subsection{Stability diagnostics}" in validation,
        ),
        (
            "Main numbers unchanged",
            all(value in public_all for value in RESULT_STRINGS),
        ),
        (
            "No PCA/oracle/architecture exploration",
            absent(public_all, r"PCA oracle|architecture pilot|sampling scaling|network replacement"),
        ),
        (
            "No hardware claim / no Colab / no Windows path / no Phase words in main text",
            absent(public_main, r"[A-Za-z]:[\\/]|Colab|\bPhase\s+\d+\b|internal phase|hardware optical experiment|hardware experiment"),
        ),
    ]

    for label, ok in checks:
        if not ok:
            failures.append(label)

    forbidden = [
        r"strict SOTA",
        r"state-of-the-art ranking",
        r"first pseudoinverse GI",
        r"first pseudo-inverse GI",
        r"first deep GI",
        r"GAN main mechanism",
        r"GAN-based method",
        r"low-frequency Hadamard.*5\\%.*high-quality",
        r"high-quality.*low-frequency Hadamard.*5\\%",
        r"TODO",
        r"Reference Placeholder",
    ]
    for pattern in forbidden:
        if re.search(pattern, public_all, flags=re.IGNORECASE):
            failures.append(f"Forbidden claim or placeholder remains: {pattern}")

    log_text = read(PROJECT / "main.log") + "\n" + read(PROJECT / "supplement.log")
    if re.search(r"undefined references|Citation `|Rerun to get cross-references right|LaTeX Warning: Reference", log_text):
        failures.append("LaTeX log contains unresolved references/citations or rerun warnings.")
    if "Overfull" in log_text:
        warnings.append("LaTeX log contains overfull box warnings; visual inspection is recommended.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Phase 37 Check Report",
        "",
        f"Status: {status}",
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
        f"- Main PDF: {OUT / 'main_v37.pdf'}",
        f"- Supplement PDF: {OUT / 'supplement_v37.pdf'}",
        f"- Figure 1: {FIG_DIR / 'fig1_minimal_mechanism_v37.pdf'}",
        f"- Figure 4: {FIG_DIR / 'fig4_measurement_attribution_v36.pdf'}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
