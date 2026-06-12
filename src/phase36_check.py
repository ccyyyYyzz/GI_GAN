from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase36_conventional_gi_aligned"
PROJECT = OUT / "latex_project_v36"
FIG_DIR = OUT / "figures"
REPORT = OUT / "PHASE36_CHECK_REPORT.md"

RESULT_STRINGS = ["22.316", "22.271", "24.781", "24.730", "27.692", "25.019"]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def pdf_text(pdf: Path) -> str:
    txt = pdf.with_suffix(".txt")
    if pdf.exists():
        subprocess.run(["pdftotext", str(pdf), str(txt)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return read(txt)


def source_text() -> str:
    parts = [read(PROJECT / "main.tex"), read(PROJECT / "supplement.tex"), read(PROJECT / "references.bib")]
    for folder in ("sections", "supplement", "tables"):
        base = PROJECT / folder
        if base.exists():
            for path in sorted(base.glob("*.tex")):
                parts.append(read(path))
    return "\n".join(parts)


def main_text() -> str:
    parts = [read(PROJECT / "main.tex")]
    for path in sorted((PROJECT / "sections").glob("*.tex")):
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


def citation_keys_from_tex(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", text):
        for key in match.group(1).split(","):
            keys.add(key.strip())
    return keys


def bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,\s]+)", text))


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []

    expected = {
        FIG_DIR / "fig1_conventional_gi_anchor.pdf": "Figure 1 PDF",
        FIG_DIR / "fig1_conventional_gi_anchor.png": "Figure 1 PNG",
        FIG_DIR / "fig1_conventional_gi_anchor.svg": "Figure 1 SVG",
        FIG_DIR / "fig4_measurement_attribution_v36.pdf": "Figure 4 PDF",
        FIG_DIR / "fig4_measurement_attribution_v36.png": "Figure 4 PNG",
        FIG_DIR / "fig4_measurement_attribution_v36.svg": "Figure 4 SVG",
        OUT / "main_v36.pdf": "main PDF",
        OUT / "supplement_v36.pdf": "supplement PDF",
        OUT / "citation_audit_phase36.md": "citation audit",
    }
    for path, label in expected.items():
        exists(path, failures, label)

    source = source_text()
    main_source = main_text()
    compiled = pdf_text(OUT / "main_v36.pdf") + "\n" + pdf_text(OUT / "supplement_v36.pdf")
    fig1_svg = strip_svg_images(read(FIG_DIR / "fig1_conventional_gi_anchor.svg"))
    fig4_svg = strip_svg_images(read(FIG_DIR / "fig4_measurement_attribution_v36.svg"))
    citation_audit = read(OUT / "citation_audit_phase36.md")
    public_text = source + "\n" + compiled

    bib = read(PROJECT / "references.bib")
    cited = citation_keys_from_tex(source)
    bib_set = bib_keys(bib)

    checks = [
        (
            r"Does Method explicitly relate \(A^Ty\) and \(x_{\rm data}\)?",
            "Relation to conventional bucket-pattern correlation" in main_source
            and r"\hat{x}_{\rm raw}=A^Ty" in main_source
            and r"x_{\rm data}=A^Tq" in main_source
            and r"x_{\rm data}\approx A^Ty" in main_source,
        ),
        (
            r"Does the manuscript say \(x_{\rm data}\) is not the novelty alone?",
            "not as a new standalone basic reconstructor" in main_source
            and "The novelty is not the linear initialization alone" in main_source,
        ),
        (
            "Does Figure 1 show conventional GI -> regularized data solution -> full method?",
            "Conventional GI" in fig1_svg
            and "raw correlation" in fig1_svg
            and "Regularized" in fig1_svg
            and "data solution" in fig1_svg
            and "Measurement-audited" in fig1_svg
            and "neural completion" in fig1_svg
            and ("A^T y" in fig1_svg or "A^Ty" in fig1_svg),
        ),
        (
            "Does Figure 1 avoid hardware experiment implication?",
            absent(fig1_svg, r"hardware|laser|lens|DMD|CCD|camera|optical setup|optical path"),
        ),
        (
            "Does Figure 4 y-axis no longer break PSNR?",
            "Delta PSNR (dB)" in fig4_svg and "PSN R" not in fig4_svg and "Neural gain PSN" not in fig4_svg,
        ),
        (
            r"Is CS-TV formula fixed to \operatorname{TV}(x)?",
            r"\lambda\operatorname{TV}(x)" in source and r"\lambda TV" not in source,
        ),
        (
            "Are all main results unchanged?",
            all(value in source for value in RESULT_STRINGS),
        ),
        (
            "No PCA/oracle/architecture exploration?",
            absent(public_text, r"PCA oracle|architecture pilot|sampling scaling|network replacement"),
        ),
        (
            "No Windows path / Colab / internal Phase words in main text?",
            absent(main_source + "\n" + compiled, r"[A-Za-z]:[\\/]|Colab|\bPhase\s+\d+\b|internal phase"),
        ),
        (
            "No strict SOTA / first pseudoinverse GI / first deep GI / GAN main mechanism claim?",
            absent(
                public_text,
                r"strict SOTA|state-of-the-art ranking|first pseudoinverse GI|first pseudo-inverse GI|first deep GI|GAN main mechanism|GAN-based method|binary learned illumination|low-frequency Hadamard.*5\\%.*high-quality|high-quality.*low-frequency Hadamard.*5\\%",
            ),
        ),
        (
            "All citation keys exist",
            cited.issubset(bib_set),
        ),
        (
            "Citation audit has manual verification note for generalized-inverse references",
            "manual" in citation_audit.lower() and "Gong" in citation_audit and "Czajkowski" in citation_audit,
        ),
    ]

    for label, ok in checks:
        if not ok:
            failures.append(label)

    for phrase in ["TODO", "Reference Placeholder", "new data solution", "first pseudoinverse", "first deep ghost"]:
        if re.search(re.escape(phrase), public_text, flags=re.IGNORECASE):
            failures.append(f"Forbidden placeholder or claim remains: {phrase}")

    missing_cites = sorted(cited - bib_set)
    if missing_cites:
        failures.append("Missing citation keys: " + ", ".join(missing_cites))

    log_text = read(PROJECT / "main.log") + "\n" + read(PROJECT / "supplement.log")
    if re.search(r"undefined references|Citation `|Rerun to get cross-references right|LaTeX Warning: Reference", log_text):
        failures.append("LaTeX log contains unresolved references/citations or rerun warnings.")
    if "Overfull" in log_text:
        warnings.append("LaTeX log contains overfull box warnings; visual inspection is recommended.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Phase 36 Check Report",
        "",
        f"Status: {status}",
        "",
        "## Required Questions",
        "",
    ]
    for i, (label, ok) in enumerate(checks[:10], 1):
        lines.append(f"{i}. {label}: {'yes' if ok else 'no'}")
    lines += [
        "",
        "## Citation Checks",
        f"- Cited keys: {len(cited)}",
        f"- Bibliography entries: {len(bib_set)}",
        f"- Missing citation keys: {', '.join(missing_cites) if missing_cites else 'none'}",
        f"- Citation audit manual-verification note present: {'yes' if checks[11][1] else 'no'}",
        "",
        "## Failures",
    ]
    lines.extend([f"- {item}" for item in failures] or ["- None."])
    lines += ["", "## Warnings"]
    lines.extend([f"- {item}" for item in warnings] or ["- None."])
    lines += [
        "",
        "## Output Paths",
        f"- Figure 1: {FIG_DIR / 'fig1_conventional_gi_anchor.pdf'}",
        f"- Figure 4: {FIG_DIR / 'fig4_measurement_attribution_v36.pdf'}",
        f"- Main PDF: {OUT / 'main_v36.pdf'}",
        f"- Supplement PDF: {OUT / 'supplement_v36.pdf'}",
        f"- Citation audit: {OUT / 'citation_audit_phase36.md'}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
