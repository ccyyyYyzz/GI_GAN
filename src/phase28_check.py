from __future__ import annotations

import re
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase28_paragraph_polish"
PROJECT = OUT / "latex_project_v28"
REPORT = OUT / "PARAGRAPH_POLISH_CHECK_REPORT.md"


MAIN_NUMBERS = [
    "22.316",
    "0.635",
    "7.297",
    "15.019",
    "22.271",
    "0.632",
    "14.310",
    "7.961",
    "24.781",
    "0.747",
    "7.756",
    "17.025",
    "24.730",
    "0.746",
    "14.533",
    "10.197",
    "27.692",
    "0.956",
    "25.019",
    "0.837",
]


REQUIRED_SNIPPETS = [
    "low sampling needs a learned prior, yet unconstrained priors may hallucinate",
    "optical measurement design and computational inversion are coupled",
    "The measurement vector therefore does not identify a single image",
    "The central issue is not only whether a neural network can improve image quality",
    "Existing learned GI/SPI methods often report improved reconstructions",
    "Geometrically, the measurements select an affine set in image space rather than a single point",
    r"x_{\rm data}\) should be interpreted as a measured-component representative",
    "The projection removes the component of the proposed residual that would be visible",
    r"The final projection \(\Pi_y\) therefore acts as an audit step",
    "Rademacher sensing uses a random measurement matrix",
    "Similar final quality does not imply the same reconstruction mechanism",
    "Final PSNR alone is insufficient to explain the role of the measurement family",
    "TV-regularized compressed-sensing baseline solved by PGD",
    r"\operatorname{TV}(x)",
    "This is a negative-control test",
    "This audit is important because random sensing results cannot be reproduced",
    "We do not claim a strict state-of-the-art ranking",
    "Binary learned illumination is not claimed",
    "More broadly, the results suggest that low-sampling ghost imaging should be designed",
]


FORBIDDEN_REGEXES = [
    (r"\bwrapper\b", "forbidden term `wrapper`"),
    (r"(?-i:fig\.)", "lowercase `fig.`"),
    (r"no-DC", "old `no-DC` terminology"),
    (r"-DC", "old `-DC` label"),
    (r"DC projection", "old `DC projection` phrase"),
    (r"TODO VERIFY", "TODO VERIFY placeholder"),
    (r"Reference Placeholders", "reference placeholder text"),
    (r"\bColab\b", "Colab mention in manuscript"),
    (r"[A-Za-z]:[\\/]", "local Windows path in manuscript source"),
    (r"\bPhase\s*\d*", "internal phase wording in manuscript"),
    (r"\bPCA\b", "PCA exploration mention in manuscript"),
    (r"\boracle\b", "oracle exploration mention in manuscript"),
    (r"architecture pilot", "architecture pilot exploration mention"),
    (r"\bNAFNet\b", "NAFNet exploration mention"),
    (r"\bunrolled\b", "unrolled exploration mention"),
    (r"\bISTA\b", "ISTA exploration mention"),
    (r"gate decision", "gate-decision exploration mention"),
    (r"recommend_full", "internal recommendation flag"),
    (r"sampling scaling", "sampling scaling exploration mention"),
    (r"network replacement", "network replacement exploration mention"),
    (r"\bGAN\b", "GAN mechanism claim or acronym"),
    (r"low-frequency Hadamard at 5\\% is (?:a )?high-quality", "low-frequency Hadamard 5% high-quality claim"),
]


PDF_TEXT_FORBIDDEN_REGEXES = [
    (r"\bno-DC\b", "old `no-DC` terminology in compiled PDF text"),
    (r"-DC", "old `-DC` label in compiled PDF text"),
    (r"\bwrapper\b", "forbidden term `wrapper` in compiled PDF text"),
    (r"(?-i:fig\.)", "lowercase `fig.` in compiled PDF text"),
    (r"\bColab\b", "Colab mention in compiled PDF text"),
    (r"\bPCA\b", "PCA exploration mention in compiled PDF text"),
    (r"\boracle\b", "oracle exploration mention in compiled PDF text"),
    (r"architecture pilot", "architecture pilot exploration mention in compiled PDF text"),
    (r"\bNAFNet\b", "NAFNet exploration mention in compiled PDF text"),
    (r"\bunrolled\b", "unrolled exploration mention in compiled PDF text"),
    (r"\bISTA\b", "ISTA exploration mention in compiled PDF text"),
    (r"TODO VERIFY", "TODO VERIFY placeholder in compiled PDF text"),
    (r"Reference Placeholders", "reference placeholder text in compiled PDF text"),
    (r"\bGAN\b", "GAN mechanism claim or acronym in compiled PDF text"),
]


def source_files() -> list[Path]:
    files: list[Path] = [PROJECT / "main.tex", PROJECT / "supplement.tex"]
    files.extend(sorted((PROJECT / "sections").glob("*.tex")))
    files.extend(sorted((PROJECT / "supplement").glob("*.tex")))
    files.extend(sorted((PROJECT / "tables").glob("*.tex")))
    return [path for path in files if path.exists()]


def read_sources() -> tuple[str, dict[str, str]]:
    by_path: dict[str, str] = {}
    for path in source_files():
        by_path[str(path)] = path.read_text(encoding="utf-8")
    return "\n".join(by_path.values()), by_path


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?", text))


def check_fig4() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    fig = OUT / "figures" / "fig4_measurement_attribution_v28.svg"
    if not fig.exists():
        return [f"Missing Figure 4 SVG: {fig}"], warnings
    text = fig.read_text(encoding="utf-8", errors="ignore")
    count5 = text.count("Lowfreq-5")
    count10 = text.count("Lowfreq-10")
    if count5 != 1 or count10 != 1:
        failures.append(f"Figure 4 Lowfreq label count is unexpected: Lowfreq-5={count5}, Lowfreq-10={count10}.")
    if re.search(r"MNIST|Fashion", text, flags=re.IGNORECASE):
        failures.append("Figure 4 contains MNIST/Fashion labels, which should not be in the main attribution figure.")
    for ext in ("pdf", "png", "svg"):
        path = OUT / "figures" / f"fig4_measurement_attribution_v28.{ext}"
        if not path.exists():
            failures.append(f"Missing Figure 4 {ext}: {path}")
    return failures, warnings


def check_fig3() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    for ext in ("pdf", "png", "svg"):
        path = OUT / "figures" / f"fig3_qualitative_reconstruction_v28.{ext}"
        if not path.exists():
            failures.append(f"Missing Figure 3 {ext}: {path}")
    warning_path = OUT / "qualitative_selection_warning_v28.md"
    if warning_path.exists() and "previous qualitative figure was retained" in warning_path.read_text(encoding="utf-8"):
        warnings.append("Figure 3 fell back to the previous qualitative figure; manual visual selection may still be needed.")
    return failures, warnings


def main() -> None:
    text, by_path = read_sources()
    failures: list[str] = []
    warnings: list[str] = []

    if not text:
        failures.append(f"No manuscript sources found under {PROJECT}.")

    abstract_md = OUT / "abstract_v28.md"
    if abstract_md.exists():
        wc = word_count(abstract_md.read_text(encoding="utf-8"))
        if not (180 <= wc <= 230):
            failures.append(f"Abstract word count is outside 180-230 words: {wc}.")
    else:
        failures.append(f"Missing abstract markdown: {abstract_md}")

    for pattern, label in FORBIDDEN_REGEXES:
        matches = []
        for path, content in by_path.items():
            if re.search(pattern, content, flags=re.IGNORECASE):
                matches.append(Path(path).name)
        if matches:
            failures.append(f"{label}: {', '.join(sorted(set(matches)))}")

    pdf_text_paths = [OUT / "main_v28.txt", OUT / "supplement_v28.txt"]
    pdf_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in pdf_text_paths if path.exists())
    if pdf_text:
        for pattern, label in PDF_TEXT_FORBIDDEN_REGEXES:
            if re.search(pattern, pdf_text, flags=re.IGNORECASE):
                failures.append(label)
    else:
        warnings.append("PDF text exports are not present yet; run pdftotext before final archival check.")

    for snippet in REQUIRED_SNIPPETS:
        if snippet not in text:
            failures.append(f"Missing required wording snippet: {snippet}")

    for num in MAIN_NUMBERS:
        if num not in text:
            failures.append(f"Main result number missing from v28 sources: {num}")

    if r"\mathcal{C}_y" not in text:
        failures.append(r"Missing \mathcal{C}_y notation.")
    if r"\mathrm{Null}(A)" not in text:
        failures.append(r"Missing \mathrm{Null}(A) notation.")

    table3 = PROJECT / "tables" / "table3_ablation_summary.tex"
    if table3.exists():
        table3_text = table3.read_text(encoding="utf-8")
        if "Method & Full & -MC & -Null & Stage1 & Raw & EMA" not in table3_text:
            failures.append("Table 3 columns are not Full, -MC, -Null, Stage1, Raw, EMA.")
        if r"\(-\mathrm{MC}\) removes the final measurement-consistency projection." not in table3_text:
            failures.append("Table 3 caption does not define -MC as final measurement-consistency projection removal.")
    else:
        failures.append(f"Missing Table 3 source: {table3}")

    dc_row_hits = re.findall(r".{0,45}DC[- ]row.{0,65}", text, flags=re.IGNORECASE)
    if dc_row_hits and "low-frequency Hadamard direct-current row" not in text:
        failures.append("DC row appears without the low-frequency Hadamard direct-current-row definition.")

    f4_fail, f4_warn = check_fig4()
    f3_fail, f3_warn = check_fig3()
    failures.extend(f4_fail)
    warnings.extend(f4_warn)
    failures.extend(f3_fail)
    warnings.extend(f3_warn)

    for stem in ("main_v28.pdf", "supplement_v28.pdf"):
        path = OUT / stem
        if not path.exists():
            warnings.append(f"Compiled output not present yet: {path}")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Paragraph Polish Check Report",
        "",
        f"Status: {status}",
        "",
        "## Core Checks",
        "",
        f"- Abstract word count: {word_count(abstract_md.read_text(encoding='utf-8')) if abstract_md.exists() else 'missing'}",
        "- no-DC / -DC terminology fixed: " + ("yes" if not re.search(r"no-DC|-DC", text) else "no"),
        "- Figure 4 Lowfreq bug fixed: " + ("yes" if not f4_fail else "no"),
        "- Figure 3 v28 files present: " + ("yes" if not f3_fail else "no"),
        "- Main result numbers unchanged in source: " + ("yes" if not any(num not in text for num in MAIN_NUMBERS) else "no"),
        "- CS-TV wording and formula present: " + ("yes" if "TV-regularized compressed-sensing baseline solved by PGD" in text and r"\operatorname{TV}(x)" in text else "no"),
        "- Compiled PDF text scan clean: " + ("yes" if pdf_text and not any(re.search(pattern, pdf_text, flags=re.IGNORECASE) for pattern, _ in PDF_TEXT_FORBIDDEN_REGEXES) else "not final"),
        "",
        "## Failures",
        "",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Main PDF: {OUT / 'main_v28.pdf'}",
            f"- Supplement PDF: {OUT / 'supplement_v28.pdf'}",
            f"- Figure 3: {OUT / 'figures' / 'fig3_qualitative_reconstruction_v28.pdf'}",
            f"- Figure 4: {OUT / 'figures' / 'fig4_measurement_attribution_v28.pdf'}",
            f"- Terminology report: {OUT / 'terminology_fix_report.md'}",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
