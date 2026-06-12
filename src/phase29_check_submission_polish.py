from __future__ import annotations

import re
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase29_final_submission_polish"
PROJECT = OUT / "latex_project_final"
REPORT = OUT / "SUBMISSION_POLISH_CHECK_REPORT.md"


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


FORBIDDEN_SOURCE_PATTERNS = [
    (r"\bPCA\b", "PCA exploration mention"),
    (r"\boracle\b", "oracle exploration mention"),
    (r"architecture pilot", "architecture pilot mention"),
    (r"\bNAFNet\b", "NAFNet mention"),
    (r"\bunrolled\b", "unrolled mention"),
    (r"\bISTA\b", "ISTA mention"),
    (r"recommend_full", "internal recommendation flag"),
    (r"sampling scaling", "sampling scaling mention"),
    (r"\bColab\b", "Colab mention"),
    (r"[A-Za-z]:[\\/]", "local path mention"),
    (r"\bPhase(?:25|26|27|28|29)?\b", "internal phase wording"),
    (r"Reference Placeholders", "reference placeholder text"),
    (r"TODO VERIFY", "TODO VERIFY placeholder"),
    (r"strict SOTA", "strict SOTA shorthand"),
    (r"GAN main mechanism", "GAN main mechanism claim"),
    (r"binary learned illumination successful", "binary learned illumination success claim"),
    (r"low-frequency Hadamard 5\\% high-quality", "low-frequency Hadamard 5% high-quality claim"),
    (r"\bwrapper\b", "forbidden wrapper wording"),
    (r"(?-i:fig\.)", "lowercase fig. reference"),
    (r"no-DC|-DC", "old DC ablation terminology"),
]


FORBIDDEN_PDF_PATTERNS = [
    (r"\bPCA\b", "PCA exploration mention in PDF"),
    (r"\boracle\b", "oracle exploration mention in PDF"),
    (r"architecture pilot", "architecture pilot mention in PDF"),
    (r"\bNAFNet\b", "NAFNet mention in PDF"),
    (r"\bunrolled\b", "unrolled mention in PDF"),
    (r"\bISTA\b", "ISTA mention in PDF"),
    (r"recommend_full", "internal recommendation flag in PDF"),
    (r"sampling scaling", "sampling scaling mention in PDF"),
    (r"\bColab\b", "Colab mention in PDF"),
    (r"[A-Za-z]:[\\/]", "local path mention in PDF"),
    (r"\bPhase(?:25|26|27|28|29)?\b", "internal phase wording in PDF"),
    (r"Reference Placeholders", "reference placeholder text in PDF"),
    (r"TODO VERIFY", "TODO VERIFY placeholder in PDF"),
    (r"strict SOTA", "strict SOTA shorthand in PDF"),
    (r"GAN main mechanism", "GAN main mechanism claim in PDF"),
    (r"binary learned illumination successful", "binary learned illumination success claim in PDF"),
    (r"low-frequency Hadamard 5% high-quality", "low-frequency Hadamard 5% high-quality claim in PDF"),
    (r"\bwrapper\b", "forbidden wrapper wording in PDF"),
    (r"(?-i:fig\.)", "lowercase fig. reference in PDF"),
    (r"no-DC|-DC", "old DC ablation terminology in PDF"),
]


REQUIRED_SOURCE_SNIPPETS = [
    r"\input{sections/data_availability.tex}",
    "TV-regularized compressed-sensing baseline solved by PGD",
    r"\operatorname{TV}(x)",
    r"\(-\mathrm{MC}\) removes the final measurement-consistency projection.",
    "Low-sampling GI is treated as measurement-constrained completion",
    "Dashed lines denote predefined operational thresholds",
    "The examples are qualitative visualizations; all quantitative conclusions are based on Table 1.",
    "Final PSNR alone hides whether performance comes from physical initialization, neural refinement, or both.",
    "they do not imply universal robustness",
    "The code, trained-checkpoint manifests, exported Rademacher measurement operators",
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


def pdf_text() -> str:
    texts = []
    for name in ("main_final_polished.txt", "supplement_final_polished.txt"):
        path = OUT / name
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(texts)


def check_figures() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    stems = [
        "fig1_mechanism_final",
        "fig2_primary_metrics_final",
        "fig3_qualitative_final",
        "fig4_measurement_attribution_final",
        "fig5_inference_ablation_final",
        "fig6_robustness_baselines_final",
    ]
    for stem in stems:
        for ext in ("pdf", "png", "svg"):
            path = OUT / "figures" / f"{stem}.{ext}"
            if not path.exists():
                failures.append(f"Missing figure file: {path}")
            project_path = PROJECT / "figures" / f"{stem}.{ext}"
            if not project_path.exists():
                failures.append(f"Missing project figure file: {project_path}")

    fig4 = OUT / "figures" / "fig4_measurement_attribution_final.svg"
    if fig4.exists():
        text = fig4.read_text(encoding="utf-8", errors="ignore")
        c5 = text.count("Lowfreq-5")
        c10 = text.count("Lowfreq-10")
        if c5 != 1 or c10 != 1:
            failures.append(f"Figure 4 Lowfreq labels unexpected: Lowfreq-5={c5}, Lowfreq-10={c10}.")
        if re.search(r"MNIST|Fashion", text, flags=re.IGNORECASE):
            failures.append("Figure 4 contains MNIST/Fashion labels.")
    else:
        warnings.append("Figure 4 SVG not available for label audit.")

    fig5 = OUT / "figures" / "fig5_inference_ablation_final.svg"
    if fig5.exists():
        text = fig5.read_text(encoding="utf-8", errors="ignore")
        if "-MC" not in text or "-DC" in text:
            failures.append("Figure 5 does not consistently use -MC terminology.")
    return failures, warnings


def citation_audit_status() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    audit = PROJECT / "citation_audit.md"
    if not audit.exists():
        return [f"Missing citation audit: {audit}"], warnings
    text = audit.read_text(encoding="utf-8")
    bad_exact = [
        "Missing cited keys: none",
        "Duplicate keys: none",
        "Duplicate obvious titles: none",
        "Malformed citation commands: none",
        "TODO or reference-placeholder text in references.bib: no",
    ]
    for snippet in bad_exact:
        if snippet not in text:
            failures.append(f"Citation audit did not pass: expected `{snippet}`.")
    if "Entries missing obvious title/year/venue fields: none" not in text:
        warnings.append("Citation audit reports entries missing obvious title/year/venue fields; inspect manually.")
    return failures, warnings


def log_status() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    for name in ("main.log", "supplement.log"):
        path = PROJECT / name
        if not path.exists():
            warnings.append(f"Missing LaTeX log: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"undefined references|Citation `|Rerun to get cross-references right", text):
            failures.append(f"LaTeX unresolved reference/citation warning remains in {name}.")
    return failures, warnings


def main() -> None:
    text, by_path = read_sources()
    pdf = pdf_text()
    failures: list[str] = []
    warnings: list[str] = []

    if not text:
        failures.append(f"No manuscript source found under {PROJECT}.")

    for pattern, label in FORBIDDEN_SOURCE_PATTERNS:
        hits = []
        for path, content in by_path.items():
            if re.search(pattern, content, flags=re.IGNORECASE):
                hits.append(Path(path).name)
        if hits:
            failures.append(f"{label}: {', '.join(sorted(set(hits)))}")

    if pdf:
        for pattern, label in FORBIDDEN_PDF_PATTERNS:
            if re.search(pattern, pdf, flags=re.IGNORECASE):
                failures.append(label)
    else:
        warnings.append("PDF text exports are missing; run pdftotext before final archival check.")

    for snippet in REQUIRED_SOURCE_SNIPPETS:
        if snippet not in text:
            failures.append(f"Missing required source snippet: {snippet}")

    for number in MAIN_NUMBERS:
        if number not in text:
            failures.append(f"Main result number missing: {number}")

    if "low-frequency Hadamard direct-current row" not in text:
        failures.append("DC row is not explicitly constrained to the low-frequency Hadamard direct-current row.")

    table3 = PROJECT / "tables" / "table3_ablation_summary.tex"
    if table3.exists():
        table3_text = table3.read_text(encoding="utf-8")
        if "Method & Full & -MC & -Null & Stage1 & Raw & EMA" not in table3_text:
            failures.append("Table 3 column header is not Full, -MC, -Null, Stage1, Raw, EMA.")
    else:
        failures.append("Table 3 source is missing.")

    for name in ("main_final_polished.pdf", "supplement_final_polished.pdf"):
        if not (OUT / name).exists():
            warnings.append(f"Compiled PDF missing: {OUT / name}")

    f_fail, f_warn = check_figures()
    c_fail, c_warn = citation_audit_status()
    l_fail, l_warn = log_status()
    failures.extend(f_fail)
    warnings.extend(f_warn)
    failures.extend(c_fail)
    warnings.extend(c_warn)
    failures.extend(l_fail)
    warnings.extend(l_warn)

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Submission Polish Check Report",
        "",
        f"Status: {status}",
        "",
        "## Core Checks",
        "",
        "- PCA / architecture exploration removed: " + ("yes" if not re.search(r"\bPCA\b|\boracle\b|architecture pilot|NAFNet|unrolled|ISTA|sampling scaling", text, flags=re.IGNORECASE) else "no"),
        "- Local paths / Colab / internal phase words removed from manuscript: " + ("yes" if not re.search(r"[A-Za-z]:[\\/]|Colab|\bPhase", text, flags=re.IGNORECASE) else "no"),
        "- Main result numbers preserved: " + ("yes" if not any(number not in text for number in MAIN_NUMBERS) else "no"),
        "- CS-TV formula corrected: " + ("yes" if r"\operatorname{TV}(x)" in text and "TV-regularized compressed-sensing baseline solved by PGD" in text else "no"),
        "- Data and code availability added: " + ("yes" if "Data and Code Availability" in text else "no"),
        "- Figure 4 Lowfreq bug fixed: " + ("yes" if not any("Figure 4 Lowfreq" in item for item in failures) else "no"),
        "- Figure 5 uses -MC: " + ("yes" if not any("Figure 5" in item for item in failures) else "no"),
        "- Compiled PDF text scan clean: " + ("yes" if pdf and not any(re.search(pattern, pdf, flags=re.IGNORECASE) for pattern, _ in FORBIDDEN_PDF_PATTERNS) else "not final"),
        "",
        "## Failures",
        "",
    ]
    lines.extend([f"- {failure}" for failure in failures] if failures else ["- None."])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- None."])
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- LaTeX project: {PROJECT}",
            f"- Main PDF: {OUT / 'main_final_polished.pdf'}",
            f"- Supplement PDF: {OUT / 'supplement_final_polished.pdf'}",
            f"- Citation audit: {PROJECT / 'citation_audit.md'}",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
