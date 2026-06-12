from __future__ import annotations

import re
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase30_submission_package"
PROJECT = OUT / "latex_project_submission"
FIG_DIR = OUT / "figures"
EXPORT_DIR = OUT / "figures_for_submission"
SOURCE_PACKAGE = OUT / "source_package"
REPORT = OUT / "SUBMISSION_CHECKLIST.md"


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


FORBIDDEN_PATTERNS = [
    (r"\bPCA\b", "PCA exploration mention"),
    (r"\boracle\b", "oracle exploration mention"),
    (r"architecture pilot", "architecture pilot mention"),
    (r"\bNAFNet\b", "NAFNet mention"),
    (r"\bunrolled\b", "unrolled mention"),
    (r"\bISTA\b", "ISTA mention"),
    (r"recommend_full", "recommend_full mention"),
    (r"sampling scaling", "sampling scaling mention"),
    (r"\bColab\b", "Colab mention"),
    (r"[A-Za-z]:[\\/]", "local path mention"),
    (r"\bPhase(?:25|26|27|28|29|30)?\b", "internal phase word"),
    (r"Reference Placeholders", "reference placeholder text"),
    (r"TODO VERIFY", "TODO VERIFY placeholder"),
    (r"strict SOTA|state-of-the-art ranking", "external-ranking overclaim wording"),
    (r"GAN main mechanism|GAN-based method", "GAN main mechanism wording"),
    (r"binary learned illumination successful", "binary learned illumination success claim"),
    (r"low-frequency Hadamard 5\\?% high-quality", "low-frequency Hadamard 5% high-quality claim"),
    (r"\bwrapper\b", "wrapper wording"),
    (r"(?-i:fig\.)", "lowercase fig."),
    (r"no-DC|-DC", "old DC ablation terminology"),
]


REQUIRED = [
    "TV-regularized compressed-sensing baseline solved by projected gradient descent (CS-TV)",
    r"\operatorname{TV}(x)",
    "Data and Code Availability",
    "Low-sampling GI is treated as measurement-constrained completion",
    "qualitative visualizations; all quantitative conclusions are based on Table 1",
    "Final PSNR alone hides whether performance comes from physical initialization",
    r"\(-\mathrm{MC}\) removes the final measurement-consistency projection",
]


FIGS = [
    "fig1_mechanism_submission",
    "fig2_primary_metrics_submission",
    "fig3_qualitative_submission",
    "fig4_measurement_attribution_submission",
    "fig5_inference_ablation_submission",
    "fig6_robustness_baselines_submission",
]


DOCS = [
    "cover_letter_draft.md",
    "highlights.md",
    "graphical_abstract_text.md",
    "significance_statement.md",
    "reviewer_summary.md",
]


def source_text() -> tuple[str, dict[str, str]]:
    paths = [PROJECT / "main.tex", PROJECT / "supplement.tex"]
    paths += list((PROJECT / "sections").glob("*.tex"))
    paths += list((PROJECT / "supplement").glob("*.tex"))
    paths += list((PROJECT / "tables").glob("*.tex"))
    by_path = {str(path): path.read_text(encoding="utf-8") for path in paths if path.exists()}
    return "\n".join(by_path.values()), by_path


def pdf_text() -> str:
    texts = []
    for name in ("main_submission.txt", "supplement_submission.txt"):
        path = OUT / name
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(texts)


def check_latex_logs() -> list[str]:
    failures = []
    for name in ("main.log", "supplement.log"):
        path = PROJECT / name
        if not path.exists():
            failures.append(f"Missing LaTeX log: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"undefined references|Citation `|Rerun to get cross-references right", text):
            failures.append(f"Unresolved LaTeX warning remains in {name}.")
    return failures


def check_figures() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    for stem in FIGS:
        for ext in ("pdf", "png", "svg"):
            if not (FIG_DIR / f"{stem}.{ext}").exists():
                failures.append(f"Missing submission figure: {FIG_DIR / f'{stem}.{ext}'}")
            if not (PROJECT / "figures" / f"{stem}.{ext}").exists():
                failures.append(f"Missing project figure: {PROJECT / 'figures' / f'{stem}.{ext}'}")
    for short in (
        "fig1_mechanism",
        "fig2_primary_metrics",
        "fig3_qualitative",
        "fig4_measurement_attribution",
        "fig5_inference_ablation",
        "fig6_robustness_baselines",
    ):
        for ext in ("pdf", "svg", "png", "tiff"):
            if not (EXPORT_DIR / f"{short}.{ext}").exists():
                warnings.append(f"Missing exported {ext.upper()} for {short}.")
    if not (EXPORT_DIR / "FIGURE_EXPORT_MANIFEST.md").exists():
        failures.append("Missing FIGURE_EXPORT_MANIFEST.md.")
    if not (FIG_DIR / "high_res" / "fig3_qualitative_submission_600dpi.png").exists():
        warnings.append("Missing explicit high-res Figure 3 PNG.")

    fig4 = FIG_DIR / "fig4_measurement_attribution_submission.svg"
    if fig4.exists():
        text = fig4.read_text(encoding="utf-8", errors="ignore")
        if text.count("Lowfreq-5") != 1 or text.count("Lowfreq-10") != 1:
            failures.append("Figure 4 Lowfreq-5/Lowfreq-10 labels are not exactly once each.")
        if re.search(r"MNIST|Fashion", text, flags=re.IGNORECASE):
            failures.append("Figure 4 contains MNIST/Fashion labels.")
    fig5 = FIG_DIR / "fig5_inference_ablation_submission.svg"
    if fig5.exists():
        text = fig5.read_text(encoding="utf-8", errors="ignore")
        if "-MC" not in text or "-DC" in text:
            failures.append("Figure 5 does not pass -MC terminology QA.")
    fig1 = FIG_DIR / "fig1_mechanism_submission.svg"
    if fig1.exists():
        text = fig1.read_text(encoding="utf-8", errors="ignore")
        for label in ("(a)", "(b)", "(c)", "(d)", "(e)"):
            if label not in text:
                failures.append(f"Figure 1 missing panel label {label}.")
        for token in ("Measurement audit", "epsilon", "lambda", "Pi", "x_", "P_N", "G_", "hat"):
            if token not in text and token.replace("_", "") not in text:
                warnings.append(f"Figure 1 token may need visual check: {token}.")
    return failures, warnings


def check_citation_audit() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    path = OUT / "citation_audit.md"
    if not path.exists():
        failures.append("Missing top-level citation_audit.md.")
        return failures, warnings
    text = path.read_text(encoding="utf-8")
    for snippet in (
        "Missing cited keys: none",
        "Duplicate keys: none",
        "Malformed citation commands: none",
        "TODO or reference-placeholder text in references.bib: no",
    ):
        if snippet not in text:
            failures.append(f"Citation audit did not pass: {snippet}")
    if "Manual verification needed" not in text:
        warnings.append("Citation audit should mention manual verification.")
    return failures, warnings


def check_source_package() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if not SOURCE_PACKAGE.exists():
        failures.append("Missing source_package directory.")
    else:
        for name in ("main.tex", "supplement.tex", "references.bib", "README_SUBMISSION_SOURCE.md"):
            if not (SOURCE_PACKAGE / name).exists():
                failures.append(f"Source package missing {name}.")
        source_texts = []
        for path in SOURCE_PACKAGE.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".tex", ".md", ".bib"}:
                source_texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        text = "\n".join(source_texts)
        if re.search(r"[A-Za-z]:[\\/]|Colab|Phase folders|PCA|architecture pilot", text, flags=re.IGNORECASE):
            failures.append("Source package contains a banned local/internal/exploration term.")
    if not (OUT / "source_package_submission.zip").exists():
        warnings.append("source_package_submission.zip was not generated.")
    return failures, warnings


def main() -> None:
    manuscript, by_path = source_text()
    pdf = pdf_text()
    failures: list[str] = []
    warnings: list[str] = []

    if not manuscript:
        failures.append("No manuscript sources found.")

    for pattern, label in FORBIDDEN_PATTERNS:
        hit_files = []
        for path, text in by_path.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                hit_files.append(Path(path).name)
        if hit_files:
            failures.append(f"{label}: {', '.join(sorted(set(hit_files)))}")
        if pdf and re.search(pattern, pdf, flags=re.IGNORECASE):
            failures.append(f"{label} in compiled PDF text.")

    for snippet in REQUIRED:
        if snippet not in manuscript:
            failures.append(f"Missing required wording: {snippet}")

    for number in MAIN_NUMBERS:
        if number not in manuscript:
            failures.append(f"Main result number missing: {number}")

    for path in (OUT / "main_submission.pdf", OUT / "supplement_submission.pdf"):
        if not path.exists():
            failures.append(f"Missing compiled PDF: {path}")

    for doc in DOCS:
        if not (OUT / "submission_docs" / doc).exists():
            failures.append(f"Missing submission doc: {doc}")

    failures.extend(check_latex_logs())
    f_fail, f_warn = check_figures()
    c_fail, c_warn = check_citation_audit()
    s_fail, s_warn = check_source_package()
    failures.extend(f_fail)
    warnings.extend(f_warn)
    failures.extend(c_fail)
    warnings.extend(c_warn)
    failures.extend(s_fail)
    warnings.extend(s_warn)

    if not pdf:
        warnings.append("PDF text export missing; final text scan not available.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Submission Checklist",
        "",
        f"Status: {status}",
        "",
        "## Automated Checks",
        "",
        f"- main PDF compiles: {'yes' if (OUT / 'main_submission.pdf').exists() else 'no'}",
        f"- supplement PDF compiles: {'yes' if (OUT / 'supplement_submission.pdf').exists() else 'no'}",
        "- figures exported: " + ("yes" if not any("Missing submission figure" in item for item in failures) else "no"),
        "- tables included: " + ("yes" if (PROJECT / "tables").exists() else "no"),
        "- no banned manuscript terms: " + ("yes" if not any("mention" in item or "wording" in item for item in failures) else "no"),
        "- no local paths/cloud-runtime/internal version words: " + ("yes" if not re.search(r"[A-Za-z]:[\\/]|Colab|\bPhase", manuscript, flags=re.IGNORECASE) else "no"),
        "- no PCA/oracle/architecture pilot content: " + ("yes" if not re.search(r"\bPCA\b|\boracle\b|architecture pilot", manuscript, flags=re.IGNORECASE) else "no"),
        "- no external benchmark ranking claim: " + ("yes" if "state-of-the-art ranking" not in manuscript else "no"),
        "- no GAN main mechanism claim: " + ("yes" if "GAN main mechanism" not in manuscript and "GAN-based method" not in manuscript else "no"),
        "- no low-frequency Hadamard 5% high-quality claim: " + ("yes" if "low-frequency Hadamard 5% high-quality" not in manuscript else "no"),
        "- CS-TV formula correct: " + ("yes" if r"\operatorname{TV}(x)" in manuscript else "no"),
        "- Data/code availability present: " + ("yes" if "Data and Code Availability" in manuscript else "no"),
        "- references audited: " + ("yes" if (OUT / "citation_audit.md").exists() else "no"),
        "- all figures cited: manual visual check still recommended",
        "- all tables cited: manual visual check still recommended",
        "- author names still placeholder: yes",
        "- affiliation still placeholder: yes",
        "- target journal template still needed: yes",
        "- manual reference verification needed: yes",
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
            "## Output Paths",
            "",
            f"- Main PDF: {OUT / 'main_submission.pdf'}",
            f"- Supplement PDF: {OUT / 'supplement_submission.pdf'}",
            f"- LaTeX project: {PROJECT}",
            f"- Figure export manifest: {EXPORT_DIR / 'FIGURE_EXPORT_MANIFEST.md'}",
            f"- Source package: {SOURCE_PACKAGE}",
            f"- Source zip: {OUT / 'source_package_submission.zip'}",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
