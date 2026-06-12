from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase44_operator_centered"
PROJECT = OUT / "latex_project_v44"
FIG_DIR = OUT / "figures"
PROV = OUT / "provenance"
REPORT = OUT / "PHASE44_CHECK_REPORT.md"

MAIN_PDF = OUT / "main_v44.pdf"
SUPP_PDF = OUT / "supplement_v44.pdf"
MAIN_TXT = OUT / "main_v44.txt"
SUPP_TXT = OUT / "supplement_v44.txt"
FIG_SVG = FIG_DIR / "fig1_operator_centered_v44.svg"
FIG_PDF = FIG_DIR / "fig1_operator_centered_v44.pdf"
FIG_PNG = FIG_DIR / "fig1_operator_centered_v44_600dpi.png"
PROV_CSV = PROV / "provenance_metrics.csv"
PROV_MD = PROV / "provenance_metrics.md"
PROV_PNG = PROV / "provenance_grid.png"
PROV_PDF = PROV / "provenance_grid.pdf"
INFO = OUT / "INKSCAPE_INFO.json"
SAMPLE_REPORT = OUT / "FIGURE1_SAMPLE_REPORT.json"

RESULT_STRINGS = [
    "22.316",
    "0.635",
    "22.271",
    "0.632",
    "24.781",
    "0.747",
    "24.730",
    "0.746",
    "27.692",
    "0.956",
    "25.019",
    "0.837",
    "15.019",
    "17.025",
    "7.297",
    "7.756",
    "14.310",
    "14.533",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def pdf_text(path: Path, txt_path: Path) -> str:
    if path.exists() and not txt_path.exists():
        try:
            subprocess.run(["pdftotext", str(path), str(txt_path)], check=False)
        except FileNotFoundError:
            pass
    return read(txt_path)


def all_tex() -> str:
    return "\n".join(read(path) for path in PROJECT.rglob("*.tex"))


def absent(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is None


def heading_order(text: str, headings: list[str]) -> bool:
    positions = []
    for heading in headings:
        idx = text.find(heading)
        if idx < 0:
            return False
        positions.append(idx)
    return positions == sorted(positions)


def abstract_word_count(text: str) -> int:
    stripped = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})?", " ", text)
    stripped = re.sub(r"\$[^$]*\$|\\\([^)]*\\\)", " ", stripped)
    return len(re.findall(r"[A-Za-z0-9%.-]+", stripped))


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    source = all_tex()
    main_text = pdf_text(MAIN_PDF, MAIN_TXT)
    supp_text = pdf_text(SUPP_PDF, SUPP_TXT)
    public_text = main_text + "\n" + supp_text
    fig_svg = read(FIG_SVG)
    fig_text = fig_svg.replace("-&gt;", "->").replace("&lt;", "<").replace("&amp;", "&")
    fig4_svg = read(PROJECT / "figures" / "fig4_measurement_attribution_v36.svg")
    method = read(PROJECT / "sections" / "method.tex")
    intro = read(PROJECT / "sections" / "introduction.tex")
    abstract = read(PROJECT / "sections" / "abstract.tex")
    results = read(PROJECT / "sections" / "results.tex")
    discussion = read(PROJECT / "sections" / "discussion.tex")
    table3 = read(PROJECT / "tables" / "table3_ablation_summary.tex")
    info = json.loads(read(INFO) or "{}")
    sample_report = json.loads(read(SAMPLE_REPORT) or "{}")
    prov_md = read(PROV_MD)

    for path in [MAIN_PDF, SUPP_PDF, FIG_SVG, FIG_PDF, FIG_PNG, PROV_CSV, PROV_MD, PROV_PNG, PROV_PDF, INFO, SAMPLE_REPORT]:
        if not path.exists():
            failures.append(f"Missing expected output: {path}")

    method_order = heading_order(
        method,
        [
            r"\subsection{Measurement model and information split}",
            r"\subsection{Calibrated inverse core}",
            r"\subsection{Anchor: measured component}",
            r"\subsection{Gate: null-space-confined proposal}",
            r"\subsection{Audit: remeasure and correct}",
            r"\subsection{Training through frozen physics}",
            r"\subsection{Two-stage implementation and exact operator handling}",
            r"\subsection{Relation to conventional bucket-pattern correlation}",
        ],
    )
    results_order = heading_order(
        results,
        [
            r"\subsection{Mechanism walkthrough and provenance}",
            r"\subsection{Primary STL-10 performance}",
            r"\subsection{Audit and perturbation as mechanism validation}",
            r"\subsection{Similar final quality, different anchor/gain regimes}",
            r"\subsection{Simple-domain sanity checks}",
        ],
    )

    checks = [
        (
            "Title and abstract are operator-centered",
            "Measurement-Audited Neural Completion for Computational Ghost Imaging" in read(PROJECT / "main.tex")
            and "operator-centered reconstruction circuit" in abstract
            and 170 <= abstract_word_count(abstract) <= 190,
        ),
        (
            "B_lambda is central and uses the requested formula",
            r"B_\lambda=A^{\mathsf T}(AA^{\mathsf T}+\lambda I)^{-1}" in source
            and source.count("B_\\lambda") >= 18,
        ),
        (
            "Method is organized as anchor, gate, audit around one operator",
            method_order
            and r"x_{\rm data}=B_\lambda y" in method
            and r"P_N=I-B_\lambda A" in method
            and r"\Pi_y(v)=v-B_\lambda(Av-y)" in method,
        ),
        (
            "Figure 1 is B_lambda-centered",
            "B_lambda = A^T(AA^T + lambda I)^-1" in fig_text and "calibrated inverse core" in fig_text,
        ),
        (
            "Figure 1 has no Rad/Scr labels, metric badges, or ablation bars",
            all(
                phrase not in fig_text
                for phrase in [
                    "Rad-5",
                    "Scr-5",
                    "Rademacher",
                    "scrambled",
                    "PSNR",
                    "SSIM",
                    "ablation",
                    "Delta",
                    "RelMeasErr",
                ]
            ),
        ),
        (
            "Figure 1 states the three roles of the same operator",
            all(term in fig_text for term in ["1. Anchor", "2. Gate", "3. Audit", "one frozen physical operator reused in three roles"]),
        ),
        (
            "Figure 1 uses one representative real sample",
            "representative_sample" in read(SAMPLE_REPORT)
            and not sample_report.get("missing_components")
            and "measured component" in fig_text
            and "final" in fig_text,
        ),
        (
            "Figure 1 avoids traditional GI as the main line",
            "Conventional GI" not in fig_text and "traditional GI" not in fig_text,
        ),
        (
            "Figure 1 avoids a misleading iterative closed loop",
            "closed loop" not in fig_text.lower() and "iterate" not in fig_text.lower(),
        ),
        (
            "Provenance decomposition is computed and exported",
            all(path.exists() for path in [PROV_CSV, PROV_MD, PROV_PNG, PROV_PDF])
            and all(term in prov_md for term in ["Rad-5", "Scr-5", "Rad-10", "Scr-10"]),
        ),
        (
            "Measured/learned components are described as regularized soft decomposition",
            "regularized soft decomposition" in prov_md
            and r"B_\lambda A\hat{x}" in results
            and r"(I-B_\lambda A)\hat{x}" in results,
        ),
        (
            "CS-TV formula uses lambda operatorname TV",
            r"\lambda\operatorname{TV}(x)" in source
            and r"\lambda TV" not in source
            and re.search(r"(?<!\\operatorname\{)TV\(x\)", source) is None,
        ),
        (
            "Figure 4 y-axis label is fixed",
            ("Delta PSNR (dB)" in fig4_svg or r"\Delta\mathrm{PSNR}" in fig4_svg)
            and absent(fig4_svg, r"PSN\s+R|Neural gain PSN R"),
        ),
        (
            "Table 3 uses -MC notation for measurement-consistency removal",
            (r"-\mathrm{MC}" in table3 or "-MC" in table3) and "-DC" not in table3,
        ),
        ("Main result numbers are unchanged", all(value in public_text for value in RESULT_STRINGS)),
        (
            "No training or new model experiments are introduced",
            absent(public_text, r"new training experiment|architecture pilot|PCA oracle|sampling scaling|network replacement")
            and "eval-only" in results,
        ),
        (
            "No hardware/SOTA/first/GAN-main-mechanism claims",
            absent(
                public_text,
                r"hardware experiment|strict SOTA|state-of-the-art ranking|first\s+(GI|ghost|pseudoinverse|pseudo-inverse|deep)|GAN main mechanism|GAN-based method|low-frequency Hadamard 5\% high-quality",
            ),
        ),
    ]

    extra = [
        (
            "Required introduction sentences present",
            "The central challenge is not merely to make the image sharper, but to let the network add only structure that the measurements cannot see." in intro
            and "The same calibrated operator that forms the data anchor also gates the neural proposal and audits the final image." in intro,
        ),
        ("Network proposes sentence present", "The network proposes; the calibrated operator anchors, gates, and audits." in source),
        ("Results subsections follow requested order", results_order),
        (
            "Discussion opening matches requested logic",
            "Low-sampling ghost imaging is a negotiation between measured evidence and learned prior." in discussion
            and "endpoint PSNR alone is incomplete" in discussion
            and "Hardware validation remains future work" in discussion,
        ),
        ("Figure is editable SVG and Inkscape-exported", bool(info.get("found")) and "<text" in fig_svg and FIG_PDF.exists() and FIG_PNG.exists()),
        ("No Colab/Windows/internal phase wording in main text", absent(main_text, r"Colab|[A-Za-z]:[\\/]|\\bPhase\\s+\\d+\\b|internal phase")),
    ]

    for label, ok in checks + extra:
        if not ok:
            failures.append(label)

    log_text = read(PROJECT / "main.log") + "\n" + read(PROJECT / "supplement.log")
    if re.search(r"undefined references|Citation `|Rerun to get cross-references right|LaTeX Warning: Reference", log_text):
        failures.append("LaTeX log contains unresolved references/citations or rerun warnings.")
    if "Overfull" in log_text:
        warnings.append("LaTeX log contains overfull box warnings; visual inspection is recommended.")
    if sample_report.get("missing_components"):
        warnings.append("Some Figure 1 component images were missing.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Phase 44 Check Report",
        "",
        f"Status: {status}",
        "",
        "## Inkscape",
        f"- Found: {'yes' if info.get('found') else 'no'}",
        f"- Command path: {info.get('command_path', '')}",
        f"- Version: {info.get('version', '')}",
        "",
        "## Figure 1 Sample",
        f"- Component directory: {sample_report.get('component_dir', '')}",
        f"- Missing components: {sample_report.get('missing_components', [])}",
        "",
        "## Required Questions",
        "",
    ]
    for i, (label, ok) in enumerate(checks, 1):
        lines.append(f"{i}. {label}: {'yes' if ok else 'no'}")
    lines += ["", "## Extra Checks"]
    for label, ok in extra:
        lines.append(f"- {label}: {'yes' if ok else 'no'}")
    lines += ["", "## Failures"]
    lines.extend([f"- {item}" for item in failures] or ["- None."])
    lines += ["", "## Warnings"]
    lines.extend([f"- {item}" for item in warnings] or ["- None."])
    lines += [
        "",
        "## Output Paths",
        f"- Main PDF: {MAIN_PDF}",
        f"- Supplement PDF: {SUPP_PDF}",
        f"- Figure 1 SVG: {FIG_SVG}",
        f"- Figure 1 PDF: {FIG_PDF}",
        f"- Figure 1 600dpi PNG: {FIG_PNG}",
        f"- Provenance CSV: {PROV_CSV}",
        f"- Provenance Markdown: {PROV_MD}",
        f"- Provenance PNG: {PROV_PNG}",
        f"- Provenance PDF: {PROV_PDF}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
