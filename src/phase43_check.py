from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase43_operator_circuit"
PROJECT = OUT / "latex_project_v43"
FIG_DIR = OUT / "figures"
REPORT = OUT / "PHASE43_CHECK_REPORT.md"

MAIN_PDF = OUT / "main_v43.pdf"
SUPP_PDF = OUT / "supplement_v43.pdf"
MAIN_TXT = OUT / "main_v43.txt"
SUPP_TXT = OUT / "supplement_v43.txt"
FIG_SVG = FIG_DIR / "fig1_operator_circuit_v43.svg"
FIG_PDF = FIG_DIR / "fig1_operator_circuit_v43.pdf"
FIG_PNG = FIG_DIR / "fig1_operator_circuit_v43_600dpi.png"
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
    info = json.loads(read(INFO) or "{}")
    sample_report = json.loads(read(SAMPLE_REPORT) or "{}")

    for path in [MAIN_PDF, SUPP_PDF, FIG_SVG, FIG_PDF, FIG_PNG, INFO, SAMPLE_REPORT]:
        if not path.exists():
            failures.append(f"Missing expected output: {path}")

    fig_has_components = all(
        term in fig_text
        for term in [
            "x_data",
            "raw residual",
            "filtered residual",
            "pre-audit",
            "final",
            "error",
            "RelMeasErr before vs after",
        ]
    )
    no_badges_or_comparison = all(
        phrase not in fig_text
        for phrase in [
            "Scr-5",
            "Rad-5",
            "Rademacher vs",
            "scrambled vs",
            "STL-10 5% final quality",
            "Audit ablation",
            "22.316 / 0.635",
            "22.271 / 0.632",
        ]
    )
    fixed_layers = all(term in fig_text for term in ["A", "B_lambda", "P_N", "Pi_y", "fixed"])
    trainable = "G_theta / R_phi" in fig_text and "trainable" in fig_text
    operator_text = (
        "B_lambda = A^T(AA^T + lambda I)^-1" in fig_text
        and "calibrated physical inverse core" in fig_text
        and "fixed, not learned" in fig_text
    )
    roles_text = all(
        term in fig_text
        for term in [
            "1. Anchor",
            "x_data = B_lambda y",
            "2. Gate",
            "P_N = I - B_lambda A",
            "3. Audit",
            "B_lambda e_y",
        ]
    )
    audit_loop = all(term in fig_text for term in ["y_tilde = A x_tilde", "e_y = y_tilde - y", "remeasure -> compare -> correct"])
    training_inset = "Training only" in fig_text and "loss -> updates neural modules only" in fig_text

    checks = [
        (r"Figure 1 centered on \(B_\lambda\)", operator_text),
        ("Figure 1 clearly shows anchor/gate/audit roles", roles_text),
        ("Figure 1 removes Rad/Scr comparison and result badges", no_badges_or_comparison),
        ("Figure 1 uses only one representative sample", "representative" in fig_text and no_badges_or_comparison),
        ("Figure 1 shows real x_data, residual, filtered residual, pre-audit, final", fig_has_components and not sample_report.get("missing_components")),
        ("Figure 1 draws audit as remeasure -> compare -> correct loop", audit_loop),
        ("Training is only a compact inset, not a disconnected long chain", training_inset and "long chain" not in fig_text),
        (r"\(A,B_\lambda,P_N,\Pi_y\) are marked fixed", fixed_layers),
        (r"\(G_\theta/R_\phi\) is marked trainable", trainable),
        (
            r"CS-TV formula uses \operatorname{TV}(x)",
            r"\lambda\operatorname{TV}(x)" in source
            and r"\lambda TV" not in source
            and re.search(r"(?<!\\operatorname\{)TV\(x\)", source) is None,
        ),
        ("Figure 4 y-axis has no broken label", "Delta PSNR (dB)" in fig4_svg and absent(fig4_svg, r"PSN\s+R|Neural gain PSN R")),
        ("Main result numbers unchanged", all(value in public_text for value in RESULT_STRINGS)),
        ("No training, new experiments, or PCA/oracle/architecture exploration", absent(public_text, r"PCA oracle|architecture pilot|sampling scaling|network replacement|new training experiment")),
        ("No hardware claim, Colab, Windows path, or phase words in main text", absent(main_text, r"hardware(?: experiment)?|Colab|[A-Za-z]:[\\/]|\\bPhase\\s+\\d+\\b|internal phase")),
    ]

    extra = [
        (
            "Abstract uses operator-centered language",
            "The same calibrated physical inverse is reused to form the data anchor, gate the neural residual, and audit the completed image." in abstract,
        ),
        (
            "Introduction required sentence present",
            "The network proposes missing structure; the calibrated measurement operator anchors, gates, and audits what can remain." in intro,
        ),
        (
            "Method has requested subsection order",
            heading_order(
                method,
                [
                    r"\subsection{Operator-centered view}",
                    r"\subsection{Data anchor}",
                    r"\subsection{Neural residual proposal}",
                    r"\subsection{Residual gate}",
                    r"\subsection{Bucket audit}",
                    r"\subsection{Relation to conventional bucket-pattern correlation}",
                    r"\subsection{Two-stage implementation and exact-operator handling}",
                ],
            ),
        ),
        (
            "Caption is operator-centered",
            "Operator-centered reconstruction circuit" in method
            and r"\hat{x}=\tilde{x}-B_\lambda(A\tilde{x}-y)" in method,
        ),
        (
            "Forbidden high-risk claims absent",
            absent(
                public_text,
                r"strict SOTA|state-of-the-art ranking|first\s+(GI|ghost|pseudoinverse|pseudo-inverse|deep)|GAN main mechanism|GAN-based method|low-frequency Hadamard 5\% high-quality",
            ),
        ),
        ("Figure is editable SVG and Inkscape-exported", bool(info.get("found")) and "<text" in fig_svg and FIG_PDF.exists() and FIG_PNG.exists()),
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
        warnings.append("Some requested component images were missing and rendered as not exported placeholders.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Phase 43 Check Report",
        "",
        f"Status: {status}",
        "",
        "## Inkscape",
        f"- Found: {'yes' if info.get('found') else 'no'}",
        f"- Command path: {info.get('command_path', '')}",
        f"- Version: {info.get('version', '')}",
        "",
        "## Representative Sample",
        f"- Component directory: {sample_report.get('component_dir', '')}",
        f"- Sample: {sample_report.get('sample', '')}",
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
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
