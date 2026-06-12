from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase42_closed_loop_figure"
PROJECT = OUT / "latex_project_v42"
FIG_DIR = OUT / "figures"
REPORT = OUT / "PHASE42_CHECK_REPORT.md"

MAIN_PDF = OUT / "main_v42.pdf"
SUPP_PDF = OUT / "supplement_v42.pdf"
MAIN_TXT = OUT / "main_v42.txt"
SUPP_TXT = OUT / "supplement_v42.txt"
FIG_SVG = FIG_DIR / "fig1_closed_loop_v42.svg"
FIG_PDF = FIG_DIR / "fig1_closed_loop_v42.pdf"
FIG_PNG = FIG_DIR / "fig1_closed_loop_v42_600dpi.png"
INFO = OUT / "INKSCAPE_INFO.json"

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
    fig4_svg = read(PROJECT / "figures" / "fig4_measurement_attribution_v36.svg")
    method = read(PROJECT / "sections" / "method.tex")
    intro = read(PROJECT / "sections" / "introduction.tex")
    abstract = read(PROJECT / "sections" / "abstract.tex")
    info = json.loads(read(INFO) or "{}")

    for path in [MAIN_PDF, SUPP_PDF, FIG_SVG, FIG_PDF, FIG_PNG, INFO]:
        if not path.exists():
            failures.append(f"Missing expected output: {path}")

    one_sample_components = [
        "known patterns A",
        "bucket vector y",
        "x_data",
        "raw residual",
        "filtered residual",
        "x_tilde",
        "final",
        "error",
        "RelMeasErr before vs after",
    ]
    has_no_badges = all(
        phrase not in fig_svg
        for phrase in [
            "STL-10 5% final quality",
            "Audit ablation",
            "Full 22.202",
            "Scr-5",
            "Rad-5",
            "22.316 / 0.635",
            "22.271 / 0.632",
        ]
    )
    checks = [
        ("Figure 1 uses only one representative sample", "Scr-5" not in fig_svg and "Rad-5" not in fig_svg),
        ("Figure 1 removes Rad/Scr comparison and final quality badge", has_no_badges),
        ("Figure 1 has measurement evidence rail", "Measurement evidence rail" in fig_svg and "Known A, bucket vector y" in fig_svg),
        (
            "Figure 1 shows y used for x_data, Pi_y, and measurement loss/RelMeasErr",
            all(term in fig_svg for term in ["forms x_data", "audits x_hat", "checks A x_hat - y", "RelMeasErr before vs after"]),
        ),
        ("Figure 1 has image completion engine", "Image completion engine" in fig_svg and all(term in fig_svg for term in one_sample_components)),
        ("Figure 1 has compact training feedback loop", "Training feedback loop" in fig_svg and "updates neural proposal" in fig_svg and "long chain" not in fig_svg),
        (
            "Figure 1 shows losses update only G_theta/R_phi, not A/P_N/Pi_y",
            "not learned" in fig_svg and "G_theta / R_phi" in fig_svg and "updates only G_theta/R_phi" in fig_svg,
        ),
        ("Figure 1 is editable SVG and has Inkscape PDF/600dpi PNG", bool(info.get("found")) and "<text" in fig_svg and FIG_PDF.exists() and FIG_PNG.exists()),
        ("Method adds one-sample signal flow", r"\subsection{One-sample signal flow}" in method),
        ("Algorithm 1 added", "Algorithm 1: Measurement-audited neural completion" in method and "update \\(G_\\theta,R_\\phi\\)" in method),
        (
            "Relation to conventional GI is downgraded to later Method subsection",
            heading_order(
                method,
                [
                    r"\subsection{One-sample signal flow}",
                    r"\subsection{Data anchor}",
                    r"\subsection{Neural residual proposal}",
                    r"\subsection{Residual admissibility filtering}",
                    r"\subsection{Bucket-measurement audit}",
                    r"\subsection{Relation to conventional bucket-pattern correlation}",
                    r"\subsection{Two-stage implementation and exact operator handling}",
                ],
            ),
        ),
        (
            r"CS-TV formula fixed to \operatorname{TV}(x)",
            r"\lambda\operatorname{TV}(x)" in source
            and r"\lambda TV" not in source
            and re.search(r"(?<!operatorname\{)TV\(x\)", source) is None,
        ),
        ("Figure 4 y-axis has no broken PSNR", "Delta PSNR (dB)" in fig4_svg and absent(fig4_svg, r"PSN\s+R|Neural gain PSN R")),
        ("Main result numbers unchanged", all(value in public_text for value in RESULT_STRINGS)),
        ("No new training / no PCA-oracle-architecture exploration", absent(public_text, r"PCA oracle|architecture pilot|sampling scaling|network replacement")),
        ("No hardware claim / Colab / Windows path / phase words in main text", absent(main_text, r"hardware(?: experiment)?|Colab|[A-Za-z]:[\\/]|\bPhase\s+\d+\b|internal phase")),
    ]

    extra = [
        ("Abstract includes bucket vector anchors/audits/training residual", "bucket vector is used not only to initialize" in abstract),
        ("Introduction required sentences present", "The key question is not whether a network can make the image sharper" in intro and "The network proposes missing structure; the measurement operator filters and audits what can remain." in intro),
        ("Figure 1 does not use traditional GI as main line", "Conventional GI" not in fig_svg and "bucket-pattern correlation" not in fig_svg),
        ("Forbidden high-risk claims absent", absent(public_text, r"strict SOTA|state-of-the-art ranking|first\s+(GI|ghost|pseudoinverse|pseudo-inverse|deep)|GAN main mechanism|GAN-based method|low-frequency Hadamard 5\\?% high-quality")),
    ]

    for label, ok in checks + extra:
        if not ok:
            failures.append(label)

    log_text = read(PROJECT / "main.log") + "\n" + read(PROJECT / "supplement.log")
    if re.search(r"undefined references|Citation `|Rerun to get cross-references right|LaTeX Warning: Reference", log_text):
        failures.append("LaTeX log contains unresolved references/citations or rerun warnings.")
    if "Overfull" in log_text:
        warnings.append("LaTeX log contains overfull box warnings; visual inspection is recommended.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Phase 42 Check Report",
        "",
        f"Status: {status}",
        "",
        "## Inkscape",
        f"- Found: {'yes' if info.get('found') else 'no'}",
        f"- Command path: {info.get('command_path', '')}",
        f"- Version: {info.get('version', '')}",
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
