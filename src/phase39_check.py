from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase39_anchor_proposal_audit"
PROJECT = OUT / "latex_project_v39"
FIG_DIR = OUT / "figures"
COMP = OUT / "mechanism_components"
IMG_DIR = COMP / "component_images"
REPORT = OUT / "PHASE39_CHECK_REPORT.md"

MAIN_PDF = OUT / "main_v39.pdf"
SUPP_PDF = OUT / "supplement_v39.pdf"
MAIN_TXT = OUT / "main_v39.txt"
SUPP_TXT = OUT / "supplement_v39.txt"
FIG_SVG = FIG_DIR / "fig1_anchor_proposal_audit_v39.svg"
FIG_PDF = FIG_DIR / "fig1_anchor_proposal_audit_v39.pdf"
FIG_PNG = FIG_DIR / "fig1_anchor_proposal_audit_v39_600dpi.png"
METRICS_CSV = COMP / "component_metrics.csv"
AVAIL_CSV = COMP / "available_components.csv"
MANIFEST = COMP / "component_manifest.md"
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
    "22.202",
    "19.399",
    "22.155",
    "6.352",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def pdf_text(path: Path, txt_path: Path) -> str:
    if not txt_path.exists() and path.exists():
        try:
            subprocess.run(["pdftotext", str(path), str(txt_path)], check=False)
        except FileNotFoundError:
            pass
    return read(txt_path)


def all_source() -> str:
    return "\n".join(read(path) for path in PROJECT.rglob("*.tex"))


def absent(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is None


def heading_order(text: str, headings: list[str]) -> bool:
    pos = []
    for heading in headings:
        idx = text.find(heading)
        if idx < 0:
            return False
        pos.append(idx)
    return pos == sorted(pos)


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    source = all_source()
    main_text = pdf_text(MAIN_PDF, MAIN_TXT)
    supp_text = pdf_text(SUPP_PDF, SUPP_TXT)
    public_all = main_text + "\n" + supp_text
    fig_svg = read(FIG_SVG)
    abstract = read(PROJECT / "sections" / "abstract.tex")
    intro = read(PROJECT / "sections" / "introduction.tex")
    method = read(PROJECT / "sections" / "method.tex")
    results = read(PROJECT / "sections" / "results.tex")
    validation = read(PROJECT / "sections" / "validation_ablation.tex")
    metric_rows = csv_rows(METRICS_CSV)
    avail_rows = csv_rows(AVAIL_CSV)
    inkscape = json.loads(read(INFO) or "{}")

    expected = [
        MAIN_PDF,
        SUPP_PDF,
        FIG_SVG,
        FIG_PDF,
        FIG_PNG,
        METRICS_CSV,
        AVAIL_CSV,
        MANIFEST,
        IMG_DIR / "rad5_gt.png",
        IMG_DIR / "rad5_x_data.png",
        IMG_DIR / "rad5_pre_audit_no_mc.png",
        IMG_DIR / "rad5_final.png",
        IMG_DIR / "rad5_abs_error.png",
        IMG_DIR / "rad5_measurement_residual_bar.png",
        IMG_DIR / "scr5_gt.png",
        IMG_DIR / "scr5_x_data.png",
        IMG_DIR / "scr5_pre_audit_no_mc.png",
        IMG_DIR / "scr5_final.png",
        IMG_DIR / "scr5_abs_error.png",
        IMG_DIR / "scr5_measurement_residual_bar.png",
    ]
    for path in expected:
        if not path.exists():
            failures.append(f"Missing expected output: {path}")

    checks = [
        (
            "Figure 1 uses real images",
            "<image" in fig_svg
            and "data:image/png;base64" in fig_svg
            and "Real STL-10 evidence strip" in fig_svg
            and (IMG_DIR / "rad5_final.png").exists()
            and (IMG_DIR / "scr5_final.png").exists(),
        ),
        (
            "Figure 1 no longer uses conventional GI comparison as main path",
            "Conventional GI" not in fig_svg and "raw BP" not in fig_svg and "Anchor -> Proposal -> Legality Filter -> Audit" in fig_svg,
        ),
        (
            "Figure 1 expresses Anchor -> Proposal -> Legality Filter -> Audit",
            all(item in fig_svg for item in ["Measured anchor", "Neural proposal", "Legality filter", "Bucket audit"]),
        ),
        (
            "Exported x_data, pre-audit/no-MC, final, RelMeasErr",
            len(metric_rows) >= 10
            and any(r.get("component") == "x_data_backprojection" for r in metric_rows)
            and any(r.get("component") == "pre_audit_no_mc" for r in metric_rows)
            and any(r.get("component") == "final_audited_output" for r in metric_rows)
            and any(r.get("component") == "measurement_residual_profile" for r in metric_rows)
            and "rel_meas_err" in (metric_rows[0].keys() if metric_rows else []),
        ),
        (
            "G_theta described as proposal network, not final reconstructor",
            "The neural network is not trained to be the final judge" in method
            and "proposes a candidate missing component" in method,
        ),
        (
            "P_N described as legality filter",
            "legality filter" in method and "suppresses residual components that the measurement operator can see" in method,
        ),
        (
            "Pi_y described as bucket audit",
            "Bucket audit" in method and "remeasuring the completed image through" in method,
        ),
        (
            "Abstract reflects anchor-proposal-filter-audit logic",
            all(term in abstract for term in ["undertermined" if False else "underdetermined", "measured anchor", "propose missing structure", "P_N", "Pi_y"])
            and len(re.findall(r"\b\w+\b", abstract)) <= 190,
        ),
        (
            "Method headings rewritten",
            heading_order(
                method,
                [
                    r"\subsection{The measurement set: low sampling leaves an affine ambiguity}",
                    r"\subsection{Data anchor: a measured point near the feasible set}",
                    r"\subsection{Neural proposal: learned missing structure}",
                    r"\subsection{Legality filter: inserting only measurement-silent residuals}",
                    r"\subsection{Bucket audit: projecting the completed image back to the data}",
                    r"\subsection{Implementation details and exact-operator handling}",
                ],
            ),
        ),
        (
            r"CS-TV formula fixed to \operatorname{TV}(x)",
            r"\lambda\operatorname{TV}(x)" in source
            and r"\lambda TV" not in source
            and re.search(r"(?<!operatorname\{)TV\(x\)", source) is None,
        ),
        ("Main result numbers unchanged", all(value in public_all for value in RESULT_STRINGS)),
        (
            "No training / no new experiment / no PCA-oracle-architecture exploration",
            absent(public_all, r"PCA oracle|architecture pilot|sampling scaling|network replacement")
            and "No training or new large experiment was run" in read(MANIFEST),
        ),
        (
            "No hardware claim / Colab / Windows path / phase words in main text",
            absent(main_text, r"hardware(?: experiment)?|Colab|[A-Za-z]:[\\/]|\bPhase\s+\d+\b|internal phase"),
        ),
        (
            "No SOTA / first pseudoinverse GI / GAN main mechanism / lowfreq 5% HQ claim",
            absent(
                public_all,
                r"strict SOTA|state-of-the-art ranking|first\s+(GI|ghost|pseudoinverse|pseudo-inverse|deep)|GAN main mechanism|GAN-based method|low-frequency Hadamard 5\\?% high-quality",
            )
            and "low-frequency Hadamard 5% high-quality" not in public_all,
        ),
    ]

    extra_checks = [
        ("Introduction has required key question sentence", "The key question is not whether a neural network can make the image sharper" in intro),
        ("Introduction has network proposes sentence", "The network proposes; the measurement operator decides what can remain." in intro),
        ("Results reordered", heading_order(results, [
            r"\subsection{Primary STL-10 performance}",
            r"\subsection{Mechanism in images: anchor, proposal, audit}",
            r"\subsection{Similar final quality, different reconstruction regimes}",
            r"\subsection{Simple-domain sanity checks}",
        ])),
        ("Validation headings rewritten", heading_order(validation, [
            r"\subsection{Random operators must be exactly audited}",
            r"\subsection{Removing the audit breaks the mechanism}",
            r"\subsection{Corrupting the bucket vector breaks the reconstruction}",
            r"\subsection{Classical CS-TV prior is not enough}",
            r"\subsection{Stability and diagnostic controls}",
        ])),
        ("Availability CSV marks residual hooks unavailable honestly", all(r.get("raw_candidate_residual_available") == "False" and r.get("filtered_residual_available") == "False" for r in avail_rows)),
        ("Inkscape available", bool(inkscape.get("found"))),
    ]

    for label, ok in checks + extra_checks:
        if not ok:
            failures.append(label)

    log_text = read(PROJECT / "main.log") + "\n" + read(PROJECT / "supplement.log")
    if re.search(r"undefined references|Citation `|Rerun to get cross-references right|LaTeX Warning: Reference", log_text):
        failures.append("LaTeX log contains unresolved references/citations or rerun warnings.")
    if "Overfull" in log_text:
        warnings.append("LaTeX log contains overfull box warnings; visual inspection is recommended.")

    status = "PASS" if not failures else "FAIL"
    lines = [
        "# Phase 39 Check Report",
        "",
        f"Status: {status}",
        "",
        "## Inkscape",
        f"- Found: {'yes' if inkscape.get('found') else 'no'}",
        f"- Command path: {inkscape.get('command_path', '')}",
        f"- Version: {inkscape.get('version', '')}",
        "",
        "## Required Questions",
        "",
    ]
    for i, (label, ok) in enumerate(checks, 1):
        lines.append(f"{i}. {label}: {'yes' if ok else 'no'}")
    lines += ["", "## Extra Checks"]
    for label, ok in extra_checks:
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
        f"- Component folder: {COMP}",
        f"- Component metrics: {METRICS_CSV}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
