from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase41_inkscape_signal_trace"
PROJECT = OUT / "latex_project_v41"
FIG_DIR = OUT / "figures"
EDIT_PACK = OUT / "figure1_edit_pack"
COMP = OUT / "components"
REPORT = OUT / "PHASE41_CHECK_REPORT.md"

MAIN_PDF = OUT / "main_v41.pdf"
SUPP_PDF = OUT / "supplement_v41.pdf"
MAIN_TXT = OUT / "main_v41.txt"
SUPP_TXT = OUT / "supplement_v41.txt"
FIG_SVG = FIG_DIR / "fig1_signal_trace_v41.svg"
FIG_PDF = FIG_DIR / "fig1_signal_trace_v41.pdf"
FIG_PNG = FIG_DIR / "fig1_signal_trace_v41_600dpi.png"
METRICS_CSV = COMP / "component_metrics.csv"
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
    if path.exists() and not txt_path.exists():
        try:
            subprocess.run(["pdftotext", str(path), str(txt_path)], check=False)
        except FileNotFoundError:
            pass
    return read(txt_path)


def all_tex() -> str:
    return "\n".join(read(path) for path in PROJECT.rglob("*.tex"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def image_count(svg: str) -> int:
    return len(re.findall(r"<image\b", svg))


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    source = all_tex()
    main_text = pdf_text(MAIN_PDF, MAIN_TXT)
    supp_text = pdf_text(SUPP_PDF, SUPP_TXT)
    public_text = main_text + "\n" + supp_text
    fig_svg = read(FIG_SVG)
    fig4_svg = read(PROJECT / "figures" / "fig4_measurement_attribution_v36.svg")
    info = json.loads(read(INFO) or "{}")
    rows = csv_rows(METRICS_CSV)
    method = read(PROJECT / "sections" / "method.tex")
    intro = read(PROJECT / "sections" / "introduction.tex")
    results = read(PROJECT / "sections" / "results.tex")

    expected = [
        MAIN_PDF,
        SUPP_PDF,
        FIG_SVG,
        FIG_PDF,
        FIG_PNG,
        EDIT_PACK / "fig1_signal_trace_v41.svg",
        EDIT_PACK / "fig1_signal_trace_v41.pdf",
        EDIT_PACK / "fig1_signal_trace_v41_600dpi.png",
        EDIT_PACK / "FIGURE1_EDITING_GUIDE.md",
        EDIT_PACK / "FIGURE1_STYLE_GUIDE.md",
        METRICS_CSV,
        MANIFEST,
    ]
    for method_key in ["rad5", "scr5"]:
        for name in [
            "ground_truth.png",
            "pattern_preview.png",
            "bucket_vector.png",
            "x_data.png",
            "raw_residual.png",
            "filtered_residual.png",
            "pre_audit.png",
            "final_audited.png",
            "abs_error_final.png",
            "measurement_residual_pre.png",
            "measurement_residual_post.png",
            "relmeaserr_bar.png",
            "psnr_ssim_bar.png",
        ]:
            expected.append(COMP / method_key / name)
    for path in expected:
        if not path.exists():
            failures.append(f"Missing expected output: {path}")

    raw_ok = rows and all(str(row.get("raw_residual_available", "")).lower() == "true" for row in rows)
    filtered_ok = rows and all(str(row.get("filtered_residual_available", "")).lower() == "true" for row in rows)
    exact_rad_ok = any(row.get("method_id") == "rademacher5_hq_noise001_colab" and str(row.get("exact_A_used", "")).lower() == "true" for row in rows)

    checks = [
        ("Inkscape found and command path recorded", bool(info.get("found")) and bool(info.get("command_path")) and bool(info.get("version"))),
        ("Real intermediate images exported", len(rows) >= 2 and all(row.get("status") == "completed_eval_only" for row in rows)),
        (
            "Pattern and bucket measurement source shown",
            (COMP / "rad5" / "pattern_preview.png").exists()
            and (COMP / "rad5" / "bucket_vector.png").exists()
            and "patterns" in fig_svg
            and "bucket y" in fig_svg,
        ),
        ("x_data shown", "Data anchor" in fig_svg and "x_data" in fig_svg),
        ("Raw and filtered residuals shown or reason recorded", raw_ok and filtered_ok and "raw residual" in fig_svg and "filtered residual" in fig_svg),
        ("Pre-audit and final audited outputs shown", "Pre-audit image" in fig_svg and "Final output" in fig_svg and "final" in fig_svg),
        ("RelMeasErr pre/post shown", "RelMeasErr" in fig_svg and "pre" in fig_svg and "post" in fig_svg),
        ("Training feedback path shown", "Training feedback path" in fig_svg and "Loss backpropagates through fixed physics layers" in fig_svg),
        ("A, P_N, Pi_y fixed and G_theta/R_phi trainable explained", "fixed differentiable physics layers" in method and "trainable components" in method and "G_\\theta/R_\\phi" in method),
        ("Figure 1 is not an abstract formula pile", image_count(fig_svg) >= 18 and "Real intermediate images" in fig_svg),
        ("Figure 1 does not use traditional GI comparison as main line", "Conventional GI" not in fig_svg and "Forward signal path" in fig_svg),
        ("Figure 1 SVG generated and Inkscape editable", FIG_SVG.exists() and "font-family" in fig_svg and "<text" in fig_svg),
        ("PDF and 600dpi PNG exported", FIG_PDF.exists() and FIG_PNG.exists() and FIG_PNG.stat().st_size > 0),
        (r"CS-TV formula fixed to \operatorname{TV}(x)", r"\lambda\operatorname{TV}(x)" in source and r"\lambda TV" not in source and re.search(r"(?<!operatorname\{)TV\(x\)", source) is None),
        ("Figure 4 y-axis has no broken PSNR label", "Delta PSNR (dB)" in fig4_svg and absent(fig4_svg, r"PSN\s+R|Neural gain PSN R")),
        ("Main result numbers unchanged", all(value in public_text for value in RESULT_STRINGS)),
        ("No training / no new experiment / no PCA/oracle/architecture exploration", "No training, new benchmark" in read(MANIFEST) and absent(public_text, r"PCA oracle|architecture pilot|sampling scaling|network replacement")),
        ("No hardware claim / Colab / Windows path / phase words in main text", absent(main_text, r"hardware(?: experiment)?|Colab|[A-Za-z]:[\\/]|\bPhase\s+\d+\b|internal phase")),
    ]

    extra = [
        ("Rademacher exact-A used", exact_rad_ok),
        ("Method signal-flow heading present", r"\subsection{Signal flow: from bucket measurements to audited reconstruction}" in method),
        ("Introduction key sentences present", "The key question is not whether a network can make the image sharper" in intro and "The network proposes missing structure; the measurement operator filters and audits what can remain." in intro),
        ("Results order present", heading_order(results, [
            r"\subsection{Primary STL-10 performance}",
            r"\subsection{Signal trace and qualitative mechanism}",
            r"\subsection{Similar final quality, different reconstruction regimes}",
            r"\subsection{Simple-domain sanity checks}",
        ])),
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
        "# Phase 41 Check Report",
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
    lines += ["", "## Residual Export"]
    if raw_ok and filtered_ok:
        lines.append("- Raw residual and filtered residual were exported by eval-only forward extras.")
    else:
        lines.append("- Raw/filtered residual export was incomplete; see component metrics.")
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
        f"- Figure 1 edit pack: {EDIT_PACK}",
        f"- Component folder: {COMP}",
        f"- Component metrics: {METRICS_CSV}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
