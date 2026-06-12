from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase32b_algorithm_optics_baselines"
PROJECT = OUT / "latex_project_algorithm_optics"
FIG_DIR = PROJECT / "figures"
BASELINE_DIR = OUT / "baseline_visuals"
REPORT = OUT / "ALGORITHM_OPTICS_BASELINE_CHECK_REPORT.md"

BANNED = [
    (r"\bPCA\b|oracle|architecture pilot|sampling scaling|network replacement", "exploratory content"),
    (r"[A-Za-z]:[\\/]|Windows path|Colab", "local/cloud path wording"),
    (r"\bPhase\s+\d+|\bPhase internal\b", "internal phase wording"),
    (r"strict SOTA|state-of-the-art ranking", "strict SOTA claim"),
    (r"GAN main mechanism|GAN-based method", "GAN main mechanism wording"),
    (r"binary learned illumination.*successful", "binary illumination success claim"),
    (r"low-frequency Hadamard.*5\\%.*high-quality|high-quality.*low-frequency Hadamard.*5\\%", "low-frequency 5% HQ claim"),
    (r"TODO|VERIFY|Reference Placeholder", "placeholder"),
]

HARDWARE_CLAIM = [
    r"hardware optical experiment is included",
    r"real optical experiment",
    r"physical experiment",
    r"experimental setup",
    r"laser",
    r"\blens\b",
    r"\bDMD\b",
    r"\bCCD\b",
]

RESULT_STRINGS = ["22.316", "22.271", "24.781", "24.730", "27.692", "25.019"]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def pdf_text(pdf: Path) -> str:
    txt = pdf.with_suffix(".txt")
    if pdf.exists():
        try:
            subprocess.run(["pdftotext", str(pdf), str(txt)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception:
            return ""
    return read(txt)


def source_text() -> str:
    parts = []
    for path in [PROJECT / "main.tex", PROJECT / "supplement.tex"]:
        parts.append(read(path))
    for folder in ["sections", "supplement", "tables"]:
        for path in sorted((PROJECT / folder).glob("*.tex")):
            parts.append(read(path))
    return "\n".join(parts)


def exists(path: Path, failures: list[str], label: str) -> None:
    if not path.exists():
        failures.append(f"Missing {label}: {path}")


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []

    main_pdf = OUT / "main_algorithm_optics.pdf"
    supp_pdf = OUT / "supplement_algorithm_optics.pdf"
    exists(main_pdf, failures, "main PDF")
    exists(supp_pdf, failures, "supplement PDF")

    for stem in [
        "fig1_forward_model_reconstruction",
        "fig7_gi_csgi_ours_visual_comparison",
        "fig7_gi_csgi_ours_all_supplement",
    ]:
        for ext in ("pdf", "png", "svg"):
            exists(FIG_DIR / f"{stem}.{ext}", failures, f"{stem}.{ext}")

    for name in [
        "gi_csgi_ours_rad5.pdf",
        "gi_csgi_ours_rad5.png",
        "gi_csgi_ours_scr5.pdf",
        "gi_csgi_ours_scr5.png",
        "gi_csgi_ours_all_supplement.pdf",
        "gi_csgi_ours_all_supplement.png",
        "baseline_visual_manifest.csv",
        "baseline_visual_manifest.md",
    ]:
        exists(BASELINE_DIR / name, failures, name)

    manuscript = source_text()
    compiled = pdf_text(main_pdf) + "\n" + pdf_text(supp_pdf)
    all_public_text = manuscript + "\n" + compiled

    for pattern, label in BANNED:
        if re.search(pattern, all_public_text, flags=re.IGNORECASE):
            failures.append(f"Banned wording found: {label}")

    no_hardware_statement = "No hardware optical experiment is included in this study" in manuscript
    positive_hardware_claim = bool(
        re.search(r"(?<!No )(?<!no )hardware optical experiment is included", all_public_text)
    )

    if "computational reconstruction problem" not in manuscript and "computational GI/SPI reconstruction study" not in manuscript:
        failures.append("Computational/simulation study framing is not explicit enough.")
    if not no_hardware_statement:
        failures.append("No-hardware-experiment statement missing.")
    if r"y=Ax+\epsilon" not in manuscript:
        failures.append("Computational forward model y=Ax+epsilon not stated.")

    if r"\lambda\operatorname{TV}(x)" not in manuscript:
        failures.append("CS-TV formula does not use lambda\\operatorname{TV}(x).")
    if "CSGI-style" not in manuscript or "CS-TV(PGD)" not in manuscript:
        failures.append("CS-TV is not clearly identified as CSGI-style CS-TV(PGD).")
    if re.search(r"fully optimized|exhaustively optimized compressed-sensing solver", manuscript, flags=re.IGNORECASE):
        if "not an exhaustively optimized" not in manuscript and "not as an exhaustively optimized" not in manuscript:
            failures.append("CS-TV may be described as fully optimized.")

    fig1_source = read(FIG_DIR / "fig1_forward_model_reconstruction.svg")
    if not fig1_source:
        failures.append("Figure 1 SVG is missing or unreadable.")
    for pattern in HARDWARE_CLAIM:
        if re.search(pattern, fig1_source, flags=re.IGNORECASE):
            failures.append(f"Figure 1 may imply hardware setup: {pattern}")
    if "hardware optical setup" not in manuscript:
        warnings.append("Figure 1 caption no-hardware wording not found verbatim.")

    if "GI/BP" not in manuscript or "CSGI" not in manuscript or "CS-TV(PGD)" not in manuscript:
        failures.append("GI/BP vs CSGI/CS-TV(PGD) vs Ours comparison wording missing.")
    if not (PROJECT / "tables" / "tableS8_gi_csgi_visual_subset.tex").exists():
        failures.append("Supplementary GI/CSGI visual subset table missing.")

    for value in RESULT_STRINGS:
        if value not in manuscript:
            failures.append(f"Main result number changed or missing: {value}")

    log_text = read(PROJECT / "main.log") + "\n" + read(PROJECT / "supplement.log")
    if re.search(r"undefined references|Citation `|Rerun to get cross-references right|LaTeX Warning: Reference", log_text):
        failures.append("LaTeX log contains unresolved references/citations or rerun warnings.")

    status = "PASS" if not failures else "FAIL"
    tv_formula_ok = r"\lambda\operatorname{TV}(x)" in manuscript
    lines = [
        "# Algorithm Optics Baseline Check Report",
        "",
        f"Status: {status}",
        "",
        "## Required Questions",
        "",
        f"1. Figure 1 avoids hardware-experiment implication: {'yes' if not any(re.search(p, fig1_source, flags=re.IGNORECASE) for p in HARDWARE_CLAIM) else 'no'}",
        f"2. Computational / simulation-based reconstruction study is explicit: {'yes' if no_hardware_statement else 'no'}",
        f"3. GI/BP vs CSGI/CS-TV vs Ours visual comparison added: {'yes' if (FIG_DIR / 'fig7_gi_csgi_ours_visual_comparison.pdf').exists() else 'no'}",
        f"4. CS-TV formula uses operatorname TV: {'yes' if tv_formula_ok else 'no'}",
        f"5. CS-TV is called CSGI-style compressed-sensing baseline: {'yes' if 'CSGI-style' in manuscript and 'CS-TV(PGD)' in manuscript else 'no'}",
        f"6. CS-TV is not called fully optimized: {'yes' if 'not as an exhaustively optimized' in manuscript or 'not an exhaustively optimized' in manuscript else 'no'}",
        f"7. No PCA / architecture / sampling scaling: {'yes' if not re.search(BANNED[0][0], all_public_text, flags=re.IGNORECASE) else 'no'}",
        f"8. No Windows path / Colab / internal phase words: {'yes' if not re.search(BANNED[1][0] + '|' + BANNED[2][0], all_public_text, flags=re.IGNORECASE) else 'no'}",
        f"9. No hardware experiment claim: {'yes' if no_hardware_statement and not positive_hardware_claim else 'no'}",
        f"10. Main result numbers unchanged: {'yes' if all(value in manuscript for value in RESULT_STRINGS) else 'no'}",
        "",
        "## Failures",
    ]
    lines.extend([f"- {item}" for item in failures] or ["- None."])
    lines += ["", "## Warnings"]
    lines.extend([f"- {item}" for item in warnings] or ["- None."])
    lines += [
        "",
        "## Output Paths",
        f"- Main PDF: {main_pdf}",
        f"- Supplement PDF: {supp_pdf}",
        f"- Figure 1: {FIG_DIR / 'fig1_forward_model_reconstruction.pdf'}",
        f"- Main GI/CSGI visual: {FIG_DIR / 'fig7_gi_csgi_ours_visual_comparison.pdf'}",
        f"- Supplement GI/CSGI visual: {FIG_DIR / 'fig7_gi_csgi_ours_all_supplement.pdf'}",
        f"- Baseline manifest: {BASELINE_DIR / 'baseline_visual_manifest.md'}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"status": status, "report": str(REPORT), "failures": len(failures), "warnings": len(warnings)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
