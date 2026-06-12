from __future__ import annotations

import re
from pathlib import Path


DRIVE_ROOT = Path("E:/ns_mc_gan_gi")
OUTPUT_ROOT = DRIVE_ROOT / "outputs_phase27_paper_purification"
PROJECT = OUTPUT_ROOT / "latex_project_purified"
REPORT = OUTPUT_ROOT / "PAPER_PURIFICATION_CHECK_REPORT.md"


FAIL_PATTERNS = [
    "Phase25",
    "Phase26",
    "PCA",
    "oracle",
    "PCA prior",
    "PCA subspace",
    "linear-prior oracle",
    "architecture pilot",
    "architecture ablation pilot",
    "current_hq_rad5_pilot",
    "current_hq_scr5_pilot",
    "nafnet_small",
    "NAFNet",
    "unrolled",
    "Unrolled-ISTA",
    "ISTA",
    "gate decision",
    "recommend_full",
    "sampling scaling",
    "two-point fit",
    "model_psnr_at_15pct",
    "model_psnr_at_20pct",
    "architecture smoke",
    "smoke PSNR",
    "train=96 / eval=8",
    "train=5000 / eval=500 PCA",
    "current architecture is strongest",
    "PCA oracle proves",
    "theoretical maximum",
    "theoretical limit",
    "upper bound",
    "Colab",
    "E:/",
    "C:/",
    "Reference Placeholders",
    "TODO VERIFY",
]

WARNING_PATTERNS = [
    "architecture",
    "state-of-the-art",
    "universal robustness",
]


def source_files() -> list[Path]:
    roots = [
        PROJECT / "main.tex",
        PROJECT / "supplement.tex",
        PROJECT / "references.bib",
        PROJECT / "citation_audit.md",
    ]
    roots.extend(sorted((PROJECT / "sections").glob("*.tex")))
    roots.extend(sorted((PROJECT / "supplement").glob("*.tex")))
    roots.extend(sorted((PROJECT / "tables").glob("*.tex")))
    return [p for p in roots if p.exists()]


def scan_file(path: Path, patterns: list[str]) -> list[tuple[str, int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    hits: list[tuple[str, int, str]] = []
    for pattern in patterns:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        for idx, line in enumerate(lines, start=1):
            if regex.search(line):
                hits.append((pattern, idx, line.strip()))
    return hits


def has_required_structure() -> list[str]:
    required = [
        PROJECT / "main.tex",
        PROJECT / "supplement.tex",
        PROJECT / "sections" / "abstract.tex",
        PROJECT / "sections" / "introduction.tex",
        PROJECT / "sections" / "related_work.tex",
        PROJECT / "sections" / "problem_formulation.tex",
        PROJECT / "sections" / "method.tex",
        PROJECT / "sections" / "measurement_families.tex",
        PROJECT / "sections" / "experimental_protocol.tex",
        PROJECT / "sections" / "results.tex",
        PROJECT / "sections" / "validation_ablation.tex",
        PROJECT / "sections" / "discussion.tex",
        PROJECT / "sections" / "limitations.tex",
        PROJECT / "sections" / "conclusion.tex",
        PROJECT / "supplement" / "supplement.tex",
        PROJECT / "figures",
        PROJECT / "tables",
        PROJECT / "references.bib",
        PROJECT / "citation_audit.md",
    ]
    return [str(path) for path in required if not path.exists()]


def check_core_results() -> list[str]:
    expected_strings = [
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
    combined = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in source_files()
        if p.suffix in {".tex", ".md", ".bib"}
    )
    return [s for s in expected_strings if s not in combined]


def write_report(
    failures: dict[str, list[tuple[str, int, str]]],
    warnings: dict[str, list[tuple[str, int, str]]],
    missing_structure: list[str],
    missing_core: list[str],
) -> None:
    failed = bool(failures or missing_structure or missing_core)
    lines: list[str] = []
    lines.append("# Paper Purification Check Report")
    lines.append("")
    lines.append(f"Status: {'FAIL' if failed else 'PASS'}")
    lines.append(f"Project: `{PROJECT}`")
    lines.append("")
    lines.append("## Structure")
    if missing_structure:
        lines.append("Missing required files/directories:")
        for item in missing_structure:
            lines.append(f"- `{item}`")
    else:
        lines.append("All required Phase27 project files/directories are present.")
    lines.append("")
    lines.append("## Forbidden Content")
    if failures:
        lines.append("Forbidden terms were found in submission sources:")
        for file, hits in failures.items():
            lines.append(f"- `{file}`")
            for pattern, line_no, excerpt in hits:
                lines.append(f"  - line {line_no}: `{pattern}` in `{excerpt[:180]}`")
    else:
        lines.append("No forbidden Phase25/26, PCA, architecture-pilot, local-path, or placeholder terms were found in submission sources.")
    lines.append("")
    lines.append("## Core Result Retention")
    if missing_core:
        lines.append("The following required metric strings were not found:")
        for item in missing_core:
            lines.append(f"- `{item}`")
    else:
        lines.append("All required core metric strings are present in the purified submission sources.")
    lines.append("")
    lines.append("## Warnings")
    if warnings:
        lines.append("Non-failing warning terms were found; review manually:")
        for file, hits in warnings.items():
            lines.append(f"- `{file}`")
            for pattern, line_no, excerpt in hits[:10]:
                lines.append(f"  - line {line_no}: `{pattern}` in `{excerpt[:180]}`")
            if len(hits) > 10:
                lines.append(f"  - ... {len(hits) - 10} additional warning hits")
    else:
        lines.append("No warning terms were found.")
    lines.append("")
    lines.append("## Manual Checks Still Required")
    lines.append("- Verify references against the target journal style and metadata.")
    lines.append("- Inspect figure aesthetics and final page layout.")
    lines.append("- Insert final author, affiliation, and data/code availability wording.")
    lines.append("- Confirm target journal template requirements.")
    lines.append("- Confirm that accompanying data packages are separated from submission PDF sources.")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not PROJECT.exists():
        raise FileNotFoundError(f"Purified project not found: {PROJECT}")

    failures: dict[str, list[tuple[str, int, str]]] = {}
    warnings: dict[str, list[tuple[str, int, str]]] = {}
    for path in source_files():
        fail_hits = scan_file(path, FAIL_PATTERNS)
        warn_hits = scan_file(path, WARNING_PATTERNS)
        if fail_hits:
            failures[str(path.relative_to(PROJECT))] = fail_hits
        if warn_hits:
            warnings[str(path.relative_to(PROJECT))] = warn_hits

    missing_structure = has_required_structure()
    missing_core = check_core_results()
    write_report(failures, warnings, missing_structure, missing_core)

    result = {
        "status": "FAIL" if failures or missing_structure or missing_core else "PASS",
        "report": str(REPORT),
        "forbidden_files": len(failures),
        "missing_structure": len(missing_structure),
        "missing_core_metrics": len(missing_core),
    }
    print(result)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
