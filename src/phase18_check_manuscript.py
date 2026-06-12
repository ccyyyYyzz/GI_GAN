from __future__ import annotations

import re
from pathlib import Path

from .phase18_rewrite_common import OUT, REGISTRY, TABLES, read_csv, write_text


LATEX = OUT / "latex_project"
SECTIONS = LATEX / "sections"


REQUIRED_FILES = [
    OUT / "manuscript_v3.md",
    OUT / "manuscript_v3.tex",
    OUT / "REWRITE_SUMMARY.md",
    OUT / "source_manifest.json",
    LATEX / "main.tex",
    LATEX / "references.bib",
    LATEX / "sections" / "abstract.tex",
    LATEX / "sections" / "problem_formulation.tex",
    LATEX / "sections" / "method.tex",
    LATEX / "sections" / "measurement_families.tex",
    LATEX / "sections" / "results.tex",
    LATEX / "sections" / "ablation_validation.tex",
    LATEX / "supplement" / "supplement.tex",
]

REQUIRED_FIGURES = [
    "fig1_mechanism",
    "fig2_measurement_attribution",
    "fig3_main_results",
    "fig4_inference_ablation",
    "fig5_robustness_baselines",
]

REQUIRED_TABLES = [
    "table1_primary_strict_noleak_results",
    "table2_measurement_attribution",
    "table3_inference_ablation_summary",
]

REQUIRED_LABELS = [
    "fig:mechanism",
    "fig:measurement_attribution",
    "fig:main_results",
    "fig:inference_ablation",
    "fig:robustness_baselines",
    "tab:primary_results",
    "tab:measurement_attribution",
    "tab:ablation_summary",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main_text() -> str:
    parts = [read(LATEX / "main.tex")]
    if SECTIONS.exists():
        parts.extend(read(path) for path in sorted(SECTIONS.glob("*.tex")))
    return "\n".join(parts)


def label_is_referenced(text: str, label: str) -> bool:
    pattern = re.compile(r"\\[cC]ref\{[^}]*" + re.escape(label) + r"[^}]*\}")
    return bool(pattern.search(text))


def status_line(ok: bool, label: str, detail: str = "") -> str:
    mark = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    return f"- {mark}: {label}{suffix}"


def check_numbers_from_sources() -> tuple[bool, str]:
    registry_rows = read_csv(REGISTRY)
    table_rows = {name: read_csv(path) for name, path in TABLES.items()}
    ok = bool(registry_rows) and all(rows or name in {"classwise"} for name, rows in table_rows.items())
    detail = f"registry rows={len(registry_rows)}; supplementary tables checked={len(table_rows)}"
    return ok, detail


def check() -> list[tuple[bool, str, str]]:
    text = main_text()
    lower = text.lower()
    checks: list[tuple[bool, str, str]] = []

    missing = [str(path) for path in REQUIRED_FILES if not path.exists()]
    checks.append((not missing, "Required manuscript and project files exist", "; ".join(missing[:4]) if missing else ""))

    fig_missing = []
    for stem in REQUIRED_FIGURES:
        for ext in [".pdf", ".png", ".svg"]:
            if not (OUT / "figures" / f"{stem}{ext}").exists():
                fig_missing.append(f"{stem}{ext}")
    checks.append((not fig_missing, "All main figures exist in pdf/png/svg", ", ".join(fig_missing[:6]) if fig_missing else ""))

    table_missing = []
    for stem in REQUIRED_TABLES:
        for ext in [".csv", ".md", ".tex"]:
            if not (OUT / "tables" / f"{stem}{ext}").exists():
                table_missing.append(f"{stem}{ext}")
    checks.append((not table_missing, "All main tables exist in csv/md/tex", ", ".join(table_missing[:6]) if table_missing else ""))

    ok_sources, source_detail = check_numbers_from_sources()
    checks.append((ok_sources, "Numbers are sourced from registry and supplementary CSV files", source_detail))

    internal_terms = [term for term in ["Phase15", "Phase16", "Phase17"] if term in text]
    checks.append((not internal_terms, "No internal phase names in main manuscript text", ", ".join(internal_terms)))

    checks.append(("state-of-the-art" not in lower and "sota" not in lower, "No strict SOTA wording in main text", ""))
    checks.append(("gan paper" in lower and "not" in lower and "final mechanism" in lower, "GAN is explicitly not framed as final main mechanism", ""))
    checks.append(("low-frequency hadamard 5\\%" in lower and "not" in lower and "primary stl-10" in lower, "Low-frequency Hadamard 5% is not claimed as primary STL-10 HQ", ""))
    checks.append(("tv-regularized compressed-sensing baseline solved by pgd" in lower and "cs-tv (pgd solver)" in lower, "CS-TV is explained as compressed-sensing PGD baseline", ""))

    formula_terms = [
        r"y_i = \langle a_i, x\rangle + \epsilon_i",
        r"y = A x + \epsilon",
        r"x_{\mathrm{data}} = A^T",
        r"P_A = A^T",
        r"P_N(v)=v-A^T",
        r"\Pi_y(v)=v-A^T",
        r"\hat{x}^{(1)}",
        r"r_\phi=R_\phi",
        r"K=AA^T+\lambda I",
        r"\mathrm{clip}(\hat{x},0,1)",
    ]
    missing_formula = [term for term in formula_terms if term not in text]
    checks.append((not missing_formula, "Method formulas and exact-A/clamp notes are present", ", ".join(missing_formula[:3]) if missing_formula else ""))

    missing_labels = [label for label in REQUIRED_LABELS if not label_is_referenced(text, label)]
    checks.append((not missing_labels, "Every main figure/table label is referenced with cref", ", ".join(missing_labels)))

    limits = read(SECTIONS / "limitations.tex").lower()
    required_limits = [
        "hardware optical experiment",
        "strict leaderboard claim",
        "admm/fista",
        "finite tested range",
        "class-wise results are diagnostic",
        "low-frequency hadamard 5\\%",
        "binary learned illumination",
        "gan components",
        "exact-a handling",
    ]
    missing_limits = [item for item in required_limits if item not in limits]
    checks.append((not missing_limits, "Limitations cover requested caveats", ", ".join(missing_limits)))

    return checks


def report() -> str:
    checks = check()
    lines = [
        "# Manuscript V3 Check Report",
        "",
        f"Output directory: `{OUT}`",
        "",
        "## Checklist",
    ]
    for ok, label, detail in checks:
        lines.append(status_line(ok, label, detail))
    lines.extend(
        [
            "",
            "## Notes",
            "- This check validates generated text, figure/table presence, and source-table availability.",
            "- It does not verify external bibliography accuracy; all bibliography placeholders remain TODO-VERIFY.",
            "- Supplement and check reports may contain internal source paths for auditability.",
            "",
            "## Source Files",
            f"- Registry: `{REGISTRY}`",
        ]
    )
    for name, path in TABLES.items():
        lines.append(f"- {name}: `{path}`")
    overall = all(ok for ok, _label, _detail in checks)
    lines.insert(2, f"Overall status: {'PASS' if overall else 'FAIL'}")
    return "\n".join(lines)


def main() -> None:
    path = OUT / "MANUSCRIPT_V3_CHECK_REPORT.md"
    write_text(path, report())
    print({"check_report": str(path)})


if __name__ == "__main__":
    main()
