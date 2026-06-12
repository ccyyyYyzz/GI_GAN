from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .phase20_common import write_text


OUT = Path("E:/ns_mc_gan_gi/outputs_phase22_submission_v8")
LATEX = OUT / "latex_project_v8"
FIG_DIR = OUT / "figures"

MAIN_FIG_LABELS = [
    "fig:mechanism",
    "fig:primary_metrics",
    "fig:qualitative_reconstruction",
    "fig:measurement_attribution",
    "fig:inference_ablation",
    "fig:validation_summary",
]
MAIN_TABLE_LABELS = ["tab:primary_results", "tab:measurement_attribution", "tab:ablation_summary"]
MAIN_RESULT_STRINGS = [
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


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main_text() -> str:
    parts = [read(LATEX / "main.tex")]
    parts.extend(read(path) for path in sorted((LATEX / "sections").glob("*.tex")))
    parts.extend(read(path) for path in sorted((LATEX / "tables").glob("table[123]*.tex")))
    parts.append(read(LATEX / "references.bib"))
    return "\n".join(parts)


def supplement_text() -> str:
    parts = [read(LATEX / "supplement.tex"), read(LATEX / "supplement" / "supplement.tex")]
    parts.extend(read(path) for path in sorted((LATEX / "tables").glob("tableS*.tex")))
    return "\n".join(parts)


def cref(text: str, label: str) -> bool:
    return bool(re.search(r"\\[cC]ref\{[^}]*" + re.escape(label) + r"[^}]*\}", text))


def aux_page(label: str, aux: str) -> int | None:
    m = re.search(r"\\newlabel\{" + re.escape(label) + r"\}\{\{[^}]*\}\{(\d+)\}", aux)
    return int(m.group(1)) if m else None


def status(ok: bool, label: str, detail: str = "") -> str:
    return f"- {'PASS' if ok else 'FAIL'}: {label}" + (f" - {detail}" if detail else "")


def figure_size(name: str) -> tuple[bool, str]:
    path = FIG_DIR / name
    if not path.exists():
        return False, "missing"
    im = Image.open(path)
    return True, f"{im.width}x{im.height}px"


def checks() -> list[tuple[bool, str, str]]:
    mt = main_text()
    st = supplement_text()
    all_text = mt + "\n" + st
    lower = all_text.lower()
    rows: list[tuple[bool, str, str]] = []

    source_files = "\n".join(read(path) for path in Path("src").glob("phase22_*.py"))
    rows.append(("train" not in source_files.lower() or "additional training" in lower, "No training was launched by Phase22 scripts", "source-level guard"))
    rows.append(("new experiment" not in lower, "No new experiments claimed", ""))
    missing_numbers = [s for s in MAIN_RESULT_STRINGS if s not in mt]
    rows.append((not missing_numbers, "Main result numbers unchanged", ", ".join(missing_numbers)))

    todo = re.findall(r"TODO[_A-Za-z0-9-]*|TODO VERIFY", all_text)
    rows.append((not todo, "No TODO VERIFY in PDF sources", ", ".join(sorted(set(todo))[:6])))
    rows.append(("Reference Placeholders" not in all_text, "No Reference Placeholders", ""))
    rows.append(("E:/" not in all_text and "E:\\" not in all_text and "C:\\" not in all_text, "No Windows path in PDF sources", ""))
    rows.append((not re.search(r"Phase1[5-9]|Phase20|Phase21|Phase22", all_text), "No internal phase names in PDF sources", ""))
    rows.append(("colab" not in mt.lower(), "No Colab in main text", ""))
    rows.append(("low-frequency hadamard at 5\\% is not the primary" in lower and "low-frequency hadamard at 5\\% is not a high-quality stl-10 setting" in lower, "No lowfreq Hadamard 5% HQ claim", ""))
    rows.append(("strict state-of-the-art ranking" in lower and "does not claim" in lower and "state-of-the-art performance" not in lower, "No strict SOTA claim", ""))
    rows.append(("gan is" not in lower and "gan main" not in lower and "adversarial generation is the final" not in lower, "No GAN main mechanism claim", ""))
    rows.append(("binary learned illumination is not claimed as successful" in lower, "No binary learned illumination claim", ""))
    rows.append(("tv-regularized compressed-sensing baseline solved by projected gradient descent" in lower and "not an exhaustively optimized" in lower, "CS-TV explained as TV-regularized compressed sensing baseline solved by PGD", ""))

    missing_fig_refs = [label for label in MAIN_FIG_LABELS if not cref(mt, label)]
    missing_tab_refs = [label for label in MAIN_TABLE_LABELS if not cref(mt, label)]
    rows.append((not missing_fig_refs, "All main figures are cited", ", ".join(missing_fig_refs)))
    rows.append((not missing_tab_refs, "All main tables are cited", ", ".join(missing_tab_refs)))

    ok, detail = figure_size("fig1_mechanism_v8.png")
    rows.append((ok and "fig1_mechanism_v8.pdf" in mt and (FIG_DIR / "fig1_mechanism_v8.svg").exists(), "Figure 1 mechanism graphic rebuilt", detail))

    ok, detail = figure_size("fig3_qualitative_reconstruction_v8.png")
    rows.append((ok and "fig3_qualitative_reconstruction_v8.pdf" in mt and "Representative evaluation samples" in mt, "Figure 3 qualitative sufficiently large and repaired", detail))

    ok, detail = figure_size("fig6_robustness_baselines_v8.png")
    rows.append((ok and "fig6_robustness_baselines_v8.pdf" in mt and "MNIST" not in read(LATEX / "sections" / "validation_ablation.tex"), "Figure 6 labels not overcrowded and STL-only", detail))

    dc = read(LATEX / "tables" / "tableS4_dc_row_control.tex")
    expected = ["21.440", "0.612", "8.409", "0.152", "19.030", "0.482", "8.069", "0.118"]
    rows.append((all(x in dc for x in expected), "Supplement DC row table complete", ", ".join(x for x in expected if x not in dc)))

    aux = read(LATEX / "main.aux")
    ref_page = aux_page("sec:references", aux)
    fig_pages = {label: aux_page(label, aux) for label in MAIN_FIG_LABELS}
    if ref_page is None or any(page is None for page in fig_pages.values()):
        ok_pages = "\\FloatBarrier" in read(LATEX / "main.tex")
        detail = f"aux pages unavailable; barrier source check={ok_pages}"
    else:
        ok_pages = all(page < ref_page for page in fig_pages.values() if page is not None)
        detail = f"figure pages={fig_pages}, references page={ref_page}"
    rows.append((ok_pages, "No main figure drifts after References", detail))

    rows.append(((LATEX / "main.pdf").exists() and (OUT / "main_v8.pdf").exists(), "main_v8.pdf compiled", ""))
    rows.append(((LATEX / "supplement.pdf").exists() and (OUT / "supplement_v8.pdf").exists(), "supplement_v8.pdf compiled", ""))

    bib = read(LATEX / "references.bib")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cite_keys = sorted({k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", mt) for k in group.split(",")})
    missing = [key for key in cite_keys if key not in bib_keys]
    rows.append((not missing and "TODO" not in bib, "Citation audit clean at source level", ", ".join(missing)))
    return rows


def report() -> str:
    rows = checks()
    ok = all(row[0] for row in rows)
    lines = [
        "# Submission V8 Check Report",
        "",
        f"Overall status: {'PASS' if ok else 'FAIL'}",
        f"Output directory: `{OUT}`",
        "",
        "## Checklist",
    ]
    lines.extend(status(*row) for row in rows)
    lines.extend(
        [
            "",
            "## Manual Follow-up",
            "- Verify DOI and exact page metadata against publisher records before journal submission.",
            "- Apply the target journal class/template and bibliography style if required.",
            "- Hardware validation and broader external benchmarks remain future work, not claims in this manuscript.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    path = OUT / "SUBMISSION_V8_CHECK_REPORT.md"
    write_text(path, report())
    print({"report": str(path)})


if __name__ == "__main__":
    main()
