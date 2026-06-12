from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .phase20_common import write_text


OUT = Path("E:/ns_mc_gan_gi/outputs_phase21_submission_polish")
LATEX = OUT / "latex_project_v7"
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


def checks() -> list[tuple[bool, str, str]]:
    mt = main_text()
    st = supplement_text()
    all_text = mt + "\n" + st
    lower = all_text.lower()
    rows: list[tuple[bool, str, str]] = []

    rows.append(("Reference Placeholders" not in all_text, "Reference Placeholders removed", ""))
    todo_in_pdf_sources = re.findall(r"TODO[_A-Za-z0-9-]*|TODO VERIFY", all_text)
    rows.append((not todo_in_pdf_sources, "TODO citations absent from PDF sources", ", ".join(sorted(set(todo_in_pdf_sources))[:6])))
    rows.append(("E:/" not in mt and "E:\\" not in mt and "C:\\" not in mt, "Main text has no Windows path", ""))
    rows.append(("E:/" not in st and "E:\\" not in st and "C:\\" not in st, "Supplement has no Windows path", ""))
    rows.append((not re.search(r"Phase1[5-9]|Phase20|Phase21", all_text), "No internal phase names in main/supplement", ""))

    missing_fig_refs = [label for label in MAIN_FIG_LABELS if not cref(mt, label)]
    missing_tab_refs = [label for label in MAIN_TABLE_LABELS if not cref(mt, label)]
    rows.append((not missing_fig_refs, "All main figures are cited", ", ".join(missing_fig_refs)))
    rows.append((not missing_tab_refs, "All main tables are cited", ", ".join(missing_tab_refs)))

    aux = read(LATEX / "main.aux")
    ref_page = aux_page("sec:references", aux)
    fig_pages = {label: aux_page(label, aux) for label in MAIN_FIG_LABELS}
    if ref_page is None or any(page is None for page in fig_pages.values()):
        ok = "\\FloatBarrier" in read(LATEX / "main.tex") and "\\bibliography" in read(LATEX / "main.tex")
        detail = f"aux pages unavailable; barrier source check={ok}"
    else:
        ok = all(page < ref_page for page in fig_pages.values() if page is not None)
        detail = f"figure pages={fig_pages}, references page={ref_page}"
    rows.append((ok, "No main figure appears after References", detail))

    q = FIG_DIR / "fig3_qualitative_reconstruction.png"
    if q.exists():
        im = Image.open(q)
        rows.append((im.width >= 1800 and im.height >= 1500, "Figure 3 qualitative is large enough", f"{im.width}x{im.height}px"))
    else:
        rows.append((False, "Figure 3 qualitative is large enough", "missing"))

    f6 = FIG_DIR / "fig6_validation_summary.png"
    if f6.exists():
        im = Image.open(f6)
        rows.append((im.width >= 1800 and im.height >= 1300, "Figure 6 is not overcrowded-sized", f"{im.width}x{im.height}px"))
    else:
        rows.append((False, "Figure 6 is not overcrowded-sized", "missing"))

    dc = read(LATEX / "tables" / "tableS4_dc_row_control.tex")
    expected = ["21.440", "0.612", "8.409", "0.152", "19.030", "0.482", "8.069", "0.118"]
    rows.append((all(x in dc for x in expected), "DC row table is complete", ", ".join(x for x in expected if x not in dc)))
    rows.append(("tv-regularized compressed-sensing baseline solved by projected gradient descent" in lower and "not an exhaustively optimized" in lower, "CS-TV explained as compressed-sensing baseline", ""))
    rows.append(("strict state-of-the-art ranking" in lower and "does not claim" in lower and "state-of-the-art performance" not in lower, "No SOTA claim", ""))
    rows.append(("gan main" not in lower and "gan is" not in lower and "adversarial generation is the final" not in lower, "No GAN main mechanism claim", ""))
    rows.append(("low-frequency hadamard at 5\\% is not the primary" in lower and "low-frequency hadamard at 5\\% is not a high-quality stl-10 setting" in lower, "No lowfreq Hadamard 5% HQ claim", ""))
    rows.append(("binary learned illumination is not claimed as successful" in lower, "No binary learned illumination claim", ""))
    rows.append(((LATEX / "main.pdf").exists() and (OUT / "main_v7.pdf").exists(), "Main PDF compiled", ""))
    rows.append(((LATEX / "supplement.pdf").exists() and (OUT / "supplement_v7.pdf").exists(), "Supplement PDF compiled", ""))
    return rows


def report() -> str:
    rows = checks()
    ok = all(row[0] for row in rows)
    lines = [
        "# Submission Polish Check Report",
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
            "- Verify DOI and exact page metadata for all references before journal submission.",
            "- Apply the target journal class/template if required.",
            "- Hardware validation and broader external baselines remain future work, not claims in this manuscript.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    path = OUT / "SUBMISSION_POLISH_CHECK_REPORT.md"
    write_text(path, report())
    print({"report": str(path)})


if __name__ == "__main__":
    main()
