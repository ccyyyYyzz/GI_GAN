from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .phase20_common import write_text


OUT = Path("E:/ns_mc_gan_gi/outputs_phase23_top_journal_rewrite")
LATEX = OUT / "latex_project_v9"
FIG_DIR = OUT / "figures"

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

MAIN_FIG_LABELS = [
    "fig:concept",
    "fig:primary_metrics",
    "fig:qualitative_reconstruction",
    "fig:regime_map",
    "fig:inference_ablation",
    "fig:validation_summary",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def pdf_source_text() -> str:
    paths = [
        LATEX / "main.tex",
        LATEX / "supplement.tex",
        LATEX / "references.bib",
        LATEX / "supplement" / "supplement.tex",
    ]
    paths.extend(sorted((LATEX / "sections").glob("*.tex")))
    paths.extend(sorted((LATEX / "tables").glob("*.tex")))
    return "\n".join(read(path) for path in paths)


def main_text() -> str:
    paths = [LATEX / "main.tex", LATEX / "references.bib"]
    paths.extend(sorted((LATEX / "sections").glob("*.tex")))
    paths.extend(sorted((LATEX / "tables").glob("table[123]*.tex")))
    return "\n".join(read(path) for path in paths)


def aux_page(label: str) -> int | None:
    aux = read(LATEX / "main.aux")
    m = re.search(r"\\newlabel\{" + re.escape(label) + r"\}\{\{[^}]*\}\{(\d+)\}", aux)
    return int(m.group(1)) if m else None


def cref(text: str, label: str) -> bool:
    return bool(re.search(r"\\[cC]ref\{[^}]*" + re.escape(label) + r"[^}]*\}", text))


def figure_size(name: str) -> tuple[bool, str]:
    path = FIG_DIR / name
    if not path.exists():
        return False, "missing"
    im = Image.open(path)
    return True, f"{im.width}x{im.height}px"


def status(ok: bool, label: str, detail: str = "") -> str:
    return f"- {'PASS' if ok else 'FAIL'}: {label}" + (f" - {detail}" if detail else "")


def checks() -> list[tuple[bool, str, str]]:
    all_text = pdf_source_text()
    mt = main_text()
    lower = all_text.lower()
    rows: list[tuple[bool, str, str]] = []

    rows.append(((OUT / "main_v9.pdf").exists(), "main_v9.pdf exists", ""))
    rows.append(((OUT / "supplement_v9.pdf").exists(), "supplement_v9.pdf exists", ""))

    forbidden = {
        "Reference Placeholders": "Reference Placeholders",
        "TODO-VERIFY": "TODO-VERIFY",
        "TODO VERIFY": "TODO VERIFY",
        "E:/": "Windows path E:/",
        "E:\\": "Windows path E:\\",
        "C:\\": "Windows path C:\\",
        "Colab results": "Colab results",
        "leaderboard superiority": "leaderboard superiority",
        "GAN main mechanism": "GAN main mechanism",
        "lowfreq Hadamard 5% HQ": "lowfreq Hadamard 5% HQ",
    }
    found = [label for token, label in forbidden.items() if token in all_text]
    rows.append((not found, "No banned phrases in PDF sources", ", ".join(found)))
    phase_terms = sorted(set(re.findall(r"Phase1[5-9]|Phase20|Phase21|Phase22", all_text)))
    rows.append((not phase_terms, "No internal phase terms in PDF sources", ", ".join(phase_terms)))
    rows.append(("colab" not in lower, "No platform wording in PDF sources", ""))

    missing_numbers = [s for s in MAIN_RESULT_STRINGS if s not in mt]
    rows.append((not missing_numbers, "Main result numbers unchanged", ", ".join(missing_numbers)))

    required_intro = (
        "The central issue is not only whether a neural network can improve image quality, "
        "but whether it can do so without losing contact with the measurements."
    )
    rows.append((required_intro in mt, "Introduction contains required physical-conflict paragraph", ""))
    rows.append((
        "How can a neural reconstructor add missing structure while remaining auditable against the bucket measurements?" in mt,
        "Introduction poses the core auditable-reconstruction question",
        "",
    ))
    rows.append((
        "measurement-constrained null-space completion" in lower
        and "measured row-space information" in lower
        and "neural null-space completion" in lower
        and "measurement audit" in lower,
        "Clear physical contradiction and answer narrative",
        "",
    ))

    method_terms = [
        "What is measured: data solution",
        "What is missing: null-space neural residual",
        "What must be checked: measurement-consistency projection",
        r"x_{\rm data}",
        r"P_N",
        r"\Pi_y",
        r"\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta",
    ]
    missing_method = [term for term in method_terms if term not in mt]
    rows.append((not missing_method, "Three-step method and required formulas present", ", ".join(missing_method)))

    result_terms = [
        "We organize the results around three questions.",
        "Can STL-10 be reconstructed at 5\\% sampling?",
        "What changes at 10\\% sampling?",
        "What do the reconstructions look like?",
        "Where does the gain come from?",
        "Simple-domain sanity checks",
    ]
    missing_results = [term for term in result_terms if term not in mt]
    rows.append((not missing_results, "Results are question-driven", ", ".join(missing_results)))

    ablation_terms = [
        "Each validation experiment is designed to answer a specific failure mode.",
        "Is random sensing reproducible? Exact-A audit.",
        "Does the measurement projection matter? No-DC ablation.",
        "Does the model depend on y? Perturbation.",
        "Is this stronger than a CS-TV compressed-sensing baseline?",
        "How stable are results across samples and classes?",
    ]
    missing_ablation = [term for term in ablation_terms if term not in mt]
    rows.append((not missing_ablation, "Validation experiments answer failure modes", ", ".join(missing_ablation)))

    discussion_terms = [
        "Low-sampling GI is measurement-constrained completion.",
        "Measurement families define different initialization-gain regimes.",
        "Physical audit matters more than adversarial realism.",
        "The learned reconstructor is useful not because it replaces the physics, but because it supplies a prior for the part of the image that the physics has not measured.",
    ]
    missing_discussion = [term for term in discussion_terms if term not in mt]
    rows.append((not missing_discussion, "Discussion contains the three required lessons", ", ".join(missing_discussion)))

    ok, detail = figure_size("fig1_concept_v9.png")
    fig1_source = read(LATEX / "sections" / "method.tex")
    fig1_words = all(term in fig1_source for term in ["measurement-constrained null-space completion", "neural prior", "measurement audit", r"\mathcal{C}_y"])
    rows.append((ok and fig1_words, "Figure 1 explains the principle", detail))

    ok, detail = figure_size("fig3_qualitative_reconstruction_v9.png")
    qsel = read(OUT / "qualitative_selection_warning.md")
    rows.append((ok and "row 5" in qsel and "row 2" in qsel and "row 3" in qsel and "row 4" in qsel, "Qualitative Figure 3 uses clearer selected samples", detail))

    ok, detail = figure_size("fig4_regime_map_v9.png")
    fig4_caption = read(LATEX / "sections" / "results.tex")
    fig4_words = all(term in fig4_caption for term in ["Backprojection PSNR", "Neural gain", "Rad-5", "Scr-5", "Lowfreq-5", "Lowfreq-10"])
    rows.append((ok and fig4_words, "Figure 4 is the requested measurement-family regime map", detail))

    ok, detail = figure_size("fig6_robustness_baselines_v9.png")
    fig6_caption = read(LATEX / "sections" / "validation_ablation.tex")
    fig6_words = "Rad-5/Scr-5 Shuffle/Wrong-y" in fig6_caption and "STL-10 comparison against CS-TV" in fig6_caption and "do not imply universal robustness" in fig6_caption
    rows.append((ok and fig6_words, "Figure 6 robustness panels are narrowed and captioned", detail))

    missing_fig_refs = [label for label in MAIN_FIG_LABELS if not cref(mt, label)]
    rows.append((not missing_fig_refs, "All main figures are cited", ", ".join(missing_fig_refs)))

    ref_page = aux_page("sec:references")
    fig_pages = {label: aux_page(label) for label in MAIN_FIG_LABELS}
    if ref_page is None or any(page is None for page in fig_pages.values()):
        ok_pages = "\\FloatBarrier" in read(LATEX / "main.tex")
        detail = f"aux incomplete; barrier source check={ok_pages}"
    else:
        ok_pages = all(page < ref_page for page in fig_pages.values() if page is not None)
        detail = f"figure pages={fig_pages}, references page={ref_page}"
    rows.append((ok_pages, "Main figures appear before References", detail))

    rows.append(((OUT / "citations_to_verify.md").exists() and "This file is intentionally separate" in read(OUT / "citations_to_verify.md"), "Citation verification kept outside PDFs", ""))
    no_training_claim = "training was launched" not in lower and (
        "additional training" not in lower or "do not introduce additional training" in lower
    )
    rows.append(("new experiment was run" not in lower and no_training_claim, "No new experiments or training claimed", ""))
    return rows


def report() -> str:
    rows = checks()
    ok = all(row[0] for row in rows)
    lines = [
        "# Top-Journal Narrative Check",
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
            "- Check the final PDF visually in the target journal template before submission.",
            "- Verify bibliography metadata and DOI/page fields against publisher records.",
            "- The qualitative figure uses saved evaluation grids; no new evaluation was run.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    path = OUT / "TOP_JOURNAL_NARRATIVE_CHECK.md"
    write_text(path, report())
    print({"report": str(path)})


if __name__ == "__main__":
    main()
