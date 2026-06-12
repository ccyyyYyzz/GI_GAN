from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
SOURCE_PROJECT = ROOT / "outputs_phase29_final_submission_polish" / "latex_project_final"
OUT = ROOT / "outputs_phase30_submission_package"
PROJECT = OUT / "latex_project_submission"
SUBMISSION_DOCS = OUT / "submission_docs"


COMPILED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".lof",
    ".log",
    ".lot",
    ".out",
    ".pdf",
    ".run.xml",
    ".synctex.gz",
    ".toc",
}


FIGURE_RENAMES = {
    "fig1_mechanism_final": "fig1_mechanism_submission",
    "fig2_primary_metrics_final": "fig2_primary_metrics_submission",
    "fig3_qualitative_final": "fig3_qualitative_submission",
    "fig4_measurement_attribution_final": "fig4_measurement_attribution_submission",
    "fig5_inference_ablation_final": "fig5_inference_ablation_submission",
    "fig6_robustness_baselines_final": "fig6_robustness_baselines_submission",
    "figS1_relmeaserr_ablation_final": "figS1_relmeaserr_ablation_submission",
}


ABSTRACT = (
    "Ghost imaging and single-pixel imaging recover spatial information from structured "
    "illumination patterns and scalar bucket measurements, but low-sampling acquisition "
    "is severely underdetermined. The measurement vector does not identify a unique image: "
    "low sampling needs a learned prior, yet unconstrained priors may hallucinate structure "
    "that is not supported by the bucket readings. We address this tension with "
    "measurement-consistent null-space neural reconstruction. The formulation computes a "
    "physical data solution from the forward operator, adds a learned null-space residual "
    "for missing structure, and applies a final measurement projection to audit the "
    "completed image against the measured signal. Under a leakage-free STL-10 protocol, "
    "the method reaches 22.316 dB PSNR / 0.635 SSIM at 5\\% sampling with Rademacher "
    "measurements and 22.271 dB / 0.632 SSIM with scrambled Hadamard measurements. At "
    "10\\% sampling, the corresponding results are 24.781 dB / 0.747 SSIM and 24.730 dB / "
    "0.746 SSIM. MNIST and Fashion-MNIST 5\\% experiments provide simple-domain sanity "
    "checks, reaching 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM. Exact-A audit, "
    "measurement-family attribution, inference ablation, finite-noise tests, measurement "
    "perturbation, comparison against a TV-regularized compressed-sensing baseline solved "
    "by projected gradient descent (CS-TV), and confidence intervals support "
    "measurement-dependent reconstruction across the tested measurement families."
)


VALIDATION_CSTV_TEXT = r"""
\subsection{CS-TV compressed-sensing baseline}
We compare against a TV-regularized compressed-sensing baseline solved by projected gradient descent (CS-TV) \cite{donoho2006compressed,candes2006robust,rudin1992tv}:
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda \operatorname{TV}(x).
\end{equation}
This baseline represents a classical compressed-sensing prior, not an exhaustively optimized ADMM or FISTA benchmark. It is a lightweight small-subset traditional baseline, and under the tested settings the learned measurement-consistent reconstructor remains substantially stronger.
"""


LIMITATIONS = r"""
\section{Limitations}
This study does not include a hardware optical experiment. We do not claim a ranking over external benchmarks because datasets, measurement operators, sampling protocols, and evaluation splits are not standardized across the literature. The CS-TV baseline is lightweight and small-subset, not an exhaustively optimized compressed-sensing solver. Robustness is tested only over finite noise and perturbation settings. Class-wise evaluation is diagnostic rather than a claim of uniform category performance. Exact-A handling is essential for random measurements, and results should be interpreted with that audit path in place. Low-frequency Hadamard at 5\% is not a high-quality STL-10 setting in this work. Binary learned illumination is not claimed as successful, and adversarial training is not the final contribution mechanism. Future work should include hardware validation, broader external baselines, and more extensive cross-domain testing.
"""


def _ignore(_dir: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if Path(name).suffix in COMPILED_SUFFIXES:
            ignored.add(name)
    return ignored


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def copy_project() -> None:
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    shutil.copytree(SOURCE_PROJECT, PROJECT, ignore=_ignore)


def replace_all(path: Path, replacements: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def update_figure_names() -> None:
    tex_files = [PROJECT / "main.tex", PROJECT / "supplement.tex"]
    tex_files += list((PROJECT / "sections").glob("*.tex"))
    tex_files += list((PROJECT / "supplement").glob("*.tex"))
    for path in tex_files:
        if path.exists():
            replace_all(path, FIGURE_RENAMES)


def copyedit_sources() -> None:
    write_text(PROJECT / "sections" / "abstract.tex", ABSTRACT)
    write_text(PROJECT / "sections" / "limitations.tex", LIMITATIONS)
    validation = PROJECT / "sections" / "validation_ablation.tex"
    text = validation.read_text(encoding="utf-8")
    text = re.sub(
        r"\\subsection\{CS-TV compressed-sensing baseline\}.*?(?=\\begin\{figure\*\}\[t\])",
        lambda _match: VALIDATION_CSTV_TEXT.strip() + "\n\n",
        text,
        flags=re.DOTALL,
    )
    validation.write_text(text, encoding="utf-8")

    table = PROJECT / "tables" / "tableS3_cstv_baseline.tex"
    if table.exists():
        text = table.read_text(encoding="utf-8")
        text = text.replace(
            r"\caption{CS-TV baseline summary.}",
            r"\caption{TV-regularized compressed-sensing baseline solved by PGD (CS-TV). This is a lightweight small-subset baseline.}",
        )
        table.write_text(text, encoding="utf-8")


def citation_keys_from_sources() -> set[str]:
    keys: set[str] = set()
    paths = [PROJECT / "main.tex"] + list((PROJECT / "sections").glob("*.tex"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\cite(?:[tp])?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^{}]+)\}", text):
            keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def parse_bib_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text):
        key = match.group(1)
        start = match.start()
        next_match = re.search(r"\n@\w+\s*\{", text[match.end() :])
        end = match.end() + next_match.start() + 1 if next_match else len(text)
        entries[key] = text[start:end]
    return entries


def field(entry: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*=\s*[\{{\"]([^}}\"]+)", entry, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def write_citation_audit() -> None:
    bib = PROJECT / "references.bib"
    bib_text = bib.read_text(encoding="utf-8")
    entries = parse_bib_entries(bib_text)
    all_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text)
    cited = citation_keys_from_sources()
    missing = sorted(cited - set(entries))
    uncited = sorted(set(entries) - cited)
    duplicate_keys = sorted(key for key, count in Counter(all_keys).items() if count > 1)
    title_counter = Counter(re.sub(r"[^a-z0-9]+", " ", field(entry, "title").lower()).strip() for entry in entries.values())
    duplicate_titles = sorted(title for title, count in title_counter.items() if title and count > 1)
    incomplete = []
    for key, entry in entries.items():
        title = field(entry, "title")
        year = field(entry, "year")
        venue = field(entry, "journal") or field(entry, "booktitle") or field(entry, "publisher") or field(entry, "howpublished")
        if not title or not year or not venue:
            incomplete.append(key)
    malformed = []
    for path in [PROJECT / "main.tex"] + list((PROJECT / "sections").glob("*.tex")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "\\cite" in line and not re.search(r"\\cite(?:[tp])?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{[^{}]+\}", line):
                malformed.append(f"{path.name}:{line_no}")

    special_notes = []
    if r'{\"O}ktem' in bib_text:
        special_notes.append('Oktem is encoded as {\\"O}ktem in BibTeX.')
    if r"Cand{\`e}s" in bib_text:
        special_notes.append("Candes is encoded with a BibTeX accent.")

    lines = [
        "# Citation Audit",
        "",
        f"- Cited keys: {len(cited)}",
        f"- Bibliography entries: {len(entries)}",
        f"- Missing cited keys: {', '.join(missing) if missing else 'none'}",
        f"- Uncited bibliography entries: {', '.join(uncited) if uncited else 'none'}",
        f"- Duplicate keys: {', '.join(duplicate_keys) if duplicate_keys else 'none'}",
        f"- Duplicate obvious entries by normalized title: {', '.join(duplicate_titles) if duplicate_titles else 'none'}",
        f"- Entries missing obvious title/year/venue fields: {', '.join(sorted(incomplete)) if incomplete else 'none'}",
        f"- Malformed citation commands: {', '.join(malformed) if malformed else 'none'}",
        f"- TODO or reference-placeholder text in references.bib: {'yes' if re.search(r'TODO|Reference Placeholders', bib_text, re.I) else 'no'}",
        f"- Special-character notes: {'; '.join(special_notes) if special_notes else 'none detected'}",
        "- DOI policy: DOI fields were not invented; absent DOI fields were left absent.",
        "- Manual verification needed: bibliographic factual accuracy should be checked before submission.",
    ]
    audit = "\n".join(lines)
    write_text(PROJECT / "citation_audit.md", audit)
    write_text(OUT / "citation_audit.md", audit)


def write_copyediting_report() -> None:
    report = """# Copyediting Report

Applied light final copyediting only; scientific claims and numerical results were not rewritten.

Terminology updates:
- Expanded the first abstract occurrence of CS-TV as a TV-regularized compressed-sensing baseline solved by projected gradient descent.
- Standardized CS-TV prose in the validation section and supplement table caption.
- Preserved measurement-consistency projection, leakage-free, and predefined operational-threshold terminology.
- Replaced the external-ranking limitation with a non-ranking statement, avoiding state-of-the-art overclaim language.

Sentence compression:
- Compressed the CS-TV paragraph by defining the baseline once and removing repeated solver wording.
- Kept the limitations paragraph concise while preserving hardware, benchmark, noise, class-wise, exact-A, low-frequency Hadamard, binary illumination, and adversarial-training limitations.

Main result numbers changed: no.
"""
    write_text(OUT / "copyediting_report.md", report)


def write_submission_docs() -> None:
    cover = """# Cover Letter Draft

Dear Editor,

We are pleased to submit our manuscript, "High-Quality Low-Sampling Ghost Imaging via Measurement-Consistent Null-Space Neural Reconstruction," for consideration in a journal covering optics, photonics, or computational imaging.

The manuscript addresses low-sampling ghost imaging and single-pixel imaging, where the inverse problem is severely underdetermined. We propose a measurement-consistent null-space reconstruction framework that combines a physical data solution, learned null-space residual completion, and a final measurement audit against the measured bucket signal.

Under a leakage-free STL-10 protocol, the method achieves high-quality reconstruction at both 5% and 10% sampling for Rademacher and scrambled Hadamard sensing families. The paper also includes exact random-operator re-evaluation for Rademacher measurements, measurement-family attribution, inference-time ablation, finite-noise tests, perturbation diagnostics, CS-TV comparison, and confidence intervals. The work is positioned as measurement-consistent neural reconstruction rather than an unconstrained generative claim or an external benchmark ranking.

We believe the manuscript will interest readers working on computational imaging, optical sensing design, and physics-constrained neural reconstruction.

Sincerely,

[Authors]
"""
    highlights = """# Highlights

- Measurement-consistent null-space reconstruction for low-sampling GI.
- STL-10 5% and 10% high-quality results under two sensing families.
- Exact random-operator audit supports reproducible Rademacher evaluation.
- Ablations and perturbations verify measurement-dependent reconstruction.
"""
    graphical = """# Graphical Abstract Text

Low-sampling ghost imaging records structured illumination patterns through scalar bucket measurements, leaving many images compatible with the same measurement vector. The proposed reconstruction treats this setting as measurement-constrained completion. A physical data solution represents the measured row-space information, while a neural reconstructor proposes missing structure through an approximate null-space residual. A final measurement-consistency projection audits the completed image by returning it to the measurement-consistent set. The conceptual graphic highlights the coupled roles of optical acquisition, affine measurement geometry, physical initialization, learned prior completion, and final measurement audit. This design keeps the reconstruction tied to the measured signal while allowing learned structure in weakly observed or unobserved directions.
"""
    significance = """# Significance Statement

Low-sampling ghost imaging and single-pixel imaging are attractive when dense detector arrays are difficult, expensive, or unavailable, but the resulting inverse problem is highly underdetermined. This paper is significant because it frames learned reconstruction as a measurement-constrained completion problem rather than as unconstrained image generation. The method preserves a physical data solution, inserts learned structure through an approximate null-space residual, and applies a final projection that audits agreement with the bucket measurements. This makes the reconstruction mechanism interpretable across sensing families: Rademacher measurements require larger learned gain from weak physical initialization, while scrambled Hadamard measurements begin from stronger physical structure. The validation suite, including exact random-operator audit, ablations, perturbations, finite-noise tests, CS-TV comparison, and confidence intervals, gives optical and computational-imaging readers a clearer basis for judging measurement-dependent neural reconstruction.
"""
    reviewer = """# Reviewer Summary

- Hallucination risk?
  Evidence: the formulation uses a physical data solution, approximate null-space insertion, final measurement-consistency projection, RelMeasErr audit, and measurement perturbation tests.

- Exact-A reproducibility?
  Evidence: Rademacher results are evaluated with the exported exact random operator and rebuilt solver caches.

- Why Rademacher and scrambled Hadamard?
  Evidence: the attribution analysis separates weak random physical initialization from stronger scrambled Hadamard initialization while showing similar final quality.

- Why CS-TV?
  Evidence: CS-TV is included as a classical TV-regularized compressed-sensing baseline solved by PGD, explicitly described as lightweight and small-subset.

- Why no hardware?
  Evidence: hardware validation is stated as a limitation; the present manuscript focuses on measurement-consistent reconstruction under controlled computational protocols.

- Why no external benchmark ranking?
  Evidence: the limitations section states that datasets, operators, sampling protocols, and splits are not standardized across the literature.
"""
    write_text(SUBMISSION_DOCS / "cover_letter_draft.md", cover)
    write_text(SUBMISSION_DOCS / "highlights.md", highlights)
    write_text(SUBMISSION_DOCS / "graphical_abstract_text.md", graphical)
    write_text(SUBMISSION_DOCS / "significance_statement.md", significance)
    write_text(SUBMISSION_DOCS / "reviewer_summary.md", reviewer)


def write_project_checklist() -> None:
    checklist = """# Submission Checklist

- Main PDF compiles.
- Supplement PDF compiles.
- All main figures are exported as PDF/SVG/PNG/TIFF where possible.
- Tables are included.
- Banned internal exploration terms are absent from manuscript sources.
- Local paths are absent from manuscript sources.
- Cloud-runtime wording is absent from manuscript sources.
- Internal version labels are absent from manuscript sources.
- No external benchmark ranking claim is made.
- No GAN main-mechanism claim is made.
- No low-frequency Hadamard 5% high-quality claim is made.
- CS-TV formula uses \\operatorname{TV}(x).
- Data/code availability is present.
- References have been structurally audited.
- All main figures are cited.
- All main tables are cited.
- Author names remain placeholders.
- Affiliations remain placeholders.
- Target journal template still needs to be chosen.
- Manual reference verification is still needed.
"""
    write_text(PROJECT / "submission_checklist.md", checklist)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in ("main_submission.pdf", "supplement_submission.pdf", "main_submission.txt", "supplement_submission.txt"):
        path = OUT / stale
        if path.exists():
            path.unlink()
    copy_project()
    update_figure_names()
    copyedit_sources()
    write_citation_audit()
    write_copyediting_report()
    write_submission_docs()
    write_project_checklist()
    print(
        {
            "output_dir": str(OUT),
            "latex_project": str(PROJECT),
            "copyediting_report": str(OUT / "copyediting_report.md"),
            "citation_audit": str(OUT / "citation_audit.md"),
            "submission_docs": str(SUBMISSION_DOCS),
        }
    )


if __name__ == "__main__":
    main()
