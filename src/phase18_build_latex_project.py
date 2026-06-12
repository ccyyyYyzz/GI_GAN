from __future__ import annotations

import shutil
from pathlib import Path

from .phase18_rewrite_common import OUT, PHASE15, PHASE16, TITLE, tex_escape_text, write_text
from .phase18_write_manuscript_v3 import tex_sections


LATEX = OUT / "latex_project"
SECTIONS = LATEX / "sections"
SUPPLEMENT = LATEX / "supplement"
TABLE_SRC = OUT / "tables"
TABLE_DST = LATEX / "tables"
FIG_SRC = OUT / "figures"
FIG_DST = LATEX / "figures"


SECTION_ORDER = [
    ("abstract", None),
    ("introduction", "Introduction"),
    ("related_work", "Related Work"),
    ("problem_formulation", "Problem Formulation"),
    ("method", "Method"),
    ("measurement_families", "Measurement Families"),
    ("training_losses", "Training Losses"),
    ("experimental_protocol", "Experimental Protocol"),
    ("results", "Results"),
    ("ablation_validation", "Ablation and Validation"),
    ("discussion", "Discussion"),
    ("limitations", "Limitations"),
    ("conclusion", "Conclusion"),
]


def copy_dir_files(src: Path, dst: Path, suffixes: tuple[str, ...]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    for path in src.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes:
            shutil.copy2(path, dst / path.name)


def write_sections() -> None:
    sections = tex_sections()
    SECTIONS.mkdir(parents=True, exist_ok=True)
    for key, title in SECTION_ORDER:
        text = sections[key].strip()
        if title is None:
            content = text
        else:
            content = rf"\section{{{title}}}" + "\n" + text
        write_text(SECTIONS / f"{key}.tex", content)


def main_tex() -> str:
    section_inputs = "\n".join(
        rf"\begin{{abstract}}\input{{sections/abstract.tex}}\end{{abstract}}"
        if key == "abstract"
        else rf"\input{{sections/{key}.tex}}"
        for key, _title in SECTION_ORDER
    )
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.74in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{subcaption}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{siunitx}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{cleveref}}
\usepackage{{url}}
\hypersetup{{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}}
\title{{{tex_escape_text(TITLE)}}}
\author{{Anonymous authors}}
\date{{}}

\begin{{document}}
\maketitle

{section_inputs}

\section*{{Reference Placeholders}}
The bibliography file \texttt{{references.bib}} contains TODO-VERIFY placeholders. Replace them with verified references before submission.

\appendix
\input{{supplement/supplement.tex}}

\end{{document}}
"""


def root_tex() -> str:
    section_inputs = "\n".join(
        rf"\begin{{abstract}}\input{{latex_project/sections/abstract.tex}}\end{{abstract}}"
        if key == "abstract"
        else rf"\input{{latex_project/sections/{key}.tex}}"
        for key, _title in SECTION_ORDER
    )
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.74in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{subcaption}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{siunitx}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{cleveref}}
\usepackage{{url}}
\hypersetup{{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}}
\title{{{tex_escape_text(TITLE)}}}
\author{{Anonymous authors}}
\date{{}}
\begin{{document}}
\maketitle
{section_inputs}
\section*{{Reference Placeholders}}
The bibliography file \path{{latex_project/references.bib}} contains TODO-VERIFY placeholders. Replace them with verified references before submission.
\appendix
\input{{latex_project/supplement/supplement.tex}}
\end{{document}}
"""


def supplement_tex() -> str:
    return rf"""\section{{Supplementary Material}}

\subsection{{Audit Sources}}
Primary registry: \path{{{PHASE15 / 'noleak_registry.csv'}}}.
Supplementary experiment directory: \path{{{PHASE16}}}.
Internal source paths are listed here for auditability and are intentionally kept out of the main text.

\subsection{{Exact-A Reproducibility}}
\input{{tables/supp_exact_a_reproducibility.tex}}

\subsection{{Noise Sweep}}
\input{{tables/supp_noise_sweep.tex}}

\subsection{{CS-TV Baseline}}
The CS-TV (PGD solver) baseline is a TV-regularized compressed-sensing baseline solved with projected gradient descent:
\begin{{equation}}
\min_x \frac{{1}}{{2}}\|Ax-y\|_2^2+\lambda\,\mathrm{{TV}}(x).
\end{{equation}}
It is a lightweight small-subset comparison and should not be described as an exhaustive ADMM/FISTA or plug-and-play benchmark.
\input{{tables/supp_cs_tv_baseline.tex}}

\subsection{{DC Row Control}}
\input{{tables/supp_dc_row_control.tex}}
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.95\textwidth]{{figures/figS1_dc_row_control.pdf}}
\caption{{Supplementary DC-row control for low-frequency Hadamard backprojection.}}
\label{{fig:supp_dc_row}}
\end{{figure*}}

\subsection{{Statistical and Class-wise Diagnostics}}
\input{{tables/supp_bootstrap_ci.tex}}
\input{{tables/supp_classwise.tex}}
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.95\textwidth]{{figures/figS2_classwise.pdf}}
\caption{{Supplementary STL-10 class-wise diagnostic.}}
\label{{fig:supp_classwise}}
\end{{figure*}}
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.95\textwidth]{{figures/figS4_histograms.pdf}}
\caption{{Supplementary bootstrap histogram diagnostic.}}
\label{{fig:supp_histograms}}
\end{{figure*}}

\subsection{{Runtime}}
\input{{tables/supp_runtime.tex}}
\begin{{figure*}}[t]
\centering
\includegraphics[width=0.8\textwidth]{{figures/figS3_runtime.pdf}}
\caption{{Supplementary learned-inference runtime diagnostic.}}
\label{{fig:supp_runtime}}
\end{{figure*}}
"""


def references_bib() -> str:
    return r"""@article{TODO_VERIFY_deep_gi_review,
  title = {TODO_VERIFY: Deep learning methods for ghost imaging and single-pixel imaging},
  author = {TODO_VERIFY},
  journal = {TODO_VERIFY},
  year = {TODO_VERIFY}
}

@article{TODO_VERIFY_null_space_learning,
  title = {TODO_VERIFY: Null-space learning and data-consistent neural inverse problems},
  author = {TODO_VERIFY},
  journal = {TODO_VERIFY},
  year = {TODO_VERIFY}
}

@article{TODO_VERIFY_tv_compressed_sensing,
  title = {TODO_VERIFY: Total variation regularized compressed sensing reconstruction},
  author = {TODO_VERIFY},
  journal = {TODO_VERIFY},
  year = {TODO_VERIFY}
}
"""


def main() -> None:
    LATEX.mkdir(parents=True, exist_ok=True)
    write_sections()
    copy_dir_files(TABLE_SRC, TABLE_DST, (".tex",))
    copy_dir_files(FIG_SRC, FIG_DST, (".pdf", ".png", ".svg"))
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    write_text(SUPPLEMENT / "supplement.tex", supplement_tex())
    write_text(LATEX / "main.tex", main_tex())
    write_text(LATEX / "references.bib", references_bib())
    write_text(OUT / "manuscript_v3.tex", root_tex())
    print({"latex_project": str(LATEX), "main_tex": str(LATEX / "main.tex"), "root_tex": str(OUT / "manuscript_v3.tex")})


if __name__ == "__main__":
    main()
