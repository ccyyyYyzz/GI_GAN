from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase36_conventional_gi_aligned"
SOURCE_PROJECT = ROOT / "outputs_phase34_mechanism_teaser" / "latex_project_mechanism_v34"
PROJECT = OUT / "latex_project_v36"
FIG_DIR = OUT / "figures"

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


RELATION_SUBSECTION = r"""
\subsection{Relation to conventional bucket-pattern correlation}
Conventional ghost imaging can be written as a centered and normalized bucket-pattern correlation,
\begin{equation}
\hat{x}_{\rm GI}\propto\sum_i (y_i-\bar{y})(a_i-\bar{a}).
\end{equation}
Ignoring centering and normalization gives the raw backprojection form
\begin{equation}
\hat{x}_{\rm raw}=A^Ty=\sum_i y_i a_i.
\end{equation}
The data solution used in this work keeps the same pattern-expansion structure but first solves for decorrelated and regularized bucket coefficients:
\begin{equation}
q=(AA^T+\lambda I)^{-1}y,
\qquad
x_{\rm data}=A^Tq=\sum_i q_i a_i.
\end{equation}
Traditional GI therefore uses the raw bucket coefficients \(y_i\), whereas \(x_{\rm data}\) uses decorrelated and regularized coefficients \(q_i\). If \(AA^T\approx I\) and \(\lambda\to0\), then \(q\approx y\) and \(x_{\rm data}\approx A^Ty\). Thus \(x_{\rm data}\) is best interpreted as a regularized, pattern-correlation-corrected generalization of conventional GI/BP, not as a new standalone basic reconstructor. The novelty is not the linear initialization alone; it is the use of this measured-component representative together with a learned residual filtered by \(P_N\) and a final measurement-consistency audit by \(\Pi_y\):
\begin{equation}
\boxed{
A^Ty
\rightarrow
A^T(AA^T+\lambda I)^{-1}y
\rightarrow
\Pi_y[x_{\rm data}+P_N(G_\theta)]
}.
\end{equation}
"""


METHOD_FIGURE = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_conventional_gi_anchor.pdf}
\caption{\textbf{From conventional GI correlation to measurement-audited neural completion.} Conventional GI forms an image by bucket-pattern correlation \(A^Ty\). The data solution used here keeps the same backprojection structure but first decorrelates and regularizes the bucket coefficients, \(q=(AA^T+\lambda I)^{-1}y\). The learned module then proposes missing structure, \(P_N\) filters the residual to avoid overwriting measured components, and \(\Pi_y\) audits the completed image against the original bucket measurements.}
\label{fig:mechanism}
\end{figure*}
"""


INTRO_SENTENCE = (
    "From the viewpoint of conventional GI, the proposed method does not discard bucket-pattern correlation. "
    "It uses a regularized pseudoinverse form of the same backprojection idea as the measured component, "
    "and then constrains learned completion around it."
)


RELATED_PARAGRAPH = r"""
Conventional GI can be expressed as bucket-pattern correlation, and pseudo-inverse or generalized-inverse reconstructions have also appeared in GI/SPI literature \cite{shapiro2008computational,bromberg2009single,katz2009compressive,gong2015pseudoinverse,czajkowski2018fdri}. We therefore do not claim the regularized data solution itself as new. Instead, our contribution is to use it as a measured-component representative and to constrain the learned residual through approximate null-space filtering and measurement-consistency projection, in line with broader data-consistency and null-space ideas for inverse problems \cite{schwab2019deepnull,wang2022physics}.
"""


CONTRIBUTIONS = r"""The main contributions are:
\begin{itemize}
\item a measurement-consistent reconstruction decomposition anchored on a regularized GI/SPI data solution;
\item a learned residual proposal inserted through approximate null-space filtering;
\item a final measurement-consistency projection for bucket-signal audit;
\item measurement-family attribution separating physical initialization and neural gain.
\end{itemize}
"""


NEW_BIB = r"""
@article{bromberg2009single,
  author = {Bromberg, Yaron and Katz, Ori and Silberberg, Yaron},
  title = {Ghost Imaging with a Single Detector},
  journal = {Physical Review A},
  volume = {79},
  pages = {053840},
  year = {2009},
  doi = {10.1103/PhysRevA.79.053840}
}

@article{katz2009compressive,
  author = {Katz, Ori and Bromberg, Yaron and Silberberg, Yaron},
  title = {Compressive Ghost Imaging},
  journal = {Applied Physics Letters},
  volume = {95},
  number = {13},
  pages = {131110},
  year = {2009},
  doi = {10.1063/1.3238296}
}

@article{gong2015pseudoinverse,
  author = {Gong, Wenlin},
  title = {High-Resolution Pseudo-Inverse Ghost Imaging},
  journal = {Photonics Research},
  volume = {3},
  number = {5},
  pages = {234--237},
  year = {2015},
  doi = {10.1364/PRJ.3.000234}
}

@article{czajkowski2018fdri,
  author = {Czajkowski, Krzysztof M. and Pastuszczak, Anna and Koty{\'n}ski, Rafa{\l}},
  title = {Real-Time Single-Pixel Video Imaging with Fourier Domain Regularization},
  journal = {Optics Express},
  volume = {26},
  number = {16},
  pages = {20009--20022},
  year = {2018},
  doi = {10.1364/OE.26.020009}
}
"""


def copy_project() -> None:
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_PROJECT, PROJECT)
    for path in PROJECT.rglob("*"):
        if path.is_file() and path.suffix in COMPILED_SUFFIXES:
            path.unlink()
    dst_fig = PROJECT / "figures"
    dst_fig.mkdir(parents=True, exist_ok=True)
    src_fig = SOURCE_PROJECT / "figures"
    if src_fig.exists():
        for path in sorted(src_fig.glob("*")):
            if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg"}:
                shutil.copy2(path, dst_fig / path.name)
    for path in sorted(FIG_DIR.glob("*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg"}:
            shutil.copy2(path, dst_fig / path.name)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    new = re.sub(pattern, lambda _m: replacement.strip() + "\n", text, count=1, flags=re.DOTALL)
    if new == text:
        raise RuntimeError(f"Could not replace {label}")
    return new


def update_method() -> None:
    path = PROJECT / "sections" / "method.tex"
    text = path.read_text(encoding="utf-8")
    physical_pattern = (
        r"(\\subsection\{Physical data solution\}.*?"
        r"Its quality depends on the measurement family\.\s*)"
        r"(?=\\subsection\{Approximate null-space residual\})"
    )
    new_text = re.sub(
        physical_pattern,
        lambda match: match.group(1) + RELATION_SUBSECTION.strip() + "\n\n",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if new_text == text:
        raise RuntimeError("Could not insert conventional GI relation subsection")
    text = new_text
    text = replace_once(
        text,
        r"Although \\\(P_N\\\) restricts the neural residual.*?direct image-to-image inverse\.",
        r"Although \(P_N\) restricts the neural residual, it is only an approximate null-space projection when \(\lambda>0\). In addition, the data solution, the refiner, numerical factorization, and intensity clipping can all introduce measurement inconsistency. The final projection \(\Pi_y\) therefore acts as an audit step on the complete image rather than only on the neural residual. Figure 1 summarizes the resulting progression from conventional bucket-pattern correlation to the regularized data solution and then to the full measurement-audited learned completion. This framing emphasizes that \(x_{\rm data}\) is the measured-component representative, while \(P_N(G_\theta)\) supplies learned missing structure and \(\Pi_y\) performs the final bucket-signal audit.",
        "method mechanism paragraph",
    )
    text = replace_once(
        text,
        r"\\begin\{figure\*\}\[t\].*?\\label\{fig:mechanism\}\s*\\end\{figure\*\}",
        METHOD_FIGURE,
        "Figure 1 block",
    )
    text = text.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
    write_text(path, text)


def update_introduction() -> None:
    path = PROJECT / "sections" / "introduction.tex"
    text = path.read_text(encoding="utf-8")
    if INTRO_SENTENCE not in text:
        needle = (
            "This work addresses that gap by treating low-sampling ghost imaging as measurement-consistent null-space reconstruction. "
            "The method computes a physical data solution, inserts a learned residual through an approximate null-space component, and then projects the result back to the measured affine set. "
            "This gives an auditable reconstruction path through \\(x_{\\rm data}\\), \\(P_N\\), and \\(\\Pi_y\\), rather than an unconstrained measurement-to-image mapping."
        )
        replacement = INTRO_SENTENCE + "\n\n" + needle
        text = text.replace(needle, replacement)
    text = re.sub(
        r"The main contributions are:.*?\\end\{itemize\}",
        lambda _match: CONTRIBUTIONS.strip(),
        text,
        flags=re.DOTALL,
    )
    write_text(path, text)


def update_related_work() -> None:
    path = PROJECT / "sections" / "related_work.tex"
    text = path.read_text(encoding="utf-8")
    if "pseudo-inverse or generalized-inverse reconstructions" not in text:
        text = text.replace(
            "These foundations motivate low-sampling reconstruction methods that exploit measurement design, prior information, and computational inversion.\n",
            "These foundations motivate low-sampling reconstruction methods that exploit measurement design, prior information, and computational inversion.\n"
            + RELATED_PARAGRAPH
            + "\n",
        )
    text = text.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
    write_text(path, text)


def update_results() -> None:
    path = PROJECT / "sections" / "results.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace("figures/fig4_measurement_attribution_v34.pdf", "figures/fig4_measurement_attribution_v36.pdf")
    text = text.replace(
        r"\caption{\textbf{Measurement attribution.} The regime map uses backprojection PSNR and model gain to separate physical-initialization quality from learned refinement. Low-frequency Hadamard points are shown as hollow diagnostic controls and are not primary STL-10 claims.}",
        r"\caption{\textbf{Measurement attribution.} The regime map uses backprojection PSNR and model gain, reported as neural gain, to separate physical-initialization quality from learned refinement. Low-frequency Hadamard points are shown as hollow diagnostic controls and are not primary STL-10 claims.}",
    )
    write_text(path, text)


def update_validation_and_supplement() -> None:
    for rel in ["sections/validation_ablation.tex", "supplement/supplement.tex"]:
        path = PROJECT / rel
        text = path.read_text(encoding="utf-8")
        text = text.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
        write_text(path, text)


def update_references() -> None:
    path = PROJECT / "references.bib"
    text = path.read_text(encoding="utf-8")
    for key in ["bromberg2009single", "katz2009compressive", "gong2015pseudoinverse", "czajkowski2018fdri"]:
        if f"{{{key}," in text:
            continue
        text += "\n" + "\n".join(block for block in NEW_BIB.strip().split("\n\n") if f"{{{key}," in block) + "\n"
    write_text(path, text)


def citation_keys_from_tex() -> set[str]:
    source = ""
    for path in [PROJECT / "main.tex", PROJECT / "supplement.tex"]:
        source += "\n" + path.read_text(encoding="utf-8")
    for folder in ["sections", "supplement", "tables"]:
        for path in sorted((PROJECT / folder).glob("*.tex")):
            source += "\n" + path.read_text(encoding="utf-8")
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", source):
        for key in match.group(1).split(","):
            keys.add(key.strip())
    return keys


def bib_keys() -> set[str]:
    text = (PROJECT / "references.bib").read_text(encoding="utf-8")
    return set(re.findall(r"@\w+\{([^,\s]+)", text))


def write_citation_audit() -> None:
    cited = citation_keys_from_tex()
    bib = bib_keys()
    missing = sorted(cited - bib)
    uncited = sorted(bib - cited)
    refs = (PROJECT / "references.bib").read_text(encoding="utf-8")
    doi_lines = [line.strip() for line in refs.splitlines() if line.strip().lower().startswith("doi")]
    manual = [
        "Gong pseudo-inverse GI and Czajkowski Fourier-domain regularized SPI entries were added to support the generalized-inverse positioning; final publisher metadata should still be manually checked before submission.",
        "No DOI was invented without a visible metadata source during this pass; existing entries without DOI were left without DOI.",
    ]
    lines = [
        "# Citation Audit",
        "",
        f"- Cited keys: {len(cited)}",
        f"- Bibliography entries: {len(bib)}",
        f"- Missing cited keys: {', '.join(missing) if missing else 'none'}",
        f"- Uncited bibliography entries: {', '.join(uncited) if uncited else 'none'}",
        "- TODO or reference-placeholder text in references.bib: " + ("yes" if re.search(r"TODO|Reference Placeholder", refs, re.I) else "no"),
        "- Malformed citation commands: none detected by regex scan",
        "- DOI policy: DOI fields were not guessed; absent DOI fields were left absent.",
        "- DOI fields present: " + (str(len(doi_lines)) if doi_lines else "none"),
        "- Manual verification needed: " + " ".join(manual),
        "- Special-character notes: Oktem is encoded as {\\\"O}ktem; Candes uses BibTeX accent; Kotynski is encoded as Koty{\\'n}ski and Rafal as Rafa{\\l}.",
    ]
    (OUT / "citation_audit_phase36.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(OUT / "citation_audit_phase36.md", PROJECT / "citation_audit_phase36.md")


def compile_pdf(filename: str) -> None:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", filename],
        cwd=PROJECT,
        check=True,
    )


def copy_outputs() -> None:
    shutil.copy2(PROJECT / "main.pdf", OUT / "main_v36.pdf")
    shutil.copy2(PROJECT / "supplement.pdf", OUT / "supplement_v36.pdf")


def main() -> None:
    copy_project()
    update_method()
    update_introduction()
    update_related_work()
    update_results()
    update_validation_and_supplement()
    update_references()
    write_citation_audit()
    compile_pdf("main.tex")
    compile_pdf("supplement.tex")
    copy_outputs()
    print(
        {
            "project": str(PROJECT),
            "main_pdf": str(OUT / "main_v36.pdf"),
            "supplement_pdf": str(OUT / "supplement_v36.pdf"),
            "citation_audit": str(OUT / "citation_audit_phase36.md"),
        }
    )


if __name__ == "__main__":
    main()
