from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase34_mechanism_teaser"
SOURCE_PROJECT = ROOT / "outputs_phase33_mechanism_overhaul" / "latex_project_mechanism_v33"
PROJECT = OUT / "latex_project_mechanism_v34"
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


METHOD_PARAGRAPH = r"""Although \(P_N\) restricts the neural residual, it is only an approximate null-space projection when \(\lambda>0\). In addition, the data solution, the refiner, numerical factorization, and intensity clipping can all introduce measurement inconsistency. The final projection \(\Pi_y\) therefore acts as an audit step on the complete image rather than only on the neural residual. Figure 1 presents the mechanism as a problem--failure--solution sequence. Low-sampling bucket measurements define an ambiguous inverse problem. A purely physical inverse is auditable but incomplete, whereas an unconstrained neural inverse may produce plausible structure that is not supported by the measurements. The proposed reconstruction resolves this tension by using \(x_{\rm data}\) for the measured component, \(P_N(G_\theta)\) for learned missing structure, and \(\Pi_y\) as a final bucket-measurement audit. This makes the network a measurement-constrained completion module rather than a direct image-to-image inverse."""


METHOD_FIGURE = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_mechanism_teaser_v34.pdf}
\caption{\textbf{Conceptual mechanism.} Low-sampling bucket measurements are insufficient to determine a unique image. A physical inverse remains tied to the measurements but is incomplete, whereas an unconstrained neural inverse may drift away from the bucket signal. The proposed reconstruction computes a data solution, filters the learned residual through an approximate null-space operator, and finally audits the completed image with a measurement-consistency projection.}
\label{fig:mechanism}
\end{figure*}
"""


SUPPLEMENT_PREFIX = r"""\section{Supplementary Material}
The supplement provides compact curated summaries for reproducibility and diagnostic interpretation. Complete CSV tables are described in the data and code availability statement.

\subsection{S1 Algebraic reconstruction mechanism}
The main text uses a conceptual mechanism teaser. The algebraic decomposition is collected here for readers who want the reconstruction operators in one place.

\begin{figure*}[h]
\centering
\includegraphics[width=\textwidth]{figures/figS1_equation_decomposition_v34.pdf}
\caption{Algebraic decomposition of the measurement-consistent null-space reconstruction.}
\label{fig:supp_mechanism_equations}
\end{figure*}

\subsection{S2 Exact-operator reproducibility}
"""


VALIDATION_CSTV = r"""
\subsection{CSGI-style CS-TV(PGD) compressed-sensing baseline}
We compare against a TV-regularized compressed-sensing baseline solved by projected gradient descent (CS-TV):
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\operatorname{TV}(x).
\end{equation}
This baseline represents a classical compressed-sensing prior, not an exhaustively tuned iterative reconstruction benchmark. GI/BP denotes the linear physical backprojection or correlation-like GI reconstruction. A selected visual comparison is provided in the Supplement.
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


def replace_first(pattern: str, replacement: str, text: str, label: str) -> str:
    new = re.sub(pattern, lambda _m: replacement.strip() + "\n", text, count=1, flags=re.DOTALL)
    if new == text:
        raise RuntimeError(f"Could not replace {label}")
    return new


def update_method() -> None:
    path = PROJECT / "sections" / "method.tex"
    text = path.read_text(encoding="utf-8")
    text = replace_first(
        r"Although \\\(P_N\\\) restricts the neural residual.*?original bucket vector\.",
        METHOD_PARAGRAPH,
        text,
        "method mechanism paragraph",
    )
    text = replace_first(
        r"\\begin\{figure\*\}\[t\].*?\\label\{fig:mechanism\}\s*\\end\{figure\*\}",
        METHOD_FIGURE,
        text,
        "Figure 1 block",
    )
    write_text(path, text)


def update_results() -> None:
    path = PROJECT / "sections" / "results.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "figures/fig4_measurement_attribution_v33.pdf",
        "figures/fig4_measurement_attribution_v34.pdf",
    )
    text = text.replace(
        r"\caption{\textbf{Measurement attribution.} Final PSNR alone hides whether performance comes from physical initialization, neural refinement, or both. Low-frequency Hadamard points are shown as hollow diagnostic controls and are not primary STL-10 claims.}",
        r"\caption{\textbf{Measurement attribution.} The regime map uses backprojection PSNR and model gain to separate physical-initialization quality from learned refinement. Low-frequency Hadamard points are shown as hollow diagnostic controls and are not primary STL-10 claims.}",
    )
    write_text(path, text)


def update_validation() -> None:
    path = PROJECT / "sections" / "validation_ablation.tex"
    text = path.read_text(encoding="utf-8")
    text = replace_first(
        r"\\subsection\{CSGI-style CS-TV\(PGD\) compressed-sensing baseline\}.*?(?=\\begin\{figure\*\}\[t\]\s*\\centering\s*\\includegraphics\[width=\\textwidth\]\{figures/fig6_robustness_baselines_submission\.pdf\})",
        VALIDATION_CSTV,
        text,
        "CS-TV paragraph",
    )
    text = text.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
    write_text(path, text)


def update_supplement() -> None:
    path = PROJECT / "supplement" / "supplement.tex"
    text = path.read_text(encoding="utf-8")
    text = replace_first(
        r"\\section\{Supplementary Material\}.*?\\subsection\{S2 Exact-operator reproducibility\}",
        SUPPLEMENT_PREFIX,
        text,
        "supplement prefix",
    )
    text = text.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
    text = text.replace(
        r"\caption{Visual comparison with GI/BP and CS-TV(PGD) is provided for reference; quantitative conclusions use the full evaluation tables.}",
        r"\caption{Visual comparison with GI/BP and CS-TV(PGD) is provided for reference; quantitative conclusions use the full evaluation tables.}",
    )
    write_text(path, text)


def update_checklist() -> None:
    path = PROJECT / "submission_checklist.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text += "\n- Figure 1 was rebuilt as a 2x3 problem-failure-solution mechanism teaser.\n"
    text += "- Figure S1 carries the algebraic decomposition so the main mechanism figure stays intuitive.\n"
    write_text(path, text)


def compile_pdf(filename: str) -> None:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", filename],
        cwd=PROJECT,
        check=True,
    )


def copy_outputs() -> None:
    shutil.copy2(PROJECT / "main.pdf", OUT / "main_v34.pdf")
    shutil.copy2(PROJECT / "supplement.pdf", OUT / "supplement_v34.pdf")


def main() -> None:
    copy_project()
    update_method()
    update_results()
    update_validation()
    update_supplement()
    update_checklist()
    compile_pdf("main.tex")
    compile_pdf("supplement.tex")
    copy_outputs()
    print(
        {
            "project": str(PROJECT),
            "main_pdf": str(OUT / "main_v34.pdf"),
            "supplement_pdf": str(OUT / "supplement_v34.pdf"),
        }
    )


if __name__ == "__main__":
    main()
