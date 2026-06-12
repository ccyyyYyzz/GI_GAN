from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
OUT = ROOT / "outputs_phase33_mechanism_overhaul"
SOURCE_PROJECT = ROOT / "outputs_phase32b_algorithm_optics_baselines" / "latex_project_algorithm_optics"
PROJECT = OUT / "latex_project_mechanism_v33"
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


METHOD_PARAGRAPH = r"""Although \(P_N\) restricts the neural residual, it is only an approximate null-space projection when \(\lambda>0\). In addition, the data solution, the refiner, numerical factorization, and intensity clipping can all introduce measurement inconsistency. The final projection \(\Pi_y\) therefore acts as an audit step on the complete image rather than only on the neural residual. The role of the neural network is therefore restricted by the measurement operator: it refines the missing component but is followed by an explicit projection back to the measured affine set. Figure 1 summarizes the mechanism. The left side shows why low-sampling bucket measurements are ambiguous. The middle shows the failure mode of an unconstrained neural inverse: the image may look plausible while drifting away from the measurements. The right side shows the proposed decomposition. \(x_{\rm data}\) carries the measured component, \(P_N(G_\theta)\) supplies learned missing structure, and \(\Pi_y\) audits the completed image against the original bucket vector."""


METHOD_FIGURE = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_mechanism_final_v33.pdf}
\caption{\textbf{Conceptual mechanism of the proposed computational reconstruction.} Low-sampling bucket measurements do not determine a unique image. An unconstrained neural inverse may generate plausible but measurement-inconsistent structure. The proposed method first computes a data solution, inserts a learned residual through an approximate null-space filter, and finally projects the output back to the measured affine set so that the reconstruction remains auditable against the bucket measurements.}
\label{fig:mechanism}
\end{figure*}
"""


VALIDATION_CSTV = r"""
\subsection{CSGI-style CS-TV(PGD) compressed-sensing baseline}
We compare against a TV-regularized compressed-sensing baseline solved by projected gradient descent (CS-TV):
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\operatorname{TV}(x).
\end{equation}
This baseline represents a classical compressed-sensing prior, not an exhaustively tuned iterative reconstruction benchmark. GI/BP denotes the linear physical backprojection or correlation-like GI reconstruction. The Supplement provides a selected-sample GI/BP, CSGI-style CS-TV(PGD), and ours visual comparison for reference; quantitative conclusions use the full evaluation tables.
"""


SUPPLEMENT_PREFIX = r"""\section{Supplementary Material}
The supplement provides compact curated summaries for reproducibility and diagnostic interpretation. Complete CSV tables are described in the data and code availability statement.

\subsection{S1 Algebraic reconstruction mechanism}
The main text uses a conceptual Figure 1. The algebraic form is collected here for readers who want the reconstruction operators in one place.

\begin{figure*}[h]
\centering
\includegraphics[width=\textwidth]{figures/figS_mechanism_equations.pdf}
\caption{Algebraic form of the measurement-consistent null-space reconstruction.}
\label{fig:supp_mechanism_equations}
\end{figure*}

\subsection{S2 Exact-operator reproducibility}
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


def replace_block(text: str, pattern: str, replacement: str) -> str:
    new = re.sub(pattern, lambda _m: replacement.strip() + "\n", text, flags=re.DOTALL)
    if new == text:
        raise RuntimeError(f"Replacement pattern not found: {pattern[:80]}")
    return new


def update_method() -> None:
    path = PROJECT / "sections" / "method.tex"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"Although \\\(P_N\\\) restricts the neural residual.*?\\Cref\{fig:mechanism\} visualizes the same logic\.",
        lambda _m: METHOD_PARAGRAPH,
        text,
        flags=re.DOTALL,
    )
    if "Figure 1 summarizes the mechanism" not in text:
        raise RuntimeError("Method mechanism paragraph was not updated.")
    text = replace_block(
        text,
        r"\\begin\{figure\*\}\[t\].*?\\label\{fig:mechanism\}\s*\\end\{figure\*\}",
        METHOD_FIGURE,
    )
    write_text(path, text)


def update_results() -> None:
    path = PROJECT / "sections" / "results.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        r"\includegraphics[width=\textwidth]{figures/fig4_measurement_attribution_submission.pdf}",
        r"\includegraphics[width=\textwidth]{figures/fig4_measurement_attribution_v33.pdf}",
    )
    text = text.replace(
        r"\caption{\textbf{Measurement attribution.} Final PSNR alone hides whether performance comes from physical initialization, neural refinement, or both. Low-frequency Hadamard points are diagnostic controls rather than primary STL-10 claims.}",
        r"\caption{\textbf{Measurement attribution.} Final PSNR alone hides whether performance comes from physical initialization, neural refinement, or both. Low-frequency Hadamard points are shown as hollow diagnostic controls and are not primary STL-10 claims.}",
    )
    write_text(path, text)


def update_validation() -> None:
    path = PROJECT / "sections" / "validation_ablation.tex"
    text = path.read_text(encoding="utf-8")
    text = replace_block(
        text,
        r"\\subsection\{CSGI-style CS-TV\(PGD\) compressed-sensing baseline\}.*?(?=\\begin\{figure\*\}\[t\]\s*\\centering\s*\\includegraphics\[width=\\textwidth\]\{figures/fig6_robustness_baselines_submission\.pdf\})",
        VALIDATION_CSTV,
    )
    text = text.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
    write_text(path, text)


def update_supplement() -> None:
    path = PROJECT / "supplement" / "supplement.tex"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"\\section\{Supplementary Material\}.*?\\subsection\{S1 Exact-operator reproducibility\}",
        lambda _m: SUPPLEMENT_PREFIX,
        text,
        flags=re.DOTALL,
    )
    text = text.replace(
        r"\caption{Supplementary GI/BP, CSGI-style CS-TV(PGD), and ours visual comparison for STL-10 5\% and 10\% settings. CS-TV(PGD) is a lightweight compressed-sensing control, not an exhaustively optimized iterative benchmark.}",
        r"\caption{Visual comparison with GI/BP and CS-TV(PGD) is provided for reference; quantitative conclusions use the full evaluation tables.}",
    )
    text = text.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
    write_text(path, text)


def update_checklist() -> None:
    checklist = PROJECT / "submission_checklist.md"
    if checklist.exists():
        text = checklist.read_text(encoding="utf-8")
        text += "\n- Figure 1 was rebuilt as a problem-risk-solution mechanism explanation.\n"
        text += "- GI/BP and CS-TV(PGD) visual comparison is kept in the Supplement.\n"
        write_text(checklist, text)


def compile_pdf(name: str) -> None:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", name],
        cwd=PROJECT,
        check=True,
    )


def copy_outputs() -> None:
    shutil.copy2(PROJECT / "main.pdf", OUT / "main_mechanism_v33.pdf")
    shutil.copy2(PROJECT / "supplement.pdf", OUT / "supplement_mechanism_v33.pdf")


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
            "main_pdf": str(OUT / "main_mechanism_v33.pdf"),
            "supplement_pdf": str(OUT / "supplement_mechanism_v33.pdf"),
        }
    )


if __name__ == "__main__":
    main()
