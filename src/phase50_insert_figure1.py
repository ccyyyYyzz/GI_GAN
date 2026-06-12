"""Insert Phase 50 Figure 1 into the latest LaTeX manuscript."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


SOURCE_LATEX = Path("E:/ns_mc_gan_gi/outputs_phase45_math_repro/latex_project_v45")
OUT_DIR = Path("E:/ns_mc_gan_gi/outputs_phase50_final_figure1")
LATEX_DIR = OUT_DIR / "latex_project_v50"
FIG_DIR = OUT_DIR / "figures"


CAPTION = r"""\caption{
\textbf{Operator-centered measurement-audited neural completion.}
(a) Computational GI/SPI measurements are represented by known patterns \(a_i\), a bucket vector \(y\), and the forward model \(y=Ax+\epsilon\).
(b) The reconstruction circuit separates a configured physical anchor from a learned completion. The data anchor is \(x_{\rm data}=D(y)\), where \(D=B_{\lambda_{\rm op}}\) for ridge/Rademacher runs and a zero-filled Hadamard inverse for Hadamard zero-fill configurations. The neural module proposes a residual \(r_\theta=G_\theta(x_{\rm data},z)\), which is admitted through the gate \(P_N^{\lambda}=I-B_{\lambda_{\rm op}}A\). The candidate \(\tilde{x}=x_{\rm data}+r_N\) is then remeasured by \(A\), compared with the original bucket vector \(y\), and corrected as \(\hat{x}=\tilde{x}-B_{\lambda_{\rm op}}(A\tilde{x}-y)\).
(c) Idealized geometry: the measurements fix an affine set \(C_y=\{x:Ax=y\}\), while the learned proposal moves along weakly measured or null-space directions toward natural-image structure. With positive \(\lambda_{\rm op}\), the gate and audit are regularized soft operations rather than exact projections.
}"""


FIGURE_BLOCK = rf"""\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig1_operator_circuit_final.pdf}}
{CAPTION}
\label{{fig:operator_circuit}}
\end{{figure*}}
"""


METHOD_SENTENCE = (
    r"Figure~\ref{fig:operator_circuit} summarizes the operator-centered circuit. "
    r"The network is not a free reconstructor: it proposes missing structure, while "
    r"the calibrated measurement operator gates and audits what can remain."
)


ANCHOR_SENTENCE = (
    r"We write the data anchor as \(D(y)\) because the implemented anchor is configurable "
    r"across measurement families; the gate and audit use the same \(B_{\lambda_{\rm op}}\) solver."
)


def copy_latex_project() -> None:
    if not SOURCE_LATEX.exists():
        raise FileNotFoundError(SOURCE_LATEX)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_LATEX, LATEX_DIR, dirs_exist_ok=True)
    (LATEX_DIR / "figures").mkdir(parents=True, exist_ok=True)
    for name in [
        "fig1_operator_circuit_final.svg",
        "fig1_operator_circuit_final.pdf",
        "fig1_operator_circuit_final_600dpi.png",
    ]:
        src = FIG_DIR / name
        if src.exists():
            shutil.copy2(src, LATEX_DIR / "figures" / name)


def replace_existing_figure(text: str) -> str:
    pattern = re.compile(
        r"\n\\begin\{figure\*\}\[t\]\s*"
        r"\\centering\s*"
        r"\\includegraphics\[width=\\textwidth\]\{figures/fig1_operator_centered_v44\.pdf\}.*?"
        r"\\label\{fig:mechanism\}\s*"
        r"\\end\{figure\*\}\s*",
        flags=re.DOTALL,
    )
    text = pattern.sub("\n", text)
    # Also remove any previous Phase 50 insertion if the script is rerun.
    phase50_pattern = re.compile(
        r"\n\\begin\{figure\*\}\[t\]\s*"
        r"\\centering\s*"
        r"\\includegraphics\[width=\\textwidth\]\{figures/fig1_operator_circuit_final\.pdf\}.*?"
        r"\\label\{fig:operator_circuit\}\s*"
        r"\\end\{figure\*\}\s*",
        flags=re.DOTALL,
    )
    return phase50_pattern.sub("\n", text)


def update_method() -> None:
    path = LATEX_DIR / "sections" / "method.tex"
    text = path.read_text(encoding="utf-8")
    text = replace_existing_figure(text)
    text = text.replace(r"\label{fig:mechanism}", r"\label{fig:operator_circuit}")
    text = text.replace(r"\Cref{fig:mechanism}", r"\Cref{fig:operator_circuit}")
    text = text.replace(r"Fig.~\ref{fig:mechanism}", r"Fig.~\ref{fig:operator_circuit}")

    if METHOD_SENTENCE not in text:
        text = text.replace(
            r"\section{Measurement-Audited Neural Completion}" + "\n",
            r"\section{Measurement-Audited Neural Completion}" + "\n" + METHOD_SENTENCE + "\n",
            1,
        )

    if ANCHOR_SENTENCE not in text:
        marker = (
            r"In both cases, the gate and audit below use the same "
            r"\(B_{\lambda_{\rm op}}\) solver."
        )
        text = text.replace(marker, marker + "\n\n" + ANCHOR_SENTENCE, 1)

    insert_marker = (
        "The reconstruction problem is therefore an information split: measured components "
        "should remain tied to \\(A\\) and \\(y\\), while learned structure should enter "
        "only through components that the measurements weakly constrain."
    )
    if FIGURE_BLOCK not in text:
        text = text.replace(insert_marker, insert_marker + "\n\n" + FIGURE_BLOCK, 1)

    path.write_text(text, encoding="utf-8")


def update_results() -> None:
    path = LATEX_DIR / "sections" / "results.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace(r"\Cref{fig:mechanism}", r"\Cref{fig:operator_circuit}")
    text = text.replace(r"Fig.~\ref{fig:mechanism}", r"Fig.~\ref{fig:operator_circuit}")
    path.write_text(text, encoding="utf-8")


def update_cstv_formulae() -> None:
    candidates = [
        LATEX_DIR / "sections" / "validation_ablation.tex",
        LATEX_DIR / "supplement" / "supplement.tex",
    ]
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        text = text.replace(r"\lambda_{\rm tv} TV(x)", r"\lambda_{\rm tv}\operatorname{TV}(x)")
        text = text.replace(r"\lambda TV(x)", r"\lambda_{\rm tv}\operatorname{TV}(x)")
        path.write_text(text, encoding="utf-8")


def main() -> None:
    copy_latex_project()
    update_method()
    update_results()
    update_cstv_formulae()
    print(f"updated LaTeX project: {LATEX_DIR}")


if __name__ == "__main__":
    main()
