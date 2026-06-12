from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
BASE_OUT = ROOT / "outputs_phase37_author_guided_rewrite"
BASE_PROJECT = BASE_OUT / "latex_project_v37"
BASE_FIGURES = BASE_OUT / "figures"
OUT = ROOT / "outputs_phase38_professional_figure"
PROJECT = OUT / "latex_project_v38"
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

FIGURE_SENTENCE = (
    "Figure 1 anchors the proposed pipeline to conventional GI. "
    "The upper path shows the raw bucket-pattern correlation \\(A^Ty\\). "
    "The lower path replaces the raw bucket weights with decorrelated coefficients to form \\(x_{\\rm data}\\), "
    "then applies learned residual completion, null-space filtering, and final measurement audit."
)

FIGURE_BLOCK = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_professional_mechanism_v38.pdf}
\caption{\textbf{From GI correlation to measurement-audited neural completion.} Conventional GI forms a raw bucket-pattern correlation image \(A^Ty=\sum_i y_i a_i\). The data solution used in this work keeps the same pattern-expansion structure but replaces raw bucket weights with decorrelated coefficients \(q=(AA^T+\lambda I)^{-1}y\). The neural module then proposes a candidate residual, \(P_N\) filters residual components visible to the measurement operator, and \(\Pi_y\) audits the completed image against the original bucket measurements.}
\label{fig:mechanism}
\end{figure*}
"""


def copy_project() -> None:
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE_PROJECT, PROJECT)
    for path in PROJECT.rglob("*"):
        if path.is_file() and path.suffix in COMPILED_SUFFIXES:
            path.unlink()


def copy_figures() -> None:
    dst = PROJECT / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for source_dir in [BASE_PROJECT / "figures", BASE_FIGURES, FIG_DIR]:
        if source_dir.exists():
            for path in source_dir.iterdir():
                if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg"}:
                    shutil.copy2(path, dst / path.name)
                    if path.name.startswith("fig4_measurement_attribution_v36") and path.parent.resolve() != FIG_DIR.resolve():
                        FIG_DIR.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, FIG_DIR / path.name)


def replace_once(text: str, pattern: str, replacement: str, label: str, flags: int = re.DOTALL) -> str:
    new = re.sub(pattern, lambda _m: replacement.strip() + "\n", text, count=1, flags=flags)
    if new == text:
        raise RuntimeError(f"Could not replace {label}")
    return new


def update_method() -> None:
    path = PROJECT / "sections" / "method.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace("Fig. 1 visualizes the same logic.", "")
    text = re.sub(
        r"Figure 1 anchors the proposed pipeline to conventional GI\..*?final measurement audit\.",
        FIGURE_SENTENCE,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if FIGURE_SENTENCE not in text:
        text = text.replace(
            "This projection is a whole-image audit, not only a residual correction. "
            "Although \\(P_N\\) filters the neural proposal, it is approximate when \\(\\lambda>0\\), and subsequent refinement, numerical factorization, and intensity clipping can still introduce measurement inconsistency. "
            "The final \\(\\Pi_y\\) step therefore checks the completed image against the bucket signal after learned completion has been inserted.",
            "This projection is a whole-image audit, not only a residual correction. "
            "Although \\(P_N\\) filters the neural proposal, it is approximate when \\(\\lambda>0\\), and subsequent refinement, numerical factorization, and intensity clipping can still introduce measurement inconsistency. "
            "The final \\(\\Pi_y\\) step therefore checks the completed image against the bucket signal after learned completion has been inserted.\n\n"
            + FIGURE_SENTENCE,
        )
    text = replace_once(
        text,
        r"\\begin\{figure\*\}\[t\].*?\\label\{fig:mechanism\}\s*\\end\{figure\*\}",
        FIGURE_BLOCK,
        "Figure 1 block",
    )
    path.write_text(text.strip() + "\n", encoding="utf-8")


def update_results() -> None:
    path = PROJECT / "sections" / "results.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace("Similar final PSNR does not mean similar sensing behavior.", "Similar final PSNR does not imply similar sensing behavior.")
    path.write_text(text.strip() + "\n", encoding="utf-8")


def update_validation() -> None:
    path = PROJECT / "sections" / "validation_ablation.tex"
    text = path.read_text(encoding="utf-8")
    replacements = {
        r"\subsection{Exact-A reproducibility}": r"\subsection{Is random sensing reproducible?}",
        r"\subsection{Inference-time ablation}": r"\subsection{Is the final measurement audit necessary?}",
        r"\subsection{Noise and perturbation tests}": r"\subsection{Does the network depend on the bucket vector?}",
        r"\subsection{CSGI-style CS-TV\(PGD\) compressed-sensing baseline}": r"\subsection{Is the method stronger than a classical CSGI-style prior?}",
        r"\subsection{DC row control}": r"\subsection{Stability diagnostics}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def clean_forbidden_text_and_tv() -> None:
    for path in PROJECT.rglob("*.tex"):
        text = path.read_text(encoding="utf-8")
        text = text.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
        text = text.replace(
            "All reported results are obtained under the computational forward model \\(y=Ax+\\epsilon\\). No hardware optical experiment is included in this study.",
            "All reported results are obtained under the computational forward model \\(y=Ax+\\epsilon\\).",
        )
        text = text.replace(
            "This study does not include a hardware optical experiment; the reported results are simulation/dataset-based evaluations of the reconstruction framework under the computational forward model \\(y=Ax+\\epsilon\\).",
            "This study reports simulation/dataset-based evaluations of the reconstruction framework under the computational forward model \\(y=Ax+\\epsilon\\).",
        )
        text = text.replace("Future work should include hardware validation", "Future work should include physical-system validation")
        path.write_text(text, encoding="utf-8")


def compile_pdf(filename: str) -> None:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", filename],
        cwd=PROJECT,
        check=True,
    )


def copy_outputs() -> None:
    shutil.copy2(PROJECT / "main.pdf", OUT / "main_v38.pdf")
    shutil.copy2(PROJECT / "supplement.pdf", OUT / "supplement_v38.pdf")
    audit = BASE_OUT / "citation_audit_phase37.md"
    if audit.exists():
        shutil.copy2(audit, OUT / "citation_audit_phase38.md")


def main() -> None:
    copy_project()
    copy_figures()
    update_method()
    update_results()
    update_validation()
    clean_forbidden_text_and_tv()
    compile_pdf("main.tex")
    compile_pdf("supplement.tex")
    copy_outputs()
    print(
        {
            "project": str(PROJECT),
            "main_pdf": str(OUT / "main_v38.pdf"),
            "supplement_pdf": str(OUT / "supplement_v38.pdf"),
            "figure1": str(FIG_DIR / "fig1_professional_mechanism_v38.pdf"),
        }
    )


if __name__ == "__main__":
    main()
