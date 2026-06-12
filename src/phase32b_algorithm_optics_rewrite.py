from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
SOURCE_PROJECT = ROOT / "outputs_phase30_submission_package" / "latex_project_submission"
OUT = ROOT / "outputs_phase32b_algorithm_optics_baselines"
PROJECT = OUT / "latex_project_algorithm_optics"

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


ABSTRACT = (
    "This paper studies the computational reconstruction problem associated with ghost imaging "
    "and single-pixel imaging under a dataset-based forward model. Structured illumination "
    "patterns are represented by a known measurement operator, and scalar bucket measurements "
    "are simulated as \(y=Ax+\\epsilon\). Low-sampling acquisition is severely underdetermined: "
    "the measurement vector does not identify a unique image. Low sampling therefore needs a "
    "learned prior, yet unconstrained priors may hallucinate structure that is not supported by "
    "the bucket readings. We address this tension with measurement-consistent null-space neural "
    "reconstruction. The formulation computes a physical data solution from the forward "
    "operator, adds a learned null-space residual for missing structure, and applies a final "
    "measurement projection to audit the completed image against the measured signal. Under a "
    "leakage-free STL-10 protocol, the method reaches 22.316 dB PSNR / 0.635 SSIM at 5\\% "
    "sampling with Rademacher measurements and 22.271 dB / 0.632 SSIM with scrambled Hadamard "
    "measurements. At 10\\% sampling, the corresponding results are 24.781 dB / 0.747 SSIM and "
    "24.730 dB / 0.746 SSIM. MNIST and Fashion-MNIST 5\\% experiments provide simple-domain "
    "sanity checks, reaching 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM. Exact-A audit, "
    "measurement-family attribution, inference ablation, finite-noise tests, measurement "
    "perturbation, comparison against a CSGI-style TV-regularized compressed-sensing baseline "
    "solved by projected gradient descent, abbreviated CS-TV(PGD), and confidence intervals "
    "support measurement-dependent reconstruction across the tested measurement families. No "
    "hardware optical experiment is included in this study."
)


METHOD_FIGURE = r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_forward_model_reconstruction.pdf}
\caption{\textbf{Computational GI forward model and measurement-consistent reconstruction.} Figure 1 shows the computational forward model and reconstruction mechanism used in this study. It is not a hardware optical setup. Low-sampling GI is treated as measurement-constrained completion: the measured component is represented by \(x_{\rm data}\), missing structure is completed through a neural residual, and the final image is projected back to the measurement-consistent set.}
\label{fig:mechanism}
\end{figure*}
"""


VALIDATION_CSTV = r"""
\subsection{CSGI-style CS-TV(PGD) compressed-sensing baseline}
To provide a conventional CSGI-style control, we compare against a TV-regularized compressed-sensing reconstruction solved by projected gradient descent,
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\operatorname{TV}(x).
\end{equation}
We refer to this baseline as CS-TV(PGD). It is intended as a lightweight compressed-sensing control, not as an exhaustively optimized ADMM or FISTA benchmark. GI/BP denotes the linear physical backprojection or correlation-like GI reconstruction. \Cref{fig:gi_csgi_visual} adds a compact visual comparison in the style of standard GI/CSGI reporting.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig7_gi_csgi_ours_visual_comparison.pdf}
\caption{\textbf{GI/BP, CSGI-style CS-TV(PGD), and ours.} GI/BP denotes the linear physical backprojection or correlation-like GI reconstruction. CSGI denotes a TV-regularized compressed-sensing baseline solved by PGD. The comparison is included to match standard GI/CSGI visual reporting, but CS-TV(PGD) is not claimed to be an exhaustively optimized compressed-sensing solver.}
\label{fig:gi_csgi_visual}
\end{figure*}
"""


LIMITATIONS = (
    "\\section{Limitations}\n"
    "This study does not include a hardware optical experiment; the reported results are "
    "simulation/dataset-based evaluations of the reconstruction framework under the "
    "computational forward model \(y=Ax+\\epsilon\). We do not claim a ranking over external "
    "benchmarks because datasets, measurement operators, sampling protocols, and evaluation "
    "splits are not standardized across the literature. The CS-TV(PGD) baseline is a "
    "CSGI-style lightweight small-subset compressed-sensing control, not an exhaustively "
    "optimized compressed-sensing solver. Robustness is tested only over finite noise and "
    "perturbation settings. Class-wise evaluation is diagnostic rather than a claim of uniform "
    "category performance. Exact-A handling is essential for random measurements, and results "
    "should be interpreted with that audit path in place. The 5\\% low-frequency Hadamard "
    "condition is retained only as a diagnostic control rather than as a primary STL-10 "
    "claim. The learned binary-illumination branch is not used as a claimed final result, "
    "and adversarial training is not the final contribution mechanism. Future work "
    "should include hardware validation, broader external baselines, and more extensive "
    "cross-domain testing.\n"
)


SUPPLEMENT_APPEND = r"""

\subsection{S8 GI/BP and CSGI visual baseline comparison}
GI/BP denotes the linear physical backprojection or correlation-like GI reconstruction from the same measurement operator. CSGI is represented by CS-TV(PGD), a TV-regularized compressed-sensing control solved by projected gradient descent:
\[
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\operatorname{TV}(x).
\]
The visual comparison is evaluated on selected samples only and is included to mirror common GI/CSGI qualitative reporting. Quantitative conclusions remain based on the full leakage-free metrics and the curated tables in the main text.

\begin{figure*}[h]
\centering
\includegraphics[width=\textwidth]{figures/fig7_gi_csgi_ours_all_supplement.pdf}
\caption{Supplementary GI/BP, CSGI-style CS-TV(PGD), and ours visual comparison for STL-10 5\% and 10\% settings. CS-TV(PGD) is a lightweight compressed-sensing control, not an exhaustively optimized iterative benchmark.}
\label{fig:supp_gi_csgi_visual}
\end{figure*}

\input{tables/tableS8_gi_csgi_visual_subset.tex}
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
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_PROJECT, PROJECT, ignore=_ignore)
    src_fig = SOURCE_PROJECT / "figures"
    dst_fig = PROJECT / "figures"
    if src_fig.exists():
        dst_fig.mkdir(parents=True, exist_ok=True)
        for path in src_fig.iterdir():
            if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg"}:
                shutil.copy2(path, dst_fig / path.name)


def replace_block(text: str, pattern: str, replacement: str) -> str:
    return re.sub(pattern, lambda _match: replacement.strip() + "\n", text, flags=re.DOTALL)


def update_main_text() -> None:
    write_text(PROJECT / "sections" / "abstract.tex", ABSTRACT)
    write_text(PROJECT / "sections" / "limitations.tex", LIMITATIONS)

    introduction = PROJECT / "sections" / "introduction.tex"
    intro = introduction.read_text(encoding="utf-8")
    needle = (
        "In such systems, optical measurement design and computational inversion are coupled: "
        "the patterns determine not only what is measured, but also what must later be inferred.\n"
    )
    insert = (
        "\nWe study the computational reconstruction problem associated with low-sampling "
        "GI/SPI, using known measurement operators and dataset-based forward simulations. "
        "All reported images and metrics are obtained from this computational model rather "
        "than from a hardware optical experiment.\n"
    )
    if insert.strip() not in intro:
        intro = intro.replace(needle, needle + insert)
    introduction.write_text(intro, encoding="utf-8")

    related = PROJECT / "sections" / "related_work.tex"
    text = related.read_text(encoding="utf-8")
    sentence = (
        "\nConventional GI studies often compare learned reconstructions with basic "
        "GI/correlation and CSGI-style compressed-sensing reconstructions. We therefore "
        "include both linear GI/BP and CS-TV(PGD) controls, while keeping the focus on the "
        "measurement-consistent reconstruction mechanism.\n"
    )
    if "linear GI/BP and CS-TV(PGD) controls" not in text:
        text += sentence
    related.write_text(text, encoding="utf-8")

    protocol = PROJECT / "sections" / "experimental_protocol.tex"
    text = protocol.read_text(encoding="utf-8")
    addition = (
        "\nAll reported results are obtained under the computational forward model "
        "\\(y=Ax+\\epsilon\\). No hardware optical experiment is included in this study. "
        "Linear GI/BP, adjoint/DGI-like, and CS-TV(PGD) baselines are evaluated using the "
        "same measurement operators and evaluation splits. CS-TV(PGD) is evaluated as a "
        "lightweight compressed-sensing control on a subset as described in the Supplement.\n"
    )
    if "No hardware optical experiment is included in this study" not in text:
        text = text.rstrip() + "\n" + addition
    protocol.write_text(text, encoding="utf-8")

    families = PROJECT / "sections" / "measurement_families.tex"
    text = families.read_text(encoding="utf-8")
    text = text.replace(
        "Low-frequency Hadamard is therefore an interpretable diagnostic family, but low-frequency Hadamard at 5\\% is not the primary STL-10 high-quality result in this work.",
        "Low-frequency Hadamard is therefore an interpretable diagnostic family, but its 5\\% STL-10 condition is not used as a primary natural-image claim in this work.",
    )
    families.write_text(text, encoding="utf-8")

    results = PROJECT / "sections" / "results.tex"
    text = results.read_text(encoding="utf-8")
    text = text.replace(
        "Low-frequency Hadamard points are diagnostic rather than primary STL-10 high-quality claims.",
        "Low-frequency Hadamard points are diagnostic controls rather than primary STL-10 claims.",
    )
    results.write_text(text, encoding="utf-8")

    method = PROJECT / "sections" / "method.tex"
    text = method.read_text(encoding="utf-8")
    text = text.replace("G_\\theta(x_{\\rm data},z)", "G_\\theta(x_{\\rm data},z)")
    text = replace_block(
        text,
        r"\\begin\{figure\*\}\[t\].*?\\label\{fig:mechanism\}\s*\\end\{figure\*\}",
        METHOD_FIGURE,
    )
    method.write_text(text, encoding="utf-8")

    validation = PROJECT / "sections" / "validation_ablation.tex"
    text = validation.read_text(encoding="utf-8")
    text = replace_block(
        text,
        r"\\subsection\{CS-TV compressed-sensing baseline\}.*?(?=\\begin\{figure\*\}\[t\]\s*\\centering\s*\\includegraphics\[width=\\textwidth\]\{figures/fig6_robustness_baselines_submission\.pdf\})",
        VALIDATION_CSTV,
    )
    text = text.replace("CS-TV comparison", "CS-TV(PGD) comparison")
    validation.write_text(text, encoding="utf-8")


def update_supplement() -> None:
    supp = PROJECT / "supplement" / "supplement.tex"
    text = supp.read_text(encoding="utf-8")
    text = text.replace(r"\subsection{S4 CS-TV compressed-sensing baseline}", r"\subsection{S4 CSGI-style CS-TV(PGD) compressed-sensing baseline}")
    text = text.replace(r"\subsection{S7 Runtime and complexity}", SUPPLEMENT_APPEND.strip() + "\n\n" + r"\subsection{S9 Runtime and complexity}")
    supp.write_text(text, encoding="utf-8")

    table = PROJECT / "tables" / "tableS3_cstv_baseline.tex"
    if table.exists():
        t = table.read_text(encoding="utf-8")
        t = t.replace(
            "TV-regularized compressed-sensing baseline solved by PGD (CS-TV). This is a lightweight small-subset baseline.",
            "CSGI-style TV-regularized compressed-sensing baseline solved by PGD, abbreviated CS-TV(PGD). This is a lightweight small-subset control.",
        )
        table.write_text(t, encoding="utf-8")


def write_report() -> None:
    report = """# Algorithm-Only Optics Framing Rewrite

Status: prepared

- Repositioned the manuscript as a computational GI/SPI reconstruction study.
- Added explicit dataset-based forward-model language and no-hardware-experiment language.
- Replaced the mechanism figure reference with a computational forward-model schematic.
- Updated CS-TV wording to CSGI-style CS-TV(PGD).
- Added main-text GI/BP vs CSGI/CS-TV(PGD) vs ours visual-comparison placeholder; figure files are generated by the export script.
- Added supplementary visual-comparison section and table placeholder.
- Main numerical results were not changed.
"""
    write_text(OUT / "ALGORITHM_OPTICS_REWRITE_REPORT.md", report)


def main() -> None:
    copy_project()
    update_main_text()
    update_supplement()
    write_report()
    print({"output_dir": str(OUT), "latex_project": str(PROJECT)})


if __name__ == "__main__":
    main()
