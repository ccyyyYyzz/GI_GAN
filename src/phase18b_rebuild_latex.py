from __future__ import annotations

import shutil
from pathlib import Path

from .phase18b_common import METHOD_LABEL, OUT, PHASE15, PHASE16, TITLE, copy_figures_to_latex, ensure_dir, registry_by_id, table, tex_escape, write_text


LATEX = OUT / "latex_project_v4"
SECTIONS = LATEX / "sections"
SUPP = LATEX / "supplement"
FIGS = LATEX / "figures"
TABLES = LATEX / "tables"


def copy_tables() -> None:
    ensure_dir(TABLES)
    src = OUT / "tables"
    for path in src.glob("*.tex"):
        shutil.copy2(path, TABLES / path.name)
    detail = src / "detailed_csv"
    if detail.exists():
        dst = TABLES / "detailed_csv"
        ensure_dir(dst)
        for path in detail.glob("*.csv"):
            shutil.copy2(path, dst / path.name)


def fmt(mid: str, key: str, digits: int = 3) -> str:
    from .phase18b_common import fmt as f

    return f(registry_by_id()[mid][key], digits)


def abstract() -> str:
    return rf"""
Ghost imaging reconstructs spatial information from structured illumination and bucket measurements. At low sampling ratios, the inverse problem is underdetermined, and reconstruction quality must be interpreted together with measurement consistency. We present a measurement-consistent null-space neural reconstruction pipeline that starts from a physical data solution, inserts learned residual structure through an approximate null-space projection, and applies a final measurement-consistency projection. On STL-10, the method reaches {fmt('rademacher5_hq_noise001_colab', 'psnr')} dB PSNR / {fmt('rademacher5_hq_noise001_colab', 'ssim')} SSIM with Rademacher measurements at 5\% sampling and {fmt('scrambled_hadamard5_hq_noise001_colab', 'psnr')} dB / {fmt('scrambled_hadamard5_hq_noise001_colab', 'ssim')} SSIM with scrambled Hadamard measurements at 5\%. At 10\%, the corresponding results are {fmt('rademacher10_full_noise001_colab', 'psnr')} dB / {fmt('rademacher10_full_noise001_colab', 'ssim')} SSIM and {fmt('scrambled_hadamard10_full_noise001_colab', 'psnr')} dB / {fmt('scrambled_hadamard10_full_noise001_colab', 'ssim')} SSIM. Large qualitative panels, attribution, inference ablations, perturbation tests, finite-noise sweeps, and a lightweight CS-TV baseline support the role of measurement-dependent reconstruction.
""".strip()


def introduction() -> str:
    return r"""
\section{Introduction}
Ghost imaging and single-pixel imaging recover an image from known illumination patterns and scalar bucket detector readings. This acquisition model is attractive when dense sensor arrays are unavailable, expensive, or undesirable. The main difficulty is low sampling: when the number of measurements is far below the number of image pixels, many images remain compatible with the same measurement vector.

This work treats low-sampling ghost imaging as constrained completion rather than ordinary denoising. The reconstruction should improve visual quality while staying tied to the measured bucket signal. \Cref{fig:mechanism} summarizes the proposed pipeline: a data solution preserves measured row-space information, a neural residual completes missing structure, and a final projection restores measurement consistency.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_mechanism.pdf}
\caption{\textbf{Physics-consistent null-space neural reconstruction.} Structured or random patterns produce bucket measurements. The data solution preserves measured row-space information, the neural module completes missing structure through an approximate null-space residual, and the final projection restores measurement consistency.}
\label{fig:mechanism}
\end{figure*}
"""


def method() -> str:
    return r"""
\section{Problem Formulation and Method}
Let \(x\in\mathbb{R}^n\) be the vectorized image, \(A\in\mathbb{R}^{m\times n}\) the measurement matrix, and \(y\in\mathbb{R}^{m}\) the bucket vector. The forward model is \(y_i=\langle a_i,x\rangle+\epsilon_i\), or in stacked form \(y=Ax+\epsilon\), with sampling ratio \(\rho=m/n\). In the low-sampling regime \(m\ll n\), the feasible set \(\mathcal{C}_y=\{x:Ax=y\}\) contains infinitely many candidates. If \(Ax_0=y\) and \(v\in\mathrm{Null}(A)\), then \(A(x_0+v)=y\).

We compute a regularized data solution
\begin{equation}
x_{\mathrm{data}}=A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
The approximate null-space projection is
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av,
\end{equation}
which is exact when \(\lambda=0\) and \(A\) has full row rank. The neural module predicts \(r_\theta=G_\theta(x_{\mathrm{data}},z)\), and the residual is inserted as \(\tilde{x}=x_{\mathrm{data}}+P_N(r_\theta)\). A final projection
\begin{equation}
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y)
\end{equation}
produces \(\hat{x}=\Pi_y(\tilde{x})\). Image metrics use clipped reconstructions, while relative measurement error is computed on the unclamped reconstruction because clipping can alter \(A\hat{x}\).
"""


def results() -> str:
    return rf"""
\section{{Results}}
\Cref{{fig:main_metrics,tab:primary_results}} summarize the primary strict no-leak results. The internal engineering thresholds are PSNR \(\geq20\), SSIM \(\geq0.60\) for STL-10 at 5\%; PSNR \(\geq22\), SSIM \(\geq0.65\) for STL-10 at 10\%; and PSNR \(\geq25\), SSIM \(\geq0.80\) for MNIST/Fashion-MNIST at 5\%. These thresholds are study-specific engineering criteria, not theoretical limits.

\input{{tables/main_table1_primary_results.tex}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig2_main_metrics.pdf}}
\caption{{\textbf{{Primary metrics.}} PSNR and SSIM are shown separately to avoid dual-axis ambiguity. Dashed lines indicate internal engineering thresholds.}}
\label{{fig:main_metrics}}
\end{{figure*}}

Measurement family affects both the physical initialization and the amount of learned refinement required. \Cref{{fig:measurement_attribution,tab:measurement_attribution}} shows that Rademacher backprojection is weak but receives a large learned gain, while scrambled Hadamard starts from a stronger backprojection and reaches similar final quality.

\input{{tables/main_table2_measurement_attribution_summary.tex}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig3_measurement_attribution.pdf}}
\caption{{\textbf{{Measurement attribution.}} Rademacher, scrambled Hadamard, and low-frequency Hadamard measurements produce different physical initializations. Final PSNR alone hides these regimes.}}
\label{{fig:measurement_attribution}}
\end{{figure*}}

\paragraph{{Qualitative reconstruction.}}
\Cref{{fig:qualitative}} replaces the previous tiny reconstruction grid with a standalone large qualitative panel. Each row shows ground truth, backprojection, reconstruction, and absolute error for one STL-10 method. The images are displayed enlarged for visibility, so the panel should be interpreted as qualitative evidence rather than a new metric source.

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig4_qualitative_reconstructions.pdf}}
\caption{{\textbf{{Qualitative reconstructions.}} Large GT/BP/Recon/Error panels show the visual improvement over backprojection for STL-10 5\% and 10\% settings. Images are displayed enlarged for visibility.}}
\label{{fig:qualitative}}
\end{{figure*}}
"""


def validation() -> str:
    return r"""
\section{Ablation and Validation}
Inference-time ablation tests whether quality depends on the measurement-consistency components. \Cref{fig:ablation} shows that removing the DC/measurement-consistency projection causes the largest degradation. Removing the null projection has limited metric effect for these checkpoints, which suggests that the final projection and trained network already constrain many measured components; the null-space step remains part of the designed pipeline.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation.pdf}
\caption{\textbf{Inference ablation.} Full reconstruction is compared with no-DC projection, no-null projection, stage-1-only output, raw weights, and EMA weights.}
\label{fig:ablation}
\end{figure*}

\Cref{fig:robustness} summarizes finite-noise, perturbation, CS-TV, and confidence-interval diagnostics. The measurement perturbation tests show that shuffled coefficients and wrong measurement vectors cause large drops, supporting dependence on \(y\). The CS-TV baseline solves
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\,\mathrm{TV}(x).
\end{equation}
It is a TV-regularized compressed-sensing baseline solved by projected gradient descent on a small subset, not an exhaustively optimized ADMM/FISTA or plug-and-play comparison.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_robustness_baselines.pdf}
\caption{\textbf{Robustness and baselines.} Finite-noise, measurement perturbation, CS-TV, and bootstrap diagnostics support measurement-dependent reconstruction. CS-TV is a lightweight PGD small-subset baseline.}
\label{fig:robustness}
\end{figure*}
"""


def discussion() -> str:
    return r"""
\section{Discussion and Limitations}
The results support high-quality STL-10 reconstruction at 5\% and 10\% under Rademacher and scrambled Hadamard measurement families. Low-frequency Hadamard 5\% is not used as a primary STL-10 high-quality claim; it remains useful for simple-domain and diagnostic controls. The final mechanism is measurement-consistent neural reconstruction, not adversarial generation.

Important limitations remain. The study does not yet include a hardware optical experiment. The noise study covers only the tested finite range. The CS-TV comparison is a lightweight small-subset baseline. Class-wise results are diagnostic. Exact random-operator handling is essential for Rademacher evaluation because the exported measurement operator and rebuilt solver cache determine reproducibility.
"""


def conclusion() -> str:
    return r"""
\section{Conclusion}
The revised presentation separates metric figures from qualitative reconstructions, uses a standalone large reconstruction panel, and moves dense supplementary tables into concise summaries plus detailed CSV files. The scientific claim remains unchanged: measurement-consistent null-space neural reconstruction can produce high-quality low-sampling GI results under suitable measurement families while preserving an explicit link to the bucket measurements.
"""


def supplement() -> str:
    return rf"""
\section{{Supplementary Material}}
The full detailed CSV files are available under \path{{tables/detailed_csv}} in this LaTeX project. Internal source paths are retained here for auditability: registry \path{{{PHASE15 / 'noleak_registry.csv'}}}; supplementary results \path{{{PHASE16}}}.

\subsection{{Simple-domain qualitative reconstructions}}
\begin{{figure*}}[h]
\centering
\includegraphics[width=\textwidth]{{figures/figS_simple_domain_reconstructions.pdf}}
\caption{{Simple-domain MNIST and Fashion-MNIST qualitative reconstructions. Images are displayed enlarged for visibility.}}
\label{{fig:simple_domain}}
\end{{figure*}}

\subsection{{Ablation measurement error}}
\begin{{figure*}}[h]
\centering
\includegraphics[width=\textwidth]{{figures/figS_ablation_relmeaserr.pdf}}
\caption{{Relative measurement error under inference-time ablations.}}
\label{{fig:supp_relmeas}}
\end{{figure*}}

\subsection{{Curated summary tables}}
\input{{tables/supplement_noise_summary_table.tex}}
\input{{tables/supplement_traditional_baseline_summary_table.tex}}
\input{{tables/supplement_classwise_summary_table.tex}}
\input{{tables/supplement_runtime_summary_table.tex}}
"""


def main_tex() -> str:
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{cleveref}}
\usepackage{{url}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\title{{{tex_escape(TITLE)}}}
\author{{Anonymous authors}}
\date{{}}
\begin{{document}}
\maketitle
\begin{{abstract}}
\input{{sections/abstract.tex}}
\end{{abstract}}
\input{{sections/introduction.tex}}
\input{{sections/method.tex}}
\input{{sections/results.tex}}
\input{{sections/validation.tex}}
\input{{sections/discussion.tex}}
\input{{sections/conclusion.tex}}
\clearpage
\appendix
\input{{supplement/supplement.tex}}
\end{{document}}
"""


def write_sections() -> None:
    ensure_dir(SECTIONS)
    write_text(SECTIONS / "abstract.tex", abstract())
    write_text(SECTIONS / "introduction.tex", introduction())
    write_text(SECTIONS / "method.tex", method())
    write_text(SECTIONS / "results.tex", results())
    write_text(SECTIONS / "validation.tex", validation())
    write_text(SECTIONS / "discussion.tex", discussion())
    write_text(SECTIONS / "conclusion.tex", conclusion())
    ensure_dir(SUPP)
    write_text(SUPP / "supplement.tex", supplement())


def main() -> None:
    ensure_dir(LATEX)
    write_sections()
    copy_figures_to_latex(OUT / "figures", FIGS)
    copy_tables()
    write_text(LATEX / "main.tex", main_tex())
    write_text(
        OUT / "MANUSCRIPT_V4_NOTES.md",
        "\n".join(
            [
                "# Manuscript V4 Notes",
                "",
                "Phase18B rebuilt the LaTeX project without overwriting Phase18 manuscript_v3.",
                "Main text uses the new standalone qualitative Figure 4 and removes the old embedded tiny grid.",
                "Dense supplement tables are represented by curated summaries; detailed CSVs are copied into latex_project_v4/tables/detailed_csv.",
            ]
        ),
    )
    print({"latex_project_v4": str(LATEX), "main_tex": str(LATEX / "main.tex")})


if __name__ == "__main__":
    main()
