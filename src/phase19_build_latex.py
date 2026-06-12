from __future__ import annotations

import shutil
from pathlib import Path

from .phase19_common import (
    METHOD_LABEL,
    METHOD_ORDER,
    OUT,
    PHASE16,
    SIMPLE_METHODS,
    STL_METHODS,
    TABLES,
    TITLE,
    as_float,
    copy_figures_to_latex,
    ensure_dir,
    fmt,
    markdown_table,
    registry_by_id,
    source_manifest,
    table,
    tex_escape,
    tex_table,
    write_csv,
    write_json,
    write_text,
)
from .phase19_rewrite_narrative import introduction_tex, metric_sentence


LATEX = OUT / "latex_project_v5"
SECTIONS = LATEX / "sections"
SUPP = LATEX / "supplement"
FIGS = LATEX / "figures"
TABLE_DIR = OUT / "tables"
LATEX_TABLES = LATEX / "tables"


def write_pack(name: str, rows: list[dict], fields: list[str], caption: str, label: str, *, wide: bool = True) -> None:
    write_csv(TABLE_DIR / f"{name}.csv", rows, fields)
    write_text(TABLE_DIR / f"{name}.md", markdown_table(rows, fields))
    write_text(TABLE_DIR / f"{name}.tex", tex_table(rows, fields, caption, label, wide=wide))


def build_tables() -> None:
    ensure_dir(TABLE_DIR)
    reg = registry_by_id()
    rows = []
    for mid in METHOD_ORDER:
        r = reg[mid]
        ratio = as_float(r["sampling_ratio"])
        psnr = as_float(r["psnr"])
        ssim = as_float(r["ssim"])
        if r["dataset"] == "STL-10" and abs(ratio - 0.05) < 1e-6:
            hq = psnr >= 20.0 and ssim >= 0.60
        elif r["dataset"] == "STL-10" and abs(ratio - 0.10) < 1e-6:
            hq = psnr >= 22.0 and ssim >= 0.65
        else:
            hq = psnr >= 25.0 and ssim >= 0.80
        rows.append(
            {
                "Dataset": r["dataset"],
                "Sampling": f"{ratio * 100:.0f}%",
                "Measurement": r["measurement_family"].replace("_", " "),
                "PSNR": fmt(r["psnr"]),
                "SSIM": fmt(r["ssim"]),
                "BP PSNR": fmt(r["backproj_psnr"]),
                "Delta PSNR": fmt(r["delta_psnr"]),
                "HQ?": "yes" if hq else "no",
            }
        )
    write_pack(
        "table1_primary_results",
        rows,
        ["Dataset", "Sampling", "Measurement", "PSNR", "SSIM", "BP PSNR", "Delta PSNR", "HQ?"],
        r"\textbf{Primary strict no-leak results.} HQ uses internal engineering thresholds stated in the text.",
        "tab:primary_results",
    )

    attr = {r["method_id"]: r for r in table("attribution")}
    rows = []
    for mid in STL_METHODS:
        a = attr[mid]
        interp = "weak initialization, large gain" if "rademacher" in a["measurement_family"] else "stronger initialization, similar final"
        rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "BP PSNR": fmt(a["backproj_psnr"]),
                "Model PSNR": fmt(a["model_psnr"]),
                "Delta PSNR": fmt(a["delta_psnr"]),
                "Interpretation": interp,
            }
        )
    write_pack(
        "table2_attribution_summary",
        rows,
        ["Method", "BP PSNR", "Model PSNR", "Delta PSNR", "Interpretation"],
        r"\textbf{Measurement attribution summary.} Final quality hides physical-initialization regimes.",
        "tab:attribution",
    )

    exact = table("exact_a")
    rows = [
        {
            "Method": METHOD_LABEL.get(r["method_id"], r["method_id"]),
            "Original PSNR": fmt(r["original_psnr"]),
            "Re-eval PSNR": fmt(r["reeval_psnr"]),
            "Abs diff": fmt(r["abs_diff_psnr"], 6),
            "Exact A loaded": r["exact_A_loaded"],
            "Cache rebuilt": r["cache_rebuilt"],
        }
        for r in exact
    ]
    write_pack("supp_exact_a_summary", rows, list(rows[0]), "Exact-A reproducibility summary.", "tab:supp_exact_a", wide=True)

    stats = table("statistics")
    rows = [
        {
            "Method": METHOD_LABEL.get(r["method_id"], r["method_id"]),
            "Mean PSNR": fmt(r["mean_psnr"]),
            "PSNR CI95": f"{fmt(r['ci95_psnr_low'])}-{fmt(r['ci95_psnr_high'])}",
            "Mean SSIM": fmt(r["mean_ssim"]),
            "SSIM CI95": f"{fmt(r['ci95_ssim_low'])}-{fmt(r['ci95_ssim_high'])}",
        }
        for r in stats
    ]
    write_pack("supp_ci_summary", rows, list(rows[0]), "Bootstrap confidence-interval summary.", "tab:supp_ci", wide=True)

    detail = ensure_dir(TABLE_DIR / "data_csv")
    for name, path in TABLES.items():
        rows = table(name)
        if rows:
            write_csv(detail / f"{name}.csv", rows)


def copy_tables_to_latex() -> None:
    ensure_dir(LATEX_TABLES)
    for path in TABLE_DIR.glob("*.tex"):
        shutil.copy2(path, LATEX_TABLES / path.name)
    src_detail = TABLE_DIR / "data_csv"
    if src_detail.exists():
        dst = ensure_dir(LATEX_TABLES / "data_csv")
        for path in src_detail.glob("*.csv"):
            shutil.copy2(path, dst / path.name)


def abstract() -> str:
    return (
        "Low-sampling ghost imaging is an underdetermined inverse problem in which many images match the same bucket measurements. "
        "We formulate reconstruction as measurement-constrained completion: a data solution carries measured row-space information, a neural residual completes missing null-space structure, and a final projection restores consistency with the measured signal. "
        + metric_sentence().replace("%", r"\%")
    )


def forward_model() -> str:
    return r"""\section{Forward Model and Inverse Problem}
For a vectorized image \(x\in\mathbb{R}^n\), ghost imaging measures scalar bucket readings
\begin{equation}
y_i=\langle a_i,x\rangle+\epsilon_i,
\end{equation}
or, in stacked form, \(y=Ax+\epsilon\), where \(A\in\mathbb{R}^{m\times n}\) and \(m\ll n\). The feasible set \(\mathcal{C}_y=\{x:Ax=y\}\) is therefore large. This makes low-sampling GI a constrained completion problem: physical measurements determine row-space content, while null-space content must be inferred.
"""


def method() -> str:
    return r"""\section{Measurement-Consistent Null-Space Reconstruction}
The method starts from a regularized data solution
\begin{equation}
x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
This term is not a learned hallucination; it is the physical component directly tied to \(y\). A neural module then predicts a residual \(G_\theta(x_{\rm data},z)\). Instead of adding this residual arbitrarily, we insert it through an approximate null-space projector
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av.
\end{equation}
The final estimate is checked with a measurement-consistency projection \(\Pi_y\). The central computational mechanism is
\begin{equation}
\boxed{\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))]}.
\end{equation}
This equation is the central computational mechanism of the paper. It separates measured row-space preservation, neural null-space completion, and final measurement checking. \Cref{fig:mechanism} visualizes the same logic.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_mechanism.pdf}
\caption{\textbf{Mechanism.} The method preserves measured row-space content, completes missing null-space structure with a neural residual, and checks the output against the bucket measurements.}
\label{fig:mechanism}
\end{figure*}
"""


def measurement_families() -> str:
    return r"""\section{Measurement Families, Network, and Losses}
Rademacher measurements use signed random rows and require exact operator reloading for reproducibility. Scrambled Hadamard measurements use selected orthogonal rows and produce stronger physical initializations. Low-frequency Hadamard rows are useful diagnostic controls, especially for DC-row behavior, but low-frequency Hadamard 5\% is not used as a primary STL-10 high-quality claim.

The network is treated as a neural residual reconstructor rather than as a generative adversarial mechanism. Losses and evaluation emphasize reconstruction fidelity, measurement consistency, strict no-leak protocol, and exact operator handling for random measurements.
"""


def results() -> str:
    return r"""\section{Results}
\subsection{Natural-image reconstruction at 5\% and 10\%}
\Cref{tab:primary_results,fig:main_metrics} report the primary strict no-leak results. STL-10 reaches the internal engineering thresholds at both 5\% and 10\% sampling under Rademacher and scrambled Hadamard measurements. These thresholds are study-specific engineering criteria, not theoretical limits.
\input{tables/table1_primary_results.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_main_metrics.pdf}
\caption{\textbf{Primary metrics.} PSNR and SSIM are shown separately for STL-10 and simple-domain sanity checks. Dashed lines denote internal engineering thresholds.}
\label{fig:main_metrics}
\end{figure*}

\subsection{Qualitative reconstruction}
\Cref{fig:qualitative} shows STL-10 ground truth, backprojection, reconstruction, and error maps. These examples illustrate the visual change from physical backprojection to measurement-consistent neural reconstruction; they are qualitative evidence and not an additional metric source.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth,height=0.82\textheight,keepaspectratio]{figures/fig3_qualitative_grid_v2.pdf}
\caption{\textbf{Qualitative STL-10 reconstructions.} Each method uses two saved strict no-leak evaluation samples, shown as GT/BP/Recon/Error.}
\label{fig:qualitative}
\end{figure*}

\subsection{Simple-domain sanity}
MNIST and Fashion-MNIST at 5\% sampling verify that the pipeline is stable on simpler structured targets. These are sanity checks rather than the main novelty.

\subsection{Measurement-family attribution}
\Cref{tab:attribution,fig:regime_map} separate physical initialization from neural gain. Rademacher has weak backprojection but large neural gain; scrambled Hadamard has stronger backprojection and similar final quality. Final PSNR alone hides physical-initialization regimes.
\input{tables/table2_attribution_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig4_measurement_regime_map.pdf}
\caption{\textbf{Measurement regime map.} Backprojection PSNR and neural gain reveal whether performance comes from physical initialization, neural refinement, or both. Low-frequency rows are shown as auxiliary diagnostics.}
\label{fig:regime_map}
\end{figure*}
"""


def validation() -> str:
    return r"""\section{Validation and Ablations}
\subsection{Exact-A reproducibility}
For Rademacher measurements, the exact exported random operator is loaded and the solver cache is rebuilt before evaluation. This avoids stale-cache mismatch and random-matrix mismatch.

\subsection{Measurement-consistency ablation}
\Cref{fig:ablation} shows that removing the DC/measurement-consistency projection causes the largest degradation. This is the strongest evidence that the output remains tied to the bucket measurements.

\subsection{Null-space and refiner ablation}
Removing the null projection has limited metric effect for these checkpoints. This is reported as a limitation of the ablation evidence: the final projection and trained network may already constrain measured components. Stage-1-only and raw-weight evaluations still show that the refiner and EMA weights contribute.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation.pdf}
\caption{\textbf{Inference ablation.} Full reconstruction is compared with no-DC projection, no-null projection, stage-1-only output, raw weights, and EMA weights.}
\label{fig:ablation}
\end{figure*}

\subsection{Noise and perturbation robustness}
\Cref{fig:validation_summary} summarizes finite-noise behavior, no-DC projection stress, and measurement perturbations. Shuffled coefficients and wrong measurement vectors cause large drops, supporting physical dependence rather than generic image-prior generation.

\subsection{CS-TV compressed-sensing baseline}
The CS-TV baseline solves
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\,\mathrm{TV}(x).
\end{equation}
It is a TV-regularized compressed-sensing baseline solved by projected gradient descent on a small subset. It is not an exhaustive ADMM/FISTA or plug-and-play benchmark.

\subsection{Confidence intervals and class-wise diagnostics}
Bootstrap confidence intervals and class-wise summaries are provided in the supplement. They are diagnostic, not the main claim.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_validation_summary.pdf}
\caption{\textbf{Validation summary.} The panels summarize finite-noise robustness, projection stress, measurement perturbation, and the CS-TV compressed-sensing baseline.}
\label{fig:validation_summary}
\end{figure*}
"""


def discussion() -> str:
    return r"""\section{Discussion}
The main innovation is a clean decomposition of the low-sampling GI inverse problem: row-space information is represented by the data solution, null-space content is completed by a neural residual, and the final image is checked against the measurement vector. This makes the method easier to audit than an unconstrained image-to-image reconstructor.

The measurement regime map clarifies why the same final PSNR can have different meanings. Rademacher results demonstrate large learned refinement from weak physical initialization. Scrambled Hadamard results show that better initialization can reach similar final quality with a different gain profile.

The limitations are explicit. There is no hardware experiment yet, no strict leaderboard claim, no validated binary learned illumination claim, and adversarial generation is not treated as the contribution mechanism. Low-frequency Hadamard 5\% remains auxiliary rather than a primary STL-10 high-quality result.
"""


def supplement() -> str:
    return r"""\section{Supplementary Material}
Detailed CSV files are packaged under the project data directory. The submission supplement contains curated summaries only.

\input{tables/supp_exact_a_summary.tex}
\input{tables/supp_ci_summary.tex}

\subsection{Graphical abstract}
\begin{figure*}[h]
\centering
\includegraphics[width=\textwidth]{figures/fig0_graphical_abstract.pdf}
\caption{Graphical abstract for presentation or slide use.}
\label{fig:supp_graphical}
\end{figure*}
"""


def references_bib() -> str:
    return r"""@article{TODO_VERIFY_ghost_imaging,
  title = {TODO_VERIFY: Ghost imaging and single-pixel imaging foundations},
  author = {TODO_VERIFY},
  journal = {TODO_VERIFY},
  year = {TODO_VERIFY}
}

@article{TODO_VERIFY_deep_gi,
  title = {TODO_VERIFY: Deep learning for ghost imaging reconstruction},
  author = {TODO_VERIFY},
  journal = {TODO_VERIFY},
  year = {TODO_VERIFY}
}

@article{TODO_VERIFY_tv_cs,
  title = {TODO_VERIFY: TV-regularized compressed sensing reconstruction},
  author = {TODO_VERIFY},
  journal = {TODO_VERIFY},
  year = {TODO_VERIFY}
}
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
\input{{sections/forward_model.tex}}
\input{{sections/method.tex}}
\input{{sections/measurement_families.tex}}
\input{{sections/results.tex}}
\input{{sections/validation.tex}}
\input{{sections/discussion.tex}}
\clearpage
\appendix
\input{{supplement/supplement.tex}}
\end{{document}}
"""


def write_sections() -> None:
    ensure_dir(SECTIONS)
    write_text(SECTIONS / "abstract.tex", abstract())
    write_text(SECTIONS / "introduction.tex", introduction_tex())
    write_text(SECTIONS / "forward_model.tex", forward_model())
    write_text(SECTIONS / "method.tex", method())
    write_text(SECTIONS / "measurement_families.tex", measurement_families())
    write_text(SECTIONS / "results.tex", results())
    write_text(SECTIONS / "validation.tex", validation())
    write_text(SECTIONS / "discussion.tex", discussion())
    ensure_dir(SUPP)
    write_text(SUPP / "supplement.tex", supplement())


def main() -> None:
    ensure_dir(LATEX)
    build_tables()
    write_sections()
    copy_figures_to_latex(LATEX / "figures")
    copy_tables_to_latex()
    write_text(LATEX / "main.tex", main_tex())
    write_text(LATEX / "references.bib", references_bib())
    write_text(OUT / "manuscript_v5.tex", main_tex())
    write_json(OUT / "internal_source_manifest.json", source_manifest())
    print({"latex_project_v5": str(LATEX), "main_tex": str(LATEX / "main.tex")})


if __name__ == "__main__":
    main()
