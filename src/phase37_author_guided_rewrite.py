from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.phase37_make_figures import main as make_phase37_figures


ROOT = Path("E:/ns_mc_gan_gi")
BASE_PROJECT = ROOT / "outputs_phase36_conventional_gi_aligned" / "latex_project_v36"
BASE_FIGURES = ROOT / "outputs_phase36_conventional_gi_aligned" / "figures"
OUT = ROOT / "outputs_phase37_author_guided_rewrite"
PROJECT = OUT / "latex_project_v37"
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

ABSTRACT = r"""
We study low-sampling ghost imaging and single-pixel imaging as a measurement-constrained reconstruction problem with known structured patterns and scalar bucket readings. Conventional GI forms a correlation image \(A^Ty\), while the low-sampling regime leaves many images compatible with the same measurements. Learned priors can improve visual quality, but they must not drift away from the bucket signal. We therefore use a regularized GI/SPI data solution as the measured-component anchor, insert a neural candidate residual through approximate null-space filtering, and apply a final measurement-consistency projection to audit the completed image. Under a leakage-free STL-10 protocol, the method reaches 22.316 dB PSNR / 0.635 SSIM at 5\% sampling with Rademacher measurements and 22.271 dB / 0.632 SSIM with scrambled Hadamard measurements. At 10\% sampling, the corresponding results are 24.781 dB / 0.747 SSIM and 24.730 dB / 0.746 SSIM. MNIST and Fashion-MNIST 5\% sanity checks reach 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM. Exact-operator audit, measurement-family attribution, inference ablation, perturbation and finite-noise diagnostics, and a CS-TV(PGD) comparison support the interpretation that reconstruction quality depends on both physical initialization and measured-signal-audited neural completion.
"""

INTRODUCTION = r"""
\section{Introduction}
Ghost imaging and single-pixel imaging reconstruct spatial information from known structured illumination patterns and scalar bucket readings. Each bucket measurement records a projection of the unknown image onto one pattern, so the reconstruction task is to recover an image from pattern-weighted scalar observations rather than from direct pixel measurements. This makes the measurement operator and the computational inverse inseparable: the chosen patterns determine both the data that are available and the ambiguities that remain.

Conventional GI can be viewed as bucket-pattern correlation. If the rows \(a_i^T\) of \(A\) denote the illumination patterns and \(y_i\) the bucket readings, the raw correlation or backprojection form is \(A^Ty=\sum_i y_i a_i\), up to centering and normalization. This expression is important because it keeps the reconstruction visibly tied to the measured buckets.

The difficulty is that low sampling is underdetermined. With \(x\in\mathbb{R}^n\), \(A\in\mathbb{R}^{m\times n}\), and \(m\ll n\), the measurement model \(y=Ax+\epsilon\) does not identify a unique image. A raw GI correlation image or a purely physical inverse can preserve measurement contact, but it usually leaves substantial missing structure.

Learned priors can supply missing image structure in this regime, but unconstrained learned reconstruction introduces a different risk. A network may improve perceptual or pixel metrics while producing details that are not supported by the measured bucket vector. The key question is therefore not only whether learning improves reconstruction quality, but whether the learned completion remains auditable against the measurements.

Our answer is not to replace GI correlation with an unconstrained image generator. Instead, we regularize the same bucket-pattern expansion to form a measured-component anchor, add a neural candidate residual only through an approximate null-space filter, and finally project the whole image back toward the measured affine set. This gives a reconstruction path from \(A^Ty\) to \(x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y\) to \(\Pi_y[x_{\rm data}+P_N(G_\theta)]\), making explicit which part comes from the measured operator and where learned completion enters.

The main contributions are:
\begin{itemize}
\item a regularized GI/SPI data solution as a measured-component anchor, explicitly related to bucket-pattern correlation;
\item a neural residual proposal inserted through approximate null-space filtering;
\item a final measurement-consistency projection for bucket-signal audit;
\item leakage-free evaluation and attribution separating physical initialization and neural gain.
\end{itemize}
"""

METHOD = r"""
\section{Measurement-Consistent Null-Space Reconstruction}
\subsection{From bucket-pattern correlation to regularized data solution}
Let \(a_i^T\) denote the \(i\)-th row of the measurement operator \(A\), and let \(y_i\) be the corresponding bucket reading. Ignoring centering and normalization, conventional GI correlation can be written as
\begin{equation}
\hat{x}_{\rm GI}=A^Ty=\sum_i y_i a_i.
\end{equation}
This raw backprojection uses the bucket readings directly as pattern weights.

The measured-component anchor used in this work keeps the same pattern-expansion form but replaces the raw bucket weights with decorrelated and regularized weights. We solve
\begin{equation}
q=(AA^T+\lambda I)^{-1}y,
\qquad
x_{\rm data}=A^Tq=\sum_i q_i a_i.
\end{equation}
Thus \(x_{\rm data}\) is a regularized and decorrelated GI/BP data solution, not a new standalone reconstructor. If \(AA^T\approx I\) and \(\lambda\to0\), then \(q\approx y\) and \(x_{\rm data}\approx A^Ty\). The role of \(x_{\rm data}\) is to provide a measured-component representative around which learned completion can be constrained.

\subsection{Candidate residual proposal}
The neural module predicts
\begin{equation}
r_\theta=G_\theta(x_{\rm data},z),
\end{equation}
where \(z\) denotes the optional latent or auxiliary input used by the implemented reconstructor. The raw network output is not interpreted as a final image, nor even as a directly supervised residual. It is a candidate residual proposal trained end-to-end through the subsequent physical operators. This distinction is important: the network proposes missing structure, but the physical operators determine how that proposal may enter the final reconstruction.

\subsection{Approximate null-space filtering}
Define \(P_A=A^T(AA^T+\lambda I)^{-1}A\) and \(P_N=I-P_A\). Applied to a vector \(v\),
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av.
\end{equation}
If \(\lambda=0\) and \(A\) has full row rank, then \(AP_N(v)=0\). With \(\lambda>0\), this becomes a regularized approximate null-space filter. The intermediate reconstruction is
\begin{equation}
\tilde{x}=x_{\rm data}+P_N(r_\theta).
\end{equation}
The filter reduces components of the proposed residual that would overwrite measured directions, so the learned module is encouraged to complete structure that is weakly observed or unobserved by \(A\).

\subsection{Measurement-consistency audit}
The final audit applies the measurement-consistency projection
\begin{equation}
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y).
\end{equation}
The final reconstruction is
\begin{equation}
\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
This projection is a whole-image audit, not only a residual correction. Although \(P_N\) filters the neural proposal, it is approximate when \(\lambda>0\), and subsequent refinement, numerical factorization, and intensity clipping can still introduce measurement inconsistency. The final \(\Pi_y\) step therefore checks the completed image against the bucket signal after learned completion has been inserted.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_minimal_mechanism_v37.pdf}
\caption{\textbf{From conventional GI correlation to measurement-audited neural completion.} Conventional GI forms an image by raw bucket-pattern correlation \(A^Ty\). The regularized data solution keeps the same pattern-expansion structure but uses decorrelated bucket weights \(q=(AA^T+\lambda I)^{-1}y\), giving \(x_{\rm data}=A^Tq\). The learned module proposes a candidate residual, \(P_N\) filters it before insertion, and \(\Pi_y\) audits the completed image against the original bucket measurements.}
\label{fig:mechanism}
\end{figure*}

\subsection{Two-stage implementation}
The implemented high-quality reconstructor uses a two-stage learned completion block. Stage 1 computes
\begin{equation}
\hat{x}^{(1)}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
A refiner predicts \(r_\phi=R_\phi(x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|)\), and the final output is
\begin{equation}
\hat{x}=\Pi_y[\hat{x}^{(1)}+r_\phi].
\end{equation}
The two-stage architecture is an implementation of the learned completion block; the conceptual contribution is the measured-component anchor, residual filtering, and final measurement audit. Image-domain metrics are computed after clipping to the valid intensity range, whereas measurement error is computed before clipping to avoid hiding projection inconsistency.

\subsection{Exact operator handling}
Rademacher sensing uses a random measurement matrix. Reproducible evaluation therefore requires reloading the exported exact operator. After replacing \(A\), all cached quantities derived from \(A\), including \(K=AA^T+\lambda I\) and its Cholesky factorization, must be rebuilt. This exact-A cache-rebuilt path is used for all reported Rademacher results.
"""

RESULTS = r"""
\section{Results}
\subsection{STL-10 reconstruction at 5\% and 10\%}
We first ask whether natural images can be reconstructed at the 5\% sampling level under a leakage-free protocol. \Cref{tab:primary_results,fig:primary_metrics} summarize the primary leakage-free evaluation results. At 5\% sampling, Rademacher measurements reach 22.316 dB PSNR and 0.635 SSIM, while scrambled Hadamard measurements reach 22.271 dB PSNR and 0.632 SSIM. Both exceed the predefined operational STL-10 5\% high-quality threshold. At 10\% sampling, Rademacher reaches 24.781 dB PSNR and 0.747 SSIM, while scrambled Hadamard reaches 24.730 dB PSNR and 0.746 SSIM. Thus, both measurement families support high-quality STL-10 reconstruction at 5\% and 10\% sampling.

\input{tables/table1_primary_results.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_primary_metrics_submission.pdf}
\caption{\textbf{Primary metrics.} PSNR and SSIM are shown for STL-10 5\%, STL-10 10\%, and simple-domain 5\% sanity checks. Dashed lines denote predefined operational thresholds used only to summarize reconstruction quality in this study.}
\label{fig:primary_metrics}
\end{figure*}

\subsection{Qualitative reconstruction}
Large qualitative reconstructions are shown in \Cref{fig:qualitative_reconstruction}. The visual comparison is meant to show what is recovered beyond the physical initialization: Rademacher backprojections are noise-like, scrambled Hadamard backprojections contain more structure, and the neural reconstruction restores object-level content. Images are enlarged for visibility and are intended as qualitative evidence; quantitative conclusions are based on the leakage-free evaluation metrics.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig3_qualitative_submission.pdf}
\caption{\textbf{Qualitative reconstruction.} STL-10 examples are shown as ground truth, backprojection, reconstruction, and absolute error. Representative evaluation samples were reselected for clearer object structure. Error maps use a shared high-percentile scale. The examples are qualitative visualizations; all quantitative conclusions are based on Table 1.}
\label{fig:qualitative_reconstruction}
\end{figure*}

\subsection{Similar final quality arises from different reconstruction regimes}
\Cref{tab:measurement_attribution,fig:measurement_attribution} separate physical initialization from learned refinement. Similar final PSNR does not mean similar sensing behavior. Rademacher measurements have weak physical backprojections: 7.297 dB at 5\% and 7.756 dB at 10\%. However, final reconstruction reaches 22.316 dB and 24.781 dB, corresponding to gains of 15.019 dB and 17.025 dB. Scrambled Hadamard measurements start from stronger backprojections, 14.310 dB at 5\% and 14.533 dB at 10\%, and reach nearly the same final quality as Rademacher.

\input{tables/table2_measurement_attribution.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig4_measurement_attribution_v36.pdf}
\caption{\textbf{Measurement attribution.} The regime map uses backprojection PSNR and model gain, reported as neural gain, to separate physical-initialization quality from learned refinement. Low-frequency Hadamard points are shown as hollow diagnostic controls and are not primary STL-10 claims.}
\label{fig:measurement_attribution}
\end{figure*}

\subsection{Simple-domain sanity checks}
On MNIST and Fashion-MNIST at 5\% sampling, the method reaches 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM, respectively. These experiments confirm that the same reconstruction pipeline works reliably on simpler structured domains, but they are not the main novelty.
"""

VALIDATION = r"""
\section{Validation and Ablation}
\subsection{Is random sensing reproducible?}
Yes, provided that exact-operator evaluation is used. Rademacher measurements require the exported exact operator rather than a regenerated nominally identical random matrix. Earlier mismatch was traced to stale solver-cache use after overriding \(A\). With safe exact-A loading and cache rebuilding, Rademacher 5\% and 10\% re-evaluations reproduce the original leakage-free evaluation with negligible differences. These reproduced results are used as primary evidence.

\subsection{Is the final measurement audit necessary?}
\Cref{tab:ablation_summary,fig:inference_ablation} report the inference-time ablations. Removing the measurement-consistency projection causes the largest degradation. This no measurement-consistency projection condition shows that \(\Pi_y\) is not merely cosmetic; it is central to maintaining physical fidelity and image quality. Removing the null projection has limited metric effect for the trained checkpoints, suggesting that the final projection and the learned network already constrain many measured components. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.

\input{tables/table3_ablation_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation_submission.pdf}
\caption{\textbf{Inference ablation.} Removing measurement-consistency projection gives the strongest degradation. \(-\mathrm{MC}\) removes the final measurement-consistency projection. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.}
\label{fig:inference_ablation}
\end{figure*}

\subsection{Does the network depend on the bucket vector?}
The perturbation diagnostics indicate that it does. \Cref{fig:validation_summary} summarizes finite-noise sweeps, measurement perturbations, CS-TV(PGD) comparison, and bootstrap confidence intervals. Finite-noise sweeps show stable degradation over the tested noise range. Measurement perturbation tests are more diagnostic: shuffled coefficients and wrong-sample measurements cause large PSNR drops. This is a negative-control test: the model should fail when the measurement vector is corrupted. The result indicates that the model depends on the bucket measurement vector rather than measurement-independent hallucination.

\subsection{Is the method stronger than a classical CSGI-style prior?}
We compare against a TV-regularized compressed-sensing baseline solved by projected gradient descent (CS-TV):
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\operatorname{TV}(x).
\end{equation}
This baseline represents a classical compressed-sensing prior, not an exhaustively tuned iterative reconstruction benchmark. GI/BP denotes the linear physical backprojection or correlation-like GI reconstruction. A selected visual comparison is provided in the Supplement.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_robustness_baselines_submission.pdf}
\caption{\textbf{Robustness and baselines.} These diagnostics support finite-noise stability and measurement dependence within the tested conditions; they do not imply universal robustness.}
\label{fig:validation_summary}
\end{figure*}

\subsection{Stability diagnostics}
For low-frequency Hadamard, including the DC row is critical for backprojection because it captures global brightness. Removing it severely degrades low-frequency Hadamard initialization. The DC-row result explains why low-frequency Hadamard backprojection behaves differently from Rademacher and scrambled Hadamard. It is a diagnostic of one measurement family, not a general explanation of all reconstructions. Bootstrap confidence intervals show that the main results are stable across samples. Class-wise STL-10 diagnostics reveal expected category variability, with some classes consistently more difficult than others. These analyses support the robustness of the main claim but are not themselves the central contribution.
"""

LIMITATIONS = r"""
\section{Limitations}
This study reports simulation/dataset-based evaluations of the reconstruction framework under the computational forward model \(y=Ax+\epsilon\). We do not claim a ranking over external benchmarks because datasets, measurement operators, sampling protocols, and evaluation splits are not standardized across the literature. The CS-TV(PGD) baseline is a CSGI-style lightweight small-subset compressed-sensing control, not an exhaustively optimized compressed-sensing solver. Robustness is tested only over finite noise and perturbation settings. Class-wise evaluation is diagnostic rather than a claim of uniform category performance. Exact-A handling is essential for random measurements, and results should be interpreted with that audit path in place. The 5\% low-frequency Hadamard condition is retained only as a diagnostic control rather than as a primary STL-10 claim. Auxiliary illumination-learning attempts are not used as claimed final results, and adversarial training is not the final contribution mechanism. Future work should include physical-system validation, broader external baselines, and more extensive cross-domain testing.
"""


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


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
                    if path.name.startswith("fig4_measurement_attribution_v36") and path.parent != FIG_DIR:
                        shutil.copy2(path, FIG_DIR / path.name)


def fix_tv_formula() -> None:
    for path in list(PROJECT.rglob("*.tex")) + [PROJECT / "references.bib"]:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            text = text.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
            text = text.replace(
                "All reported results are obtained under the computational forward model \\(y=Ax+\\epsilon\\). No hardware optical experiment is included in this study.",
                "All reported results are obtained under the computational forward model \\(y=Ax+\\epsilon\\).",
            )
            path.write_text(text, encoding="utf-8")


def write_sources() -> None:
    write(PROJECT / "sections" / "abstract.tex", ABSTRACT)
    write(PROJECT / "sections" / "introduction.tex", INTRODUCTION)
    write(PROJECT / "sections" / "method.tex", METHOD)
    write(PROJECT / "sections" / "results.tex", RESULTS)
    write(PROJECT / "sections" / "validation_ablation.tex", VALIDATION)
    write(PROJECT / "sections" / "limitations.tex", LIMITATIONS)
    fix_tv_formula()


def compile_pdf(filename: str) -> None:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", filename],
        cwd=PROJECT,
        check=True,
    )


def copy_outputs() -> None:
    shutil.copy2(PROJECT / "main.pdf", OUT / "main_v37.pdf")
    shutil.copy2(PROJECT / "supplement.pdf", OUT / "supplement_v37.pdf")
    audit = BASE_PROJECT / "citation_audit_phase36.md"
    if audit.exists():
        shutil.copy2(audit, OUT / "citation_audit_phase37.md")


def main() -> None:
    make_phase37_figures()
    copy_project()
    copy_figures()
    write_sources()
    compile_pdf("main.tex")
    compile_pdf("supplement.tex")
    copy_outputs()
    print(
        {
            "project": str(PROJECT),
            "main_pdf": str(OUT / "main_v37.pdf"),
            "supplement_pdf": str(OUT / "supplement_v37.pdf"),
            "figure1": str(FIG_DIR / "fig1_minimal_mechanism_v37.pdf"),
        }
    )


if __name__ == "__main__":
    main()
