from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
BASE_OUT = ROOT / "outputs_phase38_professional_figure"
BASE_PROJECT = BASE_OUT / "latex_project_v38"
OUT = ROOT / "outputs_phase39_anchor_proposal_audit"
PROJECT = OUT / "latex_project_v39"
FIG_DIR = OUT / "figures"

COMPILED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
    ".toc",
}


ABSTRACT = r"""Low-sampling ghost imaging is underdetermined: many images can match the same bucket measurements, so reconstruction is better viewed as measurement-constrained completion than as an unconstrained inverse problem. We use a measured anchor \(x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y\), let the network propose missing structure, filter the proposal through \(P_N\), and finally audit the completed image with \(\Pi_y\) against the bucket signal. The network is therefore not the final judge; the measurement operator decides what can remain. Under a leakage-free STL-10 protocol, the method reaches 22.316 dB PSNR / 0.635 SSIM at 5\% sampling with Rademacher measurements and 22.271 dB / 0.632 SSIM with scrambled Hadamard measurements. At 10\% sampling, the corresponding results are 24.781 dB / 0.747 SSIM and 24.730 dB / 0.746 SSIM. Exact-operator audit, inference ablation, measurement perturbation, and finite-noise diagnostics support the interpretation that reconstruction quality depends on measured-signal-audited neural completion rather than measurement-independent image synthesis."""


INTRODUCTION = r"""\section{Introduction}
Ghost imaging and single-pixel imaging reconstruct spatial information from known structured illumination patterns and scalar bucket readings. Each bucket measurement records a projection of the unknown image onto one pattern. In the low-sampling regime, \(A\in\mathbb{R}^{m\times n}\) has \(m\ll n\), so \(y=Ax+\epsilon\) leaves many images compatible with the same measurements.

This ambiguity is the central difficulty. A purely physical initialization stays tied to the measured buckets but often misses object-level structure. Learned priors can fill in that missing structure, but they introduce a risk: a network can make the image sharper while adding details that the bucket vector does not support.

The key question is not whether a neural network can make the image sharper, but whether it can add structure only where the measurements are silent. The network proposes; the measurement operator decides what can remain.

We therefore formulate reconstruction as audited completion. First, a regularized data solution \(x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y\) provides a measured anchor. Second, \(G_\theta\) proposes missing structure around that anchor. Third, \(P_N\) acts as a legality filter by suppressing residual components visible to the measurement operator. Finally, \(\Pi_y\) audits the completed image by rechecking it against the original bucket readings.

The main contributions are:
\begin{itemize}
\item an anchor-proposal-audit interpretation of low-sampling GI/SPI reconstruction;
\item a measured anchor that keeps the reconstruction tied to bucket evidence;
\item a neural proposal constrained by a legality filter and final bucket audit;
\item leakage-free evaluation and diagnostics showing measurement-dependent reconstruction.
\end{itemize}
"""


METHOD = r"""\section{Measurement-Consistent Null-Space Reconstruction}
\subsection{The measurement set: low sampling leaves an affine ambiguity}
Let \(A\in\mathbb{R}^{m\times n}\) be the measurement operator and \(y\in\mathbb{R}^m\) the bucket vector. In the low-sampling regime \(m\ll n\), the set
\begin{equation}
\mathcal{C}_y=\{x:Ax=y\}
\end{equation}
is large when noise and numerical tolerance are ignored. Reconstruction therefore cannot be an arbitrary-image inverse; it must choose a plausible image while preserving the measured signal.

\subsection{Data anchor: a measured point near the feasible set}
The measured-component anchor is
\begin{equation}
q=(AA^T+\lambda I)^{-1}y,
\qquad
x_{\rm data}=A^Tq=\sum_i q_i a_i.
\end{equation}
This image is incomplete, but it is tied to the bucket measurements and the known patterns.

\paragraph{Relation to conventional bucket-pattern correlation.}
Conventional bucket-pattern correlation can be written, up to centering and normalization, as
\begin{equation}
\hat{x}_{\rm GI}=A^Ty=\sum_i y_i a_i.
\end{equation}
The anchor keeps the same pattern-expansion structure but replaces raw bucket weights with decorrelated coefficients:
\begin{equation}
A^Ty \rightarrow A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
Thus \(x_{\rm data}\) is a regularized data anchor, not a new standalone reconstructor.

\subsection{Neural proposal: learned missing structure}
The neural module predicts
\begin{equation}
r_\theta=G_\theta(x_{\rm data},z),
\end{equation}
where \(z\) denotes the optional latent or auxiliary input used by the implemented reconstructor. The neural network is not trained to be the final judge. It proposes a candidate missing component whose usefulness is determined after filtering and audit.

\subsection{Legality filter: inserting only measurement-silent residuals}
Define \(P_A=A^T(AA^T+\lambda I)^{-1}A\) and \(P_N=I-P_A\). Applied to a vector \(v\),
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av.
\end{equation}
If \(\lambda=0\) and \(A\) has full row rank, then \(AP_N(v)=0\). With \(\lambda>0\), this becomes a regularized approximate null-space filter. \(P_N\) is best interpreted as a legality filter: it suppresses residual components that the measurement operator can see. The intermediate image is
\begin{equation}
\tilde{x}=x_{\rm data}+P_N(r_\theta).
\end{equation}

\subsection{Bucket audit: projecting the completed image back to the data}
The final audit applies the measurement-consistency projection
\begin{equation}
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y).
\end{equation}
\(\Pi_y\) closes the loop by remeasuring the completed image through \(A\) and correcting the discrepancy with \(y\). The final reconstruction is
\begin{equation}
\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
This projection is a whole-image audit, not only a residual correction.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_anchor_proposal_audit_v39.pdf}
\caption{\textbf{Mechanism of measurement-audited neural completion.} Low-sampling measurements define many feasible images rather than a unique inverse. The reconstruction first forms a measured anchor \(x_{\rm data}\). The network proposes missing structure, but this proposal is filtered before insertion and the completed image is finally audited against the bucket measurements. The bottom strip uses real STL-10 examples to show that the final audited output restores object-level structure while remaining tied to the measured signal.}
\label{fig:mechanism}
\end{figure*}

\subsection{Implementation details and exact-operator handling}
The implemented high-quality reconstructor uses a two-stage learned completion block. Stage 1 computes
\begin{equation}
\hat{x}^{(1)}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
A refiner predicts \(r_\phi=R_\phi(x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|)\), and the final output is
\begin{equation}
\hat{x}=\Pi_y[\hat{x}^{(1)}+r_\phi].
\end{equation}
The two-stage architecture is an implementation detail of the learned proposal block; the conceptual path remains measured anchor, neural proposal, legality filter, and final bucket audit. Image-domain metrics are computed after clipping to the valid intensity range, whereas measurement error is computed before clipping to avoid hiding projection inconsistency.

Rademacher sensing uses a random measurement matrix. Reproducible evaluation therefore requires reloading the exported exact operator. After replacing \(A\), all cached quantities derived from \(A\), including \(K=AA^T+\lambda I\) and its Cholesky factorization, must be rebuilt. This exact-A cache-rebuilt path is used for all reported Rademacher results.
"""


RESULTS = r"""\section{Results}
\subsection{Primary STL-10 performance}
We first ask whether natural images can be reconstructed at the 5\% sampling level under a leakage-free protocol. \Cref{tab:primary_results,fig:primary_metrics} summarize the primary leakage-free evaluation results. At 5\% sampling, Rademacher measurements reach 22.316 dB PSNR and 0.635 SSIM, while scrambled Hadamard measurements reach 22.271 dB PSNR and 0.632 SSIM. At 10\% sampling, Rademacher reaches 24.781 dB PSNR and 0.747 SSIM, while scrambled Hadamard reaches 24.730 dB PSNR and 0.746 SSIM.

\input{tables/table1_primary_results.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_primary_metrics_submission.pdf}
\caption{\textbf{Primary metrics.} PSNR and SSIM are shown for STL-10 5\%, STL-10 10\%, and simple-domain 5\% sanity checks. Dashed lines denote predefined operational thresholds used only to summarize reconstruction quality in this study.}
\label{fig:primary_metrics}
\end{figure*}

\subsection{Mechanism in images: anchor, proposal, audit}
\Cref{fig:mechanism} shows the reconstruction mechanism using real STL-10 examples. The measured anchor preserves bucket-tied evidence but is incomplete. The no-audit column shows the failure mode that appears when the neural proposal is not finally checked against the bucket signal. The final audited output restores object-level structure while maintaining measurement consistency. Larger qualitative examples are shown in \Cref{fig:qualitative_reconstruction}.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig3_qualitative_submission.pdf}
\caption{\textbf{Qualitative reconstruction.} STL-10 examples are shown as ground truth, backprojection, reconstruction, and absolute error. Representative evaluation samples were reselected for clearer object structure. Error maps use a shared high-percentile scale. The examples are qualitative visualizations; all quantitative conclusions are based on Table 1.}
\label{fig:qualitative_reconstruction}
\end{figure*}

\subsection{Similar final quality, different reconstruction regimes}
\Cref{tab:measurement_attribution,fig:measurement_attribution} separate physical initialization from learned completion. Similar final PSNR does not imply similar sensing behavior. Rademacher measurements have weak physical backprojections: 7.297 dB at 5\% and 7.756 dB at 10\%. However, final reconstruction reaches 22.316 dB and 24.781 dB, corresponding to gains of 15.019 dB and 17.025 dB. Scrambled Hadamard measurements start from stronger backprojections, 14.310 dB at 5\% and 14.533 dB at 10\%, and reach nearly the same final quality as Rademacher.

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


VALIDATION = r"""\section{Validation and Ablation}
The validation experiments are organized around failure modes of the proposed mechanism.

\subsection{Random operators must be exactly audited}
Rademacher measurements require the exported exact operator rather than a regenerated nominally identical random matrix. Earlier mismatch was traced to stale solver-cache use after overriding \(A\). With safe exact-A loading and cache rebuilding, Rademacher 5\% and 10\% re-evaluations reproduce the original leakage-free evaluation with negligible differences. These reproduced results are used as primary evidence.

\subsection{Removing the audit breaks the mechanism}
\Cref{tab:ablation_summary,fig:inference_ablation} report the inference-time ablations. Removing the measurement-consistency projection causes the largest degradation. This no measurement-consistency projection condition shows that \(\Pi_y\) is not merely cosmetic; it is central to maintaining physical fidelity and image quality. Removing the null projection has limited metric effect for the trained checkpoints, suggesting that the final projection and the learned network already constrain many measured components. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.

\input{tables/table3_ablation_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation_submission.pdf}
\caption{\textbf{Inference ablation.} Removing measurement-consistency projection gives the strongest degradation. \(-\mathrm{MC}\) removes the final measurement-consistency projection. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.}
\label{fig:inference_ablation}
\end{figure*}

\subsection{Corrupting the bucket vector breaks the reconstruction}
The perturbation diagnostics indicate that the reconstruction depends on the bucket vector. \Cref{fig:validation_summary} summarizes finite-noise sweeps, measurement perturbations, CS-TV(PGD) comparison, and bootstrap confidence intervals. Finite-noise sweeps show stable degradation over the tested noise range. Measurement perturbation tests are more diagnostic: shuffled coefficients and wrong-sample measurements cause large PSNR drops. This negative-control test indicates that the model depends on the bucket measurement vector rather than measurement-independent hallucination.

\subsection{Classical CS-TV prior is not enough}
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

\subsection{Stability and diagnostic controls}
For low-frequency Hadamard, including the DC row is critical for backprojection because it captures global brightness. Removing it severely degrades low-frequency Hadamard initialization. The DC-row result explains why low-frequency Hadamard backprojection behaves differently from Rademacher and scrambled Hadamard. It is a diagnostic of one measurement family, not a general explanation of all reconstructions. Bootstrap confidence intervals show that the main results are stable across samples. Class-wise STL-10 diagnostics reveal expected category variability, with some classes consistently more difficult than others. These analyses support the robustness of the main claim but are not themselves the central contribution.
"""


def prepare_project() -> None:
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    shutil.copytree(BASE_PROJECT, PROJECT)
    for path in PROJECT.rglob("*"):
        if path.is_file() and path.suffix in COMPILED_SUFFIXES:
            path.unlink()


def copy_figures() -> None:
    dst = PROJECT / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for source_dir in [BASE_PROJECT / "figures", BASE_OUT / "figures", FIG_DIR]:
        if source_dir.exists():
            for path in source_dir.iterdir():
                if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg"}:
                    shutil.copy2(path, dst / path.name)


def clean_tex(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
    text = re.sub(r"future hardware validation", "physical-system validation", text, flags=re.IGNORECASE)
    path.write_text(text, encoding="utf-8")


def write_sections() -> None:
    section_dir = PROJECT / "sections"
    (section_dir / "abstract.tex").write_text(ABSTRACT + "\n", encoding="utf-8")
    (section_dir / "introduction.tex").write_text(INTRODUCTION + "\n", encoding="utf-8")
    (section_dir / "method.tex").write_text(METHOD + "\n", encoding="utf-8")
    (section_dir / "results.tex").write_text(RESULTS + "\n", encoding="utf-8")
    (section_dir / "validation_ablation.tex").write_text(VALIDATION + "\n", encoding="utf-8")
    for path in PROJECT.rglob("*.tex"):
        clean_tex(path)


def run_latex(target: str) -> None:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", target],
        cwd=PROJECT,
        check=True,
    )


def copy_outputs() -> None:
    shutil.copy2(PROJECT / "main.pdf", OUT / "main_v39.pdf")
    shutil.copy2(PROJECT / "supplement.pdf", OUT / "supplement_v39.pdf")
    audit = BASE_OUT / "citation_audit_phase38.md"
    if audit.exists():
        shutil.copy2(audit, OUT / "citation_audit_phase39.md")
    for pdf_name, txt_name in [("main_v39.pdf", "main_v39.txt"), ("supplement_v39.pdf", "supplement_v39.txt")]:
        try:
            subprocess.run(["pdftotext", str(OUT / pdf_name), str(OUT / txt_name)], check=False)
        except FileNotFoundError:
            pass


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_project()
    copy_figures()
    write_sections()
    run_latex("main.tex")
    run_latex("supplement.tex")
    copy_outputs()
    print(
        json.dumps(
            {
                "project": str(PROJECT),
                "main_pdf": str(OUT / "main_v39.pdf"),
                "supplement_pdf": str(OUT / "supplement_v39.pdf"),
                "figure1": str(FIG_DIR / "fig1_anchor_proposal_audit_v39.pdf"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
