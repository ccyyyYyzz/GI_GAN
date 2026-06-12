from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
BASE_OUT = ROOT / "outputs_phase43_operator_circuit"
BASE_PROJECT = BASE_OUT / "latex_project_v43"
OUT = ROOT / "outputs_phase44_operator_centered"
PROJECT = OUT / "latex_project_v44"
FIG_DIR = OUT / "figures"
PROV = OUT / "provenance"

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


ABSTRACT = r"""Computational ghost imaging at low sampling is underdetermined: bucket measurements constrain an image but leave a large null space. This makes learned priors useful, but also makes them dangerous, because a network can sharpen a reconstruction by adding structure unsupported by the measurements. We present measurement-audited neural completion, an operator-centered reconstruction circuit for computational ghost imaging. A single calibrated inverse \(B_\lambda=A^{\mathsf T}(AA^{\mathsf T}+\lambda I)^{-1}\) is reused in three roles. It anchors the data as \(x_{\rm data}=B_\lambda y\), gates the neural proposal through \(P_N=I-B_\lambda A\), and audits the final image through \(\Pi_y(v)=v-B_\lambda(Av-y)\). The network proposes; the calibrated operator anchors, gates, and audits. This framing makes measurement consistency the organizing structure of both the reconstructor and its diagnostics. Under a leakage-free STL-10 protocol, the method reaches 22.316 dB / 0.635 SSIM and 22.271 dB / 0.632 SSIM at 5\% sampling for Rademacher and scrambled Hadamard measurements, and 24.781 dB / 0.747 and 24.730 dB / 0.746 at 10\%. Exact-A audit, ablations, measurement perturbations, and CS-TV comparisons support measurement-dependent completion rather than measurement-independent sharpening."""


INTRODUCTION = r"""\section{Introduction}
Computational ghost imaging and single-pixel imaging recover spatial structure from known illumination patterns and scalar bucket measurements. With a vectorized image \(x\in\mathbb{R}^n\), the forward model is
\begin{equation}
y=Ax+\epsilon,
\end{equation}
where \(A\in\mathbb{R}^{m\times n}\) contains the projected patterns and \(y\in\mathbb{R}^m\) is the bucket vector. In low-sampling regimes \(m\ll n\), the measurements constrain the image but do not uniquely determine it.

The central challenge is not merely to make the image sharper, but to let the network add only structure that the measurements cannot see. A physical backprojection is tied to the buckets but usually lacks object-level detail. A learned prior can fill missing structure, yet it can also create plausible details that are weakly tied to the measured bucket vector.

We therefore frame reconstruction as measurement-audited neural completion. A single calibrated inverse
\begin{equation}
B_\lambda=A^{\mathsf T}(AA^{\mathsf T}+\lambda I)^{-1}
\end{equation}
is reused throughout the reconstruction circuit. The same calibrated operator that forms the data anchor also gates the neural proposal and audits the final image. It constructs \(x_{\rm data}=B_\lambda y\), defines \(P_N=I-B_\lambda A\), and corrects a candidate image with \(\Pi_y(v)=v-B_\lambda(Av-y)\). The network proposes; the calibrated operator anchors, gates, and audits.

This operator-centered view separates two questions that are often mixed together: what the measurements directly support, and what the learned prior contributes in the regularized complement. The endpoint PSNR can be similar for different measurement families even when their physical anchors and learned gains differ substantially.

The main contributions are:
\begin{itemize}
\item measurement-audited null-space completion around one calibrated operator \(B_\lambda\);
\item an operator-centered reconstruction circuit where \(B_\lambda\) acts as anchor, gate, and audit;
\item a strict evaluation package: exact-A audit, audit ablation, wrong-\(y\)/shuffle perturbation, and CS-TV baseline;
\item attribution analysis showing similar PSNR can hide different physical-anchor / learned-completion splits.
\end{itemize}
"""


METHOD = r"""\section{Measurement-Audited Neural Completion}
\subsection{Measurement model and information split}
Let \(A\in\mathbb{R}^{m\times n}\) be the calibrated measurement operator and \(y\in\mathbb{R}^m\) the bucket vector. For low-sampling ghost imaging, \(m<n\), so arbitrary-image recovery is impossible without assumptions beyond the measurements. The affine set
\begin{equation}
\mathcal{C}_y=\{x:Ax=y\}
\end{equation}
contains many candidates when noise and numerical tolerance are ignored. The reconstruction problem is therefore an information split: measured components should remain anchored to \(A\) and \(y\), while learned structure should enter only through directions weakly visible to the measurements.

\subsection{Calibrated inverse core}
The circuit is built around one regularized inverse core,
\begin{equation}
B_\lambda=A^{\mathsf T}(AA^{\mathsf T}+\lambda I)^{-1}.
\end{equation}
This operator is not a learned network layer. It is a frozen physics-derived map from bucket space back to image space. The same \(B_\lambda\) is reused for anchoring, gating, and auditing, which makes the reconstruction path explicitly operator-centered.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_operator_centered_v44.pdf}
\caption{\textbf{Physics-gated and audited neural completion circuit.} A single calibrated physical operator \(B_\lambda=A^{\mathsf T}(AA^{\mathsf T}+\lambda I)^{-1}\) is reused in three roles. Anchor: \(x_{\rm data}=B_\lambda y\) fixes the measured component. Gate: the neural proposal \(r_\theta=G_\theta(x_{\rm data},z)\) enters only through \(P_N=I-B_\lambda A\), which suppresses components visible to the measurement operator. Audit: the candidate image is remeasured by \(A\), compared with the original bucket vector \(y\), and corrected as \(\hat{x}=\tilde{x}-B_\lambda(A\tilde{x}-y)\). The network proposes missing structure, while the calibrated measurement operator decides what can remain.}
\label{fig:mechanism}
\end{figure*}

\subsection{Anchor: measured component}
The data anchor is
\begin{equation}
x_{\rm data}=B_\lambda y.
\end{equation}
This image is not expected to be visually complete. Its role is to provide the measurement-supported component around which the learned completion is built.

\subsection{Gate: null-space-confined proposal}
The neural module predicts a residual proposal
\begin{equation}
r_\theta=G_\theta(x_{\rm data},z),
\end{equation}
where \(z\) denotes the auxiliary input used by the implemented reconstructor. The residual is admitted through
\begin{equation}
r_N=P_N r_\theta,\qquad P_N=I-B_\lambda A.
\end{equation}
For \(\lambda=0\) and full row rank \(A\), this is an exact null-space projection. With \(\lambda>0\), it is a regularized soft gate that suppresses proposal components visible to the measurement operator. The pre-audit candidate is
\begin{equation}
\tilde{x}=x_{\rm data}+r_N.
\end{equation}

\subsection{Audit: remeasure and correct}
The audit remeasures the candidate and corrects the discrepancy:
\begin{equation}
\tilde{y}=A\tilde{x},\qquad e_y=\tilde{y}-y,
\end{equation}
\begin{equation}
\hat{x}=\tilde{x}-B_\lambda e_y
       =\tilde{x}-B_\lambda(A\tilde{x}-y).
\end{equation}
Equivalently,
\begin{equation}
\Pi_y(v)=v-B_\lambda(Av-y),
\qquad
\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
The audit is not cosmetic postprocessing: the final image is explicitly rechecked against the bucket vector.

\subsection{Training through frozen physics}
Training losses are applied after the frozen physics layers. Image-domain losses compare \(\hat{x}\) to the target image, and measurement losses compare \(A\hat{x}\) to \(y\). Gradients pass through \(A\), \(B_\lambda\), \(P_N\), and \(\Pi_y\), but these physics operators are fixed; only the neural proposal modules are updated.

\subsection{Two-stage implementation and exact operator handling}
The reported high-quality reconstructor uses a two-stage proposal block. Stage one computes
\begin{equation}
\hat{x}^{(1)}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
A refiner then proposes \(r_\phi=R_\phi(x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|)\), and the final output is
\begin{equation}
\hat{x}=\Pi_y[\hat{x}^{(1)}+r_\phi].
\end{equation}
This second stage changes the proposal module, not the operator-centered logic. Image metrics are computed after clipping to the valid intensity range, while measurement error is computed before clipping so that projection inconsistency is not hidden.

Rademacher sensing uses a random measurement matrix. Reproducible evaluation therefore reloads the exported exact operator. After replacing \(A\), all quantities derived from \(A\), including \(AA^{\mathsf T}+\lambda I\) and its Cholesky factorization, are rebuilt. This exact-A path is used for all reported Rademacher results.

\subsection{Relation to conventional bucket-pattern correlation}
Conventional bucket-pattern correlation can be written, up to centering and normalization, as
\begin{equation}
\hat{x}_{\rm GI}=A^{\mathsf T}y=\sum_i y_i a_i.
\end{equation}
The anchor keeps the same pattern-expansion direction but replaces raw bucket weights with decorrelated coefficients:
\begin{equation}
A^{\mathsf T}y \rightarrow A^{\mathsf T}(AA^{\mathsf T}+\lambda I)^{-1}y.
\end{equation}
Thus \(x_{\rm data}\) is a regularized/decorrelated GI-style data solution. The contribution is the full operator-centered circuit in which the same calibrated inverse anchors the data, gates the learned proposal, and audits the completed image.
"""


RESULTS = r"""\section{Results}
\subsection{Mechanism walkthrough and provenance}
\Cref{fig:mechanism} summarizes the reconstruction circuit. The same calibrated inverse \(B_\lambda\) is used as data anchor, proposal gate, and final audit. To visualize the division between measurement-supported and learned-complement structure, \Cref{fig:provenance} decomposes one representative reconstruction per primary STL-10 setting into \(B_\lambda A\hat{x}\) and \((I-B_\lambda A)\hat{x}\). Because \(B_\lambda\) is regularized, this is a soft decomposition rather than a hard orthogonal split, but it makes the sensing/learning division visible.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/provenance_grid.pdf}
\caption{\textbf{Regularized provenance decomposition.} Each row shows ground truth, final audited reconstruction, measured component \(B_\lambda A\hat{x}\), learned/null component \((I-B_\lambda A)\hat{x}\), and absolute error for one representative sample. This diagnostic is eval-only and is not used to change the reported main metrics.}
\label{fig:provenance}
\end{figure*}

\subsection{Primary STL-10 performance}
\Cref{tab:primary_results,fig:primary_metrics} summarize the primary leakage-free evaluation results. At 5\% sampling, Rademacher measurements reach 22.316 dB PSNR and 0.635 SSIM, while scrambled Hadamard measurements reach 22.271 dB PSNR and 0.632 SSIM. At 10\% sampling, Rademacher reaches 24.781 dB PSNR and 0.747 SSIM, while scrambled Hadamard reaches 24.730 dB PSNR and 0.746 SSIM.

\input{tables/table1_primary_results.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_primary_metrics_submission.pdf}
\caption{\textbf{Primary metrics.} PSNR and SSIM are shown for STL-10 5\%, STL-10 10\%, and simple-domain 5\% sanity checks. Dashed lines denote predefined operational thresholds used only to summarize reconstruction quality in this study.}
\label{fig:primary_metrics}
\end{figure*}

\subsection{Audit and perturbation as mechanism validation}
\Cref{tab:ablation_summary,fig:inference_ablation} report inference-time ablations. Removing measurement-consistency projection gives the strongest degradation. The \(-\mathrm{MC}\) condition removes the final measurement-consistency projection, showing that \(\Pi_y\) is not merely cosmetic. Perturbation diagnostics in \Cref{fig:validation_summary} provide a complementary negative control: shuffled coefficients and wrong-sample bucket vectors cause large PSNR drops, indicating that reconstruction depends on the measured bucket vector.

\input{tables/table3_ablation_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation_submission.pdf}
\caption{\textbf{Inference ablation.} Removing measurement-consistency projection gives the strongest degradation. \(-\mathrm{MC}\) removes the final measurement-consistency projection. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.}
\label{fig:inference_ablation}
\end{figure*}

\subsection{Similar final quality, different anchor/gain regimes}
\Cref{tab:measurement_attribution,fig:measurement_attribution} separate physical initialization from learned completion. Similar final PSNR does not imply similar sensing behavior. Rademacher measurements have weak physical anchors: 7.297 dB at 5\% and 7.756 dB at 10\%. Final reconstruction reaches 22.316 dB and 24.781 dB, corresponding to gains of 15.019 dB and 17.025 dB. Scrambled Hadamard measurements start from stronger anchors, 14.310 dB at 5\% and 14.533 dB at 10\%, and reach nearly the same final quality as Rademacher.

\input{tables/table2_measurement_attribution.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig4_measurement_attribution_v36.pdf}
\caption{\textbf{Measurement attribution.} The regime map uses backprojection PSNR and \(\Delta\mathrm{PSNR}\) to separate physical-initialization quality from learned refinement. Low-frequency Hadamard points are hollow diagnostic controls and are not primary STL-10 claims.}
\label{fig:measurement_attribution}
\end{figure*}

\subsection{Simple-domain sanity checks}
On MNIST and Fashion-MNIST at 5\% sampling, the method reaches 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM, respectively. These experiments confirm that the same reconstruction pipeline works reliably on simpler structured domains, but they are not the main novelty.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig3_qualitative_submission.pdf}
\caption{\textbf{Qualitative reconstruction.} STL-10 examples are shown as ground truth, backprojection, reconstruction, and absolute error. Representative evaluation samples were selected for clear object structure. Error maps use a shared high-percentile scale. Quantitative conclusions are based on the tables.}
\label{fig:qualitative_reconstruction}
\end{figure*}
"""


DISCUSSION = r"""\section{Discussion}
Low-sampling ghost imaging is a negotiation between measured evidence and learned prior. The calibrated operator supplies the evidence: it forms the anchor, gates the proposal, and audits the final image. The learned prior fills only the regularized complement that the measurements do not determine. This is why endpoint PSNR alone is incomplete; similar final quality can hide different sensing/learning splits.

The audit is not a cosmetic correction. Removing measurement-consistency projection causes the strongest inference-time degradation, and corrupting the bucket vector breaks reconstruction quality. These diagnostics support the interpretation that the network is not simply sharpening images independently of the measurements.

The provenance decomposition further clarifies the mechanism. \(B_\lambda A\hat{x}\) visualizes the operator-recoverable component of the final reconstruction, while \((I-B_\lambda A)\hat{x}\) visualizes the regularized learned complement. The decomposition is soft because \(B_\lambda\) is regularized, but it makes clear why measurement geometry and learned prior must be studied together.

The present evidence is simulation- and dataset-based under calibrated computational forward models. Hardware validation remains future work, as do broader external baselines and more extensive cross-domain tests. The central claim here is therefore mechanism-level: a frozen calibrated operator can anchor, gate, and audit neural completion in low-sampling computational ghost imaging.
"""


VALIDATION_ABLATION = r"""\section{Validation and Ablation}
The validation experiments are organized around failure modes of the proposed mechanism. Exact-A re-evaluation checks that random Rademacher measurements are audited with the exported operator and rebuilt solver cache. The inference ablation reported in \Cref{tab:ablation_summary,fig:inference_ablation} tests whether removing the measurement-consistency projection damages reconstruction quality. Bucket perturbations test whether the model depends on the measured \(y\) rather than producing measurement-independent sharpening.

\subsection{Random operators must be exactly audited}
Rademacher measurements require the exported exact operator rather than a regenerated nominally identical random matrix. With safe exact-A loading and cache rebuilding, Rademacher 5\% and 10\% re-evaluations reproduce the leakage-free evaluation with negligible differences.

\subsection{Corrupting the bucket vector breaks the reconstruction}
The perturbation diagnostics indicate that the reconstruction depends on the bucket vector. Shuffled coefficients and wrong-sample measurements cause large PSNR drops. This negative-control test indicates measurement-dependent completion rather than measurement-independent sharpening.

\subsection{Classical CS-TV prior is not enough}
We compare against a TV-regularized compressed-sensing baseline solved by projected gradient descent (CS-TV):
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\operatorname{TV}(x).
\end{equation}
This baseline represents a classical compressed-sensing prior, not an exhaustively tuned iterative reconstruction benchmark. GI/BP denotes the linear physical backprojection or correlation-like GI reconstruction.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_robustness_baselines_submission.pdf}
\caption{\textbf{Robustness and baselines.} These diagnostics support finite-noise stability and measurement dependence within the tested conditions; they do not imply universal robustness.}
\label{fig:validation_summary}
\end{figure*}

\subsection{Stability and diagnostic controls}
For low-frequency Hadamard, including the DC row is critical for backprojection because it captures global brightness. Removing it severely degrades low-frequency Hadamard initialization. The 5\% low-frequency Hadamard condition is retained only as a diagnostic control. Bootstrap confidence intervals show that the main results are stable across samples. Class-wise STL-10 diagnostics reveal expected category variability, with some classes consistently more difficult than others.
"""


def prepare_project() -> None:
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
    source_dirs = [BASE_PROJECT / "figures", BASE_OUT / "figures", FIG_DIR, PROV]
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in source_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg"}:
                shutil.copy2(path, dst / path.name)


def clean_tex(path: Path) -> None:
    body = path.read_text(encoding="utf-8")
    body = body.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
    body = re.sub(r"(?<!\\operatorname\{)TV\(x\)", r"\\operatorname{TV}(x)", body)
    body = body.replace("-DC", "-MC")
    body = body.replace(r"-\mathrm{DC}", r"-\mathrm{MC}")
    path.write_text(body, encoding="utf-8")


def write_sections() -> None:
    main = PROJECT / "main.tex"
    main_body = main.read_text(encoding="utf-8")
    main_body = re.sub(
        r"\\title\{[^}]+\}",
        r"\\title{Measurement-Audited Neural Completion for Computational Ghost Imaging}",
        main_body,
    )
    main.write_text(main_body, encoding="utf-8")

    section_dir = PROJECT / "sections"
    (section_dir / "abstract.tex").write_text(ABSTRACT + "\n", encoding="utf-8")
    (section_dir / "introduction.tex").write_text(INTRODUCTION + "\n", encoding="utf-8")
    (section_dir / "method.tex").write_text(METHOD + "\n", encoding="utf-8")
    (section_dir / "results.tex").write_text(RESULTS + "\n", encoding="utf-8")
    (section_dir / "validation_ablation.tex").write_text(VALIDATION_ABLATION + "\n", encoding="utf-8")
    (section_dir / "discussion.tex").write_text(DISCUSSION + "\n", encoding="utf-8")
    for path in PROJECT.rglob("*.tex"):
        clean_tex(path)


def run_latex(target: str) -> None:
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", target], cwd=PROJECT, check=True)


def copy_outputs() -> None:
    shutil.copy2(PROJECT / "main.pdf", OUT / "main_v44.pdf")
    shutil.copy2(PROJECT / "supplement.pdf", OUT / "supplement_v44.pdf")
    for pdf_name, txt_name in [("main_v44.pdf", "main_v44.txt"), ("supplement_v44.pdf", "supplement_v44.txt")]:
        try:
            subprocess.run(["pdftotext", str(OUT / pdf_name), str(OUT / txt_name)], check=False)
        except FileNotFoundError:
            pass


def main() -> None:
    prepare_project()
    copy_figures()
    write_sections()
    run_latex("main.tex")
    run_latex("supplement.tex")
    copy_outputs()
    print({"project": str(PROJECT), "main_pdf": str(OUT / "main_v44.pdf"), "supplement_pdf": str(OUT / "supplement_v44.pdf")})


if __name__ == "__main__":
    main()
