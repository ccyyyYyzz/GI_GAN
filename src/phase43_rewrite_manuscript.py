from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
BASE_OUT = ROOT / "outputs_phase42_closed_loop_figure"
BASE_PROJECT = BASE_OUT / "latex_project_v42"
OUT = ROOT / "outputs_phase43_operator_circuit"
PROJECT = OUT / "latex_project_v43"
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


ABSTRACT = r"""Low-sampling ghost imaging is underdetermined: bucket measurements constrain the reconstruction but do not uniquely determine an image. A learned prior is useful in this regime, but it is also risky because sharper images can contain structure unsupported by the measurements. We formulate reconstruction as operator-centered neural completion. The same calibrated physical inverse is reused to form the data anchor, gate the neural residual, and audit the completed image. The network proposes missing structure, while the calibrated measurement operator anchors, gates, and audits what can remain. Under a leakage-free STL-10 protocol, the method reaches 22.316 dB PSNR / 0.635 SSIM at 5\% sampling with Rademacher measurements and 22.271 dB / 0.632 SSIM with scrambled Hadamard measurements. At 10\% sampling, the corresponding results are 24.781 dB / 0.747 SSIM and 24.730 dB / 0.746 SSIM. Exact-operator re-evaluation, inference ablation, measurement perturbation, and finite-noise diagnostics support the interpretation that performance comes from measurement-audited completion rather than measurement-independent sharpening."""


INTRODUCTION = r"""\section{Introduction}
Ghost imaging and single-pixel imaging reconstruct spatial information from known structured illumination patterns and scalar bucket readings. Each bucket measurement records a projection of the unknown image onto one pattern. In the low-sampling regime, \(A\in\mathbb{R}^{m\times n}\) has \(m\ll n\), so \(y=Ax+\epsilon\) leaves many images compatible with the same measurements.

This ambiguity is the central difficulty. A physical initialization stays tied to the bucket measurements but often misses object-level structure. Learned priors can fill in missing structure, but they also create a risk: a network can make an image sharper while adding details not supported by the measured bucket vector.

The key question is not whether a network can make the image sharper, but whether the added structure remains compatible with the bucket measurements. The network proposes missing structure; the calibrated measurement operator anchors, gates, and audits what can remain.

We therefore formulate reconstruction as operator-centered completion. A single regularized physical inverse \(B_\lambda=A^T(AA^T+\lambda I)^{-1}\) is reused throughout the reconstruction circuit. It forms the measured anchor \(x_{\rm data}=B_\lambda y\), defines the residual gate \(P_N=I-B_\lambda A\), and audits a completed image through \(\Pi_y(v)=v-B_\lambda(Av-y)\). During training, losses pass through these fixed differentiable physics layers and update only the neural modules.

The main contributions are:
\begin{itemize}
\item an operator-centered reconstruction circuit that reuses the same calibrated inverse for anchoring, gating, and auditing;
\item a measured anchor that keeps reconstruction tied to bucket evidence;
\item a neural residual proposal constrained by an operator-defined gate and final bucket audit;
\item leakage-free evaluation and diagnostics showing measurement-dependent reconstruction.
\end{itemize}
"""


METHOD = r"""\section{Measurement-Consistent Null-Space Reconstruction}
\subsection{Operator-centered view}
The reconstruction circuit is organized around one regularized physical inverse,
\begin{equation}
B_\lambda=A^T(AA^T+\lambda I)^{-1}.
\end{equation}
The neural module proposes a residual, but the calibrated physical operator anchors, gates, and audits the proposal. The three roles are:
\begin{enumerate}
\item \textbf{Anchor:}
\begin{equation}
x_{\rm data}=B_\lambda y.
\end{equation}
\item \textbf{Gate:}
\begin{equation}
P_N=I-B_\lambda A.
\end{equation}
\item \textbf{Audit:}
\begin{equation}
\Pi_y(v)=v-B_\lambda(Av-y).
\end{equation}
\end{enumerate}
Thus the learned module is not responsible for inventing a complete image by itself. It proposes missing structure; the known measurement operator determines how that proposal is admitted and how the completed image is corrected against the bucket vector.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_operator_circuit_v43.pdf}
\caption{\textbf{Operator-centered reconstruction circuit.} The calibrated physical inverse \(B_\lambda=A^\top(AA^\top+\lambda I)^{-1}\) is reused in three roles: it forms the measured anchor \(x_{\rm data}=B_\lambda y\), defines the residual gate \(I-B_\lambda A\), and audits the completed image through \(\hat{x}=\tilde{x}-B_\lambda(A\tilde{x}-y)\). The neural module proposes missing structure, but the known measurement operator determines how that proposal is admitted and how the final image is corrected against the bucket vector.}
\label{fig:mechanism}
\end{figure*}

\subsection{Data anchor}
Let \(A\in\mathbb{R}^{m\times n}\) be the measurement operator and \(y\in\mathbb{R}^m\) the bucket vector. In the low-sampling regime \(m\ll n\), the set
\begin{equation}
\mathcal{C}_y=\{x:Ax=y\}
\end{equation}
is large when noise and numerical tolerance are ignored. We use the measured-component anchor
\begin{equation}
x_{\rm data}=B_\lambda y=A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
This image is incomplete, but it is tied to the bucket measurements and known patterns.

\subsection{Neural residual proposal}
The neural module predicts
\begin{equation}
r_\theta=G_\theta(x_{\rm data},z),
\end{equation}
where \(z\) denotes optional auxiliary input used by the implemented reconstructor. The neural output is a proposal, not the final image.

\subsection{Residual gate}
The residual gate is
\begin{equation}
r_N=P_N r_\theta=(I-B_\lambda A)r_\theta.
\end{equation}
If \(\lambda=0\) and \(A\) has full row rank, then \(AP_N(v)=0\). With \(\lambda>0\), this becomes a regularized approximate null-space gate. The gate suppresses the part of the proposed residual that is visible to the measurement operator before insertion into the data anchor:
\begin{equation}
\tilde{x}=x_{\rm data}+r_N.
\end{equation}

\subsection{Bucket audit}
The final audit remeasures the pre-audit image and corrects the discrepancy with the original bucket vector:
\begin{equation}
\tilde{y}=A\tilde{x},\qquad e_y=\tilde{y}-y,
\end{equation}
\begin{equation}
\hat{x}=\tilde{x}-B_\lambda e_y
       =\tilde{x}-B_\lambda(A\tilde{x}-y).
\end{equation}
Equivalently, \(\hat{x}=\Pi_y(\tilde{x})\). The audit is therefore a remeasure--compare--correct loop, not a detached postprocessing block.

\subsection{Relation to conventional bucket-pattern correlation}
Conventional bucket-pattern correlation can be written, up to centering and normalization, as
\begin{equation}
\hat{x}_{\rm GI}=A^Ty=\sum_i y_i a_i.
\end{equation}
The anchor keeps the same pattern-expansion structure but replaces raw bucket weights with decorrelated coefficients:
\begin{equation}
A^Ty \rightarrow A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
Thus \(x_{\rm data}\) is a regularized/decorrelated GI/BP data solution, not a new standalone reconstructor. The contribution here is the operator-centered circuit: the same calibrated inverse anchors the data, gates the learned residual, and audits the completed image.

\subsection{Two-stage implementation and exact-operator handling}
The implemented high-quality reconstructor uses a two-stage learned completion block. The first stage computes
\begin{equation}
\hat{x}^{(1)}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
A refiner predicts \(r_\phi=R_\phi(x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|)\), and the final output is
\begin{equation}
\hat{x}=\Pi_y[\hat{x}^{(1)}+r_\phi].
\end{equation}
The two-stage architecture is an implementation detail of the learned proposal block; the conceptual path remains measurement, calibrated inverse, data anchor, residual proposal, residual gate, pre-audit completion, bucket audit, and final reconstruction. Image-domain metrics are computed after clipping to the valid intensity range, whereas measurement error is computed before clipping to avoid hiding projection inconsistency.

Rademacher sensing uses a random measurement matrix. Reproducible evaluation therefore requires reloading the exported exact operator. After replacing \(A\), all cached quantities derived from \(A\), including \(K=AA^T+\lambda I\) and its Cholesky factorization, must be rebuilt. This exact-operator cache-rebuilt path is used for all reported Rademacher results.
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

\subsection{Operator circuit and qualitative mechanism}
\Cref{fig:mechanism} shows the mechanism with one representative sample. The central calibrated inverse forms the measured anchor, defines the residual gate, and audits the completed image. The figure also shows the real measured anchor, proposed residual, gated residual, pre-audit image, final audited image, error image, and measurement-error audit. Larger qualitative examples are shown in \Cref{fig:qualitative_reconstruction}.

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
\caption{\textbf{Measurement attribution.} The regime map uses backprojection PSNR and \(\Delta\mathrm{PSNR}\) to separate physical-initialization quality from learned refinement. Low-frequency Hadamard points are shown as hollow diagnostic controls and are not primary STL-10 claims.}
\label{fig:measurement_attribution}
\end{figure*}

\subsection{Simple-domain sanity checks}
On MNIST and Fashion-MNIST at 5\% sampling, the method reaches 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM, respectively. These experiments confirm that the same reconstruction pipeline works reliably on simpler structured domains, but they are not the main novelty.
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
    for source_dir in [BASE_PROJECT / "figures", BASE_OUT / "figures", FIG_DIR]:
        if source_dir.exists():
            for path in source_dir.iterdir():
                if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg"}:
                    shutil.copy2(path, dst / path.name)


def clean_tex(path: Path) -> None:
    body = path.read_text(encoding="utf-8")
    body = body.replace(r"\lambda TV(x)", r"\lambda\operatorname{TV}(x)")
    body = re.sub(r"(?<!operatorname\{)TV\(x\)", r"\\operatorname{TV}(x)", body)
    body = re.sub(r"future hardware validation", "physical-system validation", body, flags=re.IGNORECASE)
    body = re.sub(r"hardware optical experiment", "physical-system experiment", body, flags=re.IGNORECASE)
    path.write_text(body, encoding="utf-8")


def write_sections() -> None:
    section_dir = PROJECT / "sections"
    (section_dir / "abstract.tex").write_text(ABSTRACT + "\n", encoding="utf-8")
    (section_dir / "introduction.tex").write_text(INTRODUCTION + "\n", encoding="utf-8")
    (section_dir / "method.tex").write_text(METHOD + "\n", encoding="utf-8")
    (section_dir / "results.tex").write_text(RESULTS + "\n", encoding="utf-8")
    for path in PROJECT.rglob("*.tex"):
        clean_tex(path)


def run_latex(target: str) -> None:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", target],
        cwd=PROJECT,
        check=True,
    )


def copy_outputs() -> None:
    shutil.copy2(PROJECT / "main.pdf", OUT / "main_v43.pdf")
    shutil.copy2(PROJECT / "supplement.pdf", OUT / "supplement_v43.pdf")
    for pdf_name, txt_name in [("main_v43.pdf", "main_v43.txt"), ("supplement_v43.pdf", "supplement_v43.txt")]:
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
    print({"project": str(PROJECT), "main_pdf": str(OUT / "main_v43.pdf"), "supplement_pdf": str(OUT / "supplement_v43.pdf")})


if __name__ == "__main__":
    main()
