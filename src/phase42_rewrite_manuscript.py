from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
BASE_OUT = ROOT / "outputs_phase41_inkscape_signal_trace"
BASE_PROJECT = BASE_OUT / "latex_project_v41"
OUT = ROOT / "outputs_phase42_closed_loop_figure"
PROJECT = OUT / "latex_project_v42"
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


ABSTRACT = r"""Low-sampling ghost imaging is underdetermined: bucket measurements constrain the reconstruction but do not uniquely determine an image. A learned prior is useful in this regime, but it is also risky because sharper images can contain structure unsupported by the measurements. We formulate reconstruction as measurement-audited neural completion. The bucket vector is used not only to initialize the reconstruction, but also to audit the completed image and to define the measurement residual during training. The network proposes missing structure; fixed physics layers filter and audit what can remain. Under a leakage-free STL-10 protocol, the method reaches 22.316 dB PSNR / 0.635 SSIM at 5\% sampling with Rademacher measurements and 22.271 dB / 0.632 SSIM with scrambled Hadamard measurements. At 10\% sampling, the corresponding results are 24.781 dB / 0.747 SSIM and 24.730 dB / 0.746 SSIM. Exact-operator re-evaluation, inference ablation, measurement perturbation, and finite-noise diagnostics support the interpretation that performance comes from bucket-measurement-audited completion rather than measurement-independent sharpening."""


INTRODUCTION = r"""\section{Introduction}
Ghost imaging and single-pixel imaging reconstruct spatial information from known structured illumination patterns and scalar bucket readings. Each bucket measurement records a projection of the unknown image onto one pattern. In the low-sampling regime, \(A\in\mathbb{R}^{m\times n}\) has \(m\ll n\), so \(y=Ax+\epsilon\) leaves many images compatible with the same measurements.

This ambiguity is the central difficulty. A physical initialization stays tied to the bucket measurements but often misses object-level structure. Learned priors can fill in missing structure, but they also create a risk: a network can make an image sharper while adding details not supported by the measured bucket vector.

The key question is not whether a network can make the image sharper, but whether the added structure remains compatible with the bucket measurements. The network proposes missing structure; the measurement operator filters and audits what can remain.

We therefore formulate reconstruction as audited completion. The same measured bucket vector anchors the reconstruction, audits the completed image, and defines a measurement residual used for diagnostics and training. First, a regularized data solution \(x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y\) provides a measured anchor. Second, \(G_\theta\) proposes missing structure around that anchor. Third, \(P_N\) filters residual components visible to the measurement operator. Finally, \(\Pi_y\) audits the completed image by rechecking it against the original bucket readings. During training, losses pass through the fixed differentiable physics layers and update only the neural modules.

The main contributions are:
\begin{itemize}
\item a closed-loop signal-trace interpretation of low-sampling reconstruction from bucket measurements to audited output;
\item a measured anchor that keeps the reconstruction tied to bucket evidence;
\item a neural residual proposal constrained by residual filtering and final bucket audit;
\item leakage-free evaluation and diagnostics showing measurement-dependent reconstruction.
\end{itemize}
"""


METHOD = r"""\section{Measurement-Consistent Null-Space Reconstruction}
\subsection{One-sample signal flow}
The bucket vector \(y\) is used in three places. First, it forms the measured anchor \(x_{\rm data}\). Second, it defines the final audit \(\Pi_y\), which corrects the completed image using the discrepancy \(A\tilde{x}-y\). Third, during training and diagnostics, it defines the measurement residual \(A\hat{x}-y\). Thus the reconstruction is not a free image-to-image network: the same measured signal anchors, audits, and evaluates the neural completion.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_closed_loop_v42.pdf}
\caption{\textbf{Measurement-audited neural completion.} The bucket vector is not a one-time input: it forms the measured anchor \(x_{\rm data}\), audits the completed image through \(\Pi_y\), and provides the measurement residual used for training diagnostics. The image-completion engine lets the network propose a residual, filters that proposal through \(P_N\), forms a pre-audit image, and projects the result back toward the measured affine set. During training, losses on the final image and the remeasured bucket signal backpropagate through the fixed physics layers and update only the neural modules.}
\label{fig:mechanism}
\end{figure*}

\begin{center}
\fbox{\begin{minipage}{0.92\linewidth}
\textbf{Algorithm 1: Measurement-audited neural completion}

\textbf{Input:} \(A,y\).
\begin{enumerate}
\item \(x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y\).
\item \(r_\theta=G_\theta(x_{\rm data})\).
\item \(r_N=P_N(r_\theta)\).
\item \(\tilde{x}=x_{\rm data}+r_N\).
\item \(\hat{x}=\Pi_y(\tilde{x})\).
\end{enumerate}
\textbf{Training only:}
\begin{enumerate}
\setcounter{enumi}{5}
\item compute \(\mathcal L_{\rm img}(\hat{x},x)\) and \(\mathcal L_{\rm meas}(A\hat{x},y)\);
\item update \(G_\theta,R_\phi\); keep \(A,P_N,\Pi_y\) fixed.
\end{enumerate}
\end{minipage}}
\end{center}

\subsection{Data anchor}
Let \(A\in\mathbb{R}^{m\times n}\) be the measurement operator and \(y\in\mathbb{R}^m\) the bucket vector. In the low-sampling regime \(m\ll n\), the set
\begin{equation}
\mathcal{C}_y=\{x:Ax=y\}
\end{equation}
is large when noise and numerical tolerance are ignored. We use the measured-component anchor
\begin{equation}
q=(AA^T+\lambda I)^{-1}y,
\qquad
x_{\rm data}=A^Tq=\sum_i q_i a_i.
\end{equation}
This image is incomplete, but it is tied to the bucket measurements and known patterns.

\subsection{Neural residual proposal}
The neural module predicts
\begin{equation}
r_\theta=G_\theta(x_{\rm data},z),
\end{equation}
where \(z\) denotes optional auxiliary input used by the implemented reconstructor. The neural output is a proposal, not the final image.

\subsection{Residual admissibility filtering}
Define \(P_A=A^T(AA^T+\lambda I)^{-1}A\) and \(P_N=I-P_A\). Applied to a vector \(v\),
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av.
\end{equation}
If \(\lambda=0\) and \(A\) has full row rank, then \(AP_N(v)=0\). With \(\lambda>0\), this becomes a regularized approximate null-space filter. The filter suppresses residual components visible to the measurement operator, yielding
\begin{equation}
\tilde{x}=x_{\rm data}+P_N(r_\theta).
\end{equation}

\subsection{Bucket-measurement audit}
The final audit applies the measurement-consistency projection
\begin{equation}
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y).
\end{equation}
\(\Pi_y\) remeasures the completed image through \(A\) and corrects discrepancy with \(y\). The final reconstruction is
\begin{equation}
\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
This projection is a whole-image bucket audit after learned structure has been inserted.

\subsection{Relation to conventional bucket-pattern correlation}
Conventional bucket-pattern correlation can be written, up to centering and normalization, as
\begin{equation}
\hat{x}_{\rm GI}=A^Ty=\sum_i y_i a_i.
\end{equation}
The anchor keeps the same pattern-expansion structure but replaces raw bucket weights with decorrelated coefficients:
\begin{equation}
A^Ty \rightarrow A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
Thus \(x_{\rm data}\) is a regularized/decorrelated GI/BP data solution, not a new standalone reconstructor. The contribution here is the closed signal trace with residual filtering, final audit, and training feedback through fixed physics layers.

\subsection{Two-stage implementation and exact operator handling}
The implemented high-quality reconstructor uses a two-stage learned completion block. The first stage computes
\begin{equation}
\hat{x}^{(1)}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
A refiner predicts \(r_\phi=R_\phi(x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|)\), and the final output is
\begin{equation}
\hat{x}=\Pi_y[\hat{x}^{(1)}+r_\phi].
\end{equation}
The two-stage architecture is an implementation detail of the learned proposal block; the conceptual path remains measurement, anchor, residual proposal, residual filter, pre-audit completion, bucket audit, and final reconstruction. Image-domain metrics are computed after clipping to the valid intensity range, whereas measurement error is computed before clipping to avoid hiding projection inconsistency.

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

\subsection{Closed-loop signal trace and qualitative mechanism}
\Cref{fig:mechanism} shows the mechanism with one representative sample. The measurement rail makes explicit that the same bucket vector forms the measured anchor, audits the completed image, and defines the measurement residual. The image-completion engine shows the measured anchor, proposed residual, filtered residual, pre-audit image, and final audited image. Larger qualitative examples are shown in \Cref{fig:qualitative_reconstruction}.

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
    shutil.copy2(PROJECT / "main.pdf", OUT / "main_v42.pdf")
    shutil.copy2(PROJECT / "supplement.pdf", OUT / "supplement_v42.pdf")
    for pdf_name, txt_name in [("main_v42.pdf", "main_v42.txt"), ("supplement_v42.pdf", "supplement_v42.txt")]:
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
    print({"project": str(PROJECT), "main_pdf": str(OUT / "main_v42.pdf"), "supplement_pdf": str(OUT / "supplement_v42.pdf")})


if __name__ == "__main__":
    main()
