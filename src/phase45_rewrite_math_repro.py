from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
BASE_OUT = ROOT / "outputs_phase44_operator_centered"
BASE_PROJECT = BASE_OUT / "latex_project_v44"
OUT = ROOT / "outputs_phase45_math_repro"
PROJECT = OUT / "latex_project_v45"

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


METHOD = r"""\section{Measurement-Audited Neural Completion}
\subsection{Measurement model and information split}
Let \(A\in\mathbb{R}^{m\times n}\) be the calibrated measurement operator, \(x\in[0,1]^n\) the vectorized image, and \(y\in\mathbb{R}^m\) the bucket vector. The computational forward model used in all reported experiments is
\begin{equation}
y=Ax+\epsilon,
\end{equation}
where \(\epsilon\) is additive i.i.d. Gaussian noise generated as \texttt{noise\_std} times \texttt{torch.randn\_like(y)} in the training and evaluation code. In low-sampling ghost imaging \(m<n\), so arbitrary-image recovery is impossible from \(y\) alone. The reconstruction problem is therefore an information split: measured components should remain tied to \(A\) and \(y\), while learned structure should enter only through components that the measurements weakly constrain.

\subsection{Calibrated inverse core}
The frozen operator used by the gate and audit is defined through
\begin{equation}
K_{\lambda_{\rm op}}=AA^{\mathsf T}+\lambda_{\rm op}I_m,
\qquad
B_{\lambda_{\rm op}}u=A^{\mathsf T}K_{\lambda_{\rm op}}^{-1}u,
\end{equation}
with \(A\in\mathbb{R}^{m\times n}\), \(K_{\lambda_{\rm op}}\in\mathbb{R}^{m\times m}\), and \(B_{\lambda_{\rm op}}\in\mathbb{R}^{n\times m}\). The implementation does not explicitly form \(K_{\lambda_{\rm op}}^{-1}\). It solves \(K_{\lambda_{\rm op}}s=u\) and returns \(A^{\mathsf T}s\). For dense/random \(A\), the Cholesky factorization of \(K_{\lambda_{\rm op}}\) is cached when available. If an exact exported Rademacher operator is loaded, \texttt{set\_A\_override} replaces \(A\), rebuilds \(K_{\lambda_{\rm op}}\), and rebuilds the Cholesky cache.

The default mathematical anchor is \(B_{\lambda_{\rm op}}y\). The actual code uses a configurable data-anchor map \(D(y)\): for Rademacher runs \(D(y)=B_{\lambda_{\rm op}}y\), while the Hadamard-family final configs set \texttt{backprojection\_mode=hadamard\_zero\_filled} and use a zero-filled selected-Hadamard coefficient inverse for \(D(y)\). In both cases, the gate and audit below use the same \(B_{\lambda_{\rm op}}\) solver.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_operator_centered_v44.pdf}
\caption{\textbf{Physics-gated and audited neural completion circuit.} A calibrated physical operator \(B_{\lambda_{\rm op}}=A^{\mathsf T}(AA^{\mathsf T}+\lambda_{\rm op} I)^{-1}\) anchors the default data solution, gates the neural proposal, and audits the final image. Anchor: \(x_{\rm data}=D(y)\), with \(D=B_{\lambda_{\rm op}}\) for ridge-pseudoinverse runs and a zero-filled Hadamard inverse for the Hadamard final configs. Gate: the neural proposal \(r_\theta=G_\theta(x_{\rm data},z)\) enters through \(P_N^{\lambda_{\rm op}}=I-B_{\lambda_{\rm op}}A\). Audit: the candidate image is remeasured by \(A\), compared with \(y\), and corrected as \(\Pi_y(v)=v-B_{\lambda_{\rm op}}(Av-y)\). The network proposes missing structure, while the calibrated measurement operator decides what can remain.}
\label{fig:mechanism}
\end{figure*}

\subsection{Anchor, gate, and audit}
The data anchor is
\begin{equation}
x_{\rm data}=D(y),
\end{equation}
where \(D\) is the configured backprojection described above. The first neural module predicts
\begin{equation}
r_\theta=G_\theta(x_{\rm data},z),
\end{equation}
where \(z\) is a random noise map sampled with the same spatial size as \(x_{\rm data}\). The residual is admitted through the soft null-space gate
\begin{equation}
r_N=P_N^{\lambda_{\rm op}}r_\theta,
\qquad
P_N^{\lambda_{\rm op}}=I-B_{\lambda_{\rm op}}A.
\end{equation}
The pre-audit candidate is \(x_{\rm data}+r_N\). The audit map is
\begin{equation}
\Pi_y(v)=v-B_{\lambda_{\rm op}}(Av-y).
\end{equation}
It remeasures a candidate, solves the same regularized bucket-space system, and corrects the image-space candidate through \(A^{\mathsf T}s\).

\subsection{Properties of the soft gate and audit}
Let \(P_N^{\lambda_{\rm op}}=I-B_{\lambda_{\rm op}}A\). For any vector \(v\),
\begin{equation}
AP_N^{\lambda_{\rm op}}v
=
\lambda_{\rm op}K_{\lambda_{\rm op}}^{-1}Av.
\end{equation}
Therefore the gate is an exact null-space projection only in the idealized case \(\lambda_{\rm op}=0\) with full row-rank \(A\). For the reported \(\lambda_{\rm op}>0\), it is a regularized soft gate.

For the audit residual, define \(e=Av-y\). Then
\begin{equation}
A\Pi_y(v)-y
=
\lambda_{\rm op}K_{\lambda_{\rm op}}^{-1}e.
\end{equation}
Thus \(\Pi_y\) is an exact measurement projection only when \(\lambda_{\rm op}=0\). With positive \(\lambda_{\rm op}\), it is a regularized soft audit rather than a claim of exact equality \(A\hat{x}=y\).

\begin{figure}[t]
\centering
\fbox{\begin{minipage}{0.96\linewidth}
\textbf{Algorithm 1: Inference with measurement-audited neural completion.}
\begin{enumerate}
\item Input \(A,y,\lambda_{\rm op},G_\theta,R_\phi\) and the configured backprojection mode.
\item Form \(K_{\lambda_{\rm op}}=AA^{\mathsf T}+\lambda_{\rm op}I_m\); solve \(K_{\lambda_{\rm op}}s=u\) when applying \(B_{\lambda_{\rm op}}u=A^{\mathsf T}s\).
\item Compute \(x_{\rm data}=D(y)\), where \(D=B_{\lambda_{\rm op}}\) for ridge-pseudoinverse runs and \(D\) is zero-filled Hadamard inversion for Hadamard zero-fill runs.
\item Sample \(z\) as a noise map and compute \(r_\theta=G_\theta(x_{\rm data},z)\).
\item Compute \(r_N=r_\theta-B_{\lambda_{\rm op}}Ar_\theta\).
\item Compute \(\hat{x}^{(1)}=\Pi_y(x_{\rm data}+r_N)\).
\item Compute \(r_\phi=R_\phi(x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|)\).
\item Compute the unclipped final output \(\hat{x}=\Pi_y(\hat{x}^{(1)}+r_\phi)\).
\item Return \(x_{\rm eval}=\operatorname{clip}(\hat{x},0,1)\) for image metrics and retain \(\hat{x}\) for unclipped audit diagnostics.
\end{enumerate}
\end{minipage}}
\caption{Inference algorithm implemented by \texttt{reconstruct\_from\_measurements}, including the Hadamard zero-fill anchor option.}
\label{alg:inference}
\end{figure}

\subsection{Training objective}
The following objective is extracted from \texttt{src/train.py} and \texttt{src/losses.py}. Let \(C(u)=\operatorname{clip}(u,0,1)\). In the final configs \texttt{output\_range\_mode=clamp\_eval\_only}, image-domain losses use the clipped output \(C(\hat{x})\), while the measurement-consistency training loss uses the unclipped \texttt{x\_hat\_unclamped}. The total generator loss is
\begin{align}
\mathcal{L}_{\rm total}
=&\ \omega_1\mathcal{L}_1
+\omega_{\rm meas}\mathcal{L}_{\rm meas}
+\lambda_{\rm tv}\operatorname{TV}(C(\hat{x}))
+\omega_{\rm charb}\mathcal{L}_{\rm charb}
+\omega_{\rm edge}\mathcal{L}_{\rm edge} \nonumber\\
&+\omega_{\rm ms1}\mathcal{L}_{\rm ms1}
+\omega_{\rm ssim}\mathcal{L}_{\rm ssim}
+\omega_{\rm msssim}\mathcal{L}_{\rm msssim}
+\omega_{\rm grad}\mathcal{L}_{\rm grad}
+\omega_{\rm freq}\mathcal{L}_{\rm freq} \nonumber\\
&+\omega_{\rm s1}\mathcal{L}_{\rm s1}
+\omega_{\rm adv}\mathcal{L}_{\rm adv}
+\mathcal{L}_{\rm pattern}.
\end{align}
For all reported final runs, \(\omega_{\rm adv}=0\), \texttt{use\_adversarial=false}, \texttt{use\_learned\_patterns=false}, and \(\omega_{\rm s1}=0\). Adversarial loss is disabled in all reported final experiments. Stage-1 supervision is no: the stage-1 auxiliary L1 value is computed by the code but is not weighted in the final loss because \(\omega_{\rm s1}=0\). Therefore no adversarial, pattern-learning, or weighted stage-1 supervision term contributes to the final reported training loss. The final output is supervised through the image-domain terms.

The individual losses are
\begin{align}
\mathcal{L}_1 &= \|C(\hat{x})-x\|_1 \text{ averaged over all pixels},\\
\mathcal{L}_{\rm meas}(u,y) &= \frac{1}{Bm}\sum_{b=1}^{B}\|A u_b-y_b\|_2^2,\\
\mathcal{L}_{\rm charb} &= \operatorname{mean}\sqrt{(C(\hat{x})-x)^2+10^{-6}},\\
\mathcal{L}_{\rm edge} &= \|S_x C(\hat{x})-S_x x\|_1+\|S_y C(\hat{x})-S_y x\|_1,\\
\mathcal{L}_{\rm grad} &= \|\nabla_x C(\hat{x})-\nabla_x x\|_1+\|\nabla_y C(\hat{x})-\nabla_y x\|_1,\\
\mathcal{L}_{\rm freq} &= \|\log(1+|\operatorname{rFFT2}(C(\hat{x}))|)-\log(1+|\operatorname{rFFT2}(x)|)\|_1.
\end{align}
The multiscale L1 term is \(\mathcal{L}_{\rm ms1}=\mathcal{L}_1+0.5\mathcal{L}_{1,2}+0.25\mathcal{L}_{1,4}\), where the last two terms use average pooling by factors 2 and 4. The differentiable SSIM loss uses a \(7\times7\) average-pooling window and returns \(1-\operatorname{SSIM}\); the multiscale SSIM term uses the same weights \(1,0.5,0.25\) after pooling by 1, 2, and 4. The measurement loss above is the actual code formula; it is not divided by \(\|y\|_2^2\).

\begin{figure}[t]
\centering
\fbox{\begin{minipage}{0.96\linewidth}
\textbf{Algorithm 2: Training one mini-batch.}
\begin{enumerate}
\item Load a normalized grayscale mini-batch \(x\in[0,1]^{B\times1\times64\times64}\).
\item Generate \(y=Ax+\epsilon\), where \(\epsilon\) is \texttt{noise\_std} times \texttt{torch.randn\_like(y)}.
\item Run Algorithm 1 to obtain \(\hat{x}^{(1)}\), unclipped \(\hat{x}\), and clipped \(C(\hat{x})\).
\item Compute \(\mathcal{L}_{\rm total}\) using the configured weights in the supplementary reproducibility table.
\item Backpropagate through \(A\), \(B_{\lambda_{\rm op}}\), \(P_N^{\lambda_{\rm op}}\), and \(\Pi_y\); these physics operators remain frozen.
\item Update only the trainable neural parameters \(\theta,\phi\) with Adam. Final configs use learning rate \(10^{-4}\), betas \((0.5,0.9)\), batch size 8, AMP on CUDA, and EMA decay 0.999.
\item No learning-rate scheduler or weight decay is used in the final training code path.
\end{enumerate}
\end{minipage}}
\caption{Training algorithm for the reported final runs. Exact per-run epochs and loss weights are listed in the supplementary reproducibility table.}
\label{alg:training}
\end{figure}

\subsection{Two-stage neural implementation}
The reported high-quality reconstructor uses \texttt{hq\_two\_stage} with 34,695,170 trainable parameters for base width 64. Stage one is an HQ residual U-Net receiving \(x_{\rm data}\) and a noise map. The refiner receives \([x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|]\) and predicts the second residual proposal. Stage one is active from the beginning; the refiner is enabled from the configured \texttt{refiner\_start\_epoch}. Because \(\omega_{\rm s1}=0\) in the final configs, the stage-1 output is not an additional weighted supervised target, even though it is used as the input to the refiner.

\subsection{Relation to conventional bucket-pattern correlation}
Conventional bucket-pattern correlation can be written, up to centering and normalization, as
\begin{equation}
\hat{x}_{\rm GI}=A^{\mathsf T}y=\sum_i y_i a_i.
\end{equation}
The ridge anchor replaces raw bucket weights with decorrelated coefficients,
\begin{equation}
A^{\mathsf T}y \rightarrow A^{\mathsf T}(AA^{\mathsf T}+\lambda_{\rm op}I_m)^{-1}y.
\end{equation}
The contribution is not \(x_{\rm data}\) alone, nor is \(x_{\rm data}\) claimed as a new invention. The contribution is the operator-centered circuit in which a frozen calibrated operator gates and audits neural completion.
"""


SUPPLEMENT_APPENDIX = r"""

\subsection{S10 Metric definitions}
All images are represented as single-channel \(64\times64\) tensors normalized to \([0,1]\). The reported image metrics use the clipped image \(x_{\rm eval}=\operatorname{clip}(\hat{x},0,1)\). The mean squared error is
\[
\operatorname{MSE}
=
\frac{1}{n}\|x_{\rm eval}-x\|_2^2.
\]
Since \(x_{\rm eval},x\in[0,1]^n\), PSNR uses \(x_{\max}=1\):
\[
\operatorname{PSNR}
=
10\log_{10}\frac{1}{\operatorname{MSE}}.
\]
The evaluation code uses \texttt{skimage.metrics.peak\_signal\_noise\_ratio} and \texttt{structural\_similarity} with \texttt{data\_range=1.0} when available, after removing the singleton grayscale channel. It averages per-sample metric values. If scikit-image is unavailable, the code falls back to a torch PSNR implementation and a global SSIM approximation.

The relative measurement error is
\[
\operatorname{RelMeasErr}
=
\frac{\|A\hat{x}-y\|_2}{\|y\|_2+\epsilon},
\]
implemented with the denominator clamped to \(10^{-12}\). The default \texttt{rel\_meas\_error} reported by \texttt{batch\_metrics} is computed on the tensor passed to it; final eval JSON additionally records \texttt{rel\_meas\_err\_clamped} and \texttt{rel\_meas\_err\_unclamped} for the model output. Training measurement loss uses the unclipped final output in \texttt{clamp\_eval\_only} mode.

\subsection{S11 Reproducible training and evaluation configuration}
\input{tables/tableS9_training_config_phase45.tex}

\subsection{S12 Training-code audit limitations}
The resolved configs and final metrics are stored locally for all six final no-leak runs. Exact exported Rademacher operators are also stored locally and are reloaded with solver-cache rebuilding. The local imported folders do not preserve every original Colab training log or GPU model string, so those fields are not claimed in the paper. Dataset subset membership is reproducible from the deterministic limiting rule in \texttt{src/datasets.py}: random subsets are chosen by \texttt{torch.randperm} with \texttt{seed} for training and \texttt{seed+1} for evaluation.
"""


def prepare_project() -> None:
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE_PROJECT, PROJECT)
    for path in PROJECT.rglob("*"):
        if path.is_file() and path.suffix in COMPILED_SUFFIXES:
            path.unlink()


def copy_phase45_tables() -> None:
    tables = PROJECT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    src = OUT / "tableS9_training_config_phase45.tex"
    if not src.exists():
        raise FileNotFoundError(f"Run phase45_audit_training_code first; missing {src}")
    shutil.copy2(src, tables / src.name)


def copy_figures() -> None:
    dst = PROJECT / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    source_dirs = [
        BASE_PROJECT / "figures",
        BASE_OUT / "figures",
        BASE_OUT / "provenance",
    ]
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in source_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg"}:
                shutil.copy2(path, dst / path.name)


def global_symbol_cleanup(text: str) -> str:
    text = text.replace(r"B_\lambda", r"B_{\lambda_{\rm op}}")
    text = text.replace(r"(AA^{\mathsf T}+\lambda I)", r"(AA^{\mathsf T}+\lambda_{\rm op} I)")
    text = text.replace(r"AA^{\mathsf T}+\lambda I", r"AA^{\mathsf T}+\lambda_{\rm op} I")
    text = text.replace(r"\lambda\operatorname{TV}(x)", r"\lambda_{\rm tv}\operatorname{TV}(x)")
    text = text.replace(r"\lambda TV(x)", r"\lambda_{\rm tv}\operatorname{TV}(x)")
    text = re.sub(r"(?<!\\operatorname\{)TV\(x\)", r"\\operatorname{TV}(x)", text)
    return text


def write_sections() -> None:
    section_dir = PROJECT / "sections"
    (section_dir / "method.tex").write_text(METHOD + "\n", encoding="utf-8")
    for path in PROJECT.rglob("*.tex"):
        body = path.read_text(encoding="utf-8")
        body = global_symbol_cleanup(body)
        path.write_text(body, encoding="utf-8")
    validation = section_dir / "validation_ablation.tex"
    body = validation.read_text(encoding="utf-8")
    body = body.replace(
        r"\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda_{\rm tv}\operatorname{TV}(x).",
        r"\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda_{\rm tv}\operatorname{TV}(x).",
    )
    validation.write_text(body, encoding="utf-8")
    supp = PROJECT / "supplement" / "supplement.tex"
    supp_body = supp.read_text(encoding="utf-8")
    supp_body = supp_body.replace(
        r"\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda_{\rm tv}\operatorname{TV}(x).",
        r"\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda_{\rm tv}\operatorname{TV}(x).",
    )
    supp_body = supp_body.rstrip() + "\n" + SUPPLEMENT_APPENDIX + "\n"
    supp.write_text(global_symbol_cleanup(supp_body), encoding="utf-8")


def patch_preamble() -> None:
    for tex_name in ["main.tex", "supplement.tex"]:
        path = PROJECT / tex_name
        body = path.read_text(encoding="utf-8")
        if r"\usepackage{array}" not in body:
            body = body.replace(r"\usepackage{booktabs}", r"\usepackage{booktabs}" + "\n" + r"\usepackage{array}")
        path.write_text(body, encoding="utf-8")


def run_latex(target: str) -> None:
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", target],
        cwd=PROJECT,
        check=True,
    )


def copy_outputs() -> None:
    shutil.copy2(PROJECT / "main.pdf", OUT / "main_v45.pdf")
    shutil.copy2(PROJECT / "supplement.pdf", OUT / "supplement_v45.pdf")
    for pdf_name, txt_name in [("main_v45.pdf", "main_v45.txt"), ("supplement_v45.pdf", "supplement_v45.txt")]:
        try:
            subprocess.run(["pdftotext", str(OUT / pdf_name), str(OUT / txt_name)], check=False)
        except FileNotFoundError:
            pass


def main() -> None:
    if not BASE_PROJECT.exists():
        raise FileNotFoundError(BASE_PROJECT)
    prepare_project()
    copy_figures()
    copy_phase45_tables()
    patch_preamble()
    write_sections()
    run_latex("main.tex")
    run_latex("supplement.tex")
    copy_outputs()
    print(
        {
            "project": str(PROJECT),
            "main_pdf": str(OUT / "main_v45.pdf"),
            "supplement_pdf": str(OUT / "supplement_v45.pdf"),
        }
    )


if __name__ == "__main__":
    main()
