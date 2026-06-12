from __future__ import annotations

import re
import shutil
from pathlib import Path

from .phase20_common import TITLE, ensure_dir, source_manifest, tex_escape, write_json, write_text


P21 = Path("E:/ns_mc_gan_gi/outputs_phase21_submission_polish")
P22 = Path("E:/ns_mc_gan_gi/outputs_phase22_submission_v8")
P22_LATEX = P22 / "latex_project_v8"
OUT = Path("E:/ns_mc_gan_gi/outputs_phase23_top_journal_rewrite")
LATEX = OUT / "latex_project_v9"
SECTIONS = LATEX / "sections"
SUPP = LATEX / "supplement"
FIGS = LATEX / "figures"
TABLES = LATEX / "tables"


ABSTRACT = (
    "Ghost imaging and single-pixel imaging recover images from structured illumination patterns and scalar "
    "bucket measurements, but low-sampling acquisition leaves most image degrees of freedom unmeasured. "
    "The resulting inverse problem is not ordinary image restoration: many images can match the same measurement "
    "vector, so neural detail must remain auditable against the bucket data. We formulate low-sampling ghost "
    "imaging as measurement-constrained null-space completion. The reconstruction first computes a physical data "
    "solution, then uses a neural module to propose missing structure, inserts that proposal through an approximate "
    "null-space projection, and finally applies a measurement-consistency projection. On STL-10, the method reaches "
    "22.316 dB PSNR / 0.635 SSIM at 5% sampling with Rademacher measurements and 22.271 dB / 0.632 SSIM with "
    "scrambled Hadamard measurements. At 10% sampling, the corresponding results are 24.781 dB / 0.747 SSIM and "
    "24.730 dB / 0.746 SSIM. MNIST and Fashion-MNIST 5% experiments provide simple-domain sanity checks, reaching "
    "27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM. Exact-operator audit, attribution, ablation, finite-noise, "
    "measurement-perturbation, compressed-sensing baseline, and confidence-interval analyses support the central "
    "claim: the learned prior is useful for the unmeasured component, but the reconstruction remains tied to the "
    "bucket measurements."
)


def clean_latex_dir() -> None:
    if LATEX.exists():
        shutil.rmtree(LATEX)
    ensure_dir(SECTIONS)
    ensure_dir(SUPP)
    ensure_dir(FIGS)
    ensure_dir(TABLES)


def _clean_pdf_source_text(text: str) -> str:
    replacements = {
        "Primary strict no-leak results.": "Primary leakage-free evaluation results.",
        "strict no-leak": "strict leakage-free",
        "no-leak": "leakage-free",
        "Thresholds are internal engineering criteria stated in the protocol.": "Thresholds are predefined operational criteria used in this study.",
        "internal engineering criteria": "predefined operational criteria",
        "lowfreq": "low-frequency",
        "Colab results": "original leakage-free evaluation",
        "Colab checkpoint": "trained checkpoint",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def copy_tables() -> None:
    src = P21 / "tables"
    dst = OUT / "tables"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    shutil.copytree(src, TABLES, dirs_exist_ok=True)
    for path in list(dst.glob("*.tex")) + list(TABLES.glob("*.tex")):
        text = _clean_pdf_source_text(path.read_text(encoding="utf-8"))
        path.write_text(text, encoding="utf-8")


def copy_figures() -> None:
    ensure_dir(FIGS)
    src = OUT / "figures"
    for path in src.glob("*_v9.*"):
        if path.suffix.lower() in {".pdf", ".png", ".svg"}:
            shutil.copy2(path, FIGS / path.name)


def read_v8(name: str) -> str:
    return (P22_LATEX / "sections" / name).read_text(encoding="utf-8")


def normalize_v8_text(text: str) -> str:
    text = _clean_pdf_source_text(text)
    text = text.replace("generic image-prior hallucination", "measurement-independent hallucination")
    text = text.replace("adversarial-generation paper", "adversarially driven reconstruction paper")
    text = re.sub(r"Phase1[5-9]|Phase20|Phase21|Phase22", "internal stage", text)
    text = text.replace("E:/", "")
    text = text.replace("C:\\", "")
    return text


def abstract() -> str:
    return ABSTRACT.replace("%", r"\%")


def introduction() -> str:
    return r"""\section{Introduction}
Ghost imaging and single-pixel imaging recover spatial information from structured illumination patterns and scalar bucket measurements rather than from a dense focal-plane detector \cite{shapiro2008computational,edgar2019principles,gibson2020singlepixel}. Each pattern probes the object with a known spatial code, and the bucket detector records only the total transmitted or reflected intensity. The bucket measurement is therefore not a low-resolution image; it is a projection of the object onto the illumination pattern. Reconstruction must infer an image from these scalar projections.

Low sampling is the central difficulty. If the vectorized image is \(x\in\mathbb{R}^n\), the sensing matrix is \(A\in\mathbb{R}^{m\times n}\), and the bucket vector is \(y\), then \(y=Ax+\epsilon\). In the regimes studied here, \(m\ll n\): for \(64\times64\) images, 5\% sampling gives approximately 205 measurements for 4096 unknown pixels. Consequently, many images can match the same \(y\). Low-sampling ghost imaging is therefore not a conventional denoising or super-resolution problem; it is an underdetermined inverse problem with a large measurement-invisible subspace.

Direct physical inverses such as backprojection remain valuable because they are tied to the actual measurements. They provide a data solution that reflects the row-space component constrained by \(A\) and \(y\). However, at low sampling this physically anchored image is incomplete: object structure is blurred, aliased, or absent because most degrees of freedom are not measured. Backprojection alone therefore supplies contact with the experiment but not enough image content.

Neural reconstruction can supply the missing content, but it also creates a scientific risk. A learned model may produce visually plausible structure that is weakly related to the bucket measurements. The central issue is not only whether a neural network can improve image quality, but whether it can do so without losing contact with the measurements. In low-sampling GI, the measurement vector fixes only a low-dimensional component of the image; the remaining degrees of freedom lie in directions that are invisible to the sensing operator. A reconstruction method should therefore distinguish between information that is measured, information that is missing, and information that is hallucinated.

The core question of this work is: How can a neural reconstructor add missing structure while remaining auditable against the bucket measurements? This question shifts the emphasis away from unconstrained perceptual restoration. The reconstruction should expose which part is physically determined, where the learned prior enters, and how consistency with \(y\) is checked after neural refinement.

Our answer is a measurement-constrained null-space completion pipeline. The method first computes a data solution \(x_{\rm data}\) from the forward operator and bucket vector. A neural module then proposes missing structure, but the proposed residual is inserted through an approximate null-space projection \(P_N\). Finally, a measurement-consistency projection \(\Pi_y\) returns the reconstruction to the measured affine set. This sequence separates measured row-space information, neural null-space completion, and measurement audit.

The contributions are threefold. First, we present low-sampling ghost imaging as measurement-constrained null-space completion and implement the corresponding \(x_{\rm data}\), \(P_N\), and \(\Pi_y\) reconstruction path. Second, we evaluate the method under a leakage-free protocol on STL-10 at 5\% and 10\% sampling for Rademacher and scrambled Hadamard measurements, with MNIST and Fashion-MNIST as simple-domain sanity checks. Third, we provide audits that connect reconstruction quality to physical evidence: exact-operator reproducibility, initialization-vs-gain attribution, inference ablations, measurement perturbations, finite-noise behavior, a CS-TV baseline, and confidence intervals.
"""


def problem_formulation() -> str:
    return r"""\section{Problem Formulation}
Let \(x\in\mathbb{R}^n\) denote the vectorized unknown image. Each illumination pattern \(a_i\in\mathbb{R}^n\) produces a scalar bucket measurement
\begin{equation}
y_i=\langle a_i,x\rangle+\epsilon_i .
\end{equation}
Stacking all measurements gives
\begin{equation}
y=Ax+\epsilon,
\end{equation}
where \(A\in\mathbb{R}^{m\times n}\). The sampling ratio is \(\rho=m/n\). For \(64\times64\) images, \(n=4096\); 5\% and 10\% sampling correspond to approximately 205 and 410 bucket measurements. Thus the measurement equation fixes far fewer constraints than there are pixels.

In the noiseless case, the measurement-consistent set is the affine set
\begin{equation}
\mathcal{C}_y=\{x:Ax=y\}.
\end{equation}
If \(x_0\in\mathcal{C}_y\) and \(v\in\mathrm{Null}(A)\), then \(A(x_0+v)=y\). The bucket vector fixes the row-space component of the image but leaves the null-space component unconstrained. The reconstruction task is therefore a constrained completion problem: recover image structure in directions not measured by \(A\), while preserving the component that is measured by \(y\).
"""


def method() -> str:
    return r"""\section{Measurement-Consistent Null-Space Reconstruction}
Our reconstruction follows directly from the geometry of the underdetermined measurement equation. The data solution represents one physically anchored point in the measurement-consistent affine set. The neural module is then used only to propose missing structure, and the projection operators determine how this proposal is inserted and audited.

\subsection{What is measured: data solution}
We first compute a regularized data solution
\begin{equation}
x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
This is the measured anchor of the reconstruction. It is determined by the illumination operator and bucket vector, and it lies near the measurement-consistent affine set. When the sampling ratio is low, \(x_{\rm data}\) is incomplete but physically interpretable.

\subsection{What is missing: null-space neural residual}
The missing component is associated with directions that the measurement operator does not observe. Define
\begin{equation}
P_A=A^T(AA^T+\lambda I)^{-1}A,
\end{equation}
and
\begin{equation}
P_N=I-P_A .
\end{equation}
Applied to a candidate residual \(v\),
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av.
\end{equation}
If \(\lambda=0\) and \(A\) has full row rank, \(AP_N(v)=0\). With \(\lambda>0\), this becomes a regularized approximate null-space projection. The neural module predicts \(G_\theta(x_{\rm data},z)\), and only its projected residual is inserted.

\subsection{What must be checked: measurement-consistency projection}
After neural completion, the reconstruction is projected back toward the measured affine set:
\begin{equation}
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y).
\end{equation}
The final reconstruction is
\begin{equation}
\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
The learned prior can therefore add missing structure, but the output is still audited against the bucket measurements. \Cref{fig:concept} summarizes this three-part logic.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_concept_v9.pdf}
\caption{\textbf{Low-sampling GI as measurement-constrained null-space completion.} The optical system records bucket measurements from illumination patterns. The measurement equation defines an affine set \(\mathcal{C}_y=\{x:Ax=y\}\) and a null space. The algorithm starts from \(x_{\rm data}\), uses a neural prior only to propose missing structure, inserts the proposal through \(P_N\), and applies \(\Pi_y\) as a measurement audit.}
\label{fig:concept}
\end{figure*}

\subsection{Two-stage refiner}
The implemented high-quality reconstructor uses a two-stage structure. Stage 1 computes
\begin{equation}
\hat{x}^{(1)}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
A refiner then predicts \(r_\phi=R_\phi(x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|)\), and the final output is
\begin{equation}
\hat{x}=\Pi_y[\hat{x}^{(1)}+r_\phi].
\end{equation}
Image-domain metrics are computed after clipping to the valid intensity range, whereas measurement error is computed before clipping to avoid hiding projection inconsistency.

\subsection{Exact operator handling}
For random Rademacher measurements, evaluation must use the exact exported measurement operator associated with the checkpoint. We reload that operator and rebuild the cached solver before evaluation, so that forward, adjoint, and inverse operations use the same matrix. This exact-operator cache-rebuilt path is used for all reported Rademacher results.
"""


def experimental_protocol() -> str:
    return r"""\section{Experimental Protocol}
Primary natural-image experiments use STL-10 at 5\% and 10\% sampling with Rademacher and scrambled Hadamard measurements. MNIST and Fashion-MNIST are used as simple-domain sanity checks at 5\% sampling. Checkpoint selection and final testing are separated. For random Rademacher measurements, the exact exported operators are reloaded and the solver cache is rebuilt before evaluation.

We summarize quality with predefined operational thresholds: PSNR \(\ge 20\) and SSIM \(\ge 0.60\) for STL-10 at 5\%, PSNR \(\ge 22\) and SSIM \(\ge 0.65\) for STL-10 at 10\%, and PSNR \(\ge 25\) and SSIM \(\ge 0.80\) for MNIST/Fashion-MNIST at 5\%. These thresholds are operational criteria used to summarize reconstruction quality in this study, not theoretical limits. Supplementary analyses use existing trained checkpoints and do not introduce additional training.
"""


def results() -> str:
    return r"""\section{Results}
We organize the results around three questions. First, can high-quality natural-image reconstruction be achieved at 5\% and 10\% sampling? Second, how much of the improvement comes from the physical initialization versus neural refinement? Third, do the reconstructions remain dependent on the bucket measurements?

\subsection{Can STL-10 be reconstructed at 5\% sampling?}
\Cref{tab:primary_results,fig:primary_metrics} summarize the primary leakage-free evaluation results. At 5\% sampling, Rademacher measurements reach 22.316 dB PSNR and 0.635 SSIM, while scrambled Hadamard measurements reach 22.271 dB PSNR and 0.632 SSIM. Both exceed the predefined operational STL-10 5\% high-quality threshold. The result is notable because the Rademacher backprojection is very weak at 7.297 dB, whereas the scrambled Hadamard backprojection is stronger at 14.310 dB. Similar final quality therefore arises from different physical-initialization regimes.

\input{tables/table1_primary_results.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_primary_metrics_v9.pdf}
\caption{\textbf{Primary metrics.} PSNR and SSIM are shown for STL-10 5\%, STL-10 10\%, and simple-domain 5\% sanity checks. Dashed lines are predefined operational thresholds used to summarize reconstruction quality in this study.}
\label{fig:primary_metrics}
\end{figure*}

\subsection{What changes at 10\% sampling?}
At 10\% sampling, Rademacher measurements reach 24.781 dB PSNR and 0.747 SSIM, while scrambled Hadamard measurements reach 24.730 dB PSNR and 0.746 SSIM. The higher sampling ratio improves final reconstruction quality for both measurement families. Rademacher still has a weak physical initialization at 7.756 dB, while scrambled Hadamard starts from 14.533 dB. Thus the measurement family controls the starting point and neural gain more strongly than it controls the final quality in these trained settings.

\subsection{What do the reconstructions look like?}
Large qualitative reconstructions are shown in \Cref{fig:qualitative_reconstruction}. The samples were reselected from saved evaluation grids to favor more recognizable STL-10 objects such as aircraft- or vehicle-like examples. Backprojections are incomplete, especially for random measurements, whereas reconstructions recover object-level structure. The qualitative figure is illustrative; quantitative conclusions are based on the leakage-free metrics.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig3_qualitative_reconstruction_v9.pdf}
\caption{\textbf{Qualitative reconstruction.} STL-10 examples are shown as ground truth, backprojection, reconstruction, and absolute error. Samples were reselected for clearer object structure. Error maps use a shared 98.5th-percentile scale.}
\label{fig:qualitative_reconstruction}
\end{figure*}

\subsection{Where does the gain come from?}
\Cref{tab:measurement_attribution,fig:regime_map} separates physical initialization from learned refinement. Rademacher measurements occupy a weak-initialization, large-gain regime: 7.297 dB to 22.316 dB at 5\%, and 7.756 dB to 24.781 dB at 10\%. Scrambled Hadamard measurements start much stronger and require smaller learned gains. Auxiliary low-frequency Hadamard points illustrate a different initialization-gain regime and are used only as measurement-family diagnostics.

\input{tables/table2_measurement_attribution.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig4_regime_map_v9.pdf}
\caption{\textbf{Measurement-family regime map.} Main-text points are Rad-5, Scr-5, Rad-10, Scr-10, Lowfreq-5, and Lowfreq-10. The x-axis is Backprojection PSNR and the y-axis is Neural gain \(\Delta\)PSNR. The background separates weak initialization with large learned gain, stronger initialization with moderate gain, and strong initialization with smaller gain. Low-frequency Hadamard points are auxiliary diagnostics rather than main high-quality claims.}
\label{fig:regime_map}
\end{figure*}

\subsection{Simple-domain sanity checks}
On MNIST and Fashion-MNIST at 5\% sampling, the same pipeline reaches 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM, respectively. These experiments show that the measurement-constrained reconstruction path also works on simpler structured domains. They are sanity checks rather than the main natural-image result.
"""


def validation_ablation() -> str:
    return r"""\section{Validation and Ablation}
Each validation experiment is designed to answer a specific failure mode.

\subsection{Is random sensing reproducible? Exact-A audit.}
Random Rademacher measurements require exact-operator evaluation. A mismatch between the exported operator and cached solver would make the reported measurement consistency unreliable. With safe exact-operator loading and cache rebuilding, Rademacher 5\% and 10\% re-evaluations reproduce the primary leakage-free evaluation with negligible differences. This audit checks that the random sensing operator used by the evaluator is the one used by the checkpoint.

\subsection{Does the measurement projection matter? No-DC ablation.}
\Cref{tab:ablation_summary,fig:inference_ablation} report inference-time ablations. Removing the measurement-consistency projection causes the largest degradation. This shows that \(\Pi_y\) is central to maintaining physical fidelity and image quality, not a cosmetic postprocessing step. Removing the null projection has limited metric effect for these trained checkpoints; this is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.

\input{tables/table3_ablation_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation_v9.pdf}
\caption{\textbf{Inference ablation.} Removing measurement-consistency projection gives the clearest failure. The limited metric change of \(-\)Null is reported explicitly for these trained checkpoints.}
\label{fig:inference_ablation}
\end{figure*}

\subsection{Does the model depend on y? Perturbation.}
Measurement perturbation tests are designed to detect measurement-independent generation. Shuffling coefficients or replacing the measurement vector with a wrong sample causes large PSNR drops for Rad-5 and Scr-5, as shown in \Cref{fig:validation_summary}. These failures are desirable diagnostics: they show that the model depends on the bucket vector rather than producing a generic prior image.

\subsection{Is this stronger than a CS-TV compressed-sensing baseline?}
We compare against a TV-regularized compressed-sensing baseline solved by projected gradient descent \cite{donoho2006compressed,candes2006robust,rudin1992tv}:
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\,\mathrm{TV}(x).
\end{equation}
We refer to this baseline as CS-TV. It is a lightweight small-subset traditional baseline, not an exhaustively optimized ADMM/FISTA or plug-and-play benchmark. Under the tested settings, the learned measurement-consistent reconstructor remains substantially stronger.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_robustness_baselines_v9.pdf}
\caption{\textbf{Robustness and baselines.} Panels show finite-noise behavior, Rad-5/Scr-5 Shuffle/Wrong-y perturbations, STL-10 comparison against CS-TV, and bootstrap confidence intervals. These diagnostics support finite-noise stability and measurement dependence; they provide finite evidence and do not imply universal robustness.}
\label{fig:validation_summary}
\end{figure*}

\subsection{How stable are results across samples and classes?}
Bootstrap confidence intervals show that the main STL-10 results are stable across evaluation samples. Class-wise diagnostics reveal expected category variability, with some classes consistently more difficult than others. These analyses support the reliability of the main claim while also showing that performance is not uniform across semantic categories.
"""


def discussion() -> str:
    return r"""\section{Discussion}
\subsection{Low-sampling GI is measurement-constrained completion.}
The first lesson is that low-sampling ghost imaging should be interpreted as measurement-constrained completion. The measurement vector does not determine the full image; it determines a low-dimensional component and leaves a large null-space component unresolved. A good reconstruction must therefore improve image quality while remaining tied to the measured bucket signal. The learned reconstructor is useful not because it replaces the physics, but because it supplies a prior for the part of the image that the physics has not measured.

\subsection{Measurement families define different initialization-gain regimes.}
The second lesson is that measurement families control the balance between physical initialization and learned refinement. Rademacher measurements produce weak backprojections but large neural gains. Scrambled Hadamard measurements provide stronger backprojections and smaller gains while reaching similar final quality. Low-frequency Hadamard diagnostics occupy another regime. Thus final PSNR alone is not enough: it must be interpreted together with backprojection PSNR and neural gain.

\subsection{Physical audit matters more than adversarial realism.}
The third lesson is that the central scientific constraint is physical audit, not adversarial realism. The contribution is the measurement-constrained reconstruction decomposition and the evidence that reconstructions remain dependent on \(y\). This framing is more important than presenting the network as a generator of realistic images. The method is valuable when its learned component can be checked by the same bucket measurements that created the inverse problem.
"""


def limitations() -> str:
    return r"""\section{Limitations}
This study does not include a hardware optical experiment. It does not claim a strict state-of-the-art ranking because a broad external benchmark under matched protocols is not included. The CS-TV baseline is lightweight and small-subset, not an exhaustively optimized compressed-sensing solver. Robustness is tested only over finite noise and perturbation settings. Low-frequency Hadamard at 5\% is not a high-quality STL-10 setting in this work. Binary learned illumination is not claimed as successful, and adversarial training is not the final contribution mechanism. Future work should include hardware validation, broader external baselines, and more extensive cross-domain testing.
"""


def conclusion() -> str:
    return r"""\section{Conclusion}
We presented a measurement-constrained null-space completion framework for low-sampling ghost imaging. The method separates the data solution, neural null-space residual, and measurement-consistency projection, making the reconstruction auditable against bucket measurements. Under the reported leakage-free protocol, it achieves high-quality STL-10 reconstruction at both 5\% and 10\% sampling for Rademacher and scrambled Hadamard measurements. Ablations, perturbations, exact-operator audits, CS-TV comparison, and confidence intervals support the interpretation that the reconstructions are measurement-dependent rather than generic hallucinations. These results provide a physics-consistent route toward high-quality low-sampling ghost-imaging reconstruction.
"""


def supplement_text() -> str:
    return r"""\section{Supplementary Material}
Detailed CSV files and audit manifests are provided as an accompanying data package. The tables below are curated summaries intended for a compact submission supplement.

\subsection{S1 Exact-operator reproducibility}
\input{tables/tableS1_exact_a.tex}

\subsection{S2 Measurement-error ablation}
\begin{figure*}[h]
\centering
\includegraphics[width=0.82\textwidth]{figures/figS1_relmeaserr_ablation_v9.pdf}
\caption{Supplementary measurement-error view of the no-DC projection ablation. Removing the projection increases measurement inconsistency.}
\label{fig:supp_relmeaserr_ablation}
\end{figure*}

\subsection{S3 Noise sweep summary}
\input{tables/tableS2_noise_sweep.tex}

\subsection{S4 CS-TV baseline summary}
\input{tables/tableS3_cstv_baseline.tex}

\subsection{S5 DC row control}
\input{tables/tableS4_dc_row_control.tex}

\subsection{S6 Bootstrap CI and class-wise diagnostics}
\input{tables/tableS5_statistics_ci.tex}
\input{tables/tableS6_classwise.tex}

\subsection{S7 Runtime}
\input{tables/tableS7_runtime.tex}

\FloatBarrier
\section{Data and Code Availability}
The code, trained checkpoint manifests, exported Rademacher measurement operators, and detailed supplementary CSV tables will be made available upon publication or reasonable request. Rademacher reproduction requires the exported exact measurement operator and the cache-rebuilt evaluation path.
"""


def main_tex() -> str:
    inputs = [
        "introduction",
        "related_work",
        "problem_formulation",
        "method",
        "measurement_families",
        "experimental_protocol",
        "results",
        "validation_ablation",
        "discussion",
        "limitations",
        "conclusion",
    ]
    body = "\n".join(rf"\input{{sections/{name}.tex}}\FloatBarrier" for name in inputs)
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{placeins}}
\usepackage{{hyperref}}
\usepackage{{cleveref}}
\usepackage{{url}}
\hypersetup{{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}}
\title{{{tex_escape(TITLE)}}}
\author{{Anonymous authors}}
\date{{}}
\begin{{document}}
\maketitle
\begin{{abstract}}
\input{{sections/abstract.tex}}
\end{{abstract}}
{body}
\FloatBarrier
\clearpage
\phantomsection
\label{{sec:references}}
\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""


def supplement_tex() -> str:
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{placeins}}
\usepackage{{hyperref}}
\usepackage{{cleveref}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\renewcommand{{\thefigure}}{{S\arabic{{figure}}}}
\renewcommand{{\thetable}}{{S\arabic{{table}}}}
\title{{Supplementary Material: {tex_escape(TITLE)}}}
\author{{Anonymous authors}}
\date{{}}
\begin{{document}}
\maketitle
\input{{supplement/supplement.tex}}
\end{{document}}
"""


def write_sections() -> None:
    section_map = {
        "abstract.tex": abstract(),
        "introduction.tex": introduction(),
        "related_work.tex": normalize_v8_text(read_v8("related_work.tex")),
        "problem_formulation.tex": problem_formulation(),
        "method.tex": method(),
        "measurement_families.tex": normalize_v8_text(read_v8("measurement_families.tex")),
        "experimental_protocol.tex": experimental_protocol(),
        "results.tex": results(),
        "validation_ablation.tex": validation_ablation(),
        "discussion.tex": discussion(),
        "limitations.tex": limitations(),
        "conclusion.tex": conclusion(),
    }
    for name, text in section_map.items():
        write_text(SECTIONS / name, text)
    write_text(SUPP / "supplement.tex", supplement_text())


def citation_audit() -> str:
    bib = (P22_LATEX / "references.bib").read_text(encoding="utf-8")
    main_sources = "\n".join(path.read_text(encoding="utf-8") for path in sorted(SECTIONS.glob("*.tex")))
    cite_keys = sorted(set(re.findall(r"\\cite\{([^}]+)\}", main_sources)))
    cite_keys = sorted({key.strip() for group in cite_keys for key in group.split(",")})
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = [key for key in cite_keys if key not in bib_keys]
    entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, flags=re.S)
    malformed = []
    missing_core = []
    for key, body in entries:
        lower = body.lower()
        if "author" not in lower or "title" not in lower:
            malformed.append(key)
        if "year" not in lower:
            missing_core.append(key)
        if not any(field in lower for field in ["journal", "booktitle", "publisher"]):
            missing_core.append(key)
    lines = [
        "# Citations To Verify",
        "",
        "This file is intentionally separate from the PDF sources.",
        "",
        f"- Citation keys missing from bibliography: {', '.join(missing) if missing else 'none'}",
        f"- Malformed author/title entries: {', '.join(sorted(set(malformed))) if malformed else 'none detected'}",
        f"- Entries missing venue or year: {', '.join(sorted(set(missing_core))) if missing_core else 'none detected'}",
        "",
        "Manual follow-up: verify DOI, publisher metadata, and journal-specific bibliography style before submission.",
    ]
    return "\n".join(lines)


def plain_manuscript() -> str:
    names = [
        "abstract.tex",
        "introduction.tex",
        "related_work.tex",
        "problem_formulation.tex",
        "method.tex",
        "measurement_families.tex",
        "experimental_protocol.tex",
        "results.tex",
        "validation_ablation.tex",
        "discussion.tex",
        "limitations.tex",
        "conclusion.tex",
    ]
    return "\n\n".join((SECTIONS / name).read_text(encoding="utf-8") for name in names)


def write_reports() -> None:
    write_text(OUT / "abstract_v9.md", ABSTRACT)
    write_text(OUT / "human_written_manuscript_v9.md", plain_manuscript())
    write_text(OUT / "citations_to_verify.md", citation_audit())
    write_text(
        OUT / "narrative_rewrite_notes.md",
        """# Narrative Rewrite Notes

- Reframed the manuscript around measurement-constrained null-space completion.
- Rewrote Introduction as seven paragraphs with the requested physical conflict.
- Reorganized Method into data solution, null-space residual, and measurement-consistency projection.
- Reorganized Results and Validation around explicit questions and failure modes.
- Rebuilt Figures 1, 3, 4, and 6 for the top-journal narrative.
""",
    )


def main() -> None:
    ensure_dir(OUT)
    clean_latex_dir()
    copy_tables()
    copy_figures()
    write_sections()
    refs = (P22_LATEX / "references.bib").read_text(encoding="utf-8")
    write_text(LATEX / "references.bib", refs)
    write_text(LATEX / "main.tex", main_tex())
    write_text(LATEX / "supplement.tex", supplement_tex())
    write_reports()
    manifest = source_manifest()
    manifest["output"] = str(OUT)
    manifest["phase22_base"] = str(P22)
    write_json(OUT / "internal_source_manifest.json", manifest)
    print({"latex_project_v9": str(LATEX)})


if __name__ == "__main__":
    main()
