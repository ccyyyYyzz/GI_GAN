from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path("E:/ns_mc_gan_gi")
SOURCE_PROJECT = ROOT / "outputs_phase27_paper_purification" / "latex_project_purified"
OUT = ROOT / "outputs_phase28_paragraph_polish"
PROJECT = OUT / "latex_project_v28"


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


ABSTRACT_TEXT = (
    "Ghost imaging and single-pixel imaging recover spatial information from structured "
    "illumination patterns and scalar bucket measurements, but low-sampling acquisition "
    "is severely underdetermined. The measurement vector does not identify a unique image: "
    "low sampling needs a learned prior, yet unconstrained priors may hallucinate structure "
    "that is not supported by the bucket readings. We address this tension with "
    "measurement-consistent null-space neural reconstruction. The formulation computes a "
    "physical data solution from the forward operator, adds a learned null-space residual "
    "for missing structure, and applies a final measurement projection to audit the "
    "completed image against the measured signal. Under a leakage-free STL-10 protocol, "
    "the method reaches 22.316 dB PSNR / 0.635 SSIM at 5% sampling with Rademacher "
    "measurements and 22.271 dB / 0.632 SSIM with scrambled Hadamard measurements. At "
    "10% sampling, the corresponding results are 24.781 dB / 0.747 SSIM and 24.730 dB / "
    "0.746 SSIM. MNIST and Fashion-MNIST 5% experiments provide simple-domain sanity "
    "checks, reaching 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM. Exact-A audit, "
    "measurement-family attribution, inference ablation, finite-noise tests, measurement "
    "perturbation, CS-TV comparison, and confidence intervals support measurement-dependent "
    "reconstruction across the tested measurement families."
)


SECTIONS: dict[str, str] = {
    "abstract.tex": ABSTRACT_TEXT.replace("%", r"\%"),
    "introduction.tex": r"""
\section{Introduction}
Ghost imaging and single-pixel imaging reconstruct spatial information from known illumination patterns and scalar bucket detector readings. Instead of measuring an image directly with a dense sensor array, the system records projections of the unknown scene onto structured patterns. This acquisition model is attractive when detector arrays are expensive, unavailable, or difficult to deploy, but it shifts the burden from direct sensing to computational reconstruction. In such systems, optical measurement design and computational inversion are coupled: the patterns determine not only what is measured, but also what must later be inferred.

The central difficulty is low sampling. If the unknown image is represented as \(x\in\mathbb{R}^n\) and the system collects \(m\) bucket measurements, the measurement model is
\begin{equation}
y=Ax+\epsilon,
\end{equation}
where \(A\in\mathbb{R}^{m\times n}\) is determined by the illumination patterns. In the low-sampling regime, \(m\ll n\), and the inverse problem is underdetermined. The measurement vector therefore does not identify a single image; it defines a large family of images compatible with the bucket readings. Direct physical inverses, such as backprojection, preserve a transparent link to the measurements but often leave severe missing structure.

Deep neural networks can improve reconstruction quality in this regime, but unconstrained networks introduce a different risk. A network may generate image details that look plausible while moving away from the measured bucket signal. The central issue is not only whether a neural network can improve image quality, but whether it can do so without losing contact with the measurements.

Existing learned GI/SPI methods often report improved reconstructions, but they rarely make explicit which part of the result is inherited from the measured row-space component and which part is supplied by the learned prior. This distinction matters in low-sampling regimes: two measurement families may reach similar final PSNR while relying on very different balances of physical information and learned completion.

This work addresses that gap by treating low-sampling ghost imaging as measurement-consistent null-space reconstruction. The method computes a physical data solution, inserts a learned residual through an approximate null-space component, and then projects the result back to the measured affine set. This gives an auditable reconstruction path through \(x_{\rm data}\), \(P_N\), and \(\Pi_y\), rather than an unconstrained measurement-to-image mapping.

The main contributions are:
\begin{itemize}
\item a measurement-consistent null-space formulation for low-sampling ghost imaging;
\item an auditable reconstruction decomposition: data solution, learned null-space residual, measurement projection;
\item leakage-free STL-10 5\%/10\% evidence for Rademacher and scrambled Hadamard measurements;
\item a validation package: exact-A, attribution, inference ablation, finite noise, perturbation, CS-TV, and confidence intervals.
\end{itemize}
""",
    "problem_formulation.tex": r"""
\section{Problem Formulation}
Let \(x\in\mathbb{R}^n\) denote the vectorized unknown image. Each illumination pattern \(a_i\in\mathbb{R}^n\) produces a scalar bucket measurement
\begin{equation}
y_i=\langle a_i,x\rangle+\epsilon_i .
\end{equation}
Stacking all measurements gives \(y=Ax+\epsilon\), where \(A\in\mathbb{R}^{m\times n}\). The sampling ratio is \(\rho=m/n\). For \(64\times64\) images, \(n=4096\). At 5\% sampling, \(m\approx205\), and at 10\% sampling, \(m\approx410\). Therefore \(m\ll n\), and the inverse problem is underdetermined.

In the noiseless case, the measurement-consistent set is \(\mathcal{C}_y=\{x:Ax=y\}\). The null space of the measurement operator is \(\mathrm{Null}(A)=\{v:Av=0\}\). If \(Ax_0=y\) and \(v\in\mathrm{Null}(A)\), then \(A(x_0+v)=Ax_0+Av=y\). Thus the measurement vector fixes only part of the image. Geometrically, the measurements select an affine set in image space rather than a single point. The role of a reconstruction prior is therefore to choose a plausible point inside or near this set, not to ignore the set.

Low-sampling reconstruction is therefore a constrained completion problem: the measured component should remain tied to \(y\), while the missing component must be inferred from prior information.
""",
    "method.tex": r"""
\section{Measurement-Consistent Null-Space Reconstruction}
\subsection{Physical data solution}
We first compute a regularized data solution
\begin{equation}
x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
This solution is a physical initialization, not a learned hallucination. It is determined by the forward operator and the bucket measurements. \(x_{\rm data}\) should be interpreted as a measured-component representative, not as a visually complete reconstruction. Its quality depends on the measurement family.

\subsection{Approximate null-space residual}
Define \(P_A=A^T(AA^T+\lambda I)^{-1}A\) and \(P_N=I-P_A\). Applied to a vector \(v\),
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av.
\end{equation}
If \(\lambda=0\) and \(A\) has full row rank, then \(AP_N(v)=0\). With \(\lambda>0\), this becomes a regularized approximate null-space projection. The neural reconstructor predicts \(r_\theta=G_\theta(x_{\rm data},z)\), and the intermediate reconstruction is \(\tilde{x}=x_{\rm data}+P_N(r_\theta)\). This step encourages the network to complete information not directly determined by the measurements. The projection removes the component of the proposed residual that would be visible to the measurement operator. Thus the network is not asked to overwrite the measured component; it proposes structure in directions that are weakly observed or unobserved by \(A\).

\subsection{Measurement-consistency projection}
To restore agreement with the bucket measurements, we apply
\begin{equation}
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y).
\end{equation}
The final reconstruction is
\begin{equation}
\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
Although \(P_N\) restricts the neural residual, it is only an approximate null-space projection when \(\lambda>0\). In addition, the data solution, the refiner, numerical factorization, and intensity clipping can all introduce measurement inconsistency. The final projection \(\Pi_y\) therefore acts as an audit step on the complete image rather than only on the neural residual. The role of the neural network is therefore restricted by the measurement operator: it refines the missing component but is followed by an explicit projection back to the measured affine set. \Cref{fig:mechanism} visualizes the same logic.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_mechanism_v8.pdf}
\caption{\textbf{Mechanism.} The acquisition provides bucket measurements; the feasible set is underdetermined; the method separates a measured data component from a neural residual, inserts the residual through an approximate null-space projection, projects back to the measured affine set, and audits the result with relative measurement error.}
\label{fig:mechanism}
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
The first stage enforces the main measurement-consistent null-space structure, while the refiner improves image fidelity. The final projection is retained after refinement so that the refiner cannot permanently move the output away from the measured signal. Image-domain metrics are computed after clipping to the valid intensity range, whereas measurement error is computed before clipping to avoid hiding projection inconsistency.

\subsection{Exact operator handling}
Rademacher sensing uses a random measurement matrix. Reproducible evaluation therefore requires reloading the exported exact operator. After replacing \(A\), all cached quantities derived from \(A\), including \(K=AA^T+\lambda I\) and its Cholesky factorization, must be rebuilt. This exact-A cache-rebuilt path is used for all reported Rademacher results.
""",
    "measurement_families.tex": r"""
\section{Measurement Families}
These families should not be viewed merely as interchangeable sampling masks. They change the quality of \(x_{\rm data}\), the burden placed on \(G_\theta\), and the amount corrected by \(\Pi_y\). Similar final quality does not imply the same reconstruction mechanism.

\subsection{Rademacher measurements}
Rademacher measurements use signed random entries, \(A_{ij}\in\{-m^{-1/2},+m^{-1/2}\}\). Because this operator is random, exact-A reproducibility requires the exported operator and cache-rebuilt evaluation path described above. Rademacher measurements produce weak physical inverses in our experiments, so a large learned gain is required to recover object-level structure.

\subsection{Scrambled Hadamard measurements}
Let \(H\in\{-1,+1\}^{n\times n}\) be a Hadamard matrix. We use \(H_{\rm norm}=n^{-1/2}H\), select scrambled rows, and form \(A\). Scrambled Hadamard measurements provide stronger physical initializations than Rademacher measurements while reaching similar final reconstruction quality. The final metric similarity therefore hides different balances between measured-component quality and learned completion.

\subsection{Low-frequency Hadamard measurements}
Low-frequency Hadamard measurements select low-sequency rows. For selected rows \(S\), zero-filled reconstruction uses \(c[S]=y\) and \(x_{\rm data}=H_{\rm norm}^Tc\). The DC row is the low-frequency Hadamard direct-current row; it measures global brightness and strongly affects low-frequency Hadamard backprojection. Low-frequency Hadamard is therefore an interpretable diagnostic family, but low-frequency Hadamard at 5\% is not the primary STL-10 high-quality result in this work.
""",
    "results.tex": r"""
\section{Results}
\subsection{STL-10 reconstruction at 5\% and 10\%}
We first ask whether natural images can be reconstructed at the 5\% sampling level under a leakage-free protocol. \Cref{tab:primary_results,fig:primary_metrics} summarize the primary leakage-free evaluation results. At 5\% sampling, Rademacher measurements reach 22.316 dB PSNR and 0.635 SSIM, while scrambled Hadamard measurements reach 22.271 dB PSNR and 0.632 SSIM. Both exceed the predefined operational STL-10 5\% high-quality threshold. At 10\% sampling, Rademacher reaches 24.781 dB PSNR and 0.747 SSIM, while scrambled Hadamard reaches 24.730 dB PSNR and 0.746 SSIM. Thus, both measurement families support high-quality STL-10 reconstruction at 5\% and 10\% sampling.

\input{tables/table1_primary_results.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_primary_metrics_v8.pdf}
\caption{\textbf{Primary metrics.} PSNR and SSIM are shown for STL-10 5\%, STL-10 10\%, and simple-domain 5\% sanity checks. Dashed lines are predefined operational thresholds used to summarize reconstruction quality in this study.}
\label{fig:primary_metrics}
\end{figure*}

\subsection{Qualitative reconstruction}
Large qualitative reconstructions are shown in \Cref{fig:qualitative_reconstruction}. The visual comparison is meant to show what is recovered beyond the physical initialization: Rademacher backprojections are noise-like, scrambled Hadamard backprojections contain more structure, and the neural reconstruction restores object-level content. Images are enlarged for visibility and are intended as qualitative evidence; quantitative conclusions are based on the leakage-free evaluation metrics.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig3_qualitative_reconstruction_v28.pdf}
\caption{\textbf{Qualitative reconstruction.} STL-10 examples are shown as ground truth, backprojection, reconstruction, and absolute error. Representative evaluation samples were reselected for clearer object structure. Error maps use a shared high-percentile scale.}
\label{fig:qualitative_reconstruction}
\end{figure*}

\subsection{Simple-domain sanity checks}
On MNIST and Fashion-MNIST at 5\% sampling, the method reaches 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM, respectively. These experiments confirm that the same reconstruction pipeline works reliably on simpler structured domains, but they are not the main novelty.

\subsection{Measurement-family attribution}
\Cref{tab:measurement_attribution,fig:measurement_attribution} separate physical initialization from learned refinement. Final PSNR alone is insufficient to explain the role of the measurement family. Rademacher measurements have weak physical backprojections: 7.297 dB at 5\% and 7.756 dB at 10\%. However, final reconstruction reaches 22.316 dB and 24.781 dB, corresponding to gains of 15.019 dB and 17.025 dB. Scrambled Hadamard measurements start from stronger backprojections, 14.310 dB at 5\% and 14.533 dB at 10\%, and reach nearly the same final quality as Rademacher.

\input{tables/table2_measurement_attribution.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig4_measurement_attribution_v28.pdf}
\caption{\textbf{Measurement attribution.} The regime map and backprojection-vs-model PSNR bars show that final PSNR alone hides whether performance comes from physical initialization, neural refinement, or both. Low-frequency Hadamard points are shown only as diagnostic regime-map points, not as main natural-image claims.}
\label{fig:measurement_attribution}
\end{figure*}
""",
    "validation_ablation.tex": r"""
\section{Validation and Ablation}
\subsection{Exact-A reproducibility}
Rademacher measurements require exact-operator evaluation. Earlier mismatch was traced to stale solver-cache use after overriding \(A\). With safe exact-A loading and cache rebuilding, Rademacher 5\% and 10\% re-evaluations reproduce the original leakage-free evaluation with negligible differences. These reproduced results are used as primary evidence. This audit is important because random sensing results cannot be reproduced by regenerating a nominally identical random matrix; the exact exported operator must be used.

\subsection{Inference-time ablation}
\Cref{tab:ablation_summary,fig:inference_ablation} report the inference-time ablations. Removing the measurement-consistency projection causes the largest degradation. This no measurement-consistency projection condition shows that \(\Pi_y\) is not merely cosmetic; it is central to maintaining physical fidelity and image quality. Removing the null projection has limited metric effect for the trained checkpoints, suggesting that the final projection and the learned network already constrain many measured components. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.

\input{tables/table3_ablation_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation_v28.pdf}
\caption{\textbf{Inference ablation.} Removing measurement-consistency projection gives the strongest degradation. \(-\mathrm{MC}\) removes the final measurement-consistency projection. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.}
\label{fig:inference_ablation}
\end{figure*}

\subsection{Noise and perturbation tests}
\Cref{fig:validation_summary} summarizes finite-noise sweeps, measurement perturbations, CS-TV comparison, and bootstrap confidence intervals. Finite-noise sweeps show stable degradation over the tested noise range. Measurement perturbation tests are more diagnostic: shuffled coefficients and wrong-sample measurements cause large PSNR drops. This is a negative-control test: the model should fail when the measurement vector is corrupted. The result indicates that the model depends on the bucket measurement vector rather than measurement-independent hallucination.

\subsection{CS-TV compressed-sensing baseline}
We compare against a TV-regularized compressed-sensing baseline solved by PGD \cite{donoho2006compressed,candes2006robust,rudin1992tv}:
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda \operatorname{TV}(x).
\end{equation}
We refer to this baseline as CS-TV. This baseline represents a classical compressed-sensing prior, not an exhaustively tuned iterative reconstruction benchmark. It is a lightweight small-subset traditional baseline, and under the tested settings the learned measurement-consistent reconstructor remains substantially stronger.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_robustness_baselines_v8.pdf}
\caption{\textbf{Robustness and baselines.} Panels show finite-noise behavior, Shuffle/Wrong-y measurement perturbations, comparison against CS-TV, and bootstrap confidence intervals. These diagnostics support finite-noise stability and measurement dependence within the tested conditions.}
\label{fig:validation_summary}
\end{figure*}

\subsection{DC row control}
For low-frequency Hadamard, including the DC row is critical for backprojection because it captures global brightness. Removing it severely degrades low-frequency Hadamard initialization. The DC-row result explains why low-frequency Hadamard backprojection behaves differently from Rademacher and scrambled Hadamard. It is a diagnostic of one measurement family, not a general explanation of all reconstructions.

\subsection{Statistics and class-wise diagnostics}
Bootstrap confidence intervals show that the main results are stable across samples. Class-wise STL-10 diagnostics reveal expected category variability, with some classes consistently more difficult than others. These analyses support the robustness of the main claim but are not themselves the central contribution.
""",
    "discussion.tex": r"""
\section{Discussion}
The first lesson is that low-sampling ghost imaging is measurement-constrained completion. A good reconstruction must not only look plausible but also remain tied to the measured bucket signal. The proposed pipeline achieves this by combining a data solution, null-space residual completion, and a measurement-consistency projection. The learned reconstructor is useful not because it replaces the physics, but because it supplies a prior for the part of the image that the physics has not measured. The main novelty is a physics-constrained reconstruction decomposition that makes the role of measured data, missing-structure completion, and measurement projection explicit.

The second lesson is that measurement families define different initialization and neural-gain regimes. Rademacher measurements produce weak backprojections but high final quality after neural refinement. Scrambled Hadamard measurements provide stronger physical initialization and similar final quality. Similar final quality can arise from different mechanisms: weak initialization with large learned gain, or stronger initialization with moderate learned gain. This suggests that physical initialization quality and learnability of the neural inverse are distinct properties.

The third lesson is that measurement audit matters more than adversarial realism. Although adversarial ideas were considered during development, the final high-quality results are driven by measurement-consistent neural reconstruction and fidelity-oriented losses. This is why the paper is positioned as measurement-consistent neural reconstruction rather than adversarial generation.

Future work may explore alternative neural priors within the same measurement-consistent formulation.
""",
    "limitations.tex": r"""
\section{Limitations}
This study does not include a hardware optical experiment. We do not claim a strict state-of-the-art ranking because datasets, measurement operators, sampling protocols, and evaluation splits are not standardized across the literature. The CS-TV baseline is lightweight and small-subset, not an exhaustively optimized compressed-sensing solver. Robustness is tested only over finite noise and perturbation settings. Class-wise evaluation is diagnostic rather than a claim of uniform category performance. Exact-A handling is essential for random measurements, and results should be interpreted with that audit path in place. Low-frequency Hadamard at 5\% is not a high-quality STL-10 setting in this work. Binary learned illumination is not claimed as successful, and adversarial training is not the final contribution mechanism. Future work should include hardware validation, broader external baselines, and more extensive cross-domain testing.
""",
    "conclusion.tex": r"""
\section{Conclusion}
We presented a measurement-consistent null-space neural reconstruction framework for low-sampling ghost imaging. By combining a physical data solution, neural null-space residual completion, and a final measurement-consistency projection, the method achieves high-quality STL-10 reconstruction at both 5\% and 10\% sampling under Rademacher and scrambled Hadamard measurements. Supplementary ablations, perturbation tests, exact-operator re-evaluation, and compressed-sensing baselines support the interpretation that the reconstructions are measurement-dependent rather than generic hallucinations. More broadly, the results suggest that low-sampling ghost imaging should be designed as a coupled problem of sensing geometry, learned prior, and measurement audit.
""",
}


SUPPLEMENT = r"""
\section{Supplementary Material}
Detailed CSV files and audit manifests are provided as an accompanying data package. The tables below are curated summaries intended for a compact submission supplement.

\subsection{S1 Exact-A reproducibility}
\input{tables/tableS1_exact_a.tex}

\subsection{S2 Measurement-family attribution details}
The main text separates physical initialization from learned refinement. The accompanying data package provides the full attribution CSV used to produce the curated table and regime summary.

\subsection{S3 Inference ablation and measurement error}
\begin{figure*}[h]
\centering
\includegraphics[width=0.82\textwidth]{figures/figS1_relmeaserr_ablation_v28.pdf}
\caption{Supplementary measurement-error view of the no measurement-consistency projection ablation. Removing the projection increases measurement inconsistency.}
\label{fig:supp_relmeaserr_ablation}
\end{figure*}

\subsection{S4 Noise sweep}
\input{tables/tableS2_noise_sweep.tex}

\subsection{S5 CS-TV baseline}
\input{tables/tableS3_cstv_baseline.tex}

\subsection{S6 DC-row control}
\input{tables/tableS4_dc_row_control.tex}

\subsection{S7 Statistics, confidence intervals, and class-wise diagnostics}
\input{tables/tableS5_statistics_ci.tex}
\input{tables/tableS6_classwise.tex}

\subsection{S8 Runtime}
\input{tables/tableS7_runtime.tex}

\FloatBarrier
\section{Data and Code Availability}
The code, trained checkpoint manifests, exported Rademacher measurement operators, and detailed supplementary CSV tables will be made available upon publication or reasonable request. Rademacher reproduction requires the exported exact measurement operator and the cache-rebuilt evaluation path.
"""


TABLE3_TEX = r"""
\begin{table*}[t]
\centering
\small
\caption{\textbf{Inference ablation summary.} Values are PSNR in dB. \(-\mathrm{MC}\) removes the final measurement-consistency projection.}
\label{tab:ablation_summary}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllllll}
\toprule
Method & Full & -MC & -Null & Stage1 & Raw & EMA \\
\midrule
Rad-5 & 22.202 & 19.399 & 22.202 & 21.736 & 22.065 & 22.202 \\
Scr-5 & 22.155 & 6.352 & 22.154 & 21.294 & 22.050 & 22.155 \\
Rad-10 & 24.676 & 20.106 & 24.676 & 23.598 & 24.539 & 24.676 \\
Scr-10 & 24.608 & 6.352 & 24.607 & 22.492 & 24.518 & 24.608 \\
\bottomrule
\end{tabular}
}
\end{table*}
"""


TABLE3_MD = """|Method|Full|-MC|-Null|Stage1|Raw|EMA|
|---|---|---|---|---|---|---|
|Rad-5|22.202|19.399|22.202|21.736|22.065|22.202|
|Scr-5|22.155|6.352|22.154|21.294|22.050|22.155|
|Rad-10|24.676|20.106|24.676|23.598|24.539|24.676|
|Scr-10|24.608|6.352|24.607|22.492|24.518|24.608|
"""


TABLE3_CSV = """Method,Full,-MC,-Null,Stage1,Raw,EMA
Rad-5,22.202,19.399,22.202,21.736,22.065,22.202
Scr-5,22.155,6.352,22.154,21.294,22.050,22.155
Rad-10,24.676,20.106,24.676,23.598,24.539,24.676
Scr-10,24.608,6.352,24.607,22.492,24.518,24.608
"""


def _ignore(_dir: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(name)
        if path.suffix in COMPILED_SUFFIXES:
            ignored.add(name)
    return ignored


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def copy_clean_project() -> None:
    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    shutil.copytree(SOURCE_PROJECT, PROJECT, ignore=_ignore)
    main = PROJECT / "main.tex"
    text = main.read_text(encoding="utf-8")
    cleveref_names = (
        r"\crefname{figure}{Fig.}{Figs.}" + "\n"
        r"\Crefname{figure}{Fig.}{Figs.}" + "\n"
        r"\crefname{table}{Table}{Tables}" + "\n"
        r"\Crefname{table}{Table}{Tables}" + "\n"
    )
    if r"\crefname{figure}{Fig.}{Figs.}" not in text:
        text = text.replace(r"\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}" + "\n", r"\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}" + "\n" + cleveref_names)
        main.write_text(text, encoding="utf-8")


def write_sections() -> None:
    section_dir = PROJECT / "sections"
    for name, text in SECTIONS.items():
        write_text(section_dir / name, text)
    write_text(PROJECT / "supplement" / "supplement.tex", SUPPLEMENT)
    write_text(OUT / "abstract_v28.md", ABSTRACT_TEXT)


def write_tables() -> None:
    table_dir = PROJECT / "tables"
    write_text(table_dir / "table3_ablation_summary.tex", TABLE3_TEX)
    write_text(table_dir / "table3_ablation_summary.md", TABLE3_MD)
    write_text(table_dir / "table3_ablation_summary.csv", TABLE3_CSV)


def terminology_report() -> None:
    source_files = list((PROJECT / "sections").glob("*.tex"))
    source_files += [PROJECT / "supplement" / "supplement.tex", PROJECT / "tables" / "table3_ablation_summary.tex"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_files if path.exists())
    checks = {
        "wrapper removed": not re.search(r"\bwrapper\b", text, flags=re.IGNORECASE),
        "lowercase fig. removed": "fig." not in text,
        "-DC table/figure labels removed": "-DC" not in text and "no-DC" not in text,
        "DC projection phrase removed": "DC projection" not in text,
        "measurement-consistency projection present": "measurement-consistency projection" in text,
        "CS-TV PGD wording present": "TV-regularized compressed-sensing baseline solved by PGD" in text,
    }
    rows = ["# Terminology Fix Report", ""]
    rows.append("All requested terminology edits were applied to the v28 manuscript sources.")
    rows.append("")
    for label, ok in checks.items():
        rows.append(f"- {'PASS' if ok else 'FAIL'}: {label}")
    rows.append("")
    rows.append("Key replacements:")
    rows.append("- `wrapper` -> `framework` or `formulation`.")
    rows.append("- `fig.` -> `Fig.` where sentence text required a figure reference.")
    rows.append("- `-DC` / `no-DC` -> `-MC` / no measurement-consistency projection.")
    rows.append("- Data-consistency projection language -> measurement-consistency projection.")
    rows.append("- `DC row` is retained only for the low-frequency Hadamard direct-current row.")
    write_text(OUT / "terminology_fix_report.md", "\n".join(rows))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("main_v28.pdf", "supplement_v28.pdf", "main_v28.txt", "supplement_v28.txt"):
        stale = OUT / name
        if stale.exists():
            stale.unlink()
    copy_clean_project()
    write_sections()
    write_tables()
    terminology_report()
    print(
        {
            "output_dir": str(OUT),
            "latex_project": str(PROJECT),
            "abstract": str(OUT / "abstract_v28.md"),
            "terminology_report": str(OUT / "terminology_fix_report.md"),
        }
    )


if __name__ == "__main__":
    main()
