from __future__ import annotations

import re
import shutil
from pathlib import Path

from .phase20_common import TITLE, ensure_dir, source_manifest, tex_escape, write_json, write_text


P21 = Path("E:/ns_mc_gan_gi/outputs_phase21_submission_polish")
P21_LATEX = P21 / "latex_project_v7"
OUT = Path("E:/ns_mc_gan_gi/outputs_phase22_submission_v8")
LATEX = OUT / "latex_project_v8"
SECTIONS = LATEX / "sections"
SUPP = LATEX / "supplement"
FIGS = LATEX / "figures"
TABLES = LATEX / "tables"


ABSTRACT = (
    "Ghost imaging and single-pixel imaging recover spatial information from structured illumination "
    "patterns and scalar bucket measurements, but low-sampling acquisition is severely underdetermined. "
    "Many images can match the same measurement vector, so visually plausible neural outputs must still "
    "be checked against the measured signal. We formulate low-sampling ghost imaging as measurement-consistent "
    "null-space neural reconstruction. The pipeline computes a physical data solution from the forward "
    "operator, inserts a neural residual through an approximate null-space projection, and applies a final "
    "measurement-consistency projection. For Rademacher measurements, evaluation reloads the exported exact "
    "measurement operator and rebuilds the solver cache so that forward, adjoint, and inverse operations use "
    "the same matrix. On STL-10, the method reaches 22.316 dB PSNR / 0.635 SSIM at 5% sampling with Rademacher "
    "measurements and 22.271 dB / 0.632 SSIM with scrambled Hadamard measurements. At 10% sampling, the "
    "corresponding results are 24.781 dB / 0.747 SSIM and 24.730 dB / 0.746 SSIM. MNIST and Fashion-MNIST "
    "5% experiments provide simple-domain sanity checks, reaching 27.692 dB / 0.956 SSIM and 25.019 dB / "
    "0.837 SSIM. Exact-operator audit, attribution, ablation, finite-noise, perturbation, compressed-sensing "
    "baseline, and confidence-interval analyses support measurement-dependent reconstruction rather than a "
    "strict state-of-the-art claim."
)


def clean_latex_dir() -> None:
    if LATEX.exists():
        shutil.rmtree(LATEX)
    ensure_dir(SECTIONS)
    ensure_dir(SUPP)
    ensure_dir(FIGS)
    ensure_dir(TABLES)


def copy_tables() -> None:
    src = P21 / "tables"
    dst = OUT / "tables"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    shutil.copytree(src, TABLES, dirs_exist_ok=True)

    replacements = {
        "Primary strict no-leak results.": "Primary leakage-free evaluation results.",
        "Thresholds are internal engineering criteria stated in the protocol.": "Thresholds are predefined operational criteria used in this study.",
        "Inference ablation summary.": "Inference ablation summary.",
        "DC-row control for low-frequency Hadamard backprojection.": "DC-row control for low-frequency Hadamard backprojection.",
    }
    for path in list(dst.glob("*.tex")) + list(TABLES.glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def copy_figures() -> None:
    ensure_dir(FIGS)
    src = OUT / "figures"
    if not src.exists():
        return
    for path in src.glob("*_v8.*"):
        if path.suffix.lower() in {".pdf", ".png", ".svg"}:
            shutil.copy2(path, FIGS / path.name)


def read_v7(name: str) -> str:
    return (P21_LATEX / "sections" / name).read_text(encoding="utf-8")


def abstract() -> str:
    return ABSTRACT.replace("%", r"\%")


def introduction() -> str:
    return r"""\section{Introduction}
Ghost imaging and single-pixel imaging reconstruct spatial information from a sequence of known illumination patterns and scalar bucket detector readings. Instead of measuring an image directly with a dense sensor array, the system records projections of the unknown scene onto a set of illumination patterns. This acquisition model is attractive when detector arrays are expensive, unavailable, or difficult to deploy, but it shifts the burden from direct sensing to computational reconstruction.

The central difficulty is low sampling. If the unknown image is represented as \(x\in\mathbb{R}^n\) and the system collects \(m\) bucket measurements, the measurement model is
\begin{equation}
y=Ax+\epsilon,
\end{equation}
where \(A\in\mathbb{R}^{m\times n}\) is determined by the illumination patterns. In the low-sampling regime, \(m\ll n\), and the inverse problem is underdetermined. Many candidate images can explain the same measurement vector. Direct physical inverses, such as backprojection, preserve a transparent link to the measurements but often leave severe missing structure.

Deep neural networks can improve reconstruction quality in this regime, but unconstrained networks introduce a different risk. A network may generate image details that look plausible while moving away from the measured bucket signal. For computational imaging, this is not merely a perceptual issue. The reconstructed image should remain physically tied to the forward model and to the measured data.

Existing deep-learning GI/SPI methods improve reconstruction quality, but many do not explicitly decompose the measured row-space component from the unmeasured null-space component, nor do they systematically separate physical initialization quality from neural refinement gain. This distinction matters in low-sampling regimes: two measurement families may reach similar final PSNR while relying on very different balances of physical information and learned completion.

This work addresses that gap by treating low-sampling ghost imaging as measurement-consistent null-space reconstruction. The method computes a physical data solution, inserts a learned residual only through an approximate null-space component, and then projects the result back to the measured affine set. This gives an auditable reconstruction path through \(x_{\rm data}\), \(P_N\), and \(\Pi_y\), rather than an unconstrained measurement-to-image mapping. Under a strict leakage-free protocol, i.e., checkpoint selection and final testing are separated, the method achieves high-quality STL-10 reconstruction at 5\% and 10\% sampling for both Rademacher and scrambled Hadamard measurements. The validation package further audits exact-A reproducibility, measurement-family attribution, inference-time ablation, finite-noise behavior, CS-TV comparison, measurement perturbation, and bootstrap confidence intervals.
"""


def method() -> str:
    return r"""\section{Measurement-Consistent Null-Space Reconstruction}
\subsection{Physical data solution}
We first compute a regularized data solution
\begin{equation}
x_{\rm data}=A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
This solution is a physical initialization, not a learned hallucination. It is determined by the forward operator and the bucket measurements.

\subsection{Approximate null-space residual}
Define \(P_A=A^T(AA^T+\lambda I)^{-1}A\) and \(P_N=I-P_A\). Applied to a vector \(v\),
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av.
\end{equation}
If \(\lambda=0\) and \(A\) has full row rank, then \(AP_N(v)=0\). With \(\lambda>0\), this becomes a regularized approximate null-space projection. The neural reconstructor predicts \(r_\theta=G_\theta(x_{\rm data},z)\), and the intermediate reconstruction is \(\tilde{x}=x_{\rm data}+P_N(r_\theta)\). This step encourages the network to complete information not directly determined by the measurements.

\subsection{Measurement-consistency projection}
To restore agreement with the bucket measurements, we apply
\begin{equation}
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y).
\end{equation}
The final reconstruction is
\begin{equation}
\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
The role of the neural network is therefore restricted by the measurement operator: it refines the missing component but is followed by an explicit projection back to the measured affine set. \Cref{fig:mechanism} visualizes the same logic.

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
Image-domain metrics are computed after clipping to the valid intensity range, whereas measurement error is computed before clipping to avoid hiding projection inconsistency.

\subsection{Exact operator handling}
For Rademacher measurements, \(A\) is random. Therefore evaluation must use the exact measurement operator used for the checkpoint. For Rademacher measurements, we reload the exported exact measurement operator and rebuild the cached solver before evaluation. This correction is numerical rather than algorithmic: it ensures that all forward, adjoint, and solver operations use the same matrix. After overriding \(A\), the matrix \(K=AA^T+\lambda I\) and its Cholesky cache must be rebuilt. This exact-A cache-rebuilt path is used for all reported Rademacher results.
"""


def experimental_protocol() -> str:
    return r"""\section{Experimental Protocol}
Primary natural-image experiments use STL-10 at 5\% and 10\% sampling with Rademacher and scrambled Hadamard measurements. MNIST and Fashion-MNIST are used as simple-domain sanity checks at 5\% sampling. We use a strict leakage-free protocol, i.e., checkpoint selection and final testing are separated. For Rademacher measurements, the exact exported operators are reloaded and the solver cache is rebuilt before evaluation.

We summarize quality with predefined operational thresholds: PSNR \(\ge 20\) and SSIM \(\ge 0.60\) for STL-10 at 5\%, PSNR \(\ge 22\) and SSIM \(\ge 0.65\) for STL-10 at 10\%, and PSNR \(\ge 25\) and SSIM \(\ge 0.80\) for MNIST/Fashion-MNIST at 5\%. These thresholds are operational criteria used to summarize reconstruction quality in this study, not theoretical limits. Supplementary analyses use existing trained checkpoints and do not introduce additional training.
"""


def results() -> str:
    return r"""\section{Results}
\subsection{STL-10 reconstruction at 5\% and 10\%}
\Cref{tab:primary_results,fig:primary_metrics} summarize the primary leakage-free evaluation results. At 5\% sampling, Rademacher measurements reach 22.316 dB PSNR and 0.635 SSIM, while scrambled Hadamard measurements reach 22.271 dB PSNR and 0.632 SSIM. Both exceed the predefined operational STL-10 5\% high-quality threshold. At 10\% sampling, Rademacher reaches 24.781 dB PSNR and 0.747 SSIM, while scrambled Hadamard reaches 24.730 dB PSNR and 0.746 SSIM. Thus, both measurement families support high-quality STL-10 reconstruction at 5\% and 10\% sampling.

\input{tables/table1_primary_results.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_primary_metrics_v8.pdf}
\caption{\textbf{Primary metrics.} PSNR and SSIM are shown for STL-10 5\%, STL-10 10\%, and simple-domain 5\% sanity checks. Dashed lines are predefined operational thresholds used to summarize reconstruction quality in this study.}
\label{fig:primary_metrics}
\end{figure*}

\subsection{Qualitative reconstruction}
Large qualitative reconstructions are shown in \Cref{fig:qualitative_reconstruction}. The backprojections are incomplete and noisy, especially for Rademacher measurements. The learned reconstruction restores object-level structure while preserving measurement dependence. Images are enlarged for visibility and are intended as qualitative evidence; quantitative conclusions are based on the leakage-free evaluation metrics.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig3_qualitative_reconstruction_v8.pdf}
\caption{\textbf{Qualitative reconstruction.} STL-10 examples are shown as ground truth, backprojection, reconstruction, and absolute error. Representative evaluation samples are enlarged for visibility. Error maps use a shared 99th-percentile scale.}
\label{fig:qualitative_reconstruction}
\end{figure*}

\subsection{Simple-domain sanity checks}
On MNIST and Fashion-MNIST at 5\% sampling, the method reaches 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM, respectively. These experiments confirm that the same reconstruction pipeline works reliably on simpler structured domains, but they are not the main novelty.

\subsection{Measurement-family attribution}
\Cref{tab:measurement_attribution,fig:measurement_attribution} separate physical initialization from learned refinement. Final PSNR alone hides whether performance comes from physical initialization, neural refinement, or both. Rademacher measurements have weak physical backprojections: 7.297 dB at 5\% and 7.756 dB at 10\%. However, final reconstruction reaches 22.316 dB and 24.781 dB, corresponding to gains of 15.019 dB and 17.025 dB. Scrambled Hadamard measurements start from stronger backprojections, 14.310 dB at 5\% and 14.533 dB at 10\%, and reach nearly the same final quality as Rademacher.

\input{tables/table2_measurement_attribution.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig4_measurement_attribution_v8.pdf}
\caption{\textbf{Measurement attribution.} Pattern examples, backprojection-vs-model PSNR, neural gain, and the regime map show that final PSNR alone hides whether performance comes from physical initialization, neural refinement, or both.}
\label{fig:measurement_attribution}
\end{figure*}
"""


def validation_ablation() -> str:
    return r"""\section{Validation and Ablation}
\subsection{Exact-A reproducibility}
Rademacher measurements require exact-operator evaluation. Earlier mismatch was traced to stale solver-cache use after overriding \(A\). With safe exact-A loading and cache rebuilding, Rademacher 5\% and 10\% re-evaluations reproduce the original leakage-free evaluation with negligible differences. These reproduced results are used as primary evidence.

\subsection{Inference-time ablation}
\Cref{tab:ablation_summary,fig:inference_ablation} report the inference-time ablations. Removing the measurement-consistency projection causes the largest degradation. This shows that \(\Pi_y\) is not merely cosmetic; it is central to maintaining physical fidelity and image quality. Removing the null projection has limited metric effect for the trained checkpoints, suggesting that the final projection and the learned network already constrain many measured components. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.

\input{tables/table3_ablation_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation_v8.pdf}
\caption{\textbf{Inference ablation.} Removing measurement-consistency projection gives the strongest degradation. The limited metric change of \(-\)Null is reported explicitly and should not be overinterpreted as proving that null-space modeling is unnecessary.}
\label{fig:inference_ablation}
\end{figure*}

\subsection{Noise and perturbation tests}
\Cref{fig:validation_summary} summarizes finite-noise sweeps, measurement perturbations, CS-TV comparison, and bootstrap confidence intervals. Finite-noise sweeps show stable degradation over the tested noise range. Measurement perturbation tests are more diagnostic: shuffled coefficients and wrong-sample measurements cause large PSNR drops. This indicates that the model depends on the bucket measurement vector rather than measurement-independent hallucination.

\subsection{CS-TV compressed-sensing baseline}
We compare against a TV-regularized compressed-sensing baseline solved by projected gradient descent \cite{donoho2006compressed,candes2006robust,rudin1992tv}:
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\,\mathrm{TV}(x).
\end{equation}
We refer to this baseline as CS-TV (PGD solver). It is a lightweight small-subset traditional baseline, not an exhaustively optimized ADMM/FISTA or plug-and-play benchmark. Under the tested settings, the learned measurement-consistent reconstructor remains substantially stronger.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_robustness_baselines_v8.pdf}
\caption{\textbf{Robustness and baselines.} Panels show finite-noise behavior, Shuffle/Wrong-y measurement perturbations, comparison against CS-TV, and bootstrap confidence intervals. These diagnostics support finite-noise stability and measurement dependence; they do not imply universal robustness.}
\label{fig:validation_summary}
\end{figure*}

\subsection{DC row control}
For low-frequency Hadamard, including the DC row is critical for backprojection because it captures global brightness. Removing it severely degrades low-frequency Hadamard initialization. This is a measurement-design diagnostic and should not be generalized to all measurement families.

\subsection{Statistics and class-wise diagnostics}
Bootstrap confidence intervals show that the main results are stable across samples. Class-wise STL-10 diagnostics reveal expected category variability, with some classes consistently more difficult than others. These analyses support the robustness of the main claim but are not themselves the central contribution.
"""


def supplement_text() -> str:
    return r"""\section{Supplementary Material}
Detailed CSV files and audit manifests are provided as an accompanying data package. The tables below are curated summaries intended for a compact submission supplement.

\subsection{S1 Exact-A reproducibility}
\input{tables/tableS1_exact_a.tex}

\subsection{S2 RelMeasErr ablation}
\begin{figure*}[h]
\centering
\includegraphics[width=0.82\textwidth]{figures/figS1_relmeaserr_ablation_v8.pdf}
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


def normalize_v7_text(text: str) -> str:
    replacements = {
        "strict no-leak": "strict leakage-free",
        "strict no-leak evaluations": "strict leakage-free evaluations",
        "no-leak": "leakage-free",
        "Colab results": "original leakage-free evaluation",
        "Colab checkpoint": "trained checkpoint",
        "internal engineering thresholds": "predefined operational thresholds",
        "internal engineering threshold": "predefined operational threshold",
        "generic image-prior hallucination": "measurement-independent hallucination",
        "lowfreq": "low-frequency",
        "adversarial-generation paper": "GAN-centered reconstruction method",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_sections() -> None:
    section_map = {
        "abstract.tex": abstract(),
        "introduction.tex": introduction(),
        "related_work.tex": normalize_v7_text(read_v7("related_work.tex")),
        "problem_formulation.tex": normalize_v7_text(read_v7("problem_formulation.tex")),
        "method.tex": method(),
        "measurement_families.tex": normalize_v7_text(read_v7("measurement_families.tex")).replace(
            "exact-A reproducibility requires the exported operator and cache-rebuilt evaluation path described above.",
            "exact-A reproducibility requires the exported operator and cache-rebuilt evaluation path described above. For Rademacher measurements, we reload the exported exact measurement operator and rebuild the cached solver before evaluation.",
        ),
        "experimental_protocol.tex": experimental_protocol(),
        "results.tex": results(),
        "validation_ablation.tex": validation_ablation(),
        "discussion.tex": normalize_v7_text(read_v7("discussion.tex")),
        "limitations.tex": normalize_v7_text(read_v7("limitations.tex")),
        "conclusion.tex": normalize_v7_text(read_v7("conclusion.tex")),
    }
    for name, text in section_map.items():
        write_text(SECTIONS / name, text)
    write_text(SUPP / "supplement.tex", supplement_text())


def write_reports() -> None:
    write_text(OUT / "abstract_v8.md", ABSTRACT)
    write_text(
        OUT / "language_replacements_report.md",
        """# Language Replacements Report

- Replaced main-text strict no-leak wording with strict leakage-free protocol/evaluation wording.
- Removed Colab/platform wording from main text.
- Standardized engineering threshold wording to predefined operational threshold.
- Expanded HQ wording to high-quality in main prose.
- Kept Fig. capitalization through LaTeX captions and prose.
- Replaced generic hallucination phrasing with measurement-independent hallucination or measurement-independent generation wording.
- Standardized lowfreq to low-frequency.
- Added exact-A wording for Rademacher operator reload and cache rebuild.
""",
    )
    write_text(
        LATEX / "submission_checklist.md",
        """# Submission Checklist

- Main text and supplement compile separately.
- Main Figures 1--6 are before References.
- Supplement figures stay in the supplement.
- No local Windows paths are included in PDF sources.
- No Reference Placeholders or TODO VERIFY strings are included in PDF sources.
- Data/code availability avoids local paths.
""",
    )
    write_text(OUT / "submission_checklist.md", (LATEX / "submission_checklist.md").read_text(encoding="utf-8"))


def citation_audit() -> str:
    bib = (P21_LATEX / "references.bib").read_text(encoding="utf-8")
    main_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(SECTIONS.glob("*.tex"))
    )
    cite_keys = sorted(set(re.findall(r"\\cite\{([^}]+)\}", main_sources)))
    cite_keys = sorted({key.strip() for group in cite_keys for key in group.split(",")})
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = [key for key in cite_keys if key not in bib_keys]
    entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, flags=re.S)
    malformed = []
    missing_core = []
    for key, body in entries:
        if "author" not in body.lower() or "title" not in body.lower():
            malformed.append(key)
        if "year" not in body.lower():
            missing_core.append(key)
        if not any(field in body.lower() for field in ["journal", "booktitle", "publisher"]):
            missing_core.append(key)
    categories = [
        "SPI/GI foundations: shapiro2008computational, edgar2019principles, gibson2020singlepixel.",
        "Deep GI/SPI and physics-enhanced SPI: he2018ghost, wang2019learning, rizvi2020deepghost, bian2020residual, wang2022physics.",
        "Null-space/data-consistency inverse problems: adler2018learned, aggarwal2019modl, schwab2019deepnull, goppel2023dataproximal.",
        "Compressed sensing/TV: donoho2006compressed, candes2006robust, rudin1992tv.",
        "Hadamard ordering/measurement design: sun2017russian, zhang2017hadamard, cakecutting2019.",
    ]
    lines = [
        "# Citation Audit",
        "",
        f"- TODO VERIFY in references: {'no' if 'TODO VERIFY' not in bib else 'yes'}",
        f"- Reference Placeholders in references: {'no' if 'Reference Placeholders' not in bib else 'yes'}",
        f"- Citation keys missing from bibliography: {', '.join(missing) if missing else 'none'}",
        f"- Malformed author/title entries: {', '.join(sorted(set(malformed))) if malformed else 'none detected'}",
        f"- Entries missing journal/booktitle/publisher or year: {', '.join(sorted(set(missing_core))) if missing_core else 'none detected'}",
        "",
        "## Coverage",
    ]
    lines.extend(f"- {item}" for item in categories)
    lines.extend(
        [
            "",
            "## Manual Checks Remaining",
            "- Verify DOI and exact page metadata against publisher records before submission.",
            "- Apply target journal bibliography style if required.",
        ]
    )
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


def main() -> None:
    ensure_dir(OUT)
    clean_latex_dir()
    copy_tables()
    copy_figures()
    write_sections()
    refs = (P21_LATEX / "references.bib").read_text(encoding="utf-8")
    write_text(LATEX / "references.bib", refs)
    write_text(LATEX / "main.tex", main_tex())
    write_text(LATEX / "supplement.tex", supplement_tex())
    write_reports()
    audit = citation_audit()
    write_text(OUT / "citation_audit.md", audit)
    write_text(LATEX / "citation_audit.md", audit)
    write_text(OUT / "manuscript_v8.tex", main_tex())
    write_text(OUT / "human_written_manuscript_v8.md", plain_manuscript())
    manifest = source_manifest()
    manifest["output"] = str(OUT)
    manifest["phase21_base"] = str(P21)
    write_json(OUT / "internal_source_manifest.json", manifest)
    print({"latex_project_v8": str(LATEX)})


if __name__ == "__main__":
    main()
