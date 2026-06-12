from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .phase17_common import (
    PHASE16,
    PHASE16_TABLES,
    PHASE17,
    REGISTRY,
    TITLE,
    ensure_dir,
    fnum,
    main_result_rows,
    markdown_table,
    read_csv,
    tex_escape,
    write_csv,
    write_json,
    write_text,
)


OUT = Path("E:/ns_mc_gan_gi/outputs_phase18_manuscript_v2")
LATEX = OUT / "latex_project"
SECTIONS = LATEX / "sections"
SUPP_DIR = LATEX / "supplement"
FIG_DIR = OUT / "figures"
LATEX_FIG_DIR = LATEX / "figures"


def method_short(method_id: str) -> str:
    return (
        method_id.replace("_hq_noise001_colab", "")
        .replace("_full_noise001_colab", "")
        .replace("_full_colab", "")
        .replace("scrambled_hadamard", "scrambled")
        .replace("rademacher", "rademacher")
    )


def rows_by_id() -> dict[str, dict[str, str]]:
    return {row.get("method_id", ""): row for row in read_csv(REGISTRY)}


def main_metric_sentence() -> str:
    rows = rows_by_id()
    return (
        "On STL-10 natural images, the framework reaches "
        f"{fnum(rows['rademacher5_hq_noise001_colab']['psnr'])} dB / {fnum(rows['rademacher5_hq_noise001_colab']['ssim'])} SSIM "
        "with Rademacher measurements at 5% sampling and "
        f"{fnum(rows['scrambled_hadamard5_hq_noise001_colab']['psnr'])} dB / {fnum(rows['scrambled_hadamard5_hq_noise001_colab']['ssim'])} SSIM "
        "with scrambled Hadamard measurements at 5% sampling. "
        "At 10% sampling, the corresponding results are "
        f"{fnum(rows['rademacher10_full_noise001_colab']['psnr'])} dB / {fnum(rows['rademacher10_full_noise001_colab']['ssim'])} SSIM "
        "and "
        f"{fnum(rows['scrambled_hadamard10_full_noise001_colab']['psnr'])} dB / {fnum(rows['scrambled_hadamard10_full_noise001_colab']['ssim'])} SSIM. "
        "On MNIST and Fashion-MNIST at 5% sampling, the same pipeline obtains "
        f"{fnum(rows['mnist_hadamard5_full_colab']['psnr'])} dB / {fnum(rows['mnist_hadamard5_full_colab']['ssim'])} SSIM "
        "and "
        f"{fnum(rows['fashion_hadamard5_full_colab']['psnr'])} dB / {fnum(rows['fashion_hadamard5_full_colab']['ssim'])} SSIM, respectively."
    )


def main_table_tex() -> str:
    rows = []
    for row in main_result_rows():
        rows.append(
            {
                "Dataset": row["dataset"],
                "Sampling": row["sampling"],
                "Measurements": row["family"].replace("_", " "),
                "PSNR": row["psnr"],
                "SSIM": row["ssim"],
                "BP PSNR": row["bp_psnr"],
                "$\\Delta$PSNR": row["delta_psnr"],
            }
        )
    fields = ["Dataset", "Sampling", "Measurements", "PSNR", "SSIM", "BP PSNR", "$\\Delta$PSNR"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Primary strict no-leak reconstruction results.} STL-10 high-quality claims are made only for Rademacher and scrambled Hadamard measurements. Low-frequency Hadamard rows are used for simple-domain sanity checks and diagnostic controls.}",
        r"\label{tab:main_results}",
        r"\small",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        " & ".join(fields) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(row[field]) for field in fields) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def abstract_text() -> str:
    return (
        "Ghost imaging and single-pixel imaging recover spatial information from structured illumination patterns and bucket measurements. "
        "At low sampling rates, the inverse problem is highly underdetermined, and purely data-driven reconstruction networks can generate visually plausible images that are not physically consistent with the measured signal. "
        "We formulate low-sampling ghost imaging as a measurement-consistent null-space completion problem. "
        "Starting from a physically meaningful data solution, a neural reconstructor estimates a residual component that is constrained by the measurement operator, followed by a measurement-consistency projection that enforces agreement with the bucket measurements. "
        "We evaluate Rademacher, scrambled Hadamard, and low-frequency Hadamard measurement families under strict no-leak protocols and exact-operator re-evaluation for random measurements. "
        + main_metric_sentence()
        + " Ablation, finite-noise, compressed-sensing baseline, and perturbation analyses show that the reconstructor improves substantially over physical backprojection, remains measurement-dependent, and benefits from measurement-consistency constraints. "
        "These results demonstrate a physics-consistent route toward high-quality low-sampling ghost imaging reconstruction without claiming strict state-of-the-art performance."
    )


def intro_md() -> str:
    return """Ghost imaging and single-pixel imaging reconstruct an image from known illumination patterns and scalar bucket measurements. Instead of recording a spatially resolved image directly, the system measures projections of the scene onto a sequence of patterns. This acquisition model is attractive for wavelengths, detectors, or optical configurations where dense sensor arrays are expensive or unavailable.

The central difficulty is low sampling. When the number of bucket measurements is much smaller than the number of image pixels, the inverse problem is severely underdetermined. Many candidate images can explain the same measurement vector, and direct physical backprojection often leaves substantial missing structure.

Deep reconstruction networks can improve visual quality in this regime, but an unconstrained network introduces a different risk: it may hallucinate image details that improve perceptual metrics while violating the bucket measurements. For computational imaging, this is not a cosmetic issue; the reconstructed image should remain tied to the actual optical measurement.

This work treats low-sampling ghost imaging as a measurement-consistent null-space completion problem. We first compute a data solution from the forward operator and the bucket measurements. The neural reconstructor then estimates a residual component, but this residual is inserted through the measurement null space and followed by a final measurement-consistency projection.

Measurement design also matters. Rademacher measurements, scrambled Hadamard measurements, and low-frequency Hadamard measurements produce different physical initializations and different neural refinement behavior. Separating backprojection quality from final reconstruction quality is therefore essential for interpreting performance.

Our contributions are fourfold. First, we formulate a measurement-consistent null-space neural reconstruction pipeline for low-sampling ghost imaging. Second, we evaluate random and orthogonal measurement families under strict no-leak protocols, including exact-operator re-evaluation for Rademacher measurements. Third, we demonstrate high-quality STL-10 reconstruction at both 5% and 10% sampling under Rademacher and scrambled Hadamard measurements. Fourth, we provide a reviewer-facing validation package covering attribution, inference ablation, finite noise robustness, TV-regularized compressed sensing, DC-row controls, confidence intervals, class-wise diagnostics, and measurement perturbation tests."""


def method_md() -> str:
    return r"""The forward model is

$$
y = Ax + \varepsilon,
$$

where \(x\in\mathbb{R}^n\) is the unknown image, \(A\in\mathbb{R}^{m\times n}\) is the measurement matrix, \(y\in\mathbb{R}^{m}\) is the bucket measurement vector, and \(\varepsilon\) is noise. In the low-sampling regime \(m\ll n\), direct inversion is ill posed.

We use a regularized data solution

$$
x_{\mathrm{data}} = A^T(AA^T+\lambda I)^{-1}y .
$$

The corresponding null-space projection is

$$
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av .
$$

A neural reconstructor predicts a residual from the physical initialization,

$$
r_\theta = G_\theta(x_{\mathrm{data}},z),
$$

and the residual is inserted through the measurement null space:

$$
\tilde{x}=x_{\mathrm{data}}+P_N(G_\theta(x_{\mathrm{data}},z)).
$$

Finally, we apply a measurement-consistency projection

$$
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y),
$$

yielding

$$
\hat{x}=\Pi_y(\tilde{x}).
$$

For Hadamard measurements, we use an orthonormal matrix

$$
H_{\mathrm{norm}}=\frac{H}{\sqrt{n}},
$$

select a subset of rows,

$$
A=H_{\mathrm{norm}}[\mathrm{selected\ rows},:],
$$

assign the measured coefficients by

$$
c[\mathrm{selected\ rows}]=y,
$$

and compute the zero-filled data solution

$$
x_{\mathrm{data}}=H_{\mathrm{norm}}^Tc.
$$

For Rademacher measurements, \(A\) is a signed random measurement operator. All Rademacher results are evaluated with the exported exact operator and a solver cache rebuilt after loading that operator."""


def related_work_md() -> str:
    return """Deep learning has been widely studied for ghost imaging and single-pixel reconstruction, including convolutional, generative, and conditional reconstruction models. Separately, data consistency and null-space correction are established ideas in inverse problems and computational imaging. The contribution here should therefore not be framed as the first deep-learning ghost imaging method or the first null-space neural inverse method.

The more precise positioning is that this work combines explicit measurement consistency, null-space residual reconstruction, measurement-family attribution, exact-operator reproducibility for random sensing, and a compact validation package in a low-sampling ghost imaging setting. Placeholder citations in the BibTeX file are marked TODO-VERIFY and must be replaced before submission."""


def experiments_md() -> str:
    return """We evaluate STL-10 natural images, MNIST, and Fashion-MNIST. The primary STL-10 claims use Rademacher and scrambled Hadamard measurements at 5% and 10% sampling. MNIST and Fashion-MNIST provide simple-domain sanity checks at 5% sampling. Low-frequency Hadamard is used for simple-domain and diagnostic analyses, but low-frequency Hadamard 5% is not reported as a high-quality STL-10 result.

All primary numbers are strict no-leak evaluations of final imported checkpoints. For Rademacher measurements, the exact exported measurement operator is reloaded during evaluation and the solver cache is rebuilt before applying the inverse operators. This avoids the stale-cache mismatch observed in earlier local re-evaluations."""


def results_md() -> str:
    rows = rows_by_id()
    return (
        main_metric_sentence()
        + "\n\n"
        "The main STL-10 observation is that final reconstruction quality is similar for Rademacher and scrambled Hadamard at the same sampling ratio, even though their physical backprojections differ substantially. "
        f"At 5% sampling, Rademacher backprojection is only {fnum(rows['rademacher5_hq_noise001_colab']['backproj_psnr'])} dB, while scrambled Hadamard backprojection is {fnum(rows['scrambled_hadamard5_hq_noise001_colab']['backproj_psnr'])} dB. "
        f"At 10% sampling, the corresponding backprojection values are {fnum(rows['rademacher10_full_noise001_colab']['backproj_psnr'])} dB and {fnum(rows['scrambled_hadamard10_full_noise001_colab']['backproj_psnr'])} dB. "
        "This indicates that Rademacher measurements are difficult for direct inversion but recoverable by the learned measurement-consistent inverse, while scrambled Hadamard provides a stronger physical initialization."
    )


def ablation_md() -> str:
    return """Inference-time ablations test whether the reconstruction quality is tied to the physical measurement operator. Removing the measurement-consistency projection strongly degrades reconstruction quality and measurement fidelity, especially for scrambled Hadamard measurements. Stage-1-only reconstruction is lower than the full two-stage reconstructor, indicating that the residual refiner contributes to final quality. EMA weights provide a small stabilization effect relative to raw weights.

We compare against linear physical baselines and a TV-regularized compressed-sensing baseline solved by projected gradient descent, denoted CS-TV (PGD solver). This baseline solves an optimization problem of the form

$$
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda \mathrm{TV}(x),
$$

with projected gradient steps. Thus, the comparison is a compressed-sensing-style baseline, not an unrelated heuristic. We describe it as lightweight and small-subset because it is not an exhaustively optimized ADMM/FISTA benchmark.

Finite noise sweeps show stable degradation over the tested noise range, and measurement perturbation controls show that shuffled or wrong-sample measurements cause large performance drops. These perturbation results support the claim that the model depends on the bucket signal rather than merely generating generic natural images. DC-row controls further show that the DC row is crucial for low-frequency Hadamard backprojection."""


def discussion_md() -> str:
    return """The results suggest that measurement family, physical initialization, and neural refinement should be interpreted jointly. Scrambled Hadamard measurements provide stronger direct physical reconstructions, while Rademacher measurements provide weak backprojections but comparable final results after learned refinement. This difference is hidden if only the final PSNR is reported.

The framework is also intentionally auditable. Exact-operator evaluation, relative measurement error, no-DC ablation, perturbation controls, and confidence intervals all test whether the reconstruction remains connected to the measured signal. This is important for optical inverse problems, where visual plausibility alone is not sufficient."""


def limitations_md() -> str:
    return """We do not claim strict state-of-the-art performance. The reported comparisons are intended to validate the reconstruction mechanism and measurement-family behavior under the tested protocols.

The CS-TV (PGD solver) baseline is a lightweight TV-regularized compressed-sensing baseline evaluated on a small subset, not an exhaustively optimized compressed-sensing solver. Stronger ADMM, FISTA, or plug-and-play baselines could be added in future work if the target venue requires a broader benchmark.

Robustness claims are limited to the tested finite noise range and measurement perturbation controls. The class-wise analysis is diagnostic and should not be over-interpreted. Unless hardware data are added, the results should be described as simulation-based optical inverse-problem experiments. We do not claim binary learned illumination as the main contribution, and we do not present GAN training as the final main mechanism. Low-frequency Hadamard at 5% sampling should not be described as a high-quality STL-10 result."""


def conclusion_md() -> str:
    return """This work presents a measurement-consistent null-space neural reconstruction framework for low-sampling ghost imaging. By combining a physical data solution, null-space residual completion, and a final measurement-consistency projection, the method reaches high-quality STL-10 reconstruction at 5% and 10% sampling under Rademacher and scrambled Hadamard measurements. The supplementary analyses show that the gains are not simply inherited from backprojection, that data consistency matters, and that the reconstructor remains dependent on the measured bucket signal. The evidence supports moving from broad experimentation to careful manuscript polishing, citation verification, and final figure preparation."""


def md_document() -> str:
    return f"""# {TITLE}

## Abstract

{abstract_text()}

## 1. Introduction

{intro_md()}

## 2. Related Work

{related_work_md()}

## 3. Method

{method_md()}

## 4. Experiments

{experiments_md()}

## 5. Results

{results_md()}

{markdown_table(main_result_rows(), ["method", "dataset", "sampling", "family", "psnr", "ssim", "bp_psnr", "delta_psnr"])}

## 6. Ablation and Validation

{ablation_md()}

## 7. Discussion

{discussion_md()}

## 8. Limitations

{limitations_md()}

## 9. Conclusion

{conclusion_md()}
"""


def section_tex(title: str, body_md: str, label: str | None = None) -> str:
    label_line = f"\\label{{{label}}}\n" if label else ""
    return f"\\section{{{tex_escape(title)}}}\n{label_line}{md_to_tex(body_md)}\n"


def md_to_tex(text: str) -> str:
    # The manuscript body is controlled prose with display math already in LaTeX syntax.
    # Preserve LaTeX commands in formulas and inline math; only escape prose percent signs.
    out = []
    in_math = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "$$":
            out.append(r"\[" if not in_math else r"\]")
            in_math = not in_math
        elif in_math:
            out.append(line)
        elif not stripped:
            out.append("")
        else:
            out.append(line.replace("%", r"\%"))
    return "\n".join(out)


def write_latex_project() -> None:
    ensure_dir(SECTIONS)
    ensure_dir(SUPP_DIR)
    ensure_dir(LATEX_FIG_DIR)
    section_map = {
        "abstract.tex": md_to_tex(abstract_text()),
        "introduction.tex": section_tex("Introduction", intro_md(), "sec:introduction"),
        "related_work.tex": section_tex("Related Work", related_work_md(), "sec:related"),
        "method.tex": section_tex("Method", method_md(), "sec:method"),
        "experiments.tex": section_tex("Experimental Protocol", experiments_md(), "sec:experiments"),
        "results.tex": f"\\section{{Results}}\n\\label{{sec:results}}\n{md_to_tex(results_md())}\n\n{main_table_tex()}\n",
        "ablation.tex": section_tex("Ablation and Validation", ablation_md(), "sec:ablation"),
        "discussion.tex": section_tex("Discussion", discussion_md(), "sec:discussion"),
        "limitations.tex": section_tex("Limitations", limitations_md(), "sec:limitations"),
        "conclusion.tex": section_tex("Conclusion", conclusion_md(), "sec:conclusion"),
    }
    for name, content in section_map.items():
        write_text(SECTIONS / name, content)
    write_text(SUPP_DIR / "supplement.tex", supplement_tex_body())
    write_text(LATEX / "references.bib", references_bib())
    write_text(LATEX / "main.tex", main_tex())


def main_tex() -> str:
    return r"""\documentclass[twocolumn,10pt]{article}
\usepackage[margin=0.72in]{geometry}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{cleveref}
\usepackage{stfloats}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\title{High-Quality Low-Sampling Ghost Imaging via Measurement-Consistent Null-Space Neural Reconstruction}
\author{Author names to be added}
\date{}
\begin{document}
\maketitle
\begin{abstract}
\input{sections/abstract}
\end{abstract}
\input{sections/introduction}
\input{sections/related_work}
\input{sections/method}
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/fig1_mechanism.pdf}
    \caption{\textbf{Measurement-consistent null-space neural reconstruction for low-sampling ghost imaging.} A structured or random illumination sequence produces bucket measurements \(y=Ax+\varepsilon\). The low-sampling inverse problem is underdetermined; the neural reconstructor predicts a residual that is inserted through \(P_N\), followed by a measurement-consistency projection \(\Pi_y\).}
    \label{fig:mechanism}
\end{figure*}
\input{sections/experiments}
\input{sections/results}
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/fig2_measurement_attribution.pdf}
    \caption{\textbf{Measurement-family attribution.} Rademacher measurements produce weak linear backprojections but large neural refinement gains. Scrambled Hadamard measurements provide stronger physical initialization and similar final reconstruction quality. Low-frequency Hadamard measurements are diagnostic and should not be interpreted as the primary high-quality STL-10 5\% setting.}
    \label{fig:measurement_attribution}
\end{figure*}
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/fig3_main_results.pdf}
    \caption{\textbf{Main reconstruction results.} The framework reaches high-quality reconstruction on STL-10 at 5\% and 10\% sampling under Rademacher and scrambled Hadamard measurements. It also achieves high-quality reconstruction on MNIST and Fashion-MNIST at 5\% sampling. Dashed lines indicate the internal high-quality thresholds used in this study.}
    \label{fig:main_results}
\end{figure*}
\input{sections/ablation}
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/fig4_inference_ablation.pdf}
    \caption{\textbf{Inference-time ablation.} Removing the measurement-consistency projection substantially degrades reconstruction quality and physical fidelity. The two-stage refiner improves over the stage-1 output, and EMA weights provide a small stabilization effect.}
    \label{fig:ablation}
\end{figure*}
\begin{figure*}[t]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/fig5_robustness_baselines.pdf}
    \caption{\textbf{Robustness and baseline analyses.} Noise sweeps show stable degradation over the tested finite noise range. Measurement perturbation experiments demonstrate dependence on bucket measurements. CS-TV (PGD solver), a lightweight TV-regularized compressed-sensing baseline, remains below the learned reconstruction under the tested settings.}
    \label{fig:robustness}
\end{figure*}
\input{sections/discussion}
\input{sections/limitations}
\input{sections/conclusion}
\bibliographystyle{plain}
\bibliography{references}
\end{document}
"""


def standalone_tex() -> str:
    return main_tex().replace(r"\input{sections/abstract}", md_to_tex(abstract_text()))


def supplement_md() -> str:
    exact = read_csv(PHASE16_TABLES["exact_a_reeval"])
    attr = read_csv(PHASE16_TABLES["attribution"])
    ablation = read_csv(PHASE16_TABLES["ablation"])
    noise = read_csv(PHASE16_TABLES["noise"])
    baseline = read_csv(PHASE16_TABLES["traditional_baselines"])
    dc = read_csv(PHASE16_TABLES["dc_row"])
    stats = read_csv(PHASE16_TABLES["statistics"])
    perturb = read_csv(PHASE16_TABLES["perturbation"])
    runtime = read_csv(PHASE16_TABLES["runtime"])
    baseline_display = []
    for row in baseline:
        row = dict(row)
        if row.get("baseline") == "tv_pgd":
            row["baseline"] = "CS-TV (PGD solver)"
        baseline_display.append(row)
    runtime_display = []
    for row in runtime:
        row = dict(row)
        if row.get("path") == "tv_pgd_best_observed":
            row["path"] = "CS-TV best observed"
        runtime_display.append(row)
    return f"""# Supplementary Material v2

## S1. Forward model details

The main paper uses \(y=Ax+\\varepsilon\), a ridge-stabilized data solution, a null-space residual, and a final measurement-consistency projection. This supplement records the evidence tables used to support the manuscript.

## S2. Exact-operator Rademacher reproducibility

{markdown_table(exact, ["method_id", "original_psnr", "reeval_psnr", "abs_diff_psnr", "status"])}

## S3. Measurement family attribution

{markdown_table(attr, ["method_id", "backproj_psnr", "model_psnr", "delta_psnr", "classification"])}

## S4. Inference-time ablation

{markdown_table(ablation, ["method_id", "ablation_mode", "psnr", "ssim", "delta_vs_full_psnr", "rel_meas_err"], limit=32)}

## S5. Noise robustness

{markdown_table(noise, ["method_id", "noise_std", "psnr", "ssim", "rel_meas_err"])}

## S6. Traditional baselines

The TV-regularized compressed-sensing baseline is reported as **CS-TV (PGD solver)**. It is lightweight and small-subset, not exhaustively optimized.

{markdown_table(baseline_display, ["method_id", "baseline", "num_samples", "lambda_tv", "psnr", "ssim", "notes"], limit=30)}

## S7. DC-row control

{markdown_table(dc, ["method_id", "sampling_ratio", "hadamard_include_dc", "hadamard_skip_dc", "backproj_psnr", "backproj_ssim"])}

## S8. Bootstrap confidence intervals

{markdown_table(stats, ["method_id", "mean_psnr", "ci95_psnr_low", "ci95_psnr_high", "mean_ssim", "ci95_ssim_low", "ci95_ssim_high"])}

## S9. Measurement perturbation

{markdown_table(perturb, ["method_id", "perturbation_mode", "psnr", "psnr_drop_from_normal", "rel_meas_err"], limit=24)}

## S10. Runtime and complexity

{markdown_table(runtime_display, ["method_id", "path", "runtime_sec_per_image", "model_params_m", "device", "notes"], limit=20)}

## S11. Deprecated or excluded claims

We exclude pre-fix Rademacher mismatch results, leaked/exploratory runs, strict SOTA claims, binary learned illumination claims, GAN-as-main-mechanism claims, and low-frequency Hadamard 5% as a high-quality STL-10 claim.
"""


def supplement_tex_body() -> str:
    text = supplement_md()
    lines = []
    for line in text.splitlines():
        if line.startswith("# "):
            lines.append(r"\section*{" + tex_escape(line[2:]) + "}")
        elif line.startswith("## "):
            lines.append(r"\section{" + tex_escape(line[3:]) + "}")
        elif line.startswith("|"):
            continue
        elif line.strip():
            lines.append(tex_escape(line) + "\n")
        else:
            lines.append("")
    return "\n".join(lines)


def supplement_tex() -> str:
    return r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref}
\title{Supplementary Material}
\date{}
\begin{document}
\maketitle
\input{supplement/supplement}
\end{document}
"""


def references_bib() -> str:
    return r"""@article{TODO_VERIFY_deep_gi_review,
  title = {TODO VERIFY: representative review or survey on deep learning for ghost imaging and single-pixel imaging},
  author = {TODO VERIFY},
  journal = {TODO VERIFY},
  year = {TODO VERIFY}
}

@article{TODO_VERIFY_gan_gi,
  title = {TODO VERIFY: representative GAN or conditional GAN method for ghost imaging},
  author = {TODO VERIFY},
  journal = {TODO VERIFY},
  year = {TODO VERIFY}
}

@article{TODO_VERIFY_null_space_inverse,
  title = {TODO VERIFY: null-space or data-consistent neural inverse problem method},
  author = {TODO VERIFY},
  journal = {TODO VERIFY},
  year = {TODO VERIFY}
}

@article{TODO_VERIFY_spi_foundation,
  title = {TODO VERIFY: foundational single-pixel imaging or ghost imaging reference},
  author = {TODO VERIFY},
  journal = {TODO VERIFY},
  year = {TODO VERIFY}
}

@article{TODO_VERIFY_tv_cs,
  title = {TODO VERIFY: TV-regularized compressed sensing reconstruction reference},
  author = {TODO VERIFY},
  journal = {TODO VERIFY},
  year = {TODO VERIFY}
}
"""


def figures_to_make() -> str:
    return """# Figures to make

## Figure 1: Mechanism schematic

Output: `figures/fig1_mechanism.pdf`

Panels: (a) GI/SPI acquisition \(y=Ax+\epsilon\); (b) underdetermined inverse problem and Null(A); (c) neural residual \(G_\theta\); (d) null-space and measurement-consistency projections; (e) reconstruction output.

## Figure 2: Measurement attribution

Output: `figures/fig2_measurement_attribution.pdf`

Panels: Rademacher, scrambled Hadamard, and low-frequency Hadamard pattern examples; backprojection vs model PSNR; delta PSNR.

## Figure 3: Main results

Output: `figures/fig3_main_results.pdf`

Panels: STL-10 5% PSNR/SSIM; STL-10 10% PSNR/SSIM; MNIST/Fashion 5%; thresholds.

## Figure 4: Inference ablation

Output: `figures/fig4_inference_ablation.pdf`

Panels: full/no-DC/no-null/stage1/raw/EMA where available. The no-DC drop is the most important visual message.

## Figure 5: Robustness and baselines

Output: `figures/fig5_robustness_baselines.pdf`

Panels: finite noise sweep, measurement perturbation, CS-TV (PGD solver) comparison, confidence intervals.

## Supplement figures

DC row control, class-wise STL-10 diagnostics, runtime, histograms, and optional best/median/worst examples.
"""


def tables_to_make() -> str:
    return """# Tables to make

## Main table 1

Primary strict no-leak results: dataset, sampling, measurement family, PSNR, SSIM, backprojection PSNR, and delta PSNR.

## Main table 2

Attribution: backprojection PSNR, model PSNR, delta PSNR, measurement family.

## Main table 3

Ablation summary: full model, no-DC, stage1-only, raw weights, EMA weights.

## Supplement tables

Exact-A reproducibility, full attribution table, full ablation table, finite noise sweep, CS-TV (PGD solver), DC row control, bootstrap confidence intervals, class-wise diagnostics, measurement perturbations, runtime.

## Baseline naming

Use `CS-TV (PGD solver)` in figures and tables. Expand it at first mention as: TV-regularized compressed-sensing baseline solved by projected gradient descent.
"""


def writing_notes() -> str:
    included = [
        "STL-10 5% HQ for Rademacher and scrambled Hadamard.",
        "STL-10 10% HQ for Rademacher and scrambled Hadamard.",
        "MNIST/Fashion-MNIST 5% HQ sanity results.",
        "Rademacher exact-A re-evaluation reproduced after safe cache rebuild.",
        "Model refinement strongly improves weak backprojections.",
        "Scrambled Hadamard provides stronger physical initialization than Rademacher but similar final quality.",
        "Measurement perturbation supports dependence on bucket measurements.",
        "CS-TV (PGD solver) is included as a compressed-sensing-style baseline.",
    ]
    excluded = [
        "Strict SOTA.",
        "Universal/adversarial robustness.",
        "Low-frequency Hadamard 5% as high-quality STL-10.",
        "Binary learned illumination as the contribution.",
        "GAN as the final main mechanism.",
        "CS-TV (PGD solver) as an exhaustively optimized solver.",
        "First deep-learning ghost imaging or first null-space inverse method.",
    ]
    return "# Paper writing notes\n\n## Included claims\n\n" + "\n".join(f"- {x}" for x in included) + "\n\n## Excluded claims\n\n" + "\n".join(f"- {x}" for x in excluded) + "\n\n## TV/CS wording\n\nUse: `TV-regularized compressed-sensing baseline solved by projected gradient descent`, abbreviated `CS-TV (PGD solver)`. This answers the compressed-sensing-baseline concern while keeping the limitation honest.\n"


def codex_make_figures_code() -> str:
    return r'''from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path("E:/ns_mc_gan_gi/outputs_phase18_manuscript_v2")
FIG = ROOT / "figures"
LATEX_FIG = ROOT / "latex_project" / "figures"
PHASE15 = Path("E:/ns_mc_gan_gi/outputs_phase15")
PHASE16 = Path("E:/ns_mc_gan_gi/outputs_phase16/supplementary_experiments")
REGISTRY = PHASE15 / "noleak_registry.csv"
TABLES = {
    "attribution": PHASE16 / "attribution/attribution_final.csv",
    "ablation": PHASE16 / "inference_ablation/real_inference_ablation_results.csv",
    "noise": PHASE16 / "noise_sweep/noise_sweep_results.csv",
    "baseline": PHASE16 / "traditional_baselines/tv_pgd_baseline_results.csv",
    "dc": PHASE16 / "dc_row_control/dc_row_final.csv",
    "statistics": PHASE16 / "statistics/statistics_ci.csv",
    "classwise": PHASE16 / "classwise/classwise_stl10_metrics.csv",
    "perturbation": PHASE16 / "measurement_perturbation/measurement_perturbation.csv",
    "runtime": PHASE16 / "runtime_complexity/runtime_complexity.csv",
}


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def short(mid):
    return (
        mid.replace("_hq_noise001_colab", "")
        .replace("_full_noise001_colab", "")
        .replace("_full_colab", "")
        .replace("scrambled_hadamard", "Scr.")
        .replace("rademacher", "Rad.")
        .replace("mnist_hadamard5", "MNIST")
        .replace("fashion_hadamard5", "Fashion")
    )


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, bbox_inches="tight")
    shutil.copy2(path, LATEX_FIG / name)
    plt.close(fig)


def panel_label(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def fig1():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    boxes = [
        ("Patterns\n$A_i$", 0.03),
        ("Object\n$x$", 0.21),
        ("Bucket\n$y=Ax+\\epsilon$", 0.39),
        ("Data solution\n$x_{data}$", 0.57),
        ("Null-space neural\nresidual + $\\Pi_y$", 0.75),
        ("Reconstruction\n$\\hat{x}$", 0.91),
    ]
    for text, x in boxes:
        w = 0.13 if x < 0.9 else 0.11
        box = FancyBboxPatch((x, 0.38), w, 0.28, boxstyle="round,pad=0.03", fc="#f3f7fb", ec="#2d5f83", lw=1.2)
        ax.add_patch(box)
        ax.text(x + w / 2, 0.52, text, ha="center", va="center", fontsize=10)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + (0.13 if boxes[i][1] < 0.9 else 0.11)
        x1 = boxes[i + 1][1]
        ax.add_patch(FancyArrowPatch((x0 + 0.01, 0.52), (x1 - 0.01, 0.52), arrowstyle="->", mutation_scale=12, lw=1.2, color="#333"))
    ax.text(0.5, 0.12, "Measured subspace is preserved; missing information is completed through the measurement null space.", ha="center", fontsize=11)
    save(fig, "fig1_mechanism.pdf")


def pattern(kind, n=32):
    rng = np.random.default_rng(4)
    if kind == "rademacher":
        return rng.choice([-1, 1], size=(n, n))
    x = np.arange(n)
    y = np.arange(n)[:, None]
    if kind == "lowfreq":
        return np.cos(2 * np.pi * x / n) + np.cos(2 * np.pi * y / n)
    # deterministic pseudo-Hadamard-like checker pattern with row/column scramble
    mat = ((x[None, :] * 7 + y * 11) % 17 < 8).astype(float) * 2 - 1
    return mat[rng.permutation(n)][:, rng.permutation(n)]


def fig2():
    rows = read_csv(TABLES["attribution"])
    primary = [r for r in rows if r["method_id"] in {
        "rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab",
        "rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab",
        "mnist_hadamard5_full_colab", "fashion_hadamard5_full_colab"}]
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    for idx, kind in enumerate(["rademacher", "scrambled", "lowfreq"]):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(pattern(kind), cmap="gray", interpolation="nearest")
        ax.set_title(["Rademacher", "Scrambled Hadamard", "Low-frequency Hadamard"][idx])
        ax.set_xticks([]); ax.set_yticks([])
        panel_label(ax, chr(ord("a") + idx))
    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(len(primary))
    ax.bar(x - 0.18, [f(r, "backproj_psnr") for r in primary], width=0.36, label="Backprojection", color="#9bb7c8")
    ax.bar(x + 0.18, [f(r, "model_psnr") for r in primary], width=0.36, label="Model", color="#2f6f95")
    ax.set_xticks(x); ax.set_xticklabels([short(r["method_id"]) for r in primary], rotation=25, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    panel_label(ax, "d")
    ax = fig.add_subplot(gs[1, 2])
    ax.barh([short(r["method_id"]) for r in primary], [f(r, "delta_psnr") for r in primary], color="#4c9a62")
    ax.set_xlabel("$\\Delta$PSNR")
    ax.grid(axis="x", alpha=0.25)
    panel_label(ax, "e")
    save(fig, "fig2_measurement_attribution.pdf")


def fig3():
    reg = read_csv(REGISTRY)
    stl5 = [r for r in reg if r["dataset"] == "STL-10" and abs(f(r, "sampling_ratio") - 0.05) < 1e-6]
    stl10 = [r for r in reg if r["dataset"] == "STL-10" and abs(f(r, "sampling_ratio") - 0.10) < 1e-6]
    simple = [r for r in reg if r["dataset"] != "STL-10"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, rows, title, thr in [(axes[0], stl5, "STL-10 5%", 20.0), (axes[1], stl10, "STL-10 10%", 22.0), (axes[2], simple, "MNIST/Fashion 5%", 25.0)]:
        x = np.arange(len(rows))
        ax.bar(x, [f(r, "psnr") for r in rows], color="#3b7ba5")
        ax.axhline(thr, ls="--", color="#a33", lw=1, label="HQ threshold")
        ax.set_xticks(x); ax.set_xticklabels([short(r["method_id"]) for r in rows], rotation=25, ha="right")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8)
    for i, ax in enumerate(axes):
        panel_label(ax, chr(ord("a") + i))
    save(fig, "fig3_main_results.pdf")


def fig4():
    rows = read_csv(TABLES["ablation"])
    keep = ["full_model", "no_dc_project", "no_null_project", "stage1_only", "raw_weights", "ema_weights"]
    methods = ["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab", "rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for ax, method in zip(axes.ravel(), methods):
        sub = [r for r in rows if r["method_id"] == method and r["ablation_mode"] in keep]
        x = np.arange(len(sub))
        ax.bar(x, [f(r, "psnr") for r in sub], color=["#2f6f95" if r["ablation_mode"] == "full_model" else "#9bb7c8" for r in sub])
        ax.set_xticks(x); ax.set_xticklabels([r["ablation_mode"].replace("_", "\n") for r in sub], fontsize=7)
        ax.set_title(short(method))
        ax.set_ylabel("PSNR (dB)")
        ax.grid(axis="y", alpha=0.25)
    for i, ax in enumerate(axes.ravel()):
        panel_label(ax, chr(ord("a") + i))
    save(fig, "fig4_inference_ablation.pdf")


def fig5():
    noise = read_csv(TABLES["noise"])
    pert = read_csv(TABLES["perturbation"])
    base = read_csv(TABLES["baseline"])
    stats = read_csv(TABLES["statistics"])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    ax = axes[0, 0]
    for method in sorted({r["method_id"] for r in noise}):
        sub = sorted([r for r in noise if r["method_id"] == method], key=lambda r: f(r, "noise_std"))
        ax.plot([f(r, "noise_std") for r in sub], [f(r, "psnr") for r in sub], marker="o", label=short(method))
    ax.set_xlabel("Noise std"); ax.set_ylabel("PSNR (dB)"); ax.set_title("Finite noise sweep"); ax.grid(alpha=0.25); ax.legend(fontsize=6, frameon=False)
    ax = axes[0, 1]
    wrong = [r for r in pert if r["perturbation_mode"] in {"shuffle_coefficients", "wrong_sample"}]
    ax.bar(range(len(wrong)), [f(r, "psnr_drop_from_normal") for r in wrong], color="#ba6b57")
    ax.set_xticks(range(len(wrong))); ax.set_xticklabels([short(r["method_id"]) + "\n" + r["perturbation_mode"].replace("_", " ") for r in wrong], fontsize=6, rotation=20, ha="right")
    ax.set_ylabel("PSNR drop"); ax.set_title("Measurement perturbation"); ax.grid(axis="y", alpha=0.25)
    ax = axes[1, 0]
    best = []
    for method in sorted({r["method_id"] for r in base}):
        sub = [r for r in base if r["method_id"] == method and r["baseline"] == "tv_pgd"]
        if sub:
            best.append(max(sub, key=lambda r: f(r, "psnr")))
    ax.bar(range(len(best)), [f(r, "psnr") for r in best], color="#7d9a52")
    ax.set_xticks(range(len(best))); ax.set_xticklabels([short(r["method_id"]) for r in best], rotation=25, ha="right")
    ax.set_ylabel("PSNR (dB)"); ax.set_title("CS-TV (PGD solver)"); ax.grid(axis="y", alpha=0.25)
    ax = axes[1, 1]
    x = np.arange(len(stats))
    means = [f(r, "mean_psnr") for r in stats]
    low = [f(r, "ci95_psnr_low") for r in stats]
    high = [f(r, "ci95_psnr_high") for r in stats]
    yerr = np.array([[m - l for m, l in zip(means, low)], [h - m for m, h in zip(means, high)]])
    ax.errorbar(x, means, yerr=yerr, fmt="o", color="#2f6f95", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([short(r["method_id"]) for r in stats], rotation=25, ha="right")
    ax.set_ylabel("PSNR (dB)"); ax.set_title("Bootstrap 95% CI"); ax.grid(axis="y", alpha=0.25)
    for i, ax in enumerate(axes.ravel()):
        panel_label(ax, chr(ord("a") + i))
    save(fig, "fig5_robustness_baselines.pdf")


def supplement_figures():
    copy_map = {
        "supp_dc_row_control.pdf": PHASE16 / "dc_row_control/dc_row_psnr.png",
        "supp_classwise_stl10.pdf": PHASE16 / "classwise/classwise_psnr.png",
        "supp_runtime_complexity.pdf": PHASE16 / "runtime_complexity/runtime_complexity.csv",
        "supp_psnr_histograms.pdf": PHASE16 / "statistics/psnr_histograms.png",
    }
    # Convert PNG-looking diagnostics to PDF by embedding in a simple figure.
    for name, src in copy_map.items():
        if not src.exists() or src.suffix.lower() != ".png":
            continue
        img = plt.imread(src)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.imshow(img)
        ax.axis("off")
        save(fig, name)


def main():
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    supplement_figures()
    print(f"wrote figures to {FIG} and {LATEX_FIG}")


if __name__ == "__main__":
    main()
'''


def write_outputs() -> None:
    for path in [OUT, LATEX, SECTIONS, SUPP_DIR, FIG_DIR, LATEX_FIG_DIR]:
        ensure_dir(path)
    write_text(OUT / "manuscript_v2.md", md_document())
    write_text(
        OUT / "manuscript_v2.tex",
        main_tex()
        .replace("sections/", "latex_project/sections/")
        .replace("figures/", "latex_project/figures/")
        .replace(r"\bibliography{references}", r"\bibliography{latex_project/references}"),
    )
    write_text(OUT / "supplement_v2.md", supplement_md())
    write_text(OUT / "supplement_v2.tex", supplement_tex().replace(r"\input{supplement/supplement}", r"\input{latex_project/supplement/supplement}"))
    write_text(OUT / "figures_to_make.md", figures_to_make())
    write_text(OUT / "tables_to_make.md", tables_to_make())
    write_text(OUT / "PAPER_WRITING_NOTES.md", writing_notes())
    write_text(OUT / "codex_make_figures.py", codex_make_figures_code())
    write_latex_project()
    write_json(
        OUT / "phase18_source_manifest.json",
        {
            "registry": str(REGISTRY),
            "phase16_tables": {k: str(v) for k, v in PHASE16_TABLES.items()},
            "output": str(OUT),
            "main_title": TITLE,
            "baseline_name": "CS-TV (PGD solver)",
        },
    )
    write_csv(
        OUT / "claims_included_excluded.csv",
        [
            {"type": "included", "claim": "STL-10 5% HQ supported for Rademacher and scrambled Hadamard."},
            {"type": "included", "claim": "STL-10 10% HQ supported for Rademacher and scrambled Hadamard."},
            {"type": "included", "claim": "MNIST/Fashion-MNIST 5% HQ supported."},
            {"type": "included", "claim": "Rademacher exact-A re-evaluation reproduced after safe cache rebuild."},
            {"type": "included", "claim": "Model refinement improves weak backprojections."},
            {"type": "excluded", "claim": "Strict SOTA."},
            {"type": "excluded", "claim": "Low-frequency Hadamard 5% is HQ on STL-10."},
            {"type": "excluded", "claim": "GAN is the final main mechanism."},
            {"type": "excluded", "claim": "Binary learned illumination is claimed."},
        ],
        ["type", "claim"],
    )


def main() -> None:
    write_outputs()
    print(json.dumps({"output": str(OUT), "latex_project": str(LATEX)}, indent=2))


if __name__ == "__main__":
    main()
