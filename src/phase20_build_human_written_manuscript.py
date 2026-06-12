from __future__ import annotations

import shutil
from pathlib import Path

from .phase20_common import (
    METHOD_LABEL,
    METHOD_ORDER,
    OUT,
    SIMPLE_METHODS,
    STL_METHODS,
    TABLES,
    TITLE,
    as_float,
    copy_figures_to_latex,
    ensure_dir,
    fmt,
    markdown_table,
    registry_by_id,
    source_manifest,
    table,
    tex_escape,
    tex_table,
    write_csv,
    write_json,
    write_text,
)


LATEX = OUT / "latex_project"
SECTIONS = LATEX / "sections"
SUPP = LATEX / "supplement"
FIGS = LATEX / "figures"
TABLE_DIR = OUT / "tables"
LATEX_TABLES = LATEX / "tables"


def write_pack(name: str, rows: list[dict], fields: list[str], caption: str, label: str, *, wide: bool = True) -> None:
    write_csv(TABLE_DIR / f"{name}.csv", rows, fields)
    write_text(TABLE_DIR / f"{name}.md", markdown_table(rows, fields))
    write_text(TABLE_DIR / f"{name}.tex", tex_table(rows, fields, caption, label, wide=wide))


def build_tables() -> None:
    ensure_dir(TABLE_DIR)
    reg = registry_by_id()

    rows = []
    for mid in METHOD_ORDER:
        r = reg[mid]
        ratio = as_float(r["sampling_ratio"])
        psnr = as_float(r["psnr"])
        ssim = as_float(r["ssim"])
        if r["dataset"] == "STL-10" and abs(ratio - 0.05) < 1e-6:
            hq = psnr >= 20.0 and ssim >= 0.60
        elif r["dataset"] == "STL-10" and abs(ratio - 0.10) < 1e-6:
            hq = psnr >= 22.0 and ssim >= 0.65
        else:
            hq = psnr >= 25.0 and ssim >= 0.80
        rows.append(
            {
                "Dataset": r["dataset"],
                "Sampling": f"{ratio * 100:.0f}%",
                "Measurement": r["measurement_family"].replace("_", " "),
                "PSNR": fmt(r["psnr"]),
                "SSIM": fmt(r["ssim"]),
                "BP PSNR": fmt(r["backproj_psnr"]),
                "Delta PSNR": fmt(r["delta_psnr"]),
                "Threshold met": "yes" if hq else "no",
            }
        )
    write_pack(
        "table1_primary_results",
        rows,
        ["Dataset", "Sampling", "Measurement", "PSNR", "SSIM", "BP PSNR", "Delta PSNR", "Threshold met"],
        r"\textbf{Primary strict no-leak results.} Thresholds are internal engineering criteria stated in the protocol.",
        "tab:primary_results",
    )

    attr = {r["method_id"]: r for r in table("attribution")}
    rows = []
    for mid in STL_METHODS:
        a = attr[mid]
        interp = "weak BP, large gain" if "rademacher" in a["measurement_family"] else "stronger BP, similar final"
        rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "BP PSNR": fmt(a["backproj_psnr"]),
                "Model PSNR": fmt(a["model_psnr"]),
                "Delta PSNR": fmt(a["delta_psnr"]),
                "Interpretation": interp,
            }
        )
    write_pack(
        "table2_measurement_attribution",
        rows,
        ["Method", "BP PSNR", "Model PSNR", "Delta PSNR", "Interpretation"],
        r"\textbf{Measurement attribution.} Final PSNR hides physical-initialization and neural-gain regimes.",
        "tab:measurement_attribution",
    )

    ab = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    modes = [
        ("full_model", "Full"),
        ("no_dc_project", "-DC"),
        ("no_null_project", "-Null"),
        ("stage1_only", "Stage1"),
        ("raw_weights", "Raw"),
        ("ema_weights", "EMA"),
    ]
    rows = []
    for mid in STL_METHODS:
        row = {"Method": METHOD_LABEL[mid]}
        for mode, label in modes:
            row[label] = fmt(ab[(mid, mode)]["psnr"])
        rows.append(row)
    write_pack(
        "table3_ablation_summary",
        rows,
        ["Method", "Full", "-DC", "-Null", "Stage1", "Raw", "EMA"],
        r"\textbf{Inference ablation summary.} Values are PSNR in dB.",
        "tab:ablation_summary",
    )

    exact = [
        {
            "Method": METHOD_LABEL.get(r["method_id"], r["method_id"]),
            "Original PSNR": fmt(r["original_psnr"]),
            "Re-eval PSNR": fmt(r["reeval_psnr"]),
            "Abs diff": fmt(r["abs_diff_psnr"], 6),
            "Exact A": r["exact_A_loaded"],
            "Cache rebuilt": r["cache_rebuilt"],
        }
        for r in table("exact_a")
    ]
    write_pack("supp_exact_a", exact, list(exact[0]), "Exact-operator reproducibility summary.", "tab:supp_exact_a")

    noise_rows = []
    for mid in STL_METHODS:
        sub = [r for r in table("noise") if r["method_id"] == mid]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: as_float(r["noise_std"]))
        noise_rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "Noise range": f"{fmt(sub[0]['noise_std'], 3)}-{fmt(sub[-1]['noise_std'], 3)}",
                "PSNR first": fmt(sub[0]["psnr"]),
                "PSNR last": fmt(sub[-1]["psnr"]),
                "SSIM last": fmt(sub[-1]["ssim"]),
            }
        )
    write_pack("supp_noise_sweep", noise_rows, list(noise_rows[0]), "Noise-sweep summary.", "tab:supp_noise")

    baseline_rows = []
    for mid in METHOD_ORDER:
        sub = [r for r in table("baseline") if r["method_id"] == mid and r["baseline"] == "tv_pgd"]
        if not sub:
            continue
        best = max(sub, key=lambda r: as_float(r["psnr"]))
        baseline_rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "Ours PSNR": fmt(reg[mid]["psnr"]),
                "CS-TV PSNR": fmt(best["psnr"]),
                "CS-TV SSIM": fmt(best["ssim"]),
            }
        )
    write_pack("supp_traditional_baseline", baseline_rows, list(baseline_rows[0]), "Traditional CS-TV baseline summary.", "tab:supp_cstv")

    dc = [
        {
            "Method": METHOD_LABEL.get(r["method_id"], r["method_id"]),
            "With DC PSNR": fmt(r.get("with_dc_psnr")),
            "Without DC PSNR": fmt(r.get("without_dc_psnr")),
            "PSNR drop": fmt(r.get("psnr_drop")),
        }
        for r in table("dc_row")
    ]
    if dc:
        write_pack("supp_dc_row_control", dc, list(dc[0]), "DC-row control summary.", "tab:supp_dc")

    stats = [
        {
            "Method": METHOD_LABEL.get(r["method_id"], r["method_id"]),
            "Mean PSNR": fmt(r["mean_psnr"]),
            "PSNR CI95": f"{fmt(r['ci95_psnr_low'])}-{fmt(r['ci95_psnr_high'])}",
            "Mean SSIM": fmt(r["mean_ssim"]),
            "SSIM CI95": f"{fmt(r['ci95_ssim_low'])}-{fmt(r['ci95_ssim_high'])}",
        }
        for r in table("statistics")
    ]
    write_pack("supp_statistics_ci", stats, list(stats[0]), "Bootstrap confidence intervals.", "tab:supp_ci")

    class_rows = [
        {
            "Method": METHOD_LABEL.get(r["method_id"], r["method_id"]),
            "Class": r.get("class_name", r.get("class", "")),
            "PSNR": fmt(r.get("psnr", r.get("mean_psnr"))),
            "SSIM": fmt(r.get("ssim", r.get("mean_ssim"))),
        }
        for r in table("classwise")[:10]
    ]
    if class_rows:
        write_pack("supp_classwise", class_rows, list(class_rows[0]), "Class-wise STL-10 diagnostic excerpt.", "tab:supp_classwise")

    runtime_rows = [
        {
            "Method": METHOD_LABEL.get(r["method_id"], r["method_id"]),
            "Path": r["path"],
            "Samples": r["num_samples"],
            "Sec/image": fmt(r["runtime_sec_per_image"], 5),
            "Peak MB": fmt(r["peak_cuda_mem_mb"]),
        }
        for r in table("runtime")
        if r.get("path") in {"backprojection", "ns_mc_gan_full_inference"}
    ]
    if runtime_rows:
        write_pack("supp_runtime", runtime_rows, list(runtime_rows[0]), "Runtime and memory diagnostic summary.", "tab:supp_runtime")

    detail = ensure_dir(TABLE_DIR / "data_csv")
    for name in TABLES:
        rows = table(name)
        if rows:
            write_csv(detail / f"{name}.csv", rows)


def copy_tables_to_latex() -> None:
    ensure_dir(LATEX_TABLES)
    for path in TABLE_DIR.glob("*.tex"):
        shutil.copy2(path, LATEX_TABLES / path.name)
    src_detail = TABLE_DIR / "data_csv"
    if src_detail.exists():
        dst = ensure_dir(LATEX_TABLES / "data_csv")
        for path in src_detail.glob("*.csv"):
            shutil.copy2(path, dst / path.name)


def abstract() -> str:
    return r"""Ghost imaging and single-pixel imaging recover spatial information from structured illumination patterns and scalar bucket measurements. In low-sampling regimes, the inverse problem is severely underdetermined: many images are compatible with the same measurement vector. This makes unconstrained neural reconstruction risky, since a network may generate visually plausible structures that are not tied to the measured signal. We formulate low-sampling ghost imaging as a measurement-consistent null-space completion problem. The proposed reconstruction first computes a physical data solution from the forward operator, then lets a neural reconstructor complete missing structure through an approximate null-space residual, and finally applies a measurement-consistency projection to restore agreement with the bucket measurements. We evaluate Rademacher, scrambled Hadamard, and low-frequency Hadamard measurement families under strict no-leak protocols. For random Rademacher sensing, the exact exported measurement operator is reloaded and the solver cache is rebuilt for reproducible evaluation. On STL-10, the method reaches 22.316 dB PSNR / 0.635 SSIM at 5\% sampling with Rademacher measurements and 22.271 dB / 0.632 SSIM with scrambled Hadamard measurements. At 10\% sampling, the corresponding results are 24.781 dB / 0.747 SSIM and 24.730 dB / 0.746 SSIM. On MNIST and Fashion-MNIST at 5\% sampling, the method obtains 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM. Attribution, ablation, noise, perturbation, and compressed-sensing baseline analyses show that the improvements are measurement-dependent rather than generic image-prior hallucination."""


def introduction() -> str:
    return r"""\section{Introduction}
Ghost imaging and single-pixel imaging reconstruct spatial information from a sequence of known illumination patterns and scalar bucket detector readings. Instead of measuring an image directly with a dense sensor array, the system records projections of the unknown scene onto a set of illumination patterns. This acquisition model is attractive when detector arrays are expensive, unavailable, or difficult to deploy, but it shifts the burden from direct sensing to computational reconstruction.

The central difficulty is low sampling. If the unknown image is represented as \(x\in\mathbb{R}^n\) and the system collects \(m\) bucket measurements, the measurement model is
\begin{equation}
y=Ax+\epsilon,
\end{equation}
where \(A\in\mathbb{R}^{m\times n}\) is determined by the illumination patterns. In the low-sampling regime, \(m\ll n\), and the inverse problem is underdetermined. Many candidate images can explain the same measurement vector. Direct physical inverses, such as backprojection, preserve a transparent link to the measurements but often leave severe missing structure.

Deep neural networks can improve reconstruction quality in this regime, but unconstrained networks introduce a different risk. A network may generate image details that look plausible while moving away from the measured bucket signal. For computational imaging, this is not merely a perceptual issue. The reconstructed image should remain physically tied to the forward model and to the measured data.

This work treats low-sampling ghost imaging as a constrained completion problem. Instead of asking a network to reconstruct the whole image freely, we separate the reconstruction into three components: a physical data solution, a neural residual for missing structure, and a final measurement-consistency projection. The data solution carries information directly determined by the measurements. The neural residual estimates the information not fixed by the measurements. The final projection checks the output against the original bucket signal.

This formulation also makes the role of the measurement family explicit. Rademacher measurements, scrambled Hadamard measurements, and low-frequency Hadamard measurements produce different physical initializations and different neural refinement behavior. Rademacher backprojections are weak in our experiments, but the learned inverse recovers high-quality images. Scrambled Hadamard measurements provide stronger physical initializations and reach similar final quality. Low-frequency Hadamard measurements provide interpretable backprojections and useful diagnostics, but low-frequency Hadamard at 5\% is not the primary high-quality STL-10 setting.

The contributions of this work are fourfold. First, we formulate low-sampling ghost imaging as measurement-consistent null-space neural reconstruction. Second, we implement a physically auditable reconstruction pipeline that combines a data solution, approximate null-space residual insertion, and measurement-consistency projection. Third, we demonstrate strict no-leak high-quality reconstruction on STL-10 at both 5\% and 10\% sampling under Rademacher and scrambled Hadamard measurements. Fourth, we provide validation analyses including exact-operator re-evaluation, measurement-family attribution, inference-time ablation, finite-noise sweep, measurement perturbation, TV-regularized compressed sensing, and bootstrap confidence intervals.
"""


def related_work() -> str:
    return r"""\section{Related Work}
Deep learning has been widely explored for ghost imaging and single-pixel reconstruction. Existing methods include convolutional reconstructions, residual networks, generative models, conditional networks, and physics-enhanced pipelines \cite{TODOdeepgi}. These works have shown that learned priors can improve visual quality and reconstruction speed in low-sampling regimes. However, many such methods are best understood as image-to-image or measurement-to-image mappings, and they do not always explicitly separate measured components from unmeasured null-space content.

Data consistency and null-space correction have also been studied in broader inverse-problem literature \cite{TODOnullspace}. These ideas are important because inverse problems are not ordinary image restoration tasks: the output must remain compatible with the forward model. Our work does not claim to introduce deep ghost imaging, data consistency, or null-space learning for the first time. Instead, it combines these principles into a low-sampling ghost-imaging reconstruction pipeline and evaluates them across multiple measurement families.

Measurement design is another important component of single-pixel imaging \cite{TODOghost}. Random Rademacher measurements, Hadamard measurements, and ordered or scrambled Hadamard patterns impose different structures on the inverse problem. A measurement family can influence both the quality of the physical initialization and the difficulty of neural refinement. This work explicitly separates these two effects by reporting backprojection quality, final reconstruction quality, and neural gain.
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
where \(A\in\mathbb{R}^{m\times n}\). The sampling ratio is
\begin{equation}
\rho=\frac{m}{n}.
\end{equation}
For \(64\times64\) images, \(n=4096\). At 5\% sampling, \(m\approx205\), and at 10\% sampling, \(m\approx410\). Therefore \(m\ll n\), and the inverse problem is underdetermined.

In the noiseless case, the measurement-consistent set is
\begin{equation}
\mathcal{C}_y=\{x:Ax=y\}.
\end{equation}
The null space of the measurement operator is
\begin{equation}
\mathrm{Null}(A)=\{v:Av=0\}.
\end{equation}
If \(Ax_0=y\) and \(v\in\mathrm{Null}(A)\), then
\begin{equation}
A(x_0+v)=Ax_0+Av=y.
\end{equation}
Thus the measurement vector fixes only part of the image. Low-sampling reconstruction is therefore a constrained completion problem: the measured component should remain tied to \(y\), while the missing component must be inferred from prior information.
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
Define the regularized row-space operator
\begin{equation}
P_A=A^T(AA^T+\lambda I)^{-1}A,
\end{equation}
and the corresponding approximate null-space operator
\begin{equation}
P_N=I-P_A.
\end{equation}
Applied to a vector \(v\),
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av.
\end{equation}
If \(\lambda=0\) and \(A\) has full row rank, then \(AP_N(v)=0\). With \(\lambda>0\), this becomes a regularized approximate null-space projection.

The neural reconstructor predicts a residual
\begin{equation}
r_\theta=G_\theta(x_{\rm data},z),
\end{equation}
and the intermediate reconstruction is
\begin{equation}
\tilde{x}=x_{\rm data}+P_N(r_\theta).
\end{equation}
This step encourages the network to complete information not directly determined by the measurements.

\subsection{Measurement-consistency projection}
To restore agreement with the bucket measurements, we apply
\begin{equation}
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y).
\end{equation}
The final reconstruction is
\begin{equation}
\hat{x}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
This equation is the central computational mechanism of the paper. It combines physical initialization, neural null-space completion, and final measurement checking. \Cref{fig:mechanism} visualizes the same logic.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_mechanism.pdf}
\caption{\textbf{Mechanism.} The method preserves measured row-space content, completes missing null-space structure with a neural residual, and checks the output against the bucket measurements.}
\label{fig:mechanism}
\end{figure*}

\subsection{Two-stage refiner}
The implemented high-quality reconstructor uses a two-stage structure. Stage 1 computes
\begin{equation}
\hat{x}^{(1)}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))].
\end{equation}
A refiner then predicts
\begin{equation}
r_\phi=R_\phi(x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|),
\end{equation}
and the final output is
\begin{equation}
\hat{x}=\Pi_y[\hat{x}^{(1)}+r_\phi].
\end{equation}

\subsection{Exact operator handling}
For Rademacher measurements, \(A\) is random. Therefore evaluation must use the exact measurement operator used for the checkpoint. We export the exact operator and reload it during evaluation. After overriding \(A\), the matrix \(K=AA^T+\lambda I\) and its Cholesky cache must be rebuilt. Otherwise, the forward operation and inverse solver may correspond to different matrices. This exact-A cache-rebuilt path is used for all reported Rademacher results.
"""


def measurement_families() -> str:
    return r"""\section{Measurement Families}
\subsection{Rademacher measurements}
Rademacher measurements use signed random entries,
\begin{equation}
A_{ij}\in\{-m^{-1/2},+m^{-1/2}\}.
\end{equation}
They produce weak backprojections in our experiments, but final reconstruction quality is high after learned refinement.

\subsection{Scrambled Hadamard measurements}
Let \(H\in\{-1,+1\}^{n\times n}\) be a Hadamard matrix. We use
\begin{equation}
H_{\rm norm}=n^{-1/2}H,
\end{equation}
select scrambled rows, and form \(A\). Scrambled Hadamard measurements provide stronger physical initializations than Rademacher measurements while reaching similar final reconstruction quality.

\subsection{Low-frequency Hadamard measurements}
Low-frequency Hadamard measurements select low-sequency rows. For selected rows \(S\), zero-filled reconstruction uses
\begin{equation}
c[S]=y,\qquad x_{\rm data}=H_{\rm norm}^T c.
\end{equation}
The DC row measures global brightness and strongly affects low-frequency Hadamard backprojection. However, low-frequency Hadamard at 5\% is not the primary STL-10 high-quality setting in this work.
"""


def experimental_protocol() -> str:
    return r"""\section{Experimental Protocol}
Primary natural-image experiments use STL-10 at 5\% and 10\% sampling with Rademacher and scrambled Hadamard measurements. MNIST and Fashion-MNIST are used as simple-domain sanity checks at 5\% sampling. Internal engineering thresholds are PSNR \(\ge 20\) and SSIM \(\ge 0.60\) for STL-10 at 5\%, PSNR \(\ge 22\) and SSIM \(\ge 0.65\) for STL-10 at 10\%, and PSNR \(\ge 25\) and SSIM \(\ge 0.80\) for MNIST/Fashion-MNIST at 5\%. These thresholds are not theoretical limits.

All primary results are strict no-leak evaluations of final imported checkpoints. For Rademacher measurements, we use exact exported operators and rebuild the solver cache before evaluation. Supplementary analyses do not introduce new training runs.
"""


def results() -> str:
    return r"""\section{Results}
\subsection{STL-10 reconstruction at 5\% and 10\%}
\Cref{tab:primary_results,fig:main_results} summarize the primary results. At 5\% sampling, Rademacher measurements reach 22.316 dB PSNR and 0.635 SSIM, while scrambled Hadamard measurements reach 22.271 dB PSNR and 0.632 SSIM. Both exceed the internal STL-10 5\% high-quality threshold.

At 10\% sampling, Rademacher reaches 24.781 dB PSNR and 0.747 SSIM, while scrambled Hadamard reaches 24.730 dB PSNR and 0.746 SSIM. Thus, both measurement families support high-quality STL-10 reconstruction at 5\% and 10\% sampling.

\input{tables/table1_primary_results.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_main_results.pdf}
\caption{\textbf{Primary results.} PSNR and SSIM are shown for STL-10 5\%, STL-10 10\%, and MNIST/Fashion-MNIST 5\%. Dashed lines are internal engineering thresholds.}
\label{fig:main_results}
\end{figure*}

\subsection{Qualitative reconstruction}
Large qualitative reconstructions are shown in \Cref{fig:qualitative_reconstruction}. The backprojections are incomplete and noisy, especially for Rademacher measurements. The learned reconstruction restores object-level structure while preserving measurement dependence. Images are displayed enlarged for visibility and are intended as qualitative evidence; quantitative conclusions are based on the strict no-leak metrics.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth,height=0.84\textheight,keepaspectratio]{figures/fig3_qualitative_reconstruction.pdf}
\caption{\textbf{Qualitative reconstruction.} STL-10 examples are shown as ground truth, backprojection, reconstruction, and absolute error. Images are enlarged for visibility.}
\label{fig:qualitative_reconstruction}
\end{figure*}

\subsection{Simple-domain sanity checks}
On MNIST and Fashion-MNIST at 5\% sampling, the method reaches 27.692 dB / 0.956 SSIM and 25.019 dB / 0.837 SSIM, respectively. These experiments confirm that the same reconstruction pipeline works reliably on simpler structured domains, but they are not the main novelty.

\subsection{Measurement-family attribution}
\Cref{tab:measurement_attribution,fig:measurement_attribution} separate physical initialization from learned refinement. Final PSNR alone hides how different measurement families behave. Rademacher measurements have weak physical backprojections: 7.297 dB at 5\% and 7.756 dB at 10\%. However, final reconstruction reaches 22.316 dB and 24.781 dB, corresponding to gains of 15.019 dB and 17.025 dB. Scrambled Hadamard measurements start from stronger backprojections, 14.310 dB at 5\% and 14.533 dB at 10\%, and reach nearly the same final quality as Rademacher.

This suggests that Rademacher and scrambled Hadamard occupy different regimes. Rademacher is poor for direct inversion but highly recoverable by the learned measurement-consistent inverse. Scrambled Hadamard provides a stronger physical initialization and similar final quality.

\input{tables/table2_measurement_attribution.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig4_measurement_attribution.pdf}
\caption{\textbf{Measurement attribution.} Pattern examples, backprojection-vs-model PSNR, neural gain, and the regime map separate physical initialization from learned refinement.}
\label{fig:measurement_attribution}
\end{figure*}
"""


def validation_ablation() -> str:
    return r"""\section{Validation and Ablation}
\subsection{Exact-A reproducibility}
Rademacher measurements require exact-operator evaluation. Earlier mismatch was traced to stale solver-cache use after overriding \(A\). With safe exact-A loading and cache rebuilding, Rademacher 5\% and 10\% re-evaluations reproduce the Colab results with negligible differences. These reproduced results are used as primary evidence.

\subsection{Inference-time ablation}
\Cref{tab:ablation_summary,fig:inference_ablation} report the inference-time ablations. Removing the measurement-consistency projection causes the largest degradation. This shows that \(\Pi_y\) is not merely cosmetic; it is central to maintaining physical fidelity and image quality. Removing the null projection has limited metric effect for the trained checkpoints, suggesting that the final projection and the learned network already constrain many measured components. Stage1-only reconstruction is lower than the full model, and EMA weights provide slight stabilization.

\input{tables/table3_ablation_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig5_inference_ablation.pdf}
\caption{\textbf{Inference ablation.} Full reconstruction is compared with no-DC projection, no-null projection, stage-1-only output, raw weights, and EMA weights. The no-DC condition gives the strongest degradation.}
\label{fig:inference_ablation}
\end{figure*}

\subsection{Noise and perturbation tests}
\Cref{fig:robustness_baselines} summarizes finite-noise sweeps, measurement perturbations, CS-TV comparison, and bootstrap confidence intervals. Finite-noise sweeps show stable degradation over the tested noise range. Measurement perturbation tests are more diagnostic: shuffled coefficients and wrong-sample measurements cause large PSNR drops. This indicates that the model depends on the bucket measurement vector rather than generating a generic image prior.

\subsection{CS-TV compressed-sensing baseline}
We compare against a TV-regularized compressed-sensing baseline solved by projected gradient descent \cite{TODOtvc}:
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\,\mathrm{TV}(x).
\end{equation}
We refer to this baseline as CS-TV (PGD solver). It is a lightweight small-subset traditional baseline, not an exhaustively optimized ADMM/FISTA or plug-and-play benchmark. Under the tested settings, the learned measurement-consistent reconstructor remains substantially stronger.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_robustness_baselines.pdf}
\caption{\textbf{Robustness and baselines.} Panels show finite-noise behavior, Shuffle/Wrong-y measurement perturbations, comparison against CS-TV, and bootstrap confidence intervals. CS-TV is a TV-regularized compressed-sensing baseline solved by PGD.}
\label{fig:robustness_baselines}
\end{figure*}

\subsection{DC row control}
For low-frequency Hadamard, including the DC row is critical for backprojection because it captures global brightness. Removing it severely degrades low-frequency Hadamard initialization. This is a measurement-design diagnostic and should not be generalized to all measurement families.

\subsection{Statistics and class-wise diagnostics}
Bootstrap confidence intervals show that the main results are stable across samples. Class-wise STL-10 diagnostics reveal expected category variability, with some classes consistently more difficult than others. These analyses support the robustness of the main claim but are not themselves the central contribution.
"""


def discussion() -> str:
    return r"""\section{Discussion}
The main lesson is that low-sampling ghost imaging should be interpreted as a physically constrained completion problem. A good reconstruction must not only look plausible but also remain tied to the measured bucket signal. The proposed pipeline achieves this by combining a data solution, null-space residual completion, and a measurement-consistency projection.

The second lesson is that measurement families play different roles. Rademacher measurements produce weak backprojections but high final quality after neural refinement. Scrambled Hadamard measurements provide stronger physical initialization and similar final quality. This suggests that physical initialization quality and learnability of the neural inverse are distinct properties.

The third lesson is that the method should not be presented as an adversarial-generation paper. Although adversarial ideas were considered in development, the final high-quality results are driven by measurement-consistent neural reconstruction and fidelity-oriented losses. The contribution is the physics-constrained reconstruction formulation, not adversarial generation.
"""


def limitations() -> str:
    return r"""\section{Limitations}
This study does not include a hardware optical experiment. It does not claim leaderboard superiority because a broad external benchmark under matched protocols is not included. The CS-TV baseline is lightweight and small-subset, not an exhaustively optimized compressed-sensing solver. Robustness is tested only over finite noise and perturbation settings. Low-frequency Hadamard at 5\% is not a high-quality STL-10 setting in this work. Binary learned illumination is not claimed as successful, and adversarial training is not the final contribution mechanism. Future work should include hardware validation, broader external baselines, and more extensive cross-domain testing.
"""


def conclusion() -> str:
    return r"""\section{Conclusion}
We presented a measurement-consistent null-space neural reconstruction framework for low-sampling ghost imaging. By combining a physical data solution, neural null-space residual completion, and a final measurement-consistency projection, the method achieves high-quality STL-10 reconstruction at both 5\% and 10\% sampling under Rademacher and scrambled Hadamard measurements. Supplementary ablations, perturbation tests, exact-operator re-evaluation, and compressed-sensing baselines support the interpretation that the reconstructions are measurement-dependent rather than generic hallucinations. These results provide a physics-consistent route toward high-quality low-sampling ghost imaging reconstruction.
"""


def supplement() -> str:
    return r"""\section{Supplementary Material}
The supplement contains curated tables. Full CSV files are included with the project data directory and are not reproduced as oversized submission tables.

\input{tables/supp_exact_a.tex}
\input{tables/supp_noise_sweep.tex}
\input{tables/supp_traditional_baseline.tex}
\input{tables/supp_dc_row_control.tex}
\input{tables/supp_statistics_ci.tex}
\input{tables/supp_classwise.tex}
\input{tables/supp_runtime.tex}

\subsection{Ablation measurement error}
\begin{figure*}[h]
\centering
\includegraphics[width=0.82\textwidth]{figures/figS_ablation_relmeaserr.pdf}
\caption{Supplementary measurement-error view of the no-DC projection ablation.}
\label{fig:supp_ablation_relmeaserr}
\end{figure*}
"""


def references_bib() -> str:
    return r"""@article{TODOghost,
  title = {TODO VERIFY: Ghost imaging and single-pixel imaging foundations},
  author = {TODO VERIFY},
  journal = {TODO VERIFY},
  year = {TODO VERIFY}
}

@article{TODOdeepgi,
  title = {TODO VERIFY: Deep learning for ghost imaging reconstruction},
  author = {TODO VERIFY},
  journal = {TODO VERIFY},
  year = {TODO VERIFY}
}

@article{TODOnullspace,
  title = {TODO VERIFY: Data consistency and null-space methods for inverse problems},
  author = {TODO VERIFY},
  journal = {TODO VERIFY},
  year = {TODO VERIFY}
}

@article{TODOtvc,
  title = {TODO VERIFY: TV-regularized compressed sensing reconstruction},
  author = {TODO VERIFY},
  journal = {TODO VERIFY},
  year = {TODO VERIFY}
}
"""


def citations_to_verify() -> str:
    return """# Citations To Verify

The manuscript intentionally uses TODO BibTeX entries and does not include a dedicated placeholder-reference section in the PDF.

- TODOghost: foundational ghost imaging / single-pixel imaging citation.
- TODOdeepgi: representative deep-learning ghost-imaging reconstruction citation.
- TODOnullspace: data-consistency or null-space inverse-problem citation.
- TODOtvc: TV-regularized compressed-sensing / PGD baseline citation.
"""


def main_tex() -> str:
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
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
\input{{sections/introduction.tex}}
\input{{sections/related_work.tex}}
\input{{sections/problem_formulation.tex}}
\input{{sections/method.tex}}
\input{{sections/measurement_families.tex}}
\input{{sections/experimental_protocol.tex}}
\input{{sections/results.tex}}
\input{{sections/validation_ablation.tex}}
\input{{sections/discussion.tex}}
\input{{sections/limitations.tex}}
\input{{sections/conclusion.tex}}
\bibliographystyle{{plain}}
\bibliography{{references}}
\clearpage
\appendix
\input{{supplement/supplement.tex}}
\end{{document}}
"""


def plain_manuscript() -> str:
    return "\n\n".join(
        [
            f"# {TITLE}",
            "## Abstract\n" + abstract(),
            introduction(),
            related_work(),
            problem_formulation(),
            method(),
            measurement_families(),
            experimental_protocol(),
            results(),
            validation_ablation(),
            discussion(),
            limitations(),
            conclusion(),
        ]
    )


def write_sections() -> None:
    ensure_dir(SECTIONS)
    sections = {
        "abstract.tex": abstract(),
        "introduction.tex": introduction(),
        "related_work.tex": related_work(),
        "problem_formulation.tex": problem_formulation(),
        "method.tex": method(),
        "measurement_families.tex": measurement_families(),
        "experimental_protocol.tex": experimental_protocol(),
        "results.tex": results(),
        "validation_ablation.tex": validation_ablation(),
        "discussion.tex": discussion(),
        "limitations.tex": limitations(),
        "conclusion.tex": conclusion(),
    }
    for name, text in sections.items():
        write_text(SECTIONS / name, text)
    ensure_dir(SUPP)
    write_text(SUPP / "supplement.tex", supplement())


def main() -> None:
    ensure_dir(LATEX)
    build_tables()
    write_sections()
    copy_figures_to_latex(FIGS)
    copy_tables_to_latex()
    write_text(LATEX / "main.tex", main_tex())
    write_text(LATEX / "references.bib", references_bib())
    write_text(LATEX / "citations_to_verify.md", citations_to_verify())
    write_text(OUT / "citations_to_verify.md", citations_to_verify())
    write_text(OUT / "manuscript_v6.tex", main_tex())
    write_text(OUT / "human_written_manuscript_v6.md", plain_manuscript())
    write_json(OUT / "internal_source_manifest.json", source_manifest())
    print({"latex_project": str(LATEX), "main_tex": str(LATEX / "main.tex")})


if __name__ == "__main__":
    main()
