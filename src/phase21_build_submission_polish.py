from __future__ import annotations

import shutil
from pathlib import Path

from .phase20_common import (
    METHOD_LABEL,
    METHOD_ORDER,
    OUT as PHASE20_OUT,
    SIMPLE_METHODS,
    STL_METHODS,
    TABLES,
    TITLE,
    as_float,
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


OUT = Path("E:/ns_mc_gan_gi/outputs_phase21_submission_polish")
LATEX = OUT / "latex_project_v7"
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
        rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "BP PSNR": fmt(a["backproj_psnr"]),
                "Model PSNR": fmt(a["model_psnr"]),
                "Delta PSNR": fmt(a["delta_psnr"]),
                "Interpretation": "weak BP, large gain" if "rademacher" in a["measurement_family"] else "stronger BP, similar final",
            }
        )
    write_pack(
        "table2_measurement_attribution",
        rows,
        ["Method", "BP PSNR", "Model PSNR", "Delta PSNR", "Interpretation"],
        r"\textbf{Measurement attribution summary.} Final PSNR hides physical-initialization and neural-gain regimes.",
        "tab:measurement_attribution",
    )

    ab = {(r["method_id"], r["ablation_mode"]): r for r in table("ablation")}
    modes = [("full_model", "Full"), ("no_dc_project", "-DC"), ("no_null_project", "-Null"), ("stage1_only", "Stage1"), ("raw_weights", "Raw"), ("ema_weights", "EMA")]
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
    write_pack("tableS1_exact_a", exact, list(exact[0]), "Exact-operator reproducibility summary.", "tab:supp_exact_a")

    noise_rows = []
    for mid in STL_METHODS:
        sub = sorted([r for r in table("noise") if r["method_id"] == mid], key=lambda r: as_float(r["noise_std"]))
        if sub:
            noise_rows.append({"Method": METHOD_LABEL[mid], "Noise range": f"{fmt(sub[0]['noise_std'], 3)}-{fmt(sub[-1]['noise_std'], 3)}", "PSNR first": fmt(sub[0]["psnr"]), "PSNR last": fmt(sub[-1]["psnr"]), "SSIM last": fmt(sub[-1]["ssim"])})
    write_pack("tableS2_noise_sweep", noise_rows, list(noise_rows[0]), "Noise-sweep summary.", "tab:supp_noise")

    baseline_rows = []
    for mid in METHOD_ORDER:
        sub = [r for r in table("baseline") if r["method_id"] == mid and r["baseline"] == "tv_pgd"]
        if sub:
            best = max(sub, key=lambda r: as_float(r["psnr"]))
            baseline_rows.append({"Method": METHOD_LABEL[mid], "Ours PSNR": fmt(reg[mid]["psnr"]), "CS-TV PSNR": fmt(best["psnr"]), "CS-TV SSIM": fmt(best["ssim"])})
    write_pack("tableS3_cstv_baseline", baseline_rows, list(baseline_rows[0]), "CS-TV baseline summary.", "tab:supp_cstv")

    dc_rows = []
    for r in sorted(table("dc_row"), key=lambda x: (as_float(x["sampling_ratio"]), x["hadamard_include_dc"] != "True")):
        dc_rows.append(
            {
                "Sampling": f"{as_float(r['sampling_ratio']) * 100:.0f}%",
                "DC row": "include" if r["hadamard_include_dc"] == "True" else "skip",
                "BP PSNR": fmt(r["backproj_psnr"]),
                "BP SSIM": fmt(r["backproj_ssim"]),
            }
        )
    write_pack("tableS4_dc_row_control", dc_rows, ["Sampling", "DC row", "BP PSNR", "BP SSIM"], "DC-row control for low-frequency Hadamard backprojection.", "tab:supp_dc")

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
    write_pack("tableS5_statistics_ci", stats, list(stats[0]), "Bootstrap confidence intervals.", "tab:supp_ci")

    class_rows = [
        {
            "Method": METHOD_LABEL.get(r["method_id"], r["method_id"]),
            "Class": r.get("class_name", r.get("class", "")),
            "PSNR": fmt(r.get("psnr", r.get("mean_psnr"))),
            "SSIM": fmt(r.get("ssim", r.get("mean_ssim"))),
        }
        for r in table("classwise")[:10]
    ]
    write_pack("tableS6_classwise", class_rows, list(class_rows[0]), "Class-wise STL-10 diagnostic excerpt.", "tab:supp_classwise")

    runtime_rows = [
        {
            "Method": METHOD_LABEL.get(r["method_id"], r["method_id"]),
            "Path": r["path"].replace("ns_mc_gan_full_inference", "full inference"),
            "Samples": r["num_samples"],
            "Sec/image": fmt(r["runtime_sec_per_image"], 5),
            "Peak MB": fmt(r["peak_cuda_mem_mb"]),
        }
        for r in table("runtime")
        if r.get("path") in {"backprojection", "ns_mc_gan_full_inference"}
    ]
    write_pack("tableS7_runtime", runtime_rows, list(runtime_rows[0]), "Runtime and memory diagnostic summary.", "tab:supp_runtime")

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


def copy_figures_to_latex() -> None:
    ensure_dir(FIGS)
    for path in (OUT / "figures").glob("*"):
        if path.suffix.lower() in {".pdf", ".png", ".svg"}:
            shutil.copy2(path, FIGS / path.name)


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

This formulation also makes the role of the measurement family explicit. Rademacher measurements, scrambled Hadamard measurements, and low-frequency Hadamard measurements produce different physical initializations and different neural refinement behavior. Rademacher backprojections are weak in our experiments, but the learned inverse recovers high-quality images under the internal engineering thresholds defined in the protocol. Scrambled Hadamard measurements provide stronger physical initializations and reach similar final quality. Low-frequency Hadamard measurements provide interpretable backprojections and useful diagnostics, but low-frequency Hadamard at 5\% is not the primary high-quality STL-10 setting.

The contributions of this work are fourfold. First, we formulate low-sampling ghost imaging as measurement-consistent null-space neural reconstruction. Second, we implement a physically auditable reconstruction pipeline that combines a data solution, approximate null-space residual insertion, and measurement-consistency projection. Third, we demonstrate strict no-leak high-quality reconstruction on STL-10 at both 5\% and 10\% sampling under Rademacher and scrambled Hadamard measurements. Fourth, we provide validation analyses including exact-operator re-evaluation, measurement-family attribution, inference-time ablation, finite-noise sweep, measurement perturbation, TV-regularized compressed sensing, and bootstrap confidence intervals.
"""


def related_work() -> str:
    return r"""\section{Related Work}
Single-pixel imaging and computational ghost imaging recover images from structured illumination and bucket measurements rather than direct focal-plane sampling \cite{shapiro2008computational,edgar2019principles,gibson2020singlepixel}. These foundations motivate low-sampling reconstruction methods that exploit measurement design, prior information, and computational inversion.

Deep learning has been widely explored for ghost imaging and single-pixel reconstruction. Existing methods include convolutional reconstructions, residual networks, generative models, conditional networks, and physics-enhanced pipelines \cite{he2018ghost,wang2019learning,rizvi2020deepghost,bian2020residual,wang2022physics}. These works have shown that learned priors can improve visual quality and reconstruction speed in low-sampling regimes. However, many such methods are best understood as image-to-image or measurement-to-image mappings, and they do not always explicitly separate measured components from unmeasured null-space content.

Data consistency and null-space correction have also been studied in broader inverse-problem literature \cite{adler2018learned,aggarwal2019modl,schwab2019deepnull,goppel2023dataproximal}. These ideas are important because inverse problems are not ordinary image restoration tasks: the output must remain compatible with the forward model. Our work does not claim to introduce deep ghost imaging, data consistency, or null-space learning for the first time. Instead, it combines these principles into a low-sampling ghost-imaging reconstruction pipeline and evaluates them across multiple measurement families.

Measurement design is another important component of single-pixel imaging. Random Rademacher measurements, Hadamard measurements, and ordered or scrambled Hadamard patterns impose different structures on the inverse problem \cite{sun2017russian,zhang2017hadamard,cakecutting2019}. A measurement family can influence both the quality of the physical initialization and the difficulty of neural refinement. This work explicitly separates these two effects by reporting backprojection quality, final reconstruction quality, and neural gain.
"""


def problem_formulation() -> str:
    return r"""\section{Problem Formulation}
Let \(x\in\mathbb{R}^n\) denote the vectorized unknown image. Each illumination pattern \(a_i\in\mathbb{R}^n\) produces a scalar bucket measurement
\begin{equation}
y_i=\langle a_i,x\rangle+\epsilon_i .
\end{equation}
Stacking all measurements gives \(y=Ax+\epsilon\), where \(A\in\mathbb{R}^{m\times n}\). The sampling ratio is \(\rho=m/n\). For \(64\times64\) images, \(n=4096\). At 5\% sampling, \(m\approx205\), and at 10\% sampling, \(m\approx410\). Therefore \(m\ll n\), and the inverse problem is underdetermined.

In the noiseless case, the measurement-consistent set is \(\mathcal{C}_y=\{x:Ax=y\}\). The null space of the measurement operator is \(\mathrm{Null}(A)=\{v:Av=0\}\). If \(Ax_0=y\) and \(v\in\mathrm{Null}(A)\), then \(A(x_0+v)=Ax_0+Av=y\). Thus the measurement vector fixes only part of the image. Low-sampling reconstruction is therefore a constrained completion problem: the measured component should remain tied to \(y\), while the missing component must be inferred from prior information.
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
This equation is the central computational mechanism of the paper. It combines physical initialization, neural null-space completion, and final measurement checking. \Cref{fig:mechanism} visualizes the same logic.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_mechanism.pdf}
\caption{\textbf{Mechanism.} The method preserves measured row-space content, completes missing null-space structure with a neural residual, and checks the output against the bucket measurements.}
\label{fig:mechanism}
\end{figure*}

\subsection{Two-stage refiner}
The implemented high-quality reconstructor uses a two-stage structure. Stage 1 computes \(\hat{x}^{(1)}=\Pi_y[x_{\rm data}+P_N(G_\theta(x_{\rm data},z))]\). A refiner then predicts \(r_\phi=R_\phi(x_{\rm data},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\rm data}|)\), and the final output is \(\hat{x}=\Pi_y[\hat{x}^{(1)}+r_\phi]\).

\subsection{Exact operator handling}
For Rademacher measurements, \(A\) is random. Therefore evaluation must use the exact measurement operator used for the checkpoint. We export the exact operator and reload it during evaluation. After overriding \(A\), the matrix \(K=AA^T+\lambda I\) and its Cholesky cache must be rebuilt. Otherwise, the forward operation and inverse solver may correspond to different matrices. This exact-A cache-rebuilt path is used for all reported Rademacher results.
"""


def measurement_families() -> str:
    return r"""\section{Measurement Families}
\subsection{Rademacher measurements}
Rademacher measurements use signed random entries, \(A_{ij}\in\{-m^{-1/2},+m^{-1/2}\}\). Because this operator is random, exact-A reproducibility requires the exported operator and cache-rebuilt evaluation path described above. Rademacher measurements produce weak backprojections in our experiments, but final reconstruction quality is high after learned refinement.

\subsection{Scrambled Hadamard measurements}
Let \(H\in\{-1,+1\}^{n\times n}\) be a Hadamard matrix. We use \(H_{\rm norm}=n^{-1/2}H\), select scrambled rows, and form \(A\). Scrambled Hadamard measurements provide stronger physical initializations than Rademacher measurements while reaching similar final reconstruction quality.

\subsection{Low-frequency Hadamard measurements}
Low-frequency Hadamard measurements select low-sequency rows. For selected rows \(S\), zero-filled reconstruction uses \(c[S]=y\) and \(x_{\rm data}=H_{\rm norm}^Tc\). The DC row measures global brightness and strongly affects low-frequency Hadamard backprojection. However, low-frequency Hadamard at 5\% is not the primary STL-10 high-quality setting in this work.
"""


def experimental_protocol() -> str:
    return r"""\section{Experimental Protocol}
Primary natural-image experiments use STL-10 at 5\% and 10\% sampling with Rademacher and scrambled Hadamard measurements. MNIST and Fashion-MNIST are used as simple-domain sanity checks at 5\% sampling. Internal engineering thresholds are PSNR \(\ge 20\) and SSIM \(\ge 0.60\) for STL-10 at 5\%, PSNR \(\ge 22\) and SSIM \(\ge 0.65\) for STL-10 at 10\%, and PSNR \(\ge 25\) and SSIM \(\ge 0.80\) for MNIST/Fashion-MNIST at 5\%. These thresholds are not theoretical limits.

All primary results are strict no-leak evaluations of final imported checkpoints. For Rademacher measurements, we use exact exported operators and rebuild the solver cache before evaluation. Supplementary analyses do not introduce new training runs.
"""


def results() -> str:
    return r"""\section{Results}
\subsection{STL-10 reconstruction at 5\% and 10\%}
\Cref{tab:primary_results,fig:primary_metrics} summarize the primary results. At 5\% sampling, Rademacher measurements reach 22.316 dB PSNR and 0.635 SSIM, while scrambled Hadamard measurements reach 22.271 dB PSNR and 0.632 SSIM. Both exceed the internal STL-10 5\% high-quality threshold. At 10\% sampling, Rademacher reaches 24.781 dB PSNR and 0.747 SSIM, while scrambled Hadamard reaches 24.730 dB PSNR and 0.746 SSIM. Thus, both measurement families support high-quality STL-10 reconstruction at 5\% and 10\% sampling.

\input{tables/table1_primary_results.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig2_primary_metrics.pdf}
\caption{\textbf{Primary metrics.} PSNR and SSIM are shown for STL-10 5\%, STL-10 10\%, and MNIST/Fashion-MNIST 5\%. Dashed lines are internal engineering thresholds, not theoretical limits.}
\label{fig:primary_metrics}
\end{figure*}

\subsection{Qualitative reconstruction}
Large qualitative reconstructions are shown in \Cref{fig:qualitative_reconstruction}. The backprojections are incomplete and noisy, especially for Rademacher measurements. The learned reconstruction restores object-level structure while preserving measurement dependence. Images are enlarged for visibility and are intended as qualitative evidence; quantitative conclusions are based on the strict no-leak metrics.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig3_qualitative_reconstruction.pdf}
\caption{\textbf{Qualitative reconstruction.} STL-10 examples are shown as ground truth, backprojection, reconstruction, and absolute error. Images are enlarged for visibility; error maps are contrast-enhanced with a 99th-percentile scale.}
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
\caption{\textbf{Measurement attribution.} Pattern examples, backprojection-vs-model PSNR, neural gain, and the regime map show that final PSNR hides measurement-family regimes.}
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
\caption{\textbf{Inference ablation.} Removing measurement-consistency projection gives the strongest degradation; no-null removal has limited metric effect for these trained checkpoints and is not used as the sole evidence for null-space necessity.}
\label{fig:inference_ablation}
\end{figure*}

\subsection{Noise and perturbation tests}
\Cref{fig:validation_summary} summarizes finite-noise sweeps, measurement perturbations, CS-TV comparison, and bootstrap confidence intervals. Finite-noise sweeps show stable degradation over the tested noise range. Measurement perturbation tests are more diagnostic: shuffled coefficients and wrong-sample measurements cause large PSNR drops. This indicates that the model depends on the bucket measurement vector rather than generating a generic image prior.

\subsection{CS-TV compressed-sensing baseline}
We compare against a TV-regularized compressed-sensing baseline solved by projected gradient descent \cite{donoho2006compressed,candes2006robust,rudin1992tv}:
\begin{equation}
\min_x \frac{1}{2}\|Ax-y\|_2^2+\lambda\,\mathrm{TV}(x).
\end{equation}
We refer to this baseline as CS-TV (PGD solver). It is a lightweight small-subset traditional baseline, not an exhaustively optimized ADMM/FISTA or plug-and-play benchmark. Under the tested settings, the learned measurement-consistent reconstructor remains substantially stronger.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig6_validation_summary.pdf}
\caption{\textbf{Robustness and baselines.} Panels show finite-noise behavior, Shuffle/Wrong-y measurement perturbations, comparison against CS-TV, and bootstrap confidence intervals. These tests support finite-noise stability and measurement dependence; they do not imply universal robustness.}
\label{fig:validation_summary}
\end{figure*}

\subsection{DC row control}
For low-frequency Hadamard, including the DC row is critical for backprojection because it captures global brightness. Removing it severely degrades low-frequency Hadamard initialization. This is a measurement-design diagnostic and should not be generalized to all measurement families.

\subsection{Statistics and class-wise diagnostics}
Bootstrap confidence intervals show that the main results are stable across samples. Class-wise STL-10 diagnostics reveal expected category variability, with some classes consistently more difficult than others. These analyses support the robustness of the main claim but are not themselves the central contribution.
"""


def discussion() -> str:
    return r"""\section{Discussion}
The main lesson is that low-sampling ghost imaging should be interpreted as a physically constrained completion problem. A good reconstruction must not only look plausible but also remain tied to the measured bucket signal. The proposed pipeline achieves this by combining a data solution, null-space residual completion, and a measurement-consistency projection. The main novelty is not a new generic network, but a physics-constrained reconstruction decomposition that makes the role of measured data, null-space completion, and measurement projection explicit.

The second lesson is that measurement families play different roles. Rademacher measurements produce weak backprojections but high final quality after neural refinement. Scrambled Hadamard measurements provide stronger physical initialization and similar final quality. This suggests that physical initialization quality and learnability of the neural inverse are distinct properties.

The third lesson is that the method should not be presented as an adversarial-generation paper. Although adversarial ideas were considered in development, the final high-quality results are driven by measurement-consistent neural reconstruction and fidelity-oriented losses. The contribution is the physics-constrained reconstruction formulation, not adversarial generation.
"""


def limitations() -> str:
    return r"""\section{Limitations}
This study does not include a hardware optical experiment. It does not claim a strict state-of-the-art ranking because a broad external benchmark under matched protocols is not included. The CS-TV baseline is lightweight and small-subset, not an exhaustively optimized compressed-sensing solver. Robustness is tested only over finite noise and perturbation settings. Low-frequency Hadamard at 5\% is not a high-quality STL-10 setting in this work. Binary learned illumination is not claimed as successful, and adversarial training is not the final contribution mechanism. Future work should include hardware validation, broader external baselines, and more extensive cross-domain testing.
"""


def conclusion() -> str:
    return r"""\section{Conclusion}
We presented a measurement-consistent null-space neural reconstruction framework for low-sampling ghost imaging. By combining a physical data solution, neural null-space residual completion, and a final measurement-consistency projection, the method achieves high-quality STL-10 reconstruction at both 5\% and 10\% sampling under Rademacher and scrambled Hadamard measurements. Supplementary ablations, perturbation tests, exact-operator re-evaluation, and compressed-sensing baselines support the interpretation that the reconstructions are measurement-dependent rather than generic hallucinations. These results provide a physics-consistent route toward high-quality low-sampling ghost imaging reconstruction.
"""


def references_bib() -> str:
    return r"""@article{shapiro2008computational,
  author = {Shapiro, Jeffrey H.},
  title = {Computational Ghost Imaging},
  journal = {Physical Review A},
  volume = {78},
  pages = {061802},
  year = {2008}
}

@article{edgar2019principles,
  author = {Edgar, Matthew P. and Gibson, Graham M. and Padgett, Miles J.},
  title = {Principles and Prospects for Single-Pixel Imaging},
  journal = {Nature Photonics},
  volume = {13},
  pages = {13--20},
  year = {2019}
}

@article{gibson2020singlepixel,
  author = {Gibson, Graham M. and Johnson, Steven D. and Padgett, Miles J.},
  title = {Single-Pixel Imaging 12 Years On: A Review},
  journal = {Optics Express},
  volume = {28},
  number = {19},
  pages = {28190--28208},
  year = {2020}
}

@article{he2018ghost,
  author = {He, Ya and Wang, Guoan and Dong, Guohua and Zhu, Shiyang and Chen, Hu and Zhang, Anni and Xu, Zhihai},
  title = {Ghost Imaging Based on Deep Learning},
  journal = {Scientific Reports},
  volume = {8},
  pages = {6469},
  year = {2018}
}

@article{wang2019learning,
  author = {Wang, Fei and Wang, Hao and Wang, Haichao and Li, Guowei and Situ, Guohai},
  title = {Learning from Simulation: An End-to-End Deep-Learning Approach for Computational Ghost Imaging},
  journal = {Optics Express},
  volume = {27},
  number = {18},
  pages = {25560--25572},
  year = {2019}
}

@article{rizvi2020deepghost,
  author = {Rizvi, Saad and Cao, Jie and Zhang, Kaiyu and Hao, Qun},
  title = {DeepGhost: Real-Time Computational Ghost Imaging via Deep Learning},
  journal = {Scientific Reports},
  volume = {10},
  pages = {11400},
  year = {2020}
}

@article{bian2020residual,
  author = {Bian, Tong and Yi, Yuxuan and Hu, Jiale and Zhang, Yin and Wang, Yide and Gao, Lu},
  title = {A Residual-Based Deep Learning Approach for Ghost Imaging},
  journal = {Scientific Reports},
  volume = {10},
  pages = {12149},
  year = {2020}
}

@article{wang2022physics,
  author = {Wang, Fei and Wang, Chenglong and Deng, Chenjin and Han, Shensheng and Situ, Guohai},
  title = {Single-Pixel Imaging Using Physics Enhanced Deep Learning},
  journal = {Photonics Research},
  volume = {10},
  number = {1},
  pages = {104},
  year = {2022}
}

@article{adler2018learned,
  author = {Adler, Jonas and {\"O}ktem, Ozan},
  title = {Learned Primal-Dual Reconstruction},
  journal = {IEEE Transactions on Medical Imaging},
  volume = {37},
  number = {6},
  pages = {1322--1332},
  year = {2018}
}

@article{aggarwal2019modl,
  author = {Aggarwal, Hemant K. and Mani, Merry P. and Jacob, Mathews},
  title = {MoDL: Model-Based Deep Learning Architecture for Inverse Problems},
  journal = {IEEE Transactions on Medical Imaging},
  volume = {38},
  number = {2},
  pages = {394--405},
  year = {2019}
}

@article{schwab2019deepnull,
  author = {Schwab, Johannes and Antholzer, Stephan and Haltmeier, Markus},
  title = {Deep Null Space Learning for Inverse Problems: Convergence Analysis and Rates},
  journal = {Inverse Problems},
  volume = {35},
  number = {2},
  pages = {025008},
  year = {2019}
}

@article{goppel2023dataproximal,
  author = {G{\"o}ppel, Simon and Frikel, J{\"u}rgen and Haltmeier, Markus},
  title = {Data-Proximal Null-Space Networks for Inverse Problems},
  journal = {arXiv preprint arXiv:2309.06573},
  year = {2023}
}

@article{sun2017russian,
  author = {Sun, Ming-Jie and Meng, Ling-Tong and Edgar, Matthew P. and Padgett, Miles J. and Radwell, Neal},
  title = {A Russian Dolls Ordering of the Hadamard Basis for Compressive Single-Pixel Imaging},
  journal = {Scientific Reports},
  volume = {7},
  pages = {3464},
  year = {2017}
}

@article{zhang2017hadamard,
  author = {Zhang, Zibang and Wang, Xiaoping and Zheng, Guoan and Zhong, Jingang},
  title = {Hadamard Single-Pixel Imaging versus Fourier Single-Pixel Imaging},
  journal = {Optics Express},
  volume = {25},
  number = {16},
  pages = {19619--19639},
  year = {2017}
}

@article{cakecutting2019,
  author = {Yu, Wen-Kai},
  title = {Super Sub-Nyquist Single-Pixel Imaging by Means of Cake-Cutting Hadamard Basis Sort},
  journal = {Sensors},
  volume = {19},
  number = {19},
  pages = {4122},
  year = {2019}
}

@article{donoho2006compressed,
  author = {Donoho, David L.},
  title = {Compressed Sensing},
  journal = {IEEE Transactions on Information Theory},
  volume = {52},
  number = {4},
  pages = {1289--1306},
  year = {2006}
}

@article{candes2006robust,
  author = {Cand{\`e}s, Emmanuel J. and Romberg, Justin and Tao, Terence},
  title = {Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information},
  journal = {IEEE Transactions on Information Theory},
  volume = {52},
  number = {2},
  pages = {489--509},
  year = {2006}
}

@article{rudin1992tv,
  author = {Rudin, Leonid I. and Osher, Stanley and Fatemi, Emad},
  title = {Nonlinear Total Variation Based Noise Removal Algorithms},
  journal = {Physica D: Nonlinear Phenomena},
  volume = {60},
  number = {1--4},
  pages = {259--268},
  year = {1992}
}
"""


def citations_to_verify() -> str:
    return """# Citations To Verify

The PDF uses real reference entries instead of placeholder citations. Before journal submission, manually verify DOI fields and page ranges for all entries, especially:

- Wang et al. 2019 Optics Express computational ghost imaging.
- Wang et al. 2022 Photonics Research physics-enhanced deep learning.
- Yu et al. 2019 Sensors cake-cutting Hadamard ordering.
- Any publisher-required capitalization and author initials.
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


def supplement_text() -> str:
    return r"""\section{Supplementary Material}
Detailed CSV files are provided in the accompanying data package. The tables below are curated summaries intended for a compact submission supplement.

\input{tables/tableS1_exact_a.tex}
\input{tables/tableS2_noise_sweep.tex}
\input{tables/tableS3_cstv_baseline.tex}
\input{tables/tableS4_dc_row_control.tex}
\input{tables/tableS5_statistics_ci.tex}
\input{tables/tableS6_classwise.tex}
\input{tables/tableS7_runtime.tex}

\FloatBarrier
\section{Supplementary Ablation View}
\begin{figure*}[h]
\centering
\includegraphics[width=0.82\textwidth]{figures/figS_relmeaserr_ablation.pdf}
\caption{Supplementary measurement-error view of the no-DC projection ablation. Removing the projection increases measurement inconsistency.}
\label{fig:supp_relmeaserr_ablation}
\end{figure*}

\FloatBarrier
\section{Data and Code Availability}
The code and processed experiment manifests will be made available upon publication. Detailed CSV tables for supplementary analyses are included in the accompanying data package. Random Rademacher results require the exported exact measurement operators and the cache-rebuilt evaluation path.
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
\title{{Supplementary Material: {tex_escape(TITLE)}}}
\author{{Anonymous authors}}
\date{{}}
\begin{{document}}
\maketitle
\input{{supplement/supplement.tex}}
\end{{document}}
"""


def write_sections() -> None:
    ensure_dir(SECTIONS)
    for name, text in {
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
    }.items():
        write_text(SECTIONS / name, text)
    ensure_dir(SUPP)
    write_text(SUPP / "supplement.tex", supplement_text())


def plain_manuscript() -> str:
    return "\n\n".join([f"# {TITLE}", "## Abstract\n" + abstract(), introduction(), related_work(), problem_formulation(), method(), measurement_families(), experimental_protocol(), results(), validation_ablation(), discussion(), limitations(), conclusion()])


def main() -> None:
    ensure_dir(LATEX)
    build_tables()
    write_sections()
    copy_figures_to_latex()
    copy_tables_to_latex()
    write_text(LATEX / "main.tex", main_tex())
    write_text(LATEX / "supplement.tex", supplement_tex())
    write_text(LATEX / "references.bib", references_bib())
    write_text(LATEX / "citations_to_verify.md", citations_to_verify())
    write_text(OUT / "citations_to_verify.md", citations_to_verify())
    write_text(OUT / "manuscript_v7.tex", main_tex())
    write_text(OUT / "human_written_manuscript_v7.md", plain_manuscript())
    manifest = source_manifest()
    manifest["output"] = str(OUT)
    manifest["phase20_base"] = str(PHASE20_OUT)
    write_json(OUT / "internal_source_manifest.json", manifest)
    print({"latex_project_v7": str(LATEX)})


if __name__ == "__main__":
    main()
