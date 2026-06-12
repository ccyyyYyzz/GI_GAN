from __future__ import annotations

from pathlib import Path

from .phase18_rewrite_common import (
    LONG_LABEL,
    METHOD_LABEL,
    OUT,
    TITLE,
    as_float,
    fmt,
    main_results_rows,
    markdown_table,
    registry_by_id,
    source_manifest,
    table,
    tex_escape_text,
    write_json,
    write_text,
)


MAIN_FIGURES = [
    ("fig:mechanism", "fig1_mechanism", "Method overview"),
    ("fig:measurement_attribution", "fig2_measurement_attribution", "Measurement-family attribution"),
    ("fig:main_results", "fig3_main_results", "Primary reconstruction results"),
    ("fig:inference_ablation", "fig4_inference_ablation", "Inference-time ablation"),
    ("fig:robustness_baselines", "fig5_robustness_baselines", "Robustness and CS-TV baseline"),
]

MAIN_TABLES = [
    ("tab:primary_results", "table1_primary_strict_noleak_results", "Primary strict no-leak results"),
    ("tab:measurement_attribution", "table2_measurement_attribution", "Measurement attribution"),
    ("tab:ablation_summary", "table3_inference_ablation_summary", "Inference ablation summary"),
]


def row(mid: str) -> dict[str, str]:
    return registry_by_id()[mid]


def f(mid: str, key: str, digits: int = 3) -> str:
    return fmt(row(mid).get(key), digits)


def sampling(mid: str) -> str:
    return f"{as_float(row(mid).get('sampling_ratio')) * 100:.0f}%"


def main_metrics_sentence() -> str:
    return (
        "On STL-10, the method reaches "
        f"{f('rademacher5_hq_noise001_colab', 'psnr')} dB PSNR and {f('rademacher5_hq_noise001_colab', 'ssim')} SSIM "
        "with Rademacher measurements at 5% sampling, and "
        f"{f('scrambled_hadamard5_hq_noise001_colab', 'psnr')} dB PSNR and {f('scrambled_hadamard5_hq_noise001_colab', 'ssim')} SSIM "
        "with scrambled Hadamard measurements at 5% sampling. "
        "At 10% sampling, the corresponding Rademacher and scrambled Hadamard results are "
        f"{f('rademacher10_full_noise001_colab', 'psnr')} dB / {f('rademacher10_full_noise001_colab', 'ssim')} SSIM and "
        f"{f('scrambled_hadamard10_full_noise001_colab', 'psnr')} dB / {f('scrambled_hadamard10_full_noise001_colab', 'ssim')} SSIM. "
        "On MNIST and Fashion-MNIST at 5% sampling, the same evaluation pipeline obtains "
        f"{f('mnist_hadamard5_full_colab', 'psnr')} dB / {f('mnist_hadamard5_full_colab', 'ssim')} SSIM and "
        f"{f('fashion_hadamard5_full_colab', 'psnr')} dB / {f('fashion_hadamard5_full_colab', 'ssim')} SSIM, respectively."
    )


def md_sections() -> dict[str, str]:
    exact = {r["method_id"]: r for r in table("exact_a")}
    return {
        "abstract": (
            "Ghost imaging reconstructs spatial structure from known illumination patterns and scalar bucket measurements. "
            "At low sampling ratios, the inverse problem is underdetermined: many images can explain the same measurements, "
            "while unconstrained learned reconstructions may look plausible without remaining tied to the measured signal. "
            "We formulate low-sampling ghost imaging as measurement-consistent null-space neural reconstruction. "
            "A physical data solution carries the measured row-space component, a neural residual completes missing structure, "
            "and a final projection restores measurement consistency. "
            + main_metrics_sentence()
            + " Ablation, exact-operator re-evaluation, finite noise, perturbation, and a lightweight CS-TV (PGD solver) baseline "
            "support the interpretation that the improvements are measurement-dependent rather than generic image-prior hallucination."
        ),
        "introduction": (
            "Ghost imaging and single-pixel imaging recover an image from structured illumination patterns and bucket detector readings. "
            "This acquisition model is useful when dense detector arrays are expensive, unavailable, or undesirable. "
            "However, the low-sampling regime is mathematically difficult because the number of measurements is far smaller than the number of pixels.\n\n"
            "A direct physical inverse, such as backprojection, preserves a transparent link to the measurements but often leaves severe missing structure. "
            "A purely learned inverse can improve visual quality, but without an explicit data-consistency mechanism it may move away from the bucket measurements. "
            "The central design goal in this manuscript is therefore not only to improve PSNR and SSIM, but also to keep the reconstruction anchored to the actual measurement vector.\n\n"
            "The proposed pipeline is summarized in Figure 1. We start from a data solution, insert the neural residual through an approximate null-space projection, and apply a final measurement-consistency projection. "
            "This makes low-sampling reconstruction a constrained completion problem rather than ordinary image denoising.\n\n"
            "The experiments also show that measurement design changes the interpretation of performance. Rademacher measurements have weak backprojections but large learned gains. "
            "Scrambled Hadamard measurements give stronger physical initializations and similar final quality. Low-frequency Hadamard rows are interpretable and useful for simple-domain or diagnostic controls, but low-frequency Hadamard at 5% should not be treated as a primary STL-10 high-quality claim."
        ),
        "related_work": (
            "Deep learning has been widely used for single-pixel imaging and ghost imaging, including convolutional, residual, and generative reconstruction models. "
            "Separately, data consistency, null-space correction, and projected reconstruction are established tools in inverse problems. "
            "The contribution here is therefore not a claim of being the first learned ghost-imaging method. "
            "Rather, the work combines explicit measurement consistency, null-space residual insertion, measurement-family attribution, exact random-operator re-evaluation, and a compact set of validation experiments in a low-sampling GI setting.\n\n"
            "All bibliography entries in the generated BibTeX file are marked TODO-VERIFY and should be replaced with verified references before submission."
        ),
        "problem_formulation": (
            "Let x in R^n denote the vectorized object or image, A in R^{m x n} the known measurement matrix, and y in R^m the bucket-measurement vector. "
            "Each scalar bucket reading follows y_i = <a_i, x> + epsilon_i, and the stacked forward model is y = A x + epsilon. "
            "The sampling ratio is rho = m/n, and the low-sampling regime has m << n.\n\n"
            "When m is much smaller than n, the inverse problem is underdetermined. In the noiseless case, the feasible set is C_y = {x : A x = y}. "
            "The null space is Null(A) = {v : A v = 0}. If A x_0 = y and v lies in Null(A), then A(x_0 + v) = y. "
            "Thus the measurement vector determines only part of the image. Low-sampling ghost imaging is not ordinary denoising; it is completion of missing information under a measurement constraint."
        ),
        "method": (
            "We compute a regularized data solution x_data = A^T(AA^T + lambda I)^{-1} y. "
            "This solution lives in the measured row-space direction and provides a physically meaningful starting point. "
            "The associated row-space projector is P_A = A^T(AA^T + lambda I)^{-1} A, and the approximate null-space projector is P_N = I - P_A. "
            "Applied to a residual v, this gives P_N(v) = v - A^T(AA^T + lambda I)^{-1} A v.\n\n"
            "If lambda = 0 and A has full row rank, then A P_N(v) = 0 exactly. With lambda > 0, P_N is a regularized approximate null-space projection. "
            "The neural reconstructor predicts r_theta = G_theta(x_data, z), and we form x_tilde = x_data + P_N(r_theta). "
            "A final measurement-consistency projection Pi_y(v) = v - A^T(AA^T + lambda I)^{-1}(A v - y) gives x_hat = Pi_y(x_tilde). "
            "If lambda = 0, A Pi_y(v) = y. With lambda > 0, the projection approximately enforces consistency.\n\n"
            "The implemented reconstructor has two stages. Stage 1 computes x_hat^(1) = Pi_y[x_data + P_N(G_theta(x_data, z))]. "
            "A refiner then predicts r_phi = R_phi(x_data, x_hat^(1), |x_hat^(1)-x_data|), and the final estimate is x_hat = Pi_y[x_hat^(1) + r_phi]. "
            "For Rademacher measurements, the exact random measurement operator is exported and reloaded. After overriding A, K = AA^T + lambda I and its Cholesky cache must be rebuilt; otherwise the solver can use a stale inverse for a different matrix. "
            "Image metrics use clip(x_hat, 0, 1), while measurement error is computed on the unclamped x_hat because clipping after projection can alter A x_hat."
        ),
        "measurement_families": (
            "Rademacher measurements use A_ij in {-1/sqrt(m), +1/sqrt(m)}. Their backprojections are weak in these experiments, so final quality depends strongly on the learned inverse. "
            "Because the operator is random, exact-A re-evaluation is required for reproducible evaluation.\n\n"
            "Scrambled Hadamard measurements use H_norm = H/sqrt(n) with selected scrambled rows. This produces a stronger physical initialization than Rademacher measurements, while final quality remains close to the Rademacher models at the same sampling ratio.\n\n"
            "Low-frequency Hadamard measurements select low-sequency rows. The DC row measures average brightness and can strongly affect zero-filled backprojection. "
            "For Hadamard zero filling, measured coefficients are assigned by c[selected rows] = y and x_data = H_norm^T c. "
            "Low-frequency Hadamard 5% is used here for simple-domain sanity checks and diagnostic controls, not as a primary STL-10 high-quality claim. "
            "Overall, measurement design affects both x_data quality and neural refinement difficulty."
        ),
        "training_losses": (
            "Training uses image reconstruction losses and measurement-related penalties from the existing pipeline, but the final manuscript frames the contribution as measurement-consistent neural reconstruction rather than as a GAN mechanism. "
            "Adversarial components, if present in earlier development, are not the claimed final mechanism. The reported validation focuses on strict no-leak evaluation, exact operator handling, inference ablation, and measurement perturbation."
        ),
        "experimental_protocol": (
            "The primary natural-image experiments use STL-10 at 5% and 10% sampling with Rademacher and scrambled Hadamard measurements. "
            "Simple-domain sanity checks use MNIST and Fashion-MNIST at 5% sampling. "
            "The internal engineering thresholds are PSNR >= 20 and SSIM >= 0.60 for STL-10 at 5%, PSNR >= 22 and SSIM >= 0.65 for STL-10 at 10%, and PSNR >= 25 and SSIM >= 0.80 for MNIST/Fashion-MNIST at 5%. "
            "These thresholds are engineering criteria for this study, not theoretical limits.\n\n"
            "All primary metrics are strict no-leak evaluations of imported checkpoints. "
            "For Rademacher experiments, exact measurement operators are loaded and the solver cache is rebuilt before evaluation. "
            "The reported ablations and diagnostics do not introduce new training runs."
        ),
        "results": (
            "Primary results are summarized in Table 1 and Figure 3. "
            + main_metrics_sentence()
            + " Under the engineering thresholds above, STL-10 5% and 10% are supported as high-quality under Rademacher and scrambled Hadamard measurement families.\n\n"
            "MNIST and Fashion-MNIST provide simple-domain sanity checks rather than the main novelty. "
            "They show that the same code path is stable on structured simple targets at 5% sampling.\n\n"
            "Backprojection and final reconstruction must be separated. Table 2 and Figure 2 show that Rademacher backprojection is weak but final reconstruction is strong: Rad-5 improves from "
            f"{f('rademacher5_hq_noise001_colab', 'backproj_psnr')} dB to {f('rademacher5_hq_noise001_colab', 'psnr')} dB, while Rad-10 improves from "
            f"{f('rademacher10_full_noise001_colab', 'backproj_psnr')} dB to {f('rademacher10_full_noise001_colab', 'psnr')} dB. "
            "Scrambled Hadamard starts from a stronger backprojection, but final quality is close to Rademacher at the same sampling ratio. "
            "Final PSNR alone therefore hides measurement-family behavior.\n\n"
            "A careful phrasing is important: STL-10 5% high-quality reconstruction is supported under suitable measurement families. "
            "The earlier low-frequency Hadamard 5% diagnostic should not be generalized into a failure of all STL-10 5% measurement designs."
        ),
        "ablation_validation": (
            "Exact-A reproducibility is a key validation step. The Rademacher re-evaluation reproduces the random-operator checkpoints when the exact exported A is loaded and K = AA^T + lambda I is rebuilt before solving. "
            "This avoids random-matrix mismatch and stale-cache errors. In the exact-A audit, Rad-5 reports an absolute PSNR difference of "
            f"{fmt(exact['rademacher5_hq_noise001_colab'].get('abs_diff_psnr'))} dB and Rad-10 reports "
            f"{fmt(exact['rademacher10_full_noise001_colab'].get('abs_diff_psnr'))} dB.\n\n"
            "Inference ablation is summarized in Table 3 and Figure 4. Removing the measurement-consistency projection causes the largest degradation, especially for scrambled Hadamard where removing the DC projection collapses PSNR. "
            "Removing the null projection has limited metric effect for these checkpoints. The honest interpretation is that the final measurement projection and the trained network may already constrain measured components; the null projection remains part of the designed pipeline, but inference-time removal has limited effect in these trained models. "
            "Stage-1-only reconstruction remains below the full model, and EMA weights are slightly more stable than raw weights.\n\n"
            "Figure 5 collects finite noise, measurement perturbation, CS-TV, and bootstrap confidence results. "
            "The noise sweep supports robustness only over the tested finite noise range. Shuffled coefficients and wrong-sample measurements produce large drops, indicating dependence on y rather than a generic image prior. "
            "The CS-TV baseline solves min_x 1/2 ||Ax-y||_2^2 + lambda TV(x) by projected gradient descent. We therefore describe it as a TV-regularized compressed-sensing baseline solved by PGD, abbreviated CS-TV (PGD solver). "
            "It is a lightweight small-subset baseline, not an exhaustively optimized ADMM/FISTA or plug-and-play comparison.\n\n"
            "The DC-row control applies only to low-frequency Hadamard and explains the importance of global brightness. Bootstrap intervals and class-wise STL-10 diagnostics provide uncertainty and heterogeneity checks, but class-wise trends are diagnostic rather than a central claim."
        ),
        "discussion": (
            "The first lesson is that measurement family and neural refinement are complementary. Rademacher measurements give weak physical backprojections but large learned gains. "
            "Scrambled Hadamard measurements give stronger physical initializations and similar final results. Low-frequency Hadamard gives interpretable physical structure, but it is not always the strongest final measurement family.\n\n"
            "The second lesson is that measurement consistency matters. The no-DC projection ablation, relative measurement error, exact-A handling, and perturbation tests all point to a model that depends on y. "
            "This matters because visual quality alone is insufficient for an optical inverse problem.\n\n"
            "The third lesson is about positioning. The method should not be presented as a GAN paper. The final mechanism is measurement-consistent neural reconstruction. "
            "The general idea is the formulation and projection mechanism; the present evidence is limited to the tested datasets, simulated measurement model, finite noise range, and lightweight CS-TV comparison."
        ),
        "limitations": (
            "This study has several limitations. It does not include a hardware optical experiment. It does not make a strict leaderboard claim. "
            "The CS-TV (PGD solver) baseline is lightweight and small-subset, not an exhaustive ADMM/FISTA or plug-and-play benchmark. "
            "Noise robustness is demonstrated only over the finite tested range. Class-wise results are diagnostic only. "
            "Low-frequency Hadamard 5% is not a primary STL-10 high-quality result. "
            "Binary learned illumination is not claimed as a successful final result. "
            "GAN components are not the final main mechanism. "
            "Exact-A handling is essential for random measurements, and any future random-measurement evaluation must reload the exact operator and rebuild the solver cache."
        ),
        "conclusion": (
            "Measurement-consistent null-space neural reconstruction provides a compact and auditable route to high-quality low-sampling ghost imaging. "
            "The results support STL-10 reconstruction at 5% and 10% sampling under Rademacher and scrambled Hadamard measurements, and they show stable simple-domain sanity performance on MNIST and Fashion-MNIST at 5%. "
            "The validation package clarifies why the model is measurement-dependent, why exact-A handling matters, and why measurement-family attribution is necessary for interpreting low-sampling GI results."
        ),
    }


def tex_sections() -> dict[str, str]:
    exact = {r["method_id"]: r for r in table("exact_a")}
    return {
        "abstract": (
            r"Ghost imaging reconstructs spatial structure from known illumination patterns and scalar bucket measurements. "
            r"At low sampling ratios, the inverse problem is underdetermined: many images can explain the same measurements, while unconstrained learned reconstructions may look plausible without remaining tied to the measured signal. "
            r"We formulate low-sampling ghost imaging as measurement-consistent null-space neural reconstruction. "
            r"A physical data solution carries the measured row-space component, a neural residual completes missing structure, and a final projection restores measurement consistency. "
            + tex_escape_text(main_metrics_sentence()).replace(r"\%", r"\%")
            + r" Ablation, exact-operator re-evaluation, finite noise, perturbation, and a lightweight CS-TV (PGD solver) baseline support the interpretation that the improvements are measurement-dependent rather than generic image-prior hallucination."
        ),
        "introduction": r"""
Ghost imaging and single-pixel imaging recover an image from structured illumination patterns and bucket detector readings. This acquisition model is useful when dense detector arrays are expensive, unavailable, or undesirable. However, the low-sampling regime is mathematically difficult because the number of measurements is far smaller than the number of pixels.

A direct physical inverse, such as backprojection, preserves a transparent link to the measurements but often leaves severe missing structure. A purely learned inverse can improve visual quality, but without an explicit data-consistency mechanism it may move away from the bucket measurements. The central design goal in this manuscript is therefore not only to improve PSNR and SSIM, but also to keep the reconstruction anchored to the actual measurement vector.

The proposed pipeline is summarized in \cref{fig:mechanism}. We start from a data solution, insert the neural residual through an approximate null-space projection, and apply a final measurement-consistency projection. This makes low-sampling reconstruction a constrained completion problem rather than ordinary image denoising.

The experiments also show that measurement design changes the interpretation of performance. Rademacher measurements have weak backprojections but large learned gains. Scrambled Hadamard measurements give stronger physical initializations and similar final quality. Low-frequency Hadamard rows are interpretable and useful for simple-domain or diagnostic controls, but low-frequency Hadamard at 5\% should not be treated as a primary STL-10 high-quality claim.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig1_mechanism.pdf}
\caption{\textbf{Measurement-consistent null-space neural reconstruction.} The pipeline starts from the measured row-space data solution, inserts learned residual structure through an approximate null-space projection, and restores measurement consistency with a final projection.}
\label{fig:mechanism}
\end{figure*}
""",
        "related_work": r"""
Deep learning has been widely used for single-pixel imaging and ghost imaging, including convolutional, residual, and generative reconstruction models. Separately, data consistency, null-space correction, and projected reconstruction are established tools in inverse problems. The contribution here is therefore not a claim of being the first learned ghost-imaging method. Rather, the work combines explicit measurement consistency, null-space residual insertion, measurement-family attribution, exact random-operator re-evaluation, and a compact set of validation experiments in a low-sampling GI setting.

All bibliography entries in the generated BibTeX file are marked TODO-VERIFY and should be replaced with verified references before submission.
""",
        "problem_formulation": r"""
Let \(x\in\mathbb{R}^{n}\) denote the vectorized object or image, \(A\in\mathbb{R}^{m\times n}\) the known measurement matrix, and \(y\in\mathbb{R}^{m}\) the bucket-measurement vector. Each scalar bucket reading is
\begin{equation}
y_i = \langle a_i, x\rangle + \epsilon_i,
\end{equation}
and the stacked forward model is
\begin{equation}
y = A x + \epsilon.
\end{equation}
The sampling ratio is \(\rho=m/n\), and the low-sampling regime has \(m\ll n\).

When \(m\) is much smaller than \(n\), the inverse problem is underdetermined. In the noiseless case, the feasible set is
\begin{equation}
\mathcal{C}_y=\{x:Ax=y\}.
\end{equation}
The null space is
\begin{equation}
\mathrm{Null}(A)=\{v:Av=0\}.
\end{equation}
If \(A x_0=y\) and \(v\in\mathrm{Null}(A)\), then \(A(x_0+v)=y\). Thus the measurement vector determines only part of the image. Low-sampling ghost imaging is not ordinary denoising; it is completion of missing information under a measurement constraint.
""",
        "method": r"""
We compute a regularized data solution
\begin{equation}
x_{\mathrm{data}} = A^T(AA^T+\lambda I)^{-1}y.
\end{equation}
This solution lives in the measured row-space direction and provides a physically meaningful starting point. The associated row-space projector is
\begin{equation}
P_A = A^T(AA^T+\lambda I)^{-1}A,
\end{equation}
and the approximate null-space projector is \(P_N=I-P_A\). Applied to a residual \(v\),
\begin{equation}
P_N(v)=v-A^T(AA^T+\lambda I)^{-1}Av.
\end{equation}
If \(\lambda=0\) and \(A\) has full row rank, then \(A P_N(v)=0\) exactly. With \(\lambda>0\), \(P_N\) is a regularized approximate null-space projection.

The neural reconstructor predicts
\begin{equation}
r_\theta=G_\theta(x_{\mathrm{data}},z),
\end{equation}
and the residual is inserted as
\begin{equation}
\tilde{x}=x_{\mathrm{data}}+P_N(r_\theta).
\end{equation}
A final measurement-consistency projection
\begin{equation}
\Pi_y(v)=v-A^T(AA^T+\lambda I)^{-1}(Av-y)
\end{equation}
gives
\begin{equation}
\hat{x}=\Pi_y(\tilde{x}).
\end{equation}
If \(\lambda=0\), \(A\Pi_y(v)=y\). With \(\lambda>0\), the projection approximately enforces consistency.

The implemented reconstructor has two stages. Stage 1 computes
\begin{equation}
\hat{x}^{(1)}=\Pi_y[x_{\mathrm{data}}+P_N(G_\theta(x_{\mathrm{data}},z))].
\end{equation}
A refiner predicts
\begin{equation}
r_\phi=R_\phi(x_{\mathrm{data}},\hat{x}^{(1)},|\hat{x}^{(1)}-x_{\mathrm{data}}|),
\end{equation}
and the final estimate is
\begin{equation}
\hat{x}=\Pi_y[\hat{x}^{(1)}+r_\phi].
\end{equation}

For Rademacher measurements, the exact random measurement operator is exported and reloaded. After overriding \(A\), \(K=AA^T+\lambda I\) and its Cholesky cache must be rebuilt; otherwise the solver can use a stale inverse for a different matrix. Image metrics use \(\mathrm{clip}(\hat{x},0,1)\), while measurement error is computed on the unclamped \(\hat{x}\) because clipping after projection can alter \(A\hat{x}\).
""",
        "measurement_families": r"""
Rademacher measurements use \(A_{ij}\in\{-1/\sqrt{m},+1/\sqrt{m}\}\). Their backprojections are weak in these experiments, so final quality depends strongly on the learned inverse. Because the operator is random, exact-A re-evaluation is required for reproducible evaluation.

Scrambled Hadamard measurements use
\begin{equation}
H_{\mathrm{norm}} = H/\sqrt{n},
\end{equation}
with selected scrambled rows. This produces a stronger physical initialization than Rademacher measurements, while final quality remains close to the Rademacher models at the same sampling ratio.

Low-frequency Hadamard measurements select low-sequency rows. The DC row measures average brightness and can strongly affect zero-filled backprojection. For Hadamard zero filling,
\begin{equation}
c[\mathrm{selected\ rows}] = y,\qquad x_{\mathrm{data}}=H_{\mathrm{norm}}^T c.
\end{equation}
Low-frequency Hadamard 5\% is used here for simple-domain sanity checks and diagnostic controls, not as a primary STL-10 high-quality claim. Overall, measurement design affects both \(x_{\mathrm{data}}\) quality and neural refinement difficulty.
""",
        "training_losses": r"""
Training uses image reconstruction losses and measurement-related penalties from the existing pipeline, but the final manuscript frames the contribution as measurement-consistent neural reconstruction rather than as a GAN mechanism. Adversarial components, if present in earlier development, are not the claimed final mechanism. The reported validation focuses on strict no-leak evaluation, exact operator handling, inference ablation, and measurement perturbation.
""",
        "experimental_protocol": r"""
The primary natural-image experiments use STL-10 at 5\% and 10\% sampling with Rademacher and scrambled Hadamard measurements. Simple-domain sanity checks use MNIST and Fashion-MNIST at 5\% sampling. The internal engineering thresholds are PSNR \(\geq20\) and SSIM \(\geq0.60\) for STL-10 at 5\%, PSNR \(\geq22\) and SSIM \(\geq0.65\) for STL-10 at 10\%, and PSNR \(\geq25\) and SSIM \(\geq0.80\) for MNIST/Fashion-MNIST at 5\%. These thresholds are engineering criteria for this study, not theoretical limits.

All primary metrics are strict no-leak evaluations of final imported checkpoints. For Rademacher experiments, exact measurement operators are loaded and the solver cache is rebuilt before evaluation. The reported ablations and diagnostics do not introduce new training runs.
""",
        "results": rf"""
Primary results are summarized in \cref{{tab:primary_results,fig:main_results}}. {tex_escape_text(main_metrics_sentence())} Under the engineering thresholds above, STL-10 5\% and 10\% are supported as high-quality under Rademacher and scrambled Hadamard measurement families.

\input{{tables/table1_primary_strict_noleak_results.tex}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig3_main_results.pdf}}
\caption{{\textbf{{Primary reconstruction results.}} STL-10 5\% and 10\% results exceed the engineering thresholds under Rademacher and scrambled Hadamard measurements. MNIST and Fashion-MNIST are simple-domain sanity checks at 5\%.}}
\label{{fig:main_results}}
\end{{figure*}}

MNIST and Fashion-MNIST provide simple-domain sanity checks rather than the main novelty. They show that the same code path is stable on structured simple targets at 5\% sampling.

Backprojection and final reconstruction must be separated. \Cref{{tab:measurement_attribution,fig:measurement_attribution}} show that Rademacher backprojection is weak but final reconstruction is strong: Rad-5 improves from {f('rademacher5_hq_noise001_colab', 'backproj_psnr')} dB to {f('rademacher5_hq_noise001_colab', 'psnr')} dB, while Rad-10 improves from {f('rademacher10_full_noise001_colab', 'backproj_psnr')} dB to {f('rademacher10_full_noise001_colab', 'psnr')} dB. Scrambled Hadamard starts from a stronger backprojection, but final quality is close to Rademacher at the same sampling ratio. Final PSNR alone therefore hides measurement-family behavior.

\input{{tables/table2_measurement_attribution.tex}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig2_measurement_attribution.pdf}}
\caption{{\textbf{{Measurement-family attribution.}} Pattern families differ in backprojection quality and neural refinement gain. Final PSNR alone hides these regimes.}}
\label{{fig:measurement_attribution}}
\end{{figure*}}

A careful phrasing is important: STL-10 5\% high-quality reconstruction is supported under suitable measurement families. The earlier low-frequency Hadamard 5\% diagnostic should not be generalized into a failure of all STL-10 5\% measurement designs.
""",
        "ablation_validation": rf"""
Exact-A reproducibility is a key validation step. The Rademacher re-evaluation reproduces the random-operator checkpoints when the exact exported \(A\) is loaded and \(K=AA^T+\lambda I\) is rebuilt before solving. This avoids random-matrix mismatch and stale-cache errors. In the exact-A audit, Rad-5 reports an absolute PSNR difference of {fmt(exact['rademacher5_hq_noise001_colab'].get('abs_diff_psnr'))} dB and Rad-10 reports {fmt(exact['rademacher10_full_noise001_colab'].get('abs_diff_psnr'))} dB.

Inference ablation is summarized in \cref{{tab:ablation_summary,fig:inference_ablation}}. Removing the measurement-consistency projection causes the largest degradation, especially for scrambled Hadamard where removing the DC projection collapses PSNR. Removing the null projection has limited metric effect for these checkpoints. The honest interpretation is that the final measurement projection and the trained network may already constrain measured components; the null projection remains part of the designed pipeline, but inference-time removal has limited effect in these trained models. Stage-1-only reconstruction remains below the full model, and EMA weights are slightly more stable than raw weights.

\input{{tables/table3_inference_ablation_summary.tex}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig4_inference_ablation.pdf}}
\caption{{\textbf{{Inference-time ablation.}} Removing measurement-consistency projection produces the largest degradation, while no-null removal has limited metric effect for these checkpoints.}}
\label{{fig:inference_ablation}}
\end{{figure*}}

\Cref{{fig:robustness_baselines}} collects finite noise, measurement perturbation, CS-TV, and bootstrap confidence results. The noise sweep supports robustness only over the tested finite noise range. Shuffled coefficients and wrong-sample measurements produce large drops, indicating dependence on \(y\) rather than a generic image prior. The CS-TV baseline solves
\begin{{equation}}
\min_x \frac{{1}}{{2}}\|Ax-y\|_2^2+\lambda\,\mathrm{{TV}}(x)
\end{{equation}}
by projected gradient descent. We therefore describe it as a TV-regularized compressed-sensing baseline solved by PGD, abbreviated CS-TV (PGD solver). It is a lightweight small-subset baseline, not an exhaustively optimized ADMM/FISTA or plug-and-play comparison.

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/fig5_robustness_baselines.pdf}}
\caption{{\textbf{{Robustness and baselines.}} Finite noise, measurement perturbation, CS-TV, and bootstrap diagnostics test measurement dependence and provide bounded robustness evidence.}}
\label{{fig:robustness_baselines}}
\end{{figure*}}

The DC-row control applies only to low-frequency Hadamard and explains the importance of global brightness. Bootstrap intervals and class-wise STL-10 diagnostics provide uncertainty and heterogeneity checks, but class-wise trends are diagnostic rather than a central claim.
""",
        "discussion": r"""
The first lesson is that measurement family and neural refinement are complementary. Rademacher measurements give weak physical backprojections but large learned gains. Scrambled Hadamard measurements give stronger physical initializations and similar final results. Low-frequency Hadamard gives interpretable physical structure, but it is not always the strongest final measurement family.

The second lesson is that measurement consistency matters. The no-DC projection ablation, relative measurement error, exact-A handling, and perturbation tests all point to a model that depends on \(y\). This matters because visual quality alone is insufficient for an optical inverse problem.

The third lesson is about positioning. The method should not be presented as a GAN paper. The final mechanism is measurement-consistent neural reconstruction. The general idea is the formulation and projection mechanism; the present evidence is limited to the tested datasets, simulated measurement model, finite noise range, and lightweight CS-TV comparison.
""",
        "limitations": r"""
This study has several limitations. It does not include a hardware optical experiment. It does not make a strict leaderboard claim. The CS-TV (PGD solver) baseline is lightweight and small-subset, not an exhaustive ADMM/FISTA or plug-and-play benchmark. Noise robustness is demonstrated only over the finite tested range. Class-wise results are diagnostic only. Low-frequency Hadamard 5\% is not a primary STL-10 high-quality result. Binary learned illumination is not claimed as a successful final result. GAN components are not the final main mechanism. Exact-A handling is essential for random measurements, and any future random-measurement evaluation must reload the exact operator and rebuild the solver cache.
""",
        "conclusion": r"""
Measurement-consistent null-space neural reconstruction provides a compact and auditable route to high-quality low-sampling ghost imaging. The results support STL-10 reconstruction at 5\% and 10\% sampling under Rademacher and scrambled Hadamard measurements, and they show stable simple-domain sanity performance on MNIST and Fashion-MNIST at 5\%. The validation package clarifies why the model is measurement-dependent, why exact-A handling matters, and why measurement-family attribution is necessary for interpreting low-sampling GI results.
""",
    }


def markdown_document() -> str:
    sec = md_sections()
    table1 = markdown_table(
        main_results_rows(),
        ["dataset", "sampling", "measurement", "psnr", "ssim", "bp_psnr", "delta_psnr", "hq"],
    )
    lines = [
        f"# {TITLE}",
        "",
        "## Abstract",
        sec["abstract"],
        "",
        "## 1 Introduction",
        sec["introduction"],
        "",
        "## 2 Related Work",
        sec["related_work"],
        "",
        "## 3 Problem Formulation",
        sec["problem_formulation"],
        "",
        "## 4 Method",
        sec["method"],
        "",
        "## 5 Measurement Families",
        sec["measurement_families"],
        "",
        "## 6 Training Losses and Experimental Protocol",
        sec["training_losses"],
        "",
        sec["experimental_protocol"],
        "",
        "## 7 Results",
        sec["results"],
        "",
        "### Main result table snapshot",
        table1,
        "",
        "## 8 Ablation and Validation",
        sec["ablation_validation"],
        "",
        "## 9 Discussion",
        sec["discussion"],
        "",
        "## 10 Limitations",
        sec["limitations"],
        "",
        "## 11 Conclusion",
        sec["conclusion"],
        "",
        "## Figure and Table Map",
        "",
    ]
    for label, stem, title in MAIN_FIGURES:
        lines.append(f"- {label}: {stem} ({title})")
    for label, stem, title in MAIN_TABLES:
        lines.append(f"- {label}: {stem} ({title})")
    return "\n".join(lines)


def standalone_tex() -> str:
    sec = tex_sections()
    section_order = [
        ("Introduction", "introduction"),
        ("Related Work", "related_work"),
        ("Problem Formulation", "problem_formulation"),
        ("Method", "method"),
        ("Measurement Families", "measurement_families"),
        ("Training Losses", "training_losses"),
        ("Experimental Protocol", "experimental_protocol"),
        ("Results", "results"),
        ("Ablation and Validation", "ablation_validation"),
        ("Discussion", "discussion"),
        ("Limitations", "limitations"),
        ("Conclusion", "conclusion"),
    ]
    body = "\n\n".join([rf"\section{{{title}}}" + "\n" + sec[key] for title, key in section_order])
    return rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{subcaption}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{siunitx}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{cleveref}}
\title{{{tex_escape_text(TITLE)}}}
\author{{Anonymous authors}}
\date{{}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{sec['abstract']}
\end{{abstract}}

{body}

\end{{document}}
"""


def rewrite_summary() -> str:
    lines = [
        "# Phase18 Rewrite Summary",
        "",
        f"Output directory: `{OUT}`",
        "",
        "Generated artifacts:",
        "- `manuscript_v3.md`: full narrative manuscript draft.",
        "- `manuscript_v3.tex`: standalone LaTeX draft using generated figures and tables.",
        "- `latex_project/`: structured LaTeX project generated by `phase18_build_latex_project`.",
        "- `figures/`: regenerated main and supplement figures.",
        "- `tables/`: regenerated main and supplement tables.",
        "",
        "Claim handling:",
        "- No new training or experiments are introduced.",
        "- Primary numbers are read from the no-leak registry and supplementary CSV files.",
        "- Main text avoids internal stage names.",
        "- The method is framed as measurement-consistent null-space neural reconstruction, not as a GAN contribution.",
        "- Low-frequency Hadamard 5% is not framed as a primary STL-10 high-quality result.",
        "- CS-TV is described as a TV-regularized compressed-sensing baseline solved by projected gradient descent.",
        "",
        "Manual follow-up before submission:",
        "- Replace TODO-VERIFY bibliography placeholders with verified references.",
        "- Inspect final figure aesthetics against the target venue template.",
        "- Decide whether to add hardware data or stronger optimized CS baselines for a higher-stakes submission.",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_text(OUT / "manuscript_v3.md", markdown_document())
    write_text(OUT / "manuscript_v3.tex", standalone_tex())
    write_text(OUT / "REWRITE_SUMMARY.md", rewrite_summary())
    write_json(OUT / "source_manifest.json", source_manifest())
    write_json(
        OUT / "main_result_snapshot.json",
        {
            mid: {
                "label": METHOD_LABEL.get(mid, mid),
                "long_label": LONG_LABEL.get(mid, mid),
                "sampling": sampling(mid),
                "psnr": f(mid, "psnr"),
                "ssim": f(mid, "ssim"),
                "backproj_psnr": f(mid, "backproj_psnr"),
                "delta_psnr": f(mid, "delta_psnr"),
            }
            for mid in registry_by_id()
        },
    )
    print({"manuscript_md": str(OUT / "manuscript_v3.md"), "manuscript_tex": str(OUT / "manuscript_v3.tex")})


if __name__ == "__main__":
    main()
