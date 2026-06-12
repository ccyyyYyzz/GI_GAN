from __future__ import annotations

from .phase17_common import (
    DO_NOT_CLAIM,
    PHASE16_TABLES,
    PHASE17,
    TITLE,
    CORE_CLAIM,
    latex_table,
    main_result_rows,
    markdown_table,
    simple_main_rows,
    stl_main_rows,
    tex_escape,
    write_text,
)


OUT = PHASE17 / "manuscript"


def result_sentence() -> str:
    rows = {row["method"]: row for row in main_result_rows()}
    return (
        "In strict no-leak evaluation, STL-10 reaches PSNR/SSIM of "
        f"{rows['STL-10 Rademacher 5%']['psnr']}/{rows['STL-10 Rademacher 5%']['ssim']} with Rademacher 5%, "
        f"{rows['STL-10 scrambled Hadamard 5%']['psnr']}/{rows['STL-10 scrambled Hadamard 5%']['ssim']} with scrambled Hadamard 5%, "
        f"{rows['STL-10 Rademacher 10%']['psnr']}/{rows['STL-10 Rademacher 10%']['ssim']} with Rademacher 10%, and "
        f"{rows['STL-10 scrambled Hadamard 10%']['psnr']}/{rows['STL-10 scrambled Hadamard 10%']['ssim']} with scrambled Hadamard 10%. "
        f"MNIST and Fashion-MNIST at 5% reach {rows['MNIST Hadamard 5%']['psnr']}/{rows['MNIST Hadamard 5%']['ssim']} and "
        f"{rows['Fashion-MNIST Hadamard 5%']['psnr']}/{rows['Fashion-MNIST Hadamard 5%']['ssim']}, respectively."
    )


def abstract() -> str:
    return (
        "Ghost imaging and single-pixel imaging recover an image from bucket measurements rather than a pixel array. "
        "At low sampling rates, the inverse problem is strongly underdetermined, and unconstrained neural reconstruction can improve perceptual quality while risking measurement-inconsistent hallucination. "
        "We study a measurement-consistent null-space neural reconstruction framework for low-sampling ghost imaging. "
        "The reconstruction is built from an explicit data-consistent backprojection, a learned null-space correction, and a final measurement-consistency projection. "
        f"{result_sentence()} "
        "A supplementary audit verifies strict no-leak evaluation, exact exported measurement operators for Rademacher sensing, measurement-family attribution, inference-time ablations, finite noise diagnostics, traditional baselines, per-sample confidence intervals, and runtime measurements. "
        "We do not claim strict state-of-the-art performance; instead, the contribution is a physically constrained and auditable reconstruction framework for high-quality low-sampling GI/SPI under the reported protocols."
    )


def md() -> str:
    rows = main_result_rows()
    stl_rows = stl_main_rows()
    simple_rows = simple_main_rows()
    do_not = "\n".join(f"- {item}" for item in DO_NOT_CLAIM)
    equations = r"""
The bucket measurement model is

$$
y = A x + \epsilon,
$$

where \(x\) is the vectorized image, \(A\) is the sensing matrix, and \(\epsilon\) is measurement noise. We use a ridge-stabilized data solution

$$
x_{\mathrm{data}} = A^T(AA^T + \lambda I)^{-1}y.
$$

The null-space projection is

$$
P_N(v) = v - A^T(AA^T + \lambda I)^{-1} A v.
$$

The learned correction is inserted into the measurement null space:

$$
\tilde{x} = x_{\mathrm{data}} + P_N(G_\theta(x_{\mathrm{data}}, z)).
$$

The final measurement-consistency projection is

$$
\hat{x} = \Pi_y(\tilde{x}) =
\tilde{x} - A^T(AA^T + \lambda I)^{-1}(A\tilde{x} - y).
$$

For Hadamard sensing, \(H_{\mathrm{norm}} = H/\sqrt{n}\), \(A = H_{\mathrm{norm}}[\mathrm{selected\ rows},:]\), the measured coefficients are inserted into \(c[\mathrm{selected\ rows}]=y\), and \(x_{\mathrm{data}}=H_{\mathrm{norm}}^Tc\) when the orthogonal coefficient view is used. For Rademacher sensing, \(A\) is a signed random measurement operator and is evaluated with the exported exact \(A\) and a cache-rebuilt solver.
"""
    return f"""# {TITLE}

## Abstract

{abstract()}

## 1. Introduction

Ghost imaging (GI) and single-pixel imaging (SPI) replace a focal-plane sensor with a sequence of structured illuminations and bucket measurements. This changes image formation from direct pixel sampling into a linear inverse problem. Low sampling is attractive because it reduces acquisition cost, but it also makes the inverse problem highly underdetermined. A neural network can fill in missing structure, yet an unconstrained network can also hallucinate content that is visually plausible but inconsistent with the measured buckets.

This work focuses on that constraint gap. The central idea is not to treat the network as a standalone image generator, but to place it inside a reconstruction pipeline that exposes the forward operator and enforces measurement consistency. We evaluate several measurement families because the physical initialization and the learned correction behave differently under low-frequency Hadamard, scrambled Hadamard, and Rademacher measurements.

The contributions are:

- a measurement-consistent null-space neural reconstruction formulation for low-sampling GI/SPI;
- strict no-leak evaluation on STL-10, MNIST, and Fashion-MNIST using imported final checkpoints;
- exact-A Rademacher reproducibility with a cache-rebuilt solver path;
- attribution analyses that separate backprojection strength from learned refinement;
- reviewer-defense diagnostics covering ablations, finite noise robustness, TV-PGD controls, confidence intervals, class-wise behavior, measurement perturbations, and runtime.

## 2. Related Work

Deep learning has already been applied to GI/SPI, including convolutional networks, generative models, and conditional reconstruction networks. Data consistency and null-space correction are also established ideas in inverse problems such as compressed sensing and computational imaging. Therefore, this paper should not be framed as the first deep-learning GI method or the first null-space network. To the best of our knowledge, the useful distinction here is the combination of explicit measurement consistency, null-space correction, cross-measurement-family attribution, exact-A Rademacher reproducibility, and a compact reviewer-defense audit in a low-sampling GI/SPI setting.

Representative citations are listed in `references_to_verify.bib` and must be manually verified before submission.

## 3. Forward Model and Measurement Families

{equations}

We evaluate three sensing families. Low-frequency Hadamard is useful for simple-domain sanity checks and DC-row diagnostics, but low-frequency Hadamard 5% should not be described as high-quality on STL-10. Scrambled Hadamard retains structured orthogonality while distributing information more broadly. Rademacher uses signed random measurements; because its exact exported operator matters, all Rademacher claims are tied to the safe exact-A cache-rebuilt evaluation path.

## 4. Measurement-Consistent Null-Space Reconstruction

The reconstruction starts from the data solution, adds a learned correction projected away from the measured row space, and then applies a final data-consistency projection. This design makes the measurement operator visible at inference time. It also creates testable failure modes: removing the final consistency projection, replacing the measurement vector, or shuffling measurement coefficients should measurably affect the result.

## 5. HQ Reconstructor and Training Losses

The checkpointed reconstructor is used here as an imported strict no-leak model. Phase17 does not train a new model. The manuscript should describe the HQ reconstructor as the practical neural component inside the measurement-consistent pipeline, not as evidence that a GAN mechanism is the final main contribution. Loss terms should be described in relation to reconstruction quality and measurement consistency; any adversarial or generator terminology should be kept secondary unless directly supported by the final code and ablation.

## 6. Experimental Protocol

All main numbers in this draft are read from Phase15/Phase16 artifacts. The no-leak registry is the source of primary PSNR/SSIM numbers. Phase16 provides exact-A Rademacher re-evaluation, attribution, inference ablation, finite noise sweep, traditional baselines, DC-row control, statistics and confidence intervals, class-wise diagnostics, measurement perturbation, and runtime.

For Rademacher sensing, the exported exact measurement operator is loaded and the solver cache is rebuilt before evaluation. The pre-fix local mismatch is excluded from all claims. Colab-imported final checkpoints are identified as imported no-leak results rather than local training results.

## 7. Main Results

{result_sentence()}

### Main strict no-leak results

{markdown_table(rows, ["method", "dataset", "sampling", "family", "psnr", "ssim", "bp_psnr", "delta_psnr"])}

### STL-10 primary rows

{markdown_table(stl_rows, ["method", "sampling", "family", "psnr", "ssim", "bp_psnr", "delta_psnr"])}

### Simple-domain sanity rows

{markdown_table(simple_rows, ["method", "sampling", "family", "psnr", "ssim", "bp_psnr", "delta_psnr"])}

The STL-10 5% and 10% rows support high-quality reconstruction under the operational thresholds used in this project. The MNIST/Fashion-MNIST rows support simple-domain sanity claims. None of these rows should be described as a strict SOTA result.

## 8. Ablations and Reviewer-Defense Analyses

The Phase16 evidence index supports the following interpretation. Exact-A Rademacher re-evaluation reproduced the imported results after the cache-rebuilt solver path. Attribution shows that model refinement improves weak backprojections. Removing data consistency causes strong degradation in the no-DC rows, while the stage1-only rows are below the full refiner path. EMA weights are slightly better than raw weights in the tested rows. The finite noise sweep supports robustness only over the tested noise levels. TV-PGD is included as a small-subset lightweight baseline, not as an exhaustively optimized traditional method. DC-row controls explain why low-frequency Hadamard backprojection is sensitive to DC handling. Bootstrap confidence intervals, class-wise diagnostics, measurement perturbations, and runtime tables are available in the supplement.

## 9. Discussion

The results suggest that measurement family and physical initialization matter. Scrambled Hadamard starts from a stronger backprojection, while Rademacher starts from a weak physical inverse but reaches similar final quality after the learned measurement-consistent correction. This argues for reporting the sensing family, backprojection quality, final quality, and measurement consistency together instead of presenting a single PSNR number.

## 10. Limitations

{do_not}

Additional limitations: the current package is simulation-centered unless a hardware experiment is added later; robustness is tested only over finite noise and perturbation controls; class-wise results are diagnostic; TV-PGD is a lightweight small-subset control; related work citations still need manual verification before submission.

## 11. Conclusion

{CORE_CLAIM} The Phase15/Phase16 evidence package supports the main claims under strict no-leak evaluation and provides enough reviewer-defense material to stop running broad new experiments and begin manual manuscript refinement.
"""


def tex_from_md() -> str:
    rows = main_result_rows()
    return rf"""\documentclass[11pt]{{article}}
\usepackage{{amsmath,amssymb,booktabs,geometry}}
\geometry{{margin=1in}}
\title{{{tex_escape(TITLE)}}}
\author{{Author names to be added}}
\date{{Draft generated from Phase15/Phase16 evidence}}
\begin{{document}}
\maketitle

\begin{{abstract}}
{tex_escape(abstract())}
\end{{abstract}}

\section{{Introduction}}
Ghost imaging and single-pixel imaging recover an image from bucket measurements. At low sampling rates, the inverse problem is underdetermined, and unconstrained neural networks can hallucinate measurement-inconsistent structure. We propose a measurement-consistent null-space neural reconstruction framework and evaluate it under strict no-leak protocols.

\section{{Forward Model and Measurement Families}}
The bucket model is
\[
y = A x + \epsilon.
\]
The ridge-stabilized data solution is
\[
x_{{\mathrm{{data}}}} = A^T(AA^T + \lambda I)^{{-1}}y.
\]
The null-space projection is
\[
P_N(v) = v - A^T(AA^T + \lambda I)^{{-1}} A v.
\]
The neural correction is inserted as
\[
\tilde{{x}} = x_{{\mathrm{{data}}}} + P_N(G_\theta(x_{{\mathrm{{data}}}}, z)),
\]
and the final measurement-consistent estimate is
\[
\hat{{x}} = \Pi_y(\tilde{{x}}) = \tilde{{x}} - A^T(AA^T + \lambda I)^{{-1}}(A\tilde{{x}} - y).
\]
For Hadamard sensing, \(H_{{\mathrm{{norm}}}}=H/\sqrt{{n}}\), \(A=H_{{\mathrm{{norm}}}}[\mathrm{{selected\ rows}},:]\), \(c[\mathrm{{selected\ rows}}]=y\), and \(x_{{\mathrm{{data}}}}=H_{{\mathrm{{norm}}}}^Tc\). For Rademacher sensing, \(A\) is a signed random measurement operator evaluated with the exported exact \(A\) and a cache-rebuilt solver.

\section{{Experimental Protocol}}
All main numbers are imported strict no-leak results from Phase15 and Phase16. No new training is performed in Phase17.

\section{{Main Results}}
{tex_escape(result_sentence())}

{latex_table(rows, ["method", "dataset", "sampling", "family", "psnr", "ssim", "bp_psnr", "delta_psnr"], "Primary strict no-leak reconstruction results.", "tab:main_results")}

\section{{Ablations and Supplementary Analyses}}
Phase16 includes exact-A reproducibility, measurement-family attribution, inference ablations, finite noise diagnostics, traditional baselines, DC-row controls, bootstrap confidence intervals, class-wise diagnostics, measurement perturbations, and runtime tables.

\section{{Limitations}}
We do not claim strict SOTA, universal robustness, high-quality low-frequency Hadamard 5\% on STL-10, binary learned illumination as the main contribution, GAN as the final main mechanism, or exhaustively optimized TV-PGD.

\section{{Conclusion}}
{tex_escape(CORE_CLAIM)}

\bibliographystyle{{plain}}
\bibliography{{references_to_verify}}
\end{{document}}
"""


def references() -> str:
    return r"""% All entries below are placeholders to verify manually before submission.
@article{deep_gi_review_to_verify,
  title = {Deep learning for ghost imaging and single-pixel imaging: representative review to verify},
  author = {To Verify},
  journal = {To Verify},
  year = {To Verify},
  note = {Replace with verified GI/SPI deep-learning review citation}
}

@article{gan_gi_to_verify,
  title = {Generative or conditional adversarial ghost imaging method to verify},
  author = {To Verify},
  journal = {To Verify},
  year = {To Verify},
  note = {Replace with representative GAN/cGAN GI citation}
}

@article{nullspace_inverse_to_verify,
  title = {Null-space or data-consistent neural inverse problem method to verify},
  author = {To Verify},
  journal = {To Verify},
  year = {To Verify},
  note = {Replace with representative null-space/data-consistency inverse-problem citation}
}

@article{single_pixel_imaging_to_verify,
  title = {Single-pixel imaging and compressive ghost imaging reference to verify},
  author = {To Verify},
  journal = {To Verify},
  year = {To Verify},
  note = {Replace with foundational SPI/GI citation}
}
"""


def citation_notes() -> str:
    return f"""# Citations to verify

The manuscript intentionally does not pretend that citations are verified. Before submission, manually replace the placeholder BibTeX entries in `references_to_verify.bib`.

Required citation groups:

1. Deep learning GI/SPI survey or representative methods.
2. GAN or conditional GAN ghost imaging / single-pixel reconstruction.
3. Data-consistent neural inverse problems.
4. Null-space networks or null-space correction in inverse problems.
5. Classical ghost imaging and single-pixel imaging foundations.
6. TV-regularized inverse-problem baselines, if TV-PGD is discussed in the main paper.

Evidence files used for numerical claims:

- `{PHASE16_TABLES['exact_a_reeval']}`
- `{PHASE16_TABLES['attribution']}`
- `{PHASE16_TABLES['ablation']}`
- `{PHASE16_TABLES['noise']}`
- `{PHASE16_TABLES['traditional_baselines']}`
"""


def main() -> None:
    write_text(OUT / "manuscript_draft.md", md())
    write_text(OUT / "manuscript_draft.tex", tex_from_md())
    write_text(OUT / "references_to_verify.bib", references())
    write_text(OUT / "citations_to_verify.md", citation_notes())
    print({"output": str(OUT), "files": 4})


if __name__ == "__main__":
    main()
