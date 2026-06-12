from __future__ import annotations

from .phase19_common import METHOD_LABEL, OUT, fmt, registry_by_id, write_text


def r(mid: str) -> dict[str, str]:
    return registry_by_id()[mid]


def metric_sentence() -> str:
    return (
        f"On STL-10, the method reaches {fmt(r('rademacher5_hq_noise001_colab')['psnr'])} dB / "
        f"{fmt(r('rademacher5_hq_noise001_colab')['ssim'])} SSIM at 5% with Rademacher measurements and "
        f"{fmt(r('scrambled_hadamard5_hq_noise001_colab')['psnr'])} dB / "
        f"{fmt(r('scrambled_hadamard5_hq_noise001_colab')['ssim'])} SSIM at 5% with scrambled Hadamard measurements. "
        f"At 10%, Rademacher and scrambled Hadamard reach {fmt(r('rademacher10_full_noise001_colab')['psnr'])} dB / "
        f"{fmt(r('rademacher10_full_noise001_colab')['ssim'])} SSIM and "
        f"{fmt(r('scrambled_hadamard10_full_noise001_colab')['psnr'])} dB / "
        f"{fmt(r('scrambled_hadamard10_full_noise001_colab')['ssim'])} SSIM, respectively. "
        f"MNIST and Fashion-MNIST sanity checks at 5% reach {fmt(r('mnist_hadamard5_full_colab')['psnr'])} dB / "
        f"{fmt(r('mnist_hadamard5_full_colab')['ssim'])} SSIM and "
        f"{fmt(r('fashion_hadamard5_full_colab')['psnr'])} dB / {fmt(r('fashion_hadamard5_full_colab')['ssim'])} SSIM."
    )


def narrative_outline() -> str:
    return """# Narrative Outline

## Main Thesis

Low-sampling ghost imaging is a constrained completion problem. The method separates measured row-space information from unmeasured null-space content, refines the missing structure with a neural prior, and restores measurement consistency.

## Core Logic Chain

1. GI measurement gives `y = A x + epsilon`.
2. Low sampling means `m << n`, so many images match the same `y`.
3. Backprojection gives a physical but incomplete `x_data`.
4. A neural residual estimates missing structure rather than replacing the measured signal wholesale.
5. Null-space insertion prevents arbitrary overwrite of measured components.
6. Measurement-consistency projection ties the output back to `y`.
7. Measurement family determines initial difficulty and neural refinement gain.
8. Results validate high-quality 5%/10% STL-10 under Rademacher and scrambled Hadamard measurements.
9. Ablations and perturbations show physical dependence.

## Recommended Paper Flow

Problem -> measurement-consistent mechanism -> measurement families -> primary results -> qualitative evidence -> attribution -> validation.

## Innovation Claims To Keep

- Measurement-consistent null-space formulation for low-sampling GI.
- Exact-A strict no-leak evaluation for random measurements.
- High-quality STL-10 5%/10% under suitable measurement families.
- Attribution, ablation, perturbation, finite-noise, CS-TV, and uncertainty validation package.

## Claims To Avoid

- No strict SOTA claim.
- No claim that GAN is the final main mechanism.
- No claim that binary learned illumination is validated.
- No claim that low-frequency Hadamard 5% is a high-quality STL-10 result.
"""


def introduction_md() -> str:
    return f"""# Introduction Rewrite

Ghost imaging and single-pixel imaging recover spatial structure from known illumination patterns and scalar bucket measurements. Instead of directly recording a dense image, the system measures inner products between the scene and a sequence of patterns.

The challenge is low sampling. When the number of bucket measurements is much smaller than the number of image pixels, the inverse problem is underdetermined: many candidate images are compatible with the same measurement vector.

Deep reconstruction networks can improve visual quality in this regime, but an unconstrained network can also hallucinate plausible structure that is not tied to the physical measurements. For computational imaging, the output should remain consistent with the measured bucket signal, not merely look natural.

Existing deep-learning GI methods often improve images without explicitly decomposing measured row-space content from unmeasured null-space content, and without separating the effect of measurement family from neural refinement. This makes it difficult to know whether performance comes from a better physical initialization, a stronger learned inverse, or both.

We address this gap with measurement-consistent null-space neural reconstruction. The method starts from a physical data solution, inserts a learned residual through an approximate null-space projection, and applies a final measurement-consistency projection.

{metric_sentence()}

Contributions:

- A measurement-consistent null-space formulation for low-sampling ghost imaging.
- Exact-A strict no-leak evaluation for random Rademacher measurements.
- High-quality STL-10 reconstruction at 5% and 10% sampling under Rademacher and scrambled Hadamard measurements.
- An attribution, ablation, robustness, and compressed-sensing validation package that tests physical dependence.
"""


def introduction_tex() -> str:
    metrics = metric_sentence().replace("%", r"\%")
    return rf"""\section{{Introduction}}
Ghost imaging and single-pixel imaging recover spatial structure from known illumination patterns and scalar bucket measurements. Instead of directly recording a dense image, the system measures inner products between the scene and a sequence of patterns.

The challenge is low sampling. When the number of bucket measurements is much smaller than the number of image pixels, the inverse problem is underdetermined: many candidate images are compatible with the same measurement vector.

Deep reconstruction networks can improve visual quality in this regime, but an unconstrained network can also hallucinate plausible structure that is not tied to the physical measurements. For computational imaging, the output should remain consistent with the measured bucket signal, not merely look natural.

Existing deep-learning GI methods often improve images without explicitly decomposing measured row-space content from unmeasured null-space content, and without separating the effect of measurement family from neural refinement. This makes it difficult to know whether performance comes from a better physical initialization, a stronger learned inverse, or both.

We address this gap with measurement-consistent null-space neural reconstruction. The method starts from a physical data solution, inserts a learned residual through an approximate null-space projection, and applies a final measurement-consistency projection.

{metrics}

Our contributions are: (i) a measurement-consistent null-space formulation for low-sampling ghost imaging; (ii) exact-A strict no-leak evaluation for random Rademacher measurements; (iii) high-quality STL-10 reconstruction at 5\% and 10\% sampling under Rademacher and scrambled Hadamard measurements; and (iv) an attribution, ablation, robustness, and compressed-sensing validation package that tests physical dependence.
"""


def manuscript_md() -> str:
    return f"""# {OUT.name}: Manuscript V5 Logic Draft

## Thesis

Low-sampling GI is a measurement-constrained completion problem. The proposed computation is centered on a data solution, null-space neural residual, and measurement-consistency projection.

## Main Result Snapshot

{metric_sentence()}

## Figure Logic

- Figure 0: graphical abstract of the constrained-completion story.
- Figure 1: mechanism diagram with row-space preservation, null-space completion, and measurement checking.
- Figure 2: primary metrics and engineering thresholds.
- Figure 3: large qualitative STL-10 reconstructions.
- Figure 4: measurement regime map using BP PSNR vs neural gain.
- Figure 5: inference ablation.
- Figure 6: validation summary: noise, projection, perturbation, CS-TV.

## Validation Logic

Exact-A reproducibility protects random-measurement evaluation. No-DC projection removal is the strongest measurement-consistency ablation. No-null removal has limited metric effect for these checkpoints and is reported honestly. Perturbations show dependence on the measurement vector. CS-TV is a TV-regularized compressed-sensing baseline solved by PGD on a small subset.
"""


def citations_to_verify() -> str:
    return """# Citations To Verify

Replace TODO_VERIFY entries in `references.bib` before submission.

Suggested citation groups:

- Ghost imaging and single-pixel imaging foundations.
- Deep learning for ghost imaging / single-pixel reconstruction.
- Data consistency and null-space learning in inverse problems.
- Total-variation compressed sensing and projected-gradient solvers.

No unverified citation is cited in the current compiled main text.
"""


def main() -> None:
    write_text(OUT / "NARRATIVE_OUTLINE.md", narrative_outline())
    write_text(OUT / "sections" / "introduction.tex", introduction_tex())
    write_text(OUT / "introduction_rewrite.md", introduction_md())
    write_text(OUT / "manuscript_v5.md", manuscript_md())
    write_text(OUT / "citations_to_verify.md", citations_to_verify())
    print({"outline": str(OUT / "NARRATIVE_OUTLINE.md"), "introduction": str(OUT / "sections" / "introduction.tex")})


if __name__ == "__main__":
    main()
