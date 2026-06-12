from __future__ import annotations

from .phase17_common import PHASE17, TITLE, CORE_CLAIM, main_result_rows, markdown_table, write_text


OUT = PHASE17 / "defense"

SLIDES = [
    ("Title", TITLE),
    ("Problem: low-sampling ghost imaging", "Recover images from bucket measurements at 5% and 10% sampling."),
    ("Forward model", "y = Ax + epsilon; A is the measurement family."),
    ("Why low sampling is underdetermined", "Backprojection loses information and unconstrained networks can hallucinate."),
    ("Measurement-consistent null-space reconstruction", "Data solution + null-space neural correction + final consistency projection."),
    ("Measurement families", "Rademacher, scrambled Hadamard, and low-frequency Hadamard diagnostics."),
    ("HQ reconstructor", "Imported strict no-leak checkpoints inside the physical reconstruction pipeline."),
    ("Exact-A reproducibility", "Rademacher uses exported exact A with cache-rebuilt solver."),
    ("Main STL-10 5%/10% results", "Rademacher and scrambled Hadamard meet operational HQ thresholds."),
    ("MNIST/Fashion sanity", "5% simple-domain sanity results support pipeline correctness."),
    ("Attribution", "Separate backprojection strength from learned refinement."),
    ("Inference ablation", "Data consistency and refiner contributions."),
    ("Noise and perturbation robustness", "Finite noise and measurement-dependence diagnostics."),
    ("Baselines", "Backprojection, adjoint, and small-subset TV-PGD."),
    ("Limitations", "No SOTA claim; no universal robustness; simulation-centered unless hardware is added."),
    ("Conclusion", CORE_CLAIM),
]


def main() -> None:
    outline = ["# Defense slides outline", ""]
    notes = ["# Defense speaker notes", ""]
    for idx, (title, body) in enumerate(SLIDES, 1):
        outline.extend([f"## Slide {idx}. {title}", "", body, ""])
        notes.extend([f"## Slide {idx}. {title}", "", f"Say: {body}", ""])
        if title.startswith("Main STL"):
            notes.append(markdown_table([r for r in main_result_rows() if r["dataset"] == "STL-10"], ["method", "sampling", "psnr", "ssim", "bp_psnr", "delta_psnr"]))
            notes.append("")
    qa = f"""# Defense Q&A

## What is the one-sentence contribution?

{CORE_CLAIM}

## Is this just another deep ghost imaging network?

No. Deep GI/SPI networks exist. The contribution is the measurement-consistent null-space reconstruction framing plus explicit validation: exact-A reproducibility, attribution, ablation, perturbation, confidence intervals, and runtime.

## Does the model hallucinate?

The design enforces measurement consistency, and perturbation controls show that corrupting or mismatching the measurement vector reduces reconstruction quality. This is evidence against pure generic image generation, though not a complete philosophical proof.

## Why Rademacher if backprojection is poor?

Rademacher has weak direct backprojection but reaches similar final quality after learned refinement. That contrast is useful evidence that the learned null-space correction is doing nontrivial work.

## Can you claim SOTA?

No. The safe claim is high-quality strict no-leak reconstruction under the tested sensing setups.

## Is TV-PGD fully optimized?

No. TV-PGD is reported as a small-subset lightweight reviewer-defense baseline.

## Can this go to hardware?

The formulation is compatible with hardware GI/SPI, but the current package should not imply a hardware experiment unless one is added.
"""
    write_text(OUT / "defense_slides_outline.md", "\n".join(outline))
    write_text(OUT / "defense_speaker_notes.md", "\n".join(notes))
    write_text(OUT / "defense_qa.md", qa)
    print({"output": str(OUT), "slides": len(SLIDES)})


if __name__ == "__main__":
    main()
