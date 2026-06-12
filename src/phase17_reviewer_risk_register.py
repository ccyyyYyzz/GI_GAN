from __future__ import annotations

from .phase17_common import PHASE16_TABLES, PHASE17, REGISTRY, markdown_table, write_text


OUT = PHASE17 / "reviewer_risk_register"

QUESTIONS = [
    ("Is this just another deep ghost imaging network?", "Acknowledge prior DL-GI/SPI. Emphasize measurement-consistent null-space formulation and validation package.", REGISTRY),
    ("Where is the physics?", "The forward operator A appears in the data solution, null-space projection, and final data-consistency projection.", PHASE16_TABLES["ablation"]),
    ("Does it hallucinate?", "Measurement consistency and perturbation controls reduce the risk; do not claim hallucination is impossible.", PHASE16_TABLES["perturbation"]),
    ("Does measurement consistency really matter?", "No-DC ablation rows degrade reconstruction quality and consistency.", PHASE16_TABLES["ablation"]),
    ("Why Rademacher if backprojection is poor?", "Rademacher tests a difficult random sensing regime; learned refinement recovers strong final quality from weak backprojection.", PHASE16_TABLES["attribution"]),
    ("Why does scrambled Hadamard start better but end similar?", "Scrambled Hadamard has stronger physical initialization; the learned model narrows the final-quality gap.", PHASE16_TABLES["attribution"]),
    ("Why is lowfreq Hadamard 5% not HQ?", "Low-frequency Hadamard 5% is diagnostic/auxiliary on STL-10 and should not be used as a main HQ claim.", PHASE16_TABLES["attribution"]),
    ("Is exact A reproducible?", "Yes for the safe cache-rebuilt path; pre-fix mismatch is excluded.", PHASE16_TABLES["exact_a_reeval"]),
    ("Is there data leakage?", "Main rows are strict no-leak imported results from Phase15 registry.", REGISTRY),
    ("Are baselines too weak?", "Linear baselines and small-subset TV-PGD are included; state TV-PGD limits honestly.", PHASE16_TABLES["traditional_baselines"]),
    ("Is TV-PGD optimized?", "No. It is a small-subset lightweight control, not an exhaustive optimized baseline.", PHASE16_TABLES["traditional_baselines"]),
    ("Is the method robust?", "Only over finite tested noise levels and diagnostic perturbations.", PHASE16_TABLES["noise"]),
    ("Is GAN actually used?", "The final claim should be the measurement-consistent reconstructor, not GAN as the main mechanism.", REGISTRY),
    ("Can you claim SOTA?", "No. Use strict no-leak high-quality reconstruction under the reported setup.", REGISTRY),
    ("Is this simulation-only?", "Unless hardware data are added, present the current package as simulation/evaluation protocol evidence.", PHASE16_TABLES["runtime"]),
    ("What would happen in hardware?", "The method should transfer conceptually if A is calibrated/exported, but hardware noise/model mismatch would need a separate experiment.", PHASE16_TABLES["noise"]),
]


def main() -> None:
    rows = [{"question": q, "short_answer": a, "evidence_file": str(e)} for q, a, e in QUESTIONS]
    lines = [
        "# Reviewer risk register",
        "",
        "Use this document to answer likely reviewer concerns without overclaiming. Every answer points to an evidence file.",
        "",
        markdown_table(rows, ["question", "short_answer", "evidence_file"]),
    ]
    write_text(OUT / "reviewer_risk_register.md", "\n".join(lines))
    print({"output": str(OUT / "reviewer_risk_register.md"), "questions": len(rows)})


if __name__ == "__main__":
    main()
