from __future__ import annotations

from .phase17_common import PHASE17, TITLE, CORE_CLAIM, main_result_rows, markdown_table, write_text


OUT = PHASE17 / "submission_pack"


def metric_summary() -> str:
    rows = main_result_rows()
    stl = [r for r in rows if r["dataset"] == "STL-10"]
    simple = [r for r in rows if r["dataset"] != "STL-10"]
    return (
        "Primary STL-10 rows: "
        + "; ".join(f"{r['method']} {r['psnr']}/{r['ssim']}" for r in stl)
        + ". Simple-domain rows: "
        + "; ".join(f"{r['method']} {r['psnr']}/{r['ssim']}" for r in simple)
        + "."
    )


def cover(journal: str) -> str:
    return f"""# Cover letter draft: {journal}

Dear Editor,

We are pleased to submit the manuscript entitled "{TITLE}" for consideration in {journal}.

This work addresses high-quality low-sampling ghost imaging / single-pixel imaging under strict no-leak evaluation. The proposed framework combines an explicit data-consistent reconstruction, a learned null-space correction, and a final measurement-consistency projection. The central claim is deliberately measured: {CORE_CLAIM}

{metric_summary()}

The submission package includes a supplementary reviewer-defense audit: exact-A Rademacher reproducibility, measurement-family attribution, inference-time ablations, finite noise diagnostics, traditional baselines including small-subset TV-PGD, DC-row controls, bootstrap confidence intervals, class-wise diagnostics, measurement perturbation controls, and runtime measurements. We do not claim strict state-of-the-art performance; instead, the paper emphasizes physical consistency, auditable reconstruction, and transparent limits.

Thank you for considering our manuscript.

Sincerely,

Author names to be added
"""


def main() -> None:
    rows = [
        {"rank": 1, "journal": "Photonics Research", "fit": "Strong optics/photonics fit if the narrative emphasizes GI/SPI physics and measurement consistency.", "risk": "Needs polished figures and careful novelty positioning."},
        {"rank": 2, "journal": "Journal of Physics: Photonics", "fit": "Good for physics-readable reconstruction framework with clear limitations.", "risk": "May expect stronger physical interpretation and less ML hype."},
        {"rank": 3, "journal": "Optics Express", "fit": "Broad optics venue; practical if the story is concise and well-supported.", "risk": "Novelty must be stated carefully against prior DL-GI/SPI."},
        {"rank": 4, "journal": "IEEE Transactions on Computational Imaging", "fit": "Possible if rewritten in a computational-imaging style.", "risk": "Would require stronger comparison framing and citation coverage."},
    ]
    write_text(OUT / "JOURNAL_TARGETS.md", "# Journal targets\n\n" + markdown_table(rows, ["rank", "journal", "fit", "risk"]))
    write_text(OUT / "cover_letter_photonics_research.md", cover("Photonics Research"))
    write_text(OUT / "cover_letter_optics_express.md", cover("Optics Express"))
    write_text(
        OUT / "editor_summary.md",
        f"""# Editor summary

{TITLE}

{CORE_CLAIM}

{metric_summary()}

The paper is positioned as a physically constrained and auditable low-sampling GI/SPI reconstruction framework, not as a SOTA leaderboard paper. The strongest editor-facing points are strict no-leak evaluation, exact-A Rademacher reproducibility, explicit measurement consistency, and a substantial supplementary audit.
""",
    )
    write_text(
        OUT / "highlights.md",
        """# Highlights

- Measurement-consistent null-space neural reconstruction for low-sampling GI/SPI.
- Strict no-leak STL-10 5% and 10% results under Rademacher and scrambled Hadamard sensing.
- Exact-A Rademacher reproducibility audit with cache-rebuilt solver path.
- Attribution separates physical backprojection strength from learned refinement.
- Supplement includes ablations, finite noise diagnostics, traditional baselines, confidence intervals, perturbation controls, and runtime.
""",
    )
    write_text(
        OUT / "graphical_abstract_text.md",
        """# Graphical abstract text

Bucket measurements y are first mapped to a data-consistent reconstruction. A neural reconstructor proposes missing structure, but only its measurement-null-space component is added. A final projection enforces consistency with the original bucket measurements, yielding high-quality low-sampling GI/SPI reconstructions under strict no-leak evaluation.
""",
    )
    write_text(
        OUT / "code_data_availability_statement.md",
        """# Code and data availability statement draft

The code used to generate the reported Phase15/Phase16 evaluation tables and the Phase17 manuscript package will be made available in a public repository upon acceptance or submission, subject to repository cleanup. The reported numerical results are generated from strict no-leak evaluation artifacts, exact-A Rademacher audit files, and supplementary CSV tables archived under the project output directories. Public datasets used in the experiments include STL-10, MNIST, and Fashion-MNIST. Final repository URL, archive DOI, and any model-checkpoint release constraints should be inserted before submission.
""",
    )
    print({"output": str(OUT), "files": 7})


if __name__ == "__main__":
    main()
