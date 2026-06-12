from __future__ import annotations

from .phase17_common import (
    PHASE15R_REPORT,
    PHASE16_AGGREGATE,
    PHASE16_REPORT,
    PHASE16_SUPPORTED,
    PHASE16_TABLES,
    PHASE17,
    REGISTRY,
    ensure_dir,
    fnum,
    main_result_rows,
    markdown_table,
    read_csv,
    table_rows,
    write_csv,
    write_json,
    write_text,
)


OUT = PHASE17 / "evidence_index"
FIELDS = [
    "claim_id",
    "claim_text",
    "evidence_type",
    "source_file",
    "table_or_figure",
    "metric_values",
    "safe_to_claim",
    "caveat",
    "manuscript_section",
    "supplement_section",
]


def metric_pack() -> dict[str, str]:
    rows = main_result_rows()
    stl5 = [r for r in rows if r["dataset"] == "STL-10" and r["sampling"] == "5%"]
    stl10 = [r for r in rows if r["dataset"] == "STL-10" and r["sampling"] == "10%"]
    simple = [r for r in rows if r["dataset"] != "STL-10"]
    exact = read_csv(PHASE16_TABLES["exact_a_reeval"])
    return {
        "stl5": "; ".join(f"{r['method']}: PSNR {r['psnr']}, SSIM {r['ssim']}" for r in stl5),
        "stl10": "; ".join(f"{r['method']}: PSNR {r['psnr']}, SSIM {r['ssim']}" for r in stl10),
        "simple": "; ".join(f"{r['method']}: PSNR {r['psnr']}, SSIM {r['ssim']}" for r in simple),
        "exact": "; ".join(f"{r.get('method_id')}: diff PSNR {fnum(r.get('abs_diff_psnr'))}, status {r.get('status')}" for r in exact),
    }


def build_rows() -> list[dict[str, str]]:
    m = metric_pack()
    return [
        {
            "claim_id": "C1_STL10_5pct_HQ",
            "claim_text": "STL-10 5% high-quality reconstruction is supported for Rademacher and scrambled Hadamard.",
            "evidence_type": "primary no-leak metric table",
            "source_file": str(REGISTRY),
            "table_or_figure": "noleak_registry; main result table",
            "metric_values": m["stl5"],
            "safe_to_claim": "yes",
            "caveat": "High-quality is an operational threshold, not a theoretical guarantee.",
            "manuscript_section": "Main Results",
            "supplement_section": "S3/S8",
        },
        {
            "claim_id": "C2_STL10_10pct_HQ",
            "claim_text": "STL-10 10% high-quality reconstruction is supported for Rademacher and scrambled Hadamard.",
            "evidence_type": "primary no-leak metric table",
            "source_file": str(REGISTRY),
            "table_or_figure": "noleak_registry; main result table",
            "metric_values": m["stl10"],
            "safe_to_claim": "yes",
            "caveat": "Do not compare as SOTA without a broader benchmark.",
            "manuscript_section": "Main Results",
            "supplement_section": "S3/S8",
        },
        {
            "claim_id": "C3_MNIST_FASHION_5pct_HQ",
            "claim_text": "MNIST and Fashion-MNIST 5% high-quality reconstruction is supported as a simple-domain sanity result.",
            "evidence_type": "primary no-leak metric table",
            "source_file": str(REGISTRY),
            "table_or_figure": "noleak_registry; simple-domain rows",
            "metric_values": m["simple"],
            "safe_to_claim": "yes",
            "caveat": "Use as sanity evidence, not the central novelty.",
            "manuscript_section": "Main Results",
            "supplement_section": "S8",
        },
        {
            "claim_id": "C4_EXACT_A_REPRODUCED",
            "claim_text": "Rademacher exact-A evaluation is reproducible after the cache-rebuilt override path.",
            "evidence_type": "reproducibility audit",
            "source_file": str(PHASE16_TABLES["exact_a_reeval"]),
            "table_or_figure": "exactA_reeval_results.csv",
            "metric_values": m["exact"],
            "safe_to_claim": "yes",
            "caveat": "Only cite the safe exact-A path, not pre-fix mismatch runs.",
            "manuscript_section": "Experimental Protocol",
            "supplement_section": "S2",
        },
        {
            "claim_id": "C5_MODEL_REFINEMENT_HELPFUL",
            "claim_text": "The learned model improves reconstruction over backprojection.",
            "evidence_type": "attribution table",
            "source_file": str(PHASE16_TABLES["attribution"]),
            "table_or_figure": "attribution_final.csv",
            "metric_values": "Positive delta_PSNR and delta_SSIM across primary rows.",
            "safe_to_claim": "yes",
            "caveat": "Frame as empirical improvement under tested measurement families.",
            "manuscript_section": "Ablations and Reviewer-Defense Analyses",
            "supplement_section": "S3",
        },
        {
            "claim_id": "C6_MEASUREMENT_FAMILY_ATTRIBUTION",
            "claim_text": "Scrambled Hadamard gives stronger physical initialization than Rademacher, while final quality is similar after learning.",
            "evidence_type": "attribution table",
            "source_file": str(PHASE16_TABLES["attribution"]),
            "table_or_figure": "attribution_final.csv",
            "metric_values": "Backprojection PSNR differs strongly; final STL-10 PSNR is close within each sampling ratio.",
            "safe_to_claim": "yes",
            "caveat": "Do not claim universal dominance of either family.",
            "manuscript_section": "Main Results",
            "supplement_section": "S3",
        },
        {
            "claim_id": "C7_DC_ROW_IMPORTANT",
            "claim_text": "The DC row materially affects low-frequency Hadamard backprojection.",
            "evidence_type": "DC row control",
            "source_file": str(PHASE16_TABLES["dc_row"]),
            "table_or_figure": "dc_row_final.csv",
            "metric_values": "Backprojection-only include/skip rows for 5% and 10%.",
            "safe_to_claim": "yes",
            "caveat": "This is a low-frequency Hadamard diagnostic, not a primary STL-10 HQ claim.",
            "manuscript_section": "Ablations and Reviewer-Defense Analyses",
            "supplement_section": "S7",
        },
        {
            "claim_id": "C8_MEASUREMENT_CONSISTENCY_IMPORTANT",
            "claim_text": "Measurement consistency is important at inference time.",
            "evidence_type": "inference ablation",
            "source_file": str(PHASE16_TABLES["ablation"]),
            "table_or_figure": "real_inference_ablation_results.csv",
            "metric_values": "No-DC rows degrade PSNR/SSIM and increase inconsistency.",
            "safe_to_claim": "yes",
            "caveat": "Null-space ablations should be interpreted together with the trained architecture.",
            "manuscript_section": "Ablations and Reviewer-Defense Analyses",
            "supplement_section": "S4",
        },
        {
            "claim_id": "C9_REFINER_EMA_HELPFUL",
            "claim_text": "The refiner and EMA weights provide additional empirical gains.",
            "evidence_type": "inference ablation",
            "source_file": str(PHASE16_TABLES["ablation"]),
            "table_or_figure": "stage1/refiner and raw/EMA rows",
            "metric_values": "Stage1-only is below full; EMA is slightly above raw in tested rows.",
            "safe_to_claim": "qualified",
            "caveat": "State as a small empirical gain, not a fundamental theorem.",
            "manuscript_section": "Ablations and Reviewer-Defense Analyses",
            "supplement_section": "S4",
        },
        {
            "claim_id": "C10_NOISE_ROBUSTNESS_TESTED",
            "claim_text": "The model is robust over the finite tested noise sweep.",
            "evidence_type": "noise sweep",
            "source_file": str(PHASE16_TABLES["noise"]),
            "table_or_figure": "noise_sweep_results.csv",
            "metric_values": "Noise levels 0, 0.005, 0.01, 0.02, 0.05.",
            "safe_to_claim": "qualified",
            "caveat": "Do not claim arbitrary or adversarial robustness.",
            "manuscript_section": "Ablations and Reviewer-Defense Analyses",
            "supplement_section": "S5",
        },
        {
            "claim_id": "C11_TV_PGD_BASELINE_INCLUDED",
            "claim_text": "TV-PGD is included as a small-subset lightweight traditional baseline.",
            "evidence_type": "baseline table",
            "source_file": str(PHASE16_TABLES["traditional_baselines"]),
            "table_or_figure": "tv_pgd_baseline_results.csv",
            "metric_values": "Backprojection, adjoint, and TV-PGD rows.",
            "safe_to_claim": "qualified",
            "caveat": "Do not call TV-PGD exhaustive or fully optimized.",
            "manuscript_section": "Ablations and Reviewer-Defense Analyses",
            "supplement_section": "S6",
        },
        {
            "claim_id": "C12_PER_SAMPLE_CI_AVAILABLE",
            "claim_text": "Per-sample statistics and bootstrap confidence intervals are available.",
            "evidence_type": "statistics table",
            "source_file": str(PHASE16_TABLES["statistics"]),
            "table_or_figure": "statistics_ci.csv",
            "metric_values": "Mean, median, standard deviation, and 95% bootstrap CI.",
            "safe_to_claim": "yes",
            "caveat": "Report as empirical confidence intervals for the evaluated test subset.",
            "manuscript_section": "Ablations and Reviewer-Defense Analyses",
            "supplement_section": "S8",
        },
        {
            "claim_id": "C13_CLASSWISE_DIAGNOSTIC_AVAILABLE",
            "claim_text": "STL-10 class-wise diagnostics are available.",
            "evidence_type": "class-wise table",
            "source_file": str(PHASE16_TABLES["classwise"]),
            "table_or_figure": "classwise_stl10_metrics.csv",
            "metric_values": "Per-class mean PSNR/SSIM.",
            "safe_to_claim": "qualified",
            "caveat": "Do not over-interpret class-wise differences.",
            "manuscript_section": "Ablations and Reviewer-Defense Analyses",
            "supplement_section": "S9",
        },
        {
            "claim_id": "C14_MEASUREMENT_DEPENDENCE_SHOWN",
            "claim_text": "Measurement perturbation controls show the model depends on bucket measurements.",
            "evidence_type": "measurement perturbation",
            "source_file": str(PHASE16_TABLES["perturbation"]),
            "table_or_figure": "measurement_perturbation.csv",
            "metric_values": "Gaussian, shuffled-coefficient, and wrong-sample controls.",
            "safe_to_claim": "yes",
            "caveat": "Use as a sanity test against generic hallucination, not as a complete proof.",
            "manuscript_section": "Ablations and Reviewer-Defense Analyses",
            "supplement_section": "S10",
        },
        {
            "claim_id": "C15_RUNTIME_REPORTED",
            "claim_text": "Runtime and model complexity are reported.",
            "evidence_type": "runtime table",
            "source_file": str(PHASE16_TABLES["runtime"]),
            "table_or_figure": "runtime_complexity.csv",
            "metric_values": "Subset runtime per image, parameter count, and hardware-specific notes.",
            "safe_to_claim": "qualified",
            "caveat": "Runtime is local hardware specific and approximate.",
            "manuscript_section": "Discussion",
            "supplement_section": "S11",
        },
    ]


def main() -> None:
    ensure_dir(OUT)
    rows = build_rows()
    write_csv(OUT / "evidence_index.csv", rows, FIELDS)
    write_json(OUT / "evidence_index.json", rows)
    lines = [
        "# Evidence index",
        "",
        "This index maps manuscript claims to Phase15/Phase16 evidence. It intentionally excludes old unsafe Rademacher re-evaluation results and any leaked or deprecated exploratory outputs.",
        "",
        "## Source anchors",
        "",
        f"- Phase15 no-leak registry: `{REGISTRY}`",
        f"- Phase15R report: `{PHASE15R_REPORT}`",
        f"- Phase16 supplementary report: `{PHASE16_REPORT}`",
        f"- Phase16 aggregate summary: `{PHASE16_AGGREGATE}`",
        f"- Phase16 supported claims: `{PHASE16_SUPPORTED}`",
        "",
        "## Claims",
        "",
        markdown_table(rows, FIELDS),
        "",
        "## Phase16 table inventory",
        "",
        markdown_table(
            [{"name": name, "path": str(path), "exists": path.exists(), "rows": len(table_rows(name))} for name, path in PHASE16_TABLES.items()],
            ["name", "path", "exists", "rows"],
        ),
    ]
    write_text(OUT / "EVIDENCE_INDEX.md", "\n".join(lines))
    print({"rows": len(rows), "output": str(OUT)})


if __name__ == "__main__":
    main()
