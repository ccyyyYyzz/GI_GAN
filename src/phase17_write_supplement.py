from __future__ import annotations

from .phase17_common import PHASE16_TABLES, PHASE17, markdown_table, read_csv, tex_escape, write_text


OUT = PHASE17 / "supplement"


SECTIONS = [
    ("S1. Additional details of the forward model", "The forward model is the bucket-measurement equation y = Ax + epsilon. The reported reconstructions use the same measurement family and sampling ratio described in the Phase15 no-leak registry."),
    ("S2. Exact-A reproducibility audit", "Rademacher evaluations reload the exported exact operator and rebuild the cached solver. Pre-fix mismatch runs are excluded."),
    ("S3. Measurement family attribution", "Attribution separates backprojection quality from final learned reconstruction quality."),
    ("S4. Inference-time ablation", "Ablations evaluate data consistency, null-space usage, refiner contribution, and EMA/raw checkpoints at inference time."),
    ("S5. Noise robustness", "The noise sweep is a finite diagnostic over the tested noise levels only."),
    ("S6. Traditional baselines", "Backprojection and adjoint are linear controls. TV-PGD is a small-subset lightweight baseline, not a fully optimized exhaustive baseline."),
    ("S7. DC row control", "The DC-row control is a low-frequency Hadamard diagnostic and should not be confused with the primary scrambled Hadamard/Rademacher results."),
    ("S8. Per-sample statistics and bootstrap CI", "Bootstrap confidence intervals summarize per-sample reconstruction variability."),
    ("S9. STL-10 class-wise results", "Class-wise results are diagnostic and should not be over-interpreted."),
    ("S10. Measurement perturbation sanity", "Perturbation controls test whether the model depends on bucket measurements."),
    ("S11. Runtime and model complexity", "Runtime rows are hardware-specific approximate measurements."),
    ("S12. Excluded / deprecated results", "Leaked, exploratory, and pre-fix Rademacher mismatch results are not used for manuscript claims."),
    ("S13. Implementation details", "Phase17 performs document generation only. It does not train models or add experiments."),
]


TABLE_FIELDS = {
    "exact_a_reeval": ["method_id", "original_psnr", "reeval_psnr", "abs_diff_psnr", "status"],
    "attribution": ["method_id", "backproj_psnr", "model_psnr", "delta_psnr", "classification"],
    "ablation": ["method_id", "ablation_mode", "psnr", "ssim", "delta_vs_full_psnr", "status"],
    "noise": ["method_id", "noise_std", "psnr", "ssim", "rel_meas_err", "status"],
    "traditional_baselines": ["method_id", "baseline", "num_samples", "lambda_tv", "psnr", "ssim", "notes"],
    "dc_row": ["method_id", "sampling_ratio", "hadamard_include_dc", "hadamard_skip_dc", "backproj_psnr", "backproj_ssim"],
    "statistics": ["method_id", "mean_psnr", "ci95_psnr_low", "ci95_psnr_high", "mean_ssim", "ci95_ssim_low", "ci95_ssim_high"],
    "classwise": ["method_id", "class_id", "class_name", "num_samples", "mean_psnr", "mean_ssim"],
    "perturbation": ["method_id", "perturbation_mode", "psnr", "psnr_drop_from_normal", "rel_meas_err"],
    "runtime": ["method_id", "path", "runtime_sec_per_image", "model_params_m", "device", "notes"],
}


def table_block(name: str, title: str, limit: int | None = None) -> str:
    rows = read_csv(PHASE16_TABLES[name])
    fields = TABLE_FIELDS[name]
    return f"### {title}\n\nSource: `{PHASE16_TABLES[name]}`\n\n{markdown_table(rows, fields, limit=limit)}\n"


def md() -> str:
    lines = ["# Supplementary Material", "", "This supplementary draft is generated from Phase15/Phase16 evidence only. No new training or experiments are performed in Phase17.", ""]
    for title, body in SECTIONS:
        lines.extend([f"## {title}", "", body, ""])
        if title.startswith("S2"):
            lines.append(table_block("exact_a_reeval", "Exact-A re-evaluation"))
        elif title.startswith("S3"):
            lines.append(table_block("attribution", "Attribution table"))
        elif title.startswith("S4"):
            lines.append(table_block("ablation", "Inference ablation", limit=24))
        elif title.startswith("S5"):
            lines.append(table_block("noise", "Finite noise sweep"))
        elif title.startswith("S6"):
            lines.append(table_block("traditional_baselines", "Traditional baseline controls", limit=24))
        elif title.startswith("S7"):
            lines.append(table_block("dc_row", "DC row control"))
        elif title.startswith("S8"):
            lines.append(table_block("statistics", "Bootstrap confidence intervals"))
        elif title.startswith("S9"):
            lines.append(table_block("classwise", "Class-wise STL-10 diagnostics", limit=30))
        elif title.startswith("S10"):
            lines.append(table_block("perturbation", "Measurement perturbation controls", limit=30))
        elif title.startswith("S11"):
            lines.append(table_block("runtime", "Runtime and complexity"))
    return "\n".join(lines)


def tex() -> str:
    body = md()
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage{amsmath,booktabs,geometry,longtable}",
        r"\geometry{margin=1in}",
        r"\title{Supplementary Material}",
        r"\begin{document}",
        r"\maketitle",
    ]
    for line in body.splitlines():
        if line.startswith("# "):
            lines.append(r"\section*{" + tex_escape(line[2:]) + "}")
        elif line.startswith("## "):
            lines.append(r"\section{" + tex_escape(line[3:]) + "}")
        elif line.startswith("### "):
            lines.append(r"\subsection{" + tex_escape(line[4:]) + "}")
        elif line.startswith("|"):
            continue
        elif line.strip():
            lines.append(tex_escape(line) + r"\\")
        else:
            lines.append("")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def main() -> None:
    write_text(OUT / "supplementary_material.md", md())
    write_text(OUT / "supplementary_material.tex", tex())
    print({"output": str(OUT), "files": 2})


if __name__ == "__main__":
    main()
