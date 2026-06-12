from __future__ import annotations

from .phase17_common import PHASE16, PHASE16_TABLES, PHASE17, main_result_rows, markdown_table, read_csv, write_csv, write_text


OUT = PHASE17 / "figure_table_pack"


MAIN_TABLES = [
    {"id": "Main Table 1", "title": "Primary strict no-leak results", "source": "Phase15 noleak_registry.csv", "placement": "Main Results"},
    {"id": "Main Table 2", "title": "Backprojection/model attribution", "source": str(PHASE16_TABLES["attribution"]), "placement": "Main Results or Ablations"},
    {"id": "Main Table 3", "title": "Inference ablation summary", "source": str(PHASE16_TABLES["ablation"]), "placement": "Ablations"},
]

SUPP_TABLES = [
    {"id": "Table S1", "title": "Exact-A Rademacher reproducibility audit", "source": str(PHASE16_TABLES["exact_a_reeval"])},
    {"id": "Table S2", "title": "Noise sweep", "source": str(PHASE16_TABLES["noise"])},
    {"id": "Table S3", "title": "Traditional baselines including small-subset TV-PGD", "source": str(PHASE16_TABLES["traditional_baselines"])},
    {"id": "Table S4", "title": "DC row control", "source": str(PHASE16_TABLES["dc_row"])},
    {"id": "Table S5", "title": "Bootstrap confidence intervals", "source": str(PHASE16_TABLES["statistics"])},
    {"id": "Table S6", "title": "STL-10 class-wise diagnostics", "source": str(PHASE16_TABLES["classwise"])},
    {"id": "Table S7", "title": "Measurement perturbation controls", "source": str(PHASE16_TABLES["perturbation"])},
    {"id": "Table S8", "title": "Runtime and model complexity", "source": str(PHASE16_TABLES["runtime"])},
]

FIGURES = [
    {"id": "Figure 1", "title": "Mechanism schematic", "source": "to draw", "placement": "Method", "caption": "Measurement-consistent null-space neural reconstruction. A data solution is computed from bucket measurements, a learned correction is projected into the measurement null space, and a final projection enforces consistency with y."},
    {"id": "Figure 2", "title": "Measurement families and attribution", "source": str(PHASE16 / "attribution"), "placement": "Forward model / Main Results", "caption": "Measurement-family attribution under strict no-leak evaluation. Rademacher uses exact exported A with a cache-rebuilt solver; supplementary evaluations do not train new models."},
    {"id": "Figure 3", "title": "Main STL-10 5% and 10% results", "source": "Phase15 registry and optional reconstruction grids", "placement": "Main Results", "caption": "Primary STL-10 results at 5% and 10% sampling for Rademacher and scrambled Hadamard. Metrics are strict no-leak imported results."},
    {"id": "Figure 4", "title": "Inference ablations", "source": str(PHASE16 / "inference_ablation"), "placement": "Ablations", "caption": "Inference-time ablations show the role of measurement consistency, refiner, and EMA weights. No supplementary retraining is performed."},
    {"id": "Figure 5", "title": "Robustness and reviewer defense", "source": str(PHASE16), "placement": "Ablations / Supplement", "caption": "Finite noise sweep, measurement perturbation, small-subset TV-PGD controls, and confidence intervals. Robustness is limited to the tested finite noise range."},
]


def manifest_tables() -> list[dict[str, str]]:
    rows = []
    for item in MAIN_TABLES + SUPP_TABLES:
        source = item["source"]
        count = len(read_csv(PHASE16_TABLES.get("attribution", source))) if source.endswith(".csv") else ""
        rows.append({**item, "rows": count})
    return rows


def existing_figures() -> list[dict[str, str]]:
    rows = []
    for path in PHASE16.rglob("*.png"):
        rows.append({"id": "", "title": path.stem, "source": str(path), "placement": "candidate supplementary figure", "caption": "Generated Phase16 diagnostic figure."})
    return rows


def main() -> None:
    write_text(OUT / "MAIN_TABLES.md", "# Main tables\n\n" + markdown_table(MAIN_TABLES, ["id", "title", "source", "placement"]))
    write_text(OUT / "SUPPLEMENTARY_TABLES.md", "# Supplementary tables\n\n" + markdown_table(SUPP_TABLES, ["id", "title", "source"]))
    write_text(OUT / "FIGURE_CAPTIONS.md", "# Figure captions\n\n" + "\n\n".join(f"## {f['id']}: {f['title']}\n\n{f['caption']}\n\nSource/planning: `{f['source']}`" for f in FIGURES))
    write_text(OUT / "TABLE_CAPTIONS.md", "# Table captions\n\n" + "\n\n".join(f"## {t['id']}: {t['title']}\n\nSource: `{t['source']}`" for t in MAIN_TABLES + SUPP_TABLES))
    write_text(OUT / "FIGURE_PLACEMENT_PLAN.md", "# Figure placement plan\n\n" + markdown_table(FIGURES, ["id", "title", "placement", "source"]))
    table_manifest = MAIN_TABLES + SUPP_TABLES
    figure_manifest = FIGURES + existing_figures()
    write_csv(OUT / "table_manifest.csv", table_manifest, ["id", "title", "source", "placement"])
    write_csv(OUT / "figure_manifest.csv", figure_manifest, ["id", "title", "source", "placement", "caption"])
    write_text(OUT / "MAIN_RESULT_TABLE_PREVIEW.md", "# Primary no-leak table preview\n\n" + markdown_table(main_result_rows(), ["method", "dataset", "sampling", "family", "psnr", "ssim", "bp_psnr", "delta_psnr"]))
    print({"output": str(OUT), "tables": len(table_manifest), "figures": len(figure_manifest)})


if __name__ == "__main__":
    main()
