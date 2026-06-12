from __future__ import annotations

from .phase18_rewrite_common import (
    METHOD_LABEL,
    OUT,
    fmt,
    main_results_rows,
    markdown_table,
    registry_by_id,
    table,
    tex_table,
    write_csv,
    write_text,
)


TABLE_DIR = OUT / "tables"


def write_pack(name: str, rows: list[dict], fields: list[str], caption: str, label: str, *, wide: bool = True) -> None:
    write_csv(TABLE_DIR / f"{name}.csv", rows, fields)
    write_text(TABLE_DIR / f"{name}.md", markdown_table(rows, fields))
    write_text(TABLE_DIR / f"{name}.tex", tex_table(rows, fields, caption, label, wide=wide))


def main_table_1() -> list[dict]:
    fields = ["Dataset", "Sampling", "Measurement", "PSNR", "SSIM", "BP PSNR", "Delta PSNR", "HQ?"]
    rows = []
    for row in main_results_rows():
        rows.append(
            {
                "Dataset": row["dataset"],
                "Sampling": row["sampling"],
                "Measurement": row["measurement"],
                "PSNR": row["psnr"],
                "SSIM": row["ssim"],
                "BP PSNR": row["bp_psnr"],
                "Delta PSNR": row["delta_psnr"],
                "HQ?": row["hq"],
            }
        )
    write_pack(
        "table1_primary_strict_noleak_results",
        rows,
        fields,
        r"\textbf{Primary strict no-leak reconstruction results.} HQ indicates the internal engineering threshold used in this study.",
        "tab:primary_results",
    )
    return rows


def main_table_2() -> list[dict]:
    fields = ["Measurement", "Sampling", "BP PSNR", "Model PSNR", "Delta PSNR", "Interpretation"]
    rows = []
    for row in table("attribution"):
        mid = row.get("method_id", "")
        if mid not in METHOD_LABEL:
            continue
        family = row.get("measurement_family", "").replace("_", " ")
        if family == "rademacher":
            interp = "Weak physical inverse; large learned gain"
        elif family == "scrambled hadamard":
            interp = "Stronger physical initialization; similar final quality"
        else:
            interp = "Simple-domain / diagnostic low-frequency control"
        rows.append(
            {
                "Measurement": METHOD_LABEL[mid],
                "Sampling": f"{float(row.get('sampling_ratio', 0.0)) * 100:.0f}%",
                "BP PSNR": fmt(row.get("backproj_psnr")),
                "Model PSNR": fmt(row.get("model_psnr")),
                "Delta PSNR": fmt(row.get("delta_psnr")),
                "Interpretation": interp,
            }
        )
    write_pack(
        "table2_measurement_attribution",
        rows,
        fields,
        r"\textbf{Measurement-family attribution.} Final PSNR alone hides different backprojection regimes.",
        "tab:measurement_attribution",
    )
    return rows


def main_table_3() -> list[dict]:
    fields = ["Method", "Full PSNR", "No-DC PSNR", "No-null PSNR", "Stage1 PSNR", "Raw PSNR", "EMA PSNR"]
    rows = []
    by_method: dict[str, dict[str, str]] = {}
    for row in table("ablation"):
        mid = row.get("method_id", "")
        if mid not in METHOD_LABEL:
            continue
        by_method.setdefault(mid, {})
        by_method[mid][row.get("ablation_mode", "")] = fmt(row.get("psnr"))
    for mid in ["rademacher5_hq_noise001_colab", "scrambled_hadamard5_hq_noise001_colab", "rademacher10_full_noise001_colab", "scrambled_hadamard10_full_noise001_colab"]:
        modes = by_method.get(mid, {})
        rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "Full PSNR": modes.get("full_model", ""),
                "No-DC PSNR": modes.get("no_dc_project", ""),
                "No-null PSNR": modes.get("no_null_project", ""),
                "Stage1 PSNR": modes.get("stage1_only", ""),
                "Raw PSNR": modes.get("raw_weights", ""),
                "EMA PSNR": modes.get("ema_weights", ""),
            }
        )
    write_pack(
        "table3_inference_ablation_summary",
        rows,
        fields,
        r"\textbf{Inference-time ablation summary.} No-DC projection removal is the largest degradation; no-null removal is small for these checkpoints.",
        "tab:ablation_summary",
    )
    return rows


def supplement_tables() -> None:
    exact_fields = ["method_id", "original_psnr", "reeval_psnr", "abs_diff_psnr", "status"]
    write_pack("supp_exact_a_reproducibility", table("exact_a"), exact_fields, "Exact-A Rademacher reproducibility audit.", "tab:supp_exact_a")

    noise_fields = ["method_id", "noise_std", "psnr", "ssim", "rel_meas_err"]
    write_pack("supp_noise_sweep", table("noise"), noise_fields, "Finite noise sweep.", "tab:supp_noise")

    baseline = []
    for row in table("baseline"):
        r = dict(row)
        if r.get("baseline") == "tv_pgd":
            r["baseline"] = "CS-TV (PGD solver)"
        baseline.append(r)
    baseline_fields = ["method_id", "baseline", "num_samples", "lambda_tv", "psnr", "ssim", "notes"]
    write_pack("supp_cs_tv_baseline", baseline, baseline_fields, "Traditional baselines including CS-TV (PGD solver).", "tab:supp_cstv")

    dc_fields = ["method_id", "sampling_ratio", "hadamard_include_dc", "hadamard_skip_dc", "backproj_psnr", "backproj_ssim"]
    write_pack("supp_dc_row_control", table("dc_row"), dc_fields, "Low-frequency Hadamard DC-row control.", "tab:supp_dc")

    ci_fields = ["method_id", "mean_psnr", "ci95_psnr_low", "ci95_psnr_high", "mean_ssim", "ci95_ssim_low", "ci95_ssim_high"]
    write_pack("supp_bootstrap_ci", table("statistics"), ci_fields, "Bootstrap confidence intervals.", "tab:supp_ci")

    class_fields = ["method_id", "class_id", "class_name", "num_samples", "mean_psnr", "mean_ssim"]
    write_pack("supp_classwise", table("classwise"), class_fields, "STL-10 class-wise diagnostics.", "tab:supp_classwise")

    runtime = []
    for row in table("runtime"):
        r = dict(row)
        if r.get("path") == "tv_pgd_best_observed":
            r["path"] = "CS-TV best observed"
        runtime.append(r)
    runtime_fields = ["method_id", "path", "runtime_sec_per_image", "model_params_m", "device", "notes"]
    write_pack("supp_runtime", runtime, runtime_fields, "Runtime and complexity diagnostics.", "tab:supp_runtime")


def main() -> None:
    main_table_1()
    main_table_2()
    main_table_3()
    supplement_tables()
    print({"output": str(TABLE_DIR)})


if __name__ == "__main__":
    main()
