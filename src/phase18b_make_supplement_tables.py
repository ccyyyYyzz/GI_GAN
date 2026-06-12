from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .phase18b_common import (
    METHOD_LABEL,
    METHOD_ORDER,
    OUT,
    as_float,
    fmt,
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


def main_tables() -> None:
    reg = registry_by_id()
    fields = ["Dataset", "Sampling", "Measurement", "PSNR", "SSIM", "BP PSNR", "Delta PSNR", "HQ?"]
    rows = []
    for mid in METHOD_ORDER:
        r = reg[mid]
        psnr = as_float(r["psnr"])
        ssim = as_float(r["ssim"])
        ratio = as_float(r["sampling_ratio"])
        if r["dataset"] == "STL-10" and abs(ratio - 0.05) < 1e-6:
            hq = psnr >= 20 and ssim >= 0.60
        elif r["dataset"] == "STL-10" and abs(ratio - 0.10) < 1e-6:
            hq = psnr >= 22 and ssim >= 0.65
        else:
            hq = psnr >= 25 and ssim >= 0.80
        rows.append(
            {
                "Dataset": r["dataset"],
                "Sampling": f"{ratio * 100:.0f}%",
                "Measurement": r["measurement_family"].replace("_", " "),
                "PSNR": fmt(r["psnr"]),
                "SSIM": fmt(r["ssim"]),
                "BP PSNR": fmt(r["backproj_psnr"]),
                "Delta PSNR": fmt(r["delta_psnr"]),
                "HQ?": "yes" if hq else "no",
            }
        )
    write_pack(
        "main_table1_primary_results",
        rows,
        fields,
        r"\textbf{Primary strict no-leak results.} HQ uses internal engineering thresholds defined in the text.",
        "tab:primary_results",
    )

    attr = {r["method_id"]: r for r in table("attribution")}
    fields = ["Method", "Sampling", "BP PSNR", "Model PSNR", "Delta PSNR", "Interpretation"]
    rows = []
    for mid in METHOD_ORDER[:4]:
        r = attr[mid]
        if "rademacher" in r["measurement_family"]:
            interp = "weak BP, large learned gain"
        else:
            interp = "stronger BP, similar final quality"
        rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "Sampling": f"{as_float(r['sampling_ratio']) * 100:.0f}%",
                "BP PSNR": fmt(r["backproj_psnr"]),
                "Model PSNR": fmt(r["model_psnr"]),
                "Delta PSNR": fmt(r["delta_psnr"]),
                "Interpretation": interp,
            }
        )
    write_pack(
        "main_table2_measurement_attribution_summary",
        rows,
        fields,
        r"\textbf{Measurement attribution summary.} Final quality hides different physical-initialization regimes.",
        "tab:measurement_attribution",
    )


def noise_summary() -> None:
    rows = []
    by = defaultdict(list)
    for r in table("noise"):
        by[r["method_id"]].append(r)
    for mid in METHOD_ORDER:
        sub = {fmt(r["noise_std"], 2): r for r in by.get(mid, [])}
        n0 = sub.get("0.00")
        n1 = sub.get("0.01")
        n5 = sub.get("0.05")
        if not (n0 and n1 and n5):
            continue
        rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "PSNR noise 0.00": fmt(n0["psnr"]),
                "PSNR noise 0.01": fmt(n1["psnr"]),
                "PSNR noise 0.05": fmt(n5["psnr"]),
                "Drop 0.00 to 0.05": fmt(as_float(n0["psnr"]) - as_float(n5["psnr"])),
                "SSIM noise 0.05": fmt(n5["ssim"]),
            }
        )
    fields = list(rows[0])
    write_pack("supplement_noise_summary_table", rows, fields, "Noise-sweep summary; full detailed CSV is provided separately.", "tab:supp_noise_summary")


def baseline_summary() -> None:
    reg = registry_by_id()
    rows = []
    by = defaultdict(list)
    for r in table("baseline"):
        by[r["method_id"]].append(r)
    for mid in METHOD_ORDER:
        sub = by.get(mid, [])
        if not sub:
            continue
        bp = next((r for r in sub if r["baseline"] == "backprojection"), {})
        adj = next((r for r in sub if r["baseline"] == "adjoint"), {})
        tv = max([r for r in sub if r["baseline"] == "tv_pgd"], key=lambda r: as_float(r["psnr"]))
        rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "Backprojection PSNR": fmt(bp.get("psnr")),
                "Adjoint PSNR": fmt(adj.get("psnr")),
                "Best CS-TV PSNR": fmt(tv.get("psnr")),
                "Ours PSNR": fmt(reg[mid]["psnr"]),
                "Subset size": tv.get("num_samples", ""),
                "Note": "small-subset PGD",
            }
        )
    fields = list(rows[0])
    write_pack("supplement_traditional_baseline_summary_table", rows, fields, "Traditional-baseline summary; CS-TV is a lightweight PGD small-subset baseline.", "tab:supp_baseline_summary")


def classwise_summary() -> None:
    rows = []
    by = defaultdict(list)
    for r in table("classwise"):
        by[r["method_id"]].append(r)
    for mid, sub in by.items():
        if mid not in METHOD_LABEL:
            continue
        best = max(sub, key=lambda r: as_float(r["mean_psnr"]))
        worst = min(sub, key=lambda r: as_float(r["mean_psnr"]))
        psnrs = [as_float(r["mean_psnr"]) for r in sub]
        ssims = [as_float(r["mean_ssim"]) for r in sub]
        rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "Best class": best["class_name"],
                "Worst class": worst["class_name"],
                "PSNR range": f"{min(psnrs):.2f}-{max(psnrs):.2f}",
                "SSIM range": f"{min(ssims):.3f}-{max(ssims):.3f}",
            }
        )
    fields = list(rows[0])
    write_pack("supplement_classwise_summary_table", rows, fields, "Class-wise diagnostic summary.", "tab:supp_classwise_summary")


def runtime_summary() -> None:
    rows = []
    by = defaultdict(dict)
    for r in table("runtime"):
        by[r["method_id"]][r["path"]] = r
    for mid in METHOD_ORDER:
        paths = by.get(mid, {})
        inf = paths.get("ns_mc_gan_full_inference", {})
        bp = paths.get("backprojection", {})
        tv = paths.get("tv_pgd_best_observed", {})
        if not inf:
            continue
        rows.append(
            {
                "Method": METHOD_LABEL[mid],
                "Model params M": fmt(inf.get("model_params_m")),
                "Inference sec/img": fmt(inf.get("runtime_sec_per_image"), 4),
                "BP sec/img": fmt(bp.get("runtime_sec_per_image"), 4),
                "CS-TV sec/img": fmt(tv.get("runtime_sec_per_image"), 4),
            }
        )
    fields = list(rows[0])
    write_pack("supplement_runtime_summary_table", rows, fields, "Runtime summary.", "tab:supp_runtime_summary")


def copy_detailed_csvs() -> None:
    detail = TABLE_DIR / "detailed_csv"
    detail.mkdir(parents=True, exist_ok=True)
    for name in ["noise", "baseline", "classwise", "runtime", "ablation", "perturbation", "statistics", "exact_a"]:
        rows = table(name)
        if rows:
            write_csv(detail / f"{name}_detailed.csv", rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    main_tables()
    noise_summary()
    baseline_summary()
    classwise_summary()
    runtime_summary()
    copy_detailed_csvs()
    print({"tables": str(TABLE_DIR)})


if __name__ == "__main__":
    main()
