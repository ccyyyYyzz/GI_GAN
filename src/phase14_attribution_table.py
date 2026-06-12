from __future__ import annotations

from .phase14_ablation_pack_common import classify, f, load_main_rows, plot_bar, write_rows, out_dir


def main() -> None:
    rows = []
    for row in load_main_rows():
        model_psnr = f(row.get("psnr"))
        model_ssim = f(row.get("ssim"))
        back_psnr = f(row.get("backproj_psnr"))
        back_ssim = f(row.get("backproj_ssim"))
        delta_psnr = f(row.get("delta_psnr"))
        delta_ssim = f(row.get("delta_ssim"))
        rows.append(
            {
                "method": row.get("display_name") or row.get("method_id"),
                "dataset": row.get("dataset"),
                "sampling_ratio": row.get("sampling_ratio"),
                "pattern_type": row.get("pattern_type"),
                "backproj_psnr": back_psnr if back_psnr is not None else "",
                "backproj_ssim": back_ssim if back_ssim is not None else "",
                "model_psnr": model_psnr if model_psnr is not None else "",
                "model_ssim": model_ssim if model_ssim is not None else "",
                "delta_psnr": delta_psnr if delta_psnr is not None else "",
                "delta_ssim": delta_ssim if delta_ssim is not None else "",
                "classification": classify(model_psnr, back_psnr, delta_psnr, delta_ssim),
            }
        )
    fields = [
        "method",
        "dataset",
        "sampling_ratio",
        "pattern_type",
        "backproj_psnr",
        "backproj_ssim",
        "model_psnr",
        "model_ssim",
        "delta_psnr",
        "delta_ssim",
        "classification",
    ]
    write_rows("attribution_table", rows, fields)
    plot_bar(rows, "delta_psnr", out_dir() / "attribution_delta_psnr.png", "Model gain over backprojection", "delta PSNR")
    plot_bar(rows, "delta_ssim", out_dir() / "attribution_delta_ssim.png", "Model gain over backprojection", "delta SSIM")
    print(f"Wrote attribution table with {len(rows)} rows")


if __name__ == "__main__":
    main()
