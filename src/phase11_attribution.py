from __future__ import annotations

from pathlib import Path

from .phase11_common import ROOT11, as_float, ensure_dir, plot_bar, read_metrics_for_output, write_csv_rows, write_md_table


EXPERIMENTS = [
    ("phase10", "hadamard10_full_noise001", Path("E:/ns_mc_gan_gi/outputs_phase10/hadamard10_full_noise001")),
    ("phase10", "hadamard5_medium_noise001", Path("E:/ns_mc_gan_gi/outputs_phase10/hadamard5_medium_noise001")),
    ("phase10", "hadamard5_full_noise001", Path("E:/ns_mc_gan_gi/outputs_phase10/hadamard5_full_noise001")),
    ("phase10", "rademacher10_full_noise001", Path("E:/ns_mc_gan_gi/outputs_phase10/rademacher10_full_noise001")),
    ("phase10", "scrambled_hadamard10_full_noise001", Path("E:/ns_mc_gan_gi/outputs_phase10/scrambled_hadamard10_full_noise001")),
    ("phase10", "mnist_hadamard5_full", Path("E:/ns_mc_gan_gi/outputs_phase10/mnist_hadamard5_full")),
    ("phase10", "fashion_hadamard5_full", Path("E:/ns_mc_gan_gi/outputs_phase10/fashion_hadamard5_full")),
    ("phase11", "hadamard10_seed43", Path("E:/ns_mc_gan_gi/outputs_phase11/hadamard10_seed43")),
    ("phase11", "hadamard10_seed44", Path("E:/ns_mc_gan_gi/outputs_phase11/hadamard10_seed44")),
    ("phase11", "hadamard5_push_hq", Path("E:/ns_mc_gan_gi/outputs_phase11/hadamard5_push_hq")),
]


FIELDS = [
    "phase",
    "method",
    "backproj_psnr",
    "backproj_ssim",
    "backproj_mse",
    "model_psnr",
    "model_ssim",
    "model_mse",
    "delta_psnr_model_minus_backproj",
    "delta_ssim_model_minus_backproj",
    "delta_mse_model_minus_backproj",
    "relative_error_reduction",
    "classification",
    "status",
]


def classify(delta_psnr, delta_ssim, model_psnr, backproj_psnr, model_ssim, backproj_ssim) -> str:
    if None in {model_psnr, backproj_psnr, model_ssim, backproj_ssim, delta_psnr, delta_ssim}:
        return "missing"
    if model_psnr < backproj_psnr - 0.1 or model_ssim < backproj_ssim - 0.01:
        return "model_degrades_backprojection"
    if delta_psnr < 0.3 and delta_ssim < 0.03:
        return "backprojection_dominated"
    return "model_refinement_helpful"


def row_for(phase: str, method: str, output_dir: Path) -> dict:
    metrics = read_metrics_for_output(output_dir)
    if not metrics:
        return {**{field: "" for field in FIELDS}, "phase": phase, "method": method, "status": "missing"}
    back = metrics.get("backprojection", {})
    model = metrics.get("model", {})
    bp = as_float(back.get("psnr"))
    bs = as_float(back.get("ssim"))
    bm = as_float(back.get("mse"))
    mp = as_float(model.get("psnr"))
    ms = as_float(model.get("ssim"))
    mm = as_float(model.get("mse"))
    delta_psnr = "" if bp is None or mp is None else mp - bp
    delta_ssim = "" if bs is None or ms is None else ms - bs
    delta_mse = "" if bm is None or mm is None else mm - bm
    rel_reduction = "" if bm in {None, 0.0} or mm is None else (bm - mm) / bm
    cls = classify(as_float(delta_psnr), as_float(delta_ssim), mp, bp, ms, bs)
    return {
        "phase": phase,
        "method": method,
        "backproj_psnr": back.get("psnr", ""),
        "backproj_ssim": back.get("ssim", ""),
        "backproj_mse": back.get("mse", ""),
        "model_psnr": model.get("psnr", ""),
        "model_ssim": model.get("ssim", ""),
        "model_mse": model.get("mse", ""),
        "delta_psnr_model_minus_backproj": delta_psnr,
        "delta_ssim_model_minus_backproj": delta_ssim,
        "delta_mse_model_minus_backproj": delta_mse,
        "relative_error_reduction": rel_reduction,
        "classification": cls,
        "status": "completed",
    }


def main() -> None:
    ensure_dir(ROOT11)
    rows = [row_for(phase, method, output_dir) for phase, method, output_dir in EXPERIMENTS]
    write_csv_rows(rows, ROOT11 / "attribution_results.csv", FIELDS)
    write_md_table(rows, ROOT11 / "attribution_results.md", FIELDS)
    plot_rows = [row for row in rows if row.get("status") == "completed"]
    plot_bar(plot_rows, "delta_psnr_model_minus_backproj", ROOT11 / "attribution_delta_psnr.png", "Delta PSNR")
    plot_bar(plot_rows, "delta_ssim_model_minus_backproj", ROOT11 / "attribution_delta_ssim.png", "Delta SSIM")
    print(f"Attribution written to: {ROOT11 / 'attribution_results.csv'}")


if __name__ == "__main__":
    main()
