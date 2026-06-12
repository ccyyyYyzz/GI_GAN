from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .phase48_49_common import write_csv, write_environment, write_markdown_table, write_sha256s
from .utils import ensure_dir, save_json


GAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase53C_exact_null_critic_import/session_24_optional_gan_and_posterior_sampling")
MEAN_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase15/imported_noleak/scrambled_hadamard5_hq_noise001_colab")
OUT_ROOT = Path("E:/ns_mc_gan_gi/outputs_phase59_gan_sampling_mode_g1")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def mean(vals: list[Any]) -> float:
    xs = [to_float(v) for v in vals]
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    v = to_float(value)
    return "n/a" if math.isnan(v) else f"{v:.{digits}f}"


def copy_or_missing(source: Path, dest: Path, missing: list[dict[str, Any]], label: str) -> bool:
    if source.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return True
    missing.append({"item": label, "path": str(source), "status": "missing"})
    return False


def make_labeled_composite(out: Path, mean_grid: Path, sample_grid: Path) -> None:
    if not mean_grid.exists() or not sample_grid.exists():
        return
    mean_img = Image.open(mean_grid).convert("RGB")
    sample_img = Image.open(sample_grid).convert("RGB")
    width = max(mean_img.width, sample_img.width)
    pad = 48
    total_h = mean_img.height + sample_img.height + pad * 2
    canvas = Image.new("RGB", (width, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), "Mean-mode Scr-5 reconstruction reference", fill=(0, 0, 0))
    canvas.paste(mean_img, ((width - mean_img.width) // 2, pad))
    y2 = pad + mean_img.height
    draw.text((10, y2 + 12), "GAN sampling-mode Scr-5 pilot grid", fill=(0, 0, 0))
    canvas.paste(sample_img, ((width - sample_img.width) // 2, y2 + pad))
    canvas.save(out)


def save_pdf_from_png(png: Path, pdf: Path) -> None:
    if not png.exists():
        return
    try:
        import matplotlib.pyplot as plt

        img = Image.open(png).convert("RGB")
        plt.figure(figsize=(max(4, img.width / 180), max(4, img.height / 180)))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(pdf)
        plt.close()
    except Exception:
        # PDF is useful but not required for the metric audit.
        return


def main() -> None:
    out = ensure_dir(OUT_ROOT)
    missing: list[dict[str, Any]] = []
    optional_rows = read_csv_rows(GAN_ROOT / "optional_gan_results.csv")
    posterior_rows = read_csv_rows(GAN_ROOT / "posterior_sampling_metrics.csv")
    mean_metrics = read_json(MEAN_ROOT / "eval_metrics.json")
    required = [
        ("optional_gan_results.csv", GAN_ROOT / "optional_gan_results.csv"),
        ("posterior_sampling_metrics.csv", GAN_ROOT / "posterior_sampling_metrics.csv"),
        ("sample_grid.png", GAN_ROOT / "sample_grid.png"),
        ("uncertainty_maps.png", GAN_ROOT / "uncertainty_maps.png"),
        ("variance_null_ratio.png", GAN_ROOT / "variance_null_ratio.png"),
        ("perception_distortion_curve.png", GAN_ROOT / "perception_distortion_curve.png"),
        ("mean_mode_eval_metrics.json", MEAN_ROOT / "eval_metrics.json"),
        ("mean_mode_recon_grid.png", MEAN_ROOT / "eval_samples" / "recon_grid.png"),
    ]
    for label, path in required:
        if not path.exists():
            missing.append({"item": label, "path": str(path), "status": "missing"})
    tensor_samples = [
        p
        for p in list(GAN_ROOT.rglob("*sample*.pt"))
        + list(GAN_ROOT.rglob("*samples*.pt"))
        + list(GAN_ROOT.rglob("*stochastic*.pt"))
        + list(GAN_ROOT.rglob("*sample*.npy"))
        + list(GAN_ROOT.rglob("*sample*.npz"))
        if "source_checkpoint" not in p.name and "Q_exact_null" not in p.name
    ]
    stochastic_image_files = [p for p in GAN_ROOT.rglob("*.png") if p.parent.name.lower() in {"samples", "sample_images", "stochastic_samples"}]
    if not tensor_samples and not stochastic_image_files:
        missing.append({"item": "individual stochastic samples", "path": str(GAN_ROOT), "status": "missing_or_grid_only"})
    copy_or_missing(GAN_ROOT / "sample_grid.png", out / "sampling_mode_grid.png", missing, "sampling mode grid")
    copy_or_missing(GAN_ROOT / "uncertainty_maps.png", out / "uncertainty_map.png", missing, "uncertainty map")
    copy_or_missing(GAN_ROOT / "variance_null_ratio.png", out / "null_variance_ratio_hist.png", missing, "null variance ratio histogram")
    copy_or_missing(GAN_ROOT / "perception_distortion_curve.png", out / "perception_distortion_sampling_plot.png", missing, "perception distortion plot")
    make_labeled_composite(out / "mean_vs_sampling_visual_grid.png", MEAN_ROOT / "eval_samples" / "recon_grid.png", GAN_ROOT / "sample_grid.png")
    for stem in ["sampling_mode_grid", "mean_vs_sampling_visual_grid", "uncertainty_map"]:
        save_pdf_from_png(out / f"{stem}.png", out / f"{stem}.pdf")
    scr_gan = [r for r in optional_rows if r.get("task") == "scr5" and str(r.get("status", "")).startswith("ran")]
    rad_gan = [r for r in optional_rows if r.get("task") == "rad5"]
    mean_psnr = to_float(mean_metrics.get("model", {}).get("psnr"))
    mean_ssim = to_float(mean_metrics.get("model", {}).get("ssim"))
    mean_rel = to_float(mean_metrics.get("model", {}).get("rel_meas_error"))
    gan_psnr = mean([r.get("psnr") for r in scr_gan])
    gan_ssim = mean([r.get("ssim") for r in scr_gan])
    gan_rel = mean([r.get("rel_meas_error") for r in scr_gan])
    post_null = mean([r.get("variance_null_ratio_mean") for r in posterior_rows if r.get("task") == "scr5"])
    pixel_std = mean([r.get("mean_pixel_std") for r in posterior_rows if r.get("task") == "scr5"])
    std_error_corr = mean([r.get("std_error_corr_proxy") for r in posterior_rows if r.get("task") == "scr5"])
    metric_rows = [
        {
            "metric": "PSNR",
            "mean_mode_scr5": mean_psnr,
            "sampling_mode_scr5": gan_psnr,
            "sampling_minus_mean": gan_psnr - mean_psnr,
            "status": "available_from_optional_gan_results",
        },
        {
            "metric": "SSIM",
            "mean_mode_scr5": mean_ssim,
            "sampling_mode_scr5": gan_ssim,
            "sampling_minus_mean": gan_ssim - mean_ssim,
            "status": "available_from_optional_gan_results",
        },
        {
            "metric": "RelMeasErr",
            "mean_mode_scr5": mean_rel,
            "sampling_mode_scr5": gan_rel,
            "sampling_minus_mean": gan_rel - mean_rel,
            "status": "certificate_controlled" if gan_rel <= mean_rel * 1.05 else "check_residual",
        },
        {
            "metric": "mean_pixel_std",
            "mean_mode_scr5": "",
            "sampling_mode_scr5": pixel_std,
            "sampling_minus_mean": "",
            "status": "posterior_proxy_available",
        },
        {
            "metric": "null_variance_ratio",
            "mean_mode_scr5": "",
            "sampling_mode_scr5": post_null,
            "sampling_minus_mean": "",
            "status": "posterior_proxy_available",
        },
        {
            "metric": "std_error_corr_proxy",
            "mean_mode_scr5": "",
            "sampling_mode_scr5": std_error_corr,
            "sampling_minus_mean": "",
            "status": "posterior_proxy_available",
        },
        {
            "metric": "LPIPS",
            "mean_mode_scr5": "",
            "sampling_mode_scr5": "",
            "sampling_minus_mean": "",
            "status": "unavailable_no_individual_samples_or_lpips_eval",
        },
        {
            "metric": "FID/KID",
            "mean_mode_scr5": "",
            "sampling_mode_scr5": "",
            "sampling_minus_mean": "",
            "status": "insufficient_samples_grid_only",
        },
        {
            "metric": "pairwise_sample_LPIPS_diversity",
            "mean_mode_scr5": "",
            "sampling_mode_scr5": "",
            "sampling_minus_mean": "",
            "status": "insufficient_individual_samples",
        },
    ]
    write_csv(out / "g1_key_metric_table.csv", metric_rows)
    write_markdown_table(out / "g1_key_metric_table.md", metric_rows, "G1 Key Metric Table")
    cert_rows = [
        {
            "check": "aggregate_sampling_relmeaserr",
            "value": gan_rel,
            "reference_mean_relmeaserr": mean_rel,
            "status": "controlled" if gan_rel <= mean_rel * 1.05 else "not_controlled",
            "note": "Per stochastic sample residuals cannot be recomputed because individual sample tensors/images are not saved.",
        },
        {
            "check": "audit_after_sampling",
            "value": "",
            "reference_mean_relmeaserr": "",
            "status": "not_recomputed",
            "note": "No individual stochastic outputs available; relying on stored aggregate optional_gan_results.",
        },
    ]
    write_csv(out / "certificate_invariance_check.csv", cert_rows)
    write_markdown_table(out / "certificate_invariance_check.md", cert_rows, "G1 Certificate Invariance Check")
    write_csv(out / "missing_file_report.csv", missing)
    missing_lines = ["# G1 Missing File Report", ""]
    if missing:
        missing_lines.append("|item|path|status|")
        missing_lines.append("|---|---|---|")
        for row in missing:
            missing_lines.append(f"|{row['item']}|{row['path']}|{row['status']}|")
    else:
        missing_lines.append("No required files are missing.")
    (out / "missing_file_report.md").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")
    enough_samples = bool(tensor_samples or len(stochastic_image_files) >= 32)
    diversity_nontrivial = not math.isnan(pixel_std) and pixel_std > 5e-4
    null_controlled = not math.isnan(post_null) and post_null < 0.15
    rel_controlled = not math.isnan(gan_rel) and gan_rel <= mean_rel * 1.05
    psnr_budget_ok = not math.isnan(gan_psnr) and gan_psnr >= mean_psnr - 0.5
    has_perceptual = False
    if not scr_gan:
        decision = "do_not_cite"
        reason = "No successful Scr-5 GAN sampling-mode rows were found."
    elif not enough_samples:
        decision = "supplement_exploratory_only"
        reason = "Only aggregate metrics and grids are available; individual samples are insufficient for LPIPS/FID/KID."
    elif diversity_nontrivial and null_controlled and rel_controlled and has_perceptual:
        decision = "optional_short_main_discussion"
        reason = "Strong diversity/certificate/perceptual evidence exists."
    elif diversity_nontrivial and null_controlled and rel_controlled:
        decision = "supplement_only"
        reason = "Diversity and certificate proxies exist, but perceptual metrics are unavailable."
    else:
        decision = "omit_gan_from_paper"
        reason = "No clear perceptual/diversity benefit is established."
    report = [
        "# G1 Sampling-Mode Zero-Training Evaluation",
        "",
        "Scope: exploratory sampling-mode evaluation of the existing Scr-5 optional GAN pilot. No model was trained, fine-tuned, or modified.",
        "",
        "## Answers",
        "",
        f"1. Nontrivial diversity: {'yes, weak proxy' if diversity_nontrivial else 'not established'}; mean pixel std = {fmt(pixel_std, 6)}.",
        f"2. Diversity mostly in null-space: {'yes by proxy' if null_controlled else 'not established'}; mean null variance ratio = {fmt(post_null, 5)}.",
        f"3. RelMeasErr controlled by audit/certificate: {'yes by aggregate metric' if rel_controlled else 'not clearly'}; sampling RelMeasErr = {fmt(gan_rel, 5)}, mean-mode = {fmt(mean_rel, 5)}.",
        f"4. PSNR sampling-mode budget: {'ok' if psnr_budget_ok else 'not ok'}; sampling PSNR = {fmt(gan_psnr, 3)}, mean-mode = {fmt(mean_psnr, 3)}.",
        "5. LPIPS/FID/KID: unavailable / insufficient individual samples.",
        f"6. Supplement evidence: {decision}; {reason}",
        "7. Further GAN training: no.",
        "",
        "GAN is not a certificate, not a main contribution, and not recommended for title/abstract/main method framing.",
    ]
    (out / "G1_SAMPLING_MODE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision_lines = [
        "# G1 GAN Include Or Not",
        "",
        f"Decision: **{decision}**.",
        "",
        f"Reason: {reason}",
        "",
        "- Do not use GAN as main method.",
        "- Do not call GAN/D a certificate/evaluator.",
        "- If included, include only as exploratory sampling-mode supplement.",
        "- No further GAN training is recommended.",
    ]
    (out / "G1_GAN_INCLUDE_OR_NOT.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    save_json(
        {
            "phase": 59,
            "task": "G1_sampling_mode_zero_training_eval",
            "output_dir": str(out),
            "input_dir": str(GAN_ROOT),
            "mean_reference_dir": str(MEAN_ROOT),
            "decision": decision,
            "no_training": True,
            "scr5_gan_rows": len(scr_gan),
            "rad_gan_rows": len(rad_gan),
            "individual_samples_available": enough_samples,
        },
        out / "G1_MANIFEST.json",
    )
    write_environment(out)
    write_sha256s(out)
    print(out / "G1_SAMPLING_MODE_REPORT.md")


if __name__ == "__main__":
    main()
