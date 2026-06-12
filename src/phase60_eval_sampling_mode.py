from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .phase60_common import (
    G1_ROOT,
    MEAN_METRICS,
    OUT_ROOT,
    ensure_dir,
    fmt,
    read_csv_rows,
    read_json,
    save_json,
    save_pdf_from_png,
    save_placeholder_figure,
    to_float,
    write_rows,
)


def _metric_from_g1(rows: list[dict[str, str]], metric: str, col: str) -> float:
    for row in rows:
        if row.get("metric") == metric:
            return to_float(row.get(col))
    return float("nan")


def main() -> None:
    out = ensure_dir(OUT_ROOT)
    training = read_json(out / "g2_training_status.json")
    safety = read_json(out / "g2_safety_status.json")
    provenance = read_json(out / "phase60_provenance_status.json")
    mean_metrics = read_json(MEAN_METRICS).get("model", {})
    g1_rows = read_csv_rows(G1_ROOT / "g1_key_metric_table.csv")

    mean_psnr = to_float(mean_metrics.get("psnr"))
    mean_ssim = to_float(mean_metrics.get("ssim"))
    mean_rel = to_float(mean_metrics.get("rel_meas_error"))
    g1_psnr = _metric_from_g1(g1_rows, "PSNR", "sampling_mode_scr5")
    g1_ssim = _metric_from_g1(g1_rows, "SSIM", "sampling_mode_scr5")
    g1_rel = _metric_from_g1(g1_rows, "RelMeasErr", "sampling_mode_scr5")
    g1_std = _metric_from_g1(g1_rows, "mean_pixel_std", "sampling_mode_scr5")
    g1_null = _metric_from_g1(g1_rows, "null_variance_ratio", "sampling_mode_scr5")
    delta_psnr = g1_psnr - mean_psnr if not math.isnan(g1_psnr) and not math.isnan(mean_psnr) else float("nan")
    g1_kappa = 10 ** (-delta_psnr / 10.0) if not math.isnan(delta_psnr) else float("nan")

    g2_status = training.get("status", "missing_training_status")
    g2_available = g2_status not in {"skipped_unsafe_to_run", "blocked_manual_review_required", "missing_training_status"}

    sampling_rows: list[dict[str, Any]] = [
        {
            "mode": "published_mean_scr5",
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "rel_meas_error": mean_rel,
            "status": "reference_unchanged",
        },
        {
            "mode": "g1_postmortem_pilot",
            "psnr": g1_psnr,
            "ssim": g1_ssim,
            "rel_meas_error": g1_rel,
            "status": provenance.get("conclusion", "postmortem_reference"),
        },
        {
            "mode": "g2_controlled_sampling",
            "psnr": "",
            "ssim": "",
            "rel_meas_error": "",
            "status": "not_evaluated_" + str(g2_status),
        },
    ]
    write_rows(out / "g2_sampling_results", sampling_rows, "Phase60 G2 Sampling Results")

    kappa_rows = [
        {
            "case": "g1_postmortem_proxy",
            "kappa": g1_kappa,
            "observed_psnr_drop_vs_mean": mean_psnr - g1_psnr if not math.isnan(g1_psnr) and not math.isnan(mean_psnr) else "",
            "predicted_psnr_drop": -10.0 * math.log10(g1_kappa) if not math.isnan(g1_kappa) and g1_kappa > 0 else "",
            "status": "invalid_lt_1" if not math.isnan(g1_kappa) and g1_kappa < 1 else "not_invalid",
        },
        {
            "case": "g2_controlled_sampling",
            "kappa": "",
            "observed_psnr_drop_vs_mean": "",
            "predicted_psnr_drop": "",
            "status": "unavailable_" + str(g2_status),
        },
    ]
    write_rows(out / "g2_kappa_results", kappa_rows, "Phase60 G2 Kappa Results")

    cert_rows = [
        {"mode": "published_mean_scr5", "rel_meas_error": mean_rel, "certificate_status": "reference_unchanged"},
        {"mode": "g1_postmortem_pilot", "rel_meas_error": g1_rel, "certificate_status": "aggregate_only_not_g2_evidence"},
        {"mode": "g2_controlled_sampling", "rel_meas_error": "", "certificate_status": "unavailable_" + str(g2_status)},
    ]
    write_rows(out / "g2_certificate_metrics", cert_rows, "Phase60 G2 Certificate Metrics")

    perception_rows = [
        {"metric": "LPIPS_to_GT", "value": "", "status": "unavailable_no_g2_samples"},
        {"metric": "pairwise_LPIPS_diversity", "value": "", "status": "unavailable_no_g2_samples"},
        {"metric": "FID", "value": "", "status": "insufficient_or_unavailable_no_g2_samples"},
        {"metric": "KID", "value": "", "status": "insufficient_or_unavailable_no_g2_samples"},
        {"metric": "g1_mean_pixel_std_proxy", "value": g1_std, "status": "postmortem_proxy_only"},
        {"metric": "g1_null_variance_ratio_proxy", "value": g1_null, "status": "postmortem_proxy_only"},
    ]
    write_rows(out / "g2_perception_metrics", perception_rows, "Phase60 G2 Perception Metrics")

    fig_body = (
        f"G2 status: {g2_status}\n"
        "No controlled G2 samples were produced.\n"
        "Reason: safety gate did not confirm saved main train/val/test split hashes.\n"
        f"G1 postmortem kappa proxy: {fmt(g1_kappa, 6)}\n"
        f"G1 mean pixel std proxy: {fmt(g1_std, 8)}\n"
        f"G1 null variance ratio proxy: {fmt(g1_null, 6)}"
    )
    figure_specs = {
        "g2_sample_grid.png": "Phase60 G2 Sample Grid",
        "g2_uncertainty_map.png": "Phase60 G2 Uncertainty Map",
        "g2_perception_distortion_curve.png": "Phase60 G2 Perception-Distortion Curve",
        "g2_null_variance_ratio.png": "Phase60 G2 Null Variance Ratio",
    }
    for filename, title in figure_specs.items():
        save_placeholder_figure(out / filename, title, fig_body)
    save_pdf_from_png(out / "g2_sample_grid.png", out / "g2_sample_grid.pdf")
    save_pdf_from_png(out / "g2_uncertainty_map.png", out / "g2_uncertainty_map.pdf")

    eval_status = {
        "phase": 60,
        "g2_training_status": g2_status,
        "g2_samples_available": g2_available,
        "g2_eval_status": "skipped_no_controlled_g2_samples" if not g2_available else "available",
        "mean_reference": {"psnr": mean_psnr, "ssim": mean_ssim, "rel_meas_error": mean_rel},
        "g1_postmortem_reference": {
            "psnr": g1_psnr,
            "ssim": g1_ssim,
            "rel_meas_error": g1_rel,
            "kappa_proxy": g1_kappa,
            "mean_pixel_std": g1_std,
            "null_variance_ratio": g1_null,
        },
        "safety": safety,
        "main_results_unchanged": True,
    }
    save_json(eval_status, out / "g2_eval_status.json")


if __name__ == "__main__":
    main()
