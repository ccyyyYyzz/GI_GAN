from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .phase48_49_common import write_environment, write_sha256s
from .phase60_common import OUT_ROOT, ensure_dir, read_csv_rows, read_json, save_json


def _first_metric(rows: list[dict[str, str]], mode: str, key: str) -> str:
    for row in rows:
        if row.get("mode") == mode:
            return str(row.get(key, ""))
    return ""


def main() -> None:
    out = ensure_dir(OUT_ROOT)
    provenance = read_json(out / "phase60_provenance_status.json")
    safety = read_json(out / "g2_safety_status.json")
    training = read_json(out / "g2_training_status.json")
    eval_status = read_json(out / "g2_eval_status.json")
    sampling_rows = read_csv_rows(out / "g2_sampling_results.csv")
    perception_rows = read_csv_rows(out / "g2_perception_metrics.csv")
    kappa_rows = read_csv_rows(out / "g2_kappa_results.csv")

    mean_ref = eval_status.get("mean_reference", {})
    g1_ref = eval_status.get("g1_postmortem_reference", {})
    g2_status = training.get("status", "missing_training_status")
    safe_to_run = safety.get("safe_to_run", False)
    recommendation = "omit_gan_from_main_and_supplement_g2_not_run"
    if g2_status == "skipped_unsafe_to_run":
        recommendation = "omit_controlled_g2; mention only that safety gate blocked follow-up if needed"
    elif g2_status == "blocked_manual_review_required":
        recommendation = "do_not_include_until_manual_review_and_completed_controlled_eval"

    report_lines = [
        "# Phase60 Controlled Null-Gauge GAN Sampling Mode G2",
        "",
        "Scope: Scr-5 only. GAN is treated only as exploratory sampling-mode evaluation of the unmeasured prior.",
        "",
        "## Executive Decision",
        "",
        f"- G1 provenance conclusion: **{provenance.get('conclusion', 'missing')}**",
        f"- G2 training status: **{g2_status}**",
        f"- Safe to run G2 training: **{safe_to_run}**",
        f"- Recommendation: **{recommendation}**",
        "- Main result tables and checkpoints were not changed.",
        "",
        "## Metrics",
        "",
        "|mode|PSNR|SSIM|RelMeasErr|status|",
        "|---|---:|---:|---:|---|",
        f"|published mean Scr-5|{mean_ref.get('psnr', '')}|{mean_ref.get('ssim', '')}|{mean_ref.get('rel_meas_error', '')}|reference unchanged|",
        f"|G1 postmortem pilot|{g1_ref.get('psnr', '')}|{g1_ref.get('ssim', '')}|{g1_ref.get('rel_meas_error', '')}|postmortem only|",
        f"|G2 controlled sampling||||{g2_status}|",
        "",
        "## Diversity And Null-Space Diagnostics",
        "",
        f"- G1 kappa proxy: `{g1_ref.get('kappa_proxy', '')}`. This is invalid as sampling-mode evidence when below 1.",
        f"- G1 mean pixel std proxy: `{g1_ref.get('mean_pixel_std', '')}`.",
        f"- G1 null variance ratio proxy: `{g1_ref.get('null_variance_ratio', '')}`.",
        "- G2 kappa/std/null-variance metrics are unavailable because controlled G2 training was skipped by the safety gate.",
        "",
        "## LPIPS / FID / KID",
        "",
        "- LPIPS, FID, and KID are unavailable for G2 because no controlled individual stochastic samples were produced.",
        "- G1 did not contain sufficient individual stochastic samples for claim-ready LPIPS/FID/KID.",
        "",
        "## Required Questions",
        "",
        "1. Was G1 PSNR advantage a provenance/config artifact? It cannot be ruled out; G1 is post-mortem only.",
        "2. Did G2 start from the published mean-mode checkpoint? No G2 training was started. The source checkpoint was identified, but the safety gate stopped before optimization.",
        "3. Did G2 preserve RelMeasErr/certificate? Not evaluated; no G2 samples.",
        "4. Did G2 produce nontrivial diversity? Not evaluated; no G2 samples.",
        "5. Is diversity mostly null-space? Not evaluated for G2.",
        "6. Is kappa in [1,2]? Unavailable for G2; G1 proxy is below 1 and invalid.",
        "7. Does observed PSNR drop match -10 log10(kappa)? Not evaluated for G2.",
        "8. Are LPIPS/FID/KID available and improved? No.",
        "9. Should sampling mode enter supplement? No controlled G2 evidence is available; omit for now.",
        "10. Should any GAN appear in main text? No.",
        "11. Should further GAN training continue? Only after explicit approval and saved train/val/test split hashes are available.",
        "",
        "## Safety Reasons",
        "",
    ]
    reasons = safety.get("reasons", [])
    report_lines.extend([f"- {reason}" for reason in reasons] or ["- None."])
    report_lines.extend(
        [
            "",
            "## Generated Outputs",
            "",
            "- `G1_PROVENANCE_AUDIT.md`",
            "- `g1_vs_mean_config_diff.csv`",
            "- `configs/phase60_g2_scr5_null_gauge.yaml`",
            "- `g2_safety_status.json`",
            "- `g2_training_status.json`",
            "- `g2_sampling_results.csv/md`",
            "- `g2_kappa_results.csv/md`",
            "- `g2_certificate_metrics.csv/md`",
            "- `g2_perception_metrics.csv/md`",
            "- `g2_sample_grid.png/pdf`",
            "- `g2_uncertainty_map.png/pdf`",
            "- `g2_perception_distortion_curve.png`",
            "- `g2_null_variance_ratio.png`",
        ]
    )
    (out / "PHASE60_G2_SAMPLING_MODE_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    include_lines = [
        "# Phase60 GAN Include Or Not",
        "",
        f"Decision: **{recommendation}**",
        "",
        "Rationale:",
        "",
        "- G1 is not comparable as sampling-mode evidence.",
        "- G2 was not trained because the strict provenance gate was unsafe.",
        "- No claim-ready individual stochastic samples, LPIPS/FID/KID, or controlled kappa evidence exist.",
        "- GAN must not appear in the title, abstract, main contribution, or main result table.",
        "",
        "Recommended manuscript handling: omit GAN from the main paper. At most, keep a private post-mortem note unless a future controlled G2 run passes the safety gate.",
    ]
    (out / "PHASE60_GAN_INCLUDE_OR_NOT.md").write_text("\n".join(include_lines) + "\n", encoding="utf-8")

    postmortem_lines = [
        "# Phase60 G1 Postmortem",
        "",
        f"Conclusion: **{provenance.get('conclusion', 'missing')}**",
        "",
        f"- G1 sampling PSNR - mean PSNR: `{provenance.get('g1_sampling_minus_mean_psnr', '')}`.",
        f"- G1 kappa proxy: `{provenance.get('g1_kappa_proxy', '')}`.",
        f"- G1 individual stochastic sample count: `{provenance.get('individual_stochastic_sample_count', '')}`.",
        f"- Split hash confirmed: `{provenance.get('split_hash_confirmed', '')}`.",
        "",
        "Use G1 only as a diagnostic of why uncontrolled GAN pilots can mislead. Do not use it as sampling-mode success evidence.",
    ]
    (out / "PHASE60_G1_POSTMORTEM.md").write_text("\n".join(postmortem_lines) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "phase": 60,
        "output_root": str(out),
        "g1_conclusion": provenance.get("conclusion"),
        "g2_training_status": g2_status,
        "safe_to_run": safe_to_run,
        "recommendation": recommendation,
        "main_results_unchanged": True,
        "metrics": {
            "mean_reference": mean_ref,
            "g1_postmortem_reference": g1_ref,
            "g2_controlled": {"status": g2_status},
        },
        "generated_files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
    }
    save_json(manifest, out / "PHASE60_MANIFEST.json")
    write_environment(out)
    write_sha256s(out, out / "SHA256SUMS.txt")


if __name__ == "__main__":
    main()
