from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Any

from .phase60_common import (
    G1_CONFIG,
    G1_PILOT_ROOT,
    G1_ROOT,
    G1_SCR5_ROOT,
    G1_SOURCE_CHECKPOINT,
    MEAN_CHECKPOINT,
    MEAN_CONFIG,
    MEAN_METRICS,
    MEAN_ROOT,
    OUT_ROOT,
    ensure_dir,
    find_data_split_hash_files,
    find_individual_stochastic_samples,
    flatten_config,
    fmt,
    load_config_or_empty,
    maybe_sha256,
    mean,
    read_csv_rows,
    read_json,
    save_json,
    to_float,
    write_rows,
)


def _grep_flag(text: str, flag: str) -> str:
    match = re.search(rf"{re.escape(flag)}\s+([^\s]+)", text)
    return match.group(1) if match else ""


def _copy_reference_files(out: Path) -> None:
    refs = out / "g1_reference_files"
    refs.mkdir(parents=True, exist_ok=True)
    for source in [
        G1_ROOT / "g1_key_metric_table.csv",
        G1_ROOT / "G1_SAMPLING_MODE_REPORT.md",
        G1_ROOT / "G1_GAN_INCLUDE_OR_NOT.md",
        G1_PILOT_ROOT / "optional_gan_results.csv",
        G1_PILOT_ROOT / "posterior_sampling_metrics.csv",
        G1_PILOT_ROOT / "OPTIONAL_GAN_POSTERIOR_REPORT.md",
        G1_PILOT_ROOT / "command_log.txt",
    ]:
        if source.exists():
            shutil.copy2(source, refs / source.name)


def main() -> None:
    out = ensure_dir(OUT_ROOT)
    _copy_reference_files(out)

    mean_cfg = load_config_or_empty(MEAN_CONFIG)
    g1_cfg = load_config_or_empty(G1_CONFIG)
    mean_flat = flatten_config(mean_cfg)
    g1_flat = flatten_config(g1_cfg)
    keys = sorted(set(mean_flat) | set(g1_flat))
    diff_rows: list[dict[str, Any]] = []
    for key in keys:
        mean_value = mean_flat.get(key, "")
        g1_value = g1_flat.get(key, "")
        diff_rows.append(
            {
                "key": key,
                "mean_mode_value": mean_value,
                "g1_value": g1_value,
                "same": mean_value == g1_value,
            }
        )
    write_rows(out / "g1_vs_mean_config_diff", diff_rows, "G1 vs Published Mean-Mode Scr-5 Config Diff")

    optional_rows = read_csv_rows(G1_PILOT_ROOT / "optional_gan_results.csv")
    posterior_rows = read_csv_rows(G1_PILOT_ROOT / "posterior_sampling_metrics.csv")
    g1_key_rows = read_csv_rows(G1_ROOT / "g1_key_metric_table.csv")
    mean_metrics = read_json(MEAN_METRICS)
    manifest = read_json(G1_PILOT_ROOT / "MANIFEST.json")
    command_log = (G1_PILOT_ROOT / "command_log.txt").read_text(encoding="utf-8", errors="ignore") if (G1_PILOT_ROOT / "command_log.txt").exists() else ""

    mean_ckpt_sha = maybe_sha256(MEAN_CHECKPOINT)
    g1_source_sha = maybe_sha256(G1_SOURCE_CHECKPOINT)
    g1_bundle_sha = maybe_sha256(G1_SCR5_ROOT / "_source_bundle_leaf" / "source_checkpoint_last.pt")
    checkpoint_sha_match = bool(mean_ckpt_sha and g1_source_sha and mean_ckpt_sha == g1_source_sha)
    bundle_sha_match = bool(mean_ckpt_sha and g1_bundle_sha and mean_ckpt_sha == g1_bundle_sha)

    split_files, ignored_split_manifests = find_data_split_hash_files(MEAN_ROOT, G1_PILOT_ROOT)
    split_hash_confirmed = len(split_files) > 0
    individual_samples = find_individual_stochastic_samples(G1_PILOT_ROOT)

    mean_psnr = to_float(mean_metrics.get("model", {}).get("psnr"))
    sampling_psnr = mean([r.get("psnr") for r in optional_rows if r.get("task") == "scr5" and str(r.get("status", "")).startswith("ran")])
    delta_psnr = sampling_psnr - mean_psnr if not math.isnan(mean_psnr) and not math.isnan(sampling_psnr) else float("nan")
    kappa_proxy = 10 ** (-delta_psnr / 10.0) if not math.isnan(delta_psnr) else float("nan")

    posterior_scr5 = [r for r in posterior_rows if r.get("task") == "scr5"]
    mean_pixel_std = mean([r.get("mean_pixel_std") for r in posterior_scr5])
    null_variance_ratio = mean([r.get("variance_null_ratio_mean") for r in posterior_scr5])

    reasons: list[str] = []
    if not checkpoint_sha_match:
        reasons.append("G1 source checkpoint SHA does not match the published mean-mode Scr-5 checkpoint.")
    if not split_hash_confirmed:
        reasons.append("No saved train/val/test data split hash files were found; only transfer/package split manifests were found.")
    if not individual_samples:
        reasons.append("G1 has aggregate CSVs and grids, but no individual stochastic sample tensors/images.")
    if not math.isnan(delta_psnr) and delta_psnr > 0:
        reasons.append("G1 sampling PSNR is higher than mean-mode PSNR, so it is not credible as a sampling-budget tradeoff.")
    if not math.isnan(mean_pixel_std) and mean_pixel_std < 0.002:
        reasons.append("G1 stochastic diversity proxy is tiny, consistent with z-collapse or deterministic behavior.")

    g1_comparable = checkpoint_sha_match and split_hash_confirmed and bool(individual_samples) and delta_psnr <= 0
    conclusion = (
        "G1 sampling PSNR is not comparable; use only as post-mortem."
        if not g1_comparable
        else "G1 provenance is comparable enough for a controlled follow-up."
    )

    audit_rows = [
        {"item": "G1 source checkpoint", "value": str(G1_SOURCE_CHECKPOINT), "status": "exists" if G1_SOURCE_CHECKPOINT.exists() else "missing"},
        {"item": "Published mean checkpoint", "value": str(MEAN_CHECKPOINT), "status": "exists" if MEAN_CHECKPOINT.exists() else "missing"},
        {"item": "checkpoint_sha_match", "value": checkpoint_sha_match, "status": "pass" if checkpoint_sha_match else "fail"},
        {"item": "source_bundle_sha_match", "value": bundle_sha_match, "status": "pass" if bundle_sha_match else "fail_or_missing"},
        {"item": "epoch_count", "value": _grep_flag(command_log, "--critic_epochs"), "status": "from_command_log"},
        {"item": "training_budget_max_steps", "value": _grep_flag(command_log, "--max_steps"), "status": "from_command_log"},
        {"item": "num_samples_per_y", "value": _grep_flag(command_log, "--num_samples_per_y"), "status": "from_command_log"},
        {"item": "loss_beta_grid", "value": ",".join(sorted({r.get("beta", "") for r in optional_rows if r.get("task") == "scr5"})), "status": "from_optional_gan_results"},
        {"item": "split_hash_confirmed", "value": split_hash_confirmed, "status": "pass" if split_hash_confirmed else "unconfirmed"},
        {"item": "individual_stochastic_samples", "value": len(individual_samples), "status": "available" if individual_samples else "missing"},
        {"item": "G1 trains generator", "value": manifest.get("trains_generator", ""), "status": "posthoc_optional_pilot"},
        {"item": "mean_psnr", "value": fmt(mean_psnr, 6), "status": "reference"},
        {"item": "g1_sampling_psnr", "value": fmt(sampling_psnr, 6), "status": "posthoc"},
        {"item": "g1_sampling_minus_mean_psnr", "value": fmt(delta_psnr, 6), "status": "invalid_tradeoff" if delta_psnr > 0 else "ok"},
        {"item": "g1_kappa_proxy", "value": fmt(kappa_proxy, 6), "status": "invalid_lt_1" if kappa_proxy < 1 else "not_invalid"},
        {"item": "g1_mean_pixel_std", "value": fmt(mean_pixel_std, 8), "status": "tiny" if mean_pixel_std < 0.002 else "nontrivial"},
        {"item": "g1_null_variance_ratio", "value": fmt(null_variance_ratio, 6), "status": "proxy_only"},
    ]
    write_rows(out / "g1_provenance_audit_table", audit_rows, "Phase60 G1 Provenance Audit Table")

    status = {
        "phase": 60,
        "task": "G1 provenance audit",
        "conclusion": conclusion,
        "g1_comparable_to_mean_mode": g1_comparable,
        "checkpoint_sha_match": checkpoint_sha_match,
        "mean_checkpoint_sha256": mean_ckpt_sha,
        "g1_source_checkpoint_sha256": g1_source_sha,
        "g1_source_bundle_checkpoint_sha256": g1_bundle_sha,
        "source_bundle_sha_match": bundle_sha_match,
        "split_hash_confirmed": split_hash_confirmed,
        "split_hash_files": [str(p) for p in split_files],
        "ignored_transfer_manifests": [str(p) for p in ignored_split_manifests],
        "individual_stochastic_sample_count": len(individual_samples),
        "mean_mode_metrics": mean_metrics.get("model", {}),
        "g1_sampling_psnr": sampling_psnr,
        "g1_sampling_minus_mean_psnr": delta_psnr,
        "g1_kappa_proxy": kappa_proxy,
        "g1_mean_pixel_std": mean_pixel_std,
        "g1_null_variance_ratio": null_variance_ratio,
        "g1_key_metric_rows": g1_key_rows,
        "unsafe_reasons": reasons,
    }
    save_json(status, out / "phase60_provenance_status.json")

    lines = [
        "# G1 Provenance Audit",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        "## Key Findings",
        "",
        f"- Published mean checkpoint exists: `{MEAN_CHECKPOINT.exists()}`.",
        f"- G1 source checkpoint exists: `{G1_SOURCE_CHECKPOINT.exists()}`.",
        f"- Checkpoint SHA match: `{checkpoint_sha_match}`.",
        f"- Data split hashes confirmed: `{split_hash_confirmed}`.",
        f"- Individual stochastic samples found: `{len(individual_samples)}`.",
        f"- G1 sampling PSNR - mean PSNR: `{fmt(delta_psnr, 6)}` dB.",
        f"- G1 kappa proxy: `{fmt(kappa_proxy, 6)}`.",
        f"- G1 mean pixel std: `{fmt(mean_pixel_std, 8)}`.",
        f"- G1 null variance ratio: `{fmt(null_variance_ratio, 6)}`.",
        "",
        "## Why The G1 PSNR Advantage Is Not Evidence",
        "",
        "A sampling mode should normally pay a distortion budget for diversity. Here the posthoc pilot is about 1.074 dB above the mean-mode reference, which gives kappa < 1. That violates the intended interpretation of kappa as extra null-space sampling variance. Combined with missing individual stochastic samples and missing saved split hashes, the G1 aggregate numbers should be treated as a post-mortem diagnostic only.",
        "",
        "## Unsafe / Unconfirmed Items",
        "",
    ]
    if reasons:
        lines.extend([f"- {reason}" for reason in reasons])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Config diff: `{out / 'g1_vs_mean_config_diff.csv'}`",
            f"- Audit table: `{out / 'g1_provenance_audit_table.csv'}`",
            f"- Status JSON: `{out / 'phase60_provenance_status.json'}`",
        ]
    )
    (out / "G1_PROVENANCE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
