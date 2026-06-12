from __future__ import annotations

from typing import Any

from .phase60_common import (
    MEAN_CHECKPOINT,
    MEAN_CONFIG,
    MEAN_METRICS,
    MEAN_ROOT,
    OUT_ROOT,
    ensure_dir,
    find_data_split_hash_files,
    load_config_or_empty,
    read_json,
    save_config,
    save_json,
    write_rows,
)


CRITICAL_MEASUREMENT_KEYS = [
    "seed",
    "dataset_name",
    "img_size",
    "sampling_ratio",
    "pattern_type",
    "matrix_normalization",
    "hadamard_include_dc",
    "hadamard_skip_dc",
    "hadamard_random_column_permutation",
    "hadamard_random_row_permutation",
    "hadamard_row_order",
    "backprojection_mode",
    "noise_std",
    "lambda_solver",
    "use_null_project",
    "use_dc_project",
    "use_final_dc_project",
    "output_range_mode",
]


def main() -> None:
    out = ensure_dir(OUT_ROOT)
    config_dir = ensure_dir(out / "configs")
    provenance = read_json(out / "phase60_provenance_status.json")
    mean_cfg = load_config_or_empty(MEAN_CONFIG)
    split_files, ignored_transfer_manifests = find_data_split_hash_files(MEAN_ROOT)

    critical_config = {key: mean_cfg.get(key) for key in CRITICAL_MEASUREMENT_KEYS}
    measurement_config_confirmed = bool(
        mean_cfg
        and mean_cfg.get("dataset_name") == "stl10"
        and float(mean_cfg.get("sampling_ratio", -1)) == 0.05
        and mean_cfg.get("pattern_type") == "lowfreq_hadamard"
        and bool(mean_cfg.get("hadamard_random_column_permutation")) is True
        and bool(mean_cfg.get("hadamard_random_row_permutation")) is True
    )

    reasons: list[str] = []
    if not MEAN_CHECKPOINT.exists():
        reasons.append("Published mean-mode Scr-5 checkpoint is missing.")
    if not MEAN_CONFIG.exists():
        reasons.append("Published mean-mode Scr-5 resolved_config.yaml is missing.")
    if not MEAN_METRICS.exists():
        reasons.append("Published mean-mode Scr-5 eval_metrics.json is missing.")
    if not provenance.get("checkpoint_sha_match", False):
        reasons.append("G1 provenance audit could not confirm checkpoint SHA consistency.")
    if not split_files:
        reasons.append("No saved train/val/test split hash files were found for the main no-leak run.")
    if not measurement_config_confirmed:
        reasons.append("Scr-5 measurement/exact config could not be confirmed from resolved_config.yaml.")

    safe_to_run = not reasons

    g2_config: dict[str, Any] = {
        "phase": 60,
        "task": "controlled_null_gauge_gan_sampling_mode_g2",
        "scope": "Scr-5 only; supplement exploratory sampling-mode evaluation",
        "source_mean_checkpoint": str(MEAN_CHECKPOINT),
        "source_mean_config": str(MEAN_CONFIG),
        "source_mean_metrics": str(MEAN_METRICS),
        "output_root": str(out),
        "base_config": mean_cfg,
        "critical_measurement_config": critical_config,
        "safety": {
            "safe_to_run": safe_to_run,
            "unsafe_to_run": not safe_to_run,
            "reasons": reasons,
            "split_hash_files": [str(p) for p in split_files],
            "ignored_transfer_manifests": [str(p) for p in ignored_transfer_manifests],
        },
        "gan_branch": {
            "architecture": "same generator/refiner as published mean-mode Scr-5; add null-gauge adversarial sampling branch only",
            "beta_grid": [1e-5, 3e-5, 1e-4],
            "gamma_grid": [0.01, 0.05],
            "pilot_if_grid_too_large": {"beta": 3e-5, "gamma": 0.05},
            "loss": "L_G = L_img + alpha L_meas + beta L_adv_null + gamma L_div",
            "diversity_loss": "-mean_i std_z(P0 xhat_i(z))",
            "fixed_z_seeds": list(range(16)),
            "stochastic_samples_per_y": 8,
            "save_all_individual_stochastic_samples": True,
            "no_test_based_checkpoint_selection": True,
        },
        "discriminator_visibility": {
            "allowed": ["P0 xhat", "x_data", "optional projected null component features"],
            "forbidden": ["full image xhat", "Au-y", "RelMeasErr", "delta", "Pi_y(xhat)-xhat", "audit correction"],
        },
        "audit": {
            "same_pi_y_lambda_as_main": True,
            "certificate_column_after_audit": True,
            "rademacher_gan": False,
            "main_result_table_changes": False,
        },
    }
    save_config(g2_config, config_dir / "phase60_g2_scr5_null_gauge.yaml")

    split_status = {
        "status": "confirmed" if split_files else "unconfirmed_missing_saved_split_hashes",
        "split_hash_files": [str(p) for p in split_files],
        "ignored_transfer_manifests": [str(p) for p in ignored_transfer_manifests],
        "note": "Package split manifests are not data split hashes and are not accepted as train/val/test provenance.",
    }
    save_json(split_status, out / "train_val_test_split_hashes.json")

    safety_rows = [
        {"check": "mean_checkpoint_exists", "value": MEAN_CHECKPOINT.exists(), "status": "pass" if MEAN_CHECKPOINT.exists() else "fail"},
        {"check": "mean_config_exists", "value": MEAN_CONFIG.exists(), "status": "pass" if MEAN_CONFIG.exists() else "fail"},
        {"check": "mean_metrics_exists", "value": MEAN_METRICS.exists(), "status": "pass" if MEAN_METRICS.exists() else "fail"},
        {"check": "g1_checkpoint_sha_match", "value": provenance.get("checkpoint_sha_match", False), "status": "pass" if provenance.get("checkpoint_sha_match", False) else "fail"},
        {"check": "train_val_test_split_hashes_saved", "value": len(split_files), "status": "pass" if split_files else "fail"},
        {"check": "scr5_measurement_config_confirmed", "value": measurement_config_confirmed, "status": "pass" if measurement_config_confirmed else "fail"},
        {"check": "safe_to_run_g2_training", "value": safe_to_run, "status": "pass" if safe_to_run else "unsafe_to_run"},
    ]
    write_rows(out / "g2_safety_checks", safety_rows, "Phase60 G2 Safety Checks")
    safety_status = {
        "phase": 60,
        "safe_to_run": safe_to_run,
        "unsafe_to_run": not safe_to_run,
        "reasons": reasons,
        "g2_config": str(config_dir / "phase60_g2_scr5_null_gauge.yaml"),
        "split_status": split_status,
        "measurement_config_confirmed": measurement_config_confirmed,
        "main_results_unchanged": True,
    }
    save_json(safety_status, out / "g2_safety_status.json")

    lines = [
        "# G2 Controlled Config Preparation",
        "",
        f"Safety decision: **{'safe_to_run' if safe_to_run else 'unsafe_to_run'}**",
        "",
        "The prepared config is written for traceability, but training is gated by `g2_safety_status.json`.",
        "",
        "## Reasons",
        "",
    ]
    lines.extend([f"- {reason}" for reason in reasons] or ["- None."])
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- G2 config: `{config_dir / 'phase60_g2_scr5_null_gauge.yaml'}`",
            f"- Safety status: `{out / 'g2_safety_status.json'}`",
            f"- Split hash status: `{out / 'train_val_test_split_hashes.json'}`",
        ]
    )
    (out / "G2_CONFIG_PREPARATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
