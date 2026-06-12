from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from .phase26_common import REPO_ROOT, drive_root, ensure_dir, markdown_table, output_root, write_csv, write_json, write_text
from .utils import apply_experiment_defaults, load_config, save_config


CONFIG_DIR = REPO_ROOT / "configs" / "phase26_arch_pilot"

BASE_CONFIGS = {
    "rad5": REPO_ROOT / "configs" / "phase14_colab" / "rademacher5_hq_noise001_colab.yaml",
    "scr5": REPO_ROOT / "configs" / "phase14_colab" / "scrambled_hadamard5_hq_noise001_colab.yaml",
}

FAMILIES = {
    "rad5": {
        "suffix": "rad5",
        "family": "rademacher",
        "method_id": "rademacher5_hq_noise001_colab",
        "exact_A_required": True,
    },
    "scr5": {
        "suffix": "scr5",
        "family": "scrambled_hadamard",
        "method_id": "scrambled_hadamard5_hq_noise001_colab",
        "exact_A_required": False,
    },
}

ARCHES = [
    {
        "name": "current_hq",
        "model_type": "hq_unet",
        "base_channels": 64,
        "note": "Current high-quality stage-1 generator; no Phase14 refiner.",
    },
    {
        "name": "nafnet_small",
        "model_type": "nafnet_small",
        "base_channels": 48,
        "nafnet_channels": 48,
        "nafnet_blocks": 8,
        "note": "Small NAFNet-style residual proposer.",
    },
    {
        "name": "unrolled_ista",
        "model_type": "unrolled_ista",
        "base_channels": 48,
        "unrolled_ista_steps": 5,
        "note": "Learned unrolled ISTA-style residual proposer.",
    },
]

FIELDS = [
    "config_name",
    "path",
    "output_dir",
    "family",
    "model_type",
    "epochs",
    "limit_train_samples",
    "limit_val_samples",
    "exact_A_required",
    "measurement_operator_exact_path",
]


def join_text(root: str | Path, *parts: str) -> str:
    text = str(root).replace("\\", "/").rstrip("/")
    return "/".join([text, *parts])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 26 medium architecture pilot configs.")
    parser.add_argument("--drive_root", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--limit_train_samples", type=int, default=20000)
    parser.add_argument("--limit_val_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_amp", choices=["true", "false"], default="true")
    return parser.parse_args()


def build_config(base: dict[str, Any], arch: dict[str, Any], family_key: str, root: Path, root_text: str, args: argparse.Namespace) -> dict[str, Any]:
    family = FAMILIES[family_key]
    name = f"{arch['name']}_{family['suffix']}_pilot"
    config = deepcopy(base)
    config["experiment_name"] = name
    config["dataset_root"] = join_text(root_text, "data")
    config["output_dir"] = join_text(root_text, "outputs_phase26", "arch_pilot", name)
    config["device"] = str(args.device)
    config["epochs"] = int(args.epochs)
    config["limit_train_samples"] = int(args.limit_train_samples)
    config["limit_val_samples"] = int(args.limit_val_samples)
    config["batch_size"] = int(args.batch_size)
    config["num_workers"] = int(args.num_workers)
    config["use_amp"] = args.use_amp == "true"
    config["use_ema"] = True
    config["eval_before_training"] = True
    config["checkpoint_metric_mode"] = "hq"
    config["output_range_mode"] = "clamp_eval_only"
    config["model_type"] = arch["model_type"]
    config["base_channels"] = int(arch["base_channels"])
    config["phase26_medium_pilot"] = True
    config["phase26_not_final_paper_result"] = True
    config["phase26_fixed_outer_formula"] = "x_hat = Pi_y[x_data + P_N(G_theta(x_data))]"
    config["phase26_architecture_note"] = arch["note"]
    config["phase26_measurement_lock"] = {
        "same_A": True,
        "same_split": True,
        "same_loss": True,
        "same_budget": True,
        "source_method_id": family["method_id"],
        "family": family["family"],
        "exact_A_required": family["exact_A_required"],
    }
    config["exact_A_required"] = bool(family["exact_A_required"])
    if family["exact_A_required"]:
        path = join_text(
            root_text,
            "outputs_phase15",
            "imported_noleak",
            family["method_id"],
            "measurement_operator_exact.pt",
        )
        config["measurement_operator_exact_path"] = path
        config["phase26_measurement_lock"]["exact_A_path"] = path
    else:
        config["measurement_operator_exact_path"] = ""
    for key in ["nafnet_channels", "nafnet_blocks", "unrolled_ista_steps"]:
        if key in arch:
            config[key] = arch[key]
    if arch["model_type"] == "hq_unet":
        config["training_stage"] = {
            "stage1_epochs": int(args.epochs),
            "refiner_start_epoch": 999999,
            "adversarial_start_epoch": 999999,
        }
    return apply_experiment_defaults(config)


def main() -> None:
    args = parse_args()
    root = drive_root(args.drive_root)
    root_text = str(args.drive_root or root).replace("\\", "/").rstrip("/")
    ensure_dir(CONFIG_DIR)
    drive_available = root.exists()
    if drive_available:
        ensure_dir(output_root(root) / "arch_pilot")
    records = []
    for family_key in ["rad5", "scr5"]:
        base = apply_experiment_defaults(load_config(BASE_CONFIGS[family_key]))
        for arch in ARCHES:
            config = build_config(base, arch, family_key, root, root_text, args)
            path = CONFIG_DIR / f"{config['experiment_name']}.yaml"
            save_config(config, path)
            records.append(
                {
                    "config_name": config["experiment_name"],
                    "path": str(path),
                    "output_dir": config["output_dir"],
                    "family": FAMILIES[family_key]["family"],
                    "model_type": config["model_type"],
                    "epochs": config["epochs"],
                    "limit_train_samples": config["limit_train_samples"],
                    "limit_val_samples": config["limit_val_samples"],
                    "exact_A_required": config["exact_A_required"],
                    "measurement_operator_exact_path": config.get("measurement_operator_exact_path", ""),
                }
            )
    out = output_root(root) if drive_available else CONFIG_DIR
    write_csv(out / "arch_pilot_config_manifest.csv", records, FIELDS)
    write_text(
        out / "arch_pilot_config_manifest.md",
        "# Phase 26 Architecture Pilot Config Manifest\n\n"
        + "These are medium pilot configs, not final paper results.\n\n"
        + markdown_table(records, FIELDS),
    )
    write_json(
        out / "arch_pilot_config_manifest.json",
        {
            "phase": 26,
            "drive_root": str(root),
            "config_dir": str(CONFIG_DIR),
            "records": records,
        },
    )
    print({"configs": len(records), "manifest": str(out / "arch_pilot_config_manifest.csv")})


if __name__ == "__main__":
    main()
