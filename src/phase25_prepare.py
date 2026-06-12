from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .eval import make_measurement
from .metrics import batch_metrics
from .models import build_generator
from .utils import apply_experiment_defaults, load_config, reconstruct_from_measurements, save_config, set_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
E_ROOT = Path("E:/ns_mc_gan_gi")
PHASE25 = E_ROOT / "outputs_phase25"
CONFIG_OUT = REPO_ROOT / "configs" / "phase25_arch_ablation"
ATTRIBUTION_CSV = (
    E_ROOT
    / "outputs_phase16"
    / "supplementary_experiments"
    / "attribution"
    / "attribution_final.csv"
)

SOURCE_CONFIGS = {
    "rad5": REPO_ROOT / "configs" / "phase14_colab" / "rademacher5_hq_noise001_colab.yaml",
    "scr5": REPO_ROOT / "configs" / "phase14_colab" / "scrambled_hadamard5_hq_noise001_colab.yaml",
}

ARCHITECTURES = [
    {
        "name": "current_hq",
        "model_type": "hq_unet",
        "base_channels": 64,
        "note": "Current high-capacity stage-1 generator without the Phase 14 refiner.",
    },
    {
        "name": "unet",
        "model_type": "unet",
        "base_channels": 64,
        "note": "Plain U-Net residual proposer.",
    },
    {
        "name": "resunet",
        "model_type": "resunet",
        "base_channels": 64,
        "note": "Residual U-Net residual proposer.",
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

FAMILIES = {
    "rad5": {
        "suffix": "rad5",
        "method_id": "rademacher5_hq_noise001_colab",
        "measurement_family": "rademacher",
        "exact_A_required": True,
        "exact_A_path": str(
            E_ROOT
            / "outputs_phase15"
            / "imported_noleak"
            / "rademacher5_hq_noise001_colab"
            / "measurement_operator_exact.pt"
        ),
    },
    "scr5": {
        "suffix": "scr5",
        "method_id": "scrambled_hadamard5_hq_noise001_colab",
        "measurement_family": "scrambled_hadamard",
        "exact_A_required": False,
        "exact_A_path": "",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 25 architecture-limit analysis files.")
    parser.add_argument("--output_dir", default=str(PHASE25))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_smoke", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    value = safe_float(value)
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    return ""


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["|" + "|".join(fields) + "|", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(field, "")).replace("|", "/") for field in fields) + "|")
    return "\n".join(lines)


def sampling_scaling_summary(output_dir: Path) -> dict[str, Any]:
    rows = read_csv_rows(ATTRIBUTION_CSV)
    main_rows = [
        row
        for row in rows
        if row.get("dataset") == "STL-10"
        and row.get("measurement_family") in {"rademacher", "scrambled_hadamard"}
        and safe_float(row.get("sampling_ratio")) in {0.05, 0.1}
    ]
    source_rows = []
    fit_rows = []
    for row in sorted(main_rows, key=lambda r: (r.get("measurement_family", ""), safe_float(r.get("sampling_ratio")))):
        source_rows.append(
            {
                "method_id": row.get("method_id", ""),
                "family": row.get("measurement_family", ""),
                "sampling_ratio": fmt(row.get("sampling_ratio"), 2),
                "backproj_psnr": fmt(row.get("backproj_psnr"), 3),
                "model_psnr": fmt(row.get("model_psnr"), 3),
                "model_ssim": fmt(row.get("model_ssim"), 3),
            }
        )

    for family in ["rademacher", "scrambled_hadamard"]:
        family_rows = sorted(
            [row for row in main_rows if row.get("measurement_family") == family],
            key=lambda r: safe_float(r.get("sampling_ratio")),
        )
        if len(family_rows) < 2:
            continue
        x = np.asarray([safe_float(row["sampling_ratio"]) for row in family_rows], dtype=np.float64)
        model_y = np.asarray([safe_float(row["model_psnr"]) for row in family_rows], dtype=np.float64)
        bp_y = np.asarray([safe_float(row["backproj_psnr"]) for row in family_rows], dtype=np.float64)
        model_slope, model_intercept = np.polyfit(x, model_y, 1)
        bp_slope, bp_intercept = np.polyfit(x, bp_y, 1)
        fit_rows.append(
            {
                "family": family,
                "points": len(family_rows),
                "model_fit": f"PSNR = {model_intercept:.3f} + {model_slope:.3f} * ratio",
                "model_gain_per_5pct": fmt(model_slope * 0.05, 3),
                "model_psnr_at_15pct": fmt(model_intercept + model_slope * 0.15, 3),
                "model_psnr_at_20pct": fmt(model_intercept + model_slope * 0.20, 3),
                "backproj_fit": f"PSNR = {bp_intercept:.3f} + {bp_slope:.3f} * ratio",
            }
        )

    fields_source = ["method_id", "family", "sampling_ratio", "backproj_psnr", "model_psnr", "model_ssim"]
    fields_fit = [
        "family",
        "points",
        "model_fit",
        "model_gain_per_5pct",
        "model_psnr_at_15pct",
        "model_psnr_at_20pct",
        "backproj_fit",
    ]
    write_csv_rows(output_dir / "limit_analysis" / "sampling_scaling_points.csv", source_rows, fields_source)
    write_csv_rows(output_dir / "limit_analysis" / "sampling_scaling_fit.csv", fit_rows, fields_fit)
    md = f"""# Phase 25 Sampling Scaling Summary

Source: `{ATTRIBUTION_CSV}`

This is a preliminary two-point fit for STL-10 5% and 10% no-leak main results. It should be read as a planning diagnostic, not as a sampling law.

## Source Points

{markdown_table(source_rows, fields_source)}

## Linear Fit

{markdown_table(fit_rows, fields_fit)}

## Interpretation

- With only 5% and 10% points, the fit is exactly constrained and mostly useful for estimating whether a 10% result is on the same trend as 5%.
- Both Rademacher and scrambled-Hadamard model PSNR increase by about 2.46 dB when moving from 5% to 10%.
- The empirical gap between weak backprojection and strong learned reconstruction confirms that the learned prior, not arbitrary-image inversion, is determining the practical recovery limit.
"""
    write_text(output_dir / "limit_analysis" / "sampling_scaling_summary.md", md)
    return {"source_rows": source_rows, "fit_rows": fit_rows}


def load_family_base(family_key: str) -> dict[str, Any]:
    config = apply_experiment_defaults(load_config(SOURCE_CONFIGS[family_key]))
    config["dataset_root"] = str(E_ROOT / "data")
    config["device"] = "cuda"
    config["num_workers"] = 2
    return config


def phase25_config(base_config: dict[str, Any], arch: dict[str, Any], family_key: str, output_dir: Path) -> dict[str, Any]:
    family = FAMILIES[family_key]
    name = f"{arch['name']}_{family['suffix']}"
    config = deepcopy(base_config)
    config["experiment_name"] = name
    config["output_dir"] = str(output_dir / "architecture_ablation" / name)
    config["model_type"] = arch["model_type"]
    config["base_channels"] = arch["base_channels"]
    config["phase25_do_not_autorun_full_training"] = True
    config["phase25_fixed_outer_formula"] = "x_hat = Pi_y[x_data + P_N(G_theta(x_data))]"
    config["phase25_enable_refiner"] = False
    config["phase25_same_loss_budget_as"] = str(SOURCE_CONFIGS[family_key])
    config["phase25_architecture_note"] = arch["note"]
    config["phase25_measurement_lock"] = {
        "same_A": True,
        "same_split": True,
        "same_loss": True,
        "same_budget": True,
        "measurement_family": family["measurement_family"],
        "source_method_id": family["method_id"],
        "exact_A_required": family["exact_A_required"],
        "exact_A_path": family["exact_A_path"],
        "seed": config.get("seed"),
        "sampling_ratio": config.get("sampling_ratio"),
        "pattern_type": config.get("pattern_type"),
        "matrix_normalization": config.get("matrix_normalization"),
        "hadamard_random_column_permutation": config.get("hadamard_random_column_permutation", False),
        "hadamard_random_row_permutation": config.get("hadamard_random_row_permutation", False),
    }
    for key in ["nafnet_channels", "nafnet_blocks", "unrolled_ista_steps"]:
        if key in arch:
            config[key] = arch[key]
    if arch["model_type"] == "hq_unet":
        config["training_stage"] = {
            "stage1_epochs": int(config.get("epochs", 1)),
            "refiner_start_epoch": 999999,
            "adversarial_start_epoch": 999999,
        }
    return config


def generate_configs(output_dir: Path) -> list[dict[str, Any]]:
    ensure_dir(CONFIG_OUT)
    records = []
    for family_key in ["rad5", "scr5"]:
        base = load_family_base(family_key)
        for arch in ARCHITECTURES:
            config = phase25_config(base, arch, family_key, output_dir)
            name = config["experiment_name"]
            path = CONFIG_OUT / f"{name}.yaml"
            save_config(config, path)
            records.append(
                {
                    "config_name": name,
                    "path": str(path),
                    "family": FAMILIES[family_key]["measurement_family"],
                    "model_type": config["model_type"],
                    "base_channels": config.get("base_channels"),
                    "epochs": config.get("epochs"),
                    "sampling_ratio": config.get("sampling_ratio"),
                    "exact_A_required": FAMILIES[family_key]["exact_A_required"],
                    "note": arch["note"],
                }
            )
    fields = [
        "config_name",
        "path",
        "family",
        "model_type",
        "base_channels",
        "epochs",
        "sampling_ratio",
        "exact_A_required",
        "note",
    ]
    write_csv_rows(output_dir / "architecture_ablation" / "phase25_config_manifest.csv", records, fields)
    return records


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


@torch.no_grad()
def run_architecture_smoke(config_records: list[dict[str, Any]], output_dir: Path, device_name: str) -> list[dict[str, Any]]:
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    rows = []
    for record in config_records:
        config = apply_experiment_defaults(load_config(record["path"]))
        config["device"] = str(device)
        config["batch_size"] = 1
        config["num_workers"] = 0
        set_seed(int(config.get("seed", 42)))
        try:
            measurement = make_measurement(config, device)
            generator = build_generator(config, measurement=measurement).to(device).eval()
            x = torch.rand(1, 1, int(config["img_size"]), int(config["img_size"]), device=device)
            y = measurement.measure(x)
            x_hat, x_data, _extras = reconstruct_from_measurements(
                generator,
                measurement,
                y,
                use_null_project=True,
                use_dc_project=True,
                backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=False,
                output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                return_extras=True,
            )
            metrics = batch_metrics(x_hat, x, measurement, y)
            rows.append(
                {
                    "config_name": record["config_name"],
                    "model_type": config["model_type"],
                    "family": record["family"],
                    "status": "pass",
                    "device": str(device),
                    "params": count_parameters(generator),
                    "x_hat_shape": "x".join(str(v) for v in x_hat.shape),
                    "finite": bool(torch.isfinite(x_hat).all().item()),
                    "smoke_psnr_random_input": metrics["psnr"],
                    "smoke_rel_meas_err": metrics.get("rel_meas_error", ""),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "config_name": record["config_name"],
                    "model_type": record["model_type"],
                    "family": record["family"],
                    "status": "fail",
                    "device": str(device),
                    "params": "",
                    "x_hat_shape": "",
                    "finite": "",
                    "smoke_psnr_random_input": "",
                    "smoke_rel_meas_err": "",
                    "error": repr(exc),
                }
            )
    fields = [
        "config_name",
        "model_type",
        "family",
        "status",
        "device",
        "params",
        "x_hat_shape",
        "finite",
        "smoke_psnr_random_input",
        "smoke_rel_meas_err",
        "error",
    ]
    write_csv_rows(output_dir / "smoke" / "architecture_smoke_results.csv", rows, fields)
    write_text(
        output_dir / "smoke" / "architecture_smoke_results.md",
        "# Phase 25 Architecture Smoke Results\n\n"
        + markdown_table(
            [
                {
                    **row,
                    "smoke_psnr_random_input": fmt(row.get("smoke_psnr_random_input"), 3),
                    "smoke_rel_meas_err": fmt(row.get("smoke_rel_meas_err"), 6),
                }
                for row in rows
            ],
            fields,
        ),
    )
    return rows


def write_pca_oracle_framework_note(output_dir: Path) -> None:
    text = f"""# Phase 25 PCA Prior Oracle Framework

The executable PCA oracle baseline is implemented in:

`{REPO_ROOT / "src" / "phase25_pca_oracle.py"}`

It computes a PCA basis on the STL-10 training split, then evaluates constrained subspace reconstruction
for 5% and 10% Rademacher/scrambled measurements:

`x = mean + U_k z`, with `z = argmin_z ||A(mean + U_k z) - y||_2^2`.

Planned k values: `32, 64, 128, 256`.

Default local smoke command:

```powershell
conda run -p E:\\ns_mc_gan_gi\\conda_envs\\ns_mc_gan_gi_py311 python -m src.phase25_pca_oracle --smoke
```

Full PCA oracle should be run only after explicit approval because it can be CPU/GPU and data-I/O heavy:

```powershell
conda run -p E:\\ns_mc_gan_gi\\conda_envs\\ns_mc_gan_gi_py311 python -m src.phase25_pca_oracle --max_train_samples 4096 --max_eval_samples 500 --k_list 32,64,128,256
```
"""
    write_text(output_dir / "limit_analysis" / "pca_oracle_framework.md", text)


def write_plan_report(
    output_dir: Path,
    config_records: list[dict[str, Any]],
    sampling_summary: dict[str, Any],
    smoke_rows: list[dict[str, Any]] | None,
) -> None:
    config_fields = [
        "config_name",
        "family",
        "model_type",
        "base_channels",
        "epochs",
        "sampling_ratio",
        "exact_A_required",
    ]
    smoke_status = "not run"
    if smoke_rows is not None:
        passed = sum(1 for row in smoke_rows if row.get("status") == "pass")
        smoke_status = f"{passed}/{len(smoke_rows)} pass"
    pca_manifest_path = output_dir / "limit_analysis" / "pca_oracle" / "pca_oracle_manifest.json"
    pca_status = "not run"
    pca_results = output_dir / "limit_analysis" / "pca_oracle" / "pca_oracle_results.csv"
    if pca_manifest_path.exists():
        try:
            pca_manifest = json.loads(pca_manifest_path.read_text(encoding="utf-8"))
            pca_status = (
                "smoke run, "
                f"train={pca_manifest.get('max_train_samples')}, "
                f"eval={pca_manifest.get('max_eval_samples')}, "
                f"k={pca_manifest.get('k_values')}"
            )
            pca_results = Path(pca_manifest.get("results_csv", pca_results))
        except Exception:
            pca_status = "manifest present but unreadable"
    text = f"""# Phase 25: Reconstruction Limit and Architecture Ablation Plan

## Scope

Phase 25 prepares code, configs, and report scaffolding only. It does not start full training. Full training should only be run if the user explicitly approves it.

## Why Arbitrary-Image Recovery Is Impossible

For 64 x 64 grayscale images, n = 4096 unknown pixels. At 5%, m is about 205 measurements; at 10%, m is about 410. Since m < n, the linear measurement map has a large null space. Infinitely many arbitrary images can match the same measurement vector, so theoretical arbitrary-image recovery is impossible without additional assumptions or priors.

## What Sets the Empirical Limit

The practical limit is determined by the image prior used by the reconstruction method. The current NS-MC-GAN result should therefore be interpreted as measurement-consistent reconstruction under a learned STL-10 prior, not universal inversion of arbitrary images.

## PCA Prior Oracle

The PCA oracle is a linear-prior baseline. It reconstructs within a learned PCA subspace:

`x = mean + U_k z`, solving `min_z ||A(mean + U_k z) - y||^2`.

This estimates how much quality a simple linear training-set prior can explain before using a nonlinear generator. The runnable implementation is `src/phase25_pca_oracle.py`, with planned k values `32, 64, 128, 256`.

Current PCA oracle status: `{pca_status}`.

Current PCA oracle output: `{pca_results}`.

## Architecture Ablation

All ablation configs keep the outer measurement-consistent structure fixed:

`x_hat = Pi_y[x_data + P_N(G_theta(x_data))]`

Only `G_theta` changes. This tests whether the current generator architecture is a bottleneck. The Rademacher configs require the same saved exact A as the no-leak runs; scrambled-Hadamard configs keep the same seed, Hadamard row/column permutation policy, split, loss, and training budget.

## Generated Configs

{markdown_table(config_records, config_fields)}

## Sampling Scaling Summary

Detailed output: `{output_dir / "limit_analysis" / "sampling_scaling_summary.md"}`

{markdown_table(sampling_summary["fit_rows"], ["family", "points", "model_gain_per_5pct", "model_psnr_at_15pct", "model_psnr_at_20pct"])}

## Short-Run Smoke

Smoke status: `{smoke_status}`.

Smoke output: `{output_dir / "smoke" / "architecture_smoke_results.csv"}`

The smoke checks only instantiate each architecture and run one random forward/eval pass through the shared projection wrapper. These numbers are not reconstruction evidence.

## Next-Step Gate

Do not run overnight/full architecture training until the user explicitly approves:

1. PCA oracle full run with a chosen train/eval sample budget.
2. Full architecture ablation training for the selected configs.
3. Final table/figure generation after those approved runs finish.
"""
    write_text(output_dir / "ARCHITECTURE_LIMIT_PLAN.md", text)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    sampling = sampling_scaling_summary(output_dir)
    configs = generate_configs(output_dir)
    write_pca_oracle_framework_note(output_dir)
    smoke_rows = None if args.skip_smoke else run_architecture_smoke(configs, output_dir, args.device)
    write_plan_report(output_dir, configs, sampling, smoke_rows)
    manifest = {
        "phase": 25,
        "output_dir": str(output_dir),
        "sampling_summary": str(output_dir / "limit_analysis" / "sampling_scaling_summary.md"),
        "architecture_plan": str(output_dir / "ARCHITECTURE_LIMIT_PLAN.md"),
        "config_dir": str(CONFIG_OUT),
        "configs": [record["path"] for record in configs],
        "smoke_results": str(output_dir / "smoke" / "architecture_smoke_results.csv") if smoke_rows is not None else "",
        "full_training_started": False,
    }
    write_json(output_dir / "phase25_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
