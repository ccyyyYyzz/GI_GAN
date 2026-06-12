from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .checkpoint_utils import find_best_checkpoint
from .datasets import get_val_dataloader
from .eval import make_measurement
from .exact_measurement import apply_measurement_override_from_config, torch_load
from .metrics import batch_metrics
from .models import build_generator
from .phase48_49_common import (
    TASKS,
    copy_required_bundle_leaf,
    load_bundle_task,
    save_run_config,
    write_csv,
    write_environment,
    write_markdown_table,
    write_session_manifest,
    write_sha256s,
)
from .utils import apply_experiment_defaults, ensure_dir, save_config, save_json, set_seed
from .visualize import save_recon_grid


VARIANTS = {
    "no_gate_no_final_audit": {
        "label": "train_no_gate_no_final_audit",
        "use_null_project": False,
        "use_dc_project": True,
        "use_final_dc_project": False,
        "disable_measurement_loss": False,
        "description": "P_N disabled; stage-1 audit active; final/refiner Pi_y disabled; measurement loss active.",
    },
    "no_final_audit_no_meas_loss": {
        "label": "train_no_final_audit_no_meas_loss",
        "use_null_project": True,
        "use_dc_project": True,
        "use_final_dc_project": False,
        "disable_measurement_loss": True,
        "description": "P_N active; stage-1 audit active; final/refiner Pi_y disabled; measurement-domain training loss disabled.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 51A mechanism-closure train-time ablation runner.")
    parser.add_argument("--bundle_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--session_name", required=True)
    parser.add_argument("--task", choices=["rad5", "scr5"], required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--dataset_root", default="/content/ns_mc_gan_gi_data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit_train_samples", type=int, default=None)
    parser.add_argument("--limit_val_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_limit_val_samples", type=int, default=None)
    parser.add_argument("--posthoc_limit_samples", type=int, default=500)
    return parser.parse_args()


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, command)


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten(prefix: str, obj: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            row.update(flatten(f"{prefix}{key}_", value))
        else:
            row[f"{prefix}{key}"] = value
    return row


def disable_measurement_terms(config: dict[str, Any]) -> list[dict[str, Any]]:
    disabled: list[dict[str, Any]] = []
    keys = [
        "lambda_dc_loss",
        "score_relmeas_weight",
        "lambda_measurement_loss",
        "lambda_meas_loss",
        "lambda_relmeas",
        "lambda_audit",
        "lambda_meas_consistency",
        "omega_meas",
    ]
    for key in keys:
        if key in config:
            disabled.append({"term": key, "old_value": config.get(key), "new_value": 0.0})
            config[key] = 0.0
    config["phase51A_disabled_measurement_terms"] = disabled
    return disabled


def build_config(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    info = load_bundle_task(args.bundle_root, args.task)
    config = apply_experiment_defaults(info["config"])
    variant = VARIANTS[args.variant]
    disabled_terms: list[dict[str, Any]] = []
    config["phase51A_ablation"] = {
        "session_name": args.session_name,
        "variant": args.variant,
        "variant_label": variant["label"],
        "description": variant["description"],
        "source_config": str(info["config_path"]),
        "source_checkpoint": str(info["checkpoint_path"]),
        "strict_no_leak_note": "Derived from final no-leak resolved config; no test-set checkpoint selection is introduced.",
    }
    config["device"] = args.device
    config["dataset_root"] = args.dataset_root
    config["output_dir"] = str(output_dir)
    config["num_workers"] = int(args.num_workers)
    config["use_null_project"] = bool(variant["use_null_project"])
    config["use_dc_project"] = bool(variant["use_dc_project"])
    config["use_final_dc_project"] = bool(variant["use_final_dc_project"])
    config["eval_before_training"] = False
    if bool(variant["disable_measurement_loss"]):
        disabled_terms = disable_measurement_terms(config)
    if args.limit_train_samples is not None:
        config["limit_train_samples"] = int(args.limit_train_samples)
    if args.limit_val_samples is not None:
        config["limit_val_samples"] = int(args.limit_val_samples)
    if args.epochs is not None:
        config["epochs"] = int(args.epochs)
        config["save_every"] = int(args.epochs) + 1
        config["eval_every"] = int(args.epochs) + 1
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
    if info["exact_A_path"] is not None:
        config["measurement_operator_exact_path"] = str(info["exact_A_path"])
        config["exact_A_required"] = bool(info["metadata"]["requires_exact_A"])
    return config, info, disabled_terms


def load_generator_from_checkpoint(config: dict[str, Any], checkpoint_path: Path, measurement, device: torch.device):
    checkpoint = torch_load(checkpoint_path, map_location=device)
    generator = build_generator(config, measurement=measurement).to(device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("generator_ema") or checkpoint.get("generator")
    else:
        state = checkpoint
    if state is None:
        raise RuntimeError(f"No generator state found in {checkpoint_path}")
    generator.load_state_dict(state)
    generator.eval()
    return generator


def make_loader(config: dict[str, Any], device: torch.device, limit: int):
    return get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(config.get("batch_size", 8)),
        num_workers=int(config.get("num_workers", 2)),
        limit_val_samples=int(limit),
        seed=int(config["seed"]),
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )


@torch.no_grad()
def run_posthoc_and_perturbation(
    config: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    limit_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch.device(config["device"] if torch.cuda.is_available() or str(config["device"]) == "cpu" else "cpu")
    set_seed(int(config["seed"]))
    measurement = make_measurement(config, device)
    exact_info = apply_measurement_override_from_config(config, measurement, device)
    save_json(exact_info, output_dir / "posthoc_exact_A_info.json")
    generator = load_generator_from_checkpoint(config, checkpoint_path, measurement, device)
    loader = make_loader(config, device, limit_samples)

    from .utils import reconstruct_from_measurements

    posthoc_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader):
        x = batch[0].to(device, non_blocking=True)
        y = measurement.measure(x)
        x_noaudit, x_data, extras = reconstruct_from_measurements(
            generator,
            measurement,
            y,
            use_null_project=bool(config.get("use_null_project", True)),
            use_dc_project=bool(config.get("use_dc_project", True)),
            use_final_dc_project=False,
            backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
            enable_refiner=True,
            output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
            return_extras=True,
        )
        no_flat = measurement.flatten_img(extras["x_hat_unclamped"].float())
        post_flat = measurement.dc_project(no_flat, y.float())
        post_unclamped = measurement.unflatten_img(post_flat)
        post = torch.clamp(post_unclamped, 0.0, 1.0)
        before = batch_metrics(x_noaudit, x, measurement, y)
        after = batch_metrics(post, x, measurement, y)
        before_unclamped = batch_metrics(extras["x_hat_unclamped"], x, measurement, y)
        after_unclamped = batch_metrics(post_unclamped, x, measurement, y)
        posthoc_rows.append(
            {
                "batch": batch_idx,
                "psnr_before": before["psnr"],
                "psnr_after": after["psnr"],
                "ssim_before": before["ssim"],
                "ssim_after": after["ssim"],
                "relmeas_before_clamped": before.get("rel_meas_error", ""),
                "relmeas_after_clamped": after.get("rel_meas_error", ""),
                "relmeas_before_unclamped": before_unclamped.get("rel_meas_error", ""),
                "relmeas_after_unclamped": after_unclamped.get("rel_meas_error", ""),
            }
        )
        clean_psnr = before["psnr"]
        cases = []
        if y.shape[0] > 1:
            cases.append(("wrong_y_batch_roll", torch.roll(y, shifts=1, dims=0)))
        perm = torch.randperm(y.shape[1], device=y.device)
        cases.append(("shuffle_y_columns", y[:, perm]))
        for name, y_case in cases:
            x_bad, _x_data_bad, _extras_bad = reconstruct_from_measurements(
                generator,
                measurement,
                y_case,
                use_null_project=bool(config.get("use_null_project", True)),
                use_dc_project=bool(config.get("use_dc_project", True)),
                use_final_dc_project=False,
                backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=True,
                output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                return_extras=True,
            )
            metrics = batch_metrics(x_bad, x, measurement, y_case)
            perturb_rows.append(
                {
                    "batch": batch_idx,
                    "perturbation": name,
                    "clean_psnr": clean_psnr,
                    "perturbed_psnr": metrics["psnr"],
                    "psnr_drop": clean_psnr - metrics["psnr"],
                    "perturbed_ssim": metrics["ssim"],
                    "perturbed_rel_meas_error": metrics.get("rel_meas_error", ""),
                }
            )
        if batch_idx == 0:
            save_recon_grid(x, x_noaudit, post, output_dir / "visual_grid.png", max_items=8)
    return posthoc_rows, perturb_rows


def write_loss_terms(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"term": "none", "old_value": "", "new_value": "", "note": "Measurement loss remained active."}]
    write_csv(path, rows)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    info: dict[str, Any],
    config: dict[str, Any],
    eval_metrics: dict[str, Any],
    posthoc_rows: list[dict[str, Any]],
    perturb_rows: list[dict[str, Any]],
) -> None:
    baseline = read_json(info.get("metrics_path"))
    rows = []
    if baseline:
        rows.append({"run": f"full_{args.task}", **flatten("", baseline)})
    rows.append({"run": args.session_name, **flatten("", eval_metrics)})
    write_csv(output_dir / "eval_final.csv", rows)
    write_markdown_table(output_dir / "eval_final.md", rows, f"{args.session_name} final eval")

    def mean(key: str, table: list[dict[str, Any]]) -> float | str:
        vals = []
        for row in table:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        return sum(vals) / len(vals) if vals else ""

    lines = [
        f"# {args.session_name} Session Report",
        "",
        f"- task: {args.task} ({TASKS[args.task]['display']})",
        f"- variant: {args.variant} / {VARIANTS[args.variant]['label']}",
        "- category: train-time ablation / mechanism closure",
        "- strict_no_leak: true",
        f"- use_null_project: {config.get('use_null_project')}",
        f"- use_dc_project: {config.get('use_dc_project')}",
        f"- use_final_dc_project: {config.get('use_final_dc_project')}",
        f"- lambda_dc_loss: {config.get('lambda_dc_loss')}",
        f"- score_relmeas_weight: {config.get('score_relmeas_weight')}",
        f"- exact_A_required: {config.get('exact_A_required', False)}",
        f"- exact_A_path: {config.get('measurement_operator_exact_path', '')}",
        "",
        "## Posthoc Audit Summary",
        "",
        f"- PSNR before: {mean('psnr_before', posthoc_rows)}",
        f"- PSNR after: {mean('psnr_after', posthoc_rows)}",
        f"- RelMeasErr before unclamped: {mean('relmeas_before_unclamped', posthoc_rows)}",
        f"- RelMeasErr after unclamped: {mean('relmeas_after_unclamped', posthoc_rows)}",
        "",
        "## Perturbation Summary",
        "",
        f"- Mean wrong/shuffle PSNR drop: {mean('psnr_drop', perturb_rows)}",
        "",
        "## Interpretation Guardrail",
        "",
        "Do not interpret this ablation from PSNR alone. RelMeasErr, perturbation sensitivity, and posthoc-audit recovery are the mechanism-closure endpoints.",
    ]
    (output_dir / "SESSION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    command_log = output_dir / "command_log.txt"
    write_environment(output_dir)
    config, info, disabled_terms = build_config(args, output_dir)
    run_config = save_run_config(config, output_dir)
    save_config(config, output_dir / "config_used.yaml")
    write_loss_terms(output_dir / "loss_terms_log.csv", disabled_terms)
    copied = copy_required_bundle_leaf(args.bundle_root, output_dir / "_source_bundle_leaf", args.task)
    if info.get("exact_A_path") is not None:
        shutil.copy2(info["exact_A_path"], output_dir / "measurement_operator_exact.pt")

    project_root = Path.cwd()
    run_command([sys.executable, "-m", "src.train", "--config", str(run_config)], project_root, command_log)

    checkpoint = find_best_checkpoint(output_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"Training finished but no checkpoint was found in {output_dir}.")
    exported_checkpoint = output_dir / "checkpoint_final.pt"
    shutil.copy2(checkpoint, exported_checkpoint)
    best_or_final = output_dir / "best_or_final_checkpoint.pt"
    if best_or_final != exported_checkpoint:
        shutil.copy2(checkpoint, best_or_final)
    if (output_dir / "per_epoch_metrics.csv").exists():
        shutil.copy2(output_dir / "per_epoch_metrics.csv", output_dir / "train_log.csv")

    eval_dir = ensure_dir(output_dir / "eval_final")
    eval_cmd = [
        sys.executable,
        "-m",
        "src.eval",
        "--config",
        str(run_config),
        "--checkpoint",
        str(exported_checkpoint),
        "--output_dir",
        str(eval_dir),
    ]
    if args.eval_limit_val_samples is not None:
        eval_cmd.extend(["--limit_val_samples", str(args.eval_limit_val_samples)])
    run_command(eval_cmd, project_root, command_log)
    eval_metrics = read_json(eval_dir / "eval_metrics.json")
    per_sample = eval_metrics.get("per_sample_metrics")
    if per_sample and Path(per_sample).exists():
        shutil.copy2(per_sample, output_dir / "per_sample_metrics.csv")

    posthoc_rows, perturb_rows = run_posthoc_and_perturbation(
        config=config,
        checkpoint_path=exported_checkpoint,
        output_dir=output_dir,
        limit_samples=int(args.posthoc_limit_samples),
    )
    write_csv(output_dir / "posthoc_audit_eval.csv", posthoc_rows)
    write_csv(output_dir / "measurement_perturbation_subset.csv", perturb_rows)
    write_report(output_dir, args, info, config, eval_metrics, posthoc_rows, perturb_rows)

    write_session_manifest(
        output_dir,
        args.session_name,
        {
            "trains": True,
            "task": args.task,
            "variant": args.variant,
            "variant_label": VARIANTS[args.variant]["label"],
            "source_config": str(info["config_path"]),
            "source_checkpoint": str(info["checkpoint_path"]),
            "source_exact_A": str(info["exact_A_path"] or ""),
            "output_dir": str(output_dir),
            "copied_bundle_files": copied,
            "disabled_measurement_terms": disabled_terms,
        },
    )
    save_json(
        {
            "ok": True,
            "session": args.session_name,
            "task": args.task,
            "variant": args.variant,
            "checkpoint": str(exported_checkpoint),
            "eval_metrics": str(eval_dir / "eval_metrics.json"),
            "posthoc_audit_eval": str(output_dir / "posthoc_audit_eval.csv"),
            "measurement_perturbation_subset": str(output_dir / "measurement_perturbation_subset.csv"),
        },
        output_dir / "SESSION_STATUS.json",
    )
    write_sha256s(output_dir)
    print(f"{args.session_name} complete: {output_dir}")


if __name__ == "__main__":
    main()
