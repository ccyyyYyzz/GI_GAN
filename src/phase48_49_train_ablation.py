from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .checkpoint_utils import find_best_checkpoint
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
from .utils import apply_experiment_defaults, ensure_dir, load_config, save_json


VARIANTS = {
    "no_gate": {
        "label": "train_no_gate",
        "use_null_project": False,
        "use_dc_project": True,
        "use_final_dc_project": True,
        "description": "P_N is replaced by identity during training and inference; final Pi_y audit remains active.",
    },
    "no_final_audit": {
        "label": "train_no_final_audit",
        "use_null_project": True,
        "use_dc_project": True,
        "use_final_dc_project": False,
        "description": "P_N remains active; stage-1 audit remains active; final/refiner Pi_y audit is removed.",
    },
    "no_all_audit": {
        "label": "train_no_all_audit",
        "use_null_project": True,
        "use_dc_project": False,
        "use_final_dc_project": False,
        "description": "Diagnostic fallback only: P_N remains active and all Pi_y audits are disabled.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 48/49 train-time ablation runner.")
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
    return parser.parse_args()


def _run_command(command: list[str], cwd: Path, log_path: Path) -> None:
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


def _flatten_metrics(prefix: str, obj: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            row.update(_flatten_metrics(f"{prefix}{key}_", value))
        else:
            row[f"{prefix}{key}"] = value
    return row


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_ablation_config(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    info = load_bundle_task(args.bundle_root, args.task)
    config = apply_experiment_defaults(info["config"])
    variant = VARIANTS[args.variant]
    config["phase48_49_ablation"] = {
        "session_name": args.session_name,
        "variant": args.variant,
        "variant_label": variant["label"],
        "description": variant["description"],
        "source_config": str(info["config_path"]),
        "source_checkpoint": str(info["checkpoint_path"]),
        "strict_no_leak_note": "Uses final no-leak run config and changes only documented ablation switches/output paths.",
    }
    config["device"] = args.device
    config["dataset_root"] = args.dataset_root
    config["output_dir"] = str(output_dir)
    config["num_workers"] = int(args.num_workers)
    config["use_null_project"] = bool(variant["use_null_project"])
    config["use_dc_project"] = bool(variant["use_dc_project"])
    config["use_final_dc_project"] = bool(variant["use_final_dc_project"])
    config["eval_before_training"] = False
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
    return config, info


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    info: dict[str, Any],
    config: dict[str, Any],
    eval_metrics: dict[str, Any],
) -> None:
    baseline = _load_json(info.get("metrics_path"))
    rows = []
    if baseline:
        rows.append({"run": f"full_{args.task}", **_flatten_metrics("", baseline)})
    rows.append({"run": args.session_name, **_flatten_metrics("", eval_metrics)})
    write_csv(output_dir / "eval_final.csv", rows)
    write_markdown_table(output_dir / "eval_final.md", rows, f"{args.session_name} final eval")
    lines = [
        f"# {args.session_name} Session Report",
        "",
        f"- task: {args.task} ({TASKS[args.task]['display']})",
        f"- variant: {args.variant} / {VARIANTS[args.variant]['label']}",
        f"- trains: true",
        f"- strict_no_leak: true",
        f"- use_null_project: {config.get('use_null_project')}",
        f"- use_dc_project: {config.get('use_dc_project')}",
        f"- use_final_dc_project: {config.get('use_final_dc_project')}",
        f"- exact_A_required: {config.get('exact_A_required', False)}",
        f"- exact_A_path: {config.get('measurement_operator_exact_path', '')}",
        "",
        "## Interpretation Guardrail",
        "",
        "This is an exploratory train-time ablation. It should not overwrite the main no-leak checkpoint and should not be treated as a main-table result until the user explicitly approves.",
        "",
        "## Baseline And Ablation Metrics",
        "",
        "See eval_final.csv/md for flattened baseline-vs-ablation metrics.",
    ]
    (output_dir / "SESSION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    command_log = output_dir / "command_log.txt"
    write_environment(output_dir)
    config, info = build_ablation_config(args, output_dir)
    run_config = save_run_config(config, output_dir)
    copied = copy_required_bundle_leaf(args.bundle_root, output_dir / "_source_bundle_leaf", args.task)
    if info.get("exact_A_path") is not None:
        shutil.copy2(info["exact_A_path"], output_dir / "measurement_operator_exact.pt")

    project_root = Path.cwd()
    train_cmd = [sys.executable, "-m", "src.train", "--config", str(run_config)]
    _run_command(train_cmd, project_root, command_log)

    checkpoint = find_best_checkpoint(output_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"Training finished but no checkpoint was found in {output_dir}.")
    exported_checkpoint = output_dir / "best_or_final_checkpoint.pt"
    if checkpoint.resolve() != exported_checkpoint.resolve():
        shutil.copy2(checkpoint, exported_checkpoint)

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
    _run_command(eval_cmd, project_root, command_log)
    eval_metrics = _load_json(eval_dir / "eval_metrics.json")
    per_sample = eval_metrics.get("per_sample_metrics")
    if per_sample and Path(per_sample).exists():
        shutil.copy2(per_sample, output_dir / "per_sample_metrics.csv")
    write_report(output_dir, args, info, config, eval_metrics)
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
        },
        output_dir / "SESSION_STATUS.json",
    )
    write_sha256s(output_dir)
    print(f"{args.session_name} complete: {output_dir}")


if __name__ == "__main__":
    main()
