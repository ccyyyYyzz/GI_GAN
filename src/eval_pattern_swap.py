from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from .datasets import get_val_dataloader
from .eval import make_measurement
from .metrics import batch_metrics
from .models import ResidualUNetGenerator
from .pattern_diagnostics import compare_pattern_states
from .utils import (
    apply_experiment_defaults,
    ensure_dir,
    load_config,
    mean_dict,
    reconstruct_from_measurements,
    resolve_device,
    set_seed,
)


FIELDS = [
    "method",
    "generator_source",
    "pattern_source",
    "model_psnr",
    "model_ssim",
    "model_mse",
    "model_rel_meas_err",
    "score",
    "pattern_hard_flip_fraction",
    "A_rel_fro_delta",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generator/pattern swap controls.")
    parser.add_argument("--fixed_checkpoint", required=True)
    parser.add_argument("--fixed_config", required=True)
    parser.add_argument("--learned_checkpoint", required=True)
    parser.add_argument("--learned_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit_val_samples", type=int, default=None)
    return parser.parse_args()


def merged_config(config_path: str, checkpoint: dict | None) -> dict:
    config = apply_experiment_defaults(load_config(config_path))
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        merged = dict(config)
        merged.update(checkpoint["config"])
        config = apply_experiment_defaults(merged)
    return config


def load_generator(checkpoint: dict, device: torch.device) -> ResidualUNetGenerator:
    generator = ResidualUNetGenerator().to(device)
    state = checkpoint["generator"] if isinstance(checkpoint, dict) and "generator" in checkpoint else checkpoint
    generator.load_state_dict(state)
    generator.eval()
    return generator


def learned_measurement(config: dict, checkpoint: dict, device: torch.device, *, initial: bool):
    measurement = make_measurement(config, device)
    pattern_bank = getattr(measurement, "pattern_bank", None)
    if pattern_bank is None:
        raise RuntimeError("learned_config must enable learned patterns for pattern swap.")
    if "pattern_bank" not in checkpoint:
        raise RuntimeError("learned_checkpoint has no pattern_bank state.")
    pattern_bank.load_state_dict(checkpoint["pattern_bank"])
    pattern_bank.set_tau(float(config.get("pattern_tau_final", config.get("pattern_tau", 1.0))))
    initial_state = checkpoint.get("initial_pattern_state")
    if initial:
        if not initial_state or "logits_initial" not in initial_state:
            raise RuntimeError("learned_checkpoint has no initial_pattern_state/logits_initial.")
        with torch.no_grad():
            pattern_bank.logits.copy_(initial_state["logits_initial"].to(device=device))
    pattern_bank.eval()
    return measurement


@torch.no_grad()
def evaluate_combo(generator, measurement, val_loader, device, config) -> dict:
    rows = []
    for batch in tqdm(val_loader, desc="Swap eval", leave=False):
        x = batch[0].to(device, non_blocking=True)
        y = measurement.measure(x)
        x_hat, _ = reconstruct_from_measurements(
            generator,
            measurement,
            y,
            use_null_project=bool(config.get("use_null_project", True)),
            use_dc_project=bool(config.get("use_dc_project", True)),
        )
        rows.append(batch_metrics(x_hat, x, measurement, y))
    model = mean_dict(rows)
    score = float(model.get("psnr", 0.0)) + float(config.get("score_ssim_weight", 10.0)) * float(
        model.get("ssim", 0.0)
    )
    return {
        "model_psnr": model.get("psnr", ""),
        "model_ssim": model.get("ssim", ""),
        "model_mse": model.get("mse", ""),
        "model_rel_meas_err": model.get("rel_meas_error", ""),
        "score": score,
    }


def diag_for(measurement, initial_state, device, config) -> dict:
    pattern_bank = getattr(measurement, "pattern_bank", None)
    if pattern_bank is None:
        return {}
    dummy = torch.zeros(2, 1, int(config["img_size"]), int(config["img_size"]), device=device)
    return compare_pattern_states(pattern_bank, initial_state, secant_batch=dummy, config=config)


def write_csv(rows: list[dict], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def fmt(value) -> str:
    if value in ("", None):
        return "missing"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def write_markdown(rows: list[dict], path: Path) -> None:
    cols = FIELDS
    lines = [
        "# Pattern Swap Metrics",
        "",
        "|" + "|".join(cols) + "|",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(col, "")) for col in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    fixed_ckpt = torch.load(args.fixed_checkpoint, map_location="cpu")
    learned_ckpt = torch.load(args.learned_checkpoint, map_location="cpu")
    fixed_config = merged_config(args.fixed_config, fixed_ckpt)
    learned_config = merged_config(args.learned_config, learned_ckpt)
    if args.limit_val_samples is not None:
        fixed_config["limit_val_samples"] = args.limit_val_samples
        learned_config["limit_val_samples"] = args.limit_val_samples
    device = resolve_device(args.device or learned_config.get("device", fixed_config.get("device", "cuda")))
    fixed_ckpt = torch.load(args.fixed_checkpoint, map_location=device)
    learned_ckpt = torch.load(args.learned_checkpoint, map_location=device)
    fixed_config = merged_config(args.fixed_config, fixed_ckpt)
    learned_config = merged_config(args.learned_config, learned_ckpt)
    if args.limit_val_samples is not None:
        learned_config["limit_val_samples"] = args.limit_val_samples
    set_seed(int(learned_config.get("seed", fixed_config.get("seed", 42))))

    val_loader = get_val_dataloader(
        dataset_root=learned_config["dataset_root"],
        img_size=learned_config["img_size"],
        batch_size=learned_config["batch_size"],
        num_workers=learned_config["num_workers"],
        limit_val_samples=learned_config["limit_val_samples"],
        seed=learned_config["seed"],
        pin_memory=device.type == "cuda",
    )
    fixed_generator = load_generator(fixed_ckpt, device)
    learned_generator = load_generator(learned_ckpt, device)
    fixed_measurement = make_measurement(fixed_config, device)
    initial_measurement = learned_measurement(learned_config, learned_ckpt, device, initial=True)
    learned_measurement_obj = learned_measurement(learned_config, learned_ckpt, device, initial=False)
    initial_state = learned_ckpt.get("initial_pattern_state") if isinstance(learned_ckpt, dict) else None

    specs = [
        ("Fixed G + Fixed A", fixed_generator, fixed_measurement, "fixed", "fixed", {}),
        (
            "Learned G + Initial A",
            learned_generator,
            initial_measurement,
            "learned",
            "initial",
            diag_for(initial_measurement, initial_state, device, learned_config),
        ),
        (
            "Learned G + Learned A",
            learned_generator,
            learned_measurement_obj,
            "learned",
            "learned",
            diag_for(learned_measurement_obj, initial_state, device, learned_config),
        ),
        (
            "Fixed G + Learned A",
            fixed_generator,
            learned_measurement_obj,
            "fixed",
            "learned",
            diag_for(learned_measurement_obj, initial_state, device, learned_config),
        ),
    ]
    rows = []
    for method, generator, measurement, g_source, p_source, diagnostics in specs:
        metrics = evaluate_combo(generator, measurement, val_loader, device, learned_config)
        row = {
            "method": method,
            "generator_source": g_source,
            "pattern_source": p_source,
            "status": "ok",
            **metrics,
        }
        row["pattern_hard_flip_fraction"] = diagnostics.get("hard_flip_fraction", "")
        row["A_rel_fro_delta"] = diagnostics.get("A_rel_fro_delta", "")
        rows.append(row)

    output_dir = ensure_dir(args.output_dir)
    write_csv(rows, output_dir / "pattern_swap_metrics.csv")
    write_markdown(rows, output_dir / "pattern_swap_metrics.md")
    print(f"Wrote pattern swap metrics to: {output_dir}")


if __name__ == "__main__":
    main()
