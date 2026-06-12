from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

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
from .utils import apply_experiment_defaults, ensure_dir, save_json, set_seed
from .visualize import save_recon_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 48/49 eval-only mechanistic probe pack.")
    parser.add_argument("--bundle_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_root", default="/content/ns_mc_gan_gi_data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--tasks", nargs="*", default=["rad5", "scr5", "rad10", "scr10"])
    return parser.parse_args()


def _load_generator(task_info: dict[str, Any], measurement, device: torch.device):
    config = task_info["config"]
    checkpoint = torch_load(task_info["checkpoint_path"], map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        merged = dict(config)
        merged.update(checkpoint["config"])
        merged["phase48_49_source_config_path"] = str(task_info["config_path"])
        merged["phase48_49_source_checkpoint_path"] = str(task_info["checkpoint_path"])
        if task_info.get("exact_A_path") is not None:
            merged["measurement_operator_exact_path"] = str(task_info["exact_A_path"])
            merged["exact_A_required"] = bool(task_info["metadata"]["requires_exact_A"])
        config = apply_experiment_defaults(merged)
    generator = build_generator(config, measurement=measurement).to(device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("generator_ema") or checkpoint.get("generator")
    else:
        state = checkpoint
    if state is None:
        raise RuntimeError(f"Checkpoint has no generator state: {task_info['checkpoint_path']}")
    generator.load_state_dict(state)
    generator.eval()
    return config, checkpoint, generator


def _configure_task(args: argparse.Namespace, task_key: str, device: torch.device, task_out: Path):
    info = load_bundle_task(args.bundle_root, task_key)
    config = apply_experiment_defaults(info["config"])
    config["device"] = str(device)
    config["dataset_root"] = args.dataset_root
    config["limit_val_samples"] = int(args.limit_samples)
    config["batch_size"] = int(args.batch_size or config.get("batch_size", 8))
    config["num_workers"] = int(args.num_workers)
    config["output_dir"] = str(task_out)
    if info["exact_A_path"] is not None:
        config["measurement_operator_exact_path"] = str(info["exact_A_path"])
        config["exact_A_required"] = bool(info["metadata"]["requires_exact_A"])
    info["config"] = config
    measurement = make_measurement(config, device)
    exact_info = apply_measurement_override_from_config(config, measurement, device)
    config, checkpoint, generator = _load_generator(info, measurement, device)
    config["dataset_root"] = args.dataset_root
    config["output_dir"] = str(task_out)
    save_run_config(config, task_out)
    save_json(exact_info, task_out / "exact_A_info.json")
    return info, config, checkpoint, generator, measurement, exact_info


def _val_loader(config: dict[str, Any], device: torch.device):
    return get_val_dataloader(
        dataset_root=config["dataset_root"],
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config.get("num_workers", 2)),
        limit_val_samples=int(config.get("limit_val_samples", 1000)),
        seed=int(config["seed"]),
        pin_memory=device.type == "cuda",
        dataset_name=config.get("dataset_name", "stl10"),
        class_filter=config.get("class_filter"),
    )


def _metric_row(prefix: str, pred: torch.Tensor, target: torch.Tensor, measurement, y: torch.Tensor) -> dict[str, float]:
    metrics = batch_metrics(pred, target, measurement, y)
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _flat_rel_measurement_error(measurement, flat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    err = measurement.A_forward(flat.float()) - y.float()
    return torch.linalg.norm(err, dim=1) / torch.linalg.norm(y.float(), dim=1).clamp_min(1e-12)


def _norm_ratio(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(numer.float(), dim=1) / torch.linalg.norm(denom.float(), dim=1).clamp_min(1e-12)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and row[key] == row[key]]
    return float(sum(vals) / max(1, len(vals)))


def _plot_simple(path: Path, rows: list[dict[str, Any]], x_key: str, y_keys: list[str], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    x_vals = [str(row.get(x_key, i)) for i, row in enumerate(rows)]
    plt.figure(figsize=(max(6, 0.45 * len(x_vals)), 4))
    for key in y_keys:
        plt.plot(x_vals, [float(row.get(key, float("nan"))) for row in rows], marker="o", label=key)
    plt.xticks(rotation=35, ha="right")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_hist(path: Path, values: list[float], title: str, xlabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not values:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


@torch.no_grad()
def run_task_probes(args: argparse.Namespace, task_key: str, output_dir: Path, device: torch.device) -> dict[str, Any]:
    task_out = ensure_dir(output_dir / task_key)
    info, config, _checkpoint, generator, measurement, exact_info = _configure_task(args, task_key, device, task_out)
    copied = copy_required_bundle_leaf(args.bundle_root, task_out / "_source_bundle_leaf", task_key)
    loader = _val_loader(config, device)

    raw_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    prov_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    perturb_rows: list[dict[str, Any]] = []
    gate_ratios: list[float] = []
    audit_rel_pre: list[float] = []
    audit_rel_post: list[float] = []
    first_preview = None
    first_preview_x = None

    set_seed(int(config["seed"]))
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"{task_key} probes")):
        x = batch[0].to(device, non_blocking=True)
        y = measurement.measure(x)
        x_hat, x_data, extras = reconstruct_with_extras(generator, measurement, y, config)
        raw_flat = measurement.AT_forward(y.float())
        ridge_flat = measurement.data_solution(y.float(), mode="ridge_pinv")
        actual_flat = extras["x_data_flat"]
        raw_img = torch.clamp(measurement.unflatten_img(raw_flat), 0.0, 1.0)
        ridge_img = torch.clamp(measurement.unflatten_img(ridge_flat), 0.0, 1.0)
        actual_img = torch.clamp(measurement.unflatten_img(actual_flat), 0.0, 1.0)
        raw_rows.append(
            {
                "task": task_key,
                "batch": batch_idx,
                "actual_anchor_mode": config.get("backprojection_mode", "ridge_pinv"),
                **_metric_row("raw_ATy", raw_img, x, measurement, y),
                **_metric_row("ridge_anchor", ridge_img, x, measurement, y),
                **_metric_row("actual_anchor", actual_img, x, measurement, y),
            }
        )

        raw_res_flat = extras["raw_residual_flat"]
        filt_res_flat = extras["filtered_residual_flat"]
        raw_res_A = measurement.A_forward(raw_res_flat)
        filt_res_A = measurement.A_forward(filt_res_flat)
        ratio = _norm_ratio(filt_res_A, raw_res_A).detach().cpu()
        gate_ratios.extend([float(v) for v in ratio])
        raw_candidate = torch.clamp(measurement.unflatten_img(actual_flat + raw_res_flat), 0.0, 1.0)
        filt_candidate = torch.clamp(measurement.unflatten_img(actual_flat + filt_res_flat), 0.0, 1.0)
        gate_rows.append(
            {
                "task": task_key,
                "batch": batch_idx,
                "Ar_raw_norm": float(torch.linalg.norm(raw_res_A.float(), dim=1).mean().item()),
                "Ar_filtered_norm": float(torch.linalg.norm(filt_res_A.float(), dim=1).mean().item()),
                "gate_leakage_ratio": float(ratio.mean().item()),
                "residual_change_ratio": float(_norm_ratio(raw_res_flat - filt_res_flat, raw_res_flat).mean().item()),
                **_metric_row("x_data_plus_raw_residual", raw_candidate, x, measurement, y),
                **_metric_row("x_data_plus_filtered_residual", filt_candidate, x, measurement, y),
            }
        )

        pre_final_flat = extras.get("pre_final_audit_flat", extras["pre_audit_flat"])
        post_flat = measurement.flatten_img(extras["x_hat_unclamped"].float())
        pre_img = torch.clamp(measurement.unflatten_img(pre_final_flat), 0.0, 1.0)
        post_img = x_hat
        ey = measurement.A_forward(pre_final_flat.float()) - y.float()
        delta = measurement.AT_forward(measurement.solve_K(ey))
        rel_pre = _flat_rel_measurement_error(measurement, pre_final_flat, y)
        rel_post = _flat_rel_measurement_error(measurement, post_flat, y)
        audit_rel_pre.extend([float(v) for v in rel_pre.detach().cpu()])
        audit_rel_post.extend([float(v) for v in rel_post.detach().cpu()])
        audit_rows.append(
            {
                "task": task_key,
                "batch": batch_idx,
                "final_dc_project": bool(extras.get("final_dc_project", True)),
                "audit_delta_norm_ratio": float(_norm_ratio(delta, pre_final_flat).mean().item()),
                "relmeaserr_pre_final_audit": float(rel_pre.mean().item()),
                "relmeaserr_post_final_audit": float(rel_post.mean().item()),
                **_metric_row("pre_final_audit", pre_img, x, measurement, y),
                **_metric_row("post_final_audit", post_img, x, measurement, y),
            }
        )

        xhat_flat = post_flat
        x_meas_flat = measurement.AT_forward(measurement.solve_K(measurement.A_forward(xhat_flat)))
        x_comp_flat = xhat_flat - x_meas_flat
        meas_norm = _norm_ratio(x_meas_flat, xhat_flat)
        comp_norm = _norm_ratio(x_comp_flat, xhat_flat)
        cos_meas = torch.nn.functional.cosine_similarity(x_meas_flat, xhat_flat, dim=1)
        cos_comp = torch.nn.functional.cosine_similarity(x_comp_flat, xhat_flat, dim=1)
        bp_metrics = batch_metrics(actual_img, x, measurement, y)
        model_metrics = batch_metrics(x_hat, x, measurement, y)
        prov_rows.append(
            {
                "task": task_key,
                "batch": batch_idx,
                "measured_component_indicator": float(meas_norm.mean().item()),
                "learned_complement_indicator": float(comp_norm.mean().item()),
                "cos_measured_with_xhat": float(cos_meas.mean().item()),
                "cos_complement_with_xhat": float(cos_comp.mean().item()),
                "bp_psnr": float(bp_metrics["psnr"]),
                "final_psnr": float(model_metrics["psnr"]),
                "gain_psnr": float(model_metrics["psnr"] - bp_metrics["psnr"]),
                "rel_meas_error": float(model_metrics.get("rel_meas_error", float("nan"))),
            }
        )
        comp_rows.append(
            {
                "task": task_key,
                "batch": batch_idx,
                "bp_psnr": float(bp_metrics["psnr"]),
                "final_psnr": float(model_metrics["psnr"]),
                "gain_psnr": float(model_metrics["psnr"] - bp_metrics["psnr"]),
                "bp_ssim": float(bp_metrics["ssim"]),
                "final_ssim": float(model_metrics["ssim"]),
                "gain_ssim": float(model_metrics["ssim"] - bp_metrics["ssim"]),
            }
        )

        if batch_idx == 0:
            first_preview = (raw_img, actual_img, ridge_img, filt_candidate, pre_img, post_img)
            first_preview_x = x
            perturb_rows.extend(run_perturbation_probe(task_key, generator, measurement, config, x, y))

    if first_preview and first_preview_x is not None:
        raw_img, actual_img, ridge_img, filt_candidate, pre_img, post_img = first_preview
        save_recon_grid(first_preview_x, actual_img, post_img, task_out / "rawGI_vs_anchor_examples.png", max_items=8)
        save_recon_grid(first_preview_x, filt_candidate, post_img, task_out / "raw_residual_vs_filtered_residual_grid.png", max_items=8)
        save_recon_grid(first_preview_x, pre_img, post_img, task_out / "audit_correction_grid.png", max_items=8)
        save_recon_grid(first_preview_x, ridge_img, raw_img, task_out / "provenance_examples_grid.png", max_items=8)

    write_csv(task_out / "rawGI_vs_anchor_results.csv", raw_rows)
    write_markdown_table(task_out / "rawGI_vs_anchor_results.md", raw_rows, f"{task_key} Raw GI vs anchor")
    write_csv(task_out / "gate_geometry_results.csv", gate_rows)
    write_markdown_table(task_out / "gate_geometry_results.md", gate_rows, f"{task_key} Gate geometry")
    write_csv(task_out / "audit_correction_results.csv", audit_rows)
    write_markdown_table(task_out / "audit_correction_results.md", audit_rows, f"{task_key} Audit correction")
    write_csv(task_out / "provenance_distribution.csv", prov_rows)
    write_markdown_table(task_out / "provenance_distribution.md", prov_rows, f"{task_key} Provenance distribution")
    write_csv(task_out / "per_sample_compensation.csv", comp_rows)
    write_markdown_table(task_out / "per_sample_compensation.md", comp_rows, f"{task_key} Compensation")
    write_csv(task_out / "measurement_dependence_results.csv", perturb_rows)
    write_markdown_table(task_out / "measurement_dependence_results.md", perturb_rows, f"{task_key} Measurement perturbation")

    _plot_simple(task_out / "rawGI_vs_anchor_relmeaserr.png", raw_rows, "batch", ["raw_ATy_rel_meas_error", "actual_anchor_rel_meas_error", "ridge_anchor_rel_meas_error"], f"{task_key} anchor RelMeasErr")
    _plot_hist(task_out / "Ar_raw_vs_filtered_hist.png", gate_ratios, f"{task_key} gate leakage ratio", "||A r_N|| / ||A r_theta||")
    _plot_simple(task_out / "gate_reduction_ratio_by_family.png", gate_rows, "batch", ["gate_leakage_ratio"], f"{task_key} gate leakage")
    _plot_hist(task_out / "relmeaserr_pre_post_hist.png", audit_rel_pre + audit_rel_post, f"{task_key} audit RelMeasErr values", "RelMeasErr")
    _plot_simple(task_out / "audit_delta_norm_by_family.png", audit_rows, "batch", ["audit_delta_norm_ratio"], f"{task_key} audit correction norm")
    _plot_simple(task_out / "provenance_vs_gain_scatter.png", prov_rows, "batch", ["measured_component_indicator", "learned_complement_indicator", "gain_psnr"], f"{task_key} provenance and gain")
    _plot_simple(task_out / "provenance_vs_psnr_scatter.png", prov_rows, "batch", ["bp_psnr", "final_psnr"], f"{task_key} PSNR")
    _plot_simple(task_out / "perturbation_psnr_drop.png", perturb_rows, "perturbation", ["final_psnr", "psnr_drop_vs_clean"], f"{task_key} perturbation")
    _plot_simple(task_out / "paired_vs_unpaired_shuffle.png", perturb_rows, "perturbation", ["final_psnr"], f"{task_key} paired/unpaired shuffle")
    _plot_simple(task_out / "A_mismatch_results.png", perturb_rows, "perturbation", ["final_psnr", "rel_meas_error"], f"{task_key} A mismatch")

    lambda_rows = run_lambda_sweep(args, task_key, config, generator, measurement, device, task_out)
    write_csv(task_out / "lambda_sweep_results.csv", lambda_rows)
    write_markdown_table(task_out / "lambda_sweep_results.md", lambda_rows, f"{task_key} lambda sweep")
    _plot_simple(task_out / "lambda_sweep_psnr.png", lambda_rows, "lambda_op", ["final_psnr"], f"{task_key} lambda vs PSNR")
    _plot_simple(task_out / "lambda_sweep_relmeaserr.png", lambda_rows, "lambda_op", ["rel_meas_error"], f"{task_key} lambda vs RelMeasErr")
    _plot_simple(task_out / "lambda_sweep_gate_audit.png", lambda_rows, "lambda_op", ["gate_leakage_ratio", "audit_delta_norm_ratio"], f"{task_key} lambda gate/audit")

    summary = {
        "task": task_key,
        "display": TASKS[task_key]["display"],
        "exact_A_info": exact_info,
        "copied_bundle_files": copied,
        "actual_anchor_mode": config.get("backprojection_mode"),
        "final_psnr_mean": _mean(comp_rows, "final_psnr"),
        "bp_psnr_mean": _mean(comp_rows, "bp_psnr"),
        "gain_psnr_mean": _mean(comp_rows, "gain_psnr"),
        "gate_leakage_ratio_mean": _mean(gate_rows, "gate_leakage_ratio"),
        "audit_pre_relmeaserr_mean": _mean(audit_rows, "relmeaserr_pre_final_audit"),
        "audit_post_relmeaserr_mean": _mean(audit_rows, "relmeaserr_post_final_audit"),
    }
    save_json(summary, task_out / "task_summary.json")
    return summary


def reconstruct_with_extras(generator, measurement, y: torch.Tensor, config: dict[str, Any]):
    from .utils import reconstruct_from_measurements

    return reconstruct_from_measurements(
        generator,
        measurement,
        y,
        use_null_project=bool(config.get("use_null_project", True)),
        use_dc_project=bool(config.get("use_dc_project", True)),
        use_final_dc_project=bool(config.get("use_final_dc_project", config.get("use_dc_project", True))),
        backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
        enable_refiner=True,
        output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
        return_extras=True,
    )


@torch.no_grad()
def run_perturbation_probe(task_key: str, generator, measurement, config: dict[str, Any], x: torch.Tensor, y: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean_hat, clean_data, _ = reconstruct_with_extras(generator, measurement, y, config)
    clean = batch_metrics(clean_hat, x, measurement, y)
    A0 = measurement.get_current_A().detach().clone() if hasattr(measurement, "get_current_A") else None

    def eval_case(name: str, y_case: torch.Tensor, restore_A: bool = False):
        try:
            x_hat, _x_data, _extras = reconstruct_with_extras(generator, measurement, y_case, config)
            metrics = batch_metrics(x_hat, x, measurement, y_case)
            rows.append(
                {
                    "task": task_key,
                    "perturbation": name,
                    "status": "ok",
                    "final_psnr": float(metrics["psnr"]),
                    "final_ssim": float(metrics["ssim"]),
                    "rel_meas_error": float(metrics.get("rel_meas_error", float("nan"))),
                    "clean_final_psnr": float(clean["psnr"]),
                    "psnr_drop_vs_clean": float(clean["psnr"] - metrics["psnr"]),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "task": task_key,
                    "perturbation": name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if restore_A and A0 is not None:
                measurement.set_A_override(A0, metadata={"phase48_49_restore": True}, rebuild_cache=True)

    if y.shape[0] > 1:
        eval_case("wrong_y_batch_roll", torch.roll(y, shifts=1, dims=0))
    perm = torch.randperm(y.shape[1], device=y.device)
    eval_case("row_shuffle_mismatched_y_only", y[:, perm])
    for alpha in [0.5, 0.75, 1.25, 1.5]:
        eval_case(f"y_scale_{alpha}", y * float(alpha))
    for offset in [-0.05, 0.05]:
        eval_case(f"y_offset_{offset}", y + float(offset))
    a_mismatch_supported = not (
        str(config.get("backprojection_mode", "")).lower() == "hadamard_zero_filled"
        and getattr(measurement, "hadamard_metadata", None) is not None
    )
    if A0 is not None and hasattr(measurement, "set_A_override") and a_mismatch_supported:
        row_perm = torch.randperm(A0.shape[0], device=A0.device)
        measurement.set_A_override(A0[row_perm], metadata={"phase48_49_perturbation": "paired_row_shuffle"}, rebuild_cache=True)
        eval_case("paired_row_shuffle_A_and_y", y[:, row_perm], restore_A=True)
        measurement.set_A_override(A0[row_perm], metadata={"phase48_49_perturbation": "A_row_shuffle_only"}, rebuild_cache=True)
        eval_case("A_row_shuffle_only", y, restore_A=True)
        signs = torch.where(torch.rand(A0.shape[0], device=A0.device) > 0.5, 1.0, -1.0).to(A0.dtype)
        measurement.set_A_override(A0 * signs[:, None], metadata={"phase48_49_perturbation": "A_sign_flip_only"}, rebuild_cache=True)
        eval_case("A_sign_flip_only", y, restore_A=True)
        noise = 0.01 * A0.std().clamp_min(1e-12) * torch.randn_like(A0)
        measurement.set_A_override(A0 + noise, metadata={"phase48_49_perturbation": "A_gaussian_001"}, rebuild_cache=True)
        eval_case("A_gaussian_001", y, restore_A=True)
    elif A0 is not None and hasattr(measurement, "set_A_override"):
        for name in [
            "paired_row_shuffle_A_and_y",
            "A_row_shuffle_only",
            "A_sign_flip_only",
            "A_gaussian_001",
        ]:
            rows.append(
                {
                    "task": task_key,
                    "perturbation": name,
                    "status": "skipped",
                    "reason": "A override would invalidate hadamard_zero_filled metadata; use y-only perturbations for this anchor.",
                }
            )
    return rows


@torch.no_grad()
def run_lambda_sweep(
    args: argparse.Namespace,
    task_key: str,
    config: dict[str, Any],
    generator,
    measurement,
    device: torch.device,
    task_out: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    loader = _val_loader(config, device)
    original_lambda = float(getattr(measurement, "lambda_dc", config.get("lambda_solver", 1e-3)))
    for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        if hasattr(measurement, "lambda_dc"):
            measurement.lambda_dc = float(lam)
        if hasattr(measurement, "_rebuild_solver_cache"):
            measurement._rebuild_solver_cache()
        metrics_rows: list[dict[str, float]] = []
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 8:
                break
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            x_hat, _x_data, extras = reconstruct_with_extras(generator, measurement, y, config)
            metrics = batch_metrics(x_hat, x, measurement, y)
            raw_A = measurement.A_forward(extras["raw_residual_flat"])
            filt_A = measurement.A_forward(extras["filtered_residual_flat"])
            delta = measurement.AT_forward(
                measurement.solve_K(measurement.A_forward(extras["pre_final_audit_flat"]) - y.float())
            )
            metrics_rows.append(
                {
                    "final_psnr": float(metrics["psnr"]),
                    "final_ssim": float(metrics["ssim"]),
                    "rel_meas_error": float(metrics.get("rel_meas_error", float("nan"))),
                    "gate_leakage_ratio": float(_norm_ratio(filt_A, raw_A).mean().item()),
                    "audit_delta_norm_ratio": float(_norm_ratio(delta, extras["pre_final_audit_flat"]).mean().item()),
                }
            )
        rows.append(
            {
                "task": task_key,
                "lambda_op": lam,
                "solver_uses_cholesky": bool(getattr(measurement, "_use_cholesky", False)),
                "final_psnr": _mean(metrics_rows, "final_psnr"),
                "final_ssim": _mean(metrics_rows, "final_ssim"),
                "rel_meas_error": _mean(metrics_rows, "rel_meas_error"),
                "gate_leakage_ratio": _mean(metrics_rows, "gate_leakage_ratio"),
                "audit_delta_norm_ratio": _mean(metrics_rows, "audit_delta_norm_ratio"),
            }
        )
    if hasattr(measurement, "lambda_dc"):
        measurement.lambda_dc = original_lambda
    if hasattr(measurement, "_rebuild_solver_cache"):
        measurement._rebuild_solver_cache()
    return rows


def write_final_reports(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "mechanistic_probe_summary.csv", summaries)
    write_markdown_table(output_dir / "mechanistic_probe_summary.md", summaries, "Phase 48/49 mechanistic probe summary")
    compensation = {}
    by_pct: dict[int, dict[str, dict[str, Any]]] = {}
    for row in summaries:
        pct = int(TASKS[row["task"]]["sampling_pct"])
        fam = TASKS[row["task"]]["sampling_family"]
        by_pct.setdefault(pct, {})[fam] = row
    for pct, fams in by_pct.items():
        rad = fams.get("rademacher")
        scr = fams.get("scrambled_hadamard")
        if rad and scr:
            compensation[f"{pct}pct_scr_bp_advantage"] = float(scr["bp_psnr_mean"] - rad["bp_psnr_mean"])
            compensation[f"{pct}pct_rad_gain_advantage"] = float(rad["gain_psnr_mean"] - scr["gain_psnr_mean"])
            compensation[f"{pct}pct_final_gap_rad_minus_scr"] = float(rad["final_psnr_mean"] - scr["final_psnr_mean"])
    save_json(compensation, output_dir / "compensation_summary.json")
    lines = [
        "# Phase 48/49 Mechanistic Probe Report",
        "",
        "This is an eval-only diagnostic package. It does not train models and it does not change main no-leak checkpoints.",
        "",
        "## Key Diagnostic Questions",
        "",
        "1. A^T y vs D(y)/B_lambda y: see rawGI_vs_anchor_results for each task.",
        "2. P_N gate measurement visibility: see gate_geometry_results and Ar_raw_vs_filtered_hist.",
        "3. Pi_y audit re-legalization: see audit_correction_results and pre/post RelMeasErr plots.",
        "4. Rad/Scr provenance distribution: see provenance_distribution; indicators are regularized soft diagnostics, not orthogonal percentages.",
        "5. Per-sample compensation: see per_sample_compensation and compensation_summary.json.",
        "6. Measurement dependence: see measurement_dependence_results; paired shuffles are distinguished from mismatched shuffles.",
        "7. lambda_op stability/control: see lambda_sweep_results.",
        "8. Train-time ablations are still required for causal conclusions about P_N and final audit.",
        "",
        "## Aggregate Compensation",
        "",
    ]
    for key, value in compensation.items():
        lines.append(f"- {key}: {value:.4f}")
    (output_dir / "MECHANISTIC_PROBE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    claims = [
        "# Supported Mechanistic Claims",
        "",
        "These claims are conditional on the corresponding CSV/plot values, and should remain exploratory/diagnostic until reviewed.",
        "",
        "- D(y) and ridge B_lambda y are measured explicitly against raw A^T y; only claim better measurement consistency if rawGI_vs_anchor_results show it.",
        "- P_N may be described as reducing measurement-visible residual energy only if gate_leakage_ratio is below 1.",
        "- Pi_y may be described as re-legalizing outputs only if post-audit RelMeasErr is below pre-audit RelMeasErr.",
        "- Rad completion-dominated and Scr anchor-assisted language should be tied to provenance and per-sample compensation statistics.",
        "- Perturbation results support measurement dependence, not universal robustness.",
        "",
        "Do not claim SOTA, hardware validation, universal robustness, or a completed train-time ablation conclusion from this eval-only session.",
    ]
    (output_dir / "SUPPORTED_MECHANISTIC_CLAIMS.md").write_text("\n".join(claims) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    write_environment(output_dir)
    summaries = []
    for task_key in args.tasks:
        if task_key not in TASKS:
            raise ValueError(f"Unknown task key: {task_key}")
        summaries.append(run_task_probes(args, task_key, output_dir, device))
    write_final_reports(output_dir, summaries)
    write_session_manifest(
        output_dir,
        "session_01_eval_probes",
        {
            "trains": False,
            "tasks": ",".join(args.tasks),
            "bundle_root": args.bundle_root,
            "limit_samples": args.limit_samples,
            "output_dir": str(output_dir),
        },
    )
    write_sha256s(output_dir)
    save_json({"ok": True, "session": "session_01_eval_probes", "output_dir": str(output_dir)}, output_dir / "SESSION_STATUS.json")
    print(f"Session 01 complete: {output_dir}")


if __name__ == "__main__":
    main()
