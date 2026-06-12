from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .phase15_common import (
    IMPORTED_NOLEAK,
    PHASE15,
    ensure_dir,
    load_registry,
    method_by_id,
    numeric,
    registry_row,
    sha256_file,
    write_csv,
    write_json,
    write_md_table,
)


FIELDS = [
    "method_id",
    "original_psnr",
    "original_ssim",
    "reeval_psnr",
    "reeval_ssim",
    "abs_diff_psnr",
    "abs_diff_ssim",
    "exact_A_loaded",
    "exact_A_sha256",
    "checkpoint_sha256",
    "status",
    "notes",
]


def torch_load(path: Path, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def apply_exact_a(measurement: Any, exact_payload: Any, device: torch.device, lambda_dc: float) -> None:
    A = exact_payload["A"] if isinstance(exact_payload, dict) and "A" in exact_payload else exact_payload
    A = A.to(device=device, dtype=torch.float32)
    if hasattr(measurement, "set_A_override"):
        measurement.set_A_override(A, metadata={"source": "phase15_exactA_reeval"}, rebuild_cache=True)
        return
    measurement.A = A
    measurement.m = int(A.shape[0])
    measurement.n = int(A.shape[1])
    measurement.sampling_ratio = float(measurement.m / measurement.n)
    eye = torch.eye(measurement.m, device=device, dtype=A.dtype)
    measurement.K = A @ A.T + float(lambda_dc) * eye
    measurement._chol = None
    measurement._use_cholesky = True
    try:
        measurement._chol = torch.linalg.cholesky(measurement.K)
    except RuntimeError:
        measurement._use_cholesky = False


def run_eval(method_id: str, output_dir: Path, registry: dict[str, Any]) -> dict[str, Any]:
    exact_path = output_dir / "measurement_operator_exact.pt"
    checkpoint_path = Path(registry.get("checkpoint_path") or output_dir / "last.pt")
    exact_sha = sha256_file(exact_path) if exact_path.exists() else ""
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.exists() else ""
    original_psnr = numeric(registry.get("psnr"))
    original_ssim = numeric(registry.get("ssim"))
    base = {
        "method_id": method_id,
        "original_psnr": original_psnr,
        "original_ssim": original_ssim,
        "reeval_psnr": "",
        "reeval_ssim": "",
        "abs_diff_psnr": "",
        "abs_diff_ssim": "",
        "exact_A_loaded": False,
        "exact_A_sha256": exact_sha,
        "checkpoint_sha256": checkpoint_sha,
        "status": "not_run",
        "notes": "",
    }
    if not exact_path.exists():
        base.update({"status": "missing_exact_A", "notes": "measurement_operator_exact.pt is absent."})
        return base
    if not checkpoint_path.exists():
        base.update({"status": "missing_checkpoint", "notes": "No checkpoint path found."})
        return base
    try:
        from .datasets import get_val_dataloader
        from .eval import make_measurement
        from .metrics import batch_metrics
        from .models import build_generator
        from .utils import (
            apply_experiment_defaults,
            compare_metric_sets,
            load_config,
            mean_dict,
            reconstruct_from_measurements,
            resolve_device,
            save_json,
            set_seed,
        )
        from .visualize import save_recon_grid

        cfg_path = output_dir / "resolved_config.yaml"
        config = apply_experiment_defaults(load_config(str(cfg_path)))
        config["dataset_root"] = "E:/ns_mc_gan_gi/data"
        config["output_dir"] = str(output_dir)
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        device = resolve_device(config["device"])
        set_seed(int(config["seed"]))
        checkpoint = torch_load(checkpoint_path, device)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            merged = dict(config)
            merged.update(checkpoint["config"])
            merged["dataset_root"] = "E:/ns_mc_gan_gi/data"
            merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            config = apply_experiment_defaults(merged)
            device = resolve_device(config["device"])
        measurement = make_measurement(config, device)
        exact_payload = torch_load(exact_path, device)
        apply_exact_a(measurement, exact_payload, device, float(config.get("lambda_solver", 0.001)))
        generator = build_generator(config, measurement=measurement).to(device)
        if isinstance(checkpoint, dict):
            state = checkpoint.get("generator_ema") or checkpoint["generator"]
        else:
            state = checkpoint
        generator.load_state_dict(state)
        generator.eval()
        val_loader = get_val_dataloader(
            dataset_root=config["dataset_root"],
            img_size=config["img_size"],
            batch_size=config["batch_size"],
            num_workers=0,
            limit_val_samples=config["limit_val_samples"],
            seed=config["seed"],
            pin_memory=device.type == "cuda",
            dataset_name=config.get("dataset_name", "stl10"),
            class_filter=config.get("class_filter"),
        )
        reeval_dir = ensure_dir(PHASE15 / "exactA_reeval" / ("rademacher5" if "5" in method_id else "rademacher10"))
        sample_dir = ensure_dir(reeval_dir / "eval_samples")
        backprojection_metrics = []
        model_metrics = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Exact-A {method_id}")):
                x = batch[0].to(device, non_blocking=True)
                y = measurement.measure(x)
                x_hat, x_data, extras = reconstruct_from_measurements(
                    generator,
                    measurement,
                    y,
                    use_null_project=bool(config["use_null_project"]),
                    use_dc_project=bool(config["use_dc_project"]),
                    backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                    enable_refiner=True,
                    output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                    return_extras=True,
                )
                backprojection_metrics.append(batch_metrics(x_data, x, measurement, y))
                model_batch = batch_metrics(x_hat, x, measurement, y)
                model_batch["rel_meas_err_clamped"] = model_batch.get("rel_meas_error", float("nan"))
                model_batch["rel_meas_err_unclamped"] = batch_metrics(
                    extras["x_hat_unclamped"], x, measurement, y
                ).get("rel_meas_error", float("nan"))
                model_metrics.append(model_batch)
                if batch_idx == 0:
                    save_recon_grid(
                        x,
                        x_data,
                        x_hat,
                        sample_dir / "recon_grid.png",
                        max_items=int(config.get("num_eval_samples_to_save", 8)),
                        title=f"Exact A reeval | {method_id}",
                    )
        metrics = compare_metric_sets(mean_dict(backprojection_metrics), mean_dict(model_metrics))
        save_json(metrics, reeval_dir / "eval_metrics.json")
        reeval_psnr = float(metrics["model"]["psnr"])
        reeval_ssim = float(metrics["model"]["ssim"])
        diff_psnr = abs(reeval_psnr - original_psnr)
        diff_ssim = abs(reeval_ssim - original_ssim)
        status = "reproduced" if diff_psnr <= 0.02 and diff_ssim <= 0.002 else "mismatch"
        base.update(
            {
                "reeval_psnr": reeval_psnr,
                "reeval_ssim": reeval_ssim,
                "abs_diff_psnr": diff_psnr,
                "abs_diff_ssim": diff_ssim,
                "exact_A_loaded": True,
                "status": status,
                "notes": f"Full local eval written to {reeval_dir}.",
            }
        )
        return base
    except Exception as exc:
        base.update(
            {
                "exact_A_loaded": bool(exact_path.exists()),
                "status": "failed_interface",
                "notes": f"Exact A exists, but local eval interface failed: {type(exc).__name__}: {exc}",
            }
        )
        return base


def main() -> None:
    out_dir = ensure_dir(PHASE15 / "exactA_reeval")
    registry_rows = load_registry()
    registry_by_id = {row["method_id"]: row for row in registry_rows}
    rows = []
    for method_id in ["rademacher5_hq_noise001_colab", "rademacher10_full_noise001_colab"]:
        method = method_by_id(method_id)
        row = registry_by_id.get(method_id)
        if row is None:
            row = registry_row(method, IMPORTED_NOLEAK / method_id)
        rows.append(run_eval(method_id, IMPORTED_NOLEAK / method_id, row))
    write_csv(out_dir / "exactA_reeval_results.csv", rows, FIELDS)
    write_md_table(out_dir / "exactA_reeval_results.md", rows, FIELDS)
    write_json(out_dir / "exactA_reeval_results.json", rows)
    print(json.dumps({"exactA_reeval_rows": len(rows), "output": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
