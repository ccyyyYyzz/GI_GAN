from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .datasets import get_val_dataloader
from .eval import make_measurement
from .exact_measurement import apply_measurement_override_from_config, torch_load
from .models import build_generator
from .phase53D_common import (
    add_phase53d_args,
    metrics_for_images,
    reconstruct_no_full_training,
    resolve_device,
    save_bar,
    save_image_grid,
    save_scatter,
    write_rows,
)
from .utils import apply_experiment_defaults, ensure_dir, load_config, save_config, save_json, set_seed


SESSION_SPECS = [
    ("Rad-5 no_final_audit", "rad5", "Phase48/49", "session_03_rad5_no_final_audit"),
    ("Rad-5 no_gate_no_final_audit", "rad5", "Phase51A", "session_06_rad5_no_gate_no_final_audit"),
    ("Rad-5 no_final_audit_no_meas_loss", "rad5", "Phase51A", "session_08_rad5_no_final_audit_no_meas_loss"),
    ("Scr-5 no_final_audit", "scr5", "Phase48/49", "session_05_scr5_no_final_audit"),
    ("Scr-5 no_gate_no_final_audit", "scr5", "Phase51A", "session_07_scr5_no_gate_no_final_audit"),
    ("Scr-5 no_final_audit_no_meas_loss", "scr5", "Phase51A", "session_09_scr5_no_final_audit_no_meas_loss"),
]
LAMBDA_GRID = ["hard", 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53D post-hoc certificate sweep on no-audit ablations.")
    add_phase53d_args(parser)
    parser.set_defaults(limit_samples=128)
    parser.add_argument("--lambda_grid", nargs="*", default=LAMBDA_GRID)
    return parser.parse_args()


def session_root(args, source_phase: str, session: str) -> Path:
    return (Path(args.phase51A_root) if source_phase == "Phase51A" else Path(args.phase48_root)) / session


def find_first(root: Path, names: list[str]) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing any of {names} in {root}")


def load_session(session_dir: Path, args, device: torch.device):
    config_path = find_first(session_dir, ["resolved_config.yaml", "config_used.yaml", "run_config.yaml"])
    checkpoint_path = find_first(session_dir, ["best_or_final_checkpoint.pt", "checkpoint_final.pt", "last.pt", "best_hq.pt"])
    config = apply_experiment_defaults(load_config(config_path))
    config["dataset_root"] = args.dataset_root
    config["device"] = str(device)
    config["batch_size"] = int(args.batch_size)
    config["num_workers"] = int(args.num_workers)
    config["limit_val_samples"] = int(args.limit_samples)
    exact_path = session_dir / "measurement_operator_exact.pt"
    if exact_path.exists():
        config["measurement_operator_exact_path"] = str(exact_path)
        config["exact_A_required"] = True
    measurement = make_measurement(config, device)
    exact_info = apply_measurement_override_from_config(config, measurement, device)
    checkpoint = torch_load(checkpoint_path, map_location=device)
    generator = build_generator(config, measurement=measurement).to(device)
    state = checkpoint.get("generator_ema") or checkpoint.get("generator") if isinstance(checkpoint, dict) else checkpoint
    if state is None:
        raise RuntimeError(f"No generator state found in {checkpoint_path}")
    generator.load_state_dict(state)
    generator.eval()
    return config_path, checkpoint_path, config, measurement, exact_info, generator


def apply_certificate(measurement, flat: torch.Tensor, y: torch.Tensor, lam: str | float) -> torch.Tensor:
    A = measurement.get_current_A().detach().float().to(flat.device)
    residual = measurement.A_forward(flat.float()) - y.float()
    gram = A @ A.T
    if str(lam) == "hard":
        K = gram
    else:
        K = gram + float(lam) * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    z = torch.linalg.solve(K, residual.T).T
    return flat.float() - z @ A


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    task_root = ensure_dir(out / "posthoc_certificate_sweep")
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows: list[dict[str, Any]] = []
    grid_rows = []
    titles = ["GT", "Before", "After hard", "Abs correction"]
    for display, task_key, phase, session in SESSION_SPECS:
        session_dir = session_root(args, phase, session)
        if not session_dir.exists():
            rows.append({"variant": display, "session": session, "status": "missing", "source_phase": phase})
            continue
        session_out = ensure_dir(task_root / session)
        config_path, checkpoint_path, config, measurement, exact_info, generator = load_session(session_dir, args, device)
        save_config(config, session_out / "config_used.yaml")
        save_json(
            {
                "source_phase": phase,
                "session_dir": str(session_dir),
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint_path),
                "exact_A_info": exact_info,
                "note": "Post-hoc eval-only certificate sweep; generator weights are not trained.",
            },
            session_out / "posthoc_source_manifest.json",
        )
        loader = get_val_dataloader(
            dataset_root=config["dataset_root"],
            img_size=int(config["img_size"]),
            batch_size=int(config.get("batch_size", 16)),
            num_workers=int(config.get("num_workers", 2)),
            limit_val_samples=int(config.get("limit_val_samples", args.limit_samples)),
            seed=int(config.get("seed", 123)),
            pin_memory=device.type == "cuda",
            dataset_name=config.get("dataset_name", "stl10"),
            class_filter=config.get("class_filter"),
        )
        sums: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        first_visual_done = False
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            before, _xdata, _extras = reconstruct_no_full_training(generator, measurement, y, config, final_audit=False)
            before = before.clamp(0, 1)
            before_metrics = metrics_for_images(before, x, measurement, y)
            before_flat = measurement.flatten_img(before.float())
            for lam in args.lambda_grid:
                lam_key = str(lam)
                try:
                    after_flat = apply_certificate(measurement, before_flat, y, lam)
                    status = "ok"
                except Exception as exc:
                    rows.append({"variant": display, "session": session, "lambda": lam_key, "status": f"failed: {exc}", "source_phase": phase})
                    continue
                after = measurement.unflatten_img(after_flat).clamp(0, 1)
                after_metrics = metrics_for_images(after, x, measurement, y)
                correction_norm = torch.linalg.norm(after_flat - before_flat, dim=1) / torch.linalg.norm(before_flat, dim=1).clamp_min(1e-12)
                key = lam_key
                item = sums.setdefault(
                    key,
                    {
                        "psnr_before": 0.0,
                        "psnr_after": 0.0,
                        "ssim_before": 0.0,
                        "ssim_after": 0.0,
                        "relmeas_before": 0.0,
                        "relmeas_after": 0.0,
                        "correction_norm": 0.0,
                    },
                )
                bsz = x.shape[0]
                item["psnr_before"] += before_metrics["psnr"] * bsz
                item["psnr_after"] += after_metrics["psnr"] * bsz
                item["ssim_before"] += before_metrics["ssim"] * bsz
                item["ssim_after"] += after_metrics["ssim"] * bsz
                item["relmeas_before"] += before_metrics["rel_meas_err"] * bsz
                item["relmeas_after"] += after_metrics["rel_meas_err"] * bsz
                item["correction_norm"] += float(correction_norm.mean().detach().cpu()) * bsz
                counts[key] = counts.get(key, 0) + bsz
                if not first_visual_done and lam_key == "hard":
                    for i in range(min(2, x.shape[0])):
                        grid_rows.append([x[i].cpu(), before[i].cpu(), after[i].cpu(), torch.abs(after[i] - before[i]).cpu()])
                    first_visual_done = True
        for lam_key, item in sums.items():
            n = max(1, counts[lam_key])
            avg = {k: v / n for k, v in item.items()}
            rel_drop = avg["relmeas_before"] - avg["relmeas_after"]
            psnr_loss = avg["psnr_before"] - avg["psnr_after"]
            rows.append(
                {
                    "variant": display,
                    "task": task_key,
                    "session": session,
                    "source_phase": phase,
                    "lambda": lam_key,
                    "status": "ok",
                    "n_eval": n,
                    "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                    **avg,
                    "psnr_change_after_minus_before": avg["psnr_after"] - avg["psnr_before"],
                    "ssim_change_after_minus_before": avg["ssim_after"] - avg["ssim_before"],
                    "relmeas_change_after_minus_before": avg["relmeas_after"] - avg["relmeas_before"],
                    "psnr_loss_per_relmeas_reduction": psnr_loss / max(1e-12, rel_drop),
                }
            )
        generator.cpu()
    write_rows(out, "posthoc_certificate_sweep", rows, "Phase53D Post-hoc Certificate Sweep")
    save_scatter(out / "posthoc_psnr_vs_relmeaserr.png", [r for r in rows if r.get("status") == "ok"], "relmeas_after", "psnr_after", "Posthoc PSNR vs RelMeasErr", "RelMeasErr after", "PSNR after")
    save_bar(out / "posthoc_lambda_tradeoff.png", [r for r in rows if r.get("status") == "ok"], "lambda", "psnr_change_after_minus_before", "Posthoc lambda PSNR tradeoff", "PSNR change")
    save_image_grid(out / "posthoc_visual_grid.png", grid_rows, titles, max_rows=8)
    report = [
        "# Phase53D Post-hoc Certificate Sweep Report",
        "",
        "This eval-only sweep applies analytic `Pi_y^lambda` after no-audit reconstructions.",
        "If RelMeasErr drops sharply while PSNR changes little, `Pi_y` behaves like certificate / re-legalization rather than a train-time PSNR engine.",
    ]
    (out / "POSTHOC_CERTIFICATE_SWEEP_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out / "POSTHOC_CERTIFICATE_SWEEP_REPORT.md")


if __name__ == "__main__":
    main()

