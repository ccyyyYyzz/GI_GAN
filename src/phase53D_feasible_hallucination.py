from __future__ import annotations

import argparse

import torch

from .datasets import get_val_dataloader
from .phase53C_exact_projector import build_rowspace_basis
from .phase53D_common import (
    add_phase53d_args,
    configure_light_task,
    load_eval_generator,
    metrics_for_images,
    reconstruct_no_full_training,
    relmeas_from_images,
    resolve_device,
    save_image_grid,
    write_rows,
)
from .utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53D feasible hallucination figure and metrics.")
    add_phase53d_args(parser)
    parser.add_argument("--examples_per_task", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def hard_project_cross(measurement, xj: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    flat = measurement.flatten_img(xj.float())
    corrected = measurement.dc_project(flat, y.float())
    return measurement.unflatten_img(corrected)


def save_examples(out, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = ["gt", "ours", "cross_feasible", "abs_diff", "relmeas_bar"]
    ex = ensure_dir(out / "feasible_hallucination_examples")
    for i, row in enumerate(rows):
        for name, img in zip(names, row):
            plt.imsave(ex / f"{i:03d}_{name}.png", img.detach().cpu().float().squeeze().numpy(), cmap="gray", vmin=0, vmax=1)


def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    task_root = ensure_dir(out / "feasible_hallucination")
    device = resolve_device(args.device)
    set_seed(args.seed)
    metric_rows = []
    grid_rows = []
    titles = ["GT", "Ours", "Cross-feasible", "Abs diff", "RelMeasErr bars"]
    for task in args.tasks:
        task_out = ensure_dir(task_root / task)
        info, config, measurement, exact_info = configure_light_task(args, task, task_out, device)
        A = measurement.get_current_A().detach().float().to(device)
        Q = build_rowspace_basis(A)
        torch.save({"Q": Q.detach().cpu()}, task_out / "Q_exact_null.pt")
        generator = load_eval_generator(info, config, measurement, device)
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
        seen = 0
        for batch_idx, batch in enumerate(loader):
            x = batch[0].to(device, non_blocking=True)
            if x.shape[0] < 2:
                continue
            y = measurement.measure(x)
            xhat, _xdata, _extras = reconstruct_no_full_training(generator, measurement, y, config, final_audit=True)
            xj = torch.roll(x, shifts=1, dims=0)
            cross = hard_project_cross(measurement, xj, y).clamp(0, 1)
            rel_gt = relmeas_from_images(measurement, x, y)
            rel_ours = relmeas_from_images(measurement, xhat, y)
            rel_cross = relmeas_from_images(measurement, cross, y)
            ours_metrics = metrics_for_images(xhat, x, measurement, y)
            cross_metrics = metrics_for_images(cross, x, measurement, y)
            for i in range(x.shape[0]):
                if seen >= args.examples_per_task:
                    break
                metric_rows.append(
                    {
                        "task": task,
                        "family": info["metadata"]["display"],
                        "batch": batch_idx,
                        "sample": i,
                        "gt_relmeas": float(rel_gt[i].detach().cpu()),
                        "ours_relmeas": float(rel_ours[i].detach().cpu()),
                        "cross_relmeas": float(rel_cross[i].detach().cpu()),
                        "ours_psnr": ours_metrics["psnr"],
                        "ours_ssim": ours_metrics["ssim"],
                        "cross_psnr_vs_gt": cross_metrics["psnr"],
                        "cross_ssim_vs_gt": cross_metrics["ssim"],
                        "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                        "fallback_used": False,
                    }
                )
                diff = torch.abs(cross[i : i + 1] - x[i : i + 1])[0]
                bar = torch.zeros_like(x[i])
                width = int(bar.shape[-1] * min(1.0, float(rel_cross[i].detach().cpu()) * 50.0))
                bar[..., :width] = 1.0
                grid_rows.append([x[i].cpu(), xhat[i].cpu(), cross[i].cpu(), diff.cpu(), bar.cpu()])
                seen += 1
            if seen >= args.examples_per_task:
                break
        generator.cpu()
    write_rows(out, "feasible_hallucination_metrics", metric_rows, "Phase53D Feasible Hallucination Metrics")
    save_image_grid(out / "feasible_hallucination_grid.png", grid_rows, titles, max_rows=8)
    save_image_grid(out / "feasible_hallucination_grid.pdf", grid_rows, titles, max_rows=8, pdf=True)
    save_examples(out, grid_rows)
    report = [
        "# Phase53D Feasible Hallucination Report",
        "",
        "This eval-only diagnostic constructs `u_ij = Pi_yi(x_j)` using the analytic projector.",
        "Cross-feasible images can have very low measurement error with respect to `y_i` while remaining visually different from `x_i`.",
        "Therefore measurement audit alone cannot distinguish all feasible images; null-space plausibility must be evaluated separately.",
    ]
    (out / "FEASIBLE_HALLUCINATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out / "FEASIBLE_HALLUCINATION_REPORT.md")


if __name__ == "__main__":
    main()

