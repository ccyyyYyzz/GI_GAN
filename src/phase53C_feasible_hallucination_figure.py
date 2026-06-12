from __future__ import annotations

import argparse

import torch

from .phase53C_common import (
    add_common_args,
    configure_task,
    copy_checkpoint_for_manifest,
    finalize_session,
    hard_audit,
    load_generator,
    make_loader,
    prepare_exact_projector,
    relmeas_tensor,
    resolve_device,
    save_image_grid,
    write_command_log,
    write_rows,
)
from .metrics import batch_metrics
from .utils import ensure_dir, reconstruct_from_measurements, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53C Session22 feasible hallucination figure.")
    add_common_args(parser)
    parser.add_argument("--examples_per_task", type=int, default=6)
    return parser.parse_args()


def save_pdf(path, rows, titles):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    nrows, ncols = min(len(rows), 8), len(titles)
    plt.figure(figsize=(2.2 * ncols, 2.1 * nrows))
    for r, row in enumerate(rows[:nrows]):
        for c, img in enumerate(row):
            ax = plt.subplot(nrows, ncols, r * ncols + c + 1)
            ax.imshow(img.detach().cpu().float().squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


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


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    write_command_log(out)
    device = resolve_device(args.device)
    set_seed(args.seed)
    metric_rows = []
    grid_rows = []
    titles = ["GT", "Ours", "Cross-feasible", "Abs diff", "RelMeasErr bars"]
    for task in args.tasks:
        task_out = ensure_dir(out / task)
        info, config, measurement, exact_info = configure_task(args, task, task_out, device)
        copy_checkpoint_for_manifest(info, task_out)
        prepare_exact_projector(measurement, task_out)
        generator = load_generator(info, config, measurement, device)
        loader = make_loader(config, device)
        seen = 0
        for batch_idx, batch in enumerate(loader):
            x = batch[0].to(device, non_blocking=True)
            if x.shape[0] < 2:
                continue
            y = measurement.measure(x)
            xhat, _xdata, _extras = reconstruct_from_measurements(
                generator,
                measurement,
                y,
                use_null_project=bool(config.get("use_null_project", True)),
                use_dc_project=True,
                use_final_dc_project=True,
                backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=True,
                output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                return_extras=True,
            )
            xj = torch.roll(x, shifts=1, dims=0)
            cross = torch.clamp(hard_audit(measurement, xj, y), 0.0, 1.0)
            xhat_metrics = batch_metrics(xhat, x, measurement, y)
            cross_metrics = batch_metrics(cross, x, measurement, y)
            rel_gt = relmeas_tensor(measurement, measurement.flatten_img(x.float()), y)
            rel_ours = relmeas_tensor(measurement, measurement.flatten_img(xhat.float()), y)
            rel_cross = relmeas_tensor(measurement, measurement.flatten_img(cross.float()), y)
            for i in range(x.shape[0]):
                if seen >= args.examples_per_task:
                    break
                metric_rows.append(
                    {
                        "task": task,
                        "family": info["metadata"]["display"],
                        "batch": batch_idx,
                        "sample": i,
                        "gt_relmeas": float(rel_gt[i].cpu()),
                        "ours_relmeas": float(rel_ours[i].cpu()),
                        "cross_relmeas": float(rel_cross[i].cpu()),
                        "ours_psnr": xhat_metrics["psnr"],
                        "ours_ssim": xhat_metrics["ssim"],
                        "cross_psnr_vs_gt": cross_metrics["psnr"],
                        "cross_ssim_vs_gt": cross_metrics["ssim"],
                        "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                    }
                )
                diff = torch.abs(cross[i : i + 1] - x[i : i + 1])[0]
                bar = torch.zeros_like(x[i])
                bar[..., : int(bar.shape[-1] * min(1.0, float(rel_cross[i].cpu()) * 50.0))] = 1.0
                grid_rows.append([x[i].cpu(), xhat[i].cpu(), cross[i].cpu(), diff.cpu(), bar.cpu()])
                seen += 1
            if seen >= args.examples_per_task:
                break
    write_rows(out, "feasible_hallucination_metrics", metric_rows, "Feasible Hallucination Metrics")
    save_image_grid(out / "feasible_hallucination_grid.png", grid_rows, titles, max_rows=8)
    save_pdf(out / "feasible_hallucination_grid.pdf", grid_rows, titles)
    save_examples(out, grid_rows)
    report = [
        "# Feasible Hallucination Report",
        "",
        "This eval-only session constructs cross-feasible images `Pi_yi(x_j)`.",
        "The figure is intended to show that two visually different images can satisfy nearly the same bucket measurements.",
        "Measurement audit cannot by itself distinguish null-space plausibility.",
    ]
    (out / "FEASIBLE_HALLUCINATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    finalize_session(
        out,
        args.session_name,
        True,
        {
            "session_type": "feasible_hallucination_figure",
            "trains_generator": False,
            "trains_critic": False,
            "results_csv": str(out / "feasible_hallucination_metrics.csv"),
        },
    )


if __name__ == "__main__":
    main()
