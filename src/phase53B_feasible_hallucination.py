from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .phase53B_common import (
    add_common_args,
    audited_cross_pair,
    configure_task,
    copy_checkpoint_for_manifest,
    finalize_session,
    load_generator,
    make_loader,
    relmeas_tensor,
    resolve_device,
    save_image_grid,
    write_command_log,
    write_rows,
)
from .metrics import batch_metrics
from .utils import ensure_dir, reconstruct_from_measurements, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53B Session22 feasible hallucination figure.")
    add_common_args(parser)
    parser.add_argument("--examples_per_task", type=int, default=6)
    return parser.parse_args()


def _save_grid_pdf(path: Path, rows: list[list[torch.Tensor]], titles: list[str], max_rows: int = 6) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = rows[:max_rows]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nrows, ncols = len(rows), len(titles)
    plt.figure(figsize=(2.2 * ncols, 2.1 * nrows))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = plt.subplot(nrows, ncols, r * ncols + c + 1)
            arr = img.detach().float().cpu().squeeze().numpy()
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_examples(task_out: Path, task: str, grid_rows: list[list[torch.Tensor]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    ex_dir = ensure_dir(task_out / "feasible_hallucination_examples")
    names = ["gt", "ours", "cross_feasible", "abs_diff"]
    for i, row in enumerate(grid_rows):
        for name, img in zip(names, row):
            arr = img.detach().float().cpu().squeeze().numpy()
            plt.imsave(ex_dir / f"{task}_{i:03d}_{name}.png", arr, cmap="gray", vmin=0, vmax=1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    write_command_log(out)
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows = []
    all_grid_rows: list[list[torch.Tensor]] = []
    titles = ["GT x_i", "Ours xhat_i", "Cross-feasible Pi_yi(x_j)", "Absolute diff"]
    for task in args.tasks:
        task_out = ensure_dir(out / task)
        info, config, measurement, exact_info = configure_task(args, task, task_out, device)
        copy_checkpoint_for_manifest(info, task_out)
        generator = load_generator(info, config, measurement, device)
        loader = make_loader(config, device)
        task_grid_rows: list[list[torch.Tensor]] = []
        seen = 0
        for batch_idx, batch in enumerate(loader):
            x = batch[0].to(device, non_blocking=True)
            if x.shape[0] < 2:
                continue
            y = measurement.measure(x)
            xhat, _x_data, _extras = reconstruct_from_measurements(
                generator,
                measurement,
                y,
                use_null_project=bool(config.get("use_null_project", True)),
                use_dc_project=bool(config.get("use_dc_project", True)),
                use_final_dc_project=True,
                backprojection_mode=config.get("backprojection_mode", "ridge_pinv"),
                enable_refiner=True,
                output_range_mode=config.get("output_range_mode", "clamp_eval_only"),
                return_extras=True,
            )
            cross = torch.clamp(audited_cross_pair(measurement, x, y), 0.0, 1.0)
            x_flat = measurement.flatten_img(x.float())
            xhat_flat = measurement.flatten_img(xhat.float())
            cross_flat = measurement.flatten_img(cross.float())
            rel_cross = relmeas_tensor(measurement, cross_flat, y)
            rel_ours = relmeas_tensor(measurement, xhat_flat, y)
            ours_metrics = batch_metrics(xhat, x, measurement, y)
            cross_metrics = batch_metrics(cross, x, measurement, y)
            for i in range(x.shape[0]):
                if seen >= args.examples_per_task:
                    break
                row = {
                    "task": task,
                    "family": info["metadata"]["display"],
                    "batch": batch_idx,
                    "sample": i,
                    "ours_psnr": ours_metrics["psnr"],
                    "ours_ssim": ours_metrics["ssim"],
                    "ours_relmeas": float(rel_ours[i].detach().cpu()),
                    "cross_psnr": cross_metrics["psnr"],
                    "cross_ssim": cross_metrics["ssim"],
                    "cross_relmeas": float(rel_cross[i].detach().cpu()),
                    "exact_A_loaded": bool(exact_info.get("exact_A_loaded", False)),
                }
                rows.append(row)
                diff = torch.abs(x[i : i + 1] - cross[i : i + 1])
                grid_row = [x[i].cpu(), xhat[i].cpu(), cross[i].cpu(), diff[0].cpu()]
                task_grid_rows.append(grid_row)
                all_grid_rows.append(grid_row)
                seen += 1
            if seen >= args.examples_per_task:
                break
        _save_examples(task_out, task, task_grid_rows)
    write_rows(out, "feasible_hallucination_metrics", rows, "Feasible Hallucination Metrics")
    save_image_grid(out / "feasible_hallucination_grid.png", all_grid_rows, titles, max_rows=8)
    _save_grid_pdf(out / "feasible_hallucination_grid.pdf", all_grid_rows, titles, max_rows=8)
    report = [
        "# Feasible Hallucination Report",
        "",
        "This eval-only session constructs cross-feasible examples `Pi_yi(x_j)` with `j != i`.",
        "",
        "The figure supports the physical point that different images can satisfy the same bucket measurements; the row-space certificate cannot validate null-space plausibility.",
        "",
        "Columns: GT, existing reconstruction, cross-feasible audited image, absolute difference.",
    ]
    (out / "FEASIBLE_HALLUCINATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    finalize_session(
        out,
        args.session_name,
        True,
        {
            "session_type": "feasible_hallucination_dataset",
            "trains_generator": False,
            "trains_discriminator": False,
            "results_csv": str(out / "feasible_hallucination_metrics.csv"),
            "figure_png": str(out / "feasible_hallucination_grid.png"),
            "figure_pdf": str(out / "feasible_hallucination_grid.pdf"),
        },
    )


if __name__ == "__main__":
    main()
