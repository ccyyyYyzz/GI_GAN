from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .phase53B_common import (
    add_common_args,
    configure_task,
    copy_checkpoint_for_manifest,
    finalize_session,
    load_generator,
    make_loader,
    relmeas_tensor,
    resolve_device,
    save_bar_plot,
    save_image_grid,
    write_command_log,
    write_rows,
)
from .utils import ensure_dir, reconstruct_from_measurements, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53B Session24 posterior sampling pilot.")
    add_common_args(parser)
    parser.set_defaults(tasks=["scr5", "rad5"])
    parser.add_argument("--num_samples_per_y", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=4)
    parser.add_argument("--coverage_c", type=float, default=2.0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out = ensure_dir(args.output_dir)
    write_command_log(out)
    device = resolve_device(args.device)
    set_seed(args.seed)
    rows = []
    sample_grid_rows = []
    uncertainty_rows = []
    for task in args.tasks:
        task_out = ensure_dir(out / task)
        info, config, measurement, _exact_info = configure_task(args, task, task_out, device)
        copy_checkpoint_for_manifest(info, task_out)
        generator = load_generator(info, config, measurement, device)
        loader = make_loader(config, device)
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.eval_batches:
                break
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            samples = []
            for _k in range(args.num_samples_per_y):
                xhat, x_data, _extras = reconstruct_from_measurements(
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
                samples.append(xhat)
            stack = torch.stack(samples, dim=0)
            mean = stack.mean(dim=0)
            std = stack.std(dim=0, unbiased=False)
            y_mean = measurement.measure(mean)
            rel_mean = relmeas_tensor(measurement, measurement.flatten_img(mean.float()), y)
            abs_err = torch.abs(mean - x)
            coverage = ((x >= mean - args.coverage_c * std) & (x <= mean + args.coverage_c * std)).float().mean(dim=(1, 2, 3))
            corr_rows = []
            for i in range(x.shape[0]):
                s = std[i].flatten().float()
                e = abs_err[i].flatten().float()
                if torch.std(s) > 1e-12 and torch.std(e) > 1e-12:
                    corr = torch.corrcoef(torch.stack([s, e]))[0, 1].item()
                else:
                    corr = float("nan")
                diffs = stack[:, i] - mean[i]
                diff_flat = measurement.flatten_img(diffs.float())
                a_diff = measurement.A_forward(diff_flat)
                ratio = torch.linalg.norm(a_diff, dim=1) / torch.linalg.norm(diff_flat, dim=1).clamp_min(1e-12)
                rows.append(
                    {
                        "task": task,
                        "family": info["metadata"]["display"],
                        "batch": batch_idx,
                        "sample": i,
                        "K": args.num_samples_per_y,
                        "mean_relmeas": float(rel_mean[i].detach().cpu()),
                        "measurement_mean_abs_delta": float(torch.mean(torch.abs(y_mean[i] - y[i])).detach().cpu()),
                        "variance_nullspace_ratio_mean": float(ratio.mean().detach().cpu()),
                        "variance_nullspace_ratio_max": float(ratio.max().detach().cpu()),
                        "coverage_fraction": float(coverage[i].detach().cpu()),
                        "std_abs_error_corr": corr,
                        "mean_pixel_std": float(std[i].mean().detach().cpu()),
                    }
                )
                corr_rows.append(corr)
            if len(sample_grid_rows) < 8:
                sample_grid_rows.append([x[0].cpu(), mean[0].cpu(), std[0].clamp(0, 1).cpu(), abs_err[0].clamp(0, 1).cpu()])
                uncertainty_rows.append([samples[k][0].cpu() for k in range(min(4, len(samples)))])
    write_rows(out, "posterior_sampling_metrics", rows, "Posterior Sampling Metrics")
    save_bar_plot(out / "variance_nullspace_ratio.png", rows, "task", "variance_nullspace_ratio_mean", "Variance measurement visibility", "||A d|| / ||d||")
    save_image_grid(out / "uncertainty_maps.png", sample_grid_rows, ["GT", "Mean", "Std map", "Abs error"], max_rows=8)
    if uncertainty_rows:
        save_image_grid(out / "sample_grid.png", uncertainty_rows, ["z0", "z1", "z2", "z3"], max_rows=8)
    report = [
        "# Posterior Sampling Pilot Report",
        "",
        "This pilot uses repeated stochastic reconstructions with analytic audit active.",
        "",
        "The intended evidence is that diversity should lie mostly in the null/measured-weak space while samples remain measurement-audited.",
        "",
        "This is exploratory and should not be framed as a final posterior sampler without stronger calibration.",
    ]
    (out / "POSTERIOR_SAMPLING_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    finalize_session(
        out,
        args.session_name,
        True,
        {
            "session_type": "posterior_sampling_pilot",
            "trains_generator": False,
            "trains_discriminator": False,
            "results_csv": str(out / "posterior_sampling_metrics.csv"),
        },
    )


if __name__ == "__main__":
    main()
