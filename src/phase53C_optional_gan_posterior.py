from __future__ import annotations

import argparse
import copy

import torch
import torch.nn.functional as F

from .phase53C_common import (
    ProjectionConditionedCritic,
    add_common_args,
    collect_exact_null_pair_dataset,
    configure_task,
    copy_checkpoint_for_manifest,
    exact_null_component,
    load_generator,
    make_exact_null_inputs,
    make_loader,
    make_train_loader,
    prepare_exact_projector,
    relmeas_tensor,
    resolve_device,
    save_bar_plot,
    save_image_grid,
    train_binary_critic,
    write_command_log,
    write_rows,
    finalize_session,
)
from .metrics import batch_metrics
from .phase53B_common import anchor_from_y
from .run_protocol import CheckpointManager, enforce_run_protocol
from .utils import ensure_dir, reconstruct_from_measurements, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase53C Session24 optional exact-null GAN and posterior sampling.")
    add_common_args(parser)
    parser.set_defaults(tasks=["scr5", "rad5"])
    parser.add_argument("--auc_gate", type=float, default=0.70)
    parser.add_argument("--beta_grid", nargs="*", type=float, default=[1e-5, 3e-5, 1e-4])
    parser.add_argument("--max_steps", type=int, default=180)
    parser.add_argument("--num_samples_per_y", type=int, default=8)
    parser.add_argument("--eval_batches", type=int, default=4)
    parser.add_argument("--checkpoint_every", type=int, default=60)
    return parser.parse_args()


def _critic_input(measurement, Q, u, anchor):
    return torch.cat([exact_null_component(measurement, u, Q), anchor], dim=1)


def _posterior_rows(generator, config, measurement, loader, device, task, family, K, max_batches):
    rows = []
    grids = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = batch[0].to(device, non_blocking=True)
            y = measurement.measure(x)
            samples = []
            for _ in range(K):
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
                samples.append(xhat)
            stack = torch.stack(samples)
            mean = stack.mean(dim=0)
            std = stack.std(dim=0, unbiased=False)
            abs_err = torch.abs(mean - x)
            for i in range(x.shape[0]):
                diffs = stack[:, i] - mean[i]
                diff_flat = measurement.flatten_img(diffs.float())
                a_diff = measurement.A_forward(diff_flat)
                ratio = torch.linalg.norm(a_diff, dim=1) / torch.linalg.norm(diff_flat, dim=1).clamp_min(1e-12)
                rows.append(
                    {
                        "task": task,
                        "family": family,
                        "batch": batch_idx,
                        "sample": i,
                        "K": K,
                        "variance_null_ratio_mean": float(ratio.mean().cpu()),
                        "mean_pixel_std": float(std[i].mean().cpu()),
                        "std_error_corr_proxy": float((std[i] * abs_err[i]).mean().cpu()),
                    }
                )
            if len(grids) < 8:
                grids.append([x[0].cpu(), mean[0].cpu(), std[0].clamp(0, 1).cpu(), abs_err[0].clamp(0, 1).cpu()])
    return rows, grids


def main() -> None:
    args = parse_args()
    # g2r protocol: run IDs prefixed g2r_, no paper-1 output dirs; training
    # in this pilot consumes the TRAIN split only ("none" = no val loader).
    enforce_run_protocol(args.output_dir, {"run_id": args.session_name, "val_split": "none"})
    out = ensure_dir(args.output_dir)
    write_command_log(out)
    device = resolve_device(args.device)
    set_seed(args.seed)
    gan_rows = []
    posterior_rows = []
    grids = []
    for task in args.tasks:
        task_out = ensure_dir(out / task)
        info, config, measurement, _exact_info = configure_task(args, task, task_out, device)
        copy_checkpoint_for_manifest(info, task_out)
        Q, _checks = prepare_exact_projector(measurement, task_out)
        generator = load_generator(info, config, measurement, device)
        data = collect_exact_null_pair_dataset(config, measurement, Q, device)
        critic = ProjectionConditionedCritic()
        critic, metrics, _labels, _scores = train_binary_critic(
            critic,
            make_exact_null_inputs(data, device),
            data["label"],
            epochs=args.critic_epochs,
            lr=args.critic_lr,
            device=device,
        )
        can_gan = task.startswith("scr") and float(metrics["auc"]) >= float(args.auc_gate)
        if can_gan:
            base_state = copy.deepcopy(generator.state_dict())
            for beta in args.beta_grid:
                generator.load_state_dict(base_state)
                generator.train()
                critic.train()
                g_opt = torch.optim.AdamW(generator.parameters(), lr=5e-5, weight_decay=1e-5)
                d_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)
                steps = 0
                beta_tag = f"beta_{beta:.0e}".replace("-", "m")
                beta_out = ensure_dir(task_out / beta_tag)
                # TRAIN split only: the GAN update loop must never see test data.
                train_loader = make_train_loader(config, device)

                def _save_gan_checkpoint(path, context, _generator=generator, _critic=critic, _beta=beta):
                    torch.save(
                        {
                            "generator": _generator.state_dict(),
                            "critic": _critic.state_dict(),
                            "beta": _beta,
                            "config": config,
                            **context,
                        },
                        path,
                    )

                with CheckpointManager(
                    beta_out,
                    run_id=args.session_name,
                    save_fn=_save_gan_checkpoint,
                    save_every_steps=int(args.checkpoint_every),
                    validate_dir=False,
                ) as ckpt:
                    while steps < args.max_steps:
                        for batch in train_loader:
                            x = batch[0].to(device, non_blocking=True)
                            y = measurement.measure(x)
                            anchor = anchor_from_y(measurement, y, config)
                            with torch.no_grad():
                                xhat_det, _xd, _ex = reconstruct_from_measurements(
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
                            d_pos = critic(_critic_input(measurement, Q, x, anchor))
                            d_neg = critic(_critic_input(measurement, Q, xhat_det, anchor))
                            d_loss = 0.5 * (
                                F.binary_cross_entropy_with_logits(d_pos, torch.ones_like(d_pos))
                                + F.binary_cross_entropy_with_logits(d_neg, torch.zeros_like(d_neg))
                            )
                            d_opt.zero_grad(set_to_none=True)
                            d_loss.backward()
                            d_opt.step()
                            xhat, _xd, _ex = reconstruct_from_measurements(
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
                            g_score = critic(_critic_input(measurement, Q, xhat, anchor))
                            img_loss = F.l1_loss(xhat, x)
                            # RelMeasErr is ALWAYS computed on the UNCLIPPED vector.
                            rel = relmeas_tensor(
                                measurement, measurement.flatten_img(_ex["x_hat_unclamped"].float()), y
                            ).mean()
                            adv = F.binary_cross_entropy_with_logits(g_score, torch.ones_like(g_score))
                            g_loss = img_loss + 0.05 * rel + float(beta) * adv
                            g_opt.zero_grad(set_to_none=True)
                            g_loss.backward()
                            g_opt.step()
                            steps += 1
                            ckpt.step()
                            if steps >= args.max_steps:
                                break
                generator.eval()
                # Post-training checkpoint (the historical pilot never saved one).
                _save_gan_checkpoint(
                    beta_out / "gan_posterior_trained.pt",
                    {"run_id": args.session_name, "step": steps, "reason": "post_training"},
                )
                eval_metrics = []
                with torch.no_grad():
                    for bidx, batch in enumerate(make_loader(config, device)):
                        if bidx >= args.eval_batches:
                            break
                        x = batch[0].to(device)
                        y = measurement.measure(x)
                        xhat, _xd, _ex = reconstruct_from_measurements(
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
                        m = batch_metrics(xhat, x, measurement, y)
                        # RelMeasErr is ALWAYS computed on the UNCLIPPED vector.
                        m["rel_meas_error"] = float(
                            relmeas_tensor(
                                measurement, measurement.flatten_img(_ex["x_hat_unclamped"].float()), y
                            )
                            .mean()
                            .cpu()
                        )
                        eval_metrics.append(m)
                row = {
                    "task": task,
                    "family": info["metadata"]["display"],
                    "beta": beta,
                    "auc_gate": metrics["auc"],
                    "status": "ran_gan_pilot",
                    "psnr": sum(m["psnr"] for m in eval_metrics) / max(1, len(eval_metrics)),
                    "ssim": sum(m["ssim"] for m in eval_metrics) / max(1, len(eval_metrics)),
                    "rel_meas_error": sum(m["rel_meas_error"] for m in eval_metrics) / max(1, len(eval_metrics)),
                }
                gan_rows.append(row)
        else:
            gan_rows.append({"task": task, "family": info["metadata"]["display"], "auc_gate": metrics["auc"], "status": "skipped_gan_gate_failed_or_rad"})
        generator.eval()
        pr, grid = _posterior_rows(generator, config, measurement, make_loader(config, device), device, task, info["metadata"]["display"], args.num_samples_per_y, args.eval_batches)
        posterior_rows.extend(pr)
        grids.extend(grid)
    write_rows(out, "optional_gan_results", gan_rows, "Optional GAN Results")
    write_rows(out, "posterior_sampling_metrics", posterior_rows, "Posterior Sampling Metrics")
    save_bar_plot(out / "perception_distortion_curve.png", gan_rows, "beta", "psnr", "Optional GAN pilot", "PSNR")
    save_bar_plot(out / "variance_null_ratio.png", posterior_rows, "task", "variance_null_ratio_mean", "Posterior variance null ratio", "||A d||/||d||")
    save_image_grid(out / "uncertainty_maps.png", grids, ["GT", "Mean", "Std", "Abs error"], max_rows=8)
    save_image_grid(out / "sample_grid.png", grids, ["GT", "Mean", "Std", "Abs error"], max_rows=8)
    report = [
        "# Optional GAN and Posterior Sampling Report",
        "",
        "GAN training is gated by Session20-style AUC. If the gate fails, only posterior sampling diagnostics are run.",
        "D input is exact `P0 xhat` and `x_data`; D never sees residuals or RelMeasErr.",
        "All results are exploratory and must not be used as final main GAN claims without later approval.",
    ]
    (out / "OPTIONAL_GAN_POSTERIOR_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    finalize_session(
        out,
        args.session_name,
        True,
        {
            "session_type": "optional_gan_and_posterior_sampling",
            "trains_generator": "gated_optional",
            "trains_critic": True,
            "gan_csv": str(out / "optional_gan_results.csv"),
            "posterior_csv": str(out / "posterior_sampling_metrics.csv"),
        },
    )


if __name__ == "__main__":
    main()
